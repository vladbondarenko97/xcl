//! `XCL_ACTIVITY`: where the title's submits and XCL's page mappings spend their time, per 2 s
//! (calls, total and longest), and how far the GPU is behind what was submitted.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct Timer {
    n: AtomicU64,
    us: AtomicU64,
    max: AtomicU64,
}

impl Timer {
    pub const fn new() -> Timer {
        Timer { n: AtomicU64::new(0), us: AtomicU64::new(0), max: AtomicU64::new(0) }
    }

    pub fn add(&self, d: Duration) {
        let us = d.as_micros() as u64;
        self.n.fetch_add(1, Ordering::Relaxed);
        self.us.fetch_add(us, Ordering::Relaxed);
        self.max.fetch_max(us, Ordering::Relaxed);
    }

    pub fn time<R>(&self, f: impl FnOnce() -> R) -> R {
        let t = Instant::now();
        let r = f();
        self.add(t.elapsed());
        r
    }

    /// (calls, ms, longest ms) since the last take
    pub fn take(&self) -> (u64, u64, u64) {
        (self.n.swap(0, Ordering::Relaxed), self.us.swap(0, Ordering::Relaxed) / 1000, self.max.swap(0, Ordering::Relaxed) / 1000)
    }
}

/// the title's ExecuteCommandLists, whole
pub static EXECUTE: Timer = Timer::new();
/// its `title_textures::refresh`, and the parts of the uploads in it
pub static REFRESH: Timer = Timer::new();
pub static HASH: Timer = Timer::new();
pub static DETILE: Timer = Timer::new();
pub static UPLOAD: Timer = Timer::new();
pub static UPLOAD_MB: AtomicU64 = AtomicU64::new(0);
/// its sparse readbacks and CPU-write work after each chunk
pub static READBACK: Timer = Timer::new();
pub static CPU_WRITES: Timer = Timer::new();
/// UpdateTileMappings calls (CPU time in the call) on XCL's page and sparse queues
pub static UTM_PAGES: Timer = Timer::new();
pub static UTM_SPARSE: Timer = Timer::new();

pub fn line() -> String {
    let f = |t: &Timer| {
        let (n, ms, max) = t.take();
        format!("{n}/{ms} ms (max {max})")
    };
    let (budget, local) = super::real::video_budget().unwrap_or((0, 0));
    format!(
        "costs (2 s): title submits {}, refresh {} [hash {}, detile {}, upload {}, {} MB], sparse readback {}, cpu writes {}; UpdateTileMappings pages {}, sparse {}; GPU behind: title fences {} signals, pages queue {}, sparse queue {}; local {} of {} MB budget",
        f(&EXECUTE),
        f(&REFRESH),
        f(&HASH),
        f(&DETILE),
        f(&UPLOAD),
        UPLOAD_MB.swap(0, Ordering::Relaxed),
        f(&READBACK),
        f(&CPU_WRITES),
        f(&UTM_PAGES),
        f(&UTM_SPARSE),
        super::fence_x::backlog(),
        super::vram::queue_lag().map_or("?".into(), |v| v.to_string()),
        super::sparse::queue_lag().map_or("?".into(), |v| v.to_string()),
        local >> 20,
        budget >> 20,
    )
}
