//! Console texture detiling on the GPU: a title texture's tiled bytes go to the upload buffer
//! as they are (one copy) and a compute shader writes its subresources as linear rows into a
//! scratch buffer at their `GetCopyableFootprints` places; the copies into the texture read
//! from there. The CPU did it element by element (`Chain::detile_level`) – most of a retail
//! game's render thread while it streamed textures. The same formula as `Layout::offset_z`.
//! `XCL_GPU_DETILE=0` keeps the CPU path, which is also taken when DXC or the pipeline is
//! missing and for textures the plan cannot take.

use super::tiling::{Chain, MipLevel};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use windows::Win32::Graphics::Direct3D12::*;

/// One subresource's detile: the shader's root constants, in its cbuffer order. `src_off` is
/// from the start of the texture's bytes and the `t_*` from the start of its tables until
/// `upload_batch` places them; `dst_*` are filled there from the footprints.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Params {
    pub src_off: u32,
    pub src_len: u32,
    pub lvl_off: u32,
    pub pitch_blocks: u32,
    pub ox0: u32,
    pub oy0: u32,
    pub width: u32,
    pub height: u32,
    pub bw_log2: u32,
    pub bh_log2: u32,
    pub bd_log2: u32,
    pub block_bytes: u32,
    pub base_xor: u32,
    pub e_log2: u32,
    pub t_ox: u32,
    pub t_oy: u32,
    pub t_oz: u32,
    pub t_hx: u32,
    pub t_hy: u32,
    pub t_hz: u32,
    pub m_hx: u32,
    pub m_hy: u32,
    pub m_hz: u32,
    pub z0: u32,
    pub zstride: u32,
    pub dst_off: u32,
    pub row_pitch: u32,
    pub slice_pitch: u32,
    pub row_dwords: u32,
    pub depth: u32,
}

pub const PARAMS: u32 = (std::mem::size_of::<Params>() / 4) as u32;

/// A texture to detile on the GPU: its tiled bytes, the tables its subresources index (only
/// the entries its extent reaches) and one `Params` per subresource (D3D12 order).
pub struct Tiled {
    pub raw: Vec<u8>,
    pub tables: Vec<u32>,
    pub subs: Vec<Params>,
}

const HLSL: &str = r#"
ByteAddressBuffer Src : register(t0);
ByteAddressBuffer Tab : register(t1);
RWByteAddressBuffer Dst : register(u0);
cbuffer P : register(b0) {
    uint src_off, src_len, lvl_off, pitch_blocks, ox0, oy0, width, height, bw_log2, bh_log2, bd_log2,
         block_bytes, base_xor, e_log2, t_ox, t_oy, t_oz, t_hx, t_hy, t_hz, m_hx, m_hy, m_hz, z0, zstride,
         dst_off, row_pitch, slice_pitch, row_dwords, depth;
};
uint tab(uint i) { return Tab.Load(i * 4); }
uint tiled(uint x, uint y, uint z) {
    x += ox0;
    y += oy0;
    uint block = (y >> bh_log2) * pitch_blocks + (x >> bw_log2);
    uint xr = tab(t_ox + (x & ((1u << bw_log2) - 1))) ^ tab(t_oy + (y & ((1u << bh_log2) - 1))) ^ tab(t_oz + (z & ((1u << bd_log2) - 1)))
            ^ tab(t_hx + ((x >> bw_log2) & m_hx)) ^ tab(t_hy + ((y >> bh_log2) & m_hy)) ^ tab(t_hz + ((z >> bd_log2) & m_hz));
    return (z >> bd_log2) * zstride + lvl_off + block * block_bytes + (xr ^ base_xor);
}
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= row_dwords) return;
    uint z = z0 + id.z;
    uint e = 1u << e_log2;
    uint b0 = id.x * 4;
    uint v = 0;
    if (e_log2 >= 2) {
        uint o = tiled(b0 >> e_log2, id.y, z);
        if (o + e <= src_len) v = Src.Load(src_off + o + (b0 & (e - 1)));
    } else {
        for (uint k = 0; k < 4; k++) {
            uint el = (b0 + k) >> e_log2;
            if (el >= width) break;
            uint o = tiled(el, id.y, z);
            if (o + e <= src_len) {
                uint a = src_off + o + ((b0 + k) & (e - 1));
                v |= ((Src.Load(a & ~3u) >> ((a & 3) * 8)) & 0xff) << (k * 8);
            }
        }
    }
    Dst.Store(dst_off + id.z * slice_pitch + id.y * row_pitch + b0, v);
}
"#;

/// The plan of detiling a texture (`raw` = its bytes, all slices) of `array` slices (or a
/// volume when `volume`) with elements of `bpe` bytes; `None` when the GPU path cannot take it
/// (the CPU path does).
pub fn plan(c: &Chain, bpe: usize, array: usize, volume: bool, raw: Vec<u8>) -> Option<Tiled> {
    let t = &c.layout;
    let e = 1usize << t.elem_log2;
    if e != bpe || raw.len() >= 1 << 31 || c.levels.is_empty() {
        return None;
    }
    let log2 = |n: usize| n.is_power_of_two().then_some(n.trailing_zeros());
    let (bw, bh, bd) = (log2(t.block_w)?, log2(t.block_h)?, log2(t.block_d)?);
    // element loads are whole dwords from 4 B elements up: their offsets must stay aligned
    if e >= 4 && (t.base_xor | c.slice_bytes | c.levels.iter().fold(0, |a, l| a | l.offset)) & 3 != 0 {
        return None;
    }
    // the tables, the high ones only as far as this texture reaches
    let max_x = c.levels.iter().map(|l| l.origin.0 + l.width).max()?.max(1);
    let max_y = c.levels.iter().map(|l| l.origin.1 + l.height).max()?.max(1);
    let max_z = if volume { c.levels[0].depth } else { array }.max(1);
    let all = t.tables();
    let reach = |k: usize, max: usize, l2: u32| all[k].len().min(((max - 1) >> l2) + 1);
    let lens = [all[0].len(), all[1].len(), all[2].len(), reach(3, max_x, bw), reach(4, max_y, bh), reach(5, max_z, bd)];
    let mut tables = Vec::new();
    let mut at = [0u32; 6];
    for (k, tab) in all.iter().enumerate() {
        if !tab.len().is_power_of_two() {
            return None;
        }
        at[k] = tables.len() as u32;
        tables.extend_from_slice(&tab[..lens[k]]);
    }
    let mask = |k: usize| all[k].len() as u32 - 1;
    let level = |l: &MipLevel| Params {
        lvl_off: l.offset as u32,
        pitch_blocks: l.pitch.div_ceil(t.block_w) as u32,
        ox0: l.origin.0 as u32,
        oy0: l.origin.1 as u32,
        width: l.width as u32,
        height: l.height as u32,
        bw_log2: bw,
        bh_log2: bh,
        bd_log2: bd,
        block_bytes: t.block_bytes as u32,
        base_xor: t.base_xor as u32,
        e_log2: t.elem_log2,
        t_ox: at[0],
        t_oy: at[1],
        t_oz: at[2],
        t_hx: at[3],
        t_hy: at[4],
        t_hz: at[5],
        m_hx: mask(3),
        m_hy: mask(4),
        m_hz: mask(5),
        row_dwords: (l.width * e).div_ceil(4) as u32,
        ..Default::default()
    };
    let mut subs = Vec::new();
    if volume {
        for l in &c.levels {
            subs.push(Params { src_off: 0, src_len: raw.len() as u32, z0: 0, zstride: c.slice_bytes as u32, depth: l.depth as u32, ..level(l) });
        }
    } else {
        for s in 0..array {
            let src_off = s * c.slice_bytes;
            let src_len = c.slice_bytes.min(raw.len().saturating_sub(src_off));
            for l in &c.levels {
                subs.push(Params { src_off: src_off as u32, src_len: src_len as u32, z0: s as u32, zstride: 0, depth: 1, ..level(l) });
            }
        }
    }
    Some(Tiled { raw, tables, subs })
}

pub struct Pipe {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pipe {}
unsafe impl Sync for Pipe {}

/// device (raw pointer) → its detile pipeline, or `None` when it could not be made
static PIPES: Mutex<Option<HashMap<usize, Option<Arc<Pipe>>>>> = Mutex::new(None);

/// GPU detiling is on (`XCL_GPU_DETILE`, default on)
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_DETILE").map_or(true, |v| v != "0"))
}

/// The detile pipeline of `dev`, made once; `None` when GPU detiling is off or failed.
pub unsafe fn pipe(dev: &ID3D12Device) -> Option<Arc<Pipe>> {
    if !enabled() {
        return None;
    }
    let key = windows::core::Interface::as_raw(dev) as usize;
    let mut m = PIPES.lock().unwrap();
    let m = m.get_or_insert_with(Default::default);
    if let Some(p) = m.get(&key) {
        return p.clone();
    }
    let made = make_pipe(dev);
    match &made {
        Ok(_) => crate::xlog_info!("title textures: detiled on the GPU (XCL_GPU_DETILE=0: on the CPU)"),
        Err(e) => crate::xlog_warn!("title textures: GPU detiling unavailable, detiled on the CPU: {e}"),
    }
    let p = made.ok().map(Arc::new);
    m.insert(key, p.clone());
    p
}

unsafe fn make_pipe(dev: &ID3D12Device) -> Result<Pipe, String> {
    let (dxil, _) = super::recomp::dxc::compile_cached(HLSL, "main", "cs_6_0")?;
    let descriptor = |register| D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: register, RegisterSpace: 0 } };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: PARAMS } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: descriptor(0), ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
        D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV, Anonymous: descriptor(1), ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
        D3D12_ROOT_PARAMETER { ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV, Anonymous: descriptor(0), ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
    let mut blob = None;
    D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).map_err(|e| format!("root signature: {e}"))?;
    let blob = blob.ok_or("root signature: no blob")?;
    let bytes = std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize());
    let root: ID3D12RootSignature = dev.CreateRootSignature(0, bytes).map_err(|e| format!("root signature: {e}"))?;
    let pd = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::transmute_copy(&root),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as _, BytecodeLength: dxil.len() },
        ..Default::default()
    };
    let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&pd).map_err(|e| format!("pipeline: {e}"))?;
    Ok(Pipe { root, pso })
}

/// Record the detiles of `t` (its bytes at `raw_at`, its tables at u32 index `tab_at` of the
/// buffer `src` at `src_va`) into the scratch buffer `dst_va`, its subresources at `fps` +
/// `dst_base` with `nrows` rows each. The pipeline and root signature are set by the caller.
pub unsafe fn record(list: &ID3D12GraphicsCommandList, t: &Tiled, raw_at: u32, tab_at: u32, fps: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT], nrows: &[u32], dst_base: u64) {
    for ((p, fp), &rows) in t.subs.iter().zip(fps).zip(nrows) {
        let pitch = fp.Footprint.RowPitch;
        let p = Params {
            src_off: p.src_off + raw_at,
            t_ox: p.t_ox + tab_at,
            t_oy: p.t_oy + tab_at,
            t_oz: p.t_oz + tab_at,
            t_hx: p.t_hx + tab_at,
            t_hy: p.t_hy + tab_at,
            t_hz: p.t_hz + tab_at,
            dst_off: (dst_base + fp.Offset) as u32,
            row_pitch: pitch,
            slice_pitch: pitch * rows,
            ..*p
        };
        let height = p.height.min(rows);
        let depth = p.depth.min(fp.Footprint.Depth);
        if p.row_dwords == 0 || height == 0 || depth == 0 {
            continue;
        }
        list.SetComputeRoot32BitConstants(0, PARAMS, &p as *const Params as *const _, 0);
        list.Dispatch(p.row_dwords.div_ceil(64), height, depth);
    }
}
