//! The direct lowering (`lower`) written into a DXIL module through `gpu::dxil`'s builder:
//! the per-lane form of a program as a signed container, with no HLSL and no DXC.
//!
//! `Ir` maps one to one onto the builder: blocks, phis (`phi_at` / `add_incoming`), integer
//! ops, compares, `select`, the thread / group / lane ids, `waveActiveBallot`, and raw buffer
//! loads and stores on the ABI's root UAVs (u0 `io`, u1 `res`, u2 `mem`). The builder state is
//! #1426's translator state (`gpu::dxil::lower::Lw`, built by `Lw::new` and finished by
//! `Lw::finish`), so its per-entry setup (`setup_lanes_compute`) and its instruction
//! lowering are reachable from here. A phi the SSA
//! construction finds trivial has its uses replaced (`Module::replace_all_uses`); the phi itself
//! stays in its block, unused.

use super::lower::{self, Bin, Cmp, Ir};
use super::ssa::{Sink, Var};
use crate::gpu::dxil::kernel::{Entry, KernelOptions};
use crate::gpu::dxil::lower::Lw;
use crate::gpu::dxil::module::{BinOp, Block, CastOp, Pred, Value};
use crate::gpu::dxil::shader::RawUav;
use crate::gpu::rdna2::Decoded;
use std::collections::BTreeSet;

struct DxilIr {
    lw: Lw,
    io: RawUav,
    res: Option<RawUav>,
    mem: RawUav,
    /// each `Ir` block: where it starts (its phis) and where it ends now (its branch; later than
    /// the start when #1426's lowering of an instruction in it made blocks of its own)
    blocks: Vec<Block>,
    tails: Vec<Block>,
    /// the `Ir` block being written
    cur: usize,
    /// `compute_ids` set up the lanes (`Lw::lane` is the console lane)
    lanes_set: bool,
}

impl DxilIr {
    /// The builder state for `insts` at `entry` (a kernel or a compute program), with one memory
    /// window; a compute program's is coherent, as the HLSL path declares it.
    fn new(insts: &[Decoded], wave64: bool, entry: &Entry) -> DxilIr {
        Self::with(insts, KernelOptions { wave64, split: false, mem_windows: 1, sampler_base: 0 }, entry)
    }

    /// The builder state with the runtime's options `o` (memory windows, sampler registers).
    fn with(insts: &[Decoded], o: KernelOptions, entry: &Entry) -> DxilIr {
        // block leaders for the constant-sampler scan: the first instruction, branch targets
        // and the instruction after a branch
        let mut leaders: BTreeSet<u64> = insts.first().map(|d| d.pc).into_iter().collect();
        for (i, d) in insts.iter().enumerate() {
            if let Some(t) = crate::gpu::rdna2::branch_target(d) {
                leaders.insert(t);
                leaders.extend(insts.get(i + 1).map(|n| n.pc));
            }
        }
        let lw = Lw::new(insts, &leaders, o, entry);
        let io = lw.init.io.expect("the I/O buffer");
        let (res, mem) = (lw.res, lw.mems[0]);
        DxilIr { lw, io, res, mem, blocks: Vec::new(), tails: Vec::new(), cur: 0, lanes_set: false }
    }

    fn ty(&mut self, var: Var) -> crate::gpu::dxil::module::Ty {
        match var {
            Var::M(_) | Var::Scc => self.lw.s.m.i1(),
            Var::S(_) | Var::V(_) => self.lw.s.m.i32(),
        }
    }

    fn bytes(&mut self, dword: Value) -> Value {
        let four = self.lw.s.m.i32c(4);
        self.lw.s.m.bin(BinOp::Mul, dword, four)
    }
}

impl Sink for DxilIr {
    type V = Value;
    fn phi(&mut self, block: usize, var: Var) -> Value {
        let t = self.ty(var);
        self.lw.s.m.phi_at(self.blocks[block], t)
    }
    fn incoming(&mut self, phi: Value, value: Value, pred: usize) {
        self.lw.s.m.add_incoming(phi, value, self.tails[pred]);
    }
    fn replace(&mut self, old: Value, new: Value) {
        // every use of the trivial phi becomes `new`; the phi stays in its block, unused
        self.lw.s.m.replace_all_uses(old, new);
    }
    fn undef(&mut self, var: Var) -> Value {
        let t = self.ty(var);
        self.lw.s.m.undef(t)
    }
}

impl Ir for DxilIr {
    fn new_block(&mut self) -> usize {
        let b = if self.blocks.is_empty() { self.lw.s.m.entry_block() } else { self.lw.s.m.new_block() };
        self.blocks.push(b);
        self.tails.push(b);
        self.blocks.len() - 1
    }
    fn at(&mut self, block: usize) {
        self.cur = block;
        self.lw.s.m.set_block(self.tails[block]);
    }
    fn int(&mut self, v: u32) -> Value {
        self.lw.s.m.i32c(v)
    }
    fn boolean(&mut self, v: bool) -> Value {
        self.lw.s.m.i1c(v)
    }
    fn bin(&mut self, op: Bin, a: Value, b: Value) -> Value {
        let op = match op {
            Bin::Add => BinOp::Add,
            Bin::Sub => BinOp::Sub,
            Bin::Mul => BinOp::Mul,
            Bin::And => BinOp::And,
            Bin::Or => BinOp::Or,
            Bin::Xor => BinOp::Xor,
            Bin::Shl => BinOp::Shl,
            Bin::LShr => BinOp::LShr,
            Bin::AShr => BinOp::AShr,
        };
        self.lw.s.m.bin(op, a, b)
    }
    fn icmp(&mut self, c: Cmp, a: Value, b: Value) -> Value {
        let p = match c {
            Cmp::Eq => Pred::Eq,
            Cmp::Ne => Pred::Ne,
            Cmp::Ult => Pred::ULt,
            Cmp::Ule => Pred::ULe,
            Cmp::Ugt => Pred::UGt,
            Cmp::Uge => Pred::UGe,
            Cmp::Slt => Pred::SLt,
            Cmp::Sle => Pred::SLe,
            Cmp::Sgt => Pred::SGt,
            Cmp::Sge => Pred::SGe,
        };
        self.lw.s.m.cmp(p, a, b)
    }
    fn select(&mut self, c: Value, a: Value, b: Value) -> Value {
        self.lw.s.m.select(c, a, b)
    }
    fn zext(&mut self, b: Value) -> Value {
        let t = self.lw.s.m.i32();
        self.lw.s.m.cast(CastOp::ZExt, b, t)
    }
    fn lane(&mut self) -> Value {
        if self.lanes_set {
            self.lw.lane
        } else {
            self.lw.s.wave_lane_index()
        }
    }
    fn group_id(&mut self, k: u32) -> Value {
        self.lw.s.group_id(k)
    }
    fn compute_ids(&mut self, threads: [u32; 3]) -> (Value, [Value; 3]) {
        // #1426's compute entry: 2-D / 3-D groups numbered as their waves arrive, the counter
        // in the program's top LDS word when it has LDS
        let lds = self.lw.lds.is_some();
        let ids = self.lw.setup_lanes_compute(threads, lds);
        self.lanes_set = true;
        ids
    }
    fn ballot(&mut self, c: Value) -> [Value; 2] {
        let w = self.lw.s.wave_ballot(c);
        [w[0], w[1]]
    }
    // one hardware wave per console wave (native wave64, or wave32): the half-local forms are
    // the same operations
    fn mem_windows(&self) -> u32 {
        self.lw.mems.len() as u32
    }
    fn ballot_half(&mut self, c: Value) -> [Value; 2] {
        self.ballot(c)
    }
    /// #1426's semantics of `d` (`Lw::inst_guarded`) in the current block: the inputs are written
    /// into `Lw`'s SSA there first, so it reads them and nothing from other blocks (which its SSA
    /// doesn't know). A register it touches that is neither an input nor an output refuses the
    /// instruction: `Lw` would read that register's value at the program's start. Blocks it makes
    /// (a guarded access, the memory window switch) end the `Ir` block in the last of them.
    fn foreign(&mut self, d: &Decoded, guard: Value, inputs: &[(Var, Value)], outputs: &[Var]) -> Result<Vec<Value>, String> {
        use crate::gpu::dxil::lower::{VAR_BUDGET, VAR_SCC, VAR_VGPR};
        let var = |v: Var| match v {
            Var::S(r) => r as u32,
            Var::V(n) => VAR_VGPR + n as u32,
            Var::Scc => VAR_SCC,
            Var::M(_) => unreachable!("a lane-bit view is not a register"),
        };
        for &(v, x) in inputs {
            self.lw.wr(var(v), x);
        }
        // the registers `Lw` touches for `d` alone
        let before = std::mem::take(&mut self.lw.init.touched);
        self.lw.pc = d.pc;
        let r = self.lw.inst_guarded(d, guard, None);
        let used = std::mem::replace(&mut self.lw.init.touched, before);
        self.lw.init.touched.extend(&used);
        r?;
        let known: BTreeSet<u32> = inputs.iter().map(|&(v, _)| var(v)).chain(outputs.iter().map(|&v| var(v))).collect();
        if let Some(r) = used.iter().find(|&&r| r < VAR_BUDGET && !known.contains(&r)) {
            return Err(format!("it reads register {r} the per-lane form did not pass"));
        }
        let tail = self.lw.cur();
        self.tails[self.cur] = tail;
        Ok(outputs.iter().map(|&v| self.lw.rd(var(v))).collect())
    }
    fn read_lane(&mut self, v: Value, lane: Value) -> Value {
        self.lw.s.wave_read_lane_at(v, lane)
    }
    fn read_lane_half(&mut self, v: Value, lane: Value) -> Value {
        self.read_lane(v, lane)
    }
    fn count_bits(&mut self, v: Value) -> Value {
        self.lw.s.bit_count(crate::gpu::dxil::ops::BitCount::Countbits, v)
    }
    fn io_load(&mut self, dword: Value) -> Value {
        let a = self.bytes(dword);
        let io = self.io;
        self.lw.s.raw_load(io, a, 1)[0]
    }
    fn res_store(&mut self, dword: Value, value: Value) {
        let a = self.bytes(dword);
        let res = self.res.expect("res is bound in kernel mode");
        self.lw.s.raw_store(res, a, &[value]);
    }
    fn mem_load(&mut self, addr: Value) -> Value {
        let mem = self.mem;
        self.lw.s.raw_load(mem, addr, 1)[0]
    }
    fn mem_store(&mut self, addr: Value, value: Value) {
        let mem = self.mem;
        self.lw.s.raw_store(mem, addr, &[value]);
    }
    fn br(&mut self, target: usize) {
        self.lw.s.m.br(self.blocks[target]);
    }
    fn br_if(&mut self, c: Value, then: usize, otherwise: usize) {
        self.lw.s.m.cond_br(c, self.blocks[then], self.blocks[otherwise]);
    }
    fn ret(&mut self) {
        self.lw.s.m.ret();
    }
}

/// A kernel-mode program in its per-lane form, as a signed DXIL container (one wave of
/// `WaveSize` = the program's wave size), and the masks whose final value may differ from
/// the console's (`lower::lower_kernel`).
pub fn kernel(insts: &[Decoded], wave64: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (ir, stale) = lower::lower_kernel(insts, wave64, DxilIr::new(insts, wave64, &Entry::Kernel))?;
    let lanes = if wave64 { 64 } else { 32 };
    Ok((ir.lw.finish([lanes, 1, 1], lanes)?.dxil, stale))
}

/// `compute` from the program's code (decoded, far jumps resolved).
pub fn compute_code(code: &[u8], cs: &super::CsInterface, wave64: bool) -> Result<Vec<u8>, String> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = crate::gpu::rdna2::decode(&code[off..], off as u64).map_err(|e| format!("cannot decode ({e:?})"))?;
        off += d.len;
        insts.push(d);
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    compute(&insts, cs, wave64)
}

/// Survey: every reason the per-lane form refuses a compute program (`lower::survey_compute`).
pub fn survey_compute_code(code: &[u8], cs: &super::CsInterface, wave64: bool) -> Vec<String> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        match crate::gpu::rdna2::decode(&code[off..], off as u64) {
            Ok(d) => {
                off += d.len;
                insts.push(d);
            }
            Err(e) => return vec![format!("cannot decode ({e:?})")],
        }
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    lower::survey_compute(&insts, cs, wave64, DxilIr::new(&insts, wave64, &Entry::Compute(cs.clone())))
}

/// A compute program in its per-lane form, as a signed DXIL container.
pub fn compute(insts: &[Decoded], cs: &super::CsInterface, wave64: bool) -> Result<Vec<u8>, String> {
    let ir = lower::lower_compute(insts, cs, wave64, DxilIr::new(insts, wave64, &Entry::Compute(cs.clone())))?;
    Ok(ir.lw.finish(cs.threads, if wave64 { 64 } else { 32 })?.dxil)
}


/// `XCL_PER_LANE=1` (with `XCL_DIRECT_DXIL`): compute programs try the per-lane form first, then
/// #1426's per-instruction form, then the HLSL path.
pub fn enabled() -> bool {
    std::env::var("XCL_PER_LANE").is_ok_and(|v| v == "1")
}

/// A compute program in its per-lane form with the runtime's options `o` (`pso::direct_compute`):
/// the container and its static samplers. The split wave64 form is not lowered yet.
pub fn compute_with(code: &[u8], cs: &super::CsInterface, o: KernelOptions) -> Result<crate::gpu::dxil::kernel::Kernel, String> {
    if o.split {
        return Err("the split wave64 form".into());
    }
    if o.mem_windows & super::HEAP_WINDOWS != 0 {
        return Err("heap memory windows".into());
    }
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = crate::gpu::rdna2::decode(&code[off..], off as u64).map_err(|e| format!("cannot decode ({e:?})"))?;
        off += d.len;
        insts.push(d);
    }
    crate::gpu::rdna2::resolve_far_jumps(&mut insts);
    // as `kernel::translate_entry`: a thread with a large stack, a panic is a refusal
    let cs = cs.clone();
    std::thread::Builder::new()
        .name("xcl-per-lane".into())
        .stack_size(256 << 20)
        .spawn(move || {
            let ir = lower::lower_compute(&insts, &cs, o.wave64, DxilIr::with(&insts, o, &Entry::Compute(cs.clone())))?;
            ir.lw.finish(cs.threads, if o.wave64 { 64 } else { 32 })
        })
        .map_err(|e| format!("per-lane thread: {e}"))?
        .join()
        .map_err(|e| {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            format!("the per-lane lowering panicked: {msg}")
        })?
}

#[cfg(test)]
mod tests {
    use super::super::lower::tests::*;
    use super::super::{dxc, runner, translate, Mode, Options};

    fn gpu() -> bool {
        if runner::adapter().is_err() {
            runner::skip_test("no GPU");
            return false;
        }
        true
    }

    /// The f16 and packed programs of `lower`'s tests, as `f32_programs_on_the_gpu`.
    #[test]
    fn f16_programs_on_the_gpu() {
        programs_on_the_gpu(&[("f16 arithmetic", F16_ARITHMETIC, check_f16_arithmetic, &[]), ("f16 packed", F16_PACKED, check_f16_packed, &[])]);
    }

    /// The per-lane form as real DXIL on the GPU agrees with the lowering run on the CPU (and so
    /// with the HLSL path, which `lower`'s tests compare it with): registers and memory.
    #[test]
    fn per_lane_kernels_run_on_the_gpu() {
        if !gpu() {
            return;
        }
        for (name, lines) in [("straight", STRAIGHT), ("skipped if", SKIPPED_IF), ("leaving loop", LEAVING_LOOP), ("uniform loop", UNIFORM_LOOP), ("memory", MEMORY)] {
            let insts = decode(lines);
            let (dxil, stale) = super::kernel(&insts, false).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (s, v) = state(32);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (cs, cv, cmem) = run_mem(lines, false, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let mut sregs: Vec<usize> = vec![106, 126];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = crate::gpu::rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(gpu.sgpr[r], cs[r], "{name}: s{r}");
            }
            for l in 0..32 {
                for &r in &vregs {
                    assert_eq!(gpu.vgpr[l][r], cv[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
    }

    /// The f32 programs of `lower`'s tests on the GPU: the HLSL path gives the expected
    /// registers (`check_f32_*`, wave32), and the per-lane DXIL gives the HLSL path's, register
    /// for register (rcp within `f32_close`: HLSL's `rcp` is a fast fdiv). While the per-lane
    /// form refuses a program, that half is SKIPPED with the refusal.
    #[test]
    fn f32_programs_on_the_gpu() {
        programs_on_the_gpu(&[
            ("f32 arithmetic", F32_ARITHMETIC, check_f32_arithmetic, &[]),
            ("f32 compares", F32_COMPARES, check_f32_compares, &[]),
            ("f32 conversions", F32_CONVERSIONS, check_f32_conversions, &[]),
            ("f32 rounding", F32_ROUNDING, check_f32_rounding, &[]),
            ("f32 transcendentals", F32_TRANSCENDENTALS, check_f32_transcendentals, &[3, 4]),
            ("f32 three-source", F32_THREE_SOURCE, check_f32_three_source, &[]),
        ]);
    }

    /// A program of `lower`'s tests, the check of its registers on the HLSL path (wave32), and
    /// the VGPRs the per-lane form need only match within `f32_close`.
    type Program = (&'static str, &'static [I], fn(&[u32], &[Vec<u32>]), &'static [usize]);

    /// `f32_programs_on_the_gpu`'s two halves, for any programs.
    fn programs_on_the_gpu(programs: &[Program]) {
        if !gpu() {
            return;
        }
        if dxc::location().is_err() {
            runner::skip_test("no DXC");
            return;
        }
        let wave64 = runner::wave64_supported() == Ok(true);
        for &(name, lines, check, approx) in programs {
            for w64 in [false, true] {
                let lanes = if w64 { 64 } else { 32 };
                if w64 && !wave64 {
                    runner::skip_test(&format!("{name}, wave64: no wave64"));
                    continue;
                }
                let (s, v) = state(lanes);
                let mut st = runner::KernelState::default();
                st.sgpr.copy_from_slice(&s);
                for (l, regs) in v.iter().enumerate() {
                    st.vgpr[l][..regs.len()].copy_from_slice(regs);
                }
                let bytes: Vec<u8> = lines.iter().flat_map(encode_line).collect();
                let t = translate(&bytes, &Options { wave64: w64, ..Default::default() }).unwrap();
                assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
                let hlsl = runner::run(&dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap(), &st, &[]).unwrap();
                if !w64 {
                    check(&hlsl.sgpr, &hlsl.vgpr[..32]);
                }
                let insts = decode(lines);
                let (dxil, stale) = match super::kernel(&insts, w64) {
                    Ok(k) => k,
                    Err(e) => {
                        runner::skip_test(&format!("{name}, wave{lanes}, per-lane form: refused ({e})"));
                        continue;
                    }
                };
                let direct = runner::run(&dxil, &st, &[]).unwrap();
                let mut sregs: Vec<usize> = vec![106, 126];
                let mut vregs: Vec<usize> = Vec::new();
                for d in &insts {
                    let (r, w) = crate::gpu::rdna2::sgpr_uses(d);
                    sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                    let (r, w) = crate::gpu::rdna2::vgpr_uses(d);
                    vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
                }
                if w64 {
                    sregs.extend([107, 127]);
                }
                let stale_at = |r: usize| stale.contains(&(r as u8)) || (w64 && r % 2 == 1 && stale.contains(&(r as u8 - 1)));
                for &r in sregs.iter().filter(|&&r| !stale_at(r)) {
                    assert_eq!(direct.sgpr[r], hlsl.sgpr[r], "{name}, wave{lanes}: s{r}");
                }
                for l in 0..lanes {
                    for &r in &vregs {
                        let (d, h) = (direct.vgpr[l][r], hlsl.vgpr[l][r]);
                        let same = if approx.contains(&r) { f32_close(d, h) } else { d == h };
                        assert!(same, "{name}, wave{lanes}: lane {l} v{r}: {d:#x} against {h:#x}");
                    }
                }
            }
        }
    }

    /// 2-D and 3-D groups in the per-lane form: the threads numbered as the waves arrive
    /// (`Lw::setup_lanes_compute`), each wave's lanes the console's consecutive threads, as the
    /// lowering run on the CPU has them.
    #[test]
    fn a_per_lane_2d_or_3d_group_runs_on_the_gpu() {
        if !gpu() {
            return;
        }
        for wave64 in [false, true] {
            if wave64 && runner::wave64_supported() != Ok(true) {
                runner::skip_test("no wave64");
                return;
            }
            for threads in ID_SHAPES {
                let lines = ids_program(threads, wave64);
                let cs = ids_cs(threads);
                let dxil = super::compute(&decode(&lines), &cs, wave64).unwrap();
                let mem = vec![0u32; 256];
                let gpu = runner::dispatch(&dxil, [2, 1, 1], &[100, 0], &mem).unwrap();
                let cpu = run_compute_waves(&lines, &cs, wave64, 2, &[100, 0], &mem);
                assert_eq!(&gpu[..128], &cpu[..128], "{threads:?} wave64 {wave64}");
            }
        }
    }

    #[test]
    fn a_per_lane_compute_program_runs_on_the_gpu() {
        if !gpu() {
            return;
        }
        let cs = compute_cs();
        let dxil = super::compute(&decode(COMPUTE), &cs, false).unwrap();
        let mem = vec![0u32; 256];
        let gpu = runner::dispatch(&dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&gpu[..cpu.len()], &cpu[..]);
        // and the HLSL path, for good measure
        if dxc::location().is_ok() {
            let bytes: Vec<u8> = COMPUTE.iter().flat_map(|(n, e, f)| crate::gpu::rdna2::encode(n, e, f, None).unwrap()).collect();
            let t = translate(&bytes, &Options { mode: Mode::Compute(cs.clone()), wave64: false, ..Default::default() }).unwrap();
            let hlsl = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            assert_eq!(runner::dispatch(&hlsl, [3, 1, 1], &[100, 0], &mem).unwrap(), gpu);
        }
    }

    /// With the runtime's four memory windows (`compute_with`), the memory instructions go
    /// through #1426's window switch (blocks of its own in the `Ir` block) and the program
    /// still gives the CPU lowering's memory; the split wave64 form is refused.
    #[test]
    fn a_per_lane_program_with_four_memory_windows_runs_on_the_gpu() {
        let cs = compute_cs();
        let bytes: Vec<u8> = COMPUTE.iter().flat_map(|(n, e, f)| crate::gpu::rdna2::encode(n, e, f, None).unwrap()).collect();
        let o = super::KernelOptions { wave64: false, split: false, mem_windows: 4, sampler_base: 0 };
        let split = super::compute_with(&bytes, &cs, super::KernelOptions { wave64: true, split: true, ..o }).err().unwrap_or_default();
        assert!(split.contains("split"), "{split}");
        if !gpu() {
            return;
        }
        let k = super::compute_with(&bytes, &cs, o).unwrap();
        let mem = vec![0u32; 256];
        let out = runner::dispatch(&k.dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&out[..cpu.len()], &cpu[..]);
    }
}
