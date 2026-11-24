//! The direct lowering (`lower`) written into a DXIL module through `gpu::dxil`'s builder:
//! the per-lane form of a program as a signed container, with no HLSL and no DXC.
//!
//! `Ir` maps one to one onto the builder: blocks, phis (`phi_at` / `add_incoming`), integer
//! ops, compares, `select`, the thread / group / lane ids, `waveActiveBallot`, and raw buffer
//! loads and stores on the ABI's root UAVs (u0 `io`, u1 `res`, u2 `mem`). A phi the SSA
//! construction finds trivial keeps its (identical) incoming values instead of being removed:
//! the builder has no use replacement, and the driver folds such phis.

use super::lower::{self, Bin, Cmp, Ir};
use super::ssa::{Sink, Var};
use crate::gpu::dxil::container::Stage;
use crate::gpu::dxil::module::{BinOp, Block, CastOp, Pred, Value};
use crate::gpu::dxil::shader::{RawUav, Shader};
use crate::gpu::rdna2::Decoded;

struct DxilIr {
    s: Shader,
    io: RawUav,
    res: Option<RawUav>,
    mem: RawUav,
    blocks: Vec<Block>,
}

impl DxilIr {
    fn new(kernel: bool) -> DxilIr {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let io = s.raw_uav(0, 0);
        let res = kernel.then(|| s.raw_uav(1, 0));
        // compute: other waves may write the memory this one reads (as the HLSL path declares it)
        let mem = if kernel { s.raw_uav(2, 0) } else { s.raw_uav_coherent(2, 0) };
        DxilIr { s, io, res, mem, blocks: Vec::new() }
    }

    fn ty(&mut self, var: Var) -> crate::gpu::dxil::module::Ty {
        match var {
            Var::M(_) | Var::Scc => self.s.m.i1(),
            Var::S(_) | Var::V(_) => self.s.m.i32(),
        }
    }

    fn bytes(&mut self, dword: Value) -> Value {
        let four = self.s.m.i32c(4);
        self.s.m.bin(BinOp::Mul, dword, four)
    }
}

impl Sink for DxilIr {
    type V = Value;
    fn phi(&mut self, block: usize, var: Var) -> Value {
        let t = self.ty(var);
        self.s.m.phi_at(self.blocks[block], t)
    }
    fn incoming(&mut self, phi: Value, value: Value, pred: usize) {
        self.s.m.add_incoming(phi, value, self.blocks[pred]);
    }
    fn replace(&mut self, _old: Value, _new: Value) {
        // kept: its incoming values are all `new` (or itself), which is a valid phi
    }
    fn undef(&mut self, var: Var) -> Value {
        let t = self.ty(var);
        self.s.m.undef(t)
    }
}

impl Ir for DxilIr {
    fn new_block(&mut self) -> usize {
        let b = if self.blocks.is_empty() { self.s.m.entry_block() } else { self.s.m.new_block() };
        self.blocks.push(b);
        self.blocks.len() - 1
    }
    fn at(&mut self, block: usize) {
        self.s.m.set_block(self.blocks[block]);
    }
    fn int(&mut self, v: u32) -> Value {
        self.s.m.i32c(v)
    }
    fn boolean(&mut self, v: bool) -> Value {
        self.s.m.i1c(v)
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
        self.s.m.bin(op, a, b)
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
        self.s.m.cmp(p, a, b)
    }
    fn select(&mut self, c: Value, a: Value, b: Value) -> Value {
        self.s.m.select(c, a, b)
    }
    fn zext(&mut self, b: Value) -> Value {
        let t = self.s.m.i32();
        self.s.m.cast(CastOp::ZExt, b, t)
    }
    fn lane(&mut self) -> Value {
        self.s.wave_lane_index()
    }
    fn group_id(&mut self, k: u32) -> Value {
        self.s.group_id(k)
    }
    fn thread_id(&mut self, k: u32) -> Value {
        self.s.thread_id_in_group(k)
    }
    fn ballot(&mut self, c: Value) -> [Value; 2] {
        let w = self.s.wave_ballot(c);
        [w[0], w[1]]
    }
    fn read_lane(&mut self, v: Value, lane: Value) -> Value {
        self.s.wave_read_lane_at(v, lane)
    }
    fn count_bits(&mut self, v: Value) -> Value {
        self.s.bit_count(crate::gpu::dxil::ops::BitCount::Countbits, v)
    }
    fn io_load(&mut self, dword: Value) -> Value {
        let a = self.bytes(dword);
        let io = self.io;
        self.s.raw_load(io, a, 1)[0]
    }
    fn res_store(&mut self, dword: Value, value: Value) {
        let a = self.bytes(dword);
        let res = self.res.expect("res is bound in kernel mode");
        self.s.raw_store(res, a, &[value]);
    }
    fn mem_load(&mut self, addr: Value) -> Value {
        let mem = self.mem;
        self.s.raw_load(mem, addr, 1)[0]
    }
    fn mem_store(&mut self, addr: Value, value: Value) {
        let mem = self.mem;
        self.s.raw_store(mem, addr, &[value]);
    }
    fn br(&mut self, target: usize) {
        self.s.m.br(self.blocks[target]);
    }
    fn br_if(&mut self, c: Value, then: usize, otherwise: usize) {
        self.s.m.cond_br(c, self.blocks[then], self.blocks[otherwise]);
    }
    fn ret(&mut self) {
        self.s.m.ret();
    }
}

/// A kernel-mode program in its per-lane form, as a signed DXIL container (one wave of
/// `WaveSize` = the program's wave size), and the masks whose final value may differ from
/// the console's (`lower::lower_kernel`).
pub fn kernel(insts: &[Decoded], wave64: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (ir, stale) = lower::lower_kernel(insts, wave64, DxilIr::new(true))?;
    let lanes = if wave64 { 64 } else { 32 };
    Ok((ir.s.finish([lanes, 1, 1], Some(lanes)), stale))
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

/// A compute program in its per-lane form, as a signed DXIL container.
pub fn compute(insts: &[Decoded], cs: &super::CsInterface, wave64: bool) -> Result<Vec<u8>, String> {
    let ir = lower::lower_compute(insts, cs, wave64, DxilIr::new(false))?;
    Ok(ir.s.finish(cs.threads, Some(if wave64 { 64 } else { 32 })))
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
}
