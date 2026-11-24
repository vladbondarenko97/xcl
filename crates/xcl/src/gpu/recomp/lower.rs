//! Lowering of console programs to DXIL-shaped IR, for the direct backend (`XCL_RECOMP_DXIL`).
//!
//! The console CFG is kept: blocks become blocks, and every register dword is an SSA value
//! (`ssa`). EXEC is not a number here but each lane's own bit (`i1`): a mask branch
//! (`s_cbranch_execz`, `vccz`, SCC of a mask op) is a branch of each lane on its bit, and a
//! vector instruction keeps a guard (`select(bit, new, old)`) only where `lane_masks` could not
//! prove the bit set. An SGPR therefore has two views: its number (`Var::S`) and, when it holds
//! a mask, the lane's bit of it (`Var::M`). Writing one view leaves a marker in the other – `undef`
//! – so a number read where only a mask was written (or the reverse, across a join) is found once
//! every phi is complete, and the program is refused (it stays on the HLSL path). A bit read where
//! only a number was written is the lane's bit of that number. At the end of the program every
//! lane is back, so a mask stored as a number there is the wave's ballot of the bits.
//!
//! Kernel mode only for now (the ABI of `runner`): registers come from `io` and go back to `res`.
//! The IR is written through `Ir`, the part of the DXIL module builder this needs.

use std::collections::HashSet;

use super::lane_masks::{self, Edge, Summary};
use super::ssa::{Sink, Ssa, Var};
use super::{KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS};
use crate::gpu::rdna2::{self, Decoded, Variant};

mod wave;

type R<T> = Result<T, String>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bin {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cmp {
    Eq,
    Ne,
    Ult,
    Ule,
    Ugt,
    Uge,
    Slt,
    Sle,
    Sgt,
    Sge,
}

/// f32 operations, on dword bits in and out (`Ir::fbin`). `Min` / `Max` are DXIL's `FMin` /
/// `FMax` (HLSL `min` / `max`): a NaN operand gives the other one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FBin {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// Conversions between i32 / u32 and f32 (round to nearest even; out of range is the caller's)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cvt {
    SiToF,
    UiToF,
    FToSi,
    FToUi,
}

/// Ordered f32 compares (false when either operand is NaN)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FCmp {
    Lt,
    Eq,
    Le,
    Gt,
    Ge,
}

/// `Abs` and `Neg` as HLSL's `abs` / `-x`; `Sat` is `saturate` (NaN gives 0); `Floor` is
/// `floor`; `F16ToF32` is `f16tof32` of the dword's low 16 bits; the others are the HLSL
/// intrinsics of their names (`Exp` / `Log` base 2, `RoundNe` HLSL's `round`)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FUn {
    Abs,
    Neg,
    Sat,
    Floor,
    F16ToF32,
    Ceil,
    Trunc,
    RoundNe,
    Sqrt,
    Rsq,
    Exp,
    Log,
    Sin,
    Cos,
}

/// What the lowering needs from the DXIL module builder. Blocks are numbered in the order
/// `new_block` creates them, the first being the entry; `And` / `Or` / `Xor` take `i1` or `i32`.
pub trait Ir: Sink {
    fn new_block(&mut self) -> usize;
    fn at(&mut self, block: usize);
    fn int(&mut self, v: u32) -> Self::V;
    fn boolean(&mut self, v: bool) -> Self::V;
    fn bin(&mut self, op: Bin, a: Self::V, b: Self::V) -> Self::V;
    fn icmp(&mut self, c: Cmp, a: Self::V, b: Self::V) -> Self::V;
    fn select(&mut self, c: Self::V, a: Self::V, b: Self::V) -> Self::V;
    /// `i1` → `i32` (0 / 1)
    fn zext(&mut self, b: Self::V) -> Self::V;
    /// the console lane of this thread
    fn lane(&mut self) -> Self::V;
    /// compute: the group's id and the thread's id in its group, component k (x, y, z)
    fn group_id(&mut self, k: u32) -> Self::V;
    fn thread_id(&mut self, k: u32) -> Self::V;
    /// the wave's ballot of `c`: lanes 0–31, 32–63
    fn ballot(&mut self, c: Self::V) -> [Self::V; 2];
    /// `ballot` where only the word of this lane's 32-lane half is used (the other may be
    /// anything): a wave64 split into two 32-lane hardware waves needs no exchange for it
    fn ballot_half(&mut self, c: Self::V) -> [Self::V; 2];
    /// `v` as lane `lane` holds it (`WaveReadLaneAt`); `lane` may differ between lanes and is
    /// always one running this instruction too
    fn read_lane(&mut self, v: Self::V, lane: Self::V) -> Self::V;
    /// `read_lane` of a lane in this lane's 32-lane half (DPP, permlane, swizzle)
    fn read_lane_half(&mut self, v: Self::V, lane: Self::V) -> Self::V;
    /// the number of bits set in `v` (`countbits`)
    fn count_bits(&mut self, v: Self::V) -> Self::V;
    /// f32 arithmetic on the dwords' bits, each step rounded on its own (no contraction)
    fn fbin(&mut self, op: FBin, a: Self::V, b: Self::V) -> Self::V;
    fn fun(&mut self, op: FUn, a: Self::V) -> Self::V;
    /// HLSL's `mad` (DXIL `FMad`): fused or not, as the driver chooses
    fn fmad(&mut self, a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    fn cvt(&mut self, c: Cvt, a: Self::V) -> Self::V;
    /// `i1`: an ordered compare of the dwords as f32; whether a dword is a NaN
    fn fcmp(&mut self, c: FCmp, a: Self::V, b: Self::V) -> Self::V;
    fn is_nan(&mut self, a: Self::V) -> Self::V;
    fn io_load(&mut self, dword: Self::V) -> Self::V;
    fn res_store(&mut self, dword: Self::V, value: Self::V);
    /// the dword of the flat memory (`mem`, u2) at a byte address (a multiple of 4); out of
    /// range reads 0 and writes nothing, as D3D12 raw buffers do
    fn mem_load(&mut self, addr: Self::V) -> Self::V;
    fn mem_store(&mut self, addr: Self::V, value: Self::V);
    fn br(&mut self, target: usize);
    fn br_if(&mut self, c: Self::V, then: usize, otherwise: usize);
    fn ret(&mut self);
}

const EXEC: u8 = 126;

/// The f32 arithmetic `f32_arith` lowers.
const F32_ARITH: &[&str] = &[
    "V_ADD_F32", "V_SUB_F32", "V_SUBREV_F32", "V_MUL_F32", "V_MIN_F32", "V_MAX_F32", "V_FMA_F32", "V_MAD_F32", "V_MAC_F32", "V_FMAC_F32",
    "V_CVT_F32_I32", "V_CVT_F32_U32", "V_CVT_U32_F32", "V_CVT_I32_F32", "V_CVT_RPI_I32_F32", "V_CVT_FLR_I32_F32", "V_CVT_F32_F16", "V_CVT_F16_F32",
    "V_RCP_F32", "V_RCP_IFLAG_F32", "V_RSQ_F32", "V_SQRT_F32", "V_EXP_F32", "V_LOG_F32", "V_SIN_F32", "V_COS_F32",
    "V_FLOOR_F32", "V_CEIL_F32", "V_TRUNC_F32", "V_RNDNE_F32", "V_FRACT_F32",
];
const VCC: u8 = 106;

/// The program's stage: where its registers come from and go.
#[derive(Clone, Copy)]
pub enum Stage<'a> {
    /// `runner`'s ABI: the whole register file from `io`, back to `res`
    Kernel,
    /// a compute program: user SGPRs from `io`, then the group ids and TG_SIZE it enables,
    /// thread ids in v0–v2; it ends without storing registers
    Compute(&'a super::CsInterface),
}

struct Lower<'a, I: Ir> {
    ir: I,
    ssa: Ssa<I::V>,
    sum: &'a Summary,
    guarded: HashSet<u64>,
    wave64: bool,
    /// the markers: a number where a mask was written, a bit where a number was
    num_marker: I::V,
    bit_marker: I::V,
    /// values read as numbers / as mask bits, checked once every phi is complete
    num_uses: Vec<I::V>,
    bit_uses: Vec<I::V>,
    lane: Option<I::V>,
    /// per IR block: not every lane that reached one of the program's lane branches gets here
    /// before they meet again (a loop that lanes leave one by one, an arm some lanes skip)
    partial: Vec<bool>,
    /// the IR block each mask bit was written in, and the mask bits read (block, value)
    def_block: std::collections::HashMap<I::V, usize>,
    lane_uses: Vec<(usize, I::V)>,
    /// the program writes MODE (`s_setreg*` of HW_REG_MODE)
    sets_mode: bool,
    /// a DPP instruction's first source, read from its source lane, and its write enable
    dpp_src: Option<I::V>,
    dpp_en: Option<I::V>,
}

/// A vector memory store: a side effect inactive lanes must not have.
fn is_store(d: &Decoded) -> bool {
    let n = d.inst.name;
    (n.starts_with("GLOBAL_STORE_") || n.starts_with("FLAT_STORE_")) && matches!(d.enc.name, "ENC_FLAT_GLBL" | "ENC_FLAT")
}

/// IR block of analysis block b (block 0 is the prologue).
fn blk(b: usize) -> usize {
    b + 1
}

impl<'a, I: Ir> Lower<'a, I> {
    fn lane(&mut self) -> I::V {
        *self.lane.get_or_insert_with(|| self.ir.lane())
    }

    fn read(&mut self, b: usize, var: Var) -> I::V {
        self.ssa.read(&mut self.ir, b, var)
    }

    fn read_num(&mut self, b: usize, r: u8) -> R<I::V> {
        let v = self.read(b, Var::S(r));
        if self.ssa.resolve_value(v) == self.num_marker {
            return Err(format!("s{r}: a mask read as a number"));
        }
        self.num_uses.push(v);
        self.lane_uses.push((b, v));
        Ok(v)
    }

    /// The lane's bit of the mask in s<r> (wave64: the pair from s<r>).
    fn read_bit(&mut self, b: usize, r: u8) -> R<I::V> {
        let m = self.read(b, Var::M(r));
        if self.ssa.resolve_value(m) != self.bit_marker {
            self.bit_uses.push(m);
            if r != EXEC {
                self.lane_uses.push((b, m));
            }
            return Ok(m);
        }
        // only numbers written: the lane's bit of them
        let lane = self.lane();
        let word = if self.wave64 {
            let (lo, hi) = (self.read_num(b, r)?, self.read_num(b, r + 1)?);
            let c32 = self.ir.int(32);
            let upper = self.ir.icmp(Cmp::Uge, lane, c32);
            self.ir.select(upper, hi, lo)
        } else {
            self.read_num(b, r)?
        };
        let c31 = self.ir.int(31);
        let sh = self.ir.bin(Bin::And, lane, c31);
        let shifted = self.ir.bin(Bin::LShr, word, sh);
        let one = self.ir.int(1);
        let bit = self.ir.bin(Bin::And, shifted, one);
        let zero = self.ir.int(0);
        Ok(self.ir.icmp(Cmp::Ne, bit, zero))
    }

    /// SCC, written and read like an SGPR number: a lane that skipped the block writing it
    /// keeps its own older value (`stale`).
    fn write_scc(&mut self, b: usize, v: I::V) {
        self.def_block.entry(v).or_insert(b);
        self.ssa.write(b, Var::Scc, v);
    }

    fn read_scc(&mut self, b: usize) -> I::V {
        let v = self.read(b, Var::Scc);
        self.lane_uses.push((b, v));
        v
    }

    fn write_num(&mut self, b: usize, r: u8, v: I::V) {
        self.def_block.entry(v).or_insert(b);
        self.ssa.write(b, Var::S(r), v);
        self.ssa.write(b, Var::M(r), self.bit_marker);
        if self.wave64 && r > 0 {
            // a mask in the pair from s<r - 1> is no longer that mask
            self.ssa.write(b, Var::M(r - 1), self.bit_marker);
        }
    }

    fn write_bit(&mut self, b: usize, r: u8, bit: I::V) {
        if r != EXEC {
            self.def_block.entry(bit).or_insert(b);
        }
        self.ssa.write(b, Var::M(r), bit);
        self.ssa.write(b, Var::S(r), self.num_marker);
        if self.wave64 {
            self.ssa.write(b, Var::S(r + 1), self.num_marker);
            self.ssa.write(b, Var::M(r + 1), self.bit_marker);
            if r > 0 {
                self.ssa.write(b, Var::M(r - 1), self.bit_marker);
            }
        }
    }

    /// Whether s<r> holds a mask here (its bit view is not the marker).
    fn holds_mask(&mut self, b: usize, r: u8) -> bool {
        let m = self.read(b, Var::M(r));
        self.ssa.resolve_value(m) != self.bit_marker
    }

    /// A scalar or vector source operand as a number.
    fn src(&mut self, b: usize, d: &Decoded, code: u64) -> R<I::V> {
        if let (233 | 234 | 250, Some(v)) = (code, self.dpp_src) {
            return Ok(v);
        }
        Ok(match code {
            0..=105 | 106 | 107 | 124 | 126 | 127 => self.read_num(b, code as u8)?,
            128..=192 => self.ir.int((code - 128) as u32),
            193..=208 => self.ir.int((-((code - 192) as i32)) as u32),
            240..=247 => self.ir.int([0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][(code - 240) as usize].to_bits()),
            248 => self.ir.int(0x3E22_F983),
            253 => {
                let s = self.read_scc(b);
                self.ir.zext(s)
            }
            255 => self.ir.int(d.literal.ok_or("missing literal")?),
            256..=511 => self.read(b, Var::V((code - 256) as u16)),
            _ => return Err(format!("source code {code}")),
        })
    }

    /// A scalar source operand as a mask's lane bit: a register, all lanes or none.
    fn src_bit(&mut self, b: usize, code: u64) -> R<I::V> {
        match code {
            0..=105 | 106 | 107 | 126 | 127 => self.read_bit(b, code as u8),
            128 => Ok(self.ir.boolean(false)),
            193 => Ok(self.ir.boolean(true)),
            _ => Err(format!("mask source code {code}")),
        }
    }

    fn write_v(&mut self, b: usize, d: &Decoded, r: u16, v: I::V) -> R<()> {
        let v = match self.dpp_en.take() {
            Some(en) => {
                let old = self.read(b, Var::V(r));
                self.ir.select(en, v, old)
            }
            None => v,
        };
        let v = if self.guarded.contains(&d.pc) {
            let old = self.read(b, Var::V(r));
            let e = self.read_bit(b, EXEC)?;
            self.ir.select(e, v, old)
        } else {
            v
        };
        self.ssa.write(b, Var::V(r), v);
        Ok(())
    }

    fn not(&mut self, a: I::V) -> I::V {
        let t = self.ir.boolean(true);
        self.ir.bin(Bin::Xor, a, t)
    }

    /// A bitwise mask op of SALU / saveexec on lane bits.
    fn bit_op(&mut self, op: &str, a: I::V, e: I::V) -> R<I::V> {
        Ok(match op {
            "AND" => self.ir.bin(Bin::And, a, e),
            "OR" => self.ir.bin(Bin::Or, a, e),
            "XOR" => self.ir.bin(Bin::Xor, a, e),
            "ANDN2" => {
                let ne = self.not(e);
                self.ir.bin(Bin::And, a, ne)
            }
            "ORN2" => {
                let ne = self.not(e);
                self.ir.bin(Bin::Or, a, ne)
            }
            "ANDN1" => {
                let na = self.not(a);
                self.ir.bin(Bin::And, na, e)
            }
            "ORN1" => {
                let na = self.not(a);
                self.ir.bin(Bin::Or, na, e)
            }
            "NAND" | "NOR" | "XNOR" => {
                let inner = match op {
                    "NAND" => Bin::And,
                    "NOR" => Bin::Or,
                    _ => Bin::Xor,
                };
                let x = self.ir.bin(inner, a, e);
                self.not(x)
            }
            _ => return Err(format!("mask op {op}")),
        })
    }

    fn salu(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f);
        let (s0, s1, sd) = (fv("SSRC0").unwrap_or(0), fv("SSRC1").unwrap_or(0), fv("SDST").unwrap_or(0) as u8);
        if self.wave_salu(b, d)? {
            return Ok(());
        }
        let width = if self.wave64 { "_B64" } else { "_B32" };
        // mask-width ops: on lane bits when they write EXEC, are a saveexec, or read a mask
        if let Some(base) = name.strip_suffix(width) {
            let op = base.trim_start_matches("S_");
            let saveexec = op.strip_suffix("_SAVEEXEC").or_else(|| op.strip_suffix("_WREXEC"));
            // SOP1 has no SSRC1: only the sources the encoding has
            let srcs: Vec<u64> = std::iter::once(s0).chain(d.enc.field("SSRC1").map(|_| s1)).collect();
            let reads_mask = srcs.iter().any(|&c| c <= 127 && c != 124 && self.holds_mask(b, c as u8));
            let bitwise = matches!(op, "MOV" | "NOT" | "AND" | "OR" | "XOR" | "ANDN2" | "ORN2" | "NAND" | "NOR" | "XNOR");
            if let Some(op) = saveexec {
                let a = self.src_bit(b, s0)?;
                let old = self.read_bit(b, EXEC)?;
                let new = self.bit_op(op, a, old)?;
                self.write_bit(b, EXEC, new);
                if sd != 125 {
                    self.write_bit(b, sd, if base.ends_with("_WREXEC") { new } else { old });
                }
                self.write_scc(b, new);
                return Ok(());
            }
            if bitwise && (sd == EXEC || reads_mask) {
                let a = self.src_bit(b, s0)?;
                let v = match op {
                    "MOV" => a,
                    "NOT" => self.not(a),
                    _ => {
                        let c = self.src_bit(b, s1)?;
                        self.bit_op(op, a, c)?
                    }
                };
                self.write_bit(b, sd, v);
                if op != "MOV" {
                    self.write_scc(b, v);
                }
                return Ok(());
            }
        }
        // mask compares (`s_cmp_lg_u64 exec, 0`): whether the lane's bits differ / agree
        let cmp64 = if self.wave64 { ["S_CMP_EQ_U64", "S_CMP_LG_U64"] } else { ["S_CMP_EQ_U32", "S_CMP_LG_U32"] };
        if cmp64.contains(&name) && [s0, s1].iter().any(|&c| c <= 127 && self.holds_mask(b, c as u8)) {
            let (a, c) = (self.src_bit(b, s0)?, self.src_bit(b, s1)?);
            let x = self.ir.bin(Bin::Xor, a, c);
            let v = if name == cmp64[0] { self.not(x) } else { x };
            self.write_scc(b, v);
            return Ok(());
        }
        // numbers
        let num2 = |l: &mut Self, op: Bin| -> R<I::V> {
            let (a, c) = (l.src(b, d, s0)?, l.src(b, d, s1)?);
            Ok(l.ir.bin(op, a, c))
        };
        let nonzero = |l: &mut Self, v: I::V| {
            let z = l.ir.int(0);
            l.ir.icmp(Cmp::Ne, v, z)
        };
        match name {
            "S_MOV_B32" => {
                let v = self.src(b, d, s0)?;
                self.write_num(b, sd, v);
            }
            "S_MOVK_I32" => {
                let v = self.ir.int(fv("SIMM16").unwrap_or(0) as u16 as i16 as i32 as u32);
                self.write_num(b, sd, v);
            }
            "S_MOV_B64" => {
                let (lo, hi) = self.src64(b, d, s0)?;
                self.write_num(b, sd, lo);
                self.write_num(b, sd + 1, hi);
            }
            "S_ADD_U32" | "S_SUB_U32" => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let (op, carry) = if name == "S_ADD_U32" { (Bin::Add, true) } else { (Bin::Sub, false) };
                let v = self.ir.bin(op, a, c);
                // carry out: the sum wrapped below a; borrow: a < b
                let scc = if carry { self.ir.icmp(Cmp::Ult, v, a) } else { self.ir.icmp(Cmp::Ult, a, c) };
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            "S_ADD_I32" | "S_SUB_I32" => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let op = if name == "S_ADD_I32" { Bin::Add } else { Bin::Sub };
                let v = self.ir.bin(op, a, c);
                // signed overflow: the operands' signs agree (add) / differ (sub) and the result's differs
                let (av, cv) = (self.ir.bin(Bin::Xor, a, v), self.ir.bin(Bin::Xor, c, v));
                let both = if op == Bin::Add {
                    self.ir.bin(Bin::And, av, cv)
                } else {
                    let ac = self.ir.bin(Bin::Xor, a, c);
                    self.ir.bin(Bin::And, ac, av)
                };
                let z = self.ir.int(0);
                let scc = self.ir.icmp(Cmp::Slt, both, z);
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            "S_MUL_I32" => {
                let v = num2(self, Bin::Mul)?;
                self.write_num(b, sd, v);
            }
            "S_AND_B32" | "S_OR_B32" | "S_XOR_B32" => {
                let op = match name {
                    "S_AND_B32" => Bin::And,
                    "S_OR_B32" => Bin::Or,
                    _ => Bin::Xor,
                };
                let v = num2(self, op)?;
                let scc = nonzero(self, v);
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            "S_LSHL_B32" | "S_LSHR_B32" | "S_ASHR_I32" => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let c31 = self.ir.int(31);
                let sh = self.ir.bin(Bin::And, c, c31);
                let op = match name {
                    "S_LSHL_B32" => Bin::Shl,
                    "S_LSHR_B32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let v = self.ir.bin(op, a, sh);
                let scc = nonzero(self, v);
                self.write_num(b, sd, v);
                self.write_scc(b, scc);
            }
            n if n.starts_with("S_CMP_") && (n.ends_with("_U32") || n.ends_with("_I32")) => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let signed = n.ends_with("_I32");
                let p = match &n[6..8] {
                    "EQ" => Cmp::Eq,
                    "LG" => Cmp::Ne,
                    "GT" => if signed { Cmp::Sgt } else { Cmp::Ugt },
                    "GE" => if signed { Cmp::Sge } else { Cmp::Uge },
                    "LT" => if signed { Cmp::Slt } else { Cmp::Ult },
                    "LE" => if signed { Cmp::Sle } else { Cmp::Ule },
                    _ => return Err(format!("unsupported {n}")),
                };
                let v = self.ir.icmp(p, a, c);
                self.write_scc(b, v);
            }
            "S_NOP" | "S_WAITCNT" | "S_WAITCNT_VSCNT" | "S_WAITCNT_VMCNT" | "S_WAITCNT_LGKMCNT" | "S_WAITCNT_EXPCNT" | "S_INST_PREFETCH" | "S_CLAUSE" => {}
            // MODE (float rounding / denormals): the f32 ops lowered so far ignore it, as the
            // HLSL path does for f32 (it follows MODE only for f16 / f64 rounding); the one f16
            // rounding op lowered, v_cvt_f16_f32, is refused in a program that writes MODE
            "S_SETREG_IMM32_B32" | "S_SETREG_B32" if self.sets_mode_only(d) => {}
            "S_SETREG_IMM32_B32" | "S_SETREG_B32" => {
                let s = fv("SIMM16").unwrap_or(0);
                return Err(format!("unsupported {name} hwreg {} offset {} size {}", s & 0x3F, (s >> 6) & 0x1F, ((s >> 11) & 0x1F) + 1));
            }
            n => return Err(format!("unsupported {n}")),
        }
        Ok(())
    }

    /// A 64-bit scalar source: a register pair or a constant.
    fn src64(&mut self, b: usize, d: &Decoded, code: u64) -> R<(I::V, I::V)> {
        Ok(match code {
            0..=105 | 106 | 126 => (self.read_num(b, code as u8)?, self.read_num(b, code as u8 + 1)?),
            128..=192 => (self.ir.int((code - 128) as u32), self.ir.int(0)),
            193..=208 => (self.ir.int((-((code - 192) as i32)) as u32), self.ir.int(u32::MAX)),
            255 => (self.ir.int(d.literal.ok_or("missing literal")?), self.ir.int(0)),
            _ => return Err(format!("64-bit source code {code}")),
        })
    }

    fn valu(&mut self, b: usize, d: &Decoded) -> R<()> {
        if d.inst.name == "V_NOP" {
            return Ok(());
        }
        if matches!(d.variant, Variant::Dpp16 | Variant::Dpp8) {
            self.dpp(b, d)?;
        }
        let r = self.valu_op(b, d);
        (self.dpp_src, self.dpp_en) = (None, None);
        r
    }

    fn valu_op(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        if d.variant == Variant::Sdwa {
            return Err(format!("{name}: SDWA"));
        }
        if self.wave_valu(b, d)? {
            return Ok(());
        }
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f);
        let vop3 = d.enc.name.starts_with("ENC_VOP3") || d.enc.name.starts_with("VOP3");
        // the float modifiers are the f32 ops' (`f32_arith`); OP_SEL is no one's yet
        let f32_cmp = name.starts_with("V_CMP") && name.ends_with("_F32");
        let mods: &[&str] = if matches!(name, "V_CVT_F32_F16" | "V_CVT_F16_F32") {
            &[]
        } else if F32_ARITH.contains(&name) {
            &["OP_SEL"]
        } else if f32_cmp {
            &["CLAMP", "OMOD", "OP_SEL"]
        } else {
            &["ABS", "NEG", "CLAMP", "OMOD", "OP_SEL"]
        };
        if vop3 && mods.iter().any(|f| fv(f).unwrap_or(0) != 0) {
            return Err(format!("{name}: VOP3 modifiers"));
        }
        // sources: VOP1 / VOP2 / VOPC put src1 in VSRC1 (a VGPR); VOP3 in SRC1
        let s0 = fv("SRC0").unwrap_or(0);
        let s1 = if vop3 { fv("SRC1").unwrap_or(0) } else { fv("VSRC1").map(|v| v + 256).unwrap_or(0) };
        let vdst = fv("VDST").unwrap_or(0);
        if let Some(rest) = name.strip_prefix("V_CMP") {
            let (x, cond) = match rest.strip_prefix("X_") {
                Some(c) => (true, c),
                None => (false, rest.trim_start_matches('_')),
            };
            let (pred, ty) = cond.split_once('_').ok_or_else(|| format!("unsupported {name}"))?;
            let signed = match ty {
                "U32" | "F32" => false,
                "I32" => true,
                _ => return Err(format!("unsupported {name}")),
            };
            let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
            let c = if ty == "F32" {
                self.f32_cmp(d, vop3, pred, a, c).ok_or_else(|| format!("unsupported {name}"))?
            } else {
                match pred {
                    "EQ" => self.ir.icmp(Cmp::Eq, a, c),
                    "NE" => self.ir.icmp(Cmp::Ne, a, c),
                    "GT" => self.ir.icmp(if signed { Cmp::Sgt } else { Cmp::Ugt }, a, c),
                    "GE" => self.ir.icmp(if signed { Cmp::Sge } else { Cmp::Uge }, a, c),
                    "LT" => self.ir.icmp(if signed { Cmp::Slt } else { Cmp::Ult }, a, c),
                    "LE" => self.ir.icmp(if signed { Cmp::Sle } else { Cmp::Ule }, a, c),
                    "F" => self.ir.boolean(false),
                    "T" => self.ir.boolean(true),
                    _ => return Err(format!("unsupported {name}")),
                }
            };
            // the lane's result, 0 in inactive lanes
            let e = self.read_bit(b, EXEC)?;
            let bit = self.ir.bin(Bin::And, c, e);
            let dst = if x { EXEC } else if vop3 { vdst as u8 } else { VCC };
            self.write_bit(b, dst, bit);
            return Ok(());
        }
        if F32_ARITH.contains(&name) {
            let v = self.f32_arith(b, d, vop3, s0, s1, vdst)?;
            return self.write_v(b, d, vdst as u16, v);
        }
        let v = match name {
            "V_MOV_B32" => self.src(b, d, s0)?,
            "V_CNDMASK_B32" => {
                let sel = if vop3 { fv("SRC2").unwrap_or(0) } else { VCC as u64 };
                let m = self.src_bit(b, sel)?;
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                self.ir.select(m, c, a)
            }
            _ => {
                let (a, c) = (self.src(b, d, s0)?, self.src(b, d, s1)?);
                let c31 = self.ir.int(31);
                match name {
                    "V_ADD_NC_U32" | "V_ADD_U32" => self.ir.bin(Bin::Add, a, c),
                    "V_SUB_NC_U32" | "V_SUB_U32" => self.ir.bin(Bin::Sub, a, c),
                    "V_SUBREV_NC_U32" | "V_SUBREV_U32" => self.ir.bin(Bin::Sub, c, a),
                    "V_AND_B32" => self.ir.bin(Bin::And, a, c),
                    "V_OR_B32" => self.ir.bin(Bin::Or, a, c),
                    "V_XOR_B32" => self.ir.bin(Bin::Xor, a, c),
                    "V_MUL_LO_U32" => self.ir.bin(Bin::Mul, a, c),
                    // three-source integer ops (VOP3): shifts take the low 5 bits of their amount
                    "V_LSHL_ADD_U32" | "V_ADD_LSHL_U32" | "V_LSHL_OR_B32" | "V_AND_OR_B32" | "V_OR3_B32" | "V_XOR3_B32" | "V_ADD3_U32" => {
                        let e = self.src(b, d, fv("SRC2").unwrap_or(0))?;
                        match name {
                            "V_LSHL_ADD_U32" | "V_LSHL_OR_B32" => {
                                let sh = self.ir.bin(Bin::And, c, c31);
                                let x = self.ir.bin(Bin::Shl, a, sh);
                                self.ir.bin(if name == "V_LSHL_ADD_U32" { Bin::Add } else { Bin::Or }, x, e)
                            }
                            "V_ADD_LSHL_U32" => {
                                let x = self.ir.bin(Bin::Add, a, c);
                                let sh = self.ir.bin(Bin::And, e, c31);
                                self.ir.bin(Bin::Shl, x, sh)
                            }
                            "V_AND_OR_B32" => {
                                let x = self.ir.bin(Bin::And, a, c);
                                self.ir.bin(Bin::Or, x, e)
                            }
                            _ => {
                                let op = match name {
                                    "V_OR3_B32" => Bin::Or,
                                    "V_XOR3_B32" => Bin::Xor,
                                    _ => Bin::Add,
                                };
                                let x = self.ir.bin(op, a, c);
                                self.ir.bin(op, x, e)
                            }
                        }
                    }
                    "V_LSHLREV_B32" | "V_LSHRREV_B32" | "V_ASHRREV_I32" => {
                        let sh = self.ir.bin(Bin::And, a, c31);
                        let op = match name {
                            "V_LSHLREV_B32" => Bin::Shl,
                            "V_LSHRREV_B32" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.ir.bin(op, c, sh)
                    }
                    n => return Err(format!("unsupported {n}")),
                }
            }
        };
        self.write_v(b, d, vdst as u16, v)
    }

    /// f32 VOP2 / VOP3 arithmetic as the HLSL path writes it (`emit.rs` `valu`): each source
    /// takes abs, then neg, the result omod, then clamp. `v_fma` / `v_fmac` are HLSL's `mad`;
    /// `v_mad` / `v_mac` round the product before the add (`mad_nf`).
    fn f32_arith(&mut self, b: usize, d: &Decoded, vop3: bool, s0: u64, s1: u64, vdst: u64) -> R<I::V> {
        let name = d.inst.name;
        let fv = |f: &str| if vop3 { rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) } else { 0 };
        // integer and f16 sources: no abs / neg (the HLSL path's `fsrc` is for f32 sources)
        if matches!(name, "V_CVT_F32_I32" | "V_CVT_F32_U32" | "V_CVT_F32_F16") {
            let a = self.src(b, d, s0)?;
            let r = match name {
                "V_CVT_F32_I32" => self.ir.cvt(Cvt::SiToF, a),
                "V_CVT_F32_U32" => self.ir.cvt(Cvt::UiToF, a),
                _ => {
                    let h = if fv("OP_SEL") & 1 == 1 {
                        let k = self.ir.int(16);
                        self.ir.bin(Bin::LShr, a, k)
                    } else {
                        let k = self.ir.int(0xFFFF);
                        self.ir.bin(Bin::And, a, k)
                    };
                    self.ir.fun(FUn::F16ToF32, h)
                }
            };
            return Ok(self.f32_out(d, vop3, r));
        }
        let n = match name {
            "V_FMA_F32" | "V_MAD_F32" => 3,
            n if n.starts_with("V_CVT_") => 1,
            "V_RCP_F32" | "V_RCP_IFLAG_F32" | "V_RSQ_F32" | "V_SQRT_F32" | "V_EXP_F32" | "V_LOG_F32" | "V_SIN_F32" | "V_COS_F32" => 1,
            "V_FLOOR_F32" | "V_CEIL_F32" | "V_TRUNC_F32" | "V_RNDNE_F32" | "V_FRACT_F32" => 1,
            _ => 2,
        };
        let mut x = Vec::new();
        for (i, code) in [s0, s1, fv("SRC2")].into_iter().take(n).enumerate() {
            let mut v = self.src(b, d, code)?;
            if fv("ABS") >> i & 1 == 1 {
                v = self.ir.fun(FUn::Abs, v);
            }
            if fv("NEG") >> i & 1 == 1 {
                v = self.ir.fun(FUn::Neg, v);
            }
            x.push(v);
        }
        if matches!(name, "V_MAC_F32" | "V_FMAC_F32") {
            x.push(self.read(b, Var::V(vdst as u16)));
        }
        // integer results: no output modifiers
        match name {
            "V_CVT_U32_F32" => return Ok(self.cvt_u32(x[0])),
            "V_CVT_I32_F32" => return Ok(self.cvt_i32(x[0])),
            "V_CVT_RPI_I32_F32" | "V_CVT_FLR_I32_F32" => {
                let mut y = x[0];
                if name == "V_CVT_RPI_I32_F32" {
                    let h = self.ir.int(0.5f32.to_bits());
                    y = self.ir.fbin(FBin::Add, y, h);
                }
                let y = self.ir.fun(FUn::Floor, y);
                return Ok(self.cvt_i32(y));
            }
            // omod and clamp in f32, then binary16 (round to nearest even) into one half
            "V_CVT_F16_F32" => {
                if self.sets_mode {
                    return Err(format!("{name}: MODE set by the program"));
                }
                let y = self.f32_out(d, vop3, x[0]);
                let h = self.f16_rne(y);
                let old = self.read(b, Var::V(vdst as u16));
                let (keep, h) = if fv("OP_SEL") >> 3 & 1 == 1 {
                    let k = self.ir.int(16);
                    (0xFFFF, self.ir.bin(Bin::Shl, h, k))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.ir.int(keep);
                let old = self.ir.bin(Bin::And, old, keep);
                return Ok(self.ir.bin(Bin::Or, old, h));
            }
            _ => {}
        }
        let r = match name {
            "V_ADD_F32" => self.ir.fbin(FBin::Add, x[0], x[1]),
            "V_SUB_F32" => self.ir.fbin(FBin::Sub, x[0], x[1]),
            "V_SUBREV_F32" => self.ir.fbin(FBin::Sub, x[1], x[0]),
            "V_MUL_F32" => self.ir.fbin(FBin::Mul, x[0], x[1]),
            "V_MIN_F32" => self.ir.fbin(FBin::Min, x[0], x[1]),
            "V_MAX_F32" => self.ir.fbin(FBin::Max, x[0], x[1]),
            "V_FMA_F32" | "V_FMAC_F32" => self.ir.fmad(x[0], x[1], x[2]),
            // rcp is HLSL's `rcp` (1 / x); sin and cos take turns: x · 2π first, as `emit.rs`
            "V_RCP_F32" | "V_RCP_IFLAG_F32" => {
                let one = self.ir.int(1.0f32.to_bits());
                self.ir.fbin(FBin::Div, one, x[0])
            }
            "V_SIN_F32" | "V_COS_F32" => {
                let k = self.ir.int(6.283_185_3f32.to_bits());
                let t = self.ir.fbin(FBin::Mul, x[0], k);
                self.ir.fun(if name == "V_SIN_F32" { FUn::Sin } else { FUn::Cos }, t)
            }
            // `fract_f`: x − floor(x), below 1.0
            "V_FRACT_F32" => {
                let fl = self.ir.fun(FUn::Floor, x[0]);
                let r = self.ir.fbin(FBin::Sub, x[0], fl);
                let one = self.ir.int(1.0f32.to_bits());
                let ge = self.ir.fcmp(FCmp::Ge, r, one);
                self.sel_k(ge, 0x3F7F_FFFF, r)
            }
            "V_RSQ_F32" | "V_SQRT_F32" | "V_EXP_F32" | "V_LOG_F32" | "V_FLOOR_F32" | "V_CEIL_F32" | "V_TRUNC_F32" | "V_RNDNE_F32" => {
                let op = match name {
                    "V_RSQ_F32" => FUn::Rsq,
                    "V_SQRT_F32" => FUn::Sqrt,
                    "V_EXP_F32" => FUn::Exp,
                    "V_LOG_F32" => FUn::Log,
                    "V_FLOOR_F32" => FUn::Floor,
                    "V_CEIL_F32" => FUn::Ceil,
                    "V_TRUNC_F32" => FUn::Trunc,
                    _ => FUn::RoundNe,
                };
                self.ir.fun(op, x[0])
            }
            _ => {
                let p = self.ir.fbin(FBin::Mul, x[0], x[1]);
                self.ir.fbin(FBin::Add, p, x[2])
            }
        };
        Ok(self.f32_out(d, vop3, r))
    }

    /// The output modifiers of an f32 result: omod, then clamp.
    fn f32_out(&mut self, d: &Decoded, vop3: bool, r: I::V) -> I::V {
        let fv = |f: &str| if vop3 { rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) } else { 0 };
        let r = match fv("OMOD") {
            0 => r,
            m => {
                let k = self.ir.int([2.0f32, 4.0, 0.5][m as usize - 1].to_bits());
                self.ir.fbin(FBin::Mul, r, k)
            }
        };
        if fv("CLAMP") == 1 {
            self.ir.fun(FUn::Sat, r)
        } else {
            r
        }
    }

    /// `select(c, k, v)` with k a constant
    fn sel_k(&mut self, c: I::V, k: u32, v: I::V) -> I::V {
        let k = self.ir.int(k);
        self.ir.select(c, k, v)
    }

    /// `cvt_u32` of the HLSL path: NaN and ≤ 0 give 0, ≥ 2^32 gives 0xFFFFFFFF.
    fn cvt_u32(&mut self, f: I::V) -> I::V {
        let (z, big) = (self.ir.int(0), self.ir.int(4294967296.0f32.to_bits()));
        let (n, le) = (self.ir.is_nan(f), self.ir.fcmp(FCmp::Le, f, z));
        let low = self.ir.bin(Bin::Or, n, le);
        let ge = self.ir.fcmp(FCmp::Ge, f, big);
        let v = self.ir.cvt(Cvt::FToUi, f);
        let v = self.sel_k(ge, u32::MAX, v);
        self.sel_k(low, 0, v)
    }

    /// `cvt_i32` of the HLSL path: NaN gives 0, out of range saturates.
    fn cvt_i32(&mut self, f: I::V) -> I::V {
        let (lo, hi) = (self.ir.int((-2147483648.0f32).to_bits()), self.ir.int(2147483648.0f32.to_bits()));
        let n = self.ir.is_nan(f);
        let (le, ge) = (self.ir.fcmp(FCmp::Le, f, lo), self.ir.fcmp(FCmp::Ge, f, hi));
        let v = self.ir.cvt(Cvt::FToSi, f);
        let v = self.sel_k(ge, 0x7FFF_FFFF, v);
        let v = self.sel_k(le, 0x8000_0000, v);
        self.sel_k(n, 0, v)
    }

    /// `f32_to_f16_rm(f, 0)` of the HLSL path: binary32 bits to binary16, round to nearest even.
    fn f16_rne(&mut self, x: I::V) -> I::V {
        let k = |l: &mut Self, v: u32| l.ir.int(v);
        let and = |l: &mut Self, a: I::V, m: u32| {
            let m = l.ir.int(m);
            l.ir.bin(Bin::And, a, m)
        };
        let shr = |l: &mut Self, a: I::V, n: u32| {
            let n = l.ir.int(n);
            l.ir.bin(Bin::LShr, a, n)
        };
        let ucmp = |l: &mut Self, c: Cmp, a: I::V, v: u32| {
            let v = l.ir.int(v);
            l.ir.icmp(c, a, v)
        };
        let sign = shr(self, x, 16);
        let sign = and(self, sign, 0x8000);
        let ax = and(self, x, 0x7FFF_FFFF);
        // normal: exponent rebased, 10 mantissa bits, 13 bits rounded off
        let e = shr(self, ax, 23);
        let k112 = k(self, 112);
        let e = self.ir.bin(Bin::Sub, e, k112);
        let k10 = k(self, 10);
        let e = self.ir.bin(Bin::Shl, e, k10);
        let m = shr(self, ax, 13);
        let m = and(self, m, 0x3FF);
        let h_n = self.ir.bin(Bin::Or, e, m);
        let rem_n = and(self, ax, 0x1FFF);
        // denormal: the significand shifted right by 126 - exponent
        let k126 = k(self, 126);
        let e = shr(self, ax, 23);
        let sh = self.ir.bin(Bin::Sub, k126, e);
        let sig = and(self, ax, 0x007F_FFFF);
        let k23 = k(self, 0x0080_0000);
        let sig = self.ir.bin(Bin::Or, sig, k23);
        let h_d = self.ir.bin(Bin::LShr, sig, sh);
        let one = k(self, 1);
        let sh1 = self.ir.bin(Bin::Sub, sh, one);
        let half_d = self.ir.bin(Bin::Shl, one, sh1);
        let low = self.ir.bin(Bin::Shl, one, sh);
        let low = self.ir.bin(Bin::Sub, low, one);
        let rem_d = self.ir.bin(Bin::And, sig, low);
        let normal = ucmp(self, Cmp::Uge, ax, 0x3880_0000);
        let h = self.ir.select(normal, h_n, h_d);
        let rem = self.ir.select(normal, rem_n, rem_d);
        let half_n = k(self, 0x1000);
        let half = self.ir.select(normal, half_n, half_d);
        let gt = self.ir.icmp(Cmp::Ugt, rem, half);
        let eq = self.ir.icmp(Cmp::Eq, rem, half);
        let odd = and(self, h, 1);
        let odd = ucmp(self, Cmp::Ne, odd, 0);
        let tie = self.ir.bin(Bin::And, eq, odd);
        let inc = self.ir.bin(Bin::Or, gt, tie);
        let inc = self.ir.zext(inc);
        let r = self.ir.bin(Bin::Add, h, inc);
        // below half the smallest denormal: 0 or that denormal; too large: infinity
        let tiny = ucmp(self, Cmp::Ult, ax, 0x3380_0000);
        let up = ucmp(self, Cmp::Ugt, ax, 0x3300_0000);
        let up = self.ir.zext(up);
        let r = self.ir.select(tiny, up, r);
        let ovf = ucmp(self, Cmp::Uge, ax, 0x4780_0000);
        let r = self.sel_k(ovf, 0x7C00, r);
        let zero = ucmp(self, Cmp::Eq, ax, 0);
        let z = k(self, 0);
        let r = self.ir.select(zero, z, r);
        // infinity, or NaN with the payload's top bits
        let p = shr(self, ax, 13);
        let p = and(self, p, 0x1FF);
        let k7e = k(self, 0x7E00);
        let nan = self.ir.bin(Bin::Or, p, k7e);
        let is_inf = ucmp(self, Cmp::Eq, ax, 0x7F80_0000);
        let infnan = self.sel_k(is_inf, 0x7C00, nan);
        let special = ucmp(self, Cmp::Uge, ax, 0x7F80_0000);
        let r = self.ir.select(special, infnan, r);
        self.ir.bin(Bin::Or, sign, r)
    }

    /// An f32 compare as the HLSL path writes it (`emit.rs` `float_compare`): the sources take
    /// abs, then neg; NaN is tested explicitly, the ordered compares false and the unordered
    /// ones (`N…`, `U`) true on it. `None` for a predicate it does not know (`CLASS`).
    fn f32_cmp(&mut self, d: &Decoded, vop3: bool, pred: &str, a: I::V, c: I::V) -> Option<I::V> {
        let fv = |f: &str| if vop3 { rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) } else { 0 };
        let mut x = [a, c];
        for (i, v) in x.iter_mut().enumerate() {
            if fv("ABS") >> i & 1 == 1 {
                *v = self.ir.fun(FUn::Abs, *v);
            }
            if fv("NEG") >> i & 1 == 1 {
                *v = self.ir.fun(FUn::Neg, *v);
            }
        }
        let [a, c] = x;
        let rel = |l: &mut Self, r: &str| -> Option<I::V> {
            let p = match r {
                "LT" => FCmp::Lt,
                "EQ" => FCmp::Eq,
                "LE" => FCmp::Le,
                "GT" => FCmp::Gt,
                "GE" => FCmp::Ge,
                "LG" => {
                    let (lt, gt) = (l.ir.fcmp(FCmp::Lt, a, c), l.ir.fcmp(FCmp::Gt, a, c));
                    return Some(l.ir.bin(Bin::Or, lt, gt));
                }
                _ => return None,
            };
            Some(l.ir.fcmp(p, a, c))
        };
        let nan = |l: &mut Self| {
            let (x, y) = (l.ir.is_nan(a), l.ir.is_nan(c));
            l.ir.bin(Bin::Or, x, y)
        };
        Some(match pred {
            "F" => self.ir.boolean(false),
            "TRU" => self.ir.boolean(true),
            "O" => {
                let n = nan(self);
                self.not(n)
            }
            "U" => nan(self),
            p => match p.strip_prefix('N').and_then(|r| rel(self, r)) {
                Some(r) => {
                    let (n, nr) = (nan(self), self.not(r));
                    self.ir.bin(Bin::Or, n, nr)
                }
                None => {
                    let r = rel(self, p)?;
                    let n = nan(self);
                    let nn = self.not(n);
                    self.ir.bin(Bin::And, nn, r)
                }
            },
        })
    }

    /// `s_setreg*` writing `HW_REG_MODE` (id 1 of its `SIMM16`) and nothing else.
    fn sets_mode_only(&self, d: &Decoded) -> bool {
        rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) & 0x3F == 1
    }

    /// `s_load_dword{,x2,x4,x8,x16}`: dwords of the flat memory into SGPRs, from
    /// `s[sbase] + offset + soffset` (one memory window, as `emit::smem` with `mem_windows` 1).
    fn smem(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        let count = match name.strip_prefix("S_LOAD_") {
            Some("DWORD") => 1,
            Some("DWORDX2") => 2,
            Some("DWORDX4") => 4,
            Some("DWORDX8") => 8,
            Some("DWORDX16") => 16,
            _ => return Err(format!("unsupported {name}")),
        };
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0) * rdna2::register_scale(f);
        let (sdata, sbase) = (fv("SDATA") as u8, fv("SBASE") as u8);
        let imm = (((fv("OFFSET") as u32) << 11) as i32 >> 11) as u32; // 21-bit signed
        let soff = rdna2::field_value(d.enc, &d.words, "SOFFSET").unwrap_or(125);
        let base = self.read_num(b, sbase)?;
        let imm = self.ir.int(imm);
        let mut a = self.ir.bin(Bin::Add, base, imm);
        if soff != 125 {
            let s = self.src(b, d, soff)?;
            a = self.ir.bin(Bin::Add, a, s);
        }
        let m = self.ir.int(!3);
        let a = self.ir.bin(Bin::And, a, m);
        let vals: Vec<I::V> = (0..count)
            .map(|i| {
                let o = self.ir.int(4 * i as u32);
                let ai = self.ir.bin(Bin::Add, a, o);
                self.ir.mem_load(ai)
            })
            .collect();
        for (i, v) in vals.into_iter().enumerate() {
            self.write_num(b, sdata + i as u8, v);
        }
        Ok(())
    }

    /// `global_` / `flat_` `load` / `store_dword{,x2,x3,x4}` on the flat memory. A guarded load
    /// keeps the old register in lanes without their EXEC bit (`write_v`); a guarded store is
    /// lowered in a block only those lanes enter (the caller's split).
    fn vmem(&mut self, b: usize, d: &Decoded) -> R<()> {
        let name = d.inst.name;
        let op = name.trim_start_matches("GLOBAL_").trim_start_matches("FLAT_");
        let (store, count) = match op {
            "LOAD_DWORD" => (false, 1),
            "LOAD_DWORDX2" => (false, 2),
            "LOAD_DWORDX3" => (false, 3),
            "LOAD_DWORDX4" => (false, 4),
            "STORE_DWORD" => (true, 1),
            "STORE_DWORDX2" => (true, 2),
            "STORE_DWORDX3" => (true, 3),
            "STORE_DWORDX4" => (true, 4),
            _ => return Err(format!("unsupported {name}")),
        };
        let global = d.enc.name == "ENC_FLAT_GLBL";
        if !global && d.enc.name != "ENC_FLAT" {
            return Err(format!("{name}: {}", d.enc.name));
        }
        let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
        let raw = fv("OFFSET") as u32;
        let off = if global { ((raw << 20) as i32 >> 20) as u32 } else { raw };
        let saddr = fv("SADDR");
        let mut a = self.read(b, Var::V(fv("ADDR") as u16));
        if global && saddr != 0x7D && saddr != 0x7F {
            let s = self.read_num(b, saddr as u8)?;
            a = self.ir.bin(Bin::Add, s, a);
        }
        let off = self.ir.int(off);
        let a = self.ir.bin(Bin::Add, a, off);
        for i in 0..count {
            let o = self.ir.int(4 * i as u32);
            let ai = self.ir.bin(Bin::Add, a, o);
            let m = self.ir.int(!3);
            let ai = self.ir.bin(Bin::And, ai, m);
            if store {
                let v = self.read(b, Var::V(fv("DATA") as u16 + i as u16));
                self.ir.mem_store(ai, v);
            } else {
                let v = self.ir.mem_load(ai);
                self.write_v(b, d, fv("VDST") as u16 + i as u16, v)?;
            }
        }
        Ok(())
    }

    /// The condition under which a lane takes the branch ending block b.
    fn cond(&mut self, b: usize, d: &Decoded) -> R<I::V> {
        Ok(match d.inst.name {
            "S_CBRANCH_SCC1" => self.read_scc(b),
            "S_CBRANCH_SCC0" => {
                let s = self.read_scc(b);
                self.not(s)
            }
            n @ ("S_CBRANCH_VCCZ" | "S_CBRANCH_VCCNZ") => {
                let (v, e) = (self.read_bit(b, VCC)?, self.read_bit(b, EXEC)?);
                let c = self.ir.bin(Bin::And, v, e);
                if n == "S_CBRANCH_VCCZ" {
                    self.not(c)
                } else {
                    c
                }
            }
            "S_CBRANCH_EXECNZ" => self.read_bit(b, EXEC)?,
            "S_CBRANCH_EXECZ" => {
                let e = self.read_bit(b, EXEC)?;
                self.not(e)
            }
            n => return Err(format!("unsupported branch {n}")),
        })
    }

    /// Whether a lane reading mask bit v in block c may see its own stale bit: v (or a phi
    /// operand reaching it) was written in another block that not every lane passed through.
    /// The hardware's whole-wave write left the absent lanes' bits 0 there; this lane kept its
    /// own older one. EXEC is not concerned: an absent lane's bit is 0 in both.
    fn stale(&self, c: usize, v: I::V) -> bool {
        let mut seen = HashSet::new();
        let mut work = vec![v];
        while let Some(v) = work.pop() {
            let v = self.ssa.resolve_value(v);
            if !seen.insert(v) {
                continue;
            }
            match self.ssa.phi_operands(v) {
                Some(ops) => work.extend_from_slice(ops),
                None => {
                    if let Some(&b) = self.def_block.get(&v) {
                        if b != c && self.partial[b] {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Whether v is (or is a phi reaching) `marker`.
    fn reaches(&self, v: I::V, marker: I::V) -> bool {
        let mut seen = HashSet::new();
        let mut work = vec![v];
        while let Some(v) = work.pop() {
            let v = self.ssa.resolve_value(v);
            if v == marker {
                return true;
            }
            if seen.insert(v) {
                if let Some(ops) = self.ssa.phi_operands(v) {
                    work.extend_from_slice(ops);
                }
            }
        }
        false
    }
}

/// Per analysis block: whether some lane that reached a lane branch may not pass through it
/// before the lanes meet again – it is reachable from the branch but does not post-dominate
/// both of its successors (lanes that take the other one never get there; a loop's own latch
/// counts: the lanes that leave do not come back to it).
fn partial_blocks(sum: &Summary) -> Vec<bool> {
    let n = sum.blocks.len();
    let words = (n + 1).div_ceil(64);
    // post-dominators over the blocks the entry reaches, with a virtual exit n
    let full = vec![!0u64; words];
    let mut pdom: Vec<Vec<u64>> = vec![full; n + 1];
    pdom[n] = vec![0u64; words];
    pdom[n][n / 64] |= 1 << (n % 64);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in sum.order.iter().rev() {
            let succ: Vec<usize> = if sum.blocks[b].succ.is_empty() { vec![n] } else { sum.blocks[b].succ.iter().map(|s| s.0).collect() };
            let mut new = vec![!0u64; words];
            for s in succ {
                for (w, x) in new.iter_mut().zip(&pdom[s]) {
                    *w &= x;
                }
            }
            new[b / 64] |= 1 << (b % 64);
            if new != pdom[b] {
                pdom[b] = new;
                changed = true;
            }
        }
    }
    let mut partial = vec![false; n];
    for &x in &sum.order {
        if !sum.blocks[x].lane_branch {
            continue;
        }
        let mut seen = vec![false; n];
        let mut work: Vec<usize> = sum.blocks[x].succ.iter().map(|s| s.0).collect();
        while let Some(b) = work.pop() {
            if std::mem::replace(&mut seen[b], true) {
                continue;
            }
            if sum.blocks[x].succ.iter().any(|&(s, _)| pdom[s][b / 64] >> (b % 64) & 1 == 0) {
                partial[b] = true;
            }
            work.extend(sum.blocks[b].succ.iter().map(|s| s.0));
        }
    }
    partial
}

/// Lowers a kernel-mode program (far jumps resolved) into `ir`, which gets it complete, with
/// the mask registers whose value at the end may differ from the console's: written in a block
/// some lanes skipped, so those lanes still hold their own older bits (`stale`). Nothing reads
/// them in the program (that would be refused); only the kernel ABI's store of every register.
pub fn lower_kernel<I: Ir>(insts: &[Decoded], wave64: bool, ir: I) -> R<(I, Vec<u8>)> {
    lower(insts, Stage::Kernel, wave64, ir)
}

/// Lowers a compute program (far jumps resolved) into `ir`. One-dimensional groups only for
/// now: in a 2-D / 3-D group the PC driver need not give a wave consecutive threads, which the
/// console does (`emit` reorders them; not done here yet).
pub fn lower_compute<I: Ir>(insts: &[Decoded], cs: &super::CsInterface, wave64: bool, ir: I) -> R<I> {
    if cs.threads[1] * cs.threads[2] > 1 {
        return Err("2-D / 3-D thread group".into());
    }
    Ok(lower(insts, Stage::Compute(cs), wave64, ir)?.0)
}

fn lower<I: Ir>(insts: &[Decoded], stage: Stage, wave64: bool, mut ir: I) -> R<(I, Vec<u8>)> {
    let kernel = matches!(stage, Stage::Kernel);
    let sum = lane_masks::analyze(insts, wave64).map_err(|e| e.to_string())?;
    let n = sum.blocks.len();
    if n == 0 {
        return Err("empty program".into());
    }
    // IR blocks: 0 prologue, 1..=n the program's, n + 1 exit, n + 2 lane 0's scalar stores,
    // n + 3 every lane's vector stores, n + 4 the way to the exit of lanes outside EXEC at launch,
    // then for every guarded store (only the lanes with their EXEC bit may write): the block of
    // the store and the one the rest of its block continues in. `tail[b]`: the IR block the
    // program's block b ends in.
    let (exit, sstore, vstore, skip) = (n + 1, n + 2, n + 3, n + 4);
    let guarded: HashSet<u64> = sum.guarded.iter().copied().collect();
    let mut splits: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n]; // (inst, store, cont)
    let mut count = n + 5;
    for (b, block) in sum.blocks.iter().enumerate() {
        for i in block.insts.clone() {
            if is_store(&insts[i]) && guarded.contains(&insts[i].pc) {
                splits[b].push((i, count, count + 1));
                count += 2;
            }
        }
    }
    let tail: Vec<usize> = (0..n).map(|b| splits[b].last().map_or(blk(b), |s| s.2)).collect();
    let mut partial = vec![false; count];
    for (b, p) in partial_blocks(&sum).into_iter().enumerate() {
        partial[blk(b)] = p;
        for &(_, s, c) in &splits[b] {
            partial[s] = true;
            partial[c] = p;
        }
    }
    let ends: Vec<usize> = sum.order.iter().copied().filter(|&b| sum.blocks[b].succ.is_empty()).collect();
    let mut preds = vec![Vec::new(); count];
    preds[blk(0)] = vec![0];
    for &b in &sum.order {
        let mut seen = Vec::new();
        for &(s, _) in &sum.blocks[b].succ {
            if !seen.contains(&s) {
                seen.push(s);
                preds[blk(s)].push(tail[b]);
            }
        }
        let mut prev = blk(b);
        for &(_, s, c) in &splits[b] {
            preds[s] = vec![prev];
            preds[c] = vec![prev, s];
            prev = c;
        }
    }
    // compute: every lane that runs is in EXEC, and nothing is stored at the end – the skip and
    // store blocks are left unreachable (a bare `ret`)
    if kernel {
        preds[skip] = vec![0];
        preds[sstore] = vec![exit];
        preds[vstore] = vec![exit, sstore];
    }
    preds[exit] = kernel.then_some(skip).into_iter().chain(ends.iter().map(|&b| tail[b])).collect();
    for i in 0..count {
        let id = ir.new_block();
        debug_assert_eq!(id, i);
    }
    let num_marker = ir.undef(Var::S(0));
    let bit_marker = ir.undef(Var::M(0));
    let mut l = Lower {
        ssa: Ssa::new(preds.clone()),
        ir,

        sum: &sum,
        guarded,
        wave64,
        num_marker,
        bit_marker,
        num_uses: Vec::new(),
        bit_uses: Vec::new(),
        lane: None,
        partial,
        def_block: std::collections::HashMap::new(),
        lane_uses: Vec::new(),
        sets_mode: insts.iter().any(|d| matches!(d.inst.name, "S_SETREG_IMM32_B32" | "S_SETREG_B32") && rdna2::field_value(d.enc, &d.words, "SIMM16").unwrap_or(0) & 0x3F == 1),
        dpp_src: None,
        dpp_en: None,
    };
    // the registers the program names, and the ones the runner reads back
    let mut sregs: Vec<u8> = vec![VCC, VCC + 1, 124, EXEC, EXEC + 1];
    let mut vregs: Vec<u16> = Vec::new();
    for d in insts {
        let (r, w) = rdna2::sgpr_uses(d);
        sregs.extend(r.iter().chain(&w).map(|&x| x as u8));
        let (r, w) = rdna2::vgpr_uses(d);
        vregs.extend(r.iter().chain(&w).map(|&x| x as u16));
    }
    sregs.retain(|&r| (r as usize) < KERNEL_SGPRS && r != 125);
    sregs.sort_unstable();
    sregs.dedup();
    vregs.sort_unstable();
    vregs.dedup();

    // prologue: the SGPRs from `io` (kernel: the whole register file; compute: the user data)
    l.ir.at(0);
    l.ssa.seal(&mut l.ir, 0);
    let lane = l.lane();
    for &r in &sregs {
        let a = l.ir.int(r as u32);
        let v = l.ir.io_load(a);
        l.write_num(0, r, v);
    }
    let mut row = None;
    match stage {
        Stage::Kernel => {
            let a = l.ir.int(KERNEL_SCC_SLOT as u32);
            let scc = l.ir.io_load(a);
            let z = l.ir.int(0);
            let scc = l.ir.icmp(Cmp::Ne, scc, z);
            l.ssa.write(0, Var::Scc, scc);
            let (base, stride) = (l.ir.int(KERNEL_SGPRS as u32), l.ir.int(KERNEL_VGPRS as u32));
            let r = l.ir.bin(Bin::Mul, lane, stride);
            let r = l.ir.bin(Bin::Add, base, r);
            for &v in &vregs {
                let o = l.ir.int(v as u32);
                let a = l.ir.bin(Bin::Add, r, o);
                let x = l.ir.io_load(a);
                l.ssa.write(0, Var::V(v), x);
            }
            row = Some(r);
            // lanes outside EXEC go straight to the end
            let active = l.read_bit(0, EXEC)?;
            l.write_bit(0, EXEC, active);
            l.ir.br_if(active, blk(0), skip);
            // … with their bits of every mask 0, as a whole-wave write leaves them
            l.ir.at(skip);
            l.ssa.seal(&mut l.ir, skip);
            let f = l.ir.boolean(false);
            for &r in &sregs {
                l.ssa.write(skip, Var::M(r), f);
            }
            l.ir.br(exit);
        }
        Stage::Compute(cs) => {
            // system SGPRs after the user data: the enabled group ids, then TG_SIZE
            let mut next = cs.user_sgprs as u8;
            for k in 0..3 {
                if cs.tgid[k as usize] {
                    let g = l.ir.group_id(k);
                    l.write_num(0, next, g);
                    next += 1;
                }
            }
            if cs.tg_size {
                // waves in the group (a last partial one counts) | this wave's index << 6
                let lanes = if wave64 { 64 } else { 32 };
                let waves = l.ir.int(cs.threads.iter().product::<u32>().div_ceil(lanes));
                let t = l.ir.thread_id(0);
                let sh = l.ir.int(lanes.trailing_zeros());
                let w = l.ir.bin(Bin::LShr, t, sh);
                let six = l.ir.int(6);
                let w = l.ir.bin(Bin::Shl, w, six);
                let v = l.ir.bin(Bin::Or, waves, w);
                l.write_num(0, next, v);
            }
            let f = l.ir.boolean(false);
            l.ssa.write(0, Var::Scc, f);
            for &v in &vregs {
                let x = match v {
                    0 => l.ir.thread_id(0),
                    1 if cs.tid_components >= 2 => l.ir.thread_id(1),
                    2 if cs.tid_components >= 3 => l.ir.thread_id(2),
                    _ => l.ir.int(0),
                };
                l.ssa.write(0, Var::V(v), x);
            }
            // every lane that runs is in EXEC
            let t = l.ir.boolean(true);
            l.write_bit(0, EXEC, t);
            l.ir.br(blk(0));
            l.ir.at(skip);
            l.ssa.seal(&mut l.ir, skip);
            l.ir.ret();
        }
    }

    // the program, in reverse postorder; a block is sealed once all its predecessors are in
    let mut done = vec![false; count];
    done[0] = true;
    done[skip] = true;
    let seal_ready = |l: &mut Lower<I>, done: &[bool], b: usize| {
        if preds[b].iter().all(|&p| done[p]) {
            l.ssa.seal(&mut l.ir, b);
            true
        } else {
            false
        }
    };
    let mut sealed = vec![false; count];
    sealed[0] = true;
    for &b in &sum.order.clone() {
        let ib = blk(b);
        l.ir.at(ib);
        if !sealed[ib] {
            sealed[ib] = seal_ready(&mut l, &done, ib);
        }
        if b == 0 {
            // the analysis takes the lanes that start the program as its EXEC
            let t = l.ir.boolean(true);
            l.write_bit(ib, EXEC, t);
        }
        let block = &sum.blocks[b];
        let last = block.insts.end - 1;
        let mut cur = ib;
        let mut split = splits[b].iter().peekable();
        for i in block.insts.clone() {
            let d = &insts[i];
            let name = d.inst.name;
            if i == last && (name.starts_with("S_CBRANCH") || name == "S_BRANCH" || name.starts_with("S_ENDPGM")) {
                break;
            }
            if let Some(&(_, s, c)) = split.next_if(|x| x.0 == i) {
                // a guarded store: only the lanes with their EXEC bit go through its block
                let e = l.read_bit(cur, EXEC)?;
                l.ir.br_if(e, s, c);
                done[cur] = true;
                l.ir.at(s);
                l.ssa.seal(&mut l.ir, s);
                l.vmem(s, d)?;
                l.ir.br(c);
                done[s] = true;
                l.ir.at(c);
                l.ssa.seal(&mut l.ir, c);
                cur = c;
            } else if name.starts_with("S_LOAD_") || name.starts_with("S_BUFFER_LOAD_") {
                l.smem(cur, d)?;
            } else if name.starts_with("S_") {
                l.salu(cur, d)?;
            } else if name.starts_with("V_") {
                l.valu(cur, d)?;
            } else if name.starts_with("GLOBAL_") || name.starts_with("FLAT_") {
                l.vmem(cur, d)?;
            } else if name == "DS_SWIZZLE_B32" {
                l.swizzle(cur, d)?;
            } else {
                return Err(format!("unsupported {name}"));
            }
        }
        match block.succ.as_slice() {
            [] => l.ir.br(exit),
            [(t, Edge::Always)] => l.ir.br(blk(*t)),
            [(t, _), (f, _)] if t == f => l.ir.br(blk(*t)),
            [(t, Edge::Taken), (f, Edge::NotTaken)] => {
                let c = l.cond(cur, &insts[last])?;
                l.ir.br_if(c, blk(*t), blk(*f));
            }
            s => return Err(format!("block {b}: successors {s:?}")),
        }
        done[cur] = true;
        for &(s, _) in &block.succ {
            let is = blk(s);
            if !sealed[is] && l.sum.order.contains(&s) {
                // the successor's own code comes later; sealing it early is fine once its
                // predecessors are all in
                if preds[is].iter().all(|&p| done[p]) {
                    l.ssa.seal(&mut l.ir, is);
                    sealed[is] = true;
                }
            }
        }
    }
    for b in 1..=n {
        if !sealed[b] && preds[b].iter().all(|&p| done[p]) {
            l.ssa.seal(&mut l.ir, b);
            sealed[b] = true;
        }
    }

    l.ir.at(exit);
    l.ssa.seal(&mut l.ir, exit);
    let mut stale: Vec<u8> = Vec::new();
    let Some(row) = row else {
        // compute: nothing to store
        l.ir.ret();
        for b in [sstore, vstore] {
            l.ir.at(b);
            l.ssa.seal(&mut l.ir, b);
            l.ir.ret();
        }
        return l.finish(stale);
    };
    // kernel exit: every lane is here; masks the runner reads as numbers are the wave's ballots
    let mut sval: Vec<(u8, I::V)> = Vec::new();
    let mut i = 0;
    while i < sregs.len() {
        let r = sregs[i];
        i += 1;
        let s = l.read(exit, Var::S(r));
        if l.ssa.resolve_value(s) != l.num_marker && !l.reaches(s, l.num_marker) {
            l.num_uses.push(s);
            if l.stale(exit, s) {
                stale.push(r);
            }
            sval.push((r, s));
            continue;
        }
        if !l.holds_mask(exit, r) {
            if l.wave64 && r > 0 && sval.iter().any(|&(x, _)| x == r) {
                continue; // the upper half of a mask stored with its pair
            }
            return Err(format!("s{r}: neither a number nor a mask at the end"));
        }
        // read directly: a stale bit here is reported, not refused (only the ABI's store sees it)
        let m = l.read(exit, Var::M(r));
        if r != EXEC && l.stale(exit, m) {
            stale.push(r);
            if l.wave64 {
                stale.push(r + 1); // the upper half comes from the same mask
            }
        }
        let [lo, hi] = l.ir.ballot(m);
        sval.push((r, lo));
        if l.wave64 {
            sval.retain(|&(x, _)| x != r + 1);
            sval.push((r + 1, hi));
            if sregs.get(i) == Some(&(r + 1)) {
                i += 1;
            }
        }
    }
    let scc = l.read(exit, Var::Scc);
    if l.stale(exit, scc) {
        stale.push(KERNEL_SCC_SLOT as u8);
    }
    let vval: Vec<(u16, I::V)> = vregs.iter().map(|&v| (v, l.read(exit, Var::V(v)))).collect();
    let z = l.ir.int(0);
    let first = l.ir.icmp(Cmp::Eq, lane, z);
    l.ir.br_if(first, sstore, vstore);
    l.ir.at(sstore);
    l.ssa.seal(&mut l.ir, sstore);
    for &(r, v) in &sval {
        let a = l.ir.int(r as u32);
        l.ir.res_store(a, v);
    }
    let a = l.ir.int(KERNEL_SCC_SLOT as u32);
    let s = l.ir.zext(scc);
    l.ir.res_store(a, s);
    l.ir.br(vstore);
    l.ir.at(vstore);
    l.ssa.seal(&mut l.ir, vstore);
    for &(v, x) in &vval {
        let o = l.ir.int(v as u32);
        let a = l.ir.bin(Bin::Add, row, o);
        l.ir.res_store(a, x);
    }
    l.ir.ret();
    l.finish(stale)
}

impl<I: Ir> Lower<'_, I> {
    /// Every phi is complete: no number may come from a mask, nor a mask bit from a number, nor
    /// a lane read a mask bit it kept from before a write it skipped.
    fn finish(self, mut stale: Vec<u8>) -> R<(I, Vec<u8>)> {
        for &v in &self.num_uses {
            if self.reaches(v, self.num_marker) {
                return Err("a mask read as a number (through a join)".into());
            }
        }
        for &v in &self.bit_uses {
            if self.reaches(v, self.bit_marker) {
                return Err("a number read as a mask (through a join)".into());
            }
        }
        for &(c, v) in &self.lane_uses {
            if self.stale(c, v) {
                return Err("a value written while some lanes were elsewhere, read after they met again".into());
            }
        }
        stale.sort_unstable();
        stale.dedup();
        Ok((self.ir, stale))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The IR as a small program, run lane by lane on the CPU: what the lowering means,
    /// without a DXIL writer or a GPU.
    #[derive(Clone, Debug)]
    enum Ins {
        Int(u32),
        Bool(bool),
        Bin(Bin, u32, u32),
        Icmp(Cmp, u32, u32),
        Select(u32, u32, u32),
        Zext(u32),
        Lane,
        GroupId(u32),
        ThreadId(u32),
        Ballot(u32, usize),
        ReadLane(u32, u32),
        /// as `Ballot` and `ReadLane`, but the lowering may use only its own 32-lane half's part
        BallotHalf(u32, usize),
        ReadLaneHalf(u32, u32),
        CountBits(u32),
        FBin(FBin, u32, u32),
        FUn(FUn, u32),
        FMad(u32, u32, u32),
        Cvt(Cvt, u32),
        FCmp(FCmp, u32, u32),
        IsNan(u32),
        IoLoad(u32),
        ResStore(u32, u32),
        MemLoad(u32),
        MemStore(u32, u32),
        Phi(Vec<(u32, usize)>),
    }

    #[derive(Clone, Debug)]
    enum Term {
        None,
        Br(usize),
        BrIf(u32, usize, usize),
        Ret,
    }

    #[derive(Default)]
    struct Interp {
        /// per block: (value id, instruction) in order, and the terminator
        blocks: Vec<(Vec<(u32, Ins)>, Term)>,
        cur: usize,
        next: u32,
        forward: HashMap<u32, u32>,
        undef: HashMap<bool, u32>,
        phi_block: HashMap<u32, usize>,
    }

    impl Interp {
        fn push(&mut self, ins: Ins) -> u32 {
            self.next += 1;
            let v = self.next;
            self.blocks[self.cur].0.push((v, ins));
            v
        }
        fn res(&self, mut v: u32) -> u32 {
            while let Some(&n) = self.forward.get(&v) {
                v = n;
            }
            v
        }
    }

    impl Sink for Interp {
        type V = u32;
        fn phi(&mut self, block: usize, _: Var) -> u32 {
            self.next += 1;
            let v = self.next;
            self.blocks[block].0.insert(0, (v, Ins::Phi(Vec::new())));
            self.phi_block.insert(v, block);
            v
        }
        fn incoming(&mut self, phi: u32, value: u32, pred: usize) {
            let b = self.phi_block[&phi];
            for (id, ins) in &mut self.blocks[b].0 {
                if *id == phi {
                    if let Ins::Phi(ops) = ins {
                        ops.push((value, pred));
                    }
                }
            }
        }
        fn replace(&mut self, old: u32, new: u32) {
            self.forward.insert(old, new);
            let b = self.phi_block[&old];
            self.blocks[b].0.retain(|(id, _)| *id != old);
        }
        fn undef(&mut self, var: Var) -> u32 {
            let bit = matches!(var, Var::M(_) | Var::Scc);
            if let Some(&v) = self.undef.get(&bit) {
                return v;
            }
            self.next += 1;
            let v = self.next;
            self.undef.insert(bit, v);
            v
        }
    }

    impl Ir for Interp {
        fn new_block(&mut self) -> usize {
            self.blocks.push((Vec::new(), Term::None));
            self.blocks.len() - 1
        }
        fn at(&mut self, block: usize) {
            self.cur = block;
        }
        fn int(&mut self, v: u32) -> u32 {
            self.push(Ins::Int(v))
        }
        fn boolean(&mut self, v: bool) -> u32 {
            self.push(Ins::Bool(v))
        }
        fn bin(&mut self, op: Bin, a: u32, b: u32) -> u32 {
            self.push(Ins::Bin(op, a, b))
        }
        fn icmp(&mut self, c: Cmp, a: u32, b: u32) -> u32 {
            self.push(Ins::Icmp(c, a, b))
        }
        fn select(&mut self, c: u32, a: u32, b: u32) -> u32 {
            self.push(Ins::Select(c, a, b))
        }
        fn zext(&mut self, b: u32) -> u32 {
            self.push(Ins::Zext(b))
        }
        fn lane(&mut self) -> u32 {
            self.push(Ins::Lane)
        }
        fn group_id(&mut self, k: u32) -> u32 {
            self.push(Ins::GroupId(k))
        }
        fn thread_id(&mut self, k: u32) -> u32 {
            self.push(Ins::ThreadId(k))
        }
        fn ballot(&mut self, c: u32) -> [u32; 2] {
            [self.push(Ins::Ballot(c, 0)), self.push(Ins::Ballot(c, 1))]
        }
        fn ballot_half(&mut self, c: u32) -> [u32; 2] {
            [self.push(Ins::BallotHalf(c, 0)), self.push(Ins::BallotHalf(c, 1))]
        }
        fn read_lane(&mut self, v: u32, lane: u32) -> u32 {
            self.push(Ins::ReadLane(v, lane))
        }
        fn read_lane_half(&mut self, v: u32, lane: u32) -> u32 {
            self.push(Ins::ReadLaneHalf(v, lane))
        }
        fn count_bits(&mut self, v: u32) -> u32 {
            self.push(Ins::CountBits(v))
        }
        fn fbin(&mut self, op: FBin, a: u32, b: u32) -> u32 {
            self.push(Ins::FBin(op, a, b))
        }
        fn fun(&mut self, op: FUn, a: u32) -> u32 {
            self.push(Ins::FUn(op, a))
        }
        fn fmad(&mut self, a: u32, b: u32, c: u32) -> u32 {
            self.push(Ins::FMad(a, b, c))
        }
        fn cvt(&mut self, c: Cvt, a: u32) -> u32 {
            self.push(Ins::Cvt(c, a))
        }
        fn fcmp(&mut self, c: FCmp, a: u32, b: u32) -> u32 {
            self.push(Ins::FCmp(c, a, b))
        }
        fn is_nan(&mut self, a: u32) -> u32 {
            self.push(Ins::IsNan(a))
        }
        fn io_load(&mut self, dword: u32) -> u32 {
            self.push(Ins::IoLoad(dword))
        }
        fn res_store(&mut self, dword: u32, value: u32) {
            self.push(Ins::ResStore(dword, value));
        }
        fn mem_load(&mut self, addr: u32) -> u32 {
            self.push(Ins::MemLoad(addr))
        }
        fn mem_store(&mut self, addr: u32, value: u32) {
            self.push(Ins::MemStore(addr, value));
        }
        fn br(&mut self, target: usize) {
            self.blocks[self.cur].1 = Term::Br(target);
        }
        fn br_if(&mut self, c: u32, then: usize, otherwise: usize) {
            self.blocks[self.cur].1 = Term::BrIf(c, then, otherwise);
        }
        fn ret(&mut self) {
            self.blocks[self.cur].1 = Term::Ret;
        }
    }

    /// binary16 in the low 16 bits → f32 (`f16tof32`)
    pub(crate) fn f16_to_f32(h: u32) -> f32 {
        let (s, e, m) = (h >> 15 & 1, h >> 10 & 0x1F, h & 0x3FF);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 => f32::from_bits(0x7F80_0000 | m << 13),
            _ => f32::from_bits((e + 112) << 23 | m << 13),
        };
        if s == 1 { -v } else { v }
    }

    /// One lane's position: block, instruction index, the block it came from, its values.
    struct LaneState {
        block: usize,
        at: usize,
        from: usize,
        env: HashMap<u32, u32>,
        done: bool,
    }

    impl Interp {
        /// A value in a lane: what it computed, or 0 for `undef` (a marker of the lowering).
        fn value(&self, env: &HashMap<u32, u32>, v: u32) -> u32 {
            let v = self.res(v);
            if self.undef.values().any(|&u| u == v) {
                return 0;
            }
            *env.get(&v).unwrap_or_else(|| panic!("value {v} not defined"))
        }
    }

    impl Interp {
        /// Runs the program over `lanes` lanes; `io` in, `res` out. Ballots wait for every lane.
        fn run(&self, lanes: usize, io: &[u32], res: &mut [u32], mem: &mut [u32]) {
            self.run_with(lanes, 0, [0; 3], io, res, mem)
        }

        /// One wave of a compute group: its first thread `base`, the group `group`.
        fn run_with(&self, lanes: usize, base: u32, group: [u32; 3], io: &[u32], res: &mut [u32], mem: &mut [u32]) {
            let mut ls: Vec<LaneState> = (0..lanes).map(|_| LaneState { block: 0, at: 0, from: 0, env: HashMap::new(), done: false }).collect();
            let mut steps = 0usize;
            loop {
                // run every lane up to a ballot or its end
                for (lane, s) in ls.iter_mut().enumerate() {
                    while !s.done {
                        steps += 1;
                        assert!(steps < 10_000_000, "runaway program");
                        let (ins, term) = &self.blocks[s.block];
                        if s.at == 0 {
                            // phis read their operands from the edge taken, all at once
                            let vals: Vec<(u32, u32)> = ins
                                .iter()
                                .filter_map(|(id, i)| match i {
                                    Ins::Phi(ops) => {
                                        let (v, _) = ops.iter().find(|(_, p)| *p == s.from).unwrap_or_else(|| panic!("phi {id} in block {} has no operand from {}", s.block, s.from));
                                        Some((*id, self.value(&s.env, *v)))
                                    }
                                    _ => None,
                                })
                                .collect();
                            s.env.extend(vals);
                        }
                        if s.at < ins.len() {
                            let (id, i) = &ins[s.at];
                            let g = |v: &u32| self.value(&s.env, *v);
                            let r = match i {
                                Ins::Phi(_) => None,
                                Ins::Int(x) => Some(*x),
                                Ins::Bool(x) => Some(*x as u32),
                                Ins::Bin(op, a, b) => {
                                    let (a, b) = (g(a), g(b));
                                    Some(match op {
                                        Bin::Add => a.wrapping_add(b),
                                        Bin::Sub => a.wrapping_sub(b),
                                        Bin::Mul => a.wrapping_mul(b),
                                        Bin::And => a & b,
                                        Bin::Or => a | b,
                                        Bin::Xor => a ^ b,
                                        Bin::Shl => a << (b & 31),
                                        Bin::LShr => a >> (b & 31),
                                        Bin::AShr => ((a as i32) >> (b & 31)) as u32,
                                    })
                                }
                                Ins::Icmp(c, a, b) => {
                                    let (a, b) = (g(a), g(b));
                                    let (sa, sb) = (a as i32, b as i32);
                                    Some(match c {
                                        Cmp::Eq => a == b,
                                        Cmp::Ne => a != b,
                                        Cmp::Ult => a < b,
                                        Cmp::Ule => a <= b,
                                        Cmp::Ugt => a > b,
                                        Cmp::Uge => a >= b,
                                        Cmp::Slt => sa < sb,
                                        Cmp::Sle => sa <= sb,
                                        Cmp::Sgt => sa > sb,
                                        Cmp::Sge => sa >= sb,
                                    } as u32)
                                }
                                Ins::Select(c, a, b) => Some(if g(c) != 0 { g(a) } else { g(b) }),
                                Ins::Zext(b) => Some(g(b)),
                                Ins::Lane => Some(lane as u32),
                                Ins::GroupId(k) => Some(group[*k as usize]),
                                Ins::ThreadId(k) => Some(if *k == 0 { base + lane as u32 } else { 0 }),
                                Ins::Ballot(..) | Ins::BallotHalf(..) | Ins::ReadLane(..) | Ins::ReadLaneHalf(..) => break,
                                Ins::CountBits(v) => Some(g(v).count_ones()),
                                Ins::FBin(op, a, b) => {
                                    let (a, b) = (f32::from_bits(g(a)), f32::from_bits(g(b)));
                                    // f32::min / max give the operand that is not NaN, as FMin / FMax
                                    Some(match op {
                                        FBin::Add => a + b,
                                        FBin::Sub => a - b,
                                        FBin::Mul => a * b,
                                        FBin::Div => a / b,
                                        FBin::Min => a.min(b),
                                        FBin::Max => a.max(b),
                                    }.to_bits())
                                }
                                Ins::FUn(op, a) => {
                                    let g_bits = g(a);
                                    let a = f32::from_bits(g_bits);
                                    Some(match op {
                                        FUn::Abs => a.abs(),
                                        FUn::Neg => -a,
                                        FUn::Sat => if a.is_nan() { 0.0 } else { a.clamp(0.0, 1.0) },
                                        FUn::Floor => a.floor(),
                                        FUn::F16ToF32 => f16_to_f32(g_bits),
                                        FUn::Ceil => a.ceil(),
                                        FUn::Trunc => a.trunc(),
                                        FUn::RoundNe => a.round_ties_even(),
                                        // the GPU's are approximate: tests compare these with a tolerance
                                        FUn::Sqrt => a.sqrt(),
                                        FUn::Rsq => 1.0 / a.sqrt(),
                                        FUn::Exp => a.exp2(),
                                        FUn::Log => a.log2(),
                                        FUn::Sin => a.sin(),
                                        FUn::Cos => a.cos(),
                                    }.to_bits())
                                }
                                // out of range only where the lowering selects something else
                                Ins::Cvt(c, a) => {
                                    let a = g(a);
                                    Some(match c {
                                        Cvt::SiToF => (a as i32 as f32).to_bits(),
                                        Cvt::UiToF => (a as f32).to_bits(),
                                        Cvt::FToSi => f32::from_bits(a) as i32 as u32,
                                        Cvt::FToUi => f32::from_bits(a) as u32,
                                    })
                                }
                                Ins::FCmp(c, a, b) => {
                                    let (a, b) = (f32::from_bits(g(a)), f32::from_bits(g(b)));
                                    Some(match c {
                                        FCmp::Lt => a < b,
                                        FCmp::Eq => a == b,
                                        FCmp::Le => a <= b,
                                        FCmp::Gt => a > b,
                                        FCmp::Ge => a >= b,
                                    } as u32)
                                }
                                Ins::IsNan(a) => Some(f32::from_bits(g(a)).is_nan() as u32),
                                // the product rounded first: tests keep to products that are exact
                                Ins::FMad(a, b, c) => Some((f32::from_bits(g(a)) * f32::from_bits(g(b)) + f32::from_bits(g(c))).to_bits()),
                                Ins::IoLoad(a) => Some(io[g(a) as usize]),
                                Ins::MemLoad(a) => Some(mem.get((g(a) / 4) as usize).copied().unwrap_or(0)),
                                Ins::MemStore(a, v) => {
                                    let (a, v) = (g(a), g(v));
                                    if let Some(w) = mem.get_mut((a / 4) as usize) {
                                        *w = v;
                                    }
                                    None
                                }
                                Ins::ResStore(a, v) => {
                                    res[g(a) as usize] = g(v);
                                    None
                                }
                            };
                            if let Some(r) = r {
                                s.env.insert(*id, r);
                            }
                            s.at += 1;
                            continue;
                        }
                        match term {
                            Term::Br(t) => {
                                s.from = s.block;
                                s.block = *t;
                            }
                            Term::BrIf(c, t, f) => {
                                s.from = s.block;
                                s.block = if s.env[&self.res(*c)] != 0 { *t } else { *f };
                            }
                            Term::Ret => s.done = true,
                            Term::None => panic!("block {} has no terminator", s.block),
                        }
                        s.at = 0;
                    }
                }
                if ls.iter().all(|s| s.done) {
                    return;
                }
                // the lanes waiting at the first wave operation (in block order: a loop's body
                // before what follows it) run it together, as the lanes active there
                let (b, at) = ls.iter().filter(|s| !s.done).map(|s| (s.block, s.at)).min().unwrap();
                let group: Vec<usize> = (0..ls.len()).filter(|&l| !ls[l].done && (ls[l].block, ls[l].at) == (b, at)).collect();
                let (id, ins) = &self.blocks[b].0[at];
                let out: Vec<u32> = match ins {
                    Ins::Ballot(c, half) | Ins::BallotHalf(c, half) => {
                        let mut bits = 0u64;
                        for &l in &group {
                            if self.value(&ls[l].env, *c) != 0 {
                                bits |= 1 << l;
                            }
                        }
                        let word = (bits >> (32 * half)) as u32;
                        // the half-local form: the other half's word is garbage, so a lowering
                        // that reads it gets a wrong result here
                        let local = matches!(ins, Ins::BallotHalf(..));
                        group.iter().map(|&l| if local && l / 32 != *half { 0xA5A5_0000 | l as u32 } else { word }).collect()
                    }
                    Ins::ReadLane(v, from) | Ins::ReadLaneHalf(v, from) => group
                        .iter()
                        .map(|&l| {
                            let src = self.value(&ls[l].env, *from) as usize;
                            assert!(group.contains(&src), "lane {l} reads lane {src}, which is elsewhere");
                            if matches!(ins, Ins::ReadLaneHalf(..)) {
                                assert_eq!(src / 32, l / 32, "lane {l} reads lane {src} of the other half");
                            }
                            self.value(&ls[src].env, *v)
                        })
                        .collect(),
                    _ => unreachable!(),
                };
                for (&l, x) in group.iter().zip(out) {
                    ls[l].env.insert(*id, x);
                    ls[l].at += 1;
                }
            }
        }
    }

    pub(crate) type I = (&'static str, &'static str, &'static [(&'static str, u64)]);
    const V: u64 = 256;
    const ENDPGM: I = ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]);

    // wave32 programs shared by the CPU tests and the comparison with the HLSL path
    pub(crate) const STRAIGHT: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 133)]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 0), ("SSRC1", 0)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 1)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V), ("VSRC1", 1)]),
        ENDPGM,
    ];
    /// v_cmp_gt_u32 vcc, 16, v0; s_and_saveexec s0, vcc; s_cbranch_execz L; v_mov v1, 7;
    /// L: s_or exec, exec, s0; v_add v2, v1, 1
    pub(crate) const SKIPPED_IF: &[I] = &[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129), ("VSRC1", 1)]),
        ENDPGM,
    ];
    /// s_mov s0, exec; L: v_add v1, v1, 1; v_cmp_ge_u32 vcc, v1, v0; exec &= ~vcc;
    /// s_cbranch_execnz L; exec |= s0  →  v1 = max(1, lane)
    pub(crate) const LEAVING_LOOP: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        ("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 0)]),
        ("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
        ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ENDPGM,
    ];
    /// s1 = 0; L: s1 += 3; v1 += 1; s_cmp_lt_u32 s1, 12; s_cbranch_scc1 L
    pub(crate) const UNIFORM_LOOP: &[I] = &[
        ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 128)]),
        ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 131)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        ("S_CMP_LT_U32", "ENC_SOPC", &[("SSRC0", 1), ("SSRC1", 140)]),
        ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
        ENDPGM,
    ];

    /// v1 = 4·lane; s[4:5] = mem[2..4]; v2 = mem[lane] + s4; then lanes below 16 store v2 at
    /// mem[64 + lane] (a guarded store: no skip branch), and every lane stores v0 at mem[128 + lane]
    pub(crate) const MEMORY: &[I] = &[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)]),
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 8), ("SOFFSET", 125)]),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 1), ("SADDR", 125), ("OFFSET", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 4), ("VSRC1", 2)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 2), ("SADDR", 125), ("OFFSET", 0x100)]),
        ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 6)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 0), ("SADDR", 125), ("OFFSET", 0x200)]),
        ENDPGM,
    ];

    /// A compute program: 64 threads, user SGPRs s0–s1, the group id x in s2;
    /// mem[64·group + thread] = 3·thread + s0 + group
    pub(crate) const COMPUTE: &[I] = &[
        ("S_MUL_I32", "ENC_SOP2", &[("SDST", 3), ("SSRC0", 2), ("SSRC1", 192)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 3), ("VSRC1", 0)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 130), ("VSRC1", 2)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", V + 1), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 0), ("VSRC1", 1)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 2), ("VSRC1", 1)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 2), ("DATA", 1), ("SADDR", 125), ("OFFSET", 0)]),
        ENDPGM,
    ];

    /// f32 VOP2 / VOP3 arithmetic on v1 = 2^lane and v2 = 2^lane · (1 + lane / 64), with the
    /// VOP3 modifiers, and v19 = NaN (0x7FFFFFFF) into min / max / clamp. Every product is exact,
    /// so `FMad` gives the same fused or not.
    pub(crate) const F32_ARITHMETIC: &[I] = &[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 192), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 191), ("VSRC1", 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 151), ("VSRC1", 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 145), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", V + 1), ("VSRC1", 2)]),
        ("V_LSHRREV_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", 129), ("SRC1", 193)]),
        // 0.5 + x, x - p, x - (-4), x · x, min / max(2, x)
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 240), ("VSRC1", 2)]),
        ("V_SUB_F32", "ENC_VOP2", &[("VDST", 4), ("SRC0", V + 2), ("VSRC1", 1)]),
        ("V_SUBREV_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 247), ("VSRC1", 2)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 6), ("SRC0", V + 2), ("VSRC1", 2)]),
        ("V_MIN_F32", "ENC_VOP2", &[("VDST", 7), ("SRC0", 244), ("VSRC1", 2)]),
        ("V_MAX_F32", "ENC_VOP2", &[("VDST", 8), ("SRC0", 244), ("VSRC1", 2)]),
        // fma(x, -2, p), mad(x, 0.5, v3); mac v11 = 4 · x + p, fmac v12 = -0.5 · x + v3
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 2), ("SRC1", 245), ("SRC2", V + 1)]),
        ("V_MAD_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 2), ("SRC1", 240), ("SRC2", V + 3)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 1)]),
        ("V_MAC_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 246), ("VSRC1", 2)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 3)]),
        ("V_FMAC_F32", "ENC_VOP2", &[("VDST", 12), ("SRC0", 241), ("VSRC1", 2)]),
        // -|x| · |-4| (abs before neg); omod 0.5, 2, 4 (with neg on src2); clamp at 1 and at 0
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 2), ("SRC1", 247), ("ABS", 3), ("NEG", 1)]),
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 2), ("SRC1", V + 1), ("OMOD", 3)]),
        ("V_SUB_F32", "ENC_VOP3", &[("VDST", 15), ("SRC0", V + 1), ("SRC1", V + 2), ("OMOD", 1)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 1), ("SRC1", V + 1), ("SRC2", V + 2), ("NEG", 4), ("OMOD", 2)]),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 17), ("SRC0", V + 4), ("SRC1", 240), ("CLAMP", 1)]),
        ("V_FMA_F32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 2), ("SRC1", 243), ("SRC2", 244), ("CLAMP", 1)]),
        // min(NaN, x), max(x, -NaN), clamp(NaN · 1); VOP3 fmac v23 = |-0.5| · x + p
        ("V_MIN_F32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V + 19), ("VSRC1", 2)]),
        ("V_MAX_F32", "ENC_VOP3", &[("VDST", 21), ("SRC0", V + 2), ("SRC1", V + 19), ("NEG", 2)]),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 19), ("SRC1", 242), ("CLAMP", 1)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 23), ("SRC0", V + 1)]),
        ("V_FMAC_F32", "ENC_VOP3", &[("VDST", 23), ("SRC0", 241), ("SRC1", V + 2), ("ABS", 1)]),
        ENDPGM,
    ];

    /// f32 compares of a = v3 (NaN in every fourth lane, 2^lane in the others) with 4.0: all 16
    /// predicates into s[2k], two with VOP3 abs / neg, then VOPC into VCC and a cndmask of it,
    /// and v_cmpx narrowing EXEC for a v_mov.
    pub(crate) const F32_COMPARES: &[I] = &[
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 192), ("VSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 191), ("VSRC1", 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 151), ("VSRC1", 1)]),
        ("V_LSHRREV_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", 129), ("SRC1", 193)]),
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 4), ("SRC0", 131), ("VSRC1", 0)]),
        ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 128), ("VSRC1", 4)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 1), ("VSRC1", 19)]),
        ("V_CMP_F_F32", "ENC_VOP3", &[("VDST", 0), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_LT_F32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_EQ_F32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_LE_F32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_GT_F32", "ENC_VOP3", &[("VDST", 8), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_LG_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_GE_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_O_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_U_F32", "ENC_VOP3", &[("VDST", 16), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NGE_F32", "ENC_VOP3", &[("VDST", 18), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NLG_F32", "ENC_VOP3", &[("VDST", 20), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NGT_F32", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NLE_F32", "ENC_VOP3", &[("VDST", 24), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NEQ_F32", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_NLT_F32", "ENC_VOP3", &[("VDST", 28), ("SRC0", V + 3), ("SRC1", 246)]),
        ("V_CMP_TRU_F32", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 3), ("SRC1", 246)]),
        // -|a| == -4; !(|-2| > -a)
        ("V_CMP_EQ_F32", "ENC_VOP3", &[("VDST", 32), ("SRC0", V + 3), ("SRC1", 247), ("ABS", 1), ("NEG", 1)]),
        ("V_CMP_NGT_F32", "ENC_VOP3", &[("VDST", 34), ("SRC0", 245), ("SRC1", V + 3), ("ABS", 1), ("NEG", 2)]),
        // vcc = a < 4, v6 = vcc ? p : 0; exec = a >= 4, v7 = 1.0 there
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 5), ("SRC0", 246)]),
        ("V_CMP_LT_F32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 5)]),
        ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 128), ("VSRC1", 1)]),
        ("V_CMPX_GE_F32", "ENC_VOPC", &[("SRC0", V + 3), ("VSRC1", 5)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 242)]),
        ENDPGM,
    ];

    /// f32 ↔ i32 / u32 / f16 conversions: v2 = lane − 16, v4 = (lane << 26 | lane) rounded,
    /// v6 = (lane − 16) / 4 into the f32 → integer ones, v11 = v4² (out of range) and v19 = NaN
    /// into the saturating ones; binary16 in and out (denormals, ties to even, overflow, NaN),
    /// with OP_SEL halves, omod and clamp.
    pub(crate) const F32_CONVERSIONS: &[I] = &[
        ("V_LSHRREV_B32", "ENC_VOP3", &[("VDST", 19), ("SRC0", 129), ("SRC1", 193)]),
        ("V_SUBREV_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_CVT_F32_I32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 1)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 154), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V), ("VSRC1", 3)]),
        ("V_CVT_F32_U32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 3)]),
        ("V_CVT_F32_I32", "ENC_VOP3", &[("VDST", 5), ("SRC0", V + 1), ("OMOD", 3), ("CLAMP", 1)]),
        ("V_MUL_F32", "ENC_VOP3", &[("VDST", 6), ("SRC0", V + 2), ("SRC1", 240), ("OMOD", 3)]),
        ("V_CVT_I32_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 6)]),
        ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 6)]),
        ("V_CVT_RPI_I32_F32", "ENC_VOP1", &[("VDST", 9), ("SRC0", V + 6)]),
        ("V_CVT_FLR_I32_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 6)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 4), ("VSRC1", 4)]),
        ("V_CVT_I32_F32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 11)]),
        ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 11)]),
        ("V_CVT_I32_F32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 11), ("NEG", 1)]),
        ("V_CVT_U32_F32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 19)]),
        ("V_CVT_I32_F32", "ENC_VOP1", &[("VDST", 16), ("SRC0", V + 19)]),
        // v17 = f16 (lane & 31) << 9 | (lane & 31), the lane in the high half; v20 = it in the
        // high half, the lane in the low one
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 23), ("SRC0", 159), ("VSRC1", 0)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 17), ("SRC0", 137), ("VSRC1", 23)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 17), ("SRC0", V + 23), ("VSRC1", 17)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 20), ("SRC0", 144), ("VSRC1", 17)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 20), ("SRC0", V), ("VSRC1", 20)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 18), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 17), ("SRC0", V + 18), ("VSRC1", 17)]),
        ("V_CVT_F32_F16", "ENC_VOP1", &[("VDST", 21), ("SRC0", V + 17)]),
        ("V_CVT_F32_F16", "ENC_VOP3", &[("VDST", 22), ("SRC0", V + 20), ("OP_SEL", 1)]),
        // v24 = 1 + (lane & 31) · 2^-14: f16 ties at lanes 8 and 24; into v25's low half
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 24), ("SRC0", 137), ("VSRC1", 23)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 24), ("SRC0", 255), ("VSRC1", 24), ("LITERAL", 0x3F80_0000)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", V + 20)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 25), ("SRC0", V + 24)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", V)]),
        ("V_CVT_F16_F32", "ENC_VOP3", &[("VDST", 26), ("SRC0", V + 6), ("OP_SEL", 8)]),
        // f16 denormals (· 2^-20), below the smallest one (· 2^-25), NaN, overflow, clamp
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 28), ("SRC0", 255), ("VSRC1", 24), ("LITERAL", 0x3580_0000)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 27), ("SRC0", V + 28)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 28), ("SRC0", 255), ("VSRC1", 24), ("LITERAL", 0x3300_0000)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 29), ("SRC0", V + 28)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 30), ("SRC0", V + 19)]),
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 31), ("SRC0", V + 11)]),
        ("V_CVT_F16_F32", "ENC_VOP3", &[("VDST", 32), ("SRC0", V + 6), ("CLAMP", 1)]),
        ENDPGM,
    ];

    /// floor / ceil / trunc / rndne / fract of v3 = (lane − 16) / 8 (ties, signed zeros), fract
    /// of −2^-30 (just below 1.0), and two with VOP3 modifiers.
    pub(crate) const F32_ROUNDING: &[I] = &[
        ("V_SUBREV_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 144), ("VSRC1", 0)]),
        ("V_CVT_F32_I32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 1)]),
        ("V_MUL_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 2), ("LITERAL", 0x3E00_0000)]),
        ("V_FLOOR_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 3)]),
        ("V_CEIL_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 3)]),
        ("V_TRUNC_F32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 3)]),
        ("V_RNDNE_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 3)]),
        ("V_FRACT_F32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 3)]),
        ("V_FRACT_F32", "ENC_VOP1", &[("VDST", 9), ("SRC0", 255), ("LITERAL", 0xB080_0000)]),
        ("V_FLOOR_F32", "ENC_VOP3", &[("VDST", 10), ("SRC0", V + 3), ("NEG", 1), ("OMOD", 1)]),
        ("V_FRACT_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 3), ("ABS", 1), ("CLAMP", 1)]),
        ENDPGM,
    ];

    /// rcp / rsq / sqrt / log of v1 = 1 + (lane & 31) / 32, exp / sin / cos of v2 = v1 − 1.5,
    /// and three with VOP3 modifiers (log2 |0| = −∞ at lane 16). The GPU's are approximate.
    pub(crate) const F32_TRANSCENDENTALS: &[I] = &[
        ("V_AND_B32", "ENC_VOP2", &[("VDST", 23), ("SRC0", 159), ("VSRC1", 0)]),
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 146), ("VSRC1", 23)]),
        ("V_OR_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 255), ("VSRC1", 1), ("LITERAL", 0x3F80_0000)]),
        ("V_ADD_F32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 1), ("SRC1", 243)]),
        ("V_ADD_F32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 241), ("VSRC1", 2)]),
        ("V_RCP_F32", "ENC_VOP1", &[("VDST", 3), ("SRC0", V + 1)]),
        ("V_RCP_IFLAG_F32", "ENC_VOP1", &[("VDST", 4), ("SRC0", V + 1)]),
        ("V_RSQ_F32", "ENC_VOP1", &[("VDST", 5), ("SRC0", V + 1)]),
        ("V_SQRT_F32", "ENC_VOP1", &[("VDST", 6), ("SRC0", V + 1)]),
        ("V_EXP_F32", "ENC_VOP1", &[("VDST", 7), ("SRC0", V + 2)]),
        ("V_LOG_F32", "ENC_VOP1", &[("VDST", 8), ("SRC0", V + 1)]),
        ("V_SIN_F32", "ENC_VOP1", &[("VDST", 9), ("SRC0", V + 2)]),
        ("V_COS_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V + 2)]),
        ("V_EXP_F32", "ENC_VOP3", &[("VDST", 11), ("SRC0", V + 2), ("NEG", 1), ("OMOD", 1)]),
        ("V_SIN_F32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 2), ("CLAMP", 1)]),
        ("V_LOG_F32", "ENC_VOP3", &[("VDST", 13), ("SRC0", V + 2), ("ABS", 1)]),
        ENDPGM,
    ];

    pub(crate) fn compute_cs() -> crate::gpu::recomp::CsInterface {
        crate::gpu::recomp::CsInterface { threads: [64, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 }
    }

    /// A compute program run on the CPU over `groups` groups (wave32): the memory after.
    pub(crate) fn run_compute(lines: &[I], cs: &crate::gpu::recomp::CsInterface, groups: u32, sgprs: &[u32], mem: &[u32]) -> Vec<u32> {
        let ir = lower_compute(&decode(lines), cs, false, Interp::default()).unwrap_or_else(|e| panic!("{e}"));
        let mut io = vec![0u32; KERNEL_SGPRS];
        io[..sgprs.len()].copy_from_slice(sgprs);
        let mut res = vec![0u32; KERNEL_SGPRS];
        let mut mem = mem.to_vec();
        for g in 0..groups {
            for w in 0..cs.threads[0].div_ceil(32) {
                ir.run_with(32, w * 32, [g, 0, 0], &io, &mut res, &mut mem);
            }
        }
        mem
    }

    pub(crate) fn memory_image() -> Vec<u32> {
        (0..256).map(|i| 1000 + i).collect()
    }

    /// A line's bytes; a `("LITERAL", k)` field is the instruction's literal dword.
    pub(crate) fn encode_line((n, e, f): &I) -> Vec<u8> {
        let lit = f.iter().find(|(k, _)| *k == "LITERAL").map(|&(_, v)| v as u32);
        let f: Vec<(&str, u64)> = f.iter().copied().filter(|(k, _)| *k != "LITERAL").collect();
        rdna2::encode(n, e, &f, lit).unwrap()
    }

    pub(crate) fn decode(lines: &[I]) -> Vec<Decoded> {
        let bytes: Vec<u8> = lines.iter().flat_map(encode_line).collect();
        let mut out = Vec::new();
        let mut off = 0;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        out
    }

    /// The kernel run on the CPU: initial (sgpr, vgpr[lane]) → final ones.
    fn run(lines: &[I], wave64: bool, sgpr: &[u32], vgpr: &[Vec<u32>]) -> (Vec<u32>, Vec<Vec<u32>>) {
        let (s, v, _) = run_mem(lines, wave64, sgpr, vgpr, &[]);
        (s, v)
    }

    /// … with the flat memory too: (sgpr, vgpr[lane], memory after).
    pub(crate) fn run_mem(lines: &[I], wave64: bool, sgpr: &[u32], vgpr: &[Vec<u32>], mem: &[u32]) -> (Vec<u32>, Vec<Vec<u32>>, Vec<u32>) {
        let insts = decode(lines);
        let (ir, _) = lower_kernel(&insts, wave64, Interp::default()).unwrap_or_else(|e| panic!("{e}"));
        let lanes = if wave64 { 64 } else { 32 };
        let mut io = vec![0u32; KERNEL_SGPRS + 64 * KERNEL_VGPRS];
        io[..sgpr.len()].copy_from_slice(sgpr);
        for (l, regs) in vgpr.iter().enumerate().take(lanes) {
            io[KERNEL_SGPRS + l * KERNEL_VGPRS..][..regs.len()].copy_from_slice(regs);
        }
        let mut res = io.clone();
        let mut mem = mem.to_vec();
        ir.run(lanes, &io, &mut res, &mut mem);
        let v = (0..lanes).map(|l| res[KERNEL_SGPRS + l * KERNEL_VGPRS..][..KERNEL_VGPRS].to_vec()).collect();
        (res[..KERNEL_SGPRS].to_vec(), v, mem)
    }

    pub(crate) fn state(lanes: usize) -> (Vec<u32>, Vec<Vec<u32>>) {
        let mut s = vec![0u32; KERNEL_SGPRS];
        s[126] = u32::MAX;
        s[127] = u32::MAX;
        // v0 = the lane, v1 = 0
        (s, (0..lanes).map(|l| { let mut v = vec![0u32; 8]; v[0] = l as u32; v }).collect())
    }

    #[test]
    fn straight_line_scalar_and_vector_arithmetic() {
        let (s, v) = state(32);
        let (s, v) = run(STRAIGHT, false, &s, &v);
        assert_eq!((s[0], s[1]), (5, 10));
        for (l, regs) in v.iter().enumerate() {
            assert_eq!((regs[1], regs[2]), (10, 10 + l as u32), "lane {l}");
        }
    }

    #[test]
    fn a_skipped_if_runs_only_its_lanes() {
        let (s, v) = state(32);
        let (s, v) = run(SKIPPED_IF, false, &s, &v);
        for (l, regs) in v.iter().enumerate() {
            let want = if l < 16 { 7 } else { 0 };
            assert_eq!((regs[1], regs[2]), (want, want + 1), "lane {l}");
        }
        assert_eq!(s[106], 0xFFFF, "vcc: the lanes below 16");
        assert_eq!((s[0], s[126]), (u32::MAX, u32::MAX), "saved and restored EXEC");
    }

    #[test]
    fn lanes_leave_a_loop_one_by_one() {
        let (s, v) = state(32);
        let (s, v) = run(LEAVING_LOOP, false, &s, &v);
        for (l, regs) in v.iter().enumerate() {
            assert_eq!(regs[1], (l as u32).max(1), "lane {l}");
        }
        assert_eq!(s[126], u32::MAX);
    }

    #[test]
    fn a_uniform_loop_with_a_scalar_counter() {
        let (s, v) = state(32);
        let (s, v) = run(UNIFORM_LOOP, false, &s, &v);
        assert_eq!(s[1], 12);
        assert!(v.iter().all(|r| r[1] == 4));
    }

    #[test]
    fn lanes_outside_exec_at_launch_keep_their_registers() {
        let (mut s, v) = state(32);
        s[126] = 0x0000_00FF;
        let (s, v) = run(&[("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 137)]), ENDPGM], false, &s, &v);
        for (l, regs) in v.iter().enumerate() {
            assert_eq!(regs[1], if l < 8 { 9 } else { 0 }, "lane {l}");
        }
        assert_eq!(s[126], 0xFF);
    }

    #[test]
    fn wave64_masks_span_both_words() {
        // v_cmp_gt_u32 vcc, 40, v0 over 64 lanes: lanes 0–39
        let (s, v) = state(64);
        let (s, _) = run(&[("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 168), ("VSRC1", 0)]), ENDPGM], true, &s, &v);
        assert_eq!((s[106], s[107]), (u32::MAX, 0xFF));
    }

    /// The lowering run on the CPU and the HLSL path run on the GPU give the same registers.
    #[test]
    fn the_lowering_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Options};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        for (name, lines) in [("straight", STRAIGHT), ("skipped if", SKIPPED_IF), ("leaving loop", LEAVING_LOOP), ("uniform loop", UNIFORM_LOOP), ("memory", MEMORY)] {
            let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
            let (s, v) = state(32);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let t = translate(&bytes, &Options { wave64: false, ..Default::default() }).unwrap();
            assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
            let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap();
            let (cs, cv, cmem) = run_mem(lines, false, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let insts = decode(lines);
            // masks some lanes kept their own older bits of (`lower_kernel`'s `stale`): only the
            // kernel ABI's final store sees them
            let (_, stale) = lower_kernel(&insts, false, Interp::default()).unwrap();
            let mut sregs: Vec<usize> = vec![106, 126];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(cs[r], gpu.sgpr[r], "{name}: s{r}");
            }
            for l in 0..32 {
                for &r in &vregs {
                    assert_eq!(cv[l][r], gpu.vgpr[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
    }

    // wave64 versions of the programs above: 64-bit EXEC / VCC / saved masks, and thresholds
    // past lane 32 so that both words of every mask matter
    /// v_cmp_gt_u32 vcc, 40, v0; s_and_saveexec_b64 s[0:1], vcc; s_cbranch_execz L; v_mov v1, 7;
    /// L: s_or_b64 exec, exec, s[0:1]; v_add v2, v1, 1
    const SKIPPED_IF_W64: &[I] = &[
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 168), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
        ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 1)]),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 135)]),
        ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 129), ("VSRC1", 1)]),
        ENDPGM,
    ];
    /// s_mov_b64 s[0:1], exec; L: v_add v1, v1, 1; v_cmp_ge_u32 vcc, v1, v0;
    /// exec &= ~vcc (b64); s_cbranch_execnz L; exec |= s[0:1]  →  v1 = max(1, lane), 64 lanes
    const LEAVING_LOOP_W64: &[I] = &[
        ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        ("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 0)]),
        ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
        ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFC)]),
        ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
        ENDPGM,
    ];
    /// `MEMORY` over 64 lanes: lanes below 40 take the guarded store
    const MEMORY_W64: &[I] = &[
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 130), ("VSRC1", 0)]),
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 0), ("OFFSET", 8), ("SOFFSET", 125)]),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", &[("VDST", 2), ("ADDR", 1), ("SADDR", 125), ("OFFSET", 0)]),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 4), ("VSRC1", 2)]),
        ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 168), ("VSRC1", 0)]),
        ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 106)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 2), ("SADDR", 125), ("OFFSET", 0x100)]),
        ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 6)]),
        ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 1), ("DATA", 0), ("SADDR", 125), ("OFFSET", 0x200)]),
        ENDPGM,
    ];

    #[test]
    fn a_wave64_mask_left_behind_is_reported_in_both_halves() {
        // VCC from the loop's last v_cmp: lanes that left earlier kept their own bits in both words
        let (_, stale) = lower_kernel(&decode(LEAVING_LOOP_W64), true, Interp::default()).unwrap();
        assert_eq!(stale, vec![106, 107]);
    }

    /// The wave64 lowering run on the CPU and the HLSL path's native wave64 form run on a GPU
    /// with 64-lane waves give the same registers and memory.
    #[test]
    fn the_wave64_lowering_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Options};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        for (name, lines) in [
            ("straight", STRAIGHT),
            ("skipped if", SKIPPED_IF_W64),
            ("leaving loop", LEAVING_LOOP_W64),
            ("uniform loop", UNIFORM_LOOP),
            ("memory", MEMORY_W64),
        ] {
            let bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
            let (s, v) = state(64);
            let mut st = runner::KernelState::default();
            st.sgpr.copy_from_slice(&s);
            for (l, regs) in v.iter().enumerate() {
                st.vgpr[l][..regs.len()].copy_from_slice(regs);
            }
            let t = translate(&bytes, &Options { wave64: true, split_wave64: false, ..Default::default() }).unwrap();
            assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
            let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
            let mem = memory_image();
            let gpu = runner::run(&dxil, &st, &mem).unwrap();
            let (cs, cv, cmem) = run_mem(lines, true, &s, &v, &mem);
            assert_eq!(&gpu.mem[..cmem.len()], &cmem[..], "{name}: memory");
            let insts = decode(lines);
            let (_, stale) = lower_kernel(&insts, true, Interp::default()).unwrap();
            let mut sregs: Vec<usize> = vec![106, 107, 126, 127];
            let mut vregs: Vec<usize> = Vec::new();
            for d in &insts {
                let (r, w) = rdna2::sgpr_uses(d);
                sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
                let (r, w) = rdna2::vgpr_uses(d);
                vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
            }
            for &r in sregs.iter().filter(|&&r| !stale.contains(&(r as u8))) {
                assert_eq!(cs[r], gpu.sgpr[r], "{name}: s{r}");
            }
            for l in 0..64 {
                for &r in &vregs {
                    assert_eq!(cv[l][r], gpu.vgpr[l][r], "{name}: lane {l} v{r}");
                }
            }
        }
    }

    #[test]
    fn scalar_and_vector_memory_with_a_guarded_store() {
        let (s, v) = state(32);
        let (s, v, mem) = run_mem(MEMORY, false, &s, &v, &memory_image());
        assert_eq!((s[4], s[5]), (1002, 1003));
        for l in 0..32 {
            assert_eq!(v[l][2], 2002 + l as u32, "lane {l}");
            assert_eq!(mem[64 + l], if l < 16 { 2002 + l as u32 } else { 1064 + l as u32 }, "guarded store, lane {l}");
            assert_eq!(mem[128 + l], l as u32, "store, lane {l}");
        }
        // the store under the narrowed EXEC is the guarded one
        let insts = decode(MEMORY);
        let sum = lane_masks::analyze(&insts, false).unwrap();
        assert_eq!(sum.guarded.iter().filter(|&&pc| insts.iter().any(|d| d.pc == pc && is_store(d))).count(), 1);
    }

    #[test]
    fn a_compute_program_takes_its_group_and_thread_ids() {
        let mem = run_compute(COMPUTE, &compute_cs(), 3, &[100, 0], &vec![0u32; 256]);
        for g in 0..3u32 {
            for t in 0..64u32 {
                assert_eq!(mem[(64 * g + t) as usize], 3 * t + 100 + g, "group {g} thread {t}");
            }
        }
        // 2-D groups are not lowered yet
        let cs = crate::gpu::recomp::CsInterface { threads: [8, 8, 1], ..compute_cs() };
        assert!(lower_compute(&decode(COMPUTE), &cs, false, Interp::default()).is_err());
    }

    /// The compute program through the HLSL path on the GPU and through the lowering on the CPU.
    #[test]
    fn a_compute_program_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner, translate, Mode, Options};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        let bytes: Vec<u8> = COMPUTE.iter().flat_map(|(n, e, f)| rdna2::encode(n, e, f, None).unwrap()).collect();
        let cs = compute_cs();
        let t = translate(&bytes, &Options { mode: Mode::Compute(cs.clone()), wave64: false, ..Default::default() }).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
        let mem = vec![0u32; 256];
        let gpu = runner::dispatch(&dxil, [3, 1, 1], &[100, 0], &mem).unwrap();
        let cpu = run_compute(COMPUTE, &cs, 3, &[100, 0], &mem);
        assert_eq!(&gpu[..cpu.len()], &cpu[..]);
    }

    #[test]
    fn a_mask_left_behind_by_lanes_that_skipped_its_write() {
        // VCC written inside the loop that lanes leave one by one: the console's last write left
        // the earlier leavers' bits 0, each lane here kept its own. Stored at the end: reported.
        let insts = decode(LEAVING_LOOP);
        let (_, stale) = lower_kernel(&insts, false, Interp::default()).unwrap();
        assert_eq!(stale, vec![106]);
        // read after the loop (v_cndmask on VCC): refused
        let mut lines = LEAVING_LOOP.to_vec();
        lines.insert(6, ("V_CNDMASK_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 128), ("VSRC1", 0)]));
        let err = lower_kernel(&decode(&lines), false, Interp::default()).err().expect("refused");
        assert!(err.contains("some lanes were elsewhere"), "{err}");
        // a mask written before the lanes split is read after they meet: fine
        let insts = decode(SKIPPED_IF);
        assert_eq!(lower_kernel(&insts, false, Interp::default()).unwrap().1, Vec::<u8>::new());
    }

    #[test]
    fn a_scalar_counted_in_a_loop_lanes_leave_is_not_each_lanes_own() {
        // s_add_u32 s1, s1, 1 in the loop lanes leave one by one: the console counts the wave's
        // passes, each lane here only its own (a waterfall loop's key is the same case)
        // s_mov s0, exec; s1 = 0; L: v_add v1, v1, 1; s1 += 1; v_cmp_ge vcc, v1, v0;
        // exec &= ~vcc; s_cbranch_execnz L; exec |= s0
        let mut lines: Vec<I> = vec![
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 1), ("SSRC0", 128)]),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 1), ("SSRC0", 1), ("SSRC1", 129)]),
            ("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 0)]),
            ("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)]),
            ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFB)]),
            ("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            ENDPGM,
        ];
        let (_, stale) = lower_kernel(&decode(&lines), false, Interp::default()).unwrap();
        assert!(stale.contains(&1), "{stale:?}");
        // read after the loop: refused
        lines.insert(8, ("S_MOV_B32", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 1)]));
        let err = lower_kernel(&decode(&lines), false, Interp::default()).err().expect("refused");
        assert!(err.contains("some lanes were elsewhere"), "{err}");
    }

    #[test]
    fn f32_arithmetic_with_vop3_modifiers() {
        let sat = |x: f32| x.clamp(0.0, 1.0);
        let (s, v) = state(32);
        let (_, v) = run(F32_ARITHMETIC, false, &s, &v);
        for (l, r) in v.iter().enumerate() {
            let f = |i: usize| f32::from_bits(r[i]);
            let (p, x) = (f(1), f(2));
            assert_eq!((p, x), (2f32.powi(l as i32), 2f32.powi(l as i32) * (1.0 + l as f32 / 64.0)), "lane {l}");
            let y = x + 0.5;
            let want = [y, x - p, x + 4.0, x * x, x.min(2.0), x.max(2.0), x * -2.0 + p, x * 0.5 + y, 4.0 * x + p, -0.5 * x + y, -x * 4.0, (x + p) * 0.5, (p - x) * 2.0, (p * p - x) * 4.0, sat((x - p) * 0.5), sat(2.0 - x)];
            for (i, w) in want.iter().enumerate() {
                assert_eq!(f(3 + i).to_bits(), w.to_bits(), "lane {l} v{}: {} against {w}", 3 + i, f(3 + i));
            }
            assert_eq!((r[20], r[21], r[22]), (r[2], r[2], 0), "lane {l}: NaN into min, max, clamp");
            assert_eq!(f(23), 0.5 * x + p, "lane {l}: VOP3 fmac");
        }
    }

    #[test]
    fn f32_arithmetic_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        agrees_with_hlsl("f32 arithmetic, wave32", F32_ARITHMETIC, false, &[]);
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        agrees_with_hlsl("f32 arithmetic, wave64", F32_ARITHMETIC, true, &[]);
    }

    #[test]
    fn f32_conversions_round_and_saturate() {
        let (s, v) = state(32);
        let (_, v) = run(F32_CONVERSIONS, false, &s, &v);
        for (l, r) in v.iter().enumerate() {
            let f = |i: usize| f32::from_bits(r[i]);
            let (li, q) = (l as i32 - 16, (l as i32 - 16) as f32 / 4.0);
            assert_eq!(f(2), li as f32, "lane {l}");
            assert_eq!(f(4), ((l as u32) << 26 | l as u32) as f32, "lane {l}: u32 rounded to nearest even");
            assert_eq!(f(5), (li as f32 * 0.5).clamp(0.0, 1.0), "lane {l}: omod and clamp");
            assert_eq!(f(6), q, "lane {l}");
            // the HLSL path's cvt_i32 / cvt_u32 saturate as Rust's `as` does, NaN giving 0
            assert_eq!((r[7], r[8]), (q as i32 as u32, q as u32), "lane {l}: truncated");
            assert_eq!((r[9], r[10]), ((q + 0.5).floor() as i32 as u32, q.floor() as i32 as u32), "lane {l}: rpi, flr");
            assert_eq!((r[12], r[13], r[14]), (f(11) as i32 as u32, f(11) as u32, (-f(11)) as i32 as u32), "lane {l}: out of range");
            assert_eq!((r[15], r[16]), (0, 0), "lane {l}: NaN");
            assert_eq!(f(21), f16_to_f32(r[17] & 0xFFFF), "lane {l}");
            assert_eq!(r[22], r[21], "lane {l}: the high half");
            assert_eq!(r[25] >> 16, r[20] >> 16, "lane {l}: the half kept");
            assert_eq!(r[26] & 0xFFFF, l as u32, "lane {l}: the half kept");
            assert_eq!(r[29], (l != 0) as u32, "lane {l}: below the smallest denormal");
            assert_eq!(r[30], 0x7FFF, "lane {l}: NaN");
            assert_eq!(r[31], if l == 0 { 0 } else { 0x7C00 }, "lane {l}: overflow");
        }
        let lo = |l: usize, i: usize| v[l][i] & 0xFFFF;
        assert_eq!([lo(0, 21), lo(1, 17), lo(30, 17)], [0, 0x0201, 0x3C1E]);
        assert_eq!((f32::from_bits(v[1][21]), f32::from_bits(v[30][21])), (513.0 * 2f32.powi(-24), 1.0 + 30.0 / 1024.0), "f16 in");
        // 1 + k · 2^-14 at k = 1, 8, 16, 24, 31: ties to even at 8 and 24
        assert_eq!([1, 8, 16, 24, 31].map(|l| lo(l, 25)), [0x3C00, 0x3C00, 0x3C01, 0x3C02, 0x3C02]);
        assert_eq!([0, 16, 18, 20].map(|l| v[l][26] >> 16), [0xC400, 0, 0x3800, 0x3C00], "f16 into the high half");
        assert_eq!([0, 18, 20, 31].map(|l| lo(l, 32)), [0, 0x3800, 0x3C00, 0x3C00], "clamp");
        assert_eq!(lo(0, 27), 0x0010, "2^-20: an f16 denormal");
    }

    #[test]
    fn f32_conversions_agree_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        agrees_with_hlsl("f32 conversions, wave32", F32_CONVERSIONS, false, &[]);
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        agrees_with_hlsl("f32 conversions, wave64", F32_CONVERSIONS, true, &[]);
    }

    #[test]
    fn f32_rounding() {
        let (s, v) = state(32);
        let (_, v) = run(F32_ROUNDING, false, &s, &v);
        for (l, r) in v.iter().enumerate() {
            let x = (l as i32 - 16) as f32 / 8.0;
            let want = [x.floor(), x.ceil(), x.trunc(), x.round_ties_even(), x - x.floor()];
            for (i, w) in want.iter().enumerate() {
                assert_eq!(r[4 + i], w.to_bits(), "lane {l} v{}", 4 + i);
            }
            assert_eq!(r[9], 0x3F7F_FFFF, "fract(-2^-30): below 1.0");
            assert_eq!(f32::from_bits(r[10]), 2.0 * (-x).floor(), "lane {l}");
            assert_eq!(f32::from_bits(r[11]), x.abs() - x.abs().floor(), "lane {l}");
        }
    }

    #[test]
    fn f32_rounding_agrees_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        agrees_with_hlsl("f32 rounding, wave32", F32_ROUNDING, false, &[]);
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        agrees_with_hlsl("f32 rounding, wave64", F32_ROUNDING, true, &[]);
    }

    #[test]
    fn v_cvt_f16_f32_is_refused_where_the_program_writes_mode() {
        let mode: I = ("S_SETREG_IMM32_B32", "ENC_SOPK", &[("SDST", 0), ("SIMM16", 1 | 3 << 11), ("LITERAL", 4)]);
        let cvt: I = ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 242)]);
        let err = lower_kernel(&decode(&[mode, cvt, ENDPGM]), false, Interp::default()).err().expect("refused");
        assert!(err.contains("MODE"), "{err}");
        assert!(lower_kernel(&decode(&[cvt, ENDPGM]), false, Interp::default()).is_ok());
    }

    #[test]
    fn f32_compares_with_nan_and_vop3_modifiers() {
        let preds: [fn(f32, f32) -> bool; 18] = [
            |_, _| false,
            |a, b| a < b,
            |a, b| a == b,
            |a, b| a <= b,
            |a, b| a > b,
            |a, b| a < b || a > b,
            |a, b| a >= b,
            |a, b| !a.is_nan() && !b.is_nan(),
            |a, b| a.is_nan() || b.is_nan(),
            |a, b| !(a >= b),
            |a, b| !(a < b || a > b),
            |a, b| !(a > b),
            |a, b| !(a <= b),
            |a, b| !(a == b),
            |a, b| !(a < b),
            |_, _| true,
            |a, _| -a.abs() == -4.0,
            |a, _| !(2.0 > -a),
        ];
        let (s, v) = state(32);
        let (s, v) = run(F32_COMPARES, false, &s, &v);
        let a: Vec<f32> = (0..32).map(|l| if l % 4 == 0 { f32::from_bits(0x7FFF_FFFF) } else { 2f32.powi(l) }).collect();
        let mask = |p: &dyn Fn(f32) -> bool| (0..32).filter(|&l| p(a[l])).fold(0u32, |m, l| m | 1 << l);
        for (k, p) in preds.iter().enumerate() {
            assert_eq!(s[2 * k], mask(&|x| p(x, 4.0)), "s{}", 2 * k);
        }
        assert_eq!(s[106], mask(&|x| x < 4.0), "vcc");
        assert_eq!(s[126], mask(&|x| x >= 4.0), "exec");
        for (l, r) in v.iter().enumerate() {
            assert_eq!(r[3], a[l].to_bits(), "lane {l}");
            assert_eq!(r[6], if a[l] < 4.0 { r[1] } else { 0 }, "lane {l}: cndmask");
            assert_eq!(r[7], if a[l] >= 4.0 { 1f32.to_bits() } else { 0 }, "lane {l}: under v_cmpx");
        }
    }

    #[test]
    fn f32_compares_agree_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        agrees_with_hlsl("f32 compares, wave32", F32_COMPARES, false, &[]);
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        agrees_with_hlsl("f32 compares, wave64", F32_COMPARES, true, &[]);
    }

    #[test]
    fn vop3_operand_selects_are_refused() {
        let op_sel: &[I] = &[("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", V), ("OP_SEL", 1)]), ENDPGM];
        let err = lower_kernel(&decode(op_sel), false, Interp::default()).err().expect("refused");
        assert!(err.contains("modifiers"), "{err}");
        let plain: &[I] = &[("V_ADD_NC_U32", "ENC_VOP3", &[("VDST", 1), ("SRC0", V), ("SRC1", V)]), ENDPGM];
        assert!(lower_kernel(&decode(plain), false, Interp::default()).is_ok());
    }

    #[test]
    fn a_mask_read_as_a_number_is_refused() {
        // s_bcnt1-like use: s_add_u32 s2, vcc_lo, 0 after a v_cmp
        let insts = decode(&[
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 144), ("VSRC1", 0)]),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 106), ("SSRC1", 128)]),
            ENDPGM,
        ]);
        assert!(lower_kernel(&insts, false, Interp::default()).is_err());
    }

    /// Cross-lane reads with EXEC narrowed to the lanes above 4 (`v_cmpx`): readfirstlane,
    /// the counts of EXEC as a number, mbcnt, readlane / writelane, DPP16 / DPP8 sources,
    /// permlanex16 and ds_swizzle.
    fn lane_reads(wave64: bool) -> Vec<I> {
        let mut p: Vec<I> = vec![
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", 132)]),
            ("V_CMPX_LT_U32", "ENC_VOPC", &[("SRC0", V + 2), ("VSRC1", 0)]),
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)]),
            ("V_MBCNT_LO_U32_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 126), ("SRC1", 128)]),
            ("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V), ("SRC1", 131)]),
            ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 4), ("SRC0", V + 4), ("SRC1", 137)]),
            ("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 5), ("SRC0", 2), ("SRC1", 130)]),
            // row_shr:1 with bound_ctrl; quad_perm [3,2,1,0] on rows 1 and 3; DPP8 reversed, added
            ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 6), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0x111), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF), ("BOUND_CTRL", 1)]),
            ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 7), ("SRC0", 0xFA), ("VSRC0", 4), ("DPP_CTRL", 0x1B), ("ROW_MASK", 0xA), ("BANK_MASK", 0xF)]),
            ("V_ADD_NC_U32", "VOP2_VOP_DPP8", &[("VDST", 8), ("SRC0", 0xE9), ("VSRC0", 4), ("VSRC1", 0), ("LANE_SEL_0", 7), ("LANE_SEL_1", 6), ("LANE_SEL_2", 5), ("LANE_SEL_3", 4), ("LANE_SEL_4", 3), ("LANE_SEL_5", 2), ("LANE_SEL_6", 1), ("LANE_SEL_7", 0)]),
            ("S_MOVK_I32", "ENC_SOPK", &[("SDST", 5), ("SIMM16", 0x3210)]),
            ("S_MOVK_I32", "ENC_SOPK", &[("SDST", 6), ("SIMM16", 0x7F54)]),
            ("V_PERMLANEX16_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 4), ("SRC1", 5), ("SRC2", 6)]),
            // xor 1 (and 0x1F, or 0, xor 1); quad mode [3,2,1,0]
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 10), ("ADDR", 4), ("OFFSET0", 0x1F), ("OFFSET1", 0x04)]),
            ("DS_SWIZZLE_B32", "ENC_DS", &[("VDST", 11), ("ADDR", 4), ("OFFSET0", 0x1B), ("OFFSET1", 0x80)]),
        ];
        if wave64 {
            p.extend_from_slice(&[
                ("V_MBCNT_HI_U32_B32", "ENC_VOP3", &[("VDST", 3), ("SRC0", 127), ("SRC1", V + 3)]),
                ("S_BCNT1_I32_B64", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 126)]),
                ("S_FF1_I32_B64", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 126)]),
            ]);
        } else {
            p.extend_from_slice(&[
                ("S_BCNT1_I32_B32", "ENC_SOP1", &[("SDST", 3), ("SSRC0", 126)]),
                ("S_FF1_I32_B32", "ENC_SOP1", &[("SDST", 7), ("SSRC0", 126)]),
            ]);
        }
        p.push(ENDPGM);
        p
    }

    /// A waterfall loop: each pass takes the first lane's key (readfirstlane), the lanes with
    /// that key do the work and leave; the readfirstlane is in a block lanes leave one by one.
    fn waterfall(wave64: bool) -> Vec<I> {
        let w: [I; 4] = if wave64 {
            [
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
                ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
                ("S_XOR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 4)]),
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
            ]
        } else {
            [
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 126)]),
                ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
                ("S_XOR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 4)]),
                ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 0)]),
            ]
        };
        vec![
            ("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 131), ("VSRC1", 0)]),
            w[0],
            ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V + 2)]),
            ("V_CMP_EQ_U32", "ENC_VOPC", &[("SRC0", 2), ("VSRC1", 2)]),
            w[1],
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 2), ("VSRC1", 3)]),
            w[2],
            ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFA)]),
            w[3],
            ENDPGM,
        ]
    }

    /// The lowering (on the CPU) and the HLSL path (on the GPU) give the same registers.
    fn agrees_with_hlsl(name: &str, lines: &[I], wave64: bool, skip: &[usize]) {
        use super::super::{dxc, runner, translate, Options};
        let lanes = if wave64 { 64 } else { 32 };
        let bytes: Vec<u8> = lines.iter().flat_map(encode_line).collect();
        let (s, v) = state(lanes);
        let mut st = runner::KernelState::default();
        st.sgpr.copy_from_slice(&s);
        for (l, regs) in v.iter().enumerate() {
            st.vgpr[l][..regs.len()].copy_from_slice(regs);
        }
        let t = translate(&bytes, &Options { wave64, ..Default::default() }).unwrap();
        assert!(t.complete(), "{name}: {:?}", t.unsupported_why);
        let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
        let gpu = runner::run(&dxil, &st, &[]).unwrap();
        let (cs, cv) = run(lines, wave64, &s, &v);
        let insts = decode(lines);
        let (_, stale) = lower_kernel(&insts, wave64, Interp::default()).unwrap();
        let mut sregs: Vec<usize> = vec![106, 126];
        let mut vregs: Vec<usize> = Vec::new();
        for d in &insts {
            let (r, w) = rdna2::sgpr_uses(d);
            sregs.extend(r.iter().chain(&w).map(|&x| x as usize).filter(|&x| x < 125));
            let (r, w) = rdna2::vgpr_uses(d);
            vregs.extend(r.iter().chain(&w).map(|&x| x as usize));
        }
        if wave64 {
            sregs.extend([107, 127]);
        }
        // (a stale wave64 mask is reported by its first register; the pair's second is stale too)
        let stale_at = |r: usize| stale.contains(&(r as u8)) || (wave64 && r > 0 && r % 2 == 1 && stale.contains(&(r as u8 - 1)));
        for &r in sregs.iter().filter(|&&r| !stale_at(r) && !skip.contains(&r)) {
            assert_eq!(cs[r], gpu.sgpr[r], "{name}: s{r}");
        }
        for l in 0..lanes {
            for &r in &vregs {
                assert_eq!(cv[l][r], gpu.vgpr[l][r], "{name}: lane {l} v{r}");
            }
        }
    }

    #[test]
    fn cross_lane_reads_on_the_cpu() {
        for wave64 in [false, true] {
            let lanes = if wave64 { 64 } else { 32 };
            let (s, v) = state(lanes);
            let (s, v) = run(&lane_reads(wave64), wave64, &s, &v);
            assert_eq!(s[2], 5, "readfirstlane: the first lane in EXEC");
            assert_eq!(s[3], lanes as u32 - 5, "s_bcnt1 of EXEC");
            assert_eq!(s[4], 27, "readlane 9 of 3 * lane");
            assert_eq!(s[7], 5, "s_ff1 of EXEC");
            for (l, r) in v.iter().enumerate() {
                let (l32, active) = (l as u32, l > 4);
                assert_eq!(r[3], if active { l32 - 5 } else { 0 }, "lane {l}: mbcnt");
                assert_eq!(r[5], if l == 2 { 5 } else { 0 }, "lane {l}: writelane, whatever EXEC holds");
                // row_shr:1, bound_ctrl: 0 at a row's first lane or from a lane outside EXEC
                let want = if !active || l % 16 == 0 || l - 1 <= 4 { 0 } else { 3 * (l32 - 1) };
                assert_eq!(r[6], want, "lane {l}: DPP16 row_shr");
                let rev = (l32 & !7) | (7 - (l32 & 7));
                let want = if active { l32 + if rev > 4 { 3 * rev } else { 0 } } else { 0 };
                assert_eq!(r[8], want, "lane {l}: DPP8");
                let x = l32 ^ 1;
                assert_eq!(r[10], if active && x > 4 { 3 * x } else { 0 }, "lane {l}: ds_swizzle xor 1");
            }
        }
    }

    #[test]
    fn a_waterfall_loop_reads_the_first_lane_of_each_pass() {
        for wave64 in [false, true] {
            let lanes = if wave64 { 64 } else { 32 };
            let (s, v) = state(lanes);
            let (_, v) = run(&waterfall(wave64), wave64, &s, &v);
            for (l, r) in v.iter().enumerate() {
                assert_eq!(r[3], l as u32 & 3, "lane {l}");
            }
        }
        // a lane of any lane, where some lanes have left: refused
        let mut lines = waterfall(false);
        lines.insert(3, ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", V + 2), ("SRC1", 129)]));
        lines[8] = ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFF8)]);
        let err = lower_kernel(&decode(&lines), false, Interp::default()).err().expect("refused");
        assert!(err.contains("some lanes are elsewhere"), "{err}");
    }

    #[test]
    fn cross_lane_reads_agree_with_the_hlsl_path_on_the_gpu() {
        use super::super::{dxc, runner};
        if runner::adapter().is_err() || dxc::location().is_err() {
            runner::skip_test(if runner::adapter().is_err() { "no GPU" } else { "no DXC" });
            return;
        }
        agrees_with_hlsl("lane reads, wave32", &lane_reads(false), false, &[]);
        // s2 and the saved EXEC s4 after the loop: the last pass's on the console; each lane's own
        // last pass's in the lowering (written where lanes leave one by one; not refused yet)
        agrees_with_hlsl("waterfall, wave32", &waterfall(false), false, &[2, 4]);
        if runner::wave64_supported() != Ok(true) {
            runner::skip_test("no wave64");
            return;
        }
        agrees_with_hlsl("lane reads, wave64", &lane_reads(true), true, &[]);
        agrees_with_hlsl("waterfall, wave64", &waterfall(true), true, &[2, 4, 5]);
    }
}
