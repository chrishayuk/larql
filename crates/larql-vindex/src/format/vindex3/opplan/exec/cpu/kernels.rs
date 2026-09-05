//! The dense projection kernels, each declaring who threads it.
//!
//! None of them spawns. Every one computes exactly the output rows it is
//! handed; the executor decides how the rows were cut.

use super::projector::{CpuParallelism, DenseProjector, WeightRows};

/// The literal transcription: one scalar dot per row, f32 weights.
///
/// Measured at a flat 5.6 GB/s across every Qwen3.8 projection shape,
/// which is why it is the oracle rather than the execution strategy. Kept
/// [`CpuParallelism::Serial`] deliberately: the reference path's value is
/// that it can be read line-by-line beside the source it transcribes, and
/// threading it would buy speed in the one place speed is not the point.
pub struct ScalarF32;

impl DenseProjector for ScalarF32 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::Serial
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::F32(w) = weight_rows else {
            panic!("the scalar reference kernel consumes f32 weights only");
        };
        let in_dim = x.len();
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for (a, b) in row.iter().zip(x) {
                acc += a * b;
            }
            *slot = acc;
        }
    }
}

/// BLAS `sgemv` through `larql-compute` — Accelerate on macOS, OpenBLAS
/// on Linux/FreeBSD, scalar on Windows by deliberate choice.
///
/// [`CpuParallelism::LibraryOwned`] because it already threads itself:
/// partitioning rows on top won 1.14x at best and lost on `5120 x 6144`.
pub struct BlasF32;

impl DenseProjector for BlasF32 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::LibraryOwned
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::F32(w) = weight_rows else {
            panic!("the BLAS kernel consumes f32 weights only");
        };
        let y = larql_compute::cpu::ops::moe::math::matmul_vec(x, w, out.len(), x.len());
        out.copy_from_slice(&y);
    }
}

/// **Fused BF16.** Load the compact code units, widen in REGISTERS,
/// multiply by the f32 activation, accumulate f32, discard.
///
/// The representation stays compact all the way into SIMD registers.
/// CPU-1B measured the alternative — widen a tile into scratch, then call
/// `sgemv` — at 27.3 GB/s against this kernel's 122.0, i.e. slower than
/// plain f32 despite reading half the bytes. Compact-to-registers is the
/// architecture; BF16 is only its first instance.
///
/// **Not always the right kernel.** Measured through the executor, this
/// wins 2.07-2.68x on the streaming shapes and LOSES 0.26x on the tiny
/// `48 x 5120` delta projections: at ~0.5 MB they are cache-resident, so
/// there is no RAM traffic to halve, and Accelerate's cache-resident
/// `sgemv` (262 GB/s) beats a serial widen-and-FMA loop (34 GB/s). Format
/// choice belongs per matrix class alongside worker count, not to the
/// model as a whole.
///
/// The widen is EXACT: bf16 is the top half of f32, so `(bits as u32) <<
/// 16` reproduces the value with no rounding and no table. The activation
/// stays f32 and the accumulator stays f32, so this changes
/// representation and mechanics and no numerical value — measured at
/// rel_rms 3.6e-7 against BLAS, which is summation order alone. Rounding
/// activations to bf16 to reach `BFDOT` is a separate precision decision
/// and is not made here.
pub struct FusedBf16;

impl DenseProjector for FusedBf16 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::Bf16(w) = weight_rows else {
            panic!("the fused bf16 kernel consumes bf16 weights only");
        };
        let in_dim = x.len();
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            *slot = bf16_dot(row, x);
        }
    }
}

/// **Fused Q8.** Load the packed codes, widen and scale in REGISTERS,
/// accumulate f32, discard.
///
/// The same architecture as [`FusedBf16`] and for the same measured
/// reason: CPU-1B priced widen-a-tile-into-scratch-then-`sgemv` at 27.3
/// GB/s against a fused kernel's 122.0 — slower than plain f32 while
/// reading half the bytes. A compact format that materialises before it
/// computes has residency and nothing else, and Q8 would pay that twice
/// over.
///
/// The scale applies once per BLOCK, not once per element: the block's
/// integer dot is accumulated in f32 and multiplied at the end, so the
/// per-element work is a widen and an FMA regardless of block size.
///
/// **This one is lossy**, unlike every kernel beside it. `FusedBf16`
/// changes representation and no value; this changes the values, and the
/// numbers it produces are only as good as the quantiser that made the
/// codes. That judgement belongs to whoever chooses the format — here the
/// only claim is that the kernel computes what the format DENOTES, which
/// `the_q8_kernel_computes_what_the_format_denotes` pins against a scalar
/// definition.
pub struct FusedQ8;

impl DenseProjector for FusedQ8 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::Q8 {
            codes,
            scales,
            block,
            ..
        } = weight_rows
        else {
            panic!("the fused q8 kernel consumes q8 weights only");
        };
        let in_dim = x.len();
        let per_row = in_dim.div_ceil(block);
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &codes[o * in_dim..(o + 1) * in_dim];
            let row_scales = &scales[o * per_row..(o + 1) * per_row];
            *slot = q8_dot(row, row_scales, block, x);
        }
    }
}

/// **Fused Q4.** Two codes per byte, unpacked and scaled in registers.
///
/// Same architecture as [`FusedQ8`], one representation further down, and
/// asked the same question: at 4.5 bits/weight is there still traffic
/// left to trade for the extra unpacking? Q8 answered 1.28x against a
/// 1.9x byte reduction because it left the memory-bound regime; Q4 halves
/// the bytes again and adds a nibble split, a mask and a bias on top.
///
/// Codes are symmetric `-8..=7`, stored biased by 8 so a nibble is an
/// unsigned 0..15 and the unbias is one vector subtract.
pub struct FusedQ4;

impl DenseProjector for FusedQ4 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::Q4 {
            packed,
            scales,
            block,
        } = weight_rows
        else {
            panic!("the fused q4 kernel consumes q4 weights only");
        };
        let in_dim = x.len();
        let per_row = in_dim.div_ceil(block);
        let bytes_per_row = in_dim / 2;
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &packed[o * bytes_per_row..(o + 1) * bytes_per_row];
            let row_scales = &scales[o * per_row..(o + 1) * per_row];
            let mut acc = 0.0f32;
            for (b, scale) in row_scales.iter().enumerate() {
                let lo = b * block;
                let hi = (lo + block).min(in_dim);
                if lo >= hi {
                    break;
                }
                acc += scale * q4_block_dot(&row[lo / 2..hi / 2], &x[lo..hi]);
            }
            *slot = acc;
        }
    }
}

/// NVFP4 resident, decoded in registers — the CPU's arm for a compiled
/// NVFP4 pack.
///
/// Exists because a representation with no execution path is a
/// representation that cannot be measured. Before it, every backend that
/// requested NVFP4 was a device backend, so a model whose token mixer
/// has no device kernel — Qwen3.8's 48 Gated DeltaNet layers — could
/// have an NVFP4 pack compiled, verified, and then unrunnable anywhere.
///
/// The arithmetic is `larql_models::quant::nvfp4::dequantize_into`'s
/// association, applied a group at a time: the shared CPU kernel this
/// delegates to is held bit-exact against decode-then-multiply by its
/// own tests, so this arm and the oracle are the same program.
pub struct FusedNvfp4;

impl DenseProjector for FusedNvfp4 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::Nvfp4 {
            packed,
            scales,
            tensor_scale,
        } = weight_rows
        else {
            panic!("the fused nvfp4 kernel consumes nvfp4 weights only");
        };
        let n = out.len();
        let Some(values) =
            larql_compute::cpu::nvfp4_gemv::nvfp4_gemv(packed, scales, tensor_scale, x, n, x.len())
        else {
            // The slab's geometry is settled before dispatch, so a
            // refusal here means the two disagree — which is a bug in
            // this file, not a runtime condition to absorb.
            panic!("nvfp4 slab geometry does not describe [{n}, {}]", x.len());
        };
        out.copy_from_slice(&values);
    }
}

/// **Direct K-quant.** The stored ggml blocks — Q8_0, Q6_K or Q4_K —
/// multiplied where they lie, by the kernel their codec names.
///
/// PARETO-1's v3 execution arm. Until it existed a K-quant operand had
/// one route: decode the whole matrix to f32 and run an f32 GEMV over
/// the result, which is correct and is what rung A was measured with,
/// but prices a Q8_0 model at four bytes per weight — 97.4 GB of widened
/// f32 per token on Qwen3.8-27B where the artifact holds 24.4. The bytes
/// this kernel reads are the artifact's own: no decode, no requantise,
/// no derivative.
///
/// [`CpuParallelism::LibraryOwned`] because the kernels behind it thread
/// themselves, as [`BlasF32`] does: the Q4_K and Q6_K kernels in
/// `larql-compute` split rows across their own pool, and the Q8_0 one
/// was written to match them. Partitioning on top would nest a second
/// fan-out inside the first, which is the one thing this executor's
/// module note forbids.
///
/// Whether this arm is admissible as a BEHAVIOURAL authority, rather
/// than a throughput backend, is decided by the equivalence gate in
/// `~/chris-models/pareto1/V3-QUALIFICATION.md` against
/// decode-then-multiply on the same stored bytes — never by this
/// comment.
pub struct FusedKQuant;

impl DenseProjector for FusedKQuant {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::LibraryOwned
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::KQuant { blocks, codec } = weight_rows else {
            panic!("the direct K-quant kernel consumes stored K-quant blocks only");
        };
        let n = out.len();
        let Some(values) = codec.gemv(blocks, x, n, x.len()) else {
            // Geometry is settled at `WeightSlice::rows` before dispatch,
            // so a refusal here means the two disagree — a bug in this
            // file, not a runtime condition to absorb.
            panic!(
                "{} slab geometry does not describe [{n}, {}]",
                codec.name,
                x.len()
            );
        };
        out.copy_from_slice(&values);
    }
}

/// One block's unscaled dot. `packed.len() * 2 == x.len()`.
#[inline]
fn q4_block_dot(packed: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64, and every access stays
        // inside `packed` and `x`, whose lengths the caller pairs.
        return unsafe { q4_block_dot_neon(packed, x) };
    }
    #[allow(unreachable_code)]
    q4_block_dot_portable(packed, x)
}

/// The portable definition the NEON version must agree with.
///
/// Byte `j` carries element `j` in its low nibble and element
/// `j + half` in its high one — see [`WeightRows::Q4`].
pub(super) fn q4_block_dot_portable(packed: &[u8], x: &[f32]) -> f32 {
    let half = packed.len();
    let mut acc = 0.0f32;
    for (j, byte) in packed.iter().enumerate() {
        acc += ((byte & 0x0f) as i32 - 8) as f32 * x[j];
        acc += ((byte >> 4) as i32 - 8) as f32 * x[j + half];
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q4_block_dot_neon(packed: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let half = packed.len();
    let (pp, xp) = (packed.as_ptr(), x.as_ptr());
    let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let (mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mask = vdupq_n_u8(0x0f);
    let bias = vdupq_n_s8(8);
    let mut j = 0usize;
    while j + 16 <= half {
        let raw = vld1q_u8(pp.add(j));
        // Low nibbles are elements j..j+16, high nibbles j+half..+16.
        let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, mask)), bias);
        let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), bias);
        for (n, half_off) in [(lo, 0usize), (hi, half)] {
            let w = vmovl_s8(vget_low_s8(n));
            let z = vmovl_s8(vget_high_s8(n));
            let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
            let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w)));
            let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(z)));
            let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(z)));
            let base = xp.add(j + half_off);
            a0 = vfmaq_f32(a0, f0, vld1q_f32(base));
            a1 = vfmaq_f32(a1, f1, vld1q_f32(base.add(4)));
            a2 = vfmaq_f32(a2, f2, vld1q_f32(base.add(8)));
            a3 = vfmaq_f32(a3, f3, vld1q_f32(base.add(12)));
        }
        j += 16;
    }
    let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    while j < half {
        let byte = *packed.get_unchecked(j);
        acc += ((byte & 0x0f) as i32 - 8) as f32 * *x.get_unchecked(j);
        acc += ((byte >> 4) as i32 - 8) as f32 * *x.get_unchecked(j + half);
        j += 1;
    }
    acc
}

/// One row's dot product: per-block integer accumulation, scaled once.
#[inline]
pub(super) fn q8_dot(codes: &[i8], scales: &[f32], block: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (b, scale) in scales.iter().enumerate() {
        let lo = b * block;
        let hi = (lo + block).min(codes.len());
        if lo >= hi {
            break;
        }
        acc += scale * q8_block_dot(&codes[lo..hi], &x[lo..hi]);
    }
    acc
}

/// The unscaled dot of one block.
#[inline]
fn q8_block_dot(codes: &[i8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; the loop reads only within
        // `codes` and `x`, which are equal length by the caller.
        return unsafe { q8_block_dot_neon(codes, x) };
    }
    #[allow(unreachable_code)]
    q8_block_dot_portable(codes, x)
}

/// The portable definition the NEON version must agree with.
pub(super) fn q8_block_dot_portable(codes: &[i8], x: &[f32]) -> f32 {
    codes.iter().zip(x).map(|(c, v)| *c as f32 * v).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q8_block_dot_neon(codes: &[i8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = codes.len().min(x.len());
    let (cp, xp) = (codes.as_ptr(), x.as_ptr());
    let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let (mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mut i = 0usize;
    while i + 16 <= n {
        // i8x16 -> two i16x8 -> four i32x4 -> four f32x4. The widen is
        // exact at every step; only the final multiply rounds.
        let c = vld1q_s8(cp.add(i));
        let lo = vmovl_s8(vget_low_s8(c));
        let hi = vmovl_s8(vget_high_s8(c));
        let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
        let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
        let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
        let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
        a0 = vfmaq_f32(a0, f0, vld1q_f32(xp.add(i)));
        a1 = vfmaq_f32(a1, f1, vld1q_f32(xp.add(i + 4)));
        a2 = vfmaq_f32(a2, f2, vld1q_f32(xp.add(i + 8)));
        a3 = vfmaq_f32(a3, f3, vld1q_f32(xp.add(i + 12)));
        i += 16;
    }
    let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    while i < n {
        acc += *codes.get_unchecked(i) as f32 * *x.get_unchecked(i);
        i += 1;
    }
    acc
}

/// One row's dot product, widening in registers.
#[inline]
pub(super) fn bf16_dot(w: &[u16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on every aarch64 target Rust supports,
        // and the loop reads only within `w` and `x`, which are equal
        // length by the caller's contract.
        return unsafe { bf16_dot_neon(w, x) };
    }
    #[allow(unreachable_code)]
    bf16_dot_portable(w, x)
}

/// The portable widen-and-accumulate, and the definition the NEON
/// version must agree with.
pub(super) fn bf16_dot_portable(w: &[u16], x: &[f32]) -> f32 {
    w.iter()
        .zip(x)
        .map(|(b, v)| f32::from_bits((*b as u32) << 16) * v)
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bf16_dot_neon(w: &[u16], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len().min(w.len());
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    // Four accumulators to hide FMA latency.
    let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let (mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mut i = 0usize;
    while i + 16 <= n {
        let w0 = vld1q_u16(wp.add(i));
        let w1 = vld1q_u16(wp.add(i + 8));
        let f0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(w0)), 16));
        let f1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(w0)), 16));
        let f2 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(w1)), 16));
        let f3 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(w1)), 16));
        a0 = vfmaq_f32(a0, f0, vld1q_f32(xp.add(i)));
        a1 = vfmaq_f32(a1, f1, vld1q_f32(xp.add(i + 4)));
        a2 = vfmaq_f32(a2, f2, vld1q_f32(xp.add(i + 8)));
        a3 = vfmaq_f32(a3, f3, vld1q_f32(xp.add(i + 12)));
        i += 16;
    }
    let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    while i < n {
        acc += f32::from_bits((*w.get_unchecked(i) as u32) << 16) * *x.get_unchecked(i);
        i += 1;
    }
    acc
}
