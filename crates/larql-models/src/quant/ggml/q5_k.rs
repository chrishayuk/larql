//! Q5_K — 256-element super-block, 176 bytes/block. Sits between Q4_K and
//! Q6_K in the Ollama-shaped `Q5_K_M` mixes: same 6-bit scale/min metadata
//! as Q4_K, plus one extra magnitude bit per element in a separate plane.
//!
//! Block layout (176 bytes, 256 elements):
//!   [  0..  2]  d       — f16 global scale
//!   [  2..  4]  dmin    — f16 global min
//!   [  4.. 16]  scales  — 12 bytes → 8 six-bit scales + 8 six-bit mins (same as Q4_K)
//!   [ 16.. 48]  qh      — 1 high bit per element (packed, 32 bytes = 256 bits)
//!   [ 48..176]  qs      — 4 low bits per element (packed, 128 bytes = 256 nibbles)
//!
//! Element ordering is Q4_K's: four groups of 32 `qs` bytes, each group
//! spanning two adjacent sub-blocks (low nibbles → sub-block `2g`, high
//! nibbles → sub-block `2g+1`). The only structural addition is `qh`, whose
//! bits `2g` / `2g+1` supply the fifth bit for those two halves. NEON row
//! dot + scaled-add with scalar fallbacks, matching `q4_k`.

use super::q4_k::unpack_q4k_scales;
use super::{check_block_input, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMS};
use crate::detect::ModelError;
use crate::quant::half::f16_to_f32;

/// Start of the 12-byte packed scale/min field.
const OFF_SCALES: usize = 4;
/// Start of the 32-byte high-bit plane.
const OFF_QH: usize = 16;
/// Start of the 128-byte low-nibble payload.
const OFF_QL: usize = 48;
/// Elements governed by one (scale, min) pair.
const SUB_BLOCK_ELEMS: usize = 32;
/// Sub-block pairs per super-block; each pair shares one 32-byte `qs` span.
const GROUPS: usize = 4;
/// Magnitude the fifth bit contributes on top of the 4-bit low nibble.
const HIGH_BIT_MAGNITUDE: u8 = 16;

/// Decoded super-block header — the two f16 globals plus the eight unpacked
/// 6-bit scale/min pairs.
///
/// Factored out because five walkers (dot scalar/NEON, scaled-add
/// scalar/NEON, dequantize) need exactly this preamble; `q4_k` open-codes it
/// in each one and the copies have to be kept in sync by hand.
struct BlockHeader {
    d: f32,
    dmin: f32,
    scales: [u8; 8],
    mins: [u8; 8],
}

/// Per-group decode parameters for one (low nibble, high nibble) pair.
///
/// `mask_lo` / `mask_hi` select this group's bit out of a `qh` byte: group
/// `g` owns bits `2g` (low half) and `2g+1` (high half).
struct GroupParams {
    sc_lo: f32,
    sc_hi: f32,
    mn_lo: f32,
    mn_hi: f32,
    mask_lo: u8,
    mask_hi: u8,
}

impl BlockHeader {
    #[inline]
    fn decode(block: &[u8]) -> Self {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_q4k_scales(&block[OFF_SCALES..OFF_QH]);
        Self {
            d,
            dmin,
            scales,
            mins,
        }
    }

    /// Scale/min/bit-mask set for group `g`, with `alpha` folded into the
    /// scales so the inner loop stays one multiply-subtract per element.
    #[inline]
    fn group(&self, g: usize, alpha: f32) -> GroupParams {
        let lo = 2 * g;
        let hi = 2 * g + 1;
        GroupParams {
            sc_lo: alpha * self.d * self.scales[lo] as f32,
            sc_hi: alpha * self.d * self.scales[hi] as f32,
            mn_lo: alpha * self.dmin * self.mins[lo] as f32,
            mn_hi: alpha * self.dmin * self.mins[hi] as f32,
            mask_lo: 1 << lo,
            mask_hi: 1 << hi,
        }
    }
}

/// Split one super-block into `(header, qh, qs)`.
#[inline]
fn split_block(block: &[u8]) -> (BlockHeader, &[u8], &[u8]) {
    (
        BlockHeader::decode(block),
        &block[OFF_QH..OFF_QL],
        &block[OFF_QL..Q5_K_BLOCK_BYTES],
    )
}

/// `HIGH_BIT_MAGNITUDE` when this element's fifth bit is set, else 0.
#[inline(always)]
fn high_bit(qh_byte: u8, mask: u8) -> u8 {
    if qh_byte & mask != 0 {
        HIGH_BIT_MAGNITUDE
    } else {
        0
    }
}

/// Validate row geometry shared by both fused row ops. Returns the
/// super-block count.
#[inline]
fn check_row(op: &str, data: &[u8], n: usize) -> Result<usize, ModelError> {
    if !n.is_multiple_of(Q5_K_BLOCK_ELEMS) {
        return Err(ModelError::Parse(format!(
            "{op}: row length {n} not a multiple of {Q5_K_BLOCK_ELEMS}"
        )));
    }
    let n_blocks = n / Q5_K_BLOCK_ELEMS;
    let need = n_blocks * Q5_K_BLOCK_BYTES;
    if data.len() < need {
        return Err(ModelError::Parse(format!(
            "{op}: data short: {} < {need}",
            data.len(),
        )));
    }
    Ok(n_blocks)
}

/// Fused Q5_K decode + dot product — `dot(dequant(data), x)` without
/// materialising the decoded row. Same math as `dequantize_q5_k(data,
/// x.len())` followed by a BLAS sdot, minus the `Vec<f32>` allocation and
/// the extra pass over the buffer.
#[inline]
pub fn q5k_row_dot(data: &[u8], x: &[f32]) -> Result<f32, ModelError> {
    let n_blocks = check_row("q5k_row_dot", data, x.len())?;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        Ok(q5k_row_dot_neon(data, x, n_blocks))
    }
    #[cfg(not(target_arch = "aarch64"))]
    Ok(q5k_row_dot_scalar(data, x, n_blocks))
}

/// Scalar reference used on non-aarch64 and by the NEON parity tests.
#[inline]
#[allow(dead_code)]
pub(super) fn q5k_row_dot_scalar(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let block = &data[sb * Q5_K_BLOCK_BYTES..][..Q5_K_BLOCK_BYTES];
        let (header, qh, qs) = split_block(block);
        let sb_base = sb * Q5_K_BLOCK_ELEMS;
        for g in 0..GROUPS {
            let p = header.group(g, 1.0);
            let chunk = &qs[g * SUB_BLOCK_ELEMS..(g + 1) * SUB_BLOCK_ELEMS];
            let base_lo = sb_base + g * 2 * SUB_BLOCK_ELEMS;
            let base_hi = base_lo + SUB_BLOCK_ELEMS;
            for l in 0..SUB_BLOCK_ELEMS {
                let byte = chunk[l];
                let bits = qh[l];
                let lo = (byte & 0x0F) + high_bit(bits, p.mask_lo);
                let hi = (byte >> 4) + high_bit(bits, p.mask_hi);
                acc += (p.sc_lo * lo as f32 - p.mn_lo) * x[base_lo + l];
                acc += (p.sc_hi * hi as f32 - p.mn_hi) * x[base_hi + l];
            }
        }
    }
    acc
}

/// NEON-SIMD Q5_K dequant + dot. Same shape as `q4k_row_dot_neon` — four
/// elements per lane-load, two accumulators for ILP — with the fifth bit
/// folded into the f32 staging arrays before the load, so the vector path
/// stays a plain multiply-subtract-FMA chain.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q5k_row_dot_neon(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let x_ptr = x.as_ptr();
    for sb in 0..n_blocks {
        let block = &data[sb * Q5_K_BLOCK_BYTES..][..Q5_K_BLOCK_BYTES];
        let (header, qh, qs) = split_block(block);
        let sb_base = sb * Q5_K_BLOCK_ELEMS;
        for g in 0..GROUPS {
            let p = header.group(g, 1.0);
            let sc_lo = vdupq_n_f32(p.sc_lo);
            let sc_hi = vdupq_n_f32(p.sc_hi);
            let mn_lo = vdupq_n_f32(p.mn_lo);
            let mn_hi = vdupq_n_f32(p.mn_hi);
            let chunk = qs.as_ptr().add(g * SUB_BLOCK_ELEMS);
            let qh_ptr = qh.as_ptr();
            let base_lo = x_ptr.add(sb_base + g * 2 * SUB_BLOCK_ELEMS);
            let base_hi = base_lo.add(SUB_BLOCK_ELEMS);
            for l4 in 0..SUB_BLOCK_ELEMS / 4 {
                let (lo_arr, hi_arr) = decode_quad(chunk, qh_ptr, l4, p.mask_lo, p.mask_hi);
                let lo = vld1q_f32(lo_arr.as_ptr());
                let hi = vld1q_f32(hi_arr.as_ptr());
                let v_lo = vsubq_f32(vmulq_f32(sc_lo, lo), mn_lo);
                let v_hi = vsubq_f32(vmulq_f32(sc_hi, hi), mn_hi);
                let x_lo = vld1q_f32(base_lo.add(l4 * 4));
                let x_hi = vld1q_f32(base_hi.add(l4 * 4));
                acc0 = vfmaq_f32(acc0, v_lo, x_lo);
                acc1 = vfmaq_f32(acc1, v_hi, x_hi);
            }
        }
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// Decode four consecutive elements' low and high nibbles into f32 staging
/// arrays, fifth bit included. Shared by the two NEON walkers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn decode_quad(
    chunk: *const u8,
    qh: *const u8,
    l4: usize,
    mask_lo: u8,
    mask_hi: u8,
) -> ([f32; 4], [f32; 4]) {
    let mut lo_arr = [0.0f32; 4];
    let mut hi_arr = [0.0f32; 4];
    for k in 0..4 {
        let l = l4 * 4 + k;
        let byte = *chunk.add(l);
        let bits = *qh.add(l);
        lo_arr[k] = ((byte & 0x0F) + high_bit(bits, mask_lo)) as f32;
        hi_arr[k] = ((byte >> 4) + high_bit(bits, mask_hi)) as f32;
    }
    (lo_arr, hi_arr)
}

/// Fused Q5_K decode + scaled add — `out += alpha * dequant(data)` without
/// materialising the decoded row. Counterpart to [`q5k_row_dot`] for the
/// down-projection leg of the walk.
#[inline]
pub fn q5k_row_scaled_add(data: &[u8], alpha: f32, out: &mut [f32]) -> Result<(), ModelError> {
    let n_blocks = check_row("q5k_row_scaled_add", data, out.len())?;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        q5k_row_scaled_add_neon(data, alpha, out, n_blocks);
    }
    #[cfg(not(target_arch = "aarch64"))]
    q5k_row_scaled_add_scalar(data, alpha, out, n_blocks);
    Ok(())
}

#[inline]
#[allow(dead_code)]
pub(super) fn q5k_row_scaled_add_scalar(data: &[u8], alpha: f32, out: &mut [f32], n_blocks: usize) {
    for sb in 0..n_blocks {
        let block = &data[sb * Q5_K_BLOCK_BYTES..][..Q5_K_BLOCK_BYTES];
        let (header, qh, qs) = split_block(block);
        let sb_base = sb * Q5_K_BLOCK_ELEMS;
        for g in 0..GROUPS {
            // alpha folded into the scales — one multiply-subtract per element.
            let p = header.group(g, alpha);
            let chunk = &qs[g * SUB_BLOCK_ELEMS..(g + 1) * SUB_BLOCK_ELEMS];
            let base_lo = sb_base + g * 2 * SUB_BLOCK_ELEMS;
            let base_hi = base_lo + SUB_BLOCK_ELEMS;
            for l in 0..SUB_BLOCK_ELEMS {
                let byte = chunk[l];
                let bits = qh[l];
                let lo = (byte & 0x0F) + high_bit(bits, p.mask_lo);
                let hi = (byte >> 4) + high_bit(bits, p.mask_hi);
                out[base_lo + l] += p.sc_lo * lo as f32 - p.mn_lo;
                out[base_hi + l] += p.sc_hi * hi as f32 - p.mn_hi;
            }
        }
    }
}

/// NEON-SIMD fused Q5_K dequant + scaled-add.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q5k_row_scaled_add_neon(data: &[u8], alpha: f32, out: &mut [f32], n_blocks: usize) {
    use std::arch::aarch64::*;
    let out_ptr = out.as_mut_ptr();
    for sb in 0..n_blocks {
        let block = &data[sb * Q5_K_BLOCK_BYTES..][..Q5_K_BLOCK_BYTES];
        let (header, qh, qs) = split_block(block);
        let sb_base = sb * Q5_K_BLOCK_ELEMS;
        for g in 0..GROUPS {
            let p = header.group(g, alpha);
            let sc_lo = vdupq_n_f32(p.sc_lo);
            let sc_hi = vdupq_n_f32(p.sc_hi);
            let mn_lo = vdupq_n_f32(p.mn_lo);
            let mn_hi = vdupq_n_f32(p.mn_hi);
            let chunk = qs.as_ptr().add(g * SUB_BLOCK_ELEMS);
            let qh_ptr = qh.as_ptr();
            let base_lo = out_ptr.add(sb_base + g * 2 * SUB_BLOCK_ELEMS);
            let base_hi = base_lo.add(SUB_BLOCK_ELEMS);
            for l4 in 0..SUB_BLOCK_ELEMS / 4 {
                let (lo_arr, hi_arr) = decode_quad(chunk, qh_ptr, l4, p.mask_lo, p.mask_hi);
                let lo = vld1q_f32(lo_arr.as_ptr());
                let hi = vld1q_f32(hi_arr.as_ptr());
                let v_lo = vsubq_f32(vmulq_f32(sc_lo, lo), mn_lo);
                let v_hi = vsubq_f32(vmulq_f32(sc_hi, hi), mn_hi);
                let old_lo = vld1q_f32(base_lo.add(l4 * 4));
                let old_hi = vld1q_f32(base_hi.add(l4 * 4));
                vst1q_f32(base_lo.add(l4 * 4), vaddq_f32(old_lo, v_lo));
                vst1q_f32(base_hi.add(l4 * 4), vaddq_f32(old_hi, v_hi));
            }
        }
    }
}

pub fn dequantize_q5_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let n_blocks = check_block_input("Q5_K", data, n_elements, Q5_K_BLOCK_ELEMS, Q5_K_BLOCK_BYTES)?;

    let mut out = vec![0.0f32; n_elements];

    for sb in 0..n_blocks {
        let block = &data[sb * Q5_K_BLOCK_BYTES..][..Q5_K_BLOCK_BYTES];
        let (header, qh, qs) = split_block(block);
        let sb_base = sb * Q5_K_BLOCK_ELEMS;
        for g in 0..GROUPS {
            let p = header.group(g, 1.0);
            let chunk = &qs[g * SUB_BLOCK_ELEMS..(g + 1) * SUB_BLOCK_ELEMS];
            let base_lo = sb_base + g * 2 * SUB_BLOCK_ELEMS;
            let base_hi = base_lo + SUB_BLOCK_ELEMS;
            for l in 0..SUB_BLOCK_ELEMS {
                let byte = chunk[l];
                let bits = qh[l];
                let lo = (byte & 0x0F) + high_bit(bits, p.mask_lo);
                let hi = (byte >> 4) + high_bit(bits, p.mask_hi);
                out[base_lo + l] = p.sc_lo * lo as f32 - p.mn_lo;
                out[base_hi + l] = p.sc_hi * hi as f32 - p.mn_hi;
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(d: u16, dmin: u16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(Q5_K_BLOCK_BYTES);
        b.extend_from_slice(&d.to_le_bytes());
        b.extend_from_slice(&dmin.to_le_bytes());
        b.extend_from_slice(&scales);
        b.extend_from_slice(&qh);
        b.extend_from_slice(&qs);
        assert_eq!(b.len(), Q5_K_BLOCK_BYTES);
        b
    }

    /// A block exercising every scale slot, both min planes, and a
    /// non-trivial spread of high bits — the shared fixture for the
    /// row-op-vs-dequantize agreement tests.
    fn varied_block() -> Vec<u8> {
        let mut scales = [0u8; 12];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = (i as u8 * 7 + 3) & 0x3F;
        }
        let mut qh = [0u8; 32];
        for (i, h) in qh.iter_mut().enumerate() {
            *h = (i as u8).wrapping_mul(37);
        }
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (i as u8).wrapping_mul(29).wrapping_add(11);
        }
        // d = 1.0 (0x3C00), dmin = 0.5 (0x3800) — both planes active.
        make_block(0x3C00, 0x3800, scales, qh, qs)
    }

    #[test]
    fn zero_scales_all_zero() {
        // With scales=0 and mins=0, all outputs = d*q - 0 = 0 when q=0.
        let block = make_block(0x3C00, 0x0000, [0u8; 12], [0u8; 32], [0u8; 128]);
        let out = dequantize_q5_k(&block, Q5_K_BLOCK_ELEMS).unwrap();
        assert_eq!(out.len(), Q5_K_BLOCK_ELEMS);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn high_bit_set_adds_16() {
        // d=1.0, dmin=0, scales[0]=1 (raw), mins[0]=0.
        let mut sc = [0u8; 12];
        sc[0] = 1; // scale[0]=1, all others 0
        let mut qh = [0u8; 32];
        qh[0] = 0x01; // bit0 set → group 0's low half gets +16 for elem 0
        let mut qs = [0u8; 128];
        qs[0] = 0x01; // lo nibble = 1 for elem 0

        let block = make_block(0x3C00, 0x0000, sc, qh, qs);
        let out = dequantize_q5_k(&block, Q5_K_BLOCK_ELEMS).unwrap();

        // elem 0: d=1.0, scale=1, lo=1, hi bit set → 1.0 * (1+16) - 0 = 17.0
        assert!(
            (out[0] - 17.0).abs() < 0.01,
            "expected 17.0, got {}",
            out[0]
        );
        // elem 32 is group 0's high half: qs[0]>>4 = 0, qh[0] bit1 clear,
        // and scale[1]=0 → 0.0 either way.
        assert!((out[32] - 0.0).abs() < 0.01);
    }

    /// Every group must read its own `qh` bit. A walker that hard-codes bit
    /// 0 (or forgets the shift as `g` advances) still passes
    /// `high_bit_set_adds_16`, because that fixture only exercises group 0.
    #[test]
    fn each_group_reads_its_own_high_bit() {
        for g in 0..GROUPS {
            let mut sc = [0u8; 12];
            // Set the 6-bit scale for sub-block 2g to 1. Slots 0..4 live in
            // the low 6 bits of bytes 0..4; slots 4..8 straddle bytes 8..12.
            let slot = 2 * g;
            if slot < 4 {
                sc[slot] = 1;
            } else {
                sc[slot + 4] = 1;
            }
            let mut qh = [0u8; 32];
            qh[0] = 1 << slot; // only this group's low-half bit
            let qs = [0u8; 128]; // low nibbles zero → value is purely the high bit

            let block = make_block(0x3C00, 0x0000, sc, qh, qs);
            let out = dequantize_q5_k(&block, Q5_K_BLOCK_ELEMS).unwrap();

            let idx = g * 2 * SUB_BLOCK_ELEMS;
            assert!(
                (out[idx] - 16.0).abs() < 0.01,
                "group {g}: expected 16.0 at index {idx}, got {}",
                out[idx]
            );
        }
    }

    #[test]
    fn row_dot_matches_dequantize_then_dot() {
        let block = varied_block();
        let x: Vec<f32> = (0..Q5_K_BLOCK_ELEMS)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();

        let reference: f32 = dequantize_q5_k(&block, Q5_K_BLOCK_ELEMS)
            .unwrap()
            .iter()
            .zip(&x)
            .map(|(w, v)| w * v)
            .sum();
        let fused = q5k_row_dot(&block, &x).unwrap();

        let rel = (reference - fused).abs() / reference.abs().max(1e-6);
        assert!(rel < 1e-5, "reference={reference} fused={fused}");
    }

    #[test]
    fn row_scaled_add_matches_dequantize_then_axpy() {
        let block = varied_block();
        let alpha = -0.375f32;

        let mut reference = vec![1.5f32; Q5_K_BLOCK_ELEMS];
        for (slot, w) in reference
            .iter_mut()
            .zip(dequantize_q5_k(&block, Q5_K_BLOCK_ELEMS).unwrap())
        {
            *slot += alpha * w;
        }

        let mut fused = vec![1.5f32; Q5_K_BLOCK_ELEMS];
        q5k_row_scaled_add(&block, alpha, &mut fused).unwrap();

        for (i, (r, f)) in reference.iter().zip(&fused).enumerate() {
            assert!((r - f).abs() < 1e-4, "index {i}: reference={r} fused={f}");
        }
    }

    /// On aarch64 the public entry points run NEON and the scalar walkers
    /// are the reference. Off aarch64 this degenerates to comparing the
    /// scalar path with itself, which is harmless.
    #[test]
    fn neon_matches_scalar() {
        let block = varied_block();
        let x: Vec<f32> = (0..Q5_K_BLOCK_ELEMS)
            .map(|i| ((i as f32) * 0.021).cos())
            .collect();

        let scalar = q5k_row_dot_scalar(&block, &x, 1);
        let dispatched = q5k_row_dot(&block, &x).unwrap();
        let rel = (scalar - dispatched).abs() / scalar.abs().max(1e-6);
        assert!(rel < 1e-5, "scalar={scalar} dispatched={dispatched}");

        let alpha = 0.25f32;
        let mut scalar_out = vec![0.5f32; Q5_K_BLOCK_ELEMS];
        q5k_row_scaled_add_scalar(&block, alpha, &mut scalar_out, 1);
        let mut dispatched_out = vec![0.5f32; Q5_K_BLOCK_ELEMS];
        q5k_row_scaled_add(&block, alpha, &mut dispatched_out).unwrap();
        for (i, (s, d)) in scalar_out.iter().zip(&dispatched_out).enumerate() {
            assert!((s - d).abs() < 1e-4, "index {i}: scalar={s} neon={d}");
        }
    }

    #[test]
    fn multi_block_row_accumulates_every_block() {
        // Two identical super-blocks: the fused dot over both must be
        // exactly twice the single-block dot. Catches a walker that drops
        // the `sb` stride on either the weight or the `x` side.
        let one = varied_block();
        let two: Vec<u8> = one.iter().chain(one.iter()).copied().collect();
        let x = vec![0.125f32; Q5_K_BLOCK_ELEMS * 2];

        let single = q5k_row_dot(&one, &x[..Q5_K_BLOCK_ELEMS]).unwrap();
        let double = q5k_row_dot(&two, &x).unwrap();
        let rel = (double - 2.0 * single).abs() / (2.0 * single).abs().max(1e-6);
        assert!(rel < 1e-5, "single={single} double={double}");
    }

    #[test]
    fn wrong_size_returns_error() {
        assert!(dequantize_q5_k(&[0u8; 10], 256).is_err());
    }

    #[test]
    fn row_ops_reject_bad_geometry() {
        let block = varied_block();

        // Row length not a whole number of super-blocks.
        let short_x = vec![0.0f32; 100];
        assert!(q5k_row_dot(&block, &short_x).is_err());
        let mut short_out = vec![0.0f32; 100];
        assert!(q5k_row_scaled_add(&block, 1.0, &mut short_out).is_err());

        // Well-shaped row, but the byte buffer is one block short.
        let x = vec![0.0f32; Q5_K_BLOCK_ELEMS * 2];
        assert!(q5k_row_dot(&block, &x).is_err());
        let mut out = vec![0.0f32; Q5_K_BLOCK_ELEMS * 2];
        assert!(q5k_row_scaled_add(&block, 1.0, &mut out).is_err());
    }
}
