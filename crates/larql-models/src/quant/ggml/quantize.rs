//! Encode-side helpers for the legacy GGML formats.
//!
//! Q4_K / Q6_K quantizers live in `larql_compute::cpu::ops::q4_common`
//! (per ADR-008 — they're hot enough to keep next to the SIMD kernels
//! that consume them). This module covers Q4_0 and Q8_0, which the
//! vindex write path uses for the lm_head and gate vector slices.

// ── Quantizers (f32 → packed bytes) ──

/// Quantize f32 values to Q4_0 format.
/// Input must be a multiple of 32 elements.
/// Output: 18 bytes per block (f16 scale + 16 bytes of packed 4-bit quants).
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(32),
        "Q4_0: element count must be multiple of 32"
    );
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);

    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];

        // Find max absolute value for scale
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0; // map [-7*scale, 7*scale]
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        // Write f16 scale
        let scale_f16 = crate::quant::half::f32_to_f16(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        // Quantize: each value → round(val/scale) + 8, clamp to [0, 15].
        // ggml planar nibble layout (`quantize_row_q4_0_ref`): byte j packs
        // element j in its low nibble and element j+16 in its high nibble.
        for j in 0..16 {
            let lo_val = block[j];
            let hi_val = block[j + 16];
            let lo = ((lo_val * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((hi_val * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Quantize f32 values to Q8_0 format.
/// Input must be a multiple of 32 elements.
/// Output: 34 bytes per block (f16 scale + 32 signed int8 quants).
pub fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(32),
        "Q8_0: element count must be multiple of 32"
    );
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 34);

    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];

        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let scale_f16 = crate::quant::half::f32_to_f16(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());

        for &val in &block[..32] {
            let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

// Compute operations (matvec, vecmat, NEON kernels) moved to larql-compute.
// See: crates/larql-compute/src/cpu/ops/
