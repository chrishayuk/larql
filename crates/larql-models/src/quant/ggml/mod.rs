//! GGML block quantization — encode/decode Q4_0, Q4_1, Q5_0, Q5_1,
//! Q8_0, Q4_K, Q6_K.
//!
//! Data format operations only:
//! - **Dequantize**: packed bytes → f32 (GGUF loading)
//! - **Quantize**: f32 → packed bytes (Q4_0, Q8_0 for vindex)
//! - **Metadata**: tensor_data_size, type_name
//!
//! Compute operations (matvec, vecmat, GPU shaders) are in
//! `larql-compute`. Used by GGUF model files. Each format stores
//! blocks of 32 (legacy) or 256 (K-quants) elements with shared scale
//! factors.
//!
//! Module split (post 2026-04-25 audit):
//! - `legacy`   — Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 (32-element blocks)
//! - `q4_k`     — Q4_K row-dot / row-scaled-add / dequantize (256)
//! - `q6_k`     — Q6_K row-dot / row-scaled-add / dequantize (256)
//! - `quantize` — encode-side helpers for the legacy formats
//!
//! `mod.rs` carries the type-id constants, the generic `dequantize`
//! dispatch, the shared `check_block_input` validator, and the test
//! mod.

use super::half::{decode_bf16, decode_f16};
use crate::detect::ModelError;

pub mod legacy;
pub mod q4_k;
pub mod q6_k;
pub mod quantize;

pub use legacy::{dequantize_q4_0, dequantize_q5_0, dequantize_q5_1};
pub use q4_k::{dequantize_q4_k, q4k_row_dot, q4k_row_scaled_add};
pub use q6_k::{dequantize_q6_k, q6k_row_dot, q6k_row_scaled_add};
pub use quantize::{quantize_q4_0, quantize_q8_0};

// ── Tensor-type IDs (match GGML wire format) ────────────────────────────
pub const TYPE_F32: u32 = 0;
pub const TYPE_F16: u32 = 1;
pub const TYPE_Q4_0: u32 = 2;
pub const TYPE_Q4_1: u32 = 3;
pub const TYPE_Q8_0: u32 = 6;
pub const TYPE_Q5_0: u32 = 8;
pub const TYPE_Q5_1: u32 = 9;
pub const TYPE_Q2_K: u32 = 10;
pub const TYPE_Q3_K: u32 = 11;
pub const TYPE_Q4_K: u32 = 12;
pub const TYPE_Q5_K: u32 = 13;
pub const TYPE_Q6_K: u32 = 14;
pub const TYPE_BF16: u32 = 30;

// ── Block geometry (canonical GGML wire format) ─────────────────────────
//
// Legacy quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) pack 32 elements per block.
// K-quants (Q4_K/Q6_K) pack 256 elements per super-block.
//
// Block byte sizes are exact and must never be rederived inline — they
// are part of the on-disk wire format. Q4_K and Q4_0 happen to share the
// same effective rate (0.5625 B/elem), which is exactly why we silently
// shipped a Q4_K file that the reader dispatched as Q4_0 once. Constants
// remove that footgun: callers compare to `Q4_K_BLOCK_BYTES` directly.

/// Elements per block for legacy quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0).
pub const LEGACY_BLOCK_ELEMS: usize = 32;

/// Elements per super-block for K-quants (Q4_K, Q6_K).
pub const K_QUANT_BLOCK_ELEMS: usize = 256;

/// Bytes per Q4_0 block (32 elements + f16 scale): 2 + 16.
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Elements per Q4_0 block.
pub const Q4_0_BLOCK_ELEMS: usize = LEGACY_BLOCK_ELEMS;

/// Bytes per Q4_1 block (32 elements + f16 scale + f16 min): 2 + 2 + 16.
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Bytes per Q5_0 block (32 elements + f16 scale + 4-byte high-bits + 16 nibbles).
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Bytes per Q5_1 block (32 elements + f16 scale + f16 min + 4-byte high-bits + 16 nibbles).
pub const Q5_1_BLOCK_BYTES: usize = 24;

/// Bytes per Q8_0 block (32 elements + f16 scale): 2 + 32.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Bytes per Q4_K super-block (256 elements): 2 + 2 + 12 + 128.
///
/// Layout: f16 d (2) + f16 dmin (2) + 12 packed (scale, min) bytes + 128 nibble bytes.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Elements per Q4_K super-block.
pub const Q4_K_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// Bytes per Q6_K super-block (256 elements): 128 + 64 + 16 + 2.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Elements per Q6_K super-block.
pub const Q6_K_BLOCK_ELEMS: usize = K_QUANT_BLOCK_ELEMS;

/// Validate that `data` holds at least `n_blocks` blocks of
/// `block_size` bytes for `n_elements` total elements (which must be a
/// multiple of `block_elems`). Returns the block count.
///
/// Checks `data.len() >= need` (not `==`) so callers can pass
/// over-sized buffers — the safetensors loader hands us slices that
/// sometimes carry trailing padding from the next tensor.
pub(crate) fn check_block_input(
    name: &'static str,
    data: &[u8],
    n_elements: usize,
    block_elems: usize,
    block_size: usize,
) -> Result<usize, ModelError> {
    if !n_elements.is_multiple_of(block_elems) {
        return Err(ModelError::Parse(format!(
            "{name}: n_elements {n_elements} not a multiple of {block_elems}"
        )));
    }
    let n_blocks = n_elements / block_elems;
    let need = n_blocks.checked_mul(block_size).ok_or_else(|| {
        ModelError::Parse(format!(
            "{name}: byte-size overflow ({n_blocks} blocks × {block_size} bytes)"
        ))
    })?;
    if data.len() < need {
        return Err(ModelError::Parse(format!(
            "{name}: data too short: {} bytes < expected {} ({} blocks × {} bytes)",
            data.len(),
            need,
            n_blocks,
            block_size
        )));
    }
    Ok(n_blocks)
}

/// Bytes occupied by `n_elements` quantised at `tensor_type`.
pub fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    match tensor_type {
        TYPE_F32 => Ok(n_elements * 4),
        TYPE_F16 | TYPE_BF16 => Ok(n_elements * 2),
        TYPE_Q4_0 => Ok(n_elements / LEGACY_BLOCK_ELEMS * Q4_0_BLOCK_BYTES),
        TYPE_Q4_1 => Ok(n_elements / LEGACY_BLOCK_ELEMS * Q4_1_BLOCK_BYTES),
        TYPE_Q5_0 => Ok(n_elements / LEGACY_BLOCK_ELEMS * Q5_0_BLOCK_BYTES),
        TYPE_Q5_1 => Ok(n_elements / LEGACY_BLOCK_ELEMS * Q5_1_BLOCK_BYTES),
        TYPE_Q8_0 => Ok(n_elements / LEGACY_BLOCK_ELEMS * Q8_0_BLOCK_BYTES),
        TYPE_Q4_K => Ok(n_elements / K_QUANT_BLOCK_ELEMS * Q4_K_BLOCK_BYTES),
        TYPE_Q6_K => Ok(n_elements / K_QUANT_BLOCK_ELEMS * Q6_K_BLOCK_BYTES),
        _ => Err(ModelError::Parse(format!(
            "tensor_data_size: unsupported type id {tensor_type}"
        ))),
    }
}

/// Human-readable name for a GGML tensor type. Returns `"unknown"`
/// (lowercase) for unrecognised ids — tests pin this casing.
pub fn type_name(tensor_type: u32) -> &'static str {
    match tensor_type {
        TYPE_F32 => "F32",
        TYPE_F16 => "F16",
        TYPE_Q4_0 => "Q4_0",
        TYPE_Q4_1 => "Q4_1",
        TYPE_Q8_0 => "Q8_0",
        TYPE_Q5_0 => "Q5_0",
        TYPE_Q5_1 => "Q5_1",
        TYPE_Q2_K => "Q2_K",
        TYPE_Q3_K => "Q3_K",
        TYPE_Q4_K => "Q4_K",
        TYPE_Q5_K => "Q5_K",
        TYPE_Q6_K => "Q6_K",
        TYPE_BF16 => "BF16",
        _ => "unknown",
    }
}

/// Dequantize raw bytes to f32 based on GGML tensor type.
///
/// Returns `ModelError::Parse` if `data` is too short for the
/// requested number of elements rather than panicking on a slice OOB.
pub fn dequantize(
    data: &[u8],
    tensor_type: u32,
    n_elements: usize,
) -> Result<Vec<f32>, ModelError> {
    match tensor_type {
        TYPE_F32 => {
            let need = n_elements
                .checked_mul(4)
                .ok_or_else(|| ModelError::Parse(format!("F32: size overflow ({n_elements}×4)")))?;
            if data.len() < need {
                return Err(ModelError::Parse(format!(
                    "F32: data too short: {} bytes < expected {need} ({n_elements} elements)",
                    data.len()
                )));
            }
            Ok(data[..need]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        TYPE_F16 => decode_passthrough(data, n_elements, "F16", decode_f16),
        TYPE_BF16 => decode_passthrough(data, n_elements, "BF16", decode_bf16),
        TYPE_Q4_0 => dequantize_q4_0(data, n_elements),
        TYPE_Q4_1 => legacy::dequantize_q4_1(data, n_elements),
        TYPE_Q8_0 => legacy::dequantize_q8_0(data, n_elements),
        TYPE_Q5_0 => dequantize_q5_0(data, n_elements),
        TYPE_Q5_1 => dequantize_q5_1(data, n_elements),
        TYPE_Q4_K => dequantize_q4_k(data, n_elements),
        TYPE_Q6_K => dequantize_q6_k(data, n_elements),
        other => Err(ModelError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

/// Bounds-checked decode of an f16 / bf16 byte slice via the supplied
/// half-precision decoder.
#[inline]
fn decode_passthrough(
    data: &[u8],
    n_elements: usize,
    name: &'static str,
    decoder: fn(&[u8]) -> Vec<f32>,
) -> Result<Vec<f32>, ModelError> {
    let need = n_elements
        .checked_mul(2)
        .ok_or_else(|| ModelError::Parse(format!("{name}: size overflow ({n_elements}×2)")))?;
    if data.len() < need {
        return Err(ModelError::Parse(format!(
            "{name}: data too short: {} bytes < expected {need} ({n_elements} elements)",
            data.len()
        )));
    }
    Ok(decoder(&data[..need]))
}

#[cfg(test)]
mod tests {
    use super::legacy::{dequantize_q4_1, dequantize_q8_0};
    use super::q6_k::q6k_row_dot_scalar;
    use super::*;

    // ── Q4_0 ──

    #[test]
    fn q4_0_basic() {
        // Scale = 1.0, quants = 0x12 → lo=2-8=-6, hi=1-8=-7
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0x12; 16]);
        let result = dequantize_q4_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - (-6.0)).abs() < 0.01);
        assert!((result[1] - (-7.0)).abs() < 0.01);
    }

    #[test]
    fn q4_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // f16 0.0
        block.extend_from_slice(&[0xFF; 16]);
        let result = dequantize_q4_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4_0_two_blocks() {
        let mut data = vec![0x00, 0x3C]; // block 0: scale=1.0
        data.extend_from_slice(&[0x88; 16]); // quants: lo=8-8=0, hi=8-8=0
        data.extend_from_slice(&[0x00, 0x40]); // block 1: scale=2.0
        data.extend_from_slice(&[0x19; 16]); // lo=9-8=1, hi=1-8=-7
        let result = dequantize_q4_0(&data, 64).unwrap();
        assert_eq!(result.len(), 64);
        assert!((result[0] - 0.0).abs() < 0.01); // block 0
        assert!((result[32] - 2.0).abs() < 0.01); // block 1: 1*2.0 = 2.0
        assert!((result[33] - (-14.0)).abs() < 0.01); // block 1: -7*2.0 = -14.0
    }

    // ── Q4_1 ──

    #[test]
    fn q4_1_basic() {
        // Scale=1.0, min=0.5, quants=0x00 → lo=0*1+0.5=0.5, hi=0*1+0.5=0.5
        let mut block = vec![0x00, 0x3C, 0x00, 0x38]; // scale=1.0, min=0.5
        block.extend_from_slice(&[0x00; 16]);
        let result = dequantize_q4_1(&block, 32).unwrap();
        assert!((result[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn q4_1_with_offset() {
        // Scale=2.0, min=-1.0, quants=0x31 → lo=1*2-1=1, hi=3*2-1=5
        let mut block = vec![0x00, 0x40, 0x00, 0xBC]; // scale=2.0, min=-1.0
        block.extend_from_slice(&[0x31; 16]);
        let result = dequantize_q4_1(&block, 32).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
    }

    // ── Q8_0 ──

    #[test]
    fn q8_0_basic() {
        let mut block = vec![0x00, 0x38]; // f16 scale = 0.5
        for _ in 0..16 {
            block.push(2u8); // +2 → 2*0.5 = 1.0
            block.push(0xFEu8); // -2 as i8 → -2*0.5 = -1.0
        }
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn q8_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // scale = 0
        block.extend_from_slice(&[127u8; 32]); // max int8
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_full_range() {
        let mut block = vec![0x00, 0x3C]; // scale = 1.0
        block.push(127); // max positive
        block.push(0x81); // -127 as i8
        block.extend_from_slice(&[0u8; 30]); // rest zeros
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!((result[0] - 127.0).abs() < 0.01);
        assert!((result[1] - (-127.0)).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 0.01);
    }

    // ── Type metadata ──

    #[test]
    fn tensor_sizes() {
        assert_eq!(tensor_data_size(TYPE_F32, 32).unwrap(), 128);
        assert_eq!(tensor_data_size(TYPE_F16, 32).unwrap(), 64);
        assert_eq!(tensor_data_size(TYPE_Q4_0, 32).unwrap(), 18);
        assert_eq!(tensor_data_size(TYPE_Q4_1, 32).unwrap(), 20);
        assert_eq!(tensor_data_size(TYPE_Q8_0, 32).unwrap(), 34);
    }

    #[test]
    fn type_names() {
        assert_eq!(type_name(TYPE_F32), "F32");
        assert_eq!(type_name(TYPE_Q4_0), "Q4_0");
        assert_eq!(type_name(TYPE_Q8_0), "Q8_0");
        assert_eq!(type_name(99), "unknown");
    }

    // ── F32 passthrough ──

    #[test]
    fn f32_passthrough() {
        let data: Vec<u8> = [1.0f32, -2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dequantize(&data, TYPE_F32, 3).unwrap();
        assert_eq!(result, vec![1.0, -2.0, 3.0]);
    }

    // ── Q5_0 ──

    #[test]
    fn q5_0_basic() {
        // scale=1.0, high_bits=0, quants=0x88 → lo4=8, hi4=8, hi1=0
        // combined=8, value=(8-16)*1.0=-8.0
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0x00; 4]); // high bits all zero
        block.extend_from_slice(&[0x88; 16]); // quants
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - (-8.0)).abs() < 0.01);
        assert!((result[1] - (-8.0)).abs() < 0.01);
    }

    #[test]
    fn q5_0_with_high_bits() {
        // scale=1.0, high_bits=0xFFFFFFFF (all 1), quants=0x00
        // lo4=0, hi1=1, combined=0|16=16, value=(16-16)*1.0=0.0
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0xFF; 4]); // high bits all one
        block.extend_from_slice(&[0x00; 16]); // quants all zero nibbles
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn q5_0_mixed() {
        // scale=2.0, high_bits=0x00000001 (bit 0 set), quants[0]=0x53
        // element 0: lo4=3, hi1=bit0=1, combined=3|16=19, value=(19-16)*2=6.0
        // element 1: lo4=5, hi1=bit1=0, combined=5, value=(5-16)*2=-22.0
        let mut block = vec![0x00, 0x40]; // f16 2.0
        block.extend_from_slice(&0x00000001u32.to_le_bytes()); // high bits
        block.push(0x53); // quants[0]: lo=3, hi=5
        block.extend_from_slice(&[0x00; 15]); // rest zero
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert!((result[0] - 6.0).abs() < 0.01);
        assert!((result[1] - (-22.0)).abs() < 0.01);
    }

    #[test]
    fn q5_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // scale=0
        block.extend_from_slice(&[0xFF; 4]);
        block.extend_from_slice(&[0xFF; 16]);
        let result = dequantize_q5_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ── Q5_1 ──

    #[test]
    fn q5_1_basic() {
        // scale=1.0, min=0.5, high_bits=0, quants=0x00
        // combined=0, value=0*1.0+0.5=0.5
        let mut block = vec![0x00, 0x3C, 0x00, 0x38]; // scale=1.0, min=0.5
        block.extend_from_slice(&[0x00; 4]); // high bits
        block.extend_from_slice(&[0x00; 16]); // quants
        let result = dequantize_q5_1(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn q5_1_with_high_bits() {
        // scale=2.0, min=1.0, high_bits=0xFFFFFFFF, quants=0xFF
        // lo4=15, hi1=1, combined=15|16=31, value=31*2.0+1.0=63.0
        let mut block = vec![0x00, 0x40, 0x00, 0x3C]; // scale=2.0, min=1.0
        block.extend_from_slice(&[0xFF; 4]); // high bits all one
        block.extend_from_slice(&[0xFF; 16]); // quants all 0xF nibbles
        let result = dequantize_q5_1(&block, 32).unwrap();
        assert!((result[0] - 63.0).abs() < 0.01);
    }

    #[test]
    fn q5_1_via_dequantize() {
        // Verify dispatch works through the main dequantize() function
        let mut block = vec![0x00, 0x3C, 0x00, 0x00]; // scale=1.0, min=0.0
        block.extend_from_slice(&[0x00; 4]); // high bits zero
        block.extend_from_slice(&[0x33; 16]); // lo=3, hi=3, combined=3
        let result = dequantize(&block, TYPE_Q5_1, 32).unwrap();
        assert!((result[0] - 3.0).abs() < 0.01);
        assert!((result[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn q5_0_via_dequantize() {
        // Verify dispatch works through the main dequantize() function
        let mut block = vec![0x00, 0x3C]; // scale=1.0
        block.extend_from_slice(&[0x00; 4]); // high bits zero
        block.extend_from_slice(&[0x88; 16]); // lo=8,hi=8, combined=8, value=(8-16)=-8
        let result = dequantize(&block, TYPE_Q5_0, 32).unwrap();
        assert!((result[0] - (-8.0)).abs() < 0.01);
    }

    // ── Q6_K row_dot NEON ≡ scalar ──

    fn synth_q6k_block(seed: u32) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        // Deterministic pseudo-random bytes for ql (128), qh (64), scales (16).
        let mut s = seed;
        for b in &mut block[..208] {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
        // f16 d = 0.0625
        block[208] = 0x00;
        block[209] = 0x2C;
        block
    }

    #[test]
    fn q6k_row_dot_neon_matches_scalar_single_block() {
        let data = synth_q6k_block(42);
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01).sin()).collect();
        let scalar = q6k_row_dot_scalar(&data, &x, 1);
        let dispatched = q6k_row_dot(&data, &x).unwrap();
        // Both paths should agree to within fp accumulation noise.
        assert!(
            (scalar - dispatched).abs() < 1e-3,
            "scalar={scalar} dispatched={dispatched}"
        );
    }

    #[test]
    fn q6k_row_dot_neon_matches_scalar_multi_block() {
        let mut data = Vec::with_capacity(210 * 8);
        for sb in 0..8 {
            data.extend_from_slice(&synth_q6k_block(1234 + sb as u32));
        }
        let x: Vec<f32> = (0..256 * 8)
            .map(|i| (((i as f32) * 0.003).cos() - 0.5) * 0.2)
            .collect();
        let scalar = q6k_row_dot_scalar(&data, &x, 8);
        let dispatched = q6k_row_dot(&data, &x).unwrap();
        let tol = (scalar.abs() + dispatched.abs()).max(1.0) * 1e-5;
        assert!(
            (scalar - dispatched).abs() < tol,
            "scalar={scalar} dispatched={dispatched} tol={tol}"
        );
    }

    // ── Bounds-check rejection (no panics on malformed input) ──

    fn assert_short_buffer(res: Result<Vec<f32>, ModelError>, fmt: &str) {
        match res {
            Err(ModelError::Parse(msg)) => {
                assert!(
                    msg.contains("data too short") && msg.contains(fmt),
                    "expected short-buffer error for {fmt}, got: {msg}"
                );
            }
            Err(other) => panic!("expected Parse error for {fmt}, got {other:?}"),
            Ok(v) => panic!(
                "expected short-buffer error for {fmt}, got {} elements",
                v.len()
            ),
        }
    }

    #[test]
    fn q4_0_rejects_short_buffer() {
        // 32 elements need 18 bytes; give it 10.
        assert_short_buffer(dequantize_q4_0(&[0u8; 10], 32), "Q4_0");
    }

    #[test]
    fn q4_1_rejects_short_buffer() {
        assert_short_buffer(dequantize(&[0u8; 4], TYPE_Q4_1, 32), "Q4_1");
    }

    #[test]
    fn q8_0_rejects_short_buffer() {
        // 64 elements = 2 blocks × 34 bytes = 68; give 40.
        assert_short_buffer(dequantize(&[0u8; 40], TYPE_Q8_0, 64), "Q8_0");
    }

    #[test]
    fn q5_0_rejects_short_buffer() {
        assert_short_buffer(dequantize_q5_0(&[0u8; 10], 32), "Q5_0");
    }

    #[test]
    fn q5_1_rejects_short_buffer() {
        assert_short_buffer(dequantize_q5_1(&[0u8; 10], 32), "Q5_1");
    }

    #[test]
    fn q4_k_rejects_short_buffer() {
        // 256 elements = 1 super-block = 144 bytes; give 100.
        assert_short_buffer(dequantize_q4_k(&[0u8; 100], 256), "Q4_K");
    }

    #[test]
    fn q6_k_rejects_short_buffer() {
        // 256 elements = 1 super-block = 210 bytes; give 100.
        assert_short_buffer(dequantize_q6_k(&[0u8; 100], 256), "Q6_K");
    }

    #[test]
    fn q4_0_rejects_misaligned_n_elements() {
        // 33 is not a multiple of 32.
        match dequantize_q4_0(&[0u8; 18], 33) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("not a multiple of 32"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn q6_k_rejects_misaligned_n_elements() {
        // 300 is not a multiple of 256.
        match dequantize_q6_k(&[0u8; 210], 300) {
            Err(ModelError::Parse(msg)) => {
                assert!(msg.contains("not a multiple of 256"), "got: {msg}");
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_f32_rejects_short_buffer() {
        // 8 elements = 32 bytes; give 20.
        match dequantize(&[0u8; 20], TYPE_F32, 8) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("F32"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_f16_rejects_short_buffer() {
        // 8 elements = 16 bytes; give 10.
        match dequantize(&[0u8; 10], TYPE_F16, 8) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("F16"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn passthrough_bf16_rejects_short_buffer() {
        match dequantize(&[0u8; 10], TYPE_BF16, 8) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("BF16"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_ok_when_zero_elements() {
        // Zero-element tensor should succeed with empty output across all block types.
        for &ty in &[
            TYPE_Q4_0, TYPE_Q4_1, TYPE_Q8_0, TYPE_Q5_0, TYPE_Q5_1, TYPE_Q4_K, TYPE_Q6_K,
        ] {
            let out = dequantize(&[], ty, 0).unwrap_or_else(|e| panic!("type {ty} failed: {e:?}"));
            assert!(out.is_empty(), "type {ty} produced {} elements", out.len());
        }
    }

    // ── Quantize → dequantize round-trips ──

    /// Max component-wise representation error for a given scale — Q4_0 maps
    /// every value to the nearest multiple of `scale` in `[-8*scale, 7*scale]`,
    /// so round-trip error is bounded by half a quantization step.
    #[test]
    fn q4_0_round_trip_preserves_within_half_step() {
        // Inputs fit the ±7*scale range cleanly.
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) * 0.1).collect();
        let packed = quantize_q4_0(&vals);
        assert_eq!(packed.len(), 2 * 18);
        let round = dequantize_q4_0(&packed, 64).unwrap();
        let scale = 0.1 * 31.5 / 7.0; // amax / 7 per block
        let max_step = scale * 0.5 + 1e-3;
        for (i, (v, r)) in vals.iter().zip(&round).enumerate() {
            assert!(
                (v - r).abs() <= max_step,
                "idx {i}: v={v} r={r} max_step={max_step}"
            );
        }
    }

    #[test]
    fn q4_0_round_trip_all_zero() {
        // Zero-scale corner: every value must decode to exactly 0.
        let vals = vec![0.0f32; 32];
        let packed = quantize_q4_0(&vals);
        let round = dequantize_q4_0(&packed, 32).unwrap();
        assert!(round.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_round_trip_precise() {
        // Q8_0 has 127 steps — 2 decimal places should survive cleanly.
        let vals: Vec<f32> = (0..64).map(|i| ((i as f32 - 32.0) * 0.013).sin()).collect();
        let packed = quantize_q8_0(&vals);
        assert_eq!(packed.len(), 2 * 34);
        let round = dequantize_q8_0(&packed, 64).unwrap();
        // Per-block amax / 127 ≤ 1/127 ≈ 0.008, so round-trip error < 0.004.
        for (i, (v, r)) in vals.iter().zip(&round).enumerate() {
            assert!((v - r).abs() < 0.01, "idx {i}: v={v} r={r}");
        }
    }

    #[test]
    fn q8_0_round_trip_edges() {
        // Values hitting the ±127/scale clamp edges. Scale is stored as f16
        // (11-bit mantissa), so allow ~1e-3 for the quantized representation
        // of ±1.0 after the f16-scale precision loss.
        let mut vals = Vec::with_capacity(32);
        for _ in 0..16 {
            vals.push(1.0);
            vals.push(-1.0);
        }
        let packed = quantize_q8_0(&vals);
        let round = dequantize_q8_0(&packed, 32).unwrap();
        for (i, (v, r)) in vals.iter().zip(&round).enumerate() {
            assert!((v - r).abs() < 1e-3, "idx {i}: v={v} r={r}");
        }
    }

    // ── Dispatch coverage via dequantize() for the K-quants and Q4_0 ──

    #[test]
    fn q4_0_via_dequantize() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.05).collect();
        let packed = quantize_q4_0(&vals);
        let round = dequantize(&packed, TYPE_Q4_0, 32).unwrap();
        assert_eq!(round.len(), 32);
    }

    #[test]
    fn q8_0_via_dequantize() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let packed = quantize_q8_0(&vals);
        let round = dequantize(&packed, TYPE_Q8_0, 32).unwrap();
        assert_eq!(round.len(), 32);
        // Matches in-module Q8_0 path exactly.
        let direct = dequantize_q8_0(&packed, 32).unwrap();
        assert_eq!(round, direct);
    }

    #[test]
    fn q4_k_via_dequantize_roundtrips_to_known_output() {
        // Build a 144-byte Q4K block with scale 1.0, min 0.0, all sub-scales=1,
        // sub-mins=0, nibbles = low nibble index 0..7 repeated — check shape,
        // not exact values (the scale/min packing is lossy).
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0 (f16)
        block[2] = 0x00;
        block[3] = 0x00; // dmin = 0.0
                         // bytes 4..16: scales[0..4] = 1, mins[0..4] = 0 (low 6 bits only)
        for s in &mut block[4..8] {
            *s = 0x01;
        }
        for _m in &mut block[8..12] { /* mins lo = 0 */ }
        // Leave scales[4..8] = 0 (high nibble carrier) and quants zero.
        let out = dequantize(&block, TYPE_Q4_K, 256).unwrap();
        assert_eq!(out.len(), 256);
        // First 128 elements use scales[0..4] = 1 so decoded = 0 (nibbles zero).
        // Remaining 128 use scales[4..8] = 0 so also zero.
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q6_k_via_dequantize() {
        // Dispatch-path check — uses the single-block synth helper.
        let block = synth_q6k_block(99);
        let direct = dequantize_q6_k(&block, 256).unwrap();
        let dispatched = dequantize(&block, TYPE_Q6_K, 256).unwrap();
        assert_eq!(direct, dispatched);
    }

    #[test]
    fn q6k_row_dot_matches_dequantized_dot() {
        // Ground truth: dequantize_q6_k then compute the dot manually.
        let data = synth_q6k_block(7);
        let deq = dequantize_q6_k(&data, 256).unwrap();
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let gold: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        let dispatched = q6k_row_dot(&data, &x).unwrap();
        let tol = (gold.abs() + dispatched.abs()).max(1.0) * 1e-4;
        assert!(
            (gold - dispatched).abs() < tol,
            "gold={gold} dispatched={dispatched} tol={tol}"
        );
    }

    // ── Q4_K row_dot NEON ≡ scalar ──

    fn synth_q4k_block(seed: u32) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        let mut s = seed;
        for b in &mut block[4..144] {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (s >> 16) as u8;
        }
        // d = 0.0625 (f16 0x2C00), dmin = 0.0625 — small to keep values bounded.
        block[0] = 0x00;
        block[1] = 0x2C;
        block[2] = 0x00;
        block[3] = 0x2C;
        block
    }

    #[test]
    fn q4k_row_dot_neon_matches_scalar_single_block() {
        use super::q4_k::q4k_row_dot_scalar;
        let data = synth_q4k_block(42);
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01).sin()).collect();
        let scalar = q4k_row_dot_scalar(&data, &x, 1);
        let dispatched = q4k_row_dot(&data, &x).unwrap();
        assert!(
            (scalar - dispatched).abs() < 1e-3,
            "scalar={scalar} dispatched={dispatched}"
        );
    }

    #[test]
    fn q4k_row_dot_neon_matches_scalar_multi_block() {
        use super::q4_k::q4k_row_dot_scalar;
        let mut data = Vec::with_capacity(144 * 8);
        for sb in 0..8u32 {
            data.extend_from_slice(&synth_q4k_block(1000 + sb));
        }
        let x: Vec<f32> = (0..256 * 8)
            .map(|i| (((i as f32) * 0.003).cos() - 0.5) * 0.2)
            .collect();
        let scalar = q4k_row_dot_scalar(&data, &x, 8);
        let dispatched = q4k_row_dot(&data, &x).unwrap();
        let tol = (scalar.abs() + dispatched.abs()).max(1.0) * 1e-5;
        assert!(
            (scalar - dispatched).abs() < tol,
            "scalar={scalar} dispatched={dispatched} tol={tol}"
        );
    }

    #[test]
    fn q4k_row_dot_matches_dequantized_dot() {
        let data = synth_q4k_block(7);
        let deq = dequantize_q4_k(&data, 256).unwrap();
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.001 - 0.05).collect();
        let gold: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        let dispatched = q4k_row_dot(&data, &x).unwrap();
        let tol = (gold.abs() + dispatched.abs()).max(1.0) * 1e-4;
        assert!(
            (gold - dispatched).abs() < tol,
            "gold={gold} dispatched={dispatched} tol={tol}"
        );
    }

    // ── Q4_K dequantize with nonzero known values ──

    #[test]
    fn q4_k_dequantize_known_nonzero_values() {
        // d=1.0, dmin=0.0, scales[0..4]=2, scales[4..8]=0, mins all 0.
        // All quant bytes = 0x53 → lo nibble=3, hi nibble=5.
        //
        // Expected output per sub-block group:
        //   g=0: base_lo=0..32   → d*scales[0]*3 = 6.0
        //         base_hi=32..64  → d*scales[1]*5 = 10.0
        //   g=1: base_lo=64..96  → 6.0
        //         base_hi=96..128 → 10.0
        //   g=2/3: scales[4..8]=0  → 0.0
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0 (f16)
        block[2] = 0x00;
        block[3] = 0x00; // dmin = 0.0
                         // scales_bytes[0..4] = 0x02 → scales[0..4] = 2, mins[0..4] = 0
        block[4] = 0x02;
        block[5] = 0x02;
        block[6] = 0x02;
        block[7] = 0x02;
        // scales_bytes[4..12] = 0x00 → mins[0..4] = 0, scales[4..8] = 0
        block[8..16].fill(0x00);
        block[16..144].fill(0x53);

        let out = dequantize_q4_k(&block, 256).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate().take(32) {
            assert!((v - 6.0).abs() < 1e-6, "i={i} got {v}");
        }
        for (i, &v) in out.iter().enumerate().take(64).skip(32) {
            assert!((v - 10.0).abs() < 1e-6, "i={i} got {v}");
        }
        for (i, &v) in out.iter().enumerate().take(96).skip(64) {
            assert!((v - 6.0).abs() < 1e-6, "i={i} got {v}");
        }
        for (i, &v) in out.iter().enumerate().take(128).skip(96) {
            assert!((v - 10.0).abs() < 1e-6, "i={i} got {v}");
        }
        for (i, &v) in out.iter().enumerate().skip(128) {
            assert!((v - 0.0).abs() < 1e-6, "i={i} got {v}");
        }
    }

    // ── scaled_add correctness (q4k and q6k) ──

    #[test]
    fn q4k_row_scaled_add_matches_alpha_times_deq() {
        let data = synth_q4k_block(13);
        let alpha = 0.25_f32;
        let deq = dequantize_q4_k(&data, 256).unwrap();
        let mut out = vec![0.0f32; 256];
        q4k_row_scaled_add(&data, alpha, &mut out).unwrap();
        for (i, (&o, &d)) in out.iter().zip(&deq).enumerate() {
            let expected = alpha * d;
            assert!(
                (o - expected).abs() < 1e-5,
                "idx {i}: got {o} expected {expected}"
            );
        }
    }

    #[test]
    fn q6k_row_scaled_add_matches_alpha_times_deq() {
        let data = synth_q6k_block(21);
        let alpha = 0.5_f32;
        let deq = dequantize_q6_k(&data, 256).unwrap();
        let mut out = vec![0.0f32; 256];
        q6k_row_scaled_add(&data, alpha, &mut out).unwrap();
        for (i, (&o, &d)) in out.iter().zip(&deq).enumerate() {
            let expected = alpha * d;
            assert!(
                (o - expected).abs() < 1e-5,
                "idx {i}: got {o} expected {expected}"
            );
        }
    }

    #[test]
    fn q4k_row_scaled_add_rejects_misaligned() {
        let mut out = vec![0.0f32; 300]; // not a multiple of 256
        match q4k_row_scaled_add(&[0u8; 144], 1.0, &mut out) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn q6k_row_scaled_add_rejects_misaligned() {
        let mut out = vec![0.0f32; 300];
        match q6k_row_scaled_add(&[0u8; 210], 1.0, &mut out) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    // ── q4k_row_dot / q4k_row_scaled_add error-path coverage ──

    #[test]
    fn q4k_row_dot_rejects_misaligned_x_length() {
        let x = vec![0.0f32; 200];
        match q4k_row_dot(&[0u8; 144], &x) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn q4k_row_dot_rejects_short_data() {
        let x = vec![0.0f32; 256];
        // n_blocks = 1 requires 144 bytes; supply 16.
        match q4k_row_dot(&[0u8; 16], &x) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn q4k_row_scaled_add_rejects_short_data() {
        let mut out = vec![0.0f32; 256];
        match q4k_row_scaled_add(&[0u8; 16], 1.0, &mut out) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    // ── q6k_row_dot / q6k_row_scaled_add error-path coverage ──

    #[test]
    fn q6k_row_dot_rejects_misaligned_x_length() {
        // x length 200 is not a multiple of 256.
        let x = vec![0.0f32; 200];
        match q6k_row_dot(&[0u8; 210], &x) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("not a multiple of"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn q6k_row_dot_rejects_short_data() {
        // n_blocks = 1 requires 210 bytes; supply 16.
        let x = vec![0.0f32; 256];
        match q6k_row_dot(&[0u8; 16], &x) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn q6k_row_scaled_add_rejects_short_data() {
        let mut out = vec![0.0f32; 256];
        match q6k_row_scaled_add(&[0u8; 16], 1.0, &mut out) {
            Err(ModelError::Parse(msg)) => assert!(msg.contains("data short"), "got: {msg}"),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    /// Cover `q4k_row_scaled_add_scalar` directly — on aarch64 the
    /// production path goes through the NEON variant so the scalar
    /// reference is `#[allow(dead_code)]`.
    #[test]
    fn q4k_row_scaled_add_scalar_matches_alpha_times_deq() {
        use super::q4_k::q4k_row_scaled_add_scalar;
        let data = synth_q4k_block(17);
        let alpha = 0.375_f32;
        let deq = dequantize_q4_k(&data, 256).unwrap();
        let mut out = vec![0.0f32; 256];
        q4k_row_scaled_add_scalar(&data, alpha, &mut out, 1);
        for (i, (&o, &d)) in out.iter().zip(&deq).enumerate() {
            let expected = alpha * d;
            assert!(
                (o - expected).abs() < 1e-5,
                "idx {i}: got {o} expected {expected}"
            );
        }
    }
}
