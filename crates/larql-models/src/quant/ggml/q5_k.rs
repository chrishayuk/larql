//! Q5_K — 256-element super-block, 176 bytes/block. Mid-density
//! K-quant: 5 bits per element. Used by unsloth Q4_K_M / Q4_K_S
//! variants for selected tensors (attention weights, output proj).
//!
//! Layout (matches llama.cpp's `block_q5_K`):
//!   bytes  0-1:   d    (f16 global scale)
//!   bytes  2-3:   dmin (f16 global min)
//!   bytes  4-15:  12 bytes of packed 6-bit scales + 6-bit mins (8 each)
//!                 — same `get_scale_min_k4` packing as Q4_K
//!   bytes 16-47:  qh — high-bits (1 bit per element, 256 bits = 32 bytes)
//!                 The high-bit assignment loops outer-sub-block-major:
//!                 each `qh[l]` byte holds bits for 4 different output
//!                 positions across the super-block.
//!   bytes 48-175: qs — low-nibble quants (2 nibbles per byte, 256 values)
//!
//! Each (scale, min) pair governs 32 elements within the 256-element
//! super-block. The per-element value is:
//!   q5 = (qs[i].nibble) | (qh[i] >> bit_in_qh) & 1 << 4
//!   out[i] = d * scale[sb] * q5 - dmin * min[sb]
//!
//! The `q5` range is [0, 31] (5 bits unsigned).

use crate::ModelError;

use super::check_block_input;
use crate::quant::half::f16_to_f32;

pub fn dequantize_q5_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 176; // 2 + 2 + 12 + 32 + 128
    let super_block = 256;
    let n_blocks = check_block_input("Q5_K", data, n_elements, super_block, block_size)?;
    let mut out = vec![0.0f32; n_elements];

    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        // Same 6-bit scale+min unpacking as Q4_K.
        let scales_bytes = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..8 {
            if j < 4 {
                scales[j] = scales_bytes[j] & 0x3F;
                mins[j] = scales_bytes[j + 4] & 0x3F;
            } else {
                scales[j] = (scales_bytes[j + 4] & 0x0F) | ((scales_bytes[j - 4] >> 6) << 4);
                mins[j] = (scales_bytes[j + 4] >> 4) | ((scales_bytes[j] >> 6) << 4);
            }
        }

        let qh = &block[16..48]; // 32 bytes of high bits
        let qs = &block[48..176]; // 128 bytes of low nibbles

        // Reference: llama.cpp `dequantize_row_q5_K`. Outer loop steps
        // through the 256-element super-block in chunks of 64 (= 2
        // sub-blocks per iteration). Two state-bits `u1` / `u2` walk
        // through the qh bit-plane, each shifted left by 2 per outer
        // iteration to address a different bit-pair.
        let sb_base = sb * super_block;
        let mut ql_off = 0usize;
        let mut is = 0usize; // sub-block scale index
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut out_off = sb_base;
        for _outer in 0..(super_block / 64) {
            let sc1 = scales[is] as f32;
            let mn1 = mins[is] as f32;
            let d1 = d * sc1;
            let m1 = dmin * mn1;
            let sc2 = scales[is + 1] as f32;
            let mn2 = mins[is + 1] as f32;
            let d2 = d * sc2;
            let m2 = dmin * mn2;

            // First half (32 elements) — low nibble of qs + high bit u1.
            for l in 0..32 {
                let qlo = qs[ql_off + l] & 0x0F;
                let qhi_bit = if (qh[l] & u1) != 0 { 16u8 } else { 0u8 };
                let q5 = qlo + qhi_bit;
                out[out_off + l] = d1 * q5 as f32 - m1;
            }
            out_off += 32;
            // Second half (32 elements) — high nibble of qs + high bit u2.
            for l in 0..32 {
                let qlo = qs[ql_off + l] >> 4;
                let qhi_bit = if (qh[l] & u2) != 0 { 16u8 } else { 0u8 };
                let q5 = qlo + qhi_bit;
                out[out_off + l] = d2 * q5 as f32 - m2;
            }
            out_off += 32;
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct one super-block of Q5_K bytes representing a known
    /// pattern, then verify dequantize round-trips the expected f32s.
    /// This is the load-bearing parity check — the byte layout has to
    /// match llama.cpp exactly or extraction will silently produce
    /// garbage for any tensor that uses Q5_K.
    #[test]
    fn dequantize_q5_k_single_block_uniform_zero() {
        // All-zero block: d=0, dmin=0, all quants 0 → out should be 0.
        let block = vec![0u8; 176];
        let out = dequantize_q5_k(&block, 256).unwrap();
        assert_eq!(out.len(), 256);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn dequantize_q5_k_uniform_mid_value() {
        // d = 1.0, dmin = 0, all 8 scales = 1, all 8 mins = irrelevant
        // (dmin=0), all qs nibbles = 0xF (15), all qh bits = 0
        // → q5 = 15 → out = 1.0 * 1 * 15 - 0 = 15.0 for every element.
        //
        // Scale-packing (12 bytes at block[4..16]):
        //   scales[0..4] = block[4..8] & 0x3F          → set block[4..8] = 1
        //   mins[0..4]   = block[8..12] & 0x3F         → mins value doesn't
        //                                                 matter (dmin = 0)
        //   scales[4..8] = (block[12..16] & 0x0F) |
        //                  ((block[4..8] >> 6) << 4)   → set block[12..16]
        //                                                 low-nibble = 1
        //   mins[4..8]   = (block[12..16] >> 4) |
        //                  ((block[8..12] >> 6) << 4)  → dmin = 0, ignored
        let mut block = vec![0u8; 176];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0 (f16)
        for j in 0..4 {
            block[4 + j] = 0x01; // scales[0..4] = 1
            block[12 + j] = 0x01; // scales[4..8] low-nibble = 1
        }
        for i in 48..176 {
            block[i] = 0xFF; // every qs nibble = 0xF
        }
        // qh bytes 16..48 stay zero → high bit never set → q5 = 15.

        let out = dequantize_q5_k(&block, 256).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert!((v - 15.0).abs() < 1e-5, "idx {i}: got {v}");
        }
    }

    #[test]
    fn dequantize_q5_k_high_bit_lifts_to_31() {
        // Same as above but qh[0] bit 0 = 1 → position 0 in the first
        // 32 elements should get q5 = 15 + 16 = 31, so out[0] = 31.0.
        let mut block = vec![0u8; 176];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        for j in 0..4 {
            block[4 + j] = 0x01; // scales[0..4] = 1
            block[12 + j] = 0x01; // scales[4..8] low-nibble = 1
        }
        for i in 48..176 {
            block[i] = 0xFF;
        }
        block[16] = 0b0000_0001; // qh[0] bit 0 set → applies to element 0 of first half.

        let out = dequantize_q5_k(&block, 256).unwrap();
        assert!((out[0] - 31.0).abs() < 1e-5, "out[0] = {}", out[0]);
        // Element 1 of first half (qh[1] bit 0) is unchanged at 15.
        assert!((out[1] - 15.0).abs() < 1e-5, "out[1] = {}", out[1]);
    }

    #[test]
    fn dequantize_q5_k_multiple_blocks() {
        // Two zero-blocks → 512 zeros.
        let block = vec![0u8; 176 * 2];
        let out = dequantize_q5_k(&block, 512).unwrap();
        assert_eq!(out.len(), 512);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn dequantize_q5_k_rejects_short_input() {
        let block = vec![0u8; 100]; // Less than one super-block.
        let result = dequantize_q5_k(&block, 256);
        assert!(result.is_err());
    }

    #[test]
    fn dequantize_q5_k_rejects_misaligned_n_elements() {
        let block = vec![0u8; 176];
        // n_elements not a multiple of 256.
        assert!(dequantize_q5_k(&block, 128).is_err());
        assert!(dequantize_q5_k(&block, 257).is_err());
    }
}
