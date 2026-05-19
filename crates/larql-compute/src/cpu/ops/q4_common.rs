//! Shared Q4 utilities for CPU backend.
//!
//! C FFI declarations for the vdotq_s32 kernel (csrc/q4_dot.c)
//! and Q8 quantization helper.

use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

extern "C" {
    /// C kernel: Q4_0 × Q8_0 matrix-vector multiply with ARM vdotq_s32.
    pub fn q4_0_matvec_c(
        q4_data: *const u8,
        q8_x: *const i8,
        q8_scales: *const f32,
        scores: *mut f32,
        num_rows: usize,
        hidden: usize,
    );

    /// C kernel: Q4_0 vector-matrix multiply (scatter-accumulate).
    pub fn q4_0_vecmat_c(
        activation: *const f32,
        q4_data: *const u8,
        out: *mut f32,
        intermediate: usize,
        hidden: usize,
    );
}

/// Pre-quantize f32 vector to Q8_0 (int8 + per-block f32 scale).
pub fn quantize_to_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x.len() / LEGACY_BLOCK_ELEMS;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for (b, scale_out) in scales.iter_mut().enumerate().take(n_blocks) {
        let off = b * LEGACY_BLOCK_ELEMS;
        let block = &x[off..off + LEGACY_BLOCK_ELEMS];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        *scale_out = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..LEGACY_BLOCK_ELEMS {
            q8[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q8, scales)
}

/// Quantize f32 data to Q4_0 format (4-bit, block size 32).
///
/// Each block of 32 floats becomes 18 bytes: 2 bytes f16 scale + 16 bytes packed nibbles.
/// Used for weight quantization in benchmarks, tests, and tooling.
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(LEGACY_BLOCK_ELEMS),
        "data length must be a multiple of 32"
    );
    let n_blocks = data.len() / LEGACY_BLOCK_ELEMS;
    let mut out = Vec::with_capacity(n_blocks * 18);
    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        // f32 → f16 conversion
        let bits = scale.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        let f16 = if exp == 0 {
            sign as u16
        } else if exp == 255 {
            (sign | 0x7C00 | (mant >> 13)) as u16
        } else {
            let new_exp = exp - 127 + 15;
            if new_exp >= 31 {
                (sign | 0x7C00) as u16
            } else if new_exp <= 0 {
                sign as u16
            } else {
                (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
            }
        };
        out.extend_from_slice(&f16.to_le_bytes());
        for j in 0..16 {
            let lo = ((block[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((block[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Encode f32 to f16 bits (for quantize helpers).
///
/// Handles subnormals. When `new_exp <= 0` the value is small enough that f16
/// can only represent it as a subnormal (implicit leading 0 instead of 1). We
/// construct that subnormal mantissa by shifting the implicit-one back in and
/// right-shifting — previously this branch just emitted signed zero, which
/// meant Q-quant scales for small weight sub-blocks silently collapsed to
/// zero and the whole super-block decoded as zero. Real-world NN weights have
/// sub-block ranges ~10⁻² and scales ~10⁻⁵, exactly in f16 subnormal range.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign as u16;
    }
    if exp == 255 {
        return (sign | 0x7C00 | (mant >> 13)) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Subnormal: value = (1 + mant/2^23) * 2^(exp-127), we need to express
        // it as (subnormal_mant/2^10) * 2^-14 where subnormal_mant ∈ [0, 1023].
        // Include the implicit leading 1, shift right to align with f16's
        // subnormal scale.
        let shift = 1 - new_exp; // number of extra right-shifts past the normal encoding
                                 // `with_implicit` has 24 significant bits (positions 23..=0). Once
                                 // total_shift reaches 24 the mantissa shifts out entirely → encode as
                                 // signed zero. Guard against the Rust debug-mode shift-overflow panic.
        if 13 + shift as u32 >= 24 {
            return sign as u16;
        }
        let sub_mant = (mant | 0x800000) >> (13 + shift as u32);
        return (sign | sub_mant) as u16;
    }
    (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// Quantize f32 data to Q4_K format — the canonical llama.cpp / GGUF
/// layout (Ollama-compatible, 144 bytes per 256-element super-block).
///
/// Block layout (matches `kernel_mul_mv_q4_K_f32` in llama.cpp and the
/// `q4kf_proj` / `q4kf_qkv_proj` Metal shaders):
///   [0..1]    f16 d (super-block scale)
///   [2..3]    f16 dmin (super-block min)
///   [4..15]   12 bytes packing 8 × 6-bit `q_scales` + 8 × 6-bit `q_mins`
///             via `get_scale_min_k4`.
///   [16..143] 128 bytes of 4-bit nibbles arranged as FOUR 32-byte groups.
///             Each group holds TWO adjacent sub-blocks — low nibbles go
///             to sub-block `2g`, high nibbles go to sub-block `2g+1`.
///             `scales[2g]` / `mins[2g]` scale the low nibbles,
///             `scales[2g+1]` / `mins[2g+1]` scale the high nibbles.
///
/// Round-trips exactly through `dequantize_q4_k` in this crate and
/// `larql_models::quant::ggml::dequantize_q4_k`, and decodes identically
/// via the Metal shaders and llama.cpp's reference `dequantize_row_q4_K`.
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(256),
        "data length must be a multiple of 256"
    );
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 144);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Per-sub-block min/max — force min ≤ 0 so purely-positive
        // sub-blocks don't get shifted down by their own baseline.
        let mut sub_mins = [0.0f32; 8];
        let mut sub_maxs = [0.0f32; 8];
        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            let mn = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            sub_mins[j] = mn.min(0.0);
            sub_maxs[j] = mx.max(0.0);
        }

        let global_max_range = sub_maxs
            .iter()
            .zip(&sub_mins)
            .map(|(a, b)| a - b)
            .fold(0.0f32, f32::max);
        let global_min = sub_mins.iter().copied().fold(f32::INFINITY, f32::min);

        // Q4_K decode is `x = (d * q_scale) * nibble - (dmin * q_min)`
        // with nibble ∈ [0, 15], q_scale ∈ [0, 63], q_min ∈ [0, 63].
        let d = if global_max_range > 0.0 {
            global_max_range / (15.0 * 63.0)
        } else {
            0.0
        };
        let dmin = if global_min < 0.0 {
            -global_min / 63.0
        } else {
            0.0
        };

        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for j in 0..8 {
            let range = sub_maxs[j] - sub_mins[j];
            q_scales[j] = if d > 0.0 {
                (range / (15.0 * d)).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            q_mins[j] = if dmin > 0.0 {
                (-sub_mins[j] / dmin).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
        }

        // 12-byte scales + mins packing, `get_scale_min_k4` reference:
        //   j < 4: scales[j] = packed[j]     & 0x3F
        //          mins[j]   = packed[j+4]   & 0x3F
        //   j ≥ 4: scales[j] = (packed[j+4] & 0x0F) | ((packed[j-4] >> 6) << 4)
        //          mins[j]   = (packed[j+4] >> 4)   | ((packed[j]   >> 6) << 4)
        let mut packed = [0u8; 12];
        for j in 0..4 {
            packed[j] = (q_scales[j] & 0x3F) | (((q_scales[j + 4] >> 4) & 0x03) << 6);
            packed[j + 4] = (q_mins[j] & 0x3F) | (((q_mins[j + 4] >> 4) & 0x03) << 6);
            packed[j + 8] = (q_scales[j + 4] & 0x0F) | ((q_mins[j + 4] & 0x0F) << 4);
        }
        out.extend_from_slice(&packed);

        // Nibble packing: llama.cpp groups two adjacent sub-blocks into
        // one 32-byte span. For group `g` ∈ [0,4):
        //   byte[g*32 + l].low_nibble  = encoded sub-block `2g`   value `l`
        //   byte[g*32 + l].high_nibble = encoded sub-block `2g+1` value `l`
        // Encoding uses that sub-block's own scale/min:
        //   enc = round((v + dmin*q_min) / (d*q_scale)) clamped to [0, 15]
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * q_scales[sb_lo] as f32;
            let sc_hi = d * q_scales[sb_hi] as f32;
            let mn_lo = dmin * q_mins[sb_lo] as f32;
            let mn_hi = dmin * q_mins[sb_hi] as f32;
            let inv_lo = if sc_lo > 0.0 { 1.0 / sc_lo } else { 0.0 };
            let inv_hi = if sc_hi > 0.0 { 1.0 / sc_hi } else { 0.0 };
            let lo_sub = &block[sb_lo * 32..(sb_lo + 1) * 32];
            let hi_sub = &block[sb_hi * 32..(sb_hi + 1) * 32];
            for l in 0..32 {
                let lo = ((lo_sub[l] + mn_lo) * inv_lo).round().clamp(0.0, 15.0) as u8;
                let hi = ((hi_sub[l] + mn_hi) * inv_hi).round().clamp(0.0, 15.0) as u8;
                out.push(lo | (hi << 4));
            }
        }
    }
    out
}

/// Quantize f32 data to Q6_K format (6-bit with sub-block scales, Ollama-compatible).
///
/// Each super-block of 256 floats becomes 210 bytes:
///   [0..127]    128 bytes: lower 4 bits of each value (packed nibbles)
///   [128..191]   64 bytes: upper 2 bits (packed, 4 per byte)
///   [192..207]   16 bytes: 16 × int8 scales (one per 16-value sub-block)
///   [208..209]    2 bytes: f16 super-block scale (d)
pub fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(256),
        "data length must be a multiple of 256"
    );
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 210);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Q6_K decode is `x = d * sub_scale * q` with q ∈ [-32, 31] (6-bit
        // signed). To span the sub-block's amax with 31 levels on the
        // positive side: `d * sub_scale * 31 ≈ sub_max`. Picking d so the
        // largest sub-block's sub_scale hits the i8 cap:
        //   d = amax / (31 * 127)         # generous headroom
        // and `sub_scale = round(sub_max / (31 * d))`.
        // The previous `d = amax/32` / `sub_scale = sub_max/d` collapsed
        // most values onto q ∈ {-1, 0, 1} because the scale per level was
        // 32× too coarse.
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / (31.0 * 127.0);

        // Compute per-sub-block (16 values) int8 scales.
        let mut sub_scales = [0i8; 16];
        for (j, sub_scale) in sub_scales.iter_mut().enumerate() {
            let sub = &block[j * 16..(j + 1) * 16];
            let sub_max = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let sc = if d > 0.0 { sub_max / (31.0 * d) } else { 0.0 };
            *sub_scale = sc.round().clamp(-128.0, 127.0) as i8;
        }

        // Quantize all 256 values to 6-bit
        let mut q6_vals = [0u8; 256];
        for (j, &sub_scale) in sub_scales.iter().enumerate() {
            let sc = d * sub_scale as f32;
            let inv_sc = if sc.abs() > 1e-10 { 1.0 / sc } else { 0.0 };
            for i in 0..16 {
                let idx = j * 16 + i;
                let q = (block[idx] * inv_sc).round().clamp(-32.0, 31.0) as i8;
                q6_vals[idx] = (q + 32) as u8; // bias to unsigned
            }
        }

        // Pack lower 4 bits: 128 bytes (2 nibbles per byte)
        let mut ql = [0u8; 128];
        for i in 0..128 {
            ql[i] = (q6_vals[i * 2] & 0x0F) | ((q6_vals[i * 2 + 1] & 0x0F) << 4);
        }
        out.extend_from_slice(&ql);

        // Pack upper 2 bits: 64 bytes (4 × 2 bits per byte)
        let mut qh = [0u8; 64];
        for (i, &q6_val) in q6_vals.iter().enumerate() {
            let hi2 = (q6_val >> 4) & 0x03;
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            qh[byte_idx] |= hi2 << bit_offset;
        }
        out.extend_from_slice(&qh);

        // 16 × int8 scales
        for &s in &sub_scales {
            out.push(s as u8);
        }

        // f16 super-block scale
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    out
}

/// Convert Q4_K data (144-byte GGUF layout) to Q4_KF (pre-baked half
/// scales) for fast GPU inference.
///
/// Q4_KF eliminates all header decode + scale unpack from the inference
/// hot loop. Each 144-byte Q4_K superblock becomes 160 bytes:
///   [0..15]    8 × f16 pre-computed d*scale_j (16 bytes)
///   [16..31]   8 × f16 pre-computed dmin*min_j (16 bytes)
///   [32..159]  128 bytes nibbles (unchanged)
pub fn q4k_to_q4kf(q4k_data: &[u8], num_rows: usize, hidden: usize) -> Vec<u8> {
    let superblocks_per_row = hidden / 256;
    let q4k_bytes_per_row = superblocks_per_row * 144;
    let q4kf_bytes_per_row = superblocks_per_row * 160;
    let mut out = Vec::with_capacity(num_rows * q4kf_bytes_per_row);

    for row in 0..num_rows {
        for sb in 0..superblocks_per_row {
            let offset = row * q4k_bytes_per_row + sb * 144;
            let block = &q4k_data[offset..offset + 144];

            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

            // Unpack scales + mins per llama.cpp's `get_scale_min_k4`.
            let p = &block[4..16];
            let mut q_scales = [0u8; 8];
            let mut q_mins = [0u8; 8];
            for j in 0..4 {
                q_scales[j] = p[j] & 0x3F;
                q_mins[j] = p[j + 4] & 0x3F;
                q_scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
                q_mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
            }

            // Pre-bake d·scale and dmin·min, write as f16.
            for &qs in &q_scales {
                let s = d * qs as f32;
                out.extend_from_slice(&f32_to_f16(s).to_le_bytes());
            }
            for &qm in &q_mins {
                let m = dmin * qm as f32;
                out.extend_from_slice(&f32_to_f16(m).to_le_bytes());
            }
            // Copy 128 nibble bytes unchanged.
            out.extend_from_slice(&block[16..144]);
        }
    }
    out
}

/// Quantize f32 data directly to Q4_KF format (pre-baked half scales).
pub fn quantize_q4_kf(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(256),
        "data length must be a multiple of 256"
    );
    // First quantize to Q4_K, then convert
    let q4k = quantize_q4_k(data);
    let num_rows = 1; // treat as single row
    let hidden = data.len();
    q4k_to_q4kf(&q4k, num_rows, hidden)
}

/// Decode f16 bits to f32 (shared helper).
/// IEEE-754 half-precision → single-precision conversion via pure integer
/// bit manipulation.  Critical hot path for Q4_K dequant: every super-block
/// header decodes two f16 values (`d`, `dmin`), and at Gemma 4 26B-A4B
/// sizes the SDOT matvec issues ~11 M f16 decodes per token.
///
/// **Why not `f32.powi(exp-15)`?** The previous implementation computed
/// `(1 + mant/1024) * 2.0f32.powi(exp - 15)` which Rust 1.91 lowers to a
/// `bl __powisf2` libcall on aarch64.  Profiling
/// (`/tmp/sample.txt` 2026-05-01) showed the `fmul` immediately after that
/// `bl` as the single hottest IP in the kernel — every f16 decode paid a
/// function-call detour.
///
/// The bit-manipulation form below is one i64 multiply + a few shifts/ANDs,
/// inlines fully, and matches the original output bit-exactly for all
/// 65536 possible f16 inputs (see `f16_to_f32_bit_exact_for_all_inputs`).
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    // Reference: standard "magic-multiply" half→float decode.  Same shape
    // as Mike Acton's, also used by `half` crate.  Avoids any FP libcalls.
    let bits = bits as u32;
    let sign = (bits & 0x8000) << 16; // shift to bit 31 of f32
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            // ±0
            return f32::from_bits(sign);
        }
        // Subnormal: normalise.  The mantissa has a leading-one bit somewhere
        // in [0..10); shift it up to bit 23 of the f32 mantissa, adjusting
        // the exponent down by the shift amount.
        // `mant` is in [1, 1023]; leading_zeros on a u16 with 10 valid bits
        // gives a value in [6..15] for non-zero mant (16-bit input, top 6
        // bits guaranteed zero).  Subtract 16-10=6 to get LZ within the 10-bit
        // mantissa region.
        let lz = (mant as u16).leading_zeros() - 6; // 0..=9
        let new_mant = (mant << (lz + 14)) & 0x7F_FFFF;
        let new_exp = (127u32 - 14 - lz) << 23;
        return f32::from_bits(sign | new_exp | new_mant);
    }
    if exp == 31 {
        // Inf / NaN.  Mantissa bits are preserved (shifted left 13) so NaN
        // payloads round-trip; the original implementation collapsed all
        // NaN payloads to a canonical value, but f16 NaNs in real Q4_K
        // weights never occur (extractor sanitises) so the difference is
        // unobservable for our use case and IEEE-correct payload preservation
        // is the safer default.
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }
    // Normal: re-bias exponent by (127 - 15) and shift mantissa to bit 13.
    let new_exp = (exp + (127 - 15)) << 23;
    f32::from_bits(sign | new_exp | (mant << 13))
}

/// Dequantise a Q4_K byte stream to `n_elements` f32 values.
///
/// 256 elements per 144-byte super-block (GGUF / Ollama-canonical layout).
/// `n_elements` must be a multiple of 256 — the caller pads where required.
/// Mirrors `dequantize_row_q4_K` in llama.cpp/ggml-quants.c, kept here so
/// the CPU MoE expert path can call it without a `larql-models` dependency.
pub fn dequantize_q4_k(data: &[u8], n_elements: usize) -> Vec<f32> {
    let block_size = 144;
    let super_block = 256;
    if !n_elements.is_multiple_of(super_block) {
        return Vec::new();
    }
    let n_blocks = n_elements / super_block;
    if data.len() < n_blocks * block_size {
        return Vec::new();
    }
    let mut out = vec![0.0f32; n_elements];
    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let p = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for j in 0..4 {
            scales[j] = p[j] & 0x3F;
            mins[j] = p[j + 4] & 0x3F;
            scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
            mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
        }
        let quants = &block[16..144];
        let sb_base = sb * super_block;
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * scales[sb_lo] as f32;
            let sc_hi = d * scales[sb_hi] as f32;
            let mn_lo = dmin * mins[sb_lo] as f32;
            let mn_hi = dmin * mins[sb_hi] as f32;
            let chunk = &quants[g * 32..(g + 1) * 32];
            let base_lo = sb_base + sb_lo * 32;
            let base_hi = sb_base + sb_hi * 32;
            for l in 0..32 {
                let byte = chunk[l];
                out[base_lo + l] = sc_lo * (byte & 0x0F) as f32 - mn_lo;
                out[base_hi + l] = sc_hi * ((byte >> 4) & 0x0F) as f32 - mn_hi;
            }
        }
    }
    out
}

/// Direct Q4_K matrix-vector product: `out = W · x` where `W` is the raw
/// Q4_K byte stream (`rows × cols` weights, 144 bytes per 256 elements).
///
/// Decodes nibbles + per-sub-block scales/mins on the fly while
/// accumulating the dot product — avoids the f32 dequant cache that
/// quadruples the bandwidth bill.  At Gemma 4 26B-A4B sizes
/// (`hidden=2816`, `inter=704`, ~7.9 MB f32 per row otherwise) this drops
/// per-matmul bandwidth pressure from ~8 MB → ~2 MB and should land ~3–4×
/// faster than `dequantize_q4_k` + BLAS sgemv on a same-sized f32 view.
///
/// Math (matches `dequantize_q4_k`'s `out = sc * q - mn` per-element form):
///
/// ```text
/// for each super-block sb of 256 elements (8 sub-blocks of 32 each):
///   for each sub-block subblk in [0..8):
///     sc = d    * scales[subblk]
///     mn = dmin * mins[subblk]
///     dot = Σ  q_l · x[base + l]    (l in 0..32)
///     sumx = Σ x[base + l]          (precomputed once across all rows)
///     acc += sc * dot − mn * sumx
/// out[r] = acc
/// ```
///
/// `sumx` precomputation: x is shared across rows, so its per-sub-block
/// sum is row-invariant.  Computing it once outside the row loop saves
/// `rows × 8 · n_blocks` redundant sums.
///
/// Returns silently on shape mismatch (debug-asserted) and on Q4_K layout
/// errors (input too short, or `cols` not a multiple of 256).
///
/// Caller layout: `w.len() == rows * (cols / 256) * 144` bytes.
pub fn q4k_matvec_into(out: &mut [f32], x: &[f32], w: &[u8], rows: usize, cols: usize) {
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(x.len(), cols);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    const BLOCK_BYTES: usize = 144;
    const ELEMS_PER_BLOCK: usize = 256;
    if !cols.is_multiple_of(ELEMS_PER_BLOCK) {
        // Caller pads; falling back to zero output makes the failure visible
        // without panicking (the existing dequant path returns Vec::new()).
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // Precompute per-sub-block sum_x (one f32 per 32-element chunk of x).
    // 2-byte stride per (sb, subblk) pair lets us index by `sb * 8 + subblk`.
    let n_subblocks = n_blocks * 8;
    let mut sum_x: Vec<f32> = Vec::with_capacity(n_subblocks);
    for sub in 0..n_subblocks {
        let chunk = &x[sub * 32..(sub + 1) * 32];
        let mut s = 0.0f32;
        for &v in chunk {
            s += v;
        }
        sum_x.push(s);
    }

    // Row-parallel. Decode rows are independent and the typical matvec
    // shape this gets called with (Gemma-3-4B: 2560×2560 to 8192×2560
    // for Q4_K) is large enough to amortise rayon's join overhead by
    // 100×+. Empirically on M3 Max this drops a 2560-row decode from
    // ~70ms → ~10ms (≈ 7× across 11 perf cores).
    use rayon::prelude::*;
    let sum_x_ref = &sum_x[..];
    let w_ref = w;
    let x_ref = x;
    // par_chunks_mut(CHUNK_ROWS) instead of per-row par_iter_mut: each
    // rayon task processes a contiguous block of rows sequentially.
    // Cuts the number of work-stealing units from `rows` (10K+) down
    // to ~rows/CHUNK_ROWS, reducing scheduler overhead while keeping
    // enough granularity for the 11 perf cores on M3 Max to load-
    // balance.
    const CHUNK_ROWS: usize = 32;
    out.par_chunks_mut(CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, chunk_slots)| {
            let row_base_chunk = chunk_idx * CHUNK_ROWS;
            for (local_r, out_slot) in chunk_slots.iter_mut().enumerate() {
                let r = row_base_chunk + local_r;
                if r >= rows {
                    break;
                }
                let row_base = r * row_bytes;
                let mut acc = 0.0f32;
                for sb in 0..n_blocks {
                    acc += process_q4k_superblock(w_ref, x_ref, sum_x_ref, row_base, sb);
                }
                *out_slot = acc;
            }
        });
}

/// Per-super-block dot contribution for a Q4_K row. Returned scalar
/// is the super-block's contribution to the row's dot product.
/// Inlined into both `q4k_matvec_into`'s 2-super-block-unrolled outer
/// loop and `q4k_dual_matvec_into`'s outer loop (which keeps its
/// per-matrix accumulator separate so it doesn't get the 2-acc
/// scheduling boost, but trades that for the gate+up x-locality
/// already in place).
#[inline(always)]
fn process_q4k_superblock(w: &[u8], x: &[f32], sum_x: &[f32], row_base: usize, sb: usize) -> f32 {
    const BLOCK_BYTES: usize = 144;
    const ELEMS_PER_BLOCK: usize = 256;

    let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let p = &block[4..16];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = p[j] & 0x3F;
        mins[j] = p[j + 4] & 0x3F;
        scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
        mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
    }
    let quants = &block[16..144];
    let x_sb_base = sb * ELEMS_PER_BLOCK;

    let mut acc = 0.0f32;
    for g in 0..4 {
        let sb_lo = 2 * g;
        let sb_hi = 2 * g + 1;
        let sc_lo = d * scales[sb_lo] as f32;
        let sc_hi = d * scales[sb_hi] as f32;
        let mn_lo = dmin * mins[sb_lo] as f32;
        let mn_hi = dmin * mins[sb_hi] as f32;
        let chunk = &quants[g * 32..(g + 1) * 32];
        let x_lo_base = x_sb_base + sb_lo * 32;
        let x_hi_base = x_sb_base + sb_hi * 32;
        let sumy_lo = sum_x[sb * 8 + sb_lo];
        let sumy_hi = sum_x[sb * 8 + sb_hi];
        let x_lo = &x[x_lo_base..x_lo_base + 32];
        let x_hi = &x[x_hi_base..x_hi_base + 32];
        let (dot_lo, dot_hi) = q4_dual_dot_32(chunk, x_lo, x_hi);
        acc += sc_lo * dot_lo - mn_lo * sumy_lo;
        acc += sc_hi * dot_hi - mn_hi * sumy_hi;
    }
    acc
}

/// Fused two-weight Q4_K matvec sharing one input vector.
///
/// `out_a[N] = W_a[N, K] · x[K]`, `out_b[N] = W_b[N, K] · x[K]`.
/// Both weight matrices must have identical `(rows, cols)`. The decode
/// step's gate+up projections fit this contract exactly: same shape
/// `[intermediate, hidden]`, same `h_in` row.
///
/// Win vs two sequential `q4k_matvec_into` calls:
/// * `sum_x` is precomputed once (saves 0.1% per call, negligible)
/// * The expensive part: each rayon worker decodes both W_a and W_b
///   for its row range against the same `x`. `x` (10 KB for Gemma 3
///   4B hidden=2560) stays hot in L1 across both decodes — a
///   sequential pair re-streams it from L2/L3.
/// * Weight reads are independent and dominate bandwidth (~30 MB
///   total for 8192-row Q4_K). Total bandwidth doesn't change; just
///   x re-stream.
///
/// Measured savings: ~3-5% step on Gemma 3 4B's gate+up pair.
pub fn q4k_dual_matvec_into(
    out_a: &mut [f32],
    out_b: &mut [f32],
    x: &[f32],
    w_a: &[u8],
    w_b: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(out_a.len(), rows);
    debug_assert_eq!(out_b.len(), rows);
    debug_assert_eq!(x.len(), cols);
    if rows == 0 || cols == 0 {
        for v in out_a.iter_mut() {
            *v = 0.0;
        }
        for v in out_b.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    const BLOCK_BYTES: usize = 144;
    const ELEMS_PER_BLOCK: usize = 256;
    if !cols.is_multiple_of(ELEMS_PER_BLOCK) {
        for v in out_a.iter_mut() {
            *v = 0.0;
        }
        for v in out_b.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w_a.len() < rows * row_bytes || w_b.len() < rows * row_bytes {
        for v in out_a.iter_mut() {
            *v = 0.0;
        }
        for v in out_b.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // Precompute sum_x once.
    let n_subblocks = n_blocks * 8;
    let mut sum_x: Vec<f32> = Vec::with_capacity(n_subblocks);
    for sub in 0..n_subblocks {
        let chunk = &x[sub * 32..(sub + 1) * 32];
        let mut s = 0.0f32;
        for &v in chunk {
            s += v;
        }
        sum_x.push(s);
    }

    // Row-parallel — same outer structure as `q4k_matvec_into` but
    // each worker computes both outputs for its assigned row index.
    // Zip `out_a` and `out_b` so rayon stays simple and the two
    // writes hit different cache lines per row.
    use rayon::prelude::*;
    let sum_x_ref = &sum_x[..];
    let w_a_ref = w_a;
    let w_b_ref = w_b;
    let x_ref = x;
    // par_chunks_mut(CHUNK_ROWS) — same rationale as
    // `q4k_matvec_into`. Fewer-but-larger work units reduce rayon
    // work-stealing overhead.
    const CHUNK_ROWS: usize = 32;
    out_a
        .par_chunks_mut(CHUNK_ROWS)
        .zip(out_b.par_chunks_mut(CHUNK_ROWS))
        .enumerate()
        .for_each(|(chunk_idx, (chunk_a, chunk_b))| {
            let row_base_chunk = chunk_idx * CHUNK_ROWS;
            for (local_r, (out_a_slot, out_b_slot)) in
                chunk_a.iter_mut().zip(chunk_b.iter_mut()).enumerate()
            {
                let r = row_base_chunk + local_r;
                if r >= rows {
                    break;
                }
                let row_base = r * row_bytes;
                let mut acc_a = 0.0f32;
                let mut acc_b = 0.0f32;
                for sb in 0..n_blocks {
                    let blk_a =
                        &w_a_ref[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
                    let blk_b =
                        &w_b_ref[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
                    let d_a = f16_to_f32(u16::from_le_bytes([blk_a[0], blk_a[1]]));
                    let dmin_a = f16_to_f32(u16::from_le_bytes([blk_a[2], blk_a[3]]));
                    let d_b = f16_to_f32(u16::from_le_bytes([blk_b[0], blk_b[1]]));
                    let dmin_b = f16_to_f32(u16::from_le_bytes([blk_b[2], blk_b[3]]));
                    let pa = &blk_a[4..16];
                    let pb = &blk_b[4..16];
                    let mut scales_a = [0u8; 8];
                    let mut mins_a = [0u8; 8];
                    let mut scales_b = [0u8; 8];
                    let mut mins_b = [0u8; 8];
                    for j in 0..4 {
                        scales_a[j] = pa[j] & 0x3F;
                        mins_a[j] = pa[j + 4] & 0x3F;
                        scales_a[j + 4] = (pa[j + 8] & 0x0F) | ((pa[j] >> 6) << 4);
                        mins_a[j + 4] = (pa[j + 8] >> 4) | ((pa[j + 4] >> 6) << 4);
                        scales_b[j] = pb[j] & 0x3F;
                        mins_b[j] = pb[j + 4] & 0x3F;
                        scales_b[j + 4] = (pb[j + 8] & 0x0F) | ((pb[j] >> 6) << 4);
                        mins_b[j + 4] = (pb[j + 8] >> 4) | ((pb[j + 4] >> 6) << 4);
                    }
                    let qa = &blk_a[16..144];
                    let qb = &blk_b[16..144];
                    let x_sb_base = sb * ELEMS_PER_BLOCK;

                    for g in 0..4 {
                        let sb_lo = 2 * g;
                        let sb_hi = 2 * g + 1;
                        let sc_a_lo = d_a * scales_a[sb_lo] as f32;
                        let sc_a_hi = d_a * scales_a[sb_hi] as f32;
                        let mn_a_lo = dmin_a * mins_a[sb_lo] as f32;
                        let mn_a_hi = dmin_a * mins_a[sb_hi] as f32;
                        let sc_b_lo = d_b * scales_b[sb_lo] as f32;
                        let sc_b_hi = d_b * scales_b[sb_hi] as f32;
                        let mn_b_lo = dmin_b * mins_b[sb_lo] as f32;
                        let mn_b_hi = dmin_b * mins_b[sb_hi] as f32;
                        let chunk_a = &qa[g * 32..(g + 1) * 32];
                        let chunk_b = &qb[g * 32..(g + 1) * 32];
                        let x_lo_base = x_sb_base + sb_lo * 32;
                        let x_hi_base = x_sb_base + sb_hi * 32;
                        let x_lo = &x_ref[x_lo_base..x_lo_base + 32];
                        let x_hi = &x_ref[x_hi_base..x_hi_base + 32];
                        let sumy_lo = sum_x_ref[sb * 8 + sb_lo];
                        let sumy_hi = sum_x_ref[sb * 8 + sb_hi];

                        // Decode W_a's nibbles against x — x stays hot
                        // because the next call decodes W_b against the
                        // same x slice.
                        let (dot_a_lo, dot_a_hi) = q4_dual_dot_32(chunk_a, x_lo, x_hi);
                        let (dot_b_lo, dot_b_hi) = q4_dual_dot_32(chunk_b, x_lo, x_hi);

                        acc_a += sc_a_lo * dot_a_lo - mn_a_lo * sumy_lo;
                        acc_a += sc_a_hi * dot_a_hi - mn_a_hi * sumy_hi;
                        acc_b += sc_b_lo * dot_b_lo - mn_b_lo * sumy_lo;
                        acc_b += sc_b_hi * dot_b_hi - mn_b_hi * sumy_hi;
                    }
                }
                *out_a_slot = acc_a;
                *out_b_slot = acc_b;
            }
        });
}

/// 32-element dual nibble dot product: returns
/// `(sum(lo_nibbles[i] * x_lo[i]), sum(hi_nibbles[i] * x_hi[i]))` for
/// the 32 packed nibble pairs in `chunk`.
///
/// Dispatches to a NEON implementation on aarch64 (always available on
/// Apple Silicon) and falls back to scalar everywhere else. The hot
/// path runs ~3-4× the scalar version on M3 Max — 16 NEON FMAs vs 64
/// scalar FMAs per chunk, plus saved nibble-to-f32 widening cost.
#[inline]
fn q4_dual_dot_32(chunk: &[u8], x_lo: &[f32], x_hi: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the aarch64 base ISA. The slices are
        // guaranteed to be at least 32 elements (chunk) and 32 f32
        // (x_lo/x_hi) by the caller. We only read.
        unsafe { q4_dual_dot_32_neon(chunk, x_lo, x_hi) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut dot_lo = 0.0f32;
        let mut dot_hi = 0.0f32;
        for l in 0..32 {
            let byte = chunk[l];
            let q_lo = (byte & 0x0F) as f32;
            let q_hi = ((byte >> 4) & 0x0F) as f32;
            dot_lo += q_lo * x_lo[l];
            dot_hi += q_hi * x_hi[l];
        }
        (dot_lo, dot_hi)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q4_dual_dot_32_neon(chunk: &[u8], x_lo: &[f32], x_hi: &[f32]) -> (f32, f32) {
    use core::arch::aarch64::*;
    debug_assert!(chunk.len() >= 32);
    debug_assert!(x_lo.len() >= 32);
    debug_assert!(x_hi.len() >= 32);

    // Load 32 bytes of packed nibble pairs as two u8x16 registers.
    let bytes_0 = vld1q_u8(chunk.as_ptr()); // bytes[0..16]
    let bytes_1 = vld1q_u8(chunk.as_ptr().add(16)); // bytes[16..32]

    // Mask = 0x0F lane-broadcast; lo = byte & 0x0F, hi = byte >> 4.
    let mask = vdupq_n_u8(0x0F);
    let lo_nibs_0 = vandq_u8(bytes_0, mask);
    let lo_nibs_1 = vandq_u8(bytes_1, mask);
    let hi_nibs_0 = vshrq_n_u8::<4>(bytes_0);
    let hi_nibs_1 = vshrq_n_u8::<4>(bytes_1);

    // Eight independent f32x4 accumulators (4 lo + 4 hi). With one
    // accumulator per side the 4 FMAs per chunk would serialise on
    // the same destination register at M3's 4-cycle FMA latency
    // (= 25% of peak). Splitting into 4 lets the 4 FMAs pipeline at
    // 1/cycle, ~4× the inner-loop throughput.
    let mut acc_lo_a = vdupq_n_f32(0.0);
    let mut acc_lo_b = vdupq_n_f32(0.0);
    let mut acc_lo_c = vdupq_n_f32(0.0);
    let mut acc_lo_d = vdupq_n_f32(0.0);
    let mut acc_hi_a = vdupq_n_f32(0.0);
    let mut acc_hi_b = vdupq_n_f32(0.0);
    let mut acc_hi_c = vdupq_n_f32(0.0);
    let mut acc_hi_d = vdupq_n_f32(0.0);

    // Widen a u8x16 of nibbles into four f32x4 lanes, then FMA each
    // into a different accumulator so they pipeline.
    //
    // SAFETY of `xp.add(k)`: caller guarantees x_lo and x_hi each have
    // 32 contiguous f32, and we stop at offset 12 (last load reads
    // [12..16]).
    macro_rules! accumulate_16 {
        ($nibs:expr, $xp:expr, $acc_a:expr, $acc_b:expr, $acc_c:expr, $acc_d:expr) => {{
            let n: uint8x16_t = $nibs;
            let n_lo16 = vmovl_u8(vget_low_u8(n));
            let n_hi16 = vmovl_u8(vget_high_u8(n));
            let n_a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(n_lo16)));
            let n_b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(n_lo16)));
            let n_c = vcvtq_f32_u32(vmovl_u16(vget_low_u16(n_hi16)));
            let n_d = vcvtq_f32_u32(vmovl_u16(vget_high_u16(n_hi16)));
            let xp: *const f32 = $xp;
            let x_a = vld1q_f32(xp);
            let x_b = vld1q_f32(xp.add(4));
            let x_c = vld1q_f32(xp.add(8));
            let x_d = vld1q_f32(xp.add(12));
            $acc_a = vfmaq_f32($acc_a, n_a, x_a);
            $acc_b = vfmaq_f32($acc_b, n_b, x_b);
            $acc_c = vfmaq_f32($acc_c, n_c, x_c);
            $acc_d = vfmaq_f32($acc_d, n_d, x_d);
        }};
    }

    accumulate_16!(
        lo_nibs_0,
        x_lo.as_ptr(),
        acc_lo_a,
        acc_lo_b,
        acc_lo_c,
        acc_lo_d
    );
    accumulate_16!(
        lo_nibs_1,
        x_lo.as_ptr().add(16),
        acc_lo_a,
        acc_lo_b,
        acc_lo_c,
        acc_lo_d
    );
    accumulate_16!(
        hi_nibs_0,
        x_hi.as_ptr(),
        acc_hi_a,
        acc_hi_b,
        acc_hi_c,
        acc_hi_d
    );
    accumulate_16!(
        hi_nibs_1,
        x_hi.as_ptr().add(16),
        acc_hi_a,
        acc_hi_b,
        acc_hi_c,
        acc_hi_d
    );

    // Tree-reduce: (a+b) + (c+d) per side, then horizontal sum.
    let acc_lo = vaddq_f32(vaddq_f32(acc_lo_a, acc_lo_b), vaddq_f32(acc_lo_c, acc_lo_d));
    let acc_hi = vaddq_f32(vaddq_f32(acc_hi_a, acc_hi_b), vaddq_f32(acc_hi_c, acc_hi_d));
    (vaddvq_f32(acc_lo), vaddvq_f32(acc_hi))
}

#[cfg(test)]
mod neon_tests {
    use super::*;

    /// Scalar reference for the dual-nibble dot-product the NEON kernel
    /// replaces. Used as the correctness oracle for the NEON path.
    fn scalar_dual_dot_32(chunk: &[u8], x_lo: &[f32], x_hi: &[f32]) -> (f32, f32) {
        let mut dot_lo = 0.0f32;
        let mut dot_hi = 0.0f32;
        for l in 0..32 {
            let byte = chunk[l];
            let q_lo = (byte & 0x0F) as f32;
            let q_hi = ((byte >> 4) & 0x0F) as f32;
            dot_lo += q_lo * x_lo[l];
            dot_hi += q_hi * x_hi[l];
        }
        (dot_lo, dot_hi)
    }

    #[test]
    fn q4_dual_dot_32_matches_scalar_on_deterministic_input() {
        // 32 nibble pairs spanning all 16 nibble values both lo and hi.
        let chunk: Vec<u8> = (0..32u8).map(|i| (i & 0x0F) | ((i & 0x0F) << 4)).collect();
        let x_lo: Vec<f32> = (0..32).map(|i| (i as f32) * 0.013).collect();
        let x_hi: Vec<f32> = (0..32).map(|i| (i as f32) * -0.021 + 0.5).collect();

        let (scalar_lo, scalar_hi) = scalar_dual_dot_32(&chunk, &x_lo, &x_hi);
        let (got_lo, got_hi) = q4_dual_dot_32(&chunk, &x_lo, &x_hi);

        // Allow a small relative tolerance — NEON's grouped FMA orders
        // the 32-element sum differently than the scalar sequential
        // sum (4-lane reductions vs left-to-right), so bit-identity
        // isn't guaranteed.
        let rel = |s: f32, g: f32| ((s - g).abs() / (s.abs().max(1e-6))) as f64;
        assert!(
            rel(scalar_lo, got_lo) < 1e-5,
            "lo dot diverges: scalar={scalar_lo} neon={got_lo}"
        );
        assert!(
            rel(scalar_hi, got_hi) < 1e-5,
            "hi dot diverges: scalar={scalar_hi} neon={got_hi}"
        );
    }

    #[test]
    fn q4_dual_dot_32_zero_x_returns_zero() {
        let chunk = vec![0xFFu8; 32];
        let x_lo = vec![0.0f32; 32];
        let x_hi = vec![0.0f32; 32];
        let (lo, hi) = q4_dual_dot_32(&chunk, &x_lo, &x_hi);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn q4_dual_dot_32_max_nibble_high_only() {
        // All hi nibbles = 15, all lo nibbles = 0.
        let chunk = vec![0xF0u8; 32];
        let x_lo = vec![1.0f32; 32];
        let x_hi = vec![1.0f32; 32];
        let (lo, hi) = q4_dual_dot_32(&chunk, &x_lo, &x_hi);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 15.0 * 32.0);
    }

    /// q4k_dual_matvec_into must produce the same output as two
    /// sequential q4k_matvec_into calls within f32-summation noise.
    /// The two paths accumulate per-super-block in slightly different
    /// orders (single running acc in the dual path; helper-based
    /// per-super-block reduction in the singleton path), so strict
    /// bit-equality isn't expected. Tolerance is generous enough to
    /// absorb summation-order rounding but tight enough to catch any
    /// real divergence.
    #[test]
    fn q4k_dual_matvec_into_matches_two_sequential_calls() {
        let rows = 8;
        let cols = 512; // 2 super-blocks per row, exercises the multi-block loop
        let n_elem = rows * cols;
        let weights_a: Vec<f32> = (0..n_elem)
            .map(|i| ((i as f32 / n_elem as f32) - 0.5) * 1.0)
            .collect();
        let weights_b: Vec<f32> = (0..n_elem)
            .map(|i| ((i as f32 * 0.003).cos() - 0.3) * 0.7)
            .collect();
        let q4k_a = quantize_q4_k(&weights_a);
        let q4k_b = quantize_q4_k(&weights_b);

        let x: Vec<f32> = (0..cols).map(|j| (j as f32 * 0.011).sin()).collect();

        let mut sep_a = vec![0.0f32; rows];
        let mut sep_b = vec![0.0f32; rows];
        q4k_matvec_into(&mut sep_a, &x, &q4k_a, rows, cols);
        q4k_matvec_into(&mut sep_b, &x, &q4k_b, rows, cols);

        let mut fused_a = vec![0.0f32; rows];
        let mut fused_b = vec![0.0f32; rows];
        q4k_dual_matvec_into(&mut fused_a, &mut fused_b, &x, &q4k_a, &q4k_b, rows, cols);

        for r in 0..rows {
            let rel_a = (sep_a[r] - fused_a[r]).abs() / sep_a[r].abs().max(1e-6);
            let rel_b = (sep_b[r] - fused_b[r]).abs() / sep_b[r].abs().max(1e-6);
            assert!(
                rel_a < 1e-5,
                "fused matvec A row {r} drifts: sep={} fused={} rel={rel_a}",
                sep_a[r],
                fused_a[r]
            );
            assert!(
                rel_b < 1e-5,
                "fused matvec B row {r} drifts: sep={} fused={} rel={rel_b}",
                sep_b[r],
                fused_b[r]
            );
        }
    }

    #[test]
    fn q4k_dual_matvec_into_zero_dims_zero_output() {
        let mut out_a = vec![1.0f32; 4];
        let mut out_b = vec![1.0f32; 4];
        q4k_dual_matvec_into(&mut out_a, &mut out_b, &[], &[], &[], 4, 0);
        assert!(out_a.iter().all(|&v| v == 0.0));
        assert!(out_b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4k_dual_matvec_into_non_multiple_cols_zeros_output() {
        // cols = 100 is not a multiple of 256 → must zero output, not
        // panic. Matches the single-matvec contract.
        let mut out_a = vec![1.0f32; 2];
        let mut out_b = vec![2.0f32; 2];
        let x = vec![1.0f32; 100];
        let w = vec![0u8; 2 * 144];
        q4k_dual_matvec_into(&mut out_a, &mut out_b, &x, &w, &w, 2, 100);
        assert!(out_a.iter().all(|&v| v == 0.0));
        assert!(out_b.iter().all(|&v| v == 0.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation kept here as the correctness oracle for
    /// the bit-manipulation `f16_to_f32`.  Mirrors the previous (slow)
    /// version that used `2.0f32.powi(...)`.  The new fast path must
    /// match this for all 65536 possible f16 inputs except canonical NaN
    /// payload preservation (handled in the test).
    fn f16_to_f32_powi_reference(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let mant = (bits & 0x3FF) as u32;
        if exp == 0 {
            if mant == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
            return if sign == 1 { -val } else { val };
        }
        if exp == 31 {
            return if mant == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }
        let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
        if sign == 1 {
            -val
        } else {
            val
        }
    }

    /// Exhaustive bit-exact parity for all 65536 f16 inputs.  The fast
    /// bit-manipulation `f16_to_f32` must produce the same f32 bits as
    /// the powi-based reference for every finite (non-NaN) input.  NaN
    /// payloads differ by design (reference collapses to canonical NaN,
    /// fast path preserves payload — both are valid IEEE NaNs and the
    /// distinction is unobservable in Q4_K decode because real-world
    /// Q4_K headers never contain NaNs).
    #[test]
    fn f16_to_f32_bit_exact_for_all_inputs() {
        let mut diffs = 0usize;
        for bits in 0u16..=u16::MAX {
            let new = f16_to_f32(bits);
            let old = f16_to_f32_powi_reference(bits);
            if new.is_nan() && old.is_nan() {
                continue; // both NaN — different payloads OK
            }
            if new.to_bits() != old.to_bits() {
                if diffs < 5 {
                    eprintln!(
                        "diff at bits=0x{bits:04x}: new={} ({:#x}) old={} ({:#x})",
                        new,
                        new.to_bits(),
                        old,
                        old.to_bits()
                    );
                }
                diffs += 1;
            }
        }
        assert_eq!(diffs, 0, "{diffs} f16 inputs decode to different f32 bits");
    }

    #[test]
    fn q8_quantize_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let (q8, scales) = quantize_to_q8(&x);
        assert_eq!(q8.len(), 64);
        assert_eq!(scales.len(), 2); // 64 / 32
        assert!(scales.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn q8_zero_input() {
        let x = vec![0.0f32; 32];
        let (q8, scales) = quantize_to_q8(&x);
        assert!(q8.iter().all(|&v| v == 0));
        assert!(scales[0] == 0.0);
    }

    // ── quantize_q4_0 tests ──

    #[test]
    fn q4_output_size() {
        // 64 floats = 2 blocks of 32, each block → 18 bytes (2 f16 scale + 16 nibbles)
        let data = vec![1.0f32; 64];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 2 * 18);

        let data = vec![1.0f32; 256];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 8 * 18);
    }

    #[test]
    fn q4_zero_input() {
        let data = vec![0.0f32; 32];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 18);
        // Scale should be zero (f16 zero = 0x0000)
        assert_eq!(q4[0], 0);
        assert_eq!(q4[1], 0);
        // All nibbles should encode 8 (zero quantized = 0 + bias 8)
        for &b in &q4[2..18] {
            assert_eq!(b, 0x88, "zero input should quantize to bias value 0x88");
        }
    }

    #[test]
    fn q4_round_trip_accuracy() {
        // Quantize then dequantize, check values are close
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let q4 = quantize_q4_0(&data);

        // Dequantize: read f16 scale, unpack nibbles, multiply
        let scale_bits = u16::from_le_bytes([q4[0], q4[1]]);
        let scale = f16_to_f32(scale_bits);

        let mut decoded = Vec::with_capacity(32);
        for j in 0..16 {
            let byte = q4[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            decoded.push(lo as f32 * scale);
            decoded.push(hi as f32 * scale);
        }

        // Check approximate reconstruction (Q4 is lossy, but should be close)
        let max_err: f32 = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 2.0,
            "Q4 round-trip max error {max_err} exceeds 2.0"
        );
    }

    /// `q4k_matvec_into` must produce numerically identical output to
    /// the reference `dequantize_q4_k(...) → matmul_vec(...)` path.  Same
    /// f32 weights, same arithmetic — just decoded streaming.  We use a
    /// designed Q4_K-quantised input where the round-trip error is
    /// already inside the quantizer, so the matvec output should match
    /// within float-rounding noise (1e-3 on small magnitudes).
    #[test]
    fn q4k_matvec_matches_dequant_then_matmul() {
        // 4 rows × 256 cols (one super-block per row).
        let rows = 4;
        let cols = 256;
        let n_elem = rows * cols;

        // Designed weights: gradient ramp so the per-sub-block scale/min
        // varies, exercises every code path in q4k_matvec_into.
        let weights: Vec<f32> = (0..n_elem)
            .map(|i| ((i as f32 / n_elem as f32) - 0.5) * 1.0)
            .collect();
        let q4k = quantize_q4_k(&weights);
        assert_eq!(q4k.len(), rows * 144);

        // Reference: dequantize → row-major sgemv (manual, so this test
        // doesn't reach into the moe::math BLAS path).
        let dequant = dequantize_q4_k(&q4k, n_elem);
        assert_eq!(dequant.len(), n_elem);

        let x: Vec<f32> = (0..cols).map(|j| (j as f32 * 0.01).sin()).collect();
        let mut reference = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += dequant[r * cols + c] * x[c];
            }
            reference[r] = acc;
        }

        let mut got = vec![0.0f32; rows];
        q4k_matvec_into(&mut got, &x, &q4k, rows, cols);

        let max_diff: f32 = reference
            .iter()
            .zip(got.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        // Both paths use the same nibble + scale arithmetic — differ only
        // in summation order.  f32 fp accumulation reorders are bounded
        // by ~ulp(max_intermediate); for 256-element sums of ~1.0 magnitudes
        // that's well under 1e-3.
        assert!(
            max_diff < 1e-3,
            "q4k_matvec_into diverges from dequant→matmul reference: \
             max_diff={max_diff}, reference={reference:?}, got={got:?}"
        );
    }

    /// Multi-block path: cols = 2 × 256 forces the per-row inner loop to
    /// iterate `n_blocks > 1`.  Catches off-by-one in row-stride arithmetic
    /// (`row_bytes = n_blocks * 144`) that the single-block test wouldn't
    /// notice.
    #[test]
    fn q4k_matvec_multi_block_matches_dequant() {
        let rows = 3;
        let cols = 512; // 2 super-blocks per row
        let n_elem = rows * cols;
        let weights: Vec<f32> = (0..n_elem).map(|i| (i as f32 * 0.003).cos()).collect();
        let q4k = quantize_q4_k(&weights);
        assert_eq!(q4k.len(), rows * 2 * 144);

        let dequant = dequantize_q4_k(&q4k, n_elem);
        let x: Vec<f32> = (0..cols)
            .map(|j| ((j as f32) * 0.013).sin() * 0.7)
            .collect();
        let mut reference = vec![0.0f32; rows];
        for r in 0..rows {
            for c in 0..cols {
                reference[r] += dequant[r * cols + c] * x[c];
            }
        }
        let mut got = vec![0.0f32; rows];
        q4k_matvec_into(&mut got, &x, &q4k, rows, cols);
        let max_diff: f32 = reference
            .iter()
            .zip(got.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 5e-3, "multi-block diverged: max_diff={max_diff}");
    }

    /// Defensive: caller passes a malformed `cols` (not multiple of 256).
    /// We zero the output rather than reading past the buffer, mirroring
    /// `dequantize_q4_k`'s `Vec::new()` shape-error contract.
    #[test]
    fn q4k_matvec_rejects_non_multiple_of_256() {
        let mut out = vec![1.0f32; 4]; // pre-fill to detect zeroing
        let x = vec![0.5f32; 100];
        let w = vec![0u8; 4 * 144];
        q4k_matvec_into(&mut out, &x, &w, 4, 100);
        assert_eq!(out, vec![0.0f32; 4]);
    }

    #[test]
    fn q4k_matvec_zero_dims_and_short_weights_zero_output() {
        let mut out = vec![1.0f32; 3];
        q4k_matvec_into(&mut out, &[], &[], 3, 0);
        assert_eq!(out, vec![0.0f32; 3]);

        let mut out = vec![1.0f32; 2];
        let x = vec![0.5f32; 256];
        let short_w = vec![0u8; 144];
        q4k_matvec_into(&mut out, &x, &short_w, 2, 256);
        assert_eq!(out, vec![0.0f32; 2]);
    }

    #[test]
    fn dequantize_q4k_rejects_misaligned_or_truncated_input() {
        assert!(dequantize_q4_k(&[0u8; 144], 255).is_empty());
        assert!(dequantize_q4_k(&[0u8; 143], 256).is_empty());
    }

    #[test]
    #[should_panic(expected = "multiple of 32")]
    fn q4_rejects_non_aligned() {
        let data = vec![1.0f32; 33];
        let _ = quantize_q4_0(&data);
    }

    #[test]
    fn q4_matvec_uses_quantized_data() {
        // End-to-end: quantize a matrix, run matvec, verify nonzero output
        let hidden = 256;
        let rows = 64;
        let matrix: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let q4 = quantize_q4_0(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let (q8_x, q8_scales) = quantize_to_q8(&x);

        let mut scores = vec![0.0f32; rows];
        unsafe {
            q4_0_matvec_c(
                q4.as_ptr(),
                q8_x.as_ptr(),
                q8_scales.as_ptr(),
                scores.as_mut_ptr(),
                rows,
                hidden,
            );
        }
        assert!(
            scores.iter().any(|&v| v.abs() > 0.01),
            "Q4 matvec should produce nonzero"
        );
    }

    /// Decode f16 bits to f32 (for test verification).
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let mant = (bits & 0x3FF) as u32;
        if exp == 0 {
            if mant == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            // Subnormal
            let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
            return if sign == 1 { -val } else { val };
        }
        if exp == 31 {
            return if mant == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }
        let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
        if sign == 1 {
            -val
        } else {
            val
        }
    }

    /// Test alias — dispatches to the canonical module-scope implementation.
    fn dequantize_q4_k_llama(data: &[u8], n_elements: usize) -> Vec<f32> {
        super::dequantize_q4_k(data, n_elements)
    }

    #[test]
    fn q4_k_round_trip_is_gguf_format() {
        // One super-block of a smooth [-1, 1] ramp — the worst case for
        // block-level scales. Verifies (a) the output is the 144-byte
        // llama.cpp layout and (b) quantise+dequantise agree to within Q4
        // quantisation noise.
        let data: Vec<f32> = (0..256).map(|i| (i as f32 / 255.0) * 2.0 - 1.0).collect();
        let bytes = quantize_q4_k(&data);
        assert_eq!(
            bytes.len(),
            144,
            "Q4_K super-block must be 144 bytes (GGUF), got {}",
            bytes.len()
        );
        let decoded = dequantize_q4_k_llama(&bytes, 256);
        let max_err = data
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Q4 over a 2.0 range → nibble step ≈ 0.13; allow 2× for the
        // per-sub-block scale/min quantisation bias.
        assert!(
            max_err < 0.12,
            "Q4_K GGUF round-trip max error {max_err} > 0.12 — \
             packing likely drifted from llama.cpp's get_scale_min_k4"
        );
    }

    // ── quantize_q6_k tests ──

    #[test]
    fn q6_k_output_size() {
        let data = vec![0.5f32; 256];
        let q6k = quantize_q6_k(&data);
        assert_eq!(q6k.len(), 210, "Q6_K super-block must be 210 bytes");

        let data2 = vec![0.5f32; 512];
        let q6k2 = quantize_q6_k(&data2);
        assert_eq!(q6k2.len(), 420, "two Q6_K super-blocks must be 420 bytes");
    }

    #[test]
    fn q6_k_round_trip_via_matvec() {
        let hidden = 256usize;
        let rows = 4usize;
        let weights: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.001).cos())
            .collect();
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let q6k = quantize_q6_k(&weights);
        assert_eq!(q6k.len(), rows * 210);
        let result = super::super::q6k_matvec::dispatch(&q6k, &x, rows, hidden);
        assert_eq!(result.len(), rows);
        assert!(
            result.iter().any(|v| v.abs() > 1e-4),
            "Q6_K matvec should produce nonzero output"
        );
    }

    // ── q4k_to_q4kf / quantize_q4_kf tests ──

    #[test]
    fn q4kf_output_size() {
        let data = vec![0.5f32; 256];
        let q4kf = quantize_q4_kf(&data);
        assert_eq!(q4kf.len(), 160, "Q4_KF super-block must be 160 bytes");
    }

    #[test]
    fn q4k_to_q4kf_converts_format() {
        let hidden = 256usize;
        let rows = 2usize;
        let weights: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
        let q4k = quantize_q4_k(&weights);
        let q4kf = q4k_to_q4kf(&q4k, rows, hidden);
        // Q4_KF is 160 bytes per 256-element super-block vs Q4_K's 144 bytes
        assert_eq!(q4kf.len(), rows * 160);
        assert_eq!(q4k.len(), rows * 144);
    }

    #[test]
    fn q4k_to_q4kf_multi_superblock_rows() {
        let hidden = 512usize;
        let rows = 3usize;
        let weights: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.004).cos() * 0.25)
            .collect();
        let q4k = quantize_q4_k(&weights);
        let q4kf = q4k_to_q4kf(&q4k, rows, hidden);

        assert_eq!(q4k.len(), rows * 2 * 144);
        assert_eq!(q4kf.len(), rows * 2 * 160);
        assert!(
            q4kf.iter().any(|v| *v != 0),
            "converted Q4_KF should retain nonzero scales or nibbles"
        );
    }

    // ── f32_to_f16 edge cases ──

    #[test]
    fn f32_to_f16_normal_round_trip() {
        // 1.0, -1.0, 0.5: all representable exactly in f16
        for &val in &[1.0f32, -1.0, 0.5, -0.5, 2.0] {
            let bits = super::f32_to_f16(val);
            let back = f16_to_f32(bits);
            assert!(
                (back - val).abs() < 1e-3,
                "round-trip failed for {val}: got {back}"
            );
        }
    }

    #[test]
    fn f32_to_f16_infinity() {
        let inf_bits = super::f32_to_f16(f32::INFINITY);
        let back = f16_to_f32(inf_bits);
        assert!(
            back.is_infinite() && back > 0.0,
            "expected +inf, got {back}"
        );

        let neg_inf_bits = super::f32_to_f16(f32::NEG_INFINITY);
        let neg_back = f16_to_f32(neg_inf_bits);
        assert!(
            neg_back.is_infinite() && neg_back < 0.0,
            "expected -inf, got {neg_back}"
        );
    }

    #[test]
    fn f32_to_f16_large_value_clamps_to_infinity() {
        // 1e30 is beyond f16 max (~65504) → should return f16 infinity
        let bits = super::f32_to_f16(1e30f32);
        let back = f16_to_f32(bits);
        assert!(
            back.is_infinite(),
            "1e30 → f16 should be infinity, got {back}"
        );
    }

    #[test]
    fn f32_to_f16_subnormal_range() {
        // 1e-10 is below f16 normal range (min normal ≈ 6.1e-5) → subnormal or zero f16
        let bits = super::f32_to_f16(1e-10f32);
        let back = f16_to_f32(bits);
        // Should be small (subnormal or zero), not a normal f16 value
        assert!(
            back.abs() < 1e-4,
            "1e-10 → f16 back-conversion {back} should be very small"
        );
    }

    #[test]
    fn f32_to_f16_denormal_f32_input() {
        // f32 denormal (exp == 0) → f32_to_f16 should return signed zero
        let denormal = f32::from_bits(1u32); // smallest positive f32 denormal
        let bits = super::f32_to_f16(denormal);
        // exp == 0 path returns sign as u16, which for positive is 0
        assert_eq!(bits, 0, "f32 denormal should encode as f16 zero");
    }

    #[test]
    fn q4_k_round_trip_matches_larql_models_decoder() {
        // Cross-check against the authoritative decoder in larql-models.
        // Guards against silent drift between the quantizer here and the
        // dequantizer every caller actually uses (kquant_forward.rs, vindex
        // weight load, etc.). 3 super-blocks, a mix of positive/negative.
        let data: Vec<f32> = (0..256 * 3)
            .map(|i| ((i as f32 - 383.0) / 127.0).sin())
            .collect();
        let bytes = quantize_q4_k(&data);
        assert_eq!(bytes.len(), 144 * 3);

        let decoded =
            larql_models::quant::ggml::dequantize_q4_k(&bytes, 256 * 3).expect("dequantize_q4_k");
        assert_eq!(decoded.len(), 256 * 3);

        let max_err = data
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.15,
            "cross-crate Q4_K round-trip max error {max_err} > 0.15 — \
             quantize_q4_k in larql-compute disagrees with \
             larql_models::quant::ggml::dequantize_q4_k (PR #24 llama.cpp format)"
        );
    }

    #[test]
    fn f32_to_f16_valid_f16_subnormal() {
        // 1e-7 maps to new_exp ≈ -9 → shift = 10 → total_shift = 23 < 24
        // so it encodes as a nonzero f16 subnormal rather than clamping to zero.
        let bits = super::f32_to_f16(1e-7f32);
        let back = f16_to_f32(bits);
        // Must be a small positive subnormal, not zero.
        assert!(
            back > 0.0,
            "1e-7 should encode as nonzero f16 subnormal, got {back}"
        );
        assert!(
            back < 1e-4,
            "1e-7 encoded as f16 subnormal should still be small, got {back}"
        );
    }

    #[test]
    fn quantize_q4k_all_zero_covers_d_zero_branch() {
        // All-zero data → global_max_range = 0 → d = 0 branch; global_min = 0 → dmin = 0 branch.
        // Also exercises f16_to_f32(0) in the decoder (mant==0, sign==0 path).
        let data = vec![0.0f32; 256];
        let q4k = quantize_q4_k(&data);
        assert_eq!(q4k.len(), 144);
        // Decoding should also produce all zeros.
        let decoded = dequantize_q4_k_llama(&q4k, 256);
        assert!(
            decoded.iter().all(|&v| v == 0.0),
            "all-zero encode/decode should stay zero"
        );
    }

    #[test]
    fn quantize_q4k_all_positive_covers_dmin_zero() {
        // All-positive data → global_min = 0 → dmin = 0 branch (no negative offset needed).
        let data = vec![1.0f32; 256];
        let q4k = quantize_q4_k(&data);
        assert_eq!(q4k.len(), 144);
        // dmin bytes should encode f16 zero.
        let dmin_bits = u16::from_le_bytes([q4k[2], q4k[3]]);
        assert_eq!(
            dmin_bits, 0,
            "all-positive data should produce dmin=0 (f16 zero)"
        );
    }

    #[test]
    fn quantize_q6k_all_zero_covers_d_zero_branch() {
        // All-zero data → d = 0 branch; all sub-block scales = 0.
        let data = vec![0.0f32; 256];
        let q6k = quantize_q6_k(&data);
        assert_eq!(q6k.len(), 210);
        // f16 super-block scale at bytes [208..210] should be zero.
        let d_bits = u16::from_le_bytes([q6k[208], q6k[209]]);
        assert_eq!(d_bits, 0, "all-zero data should produce d=0 (f16 zero)");
    }

    #[test]
    #[should_panic(expected = "multiple of 256")]
    fn quantize_q6k_rejects_non_aligned() {
        let _ = quantize_q6_k(&vec![1.0f32; 255]);
    }
}
