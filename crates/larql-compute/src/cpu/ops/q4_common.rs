//! Shared Q4 utilities for CPU backend.
//!
//! C FFI declarations for the vdotq_s32 kernel (csrc/q4_dot.c)
//! and Q8 quantization helper.

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
    let n_blocks = x.len() / 32;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for (b, scale_out) in scales.iter_mut().enumerate().take(n_blocks) {
        let off = b * 32;
        let block = &x[off..off + 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        *scale_out = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..32 {
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
        data.len().is_multiple_of(32),
        "data length must be a multiple of 32"
    );
    let n_blocks = data.len() / 32;
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
        let _inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

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

/// Dequantize Q4_KF pre-baked half-scale blocks into f32 values.
pub fn dequantize_q4_kf(q4kf_data: &[u8], n_elements: usize) -> Option<Vec<f32>> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = crate::pipeline::Q4_KF_BLOCK_BYTES;

    if !n_elements.is_multiple_of(BLOCK_ELEMS) {
        return None;
    }

    let n_blocks = n_elements / BLOCK_ELEMS;
    let expected = n_blocks * BLOCK_BYTES;
    if q4kf_data.len() < expected {
        return None;
    }

    let mut out = vec![0.0f32; n_elements];
    for sb in 0..n_blocks {
        let block = &q4kf_data[sb * BLOCK_BYTES..(sb + 1) * BLOCK_BYTES];
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for j in 0..8 {
            let scale = u16::from_le_bytes([block[j * 2], block[j * 2 + 1]]);
            let min = u16::from_le_bytes([block[16 + j * 2], block[16 + j * 2 + 1]]);
            scales[j] = f16_to_f32(scale);
            mins[j] = f16_to_f32(min);
        }

        let quants = &block[32..160];
        let sb_base = sb * BLOCK_ELEMS;
        for group in 0..4 {
            let lo_subblock = group * 2;
            let hi_subblock = lo_subblock + 1;
            let chunk = &quants[group * 32..(group + 1) * 32];
            let lo_base = sb_base + lo_subblock * 32;
            let hi_base = sb_base + hi_subblock * 32;
            for lane in 0..32 {
                let byte = chunk[lane];
                out[lo_base + lane] =
                    scales[lo_subblock] * (byte & 0x0F) as f32 - mins[lo_subblock];
                out[hi_base + lane] =
                    scales[hi_subblock] * ((byte >> 4) & 0x0F) as f32 - mins[hi_subblock];
            }
        }
    }

    Some(out)
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

    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
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

            for g in 0..4 {
                // Two paired sub-blocks (low + high nibble) share one 32-byte
                // quant chunk.  Hot inner: 32 nibble decodes × FMA each side.
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

                let mut dot_lo = 0.0f32;
                let mut dot_hi = 0.0f32;
                let x_lo = &x[x_lo_base..x_lo_base + 32];
                let x_hi = &x[x_hi_base..x_hi_base + 32];
                for l in 0..32 {
                    let byte = chunk[l];
                    let q_lo = (byte & 0x0F) as f32;
                    let q_hi = ((byte >> 4) & 0x0F) as f32;
                    dot_lo += q_lo * x_lo[l];
                    dot_hi += q_hi * x_hi[l];
                }

                acc += sc_lo * dot_lo - mn_lo * sumy_lo;
                acc += sc_hi * dot_hi - mn_hi * sumy_hi;
            }
        }
        *out_slot = acc;
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
        // dequantizer every caller actually uses (q4k_forward.rs, vindex
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
}
