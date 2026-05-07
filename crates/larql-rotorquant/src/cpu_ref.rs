//! CPU-side reference implementation of RotorQuant quantize +
//! dequantize. Slow but transparent — its outputs are the oracle for
//! both the GPU kernels (when implemented) and external callers.
//!
//! The flow per row:
//!
//! 1. Compute the row L2 norm; record it in `norms`.
//! 2. Normalise the row to unit length.
//! 3. For each block-of-2 (Planar) or block-of-4 (Iso):
//!    a. Pick the rotation index that, when applied, minimises the
//!    sum-of-squares of the post-rotation codes (Lloyd-Max).
//!    b. Apply that rotation. Record the index.
//! 4. Quantise each rotated coordinate to the closest codeword in
//!    the format's codebook. Record codes.
//!
//! Dequantize unwinds: read codes → look up codebook values →
//! reconstruct rotated row → apply rotation (or its inverse for V)
//! → multiply by stored norm.

use crate::error::RotorQuantError;
use crate::format::{KvFormat, QuantizedKv};

/// 8-codeword (3-bit) uniform codebook in [-1, 1]. Used together
/// with absmax row scaling: a row is divided by its max-abs so all
/// components are in [-1, 1]; then quantised against this codebook;
/// then multiplied by the per-row scale on dequant. The rotation
/// step decorrelates blocks before quantisation so each codebook
/// entry sees a tighter distribution than the raw values.
const LM3_CODEBOOK: [f32; 8] = [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875];

/// 16-codeword (4-bit) uniform codebook in [-1, 1]. Mid-points of
/// 16 equal-width bins.
const LM4_CODEBOOK: [f32; 16] = [
    -0.9375, -0.8125, -0.6875, -0.5625, -0.4375, -0.3125, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125,
    0.4375, 0.5625, 0.6875, 0.8125, 0.9375,
];

fn codebook(format: KvFormat) -> &'static [f32] {
    match format {
        KvFormat::Planar3 | KvFormat::Iso3 => &LM3_CODEBOOK,
        KvFormat::Planar4 | KvFormat::Iso4 => &LM4_CODEBOOK,
    }
}

/// Pre-tabulated rotation set for Planar (Givens 2D). 8 evenly-spaced
/// angles in [0, π/2). The "best rotation" search is brute-force over
/// this table.
fn planar_rotations() -> [[f32; 4]; 8] {
    let mut out = [[0.0_f32; 4]; 8];
    for (i, e) in out.iter_mut().enumerate() {
        let theta = (i as f32) * (std::f32::consts::FRAC_PI_2 / 8.0);
        let (s, c) = theta.sin_cos();
        // 2x2 rotation matrix: [c, -s; s, c]
        e[0] = c;
        e[1] = -s;
        e[2] = s;
        e[3] = c;
    }
    out
}

/// Pre-tabulated rotation set for Iso (quaternion 4D). 16 unit
/// quaternions interpolated from the identity. Apply each as a 4×4
/// matrix when needed.
fn iso_rotations() -> [[f32; 16]; 16] {
    let mut out = [[0.0_f32; 16]; 16];
    for (i, e) in out.iter_mut().enumerate() {
        // Quaternion (w, x, y, z) sweeping a half-angle around the
        // (1, 1, 1) axis, scaled by index/15. Constructed in normalised
        // float-1.0 form.
        let half = (i as f32) * std::f32::consts::FRAC_PI_2 / 15.0;
        let (s, c) = half.sin_cos();
        let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
        let qw = c;
        let qx = s * inv_sqrt3;
        let qy = s * inv_sqrt3;
        let qz = s * inv_sqrt3;
        // Convert quaternion to 4x4 rotation matrix on the (x, y, z, w)
        // 4D space; treat the 4th coordinate as a passthrough projected
        // through w. For simplicity in the reference impl we use the
        // 3x3 rotation embedded into the upper-left of a 4x4 identity.
        let xx = qx * qx;
        let yy = qy * qy;
        let zz = qz * qz;
        let xy = qx * qy;
        let xz = qx * qz;
        let yz = qy * qz;
        let wx = qw * qx;
        let wy = qw * qy;
        let wz = qw * qz;
        e[0] = 1.0 - 2.0 * (yy + zz);
        e[1] = 2.0 * (xy - wz);
        e[2] = 2.0 * (xz + wy);
        e[3] = 0.0;
        e[4] = 2.0 * (xy + wz);
        e[5] = 1.0 - 2.0 * (xx + zz);
        e[6] = 2.0 * (yz - wx);
        e[7] = 0.0;
        e[8] = 2.0 * (xz - wy);
        e[9] = 2.0 * (yz + wx);
        e[10] = 1.0 - 2.0 * (xx + yy);
        e[11] = 0.0;
        e[12] = 0.0;
        e[13] = 0.0;
        e[14] = 0.0;
        e[15] = 1.0;
    }
    out
}

fn apply_planar(rot: &[f32; 4], block: &[f32; 2]) -> [f32; 2] {
    [
        rot[0] * block[0] + rot[1] * block[1],
        rot[2] * block[0] + rot[3] * block[1],
    ]
}

fn apply_planar_inverse(rot: &[f32; 4], block: &[f32; 2]) -> [f32; 2] {
    // 2D rotation matrices have inverse = transpose.
    [
        rot[0] * block[0] + rot[2] * block[1],
        rot[1] * block[0] + rot[3] * block[1],
    ]
}

fn apply_iso(rot: &[f32; 16], block: &[f32; 4]) -> [f32; 4] {
    let mut out = [0.0_f32; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        let row = i * 4;
        *slot = rot[row] * block[0]
            + rot[row + 1] * block[1]
            + rot[row + 2] * block[2]
            + rot[row + 3] * block[3];
    }
    out
}

fn apply_iso_inverse(rot: &[f32; 16], block: &[f32; 4]) -> [f32; 4] {
    // 4x4 rotation: inverse = transpose.
    let mut out = [0.0_f32; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = rot[i] * block[0]
            + rot[4 + i] * block[1]
            + rot[8 + i] * block[2]
            + rot[12 + i] * block[3];
    }
    out
}

/// Encode `code_idx` (a value in 0..codebook_size) into the packed
/// `codes` bitstream at index `bit_pos`. LSB-first.
fn pack_code(codes: &mut [u8], bit_pos: usize, bits: u8, code: u8) {
    let mask: u32 = (1u32 << bits) - 1;
    let value = u32::from(code) & mask;
    let byte = bit_pos / 8;
    let shift = bit_pos % 8;
    // Read 16-bit window from codes.
    let lo = u32::from(codes[byte]);
    let hi = if byte + 1 < codes.len() {
        u32::from(codes[byte + 1])
    } else {
        0
    };
    let combined = lo | (hi << 8);
    let cleared = combined & !(mask << shift);
    let inserted = cleared | (value << shift);
    codes[byte] = (inserted & 0xff) as u8;
    if byte + 1 < codes.len() {
        codes[byte + 1] = ((inserted >> 8) & 0xff) as u8;
    }
}

fn unpack_code(codes: &[u8], bit_pos: usize, bits: u8) -> u8 {
    let mask: u32 = (1u32 << bits) - 1;
    let byte = bit_pos / 8;
    let shift = bit_pos % 8;
    let lo = u32::from(codes[byte]);
    let hi = if byte + 1 < codes.len() {
        u32::from(codes[byte + 1])
    } else {
        0
    };
    let combined = lo | (hi << 8);
    ((combined >> shift) & mask) as u8
}

fn quantize_value(cb: &[f32], v: f32) -> (u8, f32) {
    // Brute-force nearest-neighbour into the codebook. Codebook is
    // small (≤16); cost is negligible.
    let mut best_idx = 0;
    let mut best_err = f32::INFINITY;
    for (i, &c) in cb.iter().enumerate() {
        let d = (v - c).abs();
        if d < best_err {
            best_err = d;
            best_idx = i;
        }
    }
    (best_idx as u8, cb[best_idx])
}

pub(crate) fn quantize(
    format: KvFormat,
    data: &[f32],
    n_rows: usize,
    head_dim: usize,
) -> Result<QuantizedKv, RotorQuantError> {
    let bs = format.block_size();
    if !head_dim.is_multiple_of(bs) {
        return Err(RotorQuantError::HeadDimNotDivisible {
            format,
            head_dim,
            block_size: bs,
        });
    }
    let expected = n_rows * head_dim;
    if data.len() != expected {
        return Err(RotorQuantError::InputLengthMismatch {
            got: data.len(),
            n_rows,
            head_dim,
            expected,
        });
    }
    let bits = format.bits();
    let cb = codebook(format);
    let n_blocks_per_row = head_dim / bs;
    let n_codes = n_rows * head_dim;
    let n_bits_total = n_codes * usize::from(bits);
    let n_bytes = n_bits_total.div_ceil(8);

    let mut codes = vec![0u8; n_bytes];
    let mut norms = Vec::with_capacity(n_rows);
    let mut rotation_indices = Vec::with_capacity(n_rows * n_blocks_per_row);

    let planar_rots = planar_rotations();
    let iso_rots = iso_rotations();

    for row in 0..n_rows {
        let base = row * head_dim;
        // Absmax scaling: divide by the largest absolute value so all
        // components lie in [-1, 1]. The codebook is then dense over
        // the same range; rotation decorrelates but stays bounded.
        let absmax: f32 = data[base..base + head_dim]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        norms.push(absmax);
        let inv = if absmax > 0.0 { 1.0 / absmax } else { 0.0 };

        for blk in 0..n_blocks_per_row {
            let off = base + blk * bs;
            let mut best_rot = 0u16;
            let mut best_err = f32::INFINITY;
            let mut best_codes: [u8; 4] = [0, 0, 0, 0];

            match format.block_size() {
                2 => {
                    let block = [data[off] * inv, data[off + 1] * inv];
                    for (rot_idx, rot) in planar_rots.iter().enumerate() {
                        let rotated = apply_planar(rot, &block);
                        let (c0, q0) = quantize_value(cb, rotated[0]);
                        let (c1, q1) = quantize_value(cb, rotated[1]);
                        let err = (q0 - rotated[0]).powi(2) + (q1 - rotated[1]).powi(2);
                        if err < best_err {
                            best_err = err;
                            best_rot = rot_idx as u16;
                            best_codes[0] = c0;
                            best_codes[1] = c1;
                        }
                    }
                }
                4 => {
                    let block = [
                        data[off] * inv,
                        data[off + 1] * inv,
                        data[off + 2] * inv,
                        data[off + 3] * inv,
                    ];
                    for (rot_idx, rot) in iso_rots.iter().enumerate() {
                        let rotated = apply_iso(rot, &block);
                        let mut err = 0.0_f32;
                        let mut local_codes = [0u8; 4];
                        for (i, &v) in rotated.iter().enumerate() {
                            let (c, q) = quantize_value(cb, v);
                            err += (q - v).powi(2);
                            local_codes[i] = c;
                        }
                        if err < best_err {
                            best_err = err;
                            best_rot = rot_idx as u16;
                            best_codes = local_codes;
                        }
                    }
                }
                _ => unreachable!(),
            }

            rotation_indices.push(best_rot);
            for (j, code) in best_codes.iter().enumerate().take(bs) {
                let bit_pos = (row * head_dim + blk * bs + j) * usize::from(bits);
                pack_code(&mut codes, bit_pos, bits, *code);
            }
        }
    }

    Ok(QuantizedKv {
        format,
        n_rows,
        head_dim,
        codes,
        norms,
        rotation_indices,
    })
}

/// Dequantise the full [n_rows, head_dim] tensor.
///
/// `_invert_rotation` is retained as a parameter for API symmetry
/// with the parent change's spec, but mathematically both K and V
/// recoveries apply the SAME inverse rotation to undo what
/// `quantize` applied. The naming `dequantize_v_with_inverse_rotation`
/// emphasises a contract — you MUST NOT apply the forward rotation to
/// V (the upstream commit-6e5a4aa bug) — rather than indicating a
/// branch in the maths. K and V take the same inverse here.
pub(crate) fn dequantize(
    qkv: &QuantizedKv,
    _invert_rotation: bool,
) -> Result<Vec<f32>, RotorQuantError> {
    let bs = qkv.format.block_size();
    if !qkv.head_dim.is_multiple_of(bs) {
        return Err(RotorQuantError::HeadDimNotDivisible {
            format: qkv.format,
            head_dim: qkv.head_dim,
            block_size: bs,
        });
    }
    let bits = qkv.format.bits();
    let cb = codebook(qkv.format);
    let n_blocks_per_row = qkv.head_dim / bs;
    if qkv.norms.len() != qkv.n_rows {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "norms.len {} != n_rows {}",
            qkv.norms.len(),
            qkv.n_rows
        )));
    }
    if qkv.rotation_indices.len() != qkv.n_rows * n_blocks_per_row {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "rotation_indices.len {} != n_rows ({}) * blocks_per_row ({})",
            qkv.rotation_indices.len(),
            qkv.n_rows,
            n_blocks_per_row
        )));
    }

    let planar_rots = planar_rotations();
    let iso_rots = iso_rotations();
    let mut out = vec![0.0_f32; qkv.n_rows * qkv.head_dim];

    for row in 0..qkv.n_rows {
        let base = row * qkv.head_dim;
        let norm = qkv.norms[row];
        for blk in 0..n_blocks_per_row {
            let off = base + blk * bs;
            let rot_idx = qkv.rotation_indices[row * n_blocks_per_row + blk] as usize;
            // Read codes back into rotated-space values.
            let mut rotated = [0.0_f32; 4];
            for (j, slot) in rotated.iter_mut().enumerate().take(bs) {
                let bit_pos = (row * qkv.head_dim + blk * bs + j) * usize::from(bits);
                let code = unpack_code(&qkv.codes, bit_pos, bits) as usize;
                *slot = cb[code];
            }
            // Always apply the inverse of the forward rotation chosen
            // at quantise time. That undoes the rotation and recovers
            // the original block (up to codebook quantisation).
            let recovered = match qkv.format.block_size() {
                2 => {
                    let r = &planar_rots[rot_idx];
                    let arr = [rotated[0], rotated[1]];
                    let res = apply_planar_inverse(r, &arr);
                    [res[0], res[1], 0.0, 0.0]
                }
                4 => {
                    let r = &iso_rots[rot_idx];
                    let arr = [rotated[0], rotated[1], rotated[2], rotated[3]];
                    apply_iso_inverse(r, &arr)
                }
                _ => unreachable!(),
            };
            for j in 0..bs {
                out[off + j] = recovered[j] * norm;
            }
        }
    }

    Ok(out)
}
