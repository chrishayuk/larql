//! Fixture and CPU-reference helpers for
//! `test_lowering_representations_routed.rs`.
//!
//! Everything here is independent of the Metal code under test: the
//! MXFP4 bytes are produced by a small OCP-rule quantiser and *decoded*
//! by `larql-models`' own `dequantize_expert`, the f16 arm round-trips
//! through `larql-models`' half codec, and the MoE reference is written
//! straight from the served semantics (`moe_route_from_router_input`
//! for the route, `MoeGateRule::combine` for the gate, biases inside
//! the expert, weights outside).

#![allow(dead_code)]

use larql_compute::MoeLayerWeights;
use larql_models::quant::fp4::{f32_to_e2m1, pack_nibbles};
use larql_models::quant::half::{f16_to_f32, f32_to_f16};
use larql_models::quant::mxfp4::{
    dequantize_expert, e8m0_to_f32, FusedHalf, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS,
};

/// The Metal buffer cache's page size — `register_region` refuses any
/// start address that is not a multiple of it. Mirrors the crate's
/// private `buffers::PAGE_SIZE`.
pub const REGION_PAGE_BYTES: usize = 16384;

/// e8m0 exponent bias: byte `b` decodes to `2^(b - 127)`.
const E8M0_BIAS: i32 = 127;
/// Exponent of e2m1's largest magnitude (6.0 → `floor(log2 6) = 2`).
const E2M1_EMAX: i32 = 2;
/// e8m0 byte range that decodes to a finite non-zero scale (0 and 255
/// are the zero / NaN sentinels).
const E8M0_MIN_FINITE: i32 = 1;
const E8M0_MAX_FINITE: i32 = 254;

/// Deterministic pseudo-random floats in roughly `±0.5 * amplitude`.
pub fn det(n: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(17);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s as f32 / u32::MAX as f32) - 0.5) * amplitude
        })
        .collect()
}

pub fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        n += (*x as f64 - *y as f64).powi(2);
        d += (*x as f64).powi(2);
    }
    (n / d).sqrt()
}

/// `y = x * rsqrt(mean(x²) + eps) * (offset + w)`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, wv)| v * inv * (offset + wv))
        .collect()
}

/// `out[r] = Σ_c m[r, c] · x[c]`, `m` row-major `[n, k]`.
pub fn matvec(m: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|r| (0..k).map(|c| m[r * k + c] * x[c]).sum())
        .collect()
}

// ── MXFP4 ─────────────────────────────────────────────────────────────

/// One `[rows, k]` matrix in the kernel's split-scale MXFP4 layout:
/// `packed[(row * groups + g) * 16 ..][..16]` (lo nibble first) and
/// `scales[row * groups + g]` (e8m0).
pub struct Mxfp4Matrix {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub rows: usize,
    pub k: usize,
}

impl Mxfp4Matrix {
    /// OCP microscaling rule: per 32-group shared scale
    /// `2^(floor(log2 max|x|) - 2)`, elements to nearest e2m1.
    pub fn quantize(values: &[f32], rows: usize, k: usize) -> Self {
        assert_eq!(k % MXFP4_GROUP_ELEMS, 0, "MXFP4 needs k ≡ 0 mod 32");
        assert_eq!(values.len(), rows * k);
        let groups = k / MXFP4_GROUP_ELEMS;
        let mut packed = Vec::with_capacity(rows * groups * MXFP4_GROUP_BYTES);
        let mut scales = Vec::with_capacity(rows * groups);
        for row in values.chunks_exact(k) {
            for group in row.chunks_exact(MXFP4_GROUP_ELEMS) {
                let max_abs = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let byte = if max_abs == 0.0 {
                    0u8
                } else {
                    let exponent = max_abs.log2().floor() as i32 - E2M1_EMAX;
                    (exponent + E8M0_BIAS).clamp(E8M0_MIN_FINITE, E8M0_MAX_FINITE) as u8
                };
                let scale = e8m0_to_f32(byte);
                let inv = if scale == 0.0 { 0.0 } else { scale.recip() };
                let codes: Vec<u8> = group.iter().map(|v| f32_to_e2m1(v * inv)).collect();
                packed.extend_from_slice(&pack_nibbles(&codes));
                scales.push(byte);
            }
        }
        Self {
            packed,
            scales,
            rows,
            k,
        }
    }

    /// The independent decoder's view of these bytes — the reference the
    /// GPU arm is judged against.
    pub fn dequantized(&self) -> Vec<f32> {
        dequantize_expert(
            &self.packed,
            &self.scales,
            self.rows,
            self.k / MXFP4_GROUP_ELEMS,
        )
        .expect("fixture bytes decode")
    }
}

/// f16 round-trip of a matrix, and its little-endian f16 bytes.
pub fn f16_matrix(values: &[f32]) -> (Vec<f32>, Vec<u8>) {
    let rounded: Vec<f32> = values.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
    let bytes = rounded
        .iter()
        .flat_map(|v| f32_to_f16(*v).to_le_bytes())
        .collect();
    (rounded, bytes)
}

// ── page-aligned regions ─────────────────────────────────────────────

/// A byte region whose logical start is page-aligned, so it can be
/// registered with the Metal buffer cache and bound zero-copy. Backed by
/// an over-allocated `Vec` (registration rounds the length up to whole
/// pages, and `new_buffer_with_bytes_no_copy` must own that tail).
pub struct AlignedRegion {
    mem: Vec<u8>,
    off: usize,
    len: usize,
}

impl AlignedRegion {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut mem = vec![0u8; bytes.len() + 2 * REGION_PAGE_BYTES];
        let off = mem.as_ptr().align_offset(REGION_PAGE_BYTES);
        mem[off..off + bytes.len()].copy_from_slice(bytes);
        Self {
            mem,
            off,
            len: bytes.len(),
        }
    }
    pub fn as_slice(&self) -> &[u8] {
        &self.mem[self.off..self.off + self.len]
    }
    /// A slice starting one byte *past* the aligned start — the same
    /// allocation, but not a registrable region.
    pub fn misaligned_slice(&self) -> &[u8] {
        &self.mem[self.off + 1..self.off + self.len]
    }
    /// Expert `e`'s slice of a bank holding equal `per` byte slices.
    pub fn expert(&self, e: usize, per: usize) -> &[u8] {
        &self.as_slice()[e * per..(e + 1) * per]
    }
}

// ── routed FFN CPU reference ─────────────────────────────────────────

/// One routed layer's dequantised expert bank plus the f32 operands,
/// as the CPU reference reads them.
pub struct RoutedRef {
    /// Per expert, `[2·inter, hidden]` interleaved gate/up rows.
    pub gate_up: Vec<Vec<f32>>,
    /// Per expert, `[hidden, inter]`.
    pub down: Vec<Vec<f32>>,
    pub hidden: usize,
    pub inter: usize,
}

/// `h_post_attn + Σ_k w_k · (down_k · rule(gate_k·x + b, up_k·x + b) + b_down)`
/// with `x = rms(h_post_attn, pre_experts_norm)`; the route from the
/// served CPU router. Written from the semantics, not from the encode.
pub fn routed_ffn_reference(
    h_post_attn: &[f32],
    moe: &MoeLayerWeights<'_>,
    r: &RoutedRef,
    eps: f32,
    norm_offset: f32,
) -> Vec<f32> {
    let (hidden, inter) = (r.hidden, r.inter);
    let x = rms_norm(h_post_attn, moe.pre_experts_norm, eps, norm_offset);
    let (ids, weights) = larql_compute::cpu::ops::moe::moe_route_from_router_input(&x, moe);
    let mut out = h_post_attn.to_vec();
    for (&e, &w) in ids.iter().zip(&weights) {
        let gu = &r.gate_up[e];
        let gub = &moe.experts_gate_up_bias[e * 2 * inter..(e + 1) * 2 * inter];
        let act: Vec<f32> = (0..inter)
            .map(|i| {
                let (gr, ur) = (FusedHalf::Gate.fused_row(i), FusedHalf::Up.fused_row(i));
                let g: f32 = (0..hidden).map(|c| gu[gr * hidden + c] * x[c]).sum::<f32>() + gub[gr];
                let u: f32 = (0..hidden).map(|c| gu[ur * hidden + c] * x[c]).sum::<f32>() + gub[ur];
                moe.gate_rule.combine(g, u)
            })
            .collect();
        let dn = &r.down[e];
        let dnb = &moe.experts_down_bias[e * hidden..(e + 1) * hidden];
        for j in 0..hidden {
            let y: f32 = (0..inter).map(|i| dn[j * inter + i] * act[i]).sum::<f32>() + dnb[j];
            out[j] += w * y;
        }
    }
    out
}
