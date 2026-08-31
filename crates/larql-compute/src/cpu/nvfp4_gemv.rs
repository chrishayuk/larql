//! The CPU's NVFP4 gemv.
//!
//! Until this existed, `MatMul::nvfp4_gemv` was implemented only on
//! Metal, so the CPU inherited the trait's `None` and every backend that
//! asks for NVFP4 was a device backend. That is a real hole rather than
//! a preference: a model whose token mixer has no device kernel — Qwen3.8
//! is 48 Gated DeltaNet layers — could have an NVFP4 representation
//! *compiled and verified* and then no path anywhere able to execute it.
//! The compiler could produce a program the substrate could not consume.
//!
//! The arithmetic here is not a second opinion about the format. It is
//! the association [`larql_models::quant::nvfp4::dequantize_into`]
//! documents — `tensor_scale * e4m3(group scale) * e2m1(code)`, in that
//! order — applied one group at a time instead of into a whole decoded
//! matrix, so the result matches dequantise-then-multiply exactly rather
//! than approximately. The tests hold it to that oracle.

use larql_models::quant::fp4::e2m1_to_f32;
use larql_models::quant::fp8::e4m3_to_f32;
use larql_models::quant::nvfp4::{NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS};

/// `out[n] = W[n, k] · x[k]`, with `W` read straight from the stored
/// NVFP4 pack.
///
/// `None` — never a wrong answer — when the geometry does not describe
/// this pack: `k` off the group grid, or either stream the wrong length
/// for `[n, k]`. Silently accepting a short stream would read a
/// neighbouring row's codes as this row's and return a plausible vector.
pub fn nvfp4_gemv(
    packed: &[u8],
    scales: &[u8],
    tensor_scale: f32,
    x: &[f32],
    n: usize,
    k: usize,
) -> Option<Vec<f32>> {
    if k == 0 || !k.is_multiple_of(NVFP4_GROUP_ELEMS) || x.len() != k {
        return None;
    }
    let groups = k / NVFP4_GROUP_ELEMS;
    if packed.len() != n * groups * NVFP4_GROUP_BYTES || scales.len() != n * groups {
        return None;
    }

    let mut out = vec![0.0f32; n];
    for (row, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..groups {
            // One multiply per group, not per element: the scale is
            // constant across the sixteen, and folding it into each
            // weight is what the reference decoder does.
            let step = tensor_scale * e4m3_to_f32(scales[row * groups + g]);
            let base = (row * groups + g) * NVFP4_GROUP_BYTES;
            let xs = &x[g * NVFP4_GROUP_ELEMS..][..NVFP4_GROUP_ELEMS];
            for b in 0..NVFP4_GROUP_BYTES {
                let byte = packed[base + b];
                // Low nibble first, matching the encoder's pairing.
                acc += (step * e2m1_to_f32(byte & 0x0F)) * xs[2 * b];
                acc += (step * e2m1_to_f32((byte >> 4) & 0x0F)) * xs[2 * b + 1];
            }
        }
        *slot = acc;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use crate::backend::MatMul;
    use crate::cpu::CpuBackend;
    use larql_models::quant::nvfp4::{dequantize_into, quantize};

    /// Deterministic weights spanning several magnitudes, so groups get
    /// different scales and a single shared scale cannot pass by luck.
    fn weights(n: usize, k: usize) -> Vec<f32> {
        (0..n * k)
            .map(|i| {
                let t = (i % 37) as f32 / 37.0 - 0.5;
                t * (1.0 + (i / k) as f32 * 0.25)
            })
            .collect()
    }

    fn dequantised_gemv(w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
        (0..n)
            .map(|r| (0..k).map(|c| w[r * k + c] * x[c]).sum())
            .collect()
    }

    /// The oracle: decoding the pack and multiplying must give exactly
    /// what the kernel gives. Not "within a tolerance" — the kernel is
    /// the same arithmetic in the same association, so any difference is
    /// a different program, not rounding.
    #[test]
    fn the_kernel_equals_dequantise_then_multiply_bit_for_bit() {
        for (n, k) in [(1, 16), (3, 32), (8, 64), (5, 112)] {
            let w = weights(n, k);
            let m = quantize(&w, n, k).expect("quantise");
            let mut decoded = vec![0.0f32; n * k];
            dequantize_into(&m, n, k, &mut decoded).expect("dequantise");

            let x: Vec<f32> = (0..k).map(|i| ((i % 11) as f32 - 5.0) / 7.0).collect();
            let got = CpuBackend
                .nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, n, k)
                .expect("kernel");
            let want = dequantised_gemv(&decoded, &x, n, k);
            assert_eq!(got, want, "[{n},{k}] kernel diverged from the decoder");
        }
    }

    /// The kernel must be reading the stored codes, not reconstructing
    /// the original floats: quantisation is lossy, so an exact match
    /// against the SOURCE weights would mean the test is not testing a
    /// quantised path at all.
    #[test]
    fn the_result_is_lossy_against_the_source_weights() {
        let (n, k) = (4, 64);
        let w = weights(n, k);
        let m = quantize(&w, n, k).expect("quantise");
        let x: Vec<f32> = (0..k).map(|i| ((i % 11) as f32 - 5.0) / 7.0).collect();
        let got = CpuBackend
            .nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, n, k)
            .expect("kernel");
        let exact = dequantised_gemv(&w, &x, n, k);
        assert_ne!(got, exact, "an exact match means nothing was quantised");
        for (g, e) in got.iter().zip(&exact) {
            assert!((g - e).abs() < 0.5 * e.abs().max(1.0), "{g} vs {e}");
        }
    }

    /// Geometry that does not describe this pack answers `None` rather
    /// than reading a neighbouring row's codes as this row's.
    #[test]
    fn mismatched_geometry_refuses_instead_of_guessing() {
        let (n, k) = (4, 64);
        let m = quantize(&weights(n, k), n, k).expect("quantise");
        let x = [0.5f32; 64];
        let call = |xs: &[f32], n, k| {
            CpuBackend.nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, xs, n, k)
        };
        assert!(call(&x, n, k).is_some(), "the honest geometry works");
        assert!(call(&x, n + 1, k).is_none(), "too many rows for the stream");
        assert!(call(&x, n, k * 2).is_none(), "k beyond the stream");
        assert!(call(&x[..k - 1], n, k).is_none(), "x shorter than k");
        assert!(call(&[0.5; 24], n, 24).is_none(), "k off the group grid");
    }

    /// `nvfp4_gemv_multi`'s default fans out to the single kernel; with
    /// the CPU arm implemented it must now answer rather than collapsing
    /// to `None` on the first matrix.
    #[test]
    fn the_multi_arm_answers_now_that_the_single_one_does() {
        let (n, k) = (2, 32);
        let m = quantize(&weights(n, k), n, k).expect("quantise");
        let x: Vec<f32> = vec![0.25; k];
        let multi = CpuBackend
            .nvfp4_gemv_multi(
                &[
                    (&m.packed, &m.scales, m.tensor_scale, n, k),
                    (&m.packed, &m.scales, m.tensor_scale, n, k),
                ],
                &x,
            )
            .expect("multi");
        let single = CpuBackend
            .nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, n, k)
            .expect("single");
        assert_eq!(multi, vec![single.clone(), single]);
    }
}
