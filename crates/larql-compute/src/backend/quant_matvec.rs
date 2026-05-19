//! `QuantMatVec` — quantised matrix × vector operations.
//!
//! Two entry points by intent:
//!
//! - [`Self::quant_matvec`] — **the convenience API.** Takes f32
//!   input, dispatches on [`crate::QuantFormat`], internally
//!   quantises to Q8 for Q4_0 / Q8_0. New callers should reach for
//!   this.
//! - [`Self::q4_matvec`] / [`Self::q4k_matvec`] / [`Self::q6k_matvec`]
//!   — **the pre-quantised-input fast path.** Hot decode paths
//!   pre-quantise the layer's input once and reuse it across many
//!   matvecs in that layer (gate, up, LM head, …). They take
//!   already-Q8 inputs and skip the per-call quantisation.
//!
//! Adding a new quant format = `QuantFormat` variant + match arm in
//! `quant_matvec` + per-format helper for the fast path.

use crate::QuantFormat;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

/// Reverse the `quantize_to_q8` block layout: each 32-element block
/// has one f32 scale, multiplied through to recover f32 values.
fn dequantise_q8(q8_x: &[i8], q8_scales: &[f32]) -> Vec<f32> {
    let n_blocks = q8_x.len() / LEGACY_BLOCK_ELEMS;
    debug_assert!(q8_scales.len() >= n_blocks);
    let mut out = Vec::with_capacity(q8_x.len());
    for (b, &scale) in q8_scales.iter().take(n_blocks).enumerate() {
        let off = b * LEGACY_BLOCK_ELEMS;
        for &q in &q8_x[off..off + LEGACY_BLOCK_ELEMS] {
            out.push(q as f32 * scale);
        }
    }
    // Tail (if `q8_x.len()` isn't a multiple of LEGACY_BLOCK_ELEMS — defensive).
    for &q in &q8_x[n_blocks * LEGACY_BLOCK_ELEMS..] {
        out.push(q as f32);
    }
    out
}

/// Quantised matvec primitives.
pub trait QuantMatVec {
    /// Format-dispatched matvec.
    ///
    /// `out[N] = W[N, K] · x[K]`. Q4_K / Q4_KF / Q6_K consume f32 input
    /// directly; Q4_0 / Q8_0 internally re-quantise `x` to Q8 (per-32
    /// f32-scaled int8) before dispatching the kernel.
    ///
    /// Returns `None` if the backend doesn't implement the format.
    fn quant_matvec(
        &self,
        format: QuantFormat,
        weights: &[u8],
        x: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        match format {
            QuantFormat::Q4_K | QuantFormat::Q4_KF => self.q4k_matvec(weights, x, num_rows, hidden),
            QuantFormat::Q6_K => self.q6k_matvec(weights, x, num_rows, hidden),
            QuantFormat::Q4_0 => {
                let (q8_x, q8_scales) = crate::cpu::ops::q4_common::quantize_to_q8(x);
                self.q4_matvec(weights, &q8_x, &q8_scales, num_rows, hidden)
            }
            QuantFormat::Q8_0 => {
                // Q8_0 weights are NOT a Q4_0 kernel input — Q8_0 blocks are 34
                // bytes per 32 values (32 i8 quants + f16 scale) while Q4_0 is
                // 18 (16 nibble bytes + f16 scale). Pre-2026-05-09 this branch
                // routed Q8_0 weights through `q4_matvec`, which read the wrong
                // byte stride and produced garbage. Returning `None` makes the
                // missing capability loud — callers fall back to a dequant
                // path or fail explicitly. Production decode reaches Q8_0
                // weights via dedicated kernels (`q8_qkv_proj` /
                // `q8_matvec_pipeline`), not through this trait method.
                let (q8_x, q8_scales) = crate::cpu::ops::q4_common::quantize_to_q8(x);
                self.q8_matvec(weights, &q8_x, &q8_scales, num_rows, hidden)
            }
            QuantFormat::BF16 | QuantFormat::F16 | QuantFormat::F32 => None,
        }
    }

    /// Format-aware matvec on **pre-quantised** Q8 input.
    ///
    /// `out[N] = W[N, K] · q8_x[K]`. Caller has already quantised `x`
    /// to Q8 (per-32 f32-scaled int8) and passes the int8 buffer +
    /// scales directly. Hot decode loops do this once per layer and
    /// reuse the buffers across many gate/up matvecs — re-quantising
    /// per call (as `quant_matvec` does) is wasted work.
    ///
    /// - For `Q4_0` / `Q8_0` this is a direct call to `q4_matvec` /
    ///   the Q8-input kernel — zero overhead vs the per-format helper.
    /// - For `Q4_K` / `Q4_KF` / `Q6_K` the GPU shaders take f32 input,
    ///   so the default impl dequantises Q8 → f32 then dispatches the
    ///   f32 path. That's strictly slower than the f32-input
    ///   `quant_matvec`, but it's the correct fallback when the caller
    ///   has *only* the Q8 form on hand.
    ///
    /// Returns `None` if the backend doesn't implement the format.
    fn quant_matvec_q8_input(
        &self,
        format: QuantFormat,
        weights: &[u8],
        q8_x: &[i8],
        q8_scales: &[f32],
        num_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        match format {
            QuantFormat::Q4_0 => self.q4_matvec(weights, q8_x, q8_scales, num_rows, hidden),
            QuantFormat::Q8_0 => self.q8_matvec(weights, q8_x, q8_scales, num_rows, hidden),
            QuantFormat::Q4_K | QuantFormat::Q4_KF | QuantFormat::Q6_K => {
                // f32-input shaders — dequantise Q8 first.
                let x_f32 = dequantise_q8(q8_x, q8_scales);
                self.quant_matvec(format, weights, &x_f32, num_rows, hidden)
            }
            QuantFormat::BF16 | QuantFormat::F16 | QuantFormat::F32 => None,
        }
    }

    // ── Pre-quantised fast path ──
    //
    // These exist because the hot decode path pre-quantises its input
    // once and reuses it across many matvecs in a layer; the unified
    // `quant_matvec` re-quantises every call. Use these when the
    // caller already has Q8-quantised input on hand; reach for
    // `quant_matvec` otherwise.

    /// Q4_0 × Q8 matvec. `Some` if the backend supports Q4_0.
    fn q4_matvec(
        &self,
        _q4_data: &[u8],
        _q8_x: &[i8],
        _q8_scales: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Q8_0 × Q8 matvec — distinct from [`Self::q4_matvec`].
    ///
    /// `q8_data` is GGML Q8_0 wire format: 34-byte blocks (32 i8 quants
    /// plus a single f16 scale per 32-element block). `q8_x` / `q8_scales`
    /// are the standard pre-quantised input.
    ///
    /// Default returns `None`; backends with a Q8 weight kernel
    /// override. The trait used to silently route `Q8_0` weights through
    /// `q4_matvec`, which read the wrong byte stride (Q4_0 = 18 bytes /
    /// block) and produced garbage. Returning `None` from the default
    /// makes the missing capability loud.
    fn q8_matvec(
        &self,
        _q8_data: &[u8],
        _q8_x: &[i8],
        _q8_scales: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Q4 matvec + GPU argmax for greedy lm_head decode. Returns
    /// `(token_id, score)` for the top-1 element without the 1MB
    /// scores readback that `q4_matvec` requires. Returns `None` if
    /// not specialised.
    fn q4_matvec_topk1(
        &self,
        _q4_data: &[u8],
        _q8_x: &[i8],
        _q8_scales: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<(u32, f32)> {
        None
    }

    /// Q4 matvec + GPU partial top-K. Generalises
    /// [`Self::q4_matvec_topk1`] to `top_k > 1` (capped at the kernel's
    /// `K_TOPK` constant). Returns `None` when not specialised or `top_k`
    /// exceeds the per-TG capacity.
    fn q4_matvec_topk(
        &self,
        _q4_data: &[u8],
        _q8_x: &[i8],
        _q8_scales: &[f32],
        _num_rows: usize,
        _hidden: usize,
        _top_k: usize,
    ) -> Option<Vec<(u32, f32)>> {
        None
    }

    /// Q4 vector-matrix: `out[K] = activation[N] @ Q4[N, K]`.
    fn q4_vecmat(
        &self,
        _activation: &[f32],
        _q4_data: &[u8],
        _intermediate: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Batched gate+up Q4 matvec for ALL seq positions in one submission.
    #[allow(clippy::type_complexity)]
    fn q4_matvec_pair_batch(
        &self,
        _gate_q4: &[u8],
        _up_q4: &[u8],
        _x_matrix: &[f32],
        _seq_len: usize,
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        None
    }

    /// Q4_K matvec: `scores[N] = Q4_K[N, K] @ f32_x[K]`.
    fn q4k_matvec(
        &self,
        _q4k_data: &[u8],
        _x: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Fused two-weight Q4_K matvec sharing one input vector.
    ///
    /// `out_a[N] = W_a[N, K] · x[K]`, `out_b[N] = W_b[N, K] · x[K]`
    /// where W_a and W_b have identical `(num_rows, hidden)` shape and
    /// `x` is shared. The decode-step gate+up projections fit exactly.
    ///
    /// Returns `None` if the backend doesn't implement the fused path;
    /// the caller falls back to two sequential `q4k_matvec` calls.
    fn q4k_dual_matvec(
        &self,
        _q4k_a: &[u8],
        _q4k_b: &[u8],
        _x: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    /// Q4_K matvec with stride-32 lane access pattern. Same Q4_K input
    /// format as [`q4k_matvec`](Self::q4k_matvec) but the per-row
    /// reduction tree mirrors `f16_gemv` — lane `k` accumulates the
    /// dot product over elements `i % 32 == k`, then `simd_sum` across
    /// 32 lanes. Designed for the LM head when the production
    /// `q4k_matvec`'s block-aware lane split drifts enough vs CPU to
    /// flip top-1 on close-call tokens. Backends without a stable-
    /// reduction Q4_K path return `None` and the caller falls back to
    /// `f16_gemv` / `q4k_matvec` / `f32_gemv` chain.
    fn q4k_matvec_stride32(
        &self,
        _q4k_data: &[u8],
        _x: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Q4_K matmul: `C[m, n] = sum_k W[n, k] * X[m, k]`.
    ///
    /// `W` is `[num_rows, hidden]` Q4_K, `X` is `[seq_len, hidden]` f32,
    /// output is `[seq_len, num_rows]` f32 row-major. Returns `None`
    /// when the backend doesn't implement amortised matmul (callers
    /// fall back to repeated `q4k_matvec`). Used by prefill where
    /// `seq_len > 1` to amortise dequant cost across positions.
    fn q4k_matmul(
        &self,
        _q4k_data: &[u8],
        _x: &[f32],
        _num_rows: usize,
        _hidden: usize,
        _seq_len: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Q6_K matvec: `scores[N] = Q6_K[N, K] @ f32_x[K]`.
    fn q6k_matvec(
        &self,
        _q6k_data: &[u8],
        _x: &[f32],
        _num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    /// Whether this backend implements quantised matvec on `format`.
    ///
    /// "Implements" means at minimum a single-matrix matvec path for the
    /// format (e.g. `q4k_matvec`, `q6k_matvec`). Backends advertise per
    /// format so the dispatcher can route Q4_K to one backend and Q6_K to
    /// another without a separate predicate per format. Default impl:
    /// `false` for every format — backends must opt in.
    fn supports_quant(&self, format: QuantFormat) -> bool {
        let _ = format;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    /// Pin the Q4_0/Q8_0 dispatch split: Q4_0 must continue to route
    /// through `q4_matvec`, and Q8_0 must route through the new
    /// `q8_matvec` (defaulting to `None`). Pre-2026-05-09 both formats
    /// shared the Q4_0 arm, which fed Q8_0's 34-byte blocks into the
    /// Q4_0 kernel that expects 18-byte blocks — silently produced
    /// garbage instead of `None`.
    ///
    /// Buffer sizes must match the kernels' wire-format block sizes
    /// (Q4_0 = 18 B/32 vals, Q8_0 = 34 B/32 vals); otherwise the FFI
    /// debug_assert in `cpu::ops::q4_matvec::dispatch_q8` catches the
    /// mismatch before we even reach the dispatch test.
    #[test]
    fn q8_0_weights_do_not_silently_route_through_q4_kernel() {
        let backend = CpuBackend;
        let x = vec![0.1f32; 32];

        // Q4_0 dispatch — feed it a correctly-sized Q4_0 buffer so the
        // FFI assert lets us reach the kernel. Synthetic zeroed bytes
        // are fine — we're pinning routing, not output correctness.
        let q4_0_weights = vec![0u8; 18]; // one Q4_0 block
        let q4_0_result = backend.quant_matvec(QuantFormat::Q4_0, &q4_0_weights, &x, 1, 32);
        assert!(
            q4_0_result.is_some(),
            "Q4_0 must continue to route through q4_matvec"
        );

        // Q8_0 dispatch — Q8_0-shaped buffer that would have produced
        // garbage under the pre-fix shared `Q4_0 | Q8_0` arm. The fix
        // routes it through the new `q8_matvec` trait method, which
        // CpuBackend doesn't override, so the result is `None` — loud
        // missing capability rather than silent garbage.
        //
        // If a future commit adds a real Q8_0 CPU path, flip this to
        // `is_some()` and add a parity check vs a dequant+BLAS reference.
        let q8_0_weights = vec![0u8; 34]; // one Q8_0 block
        let q8_0_result = backend.quant_matvec(QuantFormat::Q8_0, &q8_0_weights, &x, 1, 32);
        assert!(
            q8_0_result.is_none(),
            "Q8_0 must NOT silently route through q4_matvec; got Some(...) — \
             the bug class this test pins is back. Add a real `q8_matvec` \
             impl on CpuBackend or restore the `None` default."
        );
    }

    /// Same shape for the pre-quantised input variant
    /// (`quant_matvec_q8_input`).
    #[test]
    fn quant_matvec_q8_input_splits_q4_and_q8() {
        let backend = CpuBackend;
        let q8_x = vec![0i8; 32];
        let q8_scales = vec![1.0f32; 1];

        // Q4_0 path needs an 18-byte Q4_0 buffer (one block).
        let q4_0_weights = vec![0u8; 18];
        let q4_0 = backend.quant_matvec_q8_input(
            QuantFormat::Q4_0,
            &q4_0_weights,
            &q8_x,
            &q8_scales,
            1,
            32,
        );
        assert!(
            q4_0.is_some(),
            "Q4_0 must continue to route through q4_matvec"
        );

        // Q8_0 returns None without ever calling the kernel; buffer
        // size is irrelevant but use 34 to keep intent clear.
        let q8_0_weights = vec![0u8; 34];
        let q8_0 = backend.quant_matvec_q8_input(
            QuantFormat::Q8_0,
            &q8_0_weights,
            &q8_x,
            &q8_scales,
            1,
            32,
        );
        assert!(
            q8_0.is_none(),
            "Q8_0 must route through q8_matvec (default None), not q4_matvec"
        );
    }

    /// Float-input formats stay `None` from the trait (the caller is
    /// expected to use a float matmul or dequantise first).
    #[test]
    fn float_input_formats_return_none() {
        let backend = CpuBackend;
        let weights = vec![0u8; 64];
        let x = vec![0.1f32; 32];

        for fmt in [QuantFormat::F16, QuantFormat::F32, QuantFormat::BF16] {
            assert!(
                backend.quant_matvec(fmt, &weights, &x, 1, 32).is_none(),
                "{fmt:?} must return None from quant_matvec"
            );
        }
    }
}
