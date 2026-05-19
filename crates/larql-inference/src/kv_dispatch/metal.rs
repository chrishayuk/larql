//! `KvDispatch` implementation for `larql_compute_metal::MetalBackend` — Step 4
//! scaffolding.
//!
//! **Behaviour:** every method delegates to
//! [`larql_compute::CpuBackend`]'s [`KvDispatch`] impl. K/V handles are
//! CPU-resident (host memory). No real GPU compute — the goal of this
//! step is to exercise the trait shape against actual Metal types so
//! engines can migrate to dispatch-through-trait safely on both
//! backends (Step 3c).
//!
//! Tok/s impact: catastrophically worse than the current Metal path
//! (every call has the same cost as CpuBackend). Acceptance criterion
//! is correctness, not speed. Real Metal kernels land in Step 5; this
//! file is the place where they bind.
//!
//! Feature-gated behind `metal` (same as `larql_compute_metal::MetalBackend`).

#![cfg(all(feature = "metal", target_os = "macos"))]

use ndarray::Array2;

use super::{CompressionCodec, KvDispatch, KvHandle, KvHandleInner, ResidualHandle};
use crate::model::ModelWeights;
use larql_compute::CpuBackend;
use larql_compute_metal::MetalBackend;

/// Convenience — the CPU backend instance every method delegates to.
/// Zero-sized type; const-construction is free.
const CPU: CpuBackend = CpuBackend;

impl KvDispatch for MetalBackend {
    fn alloc_kv_buffer(&self, layer: usize, max_tokens: usize, kv_dim: usize) -> KvHandle {
        // Handles are CPU-resident at Step 4. When real Metal kernels land
        // (Step 5), this returns a `MetalKvHandle` wrapping an
        // `MTLBuffer` instead.
        CPU.alloc_kv_buffer(layer, max_tokens, kv_dim)
    }

    fn append_kv(&self, handle: &mut KvHandle, k_row: &[f32], v_row: &[f32], abs_position: usize) {
        CPU.append_kv(handle, k_row, v_row, abs_position);
    }

    fn clip_kv(&self, handle: &mut KvHandle, window_size: usize) {
        CPU.clip_kv(handle, window_size);
    }

    fn read_kv_to_host(&self, handle: &KvHandle) -> Option<(Array2<f32>, Array2<f32>)> {
        CPU.read_kv_to_host(handle)
    }

    fn attention_step(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<Array2<f32>> {
        // A3 scaffold delegates to CPU. A4/A6 will introduce a Q4K-native
        // Metal path when `index` is `Some` and Q4K data is available.
        CPU.attention_step(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<Array2<f32>> {
        CPU.attention_step_windowed(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        CPU.attention_prefill(weights, tokens_embedded, layer, window, index)
    }

    fn recompute_kv_from_residuals(
        &self,
        weights: &ModelWeights,
        residuals: &Array2<f32>,
        layer: usize,
    ) -> Option<KvHandle> {
        CPU.recompute_kv_from_residuals(weights, residuals, layer)
    }

    fn compressed_kv_append(
        &self,
        handle: &mut KvHandle,
        k: &Array2<f32>,
        v: &Array2<f32>,
        codec: &dyn CompressionCodec,
    ) {
        CPU.compressed_kv_append(handle, k, v, codec);
    }

    fn upload_boundary_residual(&self, residual: &Array2<f32>) -> Option<ResidualHandle> {
        // CPU-resident upload. When Step 5 lands the pipelined boundary
        // upload kernel, this returns a `MetalResidualHandle` instead.
        CPU.upload_boundary_residual(residual)
    }

    fn forward_from_layer(
        &self,
        weights: &ModelWeights,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> Option<Array2<f32>> {
        CPU.forward_from_layer(weights, start_layer, residuals, token_ids)
    }

    fn residual_norm_store(
        &self,
        x: &Array2<f32>,
        residual: &Array2<f32>,
        norm_weights: &[f32],
    ) -> Array2<f32> {
        CPU.residual_norm_store(x, residual, norm_weights)
    }

    // ── Coarse fused intents ────────────────────────────────────────
    //
    // Route through Metal's fused `prefill_kquant` / `decode_token` kernels
    // — the production Metal hot path that powers `larql bench` at
    // ~87–100 tok/s on Gemma 3 4B Q4K. K/V cache state lives inside
    // `MetalBackend`'s internal `kv_cache` mutex; the returned
    // `KvHandle` is a sentinel since the engine doesn't manage the
    // state directly.

    fn coarse_prefill(
        &self,
        weights: &mut ModelWeights,
        token_ids: &[u32],
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Option<(Array2<f32>, KvHandle)> {
        let index = index?;
        let hidden = crate::vindex::fused_prefill(weights, index, token_ids, self)?;
        Some((hidden, KvHandle::new(MetalCoarseHandle)))
    }

    fn coarse_decode_step(
        &self,
        weights: &mut ModelWeights,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
        _handle: &mut KvHandle,
        _abs_position: usize,
    ) -> Option<Array2<f32>> {
        let index = index?;
        // K/V state lives inside `MetalBackend`'s internal mutex — the
        // `_handle` is a sentinel populated by `coarse_prefill`; we
        // don't read from it. `_abs_position` is tracked by the backend
        // via the K/V cache row count.
        crate::vindex::fused_decode_step(weights, index, token_id, self)
    }
}

/// Sentinel `KvHandleInner` for `MetalBackend::coarse_prefill` — the
/// actual K/V state lives in `MetalBackend`'s internal `kv_cache`
/// mutex, populated by the fused `prefill_kquant` / `decode_token` kernels.
/// The handle exists to satisfy the trait shape; engines must treat it
/// opaquely.
pub struct MetalCoarseHandle;

impl KvHandleInner for MetalCoarseHandle {
    fn cached_len(&self) -> usize {
        // Backend-side state; not exposed through the handle. Engines
        // that need the cache length should query the backend directly.
        0
    }
    fn kv_dim(&self) -> usize {
        0
    }
    fn backend_name(&self) -> &'static str {
        "metal-coarse"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// `KvHandleInner` and `ResidualHandleInner` placeholders for the
// per-layer dispatch path are not needed at Step 4 — we reuse
// `CpuKvHandle` and `CpuResidualHandle` from the CPU module since
// handles are host-resident. Step 5 will introduce `MetalKvHandle`
// (wrapping `MTLBuffer`) once real per-layer Metal compute lands.

#[cfg(test)]
mod tests {
    //! Parity test: MetalBackend's KvDispatch must produce bit-identical
    //! output to CpuBackend's KvDispatch (since it's delegation). This
    //! protects against a future divergence between MetalBackend's
    //! delegation and CpuBackend's evolving impl.

    use super::super::helpers::{kv_decode_step_via_dispatch, kv_prefill_via_dispatch};
    use super::*;
    use crate::test_utils::make_test_weights;

    /// Run `test` with a fresh `MetalBackend` when one is available;
    /// otherwise do nothing (test passes as a no-op).
    ///
    /// Concentrates the "skip when no Metal device" branch into one
    /// place so each metal test doesn't carry its own dead skip-path
    /// lines on a Metal-capable host.
    fn with_metal(test: impl FnOnce(MetalBackend)) {
        if let Some(metal) = MetalBackend::new() {
            test(metal);
        }
    }

    #[test]
    fn metal_backend_implements_kv_dispatch_compiles() {
        // Compile-time check that MetalBackend satisfies the KvDispatch
        // trait. Doesn't construct a Metal context (which would require
        // a real GPU); just proves the impl exists and resolves.
        fn assert_kv_dispatch<T: KvDispatch>() {}
        assert_kv_dispatch::<MetalBackend>();
    }

    #[test]
    fn metal_backend_implements_engine_backend_compiles() {
        // Same trick for the umbrella EngineBackend.
        fn assert_engine_backend<T: crate::kv_dispatch::EngineBackend>() {}
        assert_engine_backend::<MetalBackend>();
    }

    #[test]
    fn metal_prefill_matches_cpu_when_metal_available() {
        with_metal(|metal| {
            let weights = make_test_weights();
            let ffn = crate::ffn::WeightFfn { weights: &weights };
            let prompt = vec![0u32, 1, 2];

            let (h_metal, _) = kv_prefill_via_dispatch(&metal, &weights, &ffn, &prompt, None, None)
                .expect("metal prefill");
            let (h_cpu, _) =
                kv_prefill_via_dispatch(&CpuBackend, &weights, &ffn, &prompt, None, None)
                    .expect("cpu prefill");

            assert_eq!(
                h_metal, h_cpu,
                "MetalBackend KvDispatch must match CpuBackend bit-for-bit (Step 4 scaffolding delegates)"
            );
        });
    }

    #[test]
    fn metal_decode_step_matches_cpu_when_metal_available() {
        with_metal(|metal| {
            let weights = make_test_weights();
            let ffn = crate::ffn::WeightFfn { weights: &weights };
            let prompt = vec![0u32, 1];

            let (_, mut metal_handles) =
                kv_prefill_via_dispatch(&metal, &weights, &ffn, &prompt, None, None).unwrap();
            let (_, mut cpu_handles) =
                kv_prefill_via_dispatch(&CpuBackend, &weights, &ffn, &prompt, None, None).unwrap();

            let h_metal = kv_decode_step_via_dispatch(
                &metal,
                &weights,
                &ffn,
                &mut metal_handles,
                2u32,
                prompt.len(),
                None,
                None,
            )
            .expect("metal decode");
            let h_cpu = kv_decode_step_via_dispatch(
                &CpuBackend,
                &weights,
                &ffn,
                &mut cpu_handles,
                2u32,
                prompt.len(),
                None,
                None,
            )
            .expect("cpu decode");

            assert_eq!(
                h_metal, h_cpu,
                "MetalBackend decode must match CpuBackend bit-for-bit"
            );
        });
    }

    // ── Per-method delegation coverage ───────────────────────────────
    //
    // `MetalBackend`'s `KvDispatch` impl delegates every method to
    // `CpuBackend`. Each test exercises one delegation; `with_metal`
    // concentrates the "skip when no Metal device" branch into one
    // place so per-test dead lines on metal-capable hosts stay near
    // zero.

    #[test]
    fn metal_alloc_kv_buffer_when_available() {
        with_metal(|metal| {
            let handle = metal.alloc_kv_buffer(0, 32, 64);
            assert_eq!(handle.cached_len(), 0);
            assert_eq!(handle.kv_dim(), 64);
        });
    }

    #[test]
    fn metal_append_kv_when_available() {
        with_metal(|metal| {
            let mut handle = metal.alloc_kv_buffer(0, 32, 4);
            let row = [1.0_f32, 2.0, 3.0, 4.0];
            metal.append_kv(&mut handle, &row, &row, 0);
            assert_eq!(handle.cached_len(), 1);
        });
    }

    #[test]
    fn metal_clip_kv_when_available() {
        with_metal(|metal| {
            let mut handle = metal.alloc_kv_buffer(0, 32, 2);
            for i in 0..5 {
                let row = [i as f32, i as f32];
                metal.append_kv(&mut handle, &row, &row, i);
            }
            assert_eq!(handle.cached_len(), 5);
            metal.clip_kv(&mut handle, 3);
            assert_eq!(handle.cached_len(), 3);
        });
    }

    #[test]
    fn metal_read_kv_to_host_when_available() {
        with_metal(|metal| {
            let mut handle = metal.alloc_kv_buffer(0, 32, 2);
            let row = [9.0_f32, 8.0];
            metal.append_kv(&mut handle, &row, &row, 0);
            let (k, v) = metal.read_kv_to_host(&handle).unwrap();
            assert_eq!(k[[0, 0]], 9.0);
            assert_eq!(v[[0, 1]], 8.0);
        });
    }

    #[test]
    fn metal_attention_step_windowed_matches_cpu_when_available() {
        with_metal(|metal| {
            let weights = make_test_weights();
            let tokens = vec![0u32, 1, 2, 3];
            let h_in = crate::forward::embed_tokens_pub(&weights, &tokens);
            let (_, mut kv_metal) = metal
                .attention_prefill(&weights, &h_in, 0, None, None)
                .unwrap();
            let (_, mut kv_cpu) = CPU
                .attention_prefill(&weights, &h_in, 0, None, None)
                .unwrap();
            let h_new = crate::forward::embed_tokens_pub(&weights, &[4u32]);

            let h_metal = metal
                .attention_step_windowed(&weights, &h_new, &mut kv_metal, 0, tokens.len(), 2, None)
                .unwrap();
            let h_cpu = CPU
                .attention_step_windowed(&weights, &h_new, &mut kv_cpu, 0, tokens.len(), 2, None)
                .unwrap();
            assert_eq!(h_metal, h_cpu);
            assert_eq!(kv_metal.cached_len(), 2);
        });
    }

    #[test]
    fn metal_recompute_kv_from_residuals_when_available() {
        with_metal(|metal| {
            let weights = make_test_weights();
            let residuals = Array2::zeros((1, weights.hidden_size));
            // CpuBackend's default returns None (no impl); Metal delegates to CPU.
            assert!(metal
                .recompute_kv_from_residuals(&weights, &residuals, 0)
                .is_none());
        });
    }

    #[test]
    fn metal_compressed_kv_append_panics_when_available() {
        // `MetalBackend` delegates `compressed_kv_append` to `CpuBackend`,
        // which doesn't implement it — so the default `unimplemented!()`
        // panic fires. Use `catch_unwind` to capture the panic so the
        // test stays a no-op on hosts without Metal (no `#[should_panic]`
        // would skip cleanly otherwise).
        with_metal(|metal| {
            struct NoCodec;
            impl CompressionCodec for NoCodec {
                fn encode(&self, _: &[f32]) -> Vec<u8> {
                    vec![]
                }
                fn decode(&self, _: &[u8], _: usize) -> Vec<f32> {
                    vec![]
                }
                fn name(&self) -> &str {
                    "stub"
                }
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut handle = metal.alloc_kv_buffer(0, 32, 4);
                let k = Array2::zeros((1, 4));
                let v = Array2::zeros((1, 4));
                metal.compressed_kv_append(&mut handle, &k, &v, &NoCodec);
            }));
            let err = result.expect_err("compressed_kv_append should panic on Metal scaffold");
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .expect("panic payload should be a String / &str");
            assert!(
                msg.contains("compressed_kv_append not implemented"),
                "unexpected panic message: {msg}"
            );
        });
    }

    #[test]
    fn metal_upload_boundary_residual_when_available() {
        with_metal(|metal| {
            let residual =
                Array2::from_shape_vec((2, 3), (0..6).map(|i| i as f32).collect()).unwrap();
            let handle = metal.upload_boundary_residual(&residual).unwrap();
            assert_eq!(handle.shape(), (2, 3));
        });
    }

    #[test]
    fn metal_forward_from_layer_matches_cpu_when_available() {
        with_metal(|metal| {
            let weights = make_test_weights();
            let residual = Array2::zeros((1, weights.hidden_size));
            let h_metal_residual = metal.upload_boundary_residual(&residual).unwrap();
            let h_cpu_residual = CPU.upload_boundary_residual(&residual).unwrap();
            let tokens = vec![0u32, 1];

            let h_metal = metal
                .forward_from_layer(&weights, 1, &h_metal_residual, &tokens)
                .unwrap();
            let h_cpu = CPU
                .forward_from_layer(&weights, 1, &h_cpu_residual, &tokens)
                .unwrap();
            assert_eq!(h_metal, h_cpu);
        });
    }

    #[test]
    fn metal_residual_norm_store_matches_cpu_when_available() {
        with_metal(|metal| {
            let x = Array2::from_shape_vec((1, 4), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
            let residual = Array2::from_shape_vec((1, 4), vec![0.5_f32, 0.5, 0.5, 0.5]).unwrap();
            let norm_weights = vec![1.0_f32; 4];
            let h_metal = metal.residual_norm_store(&x, &residual, &norm_weights);
            let h_cpu = CPU.residual_norm_store(&x, &residual, &norm_weights);
            assert_eq!(h_metal, h_cpu);
        });
    }
}
