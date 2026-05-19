//! `AsyncComputeBackend` implementation for `larql_compute_metal::MetalBackend`
//! — Step A3 scaffolding.
//!
//! **Behaviour:** every async method delegates to
//! [`larql_compute::CpuBackend`]'s [`AsyncComputeBackend`] impl. Handles
//! are CPU-resident; the in-flight command buffer is conceptual only.
//! No real GPU compute, no deferred dispatch — the goal of this step is
//! to exercise the trait shape against actual `MetalBackend` ownership
//! patterns so engines can migrate to async dispatch safely on both
//! backends in Step A5.
//!
//! Tok/s impact: catastrophically worse than the current Metal fused
//! `decode_token` path (every call has CpuBackend's cost). Acceptance
//! criterion is correctness, not speed. Real deferred dispatch — one
//! `MTLCommandBuffer` per session, commit at engine checkpoints — lands
//! in Step A4. Per-engine specialised shaders land in Step A6.
//!
//! Feature-gated behind `metal` (same as `larql_compute_metal::MetalBackend`).

#![cfg(all(feature = "metal", target_os = "macos"))]

use ndarray::Array2;

use super::{AsyncComputeBackend, AttentionHandle, ResidualUploadHandle};
use crate::ffn::FfnBackend;
use crate::kv_dispatch::{KvHandle, ResidualHandle};
use crate::model::ModelWeights;
use larql_compute::CpuBackend;
use larql_compute_metal::MetalBackend;

/// Convenience — the CPU backend instance every method delegates to.
/// Zero-sized type; const-construction is free.
const CPU: CpuBackend = CpuBackend;

impl AsyncComputeBackend for MetalBackend {
    fn attention_step_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> AttentionHandle {
        // Handles are CPU-resident at Step A3. When Step A4's deferred
        // dispatch lands, this records the intent into an in-flight
        // `MTLCommandBuffer` and returns a `MetalAttentionHandle`.
        CPU.attention_step_async(weights, query, kv, layer, abs_position, index)
    }

    fn attention_step_windowed_async(
        &self,
        weights: &ModelWeights,
        query: &Array2<f32>,
        kv: &mut KvHandle,
        layer: usize,
        abs_position: usize,
        window: usize,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> AttentionHandle {
        CPU.attention_step_windowed_async(weights, query, kv, layer, abs_position, window, index)
    }

    fn attention_prefill_async(
        &self,
        weights: &ModelWeights,
        tokens_embedded: &Array2<f32>,
        layer: usize,
        window: Option<usize>,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> (AttentionHandle, KvHandle) {
        CPU.attention_prefill_async(weights, tokens_embedded, layer, window, index)
    }

    fn upload_boundary_residual_async(
        &self,
        residual: &Array2<f32>,
    ) -> (ResidualUploadHandle, ResidualHandle) {
        // CPU-resident upload at Step A3. When Step A6 lands the
        // pipelined boundary-upload kernel (Apollo's win), this returns
        // a `MetalResidualHandle` whose upload fuses with the next
        // attention encode in the same command buffer.
        CPU.upload_boundary_residual_async(residual)
    }

    fn forward_from_layer_async(
        &self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        start_layer: usize,
        residuals: &ResidualHandle,
        token_ids: &[u32],
    ) -> AttentionHandle {
        CPU.forward_from_layer_async(weights, ffn, start_layer, residuals, token_ids)
    }
}

// `recompute_kv_from_residuals_async` stays at the trait default
// (`unimplemented!()`). MarkovResidual is the only engine that needs
// it; the real Metal K/V-recompute kernel lands in Step A6 alongside
// that engine's migration. CpuBackend's sync `KvDispatch` doesn't
// implement it either, so a CPU-delegating Metal scaffold would just
// surface the same `unimplemented!()`.

#[cfg(test)]
mod tests {
    //! Parity tests: MetalBackend's async dispatch must produce bit-
    //! identical output to CpuBackend's at Step A3 (since it's pure
    //! delegation). Protects against a future drift between the two
    //! impls before real GPU work lands.

    use super::*;
    use crate::test_utils::make_test_weights;

    fn metal_backend_or_skip() -> Option<MetalBackend> {
        MetalBackend::new()
    }

    #[test]
    fn metal_backend_implements_async_compute_backend_compiles() {
        fn assert_async<T: AsyncComputeBackend>() {}
        assert_async::<MetalBackend>();
    }

    #[test]
    fn metal_attention_prefill_async_matches_cpu_when_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        let weights = make_test_weights();
        let tokens = vec![0u32, 1, 2];
        let h_in = crate::forward::embed_tokens_pub(&weights, &tokens);

        let (h_metal_handle, kv_metal) =
            metal.attention_prefill_async(&weights, &h_in, 0, None, None);
        let h_metal = h_metal_handle.read();

        let (h_cpu_handle, kv_cpu) = CPU.attention_prefill_async(&weights, &h_in, 0, None, None);
        let h_cpu = h_cpu_handle.read();

        use crate::kv_dispatch::KvDispatch;
        let (k_metal, v_metal) = metal.read_kv_to_host(&kv_metal).unwrap();
        let (k_cpu, v_cpu) = CPU.read_kv_to_host(&kv_cpu).unwrap();

        assert_eq!(
            h_metal, h_cpu,
            "MetalBackend async prefill must match CpuBackend bit-for-bit (A3 delegates)"
        );
        assert_eq!(k_metal, k_cpu, "prefill K must match");
        assert_eq!(v_metal, v_cpu, "prefill V must match");
    }

    #[test]
    fn metal_attention_step_async_matches_cpu_when_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        let weights = make_test_weights();
        let tokens = vec![0u32, 1, 2];
        let h_in = crate::forward::embed_tokens_pub(&weights, &tokens);

        let (_, mut kv_metal) = metal.attention_prefill_async(&weights, &h_in, 0, None, None);
        let (_, mut kv_cpu) = CPU.attention_prefill_async(&weights, &h_in, 0, None, None);

        let h_new = crate::forward::embed_tokens_pub(&weights, &[3u32]);
        let abs_position = tokens.len();

        let h_metal = metal
            .attention_step_async(&weights, &h_new, &mut kv_metal, 0, abs_position, None)
            .read();
        let h_cpu = CPU
            .attention_step_async(&weights, &h_new, &mut kv_cpu, 0, abs_position, None)
            .read();

        assert_eq!(
            h_metal, h_cpu,
            "MetalBackend async step must match CpuBackend bit-for-bit"
        );
    }

    #[test]
    fn metal_commit_control_defaults_are_safe_when_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        // At A3 there's no deferred state. Both default impls must hold.
        assert!(!metal.has_pending_work());
        metal.flush().expect("flush no-op at A3");
    }

    #[test]
    fn metal_attention_step_windowed_async_matches_cpu_when_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        let weights = make_test_weights();
        let tokens = vec![0u32, 1, 2, 3, 4];
        let h_in = crate::forward::embed_tokens_pub(&weights, &tokens);

        let (_, mut kv_metal) = metal.attention_prefill_async(&weights, &h_in, 0, None, None);
        let (_, mut kv_cpu) = CPU.attention_prefill_async(&weights, &h_in, 0, None, None);

        let h_new = crate::forward::embed_tokens_pub(&weights, &[5u32]);
        let abs_position = tokens.len();
        let window = 3;

        let h_metal = metal
            .attention_step_windowed_async(
                &weights,
                &h_new,
                &mut kv_metal,
                0,
                abs_position,
                window,
                None,
            )
            .read();
        let h_cpu = CPU
            .attention_step_windowed_async(
                &weights,
                &h_new,
                &mut kv_cpu,
                0,
                abs_position,
                window,
                None,
            )
            .read();

        assert_eq!(
            h_metal, h_cpu,
            "MetalBackend windowed-step async must match CpuBackend bit-for-bit"
        );
        assert_eq!(kv_metal.cached_len(), kv_cpu.cached_len());
        assert_eq!(kv_metal.cached_len(), window);
    }

    #[test]
    fn metal_forward_from_layer_async_matches_cpu_when_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        let weights = make_test_weights();
        let tokens = vec![0u32, 1, 2];

        let residual =
            Array2::from_shape_vec((1, weights.hidden_size), vec![0.0; weights.hidden_size])
                .unwrap();
        let res_metal = {
            use crate::kv_dispatch::KvDispatch;
            metal.upload_boundary_residual(&residual).unwrap()
        };
        let res_cpu = {
            use crate::kv_dispatch::KvDispatch;
            CPU.upload_boundary_residual(&residual).unwrap()
        };

        let ffn = crate::ffn::NullFfn;
        let h_metal = metal
            .forward_from_layer_async(&weights, &ffn, 1, &res_metal, &tokens)
            .read();
        let h_cpu = CPU
            .forward_from_layer_async(&weights, &ffn, 1, &res_cpu, &tokens)
            .read();

        assert_eq!(
            h_metal, h_cpu,
            "MetalBackend forward_from_layer_async must match CpuBackend bit-for-bit"
        );
    }

    #[test]
    fn metal_upload_boundary_residual_async_matches_cpu_when_available() {
        let Some(metal) = metal_backend_or_skip() else {
            eprintln!("Skipping: metal backend not available on this host");
            return;
        };
        let residual = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32).collect()).unwrap();

        let (upload_metal, res_metal) = metal.upload_boundary_residual_async(&residual);
        upload_metal.read();

        let (upload_cpu, res_cpu) = CPU.upload_boundary_residual_async(&residual);
        upload_cpu.read();

        assert_eq!(res_metal.shape(), res_cpu.shape());
        assert_eq!(res_metal.backend_name(), res_cpu.backend_name());
    }
}
