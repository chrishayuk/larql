//! `LocalWalkExecutor` — per-layer attention + FFN dispatch.
//!
//! Wraps the existing attention helpers
//! ([`crate::attention::run_attention_with_kv_backend`],
//! [`crate::attention::run_attention_block_decode_step_backend`]) +
//! the caller's `FfnBackend` into a [`LayerExecutor`] impl.
//!
//! Intended consumers:
//!
//! - **Residual-stream engines** (`markov_residual`,
//!   `markov_residual_codec`, `boundary_per_layer`) that need
//!   per-layer dispatch for their state policy to fire.
//! - **Remote-FFN bench / driver** that passes a `RemoteWalkBackend`
//!   as the FFN parameter. Per-layer dispatch is the natural fit;
//!   each FFN call is an HTTP round-trip.
//!
//! The executor holds a `ComputeBackend` for the attention's
//! projection matmuls. The `FfnBackend` is passed per-call so engines
//! can swap FFN dispatch (local walk → remote shard) without
//! reconstructing the executor.

use ndarray::Array2;

use crate::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, SharedKV,
};
use crate::ffn::FfnBackend;
use crate::forward::run_ffn;
use crate::model::ModelWeights;
use larql_compute::ComputeBackend;

use super::{ExecutorDispatchKind, LayerExecutor};

/// Per-layer Rust dispatch executor.
///
/// Construct with a `ComputeBackend` (used for the attention's Q/K/V/O
/// projections). The FFN dispatcher is passed per-call so the same
/// executor can drive both local and remote FFN setups.
pub struct LocalWalkExecutor<'a> {
    backend: &'a dyn ComputeBackend,
}

impl<'a> LocalWalkExecutor<'a> {
    pub fn new(backend: &'a dyn ComputeBackend) -> Self {
        Self { backend }
    }
}

impl<'a> LayerExecutor for LocalWalkExecutor<'a> {
    fn backend(&self) -> &dyn ComputeBackend {
        self.backend
    }

    fn dispatch_kind(&self) -> ExecutorDispatchKind {
        ExecutorDispatchKind::PerLayer
    }

    fn name(&self) -> &str {
        "local-walk"
    }

    fn run_prefill_layer(
        &self,
        weights: &ModelWeights,
        layer: usize,
        hidden_in: &Array2<f32>,
        ffn: &dyn FfnBackend,
    ) -> Option<(Array2<f32>, SharedKV)> {
        // Attention with K/V capture. The backend handles the
        // projection matmuls; `run_attention_with_kv_backend` returns
        // `(h_post_attn, k_rope, v_final)`.
        let (h_post_attn, k, v) =
            run_attention_with_kv_backend(weights, hidden_in, layer, Some(self.backend))?;
        // FFN through the caller-supplied dispatcher. This is the
        // critical decoupling: local FFN uses `WeightFfn` / `BackendFfn`,
        // remote FFN uses `RemoteWalkBackend`, MoE shards use
        // `RemoteMoeBackend`. The executor doesn't pick.
        let (h_out, _activation) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        Some((h_out, (k, v)))
    }

    fn run_decode_layer(
        &self,
        weights: &ModelWeights,
        layer: usize,
        hidden_in: &Array2<f32>,
        prior_kv: &SharedKV,
        abs_position: usize,
        ffn: &dyn FfnBackend,
    ) -> Option<(Array2<f32>, SharedKV)> {
        // Decode-step attention appends one row to the K/V and returns
        // both the post-attention hidden + the extended K/V. The
        // engine integrates the K/V per policy (store it, discard it,
        // mix with cold tier, etc.).
        let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
            weights,
            hidden_in,
            layer,
            Some(prior_kv),
            abs_position,
            Some(self.backend),
        )?;
        let (h_out, _activation) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        Some((h_out, new_kv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::WeightFfn;
    use crate::test_utils::make_test_weights;
    use larql_compute::CpuBackend;

    fn fixture() -> ModelWeights {
        make_test_weights()
    }

    #[test]
    fn executor_advertises_per_layer_dispatch() {
        let backend = CpuBackend;
        let exec = LocalWalkExecutor::new(&backend);
        assert_eq!(exec.dispatch_kind(), ExecutorDispatchKind::PerLayer);
        assert_eq!(exec.name(), "local-walk");
    }

    #[test]
    fn backend_accessor_returns_underlying_backend() {
        let backend = CpuBackend;
        let exec = LocalWalkExecutor::new(&backend);
        // backend() is now provided via the trait; smoke test it.
        assert!(LayerExecutor::backend(&exec).name().starts_with("cpu"));
    }

    // ── run_prefill_layer ─────────────────────────────────────────────────

    #[test]
    fn prefill_layer_returns_finite_hidden_and_kv() {
        let weights = fixture();
        let ffn = WeightFfn { weights: &weights };
        let backend = CpuBackend;
        let exec = LocalWalkExecutor::new(&backend);
        let hidden_in = Array2::from_elem((3, weights.hidden_size), 0.1f32);
        let (h_out, (k, v)) = exec
            .run_prefill_layer(&weights, 0, &hidden_in, &ffn)
            .expect("prefill_layer");
        assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
        assert!(h_out.iter().all(|v| v.is_finite()));
        // K/V capture: [seq_len, kv_dim]
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[3, kv_dim]);
        assert_eq!(v.shape(), &[3, kv_dim]);
        assert!(k.iter().all(|v| v.is_finite()));
        assert!(v.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_layer_chains_across_layers() {
        // Drive every layer through the executor; verify the per-layer
        // outputs compose into a coherent forward pass.
        let weights = fixture();
        let ffn = WeightFfn { weights: &weights };
        let backend = CpuBackend;
        let exec = LocalWalkExecutor::new(&backend);
        let mut h = crate::forward::embed_tokens_pub(&weights, &[0u32, 1, 2]);
        for layer in 0..weights.num_layers {
            let (h_next, _kv) = exec
                .run_prefill_layer(&weights, layer, &h, &ffn)
                .expect("layer prefill");
            assert_eq!(h_next.shape(), &[3, weights.hidden_size]);
            assert!(h_next.iter().all(|v| v.is_finite()));
            h = h_next;
        }
    }

    // ── run_decode_layer ──────────────────────────────────────────────────

    #[test]
    fn decode_layer_appends_one_kv_row() {
        let weights = fixture();
        let ffn = WeightFfn { weights: &weights };
        let backend = CpuBackend;
        let exec = LocalWalkExecutor::new(&backend);
        // Seed K/V from a 2-token prefill, then decode one step.
        let prefill_hidden = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let (_, prior_kv) = exec
            .run_prefill_layer(&weights, 0, &prefill_hidden, &ffn)
            .unwrap();
        assert_eq!(prior_kv.0.shape()[0], 2);

        let new_token_hidden = Array2::from_elem((1, weights.hidden_size), 0.05f32);
        let (h_out, new_kv) = exec
            .run_decode_layer(&weights, 0, &new_token_hidden, &prior_kv, 2, &ffn)
            .expect("decode_layer");
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert!(h_out.iter().all(|v| v.is_finite()));
        // K/V grew by exactly one row.
        assert_eq!(new_kv.0.shape()[0], 3);
        assert_eq!(new_kv.1.shape()[0], 3);
    }

    #[test]
    fn decode_layer_with_empty_prior_kv_appends_first_row() {
        let weights = fixture();
        let ffn = WeightFfn { weights: &weights };
        let backend = CpuBackend;
        let exec = LocalWalkExecutor::new(&backend);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        let empty_kv: SharedKV = (Array2::zeros((0, kv_dim)), Array2::zeros((0, kv_dim)));
        let h_in = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (h_out, new_kv) = exec
            .run_decode_layer(&weights, 0, &h_in, &empty_kv, 0, &ffn)
            .expect("decode_layer with empty prior");
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert_eq!(new_kv.0.shape()[0], 1);
    }

    /// Spec §3.5: the executor must honor the caller's FFN dispatcher.
    /// This test uses two different `FfnBackend` impls and confirms the
    /// executor's output differs accordingly. (`NullFfn` returns zero
    /// activations; `WeightFfn` runs the real FFN.)
    #[test]
    fn executor_honors_caller_ffn_dispatcher() {
        use crate::ffn::NullFfn;
        let weights = fixture();
        let backend = CpuBackend;
        let exec = LocalWalkExecutor::new(&backend);
        let h_in = Array2::from_elem((2, weights.hidden_size), 0.1f32);

        let ffn_real = WeightFfn { weights: &weights };
        let (h_real, _) = exec
            .run_prefill_layer(&weights, 0, &h_in, &ffn_real)
            .unwrap();

        let ffn_null = NullFfn;
        let (h_null, _) = exec
            .run_prefill_layer(&weights, 0, &h_in, &ffn_null)
            .unwrap();

        // The two FFN backends produce different outputs; the executor
        // is dispatching through whichever it's given.
        let diff: f32 = h_real
            .iter()
            .zip(h_null.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.0,
            "outputs should differ when FFN backend changes (real vs null)"
        );
    }
}
