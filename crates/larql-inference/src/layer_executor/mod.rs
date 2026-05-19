//! Per-layer execution surface for KV engines.
//!
//! Specification:
//! [`docs/specs/engine-state-vs-execution.md`](../../docs/specs/engine-state-vs-execution.md).
//!
//! The `LayerExecutor` trait separates **state policy** (the engine's
//! concern: what residuals/K/V/frames to retain, when to evict, what to
//! compress) from **execution strategy** (the executor's concern: how
//! to run one layer's attention + FFN — locally fused, locally
//! per-layer, with remote FFN, with sharded MoE).
//!
//! Engines hold an `&dyn LayerExecutor` and call `run_decode_layer`
//! per layer. The executor decides whether to dispatch through Metal's
//! fused kernel, a per-layer Rust path, or an HTTP round-trip to a
//! remote FFN shard. The engine doesn't know or care.
//!
//! # Phase 1 scope (this module)
//!
//! v0.1 ships the trait + `ExecutorDispatchKind` + one implementation:
//!
//! - [`LocalWalkExecutor`] — per-layer Rust dispatch. Wraps the
//!   `run_attention_with_kv_backend` / `run_attention_block_decode_step_backend`
//!   helpers + a caller-supplied `FfnBackend`. This is the
//!   `MarkovResidualEngine` walk path, generalised to honor the FFN
//!   parameter properly.
//!
//! Deferred to Phase 2 (engine migration):
//!
//! - `LocalFusedExecutor` — wraps `KvDispatch::coarse_prefill` /
//!   `coarse_decode_step`. The fused fast-path doesn't fit a per-layer
//!   trait shape; it'll either live on a sibling trait or be modelled
//!   as a degenerate executor whose `run_*_layer` methods return
//!   `None` and a different code path runs the whole sequence at once.
//! - `RemoteFfnExecutor` — distributed inference (FFN remote).
//! - `MoeShardedExecutor` — MoE experts sharded across remote nodes.

pub mod local_walk;

pub use local_walk::LocalWalkExecutor;

use crate::attention::SharedKV;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use ndarray::Array2;

/// Whether an executor owns its K/V state internally (`Fused`) or
/// produces it per-layer for the engine to manage (`PerLayer`).
///
/// Engines whose state policy requires per-layer interception (the
/// residual-stream family) declare
/// `KvEngine::requires_per_layer_dispatch() -> true` and refuse
/// construction with a `Fused` executor at runtime
/// (per spec §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorDispatchKind {
    /// Backend runs all layers at once internally and owns the K/V
    /// cache. Engine state policy is unenforceable through this
    /// executor; engines that hold `Fused` executors are transparent
    /// wrappers (or refuse construction).
    Fused,
    /// Per-layer dispatch: the engine drives the layer loop and owns
    /// the K/V state externally. Engine state policy fully expressible.
    PerLayer,
}

/// Per-layer execution surface. See module docs for the design
/// rationale.
///
/// All methods take `&self` (executors are immutable views over a
/// compute backend + optional vindex). State lives in the engine.
pub trait LayerExecutor {
    /// The compute backend the executor uses for attention projections.
    ///
    /// Exposed on the trait because engines occasionally need to pass
    /// it through to other helpers (e.g. `recompute_kv` for
    /// residual-stream engines) and to drive the default-impl fallback
    /// in `KvEngine::*_via_executor` methods.
    fn backend(&self) -> &dyn larql_compute::ComputeBackend;

    /// Which dispatch model does this executor use?
    fn dispatch_kind(&self) -> ExecutorDispatchKind;

    /// Human-readable name for logging / engine info.
    fn name(&self) -> &str;

    /// Run one layer's full forward pass over a prefill chunk.
    ///
    /// `hidden_in` is the pre-attention residual entering this layer.
    /// Returns `(hidden_out, kv)` where `kv` is the layer's K/V for
    /// the chunk. The engine may store or discard `kv` per its state
    /// policy (residual-store engines discard and rebuild from
    /// residuals; standard engines store).
    ///
    /// `PerLayer` executors implement this; `Fused` executors return
    /// `None` by default — engines should not call this on `Fused`.
    fn run_prefill_layer(
        &self,
        weights: &ModelWeights,
        layer: usize,
        hidden_in: &Array2<f32>,
        ffn: &dyn FfnBackend,
    ) -> Option<(Array2<f32>, SharedKV)> {
        let _ = (weights, layer, hidden_in, ffn);
        None
    }

    /// Run one layer for a single decode step.
    ///
    /// `hidden_in` is the residual entering this layer (shape `[1, hidden]`).
    /// `prior_kv` is the K/V state the engine wants to attend against
    /// (whatever the engine assembled from its store: hot only, hot+cold,
    /// hot only after eviction, etc.).
    ///
    /// Returns `(hidden_out, new_kv)` where `new_kv` is `prior_kv`
    /// extended by one row for the new token. The engine integrates
    /// `new_kv` into its state per policy.
    ///
    /// `PerLayer` executors implement this; `Fused` executors return
    /// `None` by default.
    fn run_decode_layer(
        &self,
        weights: &ModelWeights,
        layer: usize,
        hidden_in: &Array2<f32>,
        prior_kv: &SharedKV,
        abs_position: usize,
        ffn: &dyn FfnBackend,
    ) -> Option<(Array2<f32>, SharedKV)> {
        let _ = (weights, layer, hidden_in, prior_kv, abs_position, ffn);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubExecutor {
        backend: larql_compute::CpuBackend,
    }
    impl LayerExecutor for StubExecutor {
        fn backend(&self) -> &dyn larql_compute::ComputeBackend {
            &self.backend
        }
        fn dispatch_kind(&self) -> ExecutorDispatchKind {
            ExecutorDispatchKind::PerLayer
        }
        fn name(&self) -> &str {
            "stub"
        }
    }

    #[test]
    fn stub_executor_default_methods_return_none() {
        let exec = StubExecutor {
            backend: larql_compute::CpuBackend,
        };
        assert_eq!(exec.dispatch_kind(), ExecutorDispatchKind::PerLayer);
        assert_eq!(exec.name(), "stub");
        assert!(exec.backend().name().starts_with("cpu"));
    }

    /// Drive the default impl bodies of `run_prefill_layer` and
    /// `run_decode_layer` on a stub that doesn't override them. Both
    /// must return None (line 104, 130) — they're the safe-fallback for
    /// `Fused` executors that don't expose per-layer dispatch.
    #[test]
    fn stub_executor_default_run_prefill_and_decode_return_none() {
        let exec = StubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let weights = crate::test_utils::make_test_weights();
        let ffn = crate::ffn::WeightFfn { weights: &weights };
        let hidden = Array2::<f32>::zeros((1, weights.hidden_size));
        assert!(exec.run_prefill_layer(&weights, 0, &hidden, &ffn).is_none());

        // SharedKV = (K, V) as Array2<f32>: shape doesn't matter — the
        // default impl returns None before touching either tensor.
        let kv: SharedKV = (Array2::<f32>::zeros((1, 1)), Array2::<f32>::zeros((1, 1)));
        assert!(exec
            .run_decode_layer(&weights, 0, &hidden, &kv, 0, &ffn)
            .is_none());
    }

    #[test]
    fn dispatch_kind_enum_variants_are_distinct() {
        assert_ne!(ExecutorDispatchKind::Fused, ExecutorDispatchKind::PerLayer);
        // Clone + Copy work as expected.
        let a = ExecutorDispatchKind::PerLayer;
        let b = a;
        assert_eq!(a, b);
    }
}
