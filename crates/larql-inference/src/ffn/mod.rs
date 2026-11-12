//! Feed-forward network computation — trait-based with pluggable backends.
//!
//! Production: [`WalkFfn`](crate::vindex::WalkFfn) — the unified walk kernel.
//! Full-K and sparse-K are the same code path parameterised by [`WalkFfnConfig`](crate::vindex::WalkFfnConfig).
//!
//! Reference: [`WeightFfn`] + [`SparseFfn`] live here for correctness/bench
//! comparison (see `examples/walk_correctness.rs`); they are not used in
//! production dispatch.
//!
//! The [`FfnBackend`] trait + activation helpers moved to
//! `larql-compute` (ADR-0022 Step 2c) so substrate-level forward-pass
//! code can dispatch through it without depending on `larql-inference`.
//! All trait impls (`WeightFfn`, `SparseFfn`, `RemoteWalkBackend`, MoE
//! backends) stay here because they pull in inference-side topology.

pub mod expert_fold;
pub mod graph_backend;
pub mod local_moe;
pub mod moe_backend;
pub mod moe_bound;
pub mod moe_container;
pub mod moe_remote;
pub mod remote;
pub mod shard_auth;
pub mod sparse;
pub mod sparse_compute;
#[cfg(test)]
mod tests;
pub mod weight;

pub use larql_compute::ffn::{
    expert_weight::resolvable as expert_weights_resolvable, gelu_tanh, gelu_tanh_gate_up, sigmoid,
    silu_gate_up, ExpertWeightFfn, FfnActivations, FfnBackend, PackedExpertWeightFfn,
    SparseActivations, Q4K_Q8K_SUPERBLOCK_ELEMS,
};

// ── Re-exports ──

pub use local_moe::LocalMoeFfn;
pub use moe_backend::{
    InProcessMoeBackend, MoeBackendError, MoeExpertBackend, MoeFailurePolicy, MoeRoute,
};
pub use moe_bound::BoundMoeBackend;
pub use moe_container::{CompositionError, ContainerRoutedBackend};
pub use moe_remote::{
    MoeFfn, MoeRouterWeights, RecordedRefusal, RefusalPolicy, RemoteMoeBackend, RemoteMoeError,
    RemoteMoeFfn, ShardConfig,
};
pub use remote::{
    decode_q8k_batch_response_entries, decode_single_response, encode_binary_request,
    encode_binary_request_as, encode_q8k_batch_request, LayerShardedBackend, RemoteFfnConfig,
    RemoteFfnError, RemoteLatencyStats, RemoteWalkBackend, WireFormat, WirePreference, BINARY_CT,
    F16_CT, I8_CT, Q8K_BATCH_CT,
};
pub use sparse::SparseFfn;
pub use sparse_compute::{
    sparse_ffn_forward, sparse_ffn_forward_observed, sparse_ffn_forward_with_full_overrides,
    sparse_ffn_forward_with_full_overrides_observed, sparse_ffn_forward_with_overrides,
    FeatureSlotOverride,
};
pub use weight::{dense_ffn_forward_backend, BackendFfn, NullFfn, ViewFfn, WeightFfn};

// ── Per-layer backend selection ──

/// Selects which FFN backend to use for each layer.
pub struct LayerFfnRouter<'a> {
    backends: Vec<&'a dyn FfnBackend>,
    num_layers: usize,
}

impl<'a> LayerFfnRouter<'a> {
    pub fn uniform(backend: &'a dyn FfnBackend, num_layers: usize) -> Self {
        Self {
            backends: vec![backend; num_layers],
            num_layers,
        }
    }

    pub fn per_layer(backends: Vec<&'a dyn FfnBackend>) -> Self {
        let num_layers = backends.len();
        Self {
            backends,
            num_layers,
        }
    }

    pub fn get(&self, layer: usize) -> &dyn FfnBackend {
        if layer < self.num_layers {
            self.backends[layer]
        } else {
            self.backends[self.num_layers - 1]
        }
    }
}

// Activation functions (`sigmoid`, `silu_gate_up`, `gelu_tanh`,
// `gelu_tanh_gate_up`) moved to `larql_compute::ffn` (ADR-0022 Step 2c).
// Re-exported above at module head so existing `crate::ffn::*` paths
// keep working.

#[cfg(test)]
mod router_tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use weight::WeightFfn;

    #[test]
    fn ffn_backend_default_forward_moe_full_layer_returns_none() {
        // The trait's default `forward_moe_full_layer` impl always
        // returns None — non-MoE backends rely on this fallback.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let h = larql_vindex::ndarray::Array2::<f32>::zeros((1, weights.hidden_size));
        assert!(ffn.forward_moe_full_layer(0, &h).is_ok_and(|o| o.is_none()));
    }

    #[test]
    fn layer_ffn_router_uniform_returns_same_backend_for_every_layer() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let router = LayerFfnRouter::uniform(&ffn, 4);
        // All four layers return the same backend's name.
        for layer in 0..4 {
            assert_eq!(router.get(layer).name(), "weights");
        }
    }

    #[test]
    fn layer_ffn_router_per_layer_dispatches_per_index() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let backends: Vec<&dyn FfnBackend> = (0..3).map(|_| &ffn as &dyn FfnBackend).collect();
        let router = LayerFfnRouter::per_layer(backends);
        for layer in 0..3 {
            assert_eq!(router.get(layer).name(), "weights");
        }
    }

    #[test]
    fn layer_ffn_router_get_out_of_range_clamps_to_last() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let router = LayerFfnRouter::uniform(&ffn, 2);
        // Layer 99 > num_layers=2 → clamps to last (index 1).
        assert_eq!(router.get(99).name(), "weights");
    }
}

// Architecture-correct FFN computation is in weight::dense_ffn_forward
// and sparse_compute::sparse_ffn_forward. No legacy SiLU-only functions.
