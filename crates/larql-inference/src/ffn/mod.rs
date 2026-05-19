//! Feed-forward network computation — trait-based with pluggable backends.
//!
//! Production: [`WalkFfn`](crate::vindex::WalkFfn) — the unified walk kernel.
//! Full-K and sparse-K are the same code path parameterised by [`WalkFfnConfig`](crate::vindex::WalkFfnConfig).
//!
//! Reference: [`WeightFfn`] + [`SparseFfn`] live here for correctness/bench
//! comparison (see `examples/walk_correctness.rs`); they are not used in
//! production dispatch.

pub mod graph_backend;
pub mod moe_remote;
pub mod remote;
pub mod sparse;
pub mod sparse_compute;
#[cfg(test)]
mod tests;
pub mod weight;

use ndarray::Array2;

/// Number of elements in one Q4_K / Q8_K super-block (the block size both
/// formats share). Hidden sizes that are not a multiple of this value
/// can't use the block-quantised wire formats — `walk_ffn` and
/// `grid::remote_ffn` use this for their dispatch checks. Mirrors
/// llama.cpp's `QK_K`.
pub const Q4K_Q8K_SUPERBLOCK_ELEMS: usize = 256;

// ── Trait ──

/// FFN backend trait. Defines how a single layer's FFN is computed.
pub trait FfnBackend {
    /// Run the FFN for a given layer on the pre-FFN-normed residual.
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32>;

    /// Run FFN and also return the pre-down activation (for capture).
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>);

    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// For hybrid MoE layers: receive `h_post_attn` (post-attention, pre-FFN,
    /// unnormalized) and return the full layer output `h_out`. Returns `None`
    /// to fall back to local dispatch.
    fn forward_moe_full_layer(
        &self,
        _layer: usize,
        _h_post_attn: &larql_vindex::ndarray::Array2<f32>,
    ) -> Option<larql_vindex::ndarray::Array2<f32>> {
        None
    }
}

// ── Re-exports ──

pub use moe_remote::{MoeRouterWeights, RemoteMoeBackend, RemoteMoeError, ShardConfig};
pub use remote::{
    LayerShardedBackend, RemoteFfnConfig, RemoteFfnError, RemoteLatencyStats, RemoteWalkBackend,
    WirePreference,
};
pub use sparse::SparseFfn;
pub use sparse_compute::{
    sparse_ffn_forward, sparse_ffn_forward_with_full_overrides, sparse_ffn_forward_with_overrides,
    FeatureSlotOverride,
};
pub use weight::{dense_ffn_forward_backend, BackendFfn, NullFfn, WeightFfn};

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

// ── Activation functions ──

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(|v| v * sigmoid(v));
    &activated * up
}

pub fn gelu_tanh_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(gelu_tanh);
    &activated * up
}

pub fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

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
        assert!(ffn.forward_moe_full_layer(0, &h).is_none());
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
