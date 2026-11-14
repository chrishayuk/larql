//! `LocalMoeFfn` — the **in-process** (no remote shards) counterpart to
//! [`RemoteMoeFfn`](crate::ffn::RemoteMoeFfn).
//!
//! It lets the KvEngine layer drive CPU MoE decode with a real KV cache while
//! computing the experts locally from the resident vindex weights, instead of
//! round-tripping them to a `larql-server` shard. The engine owns attention
//! (and its KV cache) and calls [`FfnBackend::forward_moe_full_layer`] per MoE
//! layer; this adapter computes that layer's MoE FFN block via
//! [`moe_ffn_block_cpu`](crate::vindex::kquant_forward::moe_ffn_block_cpu) with
//! `moe_remote = None` — dense `h1` plus experts `h2` from the local
//! `build_moe_weights` + `cpu_moe_forward` path.
//!
//! This is the missing fast in-process MoE path: the standalone
//! `predict_kquant` route is correct but full-recompute (no KV cache), and
//! [`RemoteMoeFfn`](crate::ffn::RemoteMoeFfn) is KV-cached but pays a network
//! round-trip per layer. `LocalMoeFfn` is KV-cached *and* local — the right
//! shape for a fair single-box CPU MoE benchmark (no loopback-shard tax) and
//! for `larql run` on a MoE vindex that fits RAM.
//!
//! Byte-equivalence: this is exactly [`RemoteMoeFfn`] with the remote backend
//! swapped for the local expert kernel, so a KV-cached `LocalMoeFfn` decode
//! produces the same tokens as the full-recompute `predict_kquant` path
//! (KV-cache parity), the same way the remote adapter is byte-identical to the
//! standalone remote recompute path (larql-kv "MoE-aware KV engines (C1)").
//!
//! PLE is **not** applied on this path (`moe_ffn_block_cpu` is called with
//! `ple_input = None`), so callers must route Per-Layer-Embedding
//! architectures (Gemma 4 E-series) through the full-recompute path instead.
//! Non-PLE MoE models (Gemma 4 26B-A4B, 31B-MoE) are unaffected.

use larql_compute::ffn::FfnBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

use crate::ffn::WeightFfn;
use crate::vindex::moe_ffn_block_cpu_with_index;

/// In-process MoE [`FfnBackend`] for CPU decode through a `KvEngine`.
///
/// The dense `h1` contribution runs through [`WeightFfn`] (f32 dense FFN over
/// `weights.tensors` — the caller pre-dequantizes the client's Q4K attention +
/// dense FFN), and the expert `h2` contribution is computed locally from the
/// resident expert weights (no shards).
///
/// When `index` is set, the dense slab can additionally run quantised-direct
/// under `LARQL_Q4K_DIRECT_FFN=1` (decode steps only); with the flag unset or
/// `index: None` the path is byte-identical to the f32 `WeightFfn` slab.
pub struct LocalMoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: Option<&'a larql_vindex::VectorIndex>,
}

impl<'a> FfnBackend for LocalMoeFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        WeightFfn {
            weights: self.weights,
        }
        .forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        WeightFfn {
            weights: self.weights,
        }
        .forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        "local-moe"
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        // Local dispatch over resident weights: there is no operand this could
        // fail to reach, so it never refuses.
        Ok(Some(
            moe_ffn_block_cpu_with_index(
                self.weights,
                h_post_attn,
                layer,
                &WeightFfn {
                    weights: self.weights,
                },
                None,
                None,
                self.index,
            )
            // No route is bound, so the refusal branch is unreachable rather than
            // ignored — stated so a future caller that *does* bind one here has to
            // decide what a failure means instead of inheriting silence.
            .expect("no MoE route is bound, so no refusal is reachable"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::moe_remote::RemoteMoeBackend;
    use crate::test_utils::make_test_gemma4_moe_weights;
    use ndarray::Array2;

    /// `forward_moe_full_layer` runs the MoE FFN block locally: dense `h1` +
    /// experts `h2` computed in-process + combine. Asserts a full, finite
    /// layer output of the right shape.
    #[test]
    fn forward_moe_full_layer_returns_finite_combined_output() {
        let weights = make_test_gemma4_moe_weights();
        let ffn = LocalMoeFfn {
            weights: &weights,
            index: None,
        };
        let h_post_attn = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = ffn
            .forward_moe_full_layer(0, &h_post_attn)
            .expect("executes")
            .expect("LocalMoeFfn always returns Some");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// The local path must actually evaluate experts in-process, not take the
    /// dense early-return fallback. `build_moe_weights` returning `Some` means
    /// `moe_ffn_block_cpu(.., moe_remote = None)` runs `cpu_moe_forward` per
    /// position — the local expert kernel. A *disconnected* best-effort remote
    /// route zeroes `h2` (its dispatch errors), so the local output must carry a
    /// non-zero expert delta relative to it; we assert both signals.
    #[test]
    fn in_process_expert_path_is_live() {
        let weights = make_test_gemma4_moe_weights();

        // The per-layer MoE weights build → the local expert branch is taken,
        // not the dense early-return (which would skip experts entirely).
        let moe =
            crate::layer_graph::pipeline_layer::build_moe_weights(&weights, &*weights.arch, 0)
                .expect("the fixture declares an executable router");
        assert!(
            moe.is_some(),
            "MoE fixture must build per-layer expert weights so the in-process \
             expert kernel (cpu_moe_forward) is exercised"
        );

        // The experts contribute a non-zero delta vs. the experts-zeroed
        // (disconnected) remote — i.e. they are really computed, not left at 0.
        // (The synthetic fixture's expert weights are tiny, so the delta is
        // small but strictly non-zero; an identically-zeroed h2 would give a
        // bit-exact match.)
        let local = LocalMoeFfn {
            weights: &weights,
            index: None,
        };
        // Best-effort by name: the zero-expert baseline is exactly the
        // degradation the remote adapter no longer performs by default.
        let disconnected = RemoteMoeBackend::new_disconnected();
        let remote = crate::ffn::MoeFfn::best_effort(&weights, &disconnected);
        let h_post_attn = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out_local = local
            .forward_moe_full_layer(0, &h_post_attn)
            .unwrap()
            .unwrap();
        let out_zero_experts = remote
            .forward_moe_full_layer(0, &h_post_attn)
            .unwrap()
            .unwrap();
        assert_eq!(out_local.shape(), out_zero_experts.shape());
        let max_abs_diff = out_local
            .iter()
            .zip(out_zero_experts.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff > 0.0,
            "local experts must contribute a non-zero delta vs zeroed experts"
        );
    }

    /// `forward` / `forward_observed` run the dense FFN fallback;
    /// `name` is stable.
    #[test]
    fn dense_fallbacks_and_name() {
        let weights = make_test_gemma4_moe_weights();
        let ffn = LocalMoeFfn {
            weights: &weights,
            index: None,
        };
        assert_eq!(ffn.name(), "local-moe");
        let x = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let dense = ffn.forward(0, &x);
        assert_eq!(dense.shape()[0], 2);
        assert!(dense.iter().all(|v| v.is_finite()));
        let (out, obs) = ffn.forward_observed(0, &x);
        assert_eq!(out.shape()[0], 2);
        let act = obs.into_dense().expect("dense fallback observes densely");
        assert_eq!(act.shape()[0], 2);
    }
}
