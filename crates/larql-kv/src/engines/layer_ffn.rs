//! The per-layer FFN step every engine forward loop shares, and the refusal
//! channel it carries.
//!
//! Two functions, one concept: what happens between attention and the next
//! layer. [`apply_ple_and_layer_scalar`] is the arch-dependent tail;
//! [`layer_ffn_or_moe`] is the dispatch that decides whether the layer runs
//! densely or through a bound expert route, and — since PR #197's execution
//! ring — whether it runs at all.

use larql_execution::BoxRefusal;
use larql_inference::ffn::FfnBackend;
use larql_inference::ModelWeights;
use ndarray::Array2;

/// Post-FFN tail of the per-layer sequence: `apply_per_layer_embedding`
/// then `apply_layer_scalar`, in that order — mirroring the legacy
/// `kv_prefill_run` / `kv_decode_step_run` loops in
/// [`crate::generation`], the oracle for every engine forward. Both
/// steps are no-ops on archs without PLE / layer-scalar keys, so
/// threading this through non-PLE paths costs one clone and changes no
/// bits.
pub(crate) fn apply_ple_and_layer_scalar(
    weights: &ModelWeights,
    h_post_ffn: &Array2<f32>,
    layer: usize,
    ple_input: Option<&Array2<f32>>,
) -> Array2<f32> {
    let mut h_out = larql_inference::forward::ple::apply_per_layer_embedding(
        weights, h_post_ffn, layer, ple_input,
    );
    larql_inference::forward::layer::apply_layer_scalar(weights, &mut h_out, layer);
    h_out
}

/// Per-layer FFN dispatch for engine forward loops, MoE-aware.
///
/// On a hybrid-MoE arch, when a `moe_ffn` hook is supplied (e.g.
/// `RemoteMoeFfn` for `--moe-shards`), call its
/// [`FfnBackend::forward_moe_full_layer`] — it returns the full layer output
/// (dense `h1` + experts `h2` + combine), dispatching experts to whatever
/// route is bound behind it. Otherwise fall back to the engine's own dense
/// FFN (`dense_ffn`) followed by [`apply_ple_and_layer_scalar`], the same
/// per-layer sequence as the legacy `kv_prefill_run` / `kv_decode_step_run`
/// oracle.
///
/// `ple_input` is this layer's entry from `precompute_per_layer_inputs`
/// (`None` on non-PLE archs, where PLE + layer_scalar are no-ops).
///
/// Nothing here knows which container the experts came from. The hook is an
/// `FfnBackend`, so a VINDEX2 bank, a VINDEX3 bound route and a remote shard
/// are the same shape from this side — which is the property that lets one
/// refusal channel serve all of them.
///
/// # The refusal channel
///
/// `Err` means a bound route declined and this layer has no complete output:
///
/// ```text
/// Ok(h)          the layer ran — via the MoE hook, or densely
/// Err(refusal)   a strict route refused; this layer is incomplete
/// ```
///
/// "Not applicable" is not a third return: a backend with no MoE hook, or one
/// that ran and declined, yields `Ok(None)` from `forward_moe_full_layer` and
/// falls through to the dense path, which is exactly what it did before.
/// Conflating that with a refusal is what the typed channel exists to prevent.
///
/// Whether an `Err` can arrive at all is the *policy*'s decision, not the
/// route's: `MoeFfn::best_effort` records the refusal and returns the dense
/// half (an explicit opt-in), while `MoeFfn::strict` — and `RemoteMoeFfn`,
/// which is strict — converts it into an error here.
///
/// Before this returned a `Result`, a refusal was printed to stderr and the
/// dense half was returned — a complete-looking hidden state whose experts
/// never ran, which the engine had no way to detect.
pub(crate) fn layer_ffn_or_moe(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    dense_ffn: &dyn FfnBackend,
    moe_ffn: Option<&dyn FfnBackend>,
    ple_input: Option<&Array2<f32>>,
) -> Result<Array2<f32>, BoxRefusal> {
    // Pure MoE takes the hook exactly as hybrid does — its FFN *is* the
    // expert block; the dense fallback below would ask for gate/up/down
    // tensors the checkpoint does not have.
    if weights.arch.is_moe() || weights.arch.is_hybrid_moe() {
        if let Some(mf) = moe_ffn {
            if let Some(h_out) = mf.forward_moe_full_layer(layer, h_post_attn)? {
                // Returned as-is: `forward_moe_full_layer` is contracted to
                // produce the FULL layer output. Every production impl routes
                // through `moe_ffn_block_cpu(_with_index)`, which applies PLE
                // + layer_scalar internally (`RemoteMoeFfn` / `LocalMoeFfn`
                // directly; `LayerShardedRemote` and the ffn-policy router
                // delegate to them; the HTTP walk backend requests
                // `full_output` from the server). Applying either step again
                // here would double-apply.
                return Ok(h_out);
            }
        }
    }
    let (h_post_ffn, _) =
        larql_inference::forward::run_ffn(weights, h_post_attn, layer, dense_ffn, false);
    Ok(apply_ple_and_layer_scalar(
        weights,
        &h_post_ffn,
        layer,
        ple_input,
    ))
}

#[cfg(test)]
mod tests {
    use super::{apply_ple_and_layer_scalar, layer_ffn_or_moe};
    use larql_execution::{BoxRefusal, RefusalKind};
    use larql_inference::ffn::{FfnBackend, RecordedRefusal};
    use larql_inference::test_utils::{make_test_gemma4_moe_weights, make_test_weights};
    use ndarray::Array2;

    /// Rows and a value for the synthetic hidden states below. Any finite
    /// input works; these are named so a failure message points at a
    /// deliberate fixture rather than a bare literal.
    const ROWS: usize = 2;
    const SENTINEL: f32 = 7.0;
    const REFUSAL_MESSAGE: &str = "expert 7 is not resident";

    /// `FfnBackend` whose MoE hook returns a sentinel so we can tell the MoE
    /// branch from the dense `run_ffn` fallback.
    struct SentinelFfn;

    impl FfnBackend for SentinelFfn {
        fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
            Array2::zeros(x.raw_dim())
        }
        fn name(&self) -> &str {
            "sentinel"
        }
        fn forward_moe_full_layer(
            &self,
            _layer: usize,
            h_post_attn: &Array2<f32>,
        ) -> Result<Option<Array2<f32>>, BoxRefusal> {
            Ok(Some(Array2::from_elem(h_post_attn.raw_dim(), SENTINEL)))
        }
    }

    /// `FfnBackend` whose MoE hook always refuses, with a chosen kind.
    struct RefusingFfn(RefusalKind);

    impl FfnBackend for RefusingFfn {
        fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
            Array2::zeros(x.raw_dim())
        }
        fn name(&self) -> &str {
            "refusing"
        }
        fn forward_moe_full_layer(
            &self,
            layer: usize,
            _h_post_attn: &Array2<f32>,
        ) -> Result<Option<Array2<f32>>, BoxRefusal> {
            Err(Box::new(RecordedRefusal {
                layer,
                kind: self.0,
                message: REFUSAL_MESSAGE.into(),
            }))
        }
    }

    #[test]
    fn uses_moe_hook_on_hybrid_moe_arch() {
        let weights = make_test_gemma4_moe_weights();
        assert!(weights.arch.is_hybrid_moe());
        let h = Array2::<f32>::zeros((ROWS, weights.hidden_size));
        let out = layer_ffn_or_moe(&weights, &h, 0, &SentinelFfn, Some(&SentinelFfn), None)
            .expect("an executing hook must not refuse");
        // Took the MoE hook → sentinel output, not the dense run_ffn path.
        assert!(
            out.iter().all(|&v| v == SENTINEL),
            "expected MoE-hook sentinel output"
        );
    }

    #[test]
    fn falls_back_to_dense_when_no_hook() {
        let weights = make_test_gemma4_moe_weights();
        let h = Array2::<f32>::zeros((ROWS, weights.hidden_size));
        // No moe_ffn → dense run_ffn even on a MoE arch (no experts dispatched).
        let out = layer_ffn_or_moe(&weights, &h, 0, &SentinelFfn, None, None)
            .expect("the dense path cannot refuse");
        assert_eq!(out.shape(), &[ROWS, weights.hidden_size]);
        assert!(
            out.iter().any(|&v| v != SENTINEL),
            "must NOT be the MoE-hook sentinel"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// A refusal leaves as a refusal, with its classification and its words.
    ///
    /// Every kind is swept because the helper must not acquire an opinion
    /// about which ones are worth propagating — that decision belongs to the
    /// policy above it and the engine below it.
    #[test]
    fn a_refusing_hook_propagates_rather_than_falling_back() {
        let weights = make_test_gemma4_moe_weights();
        let h = Array2::<f32>::zeros((ROWS, weights.hidden_size));
        let mut checked = 0usize;
        for kind in RefusalKind::ALL {
            let ffn = RefusingFfn(kind);
            let err = layer_ffn_or_moe(&weights, &h, 0, &ffn, Some(&ffn), None)
                .expect_err("a refusing hook must not yield a layer output");
            assert_eq!(err.kind(), kind, "the classification must survive");
            assert!(
                err.to_string().contains(REFUSAL_MESSAGE),
                "the route's own words must survive: {err}"
            );
            checked += 1;
        }
        assert_eq!(checked, RefusalKind::ALL.len(), "coverage shrank");
    }

    /// On a dense arch the hook is never consulted, so it cannot refuse.
    ///
    /// Pinned because the alternative — asking the hook first and treating a
    /// refusal as fatal — would make every dense model's forward depend on a
    /// route it does not use.
    #[test]
    fn a_refusing_hook_is_not_consulted_on_a_dense_arch() {
        let weights = make_test_weights();
        assert!(!weights.arch.is_hybrid_moe());
        let h = Array2::<f32>::zeros((ROWS, weights.hidden_size));
        let ffn = RefusingFfn(RefusalKind::Residency);
        let out = layer_ffn_or_moe(&weights, &h, 0, &ffn, Some(&ffn), None)
            .expect("a dense arch never reaches the MoE hook");
        assert_eq!(out.shape(), &[ROWS, weights.hidden_size]);
    }

    /// The tail is a no-op on an arch with neither PLE nor a layer scalar.
    #[test]
    fn the_tail_is_identity_without_ple_or_layer_scalar() {
        let weights = make_test_weights();
        let h = Array2::<f32>::from_elem((ROWS, weights.hidden_size), SENTINEL);
        let out = apply_ple_and_layer_scalar(&weights, &h, 0, None);
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            h.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a no-op tail must not perturb the bits it passes through"
        );
    }

    /// The stub's unused trait surface, so the fixture itself stays honest.
    #[test]
    fn sentinel_ffn_trait_surface() {
        let s = SentinelFfn;
        let x = Array2::<f32>::zeros((ROWS, 4));
        assert_eq!(s.name(), "sentinel");
        assert_eq!(s.forward(0, &x).shape(), &[ROWS, 4]);
        let (o, obs) = s.forward_observed(0, &x);
        assert_eq!(o.shape(), &[ROWS, 4]);
        assert!(
            obs.is_absent(),
            "sentinel stub must not fabricate activations"
        );
        let r = RefusingFfn(RefusalKind::Unsupported);
        assert_eq!(r.name(), "refusing");
        assert_eq!(r.forward(0, &x).shape(), &[ROWS, 4]);
    }
}
