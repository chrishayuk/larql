//! `ExpertWeightFfn` — the per-expert f32 MoE backend. Reference tier.
//!
//! The sibling of [`WeightFfn`](super::WeightFfn) for mixture-of-experts
//! models: full f32 matmuls straight off `ModelWeights`, no quantisation, no
//! shards, no session state. Slow and exact, which is the point — it is the
//! local ground truth a fast path gets diffed against, and the FFN the Shannon
//! scorer needs in order to score a MoE model at all.
//!
//! Every architecture-specific decision is read from [`ModelArchitecture`]
//! rather than branched on a family name: which per-expert keys hold the
//! weights, whether the router and experts carry biases, and how gate and up
//! combine ([`larql_models::ExpertGatePolicy`]). A new MoE architecture that answers those
//! accessors is served here without touching this file.
//!
//! **Why this did not exist before.** Nothing in `larql-compute` or
//! `larql-inference` consumed per-expert f32 tensors — the quantised path
//! (`cpu_moe_forward` via `MoeLayerWeights`) needs packed bytes, which a raw
//! safetensors load does not produce. So `larql shannon score` hardcoded the
//! dense `WeightFfn` and panicked on any MoE model, which is why GB had no
//! instrument for the entire model class the K3 ladder is built on. See
//! `docs/k3-funnel.md` §4.7.

pub(crate) mod gate;
/// Expert selection — public because the VINDEX3 production backend runs
/// the served selection rule rather than a second transcription of it.
pub mod router;
// Public for the `tests/test_moe_route_trace_*.rs` end-to-end exercises:
// the env-resolved sink is process-global (`OnceLock`), so each scenario
// needs its own test binary, and those binaries can only reach a `pub` path.
pub mod trace;

use larql_models::{ModelArchitecture, ModelWeights};
use ndarray::{s, Array2, Axis};

use super::FfnBackend;
use crate::forward::{add_bias, dot_proj};

/// Number of projections fused into one `gate_up` bias row.
const FUSED_HALVES: usize = 2;

/// [`FfnActivations::Absent`](super::FfnActivations::Absent) reason for
/// MoE layers: per-token expert routing means there is no single
/// pre-down activation tensor to observe.
const REASON_MOE_NO_SINGLE_ACTIVATION: &str =
    "MoE layer has no single pre-down activation (per-token expert routing)";

/// Per-expert f32 MoE FFN over `ModelWeights`.
///
/// Construct on a model whose architecture reports [`ModelArchitecture::is_moe`]
/// and exposes per-expert weight keys. [`resolvable`] reports whether this
/// model's weights actually satisfy that, so callers can fall back rather than
/// panic deep inside a matmul.
pub struct ExpertWeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

/// Whether `weights` carries what [`ExpertWeightFfn`] needs at `layer`:
/// a router and expert 0's three projections.
///
/// Checking one expert is enough to distinguish "this is a per-expert f32 MoE
/// model" from "this is dense, or its experts are packed" — the case that
/// matters. A model missing expert 17 specifically is a corrupt checkpoint and
/// should fail loudly at the matmul, not be silently rerouted to a dense path.
pub fn resolvable(weights: &ModelWeights, layer: usize) -> bool {
    let arch = &*weights.arch;
    if !arch.is_moe() {
        return false;
    }
    let has_router = arch
        .moe_router_key(layer)
        .is_some_and(|k| weights.tensors.contains_key(&k) || weights.vectors.contains_key(&k));
    has_router
        && [
            arch.expert_ffn_gate_key(layer, 0),
            arch.expert_ffn_up_key(layer, 0),
            arch.expert_ffn_down_key(layer, 0),
        ]
        .iter()
        .all(|k| k.as_ref().is_some_and(|k| weights.tensors.contains_key(k)))
}

impl<'a> ExpertWeightFfn<'a> {
    fn arch(&self) -> &dyn ModelArchitecture {
        &*self.weights.arch
    }

    /// Router logits for every token: `[tokens, num_experts]`, bias included.
    fn router_logits(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        let arch = self.arch();
        let w = arch
            .moe_router_key(layer)
            .and_then(|k| self.weights.tensors.get(&k).cloned())?;
        let mut logits = dot_proj(x, &w);
        if let Some(bias) = arch
            .moe_router_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            add_bias(&mut logits, bias);
        }
        Some(logits)
    }

    /// One half of an expert's fused `gate_up` bias row.
    ///
    /// The bias is interleaved on its last axis exactly as the weight rows
    /// are, so it has to be de-interleaved the same way — taking the leading
    /// half here would reintroduce the mixture bug one tensor over.
    fn fused_bias_half(row: &[f32], half_offset: usize) -> Vec<f32> {
        row.iter()
            .skip(half_offset)
            .step_by(FUSED_HALVES)
            .copied()
            .collect()
    }

    /// Bias for expert `expert`'s gate and up projections, if declared.
    fn expert_gate_up_bias(&self, layer: usize, expert: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let key = self.arch().packed_gate_up_bias_key(layer)?;
        let all = self.weights.tensors.get(&key)?;
        let row = all.row(expert);
        let row = row.as_slice()?;
        Some((Self::fused_bias_half(row, 0), Self::fused_bias_half(row, 1)))
    }

    /// Bias for expert `expert`'s down projection, if declared.
    fn expert_down_bias(&self, layer: usize, expert: usize) -> Option<Vec<f32>> {
        let key = self.arch().packed_down_bias_key(layer)?;
        let all = self.weights.tensors.get(&key)?;
        Some(all.row(expert).to_vec())
    }

    /// Run one expert over the token rows routed to it.
    ///
    /// `rows` is `[n, hidden]`; the result is `[n, hidden]`, unweighted.
    fn run_expert(&self, layer: usize, expert: usize, rows: &Array2<f32>) -> Option<Array2<f32>> {
        let arch = self.arch();
        let w_gate = arch
            .expert_ffn_gate_key(layer, expert)
            .and_then(|k| self.weights.tensors.get(&k))?;
        let w_up = arch
            .expert_ffn_up_key(layer, expert)
            .and_then(|k| self.weights.tensors.get(&k))?;
        let w_down = arch
            .expert_ffn_down_key(layer, expert)
            .and_then(|k| self.weights.tensors.get(&k))?;

        let mut gate = dot_proj(rows, w_gate);
        let mut up = dot_proj(rows, w_up);
        if let Some((gate_bias, up_bias)) = self.expert_gate_up_bias(layer, expert) {
            add_bias(&mut gate, &gate_bias);
            add_bias(&mut up, &up_bias);
        }

        let activation = gate::apply(&gate, &up, arch.expert_gate_policy(), arch.activation());
        let mut out = dot_proj(&activation, w_down);
        if let Some(bias) = self.expert_down_bias(layer, expert) {
            add_bias(&mut out, &bias);
        }
        Some(out)
    }

    /// The MoE block: route every token, run each hit expert once over all of
    /// its tokens, and accumulate the weighted outputs.
    ///
    /// Grouping by expert rather than looping per token is what makes this
    /// tractable — it turns `tokens × k` matvecs into `hit_experts` matmuls,
    /// mirroring the reference's own `expert_hit` loop.
    fn moe_block(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        let arch = self.arch();
        let logits = self.router_logits(layer, x)?;
        let top_k = arch.num_experts_per_token();
        let num_experts = arch.num_experts();
        let routing_policy = arch.expert_routing_policy();

        // expert -> (token rows, routing weights)
        let mut buckets: Vec<(Vec<usize>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); num_experts];
        // `None` unless LARQL_MOE_ROUTE_TRACE is set, which keeps the untraced
        // path free of the per-token id vectors entirely.
        let mut traced = trace::buffer(x.nrows());
        for (token, row) in logits.axis_iter(Axis(0)).enumerate() {
            let row = row.to_vec();
            let routing = router::select(&row, top_k, routing_policy);
            if let Some(rows) = traced.as_mut() {
                rows.push(routing.iter().map(|&(expert, _)| expert).collect());
            }
            for (expert, weight) in routing {
                if let Some(bucket) = buckets.get_mut(expert) {
                    bucket.0.push(token);
                    bucket.1.push(weight);
                }
            }
        }
        trace::record(layer, traced);

        // Latent-axis sparsity probe — opt-in, default off = byte-identical.
        // Applied AFTER `router_logits` above so the trained router still
        // selects the same experts with the same weights; only the shared
        // input handed to them is reduced. Masking channel `j` is exactly
        // equivalent to not reading column `j` of every routed expert's gate
        // and up matrices, which is the leverage this probe is testing.
        let mask = crate::cpu::ops::moe::latent_mask::active();
        let mut retained_per_token: Vec<Vec<bool>> = Vec::new();
        let x_owned;
        let x = if let Some(m) = mask {
            let mut masked = x.clone();
            for mut row in masked.axis_iter_mut(Axis(0)) {
                let slice = row.as_slice_mut().expect("contiguous row");
                let retained = m.mask_in_place(slice);
                // Recorded PER LAYER: channel `j` in different layers are
                // unrelated coordinates, so a channel-indexed histogram
                // summed across layers is uniform by construction and
                // cannot detect a per-layer stable subset.
                crate::cpu::ops::moe::latent_mask::record_stats(layer, &retained);
                retained_per_token.push(retained);
            }
            x_owned = masked;
            &x_owned
        } else {
            x
        };

        let mut out = Array2::<f32>::zeros(x.raw_dim());
        for (expert, (tokens, weights)) in buckets.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }
            let rows = x.select(Axis(0), tokens);
            let expert_out = self.run_expert(layer, expert, &rows)?;
            for (i, (&token, &weight)) in tokens.iter().zip(weights.iter()).enumerate() {
                let contribution = &expert_out.slice(s![i, ..]) * weight;
                let mut dst = out.slice_mut(s![token, ..]);
                dst += &contribution;
            }
        }

        // Both-sides variant: mask the same channels on the aggregated
        // expert output, so `down` shrinks too. Input-side alone leaves a
        // third of expert bytes fixed and caps the lever at 1.448×.
        if let Some(m) = mask.filter(|m| m.both_sides) {
            for (t, mut row) in out.axis_iter_mut(Axis(0)).enumerate() {
                if let Some(retained) = retained_per_token.get(t) {
                    m.apply(row.as_slice_mut().expect("contiguous row"), retained);
                }
            }
        }
        Some(out)
    }
}

impl FfnBackend for ExpertWeightFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.moe_block(layer, x).unwrap_or_else(|| {
            panic!(
                "ExpertWeightFfn: layer {layer} has no resolvable per-expert MoE weights. \
                 Check `resolvable()` before constructing this backend."
            )
        })
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, super::FfnActivations) {
        // A MoE layer has no single pre-down activation — each token
        // visits a different set of experts. Report Absent rather than
        // inventing one (the pre-split impl returned the block output as
        // the "activation", which was real numbers but the wrong object).
        (
            self.forward(layer, x),
            super::FfnActivations::Absent {
                reason: REASON_MOE_NO_SINGLE_ACTIVATION,
            },
        )
    }

    fn name(&self) -> &str {
        "expert-weights"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::{Activation, ExpertGatePolicy, WeightArray};

    const HIDDEN: usize = 4;
    const INTER: usize = 4;
    const EXPERTS: usize = 3;
    const TOP_K: usize = 2;

    fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> WeightArray {
        Array2::from_shape_fn((rows, cols), |(r, c)| f(r, c)).into_shared()
    }

    /// A GPT-OSS-shaped `ModelWeights`: three experts of width 4, top-2
    /// routing, every declared MoE tensor present.
    ///
    /// The weights are chosen so each expert's contribution is identifiable:
    /// gate is the identity, up is zero (so ClampedGlu's `(up + 1)` term is
    /// exactly 1), and expert `e`'s down projection is `(e + 1) × I`. The
    /// router reads one hidden dimension per expert, so a one-hot token
    /// selects a known expert.
    fn gpt_oss_weights() -> ModelWeights {
        let mut weights = larql_models::test_fixtures::make_test_weights();
        weights.arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 1,
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": HIDDEN,
            "num_local_experts": EXPERTS,
            "num_experts_per_tok": TOP_K,
            "swiglu_limit": 7.0,
        }));
        weights.hidden_size = HIDDEN;
        weights.intermediate_size = INTER;
        weights.num_layers = 1;

        let identity = |r: usize, c: usize| if r == c { 1.0 } else { 0.0 };
        let router_key = weights.arch.moe_router_key(0).unwrap();
        let router_bias_key = weights.arch.moe_router_bias_key(0).unwrap();
        let gate_up_bias_key = weights.arch.packed_gate_up_bias_key(0).unwrap();
        let down_bias_key = weights.arch.packed_down_bias_key(0).unwrap();
        let expert_keys: Vec<(String, String, String)> = (0..EXPERTS)
            .map(|e| {
                (
                    weights.arch.expert_ffn_gate_key(0, e).unwrap(),
                    weights.arch.expert_ffn_up_key(0, e).unwrap(),
                    weights.arch.expert_ffn_down_key(0, e).unwrap(),
                )
            })
            .collect();

        // Router: expert e scores hidden dimension e.
        weights
            .tensors
            .insert(router_key, mat(EXPERTS, HIDDEN, identity));
        weights.vectors.insert(router_bias_key, vec![0.0; EXPERTS]);

        for (e, (gate_key, up_key, down_key)) in expert_keys.into_iter().enumerate() {
            weights
                .tensors
                .insert(gate_key, mat(INTER, HIDDEN, identity));
            weights
                .tensors
                .insert(up_key, mat(INTER, HIDDEN, |_, _| 0.0));
            weights.tensors.insert(
                down_key,
                mat(
                    HIDDEN,
                    INTER,
                    move |r, c| {
                        if r == c {
                            (e + 1) as f32
                        } else {
                            0.0
                        }
                    },
                ),
            );
        }

        // Expert biases, both zero by default.
        weights.tensors.insert(
            gate_up_bias_key,
            mat(EXPERTS, FUSED_HALVES * INTER, |_, _| 0.0),
        );
        weights
            .tensors
            .insert(down_bias_key, mat(EXPERTS, HIDDEN, |_, _| 0.0));
        weights
    }

    /// One token, one-hot on dimension `d`.
    fn one_hot(d: usize, scale: f32) -> Array2<f32> {
        Array2::from_shape_fn((1, HIDDEN), |(_, c)| if c == d { scale } else { 0.0 })
    }

    fn forward_one(weights: &ModelWeights, d: usize) -> Array2<f32> {
        ExpertWeightFfn { weights }.forward(0, &one_hot(d, 1.0))
    }

    // ── bias de-interleaving ────────────────────────────────────────────

    /// Bias de-interleaving must mirror the weight rows: even entries are the
    /// gate's, odd the up's. Same convention as
    /// `mxfp4::deinterleave_fused_half`, applied to a bias vector.
    #[test]
    fn fused_bias_half_takes_even_then_odd() {
        let row = [10.0f32, 20.0, 11.0, 21.0, 12.0, 22.0];
        assert_eq!(
            ExpertWeightFfn::fused_bias_half(&row, 0),
            vec![10.0, 11.0, 12.0]
        );
        assert_eq!(
            ExpertWeightFfn::fused_bias_half(&row, 1),
            vec![20.0, 21.0, 22.0]
        );
    }

    #[test]
    fn fused_bias_half_on_a_single_pair() {
        let row = [1.0f32, 2.0];
        assert_eq!(ExpertWeightFfn::fused_bias_half(&row, 0), vec![1.0]);
        assert_eq!(ExpertWeightFfn::fused_bias_half(&row, 1), vec![2.0]);
    }

    // ── resolvability ───────────────────────────────────────────────────

    #[test]
    fn resolvable_is_false_for_a_dense_model() {
        let weights = larql_models::test_fixtures::make_test_weights();
        assert!(!resolvable(&weights, 0));
    }

    #[test]
    fn resolvable_is_true_for_the_moe_fixture() {
        assert!(resolvable(&gpt_oss_weights(), 0));
    }

    /// Declaring the keys is not enough — the weights have to be present. A
    /// packed-expert model answers the accessors but stores no per-expert
    /// tensors, and must be declined rather than half-resolved.
    #[test]
    fn resolvable_is_false_when_expert_tensors_are_absent() {
        let mut weights = gpt_oss_weights();
        let key = weights.arch.expert_ffn_gate_key(0, 0).unwrap();
        weights.tensors.remove(&key);
        assert!(!resolvable(&weights, 0));
    }

    #[test]
    fn resolvable_is_false_without_a_router() {
        let mut weights = gpt_oss_weights();
        let key = weights.arch.moe_router_key(0).unwrap();
        weights.tensors.remove(&key);
        assert!(!resolvable(&weights, 0));
    }

    // ── routing behaviour ───────────────────────────────────────────────

    /// The output must be a convex combination of exactly `top_k` experts.
    /// With gate = I, up = 0 and down = (e+1)·I, a one-hot input on dim 0
    /// routes to experts 0 and 1 (logits 1, 0, 0 — ties break by index), so
    /// the dim-0 output lands strictly between 1× and 2× the base GLU.
    #[test]
    fn output_is_a_convex_combination_of_top_k_experts() {
        let weights = gpt_oss_weights();
        let out = forward_one(&weights, 0);

        assert_eq!(out.shape(), &[1, HIDDEN]);
        let base = out[[0, 0]];
        let glu = crate::ffn::sigmoid(1.702);
        assert!(
            base > glu && base < 2.0 * glu,
            "{base} should sit strictly between {glu} and {}",
            2.0 * glu
        );
    }

    /// Different tokens must route independently. A token one-hot on dim 2
    /// reaches expert 2 (the strongest); a token one-hot on dim 0 cannot.
    #[test]
    fn each_token_routes_independently() {
        let weights = gpt_oss_weights();
        let ffn = ExpertWeightFfn { weights: &weights };
        let mut x = Array2::<f32>::zeros((2, HIDDEN));
        x[[0, 0]] = 1.0;
        x[[1, 2]] = 1.0;
        let out = ffn.forward(0, &x);
        assert!(
            out[[1, 2]] > out[[0, 0]],
            "the token routed to the strongest expert should dominate: {} vs {}",
            out[[1, 2]],
            out[[0, 0]]
        );
    }

    /// The router bias has to reach the logits. Biasing expert 2 far above the
    /// others pulls it into the selection and changes the output — this fails
    /// if the bias is resolved but never added.
    #[test]
    fn router_bias_changes_the_selection() {
        let before = forward_one(&gpt_oss_weights(), 0);

        let mut biased = gpt_oss_weights();
        let key = biased.arch.moe_router_bias_key(0).unwrap();
        biased.vectors.insert(key, vec![0.0, 0.0, 10.0]);
        let after = forward_one(&biased, 0);

        assert!(
            (after[[0, 0]] - before[[0, 0]]).abs() > 1e-4,
            "router bias did not affect routing: {} vs {}",
            after[[0, 0]],
            before[[0, 0]]
        );
    }

    // ── expert biases ───────────────────────────────────────────────────

    /// Routing weights sum to 1, so a down bias of 1.0 on every expert must
    /// shift every output dimension by exactly 1.0.
    #[test]
    fn down_bias_shifts_the_output_by_itself() {
        let before = forward_one(&gpt_oss_weights(), 0);

        let mut biased = gpt_oss_weights();
        let key = biased.arch.packed_down_bias_key(0).unwrap();
        biased.tensors.insert(key, mat(EXPERTS, HIDDEN, |_, _| 1.0));
        let after = forward_one(&biased, 0);

        for d in 0..HIDDEN {
            let delta = after[[0, d]] - before[[0, d]];
            assert!(
                (delta - 1.0).abs() < 1e-5,
                "dim {d} shifted by {delta}, expected 1.0"
            );
        }
    }

    /// The gate/up bias is interleaved, so only its even entries may reach the
    /// gate. Biasing the halves separately must produce different outputs —
    /// which fails if the halves are swapped or if either is dropped.
    #[test]
    fn gate_and_up_bias_halves_have_distinct_effects() {
        let base = forward_one(&gpt_oss_weights(), 0)[[0, 0]];
        let key = gpt_oss_weights().arch.packed_gate_up_bias_key(0).unwrap();

        let biased_at = |offset: usize| {
            let mut w = gpt_oss_weights();
            w.tensors.insert(
                key.clone(),
                mat(EXPERTS, FUSED_HALVES * INTER, |_, c| {
                    if c % FUSED_HALVES == offset {
                        1.0
                    } else {
                        0.0
                    }
                }),
            );
            forward_one(&w, 0)[[0, 0]]
        };

        let gate_out = biased_at(0);
        let up_out = biased_at(1);
        assert!((gate_out - base).abs() > 1e-4, "gate bias had no effect");
        assert!((up_out - base).abs() > 1e-4, "up bias had no effect");
        assert!(
            (gate_out - up_out).abs() > 1e-4,
            "gate and up biases must not be interchangeable ({gate_out} vs {up_out})"
        );
    }

    // ── policy plumbing ─────────────────────────────────────────────────

    /// The fixture must actually exercise the clamped GLU — a suite that
    /// silently took the plain gated path would prove nothing about GPT-OSS.
    #[test]
    fn fixture_uses_the_clamped_glu_policy() {
        let weights = gpt_oss_weights();
        assert!(matches!(
            weights.arch.expert_gate_policy(),
            ExpertGatePolicy::ClampedGlu { limit, alpha }
                if limit == 7.0 && (alpha - 1.702).abs() < 1e-6
        ));
    }

    #[test]
    fn name_is_stable() {
        let weights = larql_models::test_fixtures::make_test_weights();
        assert_eq!(
            ExpertWeightFfn { weights: &weights }.name(),
            "expert-weights"
        );
    }

    #[test]
    fn dense_default_policy_is_plain_gated() {
        let weights = larql_models::test_fixtures::make_test_weights();
        assert_eq!(weights.arch.expert_gate_policy(), ExpertGatePolicy::Gated);
        let _: Activation = weights.arch.activation();
    }

    #[test]
    fn forward_observed_reports_absent_with_the_moe_reason() {
        let weights = gpt_oss_weights();
        let ffn = ExpertWeightFfn { weights: &weights };
        let (out, obs) = ffn.forward_observed(0, &one_hot(1, 0.5));
        assert_eq!(out.shape(), &[1, HIDDEN]);
        assert_eq!(
            obs.absent_reason(),
            Some(REASON_MOE_NO_SINGLE_ACTIVATION),
            "MoE must decline observation, never invent an activation tensor"
        );
    }

    #[test]
    fn output_is_finite_across_a_batch() {
        let weights = gpt_oss_weights();
        let ffn = ExpertWeightFfn { weights: &weights };
        let x = Array2::from_shape_fn((5, HIDDEN), |(r, c)| ((r + c) as f32) * 0.25 - 0.5);
        let out = ffn.forward(0, &x);
        assert_eq!(out.shape(), &[5, HIDDEN]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "no resolvable per-expert MoE weights")]
    fn forward_panics_loudly_on_an_unresolvable_layer() {
        let mut weights = gpt_oss_weights();
        let key = weights.arch.moe_router_key(0).unwrap();
        weights.tensors.remove(&key);
        let _ = forward_one(&weights, 0);
    }
}
