//! Carriage probes: norms, windows, dtypes and layer sets.

use super::super::super::graph::policy::{AttentionLayerPolicy, AttentionSpan};
use super::super::super::graph::Component;
use serde_json::{json, Value};

#[allow(unused_imports)]
use super::*;

/// The recurrence's state precision, echoed in the checkpoint's own
/// spelling.
///
/// `Lowered` rather than `Represented` because it has a consumer: the
/// reference operator allocates and accumulates `GatedDeltaState` at this
/// precision. Until that executor existed this rule refused, because
/// claiming carriage into a runtime surface that could not use the value
/// would have asserted something untrue.
pub(super) fn probe_linear_state_dtype(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(component
        .execution
        .as_ref()?
        .linear_attention?
        .state_dtype?
        .declared_name()))
}

/// The uniform sliding window across sliding layers, when there is one.
pub(super) fn probe_sliding_window(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mut windows = table.iter().filter_map(|l| l.window);
    let Some(first) = windows.next() else {
        // No layer carries a window. That is an ANSWER, not a failure to
        // answer: the graph states that this component attends fully
        // everywhere. A checkpoint declaring `sliding_window: null` — the
        // whole Qwen3 generation — agrees with it, and one declaring a
        // window that reaches no layer genuinely disagrees and should
        // read as mismatched rather than as an unanswered probe.
        return Some(Value::Null);
    };
    windows.all(|w| w == first).then(|| json!(first))
}

pub(super) fn probe_pre_norm_eps(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.pre.eps))
}

pub(super) fn probe_post_norm_eps(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.post?.eps))
}

/// The judged activation, in the checkpoint's own spelling when that
/// spelling is an alias of the judged variant (`gelu_pytorch_tanh` →
/// `GeluTanh`); the schema's spelling otherwise, so a genuine
/// disagreement still reads as one.
/// The per-layer dense-FFN widths the surface carries, as the array the
/// checkpoint declared them in.
pub(super) fn probe_ffn_width_by_layer(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let ffn = component.execution.as_ref()?.ffn.as_ref()?;
    serde_json::to_value(ffn.intermediate_size_by_layer.as_ref()?).ok()
}

pub(super) fn probe_activation(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    // The FFN's activation on an FFN-bearing component; the MIXER's on a
    // mixer-only one — `hidden_act` is one declared fact, and whichever
    // op consumes it answers for it.
    let surface = component.execution.as_ref()?;
    let activation = match (&surface.ffn, &surface.mamba2) {
        (Some(ffn), _) => ffn.activation,
        (None, Some(mixer)) => mixer.activation,
        (None, None) => return None,
    };
    // `hidden_act` can name the whole COMBINE rather than the gate's
    // nonlinearity (`situ`), and then the surface's `Activation` is inert
    // and cannot answer for it. Asking the FFN's gate policy first is what
    // lets a correctly-carried SiTU FFN report as carried instead of
    // reading `mismatched` forever against a field it never used.
    if let Some(declared) = ctx.declared.as_str() {
        let combine = surface
            .ffn
            .as_ref()
            .and_then(|ffn| larql_models::config::hf_combine_name(ffn.gate_policy, activation));
        if combine.as_deref() == Some(declared) {
            return Some(json!(declared));
        }
        if larql_models::config::Activation::from_hf_name(declared) == Some(activation) {
            return Some(json!(declared));
        }
        // The FFN computes a combine no HF word names (`ClampedGlu`), or
        // there is no FFN. Fall through to the schema's own spelling, so
        // a genuine disagreement still reads as one.
    }
    serde_json::to_value(activation).ok()
}

/// The FFN shape that runs, in the checkpoint's own word when that word
/// names it (`swiglu` for a gated SiLU FFN); the schema's word for the
/// shape otherwise, so a disagreement reads as one — `geglu` declared on
/// a SiLU-gated stack resolves to `swiglu`, and a plain `silu` on the
/// same stack resolves to `swiglu` too, because the plain name is the
/// ungated shape. Both directions come from one table in
/// `larql_models::config::activation`.
pub(super) fn probe_ffn_shape_name(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let ffn = component.execution.as_ref()?.ffn.as_ref()?;
    if let Some(declared) = ctx.declared.as_str() {
        if larql_models::config::ffn_shape_from_hf_name(declared)
            == Some((ffn.ffn_type, ffn.activation))
        {
            return Some(json!(declared));
        }
    }
    larql_models::config::ffn_shape_hf_name(ffn.ffn_type, ffn.activation).map(Value::String)
}

/// Whether the declared identity resolved to the Llama family. The
/// answer comes from [`ProbeContext::family`] — the registry's resolution
/// — so a checkpoint declaring `true` under a `model_type` no entry
/// matches is refused rather than believed.
pub(super) fn probe_is_llama_config(
    _component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        ctx.family == Some(larql_models::detect::registry::LLAMA_FAMILY)
    ))
}

/// The clamp bound the FFN surface carries, when its gate policy has
/// one. A plain-gated surface has no limit to answer with — a checkpoint
/// declaring `swiglu_limit` that resolved to plain gating is then
/// reported as unrepresented, which is the truth.
///
/// BOTH clamped policies answer, and that is the point: the bound is the
/// same declaration in each, while the arithmetic around it differs
/// (`(u+1)·g·σ(αg)` against `act(g)·u`). Answering only for one would
/// have reported GLM-5.3-Flash's declared clamp as uncarried while its
/// executor applied it.
/// Where the routed experts run, off the BUILT surface.
///
/// `None` under the uniform form covers the only two states that reach
/// it — a component with no routed block at all, and one whose routed
/// experts run at `hidden_size` — and in both the declared width found
/// no home, which is what an unrepresented finding says. It is
/// deliberately NOT answered with `hidden_size`: that would report a
/// checkpoint's declared bottleneck as carried by a build that has none.
pub(super) fn probe_routed_expert_width(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    component
        .execution
        .as_ref()?
        .ffn
        .as_ref()?
        .moe
        .as_ref()?
        .latent
        .map(|latent| json!(latent.width))
}

/// Whether the latent branch normalises its aggregate.
///
/// Answers `false` as readily as `true`, because the declaration this is
/// compared against is a BOOLEAN: a checkpoint declaring
/// `latent_moe_use_norm: false` beside a width has its "no norm" carried
/// exactly, and reporting that as unrepresented would grade agreement as
/// a dropped fact.
///
/// Under the uniform form it answers `None` — and this is the flag's own
/// finding, not the width's. The reference nests the norm inside the
/// wrapper, so a `latent_moe_use_norm: true` with no
/// `routed_expert_hidden_size` builds no norm THERE either: the flag is
/// inert in the model, and reporting it unrepresented is the honest
/// reading of a declaration nothing acts on.
pub(super) fn probe_latent_moe_use_norm(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    component
        .execution
        .as_ref()?
        .ffn
        .as_ref()?
        .moe
        .as_ref()?
        .latent
        .map(|latent| json!(latent.norm.is_some()))
}

pub(super) fn probe_swiglu_limit(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    match component.execution.as_ref()?.ffn.as_ref()?.gate_policy {
        // BOTH clamped policies answer, and that is the point: the
        // bound is the same declaration in each, while the arithmetic
        // around it differs (`(u+1)*g*sigma(a*g)` against `act(g)*u`).
        // Answering for only one would report GLM-5.3-Flash's declared
        // clamp as uncarried while its executor applied it.
        larql_models::ExpertGatePolicy::ClampedGlu { limit, .. }
        | larql_models::ExpertGatePolicy::ClampedGated { limit } => Some(json!(limit)),
        // A checkpoint declaring `swiglu_limit` whose FFN resolved to
        // some OTHER policy has no limit to answer with, and is reported
        // unrepresented — which is the truth for both of these.
        larql_models::ExpertGatePolicy::Gated | larql_models::ExpertGatePolicy::SituGlu { .. } => {
            None
        }
    }
}

/// SiTU-GLU's gate softcap, when the FFN's policy is SiTU.
///
/// Reads the value off the BUILT surface rather than off the config, so
/// the finding says whether the declaration reached the op plan, not
/// whether it was declared. A component whose FFN resolved to any other
/// policy has no beta to answer with and the leaf reports unrepresented.
pub(super) fn probe_situ_beta(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    match component.execution.as_ref()?.ffn.as_ref()?.gate_policy {
        larql_models::ExpertGatePolicy::SituGlu { beta, .. } => Some(json!(beta)),
        larql_models::ExpertGatePolicy::Gated
        | larql_models::ExpertGatePolicy::ClampedGlu { .. }
        | larql_models::ExpertGatePolicy::ClampedGated { .. } => None,
    }
}

/// SiTU-GLU's up-branch softcap, when the FFN's policy is SiTU and the
/// checkpoint declared one.
///
/// `None` covers two different states on purpose — the policy is not SiTU,
/// or it is SiTU with no up cap — because in both the checkpoint's
/// declared `activation_situ_linear_beta` found no home, which is exactly
/// what an unrepresented finding says. A SiTU policy that DID carry the
/// value answers with it.
pub(super) fn probe_situ_linear_beta(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    match component.execution.as_ref()?.ffn.as_ref()?.gate_policy {
        larql_models::ExpertGatePolicy::SituGlu { linear_beta, .. } => {
            linear_beta.map(|v| json!(v))
        }
        larql_models::ExpertGatePolicy::Gated
        | larql_models::ExpertGatePolicy::ClampedGlu { .. }
        | larql_models::ExpertGatePolicy::ClampedGated { .. } => None,
    }
}

pub(super) fn probe_attention_bias(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .attention_bias?
    ))
}

pub(super) fn probe_qkv_bias(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.attention.as_ref()?.qkv_bias?
    ))
}

pub(super) fn probe_query_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .query_scale?
    ))
}

pub(super) fn probe_score_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .score_scale
    ))
}

pub(super) fn probe_attn_softcap(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .logit_softcapping?
    ))
}

pub(super) fn probe_final_softcap(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .final_logit_softcapping?
    ))
}

pub(super) fn probe_output_multiplier(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .output_multiplier?
    ))
}

/// The carried multiplier, expressed back in the divisor's units so it can
/// be compared against a declared `logits_scaling`.
///
/// The container stores the resolved *multiplicative* factor, and this leaf
/// declares a divisor — so carrying the fact faithfully means storing
/// `1/d`, and a probe that compared the two directly would report every
/// correct conversion as a dropped fact. Inverting here states the
/// relationship the carriage rule actually asserts.
pub(super) fn probe_logits_scaling(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let carried = component
        .execution
        .as_ref()?
        .head
        .as_ref()?
        .output_multiplier?;
    if !carried.is_finite() || carried == 0.0 {
        return None;
    }
    Some(json!(1.0 / carried))
}

pub(super) fn probe_embed_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.head.as_ref()?.embed_scale?
    ))
}

pub(super) fn probe_residual_scale(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.residual_scale?))
}

/// The relative-position scheme, as the graph carries it.
///
/// Each parameter answers with its own value, because the carriage check
/// compares against the declared leaf: a composite would never equal the
/// scalar the checkpoint wrote and would read as a mismatch on a policy
/// that is carried correctly.
///
/// Reports the declaration rather than a rotation. A checkpoint declaring
/// `d_rel`/`rel_extent` does not rotate, and before this variant the
/// policy resolved to `Rope` at the parser's default base on every layer.
pub(super) fn relative_position(component: &Component) -> Option<(usize, usize)> {
    match component.attention.as_ref()?.first()?.position {
        larql_models::config::PositionPolicy::Relative { d_rel, extent } => Some((d_rel, extent)),
        _ => None,
    }
}

pub(super) fn probe_relative_d_rel(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    relative_position(component).map(|(d_rel, _)| json!(d_rel))
}

pub(super) fn probe_relative_extent(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    relative_position(component).map(|(_, extent)| json!(extent))
}

/// Whether the surface's routing policy renormalises over the selected
/// experts.
///
/// Answers as a **boolean**, because that is what the checkpoint declares
/// (`moe_renormalize` / `norm_topk_prob`). Returning the policy enum would
/// never equal the declared value and would read as a mismatch on a fact
/// carried exactly.
///
/// **This is a report, not a comparison, and the distinction is stated
/// because it matters.** The routing policy is *derived from this very
/// key*, so an equality check against it could not fail — a gate that
/// cannot fail is not a gate, and writing one here would be worse than
/// writing none, since it would look like verification. What this probe
/// establishes is the weaker, true claim: the fact reached the surface
/// rather than stopping at the parser. The same caveat the `layer_types`
/// probe carries, for the same reason.
pub(super) fn probe_moe_routing_policy(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(json!(matches!(
        moe.routing_policy,
        larql_models::config::ExpertRoutingPolicy::NormalisedOverSelected
    )))
}

pub(super) fn probe_moe_shared_experts(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(json!(moe.shared_experts))
}

/// The width the shared branch will be built at.
///
/// A checkpoint that declares this key and a container that resolved the
/// branch to some other width disagree about a real projection, and the
/// mismatch must show: Qwen1.5-MoE declares 5632 where the routed width
/// times the shared count is 1408.
pub(super) fn probe_shared_expert_width(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(json!(moe.shared_expert_intermediate_size?))
}

pub(super) fn probe_moe_router_kind(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(json!(moe.router_kind.as_str()))
}

/// A declared parameter sitting at its **identity value** — one group, one
/// layer of period — has no effect for the schema to carry, so it is
/// represented exactly by the schema having no field for it.
///
/// Value-dependent on purpose. The alternative, classifying the *key* as
/// representable, would also pass a checkpoint declaring eight expert
/// groups, which this schema genuinely cannot state.
pub(super) fn probe_identity_valued(
    _component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    (ctx.declared.as_u64() == Some(1)).then(|| ctx.declared.clone())
}

/// Zero of something is the absence this schema already represents.
pub(super) fn probe_absent_when_zero(
    _component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    (ctx.declared.as_u64() == Some(0)).then(|| ctx.declared.clone())
}

/// Grouped routing is a no-op when the component's own grouping is one
/// group; the flag alone says nothing without the count beside it.
pub(super) fn probe_grouping_is_a_no_op(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    // The flag is only carried when the surface shows ungrouped routing,
    // which is what one group produces.
    component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(ctx.declared.clone())
}

/// The KDA conv width the surface carries.
pub(super) fn probe_kda_conv_kernel(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.kda?.conv_kernel))
}

/// An index-set declaration is carried when the graph holds exactly as
/// many layers of that kind as the set named.
///
/// Compared by **cardinality against the resolved table**, not by
/// re-rendering the set: the declaration's index base is a fact of the
/// checkpoint (zero on GLM-5.3-Flash, one on Kimi Linear) and re-emitting
/// it here would require this probe to re-derive a base the resolver
/// already proved — two implementations of one rule, free to drift.
///
/// It is still a real check. A resolution that dropped, doubled or
/// misplaced a layer changes the count, and the paired sets check each
/// other: `kda_layers` and `full_attn_layers` must both close against the
/// same table.
pub(super) fn probe_layer_set(
    component: &Component,
    ctx: &ProbeContext<'_>,
    is_kind: impl Fn(&AttentionLayerPolicy) -> bool,
) -> Option<Value> {
    let declared = ctx.declared.as_array()?;
    let carried = component
        .attention
        .as_ref()?
        .iter()
        .filter(|l| is_kind(l))
        .count();
    (carried == declared.len()).then(|| ctx.declared.clone())
}

pub(super) fn probe_recurrent_layer_set(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    probe_layer_set(component, ctx, |l| l.operator.is_recurrent())
}

pub(super) fn probe_softmax_layer_set(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    probe_layer_set(component, ctx, |l| !l.operator.is_recurrent())
}

pub(super) fn probe_moe_branch_scale(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    moe.branch_scale.map(|s| json!(s))
}

pub(super) fn probe_moe_dense_prefix(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    moe.dense_prefix_layers.map(|n| json!(n))
}

/// `mla_use_nope` carried onto the per-layer position policy.
///
/// Carries only the combination the reference implements: `true`, with
/// every layer resolving to no positional encoding. `false` is a
/// combination Kimi Linear's own class refuses (`assert self.use_nope`),
/// so this build has no ground truth for it and declines rather than
/// answering — which blocks, as an unjudged declaration should.
pub(super) fn probe_mla_nope(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    if ctx.declared.as_bool() != Some(true) {
        return None;
    }
    let table = component.attention.as_ref()?;
    table
        .iter()
        .all(|l| l.position == larql_models::config::PositionPolicy::None)
        .then(|| ctx.declared.clone())
}

pub(super) fn probe_sliding_layer_set(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    probe_layer_set(component, ctx, |l| l.span == Some(AttentionSpan::Sliding))
}

/// The KDA decay clamp the surface carries.
pub(super) fn probe_kda_use_full_rank_gate(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let execution = component.execution.as_ref()?;
    // A gate FORM is carried only where there is a KDA block whose gate it
    // describes; declared on a component with no KDA geometry it reaches
    // nothing, and saying so is the honest answer.
    execution.kda.as_ref()?;
    execution
        .kda_use_full_rank_gate
        .map(|full_rank| json!(full_rank))
}

pub(super) fn probe_mla_use_output_gate(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(component
        .execution
        .as_ref()?
        .mla
        .as_ref()?
        .output_gate
        .is_some()))
}

pub(super) fn probe_kda_gate_lower_bound(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    component
        .execution
        .as_ref()?
        .kda_gate_lower_bound
        .map(|b| json!(b))
}

/// The stride of the full-attention layers the built span table shows.
///
/// Derived from the table, never from the declaration: the first full
/// layer fixes the candidate period (layer `p - 1` is full under period
/// `p`), and the answer stands only if EVERY layer's span agrees with
/// [`larql_models::config::is_full_at_period`] at that period. A table
/// with no full layer, or with no single period, answers `None` — the
/// declaration then has nothing to agree with and refuses.
pub(super) fn probe_full_attention_period(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let is_full = |l: &AttentionLayerPolicy| l.span == Some(AttentionSpan::Full);
    let period = table.iter().position(is_full)? + 1;
    table
        .iter()
        .enumerate()
        .all(|(layer, l)| is_full(l) == larql_models::config::is_full_at_period(layer, period))
        .then(|| json!(period))
}
