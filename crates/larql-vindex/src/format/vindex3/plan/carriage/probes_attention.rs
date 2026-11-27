//! Carriage probes: attention, position, gating and scalar facts.

use super::super::super::graph::policy::AttentionLayerPolicy;
use super::super::super::graph::Component;
use serde_json::{json, Value};

#[allow(unused_imports)]
use super::*;

/// A fact the schema represents only as absent: the built component
/// answers `0`, so a declared `0` agrees and anything else blocks.
pub(super) fn probe_zero(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    component.execution.as_ref()?;
    Some(json!(0))
}

/// A switch the schema represents only as off.
pub(super) fn probe_false(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    component.execution.as_ref()?;
    Some(json!(false))
}

/// The rotary fraction the layers in scope carry — `partial_rotary_factor`
/// is a per-layer-type leaf on Gemma 4 (`full_attention` only).
pub(super) fn probe_partial_rotary_factor(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let mut fractions =
        layers_in_scope(component, ctx)?.filter_map(|l| l.position.rotary_fraction());
    let first = fractions.next()?;
    fractions.all(|f| f == first).then(|| json!(first))
}

/// Whether the component carries judged attention-output-gate semantics.
///
/// Answers the DECLARED boolean rather than echoing it: `true` only when
/// a spec was actually judged for this family and reached the surface. A
/// checkpoint declaring `attn_output_gate: false` is answered `false` by
/// a surface with no spec, so the two agree without the probe ever
/// asserting a gate that is not there.
///
/// Note what is NOT claimed here. HF reads this key nowhere — the gate is
/// unconditional in the reference implementation, and its real witness is
/// the stored projection carrying `2 · num_heads · head_dim` rows. That
/// cross-examination happens in operand closure (`expected_shape`'s
/// `q_proj_rows`), which is why the config being believed here is safe:
/// a checkpoint claiming a gate it has no rows for fails there.
pub(super) fn probe_attn_output_gate(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(component
        .execution
        .as_ref()?
        .attention
        .as_ref()?
        .output_gate
        .is_some()))
}

/// The multi-axis sectioning the layers in scope carry, when every
/// rotating layer agrees.
///
/// Refuses unless the arithmetic closes:
///
/// ```text
/// sum(section) * 2 == rotary_dim == head_dim * rotary_fraction
/// ```
///
/// `sum(section)` counts FREQUENCY slots, which is `rotary_dim / 2` — not
/// `rotary_dim`. On Qwen3.8 that is `11+11+10 = 32` against a 64-dim
/// rotary block on a **256**-dim head. Taking the head width as 128 (the
/// Gated DeltaNet head dim, a different operator) makes `sum == rotary_dim`
/// close instead, which is why the identity is asserted against the
/// component's own resolved `head_dim` rather than any nearby 128.
pub(super) fn mrope_of(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<([usize; 3], bool)> {
    let head_dim = component.execution.as_ref()?.attention.as_ref()?.head_dim;
    let mut policies = layers_in_scope(component, ctx)?.filter_map(|l| {
        l.position
            .mrope()
            .zip(l.position.rotary_fraction())
            .map(|((section, interleaved), fraction)| (section, interleaved, fraction))
    });
    let first = policies.next()?;
    if !policies.all(|p| p == first) {
        return None;
    }
    let (section, interleaved, fraction) = first;
    let rotary_dim = (head_dim as f64 * fraction) as usize;
    let closes = rotary_dim > 0
        && rotary_dim.is_multiple_of(2)
        && section.iter().sum::<usize>() * 2 == rotary_dim;
    closes.then_some((section, interleaved))
}

pub(super) fn probe_mrope_section(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    mrope_of(component, ctx).map(|(section, _)| json!(section))
}

pub(super) fn probe_mrope_interleaved(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    mrope_of(component, ctx).map(|(_, interleaved)| json!(interleaved))
}

/// The YaRN block the table carries, when it carries one. `None` when the
/// table has no scaled layer — the caller's leaf then has nothing to be
/// judged against, which is the right answer for a checkpoint that
/// declares the leaf outside a `yarn` block.
pub(super) fn yarn_block(component: &Component) -> Option<larql_models::YarnRopeScaling> {
    component
        .attention
        .as_ref()?
        .iter()
        .find_map(|l| l.position.yarn())
}

/// The Llama-3 block a built layer carries, if any.
pub(super) fn llama3_block(component: &Component) -> Option<larql_models::Llama3RopeScaling> {
    component
        .attention
        .as_ref()?
        .iter()
        .find_map(|l| l.position.llama3())
}

/// `factor` means the same thing in both scaling families — the extension
/// ratio — so it is answered from whichever block the layer carries.
///
/// Asked of both deliberately. Before Llama-3 had a variant this probe
/// read the YaRN block alone, so a checkpoint declaring `rope_type:
/// "llama3"` had its `factor` reported as unanswered and blocked: the
/// carriage rule claimed a home the schema did not yet have. Answering
/// from one family and defaulting the other would have been the worse
/// fix, since the two are alternatives and a wrong `factor` is a wrong
/// long-context model.
pub(super) fn probe_scaling_factor(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let factor = yarn_block(component)
        .map(|y| y.factor)
        .or_else(|| llama3_block(component).map(|l| l.factor))
        .or_else(|| linear_block(component))?;
    Some(json!(factor))
}

/// The linear position divisor a built layer carries, if any layer does.
///
/// Asked of every layer, not the first: on Gemma 3 the declaration
/// reaches the full-attention layers only, and the sliding layers
/// rotate plain — so the first layer of the table answers nothing while
/// the block is carried five layers in.
pub(super) fn linear_block(component: &Component) -> Option<f64> {
    component
        .attention
        .as_ref()?
        .iter()
        .find_map(|l| l.position.linear())
}

pub(super) fn probe_llama3_low_freq(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(llama3_block(component)?.low_freq_factor))
}

pub(super) fn probe_llama3_high_freq(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(llama3_block(component)?.high_freq_factor))
}

pub(super) fn probe_yarn_beta_fast(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(yarn_block(component)?.beta_fast))
}

pub(super) fn probe_yarn_beta_slow(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(yarn_block(component)?.beta_slow))
}

pub(super) fn probe_yarn_truncate(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(yarn_block(component)?.truncate))
}

/// The pre-trained context window, from whichever scaling block declares
/// it. Both families define their bands against it.
pub(super) fn probe_scaling_original_max(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let original = yarn_block(component)
        .map(|y| y.original_max_position_embeddings)
        .or_else(|| llama3_block(component).map(|l| l.original_max_position_embeddings))?;
    Some(json!(original))
}

/// Per-layer span kinds in the checkpoint's own vocabulary, so the
/// comparison is against the declared spelling rather than a rendering
/// this probe invents.
///
/// Refuses (returns `None`) rather than vouching for the interleave when
/// any layer's own [`declared_span`](super::super::super::graph::policy::AttentionLayerPolicy::declared_span)
/// disagrees with what `span` resolved to. `AttentionLayerPolicy::span`
/// is built off a boolean sliding/full split that silently defaults any
/// spelling outside its three-way vocabulary (a hybrid linear-attention
/// layer, e.g.) to `Full` — echoing `span.declared_name()` back in that
/// state would report the declared interleave as carried when the graph
/// actually dropped it. See `docs/k3-funnel.md` §4.7.8.
pub(super) fn probe_layer_types(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    if !table.iter().all(AttentionLayerPolicy::matches_declaration) {
        return None;
    }
    // Every layer round-trips, so rendering the carried policy back into
    // the checkpoint's vocabulary is a report rather than a claim. A
    // layer the schema has no spelling for already refused above.
    table
        .iter()
        .map(|l| l.declared_name().map(|n| json!(n)))
        .collect::<Option<Vec<_>>>()
        .map(Value::Array)
}

/// The Gated DeltaNet geometry the surface carries, read back per field.
///
/// Each answers only if the component actually built a linear-attention
/// block. A component with no recurrence answers `None`, and the gate then
/// reports carriage without a value comparison rather than inventing a
/// disagreement — the same contract every probe here has.
///
/// These are `Lowered` rather than `Represented` because each value
/// terminates in a real operand contract: the five together derive
/// `qkv_channels` and `value_width`, which the nine `LinearAttn*` shape
/// checks close against the stored tensors.
pub(super) fn probe_linear_key_heads(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.key_heads
    ))
}

pub(super) fn probe_linear_key_head_dim(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.key_head_dim
    ))
}

pub(super) fn probe_linear_value_heads(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.value_heads
    ))
}

pub(super) fn probe_linear_value_head_dim(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .linear_attention?
            .value_head_dim
    ))
}

pub(super) fn probe_linear_conv_kernel(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.conv_kernel
    ))
}

/// The Mamba2 surface's geometry, when the component carries one.
pub(super) fn mamba2_geometry(
    component: &Component,
) -> Option<larql_models::config::Mamba2Geometry> {
    Some(component.execution.as_ref()?.mamba2?.geometry)
}

pub(super) fn probe_mamba2_state_size(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.state_size))
}

pub(super) fn probe_mamba2_expand(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.expand))
}

pub(super) fn probe_mamba2_conv_kernel(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.conv_kernel))
}

pub(super) fn probe_mamba2_n_groups(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.n_groups))
}

pub(super) fn probe_mamba2_chunk_size(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.chunk_size))
}

/// Echoes the clamp in the checkpoint's own spelling: a finite side as
/// its number, an unbounded side as the non-finite literal the judged
/// boundary quoted (`-Infinity` below, `Infinity` above) — the inverse of
/// [`DtBound::from_declared`](larql_models::config::DtBound::from_declared),
/// positional because unboundedness below and above are different signs.
pub(super) fn probe_mamba2_time_step_limit(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    use larql_models::config::DtBound;
    let geometry = mamba2_geometry(component)?;
    let side = |bound: DtBound, unbounded: &str| match bound {
        DtBound::Finite(v) => json!(v),
        DtBound::Unbounded => json!(unbounded),
    };
    Some(json!([
        side(geometry.dt_limit_min, "-Infinity"),
        side(geometry.dt_limit_max, "Infinity"),
    ]))
}

pub(super) fn probe_mamba2_rms_norm(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.rms_norm))
}

pub(super) fn probe_mamba2_use_bias(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.use_bias))
}

pub(super) fn probe_mamba2_use_conv_bias(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.use_conv_bias))
}

pub(super) fn probe_mamba2_num_heads(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.num_heads))
}

pub(super) fn probe_mamba2_head_dim(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.head_dim))
}

pub(super) fn conv_qkv_geometry(
    component: &Component,
) -> Option<larql_models::config::ConvQkvAttnGeometry> {
    component.execution.as_ref()?.conv_qkv
}

pub(super) fn probe_conv_qkv_head_dim(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(conv_qkv_geometry(component)?.head_dim))
}

pub(super) fn probe_conv_qkv_conv_kernel(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(conv_qkv_geometry(component)?.conv_kernel))
}

pub(super) fn probe_conv_qkv_rotary_dim(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(conv_qkv_geometry(component)?.rotary_dim))
}

/// A declared-FALSE bias switch is genuinely carried — closure requires
/// no bias operand, and none exists. A declared-TRUE one has no judged
/// operand role yet, so the probe must NOT echo it: answering `None`
/// blocks, which is the fail-closed direction for a bias the plan would
/// silently drop.
pub(super) fn probe_conv_qkv_qkv_bias(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    match conv_qkv_geometry(component)?.qkv_bias {
        false => Some(json!(false)),
        true => None,
    }
}

pub(super) fn probe_conv_qkv_out_bias(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    match conv_qkv_geometry(component)?.out_bias {
        false => Some(json!(false)),
        true => None,
    }
}

/// Echo the declared attention-layer index set only when the component's
/// per-layer table corresponds to it under SOME consistent index base:
/// the same conv-QKV layer count, and every declared index landing on a
/// conv-QKV layer. The base itself was proven upstream from tensor
/// evidence; this re-derivation keeps the carriage claim honest without
/// re-running that proof.
pub(super) fn probe_attention_layer_idx(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let declared: Vec<i64> = ctx
        .declared
        .as_array()?
        .iter()
        .filter_map(Value::as_i64)
        .collect();
    let table = component.attention.as_ref()?;
    let conv_layers: Vec<usize> = table
        .iter()
        .enumerate()
        .filter(|(_, l)| l.operator.is_conv_qkv())
        .map(|(i, _)| i)
        .collect();
    if conv_layers.len() != declared.len() {
        return None;
    }
    for offset in [0i64, 1] {
        let mapped: Vec<i64> = conv_layers.iter().map(|l| *l as i64 + offset).collect();
        if mapped == declared {
            return Some(json!(declared));
        }
    }
    None
}

/// `0` is the one judged declaration: no MLP blocks exist, carried as
/// every layer's absent FFN op — verified against the per-layer table
/// really holding only mixer and conv-QKV operators. A non-zero width
/// has no judged lowering in this lineage yet and must block.
pub(super) fn probe_mlp_intermediate_size(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mixer_lineage_only = !table.is_empty()
        && table
            .iter()
            .all(|l| l.operator.is_mamba2() || l.operator.is_conv_qkv());
    mixer_lineage_only.then(|| json!(0))
}

/// Inert exactly when the MLP itself is declared absent — the same
/// evidence [`probe_mlp_intermediate_size`] answers from. The declared
/// value is echoed because with no MLP anywhere, ANY padding/bias value
/// parameterises nothing a forward pass reads.
pub(super) fn probe_mlp_padding_size(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mixer_lineage_only = !table.is_empty()
        && table
            .iter()
            .all(|l| l.operator.is_mamba2() || l.operator.is_conv_qkv());
    mixer_lineage_only.then(|| ctx.declared.clone())
}

/// `ssm_cfg.layer` — the layer CLASS the package instantiates, which is
/// also its identity declaration. "Mamba2" is represented exactly when
/// the mixer surface exists; any other class name finds nothing here
/// and blocks, which is the fail-closed direction for a lineage this
/// build has not judged.
pub(super) fn probe_ssm_layer_class(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    mamba2_geometry(component).map(|_| json!("Mamba2"))
}

/// `d_conv` — declared by whichever block's config section carries it.
/// The declared width is echoed only when it matches a surface that
/// really holds it (the conv-QKV block's kernel or the mixer's); a
/// width matching neither blocks.
pub(super) fn probe_declared_conv_kernel(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let declared = ctx.declared.as_u64()? as usize;
    let conv_qkv = conv_qkv_geometry(component).map(|g| g.conv_kernel);
    let mamba2 = mamba2_geometry(component).map(|g| g.conv_kernel);
    (Some(declared) == conv_qkv || Some(declared) == mamba2).then(|| json!(declared))
}

/// `attn_cfg.causal` — the operator IS causal by construction, so a
/// declared `true` is carried and a declared `false` finds no operator
/// and blocks.
pub(super) fn probe_attn_causal(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    conv_qkv_geometry(component).map(|_| json!(true))
}

pub(super) fn probe_residual_in_fp32(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.residual_in_fp32?))
}

/// Whether the attention scales QK^T at all (`scale_attn_weights`).
pub(super) fn probe_scales_scores(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let attention = component.execution.as_ref()?.attention.as_ref()?;
    Some(json!(
        attention.score_scale != larql_models::config::UNSCALED_SCORES
    ))
}

/// The learned input position table's declared row count.
pub(super) fn probe_input_position_rows(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let head = component.execution.as_ref()?.head.as_ref()?;
    Some(json!(head.input_position?.max_positions()))
}

/// The built per-layer spans in GPT-Neo's vocabulary (`global` / `local`);
/// `None` when a layer is neither, which then cannot agree with a GPT-Neo
/// declaration.
fn gpt_neo_words(component: &Component) -> Option<Vec<&'static str>> {
    use super::super::super::graph::policy::AttentionSpan;
    use larql_models::config::layer_types::{GPT_NEO_GLOBAL, GPT_NEO_LOCAL};
    component
        .attention
        .as_ref()?
        .iter()
        .map(|l| match l.span {
            Some(AttentionSpan::Full) => Some(GPT_NEO_GLOBAL),
            Some(AttentionSpan::Sliding) => Some(GPT_NEO_LOCAL),
            _ => None,
        })
        .collect()
}

/// `attention_layers`, read back off the built spans.
pub(super) fn probe_gpt_neo_attention_layers(
    component: &Component,
    _ctx: &ProbeContext<'_>,
) -> Option<Value> {
    Some(json!(gpt_neo_words(component)?))
}

/// `attention_types`: its expansion against the built spans. Agreement
/// answers in the declaration's own run-length spelling; disagreement
/// answers with the expansion the graph DID build, so the finding shows
/// both sides.
pub(super) fn probe_gpt_neo_attention_types(
    component: &Component,
    ctx: &ProbeContext<'_>,
) -> Option<Value> {
    let built = gpt_neo_words(component)?;
    let declared = larql_models::config::layer_types::expand_attention_types(ctx.declared);
    match declared {
        Some(words) if words == built => Some(ctx.declared.clone()),
        _ => Some(json!(built)),
    }
}
