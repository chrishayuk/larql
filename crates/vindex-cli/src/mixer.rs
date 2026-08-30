//! The token mixer, as the container declares it.
//!
//! Two facts, kept apart on purpose. **Identity** comes from the
//! graph's `attention[n].operator` — the checkpoint's judgement,
//! written once at encode time and required on read since
//! GRAPH_SCHEMA 6. **Operands** come from the op plan, which binds
//! tensors to the roles that operator computes with.
//!
//! Reading identity off the operands instead is the defect this
//! module exists to prevent. Before it, `mixer_label` matched
//! `LayerAttention::softmax()` and treated every `None` as Gated
//! DeltaNet, so Kimi Linear's `KKKM KKKM …` interleave — 20 KDA
//! layers and 7 MLA — printed as twenty-seven Gated DeltaNet layers,
//! and `describe layer.3.mixer` answered `GATED DELTANET` over an
//! empty operand table rather than refusing. A container that knows
//! exactly what it is was read as something it is not, confidently.
//!
//! So every match here is exhaustive over [`LayerOperator`] with no
//! wildcard arm: a new operator variant is a compile error in this
//! file, never a silent fall-through to whichever label happens to be
//! last.

use larql_vindex::format::vindex3::graph::policy::LayerOperator;
use larql_vindex::format::vindex3::graph::Component;
use larql_vindex::format::vindex3::opplan::{LayerAttention, LayerPlan, OperandRef};

/// The operator this layer's policy declares, or the reason the
/// container cannot say.
///
/// An absent `attention` table is not "softmax by default" — it is a
/// component that never declared a per-layer programme, and saying so
/// is the honest answer.
pub fn declared_operator(component: &Component, n: usize) -> Result<LayerOperator, String> {
    let policies = component.attention.as_ref().ok_or_else(|| {
        format!(
            "component `{}` declares no per-layer attention policy — this container \
             cannot say what layer {n}'s token mixer is",
            component.id
        )
    })?;
    policies
        .get(n)
        .map(|p| p.operator)
        .ok_or_else(|| layer_range(&format!("layer {n}"), policies.len()))
}

/// The refusal for a layer index the plan does not hold.
///
/// Inclusive bounds on purpose: `0..40` is Rust's half-open spelling
/// and reads to everyone else as forty-one layers, which is a wrong
/// answer to the question the reader actually asked.
pub fn layer_range(what: &str, count: usize) -> String {
    match count {
        0 => format!("{what} — the plan holds no layers"),
        1 => format!("{what} — the plan holds layer 0 only"),
        n => format!("{what} — the plan holds layers 0\u{2013}{}", n - 1),
    }
}

/// The programme label for one declared operator.
///
/// `output_gate` refines softmax only, and comes from the bound op:
/// gating is an operand the layer either ships or does not, while
/// every other name here is the operator's own.
///
/// `SOFTMAX ATTENTION` rather than plain `ATTENTION` for the ungated
/// case, because `GATED ATTENTION` is *also* softmax attention — the
/// gate is added to the operator, not substituted for it, and a pair
/// reading `ATTENTION` / `GATED ATTENTION` invites the opposite
/// reading.
pub fn label(operator: LayerOperator, output_gate: bool) -> &'static str {
    match operator {
        LayerOperator::Softmax => {
            if output_gate {
                "GATED ATTENTION"
            } else {
                "SOFTMAX ATTENTION"
            }
        }
        LayerOperator::GatedDelta => "GATED DELTANET",
        LayerOperator::Kda => "KDA",
        LayerOperator::Mla => "MLA",
        LayerOperator::Mamba2 => "MAMBA2",
        // The schema's own word for a declared recurrence whose family
        // nothing identified. Renaming it to any operator would invent
        // the fact the variant exists to withhold.
        LayerOperator::Recurrent => "UNIDENTIFIED RECURRENCE",
    }
}

/// What the plan bound for this layer, named as the operator itself
/// names its operands — or why no such table can be shown.
pub enum MixerOperands {
    Named(Vec<(&'static str, OperandRef)>),
    /// The operator is known; its operand vocabulary is not available
    /// here. Said out loud rather than rendered as an empty table.
    Undescribed(String),
}

/// The mixer's operands under the operator the graph declared.
///
/// The plan's own variant must agree with that declaration. When it
/// does not, the container disagrees with itself and this refuses
/// naming both sides — a defect worth stopping on, never a preference
/// to resolve silently.
pub fn operands(operator: LayerOperator, layer: &LayerPlan) -> MixerOperands {
    let bound = bound_name(&layer.attention);
    match (operator, &layer.attention) {
        (LayerOperator::Softmax, LayerAttention::Softmax(a)) => {
            let mut ops = vec![
                ("query", a.q.clone()),
                ("key", a.k.clone()),
                ("value", a.v.clone()),
                ("output", a.o.clone()),
            ];
            if let Some(g) = &a.output_gate {
                ops.push(("output gate", g.projection.clone()));
            }
            MixerOperands::Named(ops)
        }
        (LayerOperator::GatedDelta, LayerAttention::GatedDelta(g)) => MixerOperands::Named(vec![
            ("fused recurrent q|k|v", g.in_proj_qkv.clone()),
            ("decay projection", g.in_proj_a.clone()),
            ("write-strength projection", g.in_proj_b.clone()),
            ("output-gate projection", g.in_proj_z.clone()),
            ("causal conv over q|k|v", g.conv1d.clone()),
            ("log decay", g.a_log.clone()),
            ("timestep bias", g.dt_bias.clone()),
            ("gated norm", g.norm.clone()),
            ("output projection", g.out_proj.clone()),
        ]),
        // Fifteen roles, split where Gated DeltaNet fuses: three
        // projections through three convs, two low-rank gate pairs.
        (LayerOperator::Kda, LayerAttention::Kda(k)) => MixerOperands::Named(vec![
            ("query projection", k.q_proj.clone()),
            ("key projection", k.k_proj.clone()),
            ("value projection", k.v_proj.clone()),
            ("causal conv over q", k.q_conv1d.clone()),
            ("causal conv over k", k.k_conv1d.clone()),
            ("causal conv over v", k.v_conv1d.clone()),
            ("decay gate down", k.f_a_proj.clone()),
            ("decay gate up", k.f_b_proj.clone()),
            ("output gate down", k.g_a_proj.clone()),
            ("output gate up", k.g_b_proj.clone()),
            ("write-strength projection", k.b_proj.clone()),
            ("log decay", k.a_log.clone()),
            ("timestep bias", k.dt_bias.clone()),
            ("gated norm", k.o_norm.clone()),
            ("output projection", k.out_proj.clone()),
        ]),
        // The compressed-KV set. `query`/`output projection` are
        // byte-identical suffixes to the softmax pair at a different
        // width — named by role here, which is the distinction.
        (LayerOperator::Mla, LayerAttention::Mla(m)) => MixerOperands::Named(vec![
            ("query projection", m.q_proj.clone()),
            ("compressed kv projection", m.kv_a_proj.clone()),
            ("kv latent norm", m.kv_a_norm.clone()),
            ("kv decompression", m.kv_b_proj.clone()),
            ("output projection", m.out_proj.clone()),
        ]),
        (LayerOperator::Mamba2, LayerAttention::Mamba2(m)) => {
            let mut ops = vec![
                ("fused in-projection z|x|B|C|dt", m.in_proj.clone()),
                ("causal conv over x|B|C", m.conv1d.clone()),
            ];
            if let Some(b) = &m.conv1d_bias {
                ops.push(("conv bias", b.clone()));
            }
            ops.push(("log decay", m.a_log.clone()));
            ops.push(("skip", m.d.clone()));
            ops.push(("timestep bias", m.dt_bias.clone()));
            if let Some(n) = &m.gated_norm {
                ops.push(("gated norm", n.weight.clone()));
            }
            ops.push(("output projection", m.out_proj.clone()));
            MixerOperands::Named(ops)
        }
        // Declared, not identified: there is no operand vocabulary to
        // print, and inventing one from whatever the plan bound is the
        // original defect wearing a different hat.
        (LayerOperator::Recurrent, _) => MixerOperands::Undescribed(
            "the container declares a recurrence whose operator family it does not name — \
             no operand vocabulary applies"
                .into(),
        ),
        (declared, _) => MixerOperands::Undescribed(format!(
            "the graph declares {} and the plan bound {bound} — the container disagrees \
             with itself about this layer",
            label(declared, false)
        )),
    }
}

/// The plan's own name for what it bound, for the disagreement message.
fn bound_name(attention: &LayerAttention) -> &'static str {
    match attention {
        LayerAttention::Softmax(_) => "SOFTMAX ATTENTION",
        LayerAttention::GatedDelta(_) => "GATED DELTANET",
        LayerAttention::Kda(_) => "KDA",
        LayerAttention::Mla(_) => "MLA",
        LayerAttention::Mamba2(_) => "MAMBA2",
    }
}

/// Whether this layer's bound op carries an attention output gate.
pub fn has_output_gate(layer: &LayerPlan) -> bool {
    layer
        .attention
        .softmax()
        .is_some_and(|a| a.output_gate.is_some())
}
