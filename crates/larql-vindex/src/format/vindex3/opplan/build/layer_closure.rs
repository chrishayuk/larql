//! Per-layer stack closure: every transformer layer's required operands
//! are present, shaped as its geometry says, and nothing unclassified is
//! left over. Records defects; builds nothing.

use super::super::super::graph::component::Component;
use super::super::super::graph::surface::{AttentionSurface, ExecutionSurface};
use super::super::super::graph::AttentionLayerPolicy;
use super::absent::StackGeometry;

#[allow(unused_imports)]
use super::*;

/// What the per-layer closure check reads.
pub(super) struct LayerClosureInputs<'a> {
    pub(super) surface: &'a ExecutionSurface,
    pub(super) component: &'a Component,
    pub(super) placement: NormPlacement,
    pub(super) hyper_connection: Option<larql_models::config::HyperConnection>,
    pub(super) attention_residual: bool,
    pub(super) attention_table: &'a [AttentionLayerPolicy],
    pub(super) tables: &'a BTreeMap<ObjectKind, (&'a LogicalObject, Vec<SegmentTensor>)>,
    pub(super) attn: Option<&'a AttentionSurface>,
    pub(super) ffn_moe: Option<MoeSurface>,
    pub(super) gated_ffn: bool,
    pub(super) by_layer: &'a BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>>,
    pub(super) bank_by_layer: &'a BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>>,
    pub(super) layer_geometry: &'a dyn Fn(usize) -> StackGeometry,
}

pub(super) fn check_layer_closure(c: LayerClosureInputs<'_>, defects: &mut Vec<ClosureDefect>) {
    let LayerClosureInputs {
        surface,
        component,
        placement,
        hyper_connection,
        attention_residual,
        attention_table,
        tables,
        attn,
        ffn_moe,
        gated_ffn,
        by_layer,
        bank_by_layer,
        layer_geometry,
    } = c;
    if let Some((stack, _)) = tables.get(&ObjectKind::DecoderStack) {
        let bank_id = tables
            .get(&ObjectKind::ExpertBank)
            .map(|(o, _)| o.id.clone())
            .unwrap_or_default();
        for (layer, policy) in attention_table.iter().enumerate() {
            let geometry = layer_geometry(layer);
            let present = by_layer.get(&layer);
            let bank = bank_by_layer.get(&layer);
            // Which layers are routed is DECLARED by the surface: every
            // layer of a MoE surface, less the dense prefix it names
            // (`dense_prefix_layers`, Kimi's 1, GLM-5.3-Flash's 3). The
            // expert bank and router operands are evidence that the
            // declaration is honoured. They never decide: a routed layer
            // whose bank is missing is a defect, not a dense layer, and a
            // dense-prefix layer carrying routed operands is the same
            // disagreement the other way. A surface with no MoE judgment
            // routes nothing; its stray operands are `absent_op`'s to
            // name.
            let evidence_routed = bank.is_some()
                || present.is_some_and(|s| s.contains_key(&OperandRole::MoeRouterWeight));
            let routed = match declared_routed(ffn_moe.as_ref(), layer) {
                Some(true) => {
                    if !evidence_routed {
                        defects.push(ClosureDefect::FfnIdentityMismatch {
                            layer,
                            declared: FfnIdentity::Routed,
                            evidence: FfnIdentity::Dense,
                        });
                    }
                    true
                }
                Some(false) => {
                    if evidence_routed {
                        defects.push(ClosureDefect::FfnIdentityMismatch {
                            layer,
                            declared: FfnIdentity::Dense,
                            evidence: FfnIdentity::Routed,
                        });
                    }
                    false
                }
                None => false,
            };
            // A hybrid layer is routed AND dense: the judgment says the
            // family runs both, and the evidence is the routed evidence.
            let hybrid = routed && ffn_moe.is_some_and(|m| m.hybrid);
            let ops = LayerOps {
                placement,
                gated_ffn,
                // A fused gate ships no operand of its own — demanding
                // one would make every Qwen3.8 layer a closure defect for
                // a tensor that correctly does not exist.
                output_gate: matches!(
                    attn.and_then(|a| a.output_gate).map(|g| g.source),
                    Some(larql_models::config::GateSource::AttentionInput)
                ),
                attention_bias: attn.and_then(|a| a.attention_bias) == Some(true),
                qkv_bias: attn.and_then(|a| a.qkv_bias) == Some(true),
                norm_bias: surface.norm.bias == Some(true),
                ffn_bias: surface.ffn.as_ref().and_then(|f| f.bias) == Some(true),
                sinks: attn.is_some_and(|a| a.sinks.is_some()),
                // Both gates are DECLARED facts (K3-REP-GATE-1): the KDA
                // gate's form and the MLA gate's presence come from the
                // surface, never from which `g_proj` spelling the layer
                // happens to ship — the operands are held to them below.
                kda_full_rank_gate: surface.kda_use_full_rank_gate == Some(true),
                mla_output_gate: surface.mla.is_some_and(|m| m.output_gate.is_some()),
                mla_q_lora: surface.mla.is_some_and(|m| m.query.is_low_rank()),
                routed,
                hybrid,
                moe: ffn_moe,
                mamba2: surface.mamba2,
                conv_qkv: surface.conv_qkv,
                v_from_k: policy.v_from_k,
                hyper_connection: hyper_connection.is_some(),
                attention_residual,
                // Which operand family this layer must supply, taken from
                // the GRAPH's operator. The op below picks its operator
                // from operand EVIDENCE instead, so the two authorities
                // meet here: a layer the graph calls recurrent while its
                // tensors say softmax (or the reverse) fails closure with
                // the missing roles named. That cross-check was recorded
                // as owed at the first real encode in QW-3.5A, and this
                // is where it lands.
                //
                // `LayerOperator::Recurrent` — a declared recurrence with
                // no identified operator — answers `false` here and would
                // therefore be asked for softmax operands. It cannot
                // reach this point: `attention_policy` blocks such a
                // stack, and encode refuses an inadmissible plan. If that
                // ever changes, this is the site that needs a third arm
                // rather than a boolean.
                operator: policy.operator,
            };
            for role in required_roles(&ops) {
                let holder = if role.is_expert_bank() { bank } else { present };
                if holder.is_none_or(|slot| !slot.contains_key(&role)) {
                    defects.push(ClosureDefect::MissingOperand { layer, role });
                }
            }
            // A hyper-connected component's mixer-only or conv-QKV layer
            // is a combination no reference this build has read describes
            // — one sublayer, two sites? none? — so the layer is refused
            // as unjudged rather than planned with no site and a topology
            // that says every layer has two. Nothing observed declares
            // this shape; the arm exists so that if something does, it
            // blocks by name instead of building a plan whose layers
            // disagree with its component.
            if ops.hyper_connection && (ops.operator.is_mamba2() || ops.operator.is_conv_qkv()) {
                defects.push(ClosureDefect::UnjudgedSemantic {
                    component: component.id.clone(),
                    fact: format!("{HC_ON_MIXER_FACT} (layer {layer})"),
                    required_by: HC_ON_MIXER_REQUIRED_BY.to_string(),
                });
            }
            // QK norms travel as a pair.
            if let Some(slot) = present {
                match (
                    slot.contains_key(&OperandRole::AttnQNorm),
                    slot.contains_key(&OperandRole::AttnKNorm),
                ) {
                    (true, false) => defects.push(ClosureDefect::MissingOperand {
                        layer,
                        role: OperandRole::AttnKNorm,
                    }),
                    (false, true) => defects.push(ClosureDefect::MissingOperand {
                        layer,
                        role: OperandRole::AttnQNorm,
                    }),
                    _ => {}
                }
            }
            // A shared branch declared by count alone takes its width from
            // its own stored gate tensor, so up and down are held to it.
            let ffn_moe = ffn_moe.map(|m| {
                resolve_shared_expert_width(
                    m,
                    present.and_then(|slot| slot.get(&OperandRole::SharedExpertGate)),
                )
            });
            let stack_operands = present
                .into_iter()
                .flatten()
                .map(|(r, t)| (r, t, &stack.id));
            let bank_operands = bank.into_iter().flatten().map(|(r, t)| (r, t, &bank_id));
            for (role, tensor, object_id) in stack_operands.chain(bank_operands) {
                // An operand whose op the surface does not carry.
                if let Some(primitive) = absent_op(*role, &ops) {
                    defects.push(ClosureDefect::OperandImpliesAbsentOp {
                        object: object_id.clone(),
                        tensor: tensor.name.clone(),
                        required_primitive: primitive.to_string(),
                    });
                    continue;
                }
                if let Some(expected) = expected_shape(*role, &geometry, ffn_moe.as_ref()) {
                    if !shape_satisfies(&tensor.shape, &expected) {
                        defects.push(ClosureDefect::GeometryMismatch {
                            tensor: format!("{object_id}/{}", tensor.name),
                            expected,
                            actual: tensor.shape.clone(),
                        });
                    }
                }
            }
        }
    }
}
