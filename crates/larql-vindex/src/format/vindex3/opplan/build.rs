//! Construct the generic operation plan for one component — or refuse
//! with the itemised closure defects.
//!
//! Two passes: closure first (classify every stack tensor, check every
//! implied op has its operands with the right geometry, and every operand
//! an implied op), then plan construction, which runs only when closure
//! holds. Nothing here reads a family name, an HF tensor name, or a layer
//! pattern — arguments come from the surface, the policy table and the
//! roles, or the plan is not built.
//!
//! Scope (5b-1): the decoder-stack text program — embedding, layers,
//! final norm, output head. A `FeatureProjector` object belongs to the
//! cross-component edge program (5e) and a perception component to the
//! perception op set (5d); their closure is deferred with their rungs.

use std::collections::BTreeMap;
use std::path::Path;

use larql_models::config::FfnType;

use super::super::encode::segment::{read_segment_header, SegmentTensor};
use super::super::encode::REPRESENTATION_ID_SEP;
use super::super::graph::roles::classify_stack_tensor;
use super::super::graph::surface::MoeSurface;
use super::super::graph::{LogicalObject, NormPlacement, ObjectKind, OperandRole};
use super::super::inspect::SystemInspection;
use super::{
    AttentionOp, ClosureDefect, ComponentOpPlan, EmbeddingOp, FfnOp, GateOp, LayerFfn, LayerPlan,
    NormOp, OpPlanOutcome, OperandRef, OutputOp, PackedProjection, QkNormOp, RoutedFfnOp, SinkOp,
};
use crate::error::VindexError;
use larql_models::config::ExpertFormat;

/// The post-norm epsilon, named as [`ClosureDefect::UnjudgedSemantic`]
/// reports it.
const POST_NORM_EPS_FACT: &str = "post-norm epsilon";
/// The structure that makes the post-norm epsilon load-bearing.
const FOUR_NORM_PLACEMENT: &str = "four-norm placement";
/// The routed-FFN op, as the requirer of its judged facts.
const ROUTED_FFN_OP: &str = "routed FFN op";
/// Judged elsewhere, not yet expressible as an op here.
const MOE_SHARED_OR_HYBRID_FACT: &str =
    "shared experts / hybrid dense+expert block (no routed-FFN op variant expresses them yet)";
/// A packed fused operand with no declared branch layout cannot be read.
const GATE_UP_LAYOUT_FACT: &str = "gate_up branch layout";

/// Build the operation plan for `component_id` from a container's
/// inspection plus its segment tables. I/O failures are hard errors;
/// every semantic shortfall is a [`ClosureDefect`].
pub fn plan_component_ops(
    inspection: &SystemInspection,
    root: &Path,
    component_id: &str,
) -> Result<OpPlanOutcome, VindexError> {
    let graph = &inspection.graph;
    let Some(component) = graph.components.iter().find(|c| c.id == component_id) else {
        return Err(VindexError::Parse(format!(
            "no component `{component_id}` in the container's graph"
        )));
    };
    let mut defects: Vec<ClosureDefect> = Vec::new();

    let surface = match &component.execution {
        Some(surface) if surface.norm.placement.is_some() => surface,
        _ => {
            return Ok(OpPlanOutcome {
                plan: None,
                defects: vec![ClosureDefect::MissingSurface {
                    component: component.id.clone(),
                }],
            })
        }
    };
    let placement = surface.norm.placement.expect("checked above");
    // A four-norm stack executes two norms whose epsilon nothing else
    // supplies. `Shared` and a declared value are both judgments;
    // absence is not — and inheriting `eps` here would build exactly the
    // executable-but-unfounded program this refuses. Returning no plan
    // means no unjudged epsilon is ever written into one.
    let post_norm: Option<larql_models::config::NormSpec> = match placement {
        NormPlacement::PreOnly => None,
        NormPlacement::PrePost => match surface.norm.post {
            Some(judged) => Some(judged),
            None => {
                return Ok(OpPlanOutcome {
                    plan: None,
                    defects: vec![ClosureDefect::UnjudgedSemantic {
                        component: component.id.clone(),
                        fact: POST_NORM_EPS_FACT.to_string(),
                        required_by: FOUR_NORM_PLACEMENT.to_string(),
                    }],
                })
            }
        },
    };
    let Some(attention_table) = component
        .attention
        .as_ref()
        .filter(|t| t.len() == component.num_layers)
    else {
        return Ok(OpPlanOutcome {
            plan: None,
            defects: vec![ClosureDefect::MissingAttentionTable {
                component: component.id.clone(),
            }],
        });
    };

    let objects: Vec<&LogicalObject> = graph
        .objects
        .iter()
        .filter(|o| o.component == component.id)
        .collect();
    let mut tables: BTreeMap<ObjectKind, (&LogicalObject, Vec<SegmentTensor>)> = BTreeMap::new();
    for object in &objects {
        if matches!(
            object.kind,
            ObjectKind::DecoderStack
                | ObjectKind::ExpertBank
                | ObjectKind::Embedding
                | ObjectKind::FinalNorm
                | ObjectKind::OutputHead
        ) {
            tables.insert(
                object.kind,
                (object, object_tensors(inspection, root, object)?),
            );
        }
    }

    // ── Stack closure ──
    let hidden = component.hidden_size;
    let attn = &surface.attention;
    let q_rows = attn.num_q_heads * attn.head_dim;
    let kv_rows = attn.num_kv_heads * attn.head_dim;
    let inter = surface.ffn.intermediate_size;
    let gated_ffn = surface.ffn.ffn_type == FfnType::Gated;
    let geometry = StackGeometry {
        hidden,
        q_rows,
        kv_rows,
        intermediate: inter,
        head_dim: attn.head_dim,
        num_q_heads: attn.num_q_heads,
        qk_scope: attn.qk_norm_scope,
    };

    // Judged routed-FFN semantics the plan can express today: pure routed
    // experts with a declared fused-operand layout. Shared experts and a
    // hybrid dense+expert block are judged for other families but have no
    // op here yet — refuse, never drop.
    if let Some(moe) = &surface.ffn.moe {
        if moe.hybrid || moe.shared_experts > 0 {
            defects.push(ClosureDefect::UnjudgedSemantic {
                component: component.id.clone(),
                fact: MOE_SHARED_OR_HYBRID_FACT.to_string(),
                required_by: ROUTED_FFN_OP.to_string(),
            });
        }
        if moe.gate_up_layout.is_none() {
            defects.push(ClosureDefect::UnjudgedSemantic {
                component: component.id.clone(),
                fact: GATE_UP_LAYOUT_FACT.to_string(),
                required_by: ROUTED_FFN_OP.to_string(),
            });
        }
    }

    // Stack operands by layer, and expert-bank operands by layer — two
    // objects, one role vocabulary, one classifier.
    let mut by_layer: BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>> = BTreeMap::new();
    let mut bank_by_layer: BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>> = BTreeMap::new();
    for kind in [ObjectKind::DecoderStack, ObjectKind::ExpertBank] {
        let Some((object, tensors)) = tables.get(&kind) else {
            continue;
        };
        for tensor in tensors {
            match classify_stack_tensor(&tensor.name) {
                None => defects.push(ClosureDefect::UnclassifiedOperand {
                    object: object.id.clone(),
                    tensor: tensor.name.clone(),
                }),
                // Expert operands belong in the bank and only there; a
                // router or any dense operand belongs in the stack.
                Some((_, role)) if role.is_expert_bank() != (kind == ObjectKind::ExpertBank) => {
                    defects.push(ClosureDefect::MisplacedOperand {
                        object: object.id.clone(),
                        tensor: tensor.name.clone(),
                        belongs_in: if role.is_expert_bank() {
                            ObjectKind::ExpertBank
                        } else {
                            ObjectKind::DecoderStack
                        },
                    })
                }
                Some((layer, role)) => {
                    let table = if kind == ObjectKind::ExpertBank {
                        &mut bank_by_layer
                    } else {
                        &mut by_layer
                    };
                    let slot = table.entry(layer).or_default();
                    if slot.insert(role, tensor.clone()).is_some() {
                        defects.push(ClosureDefect::DuplicateOperand { layer, role });
                    }
                }
            }
        }
    }

    if let Some((stack, _)) = tables.get(&ObjectKind::DecoderStack) {
        let bank_id = tables
            .get(&ObjectKind::ExpertBank)
            .map(|(o, _)| o.id.clone())
            .unwrap_or_default();
        for layer in 0..component.num_layers {
            let present = by_layer.get(&layer);
            let bank = bank_by_layer.get(&layer);
            // A layer is routed by operand evidence — it has an expert
            // bank or a router — under the surface's judgment; the
            // judgment alone routes nothing, the evidence alone is a
            // stray operand (`absent_op` names it).
            let routed = surface.ffn.moe.is_some()
                && (bank.is_some()
                    || present.is_some_and(|s| s.contains_key(&OperandRole::MoeRouterWeight)));
            let ops = LayerOps {
                placement,
                gated_ffn,
                output_gate: attn.output_gate.is_some(),
                attention_bias: attn.attention_bias == Some(true),
                sinks: attn.sinks.is_some(),
                routed,
                moe: surface.ffn.moe,
            };
            for role in required_roles(&ops) {
                let holder = if role.is_expert_bank() { bank } else { present };
                if holder.is_none_or(|slot| !slot.contains_key(&role)) {
                    defects.push(ClosureDefect::MissingOperand { layer, role });
                }
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
                if let Some(expected) = expected_shape(*role, &geometry, surface.ffn.moe.as_ref()) {
                    if tensor.shape != expected {
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

    // ── Single-tensor objects ──
    let single = |kind: ObjectKind,
                  expected: Option<Vec<usize>>,
                  defects: &mut Vec<ClosureDefect>|
     -> Option<(String, SegmentTensor)> {
        let (object, tensors) = tables.get(&kind)?;
        if tensors.len() != 1 {
            defects.push(ClosureDefect::ObjectShape {
                object: object.id.clone(),
                detail: format!("expected exactly 1 tensor, found {}", tensors.len()),
            });
            return None;
        }
        let tensor = tensors[0].clone();
        if let Some(expected) = expected {
            if tensor.shape != expected {
                defects.push(ClosureDefect::GeometryMismatch {
                    tensor: format!("{}/{}", object.id, tensor.name),
                    expected,
                    actual: tensor.shape.clone(),
                });
            }
        }
        Some((object.id.clone(), tensor))
    };

    let vocab = surface.head.as_ref().map(|h| h.vocab_size);
    let embedding_tensor = single(
        ObjectKind::Embedding,
        vocab.map(|v| vec![v, hidden]),
        &mut defects,
    );
    let final_norm_tensor = single(ObjectKind::FinalNorm, Some(vec![hidden]), &mut defects);
    let head_tensor = single(
        ObjectKind::OutputHead,
        vocab.map(|v| vec![v, hidden]),
        &mut defects,
    );
    if (embedding_tensor.is_some() || head_tensor.is_some()) && surface.head.is_none() {
        defects.push(ClosureDefect::MissingSurface {
            component: component.id.clone(),
        });
    }

    if !defects.is_empty() {
        return Ok(OpPlanOutcome {
            plan: None,
            defects,
        });
    }

    // ── Plan construction (closure holds; lookups are now total) ──
    let operand = |object: &str, tensor: &SegmentTensor| OperandRef {
        object: object.to_string(),
        tensor: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
    };
    // The spec travels whole: kind, epsilon and weight offset all come
    // from the site being built, never from a model-scope answer.
    let norm_op =
        |spec: larql_models::config::NormSpec, object: &str, tensor: &SegmentTensor| NormOp {
            kind: spec.kind,
            eps: spec.eps,
            weight_offset: spec.weight_offset,
            weight: operand(object, tensor),
        };

    let stack_id = tables
        .get(&ObjectKind::DecoderStack)
        .map(|(o, _)| o.id.clone())
        .unwrap_or_default();
    let mut layers = Vec::with_capacity(component.num_layers);
    for layer in 0..component.num_layers {
        let slot = &by_layer[&layer];
        let get = |role: OperandRole| &slot[&role];
        let policy = &attention_table[layer];
        let bias = |role: OperandRole| {
            (attn.attention_bias == Some(true)).then(|| operand(&stack_id, get(role)))
        };
        let qk_norm = slot
            .contains_key(&OperandRole::AttnQNorm)
            .then(|| QkNormOp {
                scope: attn.qk_norm_scope,
                weight_offset: attn.qk_norm_weight_offset,
                q: operand(&stack_id, get(OperandRole::AttnQNorm)),
                k: operand(&stack_id, get(OperandRole::AttnKNorm)),
            });
        // Placement decides which operand feeds the pre-FFN norm: the
        // dedicated one under four-norm, the overloaded
        // `post_attention_layernorm` under two-norm.
        let (post_attention_norm, pre_ffn_role, post_ffn_norm) = match placement {
            NormPlacement::PrePost => {
                let spec = post_norm.expect("PrePost resolves or returns above");
                (
                    Some(norm_op(
                        spec,
                        &stack_id,
                        get(OperandRole::PostAttentionNorm),
                    )),
                    OperandRole::PreFfnNorm,
                    Some(norm_op(spec, &stack_id, get(OperandRole::PostFfnNorm))),
                )
            }
            NormPlacement::PreOnly => (None, OperandRole::PostAttentionNorm, None),
        };
        let bank_slot = bank_by_layer.get(&layer);
        let bank_id = tables
            .get(&ObjectKind::ExpertBank)
            .map(|(o, _)| o.id.clone())
            .unwrap_or_default();
        let ffn = match (surface.ffn.moe, bank_slot) {
            (Some(moe), Some(bank)) => {
                let bank_operand = |role: OperandRole| operand(&bank_id, &bank[&role]);
                let optional = |role: OperandRole| bank.get(&role).map(|t| operand(&bank_id, t));
                LayerFfn::Routed(Box::new(RoutedFfnOp {
                    experts: moe.experts,
                    top_k: moe.top_k,
                    expert_intermediate_size: moe.expert_intermediate_size,
                    router_kind: moe.router_kind,
                    routing_policy: moe.routing_policy,
                    activation: surface.ffn.activation,
                    gate_policy: surface.ffn.gate_policy,
                    expert_format: moe.expert_format,
                    gate_up_layout: moe.gate_up_layout,
                    router: operand(&stack_id, get(OperandRole::MoeRouterWeight)),
                    router_bias: moe
                        .router_bias
                        .then(|| operand(&stack_id, get(OperandRole::MoeRouterBias))),
                    gate_up: PackedProjection {
                        weights: bank_operand(OperandRole::ExpertGateUp),
                        scales: optional(OperandRole::ExpertGateUpScales),
                        bias: optional(OperandRole::ExpertGateUpBias),
                    },
                    down: PackedProjection {
                        weights: bank_operand(OperandRole::ExpertDown),
                        scales: optional(OperandRole::ExpertDownScales),
                        bias: optional(OperandRole::ExpertDownBias),
                    },
                }))
            }
            _ => LayerFfn::Dense(Box::new(FfnOp {
                intermediate_size: inter,
                activation: surface.ffn.activation,
                gate_policy: surface.ffn.gate_policy,
                gate: gated_ffn.then(|| operand(&stack_id, get(OperandRole::FfnGate))),
                up: operand(&stack_id, get(OperandRole::FfnUp)),
                down: operand(&stack_id, get(OperandRole::FfnDown)),
            })),
        };
        let consumed = slot.len() + bank_slot.map_or(0, |b| b.len());
        layers.push(LayerPlan {
            layer,
            pre_attention_norm: norm_op(
                surface.norm.pre,
                &stack_id,
                get(OperandRole::PreAttentionNorm),
            ),
            attention: AttentionOp {
                num_q_heads: attn.num_q_heads,
                num_kv_heads: attn.num_kv_heads,
                head_dim: attn.head_dim,
                query_scale: attn.query_scale,
                score_scale: attn.score_scale,
                logit_softcapping: attn.logit_softcapping,
                span: policy.span,
                window: policy.window,
                position: policy.position,
                qk_norm,
                parameter_free_qk_norm: attn.parameter_free_qk_norm,
                q: operand(&stack_id, get(OperandRole::AttnQ)),
                k: operand(&stack_id, get(OperandRole::AttnK)),
                v: operand(&stack_id, get(OperandRole::AttnV)),
                o: operand(&stack_id, get(OperandRole::AttnO)),
                output_gate: attn.output_gate.map(|spec| GateOp {
                    spec,
                    projection: operand(&stack_id, get(OperandRole::AttnOutputGate)),
                }),
                // Closure held, so `Some(true)` means all four are here
                // and anything else means none is.
                q_bias: bias(OperandRole::AttnQBias),
                k_bias: bias(OperandRole::AttnKBias),
                v_bias: bias(OperandRole::AttnVBias),
                o_bias: bias(OperandRole::AttnOBias),
                sinks: attn.sinks.map(|spec| SinkOp {
                    spec,
                    logits: operand(&stack_id, get(OperandRole::AttnSinks)),
                }),
            },
            post_attention_norm,
            pre_ffn_norm: norm_op(surface.norm.pre, &stack_id, get(pre_ffn_role)),
            ffn,
            post_ffn_norm,
            operands_accounted: consumed,
            operands_present: consumed,
        });
    }

    let plan = ComponentOpPlan {
        component: component.id.clone(),
        embedding: embedding_tensor.map(|(object, tensor)| EmbeddingOp {
            table: operand(&object, &tensor),
            norm: surface.head.as_ref().and_then(|h| h.embedding_norm),
            scale: surface.head.as_ref().and_then(|h| h.embed_scale),
            vocab_size: vocab.unwrap_or(0),
        }),
        layers,
        final_norm: final_norm_tensor
            .map(|(object, tensor)| norm_op(surface.norm.final_norm, &object, &tensor)),
        output: head_tensor.map(|(object, tensor)| OutputOp {
            projection: operand(&object, &tensor),
            multiplier: surface.head.as_ref().and_then(|h| h.output_multiplier),
            softcapping: surface
                .head
                .as_ref()
                .and_then(|h| h.final_logit_softcapping),
        }),
    };
    Ok(OpPlanOutcome {
        plan: Some(plan),
        defects,
    })
}

/// Tensor table of one object's canonical segment.
fn object_tensors(
    inspection: &SystemInspection,
    root: &Path,
    object: &LogicalObject,
) -> Result<Vec<SegmentTensor>, VindexError> {
    let Some(representation) = object.representations.first() else {
        return Err(VindexError::Parse(format!(
            "object `{}` carries no representation",
            object.id
        )));
    };
    let id = format!(
        "{}{REPRESENTATION_ID_SEP}{}",
        object.id, representation.encoding
    );
    let entry = inspection.index.representations.get(&id).ok_or_else(|| {
        VindexError::Parse(format!("no directory entry for representation `{id}`"))
    })?;
    let (header, _) = read_segment_header(&root.join(&entry.segment))?;
    Ok(header.tensors)
}

/// The ops one layer's surface declares — what decides which operands
/// it must have and which it may not.
struct LayerOps {
    placement: NormPlacement,
    gated_ffn: bool,
    output_gate: bool,
    attention_bias: bool,
    sinks: bool,
    /// This layer's FFN is routed (bank/router evidence under a MoE
    /// judgment); dense otherwise.
    routed: bool,
    moe: Option<MoeSurface>,
}

/// Roles every layer must supply, given the surface's ops.
fn required_roles(ops: &LayerOps) -> Vec<OperandRole> {
    let mut roles = vec![
        OperandRole::PreAttentionNorm,
        OperandRole::PostAttentionNorm,
        OperandRole::AttnQ,
        OperandRole::AttnK,
        OperandRole::AttnV,
        OperandRole::AttnO,
    ];
    if ops.placement == NormPlacement::PrePost {
        roles.push(OperandRole::PreFfnNorm);
        roles.push(OperandRole::PostFfnNorm);
    }
    if ops.output_gate {
        roles.push(OperandRole::AttnOutputGate);
    }
    if ops.attention_bias {
        roles.extend([
            OperandRole::AttnQBias,
            OperandRole::AttnKBias,
            OperandRole::AttnVBias,
            OperandRole::AttnOBias,
        ]);
    }
    if ops.sinks {
        roles.push(OperandRole::AttnSinks);
    }
    match (ops.routed, ops.moe) {
        (true, Some(moe)) => {
            roles.extend([
                OperandRole::MoeRouterWeight,
                OperandRole::ExpertGateUp,
                OperandRole::ExpertDown,
            ]);
            if moe.router_bias {
                roles.push(OperandRole::MoeRouterBias);
            }
            if moe.expert_format.has_split_scale_streams() {
                roles.push(OperandRole::ExpertGateUpScales);
                roles.push(OperandRole::ExpertDownScales);
            }
        }
        _ => {
            roles.push(OperandRole::FfnUp);
            roles.push(OperandRole::FfnDown);
            if ops.gated_ffn {
                roles.push(OperandRole::FfnGate);
            }
        }
    }
    roles
}

/// The primitive a found operand requires when the surface does not carry
/// its op. `None` when the operand is consumed by a declared op.
fn absent_op(role: OperandRole, ops: &LayerOps) -> Option<&'static str> {
    match role {
        OperandRole::AttnOutputGate if !ops.output_gate => {
            Some("attention output gate (judged semantics)")
        }
        OperandRole::AttnQBias
        | OperandRole::AttnKBias
        | OperandRole::AttnVBias
        | OperandRole::AttnOBias
            if !ops.attention_bias =>
        {
            Some("attention projection bias (declared `attention_bias`)")
        }
        OperandRole::AttnSinks if !ops.sinks => Some("attention sinks (judged semantics)"),
        OperandRole::FfnGate if !ops.routed && !ops.gated_ffn => Some("gated FFN"),
        OperandRole::FfnGate | OperandRole::FfnUp | OperandRole::FfnDown if ops.routed => {
            Some("dense FFN (this layer is routed)")
        }
        OperandRole::MoeRouterWeight
        | OperandRole::MoeRouterBias
        | OperandRole::ExpertGateUp
        | OperandRole::ExpertGateUpScales
        | OperandRole::ExpertGateUpBias
        | OperandRole::ExpertDown
        | OperandRole::ExpertDownScales
        | OperandRole::ExpertDownBias
            if !ops.routed =>
        {
            Some("routed FFN (judged semantics)")
        }
        OperandRole::MoeRouterBias if ops.moe.is_some_and(|m| !m.router_bias) => {
            Some("router bias (declared by the routed-FFN judgment)")
        }
        OperandRole::ExpertGateUpScales | OperandRole::ExpertDownScales
            if ops
                .moe
                .is_some_and(|m| !m.expert_format.has_split_scale_streams()) =>
        {
            Some("a scaled expert format (this format carries no separate scales)")
        }
        OperandRole::PreFfnNorm | OperandRole::PostFfnNorm
            if ops.placement == NormPlacement::PreOnly =>
        {
            Some("four-norm placement")
        }
        _ => None,
    }
}

/// The surface geometry every stack operand's shape is checked against.
struct StackGeometry {
    hidden: usize,
    q_rows: usize,
    kv_rows: usize,
    intermediate: usize,
    head_dim: usize,
    num_q_heads: usize,
    qk_scope: larql_models::config::QkNormScope,
}

/// Expected stored shape per role, from the surface's geometry. `None`
/// for roles whose shape contract is not yet pinned.
fn expected_shape(
    role: OperandRole,
    g: &StackGeometry,
    moe: Option<&MoeSurface>,
) -> Option<Vec<usize>> {
    use larql_models::config::QkNormScope;
    let StackGeometry {
        hidden,
        q_rows,
        kv_rows,
        intermediate,
        head_dim,
        num_q_heads,
        qk_scope,
    } = *g;
    match role {
        OperandRole::AttnQ => Some(vec![q_rows, hidden]),
        OperandRole::AttnK | OperandRole::AttnV => Some(vec![kv_rows, hidden]),
        OperandRole::AttnO => Some(vec![hidden, q_rows]),
        OperandRole::PreAttentionNorm
        | OperandRole::PostAttentionNorm
        | OperandRole::PreFfnNorm
        | OperandRole::PostFfnNorm => Some(vec![hidden]),
        OperandRole::AttnQNorm | OperandRole::AttnKNorm => match qk_scope {
            QkNormScope::PerHead => Some(vec![head_dim]),
            // Full-projection shape contract unpinned until a real
            // instance is judged.
            QkNormScope::FullProjection => None,
        },
        OperandRole::FfnGate | OperandRole::FfnUp => Some(vec![intermediate, hidden]),
        OperandRole::FfnDown => Some(vec![hidden, intermediate]),
        // Linear(hidden -> q_heads*head_dim), per the judged spec.
        OperandRole::AttnOutputGate => Some(vec![q_rows, hidden]),
        // A bias is one value per output row of its projection.
        OperandRole::AttnQBias => Some(vec![q_rows]),
        OperandRole::AttnKBias | OperandRole::AttnVBias => Some(vec![kv_rows]),
        OperandRole::AttnOBias => Some(vec![hidden]),
        // One logit per query head, per the judged spec.
        OperandRole::AttnSinks => Some(vec![num_q_heads]),
        // Routed FFN: every shape follows from the judgment's expert count,
        // width and storage format; with no judgment there is no contract
        // (the operand is refused by `absent_op` before this is asked).
        OperandRole::MoeRouterWeight => Some(vec![moe?.experts, hidden]),
        OperandRole::MoeRouterBias => Some(vec![moe?.experts]),
        OperandRole::ExpertGateUp => {
            let m = moe?;
            Some(packed_shape(
                m,
                FUSED_BRANCHES * m.expert_intermediate_size,
                hidden,
            ))
        }
        OperandRole::ExpertGateUpScales => {
            let m = moe?;
            Some(scales_shape(
                m,
                FUSED_BRANCHES * m.expert_intermediate_size,
                hidden,
            ))
        }
        OperandRole::ExpertGateUpBias => {
            let m = moe?;
            Some(vec![m.experts, FUSED_BRANCHES * m.expert_intermediate_size])
        }
        OperandRole::ExpertDown => {
            let m = moe?;
            Some(packed_shape(m, hidden, m.expert_intermediate_size))
        }
        OperandRole::ExpertDownScales => {
            let m = moe?;
            Some(scales_shape(m, hidden, m.expert_intermediate_size))
        }
        OperandRole::ExpertDownBias => Some(vec![moe?.experts, hidden]),
    }
}

/// Gate and up: the two branches sharing one fused operand.
const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// Stored shape of a packed `[experts, rows, k]` projection under the
/// judged format: MXFP4 packs `k` as `k/32` groups of 16 bytes (32
/// nibbles); an unquantised packed store keeps `[experts, rows, k]`.
fn packed_shape(moe: &MoeSurface, rows: usize, k: usize) -> Vec<usize> {
    use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};
    match moe.expert_format {
        ExpertFormat::PackedMxfp4 => {
            vec![moe.experts, rows, k / MXFP4_GROUP_ELEMS, MXFP4_GROUP_BYTES]
        }
        ExpertFormat::PackedBF16 | ExpertFormat::PerExpert => vec![moe.experts, rows, k],
    }
}

/// Stored shape of the companion scales stream: one E8M0 byte per group.
fn scales_shape(moe: &MoeSurface, rows: usize, k: usize) -> Vec<usize> {
    use larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
    match moe.expert_format {
        ExpertFormat::PackedMxfp4 => vec![moe.experts, rows, k / MXFP4_GROUP_ELEMS],
        ExpertFormat::PackedBF16 | ExpertFormat::PerExpert => vec![moe.experts, rows],
    }
}
