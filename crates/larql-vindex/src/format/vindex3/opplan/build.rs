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

use larql_models::config::{FfnType, MoeRouterKind};

use super::super::encode::segment::{read_segment_header, SegmentTensor};
use super::super::encode::REPRESENTATION_ID_SEP;
use super::super::graph::policy::LayerOperator;
use super::super::graph::roles::classify_stack_tensor_on;
use super::super::graph::surface::LinearAttentionSurface;
use super::super::graph::surface::Mamba2Surface;
use super::super::graph::surface::MlaSurface;
use super::super::graph::surface::MoeSurface;
use super::super::graph::{LogicalObject, NormPlacement, ObjectKind, OperandRole};
use super::super::inspect::SystemInspection;
use super::{
    AttentionOp, ClosureDefect, ComponentOpPlan, EmbeddingOp, ExpertBank, FfnOp, GateOp,
    GatedDeltaOp, HybridFfnOp, KdaOp, LayerAttention, LayerFfn, LayerPlan, Mamba2Op, MlaOp, NormOp,
    OpPlanOutcome, OperandRef, OutputOp, PackedProjection, QkNormOp, RoutedFfnOp, SharedExpertOp,
    SinkOp,
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
        NormPlacement::PreOnly | NormPlacement::PreMixer => None,
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
    // Schema 6: the operation surfaces follow the program, so each is
    // optional and each family that runs must find its group present.
    // The cross-check lives here as well as in `execution_completeness`
    // because closure is the proof boundary encode gates on.
    let attn = surface.attention.as_ref();
    let ffn_surface = surface.ffn.as_ref();
    let attends = attention_table
        .iter()
        .any(|l| matches!(l.operator, LayerOperator::Softmax | LayerOperator::Mla));
    let has_non_mixer = attention_table.iter().any(|l| !l.operator.is_mamba2());
    let runs_mamba2 = attention_table.iter().any(|l| l.operator.is_mamba2());
    for (runs, present, fact) in [
        (attends, attn.is_some(), "attention surface"),
        (has_non_mixer, ffn_surface.is_some(), "ffn surface"),
        (
            runs_mamba2,
            surface.mamba2.is_some(),
            "mamba2 mixer surface",
        ),
    ] {
        if runs && !present {
            defects.push(ClosureDefect::UnjudgedSemantic {
                component: component.id.clone(),
                fact: fact.to_string(),
                required_by: "the declared operation program".to_string(),
            });
        }
    }
    let inter = ffn_surface.map_or(0, |f| f.intermediate_size);
    let gated_ffn = ffn_surface.is_some_and(|f| f.ffn_type == FfnType::Gated);
    let ffn_moe = ffn_surface.and_then(|f| f.moe);
    // Head geometry is a per-layer fact when the family varies it
    // (Gemma 4's global layers); the layer's policy is the authority and
    // the surface is what a pre-geometry container meant by "every
    // layer". Zeros when the component does not attend: only attention
    // roles consult these, and none is required on a mixer-only stack.
    let layer_geometry = |layer: usize| {
        let (head_dim, num_kv_heads) = attention_table[layer].geometry.map_or_else(
            || attn.map_or((0, 0), |a| (a.head_dim, a.num_kv_heads)),
            |g| (g.head_dim, g.num_kv_heads),
        );
        let num_q_heads = attn.map_or(0, |a| a.num_q_heads);
        StackGeometry {
            hidden,
            q_rows: num_q_heads * head_dim,
            // The independent witness for the gate. The config says
            // `attn_output_gate: true`; the stored projection says
            // `2 · 24 · 256 = 12288` against an ungated 6144, and this
            // contract is what makes the two cross-examine each other
            // instead of the config being believed on its own.
            q_proj_rows: num_q_heads
                * head_dim
                * if matches!(
                    attn.and_then(|a| a.output_gate).map(|g| g.source),
                    Some(larql_models::config::GateSource::FusedQueryProjection)
                ) {
                    2
                } else {
                    1
                },
            kv_rows: num_kv_heads * head_dim,
            intermediate: inter,
            head_dim,
            num_q_heads,
            num_kv_heads,
            qk_scope: attn.map_or(larql_models::config::QkNormScope::PerHead, |a| {
                a.qk_norm_scope
            }),
            linear: surface.linear_attention,
            kda: surface.kda,
            mla: surface.mla,
            mamba2: surface.mamba2,
        }
    };

    // Judged routed-FFN semantics the plan can express today: pure routed
    // experts, Gemma 4's hybrid dense+routed block, or a shared-expert
    // branch beside the routed one (`SharedExpertOp`) — with a declared
    // fused-operand layout wherever the format actually fuses one.
    // `ExpertFormat::PerExpert` never fuses gate and up into one operand
    // (each expert's `w1`/`w3` are already separate tensors), so no layout
    // exists to declare and none is required.
    if let Some(moe) = &ffn_moe {
        if moe.gate_up_layout.is_none() && moe.expert_format != ExpertFormat::PerExpert {
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
            // Layer-aware, and it must be: on a hybrid checkpoint the
            // suffix `self_attn.o_proj.weight` names the recurrence's
            // output projection on one layer and the softmax one on the
            // next, at the same shape (Kimi Linear, `[2304, 4096]` on
            // both). The graph's per-layer operator is the only authority
            // that separates them.
            //
            // A tensor whose layer index is not in the table cannot be
            // classified against an operator at all; it falls to the
            // layer-blind table and, if that fails, is reported
            // unclassified — never quietly assigned.
            let operator = tensor
                .name
                .split_once('.')
                .and_then(|(index, _)| index.parse::<usize>().ok())
                .and_then(|layer| attention_table.get(layer))
                .map_or(LayerOperator::Softmax, |policy| policy.operator);
            match classify_stack_tensor_on(&tensor.name, operator) {
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
        for (layer, policy) in attention_table.iter().enumerate() {
            let geometry = layer_geometry(layer);
            let present = by_layer.get(&layer);
            let bank = bank_by_layer.get(&layer);
            // A layer is routed by operand evidence — it has an expert
            // bank or a router — under the surface's judgment; the
            // judgment alone routes nothing, the evidence alone is a
            // stray operand (`absent_op` names it).
            let routed = ffn_moe.is_some()
                && (bank.is_some()
                    || present.is_some_and(|s| s.contains_key(&OperandRole::MoeRouterWeight)));
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
                sinks: attn.is_some_and(|a| a.sinks.is_some()),
                routed,
                hybrid,
                moe: ffn_moe,
                mamba2: surface.mamba2,
                v_from_k: policy.v_from_k,
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
    // No standalone `OutputHead` object is placed for a checkpoint that
    // ships no separate `lm_head`-named tensor group at all — the near-
    // universal tied-embeddings convention, not a missing object. Reusing
    // the embedding object's own tensor reference is judged here, from
    // `surface.head_reuses_embedding` alone (see [`ModelArchitecture::
    // output_head_reuses_embedding`](larql_models::config::ModelArchitecture::output_head_reuses_embedding)):
    // the container never gets a second copy of the matrix, and a
    // checkpoint that explicitly declared `tie_word_embeddings: false`
    // and still has no head tensor stays `None` here — a lost tensor, not
    // a tied one, so it must not silently reuse the embedding.
    let head_tensor = single(
        ObjectKind::OutputHead,
        vocab.map(|v| vec![v, hidden]),
        &mut defects,
    )
    .or_else(|| {
        surface
            .head
            .as_ref()
            .is_some_and(|h| h.head_reuses_embedding)
            .then(|| embedding_tensor.clone())
            .flatten()
    });
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
        let geometry = layer_geometry(layer);
        // A mixer-only layer, on operand evidence: the fused five-way
        // `in_proj` is the discriminator (no other operator's role table
        // can put it in `slot`). Its whole program is the mixer — one
        // pre-block norm, no attention wrap, no FFN — so it is built
        // here and the transformer shape below is never consulted.
        if slot.contains_key(&OperandRole::Mamba2InProj) {
            let mixer = surface.mamba2.unwrap_or_else(|| {
                panic!(
                    "layer {layer} ships a Mamba2 operand while the component declares no \
                     mixer surface; closure should have refused this before the plan was built"
                )
            });
            let consumed = slot.len();
            layers.push(LayerPlan {
                layer,
                pre_attention_norm: norm_op(
                    surface.norm.pre,
                    &stack_id,
                    get(OperandRole::Mamba2PreMixerNorm),
                ),
                attention: LayerAttention::Mamba2(Box::new(Mamba2Op {
                    geometry: mixer.geometry,
                    activation: mixer.activation,
                    residual_in_fp32: surface.residual_in_fp32,
                    in_proj: operand(&stack_id, get(OperandRole::Mamba2InProj)),
                    conv1d: operand(&stack_id, get(OperandRole::Mamba2Conv1d)),
                    conv1d_bias: slot
                        .get(&OperandRole::Mamba2Conv1dBias)
                        .map(|t| operand(&stack_id, t)),
                    a_log: operand(&stack_id, get(OperandRole::Mamba2ALog)),
                    d: operand(&stack_id, get(OperandRole::Mamba2D)),
                    dt_bias: operand(&stack_id, get(OperandRole::Mamba2DtBias)),
                    gated_norm: slot
                        .get(&OperandRole::Mamba2GatedNorm)
                        .map(|t| norm_op(surface.norm.pre, &stack_id, t)),
                    out_proj: operand(&stack_id, get(OperandRole::Mamba2OutProj)),
                })),
                post_attention_norm: None,
                pre_ffn_norm: None,
                ffn: None,
                post_ffn_norm: None,
                layer_scale: None,
                residual_scale: surface.residual_scale,
                operands_accounted: consumed,
                operands_present: consumed,
            });
            continue;
        }
        // Every non-mixer layer's program includes attention wrap norms
        // and an FFN, so their surface groups are present when closure
        // held — the panics state the invariant, mirroring KDA's below.
        let ffn_s = ffn_surface.unwrap_or_else(|| {
            panic!(
                "layer {layer} requires an FFN op while the surface carries no ffn group; \
                 closure should have refused this before the plan was built"
            )
        });
        let bias = |role: OperandRole| {
            (attn.and_then(|a| a.attention_bias) == Some(true))
                .then(|| operand(&stack_id, get(role)))
        };
        let qk_norm = match (attn, slot.contains_key(&OperandRole::AttnQNorm)) {
            (Some(a), true) => Some(QkNormOp {
                scope: a.qk_norm_scope,
                weight_offset: a.qk_norm_weight_offset,
                q: operand(&stack_id, get(OperandRole::AttnQNorm)),
                k: operand(&stack_id, get(OperandRole::AttnKNorm)),
            }),
            _ => None,
        };
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
            NormPlacement::PreMixer => panic!(
                "layer {layer} carries transformer operands under a mixer-only norm \
                 placement; closure should have refused this before the plan was built"
            ),
        };
        let bank_slot = bank_by_layer.get(&layer);
        let bank_id = tables
            .get(&ObjectKind::ExpertBank)
            .map(|(o, _)| o.id.clone())
            .unwrap_or_default();
        let dense_op = || FfnOp {
            intermediate_size: inter,
            activation: ffn_s.activation,
            gate_policy: ffn_s.gate_policy,
            gate: gated_ffn.then(|| operand(&stack_id, get(OperandRole::FfnGate))),
            up: operand(&stack_id, get(OperandRole::FfnUp)),
            down: operand(&stack_id, get(OperandRole::FfnDown)),
        };
        let ffn = match (ffn_moe, bank_slot) {
            (Some(moe), Some(bank)) => {
                let bank_operand = |role: OperandRole| operand(&bank_id, &bank[&role]);
                let optional = |role: OperandRole| bank.get(&role).map(|t| operand(&bank_id, t));
                let gemma4_router = moe.router_kind == MoeRouterKind::Gemma4Hybrid;
                // `ExpertFormat::PerExpert`: no fused operand exists, so the
                // bank is `experts` independent gate/up/down triples rather
                // than one `PackedProjection` per branch — see
                // `ExpertBank`'s docs for why this cannot reuse the packed
                // shape with a placeholder.
                let bank = if moe.expert_format == ExpertFormat::PerExpert {
                    let per_expert = |ctor: fn(u16) -> OperandRole| -> Vec<OperandRef> {
                        (0..moe.experts as u16)
                            .map(|e| bank_operand(ctor(e)))
                            .collect()
                    };
                    ExpertBank::PerExpert {
                        gate: per_expert(OperandRole::PerExpertGate),
                        up: per_expert(OperandRole::PerExpertUp),
                        down: per_expert(OperandRole::PerExpertDown),
                    }
                } else {
                    ExpertBank::Packed {
                        gate_up: Box::new(PackedProjection {
                            weights: bank_operand(OperandRole::ExpertGateUp),
                            scales: optional(OperandRole::ExpertGateUpScales),
                            bias: optional(OperandRole::ExpertGateUpBias),
                        }),
                        down: Box::new(PackedProjection {
                            weights: bank_operand(OperandRole::ExpertDown),
                            scales: optional(OperandRole::ExpertDownScales),
                            bias: optional(OperandRole::ExpertDownBias),
                        }),
                    }
                };
                // Always-active, unscaled — Kimi's `KimiSparseMoeBlock.
                // forward`: `y = moe(...); y = y + shared_experts(identity)`.
                // `required_roles`/`absent_op` paired `Some` here with
                // `moe.shared_experts > 0` exactly, so this cannot desync
                // from the closure pass that admitted the layer.
                let shared = (moe.shared_experts > 0).then(|| SharedExpertOp {
                    intermediate_size: moe.expert_intermediate_size * moe.shared_experts,
                    activation: ffn_s.activation,
                    gate_policy: ffn_s.gate_policy,
                    gate: operand(&stack_id, get(OperandRole::SharedExpertGate)),
                    up: operand(&stack_id, get(OperandRole::SharedExpertUp)),
                    down: operand(&stack_id, get(OperandRole::SharedExpertDown)),
                });
                let routed = RoutedFfnOp {
                    experts: moe.experts,
                    top_k: moe.top_k,
                    expert_intermediate_size: moe.expert_intermediate_size,
                    router_kind: moe.router_kind,
                    routing_policy: moe.routing_policy,
                    activation: ffn_s.activation,
                    gate_policy: ffn_s.gate_policy,
                    expert_format: moe.expert_format,
                    gate_up_layout: moe.gate_up_layout,
                    router: operand(&stack_id, get(OperandRole::MoeRouterWeight)),
                    router_bias: moe
                        .router_bias
                        .then(|| operand(&stack_id, get(OperandRole::MoeRouterBias))),
                    bank,
                    shared,
                    router_scale: gemma4_router
                        .then(|| operand(&stack_id, get(OperandRole::MoeRouterScale))),
                    router_per_expert_scale: gemma4_router
                        .then(|| operand(&stack_id, get(OperandRole::MoeRouterPerExpertScale))),
                    // The router's scale-less norm uses the layer's norm
                    // epsilon (HF: `Gemma4RMSNorm(eps=config.rms_norm_eps,
                    // with_scale=False)`).
                    router_norm_eps: gemma4_router.then_some(surface.norm.pre.eps),
                };
                if moe.hybrid {
                    LayerFfn::Hybrid(Box::new(HybridFfnOp {
                        dense: dense_op(),
                        routed,
                        pre_experts_norm: norm_op(
                            surface.norm.pre,
                            &stack_id,
                            get(OperandRole::PreExpertsNorm),
                        ),
                        post_dense_norm: norm_op(
                            surface.norm.pre,
                            &stack_id,
                            get(OperandRole::PostDenseFfnNorm),
                        ),
                        post_experts_norm: norm_op(
                            surface.norm.pre,
                            &stack_id,
                            get(OperandRole::PostExpertsNorm),
                        ),
                    }))
                } else {
                    LayerFfn::Routed(Box::new(routed))
                }
            }
            _ => LayerFfn::Dense(Box::new(dense_op())),
        };
        let layer_scale = slot
            .get(&OperandRole::LayerScalar)
            .map(|t| operand(&stack_id, t));
        let consumed = slot.len() + bank_slot.map_or(0, |b| b.len());
        layers.push(LayerPlan {
            layer,
            pre_attention_norm: norm_op(
                surface.norm.pre,
                &stack_id,
                get(OperandRole::PreAttentionNorm),
            ),
            // Which attention-class operator this layer runs, decided on
            // OPERAND EVIDENCE: a layer holding the fused q|k|v projection
            // of a recurrence is a DeltaNet layer, whatever else is
            // declared. Roles arrive only through exact ROLE_TABLE
            // suffixes, so nothing reaches here by lexical fallback.
            attention: if slot.contains_key(&OperandRole::KdaDtBias) {
                // KDA, on operand evidence. `dt_bias` is the discriminator
                // and not by accident: Gated DeltaNet carries one too, at
                // `[Hv]` against KDA's `[Hv·Dv]`, so the two are separated
                // by the operand whose GEOMETRY differs rather than by a
                // name either could have used. The role only reached this
                // slot because the layer's operator said KDA, so this is a
                // second, independent agreement rather than a restatement.
                let k = surface.kda.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships a KDA operand while the component declares no KDA \
                         geometry; closure should have refused this before the plan was built"
                    )
                });
                // The rank the config never declares, resolved ONCE from
                // the operand that carries it and then stated on the op.
                // `f_a_proj` is `[rank, hidden]`; taking the row count is
                // the only place this is read, so no consumer has to
                // recover it from a shape later.
                let gate_rank = get(OperandRole::KdaFAProj)
                    .shape
                    .first()
                    .copied()
                    .unwrap_or(0);
                LayerAttention::Kda(Box::new(KdaOp {
                    num_heads: k.num_heads,
                    head_dim: k.head_dim,
                    conv_kernel: k.conv_kernel,
                    gate_rank,
                    gate_lower_bound: surface.kda_gate_lower_bound,
                    q_proj: operand(&stack_id, get(OperandRole::KdaQProj)),
                    k_proj: operand(&stack_id, get(OperandRole::KdaKProj)),
                    v_proj: operand(&stack_id, get(OperandRole::KdaVProj)),
                    q_conv1d: operand(&stack_id, get(OperandRole::KdaQConv1d)),
                    k_conv1d: operand(&stack_id, get(OperandRole::KdaKConv1d)),
                    v_conv1d: operand(&stack_id, get(OperandRole::KdaVConv1d)),
                    f_a_proj: operand(&stack_id, get(OperandRole::KdaFAProj)),
                    f_b_proj: operand(&stack_id, get(OperandRole::KdaFBProj)),
                    g_a_proj: operand(&stack_id, get(OperandRole::KdaGAProj)),
                    g_b_proj: operand(&stack_id, get(OperandRole::KdaGBProj)),
                    b_proj: operand(&stack_id, get(OperandRole::KdaBProj)),
                    a_log: operand(&stack_id, get(OperandRole::KdaALog)),
                    dt_bias: operand(&stack_id, get(OperandRole::KdaDtBias)),
                    o_norm: operand(&stack_id, get(OperandRole::KdaONorm)),
                    out_proj: operand(&stack_id, get(OperandRole::KdaOutProj)),
                }))
            } else if slot.contains_key(&OperandRole::LinearAttnInProjQkv) {
                let l = surface.linear_attention.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships a Gated DeltaNet operand while the component \
                         declares no linear-attention geometry; closure should have refused \
                         this before the plan was built"
                    )
                });
                LayerAttention::GatedDelta(Box::new(GatedDeltaOp {
                    num_key_heads: l.key_heads,
                    num_value_heads: l.value_heads,
                    key_head_dim: l.key_head_dim,
                    value_head_dim: l.value_head_dim,
                    conv_kernel: l.conv_kernel,
                    state_dtype: l.state_dtype,
                    in_proj_qkv: operand(&stack_id, get(OperandRole::LinearAttnInProjQkv)),
                    in_proj_a: operand(&stack_id, get(OperandRole::LinearAttnInProjA)),
                    in_proj_b: operand(&stack_id, get(OperandRole::LinearAttnInProjB)),
                    in_proj_z: operand(&stack_id, get(OperandRole::LinearAttnInProjZ)),
                    conv1d: operand(&stack_id, get(OperandRole::LinearAttnConv1d)),
                    a_log: operand(&stack_id, get(OperandRole::LinearAttnALog)),
                    dt_bias: operand(&stack_id, get(OperandRole::LinearAttnDtBias)),
                    norm: operand(&stack_id, get(OperandRole::LinearAttnNorm)),
                    out_proj: operand(&stack_id, get(OperandRole::LinearAttnOutProj)),
                }))
            } else if slot.contains_key(&OperandRole::MlaKvAProj) {
                // MLA, on operand evidence. `kv_a_proj_with_mqa` is the
                // discriminator: no other operator's role table can put it
                // in `slot`, so its presence alone proves the layer's
                // operator said MLA — the same "role only reached here
                // because the graph already decided" reasoning KDA's
                // branch states above.
                let m = surface.mla.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships an MLA operand while the component declares no \
                         MLA geometry; closure should have refused this before the plan was \
                         built"
                    )
                });
                LayerAttention::Mla(Box::new(MlaOp {
                    num_heads: m.num_heads,
                    kv_lora_rank: m.kv_lora_rank,
                    qk_nope_head_dim: m.qk_nope_head_dim,
                    qk_rope_head_dim: m.qk_rope_head_dim,
                    v_head_dim: m.v_head_dim,
                    q_proj: operand(&stack_id, get(OperandRole::MlaQProj)),
                    kv_a_proj: operand(&stack_id, get(OperandRole::MlaKvAProj)),
                    kv_b_proj: operand(&stack_id, get(OperandRole::MlaKvBProj)),
                    kv_a_norm: operand(&stack_id, get(OperandRole::MlaKvANorm)),
                    out_proj: operand(&stack_id, get(OperandRole::MlaOutProj)),
                }))
            } else {
                let a = attn.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships softmax attention operands while the surface \
                         carries no attention group; closure should have refused this before \
                         the plan was built"
                    )
                });
                LayerAttention::Softmax(Box::new(AttentionOp {
                    num_q_heads: geometry.num_q_heads,
                    num_kv_heads: geometry.num_kv_heads,
                    head_dim: geometry.head_dim,
                    query_scale: a.query_scale,
                    score_scale: a.score_scale,
                    logit_softcapping: a.logit_softcapping,
                    // The graph carries no span exactly when it recorded
                    // a recurrence for this layer. Reaching here means the
                    // layer ships softmax operands anyway — the checkpoint
                    // contradicting itself, config against tensors — and
                    // the mirror of the panic above: an invariant the
                    // builder upholds, not a case to paper over with a
                    // default span.
                    span: policy.span.unwrap_or_else(|| {
                        panic!(
                            "layer {layer} ships softmax attention operands while the graph \
                             records a recurrence for it (no span); the checkpoint's \
                             layer_types and its tensors disagree"
                        )
                    }),
                    window: policy.window,
                    position: policy.position,
                    qk_norm,
                    parameter_free_qk_norm: a.parameter_free_qk_norm,
                    q: operand(&stack_id, get(OperandRole::AttnQ)),
                    k: operand(&stack_id, get(OperandRole::AttnK)),
                    // On a K≡V layer the value operand IS the key operand:
                    // the op reads one matrix twice, and says so.
                    v: operand(
                        &stack_id,
                        get(if policy.v_from_k {
                            OperandRole::AttnK
                        } else {
                            OperandRole::AttnV
                        }),
                    ),
                    v_from_k: policy.v_from_k,
                    o: operand(&stack_id, get(OperandRole::AttnO)),
                    // On a fused source the gate has NO operand of its
                    // own: it is the per-head second half of the query
                    // projection, so the op names `q_proj` and reads one
                    // matrix for both roles — the same "one matrix, two
                    // roles" statement `v_from_k` makes for K≡V layers.
                    output_gate: a.output_gate.map(|spec| GateOp {
                        spec,
                        projection: operand(
                            &stack_id,
                            get(match spec.source {
                                larql_models::config::GateSource::AttentionInput => {
                                    OperandRole::AttnOutputGate
                                }
                                larql_models::config::GateSource::FusedQueryProjection => {
                                    OperandRole::AttnQ
                                }
                            }),
                        ),
                    }),
                    // Closure held, so `Some(true)` means all four are here
                    // and anything else means none is.
                    q_bias: bias(OperandRole::AttnQBias),
                    k_bias: bias(OperandRole::AttnKBias),
                    v_bias: bias(OperandRole::AttnVBias),
                    o_bias: bias(OperandRole::AttnOBias),
                    sinks: a.sinks.map(|spec| SinkOp {
                        spec,
                        logits: operand(&stack_id, get(OperandRole::AttnSinks)),
                    }),
                }))
            },
            post_attention_norm,
            pre_ffn_norm: Some(norm_op(surface.norm.pre, &stack_id, get(pre_ffn_role))),
            ffn: Some(ffn),
            post_ffn_norm,
            layer_scale,
            residual_scale: surface.residual_scale,
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
    /// Routed AND dense in one layer (Gemma 4): the dense roles are
    /// required alongside the routed ones, plus the branch norms.
    hybrid: bool,
    moe: Option<MoeSurface>,
    /// The Mamba2 mixer surface, on a component that declares one — what
    /// decides the conv-bias and gated-norm operand requirements on a
    /// mixer layer.
    mamba2: Option<Mamba2Surface>,
    /// V is the K projection on this layer: no V operand is required, and
    /// one present is a stray.
    v_from_k: bool,
    /// Which attention-class operator this layer runs.
    ///
    /// The operator itself rather than an `is_recurrent` flag: the two
    /// recurrences require *different* operand sets, so a boolean would
    /// have to be paired with a second one the moment a second recurrence
    /// existed — which is the shape that let one operator stand in for
    /// another in the first place.
    operator: LayerOperator,
}

/// Roles every layer must supply, given the surface's ops.
fn required_roles(ops: &LayerOps) -> Vec<OperandRole> {
    // A mixer-only layer's program shares nothing with the transformer
    // shape below — one pre-mixer norm, no attention wrap, no FFN — so
    // it returns its own complete set rather than threading exemptions
    // through every clause that follows.
    if ops.operator.is_mamba2() {
        let mut roles = vec![
            OperandRole::Mamba2PreMixerNorm,
            OperandRole::Mamba2InProj,
            OperandRole::Mamba2Conv1d,
            OperandRole::Mamba2ALog,
            OperandRole::Mamba2D,
            OperandRole::Mamba2DtBias,
            OperandRole::Mamba2OutProj,
        ];
        if let Some(mixer) = ops.mamba2 {
            if mixer.geometry.use_conv_bias {
                roles.push(OperandRole::Mamba2Conv1dBias);
            }
            if mixer.geometry.rms_norm {
                roles.push(OperandRole::Mamba2GatedNorm);
            }
        }
        return roles;
    }
    let mut roles = vec![
        OperandRole::PreAttentionNorm,
        OperandRole::PostAttentionNorm,
    ];
    if ops.operator.is_kda() {
        // Fifteen operands, and all fifteen are required: a KDA layer
        // missing one is not a partially-specified attention layer, it is
        // an operator that cannot run. None of the softmax roles apply —
        // the recurrence retains no per-position key or value.
        roles.extend([
            OperandRole::KdaQProj,
            OperandRole::KdaKProj,
            OperandRole::KdaVProj,
            OperandRole::KdaQConv1d,
            OperandRole::KdaKConv1d,
            OperandRole::KdaVConv1d,
            OperandRole::KdaFAProj,
            OperandRole::KdaFBProj,
            OperandRole::KdaGAProj,
            OperandRole::KdaGBProj,
            OperandRole::KdaBProj,
            OperandRole::KdaALog,
            OperandRole::KdaDtBias,
            OperandRole::KdaONorm,
            OperandRole::KdaOutProj,
        ]);
    } else if ops.operator.is_gated_delta() {
        // A recurrence has no query, key, value or output projection —
        // demanding them made all 48 of Qwen3.8's linear layers report
        // four missing operands each for tensors that correctly do not
        // exist. Its nine operands are required instead, so the layer is
        // still fully pinned rather than merely exempted.
        roles.extend([
            OperandRole::LinearAttnInProjQkv,
            OperandRole::LinearAttnInProjA,
            OperandRole::LinearAttnInProjB,
            OperandRole::LinearAttnInProjZ,
            OperandRole::LinearAttnConv1d,
            OperandRole::LinearAttnALog,
            OperandRole::LinearAttnDtBias,
            OperandRole::LinearAttnNorm,
            OperandRole::LinearAttnOutProj,
        ]);
    } else if ops.operator.is_mla() {
        // No K/V projection exists to require — the compressed latent and
        // its decompression are the only KV path, so demanding AttnK/AttnV
        // would report two missing operands per layer for tensors the
        // checkpoint never shipped, the same shape GatedDelta's roles fix
        // for its own operands above.
        roles.extend([
            OperandRole::MlaQProj,
            OperandRole::MlaKvAProj,
            OperandRole::MlaKvBProj,
            OperandRole::MlaKvANorm,
            OperandRole::MlaOutProj,
        ]);
    } else {
        roles.extend([OperandRole::AttnQ, OperandRole::AttnK, OperandRole::AttnO]);
        if !ops.v_from_k {
            roles.push(OperandRole::AttnV);
        }
    }
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
    if ops.routed {
        if let Some(moe) = ops.moe {
            roles.push(OperandRole::MoeRouterWeight);
            if moe.expert_format == ExpertFormat::PerExpert {
                // No fused bank tensor exists to bind — the checkpoint
                // ships one gate/up/down triple PER EXPERT, so closure
                // requires the complete indexed set, not one flat role.
                // `absent_op` is what turns a stray expert beyond
                // `moe.experts` into a defect; this only states what MUST
                // be present.
                roles.extend((0..moe.experts as u16).flat_map(|expert| {
                    [
                        OperandRole::PerExpertGate(expert),
                        OperandRole::PerExpertUp(expert),
                        OperandRole::PerExpertDown(expert),
                    ]
                }));
            } else {
                roles.push(OperandRole::ExpertGateUp);
                roles.push(OperandRole::ExpertDown);
            }
            if moe.router_bias {
                roles.push(OperandRole::MoeRouterBias);
            }
            if moe.expert_format.has_split_scale_streams() {
                roles.push(OperandRole::ExpertGateUpScales);
                roles.push(OperandRole::ExpertDownScales);
            }
            // Gemma 4's router conditions its input and its selected
            // weights with two learned scales; the kind implies both.
            if moe.router_kind == MoeRouterKind::Gemma4Hybrid {
                roles.push(OperandRole::MoeRouterScale);
                roles.push(OperandRole::MoeRouterPerExpertScale);
            }
            // Always-active alongside the routed selection — required
            // whenever the judgment declares one, on every routed layer
            // (Kimi/DeepSeek run it beside the routed block, never as a
            // Gemma-4-style hybrid dense branch).
            if moe.shared_experts > 0 {
                roles.extend([
                    OperandRole::SharedExpertGate,
                    OperandRole::SharedExpertUp,
                    OperandRole::SharedExpertDown,
                ]);
            }
        }
    }
    if !ops.routed || ops.hybrid {
        roles.push(OperandRole::FfnUp);
        roles.push(OperandRole::FfnDown);
        if ops.gated_ffn {
            roles.push(OperandRole::FfnGate);
        }
    }
    if ops.hybrid {
        roles.extend([
            OperandRole::PreExpertsNorm,
            OperandRole::PostDenseFfnNorm,
            OperandRole::PostExpertsNorm,
        ]);
    }
    roles
}

/// The primitive a found operand requires when the surface does not carry
/// its op. `None` when the operand is consumed by a declared op.
fn absent_op(role: OperandRole, ops: &LayerOps) -> Option<&'static str> {
    match role {
        // A mixer-only layer runs neither attention nor an FFN; any
        // transformer-shaped operand on it is a stray, whatever its name.
        OperandRole::AttnQ
        | OperandRole::AttnK
        | OperandRole::AttnV
        | OperandRole::AttnO
        | OperandRole::FfnGate
        | OperandRole::FfnUp
        | OperandRole::FfnDown
        | OperandRole::PreAttentionNorm
        | OperandRole::PostAttentionNorm
        | OperandRole::PreFfnNorm
        | OperandRole::PostFfnNorm
            if ops.operator.is_mamba2() =>
        {
            Some("a mixer-only Mamba2 layer (no attention, no FFN)")
        }
        OperandRole::Mamba2Conv1dBias if !ops.mamba2.is_some_and(|m| m.geometry.use_conv_bias) => {
            Some("a conv bias (`use_conv_bias` declares none)")
        }
        OperandRole::Mamba2GatedNorm if !ops.mamba2.is_some_and(|m| m.geometry.rms_norm) => {
            Some("the mixer's gated RMSNorm (`rms_norm` declares none)")
        }
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
        OperandRole::AttnV if ops.v_from_k => {
            Some("value projection (this layer's V is its K projection — `attention_k_eq_v`)")
        }
        // MLA has no plain query/key/value/output projection to bind: its
        // `q_proj`/`o_proj` SUFFIXES are intercepted by `MLA_ROLE_TABLE`
        // before they ever reach these roles, and it never ships
        // `k_proj`/`v_proj` at all (K/V arrive only through the
        // compressed path). Without this guard a stray `k_proj` on an
        // MLA layer classified as plain `AttnK` and was checked against
        // the SOFTMAX contract (`num_kv_heads · head_dim`) — the wrong
        // question, not the right refusal.
        OperandRole::AttnQ | OperandRole::AttnK | OperandRole::AttnV | OperandRole::AttnO
            if ops.operator.is_mla() =>
        {
            Some("MLA (Multi-Latent Attention) — no plain query/key/value/output projection")
        }
        OperandRole::FfnGate if !ops.routed && !ops.gated_ffn => Some("gated FFN"),
        OperandRole::FfnGate | OperandRole::FfnUp | OperandRole::FfnDown
            if ops.routed && !ops.hybrid =>
        {
            Some("dense FFN (this layer is routed)")
        }
        OperandRole::MoeRouterScale | OperandRole::MoeRouterPerExpertScale
            if !ops
                .moe
                .is_some_and(|m| m.router_kind == MoeRouterKind::Gemma4Hybrid) =>
        {
            Some("Gemma 4 router conditioning (router kind gemma4_top_k_softmax)")
        }
        OperandRole::PreExpertsNorm
        | OperandRole::PostDenseFfnNorm
        | OperandRole::PostExpertsNorm
            if !ops.hybrid =>
        {
            Some("hybrid dense+routed FFN (judged semantics)")
        }
        OperandRole::MoeRouterWeight
        | OperandRole::MoeRouterBias
        | OperandRole::MoeRouterScale
        | OperandRole::MoeRouterPerExpertScale
        | OperandRole::ExpertGateUp
        | OperandRole::ExpertGateUpScales
        | OperandRole::ExpertGateUpBias
        | OperandRole::ExpertDown
        | OperandRole::ExpertDownScales
        | OperandRole::ExpertDownBias
        | OperandRole::PerExpertGate(_)
        | OperandRole::PerExpertUp(_)
        | OperandRole::PerExpertDown(_)
        | OperandRole::SharedExpertGate
        | OperandRole::SharedExpertUp
        | OperandRole::SharedExpertDown
            if !ops.routed =>
        {
            Some("routed FFN (judged semantics)")
        }
        OperandRole::MoeRouterBias if ops.moe.is_some_and(|m| !m.router_bias) => {
            Some("router bias (declared by the routed-FFN judgment)")
        }
        // A `PerExpert` operand whose index the routed-FFN judgment does
        // not declare — the set-closure half of expert-bank carving: an
        // index beyond `moe.experts` is as much a defect as one missing
        // from `0..experts` ([`required_roles`] states the other half).
        OperandRole::PerExpertGate(expert)
        | OperandRole::PerExpertUp(expert)
        | OperandRole::PerExpertDown(expert)
            if ops.moe.is_some_and(|m| expert as usize >= m.experts) =>
        {
            Some("an expert index the routed-FFN judgment does not declare")
        }
        OperandRole::SharedExpertGate
        | OperandRole::SharedExpertUp
        | OperandRole::SharedExpertDown
            if ops.moe.is_some_and(|m| m.shared_experts == 0) =>
        {
            Some("a shared expert (the routed-FFN judgment declares none)")
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

/// The geometry one layer's stack operands are checked against — the
/// layer's own head geometry under the component's query-head count.
struct StackGeometry {
    hidden: usize,
    /// `num_q_heads · head_dim` — the ATTENTION width. What `o_proj`
    /// consumes, and what the query half occupies.
    q_rows: usize,
    /// Rows the stored query projection actually carries.
    ///
    /// Equal to [`Self::q_rows`] on an ordinary stack, and **twice** it
    /// when the component's output gate is sourced from the query
    /// projection: that projection emits `2 · head_dim` per head, query
    /// and gate interleaved. Kept as its own field rather than doubling
    /// `q_rows`, because `o_proj` and the query-bias contract are still
    /// sized by the attention width — conflating the two would silently
    /// demand a 12288-wide `o_proj` on Qwen3.8, which carries 6144.
    q_proj_rows: usize,
    kv_rows: usize,
    intermediate: usize,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    qk_scope: larql_models::config::QkNormScope,
    /// The recurrence's geometry, on a component that declares one. Kept
    /// beside the softmax fields rather than folded into them: the key and
    /// value sides carry different head counts, so `num_q_heads`/`head_dim`
    /// cannot describe this operator.
    linear: Option<LinearAttentionSurface>,
    /// The KDA block's geometry, on a component that declares one.
    /// Disjoint from [`Self::linear`]: the two describe different
    /// operators, and a stack carrying KDA operands against a Gated
    /// DeltaNet geometry would validate the wrong contracts.
    kda: Option<larql_models::config::KdaGeometry>,
    /// The MLA operator's geometry, on a component whose full-attention
    /// layers run it. Disjoint from every field above it — MLA is neither
    /// the softmax fields' uniform per-head width nor a recurrence.
    mla: Option<MlaSurface>,
    /// The Mamba2 mixer's surface, on a component whose layers run it.
    /// Disjoint from every field above for the same reason each of them
    /// is from the others.
    mamba2: Option<Mamba2Surface>,
}

/// Whether a stored shape satisfies a contract.
///
/// Exact equality, **plus one narrow equivalence**: a contract for a
/// *vector* is satisfied by the same values carrying broadcast singleton
/// dimensions. Kimi Linear stores its per-head decay as
/// `A_log: [1, 1, 32, 1]` — the shape its reference broadcasts against
/// `[B, T, H, D]` — where the contract says `[32]`. Those are the same 32
/// numbers in the same order.
///
/// Deliberately **not** a general squeeze. The equivalence applies only
/// when the contract is one-dimensional, so it can never quietly accept a
/// re-laid-out matrix: `[2, 16]` still fails `[32]`, and a `[4096, 128]`
/// contract is unaffected by anything here. A blanket "drop all ones"
/// would accept a genuine relayout as readily as a broadcast form, and the
/// point of a shape contract is to refuse exactly that.
pub(super) fn shape_satisfies(actual: &[usize], expected: &[usize]) -> bool {
    if actual == expected {
        return true;
    }
    expected.len() == 1 && actual.iter().filter(|d| **d != 1).eq(expected.iter())
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
        q_proj_rows,
        kv_rows,
        intermediate,
        head_dim,
        num_q_heads,
        num_kv_heads: _,
        qk_scope,
        linear,
        kda,
        mla,
        mamba2,
    } = *g;
    match role {
        // Mamba2/SSD. Every contract follows from the mixer's own
        // declared geometry closing over the component width; none from
        // the softmax fields, which are zero on a mixer-only stack.
        // `mamba2` absent while such an operand exists is a refusal, for
        // the same reason `linear`/`kda`/`mla` absences are.
        OperandRole::Mamba2InProj => Some(vec![mamba2?.geometry.in_proj_rows(hidden), hidden]),
        OperandRole::Mamba2Conv1d => {
            let g = mamba2?.geometry;
            Some(vec![g.conv_dim(hidden), 1, g.conv_kernel])
        }
        OperandRole::Mamba2Conv1dBias => Some(vec![mamba2?.geometry.conv_dim(hidden)]),
        // Per-head scalars — the axis that separates this family from
        // KDA's per-channel `dt_bias`.
        OperandRole::Mamba2ALog | OperandRole::Mamba2D | OperandRole::Mamba2DtBias => {
            Some(vec![mamba2?.geometry.num_heads])
        }
        // Over the FULL inner width — unlike DeltaNet's per-head norm.
        OperandRole::Mamba2GatedNorm => Some(vec![mamba2?.geometry.d_inner(hidden)]),
        OperandRole::Mamba2OutProj => Some(vec![hidden, mamba2?.geometry.d_inner(hidden)]),
        OperandRole::Mamba2PreMixerNorm => Some(vec![hidden]),
        OperandRole::AttnQ => Some(vec![q_proj_rows, hidden]),
        OperandRole::AttnK | OperandRole::AttnV => Some(vec![kv_rows, hidden]),
        OperandRole::AttnO => Some(vec![hidden, q_rows]),
        OperandRole::PreAttentionNorm
        | OperandRole::PostAttentionNorm
        | OperandRole::PreFfnNorm
        | OperandRole::PostFfnNorm
        | OperandRole::PreExpertsNorm
        | OperandRole::PostDenseFfnNorm
        | OperandRole::PostExpertsNorm
        | OperandRole::MoeRouterScale => Some(vec![hidden]),
        OperandRole::MoeRouterPerExpertScale => Some(vec![moe?.experts]),
        OperandRole::LayerScalar => Some(vec![1]),
        OperandRole::AttnQNorm | OperandRole::AttnKNorm => match qk_scope {
            QkNormScope::PerHead => Some(vec![head_dim]),
            // Full-projection shape contract unpinned until a real
            // instance is judged.
            QkNormScope::FullProjection => None,
        },
        // Gated DeltaNet. Every shape follows from the recurrence's own
        // geometry, and none from the softmax fields above — the key and
        // value sides carry different head counts (16 and 48 on Qwen3.8),
        // so nothing there stands in for them.
        //
        // `linear` absent while such an operand exists is a refusal, not a
        // waiver: the stack ships a recurrence whose geometry the component
        // never declared, and closure must not accept an operand it cannot
        // state a contract for.
        OperandRole::LinearAttnInProjQkv => Some(vec![linear?.qkv_channels(), hidden]),
        OperandRole::LinearAttnInProjA | OperandRole::LinearAttnInProjB => {
            Some(vec![linear?.value_heads, hidden])
        }
        OperandRole::LinearAttnInProjZ => Some(vec![linear?.value_width(), hidden]),
        // Depthwise over the fused channels: one kernel per channel.
        OperandRole::LinearAttnConv1d => {
            let l = linear?;
            Some(vec![l.qkv_channels(), 1, l.conv_kernel])
        }
        // Per-value-head scalars.
        OperandRole::LinearAttnALog | OperandRole::LinearAttnDtBias => {
            Some(vec![linear?.value_heads])
        }
        // Gated RMSNorm over ONE value head's width, not the full value
        // side — the norm is applied per head.
        OperandRole::LinearAttnNorm => Some(vec![linear?.value_head_dim]),
        OperandRole::LinearAttnOutProj => Some(vec![hidden, linear?.value_width()]),
        // Kimi Delta Attention. Every contract below follows from the KDA
        // block's own geometry; none from the softmax fields, which on a
        // hybrid checkpoint describe the OTHER layers of the same stack.
        //
        // `kda` absent while a KDA operand exists is a refusal for the
        // same reason `linear` is: an operand whose contract the component
        // never declared cannot be checked, and accepting it unchecked is
        // how a wrong binding survives.
        OperandRole::KdaQProj | OperandRole::KdaKProj | OperandRole::KdaVProj => {
            Some(vec![kda?.value_width(), hidden])
        }
        // Depthwise, one kernel per channel — three independent convs, not
        // one over fused channels. This is a structural difference from
        // Gated DeltaNet, not a parameterisation of it.
        OperandRole::KdaQConv1d | OperandRole::KdaKConv1d | OperandRole::KdaVConv1d => {
            let k = kda?;
            Some(vec![k.value_width(), 1, k.conv_kernel])
        }
        OperandRole::KdaBProj => Some(vec![kda?.num_heads, hidden]),
        OperandRole::KdaALog => Some(vec![kda?.num_heads]),
        // The discriminator: per CHANNEL, where Gated DeltaNet's is per
        // head. A checkpoint whose `dt_bias` is `[Hv]` is not a KDA block,
        // and this is the contract that says so.
        OperandRole::KdaDtBias => Some(vec![kda?.value_width()]),
        OperandRole::KdaONorm => Some(vec![kda?.head_dim]),
        OperandRole::KdaOutProj => Some(vec![hidden, kda?.value_width()]),
        // The f and g gates are low-rank and the config declares no rank,
        // so no per-operand contract can be stated from geometry alone.
        // Their agreement is a CLOSURE fact between the pair — `f_a` is
        // `[rank, hidden]` and `f_b` is `[Hv·Dv, rank]` for one rank — and
        // is checked there rather than invented here. `None` is the same
        // "contract not pinned" answer `QkNormScope::FullProjection` gives.
        OperandRole::KdaFAProj
        | OperandRole::KdaFBProj
        | OperandRole::KdaGAProj
        | OperandRole::KdaGBProj => None,
        // Multi-Latent Attention. Every contract below follows from the
        // MLA block's own geometry — none from the softmax fields, which
        // on a hybrid checkpoint describe the KDA layers of the same
        // stack. `mla` absent while an MLA operand exists is a refusal
        // for the same reason `kda`/`linear` are: an operand whose
        // contract the component never declared cannot be checked.
        OperandRole::MlaQProj => Some(vec![mla?.num_heads * mla?.q_head_dim(), hidden]),
        OperandRole::MlaKvAProj => Some(vec![mla?.kv_lora_rank + mla?.qk_rope_head_dim, hidden]),
        // Fused per-head nope-K + V, decompressed from the latent.
        OperandRole::MlaKvBProj => {
            let m = mla?;
            Some(vec![
                m.num_heads * (m.qk_nope_head_dim + m.v_head_dim),
                m.kv_lora_rank,
            ])
        }
        OperandRole::MlaKvANorm => Some(vec![mla?.kv_lora_rank]),
        OperandRole::MlaOutProj => Some(vec![hidden, mla?.num_heads * mla?.v_head_dim]),
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
        // `ExpertFormat::PerExpert`: one `[inter, hidden]` gate/up and one
        // `[hidden, inter]` down PER EXPERT — the index carried on the role
        // picks which expert's operand this is, not which shape.
        OperandRole::PerExpertGate(_) | OperandRole::PerExpertUp(_) => {
            Some(vec![moe?.expert_intermediate_size, hidden])
        }
        OperandRole::PerExpertDown(_) => Some(vec![hidden, moe?.expert_intermediate_size]),
        // Always-active shared expert(s): the same gated-FFN shape as a
        // routed expert, widened by `shared_experts` — Kimi's
        // `KimiSparseMoeBlock.__init__` sizes `KimiMLP` at
        // `moe_intermediate_size * num_shared_experts`, one wider FFN
        // rather than `shared_experts` separate ones.
        OperandRole::SharedExpertGate | OperandRole::SharedExpertUp => {
            let m = moe?;
            Some(vec![m.expert_intermediate_size * m.shared_experts, hidden])
        }
        OperandRole::SharedExpertDown => {
            let m = moe?;
            Some(vec![hidden, m.expert_intermediate_size * m.shared_experts])
        }
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

/// `absent_op`/`expected_shape`/`required_roles` are private pure
/// functions over private `LayerOps`/`StackGeometry` structs, so — unlike
/// the rest of this crate's tests, which build a real component/plan
/// through `opplan/tests/` — these have to live beside the code they
/// test (same reasoning `quant/convert.rs` and `opplan/gated_delta.rs`
/// already use for their own pure-function arms). Every arm here is one
/// no dense/softmax/non-MoE fixture reaches: hybrid dense+routed FFN,
/// Gemma 4's router conditioning, a declared-false router bias, an
/// unsplit expert scale stream, and the Gated DeltaNet operand-shape
/// table (nothing in this crate encodes a `linear_attention` checkpoint
/// through the real closure path yet — Qwen3.8's ladder is tracked
/// separately).
#[cfg(test)]
mod tests {
    use super::*;

    fn base_ops() -> LayerOps {
        LayerOps {
            placement: NormPlacement::PrePost,
            gated_ffn: true,
            mamba2: None,
            output_gate: false,
            attention_bias: false,
            sinks: false,
            routed: false,
            hybrid: false,
            moe: None,
            v_from_k: false,
            // Softmax by default: these fixtures predate the hybrid
            // ladder, and a recurrent default would silently retarget
            // every one of them at the operator they were not written for.
            operator: LayerOperator::Softmax,
        }
    }

    fn moe(
        router_kind: MoeRouterKind,
        router_bias: bool,
        expert_format: ExpertFormat,
    ) -> MoeSurface {
        MoeSurface {
            branch_scale: None,
            dense_prefix_layers: None,
            experts: 8,
            top_k: 2,
            expert_intermediate_size: 64,
            router_kind,
            routing_policy: larql_models::config::ExpertRoutingPolicy::SoftmaxThenSelect,
            router_bias,
            expert_format,
            gate_up_layout: Some(larql_models::config::GateUpLayout::ContiguousHalves),
            shared_experts: 0,
            hybrid: false,
        }
    }

    // ── absent_op: hybrid / routed-FFN / MoE exclusions ──────────────

    #[test]
    fn dense_ffn_roles_absent_on_a_routed_non_hybrid_layer() {
        let ops = LayerOps {
            routed: true,
            hybrid: false,
            moe: Some(moe(
                MoeRouterKind::TopKSoftmax,
                true,
                ExpertFormat::PackedMxfp4,
            )),
            ..base_ops()
        };
        for role in [
            OperandRole::FfnGate,
            OperandRole::FfnUp,
            OperandRole::FfnDown,
        ] {
            assert_eq!(
                absent_op(role, &ops),
                Some("dense FFN (this layer is routed)"),
                "{role:?}"
            );
        }
    }

    #[test]
    fn gemma4_router_conditioning_absent_unless_the_router_kind_says_so() {
        let non_gemma4 = LayerOps {
            routed: true,
            moe: Some(moe(
                MoeRouterKind::TopKSoftmax,
                true,
                ExpertFormat::PackedMxfp4,
            )),
            ..base_ops()
        };
        for role in [
            OperandRole::MoeRouterScale,
            OperandRole::MoeRouterPerExpertScale,
        ] {
            assert_eq!(
                absent_op(role, &non_gemma4),
                Some("Gemma 4 router conditioning (router kind gemma4_top_k_softmax)"),
                "{role:?}"
            );
        }
        let gemma4 = LayerOps {
            routed: true,
            moe: Some(moe(
                MoeRouterKind::Gemma4Hybrid,
                true,
                ExpertFormat::PackedMxfp4,
            )),
            ..base_ops()
        };
        assert_eq!(
            absent_op(OperandRole::MoeRouterScale, &gemma4),
            None,
            "a declared Gemma 4 router must not be reported absent"
        );
    }

    #[test]
    fn hybrid_branch_norms_absent_on_a_non_hybrid_layer() {
        let ops = base_ops();
        for role in [
            OperandRole::PreExpertsNorm,
            OperandRole::PostDenseFfnNorm,
            OperandRole::PostExpertsNorm,
        ] {
            assert_eq!(
                absent_op(role, &ops),
                Some("hybrid dense+routed FFN (judged semantics)"),
                "{role:?}"
            );
        }
        let hybrid = LayerOps {
            hybrid: true,
            ..base_ops()
        };
        assert_eq!(absent_op(OperandRole::PreExpertsNorm, &hybrid), None);
    }

    #[test]
    fn router_bias_absent_when_the_judgment_declares_none() {
        let no_bias = LayerOps {
            routed: true,
            moe: Some(moe(
                MoeRouterKind::TopKSoftmax,
                false,
                ExpertFormat::PackedMxfp4,
            )),
            ..base_ops()
        };
        assert_eq!(
            absent_op(OperandRole::MoeRouterBias, &no_bias),
            Some("router bias (declared by the routed-FFN judgment)")
        );
        let with_bias = LayerOps {
            routed: true,
            moe: Some(moe(
                MoeRouterKind::TopKSoftmax,
                true,
                ExpertFormat::PackedMxfp4,
            )),
            ..base_ops()
        };
        assert_eq!(absent_op(OperandRole::MoeRouterBias, &with_bias), None);
    }

    #[test]
    fn expert_scale_streams_absent_when_the_format_carries_none() {
        let unsplit = LayerOps {
            routed: true,
            moe: Some(moe(
                MoeRouterKind::TopKSoftmax,
                true,
                ExpertFormat::PerExpert,
            )),
            ..base_ops()
        };
        for role in [
            OperandRole::ExpertGateUpScales,
            OperandRole::ExpertDownScales,
        ] {
            assert_eq!(
                absent_op(role, &unsplit),
                Some("a scaled expert format (this format carries no separate scales)"),
                "{role:?}"
            );
        }
        let split = LayerOps {
            routed: true,
            moe: Some(moe(
                MoeRouterKind::TopKSoftmax,
                true,
                ExpertFormat::PackedMxfp4,
            )),
            ..base_ops()
        };
        assert_eq!(absent_op(OperandRole::ExpertGateUpScales, &split), None);
    }

    // ── expected_shape: Gated DeltaNet + MoE geometry ────────────────

    fn base_geometry(linear: Option<LinearAttentionSurface>) -> StackGeometry {
        StackGeometry {
            kda: None,
            mla: None,
            mamba2: None,
            hidden: 64,
            q_rows: 32,
            kv_rows: 16,
            intermediate: 128,
            head_dim: 8,
            num_q_heads: 4,
            num_kv_heads: 2,
            qk_scope: larql_models::config::QkNormScope::PerHead,
            // Ordinary width: a fused query/gate projection is twice this,
            // and the fixtures that exercise that say so themselves.
            q_proj_rows: 32,
            linear,
        }
    }

    fn linear_surface() -> LinearAttentionSurface {
        // Qwen3.8's own geometry — see gated_delta.rs's state_elements()
        // test for why real numbers, not placeholders.
        LinearAttentionSurface {
            key_heads: 16,
            key_head_dim: 128,
            value_heads: 48,
            value_head_dim: 128,
            conv_kernel: 4,
            state_dtype: Some(larql_models::inventory::report::RecurrentStateDtype::Float32),
        }
    }

    #[test]
    fn full_projection_qk_norm_shape_is_unpinned() {
        let g = StackGeometry {
            qk_scope: larql_models::config::QkNormScope::FullProjection,
            ..base_geometry(None)
        };
        assert_eq!(expected_shape(OperandRole::AttnQNorm, &g, None), None);
    }

    #[test]
    fn linear_attention_shapes_follow_the_recurrence_geometry_not_the_softmax_fields() {
        let l = linear_surface();
        let g = base_geometry(Some(l));
        assert_eq!(
            expected_shape(OperandRole::LinearAttnInProjQkv, &g, None),
            Some(vec![l.qkv_channels(), g.hidden])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnInProjA, &g, None),
            Some(vec![l.value_heads, g.hidden])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnInProjB, &g, None),
            Some(vec![l.value_heads, g.hidden])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnInProjZ, &g, None),
            Some(vec![l.value_width(), g.hidden])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnConv1d, &g, None),
            Some(vec![l.qkv_channels(), 1, l.conv_kernel])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnALog, &g, None),
            Some(vec![l.value_heads])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnDtBias, &g, None),
            Some(vec![l.value_heads])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnNorm, &g, None),
            Some(vec![l.value_head_dim])
        );
        assert_eq!(
            expected_shape(OperandRole::LinearAttnOutProj, &g, None),
            Some(vec![g.hidden, l.value_width()])
        );
    }

    #[test]
    fn linear_attention_operands_have_no_shape_contract_without_a_declared_recurrence() {
        // `linear` absent while such an operand exists is a refusal, not
        // a waiver — every LinearAttn* role must fall through to `None`
        // via the `linear?` short-circuit, never invent a shape from the
        // softmax fields.
        let g = base_geometry(None);
        for role in [
            OperandRole::LinearAttnInProjQkv,
            OperandRole::LinearAttnInProjA,
            OperandRole::LinearAttnInProjZ,
            OperandRole::LinearAttnConv1d,
            OperandRole::LinearAttnALog,
            OperandRole::LinearAttnNorm,
            OperandRole::LinearAttnOutProj,
        ] {
            assert_eq!(expected_shape(role, &g, None), None, "{role:?}");
        }
    }

    #[test]
    fn moe_router_and_expert_shapes_follow_the_judged_geometry() {
        let g = base_geometry(None);
        let m = moe(MoeRouterKind::TopKSoftmax, true, ExpertFormat::PerExpert);
        assert_eq!(
            expected_shape(OperandRole::MoeRouterPerExpertScale, &g, Some(&m)),
            Some(vec![m.experts])
        );
        assert_eq!(
            expected_shape(OperandRole::MoeRouterWeight, &g, Some(&m)),
            Some(vec![m.experts, g.hidden])
        );
        assert_eq!(
            expected_shape(OperandRole::MoeRouterBias, &g, Some(&m)),
            Some(vec![m.experts])
        );
        // PerExpert/PackedBF16 keep the unpacked [experts, rows, k] shape
        // and a bare [experts, rows] scales stream (packed_shape/
        // scales_shape's non-MXFP4 arm).
        assert_eq!(
            expected_shape(OperandRole::ExpertGateUp, &g, Some(&m)),
            Some(vec![
                m.experts,
                FUSED_BRANCHES * m.expert_intermediate_size,
                g.hidden
            ])
        );
        assert_eq!(
            expected_shape(OperandRole::ExpertGateUpBias, &g, Some(&m)),
            Some(vec![m.experts, FUSED_BRANCHES * m.expert_intermediate_size])
        );
        assert_eq!(
            expected_shape(OperandRole::ExpertDown, &g, Some(&m)),
            Some(vec![m.experts, g.hidden, m.expert_intermediate_size])
        );
        assert_eq!(
            expected_shape(OperandRole::ExpertDownBias, &g, Some(&m)),
            Some(vec![m.experts, g.hidden])
        );
        let split = moe(MoeRouterKind::TopKSoftmax, true, ExpertFormat::PackedMxfp4);
        assert_eq!(
            expected_shape(OperandRole::ExpertGateUpScales, &g, Some(&split)),
            Some(scales_shape(
                &split,
                FUSED_BRANCHES * split.expert_intermediate_size,
                g.hidden
            ))
        );
        assert_eq!(
            expected_shape(OperandRole::ExpertDownScales, &g, Some(&m)),
            Some(scales_shape(&m, g.hidden, m.expert_intermediate_size))
        );
    }
}
