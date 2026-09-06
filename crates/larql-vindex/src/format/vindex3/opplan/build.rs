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

use larql_models::config::{
    FfnType, HyperConnection, HyperConnectionWeights, MoeRouterKind, ResidualTopology,
};

use super::super::encode::segment::{read_segment_header, SegmentTensor};
use super::super::encode::REPRESENTATION_ID_SEP;
use super::super::graph::policy::LayerOperator;
use super::super::graph::roles::{
    classify_hyper_connection_head_tensor, classify_stack_tensor_on, HcHeadOperand,
};
use super::super::graph::surface::LinearAttentionSurface;
use super::super::graph::surface::Mamba2Surface;
use super::super::graph::surface::MlaSurface;
use super::super::graph::surface::MoeSurface;
use super::super::graph::{LogicalObject, NormPlacement, ObjectKind, OperandRole};
use super::super::inspect::SystemInspection;
use super::exec::hyper_connection::{HC_HEAD_SCALE_LEN, HC_SCALE_LEN};
use super::{
    AttentionOp, ClosureDefect, ComponentOpPlan, EmbeddingOp, ExpertBank, FfnIdentity, FfnOp,
    GateOp, GatedDeltaOp, HcSiteOp, HybridFfnOp, HyperConnectionHeadOp, HyperConnectionLayerOp,
    KdaOp, LayerAttention, LayerFfn, LayerPlan, Mamba2Op, MlaOp, NormOp, OpPlanOutcome, OperandRef,
    OutputOp, PackedProjection, QkNormOp, RoutedFfnOp, SharedExpertBranchGateOp, SharedExpertOp,
    SinkOp,
};
use crate::error::VindexError;
use larql_models::config::ExpertFormat;

/// The post-norm epsilon, named as [`ClosureDefect::UnjudgedSemantic`]
/// reports it.
const POST_NORM_EPS_FACT: &str = "post-norm epsilon";
/// The shape OLMo-2, OLMo-3 and EXAONE-4 declare, named as the surface
/// names it.
const POST_ONLY_PLACEMENT_FACT: &str =
    "post-norm placement (the norm applies to the sublayer's output)";
/// The structure that makes the post-norm epsilon load-bearing.
const FOUR_NORM_PLACEMENT: &str = "four-norm placement";
/// The routed-FFN op, as the requirer of its judged facts.
const ROUTED_FFN_OP: &str = "routed FFN op";
/// A packed fused operand with no declared branch layout cannot be read.
const GATE_UP_LAYOUT_FACT: &str = "gate_up branch layout";
/// The primitive a hyper-connection site operand implies, named as
/// [`ClosureDefect::OperandImpliesAbsentOp`] reports it on a component
/// whose declared residual is one stream.
const HC_SITE_ON_SINGLE_STREAM: &str =
    "hyper-connection residual topology (the component declares a single residual stream)";
/// The same operand on a mixer-only or conv-QKV layer: the block has no
/// attention and FFN sublayers to wrap in two sites, and this build has
/// judged no hyper-connected form of it.
const HC_SITE_ON_MIXER_LAYER: &str =
    "a hyper-connected transformer layer (a mixer-only program has no attention and FFN \
     sublayers to wrap in two sites)";
/// The fact a hyper-connected component's mixer-only layer leaves
/// unjudged, named as [`ClosureDefect::UnjudgedSemantic`] reports it.
const HC_ON_MIXER_FACT: &str = "hyper-connection sites on a mixer-only layer";
/// What requires that judgment: the traversal, which has to know how a
/// one-sublayer block reduces and expands the bundle.
const HC_ON_MIXER_REQUIRED_BY: &str = "the hyper-connected residual traversal";
/// The primitive the head's operands imply on a single-stream component.
const HC_HEAD_ON_SINGLE_STREAM: &str =
    "hyper-connection head reduction (the component declares a single residual stream)";

/// Build the operation plan for `component_id` from a container's
/// inspection plus its segment tables. I/O failures are hard errors;
/// every semantic shortfall is a [`ClosureDefect`].
/// Whether the surface declares `layer` routed: `None` when the surface
/// carries no MoE judgment at all, `Some(false)` inside the dense prefix
/// the surface names (`dense_prefix_layers`), `Some(true)` otherwise.
/// The one derivation both the closure pass and the plan construction
/// read, so they cannot disagree about which layers are routed.
fn declared_routed(moe: Option<&MoeSurface>, layer: usize) -> Option<bool> {
    moe.map(|m| m.dense_prefix_layers.is_none_or(|prefix| layer >= prefix))
}

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
    // Representable, and explicitly NOT executable.
    //
    // The container states this placement exactly; what is missing is the
    // op set. `LayerPlan::pre_attention_norm` is a required `NormOp` that
    // every executor reads BEFORE its sublayer, and a post-norm stack has
    // no such operand — the norm it does carry belongs after the sublayer
    // and before the residual add. Lowering it as `PreOnly` would find
    // operands for both sites (the names collide: this family's
    // `post_attention_layernorm` is a true post-norm where a Llama stack's
    // is the pre-FFN norm) and would run, applying each norm to the wrong
    // tensor and producing fluent wrong output. So it refuses, the way the
    // unimplemented router kinds and position policies refuse.
    // The residual topology is asked FIRST, because it decides what the
    // residual even is — and since wave 18 it is asked as a CLOSURE
    // question here, not as a refusal. A hyper-connected component's six
    // per-layer site operands classify, are required on every
    // transformer layer, are checked against the topology's own geometry
    // and are bound into the plan; a single-stream component refuses the
    // same operands as strays. Since wave 19 both traversals carry the
    // bundle; what a hyper-connected component still cannot do is said
    // by name where traversal starts (`exec::prepared::PreparedOperands::load`:
    // a whole-stack image with no head, a layer scale under the topology)
    // and in the plan report, from the same facts. Refusing here would
    // hide the addressability answer behind an execution gap — the
    // structural silence the wave-18 baseline recorded, where a
    // hyper-connection checkpoint emitted no UnclassifiedOperand because
    // nothing ever asked.
    let hyper_connection = match surface.residual_topology {
        ResidualTopology::HyperConnection(hc) => Some(hc),
        ResidualTopology::SingleStream => None,
    };
    if let Some(reason) = placement.unimplemented_reason() {
        return Ok(OpPlanOutcome {
            plan: None,
            defects: vec![ClosureDefect::UnimplementedSemantic {
                component: component.id.clone(),
                fact: format!("{POST_ONLY_PLACEMENT_FACT} — {reason}"),
                representable_as: format!(
                    "NormPlacement::{placement:?}, from the operand evidence"
                ),
            }],
        });
    }
    // A four-norm stack executes two norms whose epsilon nothing else
    // supplies. `Shared` and a declared value are both judgments;
    // absence is not — and inheriting `eps` here would build exactly the
    // executable-but-unfounded program this refuses. Returning no plan
    // means no unjudged epsilon is ever written into one.
    let post_norm: Option<larql_models::config::NormSpec> = match placement {
        NormPlacement::PreOnly | NormPlacement::PreMixer => None,
        // A post-norm stack runs TWO norms whose epsilon nothing else
        // supplies, exactly as a four-norm stack does — and it has no
        // pre-norm site to borrow from, so the requirement is if anything
        // sharper here.
        NormPlacement::PostOnly | NormPlacement::PrePost => match surface.norm.post {
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
                | ObjectKind::HyperConnectionHead
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
    // The mixer and the hybrid's conv-QKV block are the two judged
    // layer programs with no FFN — the same rule the graph-level
    // completeness check states, restated here because this closure gate
    // is the one execution actually passes through (found live: the
    // 2.7B hybrid declares no FFN anywhere and this line still demanded
    // the surface).
    let has_ffn_layer = attention_table
        .iter()
        .any(|l| !l.operator.is_mamba2() && !l.operator.is_conv_qkv());
    let runs_mamba2 = attention_table.iter().any(|l| l.operator.is_mamba2());
    // A dense FFN needs a dense width. A wholly-routed component has
    // none and plans no dense layer, so the fact is required exactly when
    // some layer will run one: no routed judgment at all, a routed
    // judgment with a declared dense prefix, or Gemma 4's hybrid, where
    // both branches run every layer.
    let runs_dense_ffn = has_ffn_layer
        && ffn_surface.is_some_and(|f| {
            f.moe
                .is_none_or(|m| m.hybrid || m.dense_prefix_layers.unwrap_or(0) > 0)
        });
    for (runs, present, fact) in [
        (attends, attn.is_some(), "attention surface"),
        (has_ffn_layer, ffn_surface.is_some(), "ffn surface"),
        (
            runs_dense_ffn,
            ffn_surface.is_some_and(|f| f.intermediate_size.is_some()),
            "ffn.intermediate_size (a dense FFN layer's width)",
        ),
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
    // The DENSE width, absent on a wholly-routed stack. Zero-filled only
    // where it feeds `StackGeometry`, whose convention for a fact this
    // component does not have is already zero (see the attention
    // geometry above); the dense FFN op reads the checked value.
    let inter = ffn_surface.and_then(|f| f.intermediate_size);
    // A derived static-shard container declares each layer's dense width.
    // The declaration must cover every layer and name a width the
    // component can hold; otherwise the plan refuses here, before any
    // layer's tensors are shaped against it.
    if let Some(widths) = ffn_surface.and_then(|f| f.intermediate_size_by_layer.as_ref()) {
        let refuse = |detail: String| ClosureDefect::FfnWidthDeclaration {
            component: component.id.clone(),
            detail,
        };
        if widths.len() != component.num_layers {
            defects.push(refuse(format!(
                "declares {} per-layer FFN widths for a {}-layer component",
                widths.len(),
                component.num_layers
            )));
        }
        for (layer, &width) in widths.iter().enumerate() {
            if width == 0 || inter.is_some_and(|dense| width > dense) {
                defects.push(refuse(format!(
                    "layer {layer} declares FFN width {width} against a dense width of {}",
                    inter.map_or_else(|| "none".to_string(), |d| d.to_string())
                )));
            }
        }
    }
    // The width a layer's FFN op runs at: the declared per-layer value
    // when the container carries one, else the component's dense width.
    let inter_for = |layer: usize| -> Option<usize> {
        ffn_surface
            .and_then(|f| f.intermediate_size_by_layer.as_ref())
            .and_then(|widths| widths.get(layer).copied())
            .or(inter)
    };
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
            intermediate: inter_for(layer).unwrap_or(0),
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
            conv_qkv: surface.conv_qkv,
            hyper_connection,
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
                sinks: attn.is_some_and(|a| a.sinks.is_some()),
                routed,
                hybrid,
                moe: ffn_moe,
                mamba2: surface.mamba2,
                conv_qkv: surface.conv_qkv,
                v_from_k: policy.v_from_k,
                hyper_connection: hyper_connection.is_some(),
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
    // ── The hyper-connection head ──
    let hc_head_tensors =
        tables
            .get(&ObjectKind::HyperConnectionHead)
            .and_then(|(object, tensors)| {
                hyper_connection_head_closure(
                    object,
                    tensors,
                    hyper_connection,
                    hidden,
                    &mut defects,
                )
            });

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
                declared_norm_eps: surface.norm.pre.eps,
                layer,
                pre_attention_norm: Some(norm_op(
                    surface.norm.pre,
                    &stack_id,
                    get(OperandRole::Mamba2PreMixerNorm),
                )),
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
                hyper_connection: None,
                residual_scale: surface.residual_scale,
                operands_accounted: consumed,
                operands_present: consumed,
            });
            continue;
        }
        // A conv-QKV attention layer, on operand evidence: its fused QKV
        // `in_proj` role is the discriminator (only the conv-QKV table
        // can put it in `slot`). Its whole program is the block — one
        // pre-block norm, no attention wrap, no FFN — the same shape as
        // the mixer arm above.
        if slot.contains_key(&OperandRole::ConvQkvInProj) {
            let attn_geometry = surface.conv_qkv.unwrap_or_else(|| {
                panic!(
                    "layer {layer} ships a conv-QKV operand while the component declares no \
                     conv-QKV surface; closure should have refused this before the plan was built"
                )
            });
            let consumed = slot.len();
            layers.push(LayerPlan {
                declared_norm_eps: surface.norm.pre.eps,
                layer,
                pre_attention_norm: Some(norm_op(
                    surface.norm.pre,
                    &stack_id,
                    get(OperandRole::Mamba2PreMixerNorm),
                )),
                attention: LayerAttention::ConvQkv(Box::new(super::conv_qkv::ConvQkvOp {
                    geometry: attn_geometry,
                    residual_in_fp32: surface.residual_in_fp32,
                    in_proj: operand(&stack_id, get(OperandRole::ConvQkvInProj)),
                    conv1d: operand(&stack_id, get(OperandRole::ConvQkvConv1d)),
                    conv1d_bias: slot
                        .get(&OperandRole::ConvQkvConv1dBias)
                        .map(|t| operand(&stack_id, t)),
                    out_proj: operand(&stack_id, get(OperandRole::ConvQkvOutProj)),
                })),
                post_attention_norm: None,
                pre_ffn_norm: None,
                ffn: None,
                post_ffn_norm: None,
                layer_scale: None,
                hyper_connection: None,
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
        // Which norm operand feeds each site. `None` for the pre-FFN
        // slot means the FFN reads the raw residual — the post-norm
        // program — and is not the same as a missing operand.
        // Which pre-sublayer norm operands this placement HAS. `None`
        // means the site does not exist, and the operand is not there to
        // be fetched — a post-norm stack ships neither.
        let pre_attn_role = match placement {
            NormPlacement::PostOnly => None,
            _ => Some(OperandRole::PreAttentionNorm),
        };
        let (post_attention_norm, pre_ffn_role, post_ffn_norm) = match placement {
            NormPlacement::PrePost => {
                let spec = post_norm.expect("PrePost resolves or returns above");
                (
                    Some(norm_op(
                        spec,
                        &stack_id,
                        get(OperandRole::PostAttentionNorm),
                    )),
                    Some(OperandRole::PreFfnNorm),
                    Some(norm_op(spec, &stack_id, get(OperandRole::PostFfnNorm))),
                )
            }
            // Both wrap norms, no pre-FFN norm: each sublayer reads the
            // raw residual and its OUTPUT is normalised before the add.
            NormPlacement::PostOnly => {
                let spec = post_norm.expect("PostOnly resolves or returns above");
                (
                    Some(norm_op(
                        spec,
                        &stack_id,
                        get(OperandRole::PostAttentionNorm),
                    )),
                    None,
                    Some(norm_op(spec, &stack_id, get(OperandRole::PostFfnNorm))),
                )
            }
            NormPlacement::PreOnly => (None, Some(OperandRole::PostAttentionNorm), None),
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
            intermediate_size: inter_for(layer).unwrap_or_else(|| {
                panic!(
                    "component {} plans a dense FFN layer with no declared dense width; \
                     closure should have refused this before the plan was built",
                    component.id
                )
            }),
            activation: ffn_s.activation,
            gate_policy: ffn_s.gate_policy,
            gate: gated_ffn.then(|| operand(&stack_id, get(OperandRole::FfnGate))),
            up: operand(&stack_id, get(OperandRole::FfnUp)),
            down: operand(&stack_id, get(OperandRole::FfnDown)),
        };
        let ffn = match (ffn_moe, bank_slot) {
            // A declared-dense prefix layer plans dense even if stray bank
            // tensors exist (the mismatch defect above already refuses
            // the plan); a declared-routed layer with no bank plans dense
            // only as a placeholder behind its own defect.
            (Some(moe), Some(bank)) if declared_routed(ffn_moe.as_ref(), layer) == Some(true) => {
                let moe =
                    resolve_shared_expert_width(moe, slot.get(&OperandRole::SharedExpertGate));
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
                let shared = shared_expert_width(&moe).map(|width| SharedExpertOp {
                    intermediate_size: width,
                    activation: ffn_s.activation,
                    gate_policy: ffn_s.gate_policy,
                    gate: operand(&stack_id, get(OperandRole::SharedExpertGate)),
                    up: operand(&stack_id, get(OperandRole::SharedExpertUp)),
                    down: operand(&stack_id, get(OperandRole::SharedExpertDown)),
                    branch_gate: moe.shared_expert_gate.map(|spec| SharedExpertBranchGateOp {
                        spec,
                        weight: operand(&stack_id, get(OperandRole::SharedExpertBranchGate)),
                    }),
                });
                let routed = RoutedFfnOp {
                    experts: moe.experts,
                    top_k: moe.top_k,
                    expert_intermediate_size: moe.expert_intermediate_size,
                    router_kind: moe.router_kind,
                    routing_policy: moe.routing_policy,
                    branch_scale: moe.branch_scale,
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
        // The two sites, bound iff the component declares the topology.
        // Built HERE, in the transformer arm, and not above the mixer
        // arms: closure required all six on every transformer layer, so
        // the lookups are total for this layer kind — and a mixer layer
        // under the topology never reaches this point, because closure
        // refused it as unjudged. Binding eagerly for every layer kind
        // would turn that refusal's absence into an index panic instead
        // of the named defect.
        let hc_site = |mix_fn: OperandRole, base: OperandRole, scale: OperandRole| HcSiteOp {
            mix_fn: operand(&stack_id, get(mix_fn)),
            base: operand(&stack_id, get(base)),
            scale: operand(&stack_id, get(scale)),
        };
        let hyper_connection_sites = hyper_connection.map(|_| HyperConnectionLayerOp {
            attention: hc_site(
                OperandRole::HcAttnMixFn,
                OperandRole::HcAttnBase,
                OperandRole::HcAttnScale,
            ),
            ffn: hc_site(
                OperandRole::HcFfnMixFn,
                OperandRole::HcFfnBase,
                OperandRole::HcFfnScale,
            ),
        });
        let consumed = slot.len() + bank_slot.map_or(0, |b| b.len());
        layers.push(LayerPlan {
            declared_norm_eps: surface.norm.pre.eps,
            layer,
            pre_attention_norm: pre_attn_role
                .map(|role| norm_op(surface.norm.pre, &stack_id, get(role))),
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
                    kv_a_norm_eps: m.kv_a_norm_eps,
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
            // Absent under post-norm placement: the FFN reads the raw
            // residual there, and `post_ffn_norm` carries the site that
            // does exist.
            pre_ffn_norm: pre_ffn_role.map(|role| norm_op(surface.norm.pre, &stack_id, get(role))),
            ffn: Some(ffn),
            post_ffn_norm,
            layer_scale,
            hyper_connection: hyper_connection_sites,
            residual_scale: surface.residual_scale,
            operands_accounted: consumed,
            operands_present: consumed,
        });
    }

    let plan = ComponentOpPlan {
        component: component.id.clone(),
        residual_topology: surface.residual_topology,
        hyper_connection_head: hc_head_tensors.map(|(object, reduce_fn, base, scale)| {
            HyperConnectionHeadOp {
                reduce_fn: operand(&object, &reduce_fn),
                base: operand(&object, &base),
                scale: operand(&object, &scale),
            }
        }),
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

/// Closure over the hyper-connection head object: every tensor classifies
/// as one of the head's three operands, each operand is present exactly
/// once, each has the head's geometry, and the component declares the
/// topology the head reduces. Returns the three tensors for binding when
/// all of that holds; every shortfall is itemised into `defects`.
///
/// The head's geometry is deliberately NOT a site's — `[hc, hc·hidden]`
/// against `[(2 + hc)·hc, hc·hidden]`, `[1]` against `[3]` — because
/// `ParallelHead.hc_head` runs no Sinkhorn (see
/// [`super::exec::hyper_connection::head_reduce`]). A checkpoint that
/// stored a site's operands under the head's names fails here rather
/// than binding a split into an operation that has none.
fn hyper_connection_head_closure(
    object: &LogicalObject,
    tensors: &[SegmentTensor],
    hyper_connection: Option<HyperConnection>,
    hidden: usize,
    defects: &mut Vec<ClosureDefect>,
) -> Option<(String, SegmentTensor, SegmentTensor, SegmentTensor)> {
    // The graph only places this object under the declaration, so this
    // arm states the invariant for a container whose graph was edited
    // rather than built; it is the operand-level form of the same
    // disagreement the builder refuses by name.
    let Some(hc) = hyper_connection else {
        for tensor in tensors {
            defects.push(ClosureDefect::OperandImpliesAbsentOp {
                object: object.id.clone(),
                tensor: tensor.name.clone(),
                required_primitive: HC_HEAD_ON_SINGLE_STREAM.to_string(),
            });
        }
        return None;
    };
    let mut bound: BTreeMap<HcHeadOperand, SegmentTensor> = BTreeMap::new();
    for tensor in tensors {
        let Some(role) = classify_hyper_connection_head_tensor(&tensor.name) else {
            defects.push(ClosureDefect::UnclassifiedOperand {
                object: object.id.clone(),
                tensor: tensor.name.clone(),
            });
            continue;
        };
        let expected = match role {
            HcHeadOperand::ReduceFn => vec![hc.streams, hc.streams * hidden],
            HcHeadOperand::Base => vec![hc.streams],
            HcHeadOperand::Scale => vec![HC_HEAD_SCALE_LEN],
        };
        if !shape_satisfies(&tensor.shape, &expected) {
            defects.push(ClosureDefect::GeometryMismatch {
                tensor: format!("{}/{}", object.id, tensor.name),
                expected,
                actual: tensor.shape.clone(),
            });
        }
        if bound.insert(role, tensor.clone()).is_some() {
            defects.push(ClosureDefect::ObjectShape {
                object: object.id.clone(),
                detail: format!("two operands claim the head's {role:?}"),
            });
        }
    }
    for role in [
        HcHeadOperand::ReduceFn,
        HcHeadOperand::Base,
        HcHeadOperand::Scale,
    ] {
        if !bound.contains_key(&role) {
            defects.push(ClosureDefect::ObjectShape {
                object: object.id.clone(),
                detail: format!("no operand for the head's {role:?}"),
            });
        }
    }
    let (Some(reduce_fn), Some(base), Some(scale)) = (
        bound.remove(&HcHeadOperand::ReduceFn),
        bound.remove(&HcHeadOperand::Base),
        bound.remove(&HcHeadOperand::Scale),
    ) else {
        return None;
    };
    Some((object.id.clone(), reduce_fn, base, scale))
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
    /// The conv-QKV attention geometry, on a component that declares one
    /// — what decides the conv-bias operand requirement on a hybrid
    /// attention layer.
    conv_qkv: Option<larql_models::config::ConvQkvAttnGeometry>,
    /// V is the K projection on this layer: no V operand is required, and
    /// one present is a stray.
    v_from_k: bool,
    /// The component declares the Sinkhorn hyper-connection topology, so
    /// every transformer layer must supply its two sites' six operands —
    /// and a single-stream component refuses the same six as strays.
    hyper_connection: bool,
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
    // A conv-QKV attention layer likewise: the hybrid lineage wraps it
    // in ONE pre-mixer norm (no attention wrap, no FFN), and its operand
    // set is its own — fused QKV, the conv over it, the output
    // projection. The declared bias flags decide the bias operands the
    // same way the mixer's do; the observed checkpoint declares all
    // three projections bias-free and the conv biased.
    if ops.operator.is_conv_qkv() {
        let mut roles = vec![
            OperandRole::Mamba2PreMixerNorm,
            OperandRole::ConvQkvInProj,
            OperandRole::ConvQkvConv1d,
            OperandRole::ConvQkvOutProj,
        ];
        if let Some(attn) = ops.conv_qkv {
            // The conv-bias switch is the mixer's `use_conv_bias` — one
            // flag governs both block kinds' convs in this lineage.
            if ops.mamba2.is_some_and(|m| m.geometry.use_conv_bias) {
                roles.push(OperandRole::ConvQkvConv1dBias);
            }
            // qkv_bias / out_bias operands have no roles yet: the
            // observed checkpoint declares both false, and a role must
            // be judged from a real instance, not invented ahead of one.
            let _ = attn;
        }
        return roles;
    }
    // The attention block's two trunk norms, required per placement. A
    // post-norm stack has no pre-attention norm to require, and
    // requiring one would report a missing operand for a tensor the
    // checkpoint correctly never shipped.
    let mut roles = if ops.placement == NormPlacement::PostOnly {
        vec![OperandRole::PostAttentionNorm]
    } else {
        vec![
            OperandRole::PreAttentionNorm,
            OperandRole::PostAttentionNorm,
        ]
    };
    // Two sites per hyper-connected layer, three operands each, on EVERY
    // transformer layer of the component — DeepSeek-V4 and GLM-5.3-Flash
    // both carry all six on all 43 / 45 layers. A layer missing one is
    // not a partially hyper-connected layer; it is a bundle the traversal
    // cannot reduce or expand at that site.
    if ops.hyper_connection {
        roles.extend([
            OperandRole::HcAttnMixFn,
            OperandRole::HcAttnBase,
            OperandRole::HcAttnScale,
            OperandRole::HcFfnMixFn,
            OperandRole::HcFfnBase,
            OperandRole::HcFfnScale,
        ]);
    }
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
    match ops.placement {
        NormPlacement::PrePost => {
            roles.push(OperandRole::PreFfnNorm);
            roles.push(OperandRole::PostFfnNorm);
        }
        // The FFN reads the raw residual; only its output is normed.
        NormPlacement::PostOnly => roles.push(OperandRole::PostFfnNorm),
        NormPlacement::PreOnly | NormPlacement::PreMixer => {}
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
                // Paired with the judgment, both ways: a declared gate
                // must find its operand, and the `absent_op` arm below
                // refuses the operand where no gate is declared.
                if moe.shared_expert_gate.is_some() {
                    roles.push(OperandRole::SharedExpertBranchGate);
                }
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
        // A site operand on a component whose residual is ONE vector: the
        // declaration and the estate disagree, and the operand implies an
        // operation the surface never declared. Paired with
        // `required_roles`, which demands all six iff the topology is
        // declared, so presence and requirement cannot desync.
        OperandRole::HcAttnMixFn
        | OperandRole::HcAttnBase
        | OperandRole::HcAttnScale
        | OperandRole::HcFfnMixFn
        | OperandRole::HcFfnBase
        | OperandRole::HcFfnScale
            if !ops.hyper_connection =>
        {
            Some(HC_SITE_ON_SINGLE_STREAM)
        }
        // The same operand on a one-sublayer block, under the topology:
        // no judged form exists, so it is a stray rather than a site.
        OperandRole::HcAttnMixFn
        | OperandRole::HcAttnBase
        | OperandRole::HcAttnScale
        | OperandRole::HcFfnMixFn
        | OperandRole::HcFfnBase
        | OperandRole::HcFfnScale
            if ops.operator.is_mamba2() || ops.operator.is_conv_qkv() =>
        {
            Some(HC_SITE_ON_MIXER_LAYER)
        }
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
        OperandRole::SharedExpertBranchGate
            if ops
                .moe
                .is_some_and(|m| m.shared_experts == 0 || m.shared_expert_gate.is_none()) =>
        {
            Some("a gate on the shared-expert branch (the judgment declares none, so the branch is summed unscaled)")
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
        // Paired the other way: a post-norm stack refuses the two
        // pre-sublayer norms, so a checkpoint shipping one under this
        // placement is a disagreement rather than a spare tensor.
        OperandRole::PreAttentionNorm | OperandRole::PreFfnNorm
            if ops.placement == NormPlacement::PostOnly =>
        {
            Some("a pre-sublayer norm (post-norm placement normalises each sublayer's output)")
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
    /// The hybrid's conv-QKV attention geometry, on a component whose
    /// full layers run it.
    conv_qkv: Option<larql_models::config::ConvQkvAttnGeometry>,
    /// The declared hyper-connection topology, whose stream count sizes
    /// the site operands: `[(2 + hc)·hc, hc·hidden]`, `[(2 + hc)·hc]`,
    /// `[3]`. `None` on a single-stream component, where a site operand
    /// is refused by `absent_op` before its shape is ever asked.
    hyper_connection: Option<HyperConnection>,
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
        conv_qkv,
        hyper_connection,
    } = *g;
    match role {
        // Sinkhorn hyper-connection sites. Every contract follows from the
        // component's DECLARED stream count closing over its width — the
        // same derivation the executor's `mix_rows_for` runs — and none
        // from the tensor: a `[2·hc, hc·hidden]` Sinkhorn-free operand
        // (Hy4-preview's shape) fails here rather than binding to a split
        // it does not parameterise. `hyper_connection` absent while such
        // an operand exists is unreachable past `absent_op`, and answers
        // `None` for the same reason the other family absences do.
        OperandRole::HcAttnMixFn | OperandRole::HcFfnMixFn => {
            let hc = hyper_connection?;
            Some(vec![
                HyperConnectionWeights::mix_rows_for(hc.streams),
                hc.streams * hidden,
            ])
        }
        OperandRole::HcAttnBase | OperandRole::HcFfnBase => {
            Some(vec![HyperConnectionWeights::mix_rows_for(
                hyper_connection?.streams,
            )])
        }
        OperandRole::HcAttnScale | OperandRole::HcFfnScale => Some(vec![HC_SCALE_LEN]),
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
        // Conv-QKV attention. Every contract follows from the hybrid
        // block's own declared geometry; `conv_qkv` absent while such an
        // operand exists is a refusal, for the same reason the other
        // family absences are.
        OperandRole::ConvQkvInProj => Some(vec![conv_qkv?.qkv_rows(), hidden]),
        OperandRole::ConvQkvConv1d => {
            let a = conv_qkv?;
            Some(vec![a.qkv_rows(), 1, a.conv_kernel])
        }
        OperandRole::ConvQkvConv1dBias => Some(vec![conv_qkv?.qkv_rows()]),
        OperandRole::ConvQkvOutProj => Some(vec![hidden, conv_qkv?.attn_out_width()]),
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
        // routed expert, at the width the judgment declares. Sized from
        // `shared_expert_intermediate_size` and NOT re-derived here —
        // Kimi's `KimiSparseMoeBlock.__init__` sizes one wider `KimiMLP`
        // at `moe_intermediate_size * num_shared_experts` while Qwen's
        // block sizes it from its own key, and Qwen1.5-MoE's two answers
        // differ fourfold (5632 declared against 1408 derived).
        OperandRole::SharedExpertGate | OperandRole::SharedExpertUp => {
            Some(vec![shared_expert_width(moe?)?, hidden])
        }
        OperandRole::SharedExpertDown => Some(vec![hidden, shared_expert_width(moe?)?]),
        // The scalar that gates the branch: one logit per token.
        OperandRole::SharedExpertBranchGate => Some(vec![SCALAR_GATE_ROWS, hidden]),
    }
}

/// Gate and up: the two branches sharing one fused operand.
const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// The shared-expert branch gate projects to ONE logit per token, so its
/// operand carries a single row. Named rather than written as a literal
/// `1`, which at that position would read as a placeholder.
const SCALAR_GATE_ROWS: usize = 1;

/// The width of the always-active shared branch, or `None` when the
/// judgment declares no shared expert.
///
/// The single place this build answers the question. The two lineages
/// size the branch differently and the architecture already resolved
/// which applies (`ModelArchitecture::shared_expert_intermediate_size`);
/// re-deriving it here from the routed width would put a second answer
/// beside that one, and on Qwen1.5-MoE the two differ fourfold. A graph
/// that declares the branch by count alone has had its width filled from
/// the stored gate tensor by [`resolve_shared_expert_width`] before this
/// is asked, so a declared branch never answers `None` here.
fn shared_expert_width(moe: &MoeSurface) -> Option<usize> {
    (moe.shared_experts > 0).then_some(moe.shared_expert_intermediate_size)?
}

/// A routed-FFN judgment whose shared branch has a width, wherever the
/// graph left it: the graph's own declaration when it carries one, else
/// the stored shape of the branch's gate tensor, `[width, hidden]`.
///
/// Graphs written before the width was part of the surface (schema 6,
/// before 2026-09-03) declare the branch by count only. Reading a missing
/// optional as "no branch" planned those models WITHOUT the shared expert
/// their graph declares and their closure counted — a silent omission
/// that every later parity and residency number would have inherited.
/// The gate tensor is the container's own authority on the width: this
/// is not a re-derivation from the routed width and assumes no lineage
/// convention. The shape check then holds up and down to the same width,
/// and refuses a branch whose three tensors disagree. A declared branch
/// whose gate is absent is refused by [`required_roles`] before any of
/// this matters.
fn resolve_shared_expert_width(moe: MoeSurface, gate: Option<&SegmentTensor>) -> MoeSurface {
    if moe.shared_experts == 0 || moe.shared_expert_intermediate_size.is_some() {
        return moe;
    }
    let Some(gate) = gate else {
        return moe;
    };
    let width = match gate.shape.as_slice() {
        [width, _hidden] => Some(*width),
        _ => None,
    };
    MoeSurface {
        shared_expert_intermediate_size: width,
        ..moe
    }
}

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
            conv_qkv: None,
            hyper_connection: false,
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
            shared_expert_intermediate_size: None,
            shared_expert_gate: None,
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
            conv_qkv: None,
            hyper_connection: None,
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
