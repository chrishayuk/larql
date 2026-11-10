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

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use larql_models::config::{FfnType, ResidualTopology};

use super::super::encode::segment::SegmentTensor;
use super::super::graph::policy::LayerOperator;
use super::super::graph::roles::{classify_stack_tensor_under, is_attention_residual_site_operand};
use super::super::graph::surface::MoeSurface;
use super::super::graph::{LogicalObject, NormPlacement, ObjectKind, OperandRole};
use super::super::inspect::SystemInspection;
use super::{ClosureDefect, FfnIdentity, OpPlanOutcome};
use crate::error::VindexError;
use larql_models::config::ExpertFormat;

mod absent;
// Names the test module reaches through `use super::*`.
#[cfg(test)]
use super::super::graph::surface::LinearAttentionSurface;
#[cfg(test)]
use larql_models::config::MoeRouterKind;

mod closures;
mod construct;
mod layer_closure;
mod roles;
mod shapes;
pub(super) use absent::*;
use closures::*;
use construct::*;
use layer_closure::*;
use roles::*;
use shapes::*;

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
/// The primitive an attention-residual site operand implies on a
/// component that declares no block size, named as
/// [`ClosureDefect::OperandImpliesAbsentOp`] reports it.
///
/// Reported from the tensor's NAME rather than from a role, because the
/// role vocabulary deliberately refuses to name these spellings without
/// the declaration — see
/// [`is_attention_residual_site_operand`](crate::format::vindex3::graph::roles::is_attention_residual_site_operand).
const ATTN_RES_SITE_WITHOUT_DECLARATION: &str =
    "attention-residual sites (the component declares no attn_res_block_size, so its residual \
     is one vector with no snapshot history for a site to read)";
/// The same operand on a mixer-only or conv-QKV layer: the block has no
/// attention and FFN sublayers to carry the topology's two sites, and
/// this build has judged no attention-residual form of one.
const ATTN_RES_SITE_ON_MIXER_LAYER: &str =
    "an attention-residual transformer layer (a mixer-only program has no attention and FFN \
     sublayers to carry the topology's two sites)";
/// `self_attn.g_proj` on a KDA layer whose component does not declare the
/// full-rank form: the operand implies a gate projection the declaration
/// never chose. Identity is declared, not inferred from operands — a
/// checkpoint does not acquire a gate form by shipping a tensor spelled
/// for it (K3-REP-GATE-1).
const KDA_FULL_RANK_GATE_UNDECLARED: &str =
    "a full-rank KDA output gate (the component declares no `use_full_rank_gate`, so its gate \
     is the low-rank g_a_proj/g_b_proj pair)";
/// The low-rank pair on a KDA layer whose component declares the full-rank
/// form: the same disagreement from the other side.
const KDA_LOW_RANK_GATE_UNDER_FULL_RANK: &str =
    "a low-rank KDA output gate (the component declares `use_full_rank_gate`, so its gate is \
     one full-rank g_proj)";
/// `self_attn.g_proj` on an MLA layer whose component declares no output
/// gate: the same spelling as the KDA full-rank gate, and on Kimi-K3 the
/// same shape, so only the declaration can say the layer gates its
/// aggregated value.
const MLA_OUTPUT_GATE_UNDECLARED: &str =
    "an MLA output gate (the component declares no `mla_use_output_gate`, so the aggregated \
     value goes to o_proj ungated)";
/// The factorised query's operands on a component that declares no
/// `q_lora_rank`. Its `q_b_proj` has the same ROW count as the `q_proj`
/// this component does build, so nothing about the tensor itself says
/// which operation it belongs to — only the declaration does.
const MLA_Q_LORA_UNDECLARED: &str =
    "a factorised MLA query (the component declares no `q_lora_rank`, so its query is one \
     dense q_proj)";
/// A dense `q_proj` on a component that DOES declare `q_lora_rank`: the
/// same disagreement from the other side. The reference's `__init__` is
/// an if/else and never constructs both.
const MLA_Q_PROJ_UNDER_Q_LORA: &str =
    "a dense MLA query projection (the component declares `q_lora_rank`, so its query is \
     q_a_proj -> q_a_layernorm -> q_b_proj)";
/// A latent-wrapper operand on a component that declares no
/// `routed_expert_hidden_size`. The wrapper's spellings are distinctive,
/// but the expert bank's stored width IS the quantity the declaration
/// decides — so a build willing to read the form off the operands would
/// be deciding it from the very thing it is meant to be judging, and a
/// checkpoint carrying one stray tensor would execute as a different
/// model (K3-LATENTMOE-1).
const MOE_LATENT_BRANCH_UNDECLARED: &str =
    "a latent routed branch (the routed-FFN judgment declares no `routed_expert_hidden_size`, \
     so the experts run at the component's hidden width)";
/// The same refusal for the norm, whose reason is one step further on:
/// there is no aggregate to normalise because there is no bottleneck to
/// aggregate inside.
const MOE_LATENT_NORM_WITHOUT_BRANCH: &str =
    "a latent routed branch (the routed-FFN judgment declares no `routed_expert_hidden_size`, \
     so there is no aggregate to normalise)";
/// `routed_expert_norm` under a declared branch whose flag is off. A
/// different refusal from the one above and deliberately worded so: the
/// bottleneck exists, and the checkpoint's own flag says the aggregate
/// reaches the up-projection unnormalised.
const MOE_LATENT_NORM_UNDER_FALSE_FLAG: &str =
    "a norm on the latent routed aggregate (the judgment declares `latent_moe_use_norm` false, \
     so the aggregate reaches the up-projection unnormalised)";
/// A declared latent width the routed branch cannot be built from. The
/// form is selected by PRESENCE — the reference tests `is not None` —
/// so a zero is a declared bottleneck of no width, and demoting it to
/// the uniform form would execute a different model than the checkpoint
/// declares.
fn moe_latent_width_degenerate(width: usize) -> String {
    format!(
        "declares a latent routed branch of width {width} (`routed_expert_hidden_size`): the \
         routed experts, their bank and both wrapper projections would all be sized from it. \
         The declaration selects the form by PRESENCE, so this is a bottleneck of no width and \
         not the uniform form"
    )
}
/// The primitive the exit pair implies on a component that declares no
/// block size.
const ATTN_RES_EXIT_WITHOUT_DECLARATION: &str =
    "attention-residual exit reduction (the component declares no attn_res_block_size)";

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

/// The tensor `name` is a stream of, when it is one: a name that extends
/// another tensor's whole name by one dot-separated segment.
///
/// This is the only thing the closure pass says about streams stored
/// apart. It does not name the stream, does not resolve a codec and does
/// not decide what the suffix means — it decides that the tensor is
/// ACCOUNTED FOR by its base rather than left without a fate, which is the
/// closure question and the whole of it. What the suffix must be is the
/// codec's declaration, checked where the registry lives (the operand
/// store's stream binding), so a name that looks like a stream and is not
/// one fails there by name rather than being classified here by guess.
fn auxiliary_stream_of<'a>(name: &str, names: &BTreeSet<&'a str>) -> Option<&'a str> {
    let (base, suffix) = name.rsplit_once('.')?;
    if suffix.is_empty() {
        return None;
    }
    names.get(base).copied()
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
    // What the container declares as a dependency of something. A tensor
    // nothing plans and nothing references is unclassified; one another
    // operand REFERENCES has a fate — the codec that needs it — and the
    // loader, which holds the registry, is what checks the reference is
    // one that codec declared.
    let references = match &inspection.index.auxiliary_references {
        Some(name) => {
            crate::format::vindex3::auxiliary_references::AuxiliaryReferences::read(root, name)?
        }
        None => crate::format::vindex3::auxiliary_references::ReferenceTable::empty(),
    };

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
        ResidualTopology::SingleStream | ResidualTopology::AttentionResidual { .. } => None,
    };
    // The attention-residual topology is the same kind of closure
    // question, and asked here for the same reason: its four per-layer
    // operands classify ONLY under this declaration, are required on
    // every transformer layer, and are checked against `[hidden]` and
    // `[1, hidden]`. What the topology still cannot do is said by name
    // where traversal starts (`exec::prepared::PreparedOperands::load`,
    // from `ResidualTopology::unimplemented_reason`) and in the plan
    // report, from that same authority.
    let attention_residual = matches!(
        surface.residual_topology,
        ResidualTopology::AttentionResidual { .. }
    );
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
                | ObjectKind::AttentionResidualExit
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
    if attn.is_some_and(|a| a.attention_bias == Some(true) && a.qkv_bias == Some(true)) {
        defects.push(ClosureDefect::ContradictoryDeclaration {
            component: component.id.clone(),
            detail: "`attention_bias` (Q/K/V and output biased) and `qkv_bias` (Q/K/V only, \
                     output unbiased) are both declared true"
                .to_string(),
        });
    }
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
    // The ROUTED experts' own width, when the component declares a
    // bottleneck for them. Refused here for the same reason the per-layer
    // dense widths are: before any operand is shaped against it, and
    // naming the declaration rather than the first tensor that cannot
    // satisfy it.
    if let Some(width) = ffn_surface
        .and_then(|f| f.moe)
        .and_then(|m| m.degenerate_latent_width())
    {
        defects.push(ClosureDefect::FfnWidthDeclaration {
            component: component.id.clone(),
            detail: moe_latent_width_degenerate(width),
        });
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
    // objects, one role vocabulary, one classifier. (A stream stored apart
    // is skipped by [`auxiliary_stream_of`] before either.)
    let mut by_layer: BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>> = BTreeMap::new();
    let mut bank_by_layer: BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>> = BTreeMap::new();
    for kind in [ObjectKind::DecoderStack, ObjectKind::ExpertBank] {
        let Some((object, tensors)) = tables.get(&kind) else {
            continue;
        };
        let names: BTreeSet<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        for tensor in tensors {
            // A stream stored APART is not an operand of its own: its fate
            // is its base tensor's, and classifying it would report an
            // unclassified operand for something the representation
            // already accounts for. The planner recognises only the shape
            // of the relationship — `<base>.<stream>` beside a `<base>` in
            // the same segment — and the LOADER, which holds the codec
            // registry, is what checks the suffix names a stream the
            // representation declares. Neither half infers a role.
            if auxiliary_stream_of(&tensor.name, &names).is_some() {
                continue;
            }
            // Referenced by something: its fate is its owner's requirement.
            //
            // A fine-grained FP8 scale grid is the case this rule was
            // first paid for in production: the encoder declares the
            // `scales` dependency of every E4M3 weight, so the grid is
            // referenced here and nothing about its NAME is judged. A
            // grid nothing references — an orphan, or a container written
            // before the encoder declared dependencies — falls through to
            // classification and is reported unclassified, which is the
            // defect it is: a split pair leaves neither half bindable.
            if references.is_referenced(&object.id, &tensor.name) {
                continue;
            }
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
            match classify_stack_tensor_under(&tensor.name, operator, surface.residual_topology) {
                // An attention-residual site operand on a component that
                // declares no period is named for what it implies, not
                // merely reported as unclassified: the estate and the
                // declaration disagree, and saying which operation the
                // tensor requires is what tells a reader that from "no
                // rule has ever judged this spelling". The role
                // vocabulary itself stays silent — identity is declared,
                // not inferred from operands — so this is the one place
                // the bare spelling is recognised.
                None if is_attention_residual_site_operand(&tensor.name) => {
                    defects.push(ClosureDefect::OperandImpliesAbsentOp {
                        object: object.id.clone(),
                        tensor: tensor.name.clone(),
                        required_primitive: ATTN_RES_SITE_WITHOUT_DECLARATION.to_string(),
                    })
                }
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

    check_layer_closure(
        LayerClosureInputs {
            surface,
            component,
            placement,
            hyper_connection,
            attention_residual,
            attention_table,
            tables: &tables,
            attn,
            ffn_moe,
            gated_ffn,
            by_layer: &by_layer,
            bank_by_layer: &bank_by_layer,
            layer_geometry: &layer_geometry,
        },
        &mut defects,
    );

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
    // The primary text model generates: a declared vocabulary with no head
    // (no tensor, no declared embedding reuse) is a closure defect here,
    // not a refusal first met at scoring time (#743).
    if head_tensor.is_none()
        && !tables.contains_key(&ObjectKind::OutputHead)
        && surface.head.is_some()
        && component.role == super::super::graph::ComponentRole::PrimaryText
    {
        defects.push(ClosureDefect::NoOutputHead {
            component: component.id.clone(),
        });
    }
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

    // ── The attention-residual exit ──
    let attn_res_exit_tensors =
        tables
            .get(&ObjectKind::AttentionResidualExit)
            .and_then(|(object, tensors)| {
                attention_residual_exit_closure(
                    object,
                    tensors,
                    attention_residual,
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
    let plan = construct_plan(ClosedComponent {
        surface,
        component,
        placement,
        post_norm,
        hyper_connection,
        attention_residual,
        attention_table,
        tables: &tables,
        attn,
        ffn_surface,
        ffn_moe,
        gated_ffn,
        by_layer: &by_layer,
        bank_by_layer: &bank_by_layer,
        layer_geometry: &layer_geometry,
        inter_for: &inter_for,
        vocab,
        embedding_tensor,
        final_norm_tensor,
        head_tensor,
        hc_head_tensors,
        attn_res_exit_tensors,
    });
    Ok(OpPlanOutcome {
        plan: Some(plan),
        defects,
    })
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
mod tests;
