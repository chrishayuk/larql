//! Hyper-connection and attention-residual closures, and layer op collection.

use super::super::super::encode::segment::{read_segment_header, SegmentTensor};
use super::super::super::encode::REPRESENTATION_ID_SEP;
use super::super::super::graph::policy::LayerOperator;
use super::super::super::graph::roles::{
    classify_attention_residual_exit_tensor, classify_hyper_connection_head_tensor,
    AttentionResidualExitOperand, HcHeadOperand,
};
use super::super::super::graph::surface::Mamba2Surface;
use super::super::super::graph::surface::MoeSurface;
use super::super::super::graph::{LogicalObject, NormPlacement};
use super::super::super::inspect::SystemInspection;
use super::super::exec::hyper_connection::HC_HEAD_SCALE_LEN;
use super::super::ClosureDefect;
use crate::error::VindexError;
use larql_models::config::HyperConnection;
use std::collections::BTreeMap;
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// Closure over the hyper-connection head object: every tensor classifies
/// as one of the head's three operands, each operand is present exactly
/// once, each has the head's geometry, and the component declares the
/// topology the head reduces. Returns the three tensors for binding when
/// all of that holds; every shortfall is itemised into `defects`.
///
/// The head's geometry is deliberately NOT a site's — `[hc, hc·hidden]`
/// against `[(2 + hc)·hc, hc·hidden]`, `[1]` against `[3]` — because
/// `ParallelHead.hc_head` runs no Sinkhorn (see
/// [`super::super::exec::hyper_connection::head_reduce`]). A checkpoint that
/// stored a site's operands under the head's names fails here rather
/// than binding a split into an operation that has none.
pub(super) fn hyper_connection_head_closure(
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

/// Closure over the attention-residual exit object: both operands
/// present exactly once, each classified, each at the pair's geometry,
/// and the component declaring the topology the exit reduces.
///
/// Returns the pair for binding when all of that holds. Transition 1
/// deliberately returned nothing — there was no operation to bind into,
/// and building the argument list of one that does not exist is
/// scaffolding ahead of the oracle. The oracle exists now (`ec7da08d`)
/// and the traversal reads this pair, so the closure hands it over.
///
/// The exit's geometry is a site's — `[hidden]` and `[1, hidden]`,
/// because it is the same reduction run once over the whole history —
/// and it is checked here rather than inherited, so a checkpoint storing
/// something else under these names fails rather than binding.
pub(super) fn attention_residual_exit_closure(
    object: &LogicalObject,
    tensors: &[SegmentTensor],
    attention_residual: bool,
    hidden: usize,
    defects: &mut Vec<ClosureDefect>,
) -> Option<(String, SegmentTensor, SegmentTensor)> {
    // The graph only places this object under the declaration, so this
    // arm states the invariant for a container whose graph was edited
    // rather than built; it is the operand-level form of the same
    // disagreement the builder refuses by name.
    if !attention_residual {
        for tensor in tensors {
            defects.push(ClosureDefect::OperandImpliesAbsentOp {
                object: object.id.clone(),
                tensor: tensor.name.clone(),
                required_primitive: ATTN_RES_EXIT_WITHOUT_DECLARATION.to_string(),
            });
        }
        return None;
    }
    let mut bound: BTreeMap<AttentionResidualExitOperand, SegmentTensor> = BTreeMap::new();
    for tensor in tensors {
        let Some(role) = classify_attention_residual_exit_tensor(&tensor.name) else {
            defects.push(ClosureDefect::UnclassifiedOperand {
                object: object.id.clone(),
                tensor: tensor.name.clone(),
            });
            continue;
        };
        let expected = match role {
            AttentionResidualExitOperand::Norm => vec![hidden],
            AttentionResidualExitOperand::Proj => vec![1, hidden],
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
                detail: format!("two operands claim the exit's {role:?}"),
            });
        }
    }
    for role in [
        AttentionResidualExitOperand::Norm,
        AttentionResidualExitOperand::Proj,
    ] {
        if !bound.contains_key(&role) {
            defects.push(ClosureDefect::ObjectShape {
                object: object.id.clone(),
                detail: format!("no operand for the exit's {role:?}"),
            });
        }
    }
    let (Some(norm), Some(proj)) = (
        bound.remove(&AttentionResidualExitOperand::Norm),
        bound.remove(&AttentionResidualExitOperand::Proj),
    ) else {
        return None;
    };
    Some((object.id.clone(), norm, proj))
}

/// Tensor table of one object's canonical segment.
pub(super) fn object_tensors(
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
pub(super) struct LayerOps {
    pub(super) placement: NormPlacement,
    pub(super) gated_ffn: bool,
    pub(super) output_gate: bool,
    /// The KDA output gate's declared FORM: full-rank `g_proj` (true) or
    /// the low-rank pair (false, the reference's default when undeclared).
    pub(super) kda_full_rank_gate: bool,
    /// Whether the component declares an MLA output gate.
    pub(super) mla_output_gate: bool,
    /// Whether the component declares a factorised MLA query
    /// (`q_lora_rank`). Read from the declared FORM, never from which
    /// query operands the estate ships.
    pub(super) mla_q_lora: bool,
    pub(super) attention_bias: bool,
    /// Q/K/V biased, output not (`qkv_bias`).
    pub(super) qkv_bias: bool,
    /// The output projection biased, Q/K/V not (`output_bias`).
    pub(super) output_bias: bool,
    /// Every trunk norm weight carries a bias (`norm.bias`).
    pub(super) norm_bias: bool,
    /// The dense FFN's up/down projections carry biases (`ffn.bias`).
    pub(super) ffn_bias: bool,
    pub(super) sinks: bool,
    /// This layer's FFN is routed (bank/router evidence under a MoE
    /// judgment); dense otherwise.
    pub(super) routed: bool,
    /// Routed AND dense in one layer (Gemma 4): the dense roles are
    /// required alongside the routed ones, plus the branch norms.
    pub(super) hybrid: bool,
    pub(super) moe: Option<MoeSurface>,
    /// The Mamba2 mixer surface, on a component that declares one — what
    /// decides the conv-bias and gated-norm operand requirements on a
    /// mixer layer.
    pub(super) mamba2: Option<Mamba2Surface>,
    /// The conv-QKV attention geometry, on a component that declares one
    /// — what decides the conv-bias operand requirement on a hybrid
    /// attention layer.
    pub(super) conv_qkv: Option<larql_models::config::ConvQkvAttnGeometry>,
    /// V is the K projection on this layer: no V operand is required, and
    /// one present is a stray.
    pub(super) v_from_k: bool,
    /// The component declares the Sinkhorn hyper-connection topology, so
    /// every transformer layer must supply its two sites' six operands —
    /// and a single-stream component refuses the same six as strays.
    pub(super) hyper_connection: bool,
    /// The component declares the attention-residual topology, so every
    /// transformer layer must supply its two sites' four operands. A
    /// component that does not declare it never classifies these
    /// spellings at all (see
    /// [`classify_stack_tensor_under`](crate::format::vindex3::graph::roles::classify_stack_tensor_under)),
    /// so the stray case is caught one step earlier, where the tensor is
    /// still a name rather than a role.
    pub(super) attention_residual: bool,
    /// Which attention-class operator this layer runs.
    ///
    /// The operator itself rather than an `is_recurrent` flag: the two
    /// recurrences require *different* operand sets, so a boolean would
    /// have to be paired with a second one the moment a second recurrence
    /// existed — which is the shape that let one operator stand in for
    /// another in the first place.
    pub(super) operator: LayerOperator,
}

/// Whether the surface declares `layer` routed: `None` when the surface
/// carries no MoE judgment at all, `Some(false)` inside the dense prefix
/// the surface names (`dense_prefix_layers`), `Some(true)` otherwise.
/// The one derivation both the closure pass and the plan construction
/// read, so they cannot disagree about which layers are routed.
pub(super) fn declared_routed(moe: Option<&MoeSurface>, layer: usize) -> Option<bool> {
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
pub(super) fn auxiliary_stream_of<'a>(name: &str, names: &BTreeSet<&'a str>) -> Option<&'a str> {
    let (base, suffix) = name.rsplit_once('.')?;
    if suffix.is_empty() {
        return None;
    }
    names.get(base).copied()
}
