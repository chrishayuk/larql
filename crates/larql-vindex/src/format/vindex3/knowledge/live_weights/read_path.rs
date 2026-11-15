//! Which planned operands a pending stored-weight op may stand in for.
//!
//! A replacement reaches execution only where the loader reads the operand through the operand seam
//! ([`OperandSource`](crate::format::vindex3::opplan::exec::operands::OperandSource)). This is an allow-list of
//! the planned operations whose loaders have been read and do: the embedding table, a softmax layer's
//! projections, a dense FFN's projections and the output head. Everything else is refused by name. The matches
//! are exhaustive on purpose: an operation or a layer kind added to the plan does not compile here until someone
//! has decided which side of the list it is on.

use super::LiveWeightRefusal;
use crate::format::vindex3::opplan::exec::backend::MatrixClass;
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerAttention, LayerFfn, OperandRef};

/// A routed layer's bank is mapped from the store, or sliced out of it by byte range, without passing the seam.
pub const WHY_EXPERT_BANK: &str =
    "an expert bank is bound directly from the store and bypasses the session's operand overrides";
/// A shared expert's projections sit in a routed block whose loader has not been read for this.
pub const WHY_SHARED_EXPERT: &str =
    "a shared expert belongs to a routed block, whose loaders are not qualified for live weight ops";
/// A dense half or a projection of a routed or hybrid FFN layer.
pub const WHY_ROUTED_FFN: &str =
    "its layer's FFN is routed, and a routed block's loaders are not qualified for live weight ops";
/// Any mixer but softmax attention.
pub const WHY_OTHER_MIXER: &str =
    "its layer's mixer is not softmax attention, the only mixer qualified for live weight ops";
/// A matrix class no planned operand of a projection carries today.
pub const WHY_UNPLANNED_CLASS: &str =
    "the plan projects no operand under this class, so no loader is qualified for it";

/// Why execution would not read `planned` through the operand seam; `None` when it would.
pub fn seam_refusal(plan: &ComponentOpPlan, planned: &PlannedOperand) -> Option<&'static str> {
    let layer = planned.layer.and_then(|index| plan.layers.get(index));
    match planned.operation {
        Operation::Embed | Operation::OutputHead => None,
        Operation::Project(class) => match class {
            MatrixClass::AttentionProjection => match layer.map(|l| &l.attention) {
                Some(LayerAttention::Softmax(_)) => None,
                Some(
                    LayerAttention::GatedDelta(_)
                    | LayerAttention::Mamba2(_)
                    | LayerAttention::ConvQkv(_)
                    | LayerAttention::Kda(_)
                    | LayerAttention::Mla(_),
                )
                | None => Some(WHY_OTHER_MIXER),
            },
            MatrixClass::FfnProjection => match layer.and_then(|l| l.ffn.as_ref()) {
                Some(LayerFfn::Dense(_)) => None,
                Some(LayerFfn::Routed(_) | LayerFfn::Hybrid(_)) | None => Some(WHY_ROUTED_FFN),
            },
            MatrixClass::OutputHead | MatrixClass::RoutedExpertBank => Some(WHY_UNPLANNED_CLASS),
        },
        Operation::ExpertBankSlice | Operation::ExpertProject { .. } => Some(WHY_EXPERT_BANK),
        Operation::SharedExpertProject | Operation::SharedExpertBranchGate => {
            Some(WHY_SHARED_EXPERT)
        }
    }
}

/// The operands through which `plan` executes the tensor named `tensor`, every one read through the seam.
/// Refused when the plan lists no such operand (nothing shows execution would read the edit), or when any
/// operand naming it is read another way (the edit would be partly or wholly ignored).
pub fn seam_operands(
    plan: &ComponentOpPlan,
    tensor: &str,
) -> Result<Vec<OperandRef>, LiveWeightRefusal> {
    let mut operands: Vec<OperandRef> = Vec::new();
    for planned in plan.planned_operands() {
        if planned.operand.tensor != tensor {
            continue;
        }
        if let Some(why) = seam_refusal(plan, &planned) {
            return Err(LiveWeightRefusal::OutsideSeam {
                tensor: tensor.to_string(),
                operation: planned.operation.name(),
                why,
            });
        }
        if !operands.contains(&planned.operand) {
            operands.push(planned.operand);
        }
    }
    if operands.is_empty() {
        return Err(LiveWeightRefusal::NotExecuted {
            tensor: tensor.to_string(),
        });
    }
    Ok(operands)
}
