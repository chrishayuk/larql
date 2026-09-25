//! The relocatable part of a routed FFN: one unweighted expert transform.
//! Routing, selection weights and ordered accumulation belong to the caller.
use super::backend::{ExpertSlices, RoutedFfnCall, WeightSlice};
use crate::error::VindexError;
use larql_models::config::{Activation, GateUpLayout};
mod worker;
pub use worker::{ExpertRegion, PreparedRoutedExperts};

pub(super) fn validate_plan(plan: &super::super::ComponentOpPlan) -> Result<(), VindexError> {
    use super::super::{ExpertBank, LayerFfn};
    if plan.residual_topology != larql_models::config::ResidualTopology::SingleStream
        || plan.layers.is_empty()
    {
        return Err(VindexError::Parse(
            "routed placement requires a single-stream softmax stack".into(),
        ));
    }
    for layer in &plan.layers {
        let Some(LayerFfn::Routed(op)) = &layer.ffn else {
            return Err(VindexError::Parse(
                "routed placement requires routed FFNs on every layer".into(),
            ));
        };
        if layer.attention.softmax().is_none()
            || op.shared.is_some()
            || op.latent.is_some()
            || !matches!(op.bank, ExpertBank::Packed { .. })
            || op.expert_format != larql_models::config::ExpertFormat::PackedMxfp4
            || op.gate_up_layout.is_none()
            || op.router.shape.len() != 2
            || op.router.shape.get(1).is_none_or(|n| *n == 0)
            || op.expert_intermediate_size == 0
            || op.expert_intermediate_size.checked_mul(2).is_none()
            || op.experts == 0
            || op.top_k == 0
            || op.top_k > op.experts
        {
            return Err(VindexError::Parse("routed placement currently requires softmax attention and packed MXFP4 experts without shared or latent branches".into()));
        }
    }
    Ok(())
}

pub enum ExpertWeights<'a> {
    Fused {
        gate_up: WeightSlice<'a>,
        down: WeightSlice<'a>,
        layout: GateUpLayout,
        gate_up_bias: Option<&'a [f32]>,
        down_bias: Option<&'a [f32]>,
    },
    Separate {
        gate: WeightSlice<'a>,
        up: WeightSlice<'a>,
        down: WeightSlice<'a>,
    },
}

/// Contains neither router operands nor a routing weight. Down bias is part
/// of this transform and must be applied before the coordinator weights it.
pub struct ExpertTransformCall<'a> {
    pub x: &'a [f32],
    pub hidden: usize,
    pub intermediate: usize,
    pub activation: Activation,
    pub gate_policy: larql_models::ExpertGatePolicy,
    pub weights: ExpertWeights<'a>,
}

pub struct ExpertOutput {
    pub expert: usize,
    pub row: Vec<f32>,
}

/// Results may arrive in any order, but must name every requested expert
/// exactly once. Implementations receive IDs, never routing weights.
pub trait RoutedExpertProvider: Send + Sync {
    fn apply(
        &self,
        layer: usize,
        input: &[f32],
        experts: &[usize],
    ) -> Result<Vec<ExpertOutput>, VindexError>;
}

impl<'a> RoutedFfnCall<'a> {
    /// Resolve one expert without changing any matrix representation.
    pub fn expert_transform(&self, expert: usize) -> Result<ExpertTransformCall<'a>, VindexError> {
        if expert >= self.experts {
            return Err(VindexError::Parse("expert outside bank".into()));
        }
        let get = |weights: &'a [WeightSlice<'a>]| {
            weights
                .get(expert)
                .copied()
                .ok_or_else(|| VindexError::Parse("expert matrix missing".into()))
        };
        let bias =
            |values: Option<&'a [f32]>, width: usize| -> Result<Option<&'a [f32]>, VindexError> {
                values
                    .map(|values| {
                        let start = expert.checked_mul(width).ok_or_else(|| {
                            VindexError::Parse("expert bias offset overflow".into())
                        })?;
                        let end = start.checked_add(width).ok_or_else(|| {
                            VindexError::Parse("expert bias extent overflow".into())
                        })?;
                        values
                            .get(start..end)
                            .ok_or_else(|| VindexError::Parse("expert bias shape mismatch".into()))
                    })
                    .transpose()
            };
        let weights = match &self.weights {
            ExpertSlices::Fused {
                gate_up,
                down,
                layout,
            } => ExpertWeights::Fused {
                gate_up: get(gate_up)?,
                down: get(down)?,
                layout: *layout,
                gate_up_bias: bias(
                    self.gate_up_bias,
                    self.intermediate
                        .checked_mul(2)
                        .ok_or_else(|| VindexError::Parse("expert width overflow".into()))?,
                )?,
                down_bias: bias(self.down_bias, self.hidden)?,
            },
            ExpertSlices::Separate { gate, up, down, .. } => {
                if self.gate_up_bias.is_some() || self.down_bias.is_some() {
                    return Err(VindexError::Parse(
                        "a per-expert bank carries no expert bias; the call declares one".into(),
                    ));
                }
                ExpertWeights::Separate {
                    gate: get(gate)?,
                    up: get(up)?,
                    down: get(down)?,
                }
            }
        };
        Ok(ExpertTransformCall {
            x: self.x,
            hidden: self.hidden,
            intermediate: self.intermediate,
            activation: self.activation,
            gate_policy: self.gate_policy,
            weights,
        })
    }
}

/// Validate the complete response before accumulating anything. Arrival order
/// never determines floating-point order: use the production selection order.
pub(super) fn reduce_selected(
    selected: &[(usize, f32)],
    rows: Vec<ExpertOutput>,
    hidden: usize,
) -> Result<Vec<f32>, VindexError> {
    if rows.len() != selected.len() {
        return Err(VindexError::Parse("expert response count mismatch".into()));
    }
    let mut by_expert = std::collections::BTreeMap::new();
    for output in rows {
        if !selected.iter().any(|(id, _)| *id == output.expert) {
            return Err(VindexError::Parse("unsolicited expert response".into()));
        }
        if output.row.len() != hidden || output.row.iter().any(|v| !v.is_finite()) {
            return Err(VindexError::Parse(
                "expert output width or finiteness mismatch".into(),
            ));
        }
        if by_expert.insert(output.expert, output.row).is_some() {
            return Err(VindexError::Parse("duplicate expert response".into()));
        }
    }
    let mut out = vec![0.0; hidden];
    for (expert, weight) in selected {
        let row = by_expert
            .remove(expert)
            .ok_or_else(|| VindexError::Parse("missing expert response".into()))?;
        for (acc, value) in out.iter_mut().zip(row) {
            *acc += weight * value;
        }
    }
    Ok(out)
}
