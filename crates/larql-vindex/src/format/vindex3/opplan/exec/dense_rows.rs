//! Every dense operand a recurrent or latent mixer projects through the backend's
//! [`DenseProjections`](super::gated_delta::DenseProjections), named, so a layer can be refused before any of
//! its arithmetic when the backend pinned a representation its projector has no kernel for.
//!
//! Without this the reference projector reached a compact slab and panicked mid-layer (`metal-nvfp4` on a
//! Gated DeltaNet model). A refusal is the contract: name the layer, the operand and the representation, and
//! run nothing.

use super::super::ComponentOpPlan;
use super::conv_qkv::ConvQkvWeights;
use super::cpu::WeightRows;
use super::gated_delta::{DenseProjections, GatedDeltaWeights};
use super::kda::KdaWeights;
use super::mamba2::Mamba2Weights;
use super::mla::{MlaQueryWeights, MlaWeights};
use super::prepared::{PreparedAttention, PreparedOperands};
use crate::error::VindexError;

/// A mixer's dense operands, as (operand name, resident rows).
pub trait DenseRowSet {
    /// The mixer, for the refusal message.
    const MIXER: &'static str;
    fn dense_rows(&self) -> Vec<(&'static str, WeightRows<'_>)>;
}

/// `projector`, if it consumes every dense operand of `weights`; else a typed refusal naming the first one it
/// does not.
pub fn consuming_projector<'p, W: DenseRowSet>(
    projector: &'p dyn DenseProjections,
    weights: &W,
    layer: usize,
) -> Result<&'p dyn DenseProjections, VindexError> {
    match weights
        .dense_rows()
        .into_iter()
        .find(|(_, rows)| !projector.consumes(*rows))
    {
        Some((name, rows)) => Err(VindexError::Parse(format!(
            "layer {layer}: this backend's {} projector has no kernel for `{name}` resident as {}; \
             refusing before the layer runs rather than converting or panicking mid-layer",
            W::MIXER,
            rows.representation(),
        ))),
        None => Ok(projector),
    }
}

impl DenseRowSet for GatedDeltaWeights<'_> {
    const MIXER: &'static str = "Gated DeltaNet";
    fn dense_rows(&self) -> Vec<(&'static str, WeightRows<'_>)> {
        vec![
            ("in_proj_qkv", self.in_proj_qkv),
            ("in_proj_a", self.in_proj_a),
            ("in_proj_b", self.in_proj_b),
            ("in_proj_z", self.in_proj_z),
            ("out_proj", self.out_proj),
        ]
    }
}

impl DenseRowSet for Mamba2Weights<'_> {
    const MIXER: &'static str = "Mamba2";
    fn dense_rows(&self) -> Vec<(&'static str, WeightRows<'_>)> {
        vec![("in_proj", self.in_proj), ("out_proj", self.out_proj)]
    }
}

impl DenseRowSet for ConvQkvWeights<'_> {
    const MIXER: &'static str = "conv-QKV";
    fn dense_rows(&self) -> Vec<(&'static str, WeightRows<'_>)> {
        vec![("in_proj", self.in_proj), ("out_proj", self.out_proj)]
    }
}

impl DenseRowSet for KdaWeights<'_> {
    const MIXER: &'static str = "KDA";
    fn dense_rows(&self) -> Vec<(&'static str, WeightRows<'_>)> {
        let rows = vec![
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj),
        ];
        // A full-rank `g_proj` is not listed: the KDA step projects it with
        // the CPU executor directly (`kda/step.rs`), never through the
        // backend's projector, so naming it here would refuse a slab that
        // runs.
        rows
    }
}

impl DenseRowSet for MlaWeights<'_> {
    const MIXER: &'static str = "MLA";
    fn dense_rows(&self) -> Vec<(&'static str, WeightRows<'_>)> {
        let mut rows = match self.query {
            MlaQueryWeights::Direct { q_proj } => vec![("q_proj", q_proj)],
            MlaQueryWeights::LowRank {
                q_a_proj, q_b_proj, ..
            } => vec![("q_a_proj", q_a_proj), ("q_b_proj", q_b_proj)],
        };
        rows.extend([
            ("kv_a_proj", self.kv_a_proj),
            ("kv_b_proj", self.kv_b_proj),
            ("o_proj", self.o_proj),
        ]);
        if let Some(g) = self.output_gate {
            rows.push(("g_proj", g));
        }
        rows
    }
}

/// Every mixer layer of a prepared plan, checked against `projector` BEFORE any layer runs.
///
/// The per-layer check at each call site stays, but alone it would refuse layer N after layers 0..N had
/// advanced their recurrent state. This is the refuse-before-commit form the traversal and the decode session
/// call up front, so a plan whose softmax layers come first is refused as whole as one that starts with a mixer.
pub(super) fn refuse_unconsumable_plan(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    projector: &dyn DenseProjections,
) -> Result<(), VindexError> {
    for (offset, prepared) in ops.layers().iter().enumerate() {
        let layer = plan
            .layers
            .get(ops.first_layer() + offset)
            .map_or(offset, |l| l.layer);
        match &prepared.attention {
            PreparedAttention::Softmax(_) => {}
            PreparedAttention::GatedDelta(o) => {
                consuming_projector(projector, &o.weights()?, layer)?;
            }
            PreparedAttention::Mamba2(o) => {
                consuming_projector(projector, &o.weights()?, layer)?;
            }
            PreparedAttention::ConvQkv(o) => {
                consuming_projector(projector, &o.weights()?, layer)?;
            }
            PreparedAttention::Kda(o) => {
                consuming_projector(projector, &o.weights()?, layer)?;
            }
            PreparedAttention::Mla(o) => {
                consuming_projector(projector, &o.weights()?, layer)?;
            }
        }
    }
    Ok(())
}
