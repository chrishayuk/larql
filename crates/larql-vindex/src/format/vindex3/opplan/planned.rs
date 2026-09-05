//! The operands a plan executes through a representation, and what each
//! operation requires of them — derived from the plan, hardware-independent.
//!
//! Rung 3a of the representation/execution contract
//! (`docs/represent/forecasts/rung3-planned-realizations.json`). Every
//! matrix operand the prepared-plan loader resolves through the seam
//! appears here exactly once per operation; every glue operand — norms,
//! biases, convolution kernels, the recurrences' decay and gate vectors,
//! the router — does not. The list is a VIEW over the plan, computed on
//! demand: nothing is added to the plan's wire shape, so a plan serialised
//! before this rung and after it is the same bytes.
//!
//! Access is what the OPERATION needs over the operand as executed, never
//! what a realization needs over the stored bytes: an embedding lookup
//! gathers one row per token, a packed expert bank is sliced per expert,
//! a projection reads its whole matrix front to back. Which realization can
//! provide that — a direct kernel over stored rows, a whole decode and then
//! a gather — is the prepared plan's question (3b), not this one's.

use larql_models::config::GateSource;
use larql_models::quant::mxfp4::FUSED_HALVES;

use super::exec::backend::MatrixClass;
use super::{
    ComponentOpPlan, ExpertBank, FfnOp, LayerAttention, LayerFfn, OperandRef, RoutedFfnOp,
};
use crate::format::vindex3::represent::codec::{RepresentationExtent, RequiredAccess};

/// What a plan does with one matrix operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// One row of the table per token.
    Embed,
    /// The residual through a whole matrix — attention and recurrence
    /// projections, dense FFN projections.
    Project(MatrixClass),
    /// One expert's slice of a packed bank, chosen per token by the router.
    ExpertBankSlice,
    /// The vocabulary projection of the final hidden state.
    OutputHead,
}

impl Operation {
    /// The access the operation needs over the operand as executed.
    pub const fn access(self) -> RequiredAccess {
        match self {
            Self::Embed | Self::ExpertBankSlice => RequiredAccess::RowRandom,
            Self::Project(_) | Self::OutputHead => RequiredAccess::Sequential,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Project(MatrixClass::AttentionProjection) => "project/attention",
            Self::Project(MatrixClass::FfnProjection) => "project/ffn",
            Self::Project(MatrixClass::OutputHead) => "project/head",
            Self::Project(MatrixClass::RoutedExpertBank) => "project/bank",
            Self::ExpertBankSlice => "expert-bank-slice",
            Self::OutputHead => "output-head",
        }
    }
}

/// One matrix operand the plan will execute, and the requirement its
/// operation makes of it. Hardware-independent: no representation, no
/// backend, no realization is named here.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedOperand {
    pub operand: OperandRef,
    pub operation: Operation,
    pub access: RequiredAccess,
    pub extent: RepresentationExtent,
    /// The weights the operation touches. For every operand but a packed
    /// bank this is the operand's shape; a packed bank's `OperandRef`
    /// carries the STORED tensor's shape — MXFP4 blocks are sixteen bytes
    /// per thirty-two weights — so its logical count comes from the op's
    /// declared geometry, exactly as the bank loader sizes it.
    pub logical_elements: usize,
}

impl PlannedOperand {
    fn new(operand: &OperandRef, operation: Operation) -> Self {
        Self::sized(operand, operation, operand.shape.iter().product())
    }

    fn sized(operand: &OperandRef, operation: Operation, logical_elements: usize) -> Self {
        Self {
            operand: operand.clone(),
            operation,
            access: operation.access(),
            extent: RepresentationExtent::TERMINAL,
            logical_elements,
        }
    }

    pub fn elements(&self) -> usize {
        self.logical_elements
    }
}

impl ComponentOpPlan {
    /// Every matrix operand this plan executes through a representation, in
    /// the order the prepared plan loads them: the embedding, then each
    /// layer's attention and FFN matrices, then the output head.
    ///
    /// Mirrors the loader's own distinction, operand by operand: what
    /// `load_weight` or the packed-bank loader resolves is listed; what is
    /// read as f32 glue is not. A tied head lists the embedding operand
    /// twice, once per operation, because the loader binds it twice.
    pub fn planned_operands(&self) -> Vec<PlannedOperand> {
        let mut out = Vec::new();
        if let Some(embedding) = &self.embedding {
            out.push(PlannedOperand::new(&embedding.table, Operation::Embed));
        }
        for layer in &self.layers {
            attention(&layer.attention, &mut out);
            match &layer.ffn {
                None => {}
                Some(LayerFfn::Dense(op)) => dense(op, &mut out),
                Some(LayerFfn::Routed(op)) => routed(op, &mut out),
                Some(LayerFfn::Hybrid(op)) => {
                    dense(&op.dense, &mut out);
                    routed(&op.routed, &mut out);
                }
            }
        }
        if let Some(output) = &self.output {
            out.push(PlannedOperand::new(
                &output.projection,
                Operation::OutputHead,
            ));
        }
        out
    }
}

const ATTENTION: Operation = Operation::Project(MatrixClass::AttentionProjection);
const FFN: Operation = Operation::Project(MatrixClass::FfnProjection);

fn attention(attention: &LayerAttention, out: &mut Vec<PlannedOperand>) {
    let mut push = |operand: &OperandRef| out.push(PlannedOperand::new(operand, ATTENTION));
    match attention {
        LayerAttention::Softmax(op) => {
            for operand in [&op.q, &op.k, &op.v, &op.o] {
                push(operand);
            }
            // The one gate that is a matrix of its own: a gate fused into
            // the query projection has no operand to resolve, exactly as
            // the loader binds none.
            if let Some(gate) = &op.output_gate {
                if gate.spec.source != GateSource::FusedQueryProjection {
                    push(&gate.projection);
                }
            }
        }
        LayerAttention::GatedDelta(op) => {
            for operand in [
                &op.in_proj_qkv,
                &op.in_proj_a,
                &op.in_proj_b,
                &op.in_proj_z,
                &op.out_proj,
            ] {
                push(operand);
            }
        }
        LayerAttention::Mamba2(op) => {
            push(&op.in_proj);
            push(&op.out_proj);
        }
        LayerAttention::ConvQkv(op) => {
            push(&op.in_proj);
            push(&op.out_proj);
        }
        LayerAttention::Kda(op) => {
            for operand in [&op.q_proj, &op.k_proj, &op.v_proj, &op.out_proj] {
                push(operand);
            }
        }
        LayerAttention::Mla(op) => {
            for operand in [&op.q_proj, &op.kv_a_proj, &op.kv_b_proj, &op.out_proj] {
                push(operand);
            }
        }
    }
}

fn dense(op: &FfnOp, out: &mut Vec<PlannedOperand>) {
    if let Some(gate) = &op.gate {
        out.push(PlannedOperand::new(gate, FFN));
    }
    out.push(PlannedOperand::new(&op.up, FFN));
    out.push(PlannedOperand::new(&op.down, FFN));
}

/// The router, its biases and scales are glue; the banks are the
/// operands. A per-expert bank is a set of whole matrices, each projected
/// on its own when an executor for it exists, and is listed as such.
///
/// A packed bank's logical geometry is the loader's: `experts` matrices of
/// `[FUSED_HALVES * intermediate, hidden]` for gate/up and
/// `[hidden, intermediate]` for down, with `hidden` read off the router's
/// declared width — the same three facts `load_packed` is handed.
fn routed(op: &RoutedFfnOp, out: &mut Vec<PlannedOperand>) {
    match &op.bank {
        ExpertBank::Packed { gate_up, down } => {
            let hidden = op.router.shape.get(1).copied().unwrap_or(0);
            let inter = op.expert_intermediate_size;
            out.push(PlannedOperand::sized(
                &gate_up.weights,
                Operation::ExpertBankSlice,
                op.experts * FUSED_HALVES * inter * hidden,
            ));
            out.push(PlannedOperand::sized(
                &down.weights,
                Operation::ExpertBankSlice,
                op.experts * hidden * inter,
            ));
        }
        ExpertBank::PerExpert { gate, up, down } => {
            for operand in gate.iter().chain(up).chain(down) {
                out.push(PlannedOperand::new(operand, FFN));
            }
        }
    }
    // The shared expert is three whole projections the plan executes
    // beside the routed ones. Listed because the PLAN executes them: the
    // Metal stack binds them, and the CPU loader does not yet — a gap the
    // census cross-check pins rather than hides.
    if let Some(shared) = &op.shared {
        for operand in [&shared.gate, &shared.up, &shared.down] {
            out.push(PlannedOperand::new(operand, FFN));
        }
    }
}
