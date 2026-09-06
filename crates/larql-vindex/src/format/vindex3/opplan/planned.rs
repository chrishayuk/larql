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

use larql_models::config::{ExpertFormat, GateSource};
use larql_models::quant::mxfp4::FUSED_HALVES;

use super::exec::backend::MatrixClass;
use super::{
    ComponentOpPlan, ExpertBank, FfnOp, KdaOutputGate, LayerAttention, LayerFfn, OperandRef,
    RoutedFfnOp,
};
use crate::format::vindex3::represent::codec::codecs::float::FloatDtype;
use crate::format::vindex3::represent::codec::codecs::mxfp4::DTYPE_MXFP4;
use crate::format::vindex3::represent::codec::{RepresentationExtent, RequiredAccess};

/// The bf16 codec's label, as the float codec spells it.
const BF16_LABEL: &str = FloatDtype::Bf16.label();

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
    /// A shared expert's projection beside the routed ones — its own
    /// operation, because whether a backend can bind it is a question the
    /// prepared plan must be able to refuse by name (rung 3b).
    SharedExpertProject,
    /// The scalar gate on a shared-expert branch (Qwen MoE's
    /// `sigmoid(shared_expert_gate(x))`): planned so that a plan carrying
    /// one is refused by name rather than executed with the branch
    /// summed unscaled.
    SharedExpertBranchGate,
    /// One expert's OWN matrix of a per-expert bank, read whole when the
    /// router selects it: `top_k` of `experts` per token. The operation
    /// carries the bank's geometry so a plan can price touch per token
    /// and a backend can bind the bank ONCE as shared physical storage
    /// serving every logical access (K3-RESIDENCY-VERTICAL-1, V1).
    ExpertProject { experts: usize, top_k: usize },
}

impl Operation {
    /// The access the operation needs over the operand as executed.
    pub const fn access(self) -> RequiredAccess {
        match self {
            Self::Embed | Self::ExpertBankSlice => RequiredAccess::RowRandom,
            // An expert's matrix is read whole once selected, and a shared
            // projection is a whole matrix: sequential over the operand.
            Self::Project(_)
            | Self::OutputHead
            | Self::SharedExpertProject
            | Self::SharedExpertBranchGate
            | Self::ExpertProject { .. } => RequiredAccess::Sequential,
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
            Self::SharedExpertProject => "shared-expert-project",
            Self::SharedExpertBranchGate => "shared-expert-branch-gate",
            Self::ExpertProject { .. } => "expert-project",
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
    /// The plan layer the operand belongs to; `None` for the embedding and
    /// the head, which sit outside the stack.
    pub layer: Option<usize>,
    /// A legacy default for the operand's codec, consulted only when the
    /// container's stored label is a carrier dialect that names no codec:
    /// a packed MXFP4 bank is stored as two `U8` byte tensors, and only
    /// the architecture's declaration says they are MXFP4's two streams.
    /// The STORED representation stays authoritative — a label that names
    /// a codec is never overridden by this. `None` when the stored label
    /// is the representation.
    pub declared_representation: Option<&'static str>,
    /// The weights the operation touches. For every operand but a packed
    /// bank this is the operand's shape; a packed bank's `OperandRef`
    /// carries the STORED tensor's shape — MXFP4 blocks are sixteen bytes
    /// per thirty-two weights — so its logical count comes from the op's
    /// declared geometry, exactly as the bank loader sizes it.
    pub logical_elements: usize,
}

impl PlannedOperand {
    fn new(operand: &OperandRef, operation: Operation, layer: Option<usize>) -> Self {
        Self::sized(operand, operation, layer, operand.shape.iter().product())
    }

    fn sized(
        operand: &OperandRef,
        operation: Operation,
        layer: Option<usize>,
        logical_elements: usize,
    ) -> Self {
        Self {
            operand: operand.clone(),
            operation,
            access: operation.access(),
            extent: RepresentationExtent::BASE,
            layer,
            declared_representation: None,
            logical_elements,
        }
    }

    fn declaring(mut self, representation: Option<&'static str>) -> Self {
        self.declared_representation = representation;
        self
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
            out.push(PlannedOperand::new(
                &embedding.table,
                Operation::Embed,
                None,
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            attention(&layer.attention, index, &mut out);
            match &layer.ffn {
                None => {}
                Some(LayerFfn::Dense(op)) => dense(op, index, &mut out),
                Some(LayerFfn::Routed(op)) => routed(op, index, &mut out),
                Some(LayerFfn::Hybrid(op)) => {
                    dense(&op.dense, index, &mut out);
                    routed(&op.routed, index, &mut out);
                }
            }
        }
        if let Some(output) = &self.output {
            out.push(PlannedOperand::new(
                &output.projection,
                Operation::OutputHead,
                None,
            ));
        }
        out
    }
}

const ATTENTION: Operation = Operation::Project(MatrixClass::AttentionProjection);
const FFN: Operation = Operation::Project(MatrixClass::FfnProjection);

fn attention(attention: &LayerAttention, layer: usize, out: &mut Vec<PlannedOperand>) {
    let mut push =
        |operand: &OperandRef| out.push(PlannedOperand::new(operand, ATTENTION, Some(layer)));
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
            // The full-rank output gate is a matrix the size of the four
            // above and is projected like them; the low-rank pair is glue
            // and, like the other glue, is not listed here (K3-REP-GATE-1).
            if let KdaOutputGate::FullRank { g_proj } = &op.output_gate {
                push(g_proj);
            }
        }
        LayerAttention::Mla(op) => {
            for operand in [&op.q_proj, &op.kv_a_proj, &op.kv_b_proj, &op.out_proj] {
                push(operand);
            }
            // The declared output gate, a matrix the size of `out_proj`.
            if let Some(gate) = &op.output_gate {
                push(gate);
            }
        }
    }
}

fn dense(op: &FfnOp, layer: usize, out: &mut Vec<PlannedOperand>) {
    if let Some(gate) = &op.gate {
        out.push(PlannedOperand::new(gate, FFN, Some(layer)));
    }
    out.push(PlannedOperand::new(&op.up, FFN, Some(layer)));
    out.push(PlannedOperand::new(&op.down, FFN, Some(layer)));
}

/// The router, its biases and scales are glue; the banks are the
/// operands. A per-expert bank is a set of whole matrices, each projected
/// on its own when an executor for it exists, and is listed as such.
///
/// A packed bank's logical geometry is the loader's: `experts` matrices of
/// `[FUSED_HALVES * intermediate, hidden]` for gate/up and
/// `[hidden, intermediate]` for down, with `hidden` read off the router's
/// declared width — the same three facts `load_packed` is handed.
fn routed(op: &RoutedFfnOp, layer: usize, out: &mut Vec<PlannedOperand>) {
    match &op.bank {
        ExpertBank::Packed { gate_up, down } => {
            let hidden = op.router.shape.get(1).copied().unwrap_or(0);
            let inter = op.expert_intermediate_size;
            let declared = declared_bank_representation(op.expert_format);
            out.push(
                PlannedOperand::sized(
                    &gate_up.weights,
                    Operation::ExpertBankSlice,
                    Some(layer),
                    op.experts * FUSED_HALVES * inter * hidden,
                )
                .declaring(declared),
            );
            out.push(
                PlannedOperand::sized(
                    &down.weights,
                    Operation::ExpertBankSlice,
                    Some(layer),
                    op.experts * hidden * inter,
                )
                .declaring(declared),
            );
        }
        ExpertBank::PerExpert { gate, up, down } => {
            // Multiplicity is preserved — one planned operand per expert
            // matrix, so execution touch counts every logical access —
            // while the operation names the bank the matrices share.
            let bank = Operation::ExpertProject {
                experts: op.experts,
                top_k: op.top_k,
            };
            for operand in gate.iter().chain(up).chain(down) {
                out.push(PlannedOperand::new(operand, bank, Some(layer)));
            }
        }
    }
    // The shared expert is three whole projections the plan executes
    // beside the routed ones. Listed under their own operation because the
    // PLAN executes them and no backend binds them through the prepared
    // plan yet: the selector refuses them by name (rung 3b) rather than
    // preparing a model without its shared expert.
    if let Some(shared) = &op.shared {
        for operand in [&shared.gate, &shared.up, &shared.down] {
            out.push(PlannedOperand::new(
                operand,
                Operation::SharedExpertProject,
                Some(layer),
            ));
        }
        if let Some(gate) = &shared.branch_gate {
            out.push(PlannedOperand::new(
                &gate.weight,
                Operation::SharedExpertBranchGate,
                Some(layer),
            ));
        }
    }
}

/// The codec a packed bank's declared layout carries — the legacy default
/// behind [`PlannedOperand::declared_representation`]. A packed MXFP4 bank
/// is stored as two `U8` tensors that are MXFP4's values and group-scale
/// streams; a packed bf16 bank is stored as what it is, and its stored
/// label answers before this does. A per-expert bank is not packed and
/// declares nothing beyond its tensors' own labels.
pub fn declared_bank_representation(format: ExpertFormat) -> Option<&'static str> {
    match format {
        ExpertFormat::PackedMxfp4 => Some(DTYPE_MXFP4),
        ExpertFormat::PackedBF16 => Some(BF16_LABEL),
        ExpertFormat::PerExpert => None,
    }
}
