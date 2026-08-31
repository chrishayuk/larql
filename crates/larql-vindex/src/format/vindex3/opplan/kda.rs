//! Kimi Delta Attention as an executable operation.
//!
//! A sibling of [`GatedDeltaOp`](super::gated_delta::GatedDeltaOp), never a
//! mode of it. The two recurrences differ structurally, not by
//! parameterisation:
//!
//! | | Gated DeltaNet | KDA |
//! |---|---|---|
//! | q/k/v | one fused projection | three separate projections |
//! | short conv | one, over fused channels | **three**, one per stream |
//! | decay gate | `in_proj_a`, full rank `[Hv, hidden]` | `f_a`·`f_b`, **low rank** |
//! | output gate | `in_proj_z`, full rank | `g_a`·`g_b`, **low rank** |
//! | `dt_bias` | `[Hv]`, per head | `[Hv·Dv]`, **per channel** |
//! | head counts | key and value sides differ | one head count |
//!
//! Merging them behind a flag would reintroduce exactly the ambiguity the
//! recurrence-identification work removed: a checkpoint of either kind
//! would bind to the other's roles with plausible shapes, and `dt_bias`
//! is the only operand whose *geometry* would object.
//!
//! **The completeness rule this op is written to satisfy:** the recurrence
//! must be reconstructible from the op plus its bound operands alone. No
//! consumer may need to know that a container came from Kimi Linear or
//! from GLM-5.3-Flash in order to run it — every dimension and rank is
//! stated here, not re-derived from tensor names downstream.

use serde::Serialize;

use super::OperandRef;

/// Kimi Delta Attention: one `Dk × Dv` recurrent state per head, updated
/// by a delta rule with per-channel decay.
///
/// # Execution inputs vs carried provenance
///
/// Every field below is an **execution input** — an operand or a dimension
/// the recurrence reads — *except* [`Self::gate_lower_bound`], which is
/// carried provenance and must not be wired into the operator. The two
/// kinds sit in one struct because they describe one layer, so the
/// distinction is stated here rather than left to be inferred from a field
/// name.
///
/// It is not a hypothetical distinction. Applying the declared
/// `gate_lower_bound` to this operator changes its output by a **relative
/// 1.75** against the reference (`scripts/kda_controls.py`), with every
/// shape still closing. A future reader who sees the field and wires it in
/// "because that is obviously what it is for" gets a mathematically
/// coherent, badly wrong recurrence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct KdaOp {
    /// Hv — head count. One number, unlike Gated DeltaNet: KDA's key and
    /// value sides share it.
    pub num_heads: usize,
    /// Dk = Dv — per-head width, shared by the key and value sides.
    pub head_dim: usize,
    /// Width of each depthwise causal convolution, applied independently
    /// to the query, key and value streams.
    pub conv_kernel: usize,
    /// Inner rank of the f and g gate factorisations.
    ///
    /// Stated explicitly because the config declares no such field — it is
    /// resolved once, from the bound operands, at build time. A consumer
    /// must never have to recover it by inspecting an operand's shape, or
    /// the op is not self-describing.
    pub gate_rank: usize,
    /// The checkpoint's declared `gate_lower_bound` (-5.0 on both observed
    /// checkpoints), carried verbatim.
    ///
    /// **An executor must not apply it.** Kimi Linear's own
    /// `modeling_kimi.py` reads this field *nowhere* — neither the modeling
    /// file nor `configuration_kimi.py` mentions it — and its gate call
    /// passes no lower bound:
    ///
    /// ```text
    /// g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
    /// ```
    ///
    /// which selects the softplus form, `g = -exp(A_log)·softplus(g +
    /// dt_bias)`, not the clamped form `lower_bound·sigmoid(...)` the same
    /// upstream function also offers. Applying the declared bound would
    /// compute a different decay envelope from the model's own reference
    /// while every shape still closed.
    ///
    /// So this is a declaration carried for provenance, and the operator
    /// contract is that it is *not* an input to the recurrence. If a
    /// checkpoint ever appears whose reference does apply it, that is a
    /// second gate form and belongs in its own field rather than changing
    /// what this one means.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_lower_bound: Option<f32>,

    /// Query projection, `[Hv·Dv, hidden]`.
    pub q_proj: OperandRef,
    /// Key projection, `[Hv·Dv, hidden]`.
    pub k_proj: OperandRef,
    /// Value projection, `[Hv·Dv, hidden]`.
    pub v_proj: OperandRef,
    /// Depthwise causal conv over the query stream, `[Hv·Dv, 1, kernel]`.
    pub q_conv1d: OperandRef,
    /// Depthwise causal conv over the key stream.
    pub k_conv1d: OperandRef,
    /// Depthwise causal conv over the value stream.
    pub v_conv1d: OperandRef,
    /// Decay-gate down-projection, `[rank, hidden]`.
    pub f_a_proj: OperandRef,
    /// Decay-gate up-projection, `[Hv·Dv, rank]`.
    pub f_b_proj: OperandRef,
    /// Output-gate down-projection, `[rank, hidden]`.
    pub g_a_proj: OperandRef,
    /// Output-gate up-projection, `[Hv·Dv, rank]`.
    pub g_b_proj: OperandRef,
    /// Per-head write-strength projection, `[Hv, hidden]`.
    pub b_proj: OperandRef,
    /// Per-head log decay, `[Hv]`.
    pub a_log: OperandRef,
    /// Per-channel timestep bias, `[Hv·Dv]`.
    pub dt_bias: OperandRef,
    /// Gated RMSNorm weight over one head's width, `[Dv]`.
    pub o_norm: OperandRef,
    /// Output projection, `[hidden, Hv·Dv]`.
    pub out_proj: OperandRef,
}

impl KdaOp {
    /// Width of the value/gate side, `Hv·Dv`.
    ///
    /// Derived rather than stored so it cannot drift from the head counts
    /// beside it. On Kimi Linear this closes at `32·128 = 4096`, the
    /// observed row count of `q_proj`, `k_proj`, `v_proj` and each conv;
    /// on GLM-5.3-Flash at `64·128 = 8192`.
    pub fn value_width(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Elements in this layer's recurrent state: one `Dk × Dv` matrix per
    /// head.
    ///
    /// The number a KV planner needs instead of a span. Nothing here is
    /// indexed by position, so there is no prefix to bound and no
    /// per-position cache to size — the state is this size whatever the
    /// sequence length.
    pub fn state_elements(&self) -> usize {
        self.num_heads * self.head_dim * self.head_dim
    }

    /// The three geometry numbers in the form the operator's own reference
    /// takes them.
    ///
    /// A projection, not a second record: every field is carried on this
    /// op and the executor reads exactly these, so a continuation planner
    /// and an executor cannot disagree about the shape of the state one
    /// sizes and the other advances.
    pub fn geometry(&self) -> larql_models::config::KdaGeometry {
        larql_models::config::KdaGeometry {
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            conv_kernel: self.conv_kernel,
        }
    }
}
