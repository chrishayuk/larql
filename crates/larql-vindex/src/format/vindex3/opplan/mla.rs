//! Multi-Latent Attention as an executable operation.
//!
//! Retains a real per-position KV cache — unlike [`super::kda::KdaOp`]/
//! [`super::gated_delta::GatedDeltaOp`] it is not a recurrence — but the
//! cache is the COMPRESSED latent (`kv_lora_rank + qk_rope_head_dim`
//! elements per position), not the full `heads·(nope+v_head_dim)` a
//! decompressed K/V pair would need. That asymmetry is the entire point
//! of the operator, so [`MlaOp::compressed_kv_width`] states it rather
//! than leaving a planner to infer it from the projection shapes.
//!
//! **The completeness rule this op is written to satisfy** (same as
//! [`super::kda`]'s): the operator must be reconstructible from this
//! struct plus its bound operands alone. No consumer may need to know a
//! container came from Kimi Linear vs. the DeepSeek lineage — every
//! dimension is stated here, never re-derived from a tensor name.
//!
//! **What this rung deliberately does NOT model**: a family that
//! compresses its query too (`q_lora_rank` — DeepSeek's shape, not
//! Kimi's: `assert self.q_lora_rank is None` in the checkpoint's own
//! `modeling_kimi.py`). A `q_a_proj`/`q_b_proj` pair belongs in a second
//! op variant when a checkpoint that ships them is actually judged, not
//! guessed in ahead of one.

use serde::Serialize;

use super::OperandRef;

/// Multi-Latent Attention: one dense query projection, a shared
/// (MQA-style) low-rank KV compression, and per-head decompression into
/// an asymmetric nope+rope query width and a DIFFERENT value width.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MlaOp {
    /// Query/output head count — the decompressed K/V side always
    /// produces this many heads' worth of output.
    pub num_heads: usize,
    /// Compressed KV latent width.
    pub kv_lora_rank: usize,
    /// Non-RoPE portion of the query/key head width.
    pub qk_nope_head_dim: usize,
    /// RoPE portion of the query/key head width, one SHARED projection
    /// across every head.
    pub qk_rope_head_dim: usize,
    /// Value head width — independent of the query/key head width.
    pub v_head_dim: usize,

    /// How this layer builds its query — one dense projection, or
    /// Kimi-K3's factorisation. A typed form rather than a pair of
    /// options, so "both" and "neither" cannot be written down at all.
    pub query: MlaQueryProjection,
    /// Shared compressed KV projection: latent + one rope-K,
    /// `[kv_lora_rank + rope, hidden]`.
    pub kv_a_proj: OperandRef,
    /// KV decompression: nope-K and V per head, fused,
    /// `[Hq·(nope+v_head_dim), kv_lora_rank]`.
    pub kv_b_proj: OperandRef,
    /// RMSNorm weight over the compressed KV latent, applied before
    /// decompression, `[kv_lora_rank]`.
    pub kv_a_norm: OperandRef,
    /// Output projection, `[hidden, Hq·v_head_dim]`.
    pub out_proj: OperandRef,
    /// The output gate's projection, `[Hq·v_head_dim, hidden]`, when the
    /// checkpoint declares `mla_use_output_gate` (Kimi-K3):
    /// `sigmoid(g_proj(x)) ⊙ attn_value` before [`Self::out_proj`], with
    /// `x` the block's normalised input. `None` = no gate, the reference's
    /// own default. Present only under the declaration — an undeclared
    /// `g_proj` on an MLA layer is refused at closure, never adopted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_gate: Option<OperandRef>,

    /// Epsilon for [`Self::kv_a_norm`] — the family's own value, which on
    /// Kimi Linear is `KimiRMSNorm`'s class default `1e-6` while the
    /// layer's other norms run at `rms_norm_eps` (`1e-5`).
    ///
    /// Carried per-op, never inherited from the layer's norm: the two
    /// differ by a factor of ten on the one checkpoint judged so far, and
    /// this is the operator's own norm site. `None` = the container
    /// carries no judged value (an older container, or a family whose
    /// reference has not been read), and an executor must refuse — the
    /// alternative is running the decompression through a norm the model
    /// never used, with every shape still closing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_a_norm_eps: Option<f64>,
}

impl MlaOp {
    /// `qk_nope_head_dim + qk_rope_head_dim` — one query/key head's full
    /// width, the row width `q_proj` is fused at (`num_heads · this`).
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Elements cached PER POSITION: the compressed latent plus the one
    /// shared rope-K — not `num_heads·(qk_nope_head_dim + v_head_dim)`,
    /// which is what a decompressed K/V pair would cost. This is the
    /// number a KV planner needs; the decompression happens at read time
    /// and is never itself cached.
    pub fn compressed_kv_width(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    /// The five widths in the form the operator's own reference takes
    /// them.
    ///
    /// A projection, not a second record — the same contract
    /// [`KdaOp::geometry`](super::kda::KdaOp::geometry) states: every
    /// field is carried here, so a planner and an executor cannot
    /// disagree about the shape of what one sizes and the other fills.
    pub fn geometry(&self) -> larql_models::config::MlaGeometry {
        larql_models::config::MlaGeometry {
            num_heads: self.num_heads,
            kv_lora_rank: self.kv_lora_rank,
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim,
        }
    }
}

/// How an MLA layer produces its query, as OPERANDS.
///
/// The declaration picks the variant (`q_lora_rank`'s presence, which is
/// the `is not None` the reference branches on) and closure has already
/// held the shipped operands to it from both sides before an op is
/// built. A typed form rather than `Option`s for the same reason
/// [`KdaOutputGate`](super::kda::KdaOutputGate) is one: the two are
/// alternatives, and a shape that can express "both present" or "neither"
/// invites a reader to handle states the checkpoint cannot be in.
///
/// The two variants differ in size (one `OperandRef` against three plus
/// an epsilon) and are deliberately not boxed: an [`MlaOp`] is already
/// `Box`ed inside `LayerAttention::Mla`, so the difference costs one
/// slightly larger allocation per MLA LAYER and nothing per token. A
/// second box here would buy nothing and would put an indirection between
/// a reader and the operands — which is what `ExpertBank` boxes to AVOID,
/// its `PerExpert` variant being three `Vec`s carried on every value of
/// the packed one.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "form", rename_all = "snake_case")]
pub enum MlaQueryProjection {
    /// One dense `q_proj [Hq·(nope+rope), hidden]` — Kimi Linear, and
    /// every MLA family that does not compress its query.
    Direct { q_proj: OperandRef },
    /// `q_a_proj` -> `q_a_layernorm` -> `q_b_proj` — Kimi-K3,
    /// `q_lora_rank: 1536`.
    ///
    /// `q_b_proj` has the SAME row count as [`Self::Direct`]'s `q_proj`
    /// and differs only in its columns (the rank against `hidden`), which
    /// is why the form is declared rather than deduced.
    LowRank {
        /// Query down-projection, `[rank, hidden]`.
        q_a_proj: OperandRef,
        /// RMSNorm weight over the query latent, `[rank]`.
        q_a_norm: OperandRef,
        /// Query up-projection, `[Hq·(nope+rope), rank]`.
        q_b_proj: OperandRef,
        /// Epsilon for `q_a_norm` — the family's own value, carried
        /// beside the operand it belongs to and NOT shared with
        /// [`MlaOp::kv_a_norm_eps`]. They are equal on Kimi-K3 (`1e-6`,
        /// `KimiRMSNorm`'s class default, twice) and that is a shared
        /// cause rather than a shared authority.
        ///
        /// `None` = the container carries no judged value, and an
        /// executor refuses rather than running the compression through a
        /// norm the model never used — the same contract, and the same
        /// `Option`, as [`MlaOp::kv_a_norm_eps`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        q_a_norm_eps: Option<f64>,
    },
}

impl MlaQueryProjection {
    /// Every operand of the form, for readers that iterate an op's
    /// operands without caring which form it is.
    pub fn operands(&self) -> Vec<(&'static str, &OperandRef)> {
        match self {
            Self::Direct { q_proj } => vec![("q_proj", q_proj)],
            Self::LowRank {
                q_a_proj,
                q_a_norm,
                q_b_proj,
                ..
            } => vec![
                ("q_a_proj", q_a_proj),
                ("q_a_layernorm", q_a_norm),
                ("q_b_proj", q_b_proj),
            ],
        }
    }
}
