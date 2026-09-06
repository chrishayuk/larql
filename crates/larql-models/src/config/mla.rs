//! Multi-Latent Attention's declared geometry — the numbers an executor
//! needs, bundled the same way [`super::linear_attn::KdaGeometry`]
//! bundles KDA's, and for the same reason: five tightly related widths
//! that recur across every call site are a magic-number risk spread out,
//! not bundled in one.
//!
//! Deliberately no `q_lora_rank` field on [`MlaGeometry`]: Kimi Linear
//! ships none (`assert self.q_lora_rank is None` in the checkpoint's own
//! `modeling_kimi.py` — Q is one dense projection, only K/V are low-rank
//! compressed), and the five widths here are what EVERY MLA layer has.
//!
//! K3-MLA-Q-LORA-1 is the extension that doc asked for, and it is a
//! separate type rather than a sixth field: [`MlaQueryForm`] says which
//! query a layer builds, and carries the rank and the q-A norm epsilon
//! inside the variant that has them. A layer with no factorisation
//! therefore cannot carry a rank, and a rank cannot exist without the
//! norm epsilon that goes with it.

use serde::{Deserialize, Serialize};

/// Multi-Latent Attention: one dense query projection, a shared
/// (MQA-style) low-rank KV compression, and per-head decompression into
/// an asymmetric nope+rope query width and a DIFFERENT value width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlaGeometry {
    /// Query/output head count (`num_attention_heads`) — the
    /// decompressed K/V side always produces this many heads' worth of
    /// output, not `num_key_value_heads`: MLA's compression is the
    /// efficiency mechanism, not a GQA-style head reduction after
    /// decompression.
    pub num_heads: usize,
    /// Compressed KV latent width (`kv_lora_rank`).
    pub kv_lora_rank: usize,
    /// Non-RoPE portion of the query/key head width.
    pub qk_nope_head_dim: usize,
    /// RoPE portion of the query/key head width — one SHARED projection
    /// across every head. "RoPE" is the field's inherited DeepSeek name,
    /// not a promise this family rotates it: Kimi asserts
    /// `mla_use_nope=True` and its own `forward` never calls a rotary
    /// embedding on this component.
    pub qk_rope_head_dim: usize,
    /// Value head width — independent of the query/key head width; MLA's
    /// asymmetry is structural, not an approximation.
    pub v_head_dim: usize,
}

impl MlaGeometry {
    /// `qk_nope_head_dim + qk_rope_head_dim` — one query/key head's full
    /// width, the row width `q_proj` is fused at (`num_heads · this`).
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Elements `kv_a_proj_with_mqa` produces per position, before any
    /// split or norm: the compressed latent plus the one shared rope-K.
    pub fn compressed_kv_width(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
}

/// How a layer builds its QUERY — the one thing Kimi-K3's MLA changes
/// about the operator, and a declaration rather than an observation.
///
/// The reference branches on `config.q_lora_rank is not None`
/// (`modeling_kimi_linear.py` L364, L418): a presence test on a declared
/// field, and a q-LoRA layer has no `q_proj` attribute at all while a
/// direct layer has none of the triple. So the FORM is chosen here, by
/// the declaration, and the operand plane is held to it from both sides —
/// never the reverse. A build that picked the form from whichever
/// tensors happened to be present would answer a different question, and
/// would answer it silently for a checkpoint that shipped both.
///
/// The trap this exists to survive: `q_proj` and `q_b_proj` have the
/// SAME row count (`num_heads · q_head_dim`, 18432 on K3). Only the
/// COLUMN count separates them — `hidden` against the rank — and the
/// authority for that column count is this enum.
/// No `Eq`: `norm_eps` is a float, for the same reason `MlaSurface`
/// carries none.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "snake_case")]
pub enum MlaQueryForm {
    /// One dense `q_proj [num_heads·q_head_dim, hidden]`. Kimi Linear,
    /// and what an undeclared `q_lora_rank` means — a CHECKED default,
    /// read from the reference's own signature
    /// (`q_lora_rank: Optional[int] = None`), not an assumed one.
    Direct,
    /// `q_a_proj [rank, hidden]` -> `q_a_layernorm [rank]` ->
    /// `q_b_proj [num_heads·q_head_dim, rank]`. Kimi-K3, `q_lora_rank:
    /// 1536`.
    ///
    /// `rank` is carried verbatim, INCLUDING zero: `0 is not None` in the
    /// reference, so a declared `0` selects this form and then describes
    /// a degenerate geometry. Normalising it away here would be inventing
    /// a semantic the checkpoint did not state — and note the deliberate
    /// contrast with `activation_situ_beta`, where the same checkpoint's
    /// `beta or 1.0` DOES turn a declared zero into one. Two adjacent
    /// fields, two opposite rules, neither inferred from the other.
    LowRank {
        rank: usize,
        /// `q_a_layernorm`'s epsilon — its OWN authority, never borrowed.
        ///
        /// `KimiRMSNorm(self.q_lora_rank)` (L368) passes no `eps`, so it
        /// runs at the class default `1e-6`, not `config.rms_norm_eps`
        /// (`1e-5`, what the layer's own two norms use). That is the same
        /// property [`MlaSurface::kv_a_norm_eps`] carries for the KV
        /// latent norm — and the two are equal today because they share a
        /// CAUSE, one class default, not because they share an AUTHORITY.
        /// A single accessor for both would make a coincidence into a
        /// contract and would be silently wrong for a family that
        /// overrode one and not the other.
        ///
        /// Inside the variant rather than beside it, so an epsilon for a
        /// norm the layer does not have is unrepresentable.
        ///
        /// `None` = **unjudged** — this family factorises its query and
        /// nobody has read its reference for the epsilon. Never "use the
        /// KV one", never "use the layer eps": an executor refuses. Same
        /// contract, and the same `Option`, that `kv_a_norm_eps` already
        /// carries for the latent norm.
        ///
        /// [`MlaSurface::kv_a_norm_eps`]: crate::config::MlaGeometry
        norm_eps: Option<f64>,
    },
}

impl MlaQueryForm {
    /// The rank, when the query is factorised.
    pub fn rank(self) -> Option<usize> {
        match self {
            Self::Direct => None,
            Self::LowRank { rank, .. } => Some(rank),
        }
    }

    /// `q_a_layernorm`'s epsilon: `None` under the direct form (there is
    /// no such norm) and `None` under an unjudged low-rank form (there is
    /// no such judgment). The two are told apart by [`Self::is_low_rank`]
    /// — collapsing them here would let an executor treat "no norm" and
    /// "a norm nobody judged" the same way.
    pub fn norm_eps(self) -> Option<f64> {
        match self {
            Self::Direct => None,
            Self::LowRank { norm_eps, .. } => norm_eps,
        }
    }

    /// Whether the query is factorised. Named so call sites read as the
    /// question they are asking rather than as a pattern match.
    pub fn is_low_rank(self) -> bool {
        matches!(self, Self::LowRank { .. })
    }
}
