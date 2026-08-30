//! Multi-Latent Attention's declared geometry — the numbers an executor
//! needs, bundled the same way [`super::linear_attn::KdaGeometry`]
//! bundles KDA's, and for the same reason: five tightly related widths
//! that recur across every call site are a magic-number risk spread out,
//! not bundled in one.
//!
//! Deliberately no `q_lora_rank` field: Kimi Linear ships none
//! (`assert self.q_lora_rank is None` in the checkpoint's own
//! `modeling_kimi.py` — Q is one dense projection, only K/V are low-rank
//! compressed). A family that DOES compress Q needs its own extension
//! here, not a guess bolted onto this one — the same discipline
//! `crate::inventory::report::MlaExecution` already states for the
//! resolved-facts side of this same geometry.

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
