//! Conv-QKV attention — the hybrid Mamba2Attn stack's attention operator.
//!
//! The `mamba_ssm` lineage's `MHA` block with `d_conv > 0`: one fused QKV
//! projection, a depthwise **causal conv over the full fused QKV** before
//! the heads are split, partial rotary on the leading `rotary_emb_dim`
//! dims of each head, then ordinary causal softmax attention and an
//! output projection. It is NOT plain softmax attention — reading it as
//! one would drop the conv (a real mixing step with its own continuation
//! state) and rotate the whole head instead of the declared fraction.
//!
//! Declared by OuteAI's Mamba2Attn configs (`attention_*`/`rope_emb_dim`
//! keys). The state-spaces `mamba2attn` checkpoints run the same block
//! but declare none of this geometry — that absence is that family's own
//! admission judgment, not a default to fill here.
//!
//! Every field is read from the checkpoint's declaration; a partial
//! declaration is refused rather than completed — the contract every
//! geometry read in this module holds.

use serde::{Deserialize, Serialize};

/// The conv-QKV attention block's declared geometry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConvQkvAttnGeometry {
    /// `num_attention_heads` — query heads (16).
    pub num_heads: usize,
    /// `num_key_value_heads` — key/value heads (16; the block supports
    /// GQA, the observed checkpoint does not use it).
    pub num_kv_heads: usize,
    /// `attention_head_dim` — one head's width (128). Declared apart from
    /// the Mamba2 mixer's `head_dim`, and NOT `hidden_size / num_heads`
    /// (16 · 128 = 2048 ≠ 1024 on the observed checkpoint).
    pub head_dim: usize,
    /// `attention_conv_kernel` — the depthwise causal conv's width over
    /// the fused QKV rows (4). The per-layer conv history is
    /// `conv_kernel - 1` positions of the full QKV width.
    pub conv_kernel: usize,
    /// `rope_emb_dim` — leading dims of each head that rotate (64 of
    /// 128); the rest of the head is unrotated. The inverse-frequency
    /// series is taken over this rotary width (`base^(2i/rotary_dim)`) —
    /// the plain partial rotary.
    pub rotary_dim: usize,
    /// `rope_theta` — rotary base frequency (10000).
    pub rope_theta: f64,
    /// `use_attention_qkv_bias` — whether the fused QKV projection
    /// carries a bias.
    pub qkv_bias: bool,
    /// `use_attention_out_bias` — whether the output projection carries
    /// a bias.
    pub out_bias: bool,
}

impl ConvQkvAttnGeometry {
    /// Read the geometry from a (text) config object. All fields or none.
    pub fn read(config: &serde_json::Value) -> Option<Self> {
        let dim = |key: &str| config[key].as_u64().map(|v| v as usize).filter(|v| *v > 0);
        Some(Self {
            num_heads: dim("num_attention_heads")?,
            num_kv_heads: dim("num_key_value_heads")?,
            head_dim: dim("attention_head_dim")?,
            conv_kernel: dim("attention_conv_kernel")?,
            rotary_dim: dim("rope_emb_dim")?,
            rope_theta: config["rope_theta"].as_f64()?,
            qkv_bias: config["use_attention_qkv_bias"].as_bool()?,
            out_bias: config["use_attention_out_bias"].as_bool()?,
        })
    }

    /// Rows the fused QKV projection emits:
    /// `(num_heads + 2 · num_kv_heads) · head_dim` — also the depthwise
    /// conv's channel count and the width of one conv-history position.
    /// 6144 on the observed checkpoint, the row count that tells an
    /// attention mixer apart from a Mamba2 mixer in the tensor estate.
    pub fn qkv_rows(self) -> usize {
        (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
    }

    /// Width of the attention output — `num_heads · head_dim`, the
    /// output projection's input side (2048).
    pub fn attn_out_width(self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Cross-field defects. Empty when the declaration is internally
    /// consistent.
    pub fn geometry_defects(self) -> Vec<String> {
        let mut defects = Vec::new();
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            defects.push(format!(
                "num_attention_heads ({}) is not a multiple of num_key_value_heads ({})",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.rotary_dim > self.head_dim {
            defects.push(format!(
                "rope_emb_dim ({}) exceeds attention_head_dim ({})",
                self.rotary_dim, self.head_dim
            ));
        }
        if !self.rotary_dim.is_multiple_of(2) {
            defects.push(format!(
                "rope_emb_dim ({}) is odd — rotation pairs dims",
                self.rotary_dim
            ));
        }
        defects
    }
}
