//! Conv-QKV attention — the hybrid Mamba2Attn stack's attention block,
//! transcribed from the reference implementation's own forward
//! (`modeling_mamba2attn.py`, `Mamba2Attention`).
//!
//! One fused QKV projection, a depthwise causal conv over the FULL
//! fused QKV **with no activation** (unlike the Mamba2 mixer's conv,
//! which applies SiLU), a q|k|v split, partial rotary on the leading
//! `rotary_dim` dims of each head (rotate-half convention, frequencies
//! over the rotary width), ordinary causal softmax at `1/√head_dim`
//! with fp32 score accumulation, GQA by `repeat_interleave`, and an
//! output projection.
//!
//! Its continuation state is a KV cache **and** a conv history — the
//! first operator to declare both: the conv history holds the last
//! `conv_kernel` positions of the PRE-conv fused QKV (the reference
//! caches full kernel width, left-padded), while K/V are cached
//! post-conv, post-rotary.

use larql_models::config::ConvQkvAttnGeometry;
use serde::Serialize;

use super::OperandRef;

/// One layer's conv-QKV attention operator.
///
/// Every field is transcribed from the container's own operand roles and
/// execution surface. The geometry travels whole
/// ([`ConvQkvAttnGeometry`]) — its derived widths (`qkv_rows`,
/// `attn_out_width`) are computed, never stored, so they cannot drift
/// from the declaration they close over.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConvQkvOp {
    /// The declared block geometry (head counts and width, conv kernel,
    /// partial-rotary width and base, bias switches).
    pub geometry: ConvQkvAttnGeometry,
    /// Whether the residual stream is kept at fp32
    /// (`residual_in_fp32`). `None` = undeclared.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_in_fp32: Option<bool>,

    /// Fused QKV projection q|k|v, `[(Hq + 2·Hkv)·Dh, hidden]`.
    pub in_proj: OperandRef,
    /// Depthwise causal conv over the full fused QKV,
    /// `[(Hq + 2·Hkv)·Dh, 1, conv_kernel]`. No activation follows it.
    pub conv1d: OperandRef,
    /// Conv bias `[(Hq + 2·Hkv)·Dh]` — present iff `use_conv_bias`
    /// (one switch governs both block kinds' convs in this lineage).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conv1d_bias: Option<OperandRef>,
    /// Output projection, `[hidden, Hq·Dh]`.
    pub out_proj: OperandRef,
}
