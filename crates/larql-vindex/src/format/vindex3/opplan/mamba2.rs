//! Mamba2/SSD: an attention-class operator whose continuation state is a
//! per-head `head_dim × state_size` matrix plus a short conv history —
//! never a growing KV cache.
//!
//! mamba2-780m declares ALL 48 of its layers this operator: the pure-SSM
//! witness for schema 6, where zero attention exists anywhere in the
//! model. The operand set shares nothing with
//! [`AttentionOp`](super::AttentionOp), [`GatedDeltaOp`](super::gated_delta::GatedDeltaOp)
//! or [`KdaOp`](super::kda::KdaOp): one fused five-way input projection
//! (z|x|B|C|dt) where DeltaNet splits qkv|a|b|z; a depthwise causal conv
//! over the x|B|C channels ONLY (the gate is deliberately excluded, where
//! DeltaNet convolves its whole fused projection); per-**head** scalar
//! decay/skip/timestep against KDA's per-channel `dt_bias`; and there is
//! no FFN in the layer at all — the mixer is the whole block.

use larql_models::config::{Activation, DtBound, Mamba2Geometry};
use serde::Serialize;

use super::{NormOp, OperandRef};

/// One layer's Mamba2/SSD operator.
///
/// Every field is transcribed from the container's own operand roles and
/// execution surface. The geometry travels whole
/// ([`Mamba2Geometry`]) — its derived widths (`d_inner`, `conv_dim`,
/// `in_proj_rows`) are computed, never stored, so they cannot drift from
/// the declaration they close over.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Mamba2Op {
    /// The declared mixer geometry (`state_size`, `num_heads`,
    /// `head_dim`, `expand`, `conv_kernel`, `n_groups`, `chunk_size`, the
    /// dt clamp and the bias/norm switches).
    pub geometry: Mamba2Geometry,
    /// The mixer's nonlinearity (`hidden_act`) — applied by the conv
    /// branch and by the output gate.
    pub activation: Activation,
    /// Whether the residual stream is kept at fp32
    /// (`residual_in_fp32`). `None` = undeclared.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_in_fp32: Option<bool>,

    /// Fused input projection z|x|B|C|dt,
    /// `[2·d_inner + 2·groups·state + heads, hidden]`.
    pub in_proj: OperandRef,
    /// Depthwise causal conv over x|B|C, `[conv_dim, 1, conv_kernel]`.
    pub conv1d: OperandRef,
    /// Conv bias `[conv_dim]` — present iff `use_conv_bias`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conv1d_bias: Option<OperandRef>,
    /// Per-head log decay, `[num_heads]`.
    pub a_log: OperandRef,
    /// Per-head skip weight, `[num_heads]`.
    pub d: OperandRef,
    /// Per-head timestep bias, `[num_heads]`.
    pub dt_bias: OperandRef,
    /// Gated RMSNorm between state read-out and the output projection,
    /// over the full inner width `[d_inner]` — present iff `rms_norm`.
    /// A complete [`NormOp`] so the epsilon travels with the weight.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gated_norm: Option<NormOp>,
    /// Output projection, `[hidden, d_inner]`.
    pub out_proj: OperandRef,
}

impl Mamba2Op {
    /// Elements in this layer's SSM state: one `head_dim × state_size`
    /// matrix per head. Constant in sequence length — the property that
    /// makes this a different runtime problem from softmax attention.
    pub fn state_elements(&self) -> usize {
        self.geometry.state_elements()
    }

    /// Elements in this layer's conv history: `conv_dim` channels of the
    /// last `conv_kernel` pre-convolution inputs — the full window the
    /// reference cache keeps, given the component's hidden size.
    pub fn conv_state_elements(&self, hidden_size: usize) -> usize {
        self.geometry.conv_dim(hidden_size) * self.geometry.conv_kernel
    }

    /// Whether the forward-time dt clamp is the identity (fully
    /// unbounded on both sides) — the released checkpoints clamp below at
    /// 0.0 and not above.
    pub fn dt_clamp_is_identity(&self) -> bool {
        self.geometry.dt_limit_min == DtBound::Unbounded
            && self.geometry.dt_limit_max == DtBound::Unbounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op() -> Mamba2Op {
        let geometry = Mamba2Geometry::read(&serde_json::json!({
            "state_size": 128, "num_heads": 48, "head_dim": 64, "expand": 2,
            "conv_kernel": 4, "n_groups": 1, "chunk_size": 256,
            "time_step_limit": [0.0, "Infinity"],
            "rms_norm": true, "use_bias": false, "use_conv_bias": true
        }))
        .unwrap();
        let operand = |name: &str, shape: Vec<usize>| OperandRef {
            object: "target.decoder_stack".into(),
            tensor: name.into(),
            dtype: "F16".into(),
            shape,
        };
        Mamba2Op {
            geometry,
            activation: Activation::Silu,
            residual_in_fp32: Some(true),
            in_proj: operand("0.mixer.in_proj.weight", vec![6448, 1536]),
            conv1d: operand("0.mixer.conv1d.weight", vec![3328, 1, 4]),
            conv1d_bias: Some(operand("0.mixer.conv1d.bias", vec![3328])),
            a_log: operand("0.mixer.A_log", vec![48]),
            d: operand("0.mixer.D", vec![48]),
            dt_bias: operand("0.mixer.dt_bias", vec![48]),
            gated_norm: None,
            out_proj: operand("0.mixer.out_proj.weight", vec![1536, 3072]),
        }
    }

    /// **The state is constant in sequence length, at the real geometry:**
    /// 48 heads of `64 × 128`, plus a conv history of `3328 × 3`.
    #[test]
    fn state_and_conv_history_sizes_close_over_the_real_geometry() {
        let op = op();
        assert_eq!(op.state_elements(), 48 * 64 * 128);
        assert_eq!(op.conv_state_elements(1536), 3328 * 4);
    }

    /// The released clamp is bounded below (0.0) and unbounded above —
    /// not the identity.
    #[test]
    fn the_released_dt_clamp_is_not_the_identity() {
        assert!(!op().dt_clamp_is_identity());
    }
}
