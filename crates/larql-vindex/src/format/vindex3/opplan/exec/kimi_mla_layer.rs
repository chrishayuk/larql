//! One complete Kimi Linear MLA decoder layer, executed — the same
//! composition [`super::kimi_kda_layer`] proved for the KDA family,
//! around the OTHER attention operator. Transcribed from the same
//! `KimiDecoderLayer.forward` in the checkpoint's own `modeling_kimi.py`
//! — the residual/norm shape does not change between attention
//! families, only which `self_attn` runs:
//!
//! ```text
//! residual = x
//! h = input_layernorm(x)
//! h = self_attn(h)                          # MLA — exec::mla::mla_forward
//! h = residual + h
//! residual = h
//! h = post_attention_layernorm(h)
//! h = block_sparse_moe(h)                   # exec::kimi_moe_block, UNCHANGED from P3d-h
//! h = residual + h
//! return h
//! ```
//!
//! **Nothing new is transcribed here.** Both norms route to the crate's
//! existing trusted `exec::kernels::norm`, exactly as the KDA layer
//! composition does. The MoE side is the SAME `moe_block_forward` P3d-h
//! already proved — this rung's only new claim is the attention family,
//! not routing.

use larql_models::config::{MlaGeometry, NormType};

use super::kernels::norm;
use super::kimi_moe_block::{moe_block_forward, ExpertWeights, MoeBlockTrace};
use super::mla::{mla_forward, MlaState, MlaTrace, MlaWeights, Mutation as MlaMutation};
use super::timing::{timed, OpClass};

/// Every boundary the layer crosses, so a disagreement against the
/// reference names its own stage — attention or MoE, never "the layer".
/// Same posture as [`super::kimi_kda_layer::KdaDecoderLayerTrace`].
#[derive(Debug, Clone, PartialEq)]
pub struct MlaDecoderLayerTrace {
    /// `input_layernorm(x)` — what the attention actually reads.
    pub input_normed: Vec<f32>,
    /// The MLA operator's own full boundary trace, this position only.
    pub attention: MlaTrace,
    /// `x + attention.output` — the residual stream after attention.
    pub after_attention: Vec<f32>,
    /// `post_attention_layernorm(after_attention)` — what the MoE block
    /// actually reads.
    pub post_attention_normed: Vec<f32>,
    /// The MoE block's own full boundary trace.
    pub moe: MoeBlockTrace,
    /// `after_attention + moe.output` — the layer's output.
    pub output: Vec<f32>,
}

/// One token through one complete MLA decoder layer. `x` is the RAW
/// residual stream in; `state` carries the MLA KV cache ACROSS calls —
/// call once per position, in order, to reproduce a causal sequence
/// (see `exec::mla`'s own doc comment for why a single call cannot
/// exercise the attention math at all).
#[allow(clippy::too_many_arguments)]
pub fn mla_decoder_layer_forward<'a>(
    x: &[f32],
    hidden: usize,
    input_norm_weight: &[f32],
    post_attention_norm_weight: &[f32],
    norm_eps: f64,
    mla_weights: MlaWeights<'_>,
    mla_geometry: MlaGeometry,
    mla_state: &mut MlaState,
    inter: usize,
    router_weight: &[f32],
    router_bias: &[f32],
    experts: usize,
    top_k: usize,
    renormalize: bool,
    branch_scale: f64,
    expert_weights: impl Fn(usize) -> ExpertWeights<'a> + Sync,
    shared: Option<(ExpertWeights<'a>, usize)>,
) -> MlaDecoderLayerTrace {
    let input_normed = {
        let _t = timed(OpClass::Norm);
        norm(NormType::RmsNorm, x, input_norm_weight, 0.0, norm_eps)
    };

    let attention = {
        let _t = timed(OpClass::Mla);
        mla_forward(
            &input_normed,
            hidden,
            mla_weights,
            mla_geometry,
            mla_state,
            MlaMutation::None,
        )
    };

    let after_attention: Vec<f32> = {
        let _t = timed(OpClass::Residual);
        x.iter()
            .zip(&attention.output)
            .map(|(&r, &a)| r + a)
            .collect()
    };

    let post_attention_normed = {
        let _t = timed(OpClass::Norm);
        norm(
            NormType::RmsNorm,
            &after_attention,
            post_attention_norm_weight,
            0.0,
            norm_eps,
        )
    };

    let moe = moe_block_forward(
        &post_attention_normed,
        hidden,
        inter,
        router_weight,
        router_bias,
        experts,
        top_k,
        renormalize,
        branch_scale,
        expert_weights,
        shared,
    );

    let output: Vec<f32> = {
        let _t = timed(OpClass::Residual);
        after_attention
            .iter()
            .zip(&moe.output)
            .map(|(&r, &m)| r + m)
            .collect()
    };

    MlaDecoderLayerTrace {
        input_normed,
        attention,
        after_attention,
        post_attention_normed,
        moe,
        output,
    }
}
