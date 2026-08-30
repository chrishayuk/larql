//! One complete Kimi Linear KDA decoder layer, executed — the first place
//! every already-proven piece meets. Transcribed from `KimiDecoderLayer.
//! forward` in the checkpoint's own `modeling_kimi.py`:
//!
//! ```text
//! residual = x
//! h = input_layernorm(x)
//! h = self_attn(h)                          # KDA — exec::kda::layer_forward
//! h = residual + h
//! residual = h
//! h = post_attention_layernorm(h)
//! h = block_sparse_moe(h)                   # exec::kimi_moe_block
//! h = residual + h
//! return h
//! ```
//!
//! **Nothing new is transcribed here.** `input_layernorm`/`post_attention_
//! layernorm` are plain `KimiRMSNorm` — no offset, no gate — so they route
//! to `exec::kernels::norm`, the crate's existing trusted RMSNorm
//! reference (shared with every other norm site in the plan executor,
//! never KDA's OWN internal gated `o_norm`, which is a different operand
//! at a different width). The attention and MoE math are exactly
//! `exec::kda::layer_forward` and `exec::kimi_moe_block::moe_block_forward`
//! — this file's only job is the residual/norm composition around them.

use larql_models::config::{KdaGeometry, NormType};

use super::kda::{layer_forward, KdaPlanes, KdaState, KdaWeights, Mutation as KdaMutation};
use super::kernels::norm;
use super::kimi_moe_block::{expert_ffn, moe_block_forward, ExpertWeights, MoeBlockTrace};
use super::timing::{timed, OpClass};

/// Every boundary the layer crosses, so a disagreement against the
/// reference names its own stage — attention or MoE, never "the layer" —
/// the same posture `KdaPlanes`/`RouterTrace`/`MoeBlockTrace` each take
/// for their own pieces.
#[derive(Debug, Clone, PartialEq)]
pub struct KdaDecoderLayerTrace {
    /// `input_layernorm(x)` — what the attention actually reads.
    pub input_normed: Vec<f32>,
    /// The KDA operator's own full boundary trace.
    pub attention: KdaPlanes,
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

/// One token through one complete KDA decoder layer. `x` is the RAW
/// residual stream in — `input_norm_weight`/`post_attention_norm_weight`
/// are applied here, not pre-applied by the caller, so this function's
/// contract matches `KimiDecoderLayer.forward`'s exactly: hidden state in,
/// hidden state out.
#[allow(clippy::too_many_arguments)]
pub fn kda_decoder_layer_forward<'a>(
    x: &[f32],
    hidden: usize,
    input_norm_weight: &[f32],
    post_attention_norm_weight: &[f32],
    norm_eps: f64,
    kda_weights: KdaWeights<'_>,
    kda_geometry: KdaGeometry,
    kda_state: &mut KdaState,
    inter: usize,
    router_weight: &[f32],
    router_bias: &[f32],
    experts: usize,
    top_k: usize,
    renormalize: bool,
    branch_scale: f64,
    expert_weights: impl Fn(usize) -> ExpertWeights<'a> + Sync,
    shared: Option<(ExpertWeights<'a>, usize)>,
) -> KdaDecoderLayerTrace {
    let input_normed = {
        let _t = timed(OpClass::Norm);
        norm(NormType::RmsNorm, x, input_norm_weight, 0.0, norm_eps)
    };

    // No outer `OpClass::Kda` wrap here: `layer_forward` -> `step` carries
    // its own eleven fine-grained `Kda*` timers (P4c), and `timing.rs`'s
    // leaves-are-disjoint contract forbids nesting a coarse class around
    // them.
    let attention = layer_forward(
        &input_normed,
        hidden,
        kda_weights,
        kda_geometry,
        kda_state,
        KdaMutation::None,
    );

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

    KdaDecoderLayerTrace {
        input_normed,
        attention,
        after_attention,
        post_attention_normed,
        moe,
        output,
    }
}

/// Every boundary [`kda_dense_decoder_layer_forward`] crosses — the same
/// shape [`KdaDecoderLayerTrace`] holds, minus the router/expert-combine
/// stages a dense layer has none of.
#[derive(Debug, Clone, PartialEq)]
pub struct KdaDenseDecoderLayerTrace {
    pub input_normed: Vec<f32>,
    pub attention: KdaPlanes,
    pub after_attention: Vec<f32>,
    pub post_attention_normed: Vec<f32>,
    /// `KimiMLP.forward(post_attention_normed)` — `w2(silu(w1(x))*w3(x))`,
    /// the SAME shape a routed expert or the shared branch uses (see
    /// `kimi_moe_block::expert_ffn`'s own doc comment), just called once
    /// with this layer's own dense weights, no routing, no combine.
    pub ffn_output: Vec<f32>,
    pub output: Vec<f32>,
}

/// One token through Kimi's ONE dense KDA decoder layer — layer 0, the
/// only layer `first_k_dense_replace=1` excludes from MoE routing.
/// Otherwise identical to [`kda_decoder_layer_forward`]: same residual/
/// norm composition, same KDA attention, only the FFN branch differs —
/// `KimiMLP` in place of `KimiSparseMoeBlock`, per `KimiDecoderLayer.
/// __init__`'s own `if/else` on whether a layer carries
/// `block_sparse_moe` or `mlp`.
#[allow(clippy::too_many_arguments)]
pub fn kda_dense_decoder_layer_forward(
    x: &[f32],
    hidden: usize,
    input_norm_weight: &[f32],
    post_attention_norm_weight: &[f32],
    norm_eps: f64,
    kda_weights: KdaWeights<'_>,
    kda_geometry: KdaGeometry,
    kda_state: &mut KdaState,
    ffn_weights: ExpertWeights<'_>,
    inter: usize,
) -> KdaDenseDecoderLayerTrace {
    let input_normed = {
        let _t = timed(OpClass::Norm);
        norm(NormType::RmsNorm, x, input_norm_weight, 0.0, norm_eps)
    };

    // No outer `OpClass::Kda` wrap here: `layer_forward` -> `step` carries
    // its own eleven fine-grained `Kda*` timers (P4c), and `timing.rs`'s
    // leaves-are-disjoint contract forbids nesting a coarse class around
    // them.
    let attention = layer_forward(
        &input_normed,
        hidden,
        kda_weights,
        kda_geometry,
        kda_state,
        KdaMutation::None,
    );

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

    let ffn_output = {
        let _t = timed(OpClass::MoeRoutedExpert);
        expert_ffn(&post_attention_normed, ffn_weights, hidden, inter)
    };

    let output: Vec<f32> = {
        let _t = timed(OpClass::Residual);
        after_attention
            .iter()
            .zip(&ffn_output)
            .map(|(&r, &f)| r + f)
            .collect()
    };

    KdaDenseDecoderLayerTrace {
        input_normed,
        attention,
        after_attention,
        post_attention_normed,
        ffn_output,
        output,
    }
}
