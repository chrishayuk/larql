//! llama.cpp's `gemma3` architecture (build b9430), as data.
//!
//! Names from `gguf-py/gguf/constants.py` (`MODEL_TENSOR` → `TENSOR_NAMES`
//! for `MODEL_ARCH.GEMMA3`); behaviours from `src/models/gemma3.cpp`:
//! GELU-tanh gated FFN, `1/sqrt(head_dim)` attention scale on every size
//! but 27B, the embedding scaled by `sqrt(n_embd)`, and a sliding-window
//! period of 6 unless the file states another.

use larql_models::config::Activation;

use super::target::{AttentionScaleRule, DenseRole, DenseTarget, EmbeddingScaleRule};

/// `Gemma3TextConfig.max_position_embeddings`' default, which llama.cpp's
/// converter also falls back to.
const GEMMA3_DEFAULT_CONTEXT: u64 = 131_072;
/// `swa_period` in `llama_model_gemma3::load_arch_hparams`.
const GEMMA3_DEFAULT_SWA_PERIOD: usize = 6;

pub const GEMMA3: DenseTarget = DenseTarget {
    families: &["gemma3_text", "gemma3"],
    architecture: "gemma3",
    layer_prefix: "blk",
    tensor_names: &[
        (DenseRole::Embedding, "token_embd.weight"),
        (DenseRole::FinalNorm, "output_norm.weight"),
        (DenseRole::OutputHead, "output.weight"),
        (DenseRole::PreAttentionNorm, "attn_norm.weight"),
        (DenseRole::PostAttentionNorm, "post_attention_norm.weight"),
        (DenseRole::PreFfnNorm, "ffn_norm.weight"),
        (DenseRole::PostFfnNorm, "post_ffw_norm.weight"),
        (DenseRole::Query, "attn_q.weight"),
        (DenseRole::Key, "attn_k.weight"),
        (DenseRole::Value, "attn_v.weight"),
        (DenseRole::AttentionOutput, "attn_output.weight"),
        (DenseRole::QueryNorm, "attn_q_norm.weight"),
        (DenseRole::KeyNorm, "attn_k_norm.weight"),
        (DenseRole::FfnGate, "ffn_gate.weight"),
        (DenseRole::FfnUp, "ffn_up.weight"),
        (DenseRole::FfnDown, "ffn_down.weight"),
    ],
    required_layer_roles: &[
        DenseRole::PreAttentionNorm,
        DenseRole::PostAttentionNorm,
        DenseRole::PreFfnNorm,
        DenseRole::PostFfnNorm,
        DenseRole::Query,
        DenseRole::Key,
        DenseRole::Value,
        DenseRole::AttentionOutput,
        DenseRole::QueryNorm,
        DenseRole::KeyNorm,
        DenseRole::FfnGate,
        DenseRole::FfnUp,
        DenseRole::FfnDown,
    ],
    activation: Activation::GeluTanh,
    attention_scale: AttentionScaleRule::InverseSqrtHeadDim,
    embedding_scale: EmbeddingScaleRule::SqrtHidden,
    default_swa_period: GEMMA3_DEFAULT_SWA_PERIOD,
    default_context_length: GEMMA3_DEFAULT_CONTEXT,
    add_space_prefix: false,
};
