//! GPT-Neo architecture (`GPTNeoForCausalLM`, EleutherAI; TinyStories).
//!
//! The GPT-2 block with four differences, each read from
//! `transformers.models.gpt_neo.modeling_gpt_neo` rather than inferred:
//!
//! - **`nn.Linear`, not `Conv1D`**: every projection is stored `[out,
//!   in]`, and Q, K, V are three separate projections under
//!   `attn.attention.{q,k,v}_proj` — so the storage layout derives
//!   nothing, where GPT-2's transposes and splits.
//! - **Biases**: `q_proj`/`k_proj`/`v_proj` are `bias=False` and
//!   `out_proj` is biased — the output-only attention bias form.
//! - **No score scaling**: `GPTNeoSelfAttention._attn` is `query @
//!   key.T` with no `1/sqrt(head_dim)` anywhere (the weights were trained
//!   that way).
//! - **Global/local interleave**: `attention_types` (expanded into
//!   `attention_layers`) alternates `global` and `local` layers, and a
//!   local layer sees the trailing `window_size` positions, the query's
//!   own included (`bias ^ tril(bias, -window_size)`).
//!
//! Shared with GPT-2: LayerNorm with bias at every site
//! (`layer_norm_epsilon`), a non-gated biased MLP (`c_fc` → `gelu_new` →
//! `c_proj`, width `intermediate_size` or `4 * hidden_size`), learned
//! absolute positions (`wpe`, `max_position_embeddings` rows) added to the
//! token embedding, and a head tied to `wte`.

use crate::config::{
    Activation, ActivationDeclaration, FfnType, ModelArchitecture, ModelConfig, NonParameterBuffer,
    NormType, SourceLayout,
};

pub struct GptNeoArch {
    config: ModelConfig,
}

impl GptNeoArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

use crate::config::architecture_prelude::*;

impl ArchitectureCore for GptNeoArch {
    fn family(&self) -> &str {
        "gpt_neo"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// `GPTNeoSelfAttention`'s registered buffers: `bias` is the causal mask
/// (and on a local layer the causal mask with everything older than the
/// window removed), `masked_bias` the fill value of a masked score. The
/// declared span and window reproduce the first; the second is never read
/// by a softmax over the visible keys.
const GPT_NEO_MASK_BUFFER_REASON: &str =
    "registered attention-mask buffer, not a parameter: the attention's declared causal span \
     and window reproduce it";

const GPT_NEO_SOURCE_LAYOUT: SourceLayout = SourceLayout {
    input_major: &[],
    fused: &[],
    buffers: &[
        NonParameterBuffer {
            suffix: "attn.attention.bias",
            reason: GPT_NEO_MASK_BUFFER_REASON,
        },
        NonParameterBuffer {
            suffix: "attn.attention.masked_bias",
            reason: GPT_NEO_MASK_BUFFER_REASON,
        },
    ],
};

impl TensorKeys for GptNeoArch {
    fn position_embed_key(&self) -> Option<&str> {
        Some("wpe.weight")
    }

    fn source_layout(&self) -> SourceLayout {
        GPT_NEO_SOURCE_LAYOUT
    }
}

impl Norms for GptNeoArch {
    fn norm_type(&self) -> NormType {
        NormType::LayerNorm
    }

    /// `nn.LayerNorm` with its default affine bias, at every site.
    fn norm_bias(&self) -> Option<bool> {
        Some(true)
    }
}

impl Position for GptNeoArch {}

impl Attention for GptNeoArch {
    /// `_attn` multiplies query by key and never divides.
    fn scales_attention_scores(&self) -> bool {
        false
    }

    /// Only `out_proj` is biased.
    fn attention_output_bias(&self) -> Option<bool> {
        Some(true)
    }
}

impl FeedForward for GptNeoArch {
    /// The checkpoint's `activation_function` when it states one; a silent
    /// config runs `GPTNeoConfig`'s default, `gelu_new`.
    fn activation_declaration(&self) -> ActivationDeclaration {
        match self.config().hidden_act.as_deref() {
            None => ActivationDeclaration::Nonlinearity(Activation::GeluTanh),
            declared => ActivationDeclaration::judge(declared),
        }
    }

    fn ffn_type(&self) -> FfnType {
        FfnType::Standard
    }

    /// `c_fc` and `c_proj` are biased `nn.Linear`s.
    fn ffn_bias(&self) -> Option<bool> {
        Some(true)
    }
}

impl LatentAttention for GptNeoArch {}
impl Embeddings for GptNeoArch {}
impl ModelArchitecture for GptNeoArch {}
