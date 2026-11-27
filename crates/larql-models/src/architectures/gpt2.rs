//! GPT-2 architecture.
//!
//! Key differences from standard Llama:
//! - Standard LayerNorm (with bias) instead of RMSNorm
//! - Non-gated FFN: activation(x @ c_fc.T + bias) @ c_proj.T + bias, GELU activation
//! - Conv1D-style fused QKV (`attn_qkv.weight`) instead of separate Q/K/V
//! - Learned position embeddings (`position_embd.weight`) instead of RoPE
//! - LM head tied to input embedding
//!
//! Tensor key naming after the GGUF→HF normalization in `loading/gguf.rs`
//! matches the trait defaults (`embed_tokens.weight`, `mlp.up_proj.weight`,
//! `mlp.down_proj.weight`, …), so this arch only overrides behavior flags.

use crate::config::{
    Activation, ActivationDeclaration, BufferKind, FfnType, FusedProjection, ModelArchitecture,
    ModelConfig, NonParameterBuffer, NormType, SourceLayout,
};
use crate::detect::LayerBandSplit;
use crate::tensor_keys::attn_bias;

pub struct Gpt2Arch {
    config: ModelConfig,
}

impl Gpt2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

use crate::config::architecture_prelude::*;

impl ArchitectureCore for Gpt2Arch {
    fn family(&self) -> &str {
        "gpt2"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl TensorKeys for Gpt2Arch {
    /// GPT-2 packs Q, K, V into a single Conv1D `c_attn` projection. The
    /// GGUF→HF normaliser maps `attn_qkv.` → `self_attn.qkv_proj.`; the
    /// loader's `split_fused_qkv` pass then materialises the per-projection
    /// q/k/v tensors at the trait's standard `attn_q_key`/`attn_k_key`/
    /// `attn_v_key` so downstream code stays family-agnostic.
    fn fused_qkv_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.qkv_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn fused_qkv_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.qkv_proj.bias",
            self.layer_prefix(layer)
        ))
    }

    /// GPT-2 has bias on every projection (Conv1D layers carry bias).
    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::q(&self.layer_prefix(layer))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::k(&self.layer_prefix(layer))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::v(&self.layer_prefix(layer))
    }

    fn attn_o_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::o(&self.layer_prefix(layer))
    }

    fn ffn_up_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.up_proj.bias", self.layer_prefix(layer)))
    }

    fn ffn_down_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.down_proj.bias", self.layer_prefix(layer)))
    }

    /// GPT-2 uses learned absolute position embeddings (`wpe`) added to the
    /// token embedding at the input — no rotary positional signal.
    fn position_embed_key(&self) -> Option<&str> {
        Some("wpe.weight")
    }

    /// HF GPT-2 stores every projection as `transformers.Conv1D` — weight
    /// `[in, out]`, the transpose of `nn.Linear` — and fuses Q, K, V into
    /// one `c_attn` (`q | k | v` along the output axis, bias likewise).
    /// The checkpoint also serialises the causal-mask buffer `attn.bias`
    /// (older conversions add the scalar `attn.masked_bias`); neither is a
    /// parameter.
    fn source_layout(&self) -> SourceLayout {
        GPT2_SOURCE_LAYOUT
    }
}

/// `GPT2Attention`'s registered buffers. Neither is read as a weight:
/// `bias` is `tril(ones(n_positions, n_positions))`, reproduced exactly by
/// a causal full-span attention, and `masked_bias` is the fill value of a
/// masked score, which a softmax over the causal span never reads.
pub(crate) const GPT2_MASK_BUFFER_REASON: &str =
    "registered causal-mask buffer, not a parameter: the attention's declared causal span \
     reproduces it";

const GPT2_SOURCE_LAYOUT: SourceLayout = SourceLayout {
    input_major: &["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    fused: &[FusedProjection {
        module: "attn.c_attn",
        parts: &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    }],
    buffers: &[
        NonParameterBuffer {
            suffix: "attn.bias",
            kind: BufferKind::CausalMask,
            reason: GPT2_MASK_BUFFER_REASON,
        },
        NonParameterBuffer {
            suffix: "attn.masked_bias",
            kind: BufferKind::MaskFill,
            reason: GPT2_MASK_BUFFER_REASON,
        },
    ],
};

impl Norms for Gpt2Arch {
    fn norm_type(&self) -> NormType {
        NormType::LayerNorm
    }

    /// `nn.LayerNorm` with its default affine bias, at every site.
    fn norm_bias(&self) -> Option<bool> {
        Some(true)
    }
}

impl FeedForward for Gpt2Arch {
    /// The checkpoint's `activation_function` when it states one; a silent
    /// config runs `GPT2Config`'s default, `gelu_new`.
    fn activation_declaration(&self) -> ActivationDeclaration {
        match self.config().hidden_act.as_deref() {
            None => ActivationDeclaration::Nonlinearity(Activation::GeluTanh),
            declared => ActivationDeclaration::judge(declared),
        }
    }

    fn ffn_type(&self) -> FfnType {
        FfnType::Standard
    }

    /// `Conv1D` always carries a bias.
    fn ffn_bias(&self) -> Option<bool> {
        Some(true)
    }
}

impl Position for Gpt2Arch {}
impl Attention for Gpt2Arch {
    /// `Conv1D` always carries a bias: Q, K, V (fused in `c_attn`) and the
    /// output projection `c_proj`.
    fn attention_bias(&self) -> Option<bool> {
        Some(true)
    }
}
impl LatentAttention for Gpt2Arch {}
impl Embeddings for Gpt2Arch {}
impl ModelArchitecture for Gpt2Arch {}

/// DESCRIBE layer bands for this family at the depths they were set for.
/// Exact `model_type` only: a lookalike falls back to the proportional split.
pub(crate) const GPT2_LAYER_BANDS: &[LayerBandSplit] = &[
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 12,
        syntax_last: 4,
        knowledge_last: 9,
    },
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 24,
        syntax_last: 9,
        knowledge_last: 19,
    },
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 36,
        syntax_last: 14,
        knowledge_last: 28,
    },
    LayerBandSplit {
        model_type: "gpt2",
        num_layers: 48,
        syntax_last: 19,
        knowledge_last: 38,
    },
];
