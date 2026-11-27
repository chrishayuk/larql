//! The static architecture table — the one source of truth for every
//! `model_type` `detect_from_json` recognises: how it is matched, how its
//! architecture is built, and what it supports.

use crate::architectures::bitnet::BitnetArch;
use crate::architectures::deepseek::DeepSeekArch;
use crate::architectures::deepseek_v4::DeepSeekV4Arch;
use crate::architectures::exaone4::Exaone4Arch;
use crate::architectures::gemma2::{Gemma2Arch, GEMMA2_LAYER_BANDS};
use crate::architectures::gemma3::{Gemma3Arch, GEMMA3_LAYER_BANDS};
use crate::architectures::gemma4::{Gemma4Arch, GEMMA4_LAYER_BANDS};
use crate::architectures::gemma_gguf::{gemma4_gguf_config, GEMMA_GGUF_KEY_REPLACEMENTS};
use crate::architectures::glm5::GlmMoeDsaArch;
use crate::architectures::glm5_next::Glm5NextArch;
use crate::architectures::gpt2::{Gpt2Arch, GPT2_LAYER_BANDS};
use crate::architectures::gpt_neo::GptNeoArch;
use crate::architectures::gpt_oss::GptOssArch;
use crate::architectures::granite::GraniteArch;
use crate::architectures::kimi::KimiLinearArch;
use crate::architectures::kimi_k3::KimiK3Arch;
use crate::architectures::lfm2::Lfm2Arch;
use crate::architectures::llama::{LlamaArch, LLAMA_LAYER_BANDS};
use crate::architectures::mamba2::Mamba2Arch;
use crate::architectures::mistral::{MistralArch, MISTRAL_LAYER_BANDS};
use crate::architectures::mixtral::{MixtralArch, MIXTRAL_LAYER_BANDS};
use crate::architectures::muse_glimmer::MuseGlimmerArch;
use crate::architectures::olmo2::Olmo2Arch;
use crate::architectures::olmoe::OlmoeArch;
use crate::architectures::qwen::{QwenArch, QWEN_LAYER_BANDS};
use crate::architectures::starcoder2::StarCoder2Arch;
use crate::architectures::tinymodel::TinyModelArch;
use crate::defaults::{ROPE_BASE_DEFAULT, ROPE_BASE_GEMMA};

use super::attention::AttentionKind;
use super::chat::ChatFormat;
use super::construct;
use super::defaults::{ConfigDefaults, IntermediateSize};
use super::entry::{
    ArchitectureEntry, ComponentRole, MLA_QUANT_FORMATS, SSM_QUANT_FORMATS, STANDARD_QUANT_FORMATS,
};
use super::gguf::GgufTranslation;
use super::pattern::ModelTypeMatch;

/// Every architecture `detect_from_json` recognises, in first-match-wins
/// order: where patterns overlap, the more specific row comes first. A
/// `model_type` matching none of these falls back to
/// [`crate::architectures::generic::GenericArch`].
/// The Llama family's registry label — also the prefix it matches on, and
/// the family a checkpoint's `is_llama_config` flag is a claim about.
pub const LLAMA_FAMILY: &str = "llama";

/// Gemma family head width when undeclared (every Gemma config class).
const GEMMA_HEAD_DIM: usize = 256;

/// `transformers` `GemmaConfig` (Gemma 1) class defaults.
const GEMMA1_CONFIG_DEFAULTS: ConfigDefaults = ConfigDefaults {
    rope_theta: ROPE_BASE_DEFAULT,
    head_dim: Some(GEMMA_HEAD_DIM),
    num_attention_heads: Some(16),
    num_key_value_heads: Some(16),
    intermediate_size: IntermediateSize::Required,
    layer_count_key: None,
    head_count_key: None,
    tokenizer_adds_bos: None,
};

/// `transformers` `Gemma2Config` class defaults.
const GEMMA2_CONFIG_DEFAULTS: ConfigDefaults = ConfigDefaults {
    rope_theta: ROPE_BASE_DEFAULT,
    head_dim: Some(GEMMA_HEAD_DIM),
    num_attention_heads: Some(8),
    num_key_value_heads: Some(4),
    intermediate_size: IntermediateSize::Required,
    layer_count_key: None,
    head_count_key: None,
    tokenizer_adds_bos: None,
};

/// `transformers` `Gemma3TextConfig` class defaults; Gemma 4 inherits them.
const GEMMA3_CONFIG_DEFAULTS: ConfigDefaults = ConfigDefaults {
    rope_theta: ROPE_BASE_GEMMA,
    ..GEMMA2_CONFIG_DEFAULTS
};

/// GPT-2 ships no `n_inner`; `transformers` derives the FFN as `4 * n_embd`.
const GPT2_FFN_HIDDEN_MULTIPLE: usize = 4;

pub static ARCHITECTURE_REGISTRY: &[ArchitectureEntry] = &[
    ArchitectureEntry {
        model_type: "gemma4",
        patterns: &[ModelTypeMatch::Prefix("gemma4")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Gemma4Arch::from_config(c)),
        config_defaults: GEMMA3_CONFIG_DEFAULTS,
        gguf: GgufTranslation {
            key_replacements: GEMMA_GGUF_KEY_REPLACEMENTS,
            config: Some(gemma4_gguf_config),
            ..GgufTranslation::NONE
        },
        layer_bands: GEMMA4_LAYER_BANDS,
        chat_format: Some(ChatFormat::Gemma4Turns),
    },
    ArchitectureEntry {
        model_type: "gemma3",
        patterns: &[ModelTypeMatch::Prefix("gemma3")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Gemma3Arch::from_config(c)),
        config_defaults: GEMMA3_CONFIG_DEFAULTS,
        gguf: GgufTranslation {
            key_replacements: GEMMA_GGUF_KEY_REPLACEMENTS,
            ..GgufTranslation::NONE
        },
        layer_bands: GEMMA3_LAYER_BANDS,
        chat_format: Some(ChatFormat::GemmaTurns),
    },
    // Gemma 1 is served by the Gemma 2 architecture but keeps the llama
    // two-norm layout in GGUF, so it is its own row: same constructor, none
    // of the Gemma 2+ GGUF key rewrites.
    ArchitectureEntry {
        model_type: "gemma",
        patterns: &[ModelTypeMatch::Exact("gemma")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Gemma2Arch::from_config(c)),
        config_defaults: GEMMA1_CONFIG_DEFAULTS,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: Some(ChatFormat::GemmaTurns),
    },
    ArchitectureEntry {
        model_type: "gemma2",
        patterns: &[ModelTypeMatch::Prefix("gemma2")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Gemma2Arch::from_config(c)),
        config_defaults: GEMMA2_CONFIG_DEFAULTS,
        gguf: GgufTranslation {
            key_replacements: GEMMA_GGUF_KEY_REPLACEMENTS,
            ..GgufTranslation::NONE
        },
        layer_bands: GEMMA2_LAYER_BANDS,
        chat_format: Some(ChatFormat::GemmaTurns),
    },
    ArchitectureEntry {
        model_type: LLAMA_FAMILY,
        patterns: &[ModelTypeMatch::Prefix(LLAMA_FAMILY)],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(LlamaArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: LLAMA_LAYER_BANDS,
        chat_format: Some(ChatFormat::Llama3Headers),
    },
    // OLMo-2 and OLMo-3: one decoder shape, two labels. Exact rather
    // than prefixed, because `olmo` (v1) and `olmoe` are different
    // architectures and a prefix would swallow both.
    ArchitectureEntry {
        model_type: "olmo2",
        patterns: &[
            ModelTypeMatch::Exact("olmo2"),
            ModelTypeMatch::Exact("olmo3"),
        ],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Olmo2Arch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    // EXAONE-4: prefixed, so the nested `exaone4_5_text` spelling
    // resolves too. `exaone` (v3) is a different architecture and stays
    // on the generic path until its semantics are judged.
    // LFM2 — a hybrid whose every other layer is a short causal
    // convolution rather than attention. Prefixed so `lfm2_moe` resolves
    // too. Recognised so the identity stops resolving to `GenericArch`,
    // which was serving Llama-shaped defaults to a stack that is not one.
    ArchitectureEntry {
        model_type: "lfm2",
        patterns: &[ModelTypeMatch::Prefix("lfm2")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Lfm2Arch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "exaone4",
        patterns: &[ModelTypeMatch::Prefix("exaone4")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Exaone4Arch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "mistral",
        patterns: &[ModelTypeMatch::Exact("mistral")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(MistralArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: MISTRAL_LAYER_BANDS,
        chat_format: Some(ChatFormat::MistralInst),
    },
    // Pure SSM — exact on purpose: `mamba` (v1) is a different operator
    // and stays on the generic path until its semantics are judged. No
    // extractor path has ever quantised an SSM estate, so no quant
    // format is claimed.
    ArchitectureEntry {
        model_type: "mamba2",
        patterns: &[ModelTypeMatch::Exact("mamba2")],
        attention_kind: AttentionKind::Recurrent,
        quant_formats: SSM_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Mamba2Arch::from_config(c)),
        config_defaults: ConfigDefaults {
            intermediate_size: IntermediateSize::NotApplicable,
            ..ConfigDefaults::STANDARD
        },
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    // The assistant model_type is deliberately absent: it stays on the
    // generic path until its semantics are judged.
    ArchitectureEntry {
        model_type: "muse_glimmer",
        patterns: &[
            ModelTypeMatch::Exact("muse_glimmer"),
            ModelTypeMatch::Exact("muse_glimmer_text"),
        ],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(MuseGlimmerArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "mixtral",
        patterns: &[ModelTypeMatch::Exact("mixtral")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(MixtralArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: MIXTRAL_LAYER_BANDS,
        chat_format: Some(ChatFormat::MistralInst),
    },
    ArchitectureEntry {
        model_type: "gpt2",
        patterns: &[ModelTypeMatch::Exact("gpt2")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Gpt2Arch::from_config(c)),
        config_defaults: ConfigDefaults {
            intermediate_size: IntermediateSize::HiddenMultiple(GPT2_FFN_HIDDEN_MULTIPLE),
            // `GPT2Tokenizer`: `add_bos_token` defaults to `False`.
            tokenizer_adds_bos: Some(false),
            ..ConfigDefaults::STANDARD
        },
        gguf: GgufTranslation::NONE,
        layer_bands: GPT2_LAYER_BANDS,
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "gpt_neo",
        patterns: &[ModelTypeMatch::Exact("gpt_neo")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(GptNeoArch::from_config(c)),
        // `GPTNeoConfig`: `intermediate_size: None` means `4 * hidden_size`
        // (`GPTNeoMLP` is built at that width), and the topology is spelled
        // `num_layers` / `num_heads`.
        config_defaults: ConfigDefaults {
            intermediate_size: IntermediateSize::HiddenMultiple(GPT2_FFN_HIDDEN_MULTIPLE),
            layer_count_key: Some("num_layers"),
            head_count_key: Some("num_heads"),
            // GPT-Neo ships GPT-2's tokenizer class.
            tokenizer_adds_bos: Some(false),
            ..ConfigDefaults::STANDARD
        },
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "gpt_oss",
        patterns: &[ModelTypeMatch::Exact("gpt_oss")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(GptOssArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    // MOSS-TTS-Realtime — Qwen3 backbone nested under `language_config`;
    // audio depth-transformer weights side-load via `larql_models::speech`.
    // Listed before the `qwen` prefix entry, which would otherwise claim it.
    ArchitectureEntry {
        model_type: "moss_tts_realtime",
        patterns: &[ModelTypeMatch::Exact("moss_tts_realtime")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: construct::moss_tts_realtime,
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "qwen",
        patterns: &[ModelTypeMatch::Prefix("qwen")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(QwenArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation {
            aliases: &[("qwen", "qwen2")],
            ..GgufTranslation::NONE
        },
        layer_bands: QWEN_LAYER_BANDS,
        chat_format: Some(ChatFormat::ChatMl),
    },
    ArchitectureEntry {
        model_type: "olmoe",
        patterns: &[ModelTypeMatch::Exact("olmoe")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(OlmoeArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    // `deepseek_v4` (exact) is listed before `deepseek` (prefix) —
    // order matters, `deepseek_v4` also matches the `deepseek` prefix.
    ArchitectureEntry {
        model_type: "deepseek_v4",
        patterns: &[ModelTypeMatch::Exact("deepseek_v4")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(DeepSeekV4Arch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation {
            aliases: &[("deepseekv4", "deepseek_v4")],
            ..GgufTranslation::NONE
        },
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "deepseek",
        patterns: &[ModelTypeMatch::Prefix("deepseek")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(DeepSeekArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation {
            aliases: &[("deepseek", "deepseek_v2"), ("deepseek2", "deepseek_v2")],
            ..GgufTranslation::NONE
        },
        layer_bands: &[],
        chat_format: None,
    },
    // Hybrid KDA/MLA attention — no dedicated `AttentionKind` variant
    // exists yet for the recurrence, so this reports the MORE restrictive
    // of the two (`Mla`, `MLA_QUANT_FORMATS`): it genuinely has MLA layers
    // and the Q4K writer hard-rejects MLA regardless. A `Hybrid`/`Kda`
    // kind is future work, not a claim this entry makes today.
    // Kimi K3. Its container declares `kimi_k3` and its text component
    // declares `kimi_linear`; both are true, and the identity gate needs
    // the relationship DECLARED rather than inferred from both sides
    // resolving. Listed before `kimi_linear` so first-match-wins is
    // unambiguous, though the patterns are disjoint.
    //
    // `Mla` and `MLA_QUANT_FORMATS` for the same reason the ancestor
    // carries them: K3 genuinely has MLA layers and this is the MORE
    // restrictive of the two kinds. A `Hybrid`/`Kda` kind is future work
    // here exactly as it is there, and no claim to the contrary is made.
    // GLM-5.3-Flash. Its container declares `glm5_next` and its text
    // component declares `glm5_next_text` — the same two-level identity
    // Kimi K3 has, so it is declared the same way rather than left for
    // the identity gate to infer from both sides resolving.
    //
    // `Mla` and `MLA_QUANT_FORMATS` for the reason the Kimi entries give:
    // the stack genuinely has 11 MLA layers, this is the MORE restrictive
    // of its two attention kinds, and no `Kda`/`Hybrid` kind exists yet.
    // The 34 KDA layers and the DSA indexer are NOT claimed here.
    ArchitectureEntry {
        model_type: "glm5_next",
        patterns: &[ModelTypeMatch::Exact("glm5_next")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
        // Lineage, not substitutability. See `ArchitectureEntry::components`.
        components: &[(ComponentRole::Text, "glm5_next_text")],
        construct: |c, _| Box::new(Glm5NextArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "glm5_next_text",
        patterns: &[ModelTypeMatch::Exact("glm5_next_text")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(Glm5NextArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "kimi_k3",
        patterns: &[ModelTypeMatch::Exact("kimi_k3")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
        // Lineage, not substitutability. See `ArchitectureEntry::components`.
        components: &[(ComponentRole::Text, "kimi_linear")],
        construct: |c, _| Box::new(KimiK3Arch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "kimi_linear",
        patterns: &[ModelTypeMatch::Exact("kimi_linear")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(KimiLinearArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    // GLM-5.2 — MoE + MLA, same tensor naming as `deepseek` (V3), plus a
    // DSA sparse-attention indexer represented in config only.
    ArchitectureEntry {
        model_type: "glm_moe_dsa",
        patterns: &[ModelTypeMatch::Exact("glm_moe_dsa")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(GlmMoeDsaArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "starcoder2",
        patterns: &[ModelTypeMatch::Exact("starcoder2")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(StarCoder2Arch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "granite",
        patterns: &[ModelTypeMatch::Prefix("granite")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(GraniteArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "tinymodel",
        patterns: &[ModelTypeMatch::Exact("tinymodel")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(TinyModelArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
    ArchitectureEntry {
        model_type: "bitnet",
        patterns: &[ModelTypeMatch::Prefix("bitnet")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
        components: &[],
        construct: |c, _| Box::new(BitnetArch::from_config(c)),
        config_defaults: ConfigDefaults::STANDARD,
        gguf: GgufTranslation::NONE,
        layer_bands: &[],
        chat_format: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_non_empty_and_every_entry_declares_at_least_one_pattern() {
        assert!(!ARCHITECTURE_REGISTRY.is_empty());
        for entry in ARCHITECTURE_REGISTRY {
            assert!(
                !entry.patterns.is_empty(),
                "{} declares no patterns",
                entry.model_type
            );
        }
    }

    #[test]
    fn model_type_labels_are_unique() {
        let mut labels: Vec<&str> = ARCHITECTURE_REGISTRY.iter().map(|e| e.model_type).collect();
        let before = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(
            labels.len(),
            before,
            "duplicate model_type label in the registry"
        );
    }
}
