//! The static architecture table — one source of truth for every
//! `model_type` `detect_from_json` recognises, alongside what each one
//! supports.

use super::attention::AttentionKind;
use super::entry::{
    ArchitectureEntry, MLA_QUANT_FORMATS, SSM_QUANT_FORMATS, STANDARD_QUANT_FORMATS,
};
use super::pattern::ModelTypeMatch;

/// Every architecture `detect_from_json` recognises, in the same
/// first-match-wins order as its `match` arms. A `model_type` matching
/// none of these falls back to [`crate::architectures::generic::GenericArch`].
/// The Llama family's registry label — also the prefix it matches on, and
/// the family a checkpoint's `is_llama_config` flag is a claim about.
pub const LLAMA_FAMILY: &str = "llama";

pub static ARCHITECTURE_REGISTRY: &[ArchitectureEntry] = &[
    ArchitectureEntry {
        model_type: "gemma4",
        patterns: &[ModelTypeMatch::Prefix("gemma4")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "gemma3",
        patterns: &[ModelTypeMatch::Prefix("gemma3")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "gemma2",
        patterns: &[
            ModelTypeMatch::Prefix("gemma2"),
            ModelTypeMatch::Exact("gemma"),
        ],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: LLAMA_FAMILY,
        patterns: &[ModelTypeMatch::Prefix(LLAMA_FAMILY)],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
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
    },
    ArchitectureEntry {
        model_type: "exaone4",
        patterns: &[ModelTypeMatch::Prefix("exaone4")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "mistral",
        patterns: &[ModelTypeMatch::Exact("mistral")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
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
    },
    ArchitectureEntry {
        model_type: "mixtral",
        patterns: &[ModelTypeMatch::Exact("mixtral")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "gpt2",
        patterns: &[ModelTypeMatch::Exact("gpt2")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "gpt_oss",
        patterns: &[ModelTypeMatch::Exact("gpt_oss")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    // MOSS-TTS-Realtime — Qwen3 backbone nested under `language_config`;
    // audio depth-transformer weights side-load via `larql_models::speech`.
    // Listed before the `qwen` prefix entry to mirror the match-arm order.
    ArchitectureEntry {
        model_type: "moss_tts_realtime",
        patterns: &[ModelTypeMatch::Exact("moss_tts_realtime")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "qwen",
        patterns: &[ModelTypeMatch::Prefix("qwen")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "olmoe",
        patterns: &[ModelTypeMatch::Exact("olmoe")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    // `deepseek_v4` (exact) is listed before `deepseek` (prefix) —
    // order matters, `deepseek_v4` also matches the `deepseek` prefix.
    ArchitectureEntry {
        model_type: "deepseek_v4",
        patterns: &[ModelTypeMatch::Exact("deepseek_v4")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "deepseek",
        patterns: &[ModelTypeMatch::Prefix("deepseek")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
    },
    // Hybrid KDA/MLA attention — no dedicated `AttentionKind` variant
    // exists yet for the recurrence, so this reports the MORE restrictive
    // of the two (`Mla`, `MLA_QUANT_FORMATS`): it genuinely has MLA layers
    // and the Q4K writer hard-rejects MLA regardless. A `Hybrid`/`Kda`
    // kind is future work, not a claim this entry makes today.
    ArchitectureEntry {
        model_type: "kimi_linear",
        patterns: &[ModelTypeMatch::Exact("kimi_linear")],
        attention_kind: AttentionKind::Mla,
        quant_formats: MLA_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "starcoder2",
        patterns: &[ModelTypeMatch::Exact("starcoder2")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "granite",
        patterns: &[ModelTypeMatch::Prefix("granite")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "tinymodel",
        patterns: &[ModelTypeMatch::Exact("tinymodel")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
    },
    ArchitectureEntry {
        model_type: "bitnet",
        patterns: &[ModelTypeMatch::Prefix("bitnet")],
        attention_kind: AttentionKind::Standard,
        quant_formats: STANDARD_QUANT_FORMATS,
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
