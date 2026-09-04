//! Auto-detect model architecture from `config.json`.
//!
//! The module is organised so each concern lives in its own file:
//! - [`config_io`] reads `config.json` from disk and enforces presence of
//!   topology fields that have no defensible architecture-class default.
//! - [`parser`] turns a parsed JSON value into a [`ModelConfig`], honouring
//!   both multimodal nesting (`text_config`) and flat layouts.
//! - This file owns [`ModelError`] and the public entry points, including
//!   the family-routing dispatch that maps `model_type` → concrete
//!   [`ModelArchitecture`].

use std::path::Path;

use crate::architectures::bitnet::BitnetArch;
use crate::architectures::deepseek::DeepSeekArch;
use crate::architectures::deepseek_v4::DeepSeekV4Arch;
use crate::architectures::exaone4::Exaone4Arch;
use crate::architectures::gemma2::Gemma2Arch;
use crate::architectures::gemma3::Gemma3Arch;
use crate::architectures::gemma4::Gemma4Arch;
use crate::architectures::generic::GenericArch;
use crate::architectures::gpt2::Gpt2Arch;
use crate::architectures::gpt_oss::GptOssArch;
use crate::architectures::granite::GraniteArch;
use crate::architectures::kimi::KimiLinearArch;
use crate::architectures::kimi_k3::KimiK3Arch;
use crate::architectures::lfm2::Lfm2Arch;
use crate::architectures::llama::LlamaArch;
use crate::architectures::mamba2::{Mamba2Arch, MAMBA2_MODEL_TYPE};
use crate::architectures::mistral::MistralArch;
use crate::architectures::mixtral::MixtralArch;
use crate::architectures::moss_tts_realtime::{MossTtsRealtimeArch, MOSS_TTS_REALTIME_MODEL_TYPE};
use crate::architectures::muse_glimmer::MuseGlimmerArch;
use crate::architectures::olmo2::Olmo2Arch;
use crate::architectures::olmoe::OlmoeArch;
use crate::architectures::qwen::QwenArch;
use crate::architectures::starcoder2::StarCoder2Arch;
use crate::architectures::tinymodel::TinyModelArch;
use crate::config::ModelArchitecture;
use crate::validation::ConfigValidationError;

mod config_io;
mod parser;
pub mod registry;

use config_io::{
    config_path, read_config_json, require_config_fields, CONFIG_FILE_NAME,
    CONFIG_KEY_LANGUAGE_CONFIG, CONFIG_KEY_TEXT_CONFIG,
};
use parser::parse_model_config;

pub use registry::{
    find_architecture, ArchitectureEntry, AttentionKind, ModelTypeMatch, ARCHITECTURE_REGISTRY,
};

/// Error from model detection/config parsing.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("not a directory: {0}")]
    NotADirectory(std::path::PathBuf),
    #[error("no safetensors files in {0}")]
    NoSafetensors(std::path::PathBuf),
    #[error("config validation failed: {0:?}")]
    ConfigValidation(Vec<ConfigValidationError>),
    #[error(
        "{CONFIG_FILE_NAME} not found at {0:?} — \
         architecture cannot be inferred from safetensors alone; \
         copy {CONFIG_FILE_NAME} from the source model into this directory"
    )]
    ConfigMissing(std::path::PathBuf),
    #[error(
        "{CONFIG_FILE_NAME} at {path:?} is missing required field(s): {missing:?} \
         (checked under top level and `{CONFIG_KEY_TEXT_CONFIG}`)"
    )]
    ConfigFieldsMissing {
        path: std::path::PathBuf,
        missing: Vec<&'static str>,
    },
}

/// Read `config.json` from a model directory and return the architecture.
///
/// Errors with [`ModelError::ConfigMissing`] when the directory has no
/// `config.json`, and with [`ModelError::ConfigFieldsMissing`] when the
/// file exists but lacks topology fields without a defensible
/// architecture-class default. This prevents the silent fallback-to-defaults
/// path from inventing a wrong topology and then panicking deep inside the
/// extract pipeline (issue #22).
pub fn detect_architecture(model_dir: &Path) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let config_path = config_path(model_dir);
    let config_json = read_config_json(&config_path)?;
    require_config_fields(&config_json, &config_path)?;
    Ok(detect_from_json(&config_json))
}

/// Read `config.json` from a model directory, detect the architecture, and validate it.
pub fn detect_architecture_validated(
    model_dir: &Path,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_architecture(model_dir)?;
    validate_detected_architecture(arch)
}

/// Detect architecture from an already-parsed `config.json` value.
///
/// Infallible by design: callers building an in-memory config for tests
/// or programmatic loads (e.g. GGUF-derived configs) can keep terse setup
/// and rely on [`ModelArchitecture::validate`] downstream to catch any
/// missing required fields.
pub fn detect_from_json(config: &serde_json::Value) -> Box<dyn ModelArchitecture> {
    let model_config = parse_model_config(config);
    let model_type = model_config.model_type.as_str();

    match model_type {
        // Gemma family
        t if t.starts_with("gemma4") => Box::new(Gemma4Arch::from_config(model_config)),
        t if t.starts_with("gemma3") => Box::new(Gemma3Arch::from_config(model_config)),
        t if t.starts_with("gemma2") || t == "gemma" => {
            Box::new(Gemma2Arch::from_config(model_config))
        }
        // Llama family
        t if t.starts_with("llama") => Box::new(LlamaArch::from_config(model_config)),
        // Mistral (dense)
        "mistral" => Box::new(MistralArch::from_config(model_config)),
        // Mamba2 — pure SSM, no attention anywhere. Exact match: `mamba`
        // (v1) is a different operator and stays generic until judged.
        t if t == MAMBA2_MODEL_TYPE => Box::new(Mamba2Arch::from_config(model_config)),
        // Muse-Glimmer target: config-driven defaults plus the judged
        // gate/QK-norm semantics. The assistant is deliberately excluded
        // (weighted QK norms, no gate, unjudged) and stays generic.
        "muse_glimmer" | "muse_glimmer_text" => {
            Box::new(MuseGlimmerArch::from_config(model_config))
        }
        // Mixtral (MoE) — block_sparse_moe pattern
        "mixtral" => Box::new(MixtralArch::from_config(model_config)),
        // GPT-2 (non-gated FFN, LayerNorm, learned positional embeddings)
        "gpt2" => Box::new(Gpt2Arch::from_config(model_config)),
        // GPT-OSS (MoE, MXFP4 packed experts)
        "gpt_oss" => Box::new(GptOssArch::from_config(model_config)),
        // MOSS-TTS-Realtime — a stock Qwen3 backbone nested under
        // `language_config`, whose output is a hidden state for a
        // side-loaded audio depth transformer, never text. The nested
        // object is a complete Qwen3 config carrying its own
        // `model_type: "qwen3"`, so it is parsed directly and rebranded;
        // the flat fallback keeps in-memory test configs terse.
        t if t == MOSS_TTS_REALTIME_MODEL_TYPE => {
            let nested = config.get(CONFIG_KEY_LANGUAGE_CONFIG).unwrap_or(config);
            let mut nested_config = parse_model_config(nested);
            nested_config.model_type = MOSS_TTS_REALTIME_MODEL_TYPE.to_string();
            Box::new(MossTtsRealtimeArch::from_config(nested_config))
        }
        // Qwen family (dense and MoE share same keys)
        t if t.starts_with("qwen") => Box::new(QwenArch::from_config(model_config)),
        // OLMoE — Qwen3-MoE tensor layout, but sizes experts from
        // `intermediate_size` (no `moe_intermediate_size` field) and does not
        // renormalize top-k router probabilities.
        "olmoe" => Box::new(OlmoeArch::from_config(model_config)),
        // OLMo-2 / OLMo-3 — a POST-NORM stack (each sublayer's output is
        // normalised before its residual add) with whole-projection QK
        // norm and a 1e-5 class default for `rms_norm_eps`. Matched
        // before the bare `olmo` prefix would be, and deliberately NOT
        // aliased onto Llama: all three facts are silent when inherited.
        "olmo2" | "olmo3" => Box::new(Olmo2Arch::from_config(model_config)),
        // EXAONE-4 — the same post-norm stack as OLMo-2, with PER-HEAD
        // QK norm rather than whole-projection. Its own entry because
        // that difference is an operator, not a label.
        t if t.starts_with("exaone4") => Box::new(Exaone4Arch::from_config(model_config)),
        // LFM2 — the two-norm pre-only stack under its own spelling.
        // Its conv mixer is deliberately NOT declared; see `Lfm2Arch`.
        t if t.starts_with("lfm2") => Box::new(Lfm2Arch::from_config(model_config)),
        // DeepSeek-V4 (MoE + MLA + MXFP4 + HCA attention; new tensor naming)
        "deepseek_v4" => Box::new(DeepSeekV4Arch::from_config(model_config)),
        // DeepSeek V2/V3 family (MoE + MLA, model.* prefixed keys)
        t if t.starts_with("deepseek") => Box::new(DeepSeekArch::from_config(model_config)),
        // Kimi Linear (hybrid KDA/MLA + sigmoid-routed MoE, bias-corrected
        // selection, shared expert) — `block_sparse_moe.*` keys distinct
        // from both the DeepSeek lineage and Mixtral's routing/shared-
        // expert semantics, though it reuses Mixtral's `w1/w2/w3` spelling.
        // Kimi K3 — the container identity. IDENTIFIED, not executable:
        // `KimiK3Arch` carries only architecture facts the public config
        // establishes and declares no K3-specific execution semantics.
        // Recognised explicitly, like BitNet, so a K3 config cannot
        // collapse into the generic fallback — and NOT routed to
        // `KimiLinearArch`, which would assert K3 executes as its
        // ancestor.
        "kimi_k3" => Box::new(KimiK3Arch::from_config(model_config)),
        "kimi_linear" => Box::new(KimiLinearArch::from_config(model_config)),
        // StarCoder 2
        "starcoder2" => Box::new(StarCoder2Arch::from_config(model_config)),
        // Granite family (dense and MoE share same base keys)
        t if t.starts_with("granite") => Box::new(GraniteArch::from_config(model_config)),
        // TinyModel — research-scale decoder used for LARQL compile/walk work
        "tinymodel" => Box::new(TinyModelArch::from_config(model_config)),
        // BitNet b1.58 (HF "bitnet", GGUF "bitnet-b1.58"). Recognised
        // explicitly so a BitNet config can't silently collapse to the
        // generic fallback; native-ternary inference is served by the
        // larql-inference ternary path, not this trait. See BitnetArch docs.
        t if t.starts_with("bitnet") => Box::new(BitnetArch::from_config(model_config)),
        // Unknown — generic fallback
        _ => Box::new(GenericArch::from_config(model_config)),
    }
}

/// Detect architecture from an already-parsed `config.json` value and validate it.
pub fn detect_from_json_validated(
    config: &serde_json::Value,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let arch = detect_from_json(config);
    validate_detected_architecture(arch)
}

pub(crate) fn validate_detected_architecture(
    arch: Box<dyn ModelArchitecture>,
) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    match arch.validate() {
        Ok(()) => Ok(arch),
        Err(errors) => Err(ModelError::ConfigValidation(errors)),
    }
}

#[cfg(test)]
mod tests;
