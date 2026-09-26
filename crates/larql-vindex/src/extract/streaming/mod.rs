//! Streaming vindex extraction — build from safetensors without loading the full model.
//!
//! Instead of loading all weights into ModelWeights (which requires the entire model
//! in RAM), this module mmaps safetensors files and processes one layer at a time.
//! Peak memory = 1 layer's tensors + embeddings, not the full model.
//!
//! For a 120B MoE model: ~120 GB as ModelWeights vs ~2 GB streaming.
//!
//! Structure (round-5 phase 2, 2026-05-09):
//! - `mod.rs`     — `build_vindex_streaming` entry + orchestrator
//! - `context.rs` — `StreamingContext` struct + `new` (mmap + tensor
//!                  index + checkpoint load) + `finalize` (checksums +
//!                  drop checkpoint)
//! - `stages.rs`  — `impl StreamingContext` for each stage:
//!                  `write_gate_vectors`, `write_router_weights`,
//!                  `write_embeddings`, `write_down_meta`,
//!                  `write_tokenizer`, `write_index_json`,
//!                  `maybe_write_model_weights`
//! - `tensor_io.rs` — safetensors-mmap helpers (`MmapShard`, `GateSink`,
//!                    `get_tensor_f32`, `normalize_key`)

mod context;
mod stages;
mod tensor_io;

use std::path::Path;

use crate::config::dtype::StorageDtype;
use crate::config::types::QuantFormat;
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;

use self::context::StreamingContext;

/// Build a vindex by streaming from safetensors files (no full model load).
///
/// Peak memory: embeddings + 1 layer of gate/down weights at a time.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex_streaming(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    // Per-expert summary tier (`--summary-features-per-expert`): `> 0` caps
    // each expert's gate/down features to a top-K so many-experts MoE doesn't
    // explode; `0` = full per-expert features.
    summary_features_per_expert: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    quant: QuantFormat,
    weight_opts: crate::format::weights::WriteWeightsOptions,
    q4k_opts: crate::format::weights::KquantWriteOptions,
    // Skip writing `gate_vectors.bin` entirely. Only valid when
    // `quant == Q4K` — the loader synthesizes gate from Q4K at load
    // time. Refused otherwise because without a Q4K interleaved file
    // the gate would be unrecoverable.
    drop_gate_vectors: bool,
    // Expert-bank route (`extract::orchestrate` decides; `Legacy` = today's
    // inline behaviour, no container). `Native` REQUIRES `expert_banks_out`.
    expert_banks: crate::extract::target::ExtractionRequest,
    expert_banks_out: Option<&Path>,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    if drop_gate_vectors && quant != QuantFormat::Q4K {
        return Err(VindexError::Parse(
            "--drop-gate-vectors requires --quant q4k (the loader rebuilds gate from Q4K)".into(),
        ));
    }

    // Detect architecture. For safetensors input this reads `config.json`
    // from `model_dir`; for GGUF input the architecture lives in the
    // file metadata, so we open the entry GGUF, build a config.json
    // equivalent, and detect from that.
    let arch = if let Some(gguf_path) = context::detect_gguf_entry(model_dir)? {
        let gguf = larql_models::loading::gguf::GgufFile::open(&gguf_path)
            .map_err(|e| VindexError::Parse(format!("open GGUF for arch detection: {e}")))?;
        let cfg_json = gguf.to_config_json();
        larql_models::detect_from_json_validated(&cfg_json)
            .map_err(|e| VindexError::Parse(e.to_string()))?
    } else {
        larql_models::detect_architecture_validated(model_dir)
            .map_err(|e| VindexError::Parse(e.to_string()))?
    };
    // Reject unsupported attention layouts (e.g. MLA on standard Q/K/V/O
    // manifests) before any output directory or checkpoint is created.
    crate::format::weights::ensure_extract_level_supported(&*arch, extract_level)?;

    std::fs::create_dir_all(output_dir)?;

    let mut ctx = StreamingContext::new(
        arch,
        model_dir,
        tokenizer,
        model_name,
        output_dir,
        down_top_k,
        summary_features_per_expert,
        extract_level,
        dtype,
        quant,
        weight_opts,
        q4k_opts,
        drop_gate_vectors,
        callbacks,
    )?;

    // Runs first: a checkpoint carrying tensors nothing can address should
    // fail in seconds, not after a multi-minute extraction that quietly
    // produced a hollow vindex.
    ctx.audit_tensor_coverage()?;

    // Expert banks next: the orchestrator either declines (LegacyInline —
    // the k-quant writer will emit them later as always) or streams the
    // native container now, so an experts-only run finishes here-ish
    // rather than after the spine stages.
    if expert_banks != crate::extract::target::ExtractionRequest::Legacy {
        let Some(dest) = expert_banks_out else {
            return Err(VindexError::Parse(
                "--expert-banks native/auto requires --expert-banks-out DIR".into(),
            ));
        };
        let outcome = ctx.extract_expert_banks(expert_banks, dest)?;
        ctx.callbacks.on_stage(&format!(
            "legacy LYRW expert banks: {outcome:?} -> {}",
            dest.display()
        ));
    }
    ctx.write_gate_vectors()?;
    ctx.write_router_weights()?;
    ctx.write_embeddings()?;
    ctx.write_down_meta()?;
    ctx.write_tokenizer()?;
    ctx.write_index_json()?;
    ctx.maybe_write_model_weights()?;
    ctx.finalize()?;

    Ok(())
}
