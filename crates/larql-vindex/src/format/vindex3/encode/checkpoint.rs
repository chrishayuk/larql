//! One-checkpoint encode: the pipeline both extraction surfaces share.
//!
//! Migration rung M2: `larql extract --generation v3` and LQL
//! `EXTRACT ... FORMAT VINDEX3` must produce a container the same way
//! `larql vindex3 encode` does — so the whole pipeline lives here, once:
//!
//! ```text
//! checkpoint dir (config.json + *.safetensors)
//!   → build_inventory          interpretation, once
//!   → plan_system              admissibility, with the blocking
//!                              findings RENDERED into the refusal
//!                              (encode_system's own gate discards them)
//!   → encode_system            segments + system_graph + index
//!   → capability snapshot      tokenizer.json + the HF metadata set
//! ```
//!
//! The capability snapshot exists because the encoder proper writes only
//! what execution needs — segments, graph, index — while binding with
//! full capability (INFER, chat templates) needs the checkpoint's
//! tokenizer artifacts placed beside them. The V2 extractor has always
//! done this; a V3 container produced by an extraction surface must not
//! silently degrade to token-id-only capability.

use std::path::Path;

use larql_models::inventory::build_inventory;

use super::{encode_system, EncodeOutcome};
use crate::error::VindexError;
use crate::format::filenames::TOKENIZER_JSON;
use crate::format::vindex3::plan::plan_system;

/// Checkpoint files that carry a container's *capabilities* rather than
/// its executable bytes: the tokenizer itself plus the HF metadata set
/// the V2 extractor snapshots (`snapshot_hf_metadata`). Copied verbatim
/// when present; absence of any — including the tokenizer — is not an
/// error, it just narrows what the bound container can do.
pub const CHECKPOINT_CAPABILITY_FILES: [&str; 5] = [
    TOKENIZER_JSON,
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
];

/// What one checkpoint encode produced.
#[derive(Debug)]
pub struct CheckpointEncode {
    pub outcome: EncodeOutcome,
    /// The artifact name recorded in the container (the checkpoint
    /// directory's stem — the same rule `larql vindex3 encode` uses).
    pub artifact: String,
    /// Capability files copied in from the checkpoint, in
    /// [`CHECKPOINT_CAPABILITY_FILES`] order.
    pub capabilities: Vec<String>,
}

/// Copy the checkpoint's capability files into an encoded container.
///
/// Present files copy verbatim (a failed copy of a present file is an
/// error — never a silent degrade); absent files are skipped. Returns
/// the names that were copied.
pub fn snapshot_checkpoint_capabilities(
    checkpoint: &Path,
    container: &Path,
) -> Result<Vec<String>, VindexError> {
    let mut copied = Vec::new();
    for name in CHECKPOINT_CAPABILITY_FILES {
        let src = checkpoint.join(name);
        if !src.exists() {
            continue;
        }
        std::fs::copy(&src, container.join(name)).map_err(VindexError::Io)?;
        copied.push(name.to_string());
    }
    Ok(copied)
}

/// Encode one HF checkpoint directory into a VINDEX3 container, with
/// the capability snapshot.
///
/// Refusals name their cause: a directory `build_inventory` cannot read
/// (no `config.json`) refuses as such, and an inadmissible plan refuses
/// with every blocking finding itemised — the caller never has to
/// re-run `larql vindex3 plan` to learn why.
pub fn encode_checkpoint(checkpoint: &Path, out: &Path) -> Result<CheckpointEncode, VindexError> {
    let inventory = build_inventory(checkpoint).map_err(|e| {
        VindexError::Parse(format!(
            "the VINDEX3 encoder consumes HF checkpoint artifacts (config.json + \
             safetensors); {} is not one: {e}",
            checkpoint.display()
        ))
    })?;
    let artifact = checkpoint
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact".to_string());
    let named = [(artifact.clone(), inventory)];

    let plan = plan_system(&named);
    if !plan.admissible {
        let blocking: Vec<String> = plan
            .artifacts
            .iter()
            .flat_map(|a| &a.findings)
            .filter(|f| f.blocks())
            .map(|f| format!("{}: {}", f.subject, f.detail))
            .collect();
        return Err(VindexError::Parse(format!(
            "refusing to encode an inadmissible plan ({} blocking finding(s)):\n  {}",
            blocking.len(),
            blocking.join("\n  ")
        )));
    }

    let outcome = encode_system(&named, out)?;
    // Closure at encode (schema 6, drill F4): encoding is the proof
    // boundary, so an encoder may not leave behind a container whose
    // operands do not close. The written container is read back through
    // the same closure the runtime uses — one authority, not a
    // re-implementation — and a defect removes the output and refuses
    // with every defect itemised. Before this gate, operand closure ran
    // only at read time, so an architecture that slipped the census was
    // written to disk before its operands were ever classified.
    enforce_closure_at_encode(out)?;
    let capabilities = snapshot_checkpoint_capabilities(checkpoint, out)?;
    Ok(CheckpointEncode {
        outcome,
        artifact,
        capabilities,
    })
}

/// Run operand closure over every component of the just-written
/// container; on any defect, remove the container and refuse.
fn enforce_closure_at_encode(out: &Path) -> Result<(), VindexError> {
    let inspection = super::super::inspect::inspect_container(out, false)?;
    let component_ids: Vec<String> = inspection
        .graph
        .components
        .iter()
        .filter(|c| c.execution.is_some())
        .map(|c| c.id.clone())
        .collect();
    for id in component_ids {
        let outcome = super::super::opplan::build::plan_component_ops(&inspection, out, &id)?;
        if !outcome.defects.is_empty() {
            let defects: Vec<String> = outcome.defects.iter().map(|d| format!("{d:?}")).collect();
            // The output is this function's own product; a container
            // closure refuses must not survive to be read later.
            std::fs::remove_dir_all(out).map_err(VindexError::Io)?;
            return Err(VindexError::Parse(format!(
                "refusing to keep an encode whose operands do not close \
                 (component `{id}`, {} defect(s)):\n  {}",
                defects.len(),
                defects.join("\n  ")
            )));
        }
    }
    Ok(())
}
