//! Which source files a legacy `COMPILE INTO VINDEX` must carry, and the
//! layouts it cannot bake.
//!
//! The bake rewrites a few files (`gate_vectors.bin`, `down_weights.bin`,
//! `down_meta.bin`, `knn_store.bin`, `index.json`) and carries the rest
//! unchanged. "The rest" is every file in the source directory, not a
//! fixed list, and every file the source's `weight_manifest.json` names is
//! required: one that is missing fails the compile by name. A fixed
//! list silently dropped `lm_head.bin`, and the weight loader skips a
//! missing file without an error, so the artifact loaded and produced
//! wrong logits.

use std::collections::BTreeSet;
use std::path::Path;

use crate::error::LqlError;
use larql_vindex::format::filenames::{
    DOWN_FEATURES_BIN, DOWN_WEIGHTS_BIN, GATE_VECTORS_BIN, GATE_VECTORS_Q4_BIN, INDEX_JSON,
    INTERLEAVED_BIN, INTERLEAVED_Q4_BIN, KNN_STORE_BIN, MODEL_WEIGHTS_BIN,
    REPRESENTATION_ATTESTATIONS_JSON, UP_FEATURES_BIN, WEIGHT_MANIFEST_JSON,
};
use larql_vindex::{QuantFormat, VindexConfig};

/// Manifest field that names the file an entry's bytes live in. Empty or
/// absent means the legacy single weight file.
const MANIFEST_FILE_FIELD: &str = "file";

/// Files the bake writes itself, from the session and not from the
/// source's copy. `knn_store.bin` is here because a source copy would
/// bring back entries the session deleted. `down_meta.bin` is not: it is
/// carried, and replaced only when the overlay edits metadata.
const WRITTEN_BY_THE_BAKE: &[&str] = &[
    GATE_VECTORS_BIN,
    DOWN_WEIGHTS_BIN,
    KNN_STORE_BIN,
    INDEX_JSON,
];

/// Files computed from the weights the bake edits: other encodings of the
/// gate, up and down vectors that faster walk paths open when they exist,
/// and the attestations made over them. The bake does not rewrite them, so
/// a copy carried after a vector edit would run or attest the unedited
/// weights. Without them a loader falls back to the weights the bake did
/// write.
const DERIVED_FROM_THE_WEIGHTS: &[&str] = &[
    INTERLEAVED_BIN,
    INTERLEAVED_Q4_BIN,
    GATE_VECTORS_Q4_BIN,
    UP_FEATURES_BIN,
    DOWN_FEATURES_BIN,
    REPRESENTATION_ATTESTATIONS_JSON,
];

/// Leading character of a file that is not vindex content (`.DS_Store`,
/// `.gitattributes`, a Hub `.cache`).
const HIDDEN_FILE_PREFIX: char = '.';

/// Refuse a source whose weights are not the flat float files the bake
/// reads and rewrites.
///
/// The column bakes patch `down_weights.bin`, `gate_vectors.bin` and the
/// manifest's `up_proj` tensors as f32 or f16. A quantised, block-stored,
/// per-layer or ternary source keeps its executable weights elsewhere, so
/// a compile would publish an artifact without them, or with them
/// unedited.
pub(super) fn refuse_unbakeable_layout(config: &VindexConfig) -> Result<(), LqlError> {
    let layout = if config.quant != QuantFormat::None {
        Some(format!("quantised weights (quant = {})", config.quant))
    } else if config.fp4.is_some() {
        Some("FP4/FP8 block-stored projections".to_string())
    } else if config.ffn_layout.is_some() {
        Some("per-layer FFN weight files".to_string())
    } else if config.bitnet_layout.is_some() {
        Some("BitNet ternary weights".to_string())
    } else {
        None
    };
    match layout {
        None => Ok(()),
        Some(layout) => Err(LqlError::Execution(format!(
            "COMPILE INTO VINDEX cannot bake a source with {layout}: the bake reads and \
             writes flat f32/f16 weight files only, so the result would not contain the \
             source's executable weights"
        ))),
    }
}

/// The weight files the source's manifest names, as plain file names.
///
/// Empty for a source with no manifest and no declared model weights (a
/// browse-only vindex). A source that declares model weights but has no
/// manifest is an error: there is no other record of which files the
/// artifact needs. The manifest decides, not the `has_model_weights`
/// flag, because the flag is a legacy field and a manifest can be
/// present without it.
pub(super) fn manifest_weight_files(
    source: &Path,
    config: &VindexConfig,
) -> Result<BTreeSet<String>, LqlError> {
    let manifest_path = source.join(WEIGHT_MANIFEST_JSON);
    if !manifest_path.is_file() {
        if config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "source vindex declares model weights but has no {WEIGHT_MANIFEST_JSON}: \
                 the compiled vindex would be incomplete"
            )));
        }
        return Ok(BTreeSet::new());
    }
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| LqlError::exec(format!("read {WEIGHT_MANIFEST_JSON}"), e))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| LqlError::exec(format!("parse {WEIGHT_MANIFEST_JSON}"), e))?;

    let mut files = BTreeSet::new();
    for entry in &entries {
        let name = entry
            .get(MANIFEST_FILE_FIELD)
            .and_then(|v| v.as_str())
            .filter(|name| !name.is_empty())
            .unwrap_or(MODEL_WEIGHTS_BIN);
        if !is_plain_file_name(name) {
            return Err(LqlError::Execution(format!(
                "{WEIGHT_MANIFEST_JSON} names {name:?}, which is not a file inside the vindex"
            )));
        }
        files.insert(name.to_string());
    }
    files.insert(WEIGHT_MANIFEST_JSON.to_string());
    Ok(files)
}

/// A single path component that is a normal name: no separators, no
/// `.` or `..`, not absolute.
fn is_plain_file_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(_)), None)
    )
}

/// Carry every file of the source vindex the bake does not write itself.
///
/// Everything in the source directory is part of the vindex: the weight
/// files the manifest names, and the files loaders open by name without a
/// manifest entry (router weights, the tokenizer and generation configs,
/// label files, a legacy `down_meta.jsonl`). A fixed list of names drops
/// whichever file it does not know, so the directory is the list. A
/// subdirectory is refused by name, because nothing here knows what it
/// holds; dot-files are housekeeping, not vindex content.
///
/// A file the manifest names and the source lacks fails by name.
///
/// With `vectors_edited`, the files derived from the weights are left
/// behind, and their names are returned so the caller can say so.
pub(super) fn carry_source_files(
    source: &Path,
    dest: &Path,
    manifest_files: &BTreeSet<String>,
    vectors_edited: bool,
) -> Result<Vec<String>, LqlError> {
    let entries = std::fs::read_dir(source)
        .map_err(|e| LqlError::exec(format!("read {}", source.display()), e))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| LqlError::exec(format!("read {}", source.display()), e))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(HIDDEN_FILE_PREFIX) || is_written_by_the_bake(&name) {
            continue;
        }
        // `Path::is_dir` follows a symlink, so a linked directory is
        // refused the same as a real one.
        if entry.path().is_dir() {
            return Err(LqlError::Execution(format!(
                "source vindex has a directory {name:?}, which COMPILE INTO VINDEX does not \
                 carry: the compiled vindex would be incomplete"
            )));
        }
        names.push(name);
    }
    names.sort_unstable();
    let (left_behind, names): (Vec<String>, Vec<String>) = names
        .into_iter()
        .partition(|name| vectors_edited && DERIVED_FROM_THE_WEIGHTS.contains(&name.as_str()));
    for name in &names {
        carry_file(source, dest, name)?;
    }
    if let Some(missing) = manifest_files
        .iter()
        .find(|name| !is_written_by_the_bake(name) && !names.contains(name))
    {
        return Err(missing_required_file(missing));
    }
    Ok(left_behind)
}

/// Whether the bake produces `name` itself.
fn is_written_by_the_bake(name: &str) -> bool {
    WRITTEN_BY_THE_BAKE.contains(&name)
}

/// Hard-link (or copy) `name` from `source` into `dest`.
///
/// The link is made to the file a symlink resolves to, so a source that
/// is a tree of links (a Hub cache snapshot) is not carried as links that
/// dangle from the new location.
fn carry_file(source: &Path, dest: &Path, name: &str) -> Result<(), LqlError> {
    let src = std::fs::canonicalize(source.join(name))
        .map_err(|e| LqlError::exec(format!("source vindex file {name} cannot be read"), e))?;
    let dst = dest.join(name);
    let _ = std::fs::remove_file(&dst);
    if std::fs::hard_link(&src, &dst).is_err() {
        std::fs::copy(&src, &dst)
            .map_err(|e| LqlError::exec(format!("failed to link or copy {name}"), e))?;
    }
    Ok(())
}

/// The error for a file the manifest names and the source does not have.
pub(super) fn missing_required_file(name: &str) -> LqlError {
    LqlError::Execution(format!(
        "source vindex is missing {name}, which {WEIGHT_MANIFEST_JSON} names: \
         the compiled vindex would be incomplete"
    ))
}
