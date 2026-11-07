//! Is every file a VINDEX2 index needs actually on disk?
//!
//! A VINDEX2 directory is opened by `index.json` alone, so a download that
//! stopped after the metadata — `index.json`, the tokenizer, `down_meta.bin`
//! — opens, describes itself and lists as healthy while the gigabytes it
//! needs are absent.
//!
//! # Which files are required — positive evidence only
//!
//! `checksums` is NOT the required set. `larql slice` copies a subset of
//! files and rewrites `extract_level` / `has_model_weights`, but slices
//! published before it pruned `checksums` carry their source's whole map:
//! `chrishayuk/gemma-3-4b-it-vindex-client` declares checksums for
//! `gate_vectors.bin`, `embeddings.bin`, `down_meta.bin` and `norms.bin`
//! while shipping no gate vectors. A slice carries no preset marker, so the
//! only declarations that say what it should contain are its extract level
//! (and `has_model_weights`, which separates nothing the level does not).
//!
//! The required set is therefore the files EVERY shape claiming that level
//! must carry — the intersection over the slice presets that produce it
//! and a full extract — and of those, only the ones the index itself
//! declares by checksum:
//!
//! - `browse`: nothing. The `embed`, `browse`, `router` and `server`
//!   slices all claim it and ship disjoint files (router ships neither
//!   embeddings nor gate vectors), so no file is evidence of a broken one.
//! - `attention`: `norms.bin`. Both the `client` and `attn` slices carry
//!   attention + norms (a slice reaches this level only with both); `attn`
//!   drops the embeddings, delegated to an embed server.
//! - `inference` / `all`: `embeddings.bin` and `norms.bin`. Reaching these
//!   needs attention, norms and FFN, and the loader opens the embeddings
//!   unconditionally. Gate vectors are not required: a k-quant vindex can
//!   synthesise them from `interleaved_kquant.bin`.
//!
//! Where the declarations cannot establish that a file is required, it is
//! not checked: an entry is called incomplete only on positive evidence,
//! never on a guess. Presence and declared extents only — nothing is
//! hashed (that is `larql verify`).

use std::path::Path;

use crate::config::{ExtractLevel, VindexConfig};
use crate::format::filenames::{EMBEDDINGS_BIN, GATE_VECTORS_BIN, NORMS_BIN};

/// Files every vindex claiming `level` must carry, whatever produced it,
/// in name order. Tiers come from the level's own predicates (what an
/// extract at that level writes); see the module docs for why each file
/// survives every slice that claims the tier.
pub fn files_required_at(level: ExtractLevel) -> Vec<&'static str> {
    let mut required = Vec::new();
    if level.writes_ffn() {
        // Every shape this high carries attention, norms and FFN, and the
        // loader opens the embeddings unconditionally.
        required.push(EMBEDDINGS_BIN);
    }
    if level.writes_attn() {
        // A slice reaches attention only with the norms part.
        required.push(NORMS_BIN);
    }
    required
}

/// What a VINDEX2 directory is missing against its own `index.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncompleteVindex {
    /// How many files the index's extract level requires.
    pub required: usize,
    /// The extract level the requirement comes from.
    pub level: ExtractLevel,
    /// Required files absent from the directory, in name order.
    pub missing: Vec<String>,
    /// Files present but shorter than the index says:
    /// `(name, bytes on disk, bytes declared)`.
    pub truncated: Vec<(String, u64, u64)>,
}

impl std::fmt::Display for IncompleteVindex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        if let Some(first) = self.missing.first() {
            parts.push(format!(
                "{} of {} file(s) required at extract level {} missing (first: {first})",
                self.missing.len(),
                self.required,
                self.level
            ));
        }
        if let Some((name, have, want)) = self.truncated.first() {
            parts.push(format!(
                "{name} is {have} bytes, index.json declares at least {want}"
            ));
        }
        f.write_str(&parts.join("; "))
    }
}

/// The files `config` declares, in name order: every `checksums` entry.
/// An index written without checksums declares none.
pub fn declared_files(config: &VindexConfig) -> Vec<String> {
    let mut names: Vec<String> = config
        .checksums
        .iter()
        .flat_map(|sums| sums.keys().cloned())
        .collect();
    names.sort();
    names
}

/// The files `dir` must hold, in name order: those its extract level
/// requires ([`files_required_at`]) that its index also declares. An
/// index without checksums requires nothing — there is no evidence the
/// writer produced any of them.
pub fn required_files(config: &VindexConfig) -> Vec<String> {
    let required = files_required_at(config.extract_level);
    declared_files(config)
        .into_iter()
        .filter(|name| required.contains(&name.as_str()))
        .collect()
}

/// The least size `file` must have, when the index states one: the end of
/// the furthest layer slice for `gate_vectors.bin`. `None` when the index
/// declares no extent for it, or one that overflows (an index that cannot
/// be honoured is the loader's to refuse, not a size to compare against).
pub fn declared_min_size(config: &VindexConfig, file: &str) -> Option<u64> {
    if file != GATE_VECTORS_BIN {
        return None;
    }
    config
        .layers
        .iter()
        .map(|layer| layer.offset.checked_add(layer.length))
        .try_fold(0u64, |end, slice| slice.map(|s| end.max(s)))
        .filter(|end| *end > 0)
}

/// Check `dir` against what `config` (its own `index.json`) requires:
/// every [`required_files`] entry present, and a gate-vector file that is
/// present no shorter than the layer slices the index declares (a short
/// file is positive evidence whatever the slice).
pub fn check_declared_files(dir: &Path, config: &VindexConfig) -> Result<(), IncompleteVindex> {
    let required = required_files(config);
    let mut report = IncompleteVindex {
        required: required.len(),
        level: config.extract_level,
        missing: Vec::new(),
        truncated: Vec::new(),
    };
    for name in required {
        // `metadata` follows symlinks, so a hub snapshot's link to a blob
        // that never arrived reads as missing, as it should.
        if std::fs::metadata(dir.join(&name)).is_err() {
            report.missing.push(name);
        }
    }
    if let (Some(want), Ok(meta)) = (
        declared_min_size(config, GATE_VECTORS_BIN),
        std::fs::metadata(dir.join(GATE_VECTORS_BIN)),
    ) {
        if meta.len() < want {
            report
                .truncated
                .push((GATE_VECTORS_BIN.to_string(), meta.len(), want));
        }
    }
    if report.missing.is_empty() && report.truncated.is_empty() {
        Ok(())
    } else {
        Err(report)
    }
}

#[cfg(test)]
mod tests;
