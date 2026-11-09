//! `COMPACT INTO VINDEX` for a VINDEX3 container: **semantics-
//! preserving physical reorganisation**.
//!
//! COMPILE and COMPACT are deliberately different operations:
//!
//! - COMPILE removes logical overlays by materialising meaning into
//!   rewritten segments;
//! - COMPACT changes physical organisation while preserving meaning
//!   exactly — its proof instrument is the logical DIFF
//!   (`SemanticDiff(input, output) == ∅`), and physical change is
//!   expected, not tolerated.
//!
//! Today's physical policy is garbage collection: the output carries
//! exactly the files the index names (segments, byte-identical — the
//! index and its hashes are carried unchanged) plus the recognised
//! capability files; anything else in the container directory — a
//! crash-leftover from an interrupted bake, a stray artifact — is
//! dropped and reported. Segment re-packing and representation
//! re-choice are later policies behind the same statement.

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::path::Path;

use super::encode::{SEGMENTS_DIR, SYSTEM_GRAPH_JSON};
use super::index::Vindex3Index;
use crate::error::VindexError;
use crate::format::filenames::{INDEX_JSON, KNN_STORE_BIN};

/// What one compact did.
#[derive(Debug)]
pub struct CompactReport {
    /// Segment files carried (byte-identical, linked where possible).
    pub carried_segments: usize,
    /// Files present in the source directory that the container does
    /// not reference — dropped, and named so nothing vanishes silently.
    pub dropped: Vec<String>,
}

/// Capability files a container legitimately carries besides what the
/// index names.
const CAPABILITY_FILES: &[&str] = &[
    INDEX_JSON,
    SYSTEM_GRAPH_JSON,
    "moe_manifest.json",
    "tokenizer.json",
    KNN_STORE_BIN,
];

/// Reorganise the container at `src` into `out`, preserving semantics
/// exactly.
pub fn compact_container(src: &Path, out: &Path) -> Result<CompactReport, VindexError> {
    let raw_index = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw_index)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    std::fs::create_dir_all(out)?;

    // Everything the container REFERENCES, relative to its root.
    let mut referenced: BTreeSet<String> = CAPABILITY_FILES.iter().map(|s| s.to_string()).collect();
    for entry in index.representations.values() {
        referenced.insert(entry.segment.clone());
    }

    let mut report = CompactReport {
        carried_segments: 0,
        dropped: Vec::new(),
    };

    // Carry referenced files byte-identically; report everything else.
    let mut walk = vec![src.to_path_buf()];
    while let Some(dir) = walk.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk.push(path);
                continue;
            }
            // Index segment paths use `/`; normalise the walked path
            // so Windows' `\` separators compare equal.
            let rel = path
                .strip_prefix(src)
                .map_err(|e| VindexError::Parse(format!("walk escaped the container: {e}")))?
                .to_string_lossy()
                .replace('\\', "/");
            if referenced.contains(&rel) {
                let dst = super::container_path::prepare_write(out, &rel)?;
                if std::fs::hard_link(&path, &dst).is_err() {
                    std::fs::copy(&path, &dst)?;
                }
                if rel.starts_with(SEGMENTS_DIR) || rel.ends_with(".lyrw") {
                    report.carried_segments += 1;
                }
            } else {
                report.dropped.push(rel);
            }
        }
    }
    report.dropped.sort();

    // Every referenced segment must have made it — a container that
    // references a missing file is broken, and compact must say so
    // rather than emit a smaller broken copy.
    for entry in index.representations.values() {
        if !out.join(&entry.segment).exists() {
            return Err(VindexError::Parse(format!(
                "referenced segment `{}` is missing from the source container",
                entry.segment
            )));
        }
    }

    Ok(report)
}
