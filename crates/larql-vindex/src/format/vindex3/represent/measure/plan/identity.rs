//! Byte identity for MEASURE-PLAN-1: what each arm bound, whether the
//! representations the arms share are the same bytes, and whether every
//! bound representation still matches its seal.
//!
//! A representation is named `object@ENCODING`, the container directory's
//! own key. Digests are the directory's recorded `payload_sha256`, and the
//! seal check recomputes that hash from the segment now, through the same
//! function `vindex3 verify` uses. So the two cannot disagree about what a
//! payload digest is.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::arm::ArmDescription;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::encode::REPRESENTATION_ID_SEP;
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::verify_system::payload::payload_region_hash;

/// A representation id, `object@ENCODING`.
pub fn representation_id(object: &str, encoding: &str) -> String {
    format!("{object}{REPRESENTATION_ID_SEP}{encoding}")
}

/// The representations an arm bound, by id.
pub fn bound_representations(arm: &ArmDescription) -> BTreeSet<String> {
    arm.objects
        .iter()
        .map(|(object, bound)| representation_id(object, &bound.encoding))
        .collect()
}

/// The representations the candidate bound that the reference did not: the
/// changed variable. Empty means the arms bound the same bytes.
pub fn changed_representations(
    reference: &BTreeSet<String>,
    candidate: &BTreeSet<String>,
) -> BTreeSet<String> {
    candidate.difference(reference).cloned().collect()
}

/// The payload digest of every representation in `bound`, recomputed from
/// `root`'s segments now. Err names a representation whose segment could
/// not be read.
pub fn recompute_digests(
    root: &Path,
    index: &Vindex3Index,
    bound: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>, String> {
    let mut digests = BTreeMap::new();
    for id in bound {
        let entry = index
            .representations
            .get(id)
            .ok_or_else(|| format!("{id}: no directory entry in {}", root.display()))?;
        let path = root.join(&entry.segment);
        let (_, payload_start) = read_segment_header(&path).map_err(|e| format!("{id}: {e}"))?;
        let digest = payload_region_hash(&path, payload_start).map_err(|e| format!("{id}: {e}"))?;
        digests.insert(id.clone(), digest);
    }
    Ok(digests)
}

/// Representations in `shared` whose RECOMPUTED bytes differ between the
/// two containers: protected operands that changed. Recomputed rather than
/// recorded, because a tampered segment still carries its old recorded
/// digest.
///
/// This cannot see an in-place edit to a segment the two containers share
/// on disk. `represent` hard-links the segments it carries unchanged, so
/// such an edit changes both, and they still agree. The seal check
/// ([`seal_break`]) catches that case, because the recomputed digest no
/// longer matches the recorded one, which is why the procedure runs both.
pub fn protected_changes(
    shared: &BTreeSet<String>,
    reference: &BTreeMap<String, String>,
    candidate: &BTreeMap<String, String>,
) -> Vec<String> {
    shared
        .iter()
        .filter(|id| reference.get(*id) != candidate.get(*id))
        .cloned()
        .collect()
}

/// A bound representation whose bytes no longer match the recorded digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealBreak {
    pub representation: String,
    pub expected: String,
    pub found: String,
}

/// The first representation whose recomputed digest is not the directory's.
pub fn seal_break(
    index: &Vindex3Index,
    recomputed: &BTreeMap<String, String>,
) -> Option<SealBreak> {
    recomputed.iter().find_map(|(id, found)| {
        let expected = index
            .representations
            .get(id)
            .map_or_else(String::new, |e| e.payload_sha256.clone());
        (&expected != found).then(|| SealBreak {
            representation: id.clone(),
            expected,
            found: found.clone(),
        })
    })
}

/// A bound representation as the container's directory records it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RecordedRepresentation {
    pub payload_sha256: String,
    pub payload_bytes: u64,
}

/// The directory's record of every bound representation, for the report.
pub fn recorded_representations(
    index: &Vindex3Index,
    bound: &BTreeSet<String>,
) -> BTreeMap<String, RecordedRepresentation> {
    bound
        .iter()
        .filter_map(|id| {
            index.representations.get(id).map(|e| {
                (
                    id.clone(),
                    RecordedRepresentation {
                        payload_sha256: e.payload_sha256.clone(),
                        payload_bytes: e.payload_bytes,
                    },
                )
            })
        })
        .collect()
}
