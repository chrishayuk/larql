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
use crate::format::vindex3::index::Vindex3Index;

/// A representation id, `object@ENCODING`.
pub fn representation_id(object: &str, encoding: &str) -> String {
    let _ = (object, encoding);
    todo!("MEASURE-PLAN-1 PR 2")
}

/// The representations an arm bound, by id.
pub fn bound_representations(arm: &ArmDescription) -> BTreeSet<String> {
    let _ = arm;
    todo!("MEASURE-PLAN-1 PR 2")
}

/// The representations the candidate bound that the reference did not: the
/// changed variable. Empty means the arms bound the same bytes.
pub fn changed_representations(
    reference: &BTreeSet<String>,
    candidate: &BTreeSet<String>,
) -> BTreeSet<String> {
    let _ = (reference, candidate);
    todo!("MEASURE-PLAN-1 PR 2")
}

/// The payload digest of every representation in `bound`, recomputed from
/// `root`'s segments now. Err names a representation whose segment could
/// not be read.
pub fn recompute_digests(
    root: &Path,
    index: &Vindex3Index,
    bound: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>, String> {
    let _ = (root, index, bound);
    todo!("MEASURE-PLAN-1 PR 2")
}

/// Representations in `shared` whose RECOMPUTED bytes differ between the
/// two containers: protected operands that changed. Recomputed rather than
/// recorded, because a tampered segment still carries its old recorded
/// digest.
pub fn protected_changes(
    shared: &BTreeSet<String>,
    reference: &BTreeMap<String, String>,
    candidate: &BTreeMap<String, String>,
) -> Vec<String> {
    let _ = (shared, reference, candidate);
    todo!("MEASURE-PLAN-1 PR 2")
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
    let _ = (index, recomputed);
    todo!("MEASURE-PLAN-1 PR 2")
}

/// Recorded digests by representation id, for the report.
pub fn recorded_digests(
    index: &Vindex3Index,
    bound: &BTreeSet<String>,
) -> BTreeMap<String, String> {
    let _ = (index, bound);
    todo!("MEASURE-PLAN-1 PR 2")
}
