//! VINDEX3 payload verification at the moment a pull lands.
//!
//! The hub vouches for transport: a blob is the bytes the repo holds.
//! It says nothing about whether those bytes are the bytes the
//! container's own `index.json` seals — a repo whose segments were
//! re-uploaded under an old index, or one assembled to mislead, arrives
//! intact and wrong. Every runtime opener reads containers with payload
//! hashing off (it is a full read of the model), so the one place the
//! container's word about itself is checked is here.
//!
//! # The marker is a cache optimisation, not an attestation
//!
//! A marker beside `index.json` holds the digest of the index whose
//! segments were last verified in this directory, so a pull that
//! fetches nothing does not re-hash the whole model. It says "this pull
//! path verified these bytes and has not replaced any since"; it does
//! NOT defend against a local process rewriting cached segments in
//! place. The cache is trusted as the user's filesystem. Independent
//! verification of what is on disk now is `larql verify`, which always
//! hashes.
//!
//! # Lifecycle
//!
//! [`PendingVerification::begin`] runs after `index.json` is in place and
//! BEFORE any segment is fetched. It records whether the marker vouched
//! for this exact index, then deletes it. Any replacement that follows —
//! completed, failed, or interrupted — therefore happens with no marker
//! on disk. [`PendingVerification::finish`] restores the marker only when
//! the pull fetched nothing and the marker had vouched for this index;
//! otherwise it hashes every segment and writes the marker (atomically,
//! via rename) only if all of them agree. A pull that never reaches
//! `finish` leaves no marker, and the next pull hashes.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::segment_digest_mismatch;

/// Beside `index.json`: the hex SHA-256 of the index whose segments were
/// last verified against it in this directory.
pub(super) const VERIFIED_MARKER: &str = ".larql-payloads-verified";

/// Written first, then renamed over [`VERIFIED_MARKER`].
const MARKER_STAGING_SUFFIX: &str = ".incomplete";

/// A pull's verification, opened before its segments are fetched.
#[must_use = "a pull that never finishes verification leaves no marker"]
pub(super) struct PendingVerification {
    dir: PathBuf,
    index_digest: String,
    previously_verified: bool,
}

impl PendingVerification {
    /// Invalidate the marker before anything in `dir` can be replaced.
    pub(super) fn begin(dir: &Path) -> Result<Self, VindexError> {
        let index_digest = digest_of_index(dir)?;
        let marker = dir.join(VERIFIED_MARKER);
        let previously_verified =
            std::fs::read_to_string(&marker).ok().as_deref() == Some(index_digest.as_str());
        remove_if_present(&marker)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            index_digest,
            previously_verified,
        })
    }

    /// Verify (unless provably unnecessary) and re-arm the marker.
    pub(super) fn finish(self, fetched_any: bool) -> Result<(), VindexError> {
        // The index is re-read, not trusted from `begin`: if it changed
        // under the pull, the digest it is verified against must be its.
        let index_digest = digest_of_index(&self.dir)?;
        let unchanged = index_digest == self.index_digest;
        if !(self.previously_verified && unchanged && !fetched_any) {
            verify_segments(&self.dir)?;
        }
        write_marker(&self.dir, &index_digest)
    }
}

/// `begin` then `finish`: one pull whose fetching already happened.
#[cfg(test)]
pub(super) fn verify_v3_payloads(dir: &Path, fetched_any: bool) -> Result<(), VindexError> {
    PendingVerification::begin(dir)?.finish(fetched_any)
}

fn digest_of_index(dir: &Path) -> Result<String, VindexError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(dir.join(INDEX_JSON))?)
    ))
}

fn verify_segments(dir: &Path) -> Result<(), VindexError> {
    let index: Vindex3Index = serde_json::from_slice(&std::fs::read(dir.join(INDEX_JSON))?)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    let mut mismatches = Vec::new();
    for entry in index.representations.values() {
        match segment_digest_mismatch(dir, entry) {
            Ok(None) => {}
            Ok(Some(mismatch)) => mismatches.push(mismatch),
            Err(e) => mismatches.push(format!("`{}` cannot be read: {e}", entry.segment)),
        }
    }
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(VindexError::PayloadDigestMismatch {
            container: dir.to_path_buf(),
            mismatches,
        })
    }
}

fn write_marker(dir: &Path, index_digest: &str) -> Result<(), VindexError> {
    let marker = dir.join(VERIFIED_MARKER);
    let staging = dir.join(format!("{VERIFIED_MARKER}{MARKER_STAGING_SUFFIX}"));
    remove_if_present(&staging)?;
    std::fs::write(&staging, index_digest)?;
    std::fs::rename(&staging, &marker)?;
    Ok(())
}

fn remove_if_present(path: &Path) -> Result<(), VindexError> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}
