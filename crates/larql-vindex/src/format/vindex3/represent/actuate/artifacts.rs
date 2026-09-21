//! **A locator that was told where things are — and checks anyway.**
//!
//! The doctrine this implements is already stated twice in the tree:
//! `EvidenceBank::locator_hint` is *where a bank was last seen, never
//! part of what it is*, and `SourceDependency` says the same for
//! containers. A path is therefore an input to finding something and
//! never evidence that the thing was found.
//!
//! ```text
//! container   identity read back from its own index and compared
//! corpus      manifest.json digested and compared with the bank id's
//!             own input
//! overlay     looked up by the physical state it must present
//! ```
//!
//! Two of the three are checkable here and the third is not, and saying
//! which is which matters. A compiled overlay carries no identity this
//! locator can read — establishing that the bytes on disk present the
//! state the key names requires opening them under the same layout and
//! accounting authority the record declares, which is the executor's
//! bind step and, for admissibility, stage 6's. So the overlay is looked
//! up by state id and refused when absent, and this module does not
//! claim to have verified it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::super::compile::hash_bytes;
use super::super::state::identity::RepresentationStateId;
use super::executor::{verify_container, ArtifactLocator, LocatorRefusal};
use super::request::MeasurementRequest;

/// The corpus file whose digest the bank's identity is computed over.
pub const BANK_MANIFEST: &str = "manifest.json";

/// **Artifacts this machine holds, declared by a caller.**
///
/// Deliberately dumb about discovery: it searches nothing and infers
/// nothing. Something above it — a CLI flag, a build step, a fleet
/// service — decides what is present, and this turns that declaration
/// into checked answers.
#[derive(Debug, Clone, Default)]
pub struct DeclaredArtifacts {
    container: Option<PathBuf>,
    corpus: Option<PathBuf>,
    overlays: BTreeMap<RepresentationStateId, PathBuf>,
}

impl DeclaredArtifacts {
    pub fn new() -> Self {
        Self::default()
    }

    /// The container the baseline arm reads. Verified on every answer,
    /// not on registration: a container can be replaced under a path
    /// between one experiment and the next.
    ///
    /// Named `_at` throughout so the declaration cannot be confused with
    /// [`ArtifactLocator::container`], which ANSWERS about one — the
    /// distinction between what a caller says it holds and what this
    /// will vouch for.
    pub fn container_at(mut self, path: impl Into<PathBuf>) -> Self {
        self.container = Some(path.into());
        self
    }

    /// The exported corpus directory.
    pub fn corpus_at(mut self, path: impl Into<PathBuf>) -> Self {
        self.corpus = Some(path.into());
        self
    }

    /// An overlay that has been compiled, and the physical state it
    /// presents.
    pub fn overlay_at(mut self, state: &RepresentationStateId, path: impl Into<PathBuf>) -> Self {
        self.overlays.insert(state.clone(), path.into());
        self
    }
}

impl ArtifactLocator for DeclaredArtifacts {
    fn container(&self, request: &MeasurementRequest) -> Result<PathBuf, LocatorRefusal> {
        let path = self
            .container
            .clone()
            .ok_or_else(|| LocatorRefusal::NotHeld {
                what: "container".into(),
                identity: request.model().semantic_digest(),
            })?;
        verify_container(request, &path)?;
        Ok(path)
    }

    fn corpus(&self, request: &MeasurementRequest) -> Result<PathBuf, LocatorRefusal> {
        let bank = request.bank();
        let path = self.corpus.clone().ok_or_else(|| LocatorRefusal::NotHeld {
            what: "quality bank".into(),
            identity: bank.id().to_string(),
        })?;
        let manifest = path.join(BANK_MANIFEST);
        let unreadable = |detail: String| LocatorRefusal::NotWhatItClaims {
            what: "quality bank".into(),
            path: path.display().to_string(),
            detail,
        };
        let bytes = std::fs::read(&manifest)
            .map_err(|e| unreadable(format!("its `{BANK_MANIFEST}` could not be read: {e}")))?;
        let found = hash_bytes(&bytes);
        if found != bank.manifest_sha256 {
            return Err(unreadable(format!(
                "its `{BANK_MANIFEST}` digests to {} and this experiment's bank was identified \
                 over {}",
                short(&found),
                short(&bank.manifest_sha256)
            )));
        }
        Ok(path)
    }

    fn candidate(&self, request: &MeasurementRequest) -> Result<PathBuf, LocatorRefusal> {
        let state = request.key().state();
        self.overlays
            .get(state)
            .cloned()
            .ok_or_else(|| LocatorRefusal::NotBuilt {
                state: state.clone(),
                map: request.candidate_map().name.clone(),
            })
    }
}

fn short(digest: &str) -> &str {
    &digest[..digest.len().min(12)]
}

/// Where the overlay for one state was declared, if anywhere. Exposed
/// so a caller can report what it holds without an experiment in hand.
impl DeclaredArtifacts {
    pub fn overlay_for(&self, state: &RepresentationStateId) -> Option<&Path> {
        self.overlays.get(state).map(PathBuf::as_path)
    }
}
