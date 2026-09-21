//! Independently establish a candidate before asking which experiment requested it.

use std::path::Path;

use super::super::candidate_authority::{
    read_candidate_evidence, CandidateAuthorityRefusal, CandidateEvidence,
};
use super::super::state::identity::RepresentationStateId;

/// Only the independent filesystem reader can construct this evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct EstablishedState(CandidateEvidence);

impl EstablishedState {
    pub fn established(&self) -> &RepresentationStateId {
        self.0.state().id()
    }
    pub fn candidate_authority_digest(&self) -> &str {
        self.0.binding_sha256()
    }
    pub fn state(&self) -> &super::super::state::RepresentationState {
        self.0.state()
    }
}

/// Candidate validity refusals preserve the reader's independently observed facts.
pub type StateEvidenceRefusal = CandidateAuthorityRefusal;

pub struct ArtifactStateEvidence;

impl ArtifactStateEvidence {
    /// No requested key, map or compiler result is an input. A valid Y
    /// succeeds here even if the caller plans to compare it with X later.
    pub fn establish(candidate: &Path) -> Result<EstablishedState, StateEvidenceRefusal> {
        read_candidate_evidence(candidate).map(EstablishedState)
    }
}

#[cfg(test)]
mod tests;
