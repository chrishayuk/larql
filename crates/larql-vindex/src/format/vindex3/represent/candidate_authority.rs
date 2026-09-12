//! Persisted compilation results, independently read back from disk.
//!
//! The writer takes completed decisions and payloads, never a requested
//! measurement key. The reader supplies independently verified inputs to
//! the existing state identity function. This establishes an artifact's
//! representation, not its admission as scientific evidence or promotion.

pub(crate) mod producer;

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::compile::hash_bytes;
use super::compiler::CandidateIndex;
use super::state::{
    RepresentationState, RepresentationStateId, ResolvedDecisionVector, TensorSurface,
    STATE_ID_VERSION,
};
use crate::error::VindexError;

pub const CANDIDATE_AUTHORITY_SCHEMA: &str = "represent-candidate-authority/v1";
/// A full VINDEX3 container keeps its existing index and stores the same
/// CandidateIndex beside it. Candidate provenance must not enter the source
/// semantic identity through Vindex3Index.extra.
pub const CANDIDATE_INDEX_FILE: &str = "candidate.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatePayload {
    pub file: String,
    pub len: u64,
    pub sha256: String,
}

/// The two previously absent identity inputs, bound to completed bytes
/// and the source authority and seals already owned by `CandidateIndex`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRepresentationAuthority {
    pub schema: String,
    pub state_semantics: String,
    pub surface: TensorSurface,
    pub decisions: ResolvedDecisionVector,
    /// One payload file per compiled object. File names locate bytes;
    /// they do not participate in RepresentationStateId.
    pub payloads: BTreeMap<String, CandidatePayload>,
    pub binding_sha256: String,
    /// A cross-check only. Corrupting it cannot supply an identity.
    pub state_id: RepresentationStateId,
}

#[derive(Debug, thiserror::Error)]
pub enum CandidateAuthorityRefusal {
    #[error("candidate authority could not be read or verified: {0}")]
    Invalid(String),
    #[error("candidate independently establishes {observed}, but the requested representation state is {expected}")]
    RequestedState {
        expected: RepresentationStateId,
        observed: RepresentationStateId,
    },
    #[error("candidate has no completed representation authority")]
    Missing,
    #[error("candidate {field} declares {observed}, expected {expected}")]
    Version {
        field: String,
        expected: String,
        observed: String,
    },
    #[error("candidate {what} binding expected {expected}, observed {observed}")]
    Binding {
        what: String,
        expected: String,
        observed: String,
    },
    #[error("candidate stored state id {stored} disagrees with independently reconstructed {recomputed}")]
    StoredState {
        stored: RepresentationStateId,
        recomputed: RepresentationStateId,
    },
}

fn invalid(error: impl std::fmt::Display) -> CandidateAuthorityRefusal {
    CandidateAuthorityRefusal::Invalid(error.to_string())
}

fn relative_file(root: &Path, file: &str) -> Result<std::path::PathBuf, CandidateAuthorityRefusal> {
    if file.is_empty()
        || Path::new(file)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid(format!(
            "payload locator `{file}` is not a relative file"
        )));
    }
    Ok(root.join(file))
}

/// Bounded streaming reads; sealing a large bank never materialises it.
pub(crate) fn digest_range(path: &Path, offset: u64, len: u64) -> Result<String, VindexError> {
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut hash = Sha256::new();
    let mut remaining = len;
    let mut buffer = vec![0u8; 1 << 20];
    while remaining != 0 {
        let n = remaining.min(buffer.len() as u64) as usize;
        file.read_exact(&mut buffer[..n])?;
        hash.update(&buffer[..n]);
        remaining -= n as u64;
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn binding(
    index: &CandidateIndex,
    authority: &CandidateRepresentationAuthority,
) -> Result<String, VindexError> {
    // Provenance, requested map and promotion are deliberately absent.
    // The binding is an artifact integrity seal, NOT a second state-id recipe.
    let bytes = serde_json::to_vec(&(
        &authority.schema,
        &authority.state_semantics,
        &index.source.identity.semantic,
        &authority.surface,
        &authority.decisions,
        &index.ledger.sealed,
        &authority.payloads,
    ))
    .map_err(|e| VindexError::Parse(e.to_string()))?;
    Ok(hash_bytes(&bytes))
}

/// Called only by producers after their actual encoding/write decisions.
/// No state id, request, layout policy or precision map is an input.
pub(crate) fn finish(
    index: &mut CandidateIndex,
    root: &Path,
    surface: TensorSurface,
    decisions: ResolvedDecisionVector,
    files: BTreeMap<String, String>,
) -> Result<(), VindexError> {
    for seal in index.ledger.sealed.values() {
        if !matches!(decisions.get(&seal.object, &seal.tensor), Some(super::state::ResolvedEncoding::Compiled(enc)) if enc == &seal.encoding)
        {
            return Err(VindexError::Parse(format!(
                "completed seal {}/{} is outside the compilation's effective surface",
                seal.object, seal.tensor
            )));
        }
    }
    let mut payloads = BTreeMap::new();
    for (object, file) in files {
        let path = relative_file(root, &file).map_err(|e| VindexError::Parse(e.to_string()))?;
        let len = std::fs::metadata(&path)?.len();
        payloads.insert(
            object,
            CandidatePayload {
                file,
                len,
                sha256: digest_range(&path, 0, len)?,
            },
        );
    }
    let state =
        RepresentationState::from_decisions(&index.source.identity, &surface, decisions.clone());
    let mut authority = CandidateRepresentationAuthority {
        schema: CANDIDATE_AUTHORITY_SCHEMA.into(),
        state_semantics: STATE_ID_VERSION.into(),
        surface,
        decisions,
        payloads,
        binding_sha256: String::new(),
        state_id: state.id().clone(),
    };
    authority.binding_sha256 = binding(index, &authority)?;
    index.authority = Some(authority);
    Ok(())
}

/// Open only the persisted artifact. This path never consults compiler
/// memory, a locator's state-id key, or a requested PrecisionMap.
pub fn read_candidate(path: &Path) -> Result<RepresentationState, CandidateAuthorityRefusal> {
    let sidecar = path.join(CANDIDATE_INDEX_FILE);
    let file = if sidecar.try_exists().map_err(invalid)? {
        sidecar
    } else {
        path.join("index.json")
    };
    let index: CandidateIndex =
        serde_json::from_slice(&std::fs::read(file).map_err(invalid)?).map_err(invalid)?;
    let a = index
        .authority
        .as_ref()
        .ok_or(CandidateAuthorityRefusal::Missing)?;
    for (field, expected, observed) in [
        ("schema", CANDIDATE_AUTHORITY_SCHEMA, a.schema.as_str()),
        (
            "state semantics",
            STATE_ID_VERSION,
            a.state_semantics.as_str(),
        ),
    ] {
        if expected != observed {
            return Err(CandidateAuthorityRefusal::Version {
                field: field.into(),
                expected: expected.into(),
                observed: observed.into(),
            });
        }
    }
    let observed = binding(&index, a).map_err(invalid)?;
    if observed != a.binding_sha256 {
        return Err(CandidateAuthorityRefusal::Binding {
            what: "metadata".into(),
            expected: a.binding_sha256.clone(),
            observed,
        });
    }
    // Derived serde can bypass sorted/unique constructors. Re-establish
    // the surface and vector invariants before normative identification.
    let surface = TensorSurface::new(a.surface.entries().iter().cloned()).map_err(invalid)?;
    if surface != a.surface {
        return Err(invalid("tensor surface is not in canonical order"));
    }
    let decisions =
        ResolvedDecisionVector::from_entries(&surface, a.decisions.decisions().to_vec())
            .map_err(invalid)?;
    if decisions != a.decisions {
        return Err(invalid("decision vector is not in canonical order"));
    }
    for (object, payload) in &a.payloads {
        let file = relative_file(path, &payload.file)?;
        let len = std::fs::metadata(&file).map_err(invalid)?.len();
        let observed = digest_range(&file, 0, len).map_err(invalid)?;
        if len != payload.len || observed != payload.sha256 {
            return Err(CandidateAuthorityRefusal::Binding {
                what: format!(
                    "payload `{object}` (expected {} bytes, observed {len})",
                    payload.len
                ),
                expected: payload.sha256.clone(),
                observed,
            });
        }
    }
    let mut spans: BTreeMap<&str, Vec<(u64, u64)>> = BTreeMap::new();
    for seal in index.ledger.sealed.values() {
        let payload = a
            .payloads
            .get(&seal.object)
            .ok_or_else(|| invalid(format!("no payload for sealed object {}", seal.object)))?;
        let end = seal
            .target_offset
            .checked_add(seal.target_len)
            .filter(|end| *end <= payload.len)
            .ok_or_else(|| invalid(format!("seal for {} is outside its payload", seal.tensor)))?;
        if !matches!(decisions.get(&seal.object, &seal.tensor), Some(super::state::ResolvedEncoding::Compiled(enc)) if enc == &seal.encoding)
        {
            return Err(invalid(format!(
                "seal for {}/{} disagrees with effective decisions",
                seal.object, seal.tensor
            )));
        }
        if index.ledger.get(&seal.object, &seal.tensor) != Some(seal) {
            return Err(invalid("ledger key disagrees with sealed operand identity"));
        }
        let observed = digest_range(
            &relative_file(path, &payload.file)?,
            seal.target_offset,
            seal.target_len,
        )
        .map_err(invalid)?;
        if observed != seal.target_hash {
            return Err(CandidateAuthorityRefusal::Binding {
                what: format!("operand {}/{}", seal.object, seal.tensor),
                expected: seal.target_hash.clone(),
                observed,
            });
        }
        spans
            .entry(&seal.object)
            .or_default()
            .push((seal.target_offset, end));
    }
    for ranges in spans.values_mut() {
        ranges.sort_unstable();
        if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(invalid("compiled operand seals overlap"));
        }
    }
    for d in decisions
        .decisions()
        .iter()
        .filter(|d| d.encoding.is_compiled())
    {
        if index.ledger.get(&d.object, &d.tensor).is_none() {
            return Err(invalid(format!(
                "compiled decision {}/{} has no completed seal",
                d.object, d.tensor
            )));
        }
    }
    let state = RepresentationState::from_decisions(&index.source.identity, &surface, decisions);
    if state.id() != &a.state_id {
        return Err(CandidateAuthorityRefusal::StoredState {
            stored: a.state_id.clone(),
            recomputed: state.id().clone(),
        });
    }
    Ok(state)
}

/// Establish validity first, then compare experiment identity. A valid Y
/// remains independently readable even when a caller requested X.
pub fn verify_candidate(
    path: &Path,
    expected: &RepresentationStateId,
) -> Result<RepresentationState, CandidateAuthorityRefusal> {
    let state = read_candidate(path)?;
    if state.id() != expected {
        return Err(CandidateAuthorityRefusal::RequestedState {
            expected: expected.clone(),
            observed: state.id().clone(),
        });
    }
    Ok(state)
}

#[cfg(test)]
mod tests;
