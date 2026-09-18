//! **The raw teacher-forced observation stream, persisted.**
//!
//! REAL-EVIDENCE-1
//! (`docs/represent/forecasts/represent-real-evidence-1.json`).
//!
//! The stream is AUTHORITATIVE. Every `QualityBank` at every depth is a
//! deterministic projection of it, regenerable when the estimator
//! changes without rerunning the model — which is the whole reason the
//! summaries alone were not enough.
//!
//! **The gate this module exists to pass:** a bank rederived from the
//! persisted stream must equal the bank built directly from the same
//! observations, exactly. If it does not, the recording layer has
//! changed the experiment, and every later statistic is a property of
//! the recorder rather than of the model.
//!
//! `PositionObservation` is not redesigned. It already carries top-N ids
//! with both arms' logits at THE SAME ids, each arm's full-vocabulary
//! logsumexp so probabilities are exact rather than truncated, and each
//! arm's argmax over the full vocabulary. The stream persists it
//! verbatim.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use super::bank::{BankBuilder, PositionObservation};
use super::compile::hash_bytes;
use super::quality::QualityBank;

/// Bumped when the on-disk shape changes in a way a reader must notice.
pub const STREAM_FORMAT: &str = "represent-observation-stream/v1";

const MANIFEST: &str = "stream-manifest.json";
const BODY: &str = "observations.jsonl.gz";

/// What binds a stream to the experiment that produced it.
///
/// Identity is the scientific contract; the encoding is plumbing. Every
/// field here answers a question a later reader must not have to guess:
/// which model, which transition, which protocol, which code, which
/// sequences, which draw, and whether the bytes are the ones that were
/// written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamManifest {
    pub format: String,
    /// Model / source container identity.
    pub source_identity: String,
    /// Candidate / transition under measurement.
    pub candidate_identity: String,
    /// Measurement protocol the run declared.
    pub protocol_identity: String,
    /// Code identity — commit or build stamp.
    pub code_identity: String,
    /// Quality-bank / sequence-set identity.
    pub bank_identity: String,
    /// Which draw this is, when several share a bank.
    pub draw_identity: String,
    pub sequences: u32,
    pub positions_per_sequence: u32,
    pub observations: u64,
    /// Over the compressed body exactly as written.
    pub stream_sha256: String,
}

#[derive(Debug)]
pub enum StreamError {
    Io(String),
    Encoding(String),
    /// The body's hash is not the one the manifest names.
    Tampered {
        expected: String,
        found: String,
    },
    /// The manifest's count is not the number of observations present.
    CountMismatch {
        declared: u64,
        found: u64,
    },
    UnknownFormat(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "stream io: {e}"),
            Self::Encoding(e) => write!(f, "stream encoding: {e}"),
            Self::Tampered { expected, found } => write!(
                f,
                "observation stream does not hash to what its manifest declares; \
                 expected {expected}, found {found}. The stream is authoritative, \
                 so a mismatch invalidates every bank derived from it"
            ),
            Self::CountMismatch { declared, found } => write!(
                f,
                "manifest declares {declared} observations, stream holds {found}"
            ),
            Self::UnknownFormat(v) => write!(f, "unknown stream format {v}"),
        }
    }
}

/// What a caller knows before the observations exist.
#[derive(Debug, Clone)]
pub struct StreamIdentity {
    pub source_identity: String,
    pub candidate_identity: String,
    pub protocol_identity: String,
    pub code_identity: String,
    pub bank_identity: String,
    pub draw_identity: String,
    pub sequences: u32,
    pub positions_per_sequence: u32,
}

fn io<E: std::fmt::Display>(e: E) -> StreamError {
    StreamError::Io(e.to_string())
}

/// Persist a stream. Returns the manifest actually written.
pub fn write_stream(
    dir: &Path,
    identity: &StreamIdentity,
    observations: &[PositionObservation],
) -> Result<StreamManifest, StreamError> {
    std::fs::create_dir_all(dir).map_err(io)?;

    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    for o in observations {
        let line = serde_json::to_vec(o).map_err(|e| StreamError::Encoding(e.to_string()))?;
        gz.write_all(&line).map_err(io)?;
        gz.write_all(b"\n").map_err(io)?;
    }
    let body = gz.finish().map_err(io)?;

    let manifest = StreamManifest {
        format: STREAM_FORMAT.into(),
        source_identity: identity.source_identity.clone(),
        candidate_identity: identity.candidate_identity.clone(),
        protocol_identity: identity.protocol_identity.clone(),
        code_identity: identity.code_identity.clone(),
        bank_identity: identity.bank_identity.clone(),
        draw_identity: identity.draw_identity.clone(),
        sequences: identity.sequences,
        positions_per_sequence: identity.positions_per_sequence,
        observations: observations.len() as u64,
        stream_sha256: hash_bytes(&body),
    };

    std::fs::write(dir.join(BODY), &body).map_err(io)?;
    std::fs::write(
        dir.join(MANIFEST),
        serde_json::to_vec_pretty(&manifest).map_err(|e| StreamError::Encoding(e.to_string()))?,
    )
    .map_err(io)?;
    Ok(manifest)
}

/// Read a stream back, verifying its hash and count before returning it.
///
/// Verification is not optional. A stream that does not hash to its
/// manifest is not evidence, and returning its observations anyway would
/// let a silently altered body reach a bank.
pub fn read_stream(dir: &Path) -> Result<(StreamManifest, Vec<PositionObservation>), StreamError> {
    let manifest: StreamManifest =
        serde_json::from_slice(&std::fs::read(dir.join(MANIFEST)).map_err(io)?)
            .map_err(|e| StreamError::Encoding(e.to_string()))?;
    if manifest.format != STREAM_FORMAT {
        return Err(StreamError::UnknownFormat(manifest.format));
    }

    let body = std::fs::read(dir.join(BODY)).map_err(io)?;
    let found = hash_bytes(&body);
    if found != manifest.stream_sha256 {
        return Err(StreamError::Tampered {
            expected: manifest.stream_sha256,
            found,
        });
    }

    let mut observations = Vec::with_capacity(manifest.observations as usize);
    for line in BufReader::new(GzDecoder::new(&body[..])).lines() {
        let line = line.map_err(io)?;
        if line.is_empty() {
            continue;
        }
        observations
            .push(serde_json::from_str(&line).map_err(|e| StreamError::Encoding(e.to_string()))?);
    }
    if observations.len() as u64 != manifest.observations {
        return Err(StreamError::CountMismatch {
            declared: manifest.observations,
            found: observations.len() as u64,
        });
    }
    Ok((manifest, observations))
}

/// The projection. Every depth result is this function over a subset.
pub fn rederive_bank(observations: &[PositionObservation]) -> QualityBank {
    let mut b = BankBuilder::new();
    for o in observations {
        b.observe(o);
    }
    b.finish()
}

/// Observations belonging to the first `sequences` whole sequences.
///
/// The ladder is over WHOLE SEQUENCES, never truncated ones: positions
/// inside a teacher-forced sequence are not identically distributed, so
/// a partial sequence is not a smaller sample of the same thing.
pub fn take_sequences(
    observations: &[PositionObservation],
    sequences: &[u32],
) -> Vec<PositionObservation> {
    observations
        .iter()
        .filter(|o| sequences.contains(&o.sequence))
        .cloned()
        .collect()
}

/// Where a run's stream lives under a root.
pub fn stream_dir(root: &Path, label: &str) -> PathBuf {
    root.join(format!("stream-{label}"))
}

#[cfg(test)]
#[path = "observation_stream_tests.rs"]
mod tests;
