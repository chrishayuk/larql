//! **Replay: the persisted stream must reproduce the bank the run reported.**
//!
//! REAL-EVIDENCE-1 gate, applied to REAL artifacts. The unit gate in
//! `observation_stream` proves the recorder round-trips a fixture; this
//! proves that a specific stream on disk, read back with its hash and
//! count verified, projects to EXACTLY the `QualityBank` the run wrote
//! into its report at the time. If it does not, the stream and the
//! report describe different experiments and neither is evidence.
//!
//! Exactness is the claim: `QualityBank` equality, not a tolerance. The
//! report's bank went through JSON once; `f64` survives that verbatim.

use std::fmt;
use std::path::Path;

use serde::Serialize;

use super::observation_stream::{read_stream, rederive_bank, StreamError};
use super::quality::QualityBank;

/// The stream directory to replay.
pub const REPLAY_STREAM_ENV: &str = "LARQL_Q2A_REPLAY_STREAM";
/// The report whose `bank` the replay must reproduce.
pub const REPLAY_REPORT_ENV: &str = "LARQL_Q2A_REPLAY_REPORT";
/// Where a report keeps its bank and its position count.
pub const REPORT_BANK_KEY: &str = "bank";
pub const REPORT_POSITIONS_KEY: &str = "positions";
pub const REPORT_RUN_KEY: &str = "run";

#[derive(Debug)]
pub enum ReplayError {
    Stream(StreamError),
    Report(String),
    /// The replayed bank is not the reported one. Names every top-level
    /// bank field that differs, so the reader sees WHERE, not just that.
    Mismatch {
        observations: u64,
        differing: Vec<String>,
    },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stream(e) => write!(f, "replay: {e}"),
            Self::Report(e) => write!(f, "replay: report {e}"),
            Self::Mismatch {
                observations,
                differing,
            } => write!(
                f,
                "replay of {observations} observations does not reproduce the reported \
                 bank; differing: {}",
                differing.join(", ")
            ),
        }
    }
}

/// What an exact replay proves, for the record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayWitness {
    pub run: Option<String>,
    pub observations: u64,
    pub stream_sha256: String,
}

fn report_error<E: fmt::Display>(path: &Path, e: E) -> ReplayError {
    ReplayError::Report(format!("{}: {e}", path.display()))
}

/// Read the stream (hash and count verified), rederive its bank, and
/// compare it EXACTLY with the bank the report carries.
pub fn replay_matches_report(
    stream_dir: &Path,
    report_path: &Path,
) -> Result<ReplayWitness, ReplayError> {
    let (manifest, observations) = read_stream(stream_dir).map_err(ReplayError::Stream)?;
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(report_path).map_err(|e| report_error(report_path, e))?,
    )
    .map_err(|e| report_error(report_path, e))?;
    let reported: QualityBank = serde_json::from_value(report[REPORT_BANK_KEY].clone())
        .map_err(|e| report_error(report_path, format!("`{REPORT_BANK_KEY}`: {e}")))?;
    let replayed = rederive_bank(&observations);

    let mut differing = differing_fields(&replayed, &reported);
    let reported_positions = report[REPORT_POSITIONS_KEY].as_u64();
    if reported_positions != Some(manifest.observations) {
        differing.push(format!(
            "{REPORT_POSITIONS_KEY} (report {reported_positions:?}, stream {})",
            manifest.observations
        ));
    }
    if !differing.is_empty() {
        return Err(ReplayError::Mismatch {
            observations: manifest.observations,
            differing,
        });
    }
    Ok(ReplayWitness {
        run: report[REPORT_RUN_KEY].as_str().map(str::to_string),
        observations: manifest.observations,
        stream_sha256: manifest.stream_sha256,
    })
}

/// Top-level bank fields whose values differ — a diff a reader can act
/// on. Equality itself is the struct's own `PartialEq`; the JSON view is
/// only used to NAME what moved.
fn differing_fields(a: &QualityBank, b: &QualityBank) -> Vec<String> {
    if a == b {
        return Vec::new();
    }
    let (ja, jb) = (
        serde_json::to_value(a).expect("a bank serialises"),
        serde_json::to_value(b).expect("a bank serialises"),
    );
    let (ma, mb) = (
        ja.as_object().expect("a bank is an object"),
        jb.as_object().expect("a bank is an object"),
    );
    let mut keys: Vec<&String> = ma.keys().chain(mb.keys()).collect();
    keys.sort();
    keys.dedup();
    let mut out: Vec<String> = keys
        .into_iter()
        .filter(|k| ma.get(*k) != mb.get(*k))
        .cloned()
        .collect();
    if out.is_empty() {
        // Unequal by `PartialEq` yet identical as JSON: nothing a reader
        // could locate, so say so rather than return an empty diff.
        out.push("(unlocatable: unequal as structs, equal as JSON)".into());
    }
    out
}

#[cfg(test)]
mod tests;
