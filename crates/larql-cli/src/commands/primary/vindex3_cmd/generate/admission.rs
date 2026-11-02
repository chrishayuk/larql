//! MEM-002 I2a: the cold-binding admission decision, and the evidence
//! files that must exist before it is made.
//!
//! The decision is a pure function of the binding observation's summary
//! and the declared threshold. Nothing timed is an input — the signature
//! has no place for latency — so latency cannot influence which attempts
//! are admitted. It is computed once, recorded before any timed region
//! after binding, never revised, and it never stops the run: a refused
//! attempt is still executed, timed and recorded.
//!
//! There is no default threshold. Without one the outcome is
//! `witness_only`, never `admitted`; the number comes from a frozen
//! protocol, named in `--admission-basis`.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::opplan::exec::prepared::MappedResidency;

/// The gate's verdict on one binding observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum AdmissionOutcome {
    /// The observation is not a complete witness: a region failed, no
    /// region was observed, or the counts are inconsistent. Refused — an
    /// incomplete observation never stands for a cold image.
    WitnessFailed,
    /// Complete, and the resident total exceeds the declared threshold.
    AboveThreshold,
    /// Complete, and the resident total is at or below the threshold.
    Admitted,
    /// Complete, and no threshold was declared: calibration and
    /// development attempts.
    WitnessOnly,
}

impl AdmissionOutcome {
    pub(in super::super) fn name(self) -> &'static str {
        match self {
            Self::WitnessFailed => "witness_failed",
            Self::AboveThreshold => "above_threshold",
            Self::Admitted => "admitted",
            Self::WitnessOnly => "witness_only",
        }
    }
}

/// The counting convention the resident total uses, as records name it:
/// the sum of each read region's resident bytes.
pub(in super::super) const COUNTING: &str = "per_region_sum";

/// A threshold and where it came from, both supplied by the operator and
/// recorded verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct AdmissionThreshold {
    pub(in super::super) bytes: u64,
    /// The frozen protocol section and commit the number came from.
    pub(in super::super) basis: String,
}

/// One attempt's decision and the facts it was made from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct AdmissionDecision {
    pub(in super::super) outcome: AdmissionOutcome,
    /// Whether the observation was a complete witness.
    pub(in super::super) complete: bool,
    pub(in super::super) consistent: bool,
    pub(in super::super) regions: usize,
    pub(in super::super) failed: usize,
    /// The whole-image resident total — `None` unless complete, so an
    /// incomplete observation can never read as a zero-byte cold image.
    pub(in super::super) resident_bytes: Option<u64>,
    pub(in super::super) threshold: Option<AdmissionThreshold>,
}

/// Decide admission from the binding observation's summary alone.
pub(in super::super) fn decide(
    summary: &MappedResidency,
    threshold: Option<&AdmissionThreshold>,
) -> AdmissionDecision {
    let resident_bytes = summary.complete_resident_bytes();
    let outcome = match (resident_bytes, threshold) {
        (None, _) => AdmissionOutcome::WitnessFailed,
        (Some(_), None) => AdmissionOutcome::WitnessOnly,
        (Some(total), Some(t)) if total > t.bytes => AdmissionOutcome::AboveThreshold,
        (Some(_), Some(_)) => AdmissionOutcome::Admitted,
    };
    AdmissionDecision {
        outcome,
        complete: resident_bytes.is_some(),
        consistent: summary.is_consistent(),
        regions: summary.regions,
        failed: summary.failed_regions,
        resident_bytes,
        threshold: threshold.cloned(),
    }
}

/// The console line for a decision.
pub(in super::super) fn admission_line(decision: &AdmissionDecision) -> String {
    let resident = match decision.resident_bytes {
        Some(bytes) => format!("{:.3} GB", bytes as f64 / super::GB),
        None => "none (incomplete witness)".to_string(),
    };
    let threshold = match &decision.threshold {
        Some(t) => format!("{:.3} GB", t.bytes as f64 / super::GB),
        None => "none".to_string(),
    };
    format!(
        "admission: {} (complete {}, resident {resident} over {} regions, {} failed, threshold {threshold})",
        decision.outcome.name(),
        decision.complete,
        decision.regions,
        decision.failed,
    )
}

/// Ordinary diagnostic execution, or a MEM-002 campaign attempt — whose
/// evidence is mandatory and whose latency is sealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum RunMode {
    Diagnostic,
    Campaign,
}

impl RunMode {
    pub(in super::super) fn name(self) -> &'static str {
        match self {
            Self::Diagnostic => "diagnostic",
            Self::Campaign => "campaign",
        }
    }
}

/// What the operator asked the curve to record, before it is checked.
#[derive(Debug, Clone, Copy, Default)]
pub(in super::super) struct EvidenceRequest<'a> {
    pub(in super::super) campaign: bool,
    pub(in super::super) regions: Option<&'a Path>,
    pub(in super::super) sealed_latency: Option<&'a Path>,
    pub(in super::super) threshold_bytes: Option<u64>,
    pub(in super::super) basis: Option<&'a str>,
}

/// A checked request: the mode, the threshold if one was declared, and
/// the evidence paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct EvidencePlan {
    pub(in super::super) mode: RunMode,
    pub(in super::super) threshold: Option<AdmissionThreshold>,
    pub(in super::super) regions: Option<PathBuf>,
    pub(in super::super) sealed_latency: Option<PathBuf>,
}

/// Check a request before anything is bound. A campaign attempt needs
/// both evidence files; a threshold needs a campaign and its basis; two
/// evidence files may never be one file.
pub(in super::super) fn plan_evidence(
    request: EvidenceRequest<'_>,
) -> Result<EvidencePlan, String> {
    let threshold = match (request.threshold_bytes, request.basis) {
        (None, None) => None,
        (Some(_), None) => {
            return Err("--admission-threshold-bytes needs --admission-basis: \
                        a threshold without its source cannot be audited"
                .into())
        }
        (None, Some(_)) => {
            return Err("--admission-basis names the source of a threshold, \
                        and no --admission-threshold-bytes was given"
                .into())
        }
        (Some(_), Some(basis)) if basis.trim().is_empty() => {
            return Err("--admission-basis is blank: name the frozen protocol \
                        section and commit the threshold came from"
                .into())
        }
        (Some(bytes), Some(basis)) => Some(AdmissionThreshold {
            bytes,
            basis: basis.to_string(),
        }),
    };
    if threshold.is_some() && !request.campaign {
        return Err(
            "--admission-threshold-bytes scores a MEM-002 campaign attempt; \
                    pass --residency-campaign"
                .into(),
        );
    }
    if request.campaign {
        if request.regions.is_none() {
            return Err("a --residency-campaign attempt needs --residency-regions: \
                        an admission not written before the passes does not count"
                .into());
        }
        if request.sealed_latency.is_none() {
            return Err("a --residency-campaign attempt needs --sealed-latency: \
                        every campaign attempt's latency is sealed, admitted or not"
                .into());
        }
    }
    if let (Some(a), Some(b)) = (request.regions, request.sealed_latency) {
        if same_path(a, b) {
            return Err(format!(
                "--residency-regions and --sealed-latency are one file ({}): \
                 the sealed latency must stay out of the residency stream",
                a.display()
            ));
        }
    }
    Ok(EvidencePlan {
        mode: if request.campaign {
            RunMode::Campaign
        } else {
            RunMode::Diagnostic
        },
        threshold,
        regions: request.regions.map(Path::to_path_buf),
        sealed_latency: request.sealed_latency.map(Path::to_path_buf),
    })
}

/// Two evidence paths name one file — compared as paths, made absolute
/// first. Aliases this cannot see (a link, `..`) still cannot share a
/// file: each is created exclusively, so the second creation fails.
fn same_path(a: &Path, b: &Path) -> bool {
    match (std::path::absolute(a), std::path::absolute(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Create an evidence file that did not exist. An existing file is
/// refused and left untouched: evidence is never silently truncated.
pub(in super::super) fn create_evidence_file(path: &Path, what: &str) -> Result<File, String> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::AlreadyExists => format!(
                "{what} {} already exists; evidence files are never overwritten — choose a new path",
                path.display()
            ),
            _ => format!("cannot create {what} {}: {e}", path.display()),
        })
}

/// Proof that the admission decision is on record. The passes take one,
/// so no timed region can run before the record is written.
#[derive(Debug, PartialEq, Eq)]
pub(in super::super) enum AdmissionCommitted {
    /// Written and flushed to the residency stream.
    Recorded,
    /// Diagnostic execution with no stream: decided and printed only.
    NotRecorded,
}

/// The residency stream and the file under it, so a commit can reach
/// the disk and not only the process's buffer.
pub(in super::super) trait EvidenceSink: Write {
    fn sync(&mut self) -> std::io::Result<()>;
}

impl EvidenceSink for std::io::BufWriter<File> {
    fn sync(&mut self) -> std::io::Result<()> {
        self.flush()?;
        self.get_ref().sync_data()
    }
}

/// Write the admission record, then flush and sync everything the
/// stream holds — the binding observation included. Any failure is the
/// attempt's failure: the passes do not run. A campaign attempt with no
/// stream is refused here too, not only by [`plan_evidence`].
pub(in super::super) fn commit_admission(
    mode: RunMode,
    stream: Option<&mut dyn EvidenceSink>,
    record: &serde_json::Value,
) -> Result<AdmissionCommitted, String> {
    let Some(out) = stream else {
        return match mode {
            RunMode::Diagnostic => Ok(AdmissionCommitted::NotRecorded),
            RunMode::Campaign => {
                Err("a campaign attempt has no residency stream to record its admission".into())
            }
        };
    };
    let written = serde_json::to_writer(&mut *out, record)
        .map_err(std::io::Error::from)
        .and_then(|()| out.write_all(b"\n"))
        .and_then(|()| out.sync());
    written
        .map(|()| AdmissionCommitted::Recorded)
        .map_err(|e| format!("the admission record could not be written before the passes: {e}"))
}
