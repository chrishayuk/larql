//! MEM-002 I2b: what a run can say about the files it executed from.
//!
//! The facts come from the store's retained descriptors. This module
//! compares them with each other and, when one is supplied, with a
//! verification manifest made by another tool: it reads that evidence and
//! never makes it. A payload is never hashed here.
//!
//! Three things stay separate, and only the last two are addressed:
//! 1. what the plan was built from — **not bound**, see [`scope_block`];
//! 2. which opened files execution read — the retained descriptors;
//! 3. whether those files held the expected bytes — only as far as metadata
//!    can detect a change, and as far as the manifest's own verification
//!    goes.
//!
//! The diagnostic and the evidence class here are the run's own reading,
//! kept beside the per-segment facts. The evaluator recomputes them and
//! independently rejects invalid and unverified attempts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use larql_vindex::format::vindex3::opplan::exec::operands::{FileIdentity, StoreInstance};
use serde::Deserialize;
use serde_json::{json, Value};

use super::admission::RunMode;
use super::residency_record::{point_clock, wall_seconds, RunIdentity, SCHEMA};

/// The only manifest version this build reads.
pub(in super::super) const MANIFEST_SCHEMA: &str = "larql.verification-manifest/1";
/// A run is compared with the manifest made before the campaign.
pub(in super::super) const MANIFEST_PHASE: &str = "pre";
/// The verification route that re-hashes payloads; a legacy-bank container
/// gets structural checks only, which verify no bytes.
pub(in super::super) const MANIFEST_PATH_TAKEN: &str = "graph";
/// What a run's records call the single-descriptor binding, so bind times
/// from before it are never read as the same implementation.
pub(in super::super) const BINDING_IMPLEMENTATION: &str = "single_descriptor_i2b";

/// The identity a manifest entry records for one verified segment. The
/// device id is present in the format and deliberately not read.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(in super::super) struct ManifestIdentity {
    pub(in super::super) volume_uuid: Option<String>,
    pub(in super::super) inode: u64,
    pub(in super::super) size: u64,
    pub(in super::super) mtime_ns: i64,
    pub(in super::super) ctime_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(in super::super) struct ManifestSegment {
    pub(in super::super) declared: String,
    pub(in super::super) segment_sha256_declared: String,
    pub(in super::super) segment_sha256_computed: String,
    pub(in super::super) identity: ManifestIdentity,
}

/// Every field the format requires is required here, read or not: a file
/// missing one is not the record a verification tool writes.
#[derive(Deserialize)]
struct ManifestContainer {
    #[serde(rename = "path")]
    _path: String,
    // A `Value`, because a missing `Option` would read as null: the field
    // may be null where a platform has no UUID, and must be present.
    #[serde(rename = "volume_uuid")]
    _volume_uuid: Value,
    index_json_sha256: String,
}

#[derive(Deserialize)]
struct ManifestVerifier {
    #[serde(rename = "command")]
    _command: String,
    #[serde(rename = "binary")]
    _binary: Value,
    path_taken: String,
}

#[derive(Deserialize)]
struct ManifestFile {
    schema: String,
    phase: String,
    container: ManifestContainer,
    #[serde(rename = "verified_wall")]
    _verified_wall: f64,
    #[serde(rename = "clock")]
    _clock: Value,
    verifier: ManifestVerifier,
    segments: Vec<ManifestSegment>,
}

/// A manifest as it was read: the digest is of exactly the bytes parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct VerificationManifest {
    pub(in super::super) path: PathBuf,
    pub(in super::super) sha256: String,
    pub(in super::super) index_json_sha256: String,
    pub(in super::super) segments: BTreeMap<String, ManifestSegment>,
}

impl VerificationManifest {
    /// Read and check a manifest. Anything this cannot read as a pre-campaign
    /// graph verification is refused: an operator who supplied evidence is
    /// not silently downgraded to having none.
    pub(in super::super) fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read the manifest {}: {e}", path.display()))?;
        Self::parse(path, &bytes)
    }

    pub(in super::super) fn parse(path: &Path, bytes: &[u8]) -> Result<Self, String> {
        use sha2::{Digest, Sha256};
        let refuse = |why: String| format!("the manifest {} {why}", path.display());
        let file: ManifestFile = serde_json::from_slice(bytes)
            .map_err(|e| refuse(format!("is not a verification manifest: {e}")))?;
        if file.schema != MANIFEST_SCHEMA {
            return Err(refuse(format!(
                "has schema `{}`; this build reads `{MANIFEST_SCHEMA}`",
                file.schema
            )));
        }
        if file.phase != MANIFEST_PHASE {
            return Err(refuse(format!(
                "has phase `{}`; a run is compared with the `{MANIFEST_PHASE}` manifest",
                file.phase
            )));
        }
        if file.verifier.path_taken != MANIFEST_PATH_TAKEN {
            return Err(refuse(format!(
                "took verification path `{}`; only `{MANIFEST_PATH_TAKEN}` re-hashes payloads",
                file.verifier.path_taken
            )));
        }
        let mut segments = BTreeMap::new();
        for segment in file.segments {
            let declared = segment.declared.clone();
            if segments.insert(declared.clone(), segment).is_some() {
                return Err(refuse(format!("lists segment `{declared}` twice")));
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            sha256: Sha256::digest(bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
            index_json_sha256: file.container.index_json_sha256,
            segments,
        })
    }
}

/// One segment against the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) enum ManifestComparison {
    /// No manifest was supplied.
    NotSupplied,
    /// The key fields are equal, and the manifest's computed hash, its
    /// declared hash and the container index's hash all agree. Change
    /// detection, not proof of these bytes.
    Match,
    /// Named fields differ, or cannot be compared.
    Mismatch(Vec<&'static str>),
    /// The manifest lists no such segment.
    Absent,
}

impl ManifestComparison {
    fn name(&self) -> &'static str {
        match self {
            Self::NotSupplied => "not_supplied",
            Self::Match => "match",
            Self::Mismatch(_) => "mismatch",
            Self::Absent => "absent",
        }
    }
}

/// Compare one segment's descriptor identity with its manifest entry. The
/// key is volume UUID, inode, size, mtime and ctime — never the device id,
/// and not the link count, which the manifest's format does not carry.
///
/// A null volume UUID on either side cannot establish the same volume, so
/// it is a mismatch: a platform with no UUID cannot satisfy a manifest.
pub(in super::super) fn compare_with_manifest(
    declared: &str,
    index_sha256: &str,
    identity: &FileIdentity,
    manifest: &VerificationManifest,
) -> ManifestComparison {
    let Some(entry) = manifest.segments.get(declared) else {
        return ManifestComparison::Absent;
    };
    let same_volume = matches!(
        (&identity.volume_uuid, &entry.identity.volume_uuid),
        (Some(a), Some(b)) if a == b
    );
    let same_hash = !entry.segment_sha256_computed.is_empty()
        && entry.segment_sha256_computed == entry.segment_sha256_declared
        && entry.segment_sha256_declared == index_sha256;
    let differing: Vec<&'static str> = [
        (!same_volume, "volume_uuid"),
        (identity.inode != entry.identity.inode, "inode"),
        (identity.size != entry.identity.size, "size"),
        (identity.mtime_ns != entry.identity.mtime_ns, "mtime_ns"),
        (identity.ctime_ns != entry.identity.ctime_ns, "ctime_ns"),
        (!same_hash, "segment_sha256"),
    ]
    .into_iter()
    .filter_map(|(differs, name)| differs.then_some(name))
    .collect();
    if differing.is_empty() {
        ManifestComparison::Match
    } else {
        ManifestComparison::Mismatch(differing)
    }
}

/// The run's overall reading of one moment's facts, worst first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum IdentityDiagnostic {
    /// An identity could not be read, or no segment was opened at all.
    CaptureFailed,
    /// A retained descriptor's identity differs from when it was opened.
    Changed,
    /// The facts disagree with the supplied manifest.
    Mismatch,
    /// Nothing changed, and no manifest was supplied to compare with.
    Unverified,
    /// Nothing changed, and every selected segment matched the manifest.
    ConsistentWithManifest,
}

impl IdentityDiagnostic {
    pub(in super::super) fn name(self) -> &'static str {
        match self {
            Self::CaptureFailed => "capture_failed",
            Self::Changed => "changed",
            Self::Mismatch => "mismatch",
            Self::Unverified => "unverified",
            Self::ConsistentWithManifest => "consistent_with_manifest",
        }
    }
}

/// How an attempt's evidence may be used, as far as this run can tell. The
/// evaluator still decides, on its own reading of the facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in super::super) enum EvidenceClass {
    /// Diagnostic execution: never campaign evidence, whatever its facts.
    Diagnostic,
    /// A campaign attempt with no manifest. It may run, and it cannot
    /// become a qualified observation by succeeding.
    Unverified,
    /// A campaign attempt whose identity failed, or that ended before any
    /// identity was judged. Excluded.
    Invalid,
    /// A campaign attempt whose monitored facts agree with its manifest.
    /// Subject to the stated limits and to the evaluator: not "verified".
    IdentityConsistent,
}

impl EvidenceClass {
    pub(in super::super) fn name(self) -> &'static str {
        match self {
            Self::Diagnostic => "diagnostic",
            Self::Unverified => "unverified",
            Self::Invalid => "invalid",
            Self::IdentityConsistent => "identity_consistent",
        }
    }
}

/// What a run may conclude from one diagnostic.
pub(in super::super) fn evidence_class(
    mode: RunMode,
    diagnostic: IdentityDiagnostic,
) -> EvidenceClass {
    match (mode, diagnostic) {
        (RunMode::Diagnostic, _) => EvidenceClass::Diagnostic,
        (RunMode::Campaign, IdentityDiagnostic::ConsistentWithManifest) => {
            EvidenceClass::IdentityConsistent
        }
        (RunMode::Campaign, IdentityDiagnostic::Unverified) => EvidenceClass::Unverified,
        (RunMode::Campaign, _) => EvidenceClass::Invalid,
    }
}

/// One moment's facts, judged. The diagnostic is the worst category found;
/// the reasons name every problem in every category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct InstanceJudgement {
    pub(in super::super) diagnostic: IdentityDiagnostic,
    pub(in super::super) class: EvidenceClass,
    pub(in super::super) reasons: Vec<String>,
}

impl InstanceJudgement {
    /// A campaign attempt with this judgement must not carry on.
    pub(in super::super) fn invalid(&self) -> bool {
        self.class == EvidenceClass::Invalid
    }

    /// Why, as one line.
    pub(in super::super) fn reason(&self) -> String {
        format!(
            "instance identity {}: {}",
            self.diagnostic.name(),
            self.reasons.join("; ")
        )
    }
}

/// Judge one moment's facts. `index_json_sha256` is the run's by-path digest
/// of `index.json`: it can catch a different container, and it binds nothing.
pub(in super::super) fn judge(
    mode: RunMode,
    report: &StoreInstance,
    manifest: Option<&VerificationManifest>,
    index_json_sha256: Option<&str>,
) -> InstanceJudgement {
    let mut captures = Vec::new();
    let mut changes = Vec::new();
    let mut mismatches = Vec::new();
    if report.segments.is_empty() {
        captures.push("the store opened no segment, so nothing is bound".to_string());
    }
    for segment in &report.segments {
        if let Some(failure) = segment.capture_failure() {
            captures.push(format!("`{}`: {failure}", segment.declared));
            continue;
        }
        let changed = segment.changed_fields();
        if !changed.is_empty() {
            changes.push(format!(
                "`{}` changed since it was opened ({})",
                segment.declared,
                changed.join(", ")
            ));
        }
        if let (Some(manifest), Ok(now)) = (manifest, &segment.now) {
            match compare_with_manifest(&segment.declared, &segment.index_sha256, now, manifest) {
                ManifestComparison::Absent => {
                    mismatches.push(format!("`{}` is not in the manifest", segment.declared));
                }
                ManifestComparison::Mismatch(fields) => mismatches.push(format!(
                    "`{}` disagrees with the manifest ({})",
                    segment.declared,
                    fields.join(", ")
                )),
                ManifestComparison::Match | ManifestComparison::NotSupplied => {}
            }
        }
    }
    if let Some(manifest) = manifest {
        if index_json_sha256 != Some(manifest.index_json_sha256.as_str()) {
            mismatches.push(
                "the container's index.json digest disagrees with the manifest's".to_string(),
            );
        }
    }
    let diagnostic = if !captures.is_empty() {
        IdentityDiagnostic::CaptureFailed
    } else if !changes.is_empty() {
        IdentityDiagnostic::Changed
    } else if !mismatches.is_empty() {
        IdentityDiagnostic::Mismatch
    } else if manifest.is_none() {
        IdentityDiagnostic::Unverified
    } else {
        IdentityDiagnostic::ConsistentWithManifest
    };
    let mut reasons: Vec<String> = captures
        .into_iter()
        .chain(changes)
        .chain(mismatches)
        .collect();
    if manifest.is_none() {
        reasons.push("no verification manifest was supplied".to_string());
    }
    InstanceJudgement {
        diagnostic,
        class: evidence_class(mode, diagnostic),
        reasons,
    }
}

fn identity_json(identity: &Result<FileIdentity, String>) -> Value {
    match identity {
        Ok(i) => json!({
            "platform": i.platform,
            "volume_uuid": i.volume_uuid,
            "volume_uuid_source": i.volume_uuid_source,
            "inode": i.inode,
            "size": i.size,
            "mtime_ns": i.mtime_ns,
            "ctime_ns": i.ctime_ns,
            "nlink": i.nlink,
            "device": i.device,
        }),
        Err(error) => json!({ "platform": std::env::consts::OS, "error": error }),
    }
}

/// What a record's facts cover and what they do not, in the record itself
/// so no reader has to know it from elsewhere.
pub(in super::super) fn scope_block() -> Value {
    json!({
        "monitored": "selected segment files, mapped and copied, read through the descriptors \
                      retained from open; this associates the facts with the descriptors \
                      execution used",
        "not_bound": [
            "index.json",
            "system_graph.json",
            "plan_time_segment_headers",
            "auxiliary_reference_table",
            "representation_attestations",
            "other_pre_store_by_path_reads",
        ],
        "plan_to_bind": "selected headers are re-read through the retained descriptors; that does \
                         not show the planner read identical bytes, so this record does not \
                         establish complete verification-to-execution identity",
        "not_recorded_here": "plugins and lowerings loaded in-process are outside these facts",
        "content": "metadata detects change and does not prove content: a write can be missed, \
                    identity can be restored, and an extended attribute or a hard link moves the \
                    ctime with no content change; writable mappers are forbidden by protocol and \
                    not enforced here; only payload verification addresses content",
        "distributed_workers": "unsupported for verified campaign evidence: worker reads use a \
                                retained descriptor, but no evidence path records their identity",
        "volume_uuid": "null where the platform provides none; a null cannot match a manifest",
    })
}

/// The `instance_identity` block of a record: every selected segment's
/// facts at this moment, compared, with the scope they stand within.
pub(in super::super) fn instance_block(
    report: &StoreInstance,
    manifest: Option<&VerificationManifest>,
    judgement: &InstanceJudgement,
) -> Value {
    let segments: Vec<Value> = report
        .segments
        .iter()
        .map(|segment| {
            let comparison = match (manifest, &segment.now) {
                (None, _) => ManifestComparison::NotSupplied,
                (Some(manifest), Ok(now)) => {
                    compare_with_manifest(&segment.declared, &segment.index_sha256, now, manifest)
                }
                (Some(_), Err(_)) => ManifestComparison::Mismatch(vec!["capture"]),
            };
            let fields: &[&'static str] = match &comparison {
                ManifestComparison::Mismatch(fields) => fields,
                _ => &[],
            };
            json!({
                "object": segment.object,
                "declared": segment.declared,
                "path": segment.path.display().to_string(),
                "use": { "mapped": segment.mapped, "copied": segment.copied },
                "at_open": identity_json(&segment.at_open),
                "now": identity_json(&segment.now),
                "changed_fields": segment.changed_fields(),
                "manifest": comparison.name(),
                "manifest_mismatch_fields": fields,
            })
        })
        .collect();
    json!({
        "identity_diagnostic": judgement.diagnostic.name(),
        "reasons": judgement.reasons,
        "manifest": manifest.map(|m| json!({
            "path": m.path.display().to_string(),
            "sha256": m.sha256,
        })),
        "segments": segments,
        "absent_objects": report.absent_objects,
        "counters": {
            "load_count": report.load_count,
            "bytes_read": report.bytes_read,
            "mapped_regions": report.mapped_regions,
            "mapped_objects": report.mapped_objects,
        },
        "scope": scope_block(),
    })
}

/// The identity record written before binding. **It carries no reading of
/// any clock.** The next record is the binding observation, whose bounds are
/// on the system-wide clock, so a time here would let any reader of the
/// stream subtract its way to the bind time that `--sealed-latency` seals.
pub(in super::super) fn before_binding_record(
    run: &RunIdentity,
    block: Value,
    class: EvidenceClass,
) -> Value {
    json!({
        "schema": SCHEMA,
        "record": "instance_identity",
        "run_id": run.run_id,
        "phase": "before_binding",
        "evidence_class": class.name(),
        "instance_identity": block,
    })
}

/// The identity record written after the run, timed like the after-run
/// observation beside it.
pub(in super::super) fn after_run_record(
    run: &RunIdentity,
    block: Value,
    class: EvidenceClass,
) -> Value {
    json!({
        "schema": SCHEMA,
        "record": "instance_identity",
        "run_id": run.run_id,
        "phase": "after_run",
        "evidence_class": class.name(),
        "instance_identity": block,
        "clock": point_clock(),
        "wall": wall_seconds(SystemTime::now()),
    })
}

/// How far the store's counters moved after admission. Diagnostic evidence
/// only: it shows a later read through the store, and nothing about any
/// reader outside it.
pub(in super::super) fn post_admission_reads(
    at_admission: &StoreInstance,
    after: &StoreInstance,
) -> Value {
    json!({
        "load_count": after.load_count.saturating_sub(at_admission.load_count),
        "bytes_read": after.bytes_read.saturating_sub(at_admission.bytes_read),
        "mapped_regions": after.mapped_regions.saturating_sub(at_admission.mapped_regions),
        "mapped_objects": after.mapped_objects.saturating_sub(at_admission.mapped_objects),
    })
}

/// The console's line for one moment's reading.
pub(in super::super) fn instance_line(phase: &str, judgement: &InstanceJudgement) -> String {
    let mut line = format!(
        "instance identity ({phase}): {}, evidence class {}",
        judgement.diagnostic.name(),
        judgement.class.name()
    );
    if !judgement.reasons.is_empty() {
        line.push_str(" — ");
        line.push_str(&judgement.reasons.join("; "));
    }
    line
}
