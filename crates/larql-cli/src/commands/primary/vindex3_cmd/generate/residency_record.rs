//! What the residency curve says about an observation: the console lines
//! and the `--residency-regions` JSON Lines record.
//!
//! Both are plain functions over a [`ResidencyObservation`], so what they
//! print for a failed region can be pinned on a constructed observation —
//! no fixture can make `mincore` fail on demand.

use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use larql_vindex::format::vindex3::opplan::exec::prepared::{
    boot_session, system_mono_basis, system_mono_ns, MappedResidency, ResidencyDelta,
    ResidencyObservation,
};
use serde_json::{json, Map, Value};

use super::admission::{AdmissionDecision, RunMode, COUNTING};
use super::instance_evidence::{EvidenceClass, BINDING_IMPLEMENTATION};
use super::sealed_latency::SealedDigest;
use super::GB;

/// The record's schema. A change to any field's meaning increments it.
///
/// `/2`: a region's `segment` is the container-relative path the
/// container declared (it was the absolute path in `/1`), the absolute
/// path moved to `segment_path`, `segment_declared` says whether the
/// container declared the segment (so the region's key survives cloning),
/// and the header carries
/// `binary.commit` and `consistent`.
///
/// `/4` (MEM-002 I2b): `instance_identity` on the admission record is now
/// the facts (it said `not_recorded`), with an `evidence_class` beside it
/// and on `attempt_end`; a new `instance_identity` record before binding (it
/// carries no clock, so the stream cannot date the bind) and after the run;
/// `representation_source` and `binding_implementation` on the admission
/// record, because a bind time from a binary before I2b is a different
/// implementation's. `attempt_end` carries a finished sealed file's digest.
///
/// On `attempt_end`, `status` says whether the attempt completed and
/// `evidence_class` is its identity's last reading; both must allow an
/// attempt's use. `invalid` means its identity failed, or that it ended
/// before any was judged. An attempt that ended with `status: error` is not
/// usable whatever its class. `above_threshold` is neither: it is an
/// ordinary rejection of usable evidence.
///
/// `/3` (MEM-002 I2a): a new `admission` record, written and flushed
/// after the binding observation and before any pass; a closing
/// `attempt_end` record; a `clock` block on every record, on the
/// system-wide basis it names (`start_system_mono_ns`/`end_system_mono_ns`
/// on an observation, `system_mono_ns` on a point record); and, on the
/// `after_run` header of a sealed run, the sealed-latency file's digest.
/// `started_mono_ns`/`ended_mono_ns` stay, process-relative and never
/// comparable across processes. A reader refuses a version it does not
/// know.
///
/// A key that survives cloning is not permission to compare. Region
/// records from two observations are comparable only when both headers
/// carry equal, non-null `container.index_json_sha256`; the procedure
/// that pairs them must check it, and a null digest refuses pairing.
pub(in super::super) const SCHEMA: &str = "larql.residency-observation/4";

/// The build-time variable a measured build records its commit in. Set it
/// when building a binary whose records are evidence, e.g.
/// `LARQL_BUILD_COMMIT=$(git rev-parse HEAD) cargo build --release`.
pub(in super::super) const BUILD_COMMIT_VAR: &str = "LARQL_BUILD_COMMIT";

/// The commit this binary was built from, when the build declared one.
/// Never derived from the working tree at run time: a commit is recorded
/// only if the build said so.
const BUILD_COMMIT: Option<&str> = option_env!("LARQL_BUILD_COMMIT");

/// This binary's declared commit, if the build declared one.
pub(in super::super) fn build_commit() -> Option<String> {
    declared_commit(BUILD_COMMIT)
}

/// The build's declared commit, with a blank declaration read as none.
pub(in super::super) fn declared_commit(declared: Option<&str>) -> Option<String> {
    declared
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(str::to_string)
}

/// 64 bits with no relation to the clock. A run id made from the time
/// would date the attempt's start, before binding, and so give away the
/// bind time that `--sealed-latency` seals. The standard library seeds its
/// hasher keys from OS randomness and varies them per instance. That is
/// from its behaviour, not a documented guarantee, which is enough for an
/// id that is compared for equality and never trusted as a secret.
fn unpredictable_id() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

/// Who wrote a record: enough to pair records from separate processes
/// and check they observed one container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct RunIdentity {
    /// Unique per process run, and carrying no timestamp.
    pub(in super::super) run_id: String,
    pub(in super::super) pid: u32,
    pub(in super::super) binary_version: String,
    /// The commit the build declared through [`BUILD_COMMIT_VAR`], or
    /// `None` — recorded as unavailable, never guessed.
    pub(in super::super) binary_commit: Option<String>,
    pub(in super::super) container_path: String,
    /// SHA-256 of the container's `index.json`, when it could be read.
    pub(in super::super) index_json_sha256: Option<String>,
}

impl RunIdentity {
    /// This process, observing the container at `container`.
    pub(in super::super) fn for_container(container: &Path) -> Self {
        let pid = std::process::id();
        Self {
            run_id: format!("{pid}-{:016x}", unpredictable_id()),
            pid,
            binary_version: env!("CARGO_PKG_VERSION").to_string(),
            binary_commit: declared_commit(BUILD_COMMIT),
            container_path: container.display().to_string(),
            index_json_sha256: larql_vindex::format::checksums::sha256_file(
                &container.join("index.json"),
            )
            .ok(),
        }
    }
}

/// The binding line: address space, how many regions were read and how
/// many failed, and the resident bytes of the read regions only.
pub(in super::super) fn binding_line(summary: &MappedResidency) -> String {
    let line = format!(
        "  mapped                {:>10.2} GB address space over {} regions ({} read, {} failed), {:.3} GB resident in read regions",
        summary.mapped_bytes as f64 / GB,
        summary.regions,
        summary.read_regions(),
        summary.failed_regions,
        summary.resident_bytes as f64 / GB
    );
    if summary.is_consistent() {
        line
    } else {
        format!("{line} — INCONSISTENT COUNTS, not a witness")
    }
}

/// The line that must follow whenever any region failed: how many, and
/// the first one named. `None` when every region was read.
pub(in super::super) fn incomplete_line(observation: &ResidencyObservation) -> Option<String> {
    let failed = observation.summary().failed_regions;
    let (region, error) = observation.first_failure()?;
    let store = &observation.stores[region.store];
    Some(format!(
        "  WITNESS INCOMPLETE: {failed} region(s) could not be read (first: {}@{}: {error})",
        store.id, region.file_offset
    ))
}

/// A step's resident change, for the curve's table: the signed total in
/// GB, or `n/a` when no region was comparable — never a zero that was
/// not measured.
pub(in super::super) fn delta_cell(delta: &ResidencyDelta) -> String {
    match delta.signed_total() {
        Some(total) => format!("{:.3}", total as f64 / GB),
        None => "n/a".to_string(),
    }
}

/// The clock block of a record read at one instant: the basis, the boot
/// session, and one reading. Fields the target cannot supply are `null`.
pub(in super::super) fn point_clock() -> Value {
    json!({
        "basis": system_mono_basis(),
        "boot_session": boot_session(),
        "system_mono_ns": system_mono_ns(),
    })
}

/// The clock block of an observation: its scan's system-wide bounds.
fn observation_clock(observation: &ResidencyObservation) -> Value {
    json!({
        "basis": system_mono_basis(),
        "boot_session": boot_session(),
        "start_system_mono_ns": observation.started_system_mono_ns,
        "end_system_mono_ns": observation.ended_system_mono_ns,
    })
}

fn binary_block(run: &RunIdentity) -> Value {
    json!({
        "version": run.binary_version,
        "commit": run.binary_commit,
        "commit_source": if run.binary_commit.is_some() { BUILD_COMMIT_VAR } else { "unavailable" },
    })
}

fn container_block(run: &RunIdentity) -> Value {
    json!({
        "path": run.container_path,
        "index_json_sha256": run.index_json_sha256,
    })
}

/// The run's attempt facts the admission record carries beside the
/// decision. None of them is an input to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct AttemptFacts {
    pub(in super::super) mode: RunMode,
    pub(in super::super) warmup: usize,
    pub(in super::super) repeat: usize,
    /// `QUALIFIED`, or `UNQUALIFIED (…)`.
    pub(in super::super) qualification: String,
    pub(in super::super) disqualifiers: Vec<String>,
    pub(in super::super) unquiet_ok: bool,
    pub(in super::super) expert_access: String,
    /// `auto`, `stored`: a campaign refuses `transient`.
    pub(in super::super) representation_source: String,
    pub(in super::super) sealed_latency: Option<String>,
    /// The `instance_identity` block, at this moment.
    pub(in super::super) instance: Value,
    /// The `execution_identity` block: what was asked for, and what the
    /// store selected and the image pinned.
    pub(in super::super) execution: Value,
    pub(in super::super) evidence_class: EvidenceClass,
}

/// The admission record: the decision, the facts it was made from, and
/// the attempt's facts — written before any timed region after binding.
/// The resident total is `null` unless the witness was complete.
pub(in super::super) fn admission_record(
    run: &RunIdentity,
    binding_observation_id: u64,
    decision: &AdmissionDecision,
    facts: &AttemptFacts,
) -> Value {
    json!({
        "schema": SCHEMA,
        "record": "admission",
        "run_id": run.run_id,
        "pid": run.pid,
        "binary": binary_block(run),
        "container": container_block(run),
        "mode": facts.mode.name(),
        "binding_observation_id": binding_observation_id,
        "outcome": decision.outcome.name(),
        "complete": decision.complete,
        "consistent": decision.consistent,
        "regions": decision.regions,
        "failed": decision.failed,
        "resident_bytes": decision.resident_bytes,
        "counting": COUNTING,
        "threshold": decision.threshold.as_ref().map(|t| json!({
            "bytes": t.bytes,
            "basis": t.basis,
        })),
        "warmup": facts.warmup,
        "repeat": facts.repeat,
        "expert_access": facts.expert_access,
        "representation_source": facts.representation_source,
        "quiet": {
            "qualification": facts.qualification,
            "disqualifiers": facts.disqualifiers,
            "unquiet_ok": facts.unquiet_ok,
        },
        "sealed_latency": facts.sealed_latency.as_ref().map(|path| json!({ "path": path })),
        "binding_implementation": BINDING_IMPLEMENTATION,
        "evidence_class": facts.evidence_class.name(),
        "instance_identity": facts.instance,
        "execution_identity": facts.execution,
        "clock": point_clock(),
        "wall": wall_seconds(SystemTime::now()),
    })
}

/// The attempt's last record: whether the passes completed, and the error
/// if they did not. An admission record with no `attempt_end` after it is
/// an attempt that did not finish.
pub(in super::super) fn attempt_end_record(
    run: &RunIdentity,
    error: Option<&str>,
    class: EvidenceClass,
    sealed: Option<&SealedDigest>,
) -> Value {
    json!({
        "schema": SCHEMA,
        "record": "attempt_end",
        "run_id": run.run_id,
        "status": if error.is_some() { "error" } else { "completed" },
        "error": error,
        "evidence_class": class.name(),
        "sealed_latency": sealed.map(|d| json!({
            "path": std::path::absolute(&d.path)
                .unwrap_or_else(|_| d.path.clone())
                .display()
                .to_string(),
            "sha256": d.sha256,
        })),
        "clock": point_clock(),
        "wall": wall_seconds(SystemTime::now()),
    })
}

/// What a sealed run's `after_run` header adds: the sealed file's path
/// and digest.
pub(in super::super) fn sealed_header_fields(sealed: &SealedDigest) -> Map<String, Value> {
    let mut fields = Map::new();
    fields.insert(
        "sealed_latency".into(),
        json!({
            "path": sealed.path.display().to_string(),
            "sha256": sealed.sha256,
        }),
    );
    fields
}

pub(in super::super) fn wall_seconds(t: SystemTime) -> f64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// One observation as JSON Lines: a header record, then one record per
/// region in scan order.
pub(in super::super) fn observation_records(
    run: &RunIdentity,
    observation_id: u64,
    phase: &str,
    observation: &ResidencyObservation,
) -> Vec<Value> {
    let summary = observation.summary();
    let mut records = vec![json!({
        "schema": SCHEMA,
        "record": "observation",
        "run_id": run.run_id,
        "observation_id": observation_id,
        "phase": phase,
        "pid": run.pid,
        "binary": binary_block(run),
        "container": container_block(run),
        "started_wall": wall_seconds(observation.started_wall),
        "started_mono_ns": observation.started_mono_ns,
        "ended_wall": wall_seconds(observation.ended_wall),
        "ended_mono_ns": observation.ended_mono_ns,
        "page_size": observation.page_size,
        "regions": summary.regions,
        "read": summary.read_regions(),
        "failed": summary.failed_regions,
        "mapped_bytes_all": summary.mapped_bytes,
        "mapped_bytes_read": summary.mapped_bytes_read,
        "resident_bytes_read": summary.resident_bytes,
        "consistent": summary.is_consistent(),
        "clock": observation_clock(observation),
    })];
    for region in &observation.regions {
        let store = &observation.stores[region.store];
        let mut record = json!({
            "schema": SCHEMA,
            "record": "region",
            "run_id": run.run_id,
            "observation_id": observation_id,
            "scan_index": region.scan_index,
            "store_id": store.id,
            "segment": store.declared_segment,
            "segment_path": store.source_path.as_ref().map(|p| p.display().to_string()),
            "segment_declared": store.segment_identity().is_portable(),
            "file_offset": region.file_offset,
            "len": region.len,
        });
        let fields = record.as_object_mut().expect("a JSON object");
        match region.resident {
            Ok(bytes) => {
                fields.insert("status".into(), json!("ok"));
                fields.insert("resident_bytes".into(), json!(bytes));
            }
            Err(error) => {
                fields.insert("status".into(), json!("failed"));
                let mut detail = json!({ "kind": error.kind() });
                if let Some(errno) = error.errno() {
                    detail["errno"] = json!(errno);
                }
                fields.insert("error".into(), detail);
            }
        }
        records.push(record);
    }
    records
}

/// Write [`observation_records`] to `out`, one JSON value per line.
pub(in super::super) fn write_observation(
    out: &mut (impl Write + ?Sized),
    run: &RunIdentity,
    observation_id: u64,
    phase: &str,
    observation: &ResidencyObservation,
) -> std::io::Result<()> {
    write_observation_with(out, run, observation_id, phase, observation, Map::new())
}

/// [`write_observation`], with `header_fields` added to the header.
pub(in super::super) fn write_observation_with(
    out: &mut (impl Write + ?Sized),
    run: &RunIdentity,
    observation_id: u64,
    phase: &str,
    observation: &ResidencyObservation,
    header_fields: Map<String, Value>,
) -> std::io::Result<()> {
    let mut records = observation_records(run, observation_id, phase, observation);
    if let Some(header) = records.first_mut().and_then(Value::as_object_mut) {
        header.extend(header_fields);
    }
    for record in records {
        write_record(out, &record)?;
    }
    Ok(())
}

/// One record as one line.
pub(in super::super) fn write_record(
    out: &mut (impl Write + ?Sized),
    record: &Value,
) -> std::io::Result<()> {
    serde_json::to_writer(&mut *out, record)?;
    out.write_all(b"\n")
}
