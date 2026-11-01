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
    MappedResidency, ResidencyDelta, ResidencyObservation,
};
use serde_json::{json, Value};

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
/// A key that survives cloning is not permission to compare. Region
/// records from two observations are comparable only when both headers
/// carry equal, non-null `container.index_json_sha256`; the procedure
/// that pairs them must check it, and a null digest refuses pairing.
pub(in super::super) const SCHEMA: &str = "larql.residency-observation/2";

/// The build-time variable a measured build records its commit in. Set it
/// when building a binary whose records are evidence, e.g.
/// `LARQL_BUILD_COMMIT=$(git rev-parse HEAD) cargo build --release`.
pub(in super::super) const BUILD_COMMIT_VAR: &str = "LARQL_BUILD_COMMIT";

/// The commit this binary was built from, when the build declared one.
/// Never derived from the working tree at run time: a commit is recorded
/// only if the build said so.
const BUILD_COMMIT: Option<&str> = option_env!("LARQL_BUILD_COMMIT");

/// The build's declared commit, with a blank declaration read as none.
pub(in super::super) fn declared_commit(declared: Option<&str>) -> Option<String> {
    declared
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .map(str::to_string)
}

/// Who wrote a record: enough to pair records from separate processes
/// and check they observed one container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) struct RunIdentity {
    /// Unique per process run.
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
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self {
            run_id: format!("{pid}-{nanos}"),
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

fn wall_seconds(t: SystemTime) -> f64 {
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
        "binary": {
            "version": run.binary_version,
            "commit": run.binary_commit,
            "commit_source": if run.binary_commit.is_some() { BUILD_COMMIT_VAR } else { "unavailable" },
        },
        "container": {
            "path": run.container_path,
            "index_json_sha256": run.index_json_sha256,
        },
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
    out: &mut impl Write,
    run: &RunIdentity,
    observation_id: u64,
    phase: &str,
    observation: &ResidencyObservation,
) -> std::io::Result<()> {
    for record in observation_records(run, observation_id, phase, observation) {
        serde_json::to_writer(&mut *out, &record)?;
        out.write_all(b"\n")?;
    }
    Ok(())
}
