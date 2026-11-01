//! What the residency curve prints and records about an observation,
//! pinned on observations built by hand: successful, failed, empty and
//! mixed regions. The end-to-end CLI run cannot cover this — its fixture
//! maps no region, and no fixture can make `mincore` fail.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use larql_vindex::format::vindex3::opplan::exec::prepared::{
    ObservedStore, RegionResidency, ResidencyDelta, ResidencyObservation,
};
use larql_vindex::format::vindex3::represent::physical::ResidencyReadError;
use serde_json::Value;

use larql_vindex::format::vindex3::opplan::exec::prepared::MappedResidency;

use super::super::generate::residency_record::{
    binding_line, declared_commit, delta_cell, incomplete_line, observation_records,
    write_observation, RunIdentity, BUILD_COMMIT_VAR, SCHEMA,
};

const LEN: u64 = 1 << 20;
const MINCORE_FAILED: ResidencyReadError = ResidencyReadError::Mincore { errno: 22 };
const SEGMENT: &str = "segments/target.expert_bank.bin";
const SEGMENT_PATH: &str = "/models/c.vindex3/segments/target.expert_bank.bin";

fn region(scan_index: usize, resident: Result<u64, ResidencyReadError>) -> RegionResidency {
    RegionResidency {
        scan_index,
        store: 0,
        file_offset: scan_index as u64 * LEN,
        len: LEN,
        resident,
    }
}

fn observation(regions: Vec<RegionResidency>, page_size: Option<u64>) -> ResidencyObservation {
    ResidencyObservation {
        started_wall: SystemTime::UNIX_EPOCH + Duration::from_secs(10),
        started_mono_ns: 5,
        ended_wall: SystemTime::UNIX_EPOCH + Duration::from_secs(11),
        ended_mono_ns: 9,
        page_size,
        stores: vec![ObservedStore {
            id: "target.expert_bank".to_string(),
            source_path: Some(PathBuf::from(SEGMENT_PATH)),
            declared_segment: Some(SEGMENT.to_string()),
        }],
        regions,
    }
}

fn run() -> RunIdentity {
    RunIdentity {
        run_id: "run-1".to_string(),
        pid: 42,
        binary_version: "0.0.0".to_string(),
        binary_commit: Some("abc123".to_string()),
        container_path: "/models/c.vindex3".to_string(),
        index_json_sha256: Some("ddd9".to_string()),
    }
}

#[test]
fn the_binding_line_names_read_and_failed_regions() {
    let mixed = observation(
        vec![region(0, Ok(4096)), region(1, Err(MINCORE_FAILED))],
        Some(16_384),
    );
    let line = binding_line(&mixed.summary());
    assert!(line.contains("over 2 regions (1 read, 1 failed)"), "{line}");
    assert!(line.contains("resident in read regions"), "{line}");
}

#[test]
fn an_incomplete_witness_is_said_exactly_when_a_region_failed() {
    let complete = observation(vec![region(0, Ok(0)), region(1, Ok(1))], Some(16_384));
    assert_eq!(incomplete_line(&complete), None);
    assert_eq!(
        incomplete_line(&observation(Vec::new(), Some(16_384))),
        None
    );

    let failed = observation(
        vec![
            region(0, Ok(0)),
            region(1, Err(MINCORE_FAILED)),
            region(2, Err(MINCORE_FAILED)),
        ],
        Some(16_384),
    );
    let line = incomplete_line(&failed).expect("failures must be named");
    assert!(line.contains("WITNESS INCOMPLETE: 2 region(s)"), "{line}");
    assert!(
        line.contains(&format!("target.expert_bank@{LEN}")),
        "first failure named: {line}"
    );
    assert!(line.contains("errno 22"), "{line}");
}

#[test]
fn a_delta_with_no_comparable_region_prints_na_never_zero() {
    let unmeasured = ResidencyDelta {
        comparable: 0,
        increase: 0,
        decrease: 0,
        excluded: 3,
    };
    assert_eq!(delta_cell(&unmeasured), "n/a");
    let measured = ResidencyDelta {
        comparable: 2,
        increase: 2_000_000_000,
        decrease: 500_000_000,
        excluded: 0,
    };
    assert_eq!(delta_cell(&measured), "1.500");
    let fell = ResidencyDelta {
        comparable: 1,
        increase: 0,
        decrease: 250_000_000,
        excluded: 0,
    };
    assert_eq!(delta_cell(&fell), "-0.250");
}

#[test]
fn every_record_carries_the_schema_and_a_header_precedes_the_regions() {
    let obs = observation(
        vec![
            region(0, Ok(4096)),
            region(1, Err(MINCORE_FAILED)),
            region(2, Ok(0)),
        ],
        Some(16_384),
    );
    let records = observation_records(&run(), 3, "binding", &obs);
    assert_eq!(records.len(), 1 + obs.regions.len());
    for r in &records {
        assert_eq!(r["schema"], SCHEMA);
        assert_eq!(r["run_id"], "run-1");
        assert_eq!(r["observation_id"], 3);
    }
    let header = &records[0];
    assert_eq!(header["record"], "observation");
    assert_eq!(header["phase"], "binding");
    assert_eq!(header["pid"], 42);
    assert_eq!(header["container"]["index_json_sha256"], "ddd9");
    assert_eq!(header["page_size"], 16_384);
    assert_eq!(header["regions"], 3);
    assert_eq!(header["read"], 2);
    assert_eq!(header["failed"], 1);
    assert_eq!(header["mapped_bytes_all"], 3 * LEN);
    assert_eq!(header["mapped_bytes_read"], 2 * LEN);
    assert_eq!(header["resident_bytes_read"], 4096);
    assert_eq!(header["started_mono_ns"], 5);
    assert_eq!(header["started_wall"], 10.0);
    assert_eq!(header["binary"]["commit"], "abc123");
    assert_eq!(header["binary"]["commit_source"], BUILD_COMMIT_VAR);
    assert_eq!(header["consistent"], true);

    let ok = &records[1];
    assert_eq!(ok["record"], "region");
    assert_eq!(ok["status"], "ok");
    assert_eq!(ok["resident_bytes"], 4096);
    assert_eq!(ok["segment"], SEGMENT, "the container-relative path");
    assert_eq!(
        ok["segment_path"], SEGMENT_PATH,
        "the absolute path, diagnostic"
    );
    assert_eq!(ok["segment_declared"], true);
    assert_eq!(ok["file_offset"], 0);
    assert!(ok.get("error").is_none());

    let failed = &records[2];
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["kind"], "mincore");
    assert_eq!(failed["error"]["errno"], 22);
    assert!(
        failed.get("resident_bytes").is_none(),
        "a failure carries no count"
    );
}

#[test]
fn errno_appears_only_for_mincore_and_an_absent_page_size_is_null() {
    let obs = observation(
        vec![
            region(0, Err(ResidencyReadError::PageSizeUnavailable)),
            region(1, Err(ResidencyReadError::Unsupported)),
        ],
        None,
    );
    let records = observation_records(&run(), 0, "after_run", &obs);
    assert_eq!(records[0]["page_size"], Value::Null);
    assert_eq!(records[0]["phase"], "after_run");
    assert_eq!(records[1]["error"]["kind"], "page_size_unavailable");
    assert_eq!(records[2]["error"]["kind"], "unsupported");
    for r in &records[1..] {
        assert!(r["error"].get("errno").is_none(), "{r}");
    }
}

#[test]
fn the_writer_emits_one_json_value_per_line() {
    let obs = observation(
        vec![region(0, Ok(1)), region(1, Err(MINCORE_FAILED))],
        Some(16_384),
    );
    let mut out = Vec::new();
    write_observation(&mut out, &run(), 0, "binding", &obs).unwrap();
    let text = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    for line in lines {
        let v: Value = serde_json::from_str(line).expect("each line is one JSON value");
        assert_eq!(v["schema"], SCHEMA);
    }

    let mut empty = Vec::new();
    write_observation(
        &mut empty,
        &run(),
        1,
        "binding",
        &observation(Vec::new(), None),
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(empty).unwrap().lines().count(),
        1,
        "the header alone"
    );
}

#[test]
fn a_run_identity_names_this_process_and_hashes_the_index() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.json"), b"{}").unwrap();
    let id = RunIdentity::for_container(dir.path());
    assert_eq!(id.pid, std::process::id());
    assert!(id.run_id.starts_with(&format!("{}-", id.pid)));
    assert_eq!(id.index_json_sha256.as_deref().map(str::len), Some(64));

    let missing = RunIdentity::for_container(&dir.path().join("absent"));
    assert_eq!(
        missing.index_json_sha256, None,
        "no index, no hash — never a placeholder"
    );
}

/// NB6 in the record: `segment` is the declared container-relative path,
/// which is the same for a clone at another path; a store no container
/// declared has `segment: null` and `segment_declared: false`. Pairing
/// itself still needs equal container digests, which no key replaces.
#[test]
fn the_record_keys_regions_by_declared_segment_and_flags_undeclared_ones() {
    let a = observation(vec![region(0, Ok(1))], Some(16_384));
    let mut clone = a.clone();
    clone.stores[0].source_path = Some(PathBuf::from("/elsewhere/x.vindex3/segments/b.bin"));
    let (ra, rb) = (
        &observation_records(&run(), 0, "binding", &a)[1],
        &observation_records(&run(), 0, "binding", &clone)[1],
    );
    for field in [
        "store_id",
        "segment",
        "file_offset",
        "len",
        "segment_declared",
    ] {
        assert_eq!(ra[field], rb[field], "{field} must not move with the clone");
    }
    assert_ne!(ra["segment_path"], rb["segment_path"]);

    let mut undeclared = a.clone();
    undeclared.stores[0].declared_segment = None;
    let r = &observation_records(&run(), 0, "binding", &undeclared)[1];
    assert_eq!(r["segment"], Value::Null);
    assert_eq!(r["segment_path"], SEGMENT_PATH);
    assert_eq!(r["segment_declared"], false);
}

/// `binary.commit` is recorded only when the build declared it, and an
/// absent one says so instead of carrying a placeholder.
#[test]
fn an_undeclared_build_commit_is_recorded_as_unavailable() {
    let mut unknown = run();
    unknown.binary_commit = None;
    let header = &observation_records(&unknown, 0, "binding", &observation(Vec::new(), None))[0];
    assert_eq!(header["binary"]["commit"], Value::Null);
    assert_eq!(header["binary"]["commit_source"], "unavailable");

    assert_eq!(declared_commit(None), None);
    assert_eq!(declared_commit(Some("  ")), None, "blank is not a commit");
    assert_eq!(
        declared_commit(Some(" 58d75477\n")).as_deref(),
        Some("58d75477")
    );
}

/// NB4 at the console: an inconsistent summary prints its counts without
/// underflow and says it is not a witness.
#[test]
fn an_inconsistent_summary_is_labelled_on_the_binding_line() {
    let bad = MappedResidency {
        regions: 1,
        failed_regions: 2,
        ..MappedResidency::default()
    };
    let line = binding_line(&bad);
    assert!(line.contains("(0 read, 2 failed)"), "{line}");
    assert!(line.contains("INCONSISTENT COUNTS"), "{line}");
    let fine = observation(vec![region(0, Ok(1))], Some(16_384));
    assert!(!binding_line(&fine.summary()).contains("INCONSISTENT"));
}
