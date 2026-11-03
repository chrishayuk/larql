//! MEM-002 I2a: the admission decision and the evidence around it.
//!
//! The decision is pinned on summaries and observations built by hand,
//! because no fixture can make `mincore` fail. The end-to-end runs use the
//! dense fixture, which maps no region: its witness is empty, so it must
//! be refused — and no claim is made about residency beyond that.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use larql_vindex::format::vindex3::opplan::exec::prepared::{
    MappedResidency, ObservedStore, RegionResidency, ResidencyDelta, ResidencyObservation,
};
use larql_vindex::format::vindex3::represent::physical::ResidencyReadError;
use serde_json::{json, Value};

use super::super::generate::admission::{
    admission_line, commit_admission, create_evidence_file, decide, plan_evidence,
    refuse_evidence_outside_the_curve, AdmissionOutcome, AdmissionThreshold, EvidenceRequest,
    EvidenceSink, RunMode, COUNTING,
};
use super::super::generate::instance_evidence::EvidenceClass;
use super::super::generate::residency_record::{
    admission_record, attempt_end_record, observation_records, point_clock, sealed_header_fields,
    write_observation_with, AttemptFacts, RunIdentity, SCHEMA,
};
use super::super::generate::sealed_latency::{SealedDigest, SealedLatency, SEALED_SCHEMA};
use super::super::generate::{
    bind_line, end_before_passes, summary_lines, timing_cell, timing_tail, PassSample,
    StepObservation,
};
use super::super::{run, EncodeArgs, ExecArgs, ExecBackend, Vindex3Command};
use super::fixture_dir;

const THRESHOLD: u64 = 1 << 30;
const BASIS: &str = "docs/frozen.md §3 @ abc123";
const LEN: u64 = 1 << 20;

fn threshold(bytes: u64) -> AdmissionThreshold {
    AdmissionThreshold {
        bytes,
        basis: BASIS.to_string(),
    }
}

/// A complete witness of `resident` bytes over three regions.
fn complete(resident: u64) -> MappedResidency {
    MappedResidency {
        regions: 3,
        mapped_bytes: 3 * THRESHOLD,
        resident_bytes: resident,
        failed_regions: 0,
        mapped_bytes_read: 3 * THRESHOLD,
    }
}

fn observation(regions: Vec<RegionResidency>) -> ResidencyObservation {
    ResidencyObservation {
        started_wall: SystemTime::UNIX_EPOCH + Duration::from_secs(10),
        started_mono_ns: 5,
        ended_wall: SystemTime::UNIX_EPOCH + Duration::from_secs(11),
        ended_mono_ns: 9,
        started_system_mono_ns: Some(1_000),
        ended_system_mono_ns: Some(2_000),
        page_size: Some(16384),
        stores: vec![ObservedStore {
            id: "target.expert_bank".to_string(),
            source_path: Some(PathBuf::from("/m/c.vindex3/segments/bank.bin")),
            declared_segment: Some("segments/bank.bin".to_string()),
        }],
        regions,
    }
}

fn region(scan_index: usize, resident: Result<u64, ResidencyReadError>) -> RegionResidency {
    RegionResidency {
        scan_index,
        store: 0,
        file_offset: scan_index as u64 * LEN,
        len: LEN,
        resident,
    }
}

fn identity() -> RunIdentity {
    RunIdentity {
        run_id: "run-1".to_string(),
        pid: 42,
        binary_version: "0.0.0".to_string(),
        binary_commit: None,
        container_path: "/m/c.vindex3".to_string(),
        index_json_sha256: Some("ddd9".to_string()),
    }
}

fn facts(mode: RunMode) -> AttemptFacts {
    AttemptFacts {
        mode,
        warmup: 0,
        repeat: 1,
        qualification: "UNQUALIFIED (on battery)".to_string(),
        disqualifiers: vec!["on battery".to_string()],
        unquiet_ok: true,
        expert_access: "demand".to_string(),
        representation_source: "auto".to_string(),
        sealed_latency: Some("/e/sealed.jsonl".to_string()),
        instance: json!({ "identity_diagnostic": "unverified" }),
        evidence_class: EvidenceClass::Unverified,
    }
}

// ── The decision ──

/// The incomplete observation must be a refusal, never an apparent
/// zero-byte cold state. A partial sum of zero is the dangerous case: it
/// is below every threshold.
#[test]
fn an_incomplete_observation_is_refused_and_never_reads_as_a_zero_byte_cold_state() {
    const MINCORE: ResidencyReadError = ResidencyReadError::Mincore { errno: 12 };
    let cases = [
        vec![region(0, Err(MINCORE)), region(1, Err(MINCORE))],
        vec![region(0, Ok(0)), region(1, Err(MINCORE))],
        vec![region(0, Err(MINCORE)), region(1, Ok(0))],
    ];
    for regions in cases {
        let observation = observation(regions);
        for declared in [None, Some(threshold(THRESHOLD))] {
            let decision = decide(&observation.summary(), declared.as_ref());
            assert_eq!(decision.outcome, AdmissionOutcome::WitnessFailed);
            assert!(!decision.complete);
            assert_eq!(decision.resident_bytes, None, "never Some(0)");
            let record = admission_record(&identity(), 0, &decision, &facts(RunMode::Campaign));
            assert_eq!(record["outcome"], "witness_failed");
            assert_eq!(record["complete"], false);
            assert_eq!(record["resident_bytes"], Value::Null);
            let line = admission_line(&decision);
            assert!(line.contains("witness_failed"), "{line}");
            assert!(line.contains("incomplete witness"), "{line}");
        }
    }
}

#[test]
fn zero_regions_witness_nothing_and_are_refused() {
    let empty = observation(Vec::new());
    for declared in [None, Some(threshold(THRESHOLD))] {
        let decision = decide(&empty.summary(), declared.as_ref());
        assert_eq!(decision.outcome, AdmissionOutcome::WitnessFailed);
        assert_eq!(decision.regions, 0);
        assert_eq!(decision.resident_bytes, None);
    }
}

/// Regions of zero length are read successfully and hold nothing: that
/// witnesses no image at all, and must not be admitted as a cold one.
#[test]
fn an_image_of_empty_regions_witnesses_nothing() {
    let empty = MappedResidency {
        regions: 2,
        ..MappedResidency::default()
    };
    let decision = decide(&empty, Some(&threshold(THRESHOLD)));
    assert_eq!(decision.outcome, AdmissionOutcome::WitnessFailed);
    assert_eq!(decision.resident_bytes, None);
    let partly_read = MappedResidency {
        regions: 2,
        mapped_bytes: 2,
        mapped_bytes_read: 1,
        ..MappedResidency::default()
    };
    assert_eq!(
        decide(&partly_read, None).outcome,
        AdmissionOutcome::WitnessFailed
    );
}

#[test]
fn inconsistent_counts_are_refused() {
    let more_failed_than_regions = MappedResidency {
        regions: 1,
        failed_regions: 2,
        ..MappedResidency::default()
    };
    let more_read_than_mapped = MappedResidency {
        regions: 1,
        mapped_bytes: 1,
        mapped_bytes_read: 2,
        ..MappedResidency::default()
    };
    for summary in [more_failed_than_regions, more_read_than_mapped] {
        let decision = decide(&summary, Some(&threshold(THRESHOLD)));
        assert_eq!(decision.outcome, AdmissionOutcome::WitnessFailed);
        assert!(!decision.consistent);
        assert_eq!(decision.resident_bytes, None);
    }
}

#[test]
fn the_threshold_boundary_admits_at_or_below_and_refuses_above() {
    let cases = [
        (0, THRESHOLD, AdmissionOutcome::Admitted),
        (THRESHOLD - 1, THRESHOLD, AdmissionOutcome::Admitted),
        (THRESHOLD, THRESHOLD, AdmissionOutcome::Admitted),
        (THRESHOLD + 1, THRESHOLD, AdmissionOutcome::AboveThreshold),
        (0, 0, AdmissionOutcome::Admitted),
        (1, 0, AdmissionOutcome::AboveThreshold),
        (u64::MAX, u64::MAX, AdmissionOutcome::Admitted),
    ];
    for (resident, bytes, expected) in cases {
        let decision = decide(&complete(resident), Some(&threshold(bytes)));
        assert_eq!(decision.outcome, expected, "{resident} against {bytes}");
        assert!(decision.complete);
        assert_eq!(decision.resident_bytes, Some(resident));
    }
}

/// No threshold introduces none: a complete witness of nothing resident
/// is still never `admitted`.
#[test]
fn without_a_declared_threshold_nothing_is_ever_admitted() {
    for resident in [0, 1, u64::MAX] {
        let decision = decide(&complete(resident), None);
        assert_eq!(decision.outcome, AdmissionOutcome::WitnessOnly);
        assert_eq!(decision.threshold, None);
    }
}

#[test]
fn outcome_names_are_the_ones_the_protocol_records() {
    assert_eq!(AdmissionOutcome::WitnessFailed.name(), "witness_failed");
    assert_eq!(AdmissionOutcome::AboveThreshold.name(), "above_threshold");
    assert_eq!(AdmissionOutcome::Admitted.name(), "admitted");
    assert_eq!(AdmissionOutcome::WitnessOnly.name(), "witness_only");
    assert_eq!(RunMode::Campaign.name(), "campaign");
    assert_eq!(RunMode::Diagnostic.name(), "diagnostic");
}

#[test]
fn the_console_line_names_the_outcome_total_and_threshold() {
    let line = admission_line(&decide(&complete(THRESHOLD), Some(&threshold(THRESHOLD))));
    assert!(line.contains("admission: admitted"), "{line}");
    assert!(line.contains("complete true"), "{line}");
    assert!(line.contains("1.074 GB over 3 regions"), "{line}");
    assert!(line.contains("threshold 1.074 GB"), "{line}");
    let line = admission_line(&decide(&complete(0), None));
    assert!(line.contains("witness_only"), "{line}");
    assert!(line.contains("threshold none"), "{line}");
}

// ── The record ──

#[test]
fn the_admission_record_carries_the_decision_and_the_attempt() {
    let decision = decide(&complete(THRESHOLD + 1), Some(&threshold(THRESHOLD)));
    let record = admission_record(&identity(), 7, &decision, &facts(RunMode::Campaign));
    assert_eq!(record["schema"], SCHEMA);
    assert_eq!(SCHEMA, "larql.residency-observation/4");
    assert_eq!(record["record"], "admission");
    assert_eq!(record["run_id"], "run-1");
    assert_eq!(record["binding_observation_id"], 7);
    assert_eq!(record["mode"], "campaign");
    assert_eq!(record["outcome"], "above_threshold");
    assert_eq!(record["complete"], true);
    assert_eq!(record["consistent"], true);
    assert_eq!(record["regions"], 3);
    assert_eq!(record["failed"], 0);
    assert_eq!(record["resident_bytes"], THRESHOLD + 1);
    assert_eq!(record["counting"], COUNTING);
    assert_eq!(record["threshold"]["bytes"], THRESHOLD);
    assert_eq!(record["threshold"]["basis"], BASIS);
    assert_eq!(record["warmup"], 0);
    assert_eq!(record["repeat"], 1);
    assert_eq!(record["expert_access"], "demand");
    assert_eq!(record["quiet"]["unquiet_ok"], true);
    assert_eq!(record["quiet"]["disqualifiers"], json!(["on battery"]));
    assert_eq!(record["sealed_latency"]["path"], "/e/sealed.jsonl");
    assert_eq!(record["binary"]["commit"], Value::Null);
    assert_eq!(record["binary"]["commit_source"], "unavailable");
    assert_eq!(record["container"]["index_json_sha256"], "ddd9");
    assert_eq!(record["representation_source"], "auto");
    assert_eq!(record["evidence_class"], "unverified");
    assert_eq!(
        record["instance_identity"]["identity_diagnostic"],
        "unverified"
    );
    assert_clock_is_on_its_named_basis(&record["clock"]);
    assert_eq!(record["clock"]["system_mono_ns"].is_u64(), cfg!(unix));
}

fn assert_clock_is_on_its_named_basis(clock: &Value) {
    if cfg!(unix) {
        assert_eq!(clock["basis"], "CLOCK_MONOTONIC_RAW");
    } else {
        assert_eq!(clock["basis"], Value::Null);
    }
    assert!(clock.get("boot_session").is_some());
}

#[test]
fn a_record_with_no_threshold_or_sealed_file_says_so() {
    let mut facts = facts(RunMode::Diagnostic);
    facts.sealed_latency = None;
    let record = admission_record(&identity(), 0, &decide(&complete(0), None), &facts);
    assert_eq!(record["mode"], "diagnostic");
    assert_eq!(record["outcome"], "witness_only");
    assert_eq!(record["threshold"], Value::Null);
    assert_eq!(record["sealed_latency"], Value::Null);
}

#[test]
fn an_observation_header_carries_both_clocks_and_never_a_zero_for_a_missing_one() {
    let observed = observation(vec![region(0, Ok(LEN))]);
    let records = observation_records(&identity(), 0, "binding", &observed);
    let header = &records[0];
    assert_eq!(header["schema"], SCHEMA);
    // Process-relative, kept for compatibility.
    assert_eq!(header["started_mono_ns"], 5);
    assert_eq!(header["ended_mono_ns"], 9);
    assert_eq!(header["clock"]["start_system_mono_ns"], 1_000);
    assert_eq!(header["clock"]["end_system_mono_ns"], 2_000);
    assert_clock_is_on_its_named_basis(&header["clock"]);

    let mut no_clock = observed;
    no_clock.started_system_mono_ns = None;
    no_clock.ended_system_mono_ns = None;
    let records = observation_records(&identity(), 0, "binding", &no_clock);
    assert_eq!(records[0]["clock"]["start_system_mono_ns"], Value::Null);
    assert_eq!(records[0]["clock"]["end_system_mono_ns"], Value::Null);
}

#[test]
fn the_attempt_end_says_how_the_attempt_ended() {
    let done = attempt_end_record(&identity(), None, EvidenceClass::Unverified, None);
    assert_eq!(done["record"], "attempt_end");
    assert_eq!(done["status"], "completed");
    assert_eq!(done["error"], Value::Null);
    assert_clock_is_on_its_named_basis(&done["clock"]);
    let failed = attempt_end_record(&identity(), Some("boom"), EvidenceClass::Invalid, None);
    assert_eq!(failed["status"], "error");
    assert_eq!(failed["error"], "boom");
}

#[test]
fn only_the_after_run_header_gains_the_sealed_digest() {
    let digest = SealedDigest {
        path: PathBuf::from("/e/sealed.jsonl"),
        sha256: "feed".to_string(),
    };
    let observed = observation(vec![region(0, Ok(LEN))]);
    let mut out = Vec::new();
    write_observation_with(
        &mut out,
        &identity(),
        1,
        "after_run",
        &observed,
        sealed_header_fields(&digest),
    )
    .unwrap();
    let lines = parse_lines(std::str::from_utf8(&out).unwrap());
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["sealed_latency"]["sha256"], "feed");
    assert_eq!(lines[0]["sealed_latency"]["path"], "/e/sealed.jsonl");
    assert!(lines[1].get("sealed_latency").is_none(), "a region record");
}

// ── Written before the passes ──

/// A sink that can fail at the write or at the flush to disk, and records
/// what it was given.
#[derive(Default)]
struct Sink {
    bytes: Vec<u8>,
    fail_write: bool,
    fail_sync: bool,
    synced_bytes: Option<usize>,
}

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.fail_write {
            return Err(io::Error::other("no space left"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl EvidenceSink for Sink {
    fn sync(&mut self) -> io::Result<()> {
        if self.fail_sync {
            return Err(io::Error::other("sync failed"));
        }
        self.synced_bytes = Some(self.bytes.len());
        Ok(())
    }
}

#[test]
fn the_admission_commit_writes_one_line_and_syncs_it() {
    let mut sink = Sink::default();
    let committed = commit_admission(
        RunMode::Campaign,
        Some(&mut sink),
        &json!({"record": "admission"}),
    );
    assert!(committed.unwrap().recorded());
    assert_eq!(sink.bytes, b"{\"record\":\"admission\"}\n");
    assert_eq!(
        sink.synced_bytes,
        Some(sink.bytes.len()),
        "everything synced"
    );
}

/// The passes take an `AdmissionCommitted`; a failed commit yields none,
/// so the attempt stops here rather than running unrecorded.
#[test]
fn a_record_that_cannot_be_written_stops_the_attempt_before_execution() {
    let record = json!({"record": "admission"});
    let mut no_write = Sink {
        fail_write: true,
        ..Sink::default()
    };
    let err = commit_admission(RunMode::Campaign, Some(&mut no_write), &record).unwrap_err();
    assert!(err.contains("before the passes"), "{err}");
    assert!(err.contains("no space left"), "{err}");

    let mut no_sync = Sink {
        fail_sync: true,
        ..Sink::default()
    };
    let err = commit_admission(RunMode::Diagnostic, Some(&mut no_sync), &record).unwrap_err();
    assert!(err.contains("sync failed"), "{err}");
}

#[test]
fn only_diagnostic_execution_may_go_unrecorded() {
    let record = json!({});
    assert!(!commit_admission(RunMode::Diagnostic, None, &record)
        .unwrap()
        .recorded());
    let err = commit_admission(RunMode::Campaign, None, &record).unwrap_err();
    assert!(err.contains("no residency stream"), "{err}");
}

/// An attempt that ends before admission still says so, and a stream that
/// cannot take the record does not replace the reason.
#[test]
fn an_attempt_ended_before_the_passes_leaves_its_reason() {
    let mut stream = Some(Sink::default());
    let reason = end_before_passes(
        &mut stream,
        &identity(),
        EvidenceClass::Invalid,
        "not quiet".to_string(),
    );
    assert_eq!(reason, "not quiet");
    let sink = stream.unwrap();
    assert_eq!(sink.synced_bytes, Some(sink.bytes.len()));
    let lines = parse_lines(std::str::from_utf8(&sink.bytes).unwrap());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["record"], "attempt_end");
    assert_eq!(lines[0]["status"], "error");
    assert_eq!(lines[0]["error"], "not quiet");

    let mut broken = Some(Sink {
        fail_write: true,
        ..Sink::default()
    });
    assert_eq!(
        end_before_passes(
            &mut broken,
            &identity(),
            EvidenceClass::Invalid,
            "bind failed".to_string()
        ),
        "bind failed"
    );
    let mut none: Option<Sink> = None;
    assert_eq!(
        end_before_passes(
            &mut none,
            &identity(),
            EvidenceClass::Invalid,
            "x".to_string()
        ),
        "x"
    );
}

// ── The evidence plan ──

#[test]
fn every_incomplete_campaign_request_is_refused() {
    let regions = Path::new("/e/regions.jsonl");
    let sealed = Path::new("/e/sealed.jsonl");
    let cases: [(EvidenceRequest<'_>, &str); 7] = [
        (
            EvidenceRequest {
                threshold_bytes: Some(1),
                ..EvidenceRequest::default()
            },
            "needs --admission-basis",
        ),
        (
            EvidenceRequest {
                basis: Some(BASIS),
                ..EvidenceRequest::default()
            },
            "no --admission-threshold-bytes",
        ),
        (
            EvidenceRequest {
                campaign: true,
                regions: Some(regions),
                sealed_latency: Some(sealed),
                manifest: None,
                threshold_bytes: Some(1),
                basis: Some("  "),
            },
            "blank",
        ),
        (
            EvidenceRequest {
                regions: Some(regions),
                sealed_latency: Some(sealed),
                threshold_bytes: Some(1),
                basis: Some(BASIS),
                ..EvidenceRequest::default()
            },
            "pass --residency-campaign",
        ),
        (
            EvidenceRequest {
                campaign: true,
                sealed_latency: Some(sealed),
                ..EvidenceRequest::default()
            },
            "needs --residency-regions",
        ),
        // Calibration: a campaign attempt with no threshold is still sealed.
        (
            EvidenceRequest {
                campaign: true,
                regions: Some(regions),
                ..EvidenceRequest::default()
            },
            "needs --sealed-latency",
        ),
        (
            EvidenceRequest {
                regions: Some(regions),
                sealed_latency: Some(Path::new("/e/./regions.jsonl")),
                ..EvidenceRequest::default()
            },
            "one file",
        ),
    ];
    for (request, expected) in cases {
        let err = plan_evidence(request).unwrap_err();
        assert!(err.contains(expected), "{err} should say {expected}");
    }
}

#[test]
fn a_calibration_attempt_is_a_campaign_with_no_threshold() {
    let plan = plan_evidence(EvidenceRequest {
        campaign: true,
        regions: Some(Path::new("r")),
        sealed_latency: Some(Path::new("s")),
        ..EvidenceRequest::default()
    })
    .unwrap();
    assert_eq!(plan.mode, RunMode::Campaign);
    assert_eq!(plan.threshold, None);
}

#[test]
fn a_scored_attempt_keeps_its_threshold_and_basis_verbatim() {
    let plan = plan_evidence(EvidenceRequest {
        campaign: true,
        regions: Some(Path::new("r")),
        sealed_latency: Some(Path::new("s")),
        manifest: None,
        threshold_bytes: Some(THRESHOLD),
        basis: Some(BASIS),
    })
    .unwrap();
    assert_eq!(plan.threshold, Some(threshold(THRESHOLD)));
}

#[test]
fn diagnostic_execution_needs_no_evidence() {
    let plan = plan_evidence(EvidenceRequest::default()).unwrap();
    assert_eq!(plan.mode, RunMode::Diagnostic);
    assert_eq!(plan.regions, None);
    assert_eq!(plan.sealed_latency, None);
}

#[test]
fn an_existing_evidence_file_is_refused_and_left_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("evidence.jsonl");
    std::fs::write(&path, "earlier evidence").unwrap();
    let err = create_evidence_file(&path, "--residency-regions").unwrap_err();
    assert!(err.contains("already exists"), "{err}");
    assert!(err.contains("--residency-regions"), "{err}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "earlier evidence");

    let missing_dir = dir.path().join("no/such/dir/evidence.jsonl");
    let err = create_evidence_file(&missing_dir, "--sealed-latency").unwrap_err();
    assert!(err.contains("cannot create --sealed-latency"), "{err}");
}

// ── Sealed latency ──

#[test]
fn the_sealed_file_holds_the_latency_and_its_digest_is_of_what_reached_the_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sealed.jsonl");
    let mut sealed = SealedLatency::create(&path, &identity()).unwrap();
    assert_eq!(sealed.path(), path);
    sealed.bind(1.5).unwrap();
    sealed.sync().unwrap();
    assert!(std::fs::read_to_string(&path).unwrap().contains("\"bind\""));
    sealed
        .step(
            1,
            "cold",
            "token 1",
            12.5,
            json!({"stages_ms": {"attention": 3.0}}),
        )
        .unwrap();
    sealed.failed_step(1, "cold", "token 2", 7.5).unwrap();
    let digest = sealed.finish().unwrap();
    assert_eq!(digest.path, path);
    assert_eq!(
        digest.sha256,
        larql_vindex::format::checksums::sha256_file(&path).unwrap()
    );
    let lines = parse_lines(&std::fs::read_to_string(&path).unwrap());
    for line in &lines {
        assert_eq!(line["schema"], SEALED_SCHEMA);
        assert_eq!(line["run_id"], "run-1");
        assert_clock_is_on_its_named_basis(&line["clock"]);
    }
    assert_eq!(lines[0]["record"], "header");
    assert_eq!(lines[0]["container"]["index_json_sha256"], "ddd9");
    assert_eq!(lines[1]["record"], "bind");
    assert_eq!(lines[1]["seconds"], 1.5);
    assert_eq!(lines[2]["record"], "step");
    assert_eq!(lines[2]["pass"], 1);
    assert_eq!(lines[2]["label"], "cold");
    assert_eq!(lines[2]["step"], "token 1");
    assert_eq!(lines[2]["elapsed_ms"], 12.5);
    assert_eq!(lines[2]["status"], "ok");
    assert_eq!(lines[2]["breakdown"]["stages_ms"]["attention"], 3.0);
    assert_eq!(lines[3]["status"], "error");
    assert_eq!(lines[3]["elapsed_ms"], 7.5);
}

#[test]
fn an_existing_sealed_file_is_refused_and_left_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sealed.jsonl");
    std::fs::write(&path, "sealed earlier").unwrap();
    let err = SealedLatency::create(&path, &identity()).err().unwrap();
    assert!(err.contains("already exists"), "{err}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "sealed earlier");
}

#[test]
fn a_point_clock_reads_on_its_named_basis() {
    let clock = point_clock();
    assert_clock_is_on_its_named_basis(&clock);
    assert_eq!(clock["system_mono_ns"].is_u64(), cfg!(unix));
}

// ── Sealing the console ──

/// The summary and table markers that only ever precede a printed latency.
fn prints_a_timing(line: &str) -> bool {
    line.contains("p50")
        || line.contains("stages (ms)")
        || line.contains("[leaves")
        || line.contains(" s ")
}

fn sample(pass: usize, elapsed_ms: f64) -> PassSample {
    PassSample {
        pass,
        observation: StepObservation {
            elapsed_ms,
            resident_delta: ResidencyDelta::default(),
            minor_faults: 3,
            major_faults: 2,
            stage_ms: [1.25; larql_vindex::format::vindex3::opplan::exec::stages::Stage::ALL.len()],
            stages_nested: 0,
            leaf_ms: [0.5; 3],
        },
        routing_fingerprint: 0xabc,
        routed_layers: 4,
        first_layer_experts: vec![1, 2],
        logits_delta: 0.0,
    }
}

/// Distinctive values that may not appear in a sealed summary.
const SEALED_MS: f64 = 98_765.4;

#[test]
fn a_sealed_summary_prints_no_timing_and_keeps_the_rest() {
    let samples = [sample(1, SEALED_MS), sample(2, SEALED_MS)];
    let sealed = summary_lines(&samples, 2, 0, "QUALIFIED", Some(Path::new("/e/s.jsonl")));
    let text = sealed.join("\n");
    assert!(text.contains("SEALED in /e/s.jsonl"), "{text}");
    assert!(text.contains("routing: IDENTICAL"), "{text}");
    assert!(text.contains("faults: major 4 total"), "{text}");
    assert!(!text.contains("98765"), "{text}");
    assert!(!sealed.iter().any(|l| prints_a_timing(l)), "{text}");

    let open = summary_lines(&samples, 2, 0, "QUALIFIED", None).join("\n");
    assert!(open.contains("98765"), "{open}");
    assert!(open.contains("p50"), "{open}");
    assert!(!open.contains("SEALED"), "{open}");
}

#[test]
fn the_summary_still_reports_differing_routing_and_nothing_counted() {
    let mut other = sample(2, 1.0);
    other.routing_fingerprint = 0xdef;
    other.observation.stages_nested = 1;
    let text = summary_lines(&[sample(1, 1.0), other], 2, 0, "Q", None).join("\n");
    assert!(text.contains("DIFFERS across passes"), "{text}");
    assert!(text.contains("REFUSING TO RECONCILE STAGES"), "{text}");
    let nothing = summary_lines(&[], 1, 0, "Q", None).join("\n");
    assert!(nothing.contains("nothing counted"), "{nothing}");
}

#[test]
fn the_table_and_the_bind_line_print_no_timing_while_sealed() {
    assert_eq!(timing_cell(SEALED_MS, true), "sealed");
    assert_eq!(timing_cell(SEALED_MS, false), "98765");
    let stages = ["ffn=1.5".to_string()];
    let leaves = ["norm=2".to_string()];
    assert_eq!(timing_tail(&stages, &leaves, true), "");
    assert_eq!(
        timing_tail(&stages, &leaves, false),
        "  ffn=1.5  [leaves norm=2]"
    );
    let sealed = bind_line(123.4, 1, "demand", true);
    assert!(!sealed.contains("123.4"), "{sealed}");
    assert!(sealed.contains("bind time sealed"), "{sealed}");
    let open = bind_line(123.4, 1, "demand", false);
    assert!(open.contains("in 123.4 s"), "{open}");
}

// ── End to end, on a fixture that maps no region ──

/// An encoded dense fixture, with the temp dir that keeps it alive.
fn encoded() -> (tempfile::TempDir, PathBuf) {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    (dir, out)
}

#[derive(Default)]
struct Curve {
    tokens: Option<&'static str>,
    regions: Option<PathBuf>,
    sealed: Option<PathBuf>,
    campaign: bool,
    threshold: Option<u64>,
    basis: Option<String>,
}

fn run_curve(container: &Path, curve: Curve) -> Result<(), Box<dyn std::error::Error>> {
    run(Vindex3Command::Exec(ExecArgs {
        container: container.to_path_buf(),
        component: "target".to_string(),
        tokens: curve.tokens.unwrap_or("1,2").to_string(),
        dump_layers: None,
        resume: false,
        backend: ExecBackend::Production,
        representation_source: "auto".to_string(),
        generate: Some(2),
        speculate: None,
        stop_at_eos: false,
        residency_curve: true,
        repeat: 1,
        warmup: 0,
        unquiet_ok: true,
        residency_regions: curve.regions,
        sealed_latency: curve.sealed,
        residency_campaign: curve.campaign,
        verification_manifest: None,
        admission_threshold_bytes: curve.threshold,
        admission_basis: curve.basis,
        expert_access: "demand".to_string(),
        logit_dump: None,
        bank: None,
        dump_dir: None,
        draft_depth: None,
        profile: false,
        plugin: Default::default(),
    }))
}

fn parse_lines(text: &str) -> Vec<Value> {
    text.lines()
        .map(|l| serde_json::from_str(l).expect("one JSON value per line"))
        .collect()
}

fn read_lines(path: &Path) -> Vec<Value> {
    parse_lines(&std::fs::read_to_string(path).expect("the record is written"))
}

/// The stream without its identity records, which `instance_evidence`
/// pins, so what these tests pin stands on its own.
fn without_identity(stream: Vec<Value>) -> Vec<Value> {
    stream
        .into_iter()
        .filter(|v| v["record"] != "instance_identity")
        .collect()
}

/// A refused attempt is still executed, timed and recorded, and its
/// latency stays out of the residency stream. No region is mapped here,
/// so the witness is empty: refused, with no resident total at all.
#[test]
#[cfg_attr(
    not(unix),
    ignore = "a campaign stops here with no descriptor identity"
)]
fn a_refused_campaign_attempt_runs_and_is_recorded_before_and_after_the_passes() {
    let (dir, container) = encoded();
    let regions = dir.path().join("regions.jsonl");
    let sealed = dir.path().join("sealed.jsonl");
    run_curve(
        &container,
        Curve {
            regions: Some(regions.clone()),
            sealed: Some(sealed.clone()),
            campaign: true,
            threshold: Some(THRESHOLD),
            basis: Some(BASIS.to_string()),
            ..Curve::default()
        },
    )
    .expect("a refused attempt still completes");

    let stream = without_identity(read_lines(&regions));
    let order: Vec<&str> = stream
        .iter()
        .map(|v| v["record"].as_str().unwrap())
        .collect();
    assert_eq!(
        order,
        ["observation", "admission", "observation", "attempt_end"]
    );
    assert_eq!(stream[0]["phase"], "binding");
    assert_eq!(stream[2]["phase"], "after_run");
    let admission = &stream[1];
    assert_eq!(admission["mode"], "campaign");
    assert_eq!(admission["outcome"], "witness_failed");
    assert_eq!(admission["complete"], false);
    assert_eq!(admission["regions"], 0);
    assert_eq!(admission["resident_bytes"], Value::Null);
    assert_eq!(admission["threshold"]["bytes"], THRESHOLD);
    assert_eq!(admission["threshold"]["basis"], BASIS);
    assert_eq!(stream[3]["status"], "completed");
    let run_id = &stream[0]["run_id"];
    assert!(stream.iter().all(|v| &v["run_id"] == run_id));

    // The system-wide clock orders the records within the attempt.
    if cfg!(unix) {
        let bound = stream[0]["clock"]["end_system_mono_ns"].as_u64().unwrap();
        let decided = admission["clock"]["system_mono_ns"].as_u64().unwrap();
        let after = stream[2]["clock"]["start_system_mono_ns"].as_u64().unwrap();
        let ended = stream[3]["clock"]["system_mono_ns"].as_u64().unwrap();
        assert!(bound <= decided && decided <= after && after <= ended);
    }

    // The sealed file holds the latency, digest recorded after the run.
    let digest = larql_vindex::format::checksums::sha256_file(&sealed).unwrap();
    assert_eq!(stream[2]["sealed_latency"]["sha256"], digest.as_str());
    assert_eq!(
        admission["sealed_latency"]["path"],
        sealed.display().to_string().as_str()
    );
    let latency = read_lines(&sealed);
    assert_eq!(&latency[0]["run_id"], run_id);
    let kinds: Vec<&str> = latency
        .iter()
        .map(|v| v["record"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["header", "bind", "step", "step"]);
    assert!(latency[1]["seconds"].is_f64());
    assert_eq!(latency[2]["step"], "prompt×2");
    assert_eq!(latency[3]["step"], "token 1");
    assert!(latency[3]["elapsed_ms"].is_f64());

    let breakdown = &latency[3]["breakdown"];
    assert!(breakdown["stages_ms"].is_object());
    assert!(breakdown["leaves_ms"].is_object());
    assert_eq!(breakdown["stages_nested"], 0);

    // On record before any timed step: the sealed file's first step is
    // stamped after the stream's admission, on the one system-wide clock.
    if cfg!(unix) {
        let decided = admission["clock"]["system_mono_ns"].as_u64().unwrap();
        let first_step = latency[2]["clock"]["system_mono_ns"].as_u64().unwrap();
        assert!(decided <= first_step, "{decided} > {first_step}");
    }

    // No latency FIELD in the stream an evaluator reads. This checks key
    // names only: clock readings remain, and their difference gives the
    // attempt's whole duration (see `sealed_latency`'s module docs).
    let text = std::fs::read_to_string(&regions).unwrap();
    for key in [
        "elapsed_ms",
        "stages_ms",
        "leaves_ms",
        "\"seconds\"",
        "load_seconds",
    ] {
        assert!(
            !text.contains(key),
            "{key} leaked into the residency stream"
        );
    }
}

/// An attempt whose passes fail still closes its record, with the error.
#[test]
#[cfg_attr(
    not(unix),
    ignore = "a campaign stops here with no descriptor identity"
)]
fn an_attempt_whose_passes_fail_records_how_it_ended() {
    let (dir, container) = encoded();
    let regions = dir.path().join("regions.jsonl");
    let sealed = dir.path().join("sealed.jsonl");
    let err = run_curve(
        &container,
        Curve {
            tokens: Some("4000000000"),
            regions: Some(regions.clone()),
            sealed: Some(sealed.clone()),
            campaign: true,
            ..Curve::default()
        },
    )
    .expect_err("a token past the vocabulary cannot be stepped");
    let stream = without_identity(read_lines(&regions));
    let order: Vec<&str> = stream
        .iter()
        .map(|v| v["record"].as_str().unwrap())
        .collect();
    assert_eq!(
        order,
        ["observation", "admission", "observation", "attempt_end"]
    );
    assert_eq!(stream[3]["status"], "error");
    assert_eq!(stream[3]["error"], err.to_string().as_str());
    // The failing step's time is latency too, and it is sealed.
    let latency = read_lines(&sealed);
    let last = latency.last().unwrap();
    assert_eq!(last["record"], "step");
    assert_eq!(last["status"], "error");
    assert!(last["elapsed_ms"].is_f64());
    assert_eq!(
        stream[2]["sealed_latency"]["sha256"],
        larql_vindex::format::checksums::sha256_file(&sealed)
            .unwrap()
            .as_str()
    );
}

/// Weights that cannot be bound are an attempt too: the stream says how it
/// ended, with no admission and no pass.
#[test]
#[cfg_attr(
    not(unix),
    ignore = "a campaign stops here with no descriptor identity"
)]
fn an_attempt_whose_weights_cannot_be_bound_records_how_it_ended() {
    let (dir, container) = encoded();
    for entry in walk(&container) {
        if entry.extension().is_some_and(|e| e == "bin") {
            let len = std::fs::metadata(&entry).unwrap().len();
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&entry)
                .unwrap();
            file.set_len(len - 1).unwrap();
        }
    }
    let regions = dir.path().join("regions.jsonl");
    let err = run_curve(
        &container,
        Curve {
            regions: Some(regions.clone()),
            sealed: Some(dir.path().join("sealed.jsonl")),
            campaign: true,
            ..Curve::default()
        },
    )
    .expect_err("a segment short of its payload cannot be bound");
    assert!(
        regions.exists(),
        "failed before the curve began, so this does not test binding: {err}"
    );
    let stream = without_identity(read_lines(&regions));
    assert_eq!(stream.len(), 1, "{stream:?}");
    assert_eq!(stream[0]["record"], "attempt_end");
    assert_eq!(stream[0]["status"], "error");
    assert_eq!(stream[0]["error"], err.to_string().as_str());
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

/// `--sealed-latency` alone seals a diagnostic run too; the mode stays
/// diagnostic and an ordinary stream-less run still works.
#[test]
fn diagnostic_runs_keep_working_with_and_without_sealing() {
    let (dir, container) = encoded();
    run_curve(&container, Curve::default()).expect("a plain diagnostic run");
    let regions = dir.path().join("regions.jsonl");
    let sealed = dir.path().join("sealed.jsonl");
    run_curve(
        &container,
        Curve {
            regions: Some(regions.clone()),
            sealed: Some(sealed.clone()),
            ..Curve::default()
        },
    )
    .expect("a sealed diagnostic run");
    let stream = without_identity(read_lines(&regions));
    assert_eq!(stream[1]["mode"], "diagnostic");
    assert!(sealed.exists());

    // Sealing alone needs no stream.
    let only_sealed = dir.path().join("only-sealed.jsonl");
    run_curve(
        &container,
        Curve {
            sealed: Some(only_sealed.clone()),
            ..Curve::default()
        },
    )
    .expect("sealing without a stream");
    assert!(only_sealed.exists());
}

#[test]
fn a_duplicate_evidence_path_is_refused_before_anything_is_created() {
    let (dir, container) = encoded();
    let same = dir.path().join("evidence.jsonl");
    let err = run_curve(
        &container,
        Curve {
            regions: Some(same.clone()),
            sealed: Some(same.clone()),
            campaign: true,
            ..Curve::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("one file"), "{err}");
    assert!(!same.exists(), "refused before anything was created");
}

#[test]
fn an_existing_stream_refuses_the_run_and_survives_it() {
    let (dir, container) = encoded();
    let regions = dir.path().join("regions.jsonl");
    let sealed = dir.path().join("sealed.jsonl");
    std::fs::write(&regions, "earlier evidence").unwrap();
    let err = run_curve(
        &container,
        Curve {
            regions: Some(regions.clone()),
            sealed: Some(sealed.clone()),
            campaign: true,
            ..Curve::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("already exists"), "{err}");
    assert_eq!(
        std::fs::read_to_string(&regions).unwrap(),
        "earlier evidence"
    );
    assert!(!sealed.exists(), "the stream is created first");
}

/// The refused attempt the review found unclosed: the stream exists by the
/// time the sealed file is refused, so the stream must say why.
#[test]
fn an_existing_sealed_file_refuses_the_attempt_on_the_record() {
    let (dir, container) = encoded();
    let regions = dir.path().join("regions.jsonl");
    let sealed = dir.path().join("sealed.jsonl");
    std::fs::write(&sealed, "sealed earlier").unwrap();
    let err = run_curve(
        &container,
        Curve {
            regions: Some(regions.clone()),
            sealed: Some(sealed.clone()),
            campaign: true,
            ..Curve::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("already exists"), "{err}");
    assert_eq!(std::fs::read_to_string(&sealed).unwrap(), "sealed earlier");
    let stream = without_identity(read_lines(&regions));
    assert_eq!(stream.len(), 1, "{stream:?}");
    assert_eq!(stream[0]["record"], "attempt_end");
    assert_eq!(stream[0]["status"], "error");
    assert_eq!(stream[0]["error"], err.as_str());
}

/// A complete campaign request, which the refusal table below varies.
fn evidence_args(container: &Path, regions: &Path, sealed: &Path) -> ExecArgs {
    ExecArgs {
        container: container.to_path_buf(),
        component: "target".to_string(),
        tokens: "1,2".to_string(),
        dump_layers: None,
        resume: false,
        backend: ExecBackend::Production,
        representation_source: "auto".to_string(),
        generate: Some(2),
        speculate: None,
        stop_at_eos: false,
        residency_curve: true,
        repeat: 1,
        warmup: 0,
        unquiet_ok: true,
        residency_regions: Some(regions.to_path_buf()),
        sealed_latency: Some(sealed.to_path_buf()),
        residency_campaign: true,
        verification_manifest: None,
        admission_threshold_bytes: None,
        admission_basis: None,
        expert_access: "demand".to_string(),
        logit_dump: None,
        bank: None,
        dump_dir: None,
        draft_depth: None,
        profile: false,
        plugin: Default::default(),
    }
}

/// A path that never reaches the curve would print its timings and record
/// nothing, then exit 0: every one is refused with nothing created.
#[test]
fn evidence_flags_are_refused_on_every_path_that_bypasses_the_curve() {
    let (dir, container) = encoded();
    let regions = dir.path().join("regions.jsonl");
    let sealed = dir.path().join("sealed.jsonl");
    let args = || evidence_args(&container, &regions, &sealed);
    let cases: Vec<(ExecArgs, &str)> = vec![
        (
            ExecArgs {
                residency_curve: false,
                ..args()
            },
            "without --residency-curve",
        ),
        (
            ExecArgs {
                generate: None,
                ..args()
            },
            "without --generate",
        ),
        (
            ExecArgs {
                bank: Some(dir.path().join("bank.jsonl")),
                ..args()
            },
            "with --bank",
        ),
        (
            ExecArgs {
                logit_dump: Some(dir.path().join("logits")),
                ..args()
            },
            "with --logit-dump",
        ),
        (
            ExecArgs {
                dump_layers: Some(dir.path().join("layers")),
                ..args()
            },
            "with --dump-layers",
        ),
        (
            ExecArgs {
                draft_depth: Some(1),
                ..args()
            },
            "--draft-depth",
        ),
        (
            ExecArgs {
                speculate: Some(2),
                ..args()
            },
            "--speculate",
        ),
        (
            ExecArgs {
                stop_at_eos: true,
                ..args()
            },
            "--stop-at-eos",
        ),
    ];
    for (case, expected) in cases {
        let err = run(Vindex3Command::Exec(case)).unwrap_err().to_string();
        assert!(err.contains(expected), "{err} should name {expected}");
    }
    let err = refuse_evidence_outside_the_curve(&args(), true).unwrap_err();
    assert!(err.contains("on a lowered backend"), "{err}");
    assert!(!regions.exists() && !sealed.exists(), "nothing was created");

    // Each evidence flag is enough on its own.
    let sealed_only = ExecArgs {
        residency_regions: None,
        residency_campaign: false,
        verification_manifest: None,
        ..args()
    };
    assert!(refuse_evidence_outside_the_curve(&sealed_only, true).is_err());
    let regions_only = ExecArgs {
        sealed_latency: None,
        residency_campaign: false,
        verification_manifest: None,
        ..args()
    };
    assert!(refuse_evidence_outside_the_curve(&regions_only, true).is_err());

    // A diagnostic curve run may keep flags the curve ignores, as it did,
    // and a run with no evidence flag is none of this check's business.
    let diagnostic = ExecArgs {
        residency_campaign: false,
        verification_manifest: None,
        stop_at_eos: true,
        ..args()
    };
    assert!(refuse_evidence_outside_the_curve(&diagnostic, false).is_ok());
    let plain = ExecArgs {
        residency_regions: None,
        sealed_latency: None,
        residency_campaign: false,
        verification_manifest: None,
        bank: Some(dir.path().join("bank.jsonl")),
        ..args()
    };
    assert!(refuse_evidence_outside_the_curve(&plain, true).is_ok());
}

/// A run id made from the clock dated the attempt's start before binding,
/// which gave away the sealed bind time.
#[test]
fn a_run_id_carries_no_timestamp_and_differs_between_runs() {
    let dir = tempfile::tempdir().unwrap();
    let first = RunIdentity::for_container(dir.path());
    let second = RunIdentity::for_container(dir.path());
    assert_ne!(first.run_id, second.run_id);
    let (pid, random) = first.run_id.split_once('-').unwrap();
    assert_eq!(pid, first.pid.to_string());
    assert_eq!(random.len(), 16, "{random}");
    assert!(u64::from_str_radix(random, 16).is_ok(), "{random}");
}

/// `--repeat 0` runs one pass, so one is what the record says.
#[test]
#[cfg_attr(
    not(unix),
    ignore = "a campaign stops here with no descriptor identity"
)]
fn the_record_says_how_many_passes_ran() {
    let (dir, container) = encoded();
    let regions = dir.path().join("regions.jsonl");
    let args = ExecArgs {
        repeat: 0,
        ..evidence_args(&container, &regions, &dir.path().join("sealed.jsonl"))
    };
    run(Vindex3Command::Exec(args)).unwrap();
    assert_eq!(without_identity(read_lines(&regions))[1]["repeat"], 1);
}

#[test]
fn a_campaign_without_its_evidence_never_binds_a_weight() {
    let (dir, container) = encoded();
    let sealed = dir.path().join("sealed.jsonl");
    let err = run_curve(
        &container,
        Curve {
            sealed: Some(sealed.clone()),
            campaign: true,
            ..Curve::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("needs --residency-regions"), "{err}");
    assert!(!sealed.exists());
}
