//! `vindex3 exec --logit-dump … --capture-sites` (PREC-DIAG-1 E0a) on the
//! encoded fixture: the logit dump is byte-identical with and without
//! the capture on both interpreter backends; a complete capture writes a
//! manifest whose files and digests match what is on disk; an over-budget
//! capture records `resource_limit` and leaves no plane behind; and the
//! request is checked before the weights load.

use sha2::{Digest, Sha256};

use super::super::args::SiteCaptureArgs;
use super::super::site_capture_dump::{parse_positions, tokens_sha256, MANIFEST, RESULT};
use super::super::{run, EncodeArgs, ExecArgs, ExecBackend, Vindex3Command};
use super::fixture_dir;

const TOKENS: &str = "1,2,3";
const SEQUENCE: usize = 3;
const POSITIONS: &str = "0,2";

fn exec_args(
    container: std::path::PathBuf,
    backend: ExecBackend,
    logit_dump: std::path::PathBuf,
    capture: SiteCaptureArgs,
) -> ExecArgs {
    ExecArgs {
        container,
        component: "target".to_string(),
        tokens: TOKENS.to_string(),
        dump_layers: None,
        resume: false,
        backend,
        representation_source: "auto".to_string(),
        generate: None,
        speculate: None,
        stop_at_eos: false,
        residency_curve: false,
        repeat: 1,
        warmup: 0,
        unquiet_ok: true,
        residency_regions: None,
        sealed_latency: None,
        residency_campaign: false,
        verification_manifest: None,
        admission_threshold_bytes: None,
        admission_basis: None,
        expert_access: "demand".to_string(),
        logit_dump: Some(logit_dump),
        prefill_logit_dump: None,
        prefill_lengths: None,
        prefill_method: super::super::prefill_dump::PrefillMethod::Batched,
        gemv_witness: None,
        gemv_witness_max_operands: 1,
        capture,
        rope_angles: Default::default(),
        bank: None,
        dump_dir: None,
        draft_depth: None,
        profile: false,
        plugin: Default::default(),
    }
}

fn encoded_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = fixture_dir(true);
    let container = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: container.clone(),
    }))
    .unwrap();
    (dir, container)
}

fn capture(dir: &std::path::Path, budget: Option<u64>) -> SiteCaptureArgs {
    SiteCaptureArgs {
        capture_sites: Some(dir.to_path_buf()),
        capture_positions: Some(POSITIONS.to_string()),
        capture_budget_bytes: budget,
    }
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn read_json(path: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_logit_dump_is_byte_identical_with_and_without_capture_on_both_backends() {
    let (dir, container) = encoded_fixture();
    for (backend, name) in [
        (ExecBackend::Reference, "reference"),
        (ExecBackend::Production, "production"),
    ] {
        let plain = dir.path().join(format!("{name}-plain.f32"));
        let observed = dir.path().join(format!("{name}-observed.f32"));
        let sites = dir.path().join(format!("{name}-sites"));
        run(Vindex3Command::Exec(exec_args(
            container.clone(),
            backend,
            plain.clone(),
            SiteCaptureArgs::default(),
        )))
        .unwrap();
        run(Vindex3Command::Exec(exec_args(
            container.clone(),
            backend,
            observed.clone(),
            capture(&sites, None),
        )))
        .unwrap();
        assert_eq!(
            std::fs::read(&plain).unwrap(),
            std::fs::read(&observed).unwrap(),
            "{name}: the capture changed the logit dump"
        );

        // The one authority: `complete`, bound to these manifest bytes.
        let result = read_json(&sites.join(RESULT));
        assert_eq!(result["status"], "complete");
        let manifest_bytes = std::fs::read(sites.join(MANIFEST)).unwrap();
        assert_eq!(result["detail"]["manifest_sha256"], hex(&manifest_bytes));
        let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
        assert_eq!(manifest["positions"], serde_json::json!([0, 2]));
        assert_eq!(
            manifest["logit_dump_sha256"],
            hex(&std::fs::read(&observed).unwrap())
        );
        let identity = &manifest["identity"];
        assert_eq!(identity, &result["identity"]);
        assert_eq!(identity["binary_sha256"].as_str().unwrap().len(), 64);
        assert_eq!(
            identity["container_index_sha256"].as_str().unwrap().len(),
            64
        );
        assert_eq!(identity["tokens_sha256"], tokens_sha256(&[1, 2, 3]));
        assert_eq!(identity["backend"], format!("{backend:?}"));
        assert!(!manifest["projection_plans"].as_array().unwrap().is_empty());
        let leftovers: Vec<_> = std::fs::read_dir(&sites)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no temporary file is left");
        let files = manifest["files"].as_array().unwrap();
        assert!(!files.is_empty());
        let mut total = 0u64;
        for file in files {
            let bytes = std::fs::read(sites.join(file["file"].as_str().unwrap())).unwrap();
            let expected = file["rows"].as_u64().unwrap() * file["width"].as_u64().unwrap() * 4;
            assert_eq!(bytes.len() as u64, expected, "{file}");
            assert_eq!(file["sha256"], hex(&bytes), "{file}");
            total += expected;
        }
        assert_eq!(manifest["bytes"].as_u64().unwrap(), total);
    }
}

#[test]
fn an_over_budget_capture_records_resource_limit_and_writes_no_plane() {
    let (dir, container) = encoded_fixture();
    let sites = dir.path().join("sites");
    let err = run(Vindex3Command::Exec(exec_args(
        container,
        ExecBackend::Reference,
        dir.path().join("logits.f32"),
        capture(&sites, Some(1)),
    )))
    .unwrap_err();
    assert!(err.to_string().starts_with("resource_limit:"), "{err}");
    let result = read_json(&sites.join(RESULT));
    assert_eq!(result["status"], "resource_limit");
    assert_eq!(result["detail"]["budget_bytes"], 1);
    assert!(result["detail"]["required_bytes"].as_u64().unwrap() > 1);
    let entries: Vec<_> = std::fs::read_dir(&sites).unwrap().collect();
    assert_eq!(entries.len(), 1, "only the result record is left");
    assert!(!sites.join(MANIFEST).exists());
}

#[test]
fn a_run_that_fails_after_the_capture_began_records_fault_not_in_progress() {
    let (dir, container) = encoded_fixture();
    let sites = dir.path().join("sites");
    let mut args = exec_args(
        container,
        ExecBackend::Reference,
        dir.path().join("logits.f32"),
        capture(&sites, None),
    );
    // A token outside the embedding table fails the step after the
    // capture directory and its in-progress record exist.
    args.tokens = "1,2,4000000".to_string();
    let err = run(Vindex3Command::Exec(args)).unwrap_err();
    let result = read_json(&sites.join(RESULT));
    assert_eq!(result["status"], "fault", "{err}");
    assert!(result["message"]
        .as_str()
        .unwrap()
        .contains("the run failed"));
    assert!(!sites.join(MANIFEST).exists());
}

#[test]
fn an_existing_capture_directory_is_refused_before_the_run() {
    let (dir, container) = encoded_fixture();
    let sites = dir.path().join("sites");
    std::fs::create_dir(&sites).unwrap();
    let logits = dir.path().join("logits.f32");
    let err = run(Vindex3Command::Exec(exec_args(
        container,
        ExecBackend::Reference,
        logits.clone(),
        capture(&sites, None),
    )))
    .unwrap_err();
    assert!(err.to_string().contains("must be new"), "{err}");
    assert!(!logits.exists(), "nothing ran");
}

#[test]
fn capture_positions_are_refused_not_repaired() {
    assert_eq!(parse_positions(" 0, 2", SEQUENCE).unwrap(), vec![0, 2]);
    for bad in ["", "2,1", "1,1", "3", "0,x"] {
        assert!(parse_positions(bad, SEQUENCE).is_err(), "{bad:?}");
    }
}
