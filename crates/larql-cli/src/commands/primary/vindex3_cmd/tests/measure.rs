//! `vindex3 measure` end to end on the dense fixture: the canonical
//! container through `production` against a compiled NVFP4 pack through
//! `production-nvfp4`, exactly as a user would run it. The procedure's own
//! witnesses are in `represent/measure/plan/plan_tests.rs`; this pins that
//! the verb builds the arms `exec` would, and reports a refusal as an error.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::represent::measure::plan::{
    POSITIONS_FILE, RECEIPT_FILE, REPORT_FILE,
};
use larql_vindex::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
use larql_vindex::format::vindex3::represent::token_bank::{export, TOKENIZER_FILE};
use larql_vindex::format::vindex3::represent::{compile_representation, policy, RepresentSpec};

use crate::commands::primary::vindex3_cmd::measure::{run, MeasureArgs};
use crate::commands::primary::vindex3_cmd::ExecBackend;

/// Words the fixture tokenizer knows; ids 1..=8 are inside the dense
/// fixture's 128-token vocabulary.
const WORDS: [&str; 8] = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran"];

/// Samples the verb measures.
const SEQUENCES: usize = 2;

/// Token cap the fixture bank exports with.
const CAP: usize = 8;

struct Fixture {
    _tmp: tempfile::TempDir,
    source: PathBuf,
    pack: PathBuf,
    bank: PathBuf,
    root: PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let checkpoint = root.join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let source = root.join("source.vindex3");
    let pack = root.join("nvfp4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &source, "target");
    super::write_word_tokenizer(&source, &WORDS);
    compile_representation(
        &source,
        &pack,
        &RepresentSpec {
            encoding: DTYPE_NVFP4.to_string(),
            objects: Vec::new(),
            roles: policy::RolePolicy::default(),
            deployment: false,
            protect: policy::Protections::default(),
        },
    )
    .unwrap();
    if !pack.join(TOKENIZER_FILE).exists() {
        std::fs::copy(source.join(TOKENIZER_FILE), pack.join(TOKENIZER_FILE)).unwrap();
    }
    let prompts = root.join("prompts.json");
    let body = serde_json::json!({
        "bank": "cli-measure-fixture",
        "prompts": [
            {"id": "prose-000", "category": "prose", "text": "the cat sat on a mat"},
            {"id": "prose-001", "category": "prose", "text": "a dog ran on the mat"}
        ]
    });
    std::fs::write(&prompts, serde_json::to_vec(&body).unwrap()).unwrap();
    let bank = root.join("bank");
    export(&prompts, &source.join(TOKENIZER_FILE), CAP, &bank).unwrap();
    Fixture {
        _tmp: tmp,
        source,
        pack,
        bank,
        root,
    }
}

fn args(f: &Fixture, candidate: &Path, backend: ExecBackend, output: &str) -> MeasureArgs {
    MeasureArgs {
        reference: f.source.clone(),
        reference_backend: ExecBackend::Production,
        reference_source: "auto".into(),
        candidate: candidate.to_path_buf(),
        candidate_backend: backend,
        candidate_source: "stored".into(),
        bank: f.bank.clone(),
        sequences: SEQUENCES,
        label: output.into(),
        output: f.root.join(output),
        component: "target".into(),
        provenance: vec![("source_commit".into(), "fixture".into())],
    }
}

#[test]
fn measure_runs_the_procedure_on_the_arms_exec_would_build() {
    let f = fixture();
    run(args(&f, &f.pack, ExecBackend::ProductionNvfp4, "run")).expect("admissible");
    let out = f.root.join("run");
    for file in [REPORT_FILE, POSITIONS_FILE, RECEIPT_FILE] {
        assert!(out.join(file).exists(), "{file}");
    }
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join(RECEIPT_FILE)).unwrap()).unwrap();
    assert_eq!(receipt["admissible"], true);
    assert_eq!(receipt["receipt"]["reference"]["arm"], "production");
    assert_eq!(receipt["receipt"]["candidate"]["arm"], "production-nvfp4");
    assert_eq!(
        receipt["receipt"]["candidate"]["requested_pack"],
        DTYPE_NVFP4
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join(REPORT_FILE)).unwrap()).unwrap();
    assert_eq!(report["provenance"]["source_commit"], "fixture");
    assert_eq!(
        report["provenance"]["larql_version"],
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn a_refused_measurement_is_an_error_exit_with_a_receipt() {
    let f = fixture();
    let err = run(args(&f, &f.source, ExecBackend::Production, "same"))
        .unwrap_err()
        .to_string();
    assert!(err.contains("CandidateCompilesNothing"), "{err}");
    let receipt = std::fs::read_to_string(f.root.join("same").join(RECEIPT_FILE)).unwrap();
    assert!(receipt.contains("CandidateCompilesNothing"), "{receipt}");
}
