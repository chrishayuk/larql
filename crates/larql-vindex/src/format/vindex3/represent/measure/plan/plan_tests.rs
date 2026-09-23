//! MEASURE-PLAN-1 PR 2 witnesses, on the dense fixture with the reference
//! backend, so every one runs in CI.
//!
//! Both arms are the production CPU backend, as `vindex3 exec --backend
//! production` and `production-nvfp4` build them. The reference reads the
//! canonical container; the candidate reads a conservative NVFP4 pack
//! compiled from it, under `stored`. So the only changed variable is the
//! representation. W1, W2, W4 and
//! W7 are the frozen witnesses. W5 and W6 have format-level witnesses in
//! PR 1, and are repeated here at procedure level. W3 is the foreign-reference
//! test at the end.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

use super::arm::{ArmDescription, InterpreterArm, TeacherForcedArm};
use super::*;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::plan_component_ops;
use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
use crate::format::vindex3::represent::token_bank::{export, TOKENIZER_FILE};
use crate::format::vindex3::represent::{compile_representation, policy, RepresentSpec};

const COMPONENT: &str = "target";
const REFERENCE_ARM: &str = "production";
const CANDIDATE_ARM: &str = "production-nvfp4";
/// Samples measured, including the null arm's.
const SEQUENCES: usize = 3;
/// Token cap the fixture bank exports with.
const CAP: usize = 10;
/// Words the fixture tokenizer knows. Their ids, 1..=20, are inside the
/// dense fixture's 128-token vocabulary.
const WORDS: [&str; 20] = [
    "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "far", "sun", "rose", "over",
    "hill", "we", "saw", "it", "shine", "then", "slept",
];

fn write_tokenizer(container: &Path, rotate: u32) {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("[UNK]".into(), 0);
    for (i, w) in WORDS.iter().enumerate() {
        vocab.insert((*w).into(), 1 + (i as u32 + rotate) % WORDS.len() as u32);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tk = tokenizers::Tokenizer::new(model);
    tk.with_pre_tokenizer(Some(Whitespace {}));
    tk.save(container.join(TOKENIZER_FILE), false).unwrap();
}

struct Fixture {
    _tmp: tempfile::TempDir,
    source: PathBuf,
    pack: PathBuf,
    bank: PathBuf,
    output: PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let source = tmp.path().join("source.vindex3");
    let pack = tmp.path().join("nvfp4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &source, COMPONENT);
    write_tokenizer(&source, 0);
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
    .expect("the fixture compiles to NVFP4");
    if !pack.join(TOKENIZER_FILE).exists() {
        std::fs::copy(source.join(TOKENIZER_FILE), pack.join(TOKENIZER_FILE)).unwrap();
    }
    let prompts = tmp.path().join("prompts.json");
    let bank_prompts = serde_json::json!({
        "bank": "plan-fixture",
        "prompts": [
            {"id": "prose-000", "category": "prose", "text": "the cat sat on a mat and the dog ran far"},
            {"id": "prose-001", "category": "prose", "text": "the sun rose over the hill"},
            {"id": "longform-000", "category": "longform", "text": "we saw it shine then we slept"},
            {"id": "prose-002", "category": "prose", "text": "a dog sat on the hill"}
        ]
    });
    std::fs::write(&prompts, serde_json::to_vec(&bank_prompts).unwrap()).unwrap();
    let bank = tmp.path().join("bank");
    export(&prompts, &source.join(TOKENIZER_FILE), CAP, &bank).unwrap();
    let output = tmp.path().join("out");
    Fixture {
        source,
        pack,
        bank,
        output,
        _tmp: tmp,
    }
}

fn arm(container: &Path, pack: Option<&str>, name: &str) -> InterpreterArm<ProductionBackend> {
    arm_on(container, pack, name, ProductionBackend::new())
}

fn arm_on<B: PlanBackend>(
    container: &Path,
    pack: Option<&str>,
    name: &str,
    backend: B,
) -> InterpreterArm<B> {
    let inspection = inspect_container(container, false).unwrap();
    let plan = plan_component_ops(&inspection, container, COMPONENT)
        .unwrap()
        .plan
        .expect("a plan");
    let source = if pack.is_some() {
        RepresentationSource::Stored
    } else {
        RepresentationSource::Auto
    };
    let store = OperandStore::open_for(container, &inspection, pack, source).unwrap();
    InterpreterArm::prepare(
        name,
        container.to_path_buf(),
        pack.map(str::to_string),
        pack.is_some(),
        plan,
        &store,
        backend,
    )
    .unwrap()
}

fn request(f: &Fixture) -> PlanMeasureRequest {
    PlanMeasureRequest {
        bank: f.bank.clone(),
        sequences: SEQUENCES,
        label: "fixture".into(),
        output: f.output.clone(),
        provenance: [("larql_version".to_string(), "fixture".to_string())].into(),
    }
}

fn reference_and_candidate(
    f: &Fixture,
) -> (
    InterpreterArm<ProductionBackend>,
    InterpreterArm<ProductionBackend>,
) {
    (
        arm(&f.source, None, REFERENCE_ARM),
        arm(&f.pack, Some(DTYPE_NVFP4), CANDIDATE_ARM),
    )
}

fn inadmissible(result: Result<PlanReceipt, PlanRefusal>) -> PlanInadmissible {
    match result {
        Err(PlanRefusal::Inadmissible(i)) => i,
        other => panic!("expected an inadmissible run, got {other:?}"),
    }
}

/// Flip the last byte of a segment file, which lies in its payload region.
///
/// The file is replaced, never written in place: `represent` hard-links the
/// segments it carries unchanged, so an in-place write would change the
/// source container too, and the two would still agree.
fn tamper(container: &Path, segment: &str) {
    let path = container.join("segments").join(segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, bytes).unwrap();
}

#[test]
fn an_admissible_run_measures_every_position_and_writes_its_record() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let receipt = run(&request(&f), &mut r, &mut c).expect("admissible");
    assert_eq!(receipt.procedure, PROCEDURE);
    assert!(receipt.facts.complete(SEQUENCES), "{:?}", receipt.facts);
    assert_eq!(
        receipt.facts.null_arm_samples,
        SEQUENCES.min(NULL_ARM_SAMPLES)
    );
    assert_eq!(receipt.facts.attributed_arms, 2);
    assert_eq!(receipt.facts.tokenizer_checked_arms, 2);
    assert_eq!(receipt.facts.bank_samples_read, SEQUENCES);
    assert!(receipt
        .facts
        .changed_representations
        .iter()
        .all(|r| r.ends_with(DTYPE_NVFP4)));
    assert!(!receipt.facts.changed_representations.is_empty());
    let bank = crate::format::vindex3::represent::token_bank::TokenBank::open(&f.bank).unwrap();
    let positions: usize = (0..SEQUENCES).map(|i| bank.read(i).unwrap().len()).sum();
    assert_eq!(receipt.facts.positions, positions as u64);
    assert_eq!(receipt.summary.all.positions, positions);
    assert!(receipt.summary.all.kl_mean > 0.0, "NVFP4 is lossy");
    for file in [REPORT_FILE, POSITIONS_FILE, RECEIPT_FILE] {
        assert!(f.output.join(file).exists(), "{file}");
    }
    let lines = std::fs::read_to_string(f.output.join(POSITIONS_FILE)).unwrap();
    assert_eq!(lines.lines().count(), positions);
}

/// An arm that returns different logits the second time it scores sample 0,
/// as a non-deterministic device would.
struct Noisy<A> {
    inner: A,
    calls: usize,
}

impl<A: TeacherForcedArm> TeacherForcedArm for Noisy<A> {
    fn describe(&self) -> &ArmDescription {
        self.inner.describe()
    }
    fn score(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let mut logits = self.inner.score(ids)?;
        self.calls += 1;
        if self.calls == 2 {
            logits[0][0] = f32::from_bits(logits[0][0].to_bits() ^ 1);
        }
        Ok(logits)
    }
}

/// W1: one flipped ulp in the reference's second pass is refused.
#[test]
fn the_null_arm_catches_a_non_deterministic_reference() {
    let f = fixture();
    let (r, mut c) = reference_and_candidate(&f);
    let mut r = Noisy { inner: r, calls: 0 };
    assert_eq!(
        inadmissible(run(&request(&f), &mut r, &mut c)),
        PlanInadmissible::NullArmNotZero {
            sample: 0,
            position: 0
        }
    );
}

/// W2: the same container through the same arm is not an experiment.
#[test]
fn identity_is_not_a_measurement() {
    let f = fixture();
    let mut r = arm(&f.source, None, REFERENCE_ARM);
    let mut c = arm(&f.source, None, REFERENCE_ARM);
    assert_eq!(
        inadmissible(run(&request(&f), &mut r, &mut c)),
        PlanInadmissible::CandidateCompilesNothing
    );
}

/// W4: a byte changed in a representation both arms bind is named.
#[test]
fn a_changed_protected_representation_is_refused_by_name() {
    let f = fixture();
    tamper(&f.pack, "target.embedding.bin");
    let (mut r, mut c) = reference_and_candidate(&f);
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::ProtectedOperandChanged { representation } => {
            assert!(
                representation.starts_with("target.embedding@"),
                "{representation}"
            )
        }
        other => panic!("expected ProtectedOperandChanged, got {other:?}"),
    }
}

/// The candidate's own representation is checked against its seal.
#[test]
fn a_changed_candidate_representation_breaks_its_seal() {
    let f = fixture();
    tamper(&f.pack, "target.decoder_stack@NVFP4.bin");
    let (mut r, mut c) = reference_and_candidate(&f);
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::SealMismatch { what, .. } => {
            assert_eq!(what, "target.decoder_stack@NVFP4")
        }
        other => panic!("expected SealMismatch, got {other:?}"),
    }
}

/// W5 at procedure level: a tampered bank payload.
#[test]
fn a_tampered_bank_payload_is_refused() {
    let f = fixture();
    let payload = f.bank.join("seq-001.u32");
    let mut bytes = std::fs::read(&payload).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&payload, bytes).unwrap();
    let (mut r, mut c) = reference_and_candidate(&f);
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::SealMismatch { what, .. } => assert_eq!(what, "seq-001"),
        other => panic!("expected SealMismatch, got {other:?}"),
    }
}

/// W6 at procedure level: an arm whose container has another tokenizer.
#[test]
fn a_bank_for_another_tokenizer_is_refused() {
    let f = fixture();
    write_tokenizer(&f.pack, 7);
    let (mut r, mut c) = reference_and_candidate(&f);
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::CorpusNotForThisModel { arm, .. } => assert_eq!(arm, CANDIDATE_ARM),
        other => panic!("expected CorpusNotForThisModel, got {other:?}"),
    }
}

/// An arm reporting what a real arm would have bound, with one fact edited.
struct Redescribed<A> {
    inner: A,
    description: ArmDescription,
}

impl<A: TeacherForcedArm> TeacherForcedArm for Redescribed<A> {
    fn describe(&self) -> &ArmDescription {
        &self.description
    }
    fn score(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        self.inner.score(ids)
    }
}

/// W7: a stored-only arm that quantised at load is not measured.
#[test]
fn quantising_at_load_under_stored_is_refused() {
    let f = fixture();
    let (mut r, c) = reference_and_candidate(&f);
    let mut description = c.describe().clone();
    description.runtime_quantised = 1;
    let mut c = Redescribed {
        inner: c,
        description,
    };
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::UnexpectedPhysicalRead { arm, .. } => assert_eq!(arm, CANDIDATE_ARM),
        other => panic!("expected UnexpectedPhysicalRead, got {other:?}"),
    }
}

/// W7's other direction: an arm that asked for no pack but read one.
#[test]
fn reading_a_pack_nobody_asked_for_is_refused() {
    let f = fixture();
    let (r, mut c) = reference_and_candidate(&f);
    let mut description = r.describe().clone();
    let object = description.objects.keys().next().unwrap().clone();
    description.objects.get_mut(&object).unwrap().stored = true;
    let mut r = Redescribed {
        inner: r,
        description,
    };
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::UnexpectedPhysicalRead { arm, .. } => assert_eq!(arm, REFERENCE_ARM),
        other => panic!("expected UnexpectedPhysicalRead, got {other:?}"),
    }
}

#[test]
fn a_request_for_more_samples_than_the_bank_holds_is_refused() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let mut req = request(&f);
    req.sequences = 99;
    assert!(matches!(
        run(&req, &mut r, &mut c),
        Err(PlanRefusal::Execution(
            PlanExecutionFailure::RequestRefused { .. }
        ))
    ));
}

#[test]
fn a_refused_run_still_writes_a_receipt_naming_the_refusal() {
    let f = fixture();
    let mut r = arm(&f.source, None, REFERENCE_ARM);
    let mut c = arm(&f.source, None, REFERENCE_ARM);
    let _ = run(&request(&f), &mut r, &mut c);
    let receipt = std::fs::read_to_string(f.output.join(RECEIPT_FILE)).unwrap();
    assert!(receipt.contains("CandidateCompilesNothing"), "{receipt}");
    assert!(!f.output.join(REPORT_FILE).exists());
}

// ── W3: a foreign reference ────────────────────────────────────────────
//
// The procedure's metrics against an independent numpy computation over
// the same logits. The logits are real, not synthetic: the fixture's two
// arms (`production` on the canonical container, `production-nvfp4` on
// the pack) teacher-forced over three bank samples, the same path as
// `vindex3 exec --logit-dump`. They are committed, so the comparison is a
// comparison of arithmetic and runs on every platform whatever its BLAS.
//
// Regenerate in this order:
//   cargo test -p larql-vindex --lib write_w3_logits -- --ignored
//   python3 scripts/measure_plan_w3_reference.py

/// Where W3's committed files live, relative to the crate.
const W3_DIR: &str = "src/format/vindex3/represent/fixtures/measure-plan-w3";

/// Samples W3 scores.
const W3_SAMPLES: usize = 3;

/// W3's agreement bound per position, in nats (and absolute for margin and
/// entropy), fixed by the freeze.
const W3_TOLERANCE: f64 = 1e-9;

#[test]
#[ignore = "writes W3's committed fixture; run only to regenerate it"]
fn write_w3_logits() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let bank = crate::format::vindex3::represent::token_bank::TokenBank::open(&f.bank).unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(W3_DIR);
    std::fs::create_dir_all(&dir).unwrap();
    let mut reference = Vec::new();
    let mut candidate = Vec::new();
    let mut samples = Vec::new();
    for i in 0..W3_SAMPLES {
        let ids = bank.read(i).unwrap();
        for row in r.score(&ids).unwrap() {
            reference.extend(row.iter().flat_map(|v| v.to_le_bytes()));
        }
        for row in c.score(&ids).unwrap() {
            candidate.extend(row.iter().flat_map(|v| v.to_le_bytes()));
        }
        samples.push(serde_json::json!({
            "ids": ids,
            "category": bank.manifest().samples[i].category,
        }));
    }
    std::fs::write(dir.join("reference.f32"), reference).unwrap();
    std::fs::write(dir.join("candidate.f32"), candidate).unwrap();
    let meta = serde_json::json!({
        "vocab": crate::format::vindex3::fixtures::DENSE_VOCAB,
        "reference_arm": REFERENCE_ARM,
        "candidate_arm": CANDIDATE_ARM,
        "samples": samples,
    });
    std::fs::write(
        dir.join("samples.json"),
        serde_json::to_vec_pretty(&meta).unwrap(),
    )
    .unwrap();
}

/// W3: every per-position metric agrees with numpy.
#[test]
fn the_metrics_agree_with_an_independent_numpy_computation() {
    use super::metrics::position_metrics;
    let samples: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../fixtures/measure-plan-w3/samples.json"
    ))
    .unwrap();
    let expected: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../fixtures/measure-plan-w3/expected.json"
    ))
    .unwrap();
    let floats = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let reference = floats(include_bytes!(
        "../../fixtures/measure-plan-w3/reference.f32"
    ));
    let candidate = floats(include_bytes!(
        "../../fixtures/measure-plan-w3/candidate.f32"
    ));
    let vocab = samples["vocab"].as_u64().unwrap() as usize;
    let rows = expected["positions"].as_array().unwrap();
    assert_eq!(reference.len(), rows.len() * vocab);

    let mut row = 0;
    for sample in samples["samples"].as_array().unwrap() {
        let ids: Vec<u32> = sample["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        for position in 0..ids.len() {
            let span = row * vocab..(row + 1) * vocab;
            let ours = position_metrics(
                &reference[span.clone()],
                &candidate[span],
                ids.get(position + 1).copied(),
            )
            .unwrap();
            let theirs = &rows[row];
            let close = |ours: f64, key: &str| {
                let v = theirs[key].as_f64().unwrap();
                assert!(
                    (ours - v).abs() <= W3_TOLERANCE,
                    "row {row} {key}: ours {ours:e}, numpy {v:e}"
                );
            };
            close(ours.kl, "kl");
            close(ours.reference_margin, "reference_margin");
            close(ours.reference_entropy, "reference_entropy");
            close(ours.max_abs_delta, "max_abs_delta");
            close(ours.mean_abs_delta, "mean_abs_delta");
            match (ours.delta_nll, theirs["delta_nll"].as_f64()) {
                (Some(a), Some(_)) => close(a, "delta_nll"),
                (None, None) => {}
                other => panic!("row {row}: delta_nll presence differs: {other:?}"),
            }
            assert_eq!(
                ours.top1_agree,
                theirs["top1_agree"].as_bool().unwrap(),
                "row {row}"
            );
            assert_eq!(
                ours.top5_overlap as u64,
                theirs["top5_overlap"].as_u64().unwrap(),
                "row {row}"
            );
            row += 1;
        }
    }
    assert_eq!(row, rows.len());
    // The comparison is not vacuous: the two arms really differ.
    assert!(rows.iter().any(|r| r["kl"].as_f64().unwrap() > 0.0));
}

// ── the other refusals ───────────────────────────────────────────────────

/// How a [`Faulty`] arm misbehaves when it scores.
#[derive(Clone, Copy)]
enum Fault {
    Refuses,
    DropsAPosition,
    ReturnsNan,
}

struct Faulty<A> {
    inner: A,
    fault: Fault,
}

impl<A: TeacherForcedArm> TeacherForcedArm for Faulty<A> {
    fn describe(&self) -> &ArmDescription {
        self.inner.describe()
    }
    fn score(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let mut logits = self.inner.score(ids)?;
        match self.fault {
            Fault::Refuses => return Err("the device went away".into()),
            Fault::DropsAPosition => {
                logits.pop();
            }
            Fault::ReturnsNan => logits[0][0] = f32::NAN,
        }
        Ok(logits)
    }
}

fn execution_failure(result: Result<PlanReceipt, PlanRefusal>) -> PlanExecutionFailure {
    match result {
        Err(PlanRefusal::Execution(e)) => e,
        other => panic!("expected an execution failure, got {other:?}"),
    }
}

#[test]
fn zero_sequences_is_not_a_run() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let mut req = request(&f);
    req.sequences = 0;
    assert!(matches!(
        execution_failure(run(&req, &mut r, &mut c)),
        PlanExecutionFailure::RequestRefused { .. }
    ));
}

#[test]
fn an_earlier_record_is_never_overwritten() {
    let f = fixture();
    std::fs::create_dir_all(&f.output).unwrap();
    std::fs::write(f.output.join(RECEIPT_FILE), b"an earlier run").unwrap();
    let (mut r, mut c) = reference_and_candidate(&f);
    assert!(matches!(
        execution_failure(run(&request(&f), &mut r, &mut c)),
        PlanExecutionFailure::RequestRefused { .. }
    ));
    assert_eq!(
        std::fs::read(f.output.join(RECEIPT_FILE)).unwrap(),
        b"an earlier run"
    );
}

#[test]
fn an_arm_that_cannot_score_is_an_execution_failure() {
    let f = fixture();
    let (r, mut c) = reference_and_candidate(&f);
    let mut r = Faulty {
        inner: r,
        fault: Fault::Refuses,
    };
    match execution_failure(run(&request(&f), &mut r, &mut c)) {
        PlanExecutionFailure::StepRefused { arm, detail } => {
            assert_eq!(arm, REFERENCE_ARM);
            assert!(detail.contains("went away"), "{detail}");
        }
        other => panic!("expected StepRefused, got {other:?}"),
    }
}

#[test]
fn a_candidate_that_drops_a_position_is_inadmissible() {
    let f = fixture();
    let (mut r, c) = reference_and_candidate(&f);
    let mut c = Faulty {
        inner: c,
        fault: Fault::DropsAPosition,
    };
    assert!(matches!(
        inadmissible(run(&request(&f), &mut r, &mut c)),
        PlanInadmissible::PositionCountMismatch { sample: 0, .. }
    ));
}

#[test]
fn a_non_finite_logit_is_an_execution_failure() {
    let f = fixture();
    let (mut r, c) = reference_and_candidate(&f);
    let mut c = Faulty {
        inner: c,
        fault: Fault::ReturnsNan,
    };
    assert!(matches!(
        execution_failure(run(&request(&f), &mut r, &mut c)),
        PlanExecutionFailure::StepRefused { .. }
    ));
}

#[test]
fn an_edited_bank_manifest_breaks_the_bank_seal() {
    let f = fixture();
    let path = f.bank.join("manifest.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    v["max_tokens"] = serde_json::json!(CAP + 1);
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    let (mut r, mut c) = reference_and_candidate(&f);
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::SealMismatch { what, .. } => assert_eq!(what, "bank manifest"),
        other => panic!("expected SealMismatch, got {other:?}"),
    }
}

#[test]
fn an_unreadable_bank_or_index_is_an_execution_failure() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let mut req = request(&f);
    req.bank = f.bank.with_file_name("no-such-bank");
    assert!(matches!(
        execution_failure(run(&req, &mut r, &mut c)),
        PlanExecutionFailure::ArtifactUnreadable { .. }
    ));

    let g = fixture();
    let (mut r, mut c) = reference_and_candidate(&g);
    std::fs::remove_file(g.pack.join(INDEX_JSON)).unwrap();
    assert!(matches!(
        execution_failure(run(&request(&g), &mut r, &mut c)),
        PlanExecutionFailure::ArtifactUnreadable { .. }
    ));
}

/// W7's third case: a pack of an encoding the arm did not ask for.
#[test]
fn reading_a_pack_of_another_encoding_is_refused() {
    let f = fixture();
    let (mut r, c) = reference_and_candidate(&f);
    let mut description = c.describe().clone();
    let object = description
        .objects
        .iter()
        .find(|(_, b)| b.stored)
        .map(|(o, _)| o.clone())
        .expect("the candidate binds a pack");
    description.objects.get_mut(&object).unwrap().encoding = "MXFP4".into();
    let mut c = Redescribed {
        inner: c,
        description,
    };
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::UnexpectedPhysicalRead { arm, detail } => {
            assert_eq!(arm, CANDIDATE_ARM);
            assert!(detail.contains("asked for NVFP4"), "{detail}");
        }
        other => panic!("expected UnexpectedPhysicalRead, got {other:?}"),
    }
}

/// A container that declares a compiled program, read through canonical
/// bytes by a differently named arm: once the reference against itself in
/// disguise, scored KL 0.00000. Refused, not measured.
#[test]
fn a_candidate_that_skips_its_declared_program_is_refused() {
    let f = fixture();
    let mut r = arm(&f.source, None, REFERENCE_ARM);
    let mut c = arm(&f.pack, None, "production-canonical");
    match inadmissible(run(&request(&f), &mut r, &mut c)) {
        PlanInadmissible::UnexpectedPhysicalRead { arm, detail } => {
            assert_eq!(arm, "production-canonical");
            assert!(detail.contains("declares a NVFP4 program"), "{detail}");
        }
        other => panic!("expected UnexpectedPhysicalRead, got {other:?}"),
    }
}

#[test]
fn the_report_records_provenance_bytes_and_programs() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    run(&request(&f), &mut r, &mut c).expect("admissible");
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(f.output.join(REPORT_FILE)).unwrap()).unwrap();
    assert_eq!(report["provenance"]["larql_version"], "fixture");
    assert_eq!(
        report["candidate"]["precision_map"]["encoding"],
        DTYPE_NVFP4
    );
    assert!(report["reference"]["precision_map"].is_null());
    let reps = report["candidate"]["representations"].as_object().unwrap();
    assert!(reps
        .values()
        .all(|r| r["payload_bytes"].as_u64().unwrap() > 0));
}
