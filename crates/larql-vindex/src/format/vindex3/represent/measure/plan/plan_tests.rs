//! MEASURE-PLAN-1 PR 2 witnesses, on the dense fixture with the reference
//! backend, so every one runs in CI.
//!
//! The reference arm reads the canonical container. The candidate reads a
//! conservative NVFP4 pack compiled from it, under `stored`. W1, W2, W4 and
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
use crate::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::plan_component_ops;
use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
use crate::format::vindex3::represent::token_bank::{export, TOKENIZER_FILE};
use crate::format::vindex3::represent::{compile_representation, policy, RepresentSpec};

const COMPONENT: &str = "target";
const REFERENCE_ARM: &str = "reference";
const CANDIDATE_ARM: &str = "reference-nvfp4-stored";
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

fn arm(container: &Path, pack: Option<&str>, name: &str) -> InterpreterArm<ReferenceBackend> {
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
        ReferenceBackend::new(),
    )
    .unwrap()
}

fn request(f: &Fixture) -> PlanMeasureRequest {
    PlanMeasureRequest {
        bank: f.bank.clone(),
        sequences: SEQUENCES,
        label: "fixture".into(),
        output: f.output.clone(),
    }
}

fn reference_and_candidate(
    f: &Fixture,
) -> (
    InterpreterArm<ReferenceBackend>,
    InterpreterArm<ReferenceBackend>,
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
fn tamper(container: &Path, segment: &str) {
    let path = container.join("segments").join(segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
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
            assert_eq!(representation, "target.embedding@BF16")
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
