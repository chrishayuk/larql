//! **The seam: selected by procedure, located by identity, and unable
//! to re-aim the experiment.**

use std::path::PathBuf;

use super::super::super::compile::hash_bytes;
use super::super::super::measure::{
    TEACHER_FORCED_TWO_ARM, {DEFAULT_GATE, DEFAULT_LABEL, DEFAULT_SEQUENCES},
};
use super::super::super::measurement::EvidenceScale;
use super::super::super::state::evidence_bank::EvidenceBank;
use super::super::super::state::key::MeasurementKey;
use super::super::super::state::protocol::MeasurementProtocol;
use super::super::artifacts::{DeclaredArtifacts, BANK_MANIFEST};
use super::super::executor::{
    ArtifactLocator, ExecutionRefusal, ExecutorRegistry, ExperimentExecutor, LocatorRefusal,
    Observed,
};
use super::super::request::MeasurementRequest;
use super::super::TeacherForcedExecutor;
use super::{fixtures, glimmer, ready, record_with};

// ------------------------------------------------------------ a corpus

/// A corpus directory, and a bank identified over its actual manifest.
fn corpus() -> (tempfile::TempDir, EvidenceBank) {
    let dir = tempfile::tempdir().expect("corpus dir");
    let manifest = br#"{"sequences":256,"positions_per_sequence":32}"#;
    std::fs::write(dir.path().join(BANK_MANIFEST), manifest).expect("write manifest");
    let bank = EvidenceBank::new(
        "kimi-teacher-forced/v1",
        hash_bytes(manifest),
        (0..256).map(|i| format!("seq-{i:03}")),
        32,
    );
    (dir, bank)
}

/// A record over a real container whose bank names a real corpus.
fn record(container: &std::path::Path, bank: EvidenceBank) -> super::SearchSnapshot {
    record_with(
        container,
        MeasurementProtocol::new(bank, fixtures::instrument(), TEACHER_FORCED_TWO_ARM),
    )
}

/// Everything the run needs, all of it real.
fn everything(
    container: &std::path::Path,
    corpus_dir: &std::path::Path,
    request: &MeasurementRequest,
) -> (tempfile::TempDir, DeclaredArtifacts) {
    let overlay = tempfile::tempdir().expect("overlay dir");
    let artifacts = DeclaredArtifacts::new()
        .container_at(container)
        .corpus_at(corpus_dir)
        .overlay_at(request.key().state(), overlay.path());
    (overlay, artifacts)
}

// -------------------------------------------------------- the registry

#[test]
fn the_registry_resolves_by_procedure_and_names_what_it_has() {
    let teacher_forced = TeacherForcedExecutor;
    let registry = ExecutorRegistry::new([&teacher_forced as &dyn ExperimentExecutor])
        .expect("one executor, one procedure");
    assert_eq!(registry.implemented(), vec![TEACHER_FORCED_TWO_ARM]);
    assert_eq!(
        registry
            .for_procedure(TEACHER_FORCED_TWO_ARM)
            .expect("implemented")
            .procedure(),
        TEACHER_FORCED_TWO_ARM
    );

    let Err(refusal) = registry.for_procedure("some-future-procedure/v1") else {
        panic!("this build does not perform that procedure");
    };
    let ExecutionRefusal::NoSuchProcedure { named, implemented } = &refusal else {
        panic!("{refusal:?}");
    };
    assert_eq!(named, "some-future-procedure/v1");
    // Naming what IS implemented is the difference between a refusal
    // that can be acted on and a dead end.
    assert_eq!(implemented, &vec![TEACHER_FORCED_TWO_ARM.to_string()]);
    assert!(refusal.to_string().contains(TEACHER_FORCED_TWO_ARM));
}

/// A stub that claims a procedure and answers about whatever key it was
/// built with — the two ways the seam can be attacked.
///
/// Its note names the procedure it answers to, so a test can prove WHICH
/// executor ran and not merely that one did.
struct Stub {
    procedure: String,
    answers_about: Option<MeasurementKey>,
}

impl ExperimentExecutor for Stub {
    fn procedure(&self) -> &str {
        &self.procedure
    }

    fn execute(
        &self,
        request: &MeasurementRequest,
        _artifacts: &dyn ArtifactLocator,
    ) -> Result<Observed, ExecutionRefusal> {
        Ok(Observed {
            key: self
                .answers_about
                .clone()
                .unwrap_or_else(|| request.key().clone()),
            observation: fixtures::authority_reading(0.0, 0),
            verified: Default::default(),
            execution_note: format!("stub answering to {}", self.procedure),
        })
    }
}

#[test]
fn two_executors_claiming_one_procedure_are_refused() {
    let a = Stub {
        procedure: TEACHER_FORCED_TWO_ARM.into(),
        answers_about: None,
    };
    let b = Stub {
        procedure: TEACHER_FORCED_TWO_ARM.into(),
        answers_about: None,
    };
    let Err(refusal) = ExecutorRegistry::new([&a as &dyn ExperimentExecutor, &b]) else {
        panic!("which one ran would depend on registration order");
    };
    assert!(refusal.to_string().contains(TEACHER_FORCED_TWO_ARM));
}

/// Two procedures, two executors, and the named one runs — in both
/// registration orders, because an order-dependent answer is exactly what
/// the duplicate-claim refusal above exists to prevent and a single
/// registered executor cannot witness.
///
/// **This proves registry DISPATCH and nothing more.** It does not touch
/// ACT1-N5, which stays open: no semantic relationship between an
/// instrument's declared procedure and an execution procedure is claimed
/// or tested here, and two stubs could not establish one.
#[test]
fn two_distinct_procedures_dispatch_by_name_in_either_registration_order() {
    const ALPHA: &str = "alpha-procedure/v1";
    const BETA: &str = "beta-procedure/v1";

    let dir = glimmer();
    let declaring = |procedure: &str| {
        record_with(
            dir.path(),
            MeasurementProtocol::new(
                fixtures::selection_bank(),
                fixtures::instrument(),
                procedure,
            ),
        )
    };
    let alpha_record = declaring(ALPHA);
    let beta_record = declaring(BETA);
    let alpha_request = ready(&alpha_record).request;
    let beta_request = ready(&beta_record).request;
    assert_eq!(alpha_request.procedure(), ALPHA);
    assert_eq!(beta_request.procedure(), BETA);

    let alpha = Stub {
        procedure: ALPHA.into(),
        answers_about: None,
    };
    let beta = Stub {
        procedure: BETA.into(),
        answers_about: None,
    };
    // Nothing is declared to be anywhere: dispatch happens before an
    // executor consults a locator, and these stubs never do.
    let nowhere = DeclaredArtifacts::new();

    let orders: [[&dyn ExperimentExecutor; 2]; 2] = [[&alpha, &beta], [&beta, &alpha]];
    for order in orders {
        let registry = ExecutorRegistry::new(order).expect("two procedures, two executors");
        assert_eq!(
            registry.implemented(),
            vec![ALPHA.to_string(), BETA.to_string()]
        );

        for (request, expected) in [(&alpha_request, ALPHA), (&beta_request, BETA)] {
            let observed = registry
                .execute(request, &nowhere)
                .expect("the procedure the record declares is implemented");
            assert!(
                observed.execution_note.contains(expected),
                "the executor for `{expected}` should have run: {}",
                observed.execution_note
            );
            assert_eq!(&observed.key, request.key());
        }

        // And with two registered, the refusal for a third names both —
        // so the list is a list and not the single entry a one-executor
        // registry could not tell apart from one.
        let Err(refusal) = registry.for_procedure("gamma-procedure/v1") else {
            panic!("this build performs neither of those");
        };
        let ExecutionRefusal::NoSuchProcedure { implemented, .. } = &refusal else {
            panic!("{refusal:?}");
        };
        assert_eq!(implemented, &vec![ALPHA.to_string(), BETA.to_string()]);
    }
}

/// **The seam's own falsifier.** A request that cannot misstate its
/// experiment buys nothing if the layer below may answer about another.
#[test]
fn an_executor_that_answers_about_another_experiment_is_refused() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;
    let (_overlay, artifacts) = everything(dir.path(), corpus_dir.path(), &request);

    let elsewhere = MeasurementKey::new(
        fixtures::p().physical_id(),
        &request.bank().id(),
        EvidenceScale::Authority,
        &request.instrument().id(),
    );
    let liar = Stub {
        procedure: TEACHER_FORCED_TWO_ARM.into(),
        answers_about: Some(elsewhere.clone()),
    };
    let registry =
        ExecutorRegistry::new([&liar as &dyn ExperimentExecutor]).expect("one procedure");

    let refusal = registry
        .execute(&request, &artifacts)
        .expect_err("it answered about something else");
    let ExecutionRefusal::ObservedAnotherExperiment(misdirected) = &refusal else {
        panic!("{refusal:?}");
    };
    assert_eq!(&misdirected.requested, request.key());
    assert_eq!(&misdirected.observed, &elsewhere);

    // And an honest stub is accepted, so the refusal is not vacuous.
    let honest = Stub {
        procedure: TEACHER_FORCED_TWO_ARM.into(),
        answers_about: None,
    };
    let registry =
        ExecutorRegistry::new([&honest as &dyn ExperimentExecutor]).expect("one procedure");
    let observed = registry
        .execute(&request, &artifacts)
        .expect("an observation of the experiment that was requested");
    assert_eq!(&observed.key, request.key());
}

// --------------------------------------------------------- the locator

/// **P5.** A container that merely sits at the expected path is not the
/// container the experiment was priced against.
#[test]
fn a_container_at_the_declared_path_is_still_checked_against_the_record() {
    let dir = glimmer();
    let other = super::container::dense();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;

    let artifacts = DeclaredArtifacts::new()
        .container_at(other.path())
        .corpus_at(corpus_dir.path());
    let refusal = artifacts
        .container(&request)
        .expect_err("that is a different container");
    assert!(
        matches!(refusal, LocatorRefusal::NotWhatItClaims { .. }),
        "{refusal:?}"
    );

    // The right one is accepted, so the check is not refusing everything.
    let artifacts = DeclaredArtifacts::new()
        .container_at(dir.path())
        .corpus_at(corpus_dir.path());
    assert_eq!(
        artifacts.container(&request).expect("the priced container"),
        PathBuf::from(dir.path())
    );
}

#[test]
fn a_corpus_whose_manifest_is_not_the_banks_is_refused() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;

    let artifacts = DeclaredArtifacts::new()
        .container_at(dir.path())
        .corpus_at(corpus_dir.path());
    artifacts
        .corpus(&request)
        .expect("the manifest digests to the bank's own input");

    // The same directory after its manifest moves is not the same bank,
    // and a run over it would not be the experiment that was authorised.
    let moved = b"{}";
    std::fs::write(corpus_dir.path().join(BANK_MANIFEST), moved).expect("rewrite");
    let refusal = artifacts
        .corpus(&request)
        .expect_err("its manifest is not the bank's");
    let LocatorRefusal::NotWhatItClaims { what, path, detail } = &refusal else {
        panic!("{refusal:?}");
    };
    assert_eq!(what, "quality bank");
    assert_eq!(path, &corpus_dir.path().display().to_string());

    // BOTH digests, in the message. The refusal is composed here rather
    // than by the variant's own `Display`, so the type-level contract
    // (`every_locator_refusal_names_what_it_promises`) does not reach it
    // — a mutant that emptied the digest helper survived until this
    // assertion existed.
    let found = hash_bytes(moved);
    for digest in [&found, &request.bank().manifest_sha256] {
        assert!(
            detail.contains(&digest[..12]),
            "the refusal must name both digests for a reader to tell which bank they have;              `{}` is missing from: {detail}",
            &digest[..12]
        );
    }
    assert_ne!(found, request.bank().manifest_sha256, "the two must differ");
}

#[test]
fn an_overlay_that_was_never_compiled_refuses_with_the_map_to_build() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;

    let artifacts = DeclaredArtifacts::new()
        .container_at(dir.path())
        .corpus_at(corpus_dir.path());
    let refusal = artifacts
        .candidate(&request)
        .expect_err("nothing has been compiled");
    let LocatorRefusal::NotBuilt { state, map } = &refusal else {
        panic!("{refusal:?}");
    };
    assert_eq!(state, request.key().state());
    assert_eq!(map, &request.candidate_map().name);
    // The locator must not have built one behind the caller's back.
    assert!(artifacts.overlay_for(request.key().state()).is_none());

    // And it answers about one that WAS declared, so the absence above
    // is an answer about this state rather than a constant. A mutant
    // returning `None` unconditionally survived until this existed.
    let overlay = tempfile::tempdir().expect("overlay dir");
    let holding = DeclaredArtifacts::new()
        .container_at(dir.path())
        .corpus_at(corpus_dir.path())
        .overlay_at(request.key().state(), overlay.path());
    assert_eq!(
        holding.overlay_for(request.key().state()),
        Some(overlay.path())
    );
    assert_eq!(
        holding.candidate(&request).expect("it is built"),
        PathBuf::from(overlay.path())
    );
}

// ---------------------------------------------------------- the adapter

/// **Step 5's whole content.** The instruction a run would be given, and
/// where each field's authority is. Platform-independent on purpose: a
/// mapping only checkable on one machine is a mapping nobody checks.
#[test]
fn the_instruction_carries_the_records_controls_and_none_of_the_defaults() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;
    let (overlay, artifacts) = everything(dir.path(), corpus_dir.path(), &request);

    let instructed = TeacherForcedExecutor
        .instruct(&request, &artifacts)
        .expect("everything it names is here");

    // Paths came from the locator, and each was checked on the way.
    assert_eq!(instructed.source, PathBuf::from(dir.path()));
    assert_eq!(instructed.quality_bank, PathBuf::from(corpus_dir.path()));
    assert_eq!(instructed.candidate, PathBuf::from(overlay.path()));

    // Controls came from the record, and are each observably not the
    // adapter's historic default.
    assert_eq!(instructed.gate, "kimi-logit-balanced-v1");
    assert_ne!(instructed.gate, DEFAULT_GATE);
    assert_eq!(instructed.sequences, 256);
    assert_ne!(instructed.sequences, DEFAULT_SEQUENCES);
    assert_eq!(instructed.label, format!("exp-{}", request.key().short()));
    assert_ne!(instructed.label, DEFAULT_LABEL);

    // And the built request admits under this build, so the instruction
    // is one the procedure would actually accept.
    instructed.admit().expect("a well formed instruction");
}

#[test]
fn an_experiment_naming_another_procedure_is_not_instructed_by_this_one() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record_with(
        dir.path(),
        MeasurementProtocol::new(bank, fixtures::instrument(), "some-future-procedure/v1"),
    );
    let request = ready(&snapshot).request;
    let (_overlay, artifacts) = everything(dir.path(), corpus_dir.path(), &request);

    let refusal = TeacherForcedExecutor
        .instruct(&request, &artifacts)
        .expect_err("this executor performs one procedure");
    assert!(
        matches!(refusal, ExecutionRefusal::NoSuchProcedure { .. }),
        "{refusal:?}"
    );
}

/// A missing artifact is reported as a missing artifact, and not as a
/// missing backend. The build that cannot perform the procedure still
/// instructs first, so an operator is sent to the right problem.
#[test]
fn a_missing_artifact_is_refused_before_the_backend_is_consulted() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;

    // No overlay declared.
    let artifacts = DeclaredArtifacts::new()
        .container_at(dir.path())
        .corpus_at(corpus_dir.path());
    let refusal = TeacherForcedExecutor
        .execute(&request, &artifacts)
        .expect_err("nothing to measure against");
    assert!(
        matches!(
            refusal,
            ExecutionRefusal::Locator(LocatorRefusal::NotBuilt { .. })
        ),
        "{refusal:?}"
    );
}

/// On a build without the instrument, a fully instructable request
/// reaches the procedure and is refused for the backend — naming what
/// the build lacks and which procedure it could not perform.
#[cfg(not(all(feature = "gpu", target_os = "macos")))]
#[test]
fn a_build_without_the_instrument_refuses_after_instructing() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;
    let (_overlay, artifacts) = everything(dir.path(), corpus_dir.path(), &request);

    let refusal = TeacherForcedExecutor
        .execute(&request, &artifacts)
        .expect_err("this build cannot perform it");
    let ExecutionRefusal::Measurement(measurement) = &refusal else {
        panic!("the instruction was fine; the backend is not: {refusal:?}");
    };
    assert!(
        measurement.nothing_was_measured(),
        "no measurement was attempted, so this is an execution failure"
    );
    let said = refusal.to_string();
    assert!(said.contains("gpu"), "{said}");
    assert!(said.contains(TEACHER_FORCED_TWO_ARM), "{said}");
}

/// A locator that was told nothing refuses by IDENTITY, not by path —
/// there is no path to name, and the identity is what a caller would
/// have to go and find.
#[test]
fn a_locator_that_holds_nothing_refuses_naming_the_identity_it_lacks() {
    let dir = glimmer();
    let (_corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;
    let nothing = DeclaredArtifacts::new();

    let LocatorRefusal::NotHeld { what, identity } = nothing
        .container(&request)
        .expect_err("no container was declared")
    else {
        panic!("nothing is held, so nothing can be wrong with it");
    };
    assert_eq!(what, "container");
    assert_eq!(identity, request.model().semantic_digest());

    let LocatorRefusal::NotHeld { what, identity } = nothing
        .corpus(&request)
        .expect_err("no corpus was declared")
    else {
        panic!("nothing is held, so nothing can be wrong with it");
    };
    assert_eq!(what, "quality bank");
    assert_eq!(identity, request.bank().id().to_string());
}

/// A directory that is not a container at all is refused by the same
/// check that refuses the wrong container, carrying the read's own
/// message — so "there is nothing there" and "that is something else"
/// are one question with two answers rather than two code paths.
#[test]
fn a_path_that_is_not_a_container_is_refused_with_the_reads_own_message() {
    let dir = glimmer();
    let (corpus_dir, bank) = corpus();
    let snapshot = record(dir.path(), bank);
    let request = ready(&snapshot).request;

    let empty = tempfile::tempdir().expect("an empty dir");
    let artifacts = DeclaredArtifacts::new()
        .container_at(empty.path())
        .corpus_at(corpus_dir.path());
    let LocatorRefusal::NotWhatItClaims { what, path, detail } = artifacts
        .container(&request)
        .expect_err("there is no index there")
    else {
        panic!("it cannot identify itself, so it is not what it claims");
    };
    assert_eq!(what, "container");
    assert_eq!(path, empty.path().display().to_string());
    assert!(!detail.is_empty(), "the read's own message must survive");
}

/// A bank declaring no samples has no positions, and a gate judging on
/// tail statistics would be reading an empty distribution. Refused at
/// the DECLARATION rather than at the instruction derived from it, so
/// the message names what is actually wrong.
#[test]
fn a_bank_declaring_no_samples_cannot_instruct_a_run() {
    let dir = glimmer();
    let empty_bank = EvidenceBank::new(
        "kimi-teacher-forced/v1",
        hash_bytes(b"{}"),
        Vec::<String>::new(),
        32,
    );
    assert_eq!(empty_bank.positions(), 0);
    let snapshot = record(dir.path(), empty_bank);
    let request = ready(&snapshot).request;
    assert_eq!(request.sequences(), 0);

    // Nothing is declared to be anywhere: the declaration is refused
    // before any artifact is located, so a caller is not sent looking
    // for files that would not have helped.
    let ExecutionRefusal::NotInstructable { procedure, detail } = TeacherForcedExecutor
        .instruct(&request, &DeclaredArtifacts::new())
        .expect_err("a run over no samples measures nothing")
    else {
        panic!("the declaration is what cannot be instructed");
    };
    assert_eq!(procedure, TEACHER_FORCED_TWO_ARM);
    assert!(detail.contains("no samples"), "{detail}");
}
