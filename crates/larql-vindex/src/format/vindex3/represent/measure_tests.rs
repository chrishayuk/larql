//! **5a-0: every field a prepared experiment will seal is a control the
//! executor consumes.**
//!
//! The runner used to hold `kimi_logit_v3()` as a literal while an
//! optimiser record declared `kimi-logit-balanced-v1`. Sealing a
//! "measurement protocol" into a hash while that was true would have
//! attested to declared intent rather than to caused execution.
//!
//! What these tests can establish here is the contract: the controls
//! exist, they resolve by name, an unknown name is refused, and the
//! environment form is an ADAPTER onto the same request rather than a
//! second way to run. What they cannot establish is that the measured
//! NUMBERS move — that needs the 48 B container and a Metal device, and
//! saying so beats a test that would pass without one.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::measure::{
    MeasurementProcedure, TeacherForcedRequest, BANK_ENV, CANDIDATE_ENV, DEFAULT_GATE,
    DEFAULT_LABEL, DEFAULT_SEQUENCES, GATE_ENV, LABEL_ENV, SEQUENCES_ENV, SOURCE_ENV,
    TEACHER_FORCED_TWO_ARM,
};
use super::quality::{gate_by_id, IMPLEMENTED_GATES};

/// Three artifact directories that exist, so `admit` is exercising the
/// control fields rather than tripping on a missing path.
fn artifacts() -> (tempfile::TempDir, TeacherForcedRequest) {
    let dir = tempfile::tempdir().expect("tmp");
    for name in ["source", "candidate", "bank"] {
        std::fs::create_dir_all(dir.path().join(name)).expect("dir");
    }
    let request = TeacherForcedRequest::new(
        dir.path().join("source"),
        dir.path().join("candidate"),
        dir.path().join("bank"),
    );
    (dir, request)
}

fn vars(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

// ------------------------------------------------- gates resolve by name

#[test]
fn every_implemented_gate_resolves_and_answers_to_its_own_name() {
    for id in IMPLEMENTED_GATES {
        let gate = gate_by_id(id).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(
            gate.id, id,
            "a gate must answer to the name it is asked for"
        );
    }
    assert_eq!(IMPLEMENTED_GATES.len(), 4);
}

#[test]
fn an_unknown_gate_is_refused_and_never_falls_back() {
    // **The specific defect.** A run that quietly judged under
    // `kimi-logit-v3` whatever it was asked for would carry a verdict
    // nobody requested, and the record naming the other gate would be
    // describing a judgement that never happened.
    let err = gate_by_id("kimi-logit-v9").expect_err("must refuse");
    assert!(format!("{err}").contains("kimi-logit-v9"), "{err}");
    assert!(format!("{err}").contains("nobody asked for"), "{err}");

    // And the refusal is not a disguised default: nothing this build
    // can be asked for resolves to a gate of another name.
    for id in ["", "kimi-logit", "KIMI-LOGIT-V3", "balanced-v1"] {
        assert!(gate_by_id(id).is_err(), "`{id}` resolved to something");
    }
}

#[test]
fn the_gate_a_request_names_is_the_gate_it_admits_under() {
    // The load-bearing one. A request naming the optimiser record's own
    // gate admits under THAT gate, not under the runner's former
    // literal.
    let (_dir, request) = artifacts();
    assert_eq!(
        request
            .clone()
            .with_gate("kimi-logit-balanced-v1")
            .admit()
            .expect("a known gate")
            .id,
        "kimi-logit-balanced-v1"
    );
    assert_eq!(
        request
            .clone()
            .with_gate("kimi-logit-v1")
            .admit()
            .expect("known")
            .id,
        "kimi-logit-v1"
    );
    // Two requests differing only in gate admit differently, which is
    // what makes the field a control rather than a label.
    assert_ne!(
        request
            .clone()
            .with_gate("kimi-logit-v1")
            .admit()
            .expect("known"),
        request.with_gate("kimi-logit-v3").admit().expect("known")
    );
}

// ------------------------------------------------ procedures resolve too

#[test]
fn a_procedure_this_build_does_not_perform_is_refused() {
    assert_eq!(
        MeasurementProcedure::by_name(TEACHER_FORCED_TWO_ARM).expect("implemented"),
        MeasurementProcedure::TeacherForcedTwoArm
    );
    let err = MeasurementProcedure::by_name("single-arm-perplexity/v1").expect_err("must refuse");
    assert!(
        format!("{err}").contains("single-arm-perplexity/v1"),
        "{err}"
    );
    assert_eq!(
        MeasurementProcedure::TeacherForcedTwoArm.name(),
        TEACHER_FORCED_TWO_ARM
    );
}

// ------------------------------------------- the environment is an adapter

#[test]
fn the_environment_form_builds_the_request_a_caller_would() {
    // One procedure, two ways of naming its inputs. The historic
    // command line keeps meaning what it meant: an unset slice is 32,
    // an unset gate is the one the runner used to hold.
    let env = vars(&[
        (SOURCE_ENV, "/m/source.vindex3"),
        (CANDIDATE_ENV, "/tmp/candidate.vindex3"),
        (BANK_ENV, "/tmp/bank"),
    ]);
    let request = TeacherForcedRequest::from_vars(|k| env.get(k).cloned()).expect("all three set");

    assert_eq!(
        request,
        TeacherForcedRequest::new("/m/source.vindex3", "/tmp/candidate.vindex3", "/tmp/bank")
    );
    assert_eq!(request.sequences, DEFAULT_SEQUENCES);
    assert_eq!(request.gate, DEFAULT_GATE);
    assert_eq!(request.label, DEFAULT_LABEL);
    assert_eq!(request.procedure, MeasurementProcedure::TeacherForcedTwoArm);
    assert_eq!(request.source, PathBuf::from("/m/source.vindex3"));
}

#[test]
fn each_environment_control_reaches_the_request() {
    // Varied one at a time, because a test that moved them together
    // would pass on any one of them being wired.
    let base = [
        (SOURCE_ENV, "/m/s"),
        (CANDIDATE_ENV, "/m/c"),
        (BANK_ENV, "/m/b"),
    ];
    let build = |extra: &[(&str, &str)]| {
        let mut pairs = base.to_vec();
        pairs.extend_from_slice(extra);
        let env = vars(&pairs);
        TeacherForcedRequest::from_vars(|k| env.get(k).cloned()).expect("paths set")
    };

    assert_eq!(build(&[(SEQUENCES_ENV, "8")]).sequences, 8);
    assert_eq!(
        build(&[(GATE_ENV, "kimi-logit-balanced-v1")]).gate,
        "kimi-logit-balanced-v1"
    );
    assert_eq!(build(&[(LABEL_ENV, "q80-l25")]).label, "q80-l25");

    // Each leaves the others where they were.
    let sliced = build(&[(SEQUENCES_ENV, "8")]);
    assert_eq!(sliced.gate, DEFAULT_GATE);
    assert_eq!(sliced.label, DEFAULT_LABEL);

    // An unparseable slice is the default rather than a panic — the
    // historic reader's behaviour, kept deliberately.
    assert_eq!(
        build(&[(SEQUENCES_ENV, "many")]).sequences,
        DEFAULT_SEQUENCES
    );
}

#[test]
fn without_all_three_artifacts_the_environment_names_no_run() {
    for missing in [SOURCE_ENV, CANDIDATE_ENV, BANK_ENV] {
        let env: BTreeMap<String, String> = [SOURCE_ENV, CANDIDATE_ENV, BANK_ENV]
            .into_iter()
            .filter(|k| *k != missing)
            .map(|k| (k.to_string(), "/m/x".to_string()))
            .collect();
        assert!(
            TeacherForcedRequest::from_vars(|k| env.get(k).cloned()).is_none(),
            "a run without {missing} is not a run"
        );
    }
}

// ------------------------------------------------ refused before loading

#[test]
fn a_request_over_no_sequences_is_refused() {
    // Zero sequences is zero positions, and a gate that judges on tail
    // statistics would be reading an empty distribution.
    let (_dir, request) = artifacts();
    let err = request.with_sequences(0).admit().expect_err("must refuse");
    assert!(format!("{err}").contains("zero sequences"), "{err}");
}

#[test]
fn an_artifact_that_is_not_there_is_refused_by_name() {
    let (dir, request) = artifacts();
    request.admit().expect("all three exist");

    for (what, path) in [
        ("source", dir.path().join("source")),
        ("candidate", dir.path().join("candidate")),
        ("quality bank", dir.path().join("bank")),
    ] {
        std::fs::rename(&path, path.with_extension("moved")).expect("move it aside");
        let err = request.admit().expect_err("must refuse");
        assert!(format!("{err}").contains(what), "{err}");
        std::fs::rename(path.with_extension("moved"), &path).expect("put it back");
    }
}

#[test]
fn an_unknown_gate_is_refused_before_anything_is_loaded() {
    // The refusal that matters most for cost: a 48 B model and twenty
    // minutes of instrument time must not be spent to discover that the
    // verdict cannot be drawn.
    let (_dir, request) = artifacts();
    assert!(request.with_gate("kimi-logit-v9").admit().is_err());
}

#[test]
fn a_request_survives_a_round_trip_through_json() {
    // A prepared experiment will carry one, and a type that cannot be
    // stored cannot be sealed — `PhysicalAccountingFacts` derived
    // `Serialize` and failed at runtime for exactly this reason.
    let (_dir, request) = artifacts();
    let request = request.with_sequences(8).with_gate("kimi-logit-v2");
    let text = serde_json::to_string(&request).expect("serialise");
    assert_eq!(
        serde_json::from_str::<TeacherForcedRequest>(&text).expect("reload"),
        request
    );
    let value = serde_json::to_value(&request).expect("to value");
    assert_eq!(
        serde_json::from_value::<TeacherForcedRequest>(value).expect("from value"),
        request
    );
}
