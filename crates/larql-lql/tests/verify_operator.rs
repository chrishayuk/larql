//! `VERIFY OPERATOR` against scripted verifiers: the protocol, and every way an
//! answer can be refused. The numerics belong to the verifier.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use larql_lql::{parse, CapabilityProfile, LqlError, Session};
use serde_json::{json, Value};

const OP: &str = "0.self_attn.q_proj.weight";
const SEQ: &str = "Q8_0_GEMV_SEQ";

struct Fixture {
    dir: tempfile::TempDir,
    container: PathBuf,
    witness: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let container = dir.path().join("container");
        std::fs::create_dir(&container).unwrap();
        let witness = dir.path().join("witness.json");
        std::fs::write(&witness, "{}").unwrap();
        Self {
            container: container.canonicalize().unwrap(),
            witness: witness.canonicalize().unwrap(),
            dir,
        }
    }

    fn answer(&self, status: &str, violations: u64) -> Value {
        json!({
            "schema": "larql-verify-operator/1",
            "contract": SEQ,
            "contract_statement": "the output equals the declared program",
            "evidence_kind": "numerical_measurement",
            "status": status,
            "violations": violations,
            "operator": OP,
            "witness": {"path": self.witness, "sha256": "abc123", "basis": "engine_reported",
                        "reported": {"kernel": "cpu.k", "build_commit": "c0ffee"}},
            "container": {"path": self.container, "sha256": "def456"},
            "measurements": [{"name": "rows violating", "value": violations.to_string()}],
            "offending": if violations > 0 {
                json!([{"name": OP, "violations": violations, "detail": "first row 3"}])
            } else { json!([]) },
            "reasons": if status == "could_not_evaluate" { json!(["bytes differ"]) } else { json!([]) },
            "verifier": {"name": "scripted", "refcheck_commit": "f2788f0"},
            "report_sha256": "fed987",
        })
    }

    fn verifier(&self, stdout: &str, exit: i32) -> PathBuf {
        let body = self.dir.path().join("answer.json");
        std::fs::write(&body, stdout).unwrap();
        let argv = self.dir.path().join("argv");
        let script = self.dir.path().join("verifier");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\ncat '{}'\necho boom >&2\nexit {exit}\n",
                argv.display(),
                body.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    fn statement(&self) -> String {
        statement(OP, SEQ, &self.witness, &self.container)
    }

    fn session(&self, verifier: &Path) -> Session {
        let mut session = Session::new();
        session.set_verifier(verifier);
        session
    }

    fn run(&self, verifier: &Path) -> Result<Vec<String>, LqlError> {
        self.session(verifier)
            .execute(&parse(&self.statement()).unwrap())
    }

    fn refused(&self, answer: Value, expected: &str) {
        let v = self.verifier(&answer.to_string(), 0);
        let err = self.run(&v).unwrap_err().to_string();
        assert!(err.contains(expected), "expected {expected:?} in: {err}");
    }
}

fn statement(op: &str, contract: &str, witness: &Path, container: &Path) -> String {
    format!(
        "VERIFY OPERATOR \"{op}\" USING CONTRACT \"{contract}\" IN WITNESS \"{}\" AGAINST \"{}\";",
        witness.display(),
        container.display()
    )
}

#[test]
fn it_parses_into_its_four_parts() {
    let s = parse("VERIFY OPERATOR 'o' USING CONTRACT 'c' IN WITNESS 'w' AGAINST 'm';").unwrap();
    assert!(matches!(
        s,
        larql_lql::Statement::VerifyOperator { ref operator, ref contract, ref witness, ref container }
            if operator == "o" && contract == "c" && witness == "w" && container == "m"
    ));
}

#[test]
fn every_part_is_required() {
    for text in [
        "VERIFY OPERATOR USING CONTRACT 'c' IN WITNESS 'w' AGAINST 'm';",
        "VERIFY OPERATOR 'o' IN WITNESS 'w' AGAINST 'm';",
        "VERIFY OPERATOR 'o' USING CONTRACT 'c' AGAINST 'm';",
        "VERIFY OPERATOR 'o' USING CONTRACT 'c' IN WITNESS 'w';",
        "VERIFY OPERATOR 'o' USING CONTRACT 'c' IN 'w' AGAINST 'm';",
    ] {
        assert!(parse(text).is_err(), "{text}");
    }
}

#[test]
fn the_representation_statement_still_parses() {
    assert!(matches!(
        parse("VERIFY REPRESENTATION 'c' USING CONTRACT 'x' AGAINST 's';").unwrap(),
        larql_lql::Statement::VerifyRepresentation { .. }
    ));
}

#[test]
fn the_verifier_receives_the_exact_question() {
    let f = Fixture::new();
    f.run(&f.verifier(&f.answer("satisfied", 0).to_string(), 0))
        .unwrap();
    let argv = std::fs::read_to_string(f.dir.path().join("argv")).unwrap();
    let want = [
        "operator",
        "--contract",
        SEQ,
        "--operator",
        OP,
        "--witness",
        f.witness.to_str().unwrap(),
        "--container",
        f.container.to_str().unwrap(),
    ]
    .join("\n");
    assert_eq!(argv.trim_end(), want);
}

#[test]
fn a_satisfied_contract_reports_its_evidence_and_its_limits() {
    let f = Fixture::new();
    let out = f
        .run(&f.verifier(&f.answer("satisfied", 0).to_string(), 0))
        .unwrap()
        .join("\n");
    assert!(
        out.starts_with(&format!("Operator {OP}, contract {SEQ}: SATISFIED")),
        "{out}"
    );
    assert!(out.contains("Requires: the output equals"), "{out}");
    assert!(
        out.contains("reported by the engine, not authenticated"),
        "{out}"
    );
    assert!(
        out.contains("The witness reports: build_commit=c0ffee"),
        "{out}"
    );
    assert!(out.contains("sha256 def456"), "{out}");
    assert!(
        out.contains("Measured by:") && out.contains("as reported by the verifier"),
        "{out}"
    );
    assert!(out.contains("Report sha256: fed987"), "{out}");
    assert!(
        out.contains("of this one recorded call, not a formal proof"),
        "{out}"
    );
}

#[test]
fn a_violation_is_an_answer_and_counts_for_the_exit_status() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer("violated", 7).to_string(), 0);
    let mut session = f.session(&v);
    let out = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap()
        .join("\n");
    assert!(out.contains(": VIOLATED"), "{out}");
    assert!(out.contains("first row 3"), "{out}");
    assert_eq!(session.contract_violations(), 1);
}

#[test]
fn could_not_evaluate_is_an_error_with_the_reason_and_no_violation() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer("could_not_evaluate", 0).to_string(), 0);
    let mut session = f.session(&v);
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("could not be evaluated") && err.contains("bytes differ"),
        "{err}"
    );
    assert_eq!(session.contract_violations(), 0);
}

#[test]
fn no_verifier_is_a_refusal() {
    let f = Fixture::new();
    let mut session = Session::new();
    session.set_verifier("");
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("VERIFY OPERATOR needs an independent verifier"),
        "{err}"
    );
}

#[test]
fn a_failing_verifier_is_an_error_showing_its_stderr() {
    let f = Fixture::new();
    let err = f.run(&f.verifier("", 3)).unwrap_err().to_string();
    assert!(err.contains("boom"), "{err}");
}

#[test]
fn answers_that_are_not_about_this_question_are_refused() {
    let f = Fixture::new();
    for (path, value, expected) in [
        ("/schema", json!("larql-verify-contract/1"), "schema"),
        ("/contract", json!("Q8_0_GEMV_B1"), "contract"),
        ("/operator", json!("1.other"), "operator"),
        ("/witness/path", json!("/elsewhere"), "witness"),
        ("/container/path", json!("/elsewhere"), "container"),
        ("/evidence_kind", json!("formal_proof"), "evidence kind"),
        ("/witness/basis", json!("authenticated"), "witness basis"),
    ] {
        let mut a = f.answer("satisfied", 0);
        *a.pointer_mut(path).unwrap() = value;
        f.refused(a, expected);
    }
}

#[test]
fn answers_that_contradict_themselves_are_refused() {
    let f = Fixture::new();
    f.refused(f.answer("satisfied", 1), "satisfied, with 1");
    f.refused(f.answer("violated", 0), "violated, with 0");
    let mut a = f.answer("violated", 7);
    a["offending"][0]["violations"] = json!(6);
    f.refused(a, "7 violations but 6 listed");
    let mut a = f.answer("could_not_evaluate", 0);
    a["reasons"] = json!([]);
    f.refused(a, "no reason");
    f.refused(f.answer("proved", 0), "unknown status");
    let v = f.verifier("not json", 0);
    assert!(f
        .run(&v)
        .unwrap_err()
        .to_string()
        .contains("not a valid result"));
}

#[test]
fn the_paths_must_exist_and_be_what_they_are_said_to_be() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer("satisfied", 0).to_string(), 0);
    let run = |w: &Path, c: &Path| {
        f.session(&v)
            .execute(&parse(&statement(OP, SEQ, w, c)).unwrap())
            .unwrap_err()
            .to_string()
    };
    assert!(run(Path::new("/no/such/w"), &f.container).contains("witness"));
    assert!(run(&f.container, &f.container).contains("is not a file"));
    assert!(run(&f.witness, Path::new("/no/such/c")).contains("container"));
    assert!(run(&f.witness, &f.witness).contains("is not a directory"));
}

#[test]
fn a_public_explorer_cannot_run_a_verifier() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer("satisfied", 0).to_string(), 0);
    let mut session = f.session(&v);
    session.set_profile(CapabilityProfile::PublicExplorer);
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err();
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
    assert!(!f.dir.path().join("argv").exists(), "the verifier ran");
}

// ── A verifier that is momentarily not runnable ─────────────────────────

/// How long the verifier stays open for writing after the statement starts.
const HELD_OPEN: std::time::Duration = std::time::Duration::from_millis(150);

#[test]
fn a_verifier_still_open_for_writing_is_retried_until_it_can_run() {
    // An executable cannot be run while a process holds it open for
    // writing ("text file busy"). That happens for an instant after a
    // verifier is written, and used to fail the statement outright.
    //
    // Linux refuses the exec while the descriptor is open, so there this
    // fails without the retry. macOS allows it, so there it passes either
    // way; the retry itself is unit-tested beside `spawn_when_not_busy`.
    let fx = Fixture::new();
    let verifier = fx.verifier(&fx.answer("satisfied", 0).to_string(), 0);
    let writer = std::fs::OpenOptions::new()
        .write(true)
        .open(&verifier)
        .expect("open the verifier for writing");
    let release = std::thread::spawn(move || {
        std::thread::sleep(HELD_OPEN);
        drop(writer);
    });

    let result = fx.run(&verifier);

    release.join().expect("release thread");
    result.expect("the statement runs once the verifier is closed");
}
