//! `VERIFY REPRESENTATION` against scripted verifiers, so that these tests pin the
//! protocol and every way an answer can be refused.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use larql_lql::{parse, CapabilityProfile, LqlError, Session};
use serde_json::{json, Value};

struct Fixture {
    dir: tempfile::TempDir,
    container: PathBuf,
    source: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let container = dir.path().join("container");
        let source = dir.path().join("source");
        std::fs::create_dir(&container).unwrap();
        std::fs::create_dir(&source).unwrap();
        let (container, source) = (
            container.canonicalize().unwrap(),
            source.canonicalize().unwrap(),
        );
        Self {
            dir,
            container,
            source,
        }
    }

    fn answer(&self, status: &str, violations: u64) -> Value {
        json!({
            "schema": "larql-verify-contract/1",
            "contract": "Q8_0_T2",
            "contract_statement": "every element is within the bound",
            "evidence_kind": "numerical_measurement",
            "status": status,
            "violations": violations,
            "container": {"path": self.container, "sha256": "abc123"},
            "source": {"path": self.source, "identity": "9 files match pinned hashes"},
            "measurements": [{"name": "elements over T2", "value": violations.to_string()}],
            "offending": if violations > 0 {
                json!([{"name": "layer.0", "violations": violations, "detail": "worst 2.0"}])
            } else { json!([]) },
            "reasons": if status == "could_not_evaluate" { json!(["no Q8_0 tensors"]) } else { json!([]) },
            "verifier": {"name": "scripted", "refcheck_commit": "e505afc"},
            "report_sha256": "def456",
        })
    }

    /// A verifier that records its argv, prints `stdout`, writes to stderr and exits.
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

    fn argv(&self) -> Option<String> {
        std::fs::read_to_string(self.dir.path().join("argv")).ok()
    }

    fn statement(&self) -> String {
        verify(&self.container, &self.source)
    }

    fn run(&self, verifier: &Path) -> Result<Vec<String>, LqlError> {
        let mut session = Session::new();
        session.set_verifier(verifier);
        session.execute(&parse(&self.statement()).unwrap())
    }

    fn refused(&self, answer: Value, expected: &str) {
        let v = self.verifier(&answer.to_string(), 0);
        let err = self.run(&v).unwrap_err().to_string();
        assert!(err.contains(expected), "expected {expected:?} in: {err}");
    }
}

fn verify(container: &Path, source: &Path) -> String {
    format!(
        "VERIFY REPRESENTATION \"{}\" USING CONTRACT \"Q8_0_T2\" AGAINST \"{}\";",
        container.display(),
        source.display()
    )
}

#[test]
fn a_satisfied_contract_reports_its_evidence() {
    let f = Fixture::new();
    let out = f
        .run(&f.verifier(&f.answer("satisfied", 0).to_string(), 0))
        .unwrap()
        .join("\n");
    assert!(out.starts_with("Contract Q8_0_T2: SATISFIED"), "{out}");
    assert!(out.contains("sha256 abc123"), "{out}");
    assert!(out.contains("refcheck_commit=e505afc"), "{out}");
    assert!(out.contains("Report sha256: def456"), "{out}");
    assert!(out.contains("not a formal proof"), "{out}");
}

#[test]
fn the_output_says_the_verifier_identity_is_not_authenticated() {
    let f = Fixture::new();
    let out = f
        .run(&f.verifier(&f.answer("satisfied", 0).to_string(), 0))
        .unwrap()
        .join("\n");
    let line = out.lines().find(|l| l.contains("Measured by")).unwrap();
    assert!(line.contains("refcheck_commit=e505afc"), "{line}");
    assert!(line.contains("not authenticated"), "{line}");
}

#[test]
fn a_violation_is_an_answer_not_an_error_and_names_the_tensors() {
    let f = Fixture::new();
    let out = f
        .run(&f.verifier(&f.answer("violated", 57634).to_string(), 0))
        .unwrap()
        .join("\n");
    assert!(out.starts_with("Contract Q8_0_T2: VIOLATED"), "{out}");
    assert!(out.contains("layer.0: 57634 (worst 2.0)"), "{out}");
    assert!(out.contains("not a formal proof"), "{out}");
}

#[test]
fn long_offender_lists_are_cut_and_say_how_many_remain() {
    let f = Fixture::new();
    let mut a = f.answer("violated", 12);
    a["offending"] = (0..12)
        .map(|i| json!({"name": format!("t{i}"), "violations": 1}))
        .collect();
    let out = f.run(&f.verifier(&a.to_string(), 0)).unwrap().join("\n");
    assert!(out.contains("Offending tensors: 12"), "{out}");
    assert!(out.contains("t9: 1") && !out.contains("t10: 1"), "{out}");
    assert!(out.contains("... and 2 more"), "{out}");
}

#[test]
fn the_verifier_receives_the_exact_question() {
    let f = Fixture::new();
    f.run(&f.verifier(&f.answer("satisfied", 0).to_string(), 0))
        .unwrap();
    let want = format!(
        "contract\n--contract\nQ8_0_T2\n--container\n{}\n--source\n{}\n",
        f.container.display(),
        f.source.display()
    );
    assert_eq!(f.argv().unwrap(), want);
}

#[test]
fn could_not_evaluate_is_an_error_with_the_reason() {
    let f = Fixture::new();
    f.refused(f.answer("could_not_evaluate", 0), "no Q8_0 tensors");
}

#[test]
fn a_verifier_failure_is_an_error_with_its_stderr() {
    let f = Fixture::new();
    let err = f.run(&f.verifier("", 2)).unwrap_err().to_string();
    assert!(err.contains("failed") && err.contains("boom"), "{err}");
}

#[test]
fn a_verifier_that_cannot_start_is_an_error() {
    let f = Fixture::new();
    let err = f.run(&f.dir.path().join("absent")).unwrap_err().to_string();
    assert!(err.contains("starting verifier"), "{err}");
}

#[test]
fn malformed_answers_are_refused_not_displayed() {
    let f = Fixture::new();
    let err = f.run(&f.verifier("SATISFIED", 0)).unwrap_err().to_string();
    assert!(err.contains("not a valid result"), "{err}");
}

#[test]
fn an_answer_to_a_different_question_is_refused() {
    let f = Fixture::new();
    for (key, value, expected) in [
        ("schema", json!("larql-verify-contract/2"), "schema"),
        ("contract", json!("OTHER"), "answers contract"),
        ("status", json!("passed"), "unknown status"),
    ] {
        let mut a = f.answer("satisfied", 0);
        a[key] = value;
        f.refused(a, expected);
    }
    let mut a = f.answer("satisfied", 0);
    a["container"]["path"] = json!("/somewhere/else");
    f.refused(a, "it measured");
    let mut a = f.answer("satisfied", 0);
    a["source"]["path"] = json!("/somewhere/else");
    f.refused(a, "it used source");
}

/// A measurement must never be presentable as more than one.
#[test]
fn a_claim_of_proof_is_refused() {
    let f = Fixture::new();
    let mut a = f.answer("satisfied", 0);
    a["evidence_kind"] = json!("formal_proof");
    f.refused(a, "evidence kind");
}

#[test]
fn answers_that_contradict_themselves_are_refused() {
    let f = Fixture::new();
    let mut a = f.answer("satisfied", 0);
    a["violations"] = json!(3);
    f.refused(a, "satisfied, with 3");
    let mut a = f.answer("violated", 5);
    a["offending"][0]["violations"] = json!(4);
    f.refused(a, "5 violations but 4 listed");
    let mut a = f.answer("violated", 5);
    a["violations"] = json!(0);
    f.refused(a, "violated, with 0");
    let mut a = f.answer("could_not_evaluate", 0);
    a["reasons"] = json!([]);
    f.refused(a, "no reason");
}

#[test]
fn missing_paths_refuse_before_the_verifier_runs() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer("satisfied", 0).to_string(), 0);
    let mut session = Session::new();
    session.set_verifier(&v);
    let stmt = verify(&f.dir.path().join("nope"), &f.source);
    let err = session
        .execute(&parse(&stmt).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("container"), "{err}");
    assert!(f.argv().is_none(), "the verifier must not have run");
}

/// No verifier must never read as a pass. An empty choice turns discovery off.
#[test]
fn no_verifier_refuses_and_says_larql_does_not_evaluate() {
    let f = Fixture::new();
    let mut session = Session::new();
    session.set_verifier("");
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("LARQL_VERIFIER"), "{err}");
    assert!(err.contains("does not evaluate"), "{err}");
}

#[test]
fn the_public_profile_refuses_it_before_the_verifier_runs() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer("satisfied", 0).to_string(), 0);
    let mut session = Session::new();
    session.set_verifier(&v);
    session.set_profile(CapabilityProfile::PublicExplorer);
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err();
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
    assert!(f.argv().is_none());
}

#[test]
fn it_parses_case_insensitively_with_either_quote_and_rejects_partial_forms() {
    let ok = "verify representation 'a' using contract 'c' against 'b'";
    assert!(matches!(
        parse(ok).unwrap(),
        larql_lql::Statement::VerifyRepresentation { .. }
    ));
    for bad in [
        "VERIFY 'a' USING CONTRACT 'c' AGAINST 'b';",
        "VERIFY REPRESENTATION USING CONTRACT 'c' AGAINST 'b';",
        "VERIFY REPRESENTATION 'a' USING CONTRACT 'c';",
        "VERIFY REPRESENTATION 'a' CONTRACT 'c' AGAINST 'b';",
    ] {
        assert!(parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn the_session_counts_violations_and_nothing_else() {
    let f = Fixture::new();
    let stmt = parse(&f.statement()).unwrap();
    let mut session = Session::new();

    session.set_verifier(f.verifier(&f.answer("satisfied", 0).to_string(), 0));
    session.execute(&stmt).unwrap();
    assert_eq!(session.contract_violations(), 0);

    session.set_verifier(f.verifier(&f.answer("could_not_evaluate", 0).to_string(), 0));
    session.execute(&stmt).unwrap_err();
    assert_eq!(
        session.contract_violations(),
        0,
        "unevaluated is not a violation"
    );

    session.set_verifier(f.verifier(&f.answer("violated", 3).to_string(), 0));
    session.execute(&stmt).unwrap();
    session.execute(&stmt).unwrap();
    assert_eq!(session.contract_violations(), 2);
}
