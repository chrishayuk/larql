//! `VERIFY LEDGER` against scripted verifiers: the protocol, and every way an answer can be refused. Whether a
//! compiled container is what its own weight-patch ledger declares of a base is the verifier's to measure; LQL
//! only asks and checks the answer.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use larql_lql::{parse, CapabilityProfile, LqlError, Session, Statement};
use serde_json::{json, Value};

const CONTRACT: &str = "WEIGHT_PATCHES_V4";
const SATISFIED: &str = "satisfied";
const VIOLATED: &str = "violated";
const COULD_NOT: &str = "could_not_evaluate";
const NUMERICAL: &str = "numerical_satisfaction";
const LEDGER: &str = "ledger_integrity_and_provenance";

struct Fixture {
    dir: tempfile::TempDir,
    base: PathBuf,
    container: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let make_dir = |name: &str| {
            let p = dir.path().join(name);
            std::fs::create_dir(&p).unwrap();
            p.canonicalize().unwrap()
        };
        let (base, container) = (make_dir("base"), make_dir("container"));
        Self {
            base,
            container,
            dir,
        }
    }

    /// An answer whose overall status is the one its two halves make, as the protocol defines it.
    fn answer(&self, numerical: &str, ledger: &str) -> Value {
        let status = if numerical == VIOLATED || ledger == VIOLATED {
            VIOLATED
        } else if numerical == SATISFIED && ledger == SATISFIED {
            SATISFIED
        } else {
            COULD_NOT
        };
        let failed: Vec<&str> = [(NUMERICAL, numerical), (LEDGER, ledger)]
            .into_iter()
            .filter(|(_, s)| *s != SATISFIED)
            .map(|(half, _)| half)
            .collect();
        let violations = u64::from(status == VIOLATED) * 3;
        json!({
            "schema": "larql-verify-ledger/1",
            "contract": CONTRACT,
            "contract_statement": "the container holds what its ledger declares of the base",
            "evidence_kind": "numerical_measurement",
            "status": status,
            "violations": violations,
            "base": {"path": self.base, "sha256": "bbb222"},
            "container": {"path": self.container, "sha256": "ccc333"},
            "recomputed_from": "the base's weights and the operands the ledger records",
            "numerical_satisfaction": numerical,
            "ledger_integrity": ledger,
            "provenance": ledger,
            "failed_halves": failed,
            "measurements": [{"name": "elements compared", "value": "16384"}],
            "offending": if violations > 0 {
                json!([{"name": "target.decoder_stack@BF16/1.mlp.down_proj.weight",
                        "violations": violations, "detail": "first at (5, 33)", "half": NUMERICAL}])
            } else { json!([]) },
            "reasons": if status == COULD_NOT { json!(["unknown schema"]) } else { json!([]) },
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
        statement(&self.container, &self.base)
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

fn statement(container: &Path, base: &Path) -> String {
    format!(
        "VERIFY LEDGER \"{}\" USING CONTRACT \"{CONTRACT}\" FROM \"{}\";",
        container.display(),
        base.display()
    )
}

#[test]
fn it_parses_into_its_three_parts() {
    let s = parse("VERIFY LEDGER 'm' USING CONTRACT 'c' FROM 'b';").unwrap();
    assert!(matches!(
        s,
        Statement::VerifyLedger { ref container, ref contract, ref base }
            if container == "m" && contract == "c" && base == "b"
    ));
    assert!(parse("verify ledger 'm' using contract 'c' from 'b'").is_ok());
}

#[test]
fn every_part_is_required() {
    for text in [
        "VERIFY LEDGER USING CONTRACT 'c' FROM 'b';",
        "VERIFY LEDGER 'm' FROM 'b';",
        "VERIFY LEDGER 'm' USING CONTRACT 'c';",
        "VERIFY LEDGER 'm' USING CONTRACT 'c' AGAINST 'b';",
    ] {
        assert!(parse(text).is_err(), "{text}");
    }
}

#[test]
fn the_other_verify_statements_still_parse() {
    assert!(matches!(
        parse("VERIFY REPRESENTATION 'c' USING CONTRACT 'x' AGAINST 's';").unwrap(),
        Statement::VerifyRepresentation { .. }
    ));
    assert!(matches!(
        parse("VERIFY PATCH 'p' USING CONTRACT 'c' FROM 'b' AGAINST 'm';").unwrap(),
        Statement::VerifyPatch { .. }
    ));
}

#[test]
fn the_verifier_receives_the_exact_question() {
    let f = Fixture::new();
    f.run(&f.verifier(&f.answer(SATISFIED, SATISFIED).to_string(), 0))
        .unwrap();
    let argv = std::fs::read_to_string(f.dir.path().join("argv")).unwrap();
    let want = [
        "ledger",
        "--contract",
        CONTRACT,
        "--base",
        f.base.to_str().unwrap(),
        "--container",
        f.container.to_str().unwrap(),
    ]
    .join("\n");
    assert_eq!(argv.trim_end(), want);
}

#[test]
fn a_satisfied_contract_reports_both_halves_and_what_it_does_not_judge() {
    let f = Fixture::new();
    let out = f
        .run(&f.verifier(&f.answer(SATISFIED, SATISFIED).to_string(), 0))
        .unwrap()
        .join("\n");
    assert!(
        out.starts_with(&format!("Ledger contract {CONTRACT}: SATISFIED")),
        "{out}"
    );
    assert!(out.contains("Numerical satisfaction: satisfied"), "{out}");
    assert!(
        out.contains("Ledger integrity and provenance: satisfied"),
        "{out}"
    );
    assert!(!out.contains("Failed:"), "{out}");
    assert!(
        out.contains("Recomputed from: the base's weights and the operands the ledger records"),
        "{out}"
    );
    for digest in ["sha256 bbb222", "sha256 ccc333"] {
        assert!(out.contains(digest), "{digest} in {out}");
    }
    assert!(out.contains("elements compared: 16384"), "{out}");
    assert!(out.contains("not a formal proof"), "{out}");
    assert!(
        out.contains("does not judge the re-derived (quantised) representations"),
        "{out}"
    );
}

#[test]
fn a_violation_names_the_half_that_failed_and_counts_for_the_exit_status() {
    let f = Fixture::new();
    for (numerical, ledger, failed) in [
        (VIOLATED, SATISFIED, NUMERICAL),
        (SATISFIED, VIOLATED, LEDGER),
    ] {
        let v = f.verifier(&f.answer(numerical, ledger).to_string(), 0);
        let mut session = f.session(&v);
        let out = session
            .execute(&parse(&f.statement()).unwrap())
            .unwrap()
            .join("\n");
        assert!(out.contains(": VIOLATED"), "{out}");
        assert!(out.contains(&format!("Failed: {failed}")), "{out}");
        assert!(out.contains("first at (5, 33)"), "{out}");
        assert_eq!(session.contract_violations(), 1);
    }
}

#[test]
fn could_not_evaluate_is_an_error_with_the_reason_and_no_violation() {
    let f = Fixture::new();
    // One half judged and satisfied, the other not evaluated: not a pass.
    let v = f.verifier(&f.answer(COULD_NOT, SATISFIED).to_string(), 0);
    let mut session = f.session(&v);
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("could not be evaluated") && err.contains("unknown schema"),
        "{err}"
    );
    assert_eq!(session.contract_violations(), 0);
}

#[test]
fn an_answer_refused_before_either_half_was_judged_still_gives_its_reason() {
    // An unknown contract is answered without the halves at all.
    let f = Fixture::new();
    let mut a = f.answer(COULD_NOT, COULD_NOT);
    for field in [
        "numerical_satisfaction",
        "ledger_integrity",
        "failed_halves",
    ] {
        a.as_object_mut().unwrap().remove(field);
    }
    a["reasons"] = json!(["unknown contract"]);
    let err = f
        .run(&f.verifier(&a.to_string(), 0))
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown contract"), "{err}");
}

#[test]
fn no_verifier_is_a_refusal_not_a_pass() {
    let f = Fixture::new();
    let mut session = Session::new();
    session.set_verifier("");
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("VERIFY LEDGER needs an independent verifier"),
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
        ("/schema", json!("larql-verify-patch/1"), "schema"),
        (
            "/contract",
            json!("STRUCTURED_REMOVE_DIRECTION"),
            "contract",
        ),
        ("/base/path", json!("/elsewhere"), "base"),
        ("/container/path", json!("/elsewhere"), "container"),
        ("/evidence_kind", json!("formal_proof"), "evidence kind"),
    ] {
        let mut a = f.answer(SATISFIED, SATISFIED);
        *a.pointer_mut(path).unwrap() = value;
        f.refused(a, expected);
    }
}

#[test]
fn an_overall_status_its_halves_do_not_make_is_refused() {
    let f = Fixture::new();
    // Satisfied overall while the ledger half was never evaluated.
    let mut a = f.answer(SATISFIED, SATISFIED);
    a["ledger_integrity"] = json!(COULD_NOT);
    f.refused(a, "but its halves are");
    // Satisfied overall while a half is violated.
    let mut a = f.answer(SATISFIED, SATISFIED);
    a["numerical_satisfaction"] = json!(VIOLATED);
    f.refused(a, "but its halves are");
    // Violated overall with both halves satisfied.
    let mut a = f.answer(VIOLATED, SATISFIED);
    a["numerical_satisfaction"] = json!(SATISFIED);
    f.refused(a, "but its halves are");
    // A satisfied answer that leaves a half out.
    let mut a = f.answer(SATISFIED, SATISFIED);
    a.as_object_mut().unwrap().remove("ledger_integrity");
    f.refused(a, "unknown status");
    let mut a = f.answer(SATISFIED, SATISFIED);
    a["numerical_satisfaction"] = json!("proved");
    f.refused(a, "unknown status");
}

#[test]
fn answers_that_contradict_themselves_are_refused() {
    let f = Fixture::new();
    let mut a = f.answer(SATISFIED, SATISFIED);
    a["violations"] = json!(1);
    f.refused(a, "satisfied, with 1");
    let mut a = f.answer(VIOLATED, SATISFIED);
    a["offending"][0]["violations"] = json!(2);
    f.refused(a, "3 violations but 2 listed");
    let mut a = f.answer(COULD_NOT, COULD_NOT);
    a["reasons"] = json!([]);
    f.refused(a, "no reason");
    let v = f.verifier("not json", 0);
    assert!(f
        .run(&v)
        .unwrap_err()
        .to_string()
        .contains("not a valid result"));
}

#[test]
fn the_paths_must_exist_and_be_directories() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer(SATISFIED, SATISFIED).to_string(), 0);
    let file = f.dir.path().join("a-file");
    std::fs::write(&file, "x").unwrap();
    let run = |c: &Path, b: &Path| {
        f.session(&v)
            .execute(&parse(&statement(c, b)).unwrap())
            .unwrap_err()
            .to_string()
    };
    assert!(run(Path::new("/no/such/c"), &f.base).contains("container"));
    assert!(run(&file, &f.base).contains("is not a directory"));
    assert!(run(&f.container, Path::new("/no/such/b")).contains("base"));
    assert!(run(&f.container, &file).contains("is not a directory"));
}

#[test]
fn a_public_explorer_cannot_run_a_verifier() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer(SATISFIED, SATISFIED).to_string(), 0);
    let mut session = f.session(&v);
    session.set_profile(CapabilityProfile::PublicExplorer);
    let err = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap_err();
    assert!(matches!(err, LqlError::Refused { .. }), "{err}");
    assert!(!f.dir.path().join("argv").exists(), "the verifier ran");
}
