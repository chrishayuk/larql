//! `VERIFY PATCH` against scripted verifiers: the protocol, and every way an answer can be refused. Whether a
//! compiled container implements a patch is the verifier's to measure; LQL only asks and checks the answer.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use larql_lql::{parse, CapabilityProfile, LqlError, Session, Statement};
use serde_json::{json, Value};

const CONTRACT: &str = "STRUCTURED_REMOVE_DIRECTION";

struct Fixture {
    dir: tempfile::TempDir,
    patch: PathBuf,
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
        let patch = dir.path().join("p.vlp");
        std::fs::write(&patch, "{}").unwrap();
        Self {
            patch: patch.canonicalize().unwrap(),
            base,
            container,
            dir,
        }
    }

    fn answer(&self, status: &str, violations: u64) -> Value {
        json!({
            "schema": "larql-verify-patch/1",
            "contract": CONTRACT,
            "contract_statement": "every element equals the recomputed result",
            "evidence_kind": "numerical_measurement",
            "status": status,
            "violations": violations,
            "patch": {"path": self.patch, "sha256": "aaa111"},
            "base": {"path": self.base, "sha256": "bbb222"},
            "container": {"path": self.container, "sha256": "ccc333"},
            "recomputed_from": "operands and base weights; the weight-patch ledger is not read",
            "measurements": [{"name": "elements compared", "value": "135488"}],
            "offending": if violations > 0 {
                json!([{"name": "target.decoder_stack@BF16/1.mlp.down_proj.weight",
                        "violations": violations, "detail": "first at (5, 33)"}])
            } else { json!([]) },
            "reasons": if status == "could_not_evaluate" { json!(["stale binding"]) } else { json!([]) },
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
        statement(&self.patch, &self.base, &self.container)
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

fn statement(patch: &Path, base: &Path, container: &Path) -> String {
    format!(
        "VERIFY PATCH \"{}\" USING CONTRACT \"{CONTRACT}\" FROM \"{}\" AGAINST \"{}\";",
        patch.display(),
        base.display(),
        container.display()
    )
}

#[test]
fn it_parses_into_its_four_parts() {
    let s = parse("VERIFY PATCH 'p' USING CONTRACT 'c' FROM 'b' AGAINST 'm';").unwrap();
    assert!(matches!(
        s,
        Statement::VerifyPatch { ref patch, ref contract, ref base, ref container }
            if patch == "p" && contract == "c" && base == "b" && container == "m"
    ));
}

#[test]
fn every_part_is_required() {
    for text in [
        "VERIFY PATCH USING CONTRACT 'c' FROM 'b' AGAINST 'm';",
        "VERIFY PATCH 'p' FROM 'b' AGAINST 'm';",
        "VERIFY PATCH 'p' USING CONTRACT 'c' AGAINST 'm';",
        "VERIFY PATCH 'p' USING CONTRACT 'c' FROM 'b';",
        "VERIFY PATCH 'p' USING CONTRACT 'c' IN 'b' AGAINST 'm';",
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
        parse("VERIFY OPERATOR 'o' USING CONTRACT 'c' IN WITNESS 'w' AGAINST 'm';").unwrap(),
        Statement::VerifyOperator { .. }
    ));
}

#[test]
fn the_verifier_receives_the_exact_question() {
    let f = Fixture::new();
    f.run(&f.verifier(&f.answer("satisfied", 0).to_string(), 0))
        .unwrap();
    let argv = std::fs::read_to_string(f.dir.path().join("argv")).unwrap();
    let want = [
        "patch",
        "--contract",
        CONTRACT,
        "--patch",
        f.patch.to_str().unwrap(),
        "--base",
        f.base.to_str().unwrap(),
        "--container",
        f.container.to_str().unwrap(),
    ]
    .join("\n");
    assert_eq!(argv.trim_end(), want);
}

#[test]
fn a_satisfied_contract_reports_what_it_recomputed_from_and_its_limits() {
    let f = Fixture::new();
    let out = f
        .run(&f.verifier(&f.answer("satisfied", 0).to_string(), 0))
        .unwrap()
        .join("\n");
    assert!(
        out.starts_with(&format!("Patch contract {CONTRACT}: SATISFIED")),
        "{out}"
    );
    assert!(out.contains("Requires: every element equals"), "{out}");
    assert!(
        out.contains(
            "Recomputed from: operands and base weights; the weight-patch ledger is not read"
        ),
        "{out}"
    );
    for digest in ["sha256 aaa111", "sha256 bbb222", "sha256 ccc333"] {
        assert!(out.contains(digest), "{digest} in {out}");
    }
    assert!(out.contains("elements compared: 135488"), "{out}");
    assert!(
        out.contains("Measured by:") && out.contains("as reported by the verifier"),
        "{out}"
    );
    assert!(out.contains("not a formal proof"), "{out}");
}

#[test]
fn a_violation_is_an_answer_and_counts_for_the_exit_status() {
    let f = Fixture::new();
    let v = f.verifier(&f.answer("violated", 2).to_string(), 0);
    let mut session = f.session(&v);
    let out = session
        .execute(&parse(&f.statement()).unwrap())
        .unwrap()
        .join("\n");
    assert!(out.contains(": VIOLATED"), "{out}");
    assert!(out.contains("first at (5, 33)"), "{out}");
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
        err.contains("could not be evaluated") && err.contains("stale binding"),
        "{err}"
    );
    assert_eq!(session.contract_violations(), 0);
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
        err.contains("VERIFY PATCH needs an independent verifier"),
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
        ("/schema", json!("larql-verify-operator/1"), "schema"),
        ("/contract", json!("Q8_0_T2"), "contract"),
        ("/patch/path", json!("/elsewhere"), "patch"),
        ("/base/path", json!("/elsewhere"), "base"),
        ("/container/path", json!("/elsewhere"), "container"),
        ("/evidence_kind", json!("formal_proof"), "evidence kind"),
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
    let run = |p: &Path, b: &Path, c: &Path| {
        f.session(&v)
            .execute(&parse(&statement(p, b, c)).unwrap())
            .unwrap_err()
            .to_string()
    };
    assert!(run(Path::new("/no/such/p"), &f.base, &f.container).contains("patch"));
    assert!(run(&f.base, &f.base, &f.container).contains("is not a file"));
    assert!(run(&f.patch, Path::new("/no/such/b"), &f.container).contains("base"));
    assert!(run(&f.patch, &f.patch, &f.container).contains("is not a directory"));
    assert!(run(&f.patch, &f.base, Path::new("/no/such/c")).contains("container"));
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
