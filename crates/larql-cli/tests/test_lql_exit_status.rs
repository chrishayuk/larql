//! `larql lql`'s exit status, through the real binary.

use std::process::{Command, Output};

fn lql(args: &[&str], verifier: Option<&std::path::Path>) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_larql"));
    c.env_remove("LARQL_VERIFIER").args(args);
    if let Some(v) = verifier {
        c.env("LARQL_VERIFIER", v);
    }
    c.output().expect("runs larql")
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("exited normally")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn an_empty_script_exits_0() {
    // Leading space: an argument starting with `--` is a flag to clap.
    let o = lql(&["lql", " -- only a comment\n"], None);
    assert_eq!(code(&o), 0, "{}", stdout(&o));
}

#[test]
fn a_parse_error_exits_1_and_keeps_its_diagnostic_on_stdout() {
    let o = lql(&["lql", "NOT A STATEMENT;"], None);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).starts_with("Error: "), "{}", stdout(&o));
}

#[test]
fn an_execution_error_exits_1() {
    let o = lql(&["lql", "STATS;"], None);
    assert_eq!(code(&o), 1);
    assert!(stdout(&o).contains("No backend"), "{}", stdout(&o));
}

#[test]
fn a_later_statement_does_not_hide_an_earlier_failure() {
    let o = lql(&["lql", "NOT A STATEMENT; -- then nothing\n"], None);
    assert_eq!(code(&o), 1);
    let o = lql(&["lql", "STATS; NOT A STATEMENT; STATS;"], None);
    assert_eq!(code(&o), 1);
    assert_eq!(
        stdout(&o)
            .lines()
            .filter(|l| l.starts_with("Error: "))
            .count(),
        3,
        "all ran"
    );
}

#[test]
fn a_usage_error_exits_1_never_the_violation_code() {
    assert_eq!(code(&lql(&["lql"], None)), 1);
    assert_eq!(code(&lql(&["lql", "STATS;", "--no-such-flag"], None)), 1);
}

#[test]
fn help_and_other_subcommands_keep_clap_behaviour() {
    assert_eq!(code(&lql(&["lql", "--help"], None)), 0);
    assert_eq!(code(&lql(&["--version"], None)), 0);
    assert_eq!(code(&lql(&["no-such-command"], None)), 2);
}

#[cfg(unix)]
mod contracts {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    struct Fixture {
        dir: tempfile::TempDir,
        path: PathBuf,
    }

    /// A verifier answering by the contract it is asked about, so one script can
    /// hold statements with different outcomes.
    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().canonicalize().unwrap();
        let answer = |contract: &str,
                      status: &str,
                      violations: u32,
                      offending: &str,
                      reasons: &str| {
            format!(
                r#"{{"schema":"larql-verify-contract/1","contract":"{contract}","evidence_kind":"numerical_measurement","status":"{status}","violations":{violations},"container":{{"path":"{p}","sha256":null}},"source":{{"path":"{p}","identity":null}},"offending":[{offending}],"reasons":[{reasons}],"verifier":{{"name":"scripted"}}}}"#,
                p = path.display()
            )
        };
        let script = format!(
            "#!/bin/sh\ncase \"$3\" in\nGOOD) cat <<'A'\n{}\nA\n;;\nBAD) cat <<'A'\n{}\nA\n;;\nUNKNOWN) cat <<'A'\n{}\nA\n;;\nesac\n",
            answer("GOOD", "satisfied", 0, "", ""),
            answer("BAD", "violated", 2, r#"{"name":"t","violations":2}"#, ""),
            answer("UNKNOWN", "could_not_evaluate", 0, "", r#""no such contract""#),
        );
        let verifier = path.join("verifier");
        std::fs::write(&verifier, script).unwrap();
        std::fs::set_permissions(&verifier, std::fs::Permissions::from_mode(0o755)).unwrap();
        Fixture {
            path: verifier,
            dir,
        }
    }

    fn verify(f: &Fixture, contract: &str) -> String {
        let p = f.dir.path().canonicalize().unwrap();
        format!(
            "VERIFY REPRESENTATION '{p}' USING CONTRACT '{contract}' AGAINST '{p}';",
            p = p.display()
        )
    }

    fn run(f: &Fixture, contracts: &[&str]) -> Output {
        let script: String = contracts.iter().map(|c| verify(f, c)).collect();
        lql(&["lql", &script], Some(&f.path))
    }

    #[test]
    fn satisfied_exits_0() {
        let f = fixture();
        let o = run(&f, &["GOOD"]);
        assert_eq!(code(&o), 0, "{}", stdout(&o));
        assert!(stdout(&o).contains("SATISFIED"));
    }

    #[test]
    fn violated_exits_2_with_its_report() {
        let f = fixture();
        let o = run(&f, &["BAD"]);
        assert_eq!(code(&o), 2, "{}", stdout(&o));
        assert!(stdout(&o).contains("VIOLATED"));
    }

    #[test]
    fn could_not_evaluate_exits_1_and_never_reads_as_satisfied() {
        let f = fixture();
        let o = run(&f, &["UNKNOWN"]);
        assert_eq!(code(&o), 1, "{}", stdout(&o));
        assert!(!stdout(&o).contains("SATISFIED"), "{}", stdout(&o));
        assert_eq!(
            code(&run(&f, &["UNKNOWN", "GOOD"])),
            1,
            "a later pass must not hide it"
        );
    }

    #[test]
    fn a_missing_verifier_exits_1() {
        let f = fixture();
        let o = lql(&["lql", &verify(&f, "GOOD")], None);
        assert_eq!(code(&o), 1, "{}", stdout(&o));
    }

    #[test]
    fn violations_outrank_passes_and_errors_outrank_violations_in_either_order() {
        let f = fixture();
        assert_eq!(code(&run(&f, &["GOOD", "BAD", "GOOD"])), 2);
        assert_eq!(code(&run(&f, &["BAD", "UNKNOWN"])), 1);
        assert_eq!(code(&run(&f, &["UNKNOWN", "BAD"])), 1);
    }
}
