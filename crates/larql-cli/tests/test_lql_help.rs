//! `larql lql --help` and `larql lql "HELP;"`, through the real binary: an
//! agent driving the command can learn the process model from `--help` and
//! the language from `HELP;`.

use std::process::{Command, Output};

use larql_lql::{EXIT_CONTRACT_VIOLATED, EXIT_ERROR, EXIT_OK};

fn larql(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        // An empty choice turns verifier discovery off.
        .env("LARQL_VERIFIER", "")
        .args(args)
        .output()
        .expect("runs larql")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn code(o: &Output) -> i32 {
    o.status.code().expect("exited normally")
}

#[test]
fn dash_dash_help_states_the_process_model_and_points_at_help() {
    let o = larql(&["lql", "--help"]);
    assert_eq!(code(&o), EXIT_OK);
    let text = stdout(&o);
    for phrase in [
        "ONE process and share ONE session",
        "State persists across invocations only through files",
        "every later statement that would write an artifact",
        "larql lql \"HELP;\"",
        "`larql repl` runs the same language interactively and does not report failure",
    ] {
        assert!(text.contains(phrase), "no {phrase:?} in:\n{text}");
    }
    for exit in [EXIT_OK, EXIT_ERROR, EXIT_CONTRACT_VIOLATED] {
        assert!(
            text.contains(&format!("\n  {exit}  ")),
            "no exit {exit} in:\n{text}"
        );
    }
}

#[test]
fn every_help_statement_dash_dash_help_suggests_works() {
    let text = stdout(&larql(&["lql", "--help"]));
    let suggested: Vec<&str> = text
        .match_indices("\"HELP")
        .filter_map(|(at, _)| {
            let rest = &text[at + 1..];
            rest.find('"').map(|end| &rest[..end])
        })
        .collect();
    assert!(suggested.len() >= 4, "{suggested:?}");
    for statement in suggested {
        let o = larql(&["lql", statement]);
        assert_eq!(code(&o), EXIT_OK, "{statement}: {}", stdout(&o));
        // A page may quote the word; a failed statement is a line that starts with it.
        assert!(
            !stdout(&o).lines().any(|l| l.starts_with("Error:")),
            "{statement}"
        );
    }
}

#[test]
fn help_runs_with_nothing_bound_and_lists_the_statements() {
    let o = larql(&["lql", "HELP;"]);
    assert_eq!(code(&o), EXIT_OK);
    let text = stdout(&o);
    for line in [
        "  USE \"<path>\";",
        "  DISCARD WEIGHT EDITS;",
        "  VERIFY LEDGER \"<container>\" USING CONTRACT \"<id>\" FROM \"<base>\";",
    ] {
        assert!(
            text.lines().any(|l| l == line),
            "no line {line:?} in:\n{text}"
        );
    }
}

#[test]
fn an_unknown_topic_exits_1_and_names_the_nearest() {
    let o = larql(&["lql", "HELP REMOVE DIRECTON;"]);
    assert_eq!(code(&o), EXIT_ERROR);
    assert_eq!(
        stdout(&o).trim_end(),
        "Error: no help topic \"REMOVE DIRECTON\". Nearest: REMOVE DIRECTION. Run HELP; for every topic."
    );
}

#[test]
fn a_mistyped_statement_exits_1_and_says_what_was_meant() {
    let o = larql(&["lql", "REMOVE DIRECTON \"x\" FROM \"t\";"]);
    assert_eq!(code(&o), EXIT_ERROR);
    assert_eq!(
        stdout(&o).trim_end(),
        "Error: Parse error: after REMOVE expected DIRECTION or PATCH, got DIRECTON. \
         Did you mean REMOVE DIRECTION? Run HELP REMOVE DIRECTION; for the syntax."
    );
}

#[test]
fn the_root_help_says_where_the_language_is() {
    let text = stdout(&larql(&["--help"]));
    let lql = text
        .lines()
        .find(|l| l.trim_start().starts_with("lql "))
        .unwrap_or_else(|| panic!("no lql line in:\n{text}"));
    assert!(lql.contains("HELP;"), "{lql}");
}
