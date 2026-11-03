//! What a batch reports, and the exit code that follows from it.

use larql_lql::{
    run_batch, run_batch_outcome, BatchOutcome, EXIT_CONTRACT_VIOLATED, EXIT_ERROR, EXIT_OK,
};

fn outcome(errors: usize, contract_violations: usize) -> BatchOutcome {
    BatchOutcome {
        lines: vec![],
        errors,
        contract_violations,
    }
}

#[test]
fn exit_codes_follow_the_table_and_errors_outrank_violations() {
    assert_eq!((EXIT_OK, EXIT_ERROR, EXIT_CONTRACT_VIOLATED), (0, 1, 2));
    assert_eq!(outcome(0, 0).exit_code(), EXIT_OK);
    assert_eq!(outcome(1, 0).exit_code(), EXIT_ERROR);
    assert_eq!(outcome(0, 1).exit_code(), EXIT_CONTRACT_VIOLATED);
    assert_eq!(outcome(1, 1).exit_code(), EXIT_ERROR);
}

#[test]
fn an_empty_or_comment_only_batch_succeeds() {
    for input in ["", "   ", "-- nothing here\n"] {
        let o = run_batch_outcome(input);
        assert_eq!(o, BatchOutcome::default(), "{input:?}");
        assert_eq!(o.exit_code(), EXIT_OK);
    }
}

#[test]
fn a_batch_keeps_going_after_a_failure_and_counts_every_one() {
    let o = run_batch_outcome("NOT A STATEMENT; STATS; ALSO NOT ONE;");
    assert_eq!(o.errors, 3, "{:?}", o.lines); // STATS has no backend here
    assert_eq!(o.lines.len(), 3, "each failure keeps its own diagnostic");
    assert!(
        o.lines.iter().all(|l| l.starts_with("Error: ")),
        "{:?}",
        o.lines
    );
    assert!(o.lines[1].contains("No backend"), "{}", o.lines[1]);
    assert_eq!(o.exit_code(), EXIT_ERROR);
}

/// Callers of the older function see exactly what they did before.
#[test]
fn run_batch_still_returns_failures_as_output() {
    let input = "NOT A STATEMENT; STATS;";
    assert_eq!(run_batch(input).unwrap(), run_batch_outcome(input).lines);
}
