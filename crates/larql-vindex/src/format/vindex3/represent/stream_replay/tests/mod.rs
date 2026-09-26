//! The replay gate on a fixture, both ways, and on a real artifact pair
//! when the environment names one.

use super::super::observation_stream::tests::{identity, stream};
use super::super::observation_stream::{rederive_bank, write_stream};
use super::{
    replay_matches_report, ReplayError, REPLAY_REPORT_ENV, REPLAY_STREAM_ENV, REPORT_BANK_KEY,
    REPORT_POSITIONS_KEY, REPORT_RUN_KEY,
};

const SEQUENCES: u32 = 8;
const POSITIONS: u32 = 32;

/// A stream and the report its run would have written, side by side.
fn artifact_pair() -> (tempfile::TempDir, std::path::PathBuf, serde_json::Value) {
    let dir = tempfile::tempdir().unwrap();
    let observations = stream(SEQUENCES, POSITIONS);
    let stream_dir = dir.path().join("stream-fixture");
    write_stream(&stream_dir, &identity(SEQUENCES, POSITIONS), &observations).unwrap();
    let bank = rederive_bank(&observations);
    let report = serde_json::json!({
        REPORT_RUN_KEY: "fixture-run",
        REPORT_POSITIONS_KEY: bank.positions,
        REPORT_BANK_KEY: bank,
    });
    (dir, stream_dir, report)
}

fn write_report(dir: &std::path::Path, report: &serde_json::Value) -> std::path::PathBuf {
    let path = dir.join("report.json");
    std::fs::write(&path, serde_json::to_vec_pretty(report).unwrap()).unwrap();
    path
}

#[test]
fn a_stream_replays_exactly_to_the_bank_its_run_reported() {
    let (dir, stream_dir, report) = artifact_pair();
    let report_path = write_report(dir.path(), &report);
    let witness = replay_matches_report(&stream_dir, &report_path).unwrap();
    assert_eq!(witness.run.as_deref(), Some("fixture-run"));
    assert_eq!(witness.observations, (SEQUENCES * POSITIONS) as u64);
    assert!(!witness.stream_sha256.is_empty());
}

/// The gate can fail for the claimed reason: a report whose bank is
/// not the stream's projection is named field by field.
#[test]
fn a_report_that_disagrees_with_its_stream_is_named_field_by_field() {
    let (dir, stream_dir, mut report) = artifact_pair();
    let kl = report[REPORT_BANK_KEY]["logits"]["kl_p99"]
        .as_f64()
        .unwrap();
    report[REPORT_BANK_KEY]["logits"]["kl_p99"] = serde_json::json!(kl * 1.000_001);
    let report_path = write_report(dir.path(), &report);
    match replay_matches_report(&stream_dir, &report_path) {
        Err(ReplayError::Mismatch {
            observations,
            differing,
        }) => {
            assert_eq!(observations, (SEQUENCES * POSITIONS) as u64);
            assert_eq!(differing, vec!["logits".to_string()], "{differing:?}");
        }
        other => panic!("a perturbed bank must be refused, got {other:?}"),
    }
}

#[test]
fn a_report_claiming_a_different_position_count_is_refused() {
    let (dir, stream_dir, mut report) = artifact_pair();
    report[REPORT_POSITIONS_KEY] = serde_json::json!(8192);
    let report_path = write_report(dir.path(), &report);
    let err = replay_matches_report(&stream_dir, &report_path).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("positions (report Some(8192), stream 256)"),
        "{text}"
    );
}

#[test]
fn a_report_without_a_bank_is_an_error_not_a_pass() {
    let (dir, stream_dir, _) = artifact_pair();
    let report_path = write_report(dir.path(), &serde_json::json!({"run": "x"}));
    assert!(matches!(
        replay_matches_report(&stream_dir, &report_path),
        Err(ReplayError::Report(_))
    ));
}

/// **The real gate.** Names a stream and a report from the environment
/// and demands an exact replay. Skips, loudly, when nothing is named.
#[test]
fn a_named_real_stream_replays_exactly_to_its_reported_bank() {
    let (Some(stream_dir), Some(report)) = (
        std::env::var_os(REPLAY_STREAM_ENV),
        std::env::var_os(REPLAY_REPORT_ENV),
    ) else {
        eprintln!("skipped: set {REPLAY_STREAM_ENV} and {REPLAY_REPORT_ENV}");
        return;
    };
    let witness = replay_matches_report(
        std::path::Path::new(&stream_dir),
        std::path::Path::new(&report),
    )
    .unwrap_or_else(|e| panic!("[replay] REFUSED — {e}"));
    eprintln!(
        "[replay] EXACT: run {:?}, {} observations, stream sha256 {}",
        witness.run, witness.observations, witness.stream_sha256
    );
}
