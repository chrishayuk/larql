//! The generate verb: the argmax that picks each token, the decode
//! report's timing math, and one end-to-end greedy decode over the
//! encoded fixture.

use super::super::decode::{argmax, DecodeReport};
use super::super::{run, EncodeArgs, ExecArgs, ExecBackend, Vindex3Command};
use super::fixture_dir;

#[test]
fn argmax_picks_the_largest_and_keeps_the_first_on_ties() {
    assert_eq!(argmax(&[]), None);
    assert_eq!(argmax(&[-2.0, -0.5, -1.0]), Some((1, -0.5)));
    // Ties keep the first index — the summary path's historical fold
    // behaviour, now pinned.
    assert_eq!(argmax(&[3.0, 1.0, 3.0]), Some((0, 3.0)));
}

#[test]
fn the_report_needs_at_least_one_decode_step() {
    assert_eq!(DecodeReport::from_steps(&[]), None);
}

#[test]
fn the_report_averages_the_tail_as_steady_state() {
    let report = DecodeReport::from_steps(&[2.0, 2.0, 4.0, 4.0]).unwrap();
    assert_eq!(report.decode_tokens, 4);
    assert_eq!(report.decode_seconds, 12.0);
    assert_eq!(report.mean_seconds_per_token, 3.0);
    // Steady window = the last half of the decode steps: [4.0, 4.0].
    assert_eq!(report.steady_seconds_per_token, 4.0);
}

#[test]
fn a_single_decode_step_is_its_own_steady_window() {
    let report = DecodeReport::from_steps(&[2.5]).unwrap();
    assert_eq!(report.decode_tokens, 1);
    assert_eq!(report.steady_seconds_per_token, 2.5);
}

#[test]
fn greedy_decode_runs_end_to_end_on_the_encoded_fixture() {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    run(Vindex3Command::Exec(ExecArgs {
        container: out,
        component: "target".to_string(),
        tokens: "1,2,3".to_string(),
        dump_layers: None,
        resume: false,
        backend: ExecBackend::Reference,
        // The reference arm executes canonical bytes and never looks for a
        // pack, so the source policy cannot affect this fixture.
        representation_source: "auto".to_string(),
        generate: Some(2),
        logit_dump: None,
        bank: None,
        dump_dir: None,
        draft_depth: None,
        profile: false,
    }))
    .expect("greedy decode over the fixture must complete");
}
