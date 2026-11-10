//! `--bank` on lowered backends must refuse, not silently ignore the
//! manifest. Device-free: the check runs before any Metal session is
//! touched, so this covers the same code path a real run takes without
//! needing a GPU.

use std::path::Path;

use super::super::run::refuse_bank_on_lowered;

#[test]
fn no_bank_path_is_fine() {
    assert!(refuse_bank_on_lowered(None).is_ok());
}

#[test]
fn a_bank_path_is_refused_with_a_clear_message() {
    let err = refuse_bank_on_lowered(Some(Path::new("/tmp/some-manifest.jsonl")))
        .expect_err("--bank must be refused on lowered backends");
    let message = err.to_string();
    assert!(
        message.contains("--bank"),
        "error should name the offending flag: {message}"
    );
    assert!(
        message.contains("lowered"),
        "error should say why (lowered backends specifically): {message}"
    );
}

// ── `--stop-at-eos` (#752) on the lowered drivers ────────────────────
//
// Device-free: the stop set is resolved (or refused) before any Metal
// state, and the generated-ids contract is a pure function both lowered
// loops end with, so neither needs a GPU.

use std::collections::HashSet;

use clap::Parser;
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;

use super::super::super::plugins::Plugins;
use super::super::super::prepare::{lowered_formats, prepare, DEFAULT_COMPONENT};
use super::super::super::{ExecArgs, ExecBackend};
use super::super::run::{run_lowered, truncate_at_stop};

#[test]
fn the_generated_ids_end_at_the_first_stop_which_is_kept() {
    let stop = HashSet::from([7]);
    let mut out = vec![1, 2, 7, 3, 7];
    truncate_at_stop(&mut out, &stop, 10);
    assert_eq!(out, vec![1, 2, 7]);
    // The count still bounds a run that stops later than it.
    let mut out = vec![1, 2, 3, 7];
    truncate_at_stop(&mut out, &stop, 2);
    assert_eq!(out, vec![1, 2]);
    // No stop in the ids, or no stop at all: only the count applies.
    let mut out = vec![1, 2, 3];
    truncate_at_stop(&mut out, &stop, 10);
    assert_eq!(out, vec![1, 2, 3]);
    let mut out = vec![7, 7];
    truncate_at_stop(&mut out, &HashSet::new(), 1);
    assert_eq!(out, vec![7]);
}

#[derive(Parser)]
struct ExecCli {
    #[command(flatten)]
    exec: ExecArgs,
}

#[test]
fn a_lowered_run_with_no_resolvable_stop_is_refused_before_any_device_state() {
    let root = tempfile::tempdir().unwrap();
    let checkpoint = root.path().join("checkpoint");
    let container = root.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "dense");
    let prepared = prepare(
        &container,
        DEFAULT_COMPONENT,
        ExecBackend::Reference,
        RepresentationSource::Auto,
        &Plugins::none(),
    )
    .unwrap();
    let args = ExecCli::try_parse_from([
        "exec",
        container.to_str().unwrap(),
        "--tokens",
        "1,2,3",
        "--generate",
        "4",
        "--stop-at-eos",
        "--backend",
        "metal-lowered",
    ])
    .unwrap()
    .exec;
    let (formats, label) = lowered_formats(args.backend).expect("a lowered arm");
    let err = run_lowered(
        &args,
        &[1, 2, 3],
        &prepared.plan,
        &prepared.store,
        formats,
        label,
    )
    .expect_err("no stop resolves: refused")
    .to_string();
    assert!(err.contains("--stop-at-eos"), "{err}");
    assert!(err.contains("declares no end-of-sequence token"), "{err}");
}
