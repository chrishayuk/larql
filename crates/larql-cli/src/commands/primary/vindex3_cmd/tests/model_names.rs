//! `larql vindex3` verbs take the identifier `larql run` takes: every
//! model argument is resolved before the verb runs, a path stays the path
//! it was, and `observe` runs end to end on a container named by
//! shorthand, by `owner/name` and by path.

use std::path::{Path, PathBuf};

use clap::Parser;
use larql_inference::vindex3::{EventKind, RunRecord};

use super::super::model_args::{model_slots, resolve_with};
use super::super::{run, EncodeArgs, Vindex3Command};
use super::fixture_dir;
use crate::commands::primary::cache::installed::{resolve_installed_model_in, CacheRoots};

/// The repo the fixture container is cached under.
const REPO: &str = "acme/glimmer-mini";
/// Its shorthand.
const SHORTHAND: &str = "glimmer-mini";
/// A commit id the cache layout keys the snapshot on.
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
/// The name every model argument is spelled as in the slot table.
const NAME: &str = "Qwen3-0.6B";
/// What the stand-in resolver maps every model argument to.
const RESOLVED: &str = "resolved-container";

#[derive(Parser)]
struct Vindex3Cli {
    #[command(subcommand)]
    cmd: Vindex3Command,
}

fn parse(argv: &[&str]) -> Vindex3Command {
    Vindex3Cli::try_parse_from(std::iter::once("vindex3").chain(argv.iter().copied()))
        .unwrap_or_else(|e| panic!("{argv:?}: {e}"))
        .cmd
}

/// Every verb, with each model argument spelled [`NAME`], and how many
/// model arguments it has.
fn every_verb() -> Vec<(Vec<&'static str>, usize)> {
    vec![
        (vec!["plan", "ckpt"], 0),
        (vec!["encode", "ckpt", "--output", "out"], 0),
        (vec!["inspect", NAME], 1),
        (vec!["references", NAME], 1),
        (vec!["verify", "ckpt", "--container", NAME], 1),
        (vec!["ops", NAME], 1),
        (vec!["exec", NAME, "--tokens", "1"], 1),
        (vec!["represent", NAME, "--output", "out"], 1),
        (
            vec!["observe", NAME, "--tokens", "1", "--record", "r.jsonl"],
            1,
        ),
        (vec!["sensitivity", NAME, "--output", "out"], 1),
        (
            vec![
                "consequence",
                NAME,
                "--moments",
                "m.json",
                "--expect-token-digest",
                "d",
                "--output",
                "out",
            ],
            1,
        ),
        (
            vec![
                "token-bank",
                "export",
                NAME,
                "--prompts",
                "p.json",
                "--output",
                "bank",
            ],
            1,
        ),
        (
            vec![
                "token-bank",
                "import",
                NAME,
                "--ids",
                "ids.json",
                "--output",
                "bank",
            ],
            1,
        ),
        (vec!["token-bank", "check", "bank", "--container", NAME], 1),
        (
            vec![
                "measure",
                "--reference",
                NAME,
                "--reference-backend",
                "reference",
                "--candidate",
                NAME,
                "--candidate-backend",
                "reference",
                "--bank",
                "bank",
                "--sequences",
                "1",
                "--label",
                "l",
                "--output",
                "out",
            ],
            2,
        ),
        (
            vec![
                "auto-rep",
                "init",
                NAME,
                "--bank",
                "bank",
                "--sequences",
                "1",
                "--output",
                "out",
            ],
            1,
        ),
        (
            vec![
                "auto-rep",
                "run",
                "--campaign",
                "c.json",
                "--source",
                NAME,
                "--bank",
                "bank",
                "--workdir",
                "w",
                "--resume",
                "--output",
                "out",
                "--runs",
                "runs",
            ],
            1,
        ),
    ]
}

#[test]
fn every_model_argument_and_only_model_arguments_are_resolved() {
    for (argv, models) in every_verb() {
        let mut cmd = parse(&argv);
        let slots: Vec<PathBuf> = model_slots(&mut cmd)
            .into_iter()
            .map(|p| p.clone())
            .collect();
        assert_eq!(slots.len(), models, "{argv:?}");
        assert!(slots.iter().all(|p| p == Path::new(NAME)), "{argv:?}");

        resolve_with(&mut cmd, &|p| {
            assert_eq!(p, Path::new(NAME), "{argv:?}: only model arguments");
            Ok(PathBuf::from(RESOLVED))
        })
        .unwrap();
        let after: Vec<PathBuf> = model_slots(&mut cmd)
            .into_iter()
            .map(|p| p.clone())
            .collect();
        assert!(after.iter().all(|p| p == Path::new(RESOLVED)), "{argv:?}");
    }
}

#[test]
fn a_resolution_failure_stops_the_verb_with_the_resolver_error() {
    let mut cmd = parse(&["inspect", NAME]);
    let err = resolve_with(&mut cmd, &|_| Err("not on this machine".into())).unwrap_err();
    assert_eq!(err.to_string(), "not on this machine");
}

#[test]
fn run_refuses_a_name_that_is_not_installed_before_the_verb_starts() {
    // No cache holds this name, so the lookup misses whatever is cached.
    let cmd = parse(&["inspect", "larql-test-no-such-model-anywhere"]);
    let err = run(cmd).unwrap_err().to_string();
    assert!(err.contains("never downloads"), "{err}");
}

#[test]
fn run_hands_a_missing_path_to_the_verb_unchanged() {
    let cmd = parse(&["inspect", "./larql-test-no-such-container"]);
    let err = run(cmd).unwrap_err().to_string();
    assert!(!err.contains("never downloads"), "{err}");
    assert!(err.contains("No such file or directory"), "{err}");
}

/// The fixture encoded straight into a hub-layout cache as `REPO`, and
/// the roots that cache lives under.
fn cached_fixture() -> (tempfile::TempDir, CacheRoots, PathBuf) {
    let checkpoint = fixture_dir(true);
    let caches = tempfile::tempdir().unwrap();
    let roots = CacheRoots {
        hub: caches.path().join("hub"),
        local: caches.path().join("local"),
        encoded: caches.path().join("encoded"),
    };
    let (owner, name) = REPO.split_once('/').unwrap();
    let snapshot = roots
        .encoded
        .join(format!("models--{owner}--{name}/snapshots/{COMMIT}"));
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![checkpoint.path().to_path_buf()],
        output: snapshot.clone(),
    }))
    .unwrap();
    (caches, roots, snapshot)
}

/// `vindex3 observe <model> --tokens 1,2,3 --record <record>`, resolved
/// against `roots`, run, and read back.
fn observe(model: &str, roots: &CacheRoots, record: &Path) -> RunRecord {
    let record_arg = record.to_str().unwrap();
    let mut cmd = parse(&[
        "observe", model, "--tokens", "1,2,3", "--record", record_arg, "--run-id", "named",
    ]);
    resolve_with(&mut cmd, &|p| resolve_installed_model_in(p, roots)).unwrap();
    run(cmd).unwrap();
    RunRecord::read_jsonl(record).unwrap()
}

/// A record's events without their wall-clock timestamps.
fn trace(record: &RunRecord) -> Vec<(usize, &EventKind)> {
    record
        .events
        .iter()
        .map(|e| (e.position, &e.event))
        .collect()
}

#[test]
fn observe_runs_on_a_container_named_by_shorthand_owner_name_and_path() {
    let (caches, roots, snapshot) = cached_fixture();
    let by_path = observe(
        snapshot.to_str().unwrap(),
        &roots,
        &caches.path().join("path.jsonl"),
    );
    for model in [SHORTHAND, REPO, &format!("hf://{REPO}@{COMMIT}")] {
        let record = caches.path().join("named.jsonl");
        let named = observe(model, &roots, &record);
        assert_eq!(
            named.provenance_fingerprint, by_path.provenance_fingerprint,
            "{model}"
        );
        assert_eq!(trace(&named), trace(&by_path), "{model}");
    }
}
