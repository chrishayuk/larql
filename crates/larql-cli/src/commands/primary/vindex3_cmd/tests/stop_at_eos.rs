//! `vindex3 exec --generate --stop-at-eos` (#752): the flag is honoured on
//! the interpreter backends, ends the decode at the first stop id, and is
//! refused — never silently ignored — when the container declares no
//! end-of-sequence token.

use std::collections::HashSet;
use std::path::Path;

use clap::Parser;
use larql_vindex::format::filenames::GENERATION_CONFIG_JSON;
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;

use super::super::generate::{run_generate, stop_ids};
use super::super::plugins::Plugins;
use super::super::prepare::{prepare, DEFAULT_COMPONENT};
use super::super::{run, ExecArgs, ExecBackend, Vindex3Command};

const PROMPT: [u32; 3] = [1, 2, 3];
const NEW_TOKENS: usize = 4;
const ENGINE: &str = "test";

#[derive(Parser)]
struct ExecCli {
    #[command(flatten)]
    exec: ExecArgs,
}

/// `vindex3 exec <container> --tokens 1,2,3 --generate N` plus `extra`.
fn exec_args(container: &Path, extra: &[&str]) -> ExecArgs {
    let generate = NEW_TOKENS.to_string();
    let base = [
        "exec",
        container.to_str().unwrap(),
        "--tokens",
        "1,2,3",
        "--generate",
        generate.as_str(),
    ];
    ExecCli::try_parse_from(base.iter().chain(extra))
        .expect("valid exec arguments")
        .exec
}

/// The dense fixture, encoded into `<root>/container`.
fn encoded_dense(root: &Path) -> std::path::PathBuf {
    let checkpoint = root.join("checkpoint");
    let container = root.join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "dense");
    container
}

/// Greedy decode over the dense fixture on the reference arm.
fn decode(container: &Path, stop: &HashSet<u32>) -> Vec<u32> {
    let prepared = prepare(
        container,
        DEFAULT_COMPONENT,
        ExecBackend::Reference,
        RepresentationSource::Auto,
        &Plugins::none(),
    )
    .unwrap();
    run_generate(
        &ReferenceBackend::new(),
        ENGINE,
        &PROMPT,
        NEW_TOKENS,
        &prepared.plan,
        &prepared.store,
        stop,
    )
    .unwrap()
}

#[test]
fn without_the_flag_no_stop_is_resolved_and_the_full_count_decodes() {
    let root = tempfile::tempdir().unwrap();
    let container = encoded_dense(root.path());
    let args = exec_args(&container, &[]);
    assert!(stop_ids(&args).unwrap().is_empty());
    assert_eq!(decode(&container, &HashSet::new()).len(), NEW_TOKENS);
}

#[test]
fn the_decode_halts_after_emitting_the_first_stop_id() {
    let root = tempfile::tempdir().unwrap();
    let container = encoded_dense(root.path());
    let full = decode(&container, &HashSet::new());
    // Stop at the first generated id: exactly that id comes out.
    assert_eq!(decode(&container, &HashSet::from([full[0]])), vec![full[0]]);
    // Stop at a later id: decoding runs up to and including its first
    // occurrence.
    let last = *full.last().unwrap();
    let first_at = full.iter().position(|&id| id == last).unwrap();
    assert_eq!(
        decode(&container, &HashSet::from([last])),
        full[..=first_at].to_vec()
    );
}

#[test]
fn the_flag_resolves_the_containers_declared_stops() {
    let root = tempfile::tempdir().unwrap();
    let container = encoded_dense(root.path());
    std::fs::write(
        container.join(GENERATION_CONFIG_JSON),
        serde_json::json!({"eos_token_id": [5, 6]}).to_string(),
    )
    .unwrap();
    let args = exec_args(&container, &["--stop-at-eos"]);
    assert_eq!(stop_ids(&args).unwrap(), HashSet::from([5, 6]));
}

#[test]
fn a_container_declaring_no_stop_is_refused_on_the_production_arm() {
    // Before #752 the interpreter arms ignored the flag and decoded the
    // full count as if it had been honoured.
    let root = tempfile::tempdir().unwrap();
    let container = encoded_dense(root.path());
    let _ = std::fs::remove_file(container.join(GENERATION_CONFIG_JSON));
    let args = exec_args(&container, &["--stop-at-eos", "--backend", "production"]);
    assert!(matches!(args.backend, ExecBackend::Production));
    let err = run(Vindex3Command::Exec(args)).unwrap_err().to_string();
    assert!(err.contains("--stop-at-eos"), "{err}");
    assert!(err.contains("declares no end-of-sequence token"), "{err}");
}
