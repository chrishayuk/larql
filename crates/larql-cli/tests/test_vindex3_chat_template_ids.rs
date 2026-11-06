//! The `larql run` leg of the `larql run` / `larql serve` parity gate.
//!
//! The built binary renders a single user turn through the container's
//! own chat template and prints the prompt ids it decodes from
//! (`--emit-ids`). They must be the ids the shared derivation
//! (`larql_inference::chat::render_conversation` + `encode`) gives —
//! the derivation the server's VINDEX3 chat routes call — and the ids
//! the server's leg (`larql-server/tests/test_openai_v3_chat_template.rs`)
//! pins for the same fixture, template and turn.
#![cfg(unix)]

#[path = "support/continuation_fixture.rs"]
#[allow(dead_code)]
mod fixture;

use std::path::Path;
use std::process::Command;

use fixture::glimmer_container;
use larql_inference::chat::{render_conversation, ChatTurn};
use larql_inference::vindex3::Vindex3Runtime;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;

/// The server leg's template, verbatim.
const MARKED_TEMPLATE: &str = "[9] {% for m in messages %}{{ m.content }} {% endfor %}[8]";
/// The server leg's single user turn.
const SINGLE_TURN_USER: &str = "[3] [4]";
/// The prompt ids both legs pin.
const SINGLE_TURN_IDS: [u32; 4] = [9, 3, 4, 8];
/// Tokens to generate: the ids are printed after the generation.
const NEW_TOKENS: &str = "1";
/// The component a container's text stack is served under.
const COMPONENT: &str = "target";
/// The `--emit-ids` line that carries the prompt ids.
const PROMPT_IDS_LABEL: &str = "prompt ids: ";

/// The prompt ids `larql run --emit-ids` reports for `user`.
fn run_prompt_ids(container: &Path, user: &str) -> Vec<u32> {
    let out = Command::new(env!("CARGO_BIN_EXE_larql"))
        .arg("run")
        .arg(container)
        .args([user, "-n", NEW_TOKENS, "--emit-ids"])
        .output()
        .expect("run larql");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr:\n{stderr}");
    let line = stderr
        .lines()
        .find_map(|l| l.split_once(PROMPT_IDS_LABEL).map(|(_, ids)| ids))
        .unwrap_or_else(|| panic!("no `{PROMPT_IDS_LABEL}` line in:\n{stderr}"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("parse {line:?}: {e}"))
}

#[test]
fn larql_run_renders_the_container_template_into_the_shared_ids() {
    let root = tempfile::tempdir().unwrap();
    let container = glimmer_container(root.path());
    std::fs::write(container.join("chat_template.jinja"), MARKED_TEMPLATE).unwrap();

    let ran = run_prompt_ids(&container, SINGLE_TURN_USER);
    assert_eq!(ran, SINGLE_TURN_IDS);

    let tokenizer = larql_vindex::load_vindex_tokenizer(&container).unwrap();
    let runtime = Vindex3Runtime::open(&container, COMPONENT, ProductionBackend::new()).unwrap();
    let shared = render_conversation(
        &container,
        runtime.family(),
        &[ChatTurn::user(SINGLE_TURN_USER)],
    )
    .unwrap()
    .encode(&tokenizer)
    .unwrap();
    assert_eq!(ran, shared);
}
