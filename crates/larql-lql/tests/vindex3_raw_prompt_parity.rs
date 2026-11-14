//! LQL feeds the same raw-prompt ids as every other VINDEX3 surface.
//!
//! `TRACE` names each prompt token it executes, and it encodes through
//! the binding's declared BOS exactly as `INFER`, `GENERATE` and the
//! compose captures do. For each BOS shape in the shared table
//! ([`larql_inference::test_utils::BosShape`]) the traced ids must equal
//! the table's — the same table `larql run --raw`, `vindex3 observe` and
//! the server's plain completions are held to. Before the shared encoder
//! LQL prepended a declared BOS even under `add_bos_token: false`, and
//! the Granite/Qwen3 shape failed here.

use std::path::Path;

use larql_inference::test_utils::{BosShape, RAW_PROMPT};
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, DENSE_VOCAB,
};

/// TRACE's per-position line: `position N (prompt token T)`.
const PROMPT_TOKEN_MARKER: &str = "(prompt token ";

fn container_for(shape: BosShape) -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "raw-prompt-fixture",
    );
    shape.write_into(container.path(), DENSE_VOCAB);
    container
}

fn run(session: &mut Session, stmt: &str) -> Vec<String> {
    let parsed = parse(stmt).unwrap_or_else(|e| panic!("parse {stmt}: {e}"));
    session
        .execute(&parsed)
        .unwrap_or_else(|e| panic!("execute {stmt}: {e}"))
}

fn traced_prompt_ids(container: &Path) -> Vec<u32> {
    let path = container.display().to_string().replace('\\', "\\\\");
    let mut session = Session::new();
    run(&mut session, &format!("USE \"{path}\";"));
    run(&mut session, &format!("TRACE \"{RAW_PROMPT}\";"))
        .iter()
        .filter_map(|line| {
            let rest = line.split(PROMPT_TOKEN_MARKER).nth(1)?;
            rest.trim_end_matches(')').parse().ok()
        })
        .collect()
}

#[test]
fn lql_encodes_every_bos_shape_like_every_surface() {
    for shape in BosShape::ALL {
        let container = container_for(shape);
        assert_eq!(
            traced_prompt_ids(container.path()),
            shape.expected_ids(),
            "{shape:?}"
        );
    }
}
