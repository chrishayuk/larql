//! Shared pieces of the live-execution tests: a fixture that can run `INFER`, one edit of each kind large
//! enough to move what `INFER` prints, and what a session prints.
//!
//! `INFER` prints probabilities to a hundredth of a percent, so these tests compare the full ranking and those
//! printed probabilities, the greedy continuation and the traced next token. Bit-exact logits are compared one
//! level down, in `larql-vindex`'s `weight_patch` tests, through the same operand seam.

use std::path::{Path, PathBuf};

use larql_lql::Session;
use larql_vindex::format::vindex3::fixtures::DENSE_VOCAB;
use larql_vindex::format::vindex3::weight_patch::read_bf16_tensor;
use sha2::{Digest, Sha256};

use crate::support::*;

/// Four in-vocabulary tokens.
pub const PROMPT: &str = "[3] [17] [5] [42]";
/// How many tokens `INFER … GENERATE` continues by.
pub const NEW_TOKENS: usize = 6;
pub const PROFILE: &str = "f64-seq-rne1/v1";
const TOKENIZER_JSON: &str = "tokenizer.json";
/// A delta large enough to move the printed probabilities of a model whose stored weights are within ±0.05.
const LARGE_DELTA: &str = "64.0";
/// `64.0` and `1.0` as f64 bits: the low-rank factors.
const F64_SIXTY_FOUR: u64 = 0x4050_0000_0000_0000;
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
/// The removal's scale: it multiplies the edited row's elements by about `-2000`.
const LARGE_SCALE: &str = "2001.0";
/// Two disjoint column blocks: patch A and patch B.
pub const COLUMNS_A: std::ops::Range<usize> = 0..16;
pub const COLUMNS_B: std::ops::Range<usize> = 100..116;
/// The line `INFER` ends with: how long it took.
const TIMING_SUFFIX: &str = "ms";
const GENERATED_IDS: &str = "  ids:";
const TRACED_NEXT: &str = "next token";
const LIVE_DIGEST: &str = "  sha256 ";

/// The three kinds of stored-weight statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    UpdateWeights,
    RemoveDirection,
    AddLowRank,
}

pub const KINDS: [Kind; 3] = [Kind::UpdateWeights, Kind::RemoveDirection, Kind::AddLowRank];

/// `[i]` ↔ id i, split on whitespace only, so a prompt is several tokens.
fn tokenizer_json(vocab: usize) -> String {
    let entries: serde_json::Map<String, serde_json::Value> =
        (0..vocab).map(|i| (format!("[{i}]"), i.into())).collect();
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": entries, "unk_token": "[0]"}
    })
    .to_string()
}

/// The standard fixture with a tokenizer, so `INFER` runs on it.
pub fn inferable() -> Fixture {
    let fx = Fixture::new();
    std::fs::write(
        fx.container.join(TOKENIZER_JSON),
        tokenizer_json(DENSE_VOCAB),
    )
    .unwrap();
    fx
}

fn list(columns: std::ops::Range<usize>) -> String {
    columns
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The statement that stages one edit of `kind` over `columns`, writing any operand file it needs as `name`.
pub fn statement(fx: &Fixture, kind: Kind, columns: std::ops::Range<usize>, name: &str) -> String {
    let ([rows, _], _) = read_bf16_tensor(&fx.container, TENSOR).unwrap();
    let hex = |bits: u64, n: usize| vec![format!("{bits:016x}"); n];
    match kind {
        Kind::UpdateWeights => format!(
            "UPDATE WEIGHTS SET DELTA = {LARGE_DELTA} WHERE TENSOR = \"{TENSOR}\" AND ROW = {ROW} AND COL IN ({});",
            list(columns)
        ),
        Kind::RemoveDirection => {
            let bits: Vec<String> = (0..rows)
                .map(|r| format!("{:016x}", if r == ROW { F64_ONE } else { 0 }))
                .collect();
            let path = fx.path(name);
            std::fs::write(&path, serde_json::json!({ "bits": bits }).to_string()).unwrap();
            format!(
                "REMOVE DIRECTION \"{}\" FROM \"{TENSOR}\" COLUMNS ({}) SCALE {LARGE_SCALE} NUMERICS \"{PROFILE}\";",
                q(path),
                list(columns)
            )
        }
        Kind::AddLowRank => {
            let path = fx.path(name);
            let doc = serde_json::json!({
                "rank": 1,
                "a": hex(F64_SIXTY_FOUR, rows),
                "b": hex(F64_ONE, columns.len()),
            });
            std::fs::write(&path, doc.to_string()).unwrap();
            format!(
                "ADD LOW RANK \"{}\" TO \"{TENSOR}\" COLUMNS ({}) NUMERICS \"{PROFILE}\";",
                q(path),
                list(columns)
            )
        }
    }
}

/// Record one edit of `kind` over `columns` into a `.vlp`, from a fresh session, and return its path.
pub fn save_vlp(fx: &Fixture, kind: Kind, columns: std::ops::Range<usize>, name: &str) -> PathBuf {
    let vlp = fx.path(&format!("{name}.vlp"));
    let mut s = session_on(&fx.container);
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    run(
        &mut s,
        &statement(fx, kind, columns, &format!("{name}.json")),
    );
    run(&mut s, "SAVE PATCH;");
    vlp
}

pub fn apply(s: &mut Session, vlp: &Path) -> Vec<String> {
    run(s, &format!("APPLY PATCH \"{}\";", q(vlp)))
}

pub fn remove(s: &mut Session, vlp: &Path) -> Vec<String> {
    run(s, &format!("REMOVE PATCH \"{}\";", q(vlp)))
}

pub fn infer_statement() -> String {
    format!("INFER \"{PROMPT}\" TOP {DENSE_VOCAB};")
}

/// Everything `INFER` prints for [`PROMPT`] over the whole vocabulary but its timing: the ranking, each token's
/// id and its probability to a hundredth of a percent.
pub fn infer(s: &mut Session) -> Vec<String> {
    let mut out = run(s, &infer_statement());
    assert!(out.pop().is_some_and(|l| l.ends_with(TIMING_SUFFIX)));
    assert_eq!(
        out.len(),
        DENSE_VOCAB + 1,
        "a header and one line per token"
    );
    out
}

/// The token ids `INFER … GENERATE` continues [`PROMPT`] with.
pub fn generate(s: &mut Session) -> String {
    run(s, &format!("INFER \"{PROMPT}\" GENERATE {NEW_TOKENS};"))
        .into_iter()
        .find(|l| l.starts_with(GENERATED_IDS))
        .expect("the generated ids")
}

/// The greedy next token `TRACE` reports for [`PROMPT`].
pub fn trace(s: &mut Session) -> String {
    run(s, &format!("TRACE \"{PROMPT}\";"))
        .into_iter()
        .find(|l| l.starts_with(TRACED_NEXT))
        .expect("the traced next token")
}

/// What a session computes, at the most exact level LQL prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Printed {
    pub infer: Vec<String>,
    pub generated: String,
    pub traced: String,
}

pub fn printed(s: &mut Session) -> Printed {
    Printed {
        infer: infer(s),
        generated: generate(s),
        traced: trace(s),
    }
}

/// The SHA-256 digests `lines` report for live tensors, in order.
pub fn live_digests(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter_map(|l| l.split_once(LIVE_DIGEST).map(|(_, d)| d.trim().to_string()))
        .collect()
}

pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn has(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|l| l.contains(needle))
}
