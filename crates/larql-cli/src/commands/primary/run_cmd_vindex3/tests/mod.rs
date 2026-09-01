//! `larql run` on a VINDEX3 container: detection, the refusals, and text
//! out of the dense fixture through the container's own tokenizer.

use std::path::{Path, PathBuf};

use clap::Parser;
use larql_vindex::format::filenames::{GENERATION_CONFIG_JSON, INDEX_JSON, TOKENIZER_JSON};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, DENSE_VOCAB,
};

use super::super::run_cmd::RunArgs;
use super::{is_vindex3_container, resolved_display_name, run_to};

/// A prompt that is its own id list under the fixture tokenizer.
const PROMPT: &str = "[1] [2] [3]";

/// `RunArgs` is `clap::Args`; parsing it through a shell keeps every
/// default exactly what the binary would have used.
#[derive(Parser)]
struct Shell {
    #[command(flatten)]
    run: RunArgs,
}

fn args(argv: &[&str]) -> RunArgs {
    Shell::parse_from(std::iter::once("larql").chain(argv.iter().copied())).run
}

/// A WordLevel tokenizer over the fixture's vocabulary: token `[i]` is
/// id `i`, joined back with spaces on decode. No larql-inference
/// dependency and no merges — a prompt is its own ids.
///
/// Split on whitespace ONLY (`WhitespaceSplit`). The `Whitespace`
/// pre-tokenizer also splits at punctuation, which turned `[1]` into
/// three unknown pieces and every prompt into a run of id 0 — and every
/// test still passed, because none of them looked at the ids until the
/// status stream was captured. `ids_and_timings_go_to_the_status_stream`
/// now pins the encoding.
fn word_level_tokenizer(vocab: usize) -> String {
    let entries: Vec<String> = (0..vocab).map(|i| format!("\"[{i}]\":{i}")).collect();
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"WhitespaceSplit\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

/// The dense fixture as a container, with or without its tokenizer.
fn fixture_container(root: &Path, with_tokenizer: bool) -> PathBuf {
    let checkpoint = root.join("checkpoint");
    let container = root.join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "dense");
    if with_tokenizer {
        std::fs::write(
            container.join(TOKENIZER_JSON),
            word_level_tokenizer(DENSE_VOCAB),
        )
        .unwrap();
    }
    container
}

/// Run with `argv` after the model path, feeding `input` to the chat
/// loop, and return everything written to stdout and to the status
/// stream, separately.
fn run_capturing_with_status(
    container: &Path,
    argv: &[&str],
    input: &str,
) -> Result<(String, String), String> {
    let model = container.to_str().unwrap();
    let a = args(&[&[model], argv].concat());
    let mut out = Vec::new();
    let mut status = Vec::new();
    let mut input = std::io::Cursor::new(input.as_bytes());
    run_to(container, &a, &mut input, &mut out, &mut status).map_err(|e| e.to_string())?;
    Ok((
        String::from_utf8(out).unwrap(),
        String::from_utf8(status).unwrap(),
    ))
}

/// [`run_capturing_with_status`], stdout only.
fn run_capturing(container: &Path, argv: &[&str], input: &str) -> Result<String, String> {
    run_capturing_with_status(container, argv, input).map(|(out, _)| out)
}

/// The fixture container's declared name, read from its own index.
fn declared_name(container: &Path) -> String {
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(container.join(INDEX_JSON)).unwrap())
            .unwrap();
    index["model"].as_str().unwrap().to_string()
}

/// Rewrite the fixture container's declared name to `name`.
fn declare_name(container: &Path, name: &str) {
    let path = container.join(INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    index["model"] = serde_json::Value::String(name.to_string());
    std::fs::write(&path, index.to_string()).unwrap();
}

/// Every whitespace-separated piece of a line is a fixture token `[n]`
/// with `n` inside the vocabulary.
fn assert_fixture_tokens(line: &str, expected: usize) {
    let pieces: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(pieces.len(), expected, "{line:?}");
    for piece in pieces {
        let inner = piece
            .strip_prefix('[')
            .and_then(|p| p.strip_suffix(']'))
            .unwrap_or_else(|| panic!("{piece:?} is not a fixture token"));
        let id: usize = inner.parse().unwrap();
        assert!(id < DENSE_VOCAB, "{piece:?} is outside the vocabulary");
    }
}

#[test]
fn detection_answers_only_for_a_vindex3_container() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    assert!(is_vindex3_container(&container));

    let empty = tempfile::tempdir().unwrap();
    assert!(!is_vindex3_container(empty.path()));
    assert!(!is_vindex3_container(&empty.path().join("absent")));

    // A VINDEX2 index is the dense path's business, not this arm's.
    let v2 = tempfile::tempdir().unwrap();
    std::fs::write(v2.path().join(INDEX_JSON), r#"{"version": 1}"#).unwrap();
    assert!(!is_vindex3_container(v2.path()));
}

#[test]
fn a_prompt_streams_tokens_from_the_containers_own_tokenizer() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let out = run_capturing(&container, &[PROMPT, "--max-tokens", "3"], "").unwrap();
    let line = out
        .strip_suffix('\n')
        .expect("the generation ends with a newline");
    assert_fixture_tokens(line, 3);
}

#[test]
fn the_same_prompt_produces_the_same_text() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let argv = [PROMPT, "--max-tokens", "4"];
    assert_eq!(
        run_capturing(&container, &argv, "").unwrap(),
        run_capturing(&container, &argv, "").unwrap()
    );
}

#[test]
fn emit_ids_and_verbose_report_on_stderr_and_leave_stdout_alone() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let plain = run_capturing(&container, &[PROMPT, "--max-tokens", "2"], "").unwrap();
    let chatty = run_capturing(
        &container,
        &[PROMPT, "--max-tokens", "2", "--emit-ids", "--verbose"],
        "",
    )
    .unwrap();
    assert_eq!(plain, chatty);
}

#[test]
fn each_chat_turn_starts_from_a_fresh_continuation_state() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    // The first and third turns are the same prompt with a different
    // turn between them; blank lines are skipped, not answered.
    let out = run_capturing(
        &container,
        &["--max-tokens", "2"],
        "[1] [2]\n\n[5] [6] [7]\n[1] [2]\n",
    )
    .unwrap();
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3, "{out:?}");
    for line in &lines {
        assert_fixture_tokens(line, 2);
    }
    assert_eq!(lines[0], lines[2], "state leaked between turns");
}

#[test]
fn dense_engine_flags_are_refused_by_name_before_anything_loads() {
    let root = tempfile::tempdir().unwrap();
    // No tokenizer, no plan: the refusal has to come first.
    let container = fixture_container(root.path(), false);
    let err = run_capturing(
        &container,
        &[PROMPT, "--experts", "--top", "5", "--engine", "standard"],
        "",
    )
    .expect_err("dense-engine flags are not honoured");
    for flag in ["--experts", "--top", "--engine"] {
        assert!(err.contains(flag), "{err}");
    }
    assert!(!err.contains("--max-tokens"), "{err}");
}

#[test]
fn a_container_without_a_tokenizer_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), false);
    let err = run_capturing(&container, &[PROMPT], "")
        .expect_err("text cannot be tokenised without a tokenizer");
    assert!(err.contains(TOKENIZER_JSON), "{err}");
}

#[cfg(not(all(feature = "gpu", target_os = "macos")))]
#[test]
fn metal_is_refused_where_there_is_no_metal() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let err = run_capturing(&container, &[PROMPT, "--metal"], "")
        .expect_err("--metal cannot be honoured on this build");
    assert!(err.contains("--metal"), "{err}");
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
#[test]
fn metal_serves_the_fixture_through_the_same_shell() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let out = run_capturing(&container, &[PROMPT, "--max-tokens", "3", "--metal"], "").unwrap();
    assert_fixture_tokens(out.trim_end(), 3);
}

/// Point the container's `generation_config.json` at a stop.
fn declare_stop(container: &Path, config: serde_json::Value) {
    std::fs::write(container.join(GENERATION_CONFIG_JSON), config.to_string()).unwrap();
}

#[test]
fn generation_ends_at_the_containers_declared_eos_before_it_is_printed() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let argv = [PROMPT, "--max-tokens", "3"];
    // What the model says with nothing declared, so the stop can be
    // placed on a token it is known to produce.
    let free = run_capturing(&container, &argv, "").unwrap();
    let pieces: Vec<&str> = free.split_whitespace().collect();
    let (first, second) = (pieces[0].to_string(), pieces[1].to_string());
    let first_id: u32 = first
        .trim_matches(|c| c == '[' || c == ']')
        .parse()
        .unwrap();

    // By id: the EOS is never decoded, so nothing precedes the newline.
    declare_stop(&container, serde_json::json!({ "eos_token_id": first_id }));
    assert_eq!(run_capturing(&container, &argv, "").unwrap(), "\n");

    // By surface form, through the same path the dense engine uses.
    declare_stop(
        &container,
        serde_json::json!({ "stop_strings": [first.clone()] }),
    );
    assert_eq!(run_capturing(&container, &argv, "").unwrap(), "\n");

    // A stop on the second token leaves exactly the first one printed.
    declare_stop(&container, serde_json::json!({ "stop_strings": [second] }));
    let out = run_capturing(&container, &argv, "").unwrap();
    assert_eq!(out.trim(), first, "{out:?}");
}

#[test]
fn the_declared_name_wins_and_the_directory_is_the_explicit_fallback() {
    let dir = Path::new("/models/qwen3-0.6b.vindex3");
    assert_eq!(resolved_display_name("Qwen3-0.6B", dir), "Qwen3-0.6B");
    assert_eq!(resolved_display_name("", dir), "qwen3-0.6b.vindex3");
    assert_eq!(resolved_display_name("", Path::new("/")), "container");
}

#[test]
fn the_banner_and_the_verbose_report_show_the_containers_declared_name() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let declared = declared_name(&container);
    let directory = container.file_name().unwrap().to_str().unwrap().to_string();
    assert!(!declared.is_empty() && declared != directory);

    // Chat mode: the banner names the model; verbose names it again on
    // the load line. Both come from one helper, so both say the same.
    let (_, status) = run_capturing_with_status(&container, &["--verbose"], "").unwrap();
    assert!(
        status.contains(&format!("— {declared} (Ctrl-D to exit)")),
        "{status}"
    );
    assert!(status.contains(&format!("] {declared} (")), "{status}");
    assert!(
        !status.contains(&format!("— {directory} (")),
        "filesystem identity leaked into the banner: {status}"
    );
    assert!(
        !status.contains(&format!("] {directory} (")),
        "filesystem identity leaked into the verbose report: {status}"
    );
}

#[test]
fn a_nameless_container_falls_back_to_its_directory_and_says_so_once() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    declare_name(&container, "");
    let directory = container.file_name().unwrap().to_str().unwrap().to_string();
    let (_, status) = run_capturing_with_status(&container, &["--verbose"], "").unwrap();
    assert!(
        status.contains(&format!("— {directory} (Ctrl-D to exit)")),
        "{status}"
    );
    assert!(status.contains(&format!("] {directory} (")), "{status}");
}

#[test]
fn ids_and_timings_go_to_the_status_stream() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let (out, status) = run_capturing_with_status(
        &container,
        &[PROMPT, "--max-tokens", "2", "--emit-ids", "--verbose"],
        "",
    )
    .unwrap();
    assert_fixture_tokens(out.trim_end(), 2);
    assert!(status.contains("prompt ids: [1, 2, 3]"), "{status}");
    assert!(status.contains("generated ids: ["), "{status}");
    assert!(status.contains("prompt tokens in"), "{status}");
}
