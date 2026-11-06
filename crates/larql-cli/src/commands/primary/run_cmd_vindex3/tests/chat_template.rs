//! `larql run` / `larql chat` on a container that carries a chat
//! template: the template wraps the one-shot prompt and every chat turn,
//! history included, and `--raw` turns it off.

use super::*;

/// Renders as `[9]` followed by every turn's content — a run of fixture
/// tokens, so the prompt ids show exactly what the template produced.
const MARKED: &str = "[9] {% for m in messages %}{{ m.content }} {% endfor %}";

/// The fixture container with [`MARKED`] as its `chat_template.jinja`.
fn templated_container(root: &Path) -> PathBuf {
    let container = fixture_container(root, true);
    std::fs::write(container.join("chat_template.jinja"), MARKED).unwrap();
    container
}

/// The `prompt ids: [...]` lines `--emit-ids` wrote, parsed.
fn prompt_ids(status: &str) -> Vec<Vec<u32>> {
    status
        .lines()
        .filter_map(|l| l.split_once("prompt ids: ").map(|(_, ids)| ids))
        .map(|ids| serde_json::from_str(ids).unwrap())
        .collect()
}

#[test]
fn a_one_shot_prompt_goes_through_the_template() {
    let root = tempfile::tempdir().unwrap();
    let container = templated_container(root.path());
    let (_, status) = run_capturing_with_status(
        &container,
        &[PROMPT, "--max-tokens", "2", "--emit-ids", "--verbose"],
        "",
    )
    .unwrap();
    assert_eq!(prompt_ids(&status), vec![vec![9, 1, 2, 3]], "{status}");
    assert!(status.contains("prompt: chat template"), "{status}");
}

#[test]
fn raw_sends_the_prompt_verbatim() {
    let root = tempfile::tempdir().unwrap();
    let container = templated_container(root.path());
    let (_, status) = run_capturing_with_status(
        &container,
        &[
            PROMPT,
            "--max-tokens",
            "2",
            "--emit-ids",
            "--raw",
            "--verbose",
        ],
        "",
    )
    .unwrap();
    assert_eq!(prompt_ids(&status), vec![vec![1, 2, 3]], "{status}");
    assert!(status.contains("prompt: raw"), "{status}");
}

#[test]
fn chat_turns_carry_the_conversation_so_far() {
    let root = tempfile::tempdir().unwrap();
    let container = templated_container(root.path());
    let (out, status) = run_capturing_with_status(
        &container,
        &["--max-tokens", "2", "--emit-ids"],
        "[1] [2]\n[5]\n",
    )
    .unwrap();
    let replies: Vec<&str> = out.lines().collect();
    assert_eq!(replies.len(), 2, "{out:?}");
    let reply_ids: Vec<u32> = replies[0]
        .split_whitespace()
        .map(|t| t.trim_matches(|c| c == '[' || c == ']').parse().unwrap())
        .collect();
    let ids = prompt_ids(&status);
    assert_eq!(ids[0], vec![9, 1, 2]);
    let expected: Vec<u32> = [vec![9, 1, 2], reply_ids, vec![5]].concat();
    assert_eq!(ids[1], expected, "{status}");
}

#[test]
fn a_container_without_a_template_runs_raw_and_says_so() {
    let root = tempfile::tempdir().unwrap();
    let container = fixture_container(root.path(), true);
    let (_, status) =
        run_capturing_with_status(&container, &[PROMPT, "--max-tokens", "2", "--emit-ids"], "")
            .unwrap();
    assert_eq!(prompt_ids(&status), vec![vec![1, 2, 3]], "{status}");
    assert!(status.contains("carries no chat template"), "{status}");
}
