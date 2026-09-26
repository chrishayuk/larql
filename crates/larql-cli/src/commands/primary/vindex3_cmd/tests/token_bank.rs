//! `vindex3 token-bank`: export then check round-trips, and check refuses a
//! bank for another container. The format's own refusals are pinned in
//! `represent/token_bank_tests.rs`.

use std::path::Path;

use crate::commands::primary::vindex3_cmd::token_bank::*;
use larql_vindex::format::vindex3::represent::token_bank::TOKENIZER_FILE;

/// A container directory holding only a tokenizer over a handful of words.
fn write_container(dir: &Path) {
    super::write_word_tokenizer(dir, &["one", "two", "three", "four"]);
}

fn write_prompts(path: &Path) {
    let prompts = serde_json::json!({
        "bank": "cli-fixture",
        "prompts": [
            {"id": "p-000", "category": "prose", "text": "one two three four"},
            {"id": "p-001", "category": "prose", "text": "four three two"}
        ]
    });
    std::fs::write(path, serde_json::to_vec(&prompts).unwrap()).unwrap();
}

#[test]
fn export_then_check_accepts_the_bank_for_its_own_container() {
    let tmp = tempfile::tempdir().unwrap();
    let container = tmp.path().join("model");
    write_container(&container);
    let prompts = tmp.path().join("prompts.json");
    write_prompts(&prompts);
    let bank = tmp.path().join("bank");
    run(TokenBankArgs {
        command: TokenBankCommand::Export(ExportArgs {
            container: container.clone(),
            prompts,
            max_tokens: DEFAULT_MAX_TOKENS,
            output: bank.clone(),
        }),
    })
    .expect("export");
    run(TokenBankArgs {
        command: TokenBankCommand::Check(CheckArgs {
            bank: bank.clone(),
            container,
        }),
    })
    .expect("check");
}

#[test]
fn check_refuses_a_bank_for_another_container() {
    let tmp = tempfile::tempdir().unwrap();
    let container = tmp.path().join("model");
    write_container(&container);
    let prompts = tmp.path().join("prompts.json");
    write_prompts(&prompts);
    let bank = tmp.path().join("bank");
    run_export(&container, &prompts, DEFAULT_MAX_TOKENS, &bank).unwrap();

    let other = tmp.path().join("other");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join(TOKENIZER_FILE), b"{\"not\": \"this tokenizer\"}").unwrap();
    let err = run_check(&bank, &other).unwrap_err().to_string();
    assert!(err.contains("CorpusNotForThisModel"), "{err}");
}
