//! MEASURE-PLAN-2 W4: the token bank is re-established at ingestion.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

use super::super::super::token_bank::{export, TokenBank, MANIFEST_FILE};
use super::*;

fn bank() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let words = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran"];
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("[UNK]".into(), 0);
    for (i, w) in words.iter().enumerate() {
        vocab.insert((*w).into(), 1 + i as u32);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tk = tokenizers::Tokenizer::new(model);
    tk.with_pre_tokenizer(Some(Whitespace {}));
    let tokenizer = dir.path().join("tokenizer.json");
    tk.save(&tokenizer, false).unwrap();
    let prompts = dir.path().join("prompts.json");
    std::fs::write(
        &prompts,
        serde_json::to_vec(&serde_json::json!({
            "bank": "w4",
            "prompts": [
                {"id": "a", "category": "prose", "text": "the cat sat on a mat"},
                {"id": "b", "category": "prose", "text": "the dog ran"},
                {"id": "c", "category": "prose", "text": "a cat ran on the mat"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let out = dir.path().join("bank");
    export(&prompts, &tokenizer, 16, &out).unwrap();
    (dir, out)
}

fn declared(root: &Path, samples: &[&str], per_sample: u32) -> EvidenceBank {
    let manifest = std::fs::read(root.join(MANIFEST_FILE)).unwrap();
    EvidenceBank::new(
        TOKEN_BANK_SCHEMA,
        super::super::super::compile::hash_bytes(&manifest),
        samples.iter().copied(),
        per_sample,
    )
}

#[test]
fn positions_are_the_declared_samples_token_counts() {
    let (_dir, root) = bank();
    let tokens: Vec<usize> = TokenBank::open(&root)
        .unwrap()
        .manifest()
        .samples
        .iter()
        .map(|s| s.tokens)
        .collect();
    assert!(tokens.windows(2).any(|w| w[0] != w[1]), "lengths vary");
    let two = verify(&declared(&root, &["seq-000", "seq-001"], 0), &root).unwrap();
    assert_eq!(two, (tokens[0] + tokens[1]) as u64);
    let all = verify(
        &declared(&root, &["seq-000", "seq-001", "seq-002"], 0),
        &root,
    )
    .unwrap();
    assert_eq!(all, tokens.iter().sum::<usize>() as u64);
}

#[test]
fn a_tampered_payload_is_refused() {
    let (_dir, root) = bank();
    let payload = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "u32"))
        .unwrap();
    let mut bytes = std::fs::read(&payload).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&payload, bytes).unwrap();
    let all = declared(&root, &["seq-000", "seq-001", "seq-002"], 0);
    assert!(verify(&all, &root).is_err());
}

#[test]
fn a_bank_id_that_disagrees_with_its_manifest_is_refused() {
    let (_dir, root) = bank();
    let path = root.join(MANIFEST_FILE);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["bank_id"] = "0".repeat(64).into();
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let err = verify(&declared(&root, &["seq-000"], 0), &root)
        .unwrap_err()
        .to_string();
    assert!(err.contains("token bank"), "{err}");
}

#[test]
fn declarations_that_describe_another_bank_are_refused() {
    let (_dir, root) = bank();
    // A variable-length bank declares no per-sample length.
    assert!(verify(&declared(&root, &["seq-000"], 8), &root).is_err());
    // Samples are measured in bank order from seq-000.
    assert!(verify(&declared(&root, &["seq-001"], 0), &root).is_err());
    // More samples than the bank holds.
    let four = declared(&root, &["seq-000", "seq-001", "seq-002", "seq-003"], 0);
    assert!(verify(&four, &root).is_err());
    // Another schema.
    let mut kimi = declared(&root, &["seq-000"], 0);
    kimi.schema = "kimi-teacher-forced/v1".into();
    assert!(verify(&kimi, &root).is_err());
}
