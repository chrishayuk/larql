//! MEASURE-PLAN-1 PR 1: the token bank. W5 (seals are read) and W6 (the
//! corpus belongs to the model) are the frozen witnesses; the rest pin the
//! format's other refusals and that the exporter tokenises as
//! `run_bank.py` does.

use std::collections::HashMap;
use std::path::Path;

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

use super::*;

/// The truncation cap the tests export with, below the longest fixture
/// prompt so truncation is exercised.
const CAP: usize = 6;

/// Words the fixture tokenizer knows, in id order after `[UNK]`.
const WORDS: [&str; 10] = [
    "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran", "far",
];

/// A word-level tokenizer over [`WORDS`]. `rotate` shifts every id, so two
/// calls with different values produce different tokenizers over the same
/// words: same text, different ids, different digest.
fn write_tokenizer(path: &Path, rotate: u32) {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("[UNK]".into(), 0);
    for (i, w) in WORDS.iter().enumerate() {
        vocab.insert((*w).into(), 1 + (i as u32 + rotate) % WORDS.len() as u32);
    }
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .expect("word-level model");
    let mut tk = Tokenizer::new(model);
    tk.with_pre_tokenizer(Some(Whitespace {}));
    tk.save(path, false).expect("save tokenizer");
}

/// Four prompts: two ordinary, one longer than [`CAP`], one too short to
/// score, so it is dropped as `run_bank.py` drops it.
fn write_prompts(path: &Path) {
    let prompts = serde_json::json!({
        "bank": "fixture-bank",
        "frozen": "2026-09-23",
        "prompts": [
            {"id": "prose-000", "category": "prose", "text": "the cat sat on a mat"},
            {"id": "prose-001", "category": "prose", "text": "a dog ran far"},
            {"id": "longform-000", "category": "longform",
             "text": "the cat sat on a mat and the dog ran far"},
            {"id": "short-000", "category": "factual", "text": "cat"}
        ]
    });
    std::fs::write(path, serde_json::to_vec_pretty(&prompts).unwrap()).unwrap();
}

struct Fixture {
    _tmp: tempfile::TempDir,
    prompts: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
    bank: std::path::PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let prompts = tmp.path().join("prompts.json");
    let tokenizer = tmp.path().join(TOKENIZER_FILE);
    write_prompts(&prompts);
    write_tokenizer(&tokenizer, 0);
    let bank = tmp.path().join("bank");
    export(&prompts, &tokenizer, CAP, &bank).expect("export");
    Fixture {
        prompts,
        tokenizer,
        bank,
        _tmp: tmp,
    }
}

fn encode(tokenizer: &Path, text: &str) -> Vec<u32> {
    Tokenizer::from_file(tokenizer)
        .unwrap()
        .encode(text, true)
        .unwrap()
        .get_ids()
        .to_vec()
}

#[test]
fn an_exported_bank_reopens_with_the_ids_the_tokenizer_produces() {
    let f = fixture();
    let bank = TokenBank::open(&f.bank).expect("open");
    let m = bank.manifest();
    assert_eq!(m.schema, TOKEN_BANK_SCHEMA);
    assert_eq!(m.payload_authority, TOKEN_BANK_PAYLOAD_AUTHORITY);
    assert_eq!(m.prompts.bank, "fixture-bank");
    assert_eq!(m.template, TemplatePolicy::Raw);
    assert!(m.add_special_tokens);
    assert_eq!(m.max_tokens, CAP);
    // The one-word prompt is dropped; the others keep their order.
    let ids: Vec<&str> = m.samples.iter().map(|s| s.prompt_id.as_str()).collect();
    assert_eq!(ids, ["prose-000", "prose-001", "longform-000"]);
    assert_eq!(bank.sample_count(), 3);
    for (i, text) in [
        "the cat sat on a mat",
        "a dog ran far",
        "the cat sat on a mat and the dog ran far",
    ]
    .iter()
    .enumerate()
    {
        let mut want = encode(&f.tokenizer, text);
        want.truncate(CAP);
        assert_eq!(bank.read(i).unwrap(), want, "sample {i}");
        assert_eq!(m.samples[i].tokens, want.len());
        assert_eq!(m.samples[i].id, format!("seq-{i:03}"));
    }
}

#[test]
fn the_bank_id_is_a_function_of_the_contents() {
    let f = fixture();
    let again = f.bank.with_file_name("bank-again");
    let second = export(&f.prompts, &f.tokenizer, CAP, &again).unwrap();
    let first = TokenBank::open(&f.bank).unwrap();
    assert_eq!(first.manifest().bank_id, second.bank_id);
    assert!(!second.bank_id.is_empty());
    // A different cap is a different bank.
    let capped = f.bank.with_file_name("bank-capped");
    let third = export(&f.prompts, &f.tokenizer, CAP + 1, &capped).unwrap();
    assert_ne!(third.bank_id, second.bank_id);
}

/// W5: a tampered payload is refused, naming the sample.
#[test]
fn a_tampered_payload_is_refused_as_a_seal_mismatch() {
    let f = fixture();
    let payload = f.bank.join("seq-001.u32");
    let mut bytes = std::fs::read(&payload).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&payload, bytes).unwrap();
    let bank = TokenBank::open(&f.bank).unwrap();
    assert!(bank.read(0).is_ok(), "an untouched sample still reads");
    match bank.read(1) {
        Err(TokenBankError::SealMismatch { sample, .. }) => assert_eq!(sample, "seq-001"),
        other => panic!("expected SealMismatch, got {other:?}"),
    }
}

/// W6: a bank tokenised by another tokenizer does not belong to this model.
#[test]
fn a_bank_from_another_tokenizer_is_not_for_this_model() {
    let f = fixture();
    let bank = TokenBank::open(&f.bank).unwrap();
    let own = container_tokenizer_sha256(f.tokenizer.parent().unwrap()).unwrap();
    assert!(bank.check_tokenizer(&own).is_ok());

    let other_dir = f.bank.with_file_name("other-model");
    std::fs::create_dir_all(&other_dir).unwrap();
    write_tokenizer(&other_dir.join(TOKENIZER_FILE), 3);
    let other = container_tokenizer_sha256(&other_dir).unwrap();
    assert_ne!(other, own);
    match bank.check_tokenizer(&other) {
        Err(TokenBankError::CorpusNotForThisModel { bank: b, model }) => {
            assert_eq!(b, own);
            assert_eq!(model, other);
        }
        other => panic!("expected CorpusNotForThisModel, got {other:?}"),
    }
}

fn rewrite_manifest(bank: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let path = bank.join(MANIFEST_FILE);
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(&mut v);
    std::fs::write(&path, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
}

#[test]
fn an_edited_manifest_no_longer_names_itself() {
    let f = fixture();
    rewrite_manifest(&f.bank, |v| v["max_tokens"] = serde_json::json!(CAP + 1));
    assert!(matches!(
        TokenBank::open(&f.bank),
        Err(TokenBankError::BankIdMismatch { .. })
    ));
}

#[test]
fn another_schema_or_authority_is_refused() {
    let f = fixture();
    rewrite_manifest(&f.bank, |v| {
        v["schema"] = serde_json::json!("kimi-teacher-forced/v1")
    });
    assert!(matches!(
        TokenBank::open(&f.bank),
        Err(TokenBankError::UnknownSchema { .. })
    ));
    let g = fixture();
    rewrite_manifest(&g.bank, |v| {
        v["payload_authority"] = serde_json::json!("other")
    });
    assert!(matches!(
        TokenBank::open(&g.bank),
        Err(TokenBankError::UnknownPayloadAuthority { .. })
    ));
}

#[test]
fn samples_out_of_order_are_refused() {
    let f = fixture();
    rewrite_manifest(&f.bank, |v| {
        v["samples"].as_array_mut().unwrap().swap(0, 1);
    });
    assert!(matches!(
        TokenBank::open(&f.bank),
        Err(TokenBankError::SampleOrder { index: 0, .. })
    ));
}

#[test]
fn an_export_never_writes_over_an_existing_directory() {
    let f = fixture();
    assert!(matches!(
        export(&f.prompts, &f.tokenizer, CAP, &f.bank),
        Err(TokenBankError::OutputExists { .. })
    ));
}
