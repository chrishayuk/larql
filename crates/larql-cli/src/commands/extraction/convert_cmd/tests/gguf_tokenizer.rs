//! A GGUF conversion's sibling tokenizer: missing, unparseable and valid
//! must each be told apart (#365).

use super::super::gguf_tokenizer::load_sibling_tokenizer;
use larql_vindex::format::filenames::TOKENIZER_JSON;

fn gguf_in(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join("model.gguf")
}

#[test]
fn missing_tokenizer_says_not_found_and_names_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let err = load_sibling_tokenizer(&gguf_in(dir.path())).unwrap_err();
    assert!(err.contains("not found"), "{err}");
    assert!(
        err.contains(&dir.path().join(TOKENIZER_JSON).display().to_string()),
        "{err}"
    );
}

#[test]
fn unparseable_tokenizer_is_a_parse_error_not_not_found() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(TOKENIZER_JSON), b"{\"model\": 42}").unwrap();
    let err = load_sibling_tokenizer(&gguf_in(dir.path())).unwrap_err();
    assert!(err.starts_with("failed to parse"), "{err}");
    assert!(!err.contains("not found"), "{err}");
}

#[test]
fn a_directory_named_tokenizer_json_is_not_a_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join(TOKENIZER_JSON)).unwrap();
    let err = load_sibling_tokenizer(&gguf_in(dir.path())).unwrap_err();
    assert!(err.contains("not found"), "{err}");
}

#[test]
fn valid_tokenizer_loads() {
    use larql_vindex::tokenizers::models::wordlevel::WordLevel;
    use larql_vindex::tokenizers::Tokenizer;
    let dir = tempfile::tempdir().unwrap();
    let vocab = [("[UNK]".to_string(), 0u32), ("paris".to_string(), 1)];
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    Tokenizer::new(model)
        .save(dir.path().join(TOKENIZER_JSON), false)
        .unwrap();
    let tok = load_sibling_tokenizer(&gguf_in(dir.path())).expect("valid tokenizer");
    assert_eq!(tok.token_to_id("paris"), Some(1));
}
