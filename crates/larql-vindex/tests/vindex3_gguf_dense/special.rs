//! The SpecialVocab port: post-processor templates, named tokens, flags
//! and the chat template, in the reference's discovery order.

use larql_models::loading::gguf::GgufValue;
use larql_vindex::format::vindex3::gguf::dense::special::special_entries;
use serde_json::json;

const N_VOCAB: usize = 16;

fn added() -> serde_json::Value {
    json!([
        {"id": 0, "content": "<pad>"}, {"id": 1, "content": "<eos>"}, {"id": 2, "content": "<bos>"},
        {"id": 3, "content": "<unk>"}, {"id": 4, "content": "<sep>"}, {"id": 5, "content": "<cls>"},
        {"id": 99, "content": "<mask>"}
    ])
}

fn tokenizer(post: serde_json::Value) -> serde_json::Value {
    json!({"added_tokens": added(), "post_processor": post})
}

fn template(single: serde_json::Value, pair: serde_json::Value) -> serde_json::Value {
    json!({"type": "TemplateProcessing", "single": single, "pair": pair})
}

fn sp(id: &str) -> serde_json::Value {
    json!({"SpecialToken": {"id": id, "type_id": 0}})
}

fn seq(id: &str) -> serde_json::Value {
    json!({"Sequence": {"id": id, "type_id": 0}})
}

fn keys(entries: &[(String, GgufValue)]) -> Vec<String> {
    entries.iter().map(|(k, _)| k.clone()).collect()
}

fn get<'a>(entries: &'a [(String, GgufValue)], key: &str) -> Option<&'a GgufValue> {
    entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

#[test]
fn the_gemma_shape_orders_ids_then_flags_then_template() {
    let dir = tempfile::tempdir().unwrap();
    let tok = tokenizer(template(
        json!([sp("<bos>"), seq("A")]),
        json!([sp("<bos>"), seq("A"), sp("<bos>"), seq("B")]),
    ));
    let config = json!({"add_bos_token": true, "add_eos_token": false, "bos_token": "<bos>",
        "eos_token": {"content": "<eos>"}, "pad_token": "<pad>", "unk_token": "<unk>",
        "mask_token": "<mask>", "chat_template": "T"});
    let e = special_entries(dir.path(), &tok, Some(&config), N_VOCAB).unwrap();
    assert_eq!(
        keys(&e),
        [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_sep_token",
            "tokenizer.ggml.add_eos_token",
            "tokenizer.chat_template",
        ],
        "mask (id 99) is outside the vocabulary and skipped"
    );
    assert_eq!(
        get(&e, "tokenizer.ggml.add_sep_token"),
        Some(&GgufValue::Bool(false))
    );
}

#[test]
fn a_separator_in_the_pair_template_sets_add_sep_and_names_sep() {
    let dir = tempfile::tempdir().unwrap();
    let tok = tokenizer(template(
        json!([sp("<cls>"), seq("A")]),
        json!([sp("<cls>"), seq("A"), sp("<eos>"), sp("<eos>"), seq("B")]),
    ));
    let config = json!({"cls_token": "<cls>", "eos_token": "<eos>"});
    let e = special_entries(dir.path(), &tok, Some(&config), N_VOCAB).unwrap();
    assert_eq!(
        get(&e, "tokenizer.ggml.add_bos_token"),
        Some(&GgufValue::Bool(true))
    );
    assert_eq!(
        get(&e, "tokenizer.ggml.add_sep_token"),
        Some(&GgufValue::Bool(true))
    );
    assert_eq!(
        get(&e, "tokenizer.ggml.seperator_token_id"),
        Some(&GgufValue::U32(1))
    );
    assert_eq!(
        get(&e, "tokenizer.ggml.cls_token_id"),
        Some(&GgufValue::U32(5))
    );
}

#[test]
fn a_trailing_eos_sets_add_eos_and_a_foreign_one_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let ends = |last: &str| {
        tokenizer(json!({"type": "Sequence", "processors": [
        {"type": "ByteLevel"}, template(json!([sp("<bos>"), seq("A"), sp(last)]), json!([]))]}))
    };
    let config = json!({"bos_token": "<bos>", "eos_token": "<eos>"});
    let e = special_entries(dir.path(), &ends("<eos>"), Some(&config), N_VOCAB).unwrap();
    assert_eq!(
        get(&e, "tokenizer.ggml.add_eos_token"),
        Some(&GgufValue::Bool(true))
    );
    let err = special_entries(dir.path(), &ends("<sep>"), Some(&config), N_VOCAB).unwrap_err();
    assert!(err.to_string().contains("eot/eom"), "{err}");
}

#[test]
fn roberta_adds_all_three_and_names_cls_and_sep() {
    let dir = tempfile::tempdir().unwrap();
    let tok =
        tokenizer(json!({"type": "RobertaProcessing", "cls": ["<cls>", 5], "sep": ["<sep>", 4]}));
    let config = json!({"bos_token": "<bos>"});
    let e = special_entries(dir.path(), &tok, Some(&config), N_VOCAB).unwrap();
    for flag in ["bos", "eos", "sep"] {
        assert_eq!(
            get(&e, &format!("tokenizer.ggml.add_{flag}_token")),
            Some(&GgufValue::Bool(true))
        );
    }
    assert_eq!(
        get(&e, "tokenizer.ggml.cls_token_id"),
        Some(&GgufValue::U32(5))
    );
    assert_eq!(
        get(&e, "tokenizer.ggml.seperator_token_id"),
        Some(&GgufValue::U32(4))
    );
}

#[test]
fn without_a_config_only_the_template_flags_survive() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("chat_template.jinja"), "J").unwrap();
    let tok = tokenizer(template(
        json!([sp("<bos>"), seq("A"), sp("<eos>")]),
        json!([]),
    ));
    let e = special_entries(dir.path(), &tok, None, N_VOCAB).unwrap();
    assert_eq!(
        keys(&e),
        [
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_eos_token"
        ]
    );
    assert_eq!(
        get(&e, "tokenizer.ggml.add_bos_token"),
        Some(&GgufValue::Bool(true))
    );
}

#[test]
fn the_chat_template_falls_back_to_the_jinja_file() {
    let dir = tempfile::tempdir().unwrap();
    let tok = json!({"added_tokens": added()});
    let config = json!({"bos_token": "<bos>"});
    assert!(get(
        &special_entries(dir.path(), &tok, Some(&config), N_VOCAB).unwrap(),
        "tokenizer.chat_template"
    )
    .is_none());
    std::fs::write(dir.path().join("chat_template.jinja"), "J").unwrap();
    let e = special_entries(dir.path(), &tok, Some(&config), N_VOCAB).unwrap();
    assert_eq!(
        get(&e, "tokenizer.chat_template"),
        Some(&GgufValue::String("J".into()))
    );
    // A non-string template (a named list) is not written.
    let listed = json!({"bos_token": "<bos>", "chat_template": [{"name": "default"}]});
    assert!(get(
        &special_entries(dir.path(), &tok, Some(&listed), N_VOCAB).unwrap(),
        "tokenizer.chat_template"
    )
    .is_none());
}
