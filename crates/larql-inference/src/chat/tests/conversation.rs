use super::super::conversation::{ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE};
use super::super::{
    container_has_chat_template, render_conversation, render_user_prompt, ChatTurn, PromptShape,
};
use super::*;

/// A family with no registry default system prompt and no fallback.
const PLAIN_FAMILY: &str = "tinymodel";
/// A family whose registry row declares a default system prompt and a
/// fallback template.
const SHAPED_FAMILY: &str = "gemma4";

#[test]
fn turn_constructors_carry_the_template_roles() {
    assert_eq!(ChatTurn::user("u").role, USER_ROLE);
    assert_eq!(ChatTurn::assistant("a").role, ASSISTANT_ROLE);
    assert_eq!(ChatTurn::assistant("a").content, "a");
}

#[test]
fn one_turn_matches_the_single_prompt_path() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), SHOW_ALL);
    write_tokenizer_config(tmp.path(), r#"{"bos_token":"<s>"}"#);
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("hi")]).unwrap();
    assert!(r.templated());
    assert_eq!(r.shape, PromptShape::Templated);
    assert_eq!(
        r.prompt,
        render_user_prompt(tmp.path(), PLAIN_FAMILY, "hi").unwrap()
    );
    // enable_thinking stays undefined; bos comes from tokenizer_config.
    assert_eq!(r.prompt, "<s><user>hi<assistant>");
}

#[test]
fn history_renders_every_turn_in_order() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), SHOW_ALL);
    let turns = [
        ChatTurn::user("My name is Chris."),
        ChatTurn::assistant("Hello Chris."),
        ChatTurn::user("What is my name?"),
    ];
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &turns).unwrap();
    assert_eq!(
        r.prompt,
        "<user>My name is Chris.<assistant>Hello Chris.<user>What is my name?<assistant>"
    );
}

#[test]
fn tokenizer_config_template_is_read_in_either_form() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    write_tokenizer_config(
        tmp.path(),
        r#"{"chat_template":[{"name":"default","template":"D:{{ messages[0].content }}"}]}"#,
    );
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("x")]).unwrap();
    assert_eq!(r.prompt, "D:x");
}

#[test]
fn no_template_is_the_latest_turn_verbatim_and_says_so() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    let turns = [
        ChatTurn::user("a"),
        ChatTurn::assistant("b"),
        ChatTurn::user("c"),
    ];
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &turns).unwrap();
    assert_eq!(r.shape, PromptShape::NoTemplate);
    assert!(!r.templated());
    assert_eq!(r.prompt, "c");
}

#[test]
fn raw_env_sends_the_latest_turn_even_with_a_template() {
    let _g = env(&[(ENV_RAW_PROMPT, "1")]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), SHOW_ALL);
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("raw")]).unwrap();
    assert_eq!(r.shape, PromptShape::RawRequested);
    assert_eq!(r.prompt, "raw");
}

#[test]
fn a_system_message_comes_first_and_states_thinking() {
    let _g = env(&[(ENV_SYSTEM, "S")]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), SHOW_ALL);
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("u")]).unwrap();
    assert_eq!(
        r.prompt,
        format!("<{SYSTEM_ROLE}>S<user>u[think=false]<assistant>")
    );
}

#[test]
fn thinking_env_is_stated_true() {
    let _g = env(&[(ENV_THINKING, "1")]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), SHOW_ALL);
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("u")]).unwrap();
    assert!(
        r.prompt.ends_with("[think=true]<assistant>"),
        "{}",
        r.prompt
    );
}

#[test]
fn the_registry_fallback_serves_a_shaped_family_without_a_template() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    let r = render_conversation(tmp.path(), SHAPED_FAMILY, &[ChatTurn::user("u")]).unwrap();
    assert!(r.templated());
    assert!(r.prompt.contains('u'));
}

#[test]
fn a_conversation_must_end_with_a_user_turn() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    let empty = render_conversation(tmp.path(), PLAIN_FAMILY, &[]).unwrap_err();
    assert!(empty.contains("at least one turn"), "{empty}");
    let ends_assistant =
        render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::assistant("a")]).unwrap_err();
    assert!(ends_assistant.contains(ASSISTANT_ROLE), "{ends_assistant}");
}

#[test]
fn a_broken_template_is_an_error_not_a_raw_fallback() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), "{% for");
    let err = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("u")]).unwrap_err();
    assert!(err.contains("render chat template"), "{err}");
}

#[test]
fn a_conversations_own_system_turn_replaces_the_environments() {
    let _g = env(&[(ENV_SYSTEM, "ENV")]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), SHOW_ALL);
    let turns = [
        ChatTurn {
            role: SYSTEM_ROLE.to_string(),
            content: "OWN".to_string(),
        },
        ChatTurn::user("u"),
    ];
    let r = render_conversation(tmp.path(), SHAPED_FAMILY, &turns).unwrap();
    // No injected system, so nothing is shaped: enable_thinking stays
    // undefined, exactly as for a conversation the environment left alone.
    assert_eq!(r.prompt, format!("<{SYSTEM_ROLE}>OWN<user>u<assistant>"));
}

#[test]
fn container_template_presence_is_read_from_either_source() {
    let tmp = tempfile::tempdir().unwrap();
    assert!(!container_has_chat_template(tmp.path()));
    write_tokenizer_config(tmp.path(), r#"{"chat_template":"T"}"#);
    assert!(container_has_chat_template(tmp.path()));
    let jinja = tempfile::tempdir().unwrap();
    write_jinja(jinja.path(), SHOW_ALL);
    assert!(container_has_chat_template(jinja.path()));
}

/// A word-level tokenizer whose post-processor prepends `[BOS]`, so
/// whether encoding added special tokens is visible in the ids.
fn bos_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [{"id": BOS_ID, "content": "[BOS]", "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[BOS]", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
            "special_tokens": {"[BOS]": {"id": "[BOS]", "ids": [BOS_ID], "tokens": ["[BOS]"]}}
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[BOS]": BOS_ID, "u": WORD_ID, "<unk>": UNK_ID},
                  "unk_token": "<unk>"}
    });
    larql_vindex::tokenizers::Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap()
}

const BOS_ID: u32 = 0;
const WORD_ID: u32 = 1;
const UNK_ID: u32 = 2;

#[test]
fn a_templated_prompt_encodes_without_the_post_processor() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    write_jinja(tmp.path(), "{{ messages[0].content }}");
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("u")]).unwrap();
    assert!(!r.adds_special_tokens());
    assert_eq!(r.encode(&bos_tokenizer()).unwrap(), vec![WORD_ID]);
}

#[test]
fn a_raw_prompt_encodes_with_the_post_processor() {
    let _g = env(&[]);
    let tmp = tempfile::tempdir().unwrap();
    let r = render_conversation(tmp.path(), PLAIN_FAMILY, &[ChatTurn::user("u")]).unwrap();
    assert!(r.adds_special_tokens());
    assert_eq!(r.encode(&bos_tokenizer()).unwrap(), vec![BOS_ID, WORD_ID]);
}
