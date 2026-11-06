use super::super::conversation::{ASSISTANT_ROLE, SYSTEM_ROLE, USER_ROLE};
use super::super::{render_conversation, render_user_prompt, ChatTurn, PromptShape};
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
