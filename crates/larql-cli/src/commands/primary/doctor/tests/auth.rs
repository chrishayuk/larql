use larql_vindex::format::huggingface::TokenSource;

use super::super::auth::{checks, describe_source};
use super::super::report::Status;
use super::{home, only};

#[test]
fn a_configured_token_is_reported_by_source() {
    let c = only(checks(Some("$HF_TOKEN")));
    assert_eq!(c.status, Status::Ok);
    assert!(c.summary.contains("$HF_TOKEN"));
}

#[test]
fn no_token_warns_about_gated_models() {
    let c = only(checks(None));
    assert_eq!(c.status, Status::Warn);
    let hint = c.hint.unwrap();
    assert!(hint.contains("Llama") && hint.contains("Gemma") && hint.contains("HF_TOKEN"));
}

#[test]
fn a_token_file_under_home_is_shown_from_tilde() {
    let file = TokenSource::File(home().join(".cache/huggingface/token"));
    assert!(describe_source(&file, Some(&home())).starts_with('~'));
    let env = TokenSource::Env("HF_TOKEN");
    assert_eq!(describe_source(&env, Some(&home())), "$HF_TOKEN");
}
