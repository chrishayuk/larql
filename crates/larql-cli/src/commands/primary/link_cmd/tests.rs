//! Tests for `larql link`'s name derivation and validation.

use std::path::Path;

use super::{default_name, validate_name};

#[test]
fn default_name_strips_either_generations_dir_suffix() {
    assert_eq!(
        default_name(Path::new("/x/output/gemma3-4b-f16.vindex")).unwrap(),
        "gemma3-4b-f16"
    );
    assert_eq!(
        default_name(Path::new("/x/qwen3-0.6b.vindex3")).unwrap(),
        "qwen3-0.6b"
    );
    assert_eq!(
        default_name(Path::new("/x/plain-dir")).unwrap(),
        "plain-dir"
    );
}

#[test]
fn default_name_refuses_a_path_with_no_basename() {
    let err = default_name(Path::new("/")).unwrap_err();
    assert!(err.to_string().contains("cannot derive name"), "{err}");
}

#[test]
fn validate_name_accepts_a_plain_name_and_refuses_the_rest() {
    validate_name("qwen3-0.6b").unwrap();
    assert!(validate_name("").unwrap_err().to_string().contains("empty"));
    assert!(validate_name("owner/name")
        .unwrap_err()
        .to_string()
        .contains("path separator"));
    assert!(validate_name(".hidden")
        .unwrap_err()
        .to_string()
        .contains("cannot start with `.`"));
}
