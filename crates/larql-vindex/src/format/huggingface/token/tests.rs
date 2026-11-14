//! Tests for the token search (`super`).

use super::*;

fn no_env(_: &str) -> Option<String> {
    None
}

fn write_token(home: &Path, parts: &[&str], value: &str) -> PathBuf {
    let path = parts
        .iter()
        .fold(home.to_path_buf(), |p, part| p.join(part));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, value).unwrap();
    path
}

#[test]
fn nothing_configured_finds_nothing() {
    let home = tempfile::tempdir().unwrap();
    assert_eq!(find(no_env, home.path()), None);
}

#[test]
fn the_first_variable_wins_and_is_trimmed() {
    let home = tempfile::tempdir().unwrap();
    let env = |key: &str| match key {
        "HF_TOKEN" => Some(" hf_a \n".to_string()),
        "HUGGING_FACE_HUB_TOKEN" => Some("hf_b".to_string()),
        _ => None,
    };
    assert_eq!(
        find(env, home.path()),
        Some((TokenSource::Env("HF_TOKEN"), "hf_a".to_string()))
    );
}

#[test]
fn the_legacy_variable_is_read_when_the_first_is_unset() {
    let home = tempfile::tempdir().unwrap();
    let env = |key: &str| (key == "HUGGING_FACE_HUB_TOKEN").then(|| "hf_b".to_string());
    assert_eq!(
        find(env, home.path()).map(|(s, _)| s),
        Some(TokenSource::Env("HUGGING_FACE_HUB_TOKEN"))
    );
}

#[test]
fn an_empty_variable_falls_through_to_the_files() {
    let home = tempfile::tempdir().unwrap();
    let path = write_token(
        home.path(),
        &[".cache", "huggingface", "token"],
        "hf_file\n",
    );
    let env = |key: &str| (key == "HF_TOKEN").then(String::new);
    assert_eq!(
        find(env, home.path()),
        Some((TokenSource::File(path), "hf_file".to_string()))
    );
}

#[test]
fn the_older_file_spelling_is_read_first() {
    let home = tempfile::tempdir().unwrap();
    let old = write_token(home.path(), &[".huggingface", "token"], "hf_old");
    write_token(home.path(), &[".cache", "huggingface", "token"], "hf_new");
    assert_eq!(
        find(no_env, home.path()).map(|(s, _)| s),
        Some(TokenSource::File(old))
    );
}

#[test]
fn a_blank_token_file_is_no_token() {
    let home = tempfile::tempdir().unwrap();
    write_token(home.path(), &[".huggingface", "token"], "  \n");
    assert_eq!(find(no_env, home.path()), None);
}

#[test]
fn a_source_displays_without_the_token() {
    assert_eq!(TokenSource::Env("HF_TOKEN").to_string(), "$HF_TOKEN");
    let path = PathBuf::from("a").join("token");
    assert_eq!(
        TokenSource::File(path.clone()).to_string(),
        path.display().to_string()
    );
}

/// The live entry point runs the same search. Its answer depends on this
/// machine and on other tests setting `HF_TOKEN` concurrently, so only
/// that it runs is asserted.
#[test]
fn the_live_search_runs() {
    let _ = token_source();
}
