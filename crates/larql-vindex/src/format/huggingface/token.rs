//! Where a Hugging Face token comes from — in one place.
//!
//! Every HTTP path in this module tree authenticates the same way, so
//! the sources are named once here rather than per call site. That is
//! not tidiness: `range` previously read only the environment while
//! `publish`, `download` and `discovery` read the environment *and* the
//! files `huggingface-cli login` writes. The result was that
//! `vindex plan hf://meta-llama/...` answered `HTTP 401 Unauthorized` on
//! a machine that was logged in — a refusal about the repo, caused by
//! the client, on facts the process could see.

use std::path::{Path, PathBuf};

/// Token variables, in the order `hf-hub` consults them.
const TOKEN_ENVS: [&str; 2] = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

/// Token files under `$HOME`, oldest spelling first — the hub's tooling
/// wrote `~/.huggingface/token` before `huggingface-cli login` moved to
/// `~/.cache/huggingface/token`, and machines carry both.
const TOKEN_FILES: [&[&str]; 2] = [
    &[".huggingface", "token"],
    &[".cache", "huggingface", "token"],
];

/// Where the token this process would present comes from — never the
/// token itself, so a caller can report it (`larql doctor`) without
/// holding the secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenSource {
    /// An environment variable, by name.
    Env(&'static str),
    /// A token file, by path.
    File(PathBuf),
}

impl std::fmt::Display for TokenSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenSource::Env(name) => write!(f, "${name}"),
            TokenSource::File(path) => write!(f, "{}", path.display()),
        }
    }
}

/// The token this process should present, or `None` when it holds none.
///
/// An empty value from any source is *no token*, not an empty bearer:
/// `HF_TOKEN=` in a shell profile would otherwise send `Bearer ` and
/// turn a public read into a 401.
pub(in crate::format::huggingface) fn resolve() -> Option<String> {
    find(env_var, &home_dir()).map(|(_, token)| token)
}

/// Which source [`resolve`] would read its token from, or `None` when
/// none holds one. The same search, so the report cannot drift from the
/// credential actually sent.
pub fn token_source() -> Option<TokenSource> {
    find(env_var, &home_dir()).map(|(source, _)| source)
}

fn env_var(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
}

/// Testable core: the token, with its source, reading variables through
/// `env` and token files under `home`. The first variable that is *set*
/// decides the environment's answer and the first file that is
/// *readable* decides the files' answer; an empty value from either is
/// no token, not a reason to read the next one.
fn find(env: impl Fn(&str) -> Option<String>, home: &Path) -> Option<(TokenSource, String)> {
    if let Some(found) = TOKEN_ENVS
        .iter()
        .find_map(|key| env(key).map(|t| (TokenSource::Env(key), t)))
        .and_then(non_empty)
    {
        return Some(found);
    }
    TOKEN_FILES
        .iter()
        .map(|parts| {
            parts
                .iter()
                .fold(home.to_path_buf(), |p, part| p.join(part))
        })
        .find_map(|path| {
            std::fs::read_to_string(&path)
                .ok()
                .map(|t| (TokenSource::File(path), t))
        })
        .and_then(non_empty)
}

/// A trimmed token, or `None` when only whitespace was found.
fn non_empty((source, token): (TokenSource, String)) -> Option<(TokenSource, String)> {
    let token = token.trim().to_string();
    (!token.is_empty()).then_some((source, token))
}

#[cfg(test)]
mod tests;
