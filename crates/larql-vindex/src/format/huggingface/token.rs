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

use std::path::PathBuf;

/// Token variables, in the order `hf-hub` consults them.
const TOKEN_ENVS: [&str; 2] = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

/// Token files under `$HOME`, oldest spelling first — the hub's tooling
/// wrote `~/.huggingface/token` before `huggingface-cli login` moved to
/// `~/.cache/huggingface/token`, and machines carry both.
const TOKEN_FILES: [&[&str]; 2] = [
    &[".huggingface", "token"],
    &[".cache", "huggingface", "token"],
];

/// The token this process should present, or `None` when it holds none.
///
/// An empty value from any source is *no token*, not an empty bearer:
/// `HF_TOKEN=` in a shell profile would otherwise send `Bearer ` and
/// turn a public read into a 401.
pub(in crate::format::huggingface) fn resolve() -> Option<String> {
    if let Some(token) = TOKEN_ENVS
        .iter()
        .find_map(|key| std::env::var(key).ok())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
    {
        return Some(token);
    }
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
    TOKEN_FILES
        .iter()
        .map(|parts| parts.iter().fold(home.clone(), |p, part| p.join(part)))
        .find_map(|path| std::fs::read_to_string(path).ok())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}
