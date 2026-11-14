//! The LQL session `larql lql` and `larql repl` run in.
//!
//! LQL resolves paths itself; this installs the CLI's model cache as its
//! [`larql_lql::ModelResolver`], so `USE "Qwen3-0.6B"` names a model
//! exactly as `larql vindex3 …` does: an existing path wins, a name is
//! answered from the caches already on this machine, and nothing is ever
//! downloaded (see [`cache::resolve_installed_model`]).

use std::path::{Path, PathBuf};

use larql_lql::Session;

use super::cache;

/// A fresh session whose `USE` resolves cached model names.
pub fn session() -> Session {
    let mut session = Session::new();
    session.set_model_resolver(Some(Box::new(resolve_use_literal)));
    session
}

/// The hook: the installed-model resolver, its miss message carried
/// verbatim so `USE` gives the same guidance as every other command.
pub fn resolve_use_literal(literal: &str) -> Result<Option<PathBuf>, String> {
    cache::resolve_installed_model(Path::new(literal))
        .map(Some)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;
