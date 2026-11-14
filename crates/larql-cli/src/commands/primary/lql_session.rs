//! The LQL session `larql lql` and `larql repl` run in.
//!
//! LQL resolves paths itself; this installs the CLI's model cache as its
//! [`larql_lql::ModelResolver`], so `USE "Qwen3-0.6B"` names a model
//! exactly as `larql vindex3 …` does: an existing path wins, a name is
//! answered from the caches already on this machine, and nothing is ever
//! downloaded (see [`cache::resolve_installed_model`]).

use std::path::{Path, PathBuf};

use larql_lql::{RepresentationSource, Session};

use super::cache;

/// Names the representation source `larql lql` and `larql repl` bind VINDEX3 containers under, when the
/// `--representation-source` flag does not. The default is the session's own: `stored`.
pub const REPRESENTATION_SOURCE_ENV: &str = "LARQL_LQL_REPRESENTATION_SOURCE";

/// A representation source as `--representation-source` spells it, here and on `larql vindex3 exec`.
pub fn parse_representation_source(spec: &str) -> Result<RepresentationSource, String> {
    match spec.trim() {
        "stored" => Ok(RepresentationSource::Stored),
        "auto" => Ok(RepresentationSource::Auto),
        "transient" => Ok(RepresentationSource::Transient),
        other => Err(format!(
            "unknown representation source `{other}`; expected stored, auto or transient"
        )),
    }
}

/// A fresh session whose `USE` resolves cached model names, binding VINDEX3 containers under `source`, or
/// under [`REPRESENTATION_SOURCE_ENV`] when `source` is `None`, or else under the session's default. An
/// unrecognised word is an error rather than a silent fall back to a default: the word selects the
/// arithmetic a run performs.
pub fn session_with(source: Option<&str>) -> Result<Session, String> {
    let mut session = Session::new();
    session.set_model_resolver(Some(Box::new(resolve_use_literal)));
    let from_env = std::env::var(REPRESENTATION_SOURCE_ENV).ok();
    if let Some(spec) = source.or(from_env.as_deref()) {
        session.set_representation_source(parse_representation_source(spec)?);
    }
    Ok(session)
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
