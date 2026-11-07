//! How `USE "<name>"` finds a model that is not a path.
//!
//! LQL itself knows paths and published-vindex `hf://` URIs. An embedder
//! that keeps a model cache (the `larql` CLI) installs a [`ModelResolver`]
//! on the [`Session`](super::Session) so `USE "Qwen3-0.6B"` names a model
//! the same way its own commands do. The crate never depends on that
//! cache; it only calls the hook.
//!
//! Order, in `USE`: an `hf://` URI keeps its own path; otherwise a literal
//! that exists on disk is used as written, and only a literal that does
//! not exist is offered to the resolver. With no resolver installed,
//! behaviour is exactly what it was before the hook existed.

use std::path::{Path, PathBuf};

/// Maps a `USE` literal that is not an existing path to a model directory.
///
/// `Ok(Some(dir))` resolves it; `Ok(None)` declines, leaving the literal
/// to fail as a missing path; `Err(message)` refuses it, and `message` is
/// the error `USE` reports (an embedder's guidance on fetching the model).
pub type ModelResolver = Box<dyn Fn(&str) -> Result<Option<PathBuf>, String> + Send + Sync>;

/// The directory `USE "<literal>"` binds: the literal when it exists, else
/// what `resolver` makes of it, else the literal unchanged.
pub(crate) fn resolve_use_literal(
    literal: &str,
    resolver: Option<&ModelResolver>,
) -> Result<PathBuf, String> {
    let path = PathBuf::from(literal);
    if Path::new(literal).exists() {
        return Ok(path);
    }
    match resolver {
        None => Ok(path),
        Some(resolve) => Ok(resolve(literal)?.unwrap_or(path)),
    }
}
