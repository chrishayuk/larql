//! Whether a Hugging Face token is configured — reported by source,
//! never by value.

use std::path::Path;

use larql_vindex::format::huggingface::TokenSource;

use super::report::{Check, Group};
use super::units::display_path;

/// Shown when no token is configured.
const NO_TOKEN_HINT: &str = "only gated models (Llama, Gemma) need one: set HF_TOKEN or run \
     `hf auth login`; the quick start's model does not";

/// A token source as the report shows it: a file under `home` as `~/…`.
pub(crate) fn describe_source(source: &TokenSource, home: Option<&Path>) -> String {
    match source {
        TokenSource::File(path) => display_path(path, home),
        TokenSource::Env(_) => source.to_string(),
    }
}

/// `source`: where the token would be read from (`$HF_TOKEN`, a token
/// file), as `larql_vindex`'s own token search reports it.
pub(crate) fn checks(source: Option<&str>) -> Vec<Check> {
    const NAME: &str = "Token";
    vec![match source {
        Some(source) => Check::ok(
            Group::HuggingFace,
            NAME,
            format!("configured (from {source}); gated models can be fetched"),
        ),
        None => Check::warn(Group::HuggingFace, NAME, "not configured", NO_TOKEN_HINT),
    }]
}
