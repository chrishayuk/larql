//! Stop tokens as ids, for decode loops that compare ids only.
//!
//! [`EosConfig::is_eos`] stops on a declared EOS id *or* a stop string
//! matched by surface form. A loop that never decodes — `vindex3 exec
//! --generate --stop-at-eos`, which works in token ids end to end — can
//! only honour ids, so a container whose stops are all strings (no
//! `generation_config.json`, e.g. a VINDEX3 import of Qwen3.5) used to
//! stop at nothing while the flag claimed otherwise (#752).
//!
//! [`resolve_stop_ids`] maps every stop the container declares to the id
//! its tokenizer gives it: the `generation_config.json` EOS ids, every
//! stop string [`EosConfig`] carries (built-in terminators plus declared
//! `stop_strings`), and `tokenizer_config.json`'s `eos_token`. Only a
//! string that is exactly one token of the tokenizer becomes an id; a
//! string that would encode to several tokens is never reinterpreted as
//! one of its pieces. A *declared* string with no single token (a
//! `stop_strings` entry or the `eos_token`) is returned in
//! [`StopIds::unresolved`] so the caller can say it is not honoured; a
//! built-in terminator absent from this tokenizer belongs to another model
//! family and is simply not a stop here. When no id resolves it refuses,
//! so a caller never runs as if the stop were honoured.
//!
//! An id-only loop matches the stop *token*: a model spelling the same
//! text out of smaller pieces is not caught, as it is not by any id
//! comparison.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

use larql_vindex::format::filenames::{TOKENIZER_CONFIG_JSON, TOKENIZER_JSON};

use super::eos::{EosConfig, BUILTIN_STOP_STRINGS};

/// The `tokenizer_config.json` key naming the end-of-sequence token.
pub const KEY_EOS_TOKEN: &str = "eos_token";
/// The field holding a token's text when `eos_token` is an object.
const KEY_CONTENT: &str = "content";

/// Why a container's stop ids could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopIdsError {
    /// The container's tokenizer exists but could not be loaded, so its
    /// stop strings cannot be mapped to ids.
    Tokenizer { path: PathBuf, reason: String },
    /// No declared EOS id, and no stop string or `eos_token` is a single
    /// token of the container's tokenizer.
    NoStops { container: PathBuf },
}

impl fmt::Display for StopIdsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StopIdsError::Tokenizer { path, reason } => write!(
                f,
                "cannot load the tokenizer {} to resolve stop strings to ids: {reason}",
                path.display()
            ),
            StopIdsError::NoStops { container } => write!(
                f,
                "the container {} declares no end-of-sequence token: no {} eos_token_id, \
                 and no stop string or {TOKENIZER_CONFIG_JSON} {KEY_EOS_TOKEN} is a token of \
                 its tokenizer",
                container.display(),
                super::eos::GENERATION_CONFIG_FILENAME,
            ),
        }
    }
}

impl std::error::Error for StopIdsError {}

/// `tokenizer_config.json`'s `eos_token`, as a string or `{"content": …}`.
/// `None` when the file, the key or a string value is absent.
pub fn tokenizer_config_eos_token(container: &Path) -> Option<String> {
    let bytes = std::fs::read(container.join(TOKENIZER_CONFIG_JSON)).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    match json.get(KEY_EOS_TOKEN)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get(KEY_CONTENT)?.as_str().map(str::to_string),
        _ => None,
    }
}

/// A container's stops as token ids, and the declared stop strings an
/// id-only loop cannot honour.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StopIds {
    pub ids: HashSet<u32>,
    /// Declared stop strings that are not a single token of the
    /// container's tokenizer (or that have no tokenizer to resolve them),
    /// in declaration order.
    pub unresolved: Vec<String>,
}

/// Every stop the container declares, as token ids (see the module doc).
/// Refuses rather than returning an empty id set.
pub fn resolve_stop_ids(container: &Path) -> Result<StopIds, StopIdsError> {
    let eos = EosConfig::from_vindex_dir(container);
    let builtin = |text: &str| BUILTIN_STOP_STRINGS.contains(&text);
    let mut declared: Vec<String> = eos
        .stop_strings
        .iter()
        .filter(|text| !builtin(text))
        .cloned()
        .collect();
    if let Some(eos_token) = tokenizer_config_eos_token(container) {
        if !declared.contains(&eos_token) {
            declared.push(eos_token);
        }
    }
    let mut stops = StopIds {
        ids: eos.eos_token_ids.clone(),
        unresolved: Vec::new(),
    };
    let path = container.join(TOKENIZER_JSON);
    let tokenizer = if path.is_file() {
        Some(
            tokenizers::Tokenizer::from_file(&path).map_err(|e| StopIdsError::Tokenizer {
                path: path.clone(),
                reason: e.to_string(),
            })?,
        )
    } else {
        None
    };
    let id_of = |text: &str| tokenizer.as_ref().and_then(|t| t.token_to_id(text));
    for text in &eos.stop_strings {
        if builtin(text) {
            stops.ids.extend(id_of(text));
        }
    }
    for text in declared {
        match id_of(&text) {
            Some(id) => {
                stops.ids.insert(id);
            }
            None => stops.unresolved.push(text),
        }
    }
    if stops.ids.is_empty() {
        return Err(StopIdsError::NoStops {
            container: container.to_path_buf(),
        });
    }
    Ok(stops)
}

#[cfg(test)]
mod tests;
