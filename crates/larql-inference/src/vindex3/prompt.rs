//! The one raw-prompt encoder every VINDEX3 surface calls.
//!
//! A raw prompt — a plain completion, not rendered through a chat
//! template — must reach the model as the same token sequence whichever
//! surface sent it: `larql run --raw`, `larql vindex3 observe`, the
//! server's plain completions and LQL `INFER`/`TRACE`. Each of them used
//! to encode for itself, and they disagreed in both directions: LQL
//! prepended a BOS that Granite and Qwen3 declare is never added, while
//! the other three omitted the BOS Gemma 4 requires and its tokenizer
//! does not add (Gemma 4 12B answered "1111…" to a raw prompt).
//!
//! The rule, read from the container's own carried declarations
//! (`docs/vindex-generation-policy.md`, "Cross-generation contract:
//! prompt encoding"):
//!
//! 1. the tokenizer's own encoding, post-processor included;
//! 2. `tokenizer_config.json` declaring `add_bos_token: false` — done;
//! 3. otherwise a declared BOS id (`generation_config.json`, then
//!    `tokenizer_config.json`) is prepended unless the encoding already
//!    starts with it ([`maybe_prepend_bos`]).
//!
//! A templated prompt carries its own special tokens and is encoded by
//! the chat path without the post-processor; it never comes here.

use std::path::Path;

use larql_vindex::format::filenames::{GENERATION_CONFIG_JSON, TOKENIZER_CONFIG_JSON};

use crate::tokenizer::maybe_prepend_bos;

/// HF configs a container carries that declare the BOS token id, most
/// authoritative first. Both arrive with the M2 capability snapshot.
pub const BOS_DECLARING_FILES: [&str; 2] = [GENERATION_CONFIG_JSON, TOKENIZER_CONFIG_JSON];

/// The key that declares a BOS id in either file.
pub const BOS_TOKEN_ID_KEY: &str = "bos_token_id";

/// The `tokenizer_config.json` key with which a checkpoint states
/// whether a prompt starts with BOS.
pub const ADD_BOS_TOKEN_KEY: &str = "add_bos_token";

/// Why a raw prompt could not be encoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptEncodeError {
    /// The tokenizer refused the text.
    Tokenize(String),
    /// The text encodes to no tokens — nothing to condition on. Judged
    /// before any BOS is added: a BOS alone is not a prompt.
    Empty,
}

impl std::fmt::Display for PromptEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenize(e) => write!(f, "tokenize: {e}"),
            Self::Empty => f.write_str("prompt tokenises to empty"),
        }
    }
}

impl std::error::Error for PromptEncodeError {}

/// The BOS id a raw prompt for this container starts with, or `None`.
///
/// Two facts, two authorities: `bos_token_id` says *which* token BOS is;
/// `add_bos_token` says *whether* a prompt gets one. Granite and Qwen3
/// declare an id **and** `add_bos_token: false`, so they get none. Gemma
/// 4 declares id 2 and no `add_bos_token`, so it gets 2. A container
/// that declares nothing (one predating the capability snapshot) gets
/// none. Unreadable or malformed config is no declaration.
pub fn declared_prompt_bos(container: &Path) -> Option<u32> {
    if declares_no_bos(container) {
        return None;
    }
    BOS_DECLARING_FILES.iter().find_map(|name| {
        read_json(container, name)?
            .get(BOS_TOKEN_ID_KEY)?
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
    })
}

/// Encode a raw prompt: the tokenizer's own encoding, then `bos`
/// prepended unless already first. Pass the container's
/// [`declared_prompt_bos`], resolved once at bind.
pub fn encode_raw_prompt(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    bos: Option<u32>,
) -> Result<Vec<u32>, PromptEncodeError> {
    let ids = tokenizer
        .encode(prompt, true)
        .map_err(|e| PromptEncodeError::Tokenize(e.to_string()))?
        .get_ids()
        .to_vec();
    if ids.is_empty() {
        return Err(PromptEncodeError::Empty);
    }
    Ok(maybe_prepend_bos(ids, bos))
}

/// Whether `tokenizer_config.json` explicitly declares
/// `add_bos_token: false`. Only a boolean `false` is that declaration.
fn declares_no_bos(container: &Path) -> bool {
    read_json(container, TOKENIZER_CONFIG_JSON)
        .and_then(|json| json.get(ADD_BOS_TOKEN_KEY)?.as_bool())
        == Some(false)
}

fn read_json(container: &Path, name: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(container.join(name)).ok()?;
    serde_json::from_str(&text).ok()
}
