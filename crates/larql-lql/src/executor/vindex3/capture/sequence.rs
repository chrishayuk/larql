//! The sequence a capture runs over, and which of its positions and layers it keeps.
//!
//! The prompt is encoded exactly as `INFER` encodes it ([`encode_raw_prompt`]: the tokenizer's own special
//! tokens, plus the container's declared BOS where the tokenizer did not add it; no chat template). A `CONTINUE`
//! text is encoded with no special tokens and appended; `GENERATE` appends the ids `INFER … GENERATE` produces.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::tokenizers::Tokenizer;

use super::super::{encode_raw_prompt, generate::greedy_generate, V3Runtime};
use super::refused;
use crate::ast::{CaptureContinuation, CaptureLayers, CapturePositions, CaptureSite};
use crate::error::LqlError;

/// How a `CONTINUE` text is encoded: no BOS, no other special tokens.
const CONTINUATION_SPECIAL_TOKENS: bool = false;

/// The token ids a capture runs over: the prompt's, then the continuation's.
#[derive(Debug, Clone)]
pub(super) struct Sequence {
    pub ids: Vec<u32>,
    pub prompt_len: usize,
    /// How the continuation was given, as the sidecar records it.
    pub continuation: serde_json::Value,
}

impl Sequence {
    /// The token after `position`, where the sequence has one: what that position's logits predict.
    pub fn predicts(&self, position: usize) -> Option<u32> {
        self.ids.get(position + 1).copied()
    }
}

/// Encode the prompt as INFER does and append the continuation.
pub(super) fn build(
    runtime: &V3Runtime,
    overlay: &KnowledgeOverlay,
    tokenizer: &Tokenizer,
    bos: Option<u32>,
    prompt: &str,
    continuation: Option<&CaptureContinuation>,
) -> Result<Sequence, LqlError> {
    let mut ids = encode_raw_prompt(tokenizer, prompt, bos)?;
    let prompt_len = ids.len();
    let record = match continuation {
        None => serde_json::json!({ "kind": "none" }),
        Some(CaptureContinuation::Text(text)) => {
            let encoded = tokenizer
                .encode(text.as_str(), CONTINUATION_SPECIAL_TOKENS)
                .map_err(|e| LqlError::exec("tokenize the continuation", e))?;
            let added = encoded.get_ids().to_vec();
            if added.is_empty() {
                return Err(refused(format!(
                    "the continuation {text:?} encodes to no tokens"
                )));
            }
            ids.extend_from_slice(&added);
            serde_json::json!({ "kind": "text", "text": text, "token_ids": added })
        }
        Some(CaptureContinuation::Tokens(added)) => {
            // The embedding table is the authority on which ids the program can read.
            let vocab = runtime
                .plan()
                .embedding
                .as_ref()
                .and_then(|e| e.table.shape.first().copied());
            if let Some(vocab) = vocab {
                if let Some(bad) = added.iter().find(|&&id| id as usize >= vocab) {
                    return Err(refused(format!(
                        "token id {bad} is outside the embedding's vocabulary of {vocab}"
                    )));
                }
            }
            ids.extend_from_slice(added);
            serde_json::json!({ "kind": "tokens", "token_ids": added })
        }
        Some(CaptureContinuation::Generate(n)) => {
            let (generated, text) =
                greedy_generate(runtime, overlay, tokenizer, &ids, *n as usize)?;
            ids.extend_from_slice(&generated.tokens);
            serde_json::json!({
                "kind": "generate",
                "requested": n,
                "token_ids": generated.tokens,
                "text": text,
                "path": "INFER ... GENERATE: batch prefill of the prompt, then greedy decode steps, stopping before a built-in stop token",
            })
        }
    };
    Ok(Sequence {
        ids,
        prompt_len,
        continuation: record,
    })
}

/// The positions a clause names, in the order the array holds them.
pub(super) fn positions(
    clause: &CapturePositions,
    sequence: &Sequence,
) -> Result<Vec<usize>, LqlError> {
    let len = sequence.ids.len();
    let prompt_end = sequence.prompt_len - 1;
    let chosen: Vec<usize> = match clause {
        CapturePositions::PromptEnd => vec![prompt_end],
        CapturePositions::Last => vec![len - 1],
        CapturePositions::All => (0..len).collect(),
        CapturePositions::Reply => {
            if len == sequence.prompt_len {
                return Err(refused(
                    "POSITIONS REPLY needs a continuation (CONTINUE or GENERATE) with at least one token",
                ));
            }
            (prompt_end..len - 1).collect()
        }
        CapturePositions::List(list) => {
            if let Some(bad) = list.iter().find(|&&p| p >= len) {
                return Err(refused(format!(
                    "position {bad} is outside the {len}-token sequence (positions 0 to {})",
                    len - 1
                )));
            }
            refuse_repeats("position", list)?;
            list.clone()
        }
    };
    Ok(chosen)
}

/// The layers a capture reads, refused where the site does not name one or the plan has no such layer.
pub(super) fn layers(
    site: CaptureSite,
    clause: Option<&CaptureLayers>,
    plan_layers: usize,
) -> Result<Vec<usize>, LqlError> {
    match (site.per_layer(), clause) {
        (true, None) => Err(refused(format!(
            "the site {} is inside a layer: name it with AT LAYER <n> or AT LAYERS (<n>, ...)",
            site.name()
        ))),
        (false, Some(_)) => Err(refused(format!(
            "the site {} is after the last layer and takes no AT LAYER clause",
            site.name()
        ))),
        (false, None) => Ok(Vec::new()),
        (true, Some(clause)) => {
            let layers = clause.layers();
            if let Some(bad) = layers.iter().find(|&&l| l >= plan_layers) {
                return Err(refused(format!(
                    "layer {bad} does not exist: the program has {plan_layers} layers (0 to {})",
                    plan_layers.saturating_sub(1)
                )));
            }
            refuse_repeats("layer", &layers)?;
            Ok(layers)
        }
    }
}

fn refuse_repeats(what: &str, list: &[usize]) -> Result<(), LqlError> {
    let mut seen = std::collections::BTreeSet::new();
    match list.iter().find(|item| !seen.insert(**item)) {
        Some(repeat) => Err(refused(format!("{what} {repeat} is listed twice"))),
        None => Ok(()),
    }
}
