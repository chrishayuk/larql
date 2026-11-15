//! Greedy generation over the session's program: the one path `INFER … GENERATE` and `CAPTURE … GENERATE` both
//! run, so the tokens a capture is taken along are the tokens INFER prints.

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::vindex3::{continue_session, SessionGeneration};
use larql_inference::{EosConfig, SamplingConfig};
use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::tokenizers::Tokenizer;

use super::{confirm_patched_weights_ran, execution_overrides, v3_continuation, V3Runtime};
use crate::error::LqlError;

/// Batch-prefill `prompt_ids` into a fresh continuation, then continue greedily for up to `max_new` tokens,
/// stopping before a built-in stop token. Returns the generated ids and their decoded text.
pub(crate) fn greedy_generate(
    runtime: &V3Runtime,
    overlay: &KnowledgeOverlay,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
) -> Result<(SessionGeneration, String), LqlError> {
    let mut kv = v3_continuation(runtime)?.build();
    let overrides = execution_overrides(runtime, overlay)?;
    let prefill_logits = match &overrides {
        Some(ov) => runtime.prefill_into_overlaid(prompt_ids, ov, &mut *kv),
        None => runtime.prefill_into(prompt_ids, &mut *kv),
    }
    .map_err(|e| LqlError::exec("v3 prefill failed", e))?;
    let mut session = match &overrides {
        Some(ov) => runtime.session_with_kv_overlaid(&mut *kv, ov),
        None => runtime.session_with_kv(&mut *kv),
    }
    .map_err(|e| LqlError::exec("v3 session failed", e))?;
    confirm_patched_weights_ran(overrides.as_ref())?;
    let mut detok = Detokenizer::new(tokenizer);
    detok.seed(prompt_ids);
    let mut text = String::new();
    let result = continue_session(
        &mut session,
        prefill_logits,
        max_new,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |id| text.push_str(&detok.push(id)),
    )
    .map_err(|e| LqlError::exec("v3 decode failed", e))?;
    Ok((result, text))
}
