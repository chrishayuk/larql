//! Override composition, residual capture and continuation for VINDEX3.

use crate::error::LqlError;
use larql_kv::{shipped_continuations, CanonicalFactory};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    ContinuationFactory, SelectedContinuation,
};
use larql_vindex::tokenizers::Tokenizer;

/// The one raw-prompt encoder every V3 surface calls — the BOS rule
/// lives with it in larql-inference, not here.
pub(crate) use larql_inference::vindex3::encode_raw_prompt;

#[allow(unused_imports)]
use super::*;

/// One observed pass over the runtime's plan, returning the last
/// position's **normed FFN input** at `layer` — V2's exact install
/// statistic (its walk-FFN trace captures the post-norm vector the
/// gates multiply), so a gate built from it fires on the prompt that
/// produced it (V3-LQL-3B capture: the KNN key and the compose gate
/// direction). The pass is the same canonical traversal INFER runs;
/// the tap is a subscription, never a second executor.
/// The overlay's compose edits as executor operand overrides — `None`
/// while no compose state exists, so callers keep the plain (bit-for-
/// bit identical) execution path.
pub(crate) fn compose_overrides(
    runtime: &V3Runtime,
    overlay: &larql_vindex::format::vindex3::knowledge::KnowledgeOverlay,
) -> Result<Option<larql_inference::vindex3::OperandOverrides>, LqlError> {
    if !overlay.has_vector_state() {
        return Ok(None);
    }
    overlay
        .operand_overrides(runtime.plan())
        .map(Some)
        .map_err(|e| LqlError::exec("failed to derive operand overrides", e))
}

pub(crate) fn capture_layer_residual(
    runtime: &V3Runtime,
    tokenizer: &Tokenizer,
    prompt: &str,
    layer: usize,
    overrides: Option<&larql_inference::vindex3::OperandOverrides>,
    bos: Option<u32>,
) -> Result<Vec<f32>, LqlError> {
    let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos)?;
    let mut captured: Option<Vec<f32>> = None;
    let mut sink = |event: larql_inference::vindex3::PlaneEvent| {
        if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = event {
            if index == layer {
                captured = trace.ffn_input.last().cloned();
            }
        }
        Ok(())
    };
    // The capture runs over the same effective program INFER runs —
    // V2's contract (its capture forward observes the patch overlay).
    let continuation = v3_continuation(runtime)?;
    match overrides {
        Some(overrides) => {
            runtime.execute_streaming_overlaid(&prompt_ids, overrides, &continuation, &mut sink)
        }
        None => runtime.execute_streaming(&prompt_ids, &continuation, &mut sink),
    }
    .map_err(|e| LqlError::exec("v3 capture pass failed", e))?;
    captured.ok_or_else(|| LqlError::Execution(format!("no residual captured at layer {layer}")))
}

/// LQL's continuation provider — the canonical cache, named by its
/// factory's identity (CONTINUATION-PLUGIN-1, C3) and selected against the
/// runtime's program before anything runs.
pub(crate) fn v3_continuation(runtime: &V3Runtime) -> Result<SelectedContinuation, LqlError> {
    runtime
        .select_continuation(
            &shipped_continuations(),
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
        )
        .map_err(|e| LqlError::exec("v3 continuation selection failed", e))
}

/// Softmax over the logits, then the top-k `(token_id, probability)`
/// pairs, ties keeping the lower id (the greedy sampler's rule).
pub(crate) fn top_k_probs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut scored: Vec<(u32, f32)> = exps
        .iter()
        .enumerate()
        .map(|(id, &e)| (id as u32, e / sum))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}
