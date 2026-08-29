//! Token IDs to logits — the last new composition this ladder needs:
//! `embedding lookup → proven 27-layer stack → final RMSNorm → lm_head`.
//! Transcribed from `KimiLinearModel.forward`/`KimiLinearForCausalLM.
//! forward` in the checkpoint's own `modeling_kimi.py`:
//!
//! ```text
//! h      = embed_tokens(input_ids)          # nn.Embedding row gather
//! h      = self.layers(h)                   # exec::stack::stack_forward, UNCHANGED
//! h      = self.norm(h)                     # plain KimiRMSNorm, config.rms_norm_eps
//! logits = self.lm_head(h)                  # nn.Linear, NOT tied to embed_tokens
//! ```
//!
//! **Nothing about the stack is re-verified here** — `stack_forward` is
//! called exactly as `stack_real.rs` already proved it. This module's
//! only new claims are the embedding gather (a memcpy, not a
//! computation, so [`embed`] is a lookup, never a kernel) and the
//! `lm_head` matvec.
//!
//! `lm_head` is routed to the crate's BLAS projector, not `exec::
//! kernels::matvec` (that module's own doc comment: "Deliberately
//! naive... no BLAS, no SIMD: semantic fidelity is the only job" — a
//! Stage A oracle, never a production path). Measured at P3d-n: 262 ms
//! for ONE matvec over the full `163,840 x 2,304` vocabulary — the
//! second-largest per-token cost, right behind MLA, both on the same
//! naive kernel before this fix.
//!
//! `embed_tokens.weight` and `lm_head.weight` are SEPARATE matrices —
//! `tie_word_embeddings=False` in the checkpoint's own `config.json` —
//! so this module never assumes one derives from the other.

use super::cpu::projector::{DenseProjector, WeightRows};
use super::kernels::norm;
use super::stack::{stack_forward, LayerSpec, LayerState, StackLayerTrace};
use super::timing::{timed, OpClass};
use larql_models::config::NormType;

/// Same swap, same reasoning, as `exec::mla`'s own local `matvec`: this
/// changes no arithmetic (`BlasF32` and the naive scalar loop agree up
/// to summation-order float noise), only which kernel runs it.
fn matvec(w: &[f32], x: &[f32], out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out];
    super::cpu::kernels::BlasF32.project_rows(WeightRows::F32(w), x, &mut y);
    y
}

/// One loaded embedding row, tagged by token id — the sparse
/// `LoadedExpert`/`find_expert` pattern applied to the embedding table:
/// a real embedding lookup gathers one row out of `vocab_size` many, and
/// this fixture loads only the rows its own token ids actually select,
/// never `embed_tokens.weight` in full.
pub struct EmbeddingRow<'a> {
    pub id: usize,
    pub vector: &'a [f32],
}

/// Gather this token's embedding row. Linear scan over a handful of
/// loaded rows — correctness over cleverness, the same posture
/// `stack::find_expert` already holds for its own small per-layer lookup.
pub fn embed<'a>(rows: &[EmbeddingRow<'a>], token_id: usize) -> &'a [f32] {
    rows.iter()
        .find(|r| r.id == token_id)
        .unwrap_or_else(|| panic!("token id {token_id} has no loaded embedding row"))
        .vector
}

/// Every boundary the user asked this rung to check, for one position:
/// embedding output, the stack's own full per-layer trace (so a
/// disagreement can still be localised to a specific layer), the final
/// norm, and the logits/argmax.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenTrace {
    /// `embed_tokens(token_id)` — `[hidden]`.
    pub embedding: Vec<f32>,
    /// Every one of the 27 layers' own boundaries, for this position —
    /// `stack::stack_forward`'s own return value, untouched.
    pub layers: Vec<StackLayerTrace>,
    /// `self.norm(stack_output)` — `[hidden]`.
    pub final_normed: Vec<f32>,
    /// `lm_head(final_normed)` — `[vocab_size]`, the full distribution.
    pub logits: Vec<f32>,
    /// `argmax(logits)` — the next-token id this position predicts.
    pub argmax: usize,
}

impl TokenTrace {
    /// The stack's own final layer output — `layers.last().output`,
    /// named here so a caller need not know it is layer 26 specifically.
    pub fn stack_output(&self) -> &[f32] {
        &self
            .layers
            .last()
            .expect("a stack has at least one layer")
            .layer_output
    }
}

/// One token through the whole model: embed → stack (state threaded
/// across calls, exactly as [`stack_forward`] itself requires) → final
/// norm → `lm_head`. Call once per position, in order.
#[allow(clippy::too_many_arguments)]
pub fn token_forward(
    token_id: usize,
    hidden: usize,
    embedding_rows: &[EmbeddingRow<'_>],
    layers: &[LayerSpec<'_>],
    states: &mut [LayerState],
    final_norm_weight: &[f32],
    norm_eps: f64,
    lm_head_weight: &[f32],
    vocab_size: usize,
) -> TokenTrace {
    let embedding = {
        let _t = timed(OpClass::Embed);
        embed(embedding_rows, token_id).to_vec()
    };

    let layer_traces = stack_forward(&embedding, hidden, layers, states);
    let stack_output = layer_traces
        .last()
        .expect("a stack has at least one layer")
        .layer_output
        .clone();

    let final_normed = {
        let _t = timed(OpClass::Norm);
        norm(
            NormType::RmsNorm,
            &stack_output,
            final_norm_weight,
            0.0,
            norm_eps,
        )
    };
    let logits = {
        let _t = timed(OpClass::LmHead);
        matvec(lm_head_weight, &final_normed, vocab_size)
    };
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits are never NaN"))
        .expect("vocab_size is never zero")
        .0;

    TokenTrace {
        embedding,
        layers: layer_traces,
        final_normed,
        logits,
        argmax,
    }
}
