//! Logits at chosen positions from ONE fresh causal prefill.
//!
//! A one-shot traversal already computes every position's final residual
//! under the causal mask — the exit merely keeps the last. This applies the
//! same exit (final norm, then the head over the normed row) to each
//! requested position of that one plane, so the logits predicting the token
//! after `tokens[..=p]` come out for every `p` asked for at the cost of a
//! single forward, instead of one forward per prefix (O(n²) over a
//! sequence).
//!
//! Independent of cached decode by construction: the traversal is the
//! one-shot form ([`super::streaming`]'s `execute_prepared_streaming_with`),
//! which refuses continuation state that is not fresh and, on a
//! softmax-only stack, runs with NO state at all — no KV rows exist for a
//! later step to read. Nothing here steps a [`super::decode::DecodeSession`].

use super::continuation_registry::BoxedContinuation;
use super::hyper_connection::Mutation;
use super::kv::KvState;
use super::operands::OperandSource;
use super::prepared::{ExecutionSlice, PreparedOperands};
use super::{backend::PlanBackend, ComponentOpPlan, Plane, PlaneEvent};
use crate::error::VindexError;

/// The logits predicting the token after `tokens[..=p]`, for each `p` in
/// `positions` (0-based), from one fresh causal prefill over `tokens`.
///
/// `state` is the fresh continuation provider a stateful stack needs (the
/// `_in` forms' rule); `None` on a softmax-only stack. Single-stream stacks
/// only: a hyper-connected or attention-residual plane has no per-position
/// `[hidden]` row to exit from here, and is refused rather than reduced
/// differently from the traversal's own exit.
///
/// Fail-closed self-check: when the last position is requested, its row
/// must equal the traversal's own exit logits bit for bit — the same
/// functions over the same row — or the call refuses.
pub fn execute_slice_position_logits<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    slice: ExecutionSlice,
    state: Option<BoxedContinuation>,
    positions: &[usize],
) -> Result<Vec<Vec<f32>>, VindexError> {
    execute_slice_position_logits_with_rope_angles(
        plan,
        store,
        tokens,
        backend,
        slice,
        state,
        positions,
        super::rope_angles::RopeAngles::Larql,
    )
}

/// [`execute_slice_position_logits`] with the run's RoPE construction
/// chosen (PREC-DIAG-1 E0b). `RopeAngles::Larql` is the function above,
/// bit for bit; it delegates here.
#[allow(clippy::too_many_arguments)]
pub fn execute_slice_position_logits_with_rope_angles<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    slice: ExecutionSlice,
    mut state: Option<BoxedContinuation>,
    positions: &[usize],
    rope_angles: super::rope_angles::RopeAngles,
) -> Result<Vec<Vec<f32>>, VindexError> {
    if let Some(bad) = positions.iter().find(|&&p| p >= tokens.len()) {
        return Err(VindexError::Parse(format!(
            "position {bad} is outside a {}-token sequence",
            tokens.len()
        )));
    }
    let mut ops = PreparedOperands::load(plan, store, backend, slice)?;
    ops.set_rope_angles(rope_angles)?;
    // Only the newest layer's plane is kept: each boundary replaces the
    // last, so the plane after the final executed layer is what remains.
    let mut exit_plane: Option<Plane> = None;
    let mut sink = |event: PlaneEvent| {
        if let PlaneEvent::Layer { trace, .. } = event {
            exit_plane = Some(trace.post_layer);
        }
        Ok(())
    };
    let out = super::streaming::execute_prepared_streaming_with(
        plan,
        &ops,
        tokens,
        backend,
        None,
        &mut sink,
        state.as_mut().map(|s| &mut **s as &mut dyn KvState),
        Mutation::None,
    )?;
    let rows = exit_rows(exit_plane)?;
    let mut logits = Vec::with_capacity(positions.len());
    for &p in positions {
        let hidden = match ops.final_norm() {
            Some(norm) => norm.apply(backend, &rows[p]),
            None => rows[p].clone(),
        };
        let row = ops.head_over_normed(backend, &hidden)?;
        let row = row.ok_or_else(|| VindexError::Parse(NO_HEAD.into()))?;
        if p + 1 == tokens.len() {
            same_as_exit(&row, out.logits.as_deref())?;
        }
        logits.push(row);
    }
    Ok(logits)
}

const NO_HEAD: &str = "plan carries no output head";

/// The per-position `[hidden]` rows of the exit plane; refused on a
/// bundle or residual history (no single-stream exit), or when no layer
/// ran.
pub(super) fn exit_rows(plane: Option<Plane>) -> Result<Vec<Vec<f32>>, VindexError> {
    match plane {
        Some(Plane::Rows(rows)) => Ok(rows),
        Some(Plane::Bundles(_) | Plane::Histories(_)) => Err(VindexError::Parse(
            "per-position logits need a single-stream stack; this plane is a bundle or a \
             residual history"
                .into(),
        )),
        None => Err(VindexError::Parse("the traversal executed no layer".into())),
    }
}

/// The fail-closed self-check: the last position's row is the traversal's
/// own exit, bit for bit.
pub(super) fn same_as_exit(row: &[f32], exit: Option<&[f32]>) -> Result<(), VindexError> {
    match exit == Some(row) {
        true => Ok(()),
        false => Err(VindexError::Parse(
            "the last position's logits differ from the traversal's own exit — the \
             per-position exit is not the exit the traversal runs"
                .into(),
        )),
    }
}
