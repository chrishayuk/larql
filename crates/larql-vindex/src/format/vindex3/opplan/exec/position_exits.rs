//! One fresh causal prefill, observed: every plane the traversal produces, and the exit at chosen positions.
//!
//! The one-shot traversal `INFER` runs computes a residual for every position at every layer and keeps only the
//! last position's exit. A caller that wants to see more of that same run (the planes at a layer, the final-normed
//! row or the logits at an earlier position) subscribes here instead of running anything else: plane events are
//! forwarded to the caller's sink as they occur, and the exit [`super::position_logits`] defines (final norm, then
//! the head) is applied to each requested position of the last layer's plane.
//!
//! Fail-closed self-check: the last position's exit, whenever any position is requested, must equal the
//! traversal's own exit bit for bit, both the final-normed row and the logits, or the call refuses.

use super::continuation_registry::BoxedContinuation;
use super::hyper_connection::Mutation;
use super::kv::KvState;
use super::operands::OperandSource;
use super::position_logits::{exit_rows, row_exit, same_as_exit};
use super::prepared::{ExecutionSlice, PreparedOperands};
use super::{backend::PlanBackend, ComponentOpPlan, FinalOutput, FinalState, Plane, PlaneEvent};
use crate::error::VindexError;

/// What the exit computed at one position.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionExit {
    /// The `[hidden]` vector the head read: the final-normed residual, or the residual itself where the image
    /// declares no final norm.
    pub hidden: Vec<f32>,
    /// The logits predicting the token after this position.
    pub logits: Vec<f32>,
}

/// What [`execute_streaming_position_exits`] returns: the traversal's own output, as `INFER` receives it, and
/// one exit per requested position, in the order requested.
#[derive(Debug)]
pub struct ObservedPrefill {
    pub output: FinalOutput,
    pub exits: Vec<PositionExit>,
}

const LAST_HIDDEN_DIFFERS: &str =
    "the last position's final-normed row differs from the traversal's own \
     exit; the per-position exit is not the exit the traversal runs";

/// Execute `tokens` once over the whole stack, forward every [`PlaneEvent`] to `sink`, and apply the exit at each
/// of `positions`. With no positions requested, no plane is retained and this is the plain streaming traversal.
///
/// `state` is the fresh continuation provider a stateful stack needs, `None` on a softmax-only stack (the rule
/// [`super::requires_continuation`] states). Per-position exits need a single-stream stack.
pub fn execute_streaming_position_exits<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    mut state: Option<BoxedContinuation>,
    positions: &[usize],
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<ObservedPrefill, VindexError> {
    if let Some(bad) = positions.iter().find(|&&p| p >= tokens.len()) {
        return Err(VindexError::Parse(format!(
            "position {bad} is outside a {}-token sequence",
            tokens.len()
        )));
    }
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    let keep_exit_plane = !positions.is_empty();
    // Only the newest layer's plane is kept: each boundary replaces the last.
    let mut exit_plane: Option<Plane> = None;
    let mut forward = |event: PlaneEvent| {
        if keep_exit_plane {
            if let PlaneEvent::Layer { trace, .. } = &event {
                exit_plane = Some(trace.post_layer.clone());
            }
        }
        sink(event)
    };
    let output = super::streaming::execute_prepared_streaming_with(
        plan,
        &ops,
        tokens,
        backend,
        None,
        &mut forward,
        state.as_mut().map(|s| &mut **s as &mut dyn KvState),
        Mutation::None,
    )?;
    if !keep_exit_plane {
        return Ok(ObservedPrefill {
            output,
            exits: Vec::new(),
        });
    }
    let rows = exit_rows(exit_plane)?;
    let mut exits = Vec::with_capacity(positions.len());
    for &p in positions {
        let (hidden, logits) = row_exit(&ops, backend, &rows[p])?;
        if p + 1 == tokens.len() {
            same_as_exit(&logits, output.logits.as_deref())?;
            same_hidden(&hidden, &output.exit)?;
        }
        exits.push(PositionExit { hidden, logits });
    }
    Ok(ObservedPrefill { output, exits })
}

/// The final-normed row at the last position is the traversal's own exit state, bit for bit.
pub(super) fn same_hidden(hidden: &[f32], exit: &FinalState) -> Result<(), VindexError> {
    let same = exit.try_hidden().is_ok_and(|exit| {
        exit.len() == hidden.len()
            && exit
                .iter()
                .zip(hidden)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    });
    match same {
        true => Ok(()),
        false => Err(VindexError::Parse(LAST_HIDDEN_DIFFERS.into())),
    }
}
