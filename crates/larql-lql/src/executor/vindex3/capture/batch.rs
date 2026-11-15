//! The sites `INFER`'s own pass exposes: one fresh causal traversal of the whole sequence, over the operands
//! `INFER` resolves, observed.
//!
//! The per-layer sites are the planes the traversal hands its subscriber (`PlaneEvent::Layer`); `final` and
//! `logits` are the traversal's exit at each requested position. At the sequence's last position that exit is
//! the traversal's own output, checked bit for bit by the executor, so a `logits` capture there IS the vector
//! `INFER` ranks for the same tokens.

use std::collections::BTreeMap;

use larql_inference::vindex3::{OperandOverrides, PlaneEvent, SelectedContinuation};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::{
    execute_streaming_position_exits, requires_continuation, Plane,
};
use larql_vindex::VindexError;

use super::super::V3Runtime;
use super::Rows;
use crate::ast::CaptureSite;
use crate::error::LqlError;

/// What INFER says of a component without an output head; a capture refuses in the same words.
const NO_HEAD: &str = "the component carries no output head";

/// One traversal of `ids`, keeping `site` at `layers` (or at the exit) and `positions`. One block of rows per
/// layer, in the order given; one block for a site after the last layer.
pub(super) fn capture(
    runtime: &V3Runtime,
    overrides: Option<&OperandOverrides>,
    continuation: &SelectedContinuation,
    ids: &[u32],
    site: CaptureSite,
    layers: &[usize],
    positions: &[usize],
) -> Result<Vec<Rows>, LqlError> {
    let mut kept: BTreeMap<usize, Rows> = BTreeMap::new();
    let mut sink = |event: PlaneEvent| {
        if let PlaneEvent::Layer { index, trace } = event {
            if layers.contains(&index) {
                let rows = match site {
                    CaptureSite::Residual => plane_rows(&trace.post_layer, index)?,
                    CaptureSite::PostAttention => plane_rows(&trace.post_attention, index)?,
                    CaptureSite::FfnInput => &trace.ffn_input[..],
                    _ => unreachable!("only the per-layer sites INFER's pass exposes come here"),
                };
                kept.insert(index, positions.iter().map(|&p| rows[p].clone()).collect());
            }
        }
        Ok(())
    };
    let exit_positions = if site.per_layer() { &[][..] } else { positions };
    let plan = runtime.plan();
    let source = match overrides {
        Some(overrides) => OperandSource::overlaid(runtime.store(), overrides),
        None => runtime.operands(),
    };
    // A stack that keeps state beyond softmax attention runs over a fresh provider, as INFER's pass does.
    let state = requires_continuation(plan).then(|| continuation.build());
    let observed = execute_streaming_position_exits(
        plan,
        source,
        ids,
        runtime.backend(),
        state,
        exit_positions,
        &mut sink,
    )
    .map_err(|e| LqlError::exec("capture pass failed", e))?;
    if observed.output.logits.is_none() {
        return Err(LqlError::Execution(NO_HEAD.into()));
    }
    Ok(match site {
        CaptureSite::Final => vec![observed.exits.into_iter().map(|e| e.hidden).collect()],
        CaptureSite::Logits => vec![observed.exits.into_iter().map(|e| e.logits).collect()],
        _ => layers
            .iter()
            .map(|layer| kept.remove(layer).unwrap_or_default())
            .collect(),
    })
}

/// A plane's `[hidden]` rows, refused on a component whose residual is not single-stream.
fn plane_rows(plane: &Plane, layer: usize) -> Result<&[Vec<f32>], VindexError> {
    plane.try_rows().map_err(|e| {
        VindexError::Parse(format!(
            "layer {layer}'s residual has no single [hidden] row per position to capture: {e}"
        ))
    })
}
