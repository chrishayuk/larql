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

/// A site this pass exposes: a plane a layer hands the traversal's subscriber, or a value at its exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PassSite {
    Layer(LayerPlane),
    Exit(ExitValue),
}

/// The per-layer planes the traversal hands its subscriber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LayerPlane {
    Residual,
    PostAttention,
    FfnInput,
}

/// What the traversal's exit holds at a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExitValue {
    Hidden,
    Logits,
}

impl PassSite {
    /// The pass's name for `site`; `None` for a site only the decode steps expose.
    pub(super) fn of(site: CaptureSite) -> Option<Self> {
        match site {
            CaptureSite::Residual => Some(Self::Layer(LayerPlane::Residual)),
            CaptureSite::PostAttention => Some(Self::Layer(LayerPlane::PostAttention)),
            CaptureSite::FfnInput => Some(Self::Layer(LayerPlane::FfnInput)),
            CaptureSite::Final => Some(Self::Exit(ExitValue::Hidden)),
            CaptureSite::Logits => Some(Self::Exit(ExitValue::Logits)),
            CaptureSite::FfnDownInput | CaptureSite::FfnDownOutput => None,
        }
    }
}

/// One traversal of `ids`, keeping `site` at `layers` (or at the exit) and `positions`. One block of rows per
/// layer, in the order given; one block for a site after the last layer.
pub(super) fn capture(
    runtime: &V3Runtime,
    overrides: Option<&OperandOverrides>,
    continuation: &SelectedContinuation,
    ids: &[u32],
    site: PassSite,
    layers: &[usize],
    positions: &[usize],
) -> Result<Vec<Rows>, LqlError> {
    let mut kept: BTreeMap<usize, Rows> = BTreeMap::new();
    let mut sink = |event: PlaneEvent| {
        if let (PassSite::Layer(plane), PlaneEvent::Layer { index, trace }) = (site, event) {
            if layers.contains(&index) {
                let rows = match plane {
                    LayerPlane::Residual => plane_rows(&trace.post_layer, index)?,
                    LayerPlane::PostAttention => plane_rows(&trace.post_attention, index)?,
                    LayerPlane::FfnInput => &trace.ffn_input[..],
                };
                kept.insert(index, positions.iter().map(|&p| rows[p].clone()).collect());
            }
        }
        Ok(())
    };
    let exit_positions = match site {
        PassSite::Layer(_) => &[][..],
        PassSite::Exit(_) => positions,
    };
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
        PassSite::Exit(ExitValue::Hidden) => {
            vec![observed.exits.into_iter().map(|e| e.hidden).collect()]
        }
        PassSite::Exit(ExitValue::Logits) => {
            vec![observed.exits.into_iter().map(|e| e.logits).collect()]
        }
        PassSite::Layer(_) => layers
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
