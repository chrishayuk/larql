//! **Which objects an execution requires.**
//!
//! The question a hydration set exists to answer: given a plan and a
//! slice, which of the container's objects must be present before
//! [`PreparedOperands::load`](super::prepared::PreparedOperands::load)
//! can succeed — and, just as importantly, which must *not* be fetched.
//!
//! # Logical roles are already resolved by the time a plan exists
//!
//! It is tempting to build a resolver here: logical operand → physical
//! realisation → hydration set, with an alias case for a tied output
//! head. That resolver already exists, in
//! [`opplan::build`](super::super::build), and it runs at plan time. A
//! tied component's [`OutputOp`](super::super::OutputOp) carries an
//! operand naming the **embedding object**, because the plan builder
//! collapsed the alias when it saw a component with no head tensor and
//! `head_reuses_embedding` set. Qwen3-4B is the real instance.
//!
//! So every operand a plan names is already a physically stored
//! `(object, tensor)` pair, and a second resolver here would be a second
//! authority for one decision — the bug class the remote tensor source
//! was built to avoid. This module folds; it does not resolve.
//!
//! # Why the fold is driven by serde
//!
//! A plan holds operands in thirty-five places across a dozen operator
//! types, and every new operator adds more. A hand-written walk over
//! those fields would be correct on the day it was written and silently
//! incomplete on the day someone adds an operator — the failure would be
//! a hydration set missing an object, which surfaces as a mid-execution
//! load error rather than as a test failure.
//!
//! Serialising and walking is total by construction: it cannot miss a
//! field it does not know about. What it *can* do is drift from what
//! preparation actually asks for, so that is not argued — it is measured.
//! [`OperandStore::touched_objects`](super::operands::OperandStore::touched_objects)
//! records what a real preparation resolved, and the gate requires the
//! two to be equal.
//!
//! Those two are not the dual-authority pattern the remote source was
//! built to avoid. Neither computes the other's result: one **predicts**
//! from the plan, the other **observes** what preparation asked for.
//! Requiring equality rather than containment catches both directions —
//! underprediction is an object hydration omitted, overprediction is an
//! object hydration transferred for nothing.
//!
//! # Contract: `OperandRef` must stay structurally identifiable
//!
//! This fold recognises an operand by the shape of its serialised form —
//! an object carrying all of [`OPERAND_KEYS`]. That makes the serialised
//! representation of [`OperandRef`](super::super::OperandRef) part of
//! this subsystem's contract, which is not obvious from looking at
//! `OperandRef` itself.
//!
//! **Renaming its fields, flattening it into its parent, or giving it a
//! custom `Serialize` is therefore a hydration-prediction change**, not a
//! serialisation tidy-up. The equality gate would catch it — that is what
//! the gate is for — but the failure would read as a mysterious
//! hydration bug rather than as the consequence of an intended edit, so
//! it is written down here.

use std::collections::BTreeSet;

use serde::Serialize;

use super::super::ComponentOpPlan;
use super::prepared::ExecutionSlice;
use crate::error::VindexError;

/// Field names that together identify a serialised
/// [`OperandRef`](super::super::OperandRef).
///
/// Matching on the shape rather than on one key name: `object` alone
/// appears nowhere else today, but a future op with an `object` field of
/// its own would silently join the set.
const OPERAND_KEYS: [&str; 4] = ["object", "tensor", "dtype", "shape"];

/// The key naming the object an operand lives in.
const OBJECT_KEY: &str = "object";

/// The objects `slice` of `plan` must be able to resolve operands from.
///
/// Slicing is answered by [`ExecutionSlice`] itself, never re-derived
/// here: `layers` gives the layer range and `is_whole_stack` decides
/// whether the embedding, final norm and head are in scope. Only the
/// operand extraction is new.
pub fn required_objects(
    plan: &ComponentOpPlan,
    slice: &ExecutionSlice,
) -> Result<BTreeSet<String>, VindexError> {
    slice.validate(plan)?;
    let mut objects = BTreeSet::new();
    if slice.is_whole_stack() {
        collect(&plan.embedding, &mut objects)?;
        collect(&plan.final_norm, &mut objects)?;
        collect(&plan.output, &mut objects)?;
    }
    let range = slice.layers(plan);
    collect(&plan.layers[range], &mut objects)?;
    Ok(objects)
}

/// Serialise `value` and record every operand's object.
fn collect<T: Serialize + ?Sized>(
    value: &T,
    into: &mut BTreeSet<String>,
) -> Result<(), VindexError> {
    let json = serde_json::to_value(value)
        .map_err(|e| VindexError::Parse(format!("serialising a plan fragment: {e}")))?;
    walk(&json, into);
    Ok(())
}

/// Depth-first walk recording the `object` of every operand-shaped node.
fn walk(node: &serde_json::Value, into: &mut BTreeSet<String>) {
    match node {
        serde_json::Value::Object(map) => {
            if OPERAND_KEYS.iter().all(|key| map.contains_key(*key)) {
                if let Some(object) = map[OBJECT_KEY].as_str() {
                    into.insert(object.to_string());
                }
                // An operand carries no nested operands; not descending
                // keeps `shape` out of the walk.
                return;
            }
            for value in map.values() {
                walk(value, into);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                walk(item, into);
            }
        }
        _ => {}
    }
}
