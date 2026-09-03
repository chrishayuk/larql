//! **Stage 4: the facade renders, and derives nothing.**
//!
//! Every test below runs against the Rung 5 record — the same facts the
//! 1d replay gate uses, reloaded from JSON so that what is rendered came
//! out of storage and not out of the object that built it.
//!
//! Two kinds of check, and both are needed:
//!
//! ```text
//! origin      every rendered FIELD names a substrate call, and every
//!             declared call is reached — the registry cannot rot
//! render      every rendered VALUE equals the substrate's own answer,
//!             and the real Rung 5 numbers survive the round trip
//! ```
//!
//! The first alone would pass on a view that declared honest origins and
//! then rendered nonsense. The second alone would pass on a view that
//! grew an undeclared field nobody thought to assert.

mod compare;
mod current;
mod describe;
mod evidence;
mod explain;
mod frontier;
mod next_experiment;
mod origin;
mod render;

use super::super::state::fixtures;
use super::super::state::snapshot::SearchSnapshot;
use super::OptimizerView;

/// The Rung 5 record, stored and read back.
pub(super) fn reloaded() -> SearchSnapshot {
    let json = serde_json::to_string(&fixtures::rung5_snapshot()).expect("serialize");
    let back: SearchSnapshot = serde_json::from_str(&json).expect("deserialize");
    back.check_schema().expect("schema");
    back
}

/// A facade over the reloaded record.
pub(super) fn view(snapshot: &SearchSnapshot) -> OptimizerView<'_> {
    OptimizerView::new(snapshot)
}
