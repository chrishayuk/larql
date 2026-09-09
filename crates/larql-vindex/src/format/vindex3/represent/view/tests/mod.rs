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

use std::collections::BTreeSet;

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

/// The shared record over a real encoded container — one builder, so
/// the stage-4 views and the stage-5b actuation bridge cannot come to
/// disagree about what a record that can answer looks like.
pub(super) fn priced_record(container: &std::path::Path) -> SearchSnapshot {
    fixtures::PricedRecord::new(container).build()
}

/// The same record asked from a given applied set.
pub(super) fn priced_record_from(
    container: &std::path::Path,
    applied: BTreeSet<String>,
) -> SearchSnapshot {
    fixtures::PricedRecord::new(container)
        .applied(applied)
        .build()
}

/// The same record over a surface of a chosen SHAPE, which is what
/// makes the declared layout policy observable.
pub(super) fn priced_record_shaped(
    container: &std::path::Path,
    shape: Vec<usize>,
    declared_layout: &str,
) -> SearchSnapshot {
    fixtures::PricedRecord::new(container)
        .shape(shape)
        .layout(declared_layout)
        .build()
}

/// A record whose accounting facts were read from ANOTHER container.
pub(super) fn priced_record_with_foreign_accounting(
    container: &std::path::Path,
    other: &std::path::Path,
) -> SearchSnapshot {
    fixtures::PricedRecord::new(container)
        .accounting_from(other)
        .build()
}

/// A record whose two edits resolve DIFFERENTLY, so the policy has more
/// than one opportunity to order.
pub(super) fn priced_record_ranked(container: &std::path::Path) -> SearchSnapshot {
    fixtures::PricedRecord::new(container)
        .distinct_edits()
        .build()
}

pub(super) fn priced_record_under(
    container: &std::path::Path,
    applied: BTreeSet<String>,
    declared_layout: &str,
    declared_accounting: &str,
) -> SearchSnapshot {
    fixtures::PricedRecord::new(container)
        .applied(applied)
        .layout(declared_layout)
        .accounting(declared_accounting)
        .build()
}
