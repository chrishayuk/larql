//! **Stage 5b: the bridge restates an authorisation and cannot author
//! one.**
//!
//! Every test below runs against the shared record over a REAL encoded
//! container — the same builder the stage-4 view tests use, so the thing
//! being actuated is the thing that was rendered, and not a second idea
//! of what a record is.
//!
//! Two kinds of check, and both are needed:
//!
//! ```text
//! translation   what the record declares reaches the request, and the
//!               adapter's historic defaults never do
//! truthfulness  a request states the experiment it names, and cannot be
//!               obtained when it would not
//! ```
//!
//! The first alone would pass on a bridge that faithfully carried a
//! record's fields into a request for the wrong experiment.

mod executor;
mod prepare;
mod request;

use super::super::state::fixtures::{self, PricedRecord};
use super::super::state::protocol::MeasurementProtocol;
use super::super::state::snapshot::SearchSnapshot;
use super::super::state::tests::container;
use super::PreparedExperiment;

/// The record that can answer AND can be prepared: real container
/// prices, and the declarations its standing intent stands for.
fn ready_record(dir: &std::path::Path) -> SearchSnapshot {
    PricedRecord::new(dir)
        .with_protocol(fixtures::protocol())
        .build()
}

/// A record that DECLARES one protocol and SEARCHES under another —
/// the inconsistency a record must be refused for.
fn record_declaring(dir: &std::path::Path, protocol: MeasurementProtocol) -> SearchSnapshot {
    PricedRecord::new(dir).with_protocol_only(protocol).build()
}

/// A record that declares a protocol and searches under it.
fn record_with(dir: &std::path::Path, protocol: MeasurementProtocol) -> SearchSnapshot {
    PricedRecord::new(dir).with_protocol(protocol).build()
}

/// The prepared experiment, or a panic naming what came back instead.
fn ready(snapshot: &SearchSnapshot) -> super::Ready {
    match PreparedExperiment::of(snapshot) {
        PreparedExperiment::Ready(ready) => *ready,
        other => panic!("the record should prepare: {other:?}"),
    }
}

fn glimmer() -> tempfile::TempDir {
    container::glimmer()
}
