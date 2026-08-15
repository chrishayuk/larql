//! BW10 bump sites for the Metal decode path.
//!
//! Each submodule owns the byte accounting for one instrumented surface
//! (see `larql_compute::movement_ledger::coverage::Surface`) and is the
//! single authority for it, so two encode paths reading the same weights
//! cannot disagree about how many bytes they read.

pub(crate) mod experts;
