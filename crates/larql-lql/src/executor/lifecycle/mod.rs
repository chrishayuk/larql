//! Lifecycle executor: USE, STATS, EXTRACT, COMPILE, DIFF.
//!
//! Each verb lives in its own file; this module is a pure re-export
//! point, so `Session::exec_*` method lookups resolve unchanged.

mod compact_into;
mod compile;
mod diff;
mod diff_v3;
mod extract;
mod stats;
#[cfg(test)]
mod tests;
mod use_cmd;
