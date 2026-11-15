//! `vindex3 exec --backend metal-lowered*`: the CLI's drivers over the
//! Metal lowered session.
//!
//! The session itself — device residency, the per-position step, VERIFY-N,
//! the measure arm — is library code in
//! [`larql_vindex::format::vindex3::opplan::exec::metal_lowered`]. What
//! stays here is front-end: argument handling and output (`run`), the
//! per-layer and teacher-forced dumps, and the speculative decode loop.
//! Its policy (prompt lookup, accepted prefix, stop handling) is shared,
//! in [`larql_inference::speculation`].

mod dump;
mod run;
mod teacher_force;
#[cfg(test)]
mod tests;

pub(super) use run::run_lowered;

pub(crate) use larql_vindex::format::vindex3::opplan::exec::metal_lowered::{
    measure_arm, LoweredSession,
};
