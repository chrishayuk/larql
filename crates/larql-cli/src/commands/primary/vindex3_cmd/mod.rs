//! `larql vindex3` — VINDEX3 container programme verbs.
//!
//! `plan` is the G1 gate: a semantic representability check over one or
//! more artifacts, run *before* any conversion. It prints the full
//! [`larql_vindex::format::vindex3::plan::SystemPlan`] as JSON and exits
//! non-zero when the plan is inadmissible, so scripts can gate on it.

/// Extension distinguishing a saved inventory JSON from a checkpoint dir.
const INVENTORY_EXT: &str = "json";

pub fn run(cmd: Vindex3Command) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Vindex3Command::Plan(args) => run_plan(args),
        Vindex3Command::Encode(args) => run_encode(args),
        Vindex3Command::Inspect(args) => run_inspect(args),
        Vindex3Command::References(args) => run_references(args),
        Vindex3Command::Verify(args) => run_verify(args),
        Vindex3Command::Ops(args) => run_ops(args),
        Vindex3Command::Exec(args) => run_exec(args),
        Vindex3Command::Represent(args) => run_represent(args),
        Vindex3Command::Observe(args) => observe::run(args),
        Vindex3Command::Sensitivity(args) => sensitivity::run(args),
        Vindex3Command::Consequence(args) => consequence::run(args),
        Vindex3Command::TokenBank(args) => token_bank::run(args),
        Vindex3Command::Measure(args) => measure::run(args),
        Vindex3Command::AutoRep(args) => auto_rep::run(args),
    }
}

mod artifact;
pub(crate) mod auto_rep;
mod bank;
mod consequence;
pub(crate) mod decode;
mod exec;
mod generate;
mod input_moments;
mod intervention;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub(crate) mod lowered;
pub(crate) mod measure;
mod observe;
mod ops;
mod optional_op;
pub(crate) mod plugins;
pub(crate) mod prepare;
mod realizations;
mod reconstruction;
mod sensitivity;
mod teacher_force;
mod token_bank;
use exec::run_exec;
use ops::run_ops;

mod args;
mod inspect;
mod plan;
mod represent;
mod verify_encode;
pub use args::*;
use inspect::*;
use plan::*;
use represent::*;
use verify_encode::*;

#[cfg(test)]
pub(crate) mod tests;
