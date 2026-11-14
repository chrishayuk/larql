//! Semantic representability plan over architecture inventories (V3-G1/G2).
//!
//! Consumes the G0 inventory (`larql inspect-hf`) for one or more artifacts
//! treated as a model system, and answers: **can the VINDEX3 schema
//! faithfully describe this system — and if not, exactly why not?**
//!
//! Since G2, "representable" has one definition: **the system-graph builder
//! placed it** ([`super::graph::build_from_inventories`]). Objects the
//! builder placed are representable with their graph ids as proof; groups
//! it could not place, and interfaces it could not resolve, come back as
//! blocking findings. There is no separate capability table to drift out of
//! sync with the schema.
//!
//! The other finding sources:
//!
//! - `mismatched` — declared-vs-resolved value comparison (`consumed` is
//!   never trusted; values are compared);
//! - **every** declared config key, graded by semantic class, where a key
//!   nobody has judged (`unknown`) blocks. The census covers consumed,
//!   metadata and unconsumed keys alike, so `unrepresented: N` is a count
//!   against a stated denominator rather than a lower bound.
//! - **carriage** — for execution-semantic keys, how far VINDEX3 actually
//!   carries the fact past the parser ([`carriage`]). This is the axis
//!   that keeps `consumed` from being misread as `represented`: a key the
//!   parser reads and the schema then drops used to produce no finding at
//!   all, which is how GPT-OSS's YaRN scaling would have executed as
//!   plain rope with the plan reporting nothing.
//!
//! The verdict is fail-closed and the exit gate is mechanical:
//! `blocking == 0` before a single weight byte is converted.

pub mod admission;
mod attention_policy;
pub mod capability;
pub mod carriage;
pub mod compare;
pub mod report;
pub mod rope_scope;
pub mod semantics;

#[cfg(test)]
mod tests;
/// Glimmer-shaped inventory fixtures, shared with the graph tests.
#[cfg(test)]
pub mod tests_support;

pub use report::{
    ArtifactPlan, ArtifactSource, Finding, FindingCategory, FindingId, InterfacePlan, PlanSummary,
    PlannedFinding, PlannerIdentity, SemanticClass, SystemPlan, VerdictCacheKey,
    PLANNER_SEMANTICS_VERSION, PLAN_SCHEMA,
};

mod carriage_findings;
mod planning;
mod surface_findings;
use carriage_findings::*;
pub use planning::*;
pub use surface_findings::*;
