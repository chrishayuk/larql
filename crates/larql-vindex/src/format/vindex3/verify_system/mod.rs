//! `larql vindex3 verify` — prove source ≡ encoded (V3-G4, spec §7).
//!
//! Four explicit authorities, compared structurally and semantically:
//!
//! | Authority | Source |
//! |---|---|
//! | **Declared** | what HF said (`config.json`, shard headers) |
//! | **Resolved** | what LARQL interpreted (detection + policies) |
//! | **Graph** | the logical system constructed at G2, rebuilt now |
//! | **Encoded** | what the container actually contains |
//!
//! The invariant is `Declared ≡ Resolved ≡ Graph ≡ Encoded` for everything
//! execution-semantic, checked as three pairwise links — a failure names the
//! two authorities that disagree, which is the diagnosis.
//!
//! Two equivalence claims are kept **separate**, because each can fail while
//! the other holds:
//!
//! - **semantic equivalence** — the pairwise authority chain above;
//! - **byte payload equivalence** — per representation, the source payload
//!   hash (recomputed from the checkpoint *now*, in segment order), the
//!   encoded payload-region hash (recomputed from the container *now*) and
//!   the hash the directory recorded at encode time must all agree. Both
//!   ends are re-read, so a checkpoint that drifted after encoding fails
//!   exactly like a container that was corrupted after encoding — with a
//!   detail saying which end moved.
//!
//! "Tensor count before == after" is not verification and does not appear
//! here; the tensor *table* (names, dtypes, shapes, offsets) is compared
//! entry by entry instead, because a relabelled dtype or shape over
//! identical bytes is invisible to every hash in the container.

mod derived;
pub mod payload;
pub mod semantic;

#[cfg(test)]
mod tests;

use std::path::Path;

use larql_models::inventory::ArchitectureInventory;
use serde::Serialize;

use super::inspect::inspect_container;
use super::plan::plan_system;
use crate::error::VindexError;

/// One of the four authorities of spec §7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Authority {
    Declared,
    Resolved,
    Graph,
    Encoded,
}

impl std::fmt::Display for Authority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Declared => "declared",
            Self::Resolved => "resolved",
            Self::Graph => "graph",
            Self::Encoded => "encoded",
        };
        write!(f, "{name}")
    }
}

/// One pairwise equivalence fact between two adjacent authorities.
#[derive(Debug, Clone, Serialize)]
pub struct EquivalenceCheck {
    pub left: Authority,
    pub right: Authority,
    /// What is being compared (`target.attention_policy`, `object
    /// target.embedding`, …).
    pub subject: String,
    /// The value each authority holds, for a failure to be actionable
    /// without re-running anything.
    pub left_value: serde_json::Value,
    pub right_value: serde_json::Value,
    pub pass: bool,
    pub detail: String,
}

/// Byte payload equivalence of one representation (spec §7's second claim).
#[derive(Debug, Clone, Serialize)]
pub struct PayloadCheck {
    /// Representation id (`object@encoding`).
    pub representation: String,
    pub payload_bytes: u64,
    /// SHA-256 over the source payloads, recomputed now, in segment order.
    pub source_sha256: String,
    /// SHA-256 over the container's payload region, recomputed now.
    pub encoded_sha256: String,
    /// The hash the directory recorded at encode time.
    pub recorded_sha256: String,
    pub pass: bool,
    pub detail: String,
}

/// The full G4 verdict.
#[derive(Debug, Serialize)]
pub struct SystemVerification {
    /// The G3 structural gate still holds (graph ↔ directory ↔ segments).
    pub container_coherent: bool,
    pub container_defects: Vec<String>,
    /// The pairwise authority chain, every link itemised.
    pub semantic: Vec<EquivalenceCheck>,
    /// Byte equivalence per representation.
    pub payloads: Vec<PayloadCheck>,
    /// True iff the container is coherent and every check above passes.
    pub verified: bool,
}

impl SystemVerification {
    /// Failing semantic checks, in report order.
    pub fn semantic_failures(&self) -> impl Iterator<Item = &EquivalenceCheck> {
        self.semantic.iter().filter(|c| !c.pass)
    }

    /// Failing payload checks, in report order.
    pub fn payload_failures(&self) -> impl Iterator<Item = &PayloadCheck> {
        self.payloads.iter().filter(|p| !p.pass)
    }
}

/// The registered buffers the inventory excluded, re-checked against the
/// declared spans from today's source — the claim that let them leave
/// the operand estate. `None` when no artifact excludes anything.
fn excluded_buffer_check(
    named: &[(String, ArchitectureInventory)],
    graph: &crate::format::vindex3::graph::SystemGraph,
) -> Result<Option<EquivalenceCheck>, VindexError> {
    use crate::format::vindex3::encode::source::{ArtifactSource, TensorSource};
    let excluded: usize = named.iter().map(|(_, i)| i.tensors.excluded.len()).sum();
    if excluded == 0 {
        return Ok(None);
    }
    let opened: Vec<(&str, ArtifactSource)> = named
        .iter()
        .map(|(n, i)| ArtifactSource::open(Path::new(&i.path)).map(|s| (n.as_str(), s)))
        .collect::<Result<_, _>>()?;
    let sources: std::collections::BTreeMap<&str, &dyn TensorSource> = opened
        .iter()
        .map(|(n, s)| (*n, s as &dyn TensorSource))
        .collect();
    let outcome =
        crate::format::vindex3::encode::masks::check_excluded_buffers(graph, named, &sources);
    Ok(Some(EquivalenceCheck {
        left: Authority::Declared,
        right: Authority::Graph,
        subject: "excluded registered buffers".to_string(),
        left_value: serde_json::json!(excluded),
        right_value: serde_json::json!(outcome.as_ref().ok()),
        pass: outcome.is_ok(),
        detail: match outcome {
            Ok(n) => {
                format!("{n} stored mask buffer(s) equal the declared spans they are excluded for")
            }
            Err(e) => e.to_string(),
        },
    }))
}

/// Verify a container against its source artifacts: the G4 gate.
///
/// `named` must be the same artifact set the container was encoded from
/// (same names — they are the graph's component/binding identity). Hard
/// errors are reserved for an unreadable container or source; every
/// disagreement is a *finding* in the returned report, never an early
/// return, so one run itemises everything that moved.
pub fn verify_system(
    named: &[(String, ArchitectureInventory)],
    container: &Path,
) -> Result<SystemVerification, VindexError> {
    let plan = plan_system(named);
    let inspection = inspect_container(container, false)?;

    let mut checks = semantic::declared_vs_resolved(&plan);
    checks.extend(semantic::resolved_vs_graph(named, &plan.graph));
    checks.extend(semantic::graph_vs_encoded(&plan.graph, &inspection.graph));

    let (payloads, table_checks) =
        payload::verify_payloads(named, &plan.graph, &inspection.index, container)?;
    checks.extend(table_checks);
    if let Some(check) = excluded_buffer_check(named, &plan.graph)? {
        checks.push(check);
    }

    let container_defects: Vec<String> = inspection
        .defects
        .iter()
        .map(|d| format!("{d:?}"))
        .collect();
    let verified = container_defects.is_empty()
        && checks.iter().all(|c| c.pass)
        && payloads.iter().all(|p| p.pass);

    Ok(SystemVerification {
        container_coherent: container_defects.is_empty(),
        container_defects,
        semantic: checks,
        payloads,
        verified,
    })
}
