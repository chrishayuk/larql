//! **Promotion: evidence to precision map.**
//!
//! REPRESENT's job is not to pick a format label for a model. It is to
//! compile each semantic object to the smallest representation that
//! preserves its behaviour — which is an optimisation over a precision
//! MAP, keyed by semantic role, not a model-wide setting.
//!
//! This module is the one step that turns measurements into authority.
//! It is deliberately the narrowest surface in the crate, because it is
//! where a benchmark result could otherwise become a deployment
//! decision without anyone deciding:
//!
//! ```text
//! CanonicalOperand(role = ExpertWeight, representation = BF16)
//!     CAN_REPRESENT_AS -> Q6_K   (measured: 2.15x, 88% of roofline)
//!     CAN_REPRESENT_AS -> Q4_K   (measured: 2.62x, 74% of roofline)
//!                              ^ candidates, whatever their numbers
//!     SELECTED_REPRESENTATION -> ???
//!                              ^ only [`promote`] writes this edge
//! ```
//!
//! The rule it enforces is the one that keeps the two apart: **a
//! backend supporting a format is capability, never authority.** A
//! record may be fast, small, natively dispatched and still not
//! promotable, because none of that is evidence about the model's
//! output distribution. Only a quality bank is.

use std::collections::BTreeMap;

use super::experiment::{RepresentationExperiment, RoleScope};
use super::map::{Exception, PrecisionMap};
use super::policy::Role;

/// Why a candidate could not be promoted.
///
/// One variant per missing fact rather than a single "insufficient
/// evidence", so a report tells the reader which gate to go and run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// No logit-level bank has been run for this scope.
    QualityUnproven,
    /// A bank was run and the gate REFUSED it. Distinct from
    /// `QualityUnproven`: one says go and measure, the other says this
    /// representation is not good enough for this region — which is the
    /// signal to try a narrower scope, not a bigger bank.
    QualityGateFailed { verdict: String },
    /// Throughput came from a decoded stand-in, not the format's kernel.
    NotNativelyMeasured,
    /// The ladder says something earlier is missing.
    LadderIncomplete,
    /// Two records claim the same scope with different encodings.
    ConflictingCandidates { other: String },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::QualityUnproven => f.write_str(
                "no logit-level quality bank has been run for this scope; component error is \
                 evidence about the component, not about the model's output distribution",
            ),
            Refusal::QualityGateFailed { verdict } => write!(f, "{verdict}"),
            Refusal::NotNativelyMeasured => f.write_str(
                "throughput came from a decoded stand-in rather than this format's own kernel",
            ),
            Refusal::LadderIncomplete => {
                f.write_str("the representation is not yet represented/available/runnable/measured")
            }
            Refusal::ConflictingCandidates { other } => write!(
                f,
                "another candidate ({other}) claims the same scope; a scope has one selected \
                 representation, so the tie must be broken by policy before promotion"
            ),
        }
    }
}

/// One candidate's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub scope: RoleScope,
    pub target: String,
    pub outcome: Result<(), Refusal>,
}

/// The result of considering a set of candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Promotion {
    /// The map the promotable evidence justifies. Roles with no
    /// promotable candidate are simply absent, which `PrecisionMap`
    /// already reads as source precision.
    pub map: PrecisionMap,
    /// Every candidate considered, promoted or refused, in input order.
    pub verdicts: Vec<Verdict>,
}

impl Promotion {
    pub fn promoted(&self) -> usize {
        self.verdicts.iter().filter(|v| v.outcome.is_ok()).count()
    }

    /// A line per candidate, for a report or a commit message.
    pub fn describe(&self) -> String {
        let mut out = vec![format!(
            "{}: {} of {} candidates promoted",
            self.map.name,
            self.promoted(),
            self.verdicts.len()
        )];
        for v in &self.verdicts {
            out.push(match &v.outcome {
                Ok(()) => format!("  SELECTED  {} = {}", v.scope.role, v.target),
                Err(r) => format!("  candidate {} = {} — {r}", v.scope.role, v.target),
            });
        }
        out.join("\n")
    }
}

/// Build the precision map that `candidates` justify.
///
/// Never fails: unpromotable candidates become refusals in the report
/// and are simply absent from the map, so the fail-safe direction is
/// source precision. A caller wanting a hard error asserts on
/// [`Promotion::promoted`].
pub fn promote(
    name: impl Into<String>,
    default_encoding: impl Into<String>,
    candidates: &[RepresentationExperiment],
) -> Promotion {
    let default_encoding = default_encoding.into();

    // Per-candidate evidence FIRST. Several candidates for one scope is
    // the normal state — `CAN_REPRESENT_AS` is a many edge, and calling
    // that a conflict would report every screen as a tie and hide the
    // reason none of them was eligible in the first place.
    let eligible: Vec<Option<Refusal>> = candidates
        .iter()
        .map(|c| match c.quality.as_ref() {
            None => Some(Refusal::QualityUnproven),
            Some(q) if !q.verdict().passed() => Some(Refusal::QualityGateFailed {
                verdict: q.verdict().describe(),
            }),
            Some(_) => {
                if !c.supports_throughput_claim() {
                    Some(Refusal::NotNativelyMeasured)
                } else if !c.status.ladder_complete() {
                    Some(Refusal::LadderIncomplete)
                } else {
                    None
                }
            }
        })
        .collect();

    // A tie is only a tie between candidates that would otherwise be
    // selected, and it refuses BOTH rather than letting input order pick.
    let mut winners: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (c, e) in candidates.iter().zip(&eligible) {
        if e.is_none() {
            winners
                .entry(format!("{:?}", c.scope))
                .or_default()
                .push(&c.target);
        }
    }

    let mut verdicts = Vec::with_capacity(candidates.len());
    let mut roles: Vec<String> = Vec::new();
    let mut exceptions: Vec<Exception> = Vec::new();
    for (c, refusal) in candidates.iter().zip(&eligible) {
        let rival = winners
            .get(&format!("{:?}", c.scope))
            .and_then(|w| w.iter().find(|t| **t != c.target))
            .map(|t| (*t).to_string());
        let outcome = match (refusal, rival) {
            (Some(r), _) => Err(r.clone()),
            (None, Some(other)) => Err(Refusal::ConflictingCandidates { other }),
            (None, None) => {
                let role = c.scope.role.name().to_string();
                if !roles.contains(&role) {
                    roles.push(role);
                }
                // An exception is only needed where the candidate's
                // encoding differs from the map's default, or where the
                // scope is narrower than the whole role.
                if c.target != default_encoding
                    || c.scope.projection.is_some()
                    || c.scope.layers.is_some()
                {
                    exceptions.push(c.scope.as_exception(&c.target));
                }
                Ok(())
            }
        };
        verdicts.push(Verdict {
            scope: c.scope.clone(),
            target: c.target.clone(),
            outcome,
        });
    }

    Promotion {
        map: PrecisionMap {
            name: name.into(),
            encoding: default_encoding,
            roles,
            exceptions,
        },
        verdicts,
    }
}

/// Roles a map leaves at source precision, of the ones asked about.
///
/// The complement of the map is as much a decision as the map: a caller
/// deploying one wants to see "Router, Norm and KDAGate stay BF16"
/// written down, not inferred from an absence.
pub fn unselected(map: &PrecisionMap, considered: &[Role]) -> Vec<Role> {
    considered
        .iter()
        .copied()
        .filter(|r| !map.roles.iter().any(|m| m == r.name()))
        .collect()
}

#[cfg(test)]
mod tests;
