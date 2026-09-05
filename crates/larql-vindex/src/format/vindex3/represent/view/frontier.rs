//! **A state's standing, rendered.**
//!
//! [`Adjudication`] and [`FrontierEntry`] deliberately do not derive
//! `Serialize`: they are DERIVED verdicts, and stage 1d's whole point
//! is that a derived verdict is never stored. Rendering one is a
//! different act from storing one — the agent is being shown a
//! conclusion the optimiser reached just now, from facts, on demand —
//! but it is the act with the most room to lie, so every field below
//! names the call that produced it and [`super::origin`] checks that
//! the list is complete.

use serde::Serialize;

use super::super::constraint::{ConstraintVector, Margin};
use super::super::measurement::EvidenceScale;
use super::super::state::realization::LogicalBytes;
use super::super::state::snapshot::{Adjudication, FrontierEntry};
use super::super::state::{MeasurementKey, RepresentationStateId};
use super::origin::{Origin, Rendered};

/// One observation, and what the frozen contract makes of it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdjudicationView {
    /// Which experiment this is a verdict on.
    pub key: MeasurementKey,
    /// Every criterion the gate judges, with what the bank spent.
    pub constraints: ConstraintVector,
    /// Whether every criterion — ceiling and floor — is met.
    pub admissible: bool,
    /// Whether the instrument saw: every floor cleared. A blind
    /// instrument passes every ceiling perfectly, which is why this is
    /// reported beside `admissible` and never folded into it.
    pub sound: bool,
    /// The scarce resource — the ceiling closest to its limit.
    pub binding: Option<Margin>,
    /// Every criterion this observation failed, in the gate's order.
    pub failures: Vec<Margin>,
}

impl AdjudicationView {
    pub fn of(adjudication: &Adjudication) -> Self {
        Self {
            key: adjudication.key().clone(),
            constraints: adjudication.constraints().clone(),
            admissible: adjudication.admissible(),
            sound: adjudication.sound(),
            binding: adjudication.binding().cloned(),
            failures: adjudication.failures().into_iter().cloned().collect(),
        }
    }

    /// Every field, and the call behind it.
    pub fn origins() -> Vec<Origin> {
        vec![
            Origin::new("key", "Adjudication::key()"),
            Origin::new("constraints", "Adjudication::constraints()"),
            Origin::new("admissible", "Adjudication::admissible()"),
            Origin::new("sound", "Adjudication::sound()"),
            Origin::new("binding", "Adjudication::binding()"),
            Origin::new("failures", "Adjudication::failures()"),
        ]
    }
}

/// One state's whole standing: what it costs, what has been observed of
/// it, and what the contract says about each observation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StateStanding {
    pub state: RepresentationStateId,
    /// Whole-map footprint. NOT the per-token read — see
    /// [`super::super::byte_ledger`] for why the two must not be one
    /// number.
    pub logical_bytes: LogicalBytes,
    /// Admitted only on an AUTHORITY reading. A diagnostic pass is not
    /// an admission.
    pub admitted: bool,
    /// Refused where an authority reading failed the contract. Not the
    /// negation of `admitted`: an unmeasured state is neither.
    pub refused: bool,
    /// The scales this state carries a reading at, cheapest first.
    pub measured_at: Vec<EvidenceScale>,
    pub adjudications: Vec<AdjudicationView>,
}

impl StateStanding {
    pub fn of(entry: &FrontierEntry) -> Self {
        Self {
            state: entry.state.clone(),
            logical_bytes: entry.logical_bytes,
            admitted: entry.admitted(),
            refused: entry.refused(),
            measured_at: EvidenceScale::ALL
                .into_iter()
                .filter(|scale| entry.measured_at(*scale))
                .collect(),
            adjudications: entry
                .adjudications
                .iter()
                .map(AdjudicationView::of)
                .collect(),
        }
    }

    pub fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new("state", "FrontierEntry.state"),
            Origin::new("logical_bytes", "FrontierEntry.logical_bytes"),
            Origin::new("admitted", "FrontierEntry::admitted()"),
            Origin::new("refused", "FrontierEntry::refused()"),
            Origin::new("measured_at", "FrontierEntry::measured_at(scale)"),
        ];
        origins.extend(
            AdjudicationView::origins()
                .iter()
                .map(|o| o.under("adjudications[]")),
        );
        origins
    }
}

/// **The frontier, recomputed** — every state the graph holds, in
/// state-id order, with the objective's own ordering of the admitted
/// ones beside it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Frontier {
    pub states: Vec<StateStanding>,
    /// Admitted states, CHEAPEST FIRST — the objective, applied. The
    /// order is the optimiser's, not this view's.
    pub admitted: Vec<StateStanding>,
}

impl Rendered for Frontier {
    fn origins() -> Vec<Origin> {
        StateStanding::origins()
            .iter()
            .map(|o| o.under("states[]"))
            .chain(
                StateStanding::origins()
                    .iter()
                    .map(|o| o.under("admitted[]")),
            )
            .collect()
    }
}
