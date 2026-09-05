//! **What has been observed, raw.**
//!
//! Every observation is rendered as the bank the instrument actually
//! recorded, beside the verdict the frozen contract draws from it. Both,
//! and in that order: a derived verdict read without the arms behind it
//! inherits every assumption of the classifier that produced it, and the
//! raw bank is the only place a reader can see that a p99 stood on
//! forty-six observations.
//!
//! What the programme has learned about its own instruments travels with
//! them — a calibration is a finding ABOUT an instrument, and an
//! observation read without it can be read as more than it is.

use serde::Serialize;

use super::super::diagnostic::DiagnosticPolicy;
use super::super::measurement::TailSupportPolicy;
use super::super::quality::QualityBank;
use super::super::search_evidence::SearchCalibrationRegistry;
use super::super::state::snapshot::SearchSnapshot;
use super::super::state::{MeasurementKey, RepresentationStateId};
use super::frontier::AdjudicationView;
use super::origin::{Origin, Rendered};

/// One experiment: what ran, what it saw, and what the contract makes
/// of it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Observation {
    /// State, bank, scale and instrument — what makes two observations
    /// the same observation.
    pub key: MeasurementKey,
    /// The reading itself, whole.
    pub bank: QualityBank,
    pub adjudication: AdjudicationView,
}

impl Observation {
    pub fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new("key", "MeasurementRegistry::keys()"),
            Origin::new("bank", "MeasurementRegistry::get()"),
        ];
        origins.extend(
            AdjudicationView::origins()
                .iter()
                .map(|o| o.under("adjudication")),
        );
        origins
    }
}

/// The measurement record, and the rules for reading it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvidenceReport {
    /// The state this report was narrowed to, or `None` for the whole
    /// record.
    pub of_state: Option<RepresentationStateId>,
    pub observations: Vec<Observation>,
    /// Every registered finding about an instrument — what a statistic
    /// may be used FOR, which is not the same question as what it says.
    pub calibrations: SearchCalibrationRegistry,
    /// Which statistics a diagnostic reads, and for what purpose.
    pub diagnostic_policy: DiagnosticPolicy,
    /// When a percentile is too thin to be the quantity it is named
    /// after.
    pub tail_support: TailSupportPolicy,
}

impl EvidenceReport {
    /// The whole record, or one state's part of it.
    pub fn of(snapshot: &SearchSnapshot, state: Option<&RepresentationStateId>) -> Self {
        let registry = snapshot.measurements();
        let keys: Vec<MeasurementKey> = match state {
            Some(state) => registry.of_state(state).map(|(k, _)| k.clone()).collect(),
            None => registry.keys().cloned().collect(),
        };
        Self {
            of_state: state.cloned(),
            observations: keys
                .into_iter()
                .filter_map(|key| {
                    let bank = registry.get(&key)?.clone();
                    let adjudication = AdjudicationView::of(&snapshot.adjudicate(&key)?);
                    Some(Observation {
                        key,
                        bank,
                        adjudication,
                    })
                })
                .collect(),
            calibrations: snapshot.config().calibrations.clone(),
            diagnostic_policy: snapshot.config().diagnostic_policy.clone(),
            tail_support: snapshot.tail_support().clone(),
        }
    }
}

impl Rendered for EvidenceReport {
    fn origins() -> Vec<Origin> {
        let mut origins = vec![
            Origin::new("of_state", "the state the caller asked about"),
            Origin::new("calibrations", "SearchConfig.calibrations"),
            Origin::new("diagnostic_policy", "SearchConfig.diagnostic_policy"),
            Origin::new("tail_support", "SearchSnapshot::tail_support()"),
        ];
        origins.extend(
            Observation::origins()
                .iter()
                .map(|o| o.under("observations[]")),
        );
        origins
    }
}
