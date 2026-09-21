//! **Where the activations that produced a magnitude came from.**
//!
//! A quality bank is a set of numbers measured while *something* was fed
//! through the model. What that something was decides whether the numbers
//! mean anything, and nothing in [`QualityBank`](super::quality::QualityBank)
//! recorded it. Three failures in one programme, all invisible to every
//! other field:
//!
//! | substrate error | direction | size |
//! |---|---|---|
//! | synthetic *weights* instead of trained ones | optimistic | **10.6x** |
//! | synthetic *input scale* — a scale-sensitive operator in the wrong regime | pessimistic | **2.5x** |
//! | uniform-random vocabulary ids instead of natural text | optimistic | **1.2-1.3x** |
//!
//! **A synthetic substrate does not err in a predictable direction**, so
//! "synthetic is conservative" is not available as a defence. The two
//! larger errors also reversed a *conclusion*, not merely a number: the
//! first made a representation look adequate, the second made one look
//! RED that passes on real activations.
//!
//! The rule this module makes structural:
//!
//! > **A synthetic substrate may prove mechanism and fire controls. It may
//! > never establish a magnitude or an admission.**
//!
//! Modelled on [`Provenance::native_kernel`](super::experiment::Provenance),
//! which already gates throughput claims the same way: a boolean fact about
//! how a number was produced, checked before the number is allowed to
//! justify anything.
//!
//! ## Absence is not permission
//!
//! [`ActivationSource`] is optional on the bank so existing records
//! deserialize, but a gate that asks for model activations and finds
//! `None` **fails** — the same contract as
//! [`Criterion::CoveredMass`](super::quality::Criterion::CoveredMass). An
//! unstated substrate is indistinguishable from a synthetic one.

use serde::{Deserialize, Serialize};

/// Which population a bank belongs to.
///
/// Separated because a search may read calibration as often as it likes
/// and must read a holdout **once**. Recording the role on the evidence
/// is what lets a later reader tell a qualification from a selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BankRole {
    /// May be optimised against. Contaminated as a selection surface by
    /// construction.
    Calibration,
    /// May be read once, to qualify a candidate calibration already
    /// chose. A holdout read during search is no longer a holdout.
    Holdout,
}

/// The activations a magnitude was measured over, captured from the
/// model's own execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationAuthority {
    /// Which checkpoint produced them. A magnitude measured on one
    /// checkpoint is not evidence about another, however similar.
    pub checkpoint: String,
    /// **Where in the operator they were captured**, named exactly.
    ///
    /// This field exists because comparing against the wrong boundary is
    /// a silent error: a layer's *input stream* and the operator's own
    /// post-norm input differed by 10x on a real model, and reading the
    /// former as the latter produced a scale claim wrong by an order of
    /// magnitude. "The layer's input" is not a boundary; `post_input_
    /// layernorm_post_mhc` is.
    pub operator_boundary: String,
    /// `None` when the bank spans layers rather than sampling one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    /// The frozen bank's name and digest. The digest is what makes
    /// "qualified on holdout-v1" checkable rather than asserted.
    pub bank: String,
    pub bank_digest: String,
    pub role: BankRole,
}

/// How the activations behind a measurement were obtained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ActivationSource {
    /// Generated vectors. Legitimate for proving a mechanism runs and for
    /// demonstrating that a control fires; never for a magnitude.
    ///
    /// The note is required, not decorative — a reader has to be able to
    /// tell *which* synthetic construction was used, because that is what
    /// decides which regime the operator was in.
    Synthetic { note: String },
    /// Captured from the model executing.
    ModelExecution(ActivationAuthority),
}

impl ActivationSource {
    /// Whether a magnitude or admission may rest on this substrate.
    ///
    /// The whole point of the type: this is the only question callers
    /// ask, and it cannot be answered `true` by a synthetic record.
    pub fn supports_magnitude_claim(&self) -> bool {
        matches!(self, ActivationSource::ModelExecution(_))
    }

    /// The authority, when there is one.
    pub fn authority(&self) -> Option<&ActivationAuthority> {
        match self {
            ActivationSource::ModelExecution(a) => Some(a),
            ActivationSource::Synthetic { .. } => None,
        }
    }

    /// Whether this evidence came from a holdout population.
    pub fn is_holdout(&self) -> bool {
        self.authority()
            .is_some_and(|a| a.role == BankRole::Holdout)
    }

    /// One line for a promotion report.
    pub fn describe(&self) -> String {
        match self {
            ActivationSource::Synthetic { note } => {
                format!("SYNTHETIC ({note}) — may not support a magnitude claim")
            }
            ActivationSource::ModelExecution(a) => format!(
                "model execution: {} @ {}{} · bank {} [{}] {:?}",
                a.checkpoint,
                a.operator_boundary,
                a.layer.map(|l| format!(" L{l}")).unwrap_or_default(),
                a.bank,
                &a.bank_digest[..a.bank_digest.len().min(16)],
                a.role,
            ),
        }
    }
}

/// Why a substrate was refused, for a gate's failure text.
///
/// `Unstated` and `Synthetic` are kept apart because they call for
/// different fixes: one is a record that never said, the other is a
/// record that said something disqualifying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationRefusal {
    Unstated,
    Synthetic(String),
}

impl std::fmt::Display for ActivationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationRefusal::Unstated => write!(
                f,
                "no activation source recorded; an unstated substrate is \
                 indistinguishable from a synthetic one"
            ),
            ActivationRefusal::Synthetic(note) => write!(
                f,
                "synthetic activations ({note}) may prove mechanism and fire \
                 controls, never a magnitude"
            ),
        }
    }
}

/// Judge a bank's substrate for a gate that requires model activations.
pub fn refuse_unless_model_execution(
    source: Option<&ActivationSource>,
) -> Option<ActivationRefusal> {
    match source {
        None => Some(ActivationRefusal::Unstated),
        Some(ActivationSource::Synthetic { note }) => {
            Some(ActivationRefusal::Synthetic(note.clone()))
        }
        Some(ActivationSource::ModelExecution(_)) => None,
    }
}

#[cfg(test)]
#[path = "activation_tests.rs"]
mod tests;
