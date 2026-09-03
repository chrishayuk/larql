//! **Admission and robustness are two independent state machines.**
//!
//! Rung 5's T1 candidate failed its selection bank at `kl_p99 3.6480e-3`
//! against a `3.5e-3` threshold. That settled ADMISSION permanently. It
//! did not settle whether the refusal was DECISIVE: the margin was
//! `-1.4797e-4` against a measured cross-bank disagreement guard of
//! `1.7494e-4`, so one bank could not classify itself.
//!
//! The held-out bank returned `3.7821e-3` — a determinate refusal at
//! `-1.61` bands — and the two banks agreed to within `1.3408e-4`, inside
//! the guard. **T1 looked boundary-adjacent and was not.** The selection
//! bank was simply the more favourable slice (R5-F10).
//!
//! ```text
//! ADMISSION                      ROBUSTNESS
//! any measured bank FAILS        the WORST bank's position relative to
//!   -> Refused, FINAL              the guard, and whether banks AGREE
//! a held-out PASS cannot         classifies DECISIVENESS, never
//!   resurrect it                   admissibility
//! ```
//!
//! > Stop on selection FAIL for ADMISSION. Continue within the guard when
//! > replication is required to classify DECISIVENESS.
//!
//! # Why they must not be one value
//!
//! Collapsing them invites an unregistered escalation contract — bank 3
//! passes narrowly, so bank 4? majority? worst-of-three? — invented after
//! a result exists. K25 was admitted on a contract it satisfied (both
//! banks, no failures) while sitting `+0.08` bands inside the threshold.
//! Its admission is not in doubt; its robustness is BoundaryAdjacent. Two
//! facts, two fields.
//!
//! # What a search may do with each
//!
//! A tree search may treat [`Robustness`] as exploration priority — a
//! single-bank near-boundary reading is low-confidence. It may NOT treat
//! it as a probability of PASS: R5-F10 showed the SIGN of the resolution
//! is not predictable from the first bank. K25 resolved upward to a
//! replicated pass; T1 resolved downward to a determinate refusal, from
//! the same `INDETERMINATE` starting classification.
//!
//! [`AuthorityState`] is the contract. Nothing in a search policy may
//! write to it.
//!
//! # Why this lives in `state/` and stays TWO enums
//!
//! The rest of the search substrate is here, so a contract standing that
//! lived in `represent/` would be a second place to ask what a state's
//! status is. But being given somewhere convenient to put them is not a
//! reason to merge them into one richer status enum: admission is a
//! CONTRACT and robustness is a property of the EVIDENCE, and a single
//! value would let a search policy's confidence and a gate's verdict be
//! read off each other. They are computed from the same bank outcomes by
//! two functions that do not consult one another.

use std::fmt;

use serde::{Deserialize, Serialize};

/// One bank's verdict on one map.
///
/// `passed_contract` is the gate's own answer, not a comparison this
/// module performs: a contract may fail a map on a statistic that is not
/// `binding_value` at all, and re-deriving it here would silently narrow
/// the contract to one dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BankOutcome {
    /// Which evidence bank produced this.
    pub bank: String,
    /// The contract's verdict, as the gate reported it.
    pub passed_contract: bool,
    /// The binding statistic's value, for band classification.
    pub binding_value: f64,
}

impl BankOutcome {
    /// Record one bank's verdict.
    pub fn new(bank: impl Into<String>, passed_contract: bool, binding_value: f64) -> Self {
        Self {
            bank: bank.into(),
            passed_contract,
            binding_value,
        }
    }
}

/// The measured cross-bank disagreement guard.
///
/// Not an invented safety factor: it is the maximum selection-versus-
/// held-out disagreement this programme has observed on the SAME map.
/// A margin inside it is a slice difference, not a signal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DisagreementGuard {
    /// Threshold the binding statistic is judged against.
    pub threshold: f64,
    /// Half-width of the guard, in the statistic's own units.
    pub band: f64,
}

impl DisagreementGuard {
    /// Build a guard. `band` must be positive and finite, or the
    /// classification below is meaningless.
    pub fn new(threshold: f64, band: f64) -> Option<Self> {
        (band > 0.0 && band.is_finite() && threshold.is_finite())
            .then_some(Self { threshold, band })
    }

    /// Where one reading sits relative to the guard.
    pub fn position(&self, value: f64) -> BandPosition {
        let margin = self.threshold - value;
        if margin > self.band {
            BandPosition::ClearOfThreshold
        } else if margin < -self.band {
            BandPosition::BeyondThreshold
        } else {
            BandPosition::Indeterminate
        }
    }

    /// Margin in bands: positive is inside the threshold.
    pub fn bands(&self, value: f64) -> f64 {
        (self.threshold - value) / self.band
    }
}

/// One reading's position relative to the guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BandPosition {
    /// Inside the threshold by more than one band.
    ClearOfThreshold,
    /// Within one band of the threshold, either side. One bank cannot
    /// classify itself here.
    Indeterminate,
    /// Outside the threshold by more than one band.
    BeyondThreshold,
}

/// **The contract.** What a map is permitted to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityState {
    /// Every required bank returned a passing contract verdict.
    Admitted,
    /// A measured bank failed the contract. Final — a later passing bank
    /// cannot resurrect it.
    Refused,
    /// Passing so far, but fewer banks than the contract requires.
    Pending,
}

impl fmt::Display for AuthorityState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Admitted => "Admitted",
            Self::Refused => "Refused",
            Self::Pending => "Pending",
        })
    }
}

impl AuthorityState {
    /// Derive admission from bank verdicts alone.
    ///
    /// `banks_required` is stated by the caller because it is a property
    /// of the contract, not of this code — reading it from the number of
    /// banks that happen to have run is how an unregistered escalation
    /// contract gets invented.
    pub fn of(outcomes: &[BankOutcome], banks_required: usize) -> Self {
        if outcomes.iter().any(|o| !o.passed_contract) {
            return Self::Refused;
        }
        if outcomes.len() >= banks_required && !outcomes.is_empty() {
            Self::Admitted
        } else {
            Self::Pending
        }
    }

    /// Whether this state can still change with more evidence.
    ///
    /// `Refused` cannot: that is the asymmetry.
    pub fn is_final(&self) -> bool {
        matches!(self, Self::Admitted | Self::Refused)
    }
}

/// **Search policy only.** How decisive the evidence is.
///
/// Never consulted by the contract, and never writable from one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Robustness {
    /// One bank only. Cannot classify decisiveness at all.
    Unreplicated,
    /// Replicated pass, worst bank clear of the threshold.
    ReplicatedInterior,
    /// Replicated, but the worst bank sits inside the guard. K25's state.
    BoundaryAdjacent,
    /// Replicated refusal, worst bank determinate. T1's state.
    ReplicatedRefusal,
    /// Banks fall on opposite sides of the threshold.
    CrossBankDisagreement,
}

impl fmt::Display for Robustness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unreplicated => "Unreplicated",
            Self::ReplicatedInterior => "ReplicatedInterior",
            Self::BoundaryAdjacent => "BoundaryAdjacent",
            Self::ReplicatedRefusal => "ReplicatedRefusal",
            Self::CrossBankDisagreement => "CrossBankDisagreement",
        })
    }
}

impl Robustness {
    /// Classify decisiveness from the readings and the guard.
    ///
    /// Disagreement is judged on the CONTRACT VERDICTS, not on the
    /// binding value straddling the threshold: a map can fail a contract
    /// on a statistic this guard does not describe, and calling that
    /// agreement because two kl values sit on one side would report a
    /// stability the evidence does not have.
    pub fn of(outcomes: &[BankOutcome], guard: DisagreementGuard) -> Self {
        if outcomes.len() < 2 {
            return Self::Unreplicated;
        }
        let passed = outcomes.iter().filter(|o| o.passed_contract).count();
        if passed != 0 && passed != outcomes.len() {
            return Self::CrossBankDisagreement;
        }
        // All banks agree on the verdict. The WORST reading decides how
        // decisively — worst meaning largest, since the guard is an upper
        // threshold on a cost.
        let worst = outcomes
            .iter()
            .map(|o| o.binding_value)
            .fold(f64::NEG_INFINITY, f64::max);
        match (passed == outcomes.len(), guard.position(worst)) {
            (true, BandPosition::ClearOfThreshold) => Self::ReplicatedInterior,
            (true, _) => Self::BoundaryAdjacent,
            (false, BandPosition::BeyondThreshold) => Self::ReplicatedRefusal,
            (false, _) => Self::BoundaryAdjacent,
        }
    }

    /// Whether a search should treat this map's position as low
    /// confidence when ordering exploration.
    ///
    /// **Not a probability of PASS.** R5-F10: the sign of the resolution
    /// is not predictable from the first bank.
    pub fn is_low_confidence(&self) -> bool {
        matches!(
            self,
            Self::Unreplicated | Self::BoundaryAdjacent | Self::CrossBankDisagreement
        )
    }
}

/// A map's complete authority record: the contract's answer and the
/// evidence's decisiveness, side by side and never merged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapOutcome {
    /// The contract's answer.
    pub authority: AuthorityState,
    /// How decisive the evidence is. Search policy only.
    pub robustness: Robustness,
    /// The bank verdicts this was derived from.
    pub outcomes: Vec<BankOutcome>,
}

impl MapOutcome {
    /// Derive both state machines from the same evidence, independently.
    pub fn of(outcomes: Vec<BankOutcome>, guard: DisagreementGuard, banks_required: usize) -> Self {
        Self {
            authority: AuthorityState::of(&outcomes, banks_required),
            robustness: Robustness::of(&outcomes, guard),
            outcomes,
        }
    }

    /// The worst binding value across banks, if any bank ran.
    pub fn worst_binding(&self) -> Option<f64> {
        self.outcomes
            .iter()
            .map(|o| o.binding_value)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
    }
}

#[cfg(test)]
#[path = "authority_tests.rs"]
mod authority_tests;
