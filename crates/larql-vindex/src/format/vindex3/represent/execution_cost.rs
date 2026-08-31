//! **What a byte reduction actually bought, and where that was measured.**
//!
//! [`super::byte_ledger`] says how many bytes a map removes. This says
//! what removing them did to decode time — and that is not a property
//! of the map. It is a property of a machine, a backend, a build, a
//! baseline and a BREADTH, and the only honest way to hold it is as an
//! observation with provenance rather than as a constant.
//!
//! Three things are deliberately kept apart:
//!
//! ```text
//! physical fact         bytes/token for a representation map
//! observation           this byte reduction produced this GPU-time
//!                       reduction, on this machine, at this breadth
//! planner coefficient   beta = gpu fraction removed / byte fraction
//!                       removed — DERIVED from an observation, never
//!                       an input
//! ```
//!
//! Only the first is intrinsic. Writing `const BETA: f64 = 0.80` would
//! promote one measurement to a law of the backend, and this programme
//! has already been taught what that costs: a `~45 GB` wired-memory
//! wall inferred from a single episode was contradicted at 79 GB the
//! next day, and a scalar reading of the BEHAVIOURAL budget hid the
//! fact that route movement binds at 83 % while KL sits at 68 %.
//!
//! **Beta is currently measured at exactly one breadth.** The model
//! therefore reports [`CalibrationStatus::Provisional`] and every
//! prediction carries that status plus the id of the observation it
//! came from. That is the whole point: a search may use it, and may not
//! forget where it came from.
//!
//! **Deliberately not modelled yet.** No regression, no confidence
//! interval, no piecewise curve. With one observation those would be
//! decoration over a single point. The sophistication here is in
//! preserving provenance, not in the arithmetic — which is
//! `gpu_removed ≈ beta × bytes_removed` and nothing more. When there
//! are five or ten observations across breadths and codecs,
//! [`ExecutionCostModel`] can learn a curve without the search API
//! changing.

use serde::{Deserialize, Serialize};

use super::byte_ledger::ByteLedger;

/// How much a prediction from this model can be relied on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationStatus {
    /// Beta has not been shown to hold across breadths — either there
    /// is one observation, or every observation sits at effectively the
    /// same byte fraction, so their agreement says nothing about
    /// whether beta varies.
    Provisional,
    /// Beta has been measured at separated breadths.
    Calibrated,
}

/// Two observations count as separate operating points only when their
/// byte fractions differ by more than this.
///
/// Five percentage points. Below that, two measurements are probing the
/// same point on any curve beta might have, and agreement between them
/// is not evidence that beta is flat — which is the specific claim
/// [`CalibrationStatus::Calibrated`] makes.
const DISTINCT_BREADTH_MIN_SEPARATION: f64 = 0.05;

/// **One measurement of what a byte reduction bought.**
///
/// Every field is either provenance or a measured quantity. The derived
/// quantities — byte fraction removed, GPU fraction removed, beta — are
/// METHODS rather than fields, so a stored record cannot disagree with
/// its own inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionCostObservation {
    /// Stable id a prediction can name, e.g. `"m3max-metal-001"`.
    pub id: String,

    // ── Where ──
    pub machine: String,
    pub device: String,
    pub backend: String,
    /// Commit the measured binary was built from. A kernel change moves
    /// beta, and without this a later reader cannot tell whether an
    /// observation predates one.
    pub compiler_commit: String,

    // ── What ──
    pub model_identity: String,
    pub baseline_representation: String,
    pub candidate_representation: String,

    // ── Breadth, so beta(breadth) is an askable question ──
    /// Families with at least one changed scope.
    pub families_changed: Vec<String>,
    /// Individual scopes that moved.
    pub scopes_changed: u32,

    // ── Physical ──
    pub baseline_bytes_per_token: u64,
    pub candidate_bytes_per_token: u64,

    // ── Measured ──
    pub baseline_gpu_ms_per_token: f64,
    pub candidate_gpu_ms_per_token: f64,
    /// Wall time per token that is NOT GPU time, at the measured floor.
    ///
    /// Recorded because a later model will need it to separate fixed
    /// cost from variable, and because tokens/second is a wall figure:
    /// a GPU-time prediction alone cannot produce one.
    ///
    /// **One number for both arms, which ASSUMES the overhead does not
    /// depend on the map.** In the measurement behind
    /// [`m3max_metal_001`] the two arms' floors carried 1.05 ms and
    /// 1.09 ms, so the assumption is worth about 0.1 % on a predicted
    /// wall speedup. Held here rather than hidden: if a backend ever
    /// makes overhead depend on the representation, this field is the
    /// one that has to become two.
    pub fixed_overhead_ms: f64,

    // ── How ──
    pub benchmark_protocol: String,
    /// Files a reader can go and check.
    pub evidence: Vec<String>,
}

impl ExecutionCostObservation {
    /// Fraction of the baseline per-token read this candidate removed.
    pub fn byte_fraction_removed(&self) -> f64 {
        if self.baseline_bytes_per_token == 0 {
            return 0.0;
        }
        self.baseline_bytes_per_token
            .saturating_sub(self.candidate_bytes_per_token) as f64
            / self.baseline_bytes_per_token as f64
    }

    /// Fraction of baseline GPU time per token this candidate removed.
    pub fn gpu_fraction_removed(&self) -> f64 {
        if self.baseline_gpu_ms_per_token <= 0.0 {
            return 0.0;
        }
        (self.baseline_gpu_ms_per_token - self.candidate_gpu_ms_per_token)
            / self.baseline_gpu_ms_per_token
    }

    /// **Beta — how much of a removed byte becomes removed GPU time.**
    ///
    /// `None` when nothing was removed, because beta is then `0/0`. A
    /// pure-bandwidth-bound decode would give `1.0`; anything less is
    /// the share of the step that is not bandwidth.
    pub fn bytes_to_gpu_factor(&self) -> Option<f64> {
        let bytes = self.byte_fraction_removed();
        if bytes <= 0.0 {
            return None;
        }
        Some(self.gpu_fraction_removed() / bytes)
    }
}

/// Whether a prediction sits inside the range of breadths that were
/// actually measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Breadth {
    /// The ledger's byte fraction is close to a measured one.
    Measured,
    /// It is not. Beta is being carried somewhere it has never been
    /// checked, which is exactly where a curve would show up.
    Extrapolated {
        /// Byte fraction of the nearest observation.
        nearest_measured_fraction: f64,
    },
}

/// A predicted decode cost, inseparable from the evidence it came from.
///
/// There is no constructor that takes a bare coefficient. A search can
/// only obtain one of these from [`ExecutionCostModel::predict`], which
/// means every predicted throughput in this programme can name the
/// observation behind it and say whether that observation has been
/// replicated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostPrediction {
    pub gpu_ms_per_token: f64,
    pub wall_ms_per_token: f64,
    pub tokens_per_second: f64,
    /// Against the baseline of the observation used.
    pub speedup: f64,
    /// Id of the observation this came from.
    pub calibration_id: String,
    pub status: CalibrationStatus,
    pub breadth: Breadth,
}

impl CostPrediction {
    /// One line a report or a `.represent` record can carry verbatim,
    /// so a predicted figure never appears without its epistemic
    /// status.
    pub fn describe(&self) -> String {
        let status = match self.status {
            CalibrationStatus::Provisional => "provisional",
            CalibrationStatus::Calibrated => "calibrated",
        };
        let breadth = match self.breadth {
            Breadth::Measured => String::new(),
            Breadth::Extrapolated {
                nearest_measured_fraction,
            } => format!(
                ", extrapolated from a breadth of {:.1}%",
                100.0 * nearest_measured_fraction
            ),
        };
        format!(
            "predicted {:.1} tok/s ({:.2}x) using {} execution-cost observation `{}`{breadth}",
            self.tokens_per_second, self.speedup, status, self.calibration_id,
        )
    }
}

/// Why a cost could not be predicted.
///
/// One variant per missing fact, so a refusal says which measurement to
/// go and take — the same shape [`super::selection`] uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostRefusal {
    /// Nothing has been measured on this machine at all.
    NoObservations,
    /// Every observation is of a different model. Byte economics do not
    /// transfer across models: the families, their shares and the
    /// kernels all differ.
    DifferentModel {
        ledger_model: String,
        observed_models: Vec<String>,
    },
}

impl std::fmt::Display for CostRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoObservations => write!(
                f,
                "no execution-cost observation has been recorded — run the decode benchmark"
            ),
            Self::DifferentModel {
                ledger_model,
                observed_models,
            } => write!(
                f,
                "no observation for {ledger_model}; measured models are {}",
                observed_models.join(", ")
            ),
        }
    }
}

/// **The measured execution cost of this backend, as evidence.**
///
/// A bag of observations plus the rule for reading them. It gains
/// resolution by accumulating measurements, not by fitting a better
/// curve to the ones it has.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionCostModel {
    pub observations: Vec<ExecutionCostObservation>,
}

impl ExecutionCostModel {
    pub fn new(observations: Vec<ExecutionCostObservation>) -> Self {
        Self { observations }
    }

    /// Whether beta has been shown to hold across separated breadths.
    ///
    /// Not a count of observations: ten measurements at the same byte
    /// fraction still say nothing about whether beta varies with
    /// breadth, which is the specific thing a planner extrapolates on.
    pub fn status(&self) -> CalibrationStatus {
        let mut fractions: Vec<f64> = self
            .observations
            .iter()
            .filter(|o| o.bytes_to_gpu_factor().is_some())
            .map(|o| o.byte_fraction_removed())
            .collect();
        fractions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let separated = fractions
            .windows(2)
            .any(|w| w[1] - w[0] > DISTINCT_BREADTH_MIN_SEPARATION);
        if separated {
            CalibrationStatus::Calibrated
        } else {
            CalibrationStatus::Provisional
        }
    }

    /// Predict what `ledger`'s candidate representation costs to decode.
    ///
    /// Uses the observation of the same model whose breadth is nearest
    /// the ledger's — nearest, not fitted, because with one measured
    /// point a fit is a straight line through it and a pretence.
    pub fn predict(&self, ledger: &ByteLedger) -> Result<CostPrediction, CostRefusal> {
        if self.observations.is_empty() {
            return Err(CostRefusal::NoObservations);
        }
        let wanted = ledger.fraction_removed();
        let chosen = self
            .observations
            .iter()
            .filter(|o| o.model_identity == ledger.model)
            .filter(|o| o.bytes_to_gpu_factor().is_some())
            .min_by(|a, b| {
                let (x, y) = (
                    (a.byte_fraction_removed() - wanted).abs(),
                    (b.byte_fraction_removed() - wanted).abs(),
                );
                x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some(o) = chosen else {
            return Err(CostRefusal::DifferentModel {
                ledger_model: ledger.model.clone(),
                observed_models: self
                    .observations
                    .iter()
                    .map(|o| o.model_identity.clone())
                    .collect(),
            });
        };

        let beta = o.bytes_to_gpu_factor().expect("filtered to Some above");
        let gpu = o.baseline_gpu_ms_per_token * (1.0 - beta * wanted);
        let wall = gpu + o.fixed_overhead_ms;
        let baseline_wall = o.baseline_gpu_ms_per_token + o.fixed_overhead_ms;
        let measured = o.byte_fraction_removed();
        Ok(CostPrediction {
            gpu_ms_per_token: gpu,
            wall_ms_per_token: wall,
            tokens_per_second: if wall > 0.0 { 1000.0 / wall } else { 0.0 },
            speedup: if wall > 0.0 {
                baseline_wall / wall
            } else {
                0.0
            },
            calibration_id: o.id.clone(),
            status: self.status(),
            breadth: if (wanted - measured).abs() <= DISTINCT_BREADTH_MIN_SEPARATION {
                Breadth::Measured
            } else {
                Breadth::Extrapolated {
                    nearest_measured_fraction: measured,
                }
            },
        })
    }
}

/// **The one execution-cost observation this programme has measured.**
///
/// Apple M3 Max (40-core GPU), Metal, on Kimi-Linear-48B-A3B: the
/// four-family Q8_0 map against its own BF16 baseline, both arms in one
/// process, interleaved blocks with alternating order, per-arm minimum
/// over four blocks of 128 tokens, two sessions, on AC power.
///
/// ```text
/// bytes removed     957.0 MB of 5.985 GB   15.99 %
/// GPU time removed  26.87 -> 23.43 ms      12.80 %
/// beta                                     0.8006
/// ```
///
/// Session 2 measured 26.97 -> 23.56 ms, beta 0.7907. The figures here
/// are session 1's floors; the two sessions agree on beta to 0.01,
/// which is why one record is honest and a spread is not yet worth
/// modelling.
///
/// **This is one breadth.** A prediction made from it at a very
/// different byte fraction is an extrapolation and says so.
pub fn m3max_metal_001() -> ExecutionCostObservation {
    ExecutionCostObservation {
        id: "m3max-metal-001".into(),
        machine: "Mac15,8".into(),
        device: "Apple M3 Max (40-core GPU)".into(),
        backend: "metal".into(),
        compiler_commit: "a477765a".into(),
        model_identity: "Kimi-Linear-48B-A3B-Instruct".into(),
        baseline_representation: "BF16".into(),
        candidate_representation: "experts L20-26 + KDA{20,21,22,24,25} + MLA{23,26} + head, Q8_0"
            .into(),
        families_changed: vec![
            "routed experts".into(),
            "KDA projections".into(),
            "MLA projections".into(),
            "output head".into(),
        ],
        scopes_changed: 15,
        baseline_bytes_per_token: 5_984_976_896,
        candidate_bytes_per_token: 5_027_956_736,
        baseline_gpu_ms_per_token: 26.87,
        candidate_gpu_ms_per_token: 23.43,
        fixed_overhead_ms: 1.05,
        benchmark_protocol: "both arms in-process, interleaved blocks with alternating order, \
                             per-arm minimum of 4 x 128 tokens, 2 sessions, AC power"
            .into(),
        evidence: vec![
            "kimi-quality-bank-provenance/bench-full4-session1.log".into(),
            "kimi-quality-bank-provenance/bench-full4-session2.log".into(),
        ],
    }
}

#[cfg(test)]
#[path = "execution_cost_tests.rs"]
mod tests;
