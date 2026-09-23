//! Per-position metrics and their aggregates, for MEASURE-PLAN-1.
//!
//! Everything is over the FULL vocabulary, in nats, and computed in f64
//! from f32 logits. The Kimi procedure's top-2048 truncation is its own
//! and is not reproduced here.
//!
//! Two of the per-position quantities exist to interpret the others: the
//! reference's top-1 margin (`p1 - p2`) and its entropy. A top-1 flip where
//! the reference could barely separate its first two choices is a
//! different event from one where it was certain, and a mean over all
//! positions hides the difference. So the aggregates are also split by
//! margin band.

use serde::{Deserialize, Serialize};

/// How many of the reference's top tokens the overlap metric compares.
pub const TOP_K_OVERLAP: usize = 5;

/// Reference-margin bands, `[lo, hi)` over `p1 - p2`, the last closed at 1.
/// Chosen before any measurement: near-tie, contested, confident.
pub const MARGIN_BANDS: [(f64, f64); 3] = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)];

/// One scored position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PositionMetrics {
    /// Bank sample index.
    pub sample: usize,
    /// Position within the sample. Position `i` predicts token `i + 1`.
    pub position: usize,
    /// The sample's category in the bank.
    pub category: String,
    /// KL(reference ‖ candidate), nats.
    pub kl: f64,
    /// Whether the two arms' argmax agree.
    pub top1_agree: bool,
    /// How many of the reference's top-5 tokens are in the candidate's top-5.
    pub top5_overlap: usize,
    /// NLL(candidate) − NLL(reference) of the actual next token, when the
    /// sample has one.
    pub delta_nll: Option<f64>,
    /// The reference's `p1 - p2`.
    pub reference_margin: f64,
    /// The reference's entropy, nats.
    pub reference_entropy: f64,
}

/// Why two logit rows cannot be compared.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricError {
    /// The rows have different vocabulary sizes.
    VocabularyMismatch { reference: usize, candidate: usize },
    /// A row holds a NaN or an infinity.
    NonFinite { arm: &'static str },
    /// A row is empty.
    Empty,
}

/// Score one position.
pub fn position_metrics(
    reference: &[f32],
    candidate: &[f32],
    next_token: Option<u32>,
) -> Result<PositionScore, MetricError> {
    let _ = (reference, candidate, next_token);
    todo!("MEASURE-PLAN-1 PR 2")
}

/// The per-position numbers, before a sample, position and category are
/// attached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionScore {
    pub kl: f64,
    pub top1_agree: bool,
    pub top5_overlap: usize,
    pub delta_nll: Option<f64>,
    pub reference_margin: f64,
    pub reference_entropy: f64,
}

/// Aggregates over a set of positions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Aggregate {
    pub positions: usize,
    pub kl_mean: f64,
    /// Nearest-rank, as the quality bank's percentiles are.
    pub kl_p50: f64,
    /// Nearest-rank: a p99 that names a position that actually occurred.
    pub kl_p99: f64,
    pub kl_max: f64,
    /// Share of positions whose argmax agrees, in [0, 1].
    pub top1_agreement: f64,
    pub top5_overlap_mean: f64,
    /// Mean over positions that have a next token; `None` if none do.
    pub delta_nll_mean: Option<f64>,
}

/// Aggregate `positions`. `None` for an empty set, because a mean of
/// nothing is not zero.
pub fn aggregate(positions: &[PositionMetrics]) -> Option<Aggregate> {
    let _ = positions;
    todo!("MEASURE-PLAN-1 PR 2")
}

/// The report's summary: all positions, then per category, then per margin
/// band. Categories are in first-seen order. A band or category with no
/// positions is omitted, never reported as zeros.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub all: Aggregate,
    pub by_category: Vec<(String, Aggregate)>,
    pub by_margin_band: Vec<((f64, f64), Aggregate)>,
}

/// Summarise `positions`; `None` if there are none.
pub fn summarise(positions: &[PositionMetrics]) -> Option<Summary> {
    let _ = positions;
    todo!("MEASURE-PLAN-1 PR 2")
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;
