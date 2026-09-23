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
    /// The next token is not in the vocabulary.
    TokenOutOfRange { token: u32, vocabulary: usize },
}

/// Score one position.
pub fn position_metrics(
    reference: &[f32],
    candidate: &[f32],
    next_token: Option<u32>,
) -> Result<PositionScore, MetricError> {
    if reference.len() != candidate.len() {
        return Err(MetricError::VocabularyMismatch {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }
    if reference.is_empty() {
        return Err(MetricError::Empty);
    }
    let lr = log_softmax(reference).ok_or(MetricError::NonFinite { arm: "reference" })?;
    let lc = log_softmax(candidate).ok_or(MetricError::NonFinite { arm: "candidate" })?;

    let kl = lr
        .iter()
        .zip(&lc)
        .map(|(&r, &c)| r.exp() * (r - c))
        .sum::<f64>();
    let delta_nll = match next_token {
        None => None,
        Some(t) => {
            let i = t as usize;
            if i >= lr.len() {
                return Err(MetricError::TokenOutOfRange {
                    token: t,
                    vocabulary: lr.len(),
                });
            }
            // NLL(candidate) - NLL(reference) = -lc + lr.
            Some(lr[i] - lc[i])
        }
    };
    let reference_top = top_k(&lr, TOP_K_OVERLAP);
    let candidate_top = top_k(&lc, TOP_K_OVERLAP);
    let top5_overlap = reference_top
        .iter()
        .filter(|i| candidate_top.contains(i))
        .count();
    let p1 = lr[reference_top[0]].exp();
    let p2 = reference_top.get(1).map_or(0.0, |&i| lr[i].exp());
    let reference_entropy = -lr.iter().map(|&l| l.exp() * l).sum::<f64>();
    Ok(PositionScore {
        kl,
        top1_agree: reference_top[0] == candidate_top[0],
        top5_overlap,
        delta_nll,
        reference_margin: p1 - p2,
        reference_entropy,
    })
}

/// Log-probabilities in f64, or `None` if a logit is not finite.
fn log_softmax(logits: &[f32]) -> Option<Vec<f64>> {
    if logits.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    Some(logits.iter().map(|&v| v as f64 - lse).collect())
}

/// Indices of the `k` largest values, largest first; ties resolve to the
/// lower index, so argmax is the first maximum.
fn top_k(values: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    order.truncate(k.min(values.len()));
    order
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
    use crate::format::vindex3::represent::bank::nearest_rank_percentile;
    if positions.is_empty() {
        return None;
    }
    let n = positions.len() as f64;
    let mut kls: Vec<f64> = positions.iter().map(|p| p.kl).collect();
    let kl_mean = kls.iter().sum::<f64>() / n;
    kls.sort_by(f64::total_cmp);
    let deltas: Vec<f64> = positions.iter().filter_map(|p| p.delta_nll).collect();
    Some(Aggregate {
        positions: positions.len(),
        kl_mean,
        kl_p50: nearest_rank_percentile(&kls, 0.50),
        kl_p99: nearest_rank_percentile(&kls, 0.99),
        kl_max: *kls.last().expect("non-empty"),
        top1_agreement: positions.iter().filter(|p| p.top1_agree).count() as f64 / n,
        top5_overlap_mean: positions.iter().map(|p| p.top5_overlap as f64).sum::<f64>() / n,
        delta_nll_mean: (!deltas.is_empty())
            .then(|| deltas.iter().sum::<f64>() / deltas.len() as f64),
    })
}

/// Whether `margin` falls in `band`: `[lo, hi)`, except that the last band
/// is closed so a certain reference (`p1 - p2 = 1`) lands in it.
fn in_band(margin: f64, band: (f64, f64)) -> bool {
    let last = band == MARGIN_BANDS[MARGIN_BANDS.len() - 1];
    margin >= band.0 && (margin < band.1 || (last && margin <= band.1))
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
    let all = aggregate(positions)?;
    let mut categories: Vec<&str> = Vec::new();
    for p in positions {
        if !categories.contains(&p.category.as_str()) {
            categories.push(&p.category);
        }
    }
    let by_category = categories
        .into_iter()
        .filter_map(|c| {
            let group: Vec<PositionMetrics> = positions
                .iter()
                .filter(|p| p.category == c)
                .cloned()
                .collect();
            aggregate(&group).map(|a| (c.to_string(), a))
        })
        .collect();
    let by_margin_band = MARGIN_BANDS
        .iter()
        .filter_map(|&band| {
            let group: Vec<PositionMetrics> = positions
                .iter()
                .filter(|p| in_band(p.reference_margin, band))
                .cloned()
                .collect();
            aggregate(&group).map(|a| (band, a))
        })
        .collect();
    Some(Summary {
        all,
        by_category,
        by_margin_band,
    })
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;
