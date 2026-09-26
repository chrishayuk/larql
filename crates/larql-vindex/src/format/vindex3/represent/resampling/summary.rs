//! **Per-depth summaries of a relative distribution.**
//!
//! Every number is relative to the bank's own full value, so a depth
//! table reads the same way for a rate near 0.16 and a tail near 2e-3.

use std::collections::BTreeMap;

use serde::Serialize;

use super::{BAND_HIGH, BAND_LOW, WITHIN_BANDS};

/// Nearest-rank quantile over `sorted`, ascending.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    s.sort_by(f64::total_cmp);
    s
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Quantiles {
    pub mean: f64,
    pub p05: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
}

impl Quantiles {
    pub fn of(values: &[f64]) -> Self {
        let s = sorted(values);
        Self {
            mean: s.iter().sum::<f64>() / s.len() as f64,
            p05: quantile(&s, 0.05),
            p25: quantile(&s, 0.25),
            p50: quantile(&s, 0.50),
            p75: quantile(&s, 0.75),
            p95: quantile(&s, 0.95),
        }
    }
}

/// A [low, high] interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Band {
    pub low: f64,
    pub high: f64,
}

impl Band {
    pub fn of(values: &[f64]) -> Self {
        let s = sorted(values);
        Self {
            low: quantile(&s, BAND_LOW),
            high: quantile(&s, BAND_HIGH),
        }
    }

    pub fn contains(&self, v: f64) -> bool {
        v >= self.low && v <= self.high
    }
}

/// One depth, one statistic.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DepthSummary {
    pub sequences: usize,
    pub positions: usize,
    pub samples: usize,
    pub full: f64,
    /// `(est - full) / full` across samples.
    pub relative: Quantiles,
    pub mean_abs_relative: f64,
    /// Standard deviation / mean of the ABSOLUTE values.
    pub coefficient_of_variation: f64,
    /// Fraction of samples within ±tolerance of full, keyed "±5%" etc.
    pub within: BTreeMap<String, f64>,
    /// The frozen nominal-90% band of relative values.
    pub band: Band,
    pub absolute_band: Band,
    /// Fraction of samples at or below `limit`, when a limit applies —
    /// for a statistic the full bank FAILS, every one is a false pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass_fraction: Option<f64>,
    /// `|est(next) - est(this)| / full` across draws, ladder only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_to_next: Option<Quantiles>,
}

pub fn relative(values: &[f64], full: f64) -> Vec<f64> {
    values.iter().map(|v| (v - full) / full).collect()
}

pub fn within_label(tolerance: f64) -> String {
    format!("±{}%", (tolerance * 100.0).round() as i64)
}

/// Build the summary. `limit` is the gate limit this statistic is
/// judged against, if any; `steps` the ladder's step movements, if any.
pub fn summarise(
    sequences: usize,
    positions_per_sequence: usize,
    values: &[f64],
    full: f64,
    limit: Option<f64>,
    steps: Option<&[f64]>,
) -> DepthSummary {
    let rel = relative(values, full);
    let abs: Vec<f64> = rel.iter().map(|r| r.abs()).collect();
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let within = WITHIN_BANDS
        .iter()
        .map(|t| {
            let hit = abs.iter().filter(|a| **a <= *t).count() as f64 / abs.len() as f64;
            (within_label(*t), hit)
        })
        .collect();
    DepthSummary {
        sequences,
        positions: sequences * positions_per_sequence,
        samples: values.len(),
        full,
        relative: Quantiles::of(&rel),
        mean_abs_relative: abs.iter().sum::<f64>() / abs.len() as f64,
        coefficient_of_variation: var.sqrt() / mean,
        within,
        band: Band::of(&rel),
        absolute_band: Band::of(values),
        pass_fraction: limit
            .map(|l| finite.iter().filter(|v| **v <= l).count() as f64 / finite.len() as f64),
        step_to_next: steps.map(Quantiles::of),
    }
}
