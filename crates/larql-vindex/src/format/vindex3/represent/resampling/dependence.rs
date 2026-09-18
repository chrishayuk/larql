//! **Within-sequence dependence — descriptive, not a prerequisite.**
//!
//! REAL-EVIDENCE-1 ASSUMED within-sequence independence and said so.
//! These readings describe it from the raw observations. Whatever they
//! say, the sequence remains the resampling unit of the primary
//! analysis, because it is the natural independent experimental unit.

use std::collections::BTreeMap;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use super::super::bank::PositionObservation;

/// KL is heavy-tailed and can sit at zero; the log is taken above this.
pub const KL_FLOOR: f64 = 1e-9;
/// The tail is the top this fraction of observations by KL.
pub const TAIL_FRACTION: f64 = 0.01;
/// Position quartiles of a 32-position sequence.
pub const POSITION_QUARTILES: usize = 4;

/// One observation's per-position readings.
#[derive(Debug, Clone, Copy)]
pub struct Reading {
    pub sequence: u32,
    pub position: u32,
    pub kl: f64,
    pub flips: f64,
}

pub fn readings(observations: &[PositionObservation]) -> Vec<Reading> {
    observations
        .iter()
        .map(|o| Reading {
            sequence: o.sequence,
            position: o.position,
            kl: o.kl(),
            flips: o.route_changes().1 as f64,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Dependence {
    /// One-way intraclass correlation of sequence membership.
    pub icc_log_kl: f64,
    pub icc_flips: f64,
    /// Lag-1 autocorrelation of within-sequence residuals after the
    /// position profile and the sequence mean are removed.
    pub lag1_log_kl: f64,
    pub lag1_flips: f64,
    /// Share of the top-1% KL observations in each position quartile.
    pub tail_share_by_position_quartile: Vec<f64>,
    pub tail_size: usize,
    /// Within-sequence tail pairs, observed / permutation mean.
    pub tail_clustering_ratio: f64,
    pub tail_pairs_observed: u64,
    pub tail_pairs_permuted_mean: f64,
}

/// Group values by sequence, in position order.
fn grouped(r: &[Reading], f: impl Fn(&Reading) -> f64) -> Vec<Vec<f64>> {
    let mut m: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
    for x in r {
        m.entry(x.sequence).or_default().push((x.position, f(x)));
    }
    m.into_values()
        .map(|mut v| {
            v.sort_by_key(|(p, _)| *p);
            v.into_iter().map(|(_, x)| x).collect()
        })
        .collect()
}

/// ICC(1) from a one-way layout with equal group sizes.
pub fn icc(groups: &[Vec<f64>]) -> f64 {
    let g = groups.len() as f64;
    let k = groups[0].len() as f64;
    let grand = groups.iter().flatten().sum::<f64>() / (g * k);
    let means: Vec<f64> = groups.iter().map(|v| v.iter().sum::<f64>() / k).collect();
    let msb = k * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>() / (g - 1.0);
    let msw = groups
        .iter()
        .zip(&means)
        .map(|(v, m)| v.iter().map(|x| (x - m).powi(2)).sum::<f64>())
        .sum::<f64>()
        / (g * (k - 1.0));
    (msb - msw) / (msb + (k - 1.0) * msw)
}

/// Lag-1 autocorrelation of residuals `x - seq_mean - pos_mean + grand`.
pub fn lag1(groups: &[Vec<f64>]) -> f64 {
    let g = groups.len() as f64;
    let k = groups[0].len();
    let grand = groups.iter().flatten().sum::<f64>() / (g * k as f64);
    let pos_mean: Vec<f64> = (0..k)
        .map(|i| groups.iter().map(|v| v[i]).sum::<f64>() / g)
        .collect();
    let residuals: Vec<Vec<f64>> = groups
        .iter()
        .map(|v| {
            let m = v.iter().sum::<f64>() / k as f64;
            v.iter()
                .enumerate()
                .map(|(i, x)| x - m - pos_mean[i] + grand)
                .collect()
        })
        .collect();
    let pairs: Vec<(f64, f64)> = residuals
        .iter()
        .flat_map(|r| r.windows(2).map(|w| (w[0], w[1])))
        .collect();
    let n = pairs.len() as f64;
    let (ma, mb) = (
        pairs.iter().map(|p| p.0).sum::<f64>() / n,
        pairs.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let cov = pairs.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>();
    let va = pairs.iter().map(|p| (p.0 - ma).powi(2)).sum::<f64>();
    let vb = pairs.iter().map(|p| (p.1 - mb).powi(2)).sum::<f64>();
    cov / (va * vb).sqrt()
}

fn pairs_within_sequences(sequences: &[u32]) -> u64 {
    let mut counts: BTreeMap<u32, u64> = BTreeMap::new();
    for s in sequences {
        *counts.entry(*s).or_default() += 1;
    }
    counts.values().map(|c| c * (c - 1) / 2).sum()
}

/// The full descriptive reading. `positions` is the sequence length.
pub fn describe(r: &[Reading], positions: usize, permutations: usize, seed: u64) -> Dependence {
    let log_kl = grouped(r, |x| x.kl.max(KL_FLOOR).log10());
    let flips = grouped(r, |x| x.flips);

    let mut by_kl: Vec<&Reading> = r.iter().collect();
    by_kl.sort_by(|a, b| b.kl.total_cmp(&a.kl));
    let tail_size = ((TAIL_FRACTION * r.len() as f64).ceil() as usize).max(1);
    let tail = &by_kl[..tail_size];
    let quartile = positions.div_ceil(POSITION_QUARTILES);
    let mut share = vec![0.0; POSITION_QUARTILES];
    for t in tail {
        share[(t.position as usize / quartile).min(POSITION_QUARTILES - 1)] += 1.0;
    }
    for s in &mut share {
        *s /= tail_size as f64;
    }
    let observed = pairs_within_sequences(&tail.iter().map(|t| t.sequence).collect::<Vec<_>>());
    let mut labels: Vec<u32> = r.iter().map(|x| x.sequence).collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let tail_index: Vec<usize> = {
        let mut idx: Vec<usize> = (0..r.len()).collect();
        idx.sort_by(|a, b| r[*b].kl.total_cmp(&r[*a].kl));
        idx[..tail_size].to_vec()
    };
    let permuted_mean = (0..permutations)
        .map(|_| {
            labels.shuffle(&mut rng);
            pairs_within_sequences(&tail_index.iter().map(|i| labels[*i]).collect::<Vec<_>>())
                as f64
        })
        .sum::<f64>()
        / permutations as f64;

    Dependence {
        icc_log_kl: icc(&log_kl),
        icc_flips: icc(&flips),
        lag1_log_kl: lag1(&log_kl),
        lag1_flips: lag1(&flips),
        tail_share_by_position_quartile: share,
        tail_size,
        tail_clustering_ratio: observed as f64 / permuted_mean,
        tail_pairs_observed: observed,
        tail_pairs_permuted_mean: permuted_mean,
    }
}
