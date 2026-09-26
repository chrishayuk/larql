//! **Sequence-level resampling over a persisted observation stream.**
//!
//! UNCERTAINTY-2 (`docs/represent/forecasts/represent-uncertainty-2.json`).
//! Everything here is a projection of evidence already recorded: no
//! model runs, and every subset statistic is read through the single
//! canonical mapping `Statistic::observe` on a bank rederived from the
//! chosen observations.
//!
//! The sampling unit is the SEQUENCE. Positions inside a teacher-forced
//! sequence are not identically distributed (REAL-EVIDENCE-1 measured a
//! U-shaped mean and a max growing ~23x across a sequence), so every
//! depth is a whole number of sequences and no analysis here resamples
//! positions.

use std::collections::BTreeMap;

use super::bank::{BankBuilder, PositionObservation};
use super::quality::QualityBank;
use super::statistic::Statistic;

pub mod blocks;
pub mod dependence;
pub mod ladder;
pub mod summary;
pub mod transfer;

/// The authority ladder, in whole sequences. Frozen.
pub const DEPTHS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
/// Random sequence orders per bank for the progressive ladder. Frozen.
pub const PROGRESSIVE_DRAWS: usize = 200;
/// Random partitions per bank for the independent blocks. Frozen.
pub const INDEPENDENT_PARTITIONS: usize = 50;
/// Permutations behind the tail-clustering null. Frozen.
pub const CLUSTERING_PERMUTATIONS: usize = 200;
/// The nominal-90% band: these quantiles of the relative distribution.
pub const BAND_LOW: f64 = 0.05;
pub const BAND_HIGH: f64 = 0.95;
/// Relative tolerances whose hit-fractions are reported.
pub const WITHIN_BANDS: [f64; 4] = [0.05, 0.10, 0.25, 0.50];
/// Every seed derives from this so a result names its own randomness.
pub const SEED_BASE: u64 = 20_260_919;

/// The two statistics this rung characterises, never averaged.
pub const STATISTICS: [Statistic; 2] = [Statistic::RouteFlipRate, Statistic::KlP99];

/// Which analysis a seed belongs to, so the two banks and the three
/// analyses never share a random stream by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Analysis {
    Progressive,
    Independent,
    Clustering,
}

/// A deterministic seed for one (bank, analysis) pair.
pub fn seed_for(bank_index: u64, analysis: Analysis) -> u64 {
    let a = match analysis {
        Analysis::Progressive => 1,
        Analysis::Independent => 2,
        Analysis::Clustering => 3,
    };
    SEED_BASE + bank_index * 10 + a
}

/// Observations grouped by sequence id, so a subset is chosen by
/// sequence and never by position.
pub struct SequenceSet<'a> {
    by_sequence: BTreeMap<u32, Vec<&'a PositionObservation>>,
}

impl<'a> SequenceSet<'a> {
    pub fn new(observations: &'a [PositionObservation]) -> Self {
        let mut by_sequence: BTreeMap<u32, Vec<&'a PositionObservation>> = BTreeMap::new();
        for o in observations {
            by_sequence.entry(o.sequence).or_default().push(o);
        }
        Self { by_sequence }
    }

    /// Sequence ids, ascending.
    pub fn ids(&self) -> Vec<u32> {
        self.by_sequence.keys().copied().collect()
    }

    pub fn len(&self) -> usize {
        self.by_sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_sequence.is_empty()
    }

    /// The ladder depths this set can actually support.
    pub fn depths(&self) -> Vec<usize> {
        DEPTHS
            .iter()
            .copied()
            .filter(|d| *d <= self.len())
            .collect()
    }

    /// The bank over exactly these sequences, without cloning a single
    /// observation.
    pub fn bank(&self, ids: &[u32]) -> QualityBank {
        let mut b = BankBuilder::new();
        for id in ids {
            for o in self.by_sequence.get(id).into_iter().flatten() {
                b.observe(o);
            }
        }
        b.finish()
    }

    /// Both statistics over these sequences, in [`STATISTICS`] order.
    pub fn statistics(&self, ids: &[u32]) -> [f64; 2] {
        read(&self.bank(ids))
    }
}

/// [`STATISTICS`] read off one bank. A statistic a bank cannot supply
/// (an empty bank) is NaN, which no summary silently averages.
pub fn read(bank: &QualityBank) -> [f64; 2] {
    let mut out = [f64::NAN; 2];
    for (slot, s) in STATISTICS.iter().enumerate() {
        out[slot] = s.observe(bank).0.unwrap_or(f64::NAN);
    }
    out
}

#[cfg(test)]
mod tests;
