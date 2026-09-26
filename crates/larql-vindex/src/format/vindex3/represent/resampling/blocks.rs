//! **A2 — independent blocks.**
//!
//! A random partition of the sequences into disjoint blocks of `n`,
//! each evaluated on its own: the spread of a FRESH n-sequence
//! measurement. Not pooled with the ladder — a prefix's marginal law is
//! a random subset's, but the ladder's quantity is a trajectory.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use super::SequenceSet;

/// Every block's value at every depth: `values[depth_index]` holds one
/// entry per block over all partitions, `[statistic]`-indexed.
#[derive(Debug, Clone, Serialize)]
pub struct Blocks {
    pub seed: u64,
    pub depths: Vec<usize>,
    pub partitions: usize,
    pub values: Vec<Vec<[f64; 2]>>,
}

impl Blocks {
    pub fn at(&self, depth_index: usize, statistic: usize) -> Vec<f64> {
        self.values[depth_index]
            .iter()
            .map(|v| v[statistic])
            .collect()
    }
}

/// `partitions` random partitions at every supported depth. At the
/// full depth there is exactly one block, the bank itself, so it is
/// evaluated once rather than `partitions` times.
pub fn independent(set: &SequenceSet<'_>, partitions: usize, seed: u64) -> Blocks {
    let depths = set.depths();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let ids = set.ids();
    let values = depths
        .iter()
        .map(|d| {
            if *d == ids.len() {
                return vec![set.statistics(&ids)];
            }
            let mut out = Vec::with_capacity(partitions * (ids.len() / d));
            for _ in 0..partitions {
                let mut order = ids.clone();
                order.shuffle(&mut rng);
                for block in order.chunks_exact(*d) {
                    out.push(set.statistics(block));
                }
            }
            out
        })
        .collect();
    Blocks {
        seed,
        depths,
        partitions,
        values,
    }
}
