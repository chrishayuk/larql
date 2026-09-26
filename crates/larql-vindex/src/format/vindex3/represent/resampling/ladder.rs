//! **A1 — the progressive nested ladder.**
//!
//! One random sequence order, evaluated on its prefixes of 1, 2, 4, ...
//! sequences: the trajectory of ONE growing measurement, which is what
//! REPRESENT does when it spends more authority on a candidate it is
//! already measuring. Estimates along a path are nested and correlated
//! BY DESIGN; that is the quantity, not a defect.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use super::SequenceSet;

/// One draw's trajectory: `values[depth_index][statistic]`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Path {
    pub order: Vec<u32>,
    pub values: Vec<[f64; 2]>,
}

/// Every draw's trajectory over the same depths.
#[derive(Debug, Clone, Serialize)]
pub struct Ladder {
    pub seed: u64,
    pub depths: Vec<usize>,
    pub paths: Vec<Path>,
}

impl Ladder {
    /// The values at one depth across draws, for one statistic.
    pub fn at(&self, depth_index: usize, statistic: usize) -> Vec<f64> {
        self.paths
            .iter()
            .map(|p| p.values[depth_index][statistic])
            .collect()
    }

    /// `|est(next) - est(this)| / full` for every draw, from one depth
    /// to the next. `None` at the last depth.
    pub fn steps(&self, depth_index: usize, statistic: usize, full: f64) -> Option<Vec<f64>> {
        if depth_index + 1 >= self.depths.len() {
            return None;
        }
        Some(
            self.paths
                .iter()
                .map(|p| {
                    (p.values[depth_index + 1][statistic] - p.values[depth_index][statistic]).abs()
                        / full
                })
                .collect(),
        )
    }
}

/// Evaluate `draws` random orders on their prefixes at every depth the
/// set supports. Deterministic in `seed`.
pub fn progressive(set: &SequenceSet<'_>, draws: usize, seed: u64) -> Ladder {
    let depths = set.depths();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let ids = set.ids();
    let paths = (0..draws)
        .map(|_| {
            let mut order = ids.clone();
            order.shuffle(&mut rng);
            let values = depths
                .iter()
                .map(|d| set.statistics(&order[..*d]))
                .collect();
            Path { order, values }
        })
        .collect();
    Ladder {
        seed,
        depths,
        paths,
    }
}
