//! CONTINUATION-CODEC-MAP-1 reconnaissance: the tail-localisation report,
//! pure and model-free. Every map is scored against R at the same
//! positions as C and full CH; its incremental damage is
//! d_i = KL(R‖map)_i − KL(R‖C)_i, and its common scale is the fraction of
//! full CH's tail it reproduces: Σ_{i∈T} d_map,i / Σ_{i∈T} d_CH,i over
//! CH's worst positions T. Reconnaissance numbers, not a rule.

use std::collections::BTreeSet;

use larql_vindex::format::vindex3::represent::measure::plan::metrics::PositionMetrics;

use super::codec_2_stats::nearest_rank;

/// CH's worst positions: the wide set (stable enough to rank maps by) and
/// the narrow set (CODEC-2's B3 quantile).
pub const TAIL_WIDE: f64 = 0.05;
pub const TAIL_NARROW: f64 = 0.01;
pub const TAIL_QUANTILE: f64 = 0.99;

/// Per-position incremental damage over C.
pub fn incremental(arm: &[PositionMetrics], c: &[PositionMetrics]) -> Vec<f64> {
    assert_eq!(arm.len(), c.len(), "paired positions");
    arm.iter()
        .zip(c)
        .map(|(a, b)| {
            assert_eq!(a.position, b.position, "paired positions");
            a.kl - b.kl
        })
        .collect()
}

/// Indices of the `fraction` largest values (at least one), ties broken by
/// index so the set is deterministic.
pub fn worst(d: &[f64], fraction: f64) -> BTreeSet<usize> {
    let k = ((d.len() as f64 * fraction).ceil() as usize).clamp(1, d.len().max(1));
    let mut order: Vec<usize> = (0..d.len()).collect();
    order.sort_by(|&a, &b| d[b].total_cmp(&d[a]).then(a.cmp(&b)));
    order.into_iter().take(k).collect()
}

/// Σ over `set` of `d_map` as a share of the same Σ of `d_ch`.
pub fn explained(d_map: &[f64], d_ch: &[f64], set: &BTreeSet<usize>) -> f64 {
    let ch: f64 = set.iter().map(|&i| d_ch[i]).sum();
    let map: f64 = set.iter().map(|&i| d_map[i]).sum();
    if ch == 0.0 {
        0.0
    } else {
        map / ch
    }
}

pub fn jaccard(a: &BTreeSet<usize>, b: &BTreeSet<usize>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

/// One map's line of the report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MapLine {
    pub name: String,
    pub spec: String,
    pub exact_fraction: f64,
    pub top1_agreement: f64,
    pub mean_d: f64,
    pub p99_d: f64,
    pub explained_wide: f64,
    pub explained_narrow: f64,
    /// Overlap of the map's own wide-tail positions with CH's.
    pub tail_overlap_with_ch: f64,
}

pub fn line(
    name: &str,
    spec: &str,
    exact_fraction: f64,
    arm: &[PositionMetrics],
    c: &[PositionMetrics],
    d_ch: &[f64],
) -> MapLine {
    let d = incremental(arm, c);
    let (wide, narrow) = (worst(d_ch, TAIL_WIDE), worst(d_ch, TAIL_NARROW));
    MapLine {
        name: name.to_string(),
        spec: spec.to_string(),
        exact_fraction,
        top1_agreement: arm.iter().filter(|m| m.top1_agree).count() as f64
            / arm.len().max(1) as f64,
        mean_d: d.iter().sum::<f64>() / d.len().max(1) as f64,
        p99_d: nearest_rank(&d, TAIL_QUANTILE),
        explained_wide: explained(&d, d_ch, &wide),
        explained_narrow: explained(&d, d_ch, &narrow),
        tail_overlap_with_ch: jaccard(&worst(&d, TAIL_WIDE), &wide),
    }
}

/// For each of CH's wide-tail positions, in how many maps it is also in
/// that map's own wide tail — high counts are stable (fragile whatever is
/// compressed), low counts are specific to what a map compresses.
pub fn tail_persistence(d_ch: &[f64], maps: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let own: Vec<BTreeSet<usize>> = maps.iter().map(|d| worst(d, TAIL_WIDE)).collect();
    worst(d_ch, TAIL_WIDE)
        .into_iter()
        .map(|i| (i, own.iter().filter(|s| s.contains(&i)).count()))
        .collect()
}
