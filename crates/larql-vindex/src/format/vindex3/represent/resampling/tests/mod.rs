//! The resampling machinery on a fixture: whole sequences only, the
//! full value as the reference, determinism in the seed, and summaries
//! whose numbers can be checked by hand.

use super::super::observation_stream::rederive_bank;
use super::super::observation_stream::tests::stream;
use super::blocks::independent;
use super::dependence::{describe, icc, lag1, readings, Reading};
use super::ladder::progressive;
use super::summary::{quantile, summarise, within_label};
use super::transfer::coverage;
use super::{read, seed_for, Analysis, SequenceSet, STATISTICS};

const SEQUENCES: u32 = 8;
const POSITIONS: u32 = 32;
const SEED: u64 = 7;

#[test]
fn a_subset_is_whole_sequences_and_the_full_value_is_the_bank_itself() {
    let observations = stream(SEQUENCES, POSITIONS);
    let set = SequenceSet::new(&observations);
    assert_eq!(set.len(), 8);
    assert_eq!(set.ids(), (0..8).collect::<Vec<u32>>());
    assert_eq!(set.depths(), vec![1, 2, 4, 8]);
    assert_eq!(set.bank(&[0, 1]).positions, 64);
    assert_eq!(
        set.statistics(&set.ids()),
        read(&rederive_bank(&observations))
    );
    assert_eq!(STATISTICS.len(), 2);
}

#[test]
fn every_ladder_path_ends_at_the_full_value_and_the_seed_fixes_the_orders() {
    let observations = stream(SEQUENCES, POSITIONS);
    let set = SequenceSet::new(&observations);
    let full = set.statistics(&set.ids());
    let ladder = progressive(&set, 5, SEED);
    assert_eq!(ladder.depths, vec![1, 2, 4, 8]);
    assert_eq!(ladder.paths.len(), 5);
    for p in &ladder.paths {
        assert_eq!(p.values.len(), 4);
        assert_eq!(p.values[3], full, "the full depth is the bank itself");
        let mut ids = p.order.clone();
        ids.sort_unstable();
        assert_eq!(ids, set.ids(), "an order is a permutation");
    }
    assert_eq!(
        progressive(&set, 5, SEED).paths,
        ladder.paths,
        "deterministic in the seed"
    );
    assert_ne!(
        progressive(&set, 5, SEED + 1).paths[0].order,
        ladder.paths[0].order
    );
    assert_eq!(ladder.at(3, 0).len(), 5);
    assert_eq!(
        ladder.steps(3, 0, full[0]),
        None,
        "no step after the last depth"
    );
    assert_eq!(ladder.steps(0, 1, full[1]).unwrap().len(), 5);
}

#[test]
fn independent_blocks_partition_the_set_at_every_depth() {
    let observations = stream(SEQUENCES, POSITIONS);
    let set = SequenceSet::new(&observations);
    let blocks = independent(&set, 3, SEED);
    assert_eq!(blocks.depths, vec![1, 2, 4, 8]);
    assert_eq!(
        blocks.at(0, 0).len(),
        3 * 8,
        "depth 1: eight blocks per partition"
    );
    assert_eq!(blocks.at(1, 0).len(), 3 * 4);
    assert_eq!(blocks.at(2, 0).len(), 3 * 2);
    assert_eq!(
        blocks.at(3, 1),
        vec![set.statistics(&set.ids())[1]],
        "the full depth is one block"
    );
}

#[test]
fn nearest_rank_quantiles_are_observed_values() {
    let s = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(quantile(&s, 0.05), 1.0);
    assert_eq!(quantile(&s, 0.50), 3.0);
    assert_eq!(quantile(&s, 0.95), 5.0);
}

#[test]
fn a_summary_can_be_checked_by_hand() {
    // Off the ±10% boundary on purpose: a relative value of exactly 0.1
    // is a floating-point coin toss, and real data never lands there.
    let values = [0.92, 1.0, 1.08, 1.2];
    let s = summarise(2, 32, &values, 1.0, Some(1.0), Some(&[0.1, 0.3]));
    assert_eq!((s.sequences, s.positions, s.samples), (2, 64, 4));
    assert!((s.relative.mean - 0.05).abs() < 1e-12);
    assert_eq!(s.within[&within_label(0.10)], 0.75);
    assert_eq!(s.within[&within_label(0.25)], 1.0);
    assert_eq!(
        s.pass_fraction,
        Some(0.5),
        "two of four at or below the limit"
    );
    assert!((s.band.low + 0.08).abs() < 1e-12);
    assert!((s.band.high - 0.2).abs() < 1e-12);
    assert_eq!(s.step_to_next.as_ref().unwrap().p95, 0.3);
    assert!((s.mean_abs_relative - 0.09).abs() < 1e-12);
    let no_limit = summarise(1, 32, &values, 1.0, None, None);
    assert_eq!(no_limit.pass_fraction, None);
    assert_eq!(no_limit.step_to_next, None);
}

#[test]
fn a_band_covers_its_own_values_and_not_a_shifted_set() {
    let values: Vec<f64> = (0..100).map(|i| 1.0 + i as f64 * 0.001).collect();
    let own = coverage(4, &values, 1.05, &values, 1.05);
    assert!(own.relative_coverage >= 0.9 && own.relative_coverage <= 1.0);
    assert_eq!(own.relative_coverage, own.absolute_coverage);
    let shifted: Vec<f64> = values.iter().map(|v| v + 10.0).collect();
    let far = coverage(4, &values, 1.05, &shifted, 1.05);
    assert_eq!(far.absolute_coverage, 0.0);
    assert!((far.nominal - 0.9).abs() < 1e-12);
}

#[test]
fn icc_and_lag1_read_the_structure_they_are_built_for() {
    let strong: Vec<Vec<f64>> = (0..4)
        .map(|g| {
            (0..8)
                .map(|i| 10.0 * g as f64 + 0.01 * (i % 2) as f64)
                .collect()
        })
        .collect();
    assert!(
        icc(&strong) > 0.99,
        "constant groups: nearly all variance between"
    );
    let alternating: Vec<Vec<f64>> = (0..4)
        .map(|_| {
            (0..8)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect()
        })
        .collect();
    // Removing the position profile leaves zero residuals; a shifted
    // alternation survives the profile and anticorrelates at lag 1.
    let shifted: Vec<Vec<f64>> = (0..4)
        .map(|g| {
            (0..8)
                .map(|i| if (i + g) % 2 == 0 { 1.0 } else { -1.0 })
                .collect()
        })
        .collect();
    assert!(icc(&alternating) < 0.0 || icc(&alternating).abs() < 1e-9);
    assert!(lag1(&shifted) < -0.9, "{}", lag1(&shifted));
}

#[test]
fn a_tail_concentrated_in_one_sequence_clusters_above_the_null() {
    let mut r: Vec<Reading> = (0..8u32)
        .flat_map(|s| {
            (0..32u32).map(move |p| Reading {
                sequence: s,
                position: p,
                kl: 1e-5 + (s * 32 + p) as f64 * 1e-9,
                flips: (p % 3) as f64,
            })
        })
        .collect();
    for x in r.iter_mut().filter(|x| x.sequence == 3 && x.position >= 29) {
        x.kl = 1.0;
    }
    let d = describe(&r, 32, 50, SEED);
    assert_eq!(d.tail_size, 3);
    assert_eq!(d.tail_pairs_observed, 3);
    assert!(d.tail_clustering_ratio > 5.0, "{}", d.tail_clustering_ratio);
    assert_eq!(d.tail_share_by_position_quartile[3], 1.0);
    assert!((d.tail_share_by_position_quartile.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn the_fixture_stream_describes_without_a_nan() {
    let observations = stream(SEQUENCES, POSITIONS);
    let d = describe(&readings(&observations), POSITIONS as usize, 20, SEED);
    for v in [
        d.icc_log_kl,
        d.icc_flips,
        d.lag1_log_kl,
        d.lag1_flips,
        d.tail_clustering_ratio,
    ] {
        assert!(v.is_finite(), "{d:?}");
    }
}

#[test]
fn seeds_never_collide_across_banks_or_analyses() {
    let all = [
        seed_for(0, Analysis::Progressive),
        seed_for(0, Analysis::Independent),
        seed_for(0, Analysis::Clustering),
        seed_for(1, Analysis::Progressive),
        seed_for(1, Analysis::Independent),
        seed_for(1, Analysis::Clustering),
    ];
    let mut sorted = all.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), all.len());
}
