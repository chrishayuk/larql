//! Exhaustive over the vocabulary: every statistic reads off a bank,
//! names itself, and knows which way is better. A closed vocabulary
//! deserves a closed test — a wildcard arm here would let a new variant
//! ship with no direction and no reader.

use super::super::diagnostic::tests::guard_256;
use super::super::quality::Distribution;
use super::*;

/// Every variant, written out. Adding one to the enum without adding it
/// here fails the count assertion below rather than passing silently.
const ALL: [Statistic; 11] = [
    Statistic::KlP99,
    Statistic::Top1Flips,
    Statistic::Top10Changes,
    Statistic::RouteFlips,
    Statistic::RouteFlipRate,
    Statistic::Top1MassDisplaced,
    Statistic::Top10MassDisplacedP99,
    Statistic::RouteMixtureMassP99,
    Statistic::RouteMixtureMassMax,
    Statistic::Positions,
    Statistic::CoveredMass,
];

fn dist(p99: f64, max: f64, count: u64) -> Distribution {
    Distribution {
        count,
        min: 0.0,
        p50: 0.0,
        p95: 0.0,
        p99,
        max,
    }
}

/// A bank with every optional field POPULATED, so no `observe` arm can
/// pass by returning `None`.
fn full_bank() -> QualityBank {
    let mut b = guard_256();
    b.top1_mass_displaced = Some(dist(0.004, 0.0053, 2));
    b.top10_mass_displaced = Some(dist(0.0789, 0.0789, 74));
    b.routing.route_weight_mass_moved = Some(dist(0.1028, 0.1028, 46));
    b
}

#[test]
fn every_statistic_reads_a_value_off_a_populated_bank() {
    let b = full_bank();
    for s in ALL {
        let (v, _) = s.observe(&b);
        assert!(v.is_some(), "{s} read nothing off a fully populated bank");
    }
    assert_eq!(ALL.len(), 11, "a new Statistic must be added to ALL");
}

#[test]
fn only_percentiles_carry_tail_support() {
    let b = full_bank();
    for s in ALL {
        let (_, tail) = s.observe(&b);
        let expected = matches!(
            s,
            Statistic::KlP99 | Statistic::Top10MassDisplacedP99 | Statistic::RouteMixtureMassP99
        );
        assert_eq!(tail.is_some(), expected, "{s} tail support");
    }
}

#[test]
fn an_unrecorded_optional_reads_none_rather_than_zero() {
    let mut b = full_bank();
    b.top1_mass_displaced = None;
    b.top10_mass_displaced = None;
    b.routing.route_weight_mass_moved = None;
    b.min_covered_mass = None;
    for s in [
        Statistic::Top1MassDisplaced,
        Statistic::Top10MassDisplacedP99,
        Statistic::RouteMixtureMassP99,
        Statistic::RouteMixtureMassMax,
        Statistic::CoveredMass,
    ] {
        assert_eq!(s.observe(&b).0, None, "{s} must be None, never 0.0");
    }
}

#[test]
fn the_rate_is_flips_over_positions_and_refuses_an_empty_bank() {
    let mut b = full_bank();
    b.routing.route_flips = 46;
    b.positions = 256;
    assert_eq!(
        Statistic::RouteFlipRate.observe(&b).0,
        Some(46.0 / 256.0),
        "the RATE, not the count"
    );
    b.positions = 0;
    assert_eq!(
        Statistic::RouteFlipRate.observe(&b).0,
        None,
        "no positions is not a rate of zero"
    );
}

#[test]
fn costs_are_lower_better_and_sufficiencies_are_higher_better() {
    for s in ALL {
        let expected = match s {
            Statistic::Positions | Statistic::CoveredMass => Better::Higher,
            _ => Better::Lower,
        };
        assert_eq!(s.better(), expected, "{s}");
    }
}

#[test]
fn order_reports_which_is_better_in_both_directions() {
    use std::cmp::Ordering;
    // A cost: less movement wins.
    assert_eq!(Statistic::KlP99.order(1.0, 2.0), Ordering::Less);
    assert_eq!(Statistic::KlP99.order(2.0, 1.0), Ordering::Greater);
    assert_eq!(Statistic::KlP99.order(1.0, 1.0), Ordering::Equal);
    // A sufficiency: more evidence wins.
    assert_eq!(Statistic::CoveredMass.order(0.8, 0.6), Ordering::Less);
    assert_eq!(Statistic::CoveredMass.order(0.6, 0.8), Ordering::Greater);
    assert_eq!(Statistic::Positions.order(8192.0, 256.0), Ordering::Less);
}

#[test]
fn every_label_is_distinct_and_display_matches_it() {
    let labels: std::collections::BTreeSet<&str> = ALL.iter().map(|s| s.label()).collect();
    assert_eq!(labels.len(), ALL.len(), "labels must identify uniquely");
    for s in ALL {
        assert_eq!(s.to_string(), s.label());
    }
}
