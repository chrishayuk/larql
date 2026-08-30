//! `tests` for [`super`].

use super::*;

/// A position where both arms agree exactly.
fn identical(seq: u32, pos: u32) -> PositionObservation {
    let logits = vec![9.0f32, 6.0, 5.0];
    // logsumexp over a vocabulary whose remaining mass is negligible.
    let lse = 9.0f32 + ((0.0f32).exp() + (-3.0f32).exp() + (-4.0f32).exp()).ln();
    PositionObservation {
        sequence: seq,
        position: pos,
        top_ids: vec![7, 3, 11],
        baseline_logits: logits.clone(),
        baseline_logsumexp: lse,
        candidate_logits: logits,
        candidate_logsumexp: lse,
        baseline_argmax: 7,
        candidate_argmax: 7,
        baseline_top10: vec![7, 3, 11],
        candidate_top10: vec![7, 3, 11],
        baseline_routes: vec![vec![4, 9], vec![1, 2]],
        candidate_routes: vec![vec![4, 9], vec![1, 2]],
        route_changes: Vec::new(),
        top10_change: None,
        top1_change: None,
    }
}

/// The instrument must read ZERO on an unchanged arm, or every number it
/// reports elsewhere is measuring itself.
#[test]
fn an_identical_candidate_produces_an_empty_bank() {
    let mut b = BankBuilder::new();
    for p in 0..8 {
        b.observe(&identical(0, p));
    }
    let bank = b.finish();
    assert_eq!(bank.positions, 8);
    assert!(bank.logits.kl_p99.abs() < 1e-12, "{}", bank.logits.kl_p99);
    assert_eq!(bank.logits.max_logit_delta, 0.0);
    assert_eq!(bank.logits.top1_flips, 0);
    assert_eq!(bank.logits.top10_changes, 0);
    assert_eq!(bank.routing.route_flips, 0);
    assert_eq!(bank.routing.layers_with_route_change, 0);
}

/// KL is computed between two genuine distributions, each normalised by
/// its OWN full-vocabulary logsumexp — not between two truncations.
#[test]
fn kl_matches_a_hand_computed_value() {
    let mut o = identical(0, 0);
    // Baseline p = softmax(9,6,5) restricted to these three; candidate
    // shifts one logit, keeping its own normaliser honest.
    o.candidate_logits = vec![9.0, 5.0, 5.0];
    let cl = 9.0f32 + ((0.0f32).exp() + (-4.0f32).exp() + (-4.0f32).exp()).ln();
    o.candidate_logsumexp = cl;

    let want: f64 = o
        .baseline_logits
        .iter()
        .zip(&o.candidate_logits)
        .map(|(b, c)| {
            let lp = (*b - o.baseline_logsumexp) as f64;
            let lq = (*c - o.candidate_logsumexp) as f64;
            lp.exp() * (lp - lq)
        })
        .sum();
    assert!((o.kl() - want).abs() < 1e-15);
    assert!(o.kl() > 0.0, "a moved distribution must have positive KL");
}

/// Truncation is visible: a flat distribution covers little mass, and
/// the builder carries the worst case so a blind KL cannot pass unnoticed.
#[test]
fn the_covered_mass_is_reported_and_falls_when_the_tail_is_fat() {
    let peaked = identical(0, 0);
    assert!(
        peaked.covered_mass() > 0.99,
        "peaked: {}",
        peaked.covered_mass()
    );

    let mut flat = identical(0, 1);
    // Three near-equal logits out of a vocabulary carrying far more.
    flat.baseline_logits = vec![1.0, 1.0, 1.0];
    flat.baseline_logsumexp = 1.0 + (1000.0f32).ln();
    assert!(flat.covered_mass() < 0.01, "flat: {}", flat.covered_mass());

    let mut b = BankBuilder::new();
    b.observe(&peaked);
    b.observe(&flat);
    assert!(b.min_covered_mass() < 0.01, "the worst case must survive");
}

/// **Routing is compared as a SET.** The router emits ties in a defined
/// order; two arms running the same experts in a different order have
/// not rerouted, and counting that as a flip would send the precision
/// search after a difference that does not exist.
#[test]
fn a_reordered_route_is_not_a_route_change() {
    let mut o = identical(0, 0);
    o.candidate_routes = vec![vec![9, 4], vec![2, 1]];
    assert_eq!(o.route_changes(), (0, 0));

    o.candidate_routes = vec![vec![9, 5], vec![2, 1]];
    assert_eq!(o.route_changes(), (1, 1), "one layer, one substituted id");
}

/// Route movement is attributable to depth, and a burst in one position
/// is distinguishable from the same count spread across many.
#[test]
fn routing_movement_is_attributed_to_positions_and_layers() {
    let mut b = BankBuilder::new();
    // Two positions, both rerouting in layer 1 only.
    for p in 0..2u32 {
        let mut o = identical(0, p);
        o.candidate_routes = vec![vec![4, 9], vec![1, 5]];
        b.observe(&o);
    }
    // Six clean positions.
    for p in 2..8u32 {
        b.observe(&identical(0, p));
    }
    let bank = b.finish();
    assert_eq!(bank.routing.route_flips, 2);
    assert_eq!(bank.routing.positions_with_route_change, 2);
    assert_eq!(
        bank.routing.layers_with_route_change, 1,
        "both flips were the same layer, which is what a depth-scoped fix needs to know"
    );
}

/// An argmax that leaves the recorded top-N still counts as a flip —
/// which is the case that matters most, so it must not be derived from
/// the truncation.
#[test]
fn a_top1_flip_outside_the_recorded_ids_is_still_counted() {
    let mut o = identical(0, 0);
    o.candidate_argmax = 4096;
    assert!(!o.top_ids.contains(&4096));
    assert!(o.top1_flipped());
}

/// Nearest-rank percentiles name a value some position actually
/// produced, rather than interpolating one nobody saw.
#[test]
fn percentiles_are_nearest_rank_over_real_observations() {
    let mut b = BankBuilder::new();
    for (p, cand) in (0..100u32).zip(0..100) {
        let mut o = identical(0, p);
        // A spread of divergences, one clearly worst.
        o.candidate_logits = vec![9.0 - cand as f32 * 0.01, 6.0, 5.0];
        b.observe(&o);
    }
    let bank = b.finish();
    assert!(bank.logits.kl_p50 > 0.0);
    assert!(bank.logits.kl_p95 >= bank.logits.kl_p50);
    assert!(bank.logits.kl_p99 >= bank.logits.kl_p95);
    assert_eq!(bank.positions, 100);
}

/// An empty bank is a bank of zero positions, not a bank that passed.
/// `QualityGate::positions_min` is what refuses it, and it must have a
/// count to refuse.
#[test]
fn an_empty_bank_reports_zero_positions() {
    let bank = BankBuilder::new().finish();
    assert_eq!(bank.positions, 0);
    assert_eq!(bank.logits.kl_p99, 0.0);
}

/// **The shallowest layer that moved is recorded**, because a count of
/// affected layers cannot separate the two failure modes.
///
/// A perturbed layer changing its OWN routing means its experts were
/// selected differently; a perturbed layer leaving its own routing
/// intact while LATER ones move is a cascade through the residual
/// stream, and the response to each is different.
#[test]
fn the_first_layer_that_reroutes_is_recorded_not_just_the_count() {
    let observe = |baseline: Vec<Vec<u32>>, candidate: Vec<Vec<u32>>| {
        let mut b = BankBuilder::new();
        b.observe(&PositionObservation {
            sequence: 0,
            position: 0,
            top_ids: vec![0],
            baseline_logits: vec![1.0],
            baseline_logsumexp: 1.0,
            candidate_logits: vec![1.0],
            candidate_logsumexp: 1.0,
            baseline_argmax: 0,
            candidate_argmax: 0,
            baseline_top10: vec![0],
            candidate_top10: vec![0],
            baseline_routes: baseline,
            candidate_routes: candidate,
            route_changes: Vec::new(),
            top10_change: None,
            top1_change: None,
        });
        b.finish().routing
    };

    // A CASCADE: the perturbed layer 0 keeps its own experts, later
    // layers move. This is Kimi layer 1 at Q6_K.
    let cascade = observe(
        vec![vec![1, 2], vec![3, 4], vec![5, 6]],
        vec![vec![2, 1], vec![3, 9], vec![7, 6]],
    );
    assert_eq!(cascade.first_layer_with_route_change, Some(1));
    assert_eq!(cascade.layers_with_route_change, 2);

    // LOCAL: the perturbed layer itself reroutes.
    let local = observe(vec![vec![1, 2], vec![3, 4]], vec![vec![1, 8], vec![3, 4]]);
    assert_eq!(local.first_layer_with_route_change, Some(0));
    assert_eq!(local.layers_with_route_change, 1);

    // The two are indistinguishable by count alone, which is why the
    // shallowest layer is carried.
    let same_count = observe(vec![vec![1, 2], vec![3, 4]], vec![vec![1, 2], vec![3, 9]]);
    assert_eq!(
        same_count.layers_with_route_change,
        local.layers_with_route_change
    );
    assert_ne!(
        same_count.first_layer_with_route_change,
        local.first_layer_with_route_change
    );

    // Nothing moved: no layer to name.
    let stable = observe(vec![vec![1, 2]], vec![vec![2, 1]]);
    assert_eq!(stable.first_layer_with_route_change, None);
    assert_eq!(stable.route_flips, 0, "reordering is not a reroute");
}

/// The bank carries the worst position's covered mass, so a gate can
/// judge whether its KL saw enough of the distribution.
#[test]
fn the_bank_carries_the_worst_covered_mass_it_saw() {
    let mut b = BankBuilder::new();
    // Two positions: one sharp, one flat. `logsumexp` is set so the
    // covered mass is computable and different for each.
    for (logit, lse) in [(0.0f32, 0.0f32), (0.0, 1.0)] {
        b.observe(&PositionObservation {
            sequence: 0,
            position: 0,
            top_ids: vec![0],
            baseline_logits: vec![logit],
            baseline_logsumexp: lse,
            candidate_logits: vec![logit],
            candidate_logsumexp: lse,
            baseline_argmax: 0,
            candidate_argmax: 0,
            baseline_top10: vec![0],
            candidate_top10: vec![0],
            baseline_routes: vec![],
            candidate_routes: vec![],
            route_changes: Vec::new(),
            top10_change: None,
            top1_change: None,
        });
    }
    let bank = b.finish();
    let worst = bank.min_covered_mass.expect("the builder records coverage");
    assert!(
        (worst - (-1.0f64).exp()).abs() < 1e-9,
        "the WORST position's mass, not the mean or the last: {worst}"
    );
}

/// **A routing change is WEIGHED, not counted**: how close the decision
/// was, and how much mixture mass actually moved.
///
/// The two questions a flip count cannot answer, and they dissociate: a
/// swap of the lowest-weighted expert for its near-equal neighbour and
/// an overturned high-weight decision are both "one flip".
#[test]
fn a_route_change_carries_its_margin_and_the_mass_it_moved() {
    // Layer 0's route changes; layer 1's does not. Scores are the
    // BIASED selection scores the router ranked by.
    let scores = vec![vec![0.90, 0.80, 0.7999, 0.10], vec![0.90, 0.80, 0.10, 0.05]];
    let base_routes = vec![vec![0u32, 1], vec![0, 1]];
    let cand_routes = vec![vec![0u32, 2], vec![0, 1]];
    // Equal weights: the swap moves the smallest possible mass.
    let base_w = vec![vec![0.5, 0.5, 1.0], vec![0.5, 0.5, 1.0]];
    let cand_w = base_w.clone();

    let changes = PositionObservation::weigh_route_changes(
        &base_routes,
        &cand_routes,
        &scores,
        &base_w,
        &cand_w,
    );
    assert_eq!(changes.len(), 1, "only the layer that changed is reported");
    assert_eq!(changes[0].layer, 0);
    // Expert 1 (0.80) was the lowest SELECTED; expert 2 (0.7999) the
    // best unselected — a near-tie, and the margin says so.
    assert!(
        (changes[0].boundary_margin - 0.0001).abs() < 1e-5,
        "margin {} is not the near-tie gap",
        changes[0].boundary_margin
    );
    // Half the L1 between two normalised mixtures that put equal mass
    // on a different second expert: |0.5| moved out, |0.5| moved in.
    assert!(
        (changes[0].weight_mass_moved - 0.5).abs() < 1e-5,
        "mass {} ",
        changes[0].weight_mass_moved
    );

    // The SAME flip count with a confidently-held decision and an
    // unequal mixture reports a large margin and less moved mass.
    let confident = vec![vec![0.90, 0.80, 0.20, 0.10]];
    let heavy = vec![vec![0.9, 0.1, 1.0]];
    let changes = PositionObservation::weigh_route_changes(
        &[vec![0, 1]],
        &[vec![0, 2]],
        &confident,
        &heavy,
        &heavy,
    );
    assert_eq!(changes.len(), 1);
    assert!(
        changes[0].boundary_margin > 0.5,
        "an overturned confident decision has a LARGE margin, got {}",
        changes[0].boundary_margin
    );
    assert!(
        changes[0].weight_mass_moved < 0.2,
        "swapping the light expert moves little mass, got {}",
        changes[0].weight_mass_moved
    );

    // Without score evidence the severity is NaN and the builder drops
    // it rather than folding a fabricated zero into the distribution.
    let blind = PositionObservation::weigh_route_changes(&base_routes, &cand_routes, &[], &[], &[]);
    assert_eq!(blind.len(), 1);
    assert!(blind[0].boundary_margin.is_nan());
    assert!(blind[0].weight_mass_moved.is_nan());
}

/// A distribution reports the SHAPE, and refuses to invent one from
/// nothing.
#[test]
fn a_distribution_reports_shape_and_declines_when_empty() {
    use crate::format::vindex3::represent::quality::Distribution;

    assert_eq!(Distribution::of(&mut []), None, "no observations, no shape");
    let mut one = [4.0];
    let d = Distribution::of(&mut one).expect("one observation is a shape");
    assert_eq!((d.count, d.min, d.p50, d.max), (1, 4.0, 4.0, 4.0));

    let mut many: Vec<f64> = (1..=100).map(f64::from).collect();
    let d = Distribution::of(&mut many).expect("shape");
    assert_eq!(d.count, 100);
    assert_eq!(d.min, 1.0);
    assert_eq!(d.max, 100.0);
    // Nearest-rank: every reported value is one an observation produced.
    assert_eq!(d.p50, 50.0);
    assert_eq!(d.p95, 95.0);
    assert_eq!(d.p99, 99.0);
}

/// **Severity reaches the bank as DISTRIBUTIONS**, and unmeasured
/// severity is dropped rather than folded in as a number.
#[test]
fn severity_accumulates_into_distributions_and_drops_the_unmeasured() {
    use crate::format::vindex3::represent::bank::{Top1Change, TopKChange};

    let obs = |margin: f32, mass: f32, top1: f32, topk: f32| PositionObservation {
        route_changes: vec![RouteChange {
            layer: 3,
            boundary_margin: margin,
            weight_mass_moved: mass,
        }],
        top1_change: Some(Top1Change {
            boundary_margin: 0.01,
            candidate_margin_same_ids: 0.01,
            mass_displaced: top1,
        }),
        top10_change: Some(TopKChange {
            boundary_margin: 0.07,
            candidate_margin_same_ids: 0.07,
            mass_displaced: topk,
            max_rank_displacement: 1,
        }),
        ..identical(0, 0)
    };

    let mut b = BankBuilder::new();
    b.observe(&obs(1e-3, 0.05, 0.001, 0.002));
    b.observe(&obs(2e-3, 0.09, 0.004, 0.006));
    // A change whose severity was NOT measured: NaN, and it must not
    // become a zero in any distribution.
    b.observe(&obs(f32::NAN, f32::NAN, f32::NAN, f32::NAN));
    let bank = b.finish();

    let route = bank.routing.route_margin.expect("margins recorded");
    assert_eq!(
        route.count, 2,
        "the unmeasured change is dropped, not zeroed"
    );
    // f32 severities widened to f64, so compare with tolerance.
    assert!((route.min - 1e-3).abs() < 1e-9 && (route.max - 2e-3).abs() < 1e-9);
    let mass = bank.routing.route_weight_mass_moved.expect("mass recorded");
    assert_eq!(mass.count, 2);
    assert!((mass.max - 0.09).abs() < 1e-6);

    let t1 = bank.top1_mass_displaced.expect("top1 recorded");
    assert_eq!(t1.count, 2);
    assert!((t1.max - 0.004).abs() < 1e-6);
    let tk = bank.top10_mass_displaced.expect("topk recorded");
    assert_eq!(tk.count, 2);
    assert!((tk.max - 0.006).abs() < 1e-6);
    // The rank move is an integer count and is always recorded.
    assert_eq!(
        bank.top10_rank_displacement.expect("ranks recorded").count,
        3
    );

    // And a bank with no severity at all reports none rather than zero.
    let mut plain = BankBuilder::new();
    plain.observe(&identical(0, 0));
    let bank = plain.finish();
    assert!(bank.routing.route_margin.is_none());
    assert!(bank.top1_mass_displaced.is_none());
}

/// A projection whose two arms put ALL their weight on one expert each
/// moves the whole mixture; one that cannot be normalised reports NaN
/// rather than a fabricated zero.
#[test]
fn mixture_distance_spans_its_whole_range_and_declines_when_undefined() {
    // Disjoint single-expert mixtures: everything moved.
    let changes = PositionObservation::weigh_route_changes(
        &[vec![0]],
        &[vec![1]],
        &[vec![0.9, 0.8]],
        &[vec![1.0, 1.0]],
        &[vec![1.0, 1.0]],
    );
    assert_eq!(changes.len(), 1);
    assert!(
        (changes[0].weight_mass_moved - 1.0).abs() < 1e-6,
        "disjoint mixtures are a complete replacement, got {}",
        changes[0].weight_mass_moved
    );

    // All-zero weights cannot be normalised into a mixture at all.
    let changes = PositionObservation::weigh_route_changes(
        &[vec![0]],
        &[vec![1]],
        &[vec![0.9, 0.8]],
        &[vec![0.0, 0.0]],
        &[vec![0.0, 0.0]],
    );
    assert!(changes[0].weight_mass_moved.is_nan());
}
