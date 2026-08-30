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
        });
    }
    let bank = b.finish();
    let worst = bank.min_covered_mass.expect("the builder records coverage");
    assert!(
        (worst - (-1.0f64).exp()).abs() < 1e-9,
        "the WORST position's mass, not the mean or the last: {worst}"
    );
}
