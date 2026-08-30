//! `tests` for [`super`].

use super::*;

fn gate() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v1".into(),
        positions_min: 512,
        kl_p99_max: 1e-3,
        top1_flip_max: 0,
        top10_change_max: 8,
        route_flip_max: 0,
        covered_mass_min: None,
    }
}

fn clean_bank() -> QualityBank {
    QualityBank {
        positions: 512,
        logits: LogitEvidence {
            kl_p50: 4.0e-5,
            kl_p95: 3.1e-4,
            kl_p99: 8.2e-4,
            max_logit_delta: 1.4e-2,
            top1_flips: 0,
            top10_changes: 3,
        },
        routing: RoutingEvidence {
            route_flips: 0,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: None,
        },
        min_covered_mass: None,
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
    }
}

#[test]
fn a_clean_bank_passes_and_names_the_gate_it_passed() {
    let e = QualityEvidence {
        gate: gate(),
        bank: clean_bank(),
    };
    assert!(e.verdict().passed());
    assert_eq!(e.proven_by(), Some("kimi-logit-v1"));
    assert_eq!(e.verdict().describe(), "passed kimi-logit-v1");
}

/// **The point of versioning.** The same measurements pass one gate and
/// fail a tighter one, so a claim is only meaningful with its gate id
/// attached.
#[test]
fn the_same_bank_passes_one_gate_and_fails_a_tighter_one() {
    let bank = clean_bank();
    assert!(gate().evaluate(&bank).passed());
    let tighter = QualityGate {
        id: "kimi-logit-v2".into(),
        kl_p99_max: 1e-4,
        top10_change_max: 0,
        ..gate()
    };
    let v = tighter.evaluate(&bank);
    assert!(!v.passed());
    assert_eq!(v.gate_id, "kimi-logit-v2");
    assert!(v.describe().contains("kl_p99"));
    assert!(v.describe().contains("top10_changes"));
}

/// A bank too small for its own tail statistic is refused on that
/// ground, not silently accepted because the p99 of nineteen samples
/// happened to look small.
#[test]
fn a_bank_shorter_than_the_gate_requires_is_refused() {
    let mut bank = clean_bank();
    bank.positions = 19;
    let v = gate().evaluate(&bank);
    assert!(!v.passed());
    assert_eq!(v.failures[0].0, Criterion::Positions);
    assert!(v.describe().contains("19 < 512"));
}

/// Every criterion is reported, not just the first, so one run says
/// everything that is wrong.
#[test]
fn all_failing_criteria_are_reported() {
    let mut bank = clean_bank();
    bank.logits.kl_p99 = 1.0;
    bank.logits.top1_flips = 4;
    bank.routing.route_flips = 11;
    let v = gate().evaluate(&bank);
    let names: Vec<&str> = v.failures.iter().map(|(c, _)| c.name()).collect();
    assert_eq!(names, vec!["kl_p99", "top1_flips", "route_flips"]);
}

/// **Routing and logit evidence stay apart.** Two banks with identical
/// logit movement are different findings when one of them rerouted, and
/// the precision-map response differs: more bits on the experts may fix
/// the first and be beside the point for the second.
#[test]
fn routing_movement_is_distinguishable_from_arithmetic_movement() {
    let arithmetic = QualityEvidence {
        gate: gate(),
        bank: clean_bank(),
    };
    let mut rerouted_bank = clean_bank();
    rerouted_bank.routing = RoutingEvidence {
        route_flips: 6,
        positions_with_route_change: 5,
        layers_with_route_change: 2,
        first_layer_with_route_change: None,
        route_margin: None,
        route_weight_mass_moved: None,
    };
    let rerouted = QualityEvidence {
        gate: gate(),
        bank: rerouted_bank,
    };

    assert_eq!(
        arithmetic.bank.logits, rerouted.bank.logits,
        "the two differ ONLY in routing"
    );
    assert!(arithmetic.is_arithmetic_only());
    assert!(!rerouted.is_arithmetic_only());
    assert!(arithmetic.verdict().passed());
    assert!(!rerouted.verdict().passed());
    assert_eq!(rerouted.verdict().failures[0].0, Criterion::RouteFlips);
}

/// The verdict is DERIVED, so a record cannot carry a passing claim over
/// failing numbers — there is no field to disagree with.
#[test]
fn a_verdict_cannot_be_stored_out_of_step_with_its_bank() {
    let mut e = QualityEvidence {
        gate: gate(),
        bank: clean_bank(),
    };
    assert!(e.proven_by().is_some());
    e.bank.logits.kl_p99 = 9.9;
    assert!(
        e.proven_by().is_none(),
        "editing the numbers must invalidate the claim immediately"
    );
    // And it survives a round trip as the numbers, not as a verdict.
    let json = serde_json::to_string(&e).expect("serialises");
    assert!(!json.contains("passed"), "no verdict is persisted");
    let back: QualityEvidence = serde_json::from_str(&json).expect("round trips");
    assert!(back.proven_by().is_none());
}

/// **`kimi-logit-v1` is a frozen contract.** Its identity and every
/// threshold are pinned here, so changing one fails this test and forces
/// the author to publish a v2 rather than re-date every claim that cited
/// v1.
#[test]
fn kimi_logit_v1_is_frozen() {
    let g = kimi_logit_v1();
    assert_eq!(g.id, "kimi-logit-v1");
    assert_eq!(g.positions_min, 4096);
    assert_eq!(g.kl_p99_max, 1e-3);
    assert_eq!(g.top1_flip_max, 8);
    assert_eq!(g.top10_change_max, 82);
    assert_eq!(g.route_flip_max, 82);
}

fn null_bank() -> QualityBank {
    QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: 0.0,
            kl_p95: 0.0,
            kl_p99: 0.0,
            max_logit_delta: 0.0,
            top1_flips: 0,
            top10_changes: 0,
        },
        routing: RoutingEvidence {
            route_flips: 0,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: None,
        },
        min_covered_mass: None,
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
    }
}

/// **The bar the gate itself has to clear.** A null arm — the reference
/// compared against itself — must pass with every count at zero. A gate
/// its own baseline cannot satisfy is measuring the harness, and no
/// candidate result through it would mean anything.
#[test]
fn the_null_arm_passes_kimi_logit_v1() {
    let v = kimi_logit_v1().evaluate(&null_bank());
    assert!(v.passed(), "{}", v.describe());
}

/// And it must still refuse a full-size bank that moved — otherwise
/// "the null arm passed" would be no evidence at all.
#[test]
fn kimi_logit_v1_refuses_a_full_size_bank_that_moved() {
    let mut moved = null_bank();
    moved.logits.kl_p99 = 2e-3;
    moved.logits.top1_flips = 9;
    let v = kimi_logit_v1().evaluate(&moved);
    assert!(!v.passed());
    let names: Vec<&str> = v.failures.iter().map(|(c, _)| c.name()).collect();
    assert_eq!(names, vec!["kl_p99", "top1_flips"]);

    // The thresholds are real edges: exactly at the bar passes.
    moved.logits.kl_p99 = 1e-3;
    moved.logits.top1_flips = 8;
    assert!(kimi_logit_v1().evaluate(&moved).passed());
}

/// **v2 is v1 plus a bank-validity criterion, and NOTHING else.**
///
/// Pinned field by field: a future edit that also re-tuned a threshold
/// while adding coverage would make "passed v2" mean two changes at
/// once, and no reader could tell which one a candidate cleared.
#[test]
fn v2_changes_only_the_coverage_criterion() {
    let (v1, v2) = (kimi_logit_v1(), kimi_logit_v2());
    assert_eq!(v2.id, "kimi-logit-v2");
    assert_ne!(v1.id, v2.id, "a new criterion needs a new id");
    assert_eq!(v2.positions_min, v1.positions_min);
    assert_eq!(v2.kl_p99_max, v1.kl_p99_max);
    assert_eq!(v2.top1_flip_max, v1.top1_flip_max);
    assert_eq!(v2.top10_change_max, v1.top10_change_max);
    assert_eq!(v2.route_flip_max, v1.route_flip_max);
    assert_eq!(v1.covered_mass_min, None, "v1 does not ask");
    assert_eq!(v2.covered_mass_min, Some(0.60));
}

/// A bank whose truncation was too narrow fails v2 and passes v1 — the
/// whole point of the new id.
#[test]
fn v2_refuses_a_bank_whose_kl_is_blind_to_the_distribution() {
    let perfect = QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: 0.0,
            kl_p95: 0.0,
            kl_p99: 0.0,
            max_logit_delta: 0.0,
            top1_flips: 0,
            top10_changes: 0,
        },
        routing: RoutingEvidence {
            route_flips: 0,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: None,
        },
        min_covered_mass: Some(0.307),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
    };
    assert!(
        kimi_logit_v1().evaluate(&perfect).passed(),
        "v1 asks nothing about coverage, so a narrow bank still passes it"
    );
    let verdict = kimi_logit_v2().evaluate(&perfect);
    assert!(!verdict.passed());
    assert!(verdict
        .failures
        .iter()
        .any(|(c, d)| *c == Criterion::CoveredMass && d.contains("0.3070")));

    // The measured wide truncation clears it.
    let wide = QualityBank {
        min_covered_mass: Some(0.729),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
        ..perfect.clone()
    };
    assert!(kimi_logit_v2().evaluate(&wide).passed());

    // A bank that did not RECORD its coverage fails: unknown coverage
    // is not evidence, and defaulting it to "wide enough" is exactly
    // the unfalsifiable claim this module exists to prevent.
    let silent = QualityBank {
        min_covered_mass: None,
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
        ..perfect
    };
    let verdict = kimi_logit_v2().evaluate(&silent);
    assert!(!verdict.passed());
    assert!(verdict
        .failures
        .iter()
        .any(|(c, d)| *c == Criterion::CoveredMass && d.contains("not recorded")));
    assert!(kimi_logit_v1().evaluate(&silent).passed());
}
