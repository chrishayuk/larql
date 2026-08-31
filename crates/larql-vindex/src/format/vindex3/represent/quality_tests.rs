//! `tests` for [`super`].

use super::*;

fn gate() -> QualityGate {
    QualityGate {
        id: "kimi-logit-v1".into(),
        positions_min: 512,
        kl_p99_max: 1e-3,
        top1_flip_max: Some(0),
        top10_change_max: Some(8),
        route_flip_max: Some(0),
        covered_mass_min: None,
        top1_mass_displaced_max: None,
        top10_mass_displaced_p99_max: None,
        route_mixture_mass_p99_max: None,
        route_mixture_mass_max: None,
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
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
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
        top10_change_max: Some(0),
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
    assert_eq!(g.top1_flip_max, Some(8));
    assert_eq!(g.top10_change_max, Some(82));
    assert_eq!(g.route_flip_max, Some(82));
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
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
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
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
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
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
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
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
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

/// **v3 judges CONSEQUENCE and drops counts as authority** — and that
/// is a tightening, not a loosening.
///
/// Built from the two banks the measurement actually produced: a
/// late-layer candidate whose many discrete changes are all near-ties,
/// and an early-layer one whose few-but-large mixture replacements v1
/// would have waved through inside its 82-flip allowance.
#[test]
fn v3_passes_marginal_churn_and_fails_material_displacement() {
    let dist = |min: f64, p99: f64, max: f64| Distribution {
        count: 100,
        min,
        p50: min,
        p95: p99,
        p99,
        max,
    };
    // Layer 26's real shape: 6 argmax flips giving up 0.1 % of
    // probability, 232 top-10 changes moving 0.33 % one rank, no
    // routing change at all.
    let marginal = QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: 2.2e-5,
            kl_p95: 2.7e-4,
            kl_p99: 6.1e-4,
            max_logit_delta: 0.218,
            top1_flips: 48,
            top10_changes: 1856,
        },
        routing: RoutingEvidence {
            route_flips: 0,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: None,
        },
        min_covered_mass: Some(0.729),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: Some(dist(0.0033, 0.030, 0.068)),
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: Some(dist(0.0010, 0.0040, 0.0049)),
    };
    assert!(
        kimi_logit_v3().evaluate(&marginal).passed(),
        "thousands of near-tie reorderings that move almost no mass must pass: {:?}",
        kimi_logit_v3().evaluate(&marginal).describe()
    );
    // v1 REJECTS the very same bank, on counts alone.
    let v1 = kimi_logit_v1().evaluate(&marginal);
    assert!(!v1.passed());
    assert!(v1
        .failures
        .iter()
        .all(|(c, _)| matches!(c, Criterion::Top1Flips | Criterion::Top10Changes)));

    // The opposite bank: FEW discrete changes, but they replace a
    // third of the routed mixture — layer 1's measured maximum.
    let material = QualityBank {
        logits: LogitEvidence {
            top1_flips: 4,
            top10_changes: 40,
            ..marginal.logits
        },
        routing: RoutingEvidence {
            route_flips: 40,
            positions_with_route_change: 40,
            layers_with_route_change: 25,
            first_layer_with_route_change: Some(2),
            route_margin: None,
            route_weight_mass_moved: Some(dist(0.08, 0.30, 0.361)),
        },
        ..marginal.clone()
    };
    assert!(
        kimi_logit_v1().evaluate(&material).passed(),
        "v1 waves this through: every count is inside its allowance"
    );
    let v3 = kimi_logit_v3().evaluate(&material);
    assert!(!v3.passed(), "v3 must refuse a large mixture replacement");
    assert!(v3
        .failures
        .iter()
        .any(|(c, _)| *c == Criterion::RouteDisplacement));
}

/// A consequence the gate asks for and the bank did not record FAILS —
/// unless nothing changed, in which case there is genuinely nothing to
/// displace.
#[test]
fn v3_refuses_unmeasured_consequence_but_not_an_absent_one() {
    let base = QualityBank {
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
        min_covered_mass: Some(0.9),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
    };
    assert!(
        kimi_logit_v3().evaluate(&base).passed(),
        "nothing changed, so nothing was displaced — that is not a missing measurement"
    );

    // Changes DID occur and no severity was recorded: refuse.
    let silent = QualityBank {
        logits: LogitEvidence {
            top1_flips: 3,
            top10_changes: 9,
            ..base.logits
        },
        routing: RoutingEvidence {
            route_flips: 5,
            ..base.routing
        },
        ..base
    };
    let v = kimi_logit_v3().evaluate(&silent);
    assert!(!v.passed());
    for c in [
        Criterion::Top1Displacement,
        Criterion::TopKDisplacement,
        Criterion::RouteDisplacement,
    ] {
        assert!(
            v.failures
                .iter()
                .any(|(k, d)| *k == c && d.contains("not recorded")),
            "{c:?} must refuse an unmeasured consequence: {v:?}"
        );
    }
}

/// **The report separates what DECIDED from what merely happened.**
///
/// The counts are the numbers a reader will find alarming, so they are
/// printed in full — under DIAGNOSTICS, beside the measured
/// consequence that actually decided. Hiding them would be worse than
/// either alternative.
#[test]
fn the_report_names_authority_and_diagnostics_separately() {
    let dist = |v: f64| Distribution {
        count: 10,
        min: v,
        p50: v,
        p95: v,
        p99: v,
        max: v,
    };
    let evidence = QualityEvidence {
        gate: kimi_logit_v3(),
        bank: QualityBank {
            positions: 8192,
            logits: LogitEvidence {
                kl_p50: 1e-5,
                kl_p95: 1e-4,
                kl_p99: 5e-4,
                max_logit_delta: 0.2,
                // Numbers that look alarming and decided nothing.
                top1_flips: 47,
                top10_changes: 1832,
            },
            routing: RoutingEvidence {
                route_flips: 211,
                positions_with_route_change: 180,
                layers_with_route_change: 1,
                first_layer_with_route_change: Some(26),
                route_margin: Some(dist(1e-3)),
                route_weight_mass_moved: Some(dist(0.08)),
            },
            min_covered_mass: Some(0.73),
            top10_margin: None,
            top10_candidate_margin: None,
            top10_mass_displaced: Some(dist(0.003)),
            top10_rank_displacement: None,
            top1_margin: None,
            top1_candidate_margin: None,
            top1_mass_displaced: Some(dist(0.001)),
        },
    };
    assert!(
        evidence.verdict().passed(),
        "the consequences are all tiny: {}",
        evidence.verdict().describe()
    );

    let report = evidence.report();
    assert!(report.contains("QUALITY_GATE: kimi-logit-v3"));
    // Every criterion the gate ASKS FOR is named under authority.
    for criterion in [
        "kl_p99",
        "covered_mass",
        "top1_mass_displaced",
        "top10_mass_displaced",
        "route_mass",
    ] {
        assert!(
            report.contains(criterion),
            "authority must name {criterion}"
        );
    }
    // v3 does not judge on counts, so they must NOT appear as authority.
    let authority = report
        .split("DIAGNOSTICS")
        .next()
        .expect("an authority section");
    assert!(
        !authority.contains("discrete_counts"),
        "v3 judges no counts, so none may be reported as authority:\n{authority}"
    );
    // But the alarming raw numbers ARE reported.
    for n in ["47", "1832", "211", "8192"] {
        assert!(report.contains(n), "diagnostics must report {n}:\n{report}");
    }
    assert!(report.contains("not authoritative"));
    assert!(report.contains("first changed layer    26"));

    // Under v1, the counts DO decide, and the report says so.
    let v1 = QualityEvidence {
        gate: kimi_logit_v1(),
        bank: evidence.bank.clone(),
    };
    assert!(v1.report().contains("discrete_counts"));
    assert!(!v1.verdict().passed(), "v1 rejects this bank on counts");
}

// ── kimi-logit-balanced-v1: the frozen calibration, as executable
//    anchors. Each bank below carries the MEASURED values of one
//    8,192-position characterization from the calibration ladder
//    (worst bank where two were measured), so the boundary the gate
//    draws is checked against the exact evidence it was drawn from. ──

fn dist(p99: f64, max: f64) -> Distribution {
    Distribution {
        count: 100,
        min: 1e-6,
        p50: p99 / 10.0,
        p95: p99 / 2.0,
        p99,
        max,
    }
}

/// One calibration anchor's authority-scale evidence.
fn anchor_bank(kl_p99: f64, top1_max: f64, top10_p99: f64, covered: f64) -> QualityBank {
    QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: kl_p99 / 100.0,
            kl_p95: kl_p99 / 3.0,
            kl_p99,
            max_logit_delta: 1.0,
            top1_flips: 30,
            top10_changes: 1000,
        },
        routing: RoutingEvidence {
            route_flips: 200,
            positions_with_route_change: 150,
            layers_with_route_change: 8,
            first_layer_with_route_change: Some(20),
            route_margin: Some(dist(5e-3, 5e-2)),
            route_weight_mass_moved: Some(dist(0.13, 0.21)),
        },
        min_covered_mass: Some(covered),
        top10_margin: Some(dist(0.1, 1.0)),
        top10_candidate_margin: Some(dist(0.1, 1.0)),
        top10_mass_displaced: Some(dist(top10_p99, top10_p99 * 2.0)),
        top10_rank_displacement: None,
        top1_margin: Some(dist(0.02, 0.5)),
        top1_candidate_margin: Some(dist(0.02, 0.5)),
        top1_mass_displaced: Some(dist(top1_max / 2.0, top1_max)),
    }
}

/// The ladder, exactly as measured: strict, wide and the flagship pass
/// balanced-v1; B3 — the map whose consequences changed character — is
/// refused on BOTH of the dimensions that define the boundary.
#[test]
fn balanced_v1_draws_the_line_where_the_ladder_did() {
    let g = kimi_logit_balanced_v1();
    // strict map: sel 5.99e-4 kl, worst top-1 give-up 0.020.
    assert!(g
        .evaluate(&anchor_bank(5.99e-4, 0.020, 6.2e-2, 0.631))
        .passed());
    // wide map, worst bank: kl 1.233e-3 (held-out), top1 0.055 (sel).
    assert!(g
        .evaluate(&anchor_bank(1.233e-3, 0.055, 7.5e-2, 0.577))
        .passed());
    // flagship, worst bank each dimension: kl 2.60e-3, top1 0.094.
    assert!(g
        .evaluate(&anchor_bank(2.60e-3, 0.094, 7.6e-2, 0.577))
        .passed());
    // B3: kl 4.74e-3, worst give-up 0.181 — refused on both.
    let v = g.evaluate(&anchor_bank(4.74e-3, 0.181, 6.2e-2, 0.631));
    assert!(!v.passed());
    let names: Vec<_> = v.failures.iter().map(|(c, _)| *c).collect();
    assert!(names.contains(&Criterion::KlP99));
    assert!(names.contains(&Criterion::Top1Displacement));
}

/// Balanced loosens NOTHING it has no evidence for: the route limits
/// are v3's own (measured non-discriminating in the corridor), the
/// positions floor stands, and the covered-mass floor moves only far
/// enough to admit the held-out bank's own flattest position.
#[test]
fn balanced_v1_inherits_what_the_corridor_did_not_discriminate() {
    let g = kimi_logit_balanced_v1();
    assert_eq!(
        g.route_mixture_mass_p99_max,
        kimi_logit_v3().route_mixture_mass_p99_max
    );
    assert_eq!(
        g.route_mixture_mass_max,
        kimi_logit_v3().route_mixture_mass_max
    );
    assert_eq!(g.positions_min, kimi_logit_v3().positions_min);
    assert_eq!(g.covered_mass_min, Some(0.55));
    // A diagnostic-scale bank still cannot pass balanced.
    let mut b = anchor_bank(1.0e-3, 0.03, 6e-2, 0.63);
    b.positions = 256;
    assert!(!g.evaluate(&b).passed());
    // And a blind instrument is still refused.
    assert!(!g.evaluate(&anchor_bank(1.0e-3, 0.03, 6e-2, 0.50)).passed());
}
