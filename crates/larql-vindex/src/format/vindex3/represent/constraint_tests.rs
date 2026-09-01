//! `tests` for [`super`].
//!
//! The banks here are not invented. `four_family_selection` and
//! `four_family_heldout` carry the numbers the composed Kimi map
//! actually measured at 8,192 positions on 2026-08-31, and
//! `flat_instrument` carries the numbers one of the two catalogued
//! blind-instrument episodes produced. A type whose job is to say which
//! resource is scarce should be tested against the measurement that
//! made the question worth asking.

use super::super::quality::Statistic;
use super::super::quality::{
    kimi_logit_balanced_v1, kimi_logit_v1, Distribution, LogitEvidence, QualityBank, QualityGate,
    RoutingEvidence,
};
use super::*;

fn dist(p99: f64, max: f64) -> Option<Distribution> {
    Some(Distribution {
        count: 1,
        min: 0.0,
        p50: 0.0,
        p95: 0.0,
        p99,
        max,
    })
}

/// The four-family map on the SELECTION bank: experts L20-26 x KDA
/// {20,21,22,24,25} x MLA{23,26} x output head, all Q8_0, 8,192
/// positions. `kimi_full4-selection-8192_report.json`.
fn four_family_selection() -> QualityBank {
    QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p50: 0.0,
            kl_p95: 0.0,
            kl_p99: 2.3791e-3,
            max_logit_delta: 0.0,
            top1_flips: 129,
            top10_changes: 2527,
        },
        routing: RoutingEvidence {
            route_flips: 1305,
            positions_with_route_change: 0,
            layers_with_route_change: 0,
            first_layer_with_route_change: None,
            route_margin: None,
            route_weight_mass_moved: dist(0.1248, 0.1993),
        },
        min_covered_mass: Some(0.6315),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: dist(0.064652, 0.064652),
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: dist(0.055463, 0.055463),
    }
}

/// The same map on the HELD-OUT bank, which chose no scope.
/// `kimi_full4-heldout-8192_report.json`.
fn four_family_heldout() -> QualityBank {
    QualityBank {
        positions: 8192,
        logits: LogitEvidence {
            kl_p99: 2.6174e-3,
            top1_flips: 122,
            top10_changes: 2341,
            ..four_family_selection().logits
        },
        routing: RoutingEvidence {
            route_flips: 1270,
            route_weight_mass_moved: dist(0.1257, 0.2099),
            ..four_family_selection().routing
        },
        min_covered_mass: Some(0.5773),
        top10_mass_displaced: dist(0.068805, 0.068805),
        top1_mass_displaced: dist(0.058313, 0.058313),
        ..four_family_selection()
    }
}

/// One of the two catalogued blind-instrument episodes: the GPU
/// completed command buffers without executing them, so both arms were
/// constant and coverage sat at the uniform floor `TOP_N/vocab`
/// (2048/163840). Every CEILING is perfect. Only a floor catches it.
fn flat_instrument() -> QualityBank {
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
        min_covered_mass: Some(2048.0 / 163_840.0),
        top10_margin: None,
        top10_candidate_margin: None,
        top10_mass_displaced: None,
        top10_rank_displacement: None,
        top1_margin: None,
        top1_candidate_margin: None,
        top1_mass_displaced: None,
    }
}

// ── The finding this type exists for. ──

#[test]
fn route_movement_is_the_binding_constraint_on_the_four_family_map() {
    for (name, bank) in [
        ("selection", four_family_selection()),
        ("held-out", four_family_heldout()),
    ] {
        let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &bank);
        let binding = v.binding().expect("every ceiling was measured");
        assert_eq!(
            binding.criterion,
            Criterion::RouteDisplacement,
            "{name}: the scarce resource is routing, not KL"
        );
        assert!(
            binding.utilisation().expect("scored") > 0.8,
            "{name}: routing should sit above 80% of its limit"
        );
    }
}

#[test]
fn kl_has_far_more_headroom_than_routing_does() {
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &four_family_selection());
    let kl = v.utilisation_of(Criterion::KlP99).expect("kl scored");
    let route = v
        .utilisation_of(Criterion::RouteDisplacement)
        .expect("route scored");
    // 2.3791e-3 / 3.5e-3 and 0.1248 / 0.15.
    assert!((kl - 0.6797).abs() < 1e-3, "kl utilisation was {kl}");
    assert!(
        (route - 0.8320).abs() < 1e-3,
        "route utilisation was {route}"
    );
    assert!(
        route > kl,
        "the scalar 'quality cost' framing hides exactly this ordering"
    );
}

#[test]
fn a_criterion_carrying_two_limits_reports_the_worse_one() {
    // balanced-v1 judges routed mixture mass at BOTH p99 (0.15) and max
    // (0.25). On the held-out bank p99 is at 83.8% and max at 84.0%, so
    // the coarser `Criterion` key must not average or pick the first.
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &four_family_heldout());
    let worst = v
        .utilisation_of(Criterion::RouteDisplacement)
        .expect("scored");
    let both: Vec<f64> = v
        .spendable()
        .filter(|m| m.criterion == Criterion::RouteDisplacement)
        .filter_map(|m| m.utilisation())
        .collect();
    assert_eq!(both.len(), 2, "both route limits are judged");
    assert!((worst - both.iter().cloned().fold(f64::MIN, f64::max)).abs() < 1e-12);
}

// ── The floors are not budgets. ──

#[test]
fn a_blind_instrument_passes_every_ceiling_and_is_still_refused() {
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &flat_instrument());
    for m in v.spendable() {
        assert!(
            m.satisfied(),
            "{}: a constant-logit run spends nothing",
            m.what
        );
    }
    assert!(
        !v.sound(),
        "coverage at TOP_N/vocab is the uniform floor — the run saw nothing"
    );
    assert!(
        !v.admissible(),
        "a perfect score on every budget must not be admissible when the \
         measurement itself is unsound"
    );
}

#[test]
fn binding_ranks_ceilings_only_and_never_reports_a_floor() {
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &flat_instrument());
    // Coverage is at 1.25% of a 55% floor — catastrophically the worst
    // number in the vector — and must still not be "the binding
    // constraint", which names a resource to spend less of.
    let binding = v.binding().expect("ceilings were scored");
    assert_ne!(binding.criterion, Criterion::CoveredMass);
    assert_eq!(binding.kind, LimitKind::Ceiling);
}

#[test]
fn a_sound_measurement_clears_both_floors() {
    for bank in [four_family_selection(), four_family_heldout()] {
        let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &bank);
        assert!(v.sound());
        assert!(v.admissible());
    }
}

// ── Unmeasured is not zero. ──

#[test]
fn an_unmeasured_consequence_is_not_ranked_and_not_satisfied() {
    let mut bank = four_family_selection();
    // Flips happened, so a displacement magnitude SHOULD exist; the
    // bank simply failed to record it. That is missing evidence.
    bank.top1_mass_displaced = None;
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &bank);
    let m = v
        .spendable()
        .find(|m| m.criterion == Criterion::Top1Displacement)
        .expect("the gate judges it");
    assert!(!m.vacuous, "129 top-1 flips is not 'nothing changed'");
    assert_eq!(m.utilisation(), None, "unmeasured is not zero");
    assert!(!m.satisfied(), "silence is not a pass");
    assert!(!v.admissible());
}

#[test]
fn a_vacuous_consequence_costs_nothing() {
    let mut bank = four_family_selection();
    // Nothing routed differently, so there is no magnitude to record.
    bank.routing.route_flips = 0;
    bank.routing.route_weight_mass_moved = None;
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &bank);
    for m in v
        .spendable()
        .filter(|m| m.criterion == Criterion::RouteDisplacement)
    {
        assert!(m.vacuous);
        assert_eq!(m.utilisation(), Some(0.0));
        assert!(m.satisfied());
    }
    // With routing spending nothing, the scarce resource moves.
    assert_eq!(
        v.binding().expect("scored").criterion,
        Criterion::KlP99,
        "the binding criterion is a property of the candidate, not a constant"
    );
}

// ── The drift guard. ──

#[test]
fn the_margin_table_agrees_with_the_gate_criterion_for_criterion() {
    let gate = kimi_logit_balanced_v1();
    for (name, bank) in [
        ("four-family selection", four_family_selection()),
        ("four-family held-out", four_family_heldout()),
        ("flat instrument", flat_instrument()),
        ("unmeasured displacement", {
            let mut b = four_family_selection();
            b.top1_mass_displaced = None;
            b
        }),
        ("short bank", {
            let mut b = four_family_selection();
            b.positions = 256;
            b
        }),
    ] {
        let v = ConstraintVector::of(&gate, &bank);
        let verdict = gate.evaluate(&bank);
        assert_eq!(
            v.admissible(),
            verdict.passed(),
            "{name}: the vector and the gate disagree on the verdict"
        );
        for m in &v.margins {
            let gate_failed = verdict.failures.iter().any(|(c, _)| *c == m.criterion);
            let margin_failed = v
                .margins
                .iter()
                .filter(|o| o.criterion == m.criterion)
                .any(|o| !o.satisfied());
            assert_eq!(
                margin_failed, gate_failed,
                "{name}: {} disagrees with the gate",
                m.what
            );
        }
    }
}

#[test]
fn every_criterion_the_gate_judges_appears_in_the_vector() {
    let gate = kimi_logit_balanced_v1();
    let v = ConstraintVector::of(&gate, &four_family_selection());
    // balanced-v1: kl, three displacement limits (route twice),
    // positions, covered mass. It judges no raw counts.
    assert_eq!(v.spendable().count(), 5);
    assert_eq!(v.margins.len(), 7);
    assert!(v
        .margins
        .iter()
        .all(|m| m.criterion != Criterion::Top1Flips));
}

// ── The accessors a search actually ranks with. ──

#[test]
fn headroom_is_what_is_left_of_a_budget_and_floors_have_none() {
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &four_family_selection());
    let route = v
        .spendable()
        .find(|m| m.what == Statistic::RouteMixtureMassP99)
        .expect("judged");
    // 1 - 0.1248/0.15: about a sixth of the route budget is left, which
    // is the number a search should be allocating against.
    let left = route.headroom().expect("a ceiling has headroom");
    assert!((left - 0.1680).abs() < 1e-3, "headroom was {left}");
    assert!((left + route.utilisation().expect("scored") - 1.0).abs() < 1e-12);

    // A floor is not a budget: there is nothing to spend and nothing
    // left, and reporting a number here would invite ranking against it.
    let floor = v
        .margins
        .iter()
        .find(|m| m.criterion == Criterion::CoveredMass)
        .expect("balanced-v1 asks for coverage");
    assert_eq!(floor.kind, LimitKind::Floor);
    assert_eq!(floor.utilisation(), None);
    assert_eq!(floor.headroom(), None);
    assert!(floor.satisfied(), "0.6315 clears the 0.55 floor");
}

#[test]
fn headroom_goes_negative_once_a_budget_is_overspent() {
    let mut bank = four_family_selection();
    // B3 (experts 16-26) — the map whose consequences changed character,
    // and the candidate balanced-v1 was calibrated to refuse.
    bank.logits.kl_p99 = 4.735e-3;
    let v = ConstraintVector::of(&kimi_logit_balanced_v1(), &bank);
    let kl = v
        .spendable()
        .find(|m| m.criterion == Criterion::KlP99)
        .expect("judged");
    assert!(kl.utilisation().expect("scored") > 1.0);
    assert!(kl.headroom().expect("scored") < 0.0);
    assert!(!kl.satisfied());
    assert!(!v.admissible());
    // Overspent, so it is also the scarcest thing in the vector.
    assert_eq!(v.binding().expect("scored").criterion, Criterion::KlP99);
}

#[test]
fn a_zero_budget_is_judged_but_never_ranked() {
    // No gate in this crate sets a zero ceiling, and the type must not
    // assume that stays true. `observed / 0.0` is either infinity or —
    // when nothing was observed either — NaN, and a NaN utilisation
    // sorts arbitrarily against every other criterion, which would
    // silently mis-rank the whole vector rather than fail. A zero
    // budget is therefore JUDGED but never RANKED.
    let gate = QualityGate {
        top1_flip_max: Some(0),
        route_flip_max: Some(0),
        ..kimi_logit_v1()
    };
    let v = ConstraintVector::of(&gate, &four_family_selection());
    for what in [Statistic::Top1Flips, Statistic::RouteFlips] {
        let m = v.spendable().find(|m| m.what == what).expect("judged");
        assert_eq!(m.limit, 0.0);
        assert_eq!(
            m.utilisation(),
            None,
            "{what}: a fraction of a zero budget is not a number"
        );
        assert_eq!(m.headroom(), None);
        assert!(!m.satisfied(), "{what}: 129 and 1305 are over zero");
    }
    assert!(!v.admissible());
    // Ranking still works off the criteria that CAN be scored, instead
    // of collapsing because one of them is unrankable. Under v1 that is
    // top-10 CHANGES — 2527 against a limit of 82, thirty-one times
    // over, where kl is only 2.4x over. Which is the count-based
    // gate's whole problem, and why v3 stopped judging on counts: the
    // scarce resource it names is the one that cannot tell a near-tie
    // reordering from an overturned decision.
    assert_eq!(
        v.binding().expect("scored").criterion,
        Criterion::Top10Changes
    );
}

#[test]
fn a_zero_budget_that_was_never_spent_does_not_poison_the_ranking() {
    // The NaN case specifically: limit 0 AND observed 0. Satisfied —
    // nothing was spent — but still unrankable, and `binding()` must
    // return a real criterion rather than this one.
    let gate = QualityGate {
        top1_flip_max: Some(0),
        ..kimi_logit_v1()
    };
    let mut bank = four_family_selection();
    bank.logits.top1_flips = 0;
    let v = ConstraintVector::of(&gate, &bank);
    let m = v
        .spendable()
        .find(|m| m.what == Statistic::Top1Flips)
        .expect("judged");
    assert!(m.satisfied(), "zero spent against a zero budget is fine");
    assert_eq!(
        m.utilisation(),
        None,
        "0.0 / 0.0 must not reach the ranking"
    );
    assert_ne!(
        v.binding().expect("scored").criterion,
        Criterion::Top1Flips,
        "an unrankable criterion must never be reported as the scarce one"
    );
}

/// **BS2-F2.** A margin's key looks the calibration up — the registry
/// and the constraint vector share ONE vocabulary.
///
/// The defect this pins: the registry keyed `"route flip rate"` while
/// `ConstraintVector::of` emitted `"route flips"`. The lookup missed,
/// `evidence_for` fell through to its `is_priceable()` arm, and route
/// flips — a COUNT, so always `Measured` — came back `Direct`. The one
/// statistic ROUTE-CAL-1 calibrated as ordering-ONLY was silently
/// PRICED, which is the failure the ladder exists to prevent. Two of
/// the four keys did match, so nothing looked wrong.
///
/// A free string could drift again; `Statistic` cannot. This test would
/// not have caught the original defect had it passed its own literal to
/// both sides — which is exactly what the fixtures did — so it takes the
/// key from a REAL margin built by `ConstraintVector::of`.
#[test]
fn a_counted_proxy_is_ordered_never_priced_when_keyed_from_a_real_margin() {
    use super::super::measurement::{EvidenceScale, MeasurementStatus, TailSupportPolicy};
    use super::super::search_evidence::{SearchCalibrationRegistry, SearchEvidence};

    // A gate that DOES limit route flips, so the vector emits that margin.
    let gate = QualityGate {
        route_flip_max: Some(64),
        ..kimi_logit_v1()
    };
    let v = ConstraintVector::of(&gate, &four_family_selection());
    let m = v
        .margins
        .iter()
        .find(|m| m.criterion == Criterion::RouteFlips)
        .expect("the gate limits route flips, so a margin exists");

    // A count has no tail to be thin: it IS well measured.
    let status = m.measurement_status(&TailSupportPolicy::route_cal_1());
    assert_eq!(status, MeasurementStatus::Measured);

    // And is STILL only a proxy. Before the typed key this returned
    // Direct, because the lookup missed and `Measured` is priceable.
    let r = SearchCalibrationRegistry::route_cal_1();
    let e = r.evidence_for(m.what, EvidenceScale::Diagnostic, &status);
    assert!(
        matches!(e, SearchEvidence::OrderingProxy { .. }),
        "a well-measured COUNT is still ordering-only evidence, got {e:?}"
    );
    assert!(e.orders());
    assert!(
        !e.is_priceable(),
        "BS2-F2: a calibrated proxy must never become price"
    );
}
