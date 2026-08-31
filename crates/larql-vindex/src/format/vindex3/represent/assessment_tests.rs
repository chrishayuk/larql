//! `tests` for [`super`].
//!
//! The before/after vectors here are built directly rather than derived
//! from banks: what this module does is the ARITHMETIC of a move, and
//! `constraint_tests` already pins the bank-to-vector mapping against
//! the real measurements. Limits are 1.0 throughout so that an
//! `observed` value reads as its own utilisation and the fractions in
//! each assertion can be checked by eye.

use super::super::byte_ledger::{ByteLedger, ScopeBytes};
use super::super::constraint::Margin;
use super::super::execution_cost::{m3max_metal_001, ExecutionCostModel};
use super::super::measurement::TailSupport;
use super::*;

const KIMI: &str = "Kimi-Linear-48B-A3B-Instruct";
const BASELINE_BYTES: u64 = 5_984_976_896;

fn costs() -> ExecutionCostModel {
    ExecutionCostModel::new(vec![m3max_metal_001()])
}

fn ledger_removing(fraction: f64) -> ByteLedger {
    ByteLedger {
        model: KIMI.into(),
        baseline_representation: "BF16".into(),
        candidate_representation: format!("removes {:.1}%", 100.0 * fraction),
        scopes: vec![ScopeBytes {
            scope: "whole decoder".into(),
            family: "whole decoder".into(),
            baseline_bytes: BASELINE_BYTES,
            candidate_bytes: BASELINE_BYTES - (BASELINE_BYTES as f64 * fraction) as u64,
        }],
    }
}

fn ceiling(what: &str, criterion: Criterion, utilisation: f64) -> Margin {
    // Well-supported by default; the thin-tail cases construct their own.
    ceiling_with(what, criterion, utilisation, Some(2000))
}

fn ceiling_with(
    what: &str,
    criterion: Criterion,
    utilisation: f64,
    observations: Option<u64>,
) -> Margin {
    Margin {
        criterion,
        what: what.into(),
        kind: LimitKind::Ceiling,
        limit: 1.0,
        observed: Some(utilisation),
        vacuous: false,
        tail_support: observations.map(|observations| TailSupport {
            quantile: 0.99,
            observations,
        }),
    }
}

fn floor(what: &str, criterion: Criterion, observed: f64, limit: f64) -> Margin {
    Margin {
        criterion,
        what: what.into(),
        kind: LimitKind::Floor,
        limit,
        observed: Some(observed),
        vacuous: false,
        tail_support: None,
    }
}

/// A vector with the two criteria this programme actually watches, plus
/// sound floors.
fn vector(kl: f64, route: f64) -> ConstraintVector {
    ConstraintVector {
        gate_id: "kimi-logit-balanced-v1".into(),
        margins: vec![
            ceiling("kl p99", Criterion::KlP99, kl),
            ceiling(
                "routed mixture moved at p99",
                Criterion::RouteDisplacement,
                route,
            ),
            floor("positions", Criterion::Positions, 8192.0, 4096.0),
            floor(
                "covered mass at the worst position",
                Criterion::CoveredMass,
                0.63,
                0.55,
            ),
        ],
    }
}

fn assess(
    scale: EvidenceScale,
    parent_removes: f64,
    candidate_removes: f64,
    before: ConstraintVector,
    after: ConstraintVector,
) -> CandidateAssessment {
    CandidateAssessment::of(
        &EvidenceContext::route_cal_1(scale),
        &costs(),
        &ledger_removing(parent_removes),
        &ledger_removing(candidate_removes),
        before,
        after,
    )
    .expect("both ledgers are for the measured model")
}

// ── The correction that made this module marginal. ──

#[test]
fn cost_is_a_fraction_of_what_remained_not_of_the_limit() {
    // Routing at 83% moving to 88% has not "cost 88%". Seventeen points
    // remained and the move took five of them.
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.68, 0.88),
    );
    let route = a
        .marginal
        .costs
        .iter()
        .find(|c| c.criterion == Criterion::RouteDisplacement)
        .expect("paired");
    assert!((route.remaining_before.expect("scored") - 0.17).abs() < 1e-9);
    assert!((route.delta.expect("scored") - 0.05).abs() < 1e-9);
    let f = route.fraction_of_remaining_consumed.expect("scored");
    assert!((f - 5.0 / 17.0).abs() < 1e-6, "consumed {f}");
    assert!((a.ranking_score.scarce_fraction_consumed.expect("scored") - f).abs() < 1e-9);
}

#[test]
fn a_move_that_frees_the_binding_constraint_outranks_one_that_eats_it() {
    // The failure mode this ranking exists to avoid. Both moves buy the
    // same bytes. FREEING costs a point of KL, where 32 points remain;
    // EATING costs nothing in KL and three points of routing, where
    // only 17 remain. A naive "cost = absolute KL delta" ranks EATING
    // first, because its KL cost is zero.
    let freeing = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.69, 0.80),
    );
    let eating = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.68, 0.86),
    );

    // The naive metric, stated so the comparison is explicit.
    let kl_delta = |a: &CandidateAssessment| {
        a.marginal
            .costs
            .iter()
            .find(|c| c.criterion == Criterion::KlP99)
            .and_then(|c| c.delta)
            .expect("scored")
    };
    assert!(
        kl_delta(&freeing) > kl_delta(&eating),
        "the freeing move IS more expensive in absolute KL — that is the trap"
    );

    // The marginal metric prefers it anyway, because a point of KL out
    // of 32 remaining is cheap and three points of routing out of 17 is
    // not.
    let (f, e) = (
        freeing.ranking_score.scarce_fraction_consumed.expect("s"),
        eating.ranking_score.scarce_fraction_consumed.expect("s"),
    );
    assert!((f - 1.0 / 32.0).abs() < 1e-6, "freeing consumed {f}");
    assert!((e - 3.0 / 17.0).abs() < 1e-6, "eating consumed {e}");
    assert_eq!(
        freeing.ranking_score.cmp_rank(&eating.ranking_score),
        std::cmp::Ordering::Less,
        "freeing must sort ahead of eating"
    );
    // And the vector, not the scalar, is what says WHY.
    assert_eq!(freeing.marginal.freed().count(), 1);
    assert_eq!(eating.marginal.freed().count(), 0);
}

#[test]
fn the_binding_criterion_can_change_across_a_move() {
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.90, 0.60),
    );
    assert_eq!(
        a.binding_before().expect("scored").criterion,
        Criterion::RouteDisplacement
    );
    assert_eq!(
        a.binding_after().expect("scored").criterion,
        Criterion::KlP99,
        "a search must recheck what binds after every accepted move"
    );
}

// ── Ranking edges. ──

#[test]
fn a_move_that_consumes_no_headroom_is_unpriced_and_ranks_first() {
    let free = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.68, 0.83),
    );
    assert_eq!(free.ranking_score.scarce_fraction_consumed, None);
    assert_eq!(
        free.ranking_score.score, None,
        "unpriced, not infinitely good"
    );
    assert!(free.ranking_score.gpu_ms_saved > 0.0);

    let priced = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.70, 0.85),
    );
    assert_eq!(
        free.ranking_score.cmp_rank(&priced.ranking_score),
        std::cmp::Ordering::Less
    );
}

#[test]
fn a_move_that_buys_nothing_ranks_last_however_cheap_it_was() {
    let nothing = assess(
        EvidenceScale::Authority,
        0.20,
        0.20,
        vector(0.68, 0.83),
        vector(0.68, 0.83),
    );
    assert!(nothing.ranking_score.gpu_ms_saved.abs() < 1e-9);
    let priced = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.70, 0.85),
    );
    assert_eq!(
        nothing.ranking_score.cmp_rank(&priced.ranking_score),
        std::cmp::Ordering::Greater,
        "a free lunch that feeds nobody still ranks last"
    );
}

#[test]
fn spending_past_an_exhausted_budget_is_flagged_rather_than_scored_cheap() {
    // Routing already over its limit, and the move takes more. No
    // fraction of a negative remainder is meaningful, so it is not
    // scored — and the flag exists so a search cannot read the
    // resulting `None` as "costs nothing".
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 1.05),
        vector(0.68, 1.10),
    );
    let route = a
        .marginal
        .costs
        .iter()
        .find(|c| c.criterion == Criterion::RouteDisplacement)
        .expect("paired");
    assert!(route.remaining_before.expect("scored") < 0.0);
    assert_eq!(route.fraction_of_remaining_consumed, None);
    assert!(a.marginal.spent_past_an_exhausted_budget());
    assert_eq!(a.ranking_score.scarce_fraction_consumed, None);
}

// ── A ranking is not an admission. ──

#[test]
fn diagnostic_evidence_can_be_estimated_but_never_earned() {
    let d = assess(
        EvidenceScale::Diagnostic,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.70, 0.85),
    );
    assert_eq!(d.admission(), Admission::Estimated);
    assert_ne!(d.admission(), Admission::Earned);
    // But it is NOT priced. Under ROUTE-CAL-1 no criterion balanced-v1
    // judges is priceable from a 256-position bank: kl is an ordering
    // proxy and route mass p99 is unusable. The candidate is therefore
    // Unscorable — which now means "cannot be priced at this evidence
    // scale", not "the search knows nothing about it".
    assert_eq!(d.ranking_score.class, MoveClass::Unscorable);
    assert_eq!(d.ranking_score.score, None);
    assert_eq!(
        d.marginal.unpriceable_costs().count(),
        d.marginal.costs.len()
    );
}

#[test]
fn authority_evidence_with_every_criterion_met_is_earned() {
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.70, 0.85),
    );
    assert_eq!(a.admission(), Admission::Earned);
}

#[test]
fn a_short_bank_is_not_held_against_a_diagnostic_but_is_against_authority() {
    let short = |kl, route| {
        let mut v = vector(kl, route);
        v.margins
            .iter_mut()
            .find(|m| m.criterion == Criterion::Positions)
            .expect("present")
            .observed = Some(256.0);
        v
    };
    // A diagnostic bank is short BY DEFINITION; failing the position
    // floor says nothing about the candidate.
    let d = assess(
        EvidenceScale::Diagnostic,
        0.16,
        0.20,
        short(0.68, 0.83),
        short(0.70, 0.85),
    );
    assert_eq!(d.admission(), Admission::Estimated);
    // At authority scale the same bank is exactly what the floor is for.
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        short(0.68, 0.83),
        short(0.70, 0.85),
    );
    assert_eq!(
        a.admission(),
        Admission::Refused {
            failures: vec!["positions".into()]
        }
    );
}

#[test]
fn a_blind_diagnostic_is_refused_at_either_scale() {
    // Coverage is NOT scale-dependent. A ranking computed from a blind
    // instrument is worthless in exactly the way an admission from one
    // would be — and a blind run scores perfectly on every ceiling.
    let blind = |kl, route| {
        let mut v = vector(kl, route);
        v.margins
            .iter_mut()
            .find(|m| m.criterion == Criterion::CoveredMass)
            .expect("present")
            .observed = Some(2048.0 / 163_840.0);
        v
    };
    for scale in [EvidenceScale::Diagnostic, EvidenceScale::Authority] {
        let a = assess(scale, 0.16, 0.20, blind(0.0, 0.0), blind(0.0, 0.0));
        assert_eq!(
            a.admission(),
            Admission::Refused {
                failures: vec!["covered mass at the worst position".into()]
            },
            "{scale:?}"
        );
    }
}

// ── The trace. ──

#[test]
fn the_trace_names_the_scarce_resource_and_the_evidence_scale() {
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.70, 0.88),
    );
    let t = a.describe();
    assert!(t.contains("<- scarce resource"), "{t}");
    let scarce_line = t
        .lines()
        .find(|l| l.contains("<- scarce resource"))
        .expect("marked");
    assert!(
        scarce_line.contains("routed mixture"),
        "routing consumed 5/17 against kl's 2/32 — {scarce_line}"
    );
    assert!(t.contains("Authority"), "{t}");
    assert!(t.contains("Earned"), "{t}");
    assert!(t.contains("MB/token"), "{t}");
}

#[test]
fn a_diagnostic_trace_says_it_is_not_priced_rather_than_showing_a_number() {
    let d = assess(
        EvidenceScale::Diagnostic,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.70, 0.88),
    );
    let t = d.describe();
    assert!(t.contains("NOT priced"), "{t}");
    assert!(t.contains("NOT PRICEABLE at this scale"), "{t}");
    assert!(
        !t.contains("<- scarce resource"),
        "nothing is scarce if nothing is priced — {t}"
    );
}

#[test]
fn an_unpriced_move_says_so_in_the_trace() {
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.60, 0.80),
    );
    let t = a.describe();
    assert!(t.contains("at no behavioural cost"), "{t}");
    assert!(!t.contains("<- scarce resource"), "{t}");
}

#[test]
fn byte_economics_still_do_not_transfer_across_models() {
    let other = ByteLedger {
        model: "gpt-oss-20b".into(),
        ..ledger_removing(0.20)
    };
    let r = CandidateAssessment::of(
        &EvidenceContext::route_cal_1(EvidenceScale::Authority),
        &costs(),
        &ledger_removing(0.16),
        &other,
        vector(0.68, 0.83),
        vector(0.70, 0.85),
    );
    assert!(r.is_err(), "a move may not be priced from another model");
}

#[test]
fn the_trace_shows_an_unscorable_criterion_as_unscored_not_as_free() {
    // Routing was already over budget, so its cost has no fraction. The
    // trace must SAY that rather than silently omit the line, which
    // would read as though routing had cost nothing.
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 1.05),
        vector(0.68, 1.10),
    );
    let t = a.describe();
    let route_line = t
        .lines()
        .find(|l| l.contains("routed mixture"))
        .expect("the criterion is still listed");
    assert!(route_line.contains("not scored"), "{route_line}");
    assert!(a.marginal.spent_past_an_exhausted_budget());
    // And it is refused, so no ranking of it can become an admission.
    assert!(matches!(a.admission(), Admission::Refused { .. }));
}

// ── The ranking policy, as a total order. ──

#[test]
fn a_move_that_spent_past_an_exhausted_budget_is_unscorable_not_unpriced() {
    // The regression this class exists for. Both states arrive as
    // `scarce_fraction_consumed == None`, and they rank at opposite
    // ends: consuming nothing is the BEST kind of move, and blowing
    // past an exhausted budget is nearly the worst. Reading the one as
    // the other put the single worst move at the top of the list.
    let exhausted = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 1.05),
        vector(0.68, 1.10),
    );
    assert_eq!(exhausted.ranking_score.scarce_fraction_consumed, None);
    assert_eq!(exhausted.ranking_score.class, MoveClass::Unscorable);

    let free = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.68, 0.83),
    );
    assert_eq!(free.ranking_score.scarce_fraction_consumed, None);
    assert_eq!(free.ranking_score.class, MoveClass::Unpriced);

    assert_eq!(
        free.cmp_rank(&exhausted),
        std::cmp::Ordering::Less,
        "the same None must not put them in the same tier"
    );
}

#[test]
fn the_four_classes_rank_in_the_stated_order() {
    let unpriced = assess(
        EvidenceScale::Authority,
        0.16,
        0.22,
        vector(0.68, 0.83),
        vector(0.68, 0.83),
    );
    let priced = assess(
        EvidenceScale::Authority,
        0.16,
        0.24,
        vector(0.68, 0.83),
        vector(0.70, 0.85),
    );
    let unscorable = assess(
        EvidenceScale::Authority,
        0.16,
        0.26,
        vector(0.68, 1.05),
        vector(0.68, 1.10),
    );
    let worthless = assess(
        EvidenceScale::Authority,
        0.20,
        0.20,
        vector(0.68, 0.83),
        vector(0.68, 0.83),
    );
    // Deliberately built so the WORST classes buy the most bytes: a
    // policy driven by physical gain alone would invert this.
    assert!(unscorable.ranking_score.gpu_ms_saved > priced.ranking_score.gpu_ms_saved);
    assert!(priced.ranking_score.gpu_ms_saved > unpriced.ranking_score.gpu_ms_saved);

    let mut all = [
        worthless.clone(),
        unscorable.clone(),
        priced.clone(),
        unpriced.clone(),
    ];
    all.sort_by(CandidateAssessment::cmp_rank);
    let classes: Vec<MoveClass> = all.iter().map(|a| a.ranking_score.class).collect();
    assert_eq!(
        classes,
        [
            MoveClass::Unpriced,
            MoveClass::Priced,
            MoveClass::Unscorable,
            MoveClass::Worthless,
        ]
    );
}

#[test]
fn unpriced_moves_rank_among_themselves_by_physical_gain() {
    let small = assess(
        EvidenceScale::Authority,
        0.16,
        0.18,
        vector(0.68, 0.83),
        vector(0.68, 0.83),
    );
    let large = assess(
        EvidenceScale::Authority,
        0.16,
        0.30,
        vector(0.68, 0.83),
        vector(0.68, 0.83),
    );
    assert_eq!(large.cmp_rank(&small), std::cmp::Ordering::Less);
}

#[test]
fn the_order_is_total_and_reproducible() {
    // Two moves of IDENTICAL economics must still order deterministically,
    // or a search trace cannot be reproduced from its inputs.
    let a = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.70, 0.85),
    );
    let mut b = a.clone();
    b.candidate_map = "zzz identical economics".into();
    assert_eq!(
        a.ranking_score.cmp_rank(&b.ranking_score),
        std::cmp::Ordering::Equal,
        "the scores really are tied"
    );
    assert_eq!(a.cmp_rank(&b), std::cmp::Ordering::Less, "identity decides");
    assert_eq!(
        b.cmp_rank(&a),
        std::cmp::Ordering::Greater,
        "and is antisymmetric"
    );

    // A sort is then independent of input order.
    let mut one = [a.clone(), b.clone()];
    let mut two = [b, a];
    one.sort_by(CandidateAssessment::cmp_rank);
    two.sort_by(CandidateAssessment::cmp_rank);
    assert_eq!(
        one.iter()
            .map(|x| x.candidate_map.clone())
            .collect::<Vec<_>>(),
        two.iter()
            .map(|x| x.candidate_map.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_tie_on_score_is_broken_toward_the_more_frugal_move() {
    // Same gain per unit of scarce headroom, different absolute
    // consumption. The one that leaves more budget for later moves wins.
    let frugal = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.68, 0.85),
    );
    let spendy = assess(
        EvidenceScale::Authority,
        0.16,
        0.20,
        vector(0.68, 0.83),
        vector(0.80, 0.85),
    );
    assert!(
        frugal.ranking_score.scarce_fraction_consumed.expect("s")
            < spendy.ranking_score.scarce_fraction_consumed.expect("s")
    );
    assert_eq!(frugal.cmp_rank(&spendy), std::cmp::Ordering::Less);
}
