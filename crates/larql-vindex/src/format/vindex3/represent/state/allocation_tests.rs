//! R5-F11, ported from rung 5's N3 onto the canonical substrate.
//!
//! The N3 numbers are the fixture on purpose: an allocator that stops
//! reproducing U1/U2's behaviour has changed meaning, whatever it says in
//! the abstract.

use super::super::search_policy::SelectionShape;
use super::*;

/// N3's two candidates. U2 better on kl (3.0700e-3 vs 3.1217e-3), U1
/// better on route flip rate (0.19922 vs 0.20312). Both validly
/// participate in both statistics — the conflict is REAL, unlike R4-F10's.
const U1: &str = "U1";
const U2: &str = "U2";

fn ranked() -> SelectionShape {
    SelectionShape::Ranked
}

// ---- the invariant ---------------------------------------------------

#[test]
fn incomparable_valid_evidence_expands_rather_than_choosing() {
    let a = allocate(ranked(), U1, &[U1, U2], FrontierOrdering::Incomparable);
    assert_eq!(a, EvidenceAllocation::Frontier(vec![U1, U2]));
    assert!(a.is_expanded(), "the search could not legitimately choose");
    assert_eq!(a.len(), 2);
    assert_eq!(a.subjects(), &[U1, U2]);
}

#[test]
fn resolved_evidence_takes_the_leader_and_does_not_expand() {
    let a = allocate(ranked(), U2, &[U1, U2], FrontierOrdering::Resolved);
    assert_eq!(a, EvidenceAllocation::One(U2));
    assert!(!a.is_expanded());
}

#[test]
fn a_sole_candidate_cannot_be_ambiguous_whatever_the_caller_claims() {
    // Ruling 3: it is selected because there is nothing to rank. An
    // "incomparable" claim over one member is a caller error.
    assert_eq!(
        allocate(
            SelectionShape::Sole,
            U1,
            &[U1, U2],
            FrontierOrdering::Incomparable
        ),
        EvidenceAllocation::One(U1)
    );
}

#[test]
fn an_incomparable_claim_over_one_member_is_not_a_frontier() {
    let a = allocate(ranked(), U1, &[U1], FrontierOrdering::Incomparable);
    assert_eq!(a, EvidenceAllocation::One(U1));
    assert!(!a.is_expanded());
}

#[test]
fn an_exhausted_selection_buys_nothing() {
    let a: EvidenceAllocation<&str> = allocate(
        SelectionShape::Exhausted,
        U1,
        &[U1, U2],
        FrontierOrdering::Incomparable,
    );
    assert_eq!(a, EvidenceAllocation::None);
    assert!(a.is_empty());
    assert_eq!(a.len(), 0);
    assert!(a.subjects().is_empty());
}

// ---- the R5-N3 witness, in full --------------------------------------

#[test]
fn r5_n3_the_frontier_survives_input_order_and_cannot_be_collapsed() {
    // Forward and reverse must allocate the same SET — R4-F12 recurred
    // once already while this substrate was being built, and it was a
    // permutation test that normalised both sides which hid it. Compare
    // the records the allocator produced.
    let fwd = allocate(ranked(), U1, &[U1, U2], FrontierOrdering::Incomparable);
    let rev = allocate(ranked(), U2, &[U2, U1], FrontierOrdering::Incomparable);
    assert!(fwd.is_expanded() && rev.is_expanded());
    let (mut a, mut b) = (fwd.subjects().to_vec(), rev.subjects().to_vec());
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(
        a, b,
        "the same two members, whichever order they arrived in"
    );

    // And the leader argument — which differs between the two calls — must
    // not appear as a winner anywhere in an expanded allocation.
    assert_eq!(fwd.len(), 2);
    assert_eq!(rev.len(), 2);
}

#[test]
fn physical_gain_cannot_collapse_an_incomparable_frontier() {
    // U1 saved 431,777,920 bytes against U2's 429,686,784 — MORE — and
    // the round still refuses to choose. R4-F6: gain separates only what
    // the proxies call EQUAL; it may never buy past a conflict.
    let a = allocate(ranked(), U1, &[U1, U2], FrontierOrdering::Incomparable);
    assert!(a.is_expanded(), "a 2,091,136-byte lead is not an ordering");
}

// ---- allocation is not authority -------------------------------------

#[test]
fn the_scale_is_the_policys_and_is_not_baked_into_the_allocation() {
    // The historical ruling read "ambiguous -> measure all at authority".
    // N3 needed authority because that was ITS next registered scale.
    let diag = AllocationPolicy::at(EvidenceScale::Diagnostic);
    let auth = AllocationPolicy::at(EvidenceScale::Authority);
    assert_ne!(diag, auth);
    assert_eq!(diag.next_scale, EvidenceScale::Diagnostic);
    assert!(diag.max_experiments.is_none());

    // The same allocation is purchasable at either scale.
    let a = allocate(ranked(), U1, &[U1, U2], FrontierOrdering::Incomparable);
    assert!(diag.admits(a.clone()).is_ok());
    assert!(auth.admits(a).is_ok());
}

#[test]
fn a_budget_refuses_a_round_it_cannot_buy_and_never_trims_it() {
    // Trimming is choosing, and choosing is what the frontier says cannot
    // be done. There is no third answer.
    let policy = AllocationPolicy::at(EvidenceScale::Authority).with_budget(1);
    let a = allocate(ranked(), U1, &[U1, U2], FrontierOrdering::Incomparable);
    match policy.admits(a) {
        Err(BudgetRefusal {
            required,
            budget,
            scale,
        }) => {
            assert_eq!(required, 2);
            assert_eq!(budget, 1);
            assert_eq!(scale, EvidenceScale::Authority);
        }
        Ok(other) => panic!("a budget must refuse, never shorten: {:?}", other.len()),
    }
}

#[test]
fn an_underfunded_policy_refuses_the_whole_frontier_rather_than_buying_a_subset() {
    // The invariant most likely to be broken by a future "helpful"
    // optimisation, and the one that matters most when authority runs are
    // expensive. An agent may say "budget permits one run but the
    // unresolved allocation requires two". The optimizer's answer is
    // INSUFFICIENT EVIDENCE BUDGET — never "I will measure the one I
    // think is better", because that is a resource decision masquerading
    // as behavioural evidence.
    let frontier = allocate(ranked(), U1, &[U1, U2], FrontierOrdering::Incomparable);
    let members = frontier.len();

    for budget in 1..members {
        let policy = AllocationPolicy::at(EvidenceScale::Authority).with_budget(budget);
        match policy.admits(frontier.clone()) {
            Err(refusal) => {
                assert_eq!(refusal.required, members);
                assert_eq!(refusal.budget, budget);
            }
            Ok(bought) => panic!(
                "budget {budget} bought {} of {members} — a PROPER SUBSET of an \
                 incomparable frontier is a choice the evidence did not support",
                bought.len()
            ),
        }
    }

    // And it is the WHOLE frontier or nothing: a sufficient budget buys
    // every member, never a truncation that happens to fit.
    let enough = AllocationPolicy::at(EvidenceScale::Authority).with_budget(members);
    assert_eq!(enough.admits(frontier.clone()).unwrap().len(), members);
}

#[test]
fn a_budget_that_fits_passes_the_allocation_through_unchanged() {
    let policy = AllocationPolicy::at(EvidenceScale::Authority).with_budget(2);
    let a = allocate(ranked(), U1, &[U1, U2], FrontierOrdering::Incomparable);
    assert_eq!(policy.admits(a.clone()).unwrap(), a);
}

#[test]
fn a_one_experiment_round_fits_a_one_experiment_budget() {
    let policy = AllocationPolicy::at(EvidenceScale::Diagnostic).with_budget(1);
    let a = allocate(ranked(), U2, &[U1, U2], FrontierOrdering::Resolved);
    assert!(policy.admits(a).is_ok());
}

#[test]
fn allocations_round_trip_through_serde() {
    for a in [
        EvidenceAllocation::<&str>::None,
        EvidenceAllocation::One(U1),
        EvidenceAllocation::Frontier(vec![U1, U2]),
    ] {
        let back: EvidenceAllocation<String> =
            serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(back.len(), a.len());
    }
    let p = AllocationPolicy::at(EvidenceScale::Authority).with_budget(3);
    let back: AllocationPolicy = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(p, back);
}
