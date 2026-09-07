//! **B3's forcing case, through the real planner.**
//!
//! [`attested_fidelity`](super::super::attested_fidelity) proves the
//! composition algebra against hand-built options. This file proves the
//! CARRIAGE: that `select_realizations_within` — the function a caller
//! actually plans through — hands the fidelity floor the COMPOSED bound
//! and not the codec's declared one.
//!
//! The two are different claims, and the gap between them is where a
//! guarantee goes missing. Until carriage, `shallowest_saving` read
//! `option.certificate.radius` exactly as the codec declared it, with no
//! knowledge of the dependency the extent decodes through. That was
//! invisible rather than benign: no SHIPPED codec both declares a radius
//! and requires an auxiliary, so nothing in the suite could tell the two
//! readings apart. These codecs exist precisely to tell them apart.
//!
//! ```text
//! parent radius (declared, depth 0): 0.003
//! coarse codebook:                   0.003  -> composed 0.006 -> refused
//! fine codebook:                     0.001  -> composed 0.004 -> selected
//! floor:                             0.005
//! ```
//!
//! Neither half decides it: 0.003 is inside the floor and so is each
//! codebook's bound. Only the composition separates them, which is what
//! makes this a witness for carriage rather than for arithmetic.

mod codecs;
mod container;
mod floor_vocabulary;
mod quantised_refusal;
mod vocabulary;

use codecs::registry;
use container::{build, shallow};
use vocabulary::*;

use crate::format::vindex3::opplan::exec::accounting::{RepresentationFloor, ResidencyBudget};
use crate::format::vindex3::opplan::exec::operands::OperandSource;
use crate::format::vindex3::opplan::exec::prepared::{select_realizations_within, ExecutionSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::represent::codec::RepresentationExtent;
use crate::format::vindex3::representation_attestations::recognition::RecognisedMethods;

/// **The decisive one.** The floor's answer moves on the composition, in
/// the planner, with one artifact and one floor.
#[test]
fn the_floor_judges_the_composed_bound_and_not_the_codecs_declared_one() {
    let terminal = RepresentationExtent::at_depth(1);
    let floor = RepresentationFloor::Within(FLOOR);

    let (coarse, _) = build(COARSE_LABEL)
        .select(&ResidencyBudget::UNBOUNDED)
        .unwrap();
    let option = shallow(&coarse);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the composition is available");
    assert!(
        (radius.radius() - (PARENT + COARSE)).abs() < 1e-12,
        "expected the composed bound, got {radius:?}"
    );
    assert!(
        !floor.admits(option, terminal),
        "0.006 must not satisfy a 0.005 floor"
    );

    let (fine, _) = build(FINE_LABEL)
        .select(&ResidencyBudget::UNBOUNDED)
        .unwrap();
    let option = shallow(&fine);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the composition is available");
    assert!(
        (radius.radius() - (PARENT + FINE)).abs() < 1e-12,
        "expected the composed bound, got {radius:?}"
    );
    assert!(
        floor.admits(option, terminal),
        "0.004 must satisfy a 0.005 floor"
    );
}

/// The over-promise, stated as a fact: the codec's DECLARED bound is the
/// same in both containers and inside the floor in both, so a planner
/// reading it — which is what this build did until carriage — would have
/// admitted the coarse artifact too.
#[test]
fn the_declared_bound_alone_would_have_admitted_both() {
    let declared = registry()
        .by_label(OWNER_LABEL)
        .expect("registered")
        .extents()
        .into_iter()
        .find(|c| c.extent == RepresentationExtent::BASE)
        .and_then(|c| c.radius)
        .expect("the owner declares a radius at depth 0");
    assert!(
        (declared.radius() - PARENT).abs() < 1e-12,
        "the declaration is the same whatever the codebook"
    );
    assert!(
        declared.radius() <= FLOOR,
        "which is why reading it alone was an over-promise"
    );
}

/// And the end of it: under a preparation budget the terminal extent
/// cannot meet, the fine container gives up depth and the coarse one is
/// REFUSED — because the only extent that would have saved anything is
/// one its composed bound does not admit.
#[test]
fn a_budget_may_only_spend_fidelity_the_composition_still_admits() {
    let fine = build(FINE_LABEL);
    let (whole_records, whole) = fine.select(&ResidencyBudget::UNBOUNDED).unwrap();
    for record in whole_records
        .iter()
        .filter(|r| r.representation == OWNER_LABEL)
    {
        assert_eq!(
            record.extent.selected,
            RepresentationExtent::at_depth(1),
            "an unbudgeted plan pins the whole artifact"
        );
    }

    // A deficit ONE owner's move can cover, so the arms differ in whether
    // the move is permitted and not in whether it would have been enough.
    let saving = whole_records
        .iter()
        .filter(|r| r.representation == OWNER_LABEL)
        .map(|r| r.planned.operand.shape.iter().product::<usize>() as u64)
        .max()
        .expect("the owner is in the plan");
    let budget = ResidencyBudget::UNBOUNDED
        .with_prepare_bytes(whole.read_to_prepare - saving)
        .with_fidelity(RepresentationFloor::Within(FLOOR));

    let (shallower, priced) = fine.select(&budget).expect("0.004 is inside the floor");
    assert!(
        shallower
            .iter()
            .filter(|r| r.representation == OWNER_LABEL)
            .any(|r| r.extent.selected == RepresentationExtent::BASE),
        "the fine container may spend the depth"
    );
    assert!(
        priced.read_to_prepare < whole.read_to_prepare,
        "and spending it opened less"
    );

    // The same budget, the same owner codec, the same declared radius —
    // and a refusal, because the only extent that would have saved
    // anything is one the COMPOSED bound does not admit.
    let refusal = build(COARSE_LABEL)
        .select(&budget)
        .expect_err("0.006 is outside the floor, so there is nothing to give up");
    assert!(
        refusal.contains("relative RMS at or under 5.000e-3"),
        "the refusal names the floor it could not meet: {refusal}"
    );
    assert!(
        !refusal.contains("extent depth 1 → 0"),
        "and no shallower extent was ever taken: {refusal}"
    );
}

/// The wave's exit claim, in the planner: a MEASURED error on this
/// instance reaches the floor that gates selection.
///
/// The same coarse codebook whose composition the floor refuses above.
/// Nothing about the artifact changed except that an encoder measured it
/// and said so: 0.001 against the scheme's conservative 0.003, composed
/// with the codebook's 0.003, is 0.004 and inside the floor. A lossy
/// artifact satisfies a quality requirement without anyone inventing a
/// number.
#[test]
fn a_verified_measurement_replaces_the_declared_bound_and_reaches_the_floor() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let trusting = RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1);

    let records = built
        .select_trusting(&ResidencyBudget::UNBOUNDED, trusting)
        .unwrap();
    let option = shallow(&records);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the measurement composed");
    assert!(
        (radius.radius() - (ATTESTED + COARSE)).abs() < 1e-12,
        "the attested claim REPLACES the declared one rather than widening it: {radius:?}"
    );
    assert!(
        radius.radius() < PARENT + COARSE,
        "and it is the better of the two, which is the whole point"
    );
    assert!(
        RepresentationFloor::Within(FLOOR).admits(option, RepresentationExtent::at_depth(1)),
        "0.004 satisfies a 0.005 floor"
    );
}

/// And the control that makes the arm above mean something: the same
/// container, the same attestation, the same bytes — read by a build that
/// recognises nobody.
///
/// The guarantee is UNAVAILABLE, not optimistic: selection falls back to
/// what the codec declares, the composition is 0.006, and the floor
/// refuses it. An attestation is carried and checked either way; whether
/// it may influence a decision is a trust question, and the container
/// making the claim does not get to answer it.
#[test]
fn an_unrecognised_measurement_leaves_the_declared_bound_standing() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);

    let records = built
        .select_trusting(&ResidencyBudget::UNBOUNDED, RecognisedMethods::none())
        .unwrap();
    let option = shallow(&records);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the declared bound still composes");
    assert!(
        (radius.radius() - (PARENT + COARSE)).abs() < 1e-12,
        "an unrecognised measurement must not be acted on: {radius:?}"
    );
    assert!(
        !RepresentationFloor::Within(FLOOR).admits(option, RepresentationExtent::at_depth(1)),
        "0.006 is refused exactly as if nothing had been attested"
    );
}

/// Verification is not free, and the reads are real: hashing the owners'
/// payloads moves the store's consumption ledger.
///
/// Priced honestly rather than absorbed — a plan that admits nothing
/// reads nothing, and one that admits two operands reads both. (The
/// `ResourceLedger`'s own `prepare` figure is computed from declared
/// extents and does not yet include these; that is debt D1, named by the
/// freeze and not paid here.)
#[test]
fn verifying_evidence_is_recorded_as_preparation_work() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let trusting = RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1);

    let (plan, store) = built.plan_and_store_trusting(trusting);
    let before = store.load_count();
    select_realizations_within(
        &plan,
        OperandSource::from(&store),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    assert_eq!(
        store.load_count() - before,
        OWNERS.len() as u64,
        "one payload read per attested operand, and not one more"
    );

    // The same plan with nothing attested opens nothing to plan it.
    let quiet = build(COARSE_LABEL);
    let (plan, store) = quiet.plan_and_store();
    let before = store.load_count();
    select_realizations_within(
        &plan,
        OperandSource::from(&store),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    assert_eq!(store.load_count(), before, "planning read nothing");
}

/// The refusal names the bound when there IS one.
///
/// The coarse container's terminal extent certifies `0.0` on its own and
/// still composes to the codebook's 0.003, so a floor under that refuses
/// the plan at its initial pin — before any budget pressure, and with a
/// number rather than an absence. The Q4 witness beside this covers the
/// other half: an operand with no bound at all.
#[test]
fn a_composed_bound_that_is_merely_too_large_is_refused_by_its_number() {
    let refusal = build(COARSE_LABEL)
        .select(
            &ResidencyBudget::UNBOUNDED
                .with_fidelity(RepresentationFloor::Within(TIGHTER_THAN_COMPOSED)),
        )
        .expect_err("0.003 does not fit inside 0.001");
    assert!(
        refusal.contains("composes to"),
        "the refusal states the composed bound: {refusal}"
    );
    assert!(
        refusal.contains("relative-rms"),
        "in the metric it was stated in: {refusal}"
    );
    assert!(
        !refusal.contains("states no source-fidelity bound"),
        "this operand HAS a bound; it is just too large: {refusal}"
    );
}

/// And the same container plans without complaint under the structural
/// floor, which asks a different question and makes no such claim.
#[test]
fn the_structural_floor_plans_what_the_evidence_floor_refuses() {
    build(COARSE_LABEL)
        .select(&ResidencyBudget::UNBOUNDED.with_fidelity(RepresentationFloor::TerminalExtent))
        .expect("the complete stored representation is what the pin already holds");
}
