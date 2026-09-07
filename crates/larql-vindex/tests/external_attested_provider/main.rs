//! **ATTESTATION-1 step 7: the out-of-tree acceptance test.**
//!
//! An integration test is a separate crate in cargo's model, so this file
//! sees only what `larql-vindex` exports. Nothing under `src/` was edited
//! for this provider: no match arm names it, no planner branch knows it
//! has two extents, and no policy anywhere knows who `acme-metrology` is.
//!
//! What it has to demonstrate, end to end and through exported API only:
//!
//! 1. an external parent codec, an external auxiliary codec, an external
//!    ROLE name and an external attestation method all reach selection;
//! 2. none of those identities appears in production code (`genericity`);
//! 3. the evidence is INERT under `RecognisedMethods::none()`;
//! 4. explicit recognition admits `Within(r)` — but only after the
//!    payload is verified and the auxiliary is composed in;
//! 5. `CertifiedExact` refuses a composed radius that is not zero, even
//!    when the parent itself is exact;
//! 6. removing or substituting either provider invalidates preparation,
//!    by identity.
//!
//! Each positive arm has the control that isolates its mechanism, so a
//! green arm cannot be explained by anything but the mechanism it names.

mod container;
mod genericity;
mod provider;

use container::{build, recognising, Attest, OWNERS};
use larql_vindex::format::vindex3::opplan::exec::accounting::{
    RepresentationFloor, ResidencyBudget,
};
use larql_vindex::format::vindex3::opplan::exec::realization::RealizationRecord;
use larql_vindex::format::vindex3::represent::codec::{
    CodecRegistry, RepresentationCodec, RepresentationExtent,
};
use larql_vindex::format::vindex3::representation_attestations::recognition::RecognisedMethods;
use provider::*;

/// A budget that forces ONE owner off its terminal extent, so the floor
/// has to judge the shallow option — which is the only extent the
/// attestation speaks about.
fn pressing(whole_prepare: u64, saving: u64, floor: RepresentationFloor) -> ResidencyBudget {
    ResidencyBudget::UNBOUNDED
        .with_prepare_bytes(whole_prepare - saving)
        .with_fidelity(floor)
}

/// The largest single owner's depth-1 → depth-0 saving: one byte per
/// element, since the shallow extent drops the residual plane.
fn one_owner_saving(records: &[RealizationRecord]) -> u64 {
    records
        .iter()
        .filter(|r| r.representation == LATTICE)
        .map(|r| r.planned.operand.shape.iter().product::<usize>() as u64)
        .max()
        .expect("the provider's operands are in the plan")
}

/// **The Q4 lesson, kept.** A test of representation policy must prove
/// the plan actually contains that representation — otherwise it can pass
/// by planning something else entirely and never exercise the policy at
/// all.
fn assert_plan_is_the_providers(records: &[RealizationRecord]) {
    let owned: Vec<&str> = records
        .iter()
        .filter(|r| r.representation == LATTICE)
        .map(|r| r.planned.operand.tensor.as_str())
        .collect();
    assert_eq!(
        owned.len(),
        OWNERS.len(),
        "the plan must actually store the external representation, not merely be planned \
         beside it: {owned:?}"
    );
    for record in records.iter().filter(|r| r.representation == LATTICE) {
        assert_eq!(
            record.extent.options.len(),
            2,
            "both of the provider's extents reached the pin"
        );
        assert_eq!(
            record.dependencies.len(),
            1,
            "and its dependency was resolved through the role it declared"
        );
        assert_eq!(record.dependencies[0].name, ANCHOR_ROLE);
        assert_eq!(record.dependencies[0].label, ANCHORS);
    }
}

#[test]
fn the_external_representation_and_its_dependency_reach_the_plan() {
    let built = build(Attest::Truthful);
    let (records, _) = built
        .select(
            with_providers(),
            recognising(),
            &ResidencyBudget::UNBOUNDED.with_fidelity(RepresentationFloor::TerminalExtent),
        )
        .expect("the whole artifact is what a terminal pin holds");
    assert_plan_is_the_providers(&records);
    // Neither label is one the shipped registry knows.
    assert!(CodecRegistry::builtin().by_label(LATTICE).is_none());
    assert!(CodecRegistry::builtin().by_label(ANCHORS).is_none());
}

/// **Recognition is the gate.** Same container, same bytes, same
/// attestation, same floor — and the only difference is whether this
/// build takes `acme-metrology`'s word.
#[test]
fn recognition_admits_within_and_its_absence_refuses() {
    let built = build(Attest::Truthful);
    let (whole, ledger) = built
        .select(
            with_providers(),
            recognising(),
            &ResidencyBudget::UNBOUNDED.with_fidelity(RepresentationFloor::TerminalExtent),
        )
        .expect("unpressed");
    assert_plan_is_the_providers(&whole);
    let budget = pressing(
        ledger.read_to_prepare,
        one_owner_saving(&whole),
        RepresentationFloor::Within(FLOOR),
    );

    // Recognised: the measured 0.001 replaces the declared 0.004, composes
    // with the anchor table's 0.002 to 0.003, and fits inside 0.005.
    let (pressed, _) = built
        .select(with_providers(), recognising(), &budget)
        .expect("a verified measurement makes the shallow extent admissible");
    assert!(
        pressed
            .iter()
            .filter(|r| r.representation == LATTICE)
            .any(|r| r.extent.selected == RepresentationExtent::BASE),
        "the pin spent the depth the measurement paid for"
    );

    // THE CONTROL: recognise nobody. The declared 0.004 stands, composes
    // to 0.006, and the shallow extent is no longer admissible — so the
    // budget has nothing it may give up.
    let refusal = built
        .select(with_providers(), RecognisedMethods::none(), &budget)
        .expect_err("an unrecognised measurement must not be acted on");
    assert!(refusal.contains("preparation opens"), "{refusal}");
}

/// **Verification is a second gate, and the payload is the only thing
/// that can open it.** The tampered attestation differs from the truthful
/// one in the content digest alone: same operand, same shape, same codec
/// revision, same baselines, same authority and method. Every
/// metadata check passes and the bytes still refuse it.
#[test]
fn a_recognised_measurement_still_needs_its_payload_to_verify() {
    let truthful = build(Attest::Truthful);
    let (whole, ledger) = truthful
        .select(
            with_providers(),
            recognising(),
            &ResidencyBudget::UNBOUNDED.with_fidelity(RepresentationFloor::TerminalExtent),
        )
        .expect("unpressed");
    let budget = pressing(
        ledger.read_to_prepare,
        one_owner_saving(&whole),
        RepresentationFloor::Within(FLOOR),
    );
    truthful
        .select(with_providers(), recognising(), &budget)
        .expect("the control verifies");

    let tampered = build(Attest::Tampered);
    let refusal = tampered
        .select(with_providers(), recognising(), &budget)
        .expect_err("bytes that do not match the digest cannot settle a claim");
    assert!(refusal.contains("preparation opens"), "{refusal}");
}

/// **Composition is the third gate.** With no attestation at all the
/// declared bound stands, and it is the composition with the anchor
/// table that puts it out of reach — so a floor that read the parent
/// alone would have admitted this.
#[test]
fn the_dependencys_bound_is_composed_in_and_decides() {
    let built = build(Attest::None);
    let (whole, ledger) = built
        .select(
            with_providers(),
            recognising(),
            &ResidencyBudget::UNBOUNDED.with_fidelity(RepresentationFloor::TerminalExtent),
        )
        .expect("unpressed");
    let saving = one_owner_saving(&whole);

    // The declared bound alone is inside the floor and composed with the
    // dependency it is not — both asserted at COMPILE time in `provider`,
    // so this arm cannot be read as arithmetic.
    let refusal = built
        .select(
            with_providers(),
            recognising(),
            &pressing(
                ledger.read_to_prepare,
                saving,
                RepresentationFloor::Within(FLOOR),
            ),
        )
        .expect_err("0.004 + 0.002 does not fit inside 0.005");
    assert!(refusal.contains("preparation opens"), "{refusal}");

    // And a floor above the COMPOSED bound admits the same extent, which
    // is what makes the refusal above about the composition and not about
    // the extent being unreachable for some other reason.
    let (pressed, _) = built
        .select(
            with_providers(),
            recognising(),
            &pressing(
                ledger.read_to_prepare,
                saving,
                RepresentationFloor::Within(DECLARED + ANCHOR_RADIUS),
            ),
        )
        .expect("a floor at the composed bound admits it");
    assert!(pressed
        .iter()
        .filter(|r| r.representation == LATTICE)
        .any(|r| r.extent.selected == RepresentationExtent::BASE));
}

/// **`CertifiedExact` refuses a non-zero composed radius**, even though
/// the parent is exact at the extent selected. The anchor table is not,
/// and a caller asking for an exact reconstruction is asking about what
/// they will actually receive.
#[test]
fn certified_exact_refuses_because_the_dependency_is_not_exact() {
    let built = build(Attest::Truthful);
    // The parent's terminal extent certifies 0.0 on its own …
    let terminal = LATTICE12
        .extents()
        .into_iter()
        .find(|c| c.extent == RepresentationExtent::at_depth(1))
        .and_then(|c| c.radius)
        .expect("the provider certifies its terminal extent");
    assert_eq!(terminal.radius(), 0.0);

    // … and the plan is still refused, because what reaches the caller is
    // decoded through a table that certifies 0.002.
    let refusal = built
        .select(
            with_providers(),
            recognising(),
            &ResidencyBudget::UNBOUNDED.with_fidelity(RepresentationFloor::CertifiedExact),
        )
        .expect_err("an exact parent through a lossy dependency is not exact");
    assert!(refusal.contains("composes to"), "{refusal}");
    assert!(refusal.contains(LATTICE), "{refusal}");

    // The structural floor plans the identical container without
    // complaint: it asks a different question and claims nothing.
    built
        .select(
            with_providers(),
            recognising(),
            &ResidencyBudget::UNBOUNDED.with_fidelity(RepresentationFloor::TerminalExtent),
        )
        .expect("the complete stored representation is what the pin holds");
}

/// **Either provider, removed or substituted, invalidates preparation by
/// identity** — and the refusal names which one.
#[test]
fn losing_or_substituting_either_provider_invalidates_preparation() {
    let built = build(Attest::Truthful);
    let prepared = built.prepared(with_providers()).expect("both providers");
    prepared
        .ensure_providers_in(with_providers())
        .expect("nothing changed");

    for (registry, named) in [
        (without_anchor_provider(), ANCHORS),
        (without_parent_provider(), LATTICE),
        (with_substituted_parent(), LATTICE),
    ] {
        let err = prepared
            .ensure_providers_in(registry)
            .expect_err("the image is not executable against this registry")
            .to_string();
        assert!(err.contains(named), "the refusal names the provider: {err}");
    }

    // And preparing afresh against a registry missing either provider is
    // refused too, before any execution.
    for (registry, named) in [
        (without_anchor_provider(), ANCHORS),
        (without_parent_provider(), LATTICE),
    ] {
        let refusal = built
            .prepared(registry)
            .err()
            .unwrap_or_else(|| panic!("a missing provider has no decode"));
        assert!(refusal.contains(named), "{refusal}");
    }
}
