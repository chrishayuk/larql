//! **L4 of LOWERING-PLUGIN-1: the decision survives preparation.**
//!
//! The forecast predicts that every selected pin records both provider
//! identities and that "loss or revision substitution of either
//! invalidates preparation distinctly by name", with fallback to another
//! lowering provider named as the thing that must not happen.
//!
//! What is held here is the seven controls the wave froze before it was
//! built: the pinned provider still registered keeps the preparation
//! valid; removed invalidates it; the same family at another revision
//! invalidates it; a registry full of other providers is STILL invalid
//! and never a substitution; each plane refuses on its own authority and
//! not the other's; and a pin that was serialized and reloaded yields the
//! same two verdicts.
//!
//! Plus the question L1 left open and this wave had to answer: whether a
//! provider's CONFIGURATION — the injected device, the format table — is
//! a second authority prepared state must record. The experiment is at
//! the bottom of the file, with the decision rule frozen in the notes
//! before it ran.

use super::super::backend::{PlanBackend, WeightFormat, WeightFormats};
use super::super::device::DevicePlanBackend;
use super::super::lowering::{LoweringIdentity, LoweringRegistry};
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::ProductionBackend;
use super::super::realization::PinnedAuthorities;
use super::super::reference::ReferenceBackend;
use super::super::{execute_prepared_streaming, PlaneEvent};
use super::accounting::ProviderStub;
use super::device::LoopDevice;
use super::lowering_identity::Relabelled;
use super::lowering_registry::{bits, dense, Fixture, SCRATCH_FAMILY, TOKENS};
use crate::format::vindex3::represent::codec::CodecRegistry;

fn scratch_id(revision: u32) -> LoweringIdentity {
    LoweringIdentity::new(SCRATCH_FAMILY, revision)
}

/// A registry holding one scratch provider, at `revision`.
fn scratch_registry(revision: u32) -> LoweringRegistry {
    LoweringRegistry::new()
        .register(Box::new(Relabelled::new(
            "scratch",
            SCRATCH_FAMILY,
            revision,
        )))
        .unwrap()
}

/// An image prepared by the scratch provider at revision 1 — a provider
/// this build does not ship, so every verdict below is about the pin and
/// not about anything privileged.
fn pinned_by_scratch(f: &Fixture) -> PreparedOperands {
    PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &scratch_registry(1),
        &scratch_id(1),
        ExecutionSlice::Full,
    )
    .unwrap()
}

/// A codec registry that answers for the fixture's stored label under a
/// different identity — the codec plane moving while the lowering plane
/// stands still.
fn codecs_moved() -> CodecRegistry {
    CodecRegistry::new()
        .register(Box::new(ProviderStub { label: "F32" }))
        .unwrap()
}

/// Both authorities, on every pin, neither derived from the other: the
/// lowering provider changes while the codec identity does not, and the
/// pin records both.
#[test]
fn every_pin_names_both_authorities_and_the_two_move_independently() {
    let f = dense();
    let shipped = LoweringRegistry::shipped();
    let by_production = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &shipped,
        &LoweringIdentity::cpu_production(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let by_reference = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &shipped,
        &LoweringIdentity::reference(),
        ExecutionSlice::Full,
    )
    .unwrap();

    assert!(!by_production.realizations().is_empty());
    for record in by_production.realizations() {
        assert_eq!(record.lowering_provider, LoweringIdentity::cpu_production());
        assert_eq!(
            record.codec_provider.as_ref().map(|i| i.family.as_str()),
            Some("F32"),
            "the fixture is stored f32 and the codec plane says so"
        );
    }
    assert_eq!(
        by_production.lowerings(),
        [LoweringIdentity::cpu_production()],
        "one image, one lowering provider"
    );
    assert_eq!(by_reference.lowerings(), [LoweringIdentity::reference()]);

    // The lowering authority moved; the codec authority did not.
    assert_eq!(
        by_production.authorities().codecs,
        by_reference.authorities().codecs
    );
    assert_ne!(
        by_production.authorities().lowerings,
        by_reference.authorities().lowerings
    );

    // And a provider this build does not ship is recorded exactly as a
    // shipped one is.
    assert_eq!(pinned_by_scratch(&f).lowerings(), [scratch_id(1)]);
}

/// Controls 1–4. One image, four registries: the provider that pinned it,
/// no provider, the same family at another revision, and a registry full
/// of perfectly good other providers. Only the first stands.
#[test]
fn only_the_provider_that_pinned_the_image_keeps_it_valid() {
    let f = dense();
    let ops = pinned_by_scratch(&f);
    let before = ops.authorities();

    // 1 — the same provider is still registered.
    ops.ensure_lowerings_in(&scratch_registry(1)).unwrap();

    // 2 — the provider is gone.
    let err = ops
        .ensure_lowerings_in(&LoweringRegistry::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("scratch-provider/v1"), "{err}");
    assert!(err.contains("holds none"), "{err}");
    assert!(err.contains("re-prepare"), "{err}");

    // 3 — the same family at another revision is another provider.
    let err = ops
        .ensure_lowerings_in(&scratch_registry(2))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("scratch-provider/v1") && err.contains("scratch-provider/v2"),
        "the refusal names the pin's revision and the one on offer: {err}"
    );

    // 4 — other providers are not a fallback. The shipped registry holds
    // two working providers, and neither stands in for the one that made
    // these pins.
    let err = ops
        .ensure_lowerings_in(&LoweringRegistry::shipped())
        .unwrap_err()
        .to_string();
    assert!(err.contains("scratch-provider/v1"), "{err}");
    assert!(
        err.contains("reference/v1") && err.contains("cpu-production/v1"),
        "the refusal names what IS registered: {err}"
    );
    assert!(err.contains("no other provider stands in for it"), "{err}");

    // Nothing re-selected on the way through any of that: a refusal is a
    // refusal, not a re-pin.
    assert_eq!(ops.authorities(), before);
}

/// Controls 5 and 6. Each plane refuses on its own authority, in its own
/// vocabulary, and neither refusal can be mistaken for the other's.
#[test]
fn each_plane_refuses_on_its_own_authority() {
    let f = dense();
    let ops = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &LoweringRegistry::shipped(),
        &LoweringIdentity::cpu_production(),
        ExecutionSlice::Full,
    )
    .unwrap();

    // 5 — codec valid, lowering invalid: fails on the lowering plane.
    ops.ensure_providers_in(CodecRegistry::builtin()).unwrap();
    let err = ops
        .ensure_lowerings_in(&scratch_registry(1))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("lowering provider `cpu-production/v1`"),
        "{err}"
    );
    assert!(
        !err.contains("representation"),
        "a lowering refusal does not talk about representations: {err}"
    );

    // 6 — lowering valid, codec invalid: fails on the codec plane.
    ops.ensure_lowerings_in(&LoweringRegistry::shipped())
        .unwrap();
    let err = ops
        .ensure_providers_in(&codecs_moved())
        .unwrap_err()
        .to_string();
    assert!(err.contains("representation `F32`"), "{err}");
    assert!(
        err.contains("F32 r1") && err.contains("stub-F32 r7"),
        "{err}"
    );
    assert!(
        !err.contains("lowering"),
        "a codec refusal does not talk about lowering providers: {err}"
    );
}

/// Control 7. The authorities are a value: written down, read back, and
/// judged by exactly the code that judges a live image — so a pin that
/// outlives the process it was made in still knows what decided it.
#[test]
fn a_serialized_and_reloaded_pin_retains_both_authorities() {
    let f = dense();
    let ops = pinned_by_scratch(&f);
    let authorities = ops.authorities();

    let json = serde_json::to_string(&authorities).unwrap();
    assert!(
        json.contains("scratch-provider") && json.contains("F32"),
        "both planes are written down: {json}"
    );
    let reloaded: PinnedAuthorities = serde_json::from_str(&json).unwrap();
    assert_eq!(reloaded, authorities);

    // The same seven-control verdicts, from the reloaded value.
    reloaded.ensure_lowerings_in(&scratch_registry(1)).unwrap();
    assert!(reloaded.ensure_lowerings_in(&scratch_registry(2)).is_err());
    assert!(reloaded
        .ensure_lowerings_in(&LoweringRegistry::shipped())
        .is_err());
    reloaded.ensure_codecs_in(CodecRegistry::builtin()).unwrap();
    assert!(reloaded.ensure_codecs_in(&codecs_moved()).is_err());
}

/// The seam where no registry is available: execution is handed a
/// provider directly, and the pin says which provider decided it. A
/// different provider running these pins is the decision reinterpreted,
/// which is the fallback this wave forbids — so it is refused, by both
/// identities.
#[test]
fn a_pin_is_executed_by_the_provider_that_decided_it_or_not_at_all() {
    let f = dense();
    let ops = PreparedOperands::load(
        &f.plan,
        &f.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let mut sink = |_: PlaneEvent| Ok(());
    execute_prepared_streaming(
        &f.plan,
        &ops,
        &TOKENS,
        &ProductionBackend::new(),
        None,
        &mut sink,
    )
    .expect("the provider that pinned the image executes it");

    let err = execute_prepared_streaming(
        &f.plan,
        &ops,
        &TOKENS,
        &ReferenceBackend::new(),
        None,
        &mut sink,
    )
    .expect_err("another provider may not run another's pins")
    .to_string();
    assert!(
        err.contains("cpu-production/v1") && err.contains("reference/v1"),
        "{err}"
    );
    assert!(err.contains("re-prepare"), "{err}");
}

// ── the device-instance question L1 left open ────────────────────────

/// **The experiment.** Same `LoweringIdentity`, two configurations: does
/// prepared state become invalid or semantically different?
///
/// Two `DevicePlanBackend`s over the same injected device, differing only
/// in the per-class format table they ask the loader for. Both answer
/// `device-matmul/v1`, so identity-level authority cannot tell them
/// apart — and the question is whether it needed to.
///
/// The decision rule was frozen in the wave's notes before this ran: if
/// the image refuses or computes different numbers under the other
/// instance, a SECOND authority exists and is named separately; if it
/// does not, none is invented.
///
/// The answer is in two halves. The configuration IS load-bearing — the
/// two instances prepare different images from the same plan — and yet
/// any one image means the same thing under either instance, because the
/// format table is asked at preparation and never again: what it decided
/// is already in the pinned realization and in the operands materialised
/// under it. So the pin needs the provider's identity and not its
/// configuration, and no second authority is invented here.
#[test]
fn one_identity_over_two_configurations_prepares_differently_and_executes_one_image_identically() {
    let f = dense();
    let asks_f32 = DevicePlanBackend::new(LoopDevice, "loop-device-f32", WeightFormat::F32);
    let asks_f16 = DevicePlanBackend::with_formats(
        LoopDevice,
        "loop-device-f16",
        WeightFormats::uniform(WeightFormat::F16),
    );
    assert_eq!(asks_f32.identity(), asks_f16.identity());
    assert_ne!(asks_f32.name(), asks_f16.name());

    let image = PreparedOperands::load(&f.plan, &f.store, &asks_f32, ExecutionSlice::Full).unwrap();
    assert_eq!(image.lowerings(), [LoweringIdentity::device_matmul()]);

    // Identity-level authority is satisfied by the OTHER instance. That
    // is the sharp edge of the question, not a defect: the registry holds
    // one configured instance per identity (L2) and the pin records the
    // identity, not the instance.
    let other_instance = LoweringRegistry::new()
        .register(Box::new(DevicePlanBackend::with_formats(
            LoopDevice,
            "loop-device-f16",
            WeightFormats::uniform(WeightFormat::F16),
        )))
        .unwrap();
    image.ensure_lowerings_in(&other_instance).unwrap();
    image.ensure_lowered_by(&asks_f16).unwrap();

    // Half one: the configuration is load-bearing. The same plan
    // prepared by the other instance is a different image — different
    // resident representation, different numbers — which is why
    // configuration could have been a second authority.
    let other_image =
        PreparedOperands::load(&f.plan, &f.store, &asks_f16, ExecutionSlice::Full).unwrap();
    let mut sink = |_: PlaneEvent| Ok(());
    let own_preparation =
        execute_prepared_streaming(&f.plan, &image, &TOKENS, &asks_f32, None, &mut sink).unwrap();
    let other_preparation =
        execute_prepared_streaming(&f.plan, &other_image, &TOKENS, &asks_f16, None, &mut sink)
            .unwrap();
    assert_ne!(
        bits(own_preparation.logits.as_deref().unwrap()),
        bits(other_preparation.logits.as_deref().unwrap()),
        "two configurations under one identity prepare two different images"
    );

    // Half two, and the answer: ONE image executes identically under
    // either instance. The format table was consumed at preparation; what
    // it decided is pinned, so the other instance reinterprets nothing.
    let under_other =
        execute_prepared_streaming(&f.plan, &image, &TOKENS, &asks_f16, None, &mut sink).unwrap();
    assert_eq!(
        bits(own_preparation.logits.as_deref().unwrap()),
        bits(under_other.logits.as_deref().unwrap()),
        "one preparation, two same-identity configurations, identical numbers"
    );
    assert_eq!(own_preparation.final_hidden(), under_other.final_hidden());
}
