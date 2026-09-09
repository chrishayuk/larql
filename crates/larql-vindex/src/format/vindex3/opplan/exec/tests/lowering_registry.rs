//! **L2 of LOWERING-PLUGIN-1: lowering providers are registered and
//! carried as explicit authority.**
//!
//! One question, per the forecast: can providers be registered and carried
//! as a VALUE without changing what any existing backend selects or
//! executes? Held here as the five properties the wave named — registered
//! is discoverable, unregistered is refused by identity, a duplicate is
//! refused, another revision is another provider, registration order
//! decides nothing — and the F4 anti-cheat: a registry that omits the
//! production provider, handed to the carried path, refuses rather than
//! constructing the provider anyway.

use super::super::backend::{PlanBackend, WeightFormat, WeightFormats};
use super::super::device::DevicePlanBackend;
use super::super::lowering::{LoweringError, LoweringIdentity, LoweringRegistry};
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::production::{self, ProductionBackend};
use super::super::reference::{self, ReferenceBackend};
use super::super::{execute_plan, execute_plan_via};
use super::lowering_identity::Relabelled;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_compute::cpu::CpuBackend;

const SCRATCH_FAMILY: &str = "scratch-provider";
/// A prompt over the dense fixture's 128-token vocabulary.
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];

fn reference_id() -> LoweringIdentity {
    LoweringIdentity::new(reference::IDENTITY_FAMILY, reference::IDENTITY_REVISION)
}

fn production_id() -> LoweringIdentity {
    LoweringIdentity::new(production::IDENTITY_FAMILY, production::IDENTITY_REVISION)
}

fn scratch(name: &'static str, revision: u32) -> Box<dyn PlanBackend + Send> {
    Box::new(Relabelled::new(name, SCRATCH_FAMILY, revision))
}

#[test]
fn a_registered_provider_is_discoverable_and_an_unregistered_one_is_refused_naming_the_registered()
{
    let registry = LoweringRegistry::shipped()
        .register(scratch("scratch", 1))
        .unwrap();
    let scratch_id = LoweringIdentity::new(SCRATCH_FAMILY, 1);
    assert_eq!(registry.provider(&scratch_id).unwrap().name(), "scratch");
    assert_eq!(
        registry.provider(&production_id()).unwrap().name(),
        ProductionBackend::new().name()
    );

    let absent = LoweringIdentity::new("device-matmul", 1);
    let err = registry.provider(&absent).unwrap_err();
    match &err {
        LoweringError::Unregistered {
            identity,
            registered,
        } => {
            assert_eq!(*identity, absent);
            assert_eq!(
                *registered,
                [reference_id(), production_id(), scratch_id.clone()],
                "every registered identity, in registration order"
            );
        }
        other => panic!("{other}"),
    }
    let text = err.to_string();
    assert!(text.contains("device-matmul/v1"), "{text}");
    assert!(
        text.contains("reference/v1") && text.contains("cpu-production/v1"),
        "the refusal names what IS registered: {text}"
    );
}

#[test]
fn a_duplicate_identity_is_refused_and_an_invalid_one_never_enters() {
    let err = LoweringRegistry::shipped()
        .register(Box::new(ProductionBackend::new()))
        .expect_err("the production provider is already there");
    assert!(
        matches!(&err, LoweringError::Duplicate { identity } if *identity == production_id()),
        "{err}"
    );
    // A scratch provider under the same identity as a shipped one is a
    // duplicate too — the identity is what is keyed, not the type.
    let impostor = Box::new(Relabelled::new(
        "impostor",
        production::IDENTITY_FAMILY,
        production::IDENTITY_REVISION,
    ));
    let err = LoweringRegistry::shipped().register(impostor).unwrap_err();
    assert!(matches!(err, LoweringError::Duplicate { .. }), "{err}");

    let err = LoweringRegistry::new()
        .register(Box::new(Relabelled::new("x", "not a family", 1)))
        .unwrap_err();
    assert!(matches!(err, LoweringError::Invalid(_)), "{err}");
    let err = LoweringRegistry::new()
        .register(Box::new(Relabelled::new("x", "unstated", 0)))
        .unwrap_err();
    assert!(err.to_string().contains("revision 0"), "{err}");
}

#[test]
fn two_revisions_of_one_family_are_distinct_providers() {
    let registry = LoweringRegistry::new()
        .register(scratch("one", 1))
        .and_then(|r| r.register(scratch("two", 2)))
        .unwrap();
    assert_eq!(registry.len(), 2);
    assert_eq!(
        registry
            .provider(&LoweringIdentity::new(SCRATCH_FAMILY, 1))
            .unwrap()
            .name(),
        "one"
    );
    assert_eq!(
        registry
            .provider(&LoweringIdentity::new(SCRATCH_FAMILY, 2))
            .unwrap()
            .name(),
        "two"
    );
    assert_eq!(registry.family(SCRATCH_FAMILY).len(), 2);
    // A third revision is refused, and the refusal says which exist.
    let err = registry
        .provider(&LoweringIdentity::new(SCRATCH_FAMILY, 3))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("scratch-provider/v1") && err.contains("scratch-provider/v2"),
        "{err}"
    );
}

#[test]
fn registration_order_has_no_effect_on_who_answers() {
    let forward = LoweringRegistry::new()
        .register(Box::new(ReferenceBackend::new()))
        .and_then(|r| r.register(Box::new(ProductionBackend::new())))
        .and_then(|r| r.register(scratch("scratch", 1)))
        .unwrap();
    let backward = LoweringRegistry::new()
        .register(scratch("scratch", 1))
        .and_then(|r| r.register(Box::new(ProductionBackend::new())))
        .and_then(|r| r.register(Box::new(ReferenceBackend::new())))
        .unwrap();
    assert_ne!(forward.identities(), backward.identities());
    let mut sorted = (forward.identities(), backward.identities());
    sorted.0.sort();
    sorted.1.sort();
    assert_eq!(sorted.0, sorted.1);
    for identity in forward.identities() {
        assert_eq!(
            forward.provider(&identity).unwrap().name(),
            backward.provider(&identity).unwrap().name(),
            "{identity}"
        );
        assert_eq!(
            forward.provider(&identity).unwrap().identity(),
            backward.provider(&identity).unwrap().identity()
        );
    }
}

/// `shipped()` is a value: what a build registers when a caller says
/// nothing else, built fresh on each call, holding the two providers that
/// need no device. A device provider is the caller's to add — and a
/// SECOND device instance is a duplicate, which is the registry answering
/// L1's open question the only way it can: one configured instance per
/// identity.
#[test]
fn the_shipped_registry_is_a_value_and_one_instance_per_identity() {
    let a = LoweringRegistry::shipped();
    let b = LoweringRegistry::shipped();
    assert_eq!(a.identities(), b.identities());
    assert_eq!(a.identities(), [reference_id(), production_id()]);
    assert!(a
        .provider(&LoweringIdentity::new("device-matmul", 1))
        .is_err());

    let with_device = a
        .register(Box::new(DevicePlanBackend::new(
            CpuBackend,
            "cpu-as-device-f32",
            WeightFormat::F32,
        )))
        .unwrap();
    assert_eq!(with_device.len(), 3);
    let err = with_device
        .register(Box::new(DevicePlanBackend::with_formats(
            CpuBackend,
            "cpu-as-device-f16",
            WeightFormats::uniform(WeightFormat::F16),
        )))
        .unwrap_err();
    assert!(
        matches!(&err, LoweringError::Duplicate { identity } if identity.to_string() == "device-matmul/v1"),
        "{err}"
    );
}

// ── F4: the carried path constructs nothing ──────────────────────────

struct Fixture {
    _tmp: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn dense() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let container = tmp.path().join("out.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "target");
    let inspection = inspect_container(&container, false).unwrap();
    let plan = plan_component_ops(&inspection, &container, "target")
        .unwrap()
        .plan
        .expect("the dense fixture plans");
    let store = OperandStore::open(&container, &inspection).unwrap();
    Fixture {
        _tmp: tmp,
        plan,
        store,
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A registry that omits the production provider, handed to the carried
/// path with the production identity: refused by identity, before
/// preparation and before execution, naming what the registry holds. No
/// path underneath constructs `ProductionBackend` to answer anyway.
#[test]
fn the_carried_path_refuses_an_absent_provider_rather_than_constructing_one() {
    let fixture = dense();
    let without_production = LoweringRegistry::new()
        .register(Box::new(ReferenceBackend::new()))
        .unwrap();
    assert!(without_production.provider(&production_id()).is_err());

    let err = PreparedOperands::load_via(
        &fixture.plan,
        &fixture.store,
        &without_production,
        &production_id(),
        ExecutionSlice::Full,
    )
    .err()
    .expect("preparation through a registry without the provider is refused")
    .to_string();
    assert!(err.contains("cpu-production/v1"), "{err}");
    assert!(err.contains("registered: reference/v1"), "{err}");
    assert_eq!(
        fixture.store.load_count(),
        0,
        "refused before any operand was read"
    );

    let err = execute_plan_via(
        &fixture.plan,
        &fixture.store,
        &TOKENS,
        &without_production,
        &production_id(),
    )
    .expect_err("execution through a registry without the provider is refused")
    .to_string();
    assert!(err.contains("cpu-production/v1"), "{err}");
    assert_eq!(fixture.store.load_count(), 0);
}

/// And the positive control: through a registry that holds the provider,
/// the carried path is the direct path — bit for bit, for both shipped
/// providers — so carrying the registry changed nothing about what
/// executes.
#[test]
fn the_carried_path_executes_exactly_what_the_direct_path_executes() {
    let fixture = dense();
    let shipped = LoweringRegistry::shipped();
    for (identity, direct) in [
        (
            reference_id(),
            execute_plan(
                &fixture.plan,
                &fixture.store,
                &TOKENS,
                &ReferenceBackend::new(),
            ),
        ),
        (
            production_id(),
            execute_plan(
                &fixture.plan,
                &fixture.store,
                &TOKENS,
                &ProductionBackend::new(),
            ),
        ),
    ] {
        let direct = direct.unwrap();
        let carried =
            execute_plan_via(&fixture.plan, &fixture.store, &TOKENS, &shipped, &identity).unwrap();
        assert_eq!(
            bits(carried.logits.as_deref().unwrap()),
            bits(direct.logits.as_deref().unwrap()),
            "{identity}"
        );
        assert_eq!(carried.final_hidden(), direct.final_hidden(), "{identity}");
        let prepared = PreparedOperands::load_via(
            &fixture.plan,
            &fixture.store,
            &shipped,
            &identity,
            ExecutionSlice::Full,
        )
        .unwrap();
        let pinned: Vec<String> = prepared
            .realizations()
            .iter()
            .map(|r| r.selection.realization.name())
            .collect();
        let direct_pins: Vec<String> = PreparedOperands::load(
            &fixture.plan,
            &fixture.store,
            shipped.provider(&identity).unwrap(),
            ExecutionSlice::Full,
        )
        .unwrap()
        .realizations()
        .iter()
        .map(|r| r.selection.realization.name())
        .collect();
        assert_eq!(pinned, direct_pins, "{identity}: the same pins");
    }
}
