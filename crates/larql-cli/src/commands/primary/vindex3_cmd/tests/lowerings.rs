//! LOWERING-PLUGIN-1, L3: every CLI verb obtains its provider from the
//! registry `lowerings_for` composes, and the seam it resolves through
//! refuses a provider the registry lacks before any operand is read.

use super::*;
use crate::commands::primary::vindex3_cmd::prepare::{
    lowerings_for, with_lowerings, BackendVisitor,
};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::lowering::{LoweringIdentity, LoweringRegistry};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

fn encoded_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = fixture_dir(true);
    let out = dir.path().join("container");
    run(Vindex3Command::Encode(EncodeArgs {
        capability: None,
        artifacts: vec![dir.path().to_path_buf()],
        output: out.clone(),
    }))
    .unwrap();
    (dir, out)
}

/// A visitor that would prepare the plan — the first thing every verb
/// does with its provider — so that if the seam handed it anything, the
/// store's load count would move.
struct Prepare<'a> {
    plan: &'a ComponentOpPlan,
    store: &'a OperandStore,
}

impl BackendVisitor for Prepare<'_> {
    type Out = String;

    fn visit<B: PlanBackend>(self, backend: &B) -> Result<String, Box<dyn std::error::Error>> {
        PreparedOperands::load(self.plan, self.store, backend, ExecutionSlice::Full)?;
        Ok(backend.identity().to_string())
    }
}

/// The CLI's backend choices resolve to the same providers they used to
/// construct, now by identity from the composed registry.
#[test]
fn every_interpreted_backend_choice_resolves_to_its_provider_by_identity() {
    for (choice, identity, name) in [
        (
            ExecBackend::Reference,
            LoweringIdentity::reference(),
            "reference-f32",
        ),
        (
            ExecBackend::Production,
            LoweringIdentity::cpu_production(),
            "production-larql-compute",
        ),
        (
            ExecBackend::ProductionNvfp4,
            LoweringIdentity::cpu_production(),
            "production-larql-compute",
        ),
        (
            ExecBackend::ProductionQ8,
            LoweringIdentity::cpu_production(),
            "production-larql-compute",
        ),
        (
            ExecBackend::ProductionQ6k,
            LoweringIdentity::cpu_production(),
            "production-larql-compute",
        ),
        (
            ExecBackend::ProductionQ4k,
            LoweringIdentity::cpu_production(),
            "production-larql-compute",
        ),
    ] {
        let (lowerings, resolved) = lowerings_for(choice).unwrap();
        assert_eq!(resolved, identity, "{choice:?}");
        let provider = lowerings.provider(&resolved).unwrap();
        assert_eq!(provider.name(), name, "{choice:?}");
        assert_eq!(provider.identity(), identity, "{choice:?}");
        // The interpreted arms compose exactly the shipped providers.
        assert_eq!(
            lowerings.identities(),
            LoweringRegistry::shipped().identities(),
            "{choice:?}"
        );
    }
}

/// F4 through the CLI seam: a registry without the production provider,
/// asked to prepare as it, is refused by identity naming what it holds,
/// and the store has read nothing — no provider was constructed
/// underneath to answer anyway. The same seam with the shipped registry
/// prepares, so the refusal is the registry's and not the fixture's.
#[test]
fn the_seam_refuses_an_absent_provider_with_nothing_read() {
    let (_dir, out) = encoded_fixture();
    let inspection = inspect_container(&out, false).unwrap();
    let plan = plan_component_ops(&inspection, &out, "target")
        .unwrap()
        .plan
        .expect("the fixture plans");
    let store = OperandStore::open(&out, &inspection).unwrap();

    let without_production = LoweringRegistry::new()
        .register(Box::new(ReferenceBackend::new()))
        .unwrap();
    let err = with_lowerings(
        &without_production,
        &LoweringIdentity::cpu_production(),
        Prepare {
            plan: &plan,
            store: &store,
        },
    )
    .expect_err("refused by identity")
    .to_string();
    assert!(err.contains("cpu-production/v1"), "{err}");
    assert!(err.contains("registered: reference/v1"), "{err}");
    assert_eq!(store.load_count(), 0, "nothing was read");

    let identity = with_lowerings(
        &LoweringRegistry::shipped(),
        &LoweringIdentity::cpu_production(),
        Prepare {
            plan: &plan,
            store: &store,
        },
    )
    .expect("the shipped registry prepares");
    assert_eq!(identity, "cpu-production/v1");
    assert!(
        store.load_count() > 0,
        "the positive control read its operands"
    );
}
