//! **L5 of LOWERING-PLUGIN-1: a lowering provider from outside the tree
//! goes the whole way.**
//!
//! The forecast predicts that a crate which is not `larql-vindex` can
//! register a lowering provider this build does not ship and reach
//! register → candidate derivation → selection → accounting → pin →
//! preparation → authority validation → execute, "without core naming its
//! identity" and without opening the kernel plane.
//!
//! This is that journey, in one test, on a container this build wrote —
//! with the controls that keep it from being a demonstration of nothing:
//! the numbers are the provider's own (its kernel multiplies every
//! feed-forward matrix and the vocabulary head, and perturbing it moves
//! the logits); its pins are its own (they record `outside-lowering/v1`, and
//! L4's invalidation refuses by name once it is gone, with the shipped
//! registry NOT a fallback); its refusal is its own; and the source scan
//! in `genericity.rs` says core never learned its name.
//!
//! An integration test is a separate crate in cargo's model, so this file
//! sees only what `larql-vindex` exports. Anything the proof had needed
//! from the crate's internals would have been a leak in the contract.

mod genericity;
mod provider;

use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::accounting::{
    execution_touch, expectations, render_selection_summary, stored_footprint, BlockGeometry,
    ResourceLedger,
};
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use larql_vindex::format::vindex3::opplan::exec::lowering::{LoweringIdentity, LoweringRegistry};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::realization::{
    RealizationForm, RealizationId, RefusalKind, RepresentationFacts,
};
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::exec::{execute_plan, execute_plan_via};
use larql_vindex::format::vindex3::opplan::planned::Operation;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_vindex::format::vindex3::represent::codec::CodecRegistry;

use provider::{identity, sibling_identity, Dispatches, OutsideProvider, Tally, NAME};

/// A prompt over the dense fixture's 128-token vocabulary — the same one
/// the L2 and L4 witnesses read.
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];

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

/// The registry a caller composes: what this build ships, plus a provider
/// it has never heard of.
fn registry_with_outside(provider: OutsideProvider) -> LoweringRegistry {
    LoweringRegistry::shipped()
        .register(Box::new(provider))
        .expect("an identity no shipped provider claims")
}

/// This provider's own pin: decode to f32, then its own scalar kernel.
fn decode_pin() -> RealizationId {
    RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32))
}

/// **The journey.** Register → candidates → selection → accounting → pin
/// → preparation → authority validation → execute, with nothing in core
/// naming the provider at any step.
#[test]
fn an_external_provider_reaches_execution_through_registration_alone() {
    let f = dense();
    let outside = OutsideProvider::new();
    let tally: Tally = outside.tally();

    // Register. The registry keys on the identity the provider states,
    // and the provider is a `Box<dyn PlanBackend>` from here on — its
    // concrete type is gone.
    let registry = registry_with_outside(outside);
    assert!(registry.identities().contains(&identity()));
    assert_eq!(registry.provider(&identity()).unwrap().name(), NAME);

    // Candidates and selection: derived from the codec's declarations,
    // and the choice is the provider's. The f32 codec declares a direct
    // BLAS realization; this provider records it as considered and takes
    // the decode it has a kernel for.
    let ops = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &registry,
        &identity(),
        ExecutionSlice::Full,
    )
    .expect("the external provider prepares the plan");
    let records = ops.realizations();
    assert!(!records.is_empty());
    let projections: Vec<_> = records
        .iter()
        .filter(|r| r.selection.candidates.len() > 1)
        .collect();
    assert!(
        !projections.is_empty(),
        "no operand offered this provider a choice, so nothing was derived"
    );
    for record in &projections {
        assert_eq!(record.selection.realization, decode_pin(), "{record:?}");
        assert!(
            record
                .selection
                .candidates
                .contains(&RealizationId::cpu(RealizationForm::Direct(
                    PhysicalProjectionPlan::BlasF32
                ))),
            "the declined direct realization is recorded as considered: {record:?}"
        );
    }

    // The pin is the provider's, on both authorities (L4).
    assert_eq!(ops.lowerings(), [identity()]);
    for record in records {
        assert_eq!(record.lowering_provider, identity());
    }

    // Accounting prices what THIS provider selected, from tensor tables
    // alone — no payload has been read to get here beyond preparation's
    // own loads.
    let priced = expectations(
        records,
        |op| f.store.stored_len(op),
        BlockGeometry::executor(),
    );
    let ledger = ResourceLedger::aggregate(&priced);
    assert!(ledger.stored > 0 && ledger.resident > 0, "{ledger:?}");
    assert!(execution_touch(&priced) > 0);
    assert!(stored_footprint(&priced).operands > 0);
    let summary = render_selection_summary(records);
    assert!(
        summary.contains(&decode_pin().name()),
        "the summary names what the provider pinned: {summary}"
    );

    // Authority validation (L4): the registry that holds it says the
    // preparation stands; the shipped registry — two working providers,
    // neither of them this one — says it does not.
    ops.ensure_lowerings_in(&registry).unwrap();
    let err = ops
        .ensure_lowerings_in(&LoweringRegistry::shipped())
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside-lowering/v1"), "{err}");
    assert!(err.contains("no other provider stands in for it"), "{err}");

    // Execute. The interpreter drives the whole traversal through this
    // provider, and the logits are what its own kernel computed — here,
    // bit for bit what the reference oracle computes, because this
    // provider's kernel is an honest scalar transcription and its glue is
    // the oracle's.
    let trace = execute_plan_via(&f.plan, &f.store, &TOKENS, &registry, &identity())
        .expect("the external provider executes");
    let oracle = execute_plan(&f.plan, &f.store, &TOKENS, &ReferenceBackend::new()).unwrap();
    assert_eq!(
        bits(trace.logits.as_deref().unwrap()),
        bits(oracle.logits.as_deref().unwrap())
    );
    assert_eq!(trace.final_hidden(), oracle.final_hidden());

    // And what the seam actually asked this provider for — reported
    // rather than assumed.
    let d = tally.dispatches();
    eprintln!("the interpreter's dispatches to the external provider: {d:?}");
    assert!(
        d.output_head > 0,
        "its own kernel produced the logits: {d:?}"
    );
    assert!(
        d.matrices >= 3 * d.ffn + d.output_head,
        "every feed-forward matrix and the head went through its own kernel: {d:?}"
    );
    assert!(d.embed > 0 && d.norm > 0 && d.residual_add > 0, "{d:?}");
    assert!(d.attention > 0 || d.attention_step > 0, "{d:?}");
    assert!(d.total() > 0);
}

/// **Named, not merely available.** A second external provider with the
/// same capabilities — the same candidate derivation, the same kernel,
/// its own identity — is registered beside the first, and the caller's
/// named provider is still the one that runs: its counters move and the
/// sibling's stay at zero, and asking for the sibling instead reaches the
/// sibling. Without this, "the registry resolved the identity the caller
/// named" and "it was the only implementation that could have run this
/// plan" would look identical from the outside.
///
/// This is about RESOLUTION BY IDENTITY, which is L2's contract, and not
/// about a registry matching declared capabilities to pick a provider on
/// the caller's behalf — that is F2, and it stays out of scope. Nothing
/// here asks the registry to choose; the caller chooses, and the registry
/// is held to the choice.
#[test]
fn the_named_provider_is_reached_even_beside_an_equally_capable_one() {
    let f = dense();
    let outside = OutsideProvider::new();
    let sibling = OutsideProvider::sibling();
    let (named, other) = (outside.tally(), sibling.tally());
    let registry = LoweringRegistry::shipped()
        .register(Box::new(outside))
        .and_then(|r| r.register(Box::new(sibling)))
        .expect("two external identities, both free");
    assert!(registry.identities().contains(&identity()));
    assert!(registry.identities().contains(&sibling_identity()));

    let ops = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &registry,
        &identity(),
        ExecutionSlice::Full,
    )
    .unwrap();
    assert_eq!(ops.lowerings(), [identity()], "pinned by the one named");
    execute_plan_via(&f.plan, &f.store, &TOKENS, &registry, &identity()).unwrap();
    assert!(named.dispatches().total() > 0);
    assert_eq!(
        other.dispatches(),
        Dispatches::default(),
        "the equally capable sibling was never asked for anything"
    );

    // And the sibling is reachable in its turn, by its own name — so the
    // registry is resolving identities and not preferring the first
    // provider that could have answered.
    let theirs = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &registry,
        &sibling_identity(),
        ExecutionSlice::Full,
    )
    .unwrap();
    assert_eq!(theirs.lowerings(), [sibling_identity()]);
    execute_plan_via(&f.plan, &f.store, &TOKENS, &registry, &sibling_identity()).unwrap();
    assert!(other.dispatches().total() > 0);
}

/// The instrument control. A defect in the provider's OWN kernel moves
/// the logits — so the numbers above came from it, and not from a shipped
/// path quietly answering underneath.
#[test]
fn the_numbers_are_the_external_providers_own() {
    let f = dense();
    let honest = registry_with_outside(OutsideProvider::new());
    let defective = registry_with_outside(OutsideProvider::with_defect(1e-3));

    let good = execute_plan_via(&f.plan, &f.store, &TOKENS, &honest, &identity()).unwrap();
    let bad = execute_plan_via(&f.plan, &f.store, &TOKENS, &defective, &identity()).unwrap();
    assert_ne!(
        bits(good.logits.as_deref().unwrap()),
        bits(bad.logits.as_deref().unwrap()),
        "the provider's kernel is not what produced the logits"
    );
}

/// The plan follows the PROVIDER's policy, not a shipped default: the
/// production provider, handed the same operands, pins the direct BLAS
/// realization this one declines.
#[test]
fn the_pins_are_the_providers_choice_and_not_a_shipped_default() {
    let f = dense();
    let registry = registry_with_outside(OutsideProvider::new());
    let outside = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &registry,
        &identity(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let shipped = PreparedOperands::load(
        &f.plan,
        &f.store,
        &ProductionBackend::new(),
        ExecutionSlice::Full,
    )
    .unwrap();

    let pins = |ops: &PreparedOperands| -> Vec<String> {
        ops.realizations()
            .iter()
            .filter(|r| r.selection.candidates.len() > 1)
            .map(|r| r.selection.realization.name())
            .collect()
    };
    let (mine, theirs) = (pins(&outside), pins(&shipped));
    assert!(!mine.is_empty() && mine.len() == theirs.len());
    assert_ne!(mine, theirs, "the two providers pin the same realizations");
    assert!(
        theirs.iter().all(|p| p.contains("BlasF32")),
        "the production provider takes the codec's declared direct kernel: {theirs:?}"
    );
    assert!(
        mine.iter().all(|p| p == &decode_pin().name()),
        "and this one decodes, every time, because that is what it has a kernel for: {mine:?}"
    );
}

/// Its refusal is its own, and it is a refusal rather than a fallback:
/// facts that name no registered codec admit nothing, and the provider
/// says so in the contract's vocabulary without any shipped provider
/// answering in its place.
#[test]
fn it_refuses_on_its_own_terms_when_the_facts_admit_nothing() {
    let f = dense();
    let outside = OutsideProvider::new();
    // A projection-class operand: the operations the CONTRACT answers
    // for every backend (an embedding gather, a bank slice) are not this
    // provider's choice to make, and say nothing about its policy.
    let planned = f
        .plan
        .planned_operands()
        .into_iter()
        .find(|p| matches!(p.operation, Operation::Project(_)))
        .expect("the fixture plans projections");

    // Facts resolved through a registry that knows nothing: the label
    // names no codec, so there is no decode and nothing to derive from.
    let empty: &'static CodecRegistry = Box::leak(Box::new(CodecRegistry::new()));
    let nothing = RepresentationFacts::resolve_in(empty, "F32");
    let refusal = outside
        .select(&planned, &nothing)
        .expect_err("nothing is admissible");
    assert_eq!(refusal.kind, RefusalKind::UnregisteredRepresentation);
    assert!(refusal.considered.is_empty());
    let text = refusal.to_string();
    assert!(text.contains("F32"), "{text}");

    // And with the codec registered, the same provider selects — so the
    // refusal was about the facts, not about the provider being unable.
    let known = RepresentationFacts::resolve_in(CodecRegistry::builtin(), "F32");
    let selection = outside.select(&planned, &known).expect("f32 is admissible");
    assert_eq!(selection.realization, decode_pin());
}

/// L4's contract, inherited by a provider core has never heard of: an
/// image it pinned may not be executed by another provider.
#[test]
fn another_provider_may_not_run_the_external_providers_pins() {
    let f = dense();
    let registry = registry_with_outside(OutsideProvider::new());
    let ops = PreparedOperands::load_via(
        &f.plan,
        &f.store,
        &registry,
        &identity(),
        ExecutionSlice::Full,
    )
    .unwrap();
    let err = ops
        .ensure_lowered_by(&ProductionBackend::new())
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside-lowering/v1"), "{err}");
    assert!(err.contains("cpu-production/v1"), "{err}");
    ops.ensure_lowered_by(registry.provider(&identity()).unwrap())
        .expect("its own provider may");
}

/// A control on the identity itself: an external provider is refused
/// registration under an identity a shipped provider already holds, and a
/// second revision of its own family is a different provider — the
/// registry does not care that it comes from outside.
#[test]
fn the_registry_treats_an_outside_identity_exactly_as_a_shipped_one() {
    let registry = registry_with_outside(OutsideProvider::new());
    let again = registry.register(Box::new(OutsideProvider::new()));
    assert!(again.is_err(), "a duplicate identity is refused");

    let other_revision = LoweringIdentity::new(provider::FAMILY, provider::REVISION + 1);
    let registry = registry_with_outside(OutsideProvider::new());
    let err = registry.provider(&other_revision).unwrap_err().to_string();
    assert!(err.contains("outside-lowering/v2"), "{err}");
    assert!(err.contains("outside-lowering/v1"), "{err}");
}
