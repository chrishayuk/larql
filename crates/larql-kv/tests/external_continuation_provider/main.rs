//! **C5 of CONTINUATION-PLUGIN-1: a continuation provider core has never
//! heard of carries a conversation end to end.**
//!
//! The forecast (`docs/represent/forecasts/continuation-plugin-1.json`)
//! asks for an outside provider to go register → capability match →
//! factory construction → prepare → prefill → decode → saved handoff →
//! resume, bit-identical in rows and logits to `canonical/v1`, without
//! core naming it. The design and controls were frozen in
//! `continuation-plugin-1-notes.json` before this file existed. This is
//! an integration test target, so it reaches the crates only through what
//! they export; the provider is in `provider.rs`, the source scan in
//! `genericity.rs`.

mod genericity;
mod provider;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use larql_kv::{shipped_continuations, CanonicalFactory};
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_TOKENS,
};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_handoff::{
    ContinuationHandoff, ResumeRefusal,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion, ContinuationRegistry,
    ContinuationRegistryError, SelectedContinuation,
};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{KvState, RowFactory};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

use provider::{hostile_identity, HostileFactory, BITS_OPTION, HOSTILE_REVISION};

/// Decode steps before the handoff is saved, and again after resume.
const STEPS: usize = 3;

// ---- fixtures -------------------------------------------------------------

struct Fixture {
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
    geometry: Vec<LayerContinuationGeometry>,
}

fn open(model: fn(&std::path::Path), name: &str) -> Fixture {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(model, checkpoint.path(), container.path(), name);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "{name} must close: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    Fixture {
        _container: container,
        plan,
        store,
        geometry,
    }
}

/// The sliding-window stack: every layer keeps K/V rows, so a KV-only
/// provider can hold it.
fn sliding() -> Fixture {
    let fixture = open(miniature_glimmer, "c5-sliding");
    assert!(
        fixture.geometry.iter().all(|g| g.kv().is_some()),
        "the journey fixture must be KV-only"
    );
    fixture
}

fn config(options: &[&str]) -> ContinuationConfig {
    ContinuationConfig::parse(options).unwrap()
}

fn bits(value: u32) -> String {
    format!("{BITS_OPTION}={value}")
}

/// Any factory, with every build counted.
struct Counted<F> {
    inner: F,
    builds: Arc<AtomicUsize>,
}

impl<F: ContinuationFactory> ContinuationFactory for Counted<F> {
    fn identity(&self) -> ContinuationIdentity {
        self.inner.identity()
    }
    fn regions(&self) -> &[ContinuationRegion] {
        self.inner.regions()
    }
    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        self.inner.validate_config(config)
    }
    fn build(&self, config: &ContinuationConfig) -> BoxedContinuation {
        self.builds.fetch_add(1, Ordering::SeqCst);
        self.inner.build(config)
    }
}

/// A registry of `canonical/v1`, `row/v1` and `factory`, every build
/// counted: (external, canonical, row).
struct Beside {
    registry: ContinuationRegistry,
    external: Arc<AtomicUsize>,
    canonical: Arc<AtomicUsize>,
    row: Arc<AtomicUsize>,
}

impl Beside {
    fn new(factory: HostileFactory) -> Self {
        let external = factory.builds();
        let (canonical, row) = (Arc::default(), Arc::default());
        let mut registry = ContinuationRegistry::new();
        registry
            .register(Box::new(Counted {
                inner: CanonicalFactory,
                builds: Arc::clone(&canonical),
            }))
            .unwrap();
        registry
            .register(Box::new(Counted {
                inner: RowFactory,
                builds: Arc::clone(&row),
            }))
            .unwrap();
        registry.register(Box::new(factory)).unwrap();
        Self {
            registry,
            external,
            canonical,
            row,
        }
    }

    fn select(
        &self,
        identity: &ContinuationIdentity,
        options: &[&str],
        geometry: &[LayerContinuationGeometry],
    ) -> SelectedContinuation {
        self.registry
            .select(identity, &config(options), geometry)
            .unwrap()
    }

    fn builds(&self) -> (usize, usize, usize) {
        (
            self.external.load(Ordering::SeqCst),
            self.canonical.load(Ordering::SeqCst),
            self.row.load(Ordering::SeqCst),
        )
    }
}

// ---- the journey ----------------------------------------------------------

/// Everything a conversation produced, and everything its state held.
#[derive(Debug, PartialEq)]
struct Record {
    prefill_logits: Vec<f32>,
    step_logits: Vec<Vec<f32>>,
    ids: Vec<u32>,
    position: usize,
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0 as u32
}

/// Greedy decode for [`STEPS`] from `state`, extending `ids` and `logits`.
fn decode(
    fixture: &Fixture,
    state: &mut dyn KvState,
    ids: &mut Vec<u32>,
    logits: &mut Vec<Vec<f32>>,
) {
    let backend = ReferenceBackend::new();
    let mut session =
        DecodeSession::with_kv_state(&fixture.plan, &fixture.store, &backend, state).unwrap();
    for _ in 0..STEPS {
        let next = session.step(*ids.last().unwrap()).unwrap().logits.unwrap();
        ids.push(argmax(&next));
        logits.push(next);
    }
}

/// Prefill and decode under `writer`, save the sealed handoff apart from
/// every session, resume it under `reader`'s authority, and decode on.
fn journey(
    fixture: &Fixture,
    writer: &SelectedContinuation,
    reader: &SelectedContinuation,
) -> Record {
    let mut handoff = writer.begin();
    let prefill_logits = prefill_plan(
        &fixture.plan,
        &fixture.store,
        &G_TOKENS,
        &ReferenceBackend::new(),
        handoff.state_mut(),
    )
    .unwrap()
    .logits
    .unwrap();
    let mut ids = vec![argmax(&prefill_logits)];
    let mut step_logits = Vec::new();
    decode(fixture, handoff.state_mut(), &mut ids, &mut step_logits);

    // Saved: the handoff outlives every session that drove it.
    let saved: Vec<ContinuationHandoff> = vec![handoff];
    let mut resumed = saved
        .into_iter()
        .next()
        .unwrap()
        .resume(reader.authority())
        .expect("the authority that wrote the state resumes it");
    decode(fixture, resumed.state_mut(), &mut ids, &mut step_logits);

    let state = resumed.state();
    let layers = 0..fixture.geometry.len();
    Record {
        prefill_logits,
        step_logits,
        ids,
        position: state.position(),
        keys: layers.clone().map(|l| state.keys(l).to_vec()).collect(),
        values: layers.map(|l| state.values(l).to_vec()).collect(),
    }
}

/// The same journey under `canonical/v1` from the shipped registry.
fn canonical_reference(fixture: &Fixture) -> Record {
    let shipped = shipped_continuations();
    let canonical = shipped
        .select(
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
            &fixture.geometry,
        )
        .unwrap();
    journey(fixture, &canonical, &canonical)
}

#[test]
fn an_external_provider_carries_a_conversation_bit_identical_to_canonical() {
    let fixture = sliding();
    let writer = Beside::new(HostileFactory::new(HOSTILE_REVISION));
    let hostile = hostile_identity(HOSTILE_REVISION);
    let selected = writer.select(&hostile, &[&bits(4)], &fixture.geometry);
    assert_eq!(selected.authority().identity, hostile);

    // Resume under an authority resolved from a SECOND registry: equality
    // is by authority, never by the object that built the state.
    let reader_registry = Beside::new(HostileFactory::new(HOSTILE_REVISION));
    let reader = reader_registry.select(&hostile, &[&bits(4)], &fixture.geometry);

    let record = journey(&fixture, &selected, &reader);
    let reference = canonical_reference(&fixture);
    assert!(
        record == reference,
        "the external provider's conversation diverges from canonical/v1"
    );

    // The caller named it, and nothing else answered: of three providers
    // that could hold this plan, only the external one was built, once.
    assert_eq!(writer.builds(), (1, 0, 0), "(external, canonical, row)");
    assert_eq!(reader_registry.builds(), (0, 0, 0), "resume built nothing");

    // The journey must cross the sliding window, or it says nothing about
    // windowed layers.
    let window = fixture
        .geometry
        .iter()
        .filter_map(|g| g.kv().and_then(|kv| kv.window))
        .min()
        .expect("a sliding layer");
    assert!(
        record.position > window,
        "{} ≤ window {window}",
        record.position
    );
    assert_eq!(record.ids.len(), 1 + 2 * STEPS);
}

#[test]
fn the_numbers_are_the_external_providers_own() {
    // A sibling identical except that it moves one stored value: if a
    // shipped path were answering underneath, the record would not move.
    let fixture = sliding();
    let honest = Beside::new(HostileFactory::new(HOSTILE_REVISION));
    let sibling = Beside::new(HostileFactory::corrupting(HOSTILE_REVISION));
    let hostile = hostile_identity(HOSTILE_REVISION);
    let honest_sel = honest.select(&hostile, &[], &fixture.geometry);
    let sibling_sel = sibling.select(&hostile, &[], &fixture.geometry);
    let reference = canonical_reference(&fixture);
    assert!(journey(&fixture, &honest_sel, &honest_sel) == reference);
    assert!(
        journey(&fixture, &sibling_sel, &sibling_sel) != reference,
        "a one-ULP change in the external provider's own storage went unseen"
    );
    assert_eq!(sibling.builds(), (1, 0, 0));
}

// ---- F5: refused at selection, before any state ---------------------------

#[test]
fn a_hybrid_plan_is_refused_at_selection_with_nothing_built() {
    let hybrid = open(hybrid_kda_mla_f32_model, "c5-hybrid");
    let first_stateful_non_kv = hybrid
        .geometry
        .iter()
        .position(|g| {
            ContinuationRegion::required_by(g).is_some_and(|r| r != ContinuationRegion::Kv)
        })
        .expect("the hybrid keeps recurrent or latent state");

    let beside = Beside::new(HostileFactory::new(HOSTILE_REVISION));
    let refusal = beside
        .registry
        .select(
            &hostile_identity(HOSTILE_REVISION),
            &ContinuationConfig::empty(),
            &hybrid.geometry,
        )
        .unwrap_err();
    match &refusal {
        ContinuationRegistryError::Unsupported {
            identity,
            layer,
            declared,
            ..
        } => {
            assert_eq!(identity, &hostile_identity(HOSTILE_REVISION));
            assert_eq!(*layer, first_stateful_non_kv);
            assert_eq!(declared, &[ContinuationRegion::Kv]);
        }
        other => panic!("expected a capability refusal, got {other}"),
    }
    assert_eq!(
        beside.builds(),
        (0, 0, 0),
        "no partial state: nothing was built"
    );
}

#[test]
fn control_the_late_refusal_path_exists_so_the_early_one_is_selections() {
    // Bypass selection: the same provider, built directly, reaches the
    // hybrid and is refused MID-prefill by the provider itself. Without
    // this, the test above could pass because the provider never gets
    // that far for some other reason.
    let hybrid = open(hybrid_kda_mla_f32_model, "c5-hybrid-late");
    let mut state = HostileFactory::build_unselected();
    let error = prefill_plan(
        &hybrid.plan,
        &hybrid.store,
        &[1, 0, 2],
        &ReferenceBackend::new(),
        &mut *state,
    )
    .unwrap_err()
    .to_string();
    // The refusal is the trait's DEFAULT `prepare_continuation`, which
    // names "a KV-only provider" rather than the provider's identity —
    // recorded as a C5 finding; the selection refusal above names it.
    assert!(
        error.contains("`a KV-only provider` holds no recurrent state")
            && error.contains("layer 0"),
        "the late refusal must come from the provider path: {error}"
    );
}

// ---- C4 inherited ---------------------------------------------------------

/// A prefilled handoff written by the external provider under `options`.
fn written(fixture: &Fixture, options: &[&str]) -> ContinuationHandoff {
    let writer = Beside::new(HostileFactory::new(HOSTILE_REVISION));
    let mut handoff = writer
        .select(
            &hostile_identity(HOSTILE_REVISION),
            options,
            &fixture.geometry,
        )
        .begin();
    prefill_plan(
        &fixture.plan,
        &fixture.store,
        &G_TOKENS,
        &ReferenceBackend::new(),
        handoff.state_mut(),
    )
    .unwrap();
    handoff
}

#[test]
fn a_bumped_revision_refuses_the_external_providers_state() {
    let fixture = sliding();
    let handoff = written(&fixture, &[&bits(4)]);
    let bumped = Beside::new(HostileFactory::new(HOSTILE_REVISION + 1));
    let reader = bumped.select(
        &hostile_identity(HOSTILE_REVISION + 1),
        &[&bits(4)],
        &fixture.geometry,
    );
    assert_eq!(
        handoff.resume(reader.authority()).unwrap_err(),
        ResumeRefusal::RevisionChanged {
            recorded: hostile_identity(HOSTILE_REVISION),
            resolved: hostile_identity(HOSTILE_REVISION + 1),
        }
    );
    assert_eq!(bumped.builds(), (0, 0, 0));
}

#[test]
fn another_configuration_refuses_the_external_providers_state() {
    let fixture = sliding();
    let handoff = written(&fixture, &[&bits(4)]);
    let reader_registry = Beside::new(HostileFactory::new(HOSTILE_REVISION));
    let reader = reader_registry.select(
        &hostile_identity(HOSTILE_REVISION),
        &[&bits(3)],
        &fixture.geometry,
    );
    let refusal = handoff.resume(reader.authority()).unwrap_err();
    assert_eq!(refusal.kind(), "configuration_changed", "{refusal}");
    assert_eq!(reader_registry.builds(), (0, 0, 0));
}

#[test]
fn an_unknown_option_is_refused_by_the_provider_that_reads_it() {
    let fixture = sliding();
    let beside = Beside::new(HostileFactory::new(HOSTILE_REVISION));
    let refusal = beside
        .registry
        .select(
            &hostile_identity(HOSTILE_REVISION),
            &config(&["mode=x"]),
            &fixture.geometry,
        )
        .unwrap_err()
        .to_string();
    assert!(
        refusal.contains("mode") && refusal.contains(BITS_OPTION),
        "{refusal}"
    );
    assert_eq!(beside.builds(), (0, 0, 0));
}
