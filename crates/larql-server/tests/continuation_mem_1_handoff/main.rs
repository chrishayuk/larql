//! **CONTINUATION-MEM-1 M6: handoff and resume move state and copy
//! nothing.**
//!
//! seal (`SelectedContinuation::begin`) → prefill/decode → the server's
//! `ResponseKvCache` insert → take → `ContinuationHandoff::resume`, under
//! the measuring allocator of `larql-kv/tests/continuation_mem_1` (the
//! same files, included by path, so both targets run one instrument).
//! I6: the backing inventory — every row, matrix, latent row and
//! recurrent buffer, pointer and capacity — is taken through the
//! type-erased handoff (deviation D5) before insert and after resume, and
//! must be identical. Forecast and deviations:
//! `docs/represent/forecasts/continuation-mem-1{,-notes}.json`.

// Shared with the larql-kv target, which uses the parts this one does not.
#[allow(dead_code)]
#[path = "../../../larql-kv/tests/continuation_mem_1/alloc.rs"]
mod alloc;
#[allow(dead_code)]
#[path = "../../../larql-kv/tests/continuation_mem_1/measured.rs"]
mod measured;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

use serde_json::{json, Value};

use larql_kv::{CanonicalFactory, CanonicalKvState};
use larql_server::response_kv::ResponseKvCache;
use larql_server::vindex3::V3KvHandoff;
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer, G_TOKENS,
};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion, ContinuationRegistry,
};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{RowFactory, RowKvState};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

use measured::{Backing, Inspect, InventorySink, Measured};

#[global_allocator]
static ALLOCATOR: alloc::MeasuringAllocator = alloc::MeasuringAllocator;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A shipped factory whose provider is wrapped in `Measured` with the
/// inventory sink — same identity, same regions, same config.
struct MeasuredFactory<F> {
    inner: F,
    build: fn() -> BoxedContinuation,
}

impl<F: ContinuationFactory> ContinuationFactory for MeasuredFactory<F> {
    fn identity(&self) -> ContinuationIdentity {
        self.inner.identity()
    }
    fn regions(&self) -> &[ContinuationRegion] {
        self.inner.regions()
    }
    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        self.inner.validate_config(config)
    }
    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        (self.build)()
    }
}

// The sink is process-global because a factory's `build` takes no
// argument through which to hand it one; SERIAL makes that safe.
static TRIGGER: std::sync::OnceLock<Arc<AtomicBool>> = std::sync::OnceLock::new();
static SINK: std::sync::OnceLock<InventorySink> = std::sync::OnceLock::new();

fn trigger() -> Arc<AtomicBool> {
    TRIGGER
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

fn sink() -> InventorySink {
    SINK.get_or_init(|| Arc::new(Mutex::new(None))).clone()
}

fn wrapped<P: Inspect + Send + 'static>(provider: P) -> BoxedContinuation {
    Box::new(Measured::new(provider).with_sink(trigger(), sink()))
}

struct Subject {
    _container: tempfile::TempDir,
    name: &'static str,
    plan: ComponentOpPlan,
    store: OperandStore,
    geometry: Vec<LayerContinuationGeometry>,
}

fn fixture(model: fn(&std::path::Path), name: &'static str) -> Subject {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(model, checkpoint.path(), container.path(), name);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "{name}: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    Subject {
        _container: container,
        name,
        plan,
        store,
        geometry,
    }
}

/// Take the inventory through the type-erased handoff (D5): arm the
/// trigger, re-set the current position through `state_mut()`.
fn inventory(
    state: &mut (dyn larql_vindex::format::vindex3::opplan::exec::kv::ContinuationProvider + Send),
) -> Vec<Backing> {
    trigger().store(true, std::sync::atomic::Ordering::SeqCst);
    let position = state.position();
    state.set_position(position);
    sink()
        .lock()
        .unwrap()
        .take()
        .expect("the wrapper wrote its inventory")
}

fn journey(
    subject: &Subject,
    registry: &ContinuationRegistry,
    identity: &ContinuationIdentity,
    tokens: &[u32],
    decode: &[u32],
) -> Value {
    let selected = registry
        .select(identity, &ContinuationConfig::empty(), &subject.geometry)
        .unwrap();
    let authority = selected.authority().clone();
    let backend = ReferenceBackend::new();

    let seal_scope = alloc::enter();
    let mut handoff = selected.begin();
    let seal = seal_scope.leave();

    prefill_plan(
        &subject.plan,
        &subject.store,
        tokens,
        &backend,
        handoff.state_mut(),
    )
    .unwrap();
    {
        let mut session = DecodeSession::with_kv_state(
            &subject.plan,
            &subject.store,
            &backend,
            handoff.state_mut(),
        )
        .unwrap();
        for &token in decode {
            session.step(token).unwrap();
        }
    }
    let before = inventory(handoff.state_mut());
    let position = handoff.position();

    let cache = ResponseKvCache::new(4, 600);
    let v3 = V3KvHandoff {
        continuation: handoff,
        absorbed_ids: tokens.iter().chain(decode).copied().collect(),
    };
    // One scope over insert → take → resume. Its keys and model id are
    // borrowed `&str`s built before it opens.
    let (id, model) = ("resp-mem1", "mem1-model");
    let scope = alloc::enter();
    cache.insert(id, model, None, v3);
    let taken = cache.take(id, model);
    let resumed = taken.map(|t| (t.continuation.resume(&authority), t.absorbed_ids));
    let transfer = scope.leave();
    let events = alloc::events(&transfer);

    let (resumed, absorbed) = resumed.expect("take-once returns the inserted handoff");
    let mut resumed = resumed.expect("same authority resumes");
    let after = inventory(resumed.state_mut());
    drop(absorbed);

    let backing_sizes: BTreeSet<usize> = before.iter().filter_map(|b| b.bytes).collect();
    let continuation_sized: Vec<usize> = events
        .iter()
        .filter(|e| e.kind == alloc::EventKind::Alloc && backing_sizes.contains(&e.new_size))
        .map(|e| e.new_size)
        .collect();
    let alloc_sizes: Vec<usize> = events
        .iter()
        .filter(|e| e.kind == alloc::EventKind::Alloc)
        .map(|e| e.new_size)
        .collect();
    let identical = before == after;
    let kinds: BTreeSet<&str> = before.iter().map(|b| b.kind).collect();
    json!({
        "provider": identity.to_string(),
        "position": position,
        "resumed_position": resumed.position(),
        "backings": before.len(),
        "backing_kinds": kinds,
        "backing_bytes": before.iter().filter_map(|b| b.bytes).sum::<usize>(),
        "unresolved_backings": before.iter().filter(|b| b.bytes.is_none() && !b.kind.ends_with("_header")).count(),
        "seal_scope": { "allocs": seal.allocs, "alloc_bytes": seal.alloc_bytes },
        "transfer_scope": {
            "allocs": transfer.allocs,
            "alloc_bytes": transfer.alloc_bytes,
            "alloc_sizes": alloc_sizes,
            "frees": transfer.frees,
            "realloc_moved": transfer.moved,
            "foreign": transfer.foreign,
        },
        "M6_continuation_sized_allocs": continuation_sized.len(),
        "M6_continuation_sized_alloc_bytes": continuation_sized,
        "I6_backing_identical": identical,
        "M6_held": continuation_sized.is_empty() && identical,
    })
}

fn registry() -> ContinuationRegistry {
    let mut registry = ContinuationRegistry::new();
    registry
        .register(Box::new(MeasuredFactory {
            inner: RowFactory,
            build: || wrapped(RowKvState::default()),
        }))
        .unwrap();
    registry
        .register(Box::new(MeasuredFactory {
            inner: CanonicalFactory,
            build: || wrapped(CanonicalKvState::new()),
        }))
        .unwrap();
    registry
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("LARQL_MEM1_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("continuation-mem-1"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn m6_handoff_moves_state_and_copies_nothing() {
    let _serial = serial();
    let registry = registry();
    let subjects = [
        (
            fixture(dense_f32_model, "mem1-dense"),
            (1..9).collect::<Vec<u32>>(),
            vec![30u32, 31, 32],
        ),
        (
            fixture(miniature_glimmer, "mem1-sliding"),
            G_TOKENS.to_vec(),
            vec![1, 2, 3],
        ),
        (
            fixture(hybrid_kda_mla_f32_model, "mem1-kda-mla"),
            vec![1, 2, 3, 4, 5, 6],
            vec![10, 11],
        ),
    ];
    let mut records = Vec::new();
    for (subject, tokens, decode) in &subjects {
        for identity in [RowKvState::identity(), CanonicalKvState::identity()] {
            let mut record = journey(subject, &registry, &identity, tokens, decode);
            record["subject"] = json!(subject.name);
            records.push(record);
        }
    }
    let path = out_dir().join("m6-handoff.json");
    std::fs::write(&path, serde_json::to_string_pretty(&records).unwrap()).unwrap();
    eprintln!("wrote {}", path.display());
    for r in &records {
        // Instrument validity only; M6 itself is recorded, not asserted.
        assert_eq!(
            r["transfer_scope"]["foreign"], 0,
            "foreign allocation in the transfer: {r}"
        );
        assert_eq!(
            r["unresolved_backings"], 0,
            "unresolved backing capacity: {r}"
        );
        assert_eq!(
            r["position"], r["resumed_position"],
            "resume keeps the position: {r}"
        );
    }
}
