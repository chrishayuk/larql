//! VI3-KV-1 gates: the canonical cache IS the V3 continuation state.
//!
//! The headline gate compares [`RowKvState`] and [`CanonicalKvState`]
//! after every significant boundary — geometry accepted, prefill
//! state, logical position, per-layer K/V rows, prefill logits, first
//! resumed logits, N continuation logits, generated ids, final state —
//! and demands **bit identity** throughout. With INF-3's gates that
//! chains:
//!
//! ```text
//! V3 batch == V3 tokenwise == RowKvState == larql-kv canonical cache
//! ```
//!
//! and `tests/parity.rs` in larql-vindex chains the production forward
//! onto the same equality.
//!
//! The second architectural gate is geometry closure: the adapter
//! knows the miniature's sliding(3)/full split and row width from the
//! executable plan alone — `larql-kv` consults no `ModelArchitecture`
//! anywhere on this path.

mod registry_parity;
mod resume_anti_cheat;
mod window;

use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_HEAD_DIM, G_KV_HEADS, G_LAYERS, G_TOKENS,
    G_WINDOW,
};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{
    plan_kv_geometry, HistoryRange, KvState, LayerKvGeometry, RowKvState,
};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

use crate::cache::KvCache;

use super::CanonicalKvState;

const CONTINUATION_STEPS: usize = 8;

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "kv1-fixture",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(
        outcome.closed(),
        "fixture must close: {:?}",
        outcome.defects
    );
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Ties keep the first index — the harness rule.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 {
                (index, value)
            } else {
                best
            }
        })
        .0 as u32
}

/// Compare two providers' observable state bit-for-bit.
fn assert_state_equal(a: &dyn KvState, b: &dyn KvState, layers: usize, boundary: &str) {
    assert_eq!(a.position(), b.position(), "{boundary}: positions diverge");
    for layer in 0..layers {
        assert_eq!(
            a.rows(layer).to_owned_rows().0,
            b.rows(layer).to_owned_rows().0,
            "{boundary}: K rows diverge at layer {layer}"
        );
        assert_eq!(
            a.rows(layer).to_owned_rows().1,
            b.rows(layer).to_owned_rows().1,
            "{boundary}: V rows diverge at layer {layer}"
        );
    }
}

/// The cache's matrices — not the adapter's row views — must hold the
/// same bits as the reference provider: the matrices are the storage
/// authority, and this is the check that keeps the view honest.
fn assert_cache_matrices_equal(cache: &KvCache, reference: &RowKvState, layers: usize) {
    for layer in 0..layers {
        let (k, v) = cache.get_layer(layer).expect("layer holds state");
        let k_rows: Vec<Vec<f32>> = k.rows().into_iter().map(|row| row.to_vec()).collect();
        let v_rows: Vec<Vec<f32>> = v.rows().into_iter().map(|row| row.to_vec()).collect();
        assert_eq!(
            k_rows,
            reference.rows(layer).to_owned_rows().0,
            "matrix K diverges at {layer}"
        );
        assert_eq!(
            v_rows,
            reference.rows(layer).to_owned_rows().1,
            "matrix V diverges at {layer}"
        );
    }
}

#[test]
fn canonical_cache_matches_rowkvstate_at_every_boundary() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let layers = plan.layers.len();

    // Prefill both providers from the same program.
    let mut row_state = RowKvState::default();
    let row_prefill = prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut row_state).unwrap();
    let mut canonical = CanonicalKvState::new();
    let canonical_prefill =
        prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut canonical).unwrap();

    // Geometry accepted.
    assert_eq!(canonical.geometry(), plan_kv_geometry(&plan).as_slice());
    // Prefill state, logical position, prefill logits.
    assert_state_equal(&row_state, &canonical, layers, "after prefill");
    assert_cache_matrices_equal(canonical.cache(), &row_state, layers);
    assert_eq!(row_prefill.logits, canonical_prefill.logits);

    // Resume decode over each provider; greedy ids from the shared
    // prefill logits onward.
    let mut ids_a = vec![argmax(row_prefill.logits.as_ref().unwrap())];
    let mut ids_b = ids_a.clone();
    {
        let mut session_a =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut row_state).unwrap();
        let mut session_b =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut canonical).unwrap();
        for step in 0..CONTINUATION_STEPS {
            let logits_a = session_a
                .step(*ids_a.last().unwrap())
                .unwrap()
                .logits
                .unwrap();
            let logits_b = session_b
                .step(*ids_b.last().unwrap())
                .unwrap()
                .logits
                .unwrap();
            // First resumed logits, then every continuation step.
            assert_eq!(logits_a, logits_b, "continuation step {step} diverges");
            ids_a.push(argmax(&logits_a));
            ids_b.push(argmax(&logits_b));
        }
    }
    // Generated ids and final state.
    assert_eq!(ids_a, ids_b, "generated ids diverge");
    assert_state_equal(&row_state, &canonical, layers, "final");
    assert_cache_matrices_equal(canonical.cache(), &row_state, layers);
    assert_eq!(
        canonical.position(),
        G_TOKENS.len() + CONTINUATION_STEPS,
        "final logical position"
    );
}

/// The architectural closure, as its own gate: `larql-kv` knows a
/// VINDEX3 model's continuation geometry — row width AND the
/// sliding(3)/full split — from the executable plan alone. No
/// `ModelArchitecture` appears anywhere in this test or in the
/// adapter.
#[test]
fn continuation_geometry_reaches_larql_kv_from_the_plan_alone() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut canonical = CanonicalKvState::new();
    prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut canonical).unwrap();

    assert_eq!(
        canonical.geometry(),
        &[
            LayerKvGeometry {
                kv_dim: G_KV_HEADS * G_HEAD_DIM,
                window: Some(G_WINDOW),
                history: HistoryRange::Trailing(G_WINDOW),
            },
            LayerKvGeometry {
                kv_dim: G_KV_HEADS * G_HEAD_DIM,
                window: None,
                history: HistoryRange::Full,
            },
        ],
        "the sliding/full split must arrive from the plan and be preserved"
    );
    // Canonical means canonical: the geometry's window is knowledge,
    // not policy — the held cache stays unwindowed (VI3-KV-2 is where
    // windowing becomes a provider policy under its own gates).
    assert!(canonical.cache().max_window.is_none());
}

/// The same conversation state crossing the `KvCache` boundary and
/// coming back: prefill through the adapter, surrender the cache to
/// the existing KV world, adopt it again, and resume decode — every
/// continuation step bit-identical to a session that never crossed.
#[test]
fn state_crosses_the_kvcache_boundary_intact() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();

    // Oracle: prefill + decode over one uninterrupted provider.
    let mut oracle_state = RowKvState::default();
    let oracle_prefill =
        prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut oracle_state).unwrap();
    let mut oracle_ids = vec![argmax(oracle_prefill.logits.as_ref().unwrap())];
    let mut oracle_logits = Vec::new();
    let mut oracle =
        DecodeSession::with_kv_state(&plan, &store, &backend, &mut oracle_state).unwrap();
    for _ in 0..CONTINUATION_STEPS {
        let logits = oracle
            .step(*oracle_ids.last().unwrap())
            .unwrap()
            .logits
            .unwrap();
        oracle_ids.push(argmax(&logits));
        oracle_logits.push(logits);
    }

    // Arm: prefill through the adapter, cross the boundary both ways.
    let mut canonical = CanonicalKvState::new();
    let prefill = prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut canonical).unwrap();
    let cache = canonical.into_cache();
    assert_eq!(cache.next_position, G_TOKENS.len());
    for layer in 0..G_LAYERS {
        assert_eq!(cache.cached_len(layer), G_TOKENS.len());
        let (k, _) = cache.get_layer(layer).unwrap();
        assert_eq!(k.shape()[1], G_KV_HEADS * G_HEAD_DIM);
    }

    let mut adopted = CanonicalKvState::from_cache(cache);
    let mut ids = vec![argmax(prefill.logits.as_ref().unwrap())];
    let mut resumed = DecodeSession::with_kv_state(&plan, &store, &backend, &mut adopted).unwrap();
    assert_eq!(resumed.position(), G_TOKENS.len());
    for (step, expected) in oracle_logits.iter().enumerate() {
        let logits = resumed.step(*ids.last().unwrap()).unwrap().logits.unwrap();
        assert_eq!(&logits, expected, "step {step} diverges after the crossing");
        ids.push(argmax(&logits));
    }
    assert_eq!(ids, oracle_ids);
}

#[test]
#[should_panic(expected = "unwindowed")]
fn a_windowed_cache_is_refused_as_canonical_state() {
    let _ = CanonicalKvState::from_cache(KvCache::with_window(2, 3));
}

#[test]
#[should_panic(expected = "the plan says")]
fn a_misfit_row_width_is_refused() {
    let mut canonical = CanonicalKvState::new();
    canonical.prepare(&[LayerKvGeometry {
        kv_dim: 4,
        window: None,
        history: HistoryRange::Full,
    }]);
    canonical.append(0, vec![0.0; 3], vec![0.0; 3]);
}

// ═══════════════════════════════════════════════════════════════
// Adoption-boundary and refusal contracts
// ═══════════════════════════════════════════════════════════════

#[test]
fn default_is_the_empty_provider() {
    let mut provider = CanonicalKvState::default();
    provider.prepare(&[LayerKvGeometry {
        kv_dim: 4,
        window: None,
        history: HistoryRange::Full,
    }]);
    assert_eq!(provider.geometry().len(), 1);
    assert_eq!(provider.position(), 0);
}

#[test]
fn an_adopted_cache_with_unwritten_layers_prepares_cleanly() {
    // with_layers(2) holds two None layers: LayerRows must default (no
    // rows to serve) and prepare's width check must skip what was never
    // written rather than refuse it.
    let mut adopted = CanonicalKvState::from_cache(KvCache::with_layers(2));
    adopted.prepare(&[
        LayerKvGeometry {
            kv_dim: 4,
            window: None,
            history: HistoryRange::Full,
        },
        LayerKvGeometry {
            kv_dim: 4,
            window: None,
            history: HistoryRange::Full,
        },
    ]);
    assert!(adopted.rows(0).to_owned_rows().0.is_empty());
    assert!(adopted.rows(1).to_owned_rows().1.is_empty());
}

#[test]
#[should_panic(expected = "adopted cache holds")]
fn an_adopted_cache_for_a_different_layer_count_is_refused() {
    let mut adopted = CanonicalKvState::from_cache(KvCache::with_layers(2));
    adopted.prepare(&[LayerKvGeometry {
        kv_dim: 4,
        window: None,
        history: HistoryRange::Full,
    }]);
}

#[test]
#[should_panic(expected = "the plan says")]
fn an_adopted_cache_with_misfit_rows_is_refused() {
    let mut cache = KvCache::with_layers(1);
    cache.set_layer(
        0,
        (
            ndarray::Array2::zeros((1, 3)),
            ndarray::Array2::zeros((1, 3)),
        ),
    );
    let mut adopted = CanonicalKvState::from_cache(cache);
    adopted.prepare(&[LayerKvGeometry {
        kv_dim: 4,
        window: None,
        history: HistoryRange::Full,
    }]);
}

#[test]
#[should_panic(expected = "V row at layer")]
fn a_misfit_value_row_is_refused_even_when_the_key_fits() {
    let mut canonical = CanonicalKvState::new();
    canonical.prepare(&[LayerKvGeometry {
        kv_dim: 4,
        window: None,
        history: HistoryRange::Full,
    }]);
    canonical.append(0, vec![0.0; 4], vec![0.0; 3]);
}

#[test]
fn recurrent_state_is_explicitly_unsupported_not_absent() {
    use larql_vindex::format::vindex3::opplan::exec::kv::ContinuationError;
    let mut provider = CanonicalKvState::new();
    provider.prepare(&[LayerKvGeometry {
        kv_dim: 4,
        window: None,
        history: HistoryRange::Full,
    }]);
    match provider.recurrent_state(7) {
        Err(ContinuationError::RecurrentUnsupported { provider, layer }) => {
            assert_eq!(provider, "CanonicalKvState");
            assert_eq!(layer, 7);
        }
        other => panic!("must refuse with the provider and layer named: {other:?}"),
    }
}

/// **SERVE-HYBRID: the canonical provider holds recurrent buffers where
/// the program declares them — and only there.** A mixed KV+recurrent
/// geometry prepares both sides at absolute indices; the KV layer still
/// refuses `recurrent_state` by name (a dispatch defect, never "empty
/// state"); and a resumed preparation keeps the mutated buffers, because
/// that persistence is the continuation.
#[test]
fn recurrent_layers_get_buffers_and_kv_layers_still_refuse() {
    use larql_vindex::format::vindex3::opplan::exec::continuation::{
        LayerContinuationGeometry, RecurrentBufferGeometry, RecurrentGeometry, StateInitialization,
    };
    use larql_vindex::format::vindex3::opplan::exec::kv::ContinuationError;

    let geometry = vec![
        LayerContinuationGeometry::Kv(LayerKvGeometry {
            kv_dim: 4,
            window: None,
            history: HistoryRange::Full,
        }),
        LayerContinuationGeometry::Recurrent(RecurrentGeometry::single(RecurrentBufferGeometry {
            shape: vec![2, 3],
            dtype: larql_vindex::format::vindex3::opplan::gated_delta::StateDtype::Float32,
            initialization: StateInitialization::Zeros,
        })),
    ];
    let mut provider = CanonicalKvState::new();
    provider.prepare_continuation(&geometry).unwrap();

    // The recurrent layer serves its declared buffer, zero-initialised.
    let state = provider.recurrent_state(1).expect("declared recurrent");
    assert_eq!(state.buffer(0).cells().len(), 6);
    state.buffer_mut(0).cells_mut()[0] = 7.0;

    // The KV layer refuses by name — not with an empty buffer.
    assert!(matches!(
        provider.recurrent_state(0),
        Err(ContinuationError::NotRecurrent { layer: 0, .. })
    ));
    // And it still takes rows, at its absolute index.
    provider.append(0, vec![1.0; 4], vec![2.0; 4]);
    assert_eq!(provider.rows(0).to_owned_rows().0.len(), 1);

    // Resume: the same program keeps the mutated state.
    provider.prepare_continuation(&geometry).unwrap();
    assert_eq!(
        provider.recurrent_state(1).unwrap().buffer(0).cells()[0],
        7.0,
        "a resumed preparation must not reset the continuation"
    );
}

/// Spare matrix capacity must survive appends. This detects the old full-prefix
/// allocation/copy without a noisy wall-clock performance assertion.
#[test]
fn appends_reuse_matrix_capacity_and_preserve_every_stored_bit() {
    let mut cache = KvCache::with_layers(1);
    let mut k = ndarray::Array2::zeros((0, 2));
    let mut v = ndarray::Array2::zeros((0, 2));
    k.reserve_rows(128).unwrap();
    v.reserve_rows(128).unwrap();
    let kp = k.as_ptr();
    let vp = v.as_ptr();
    cache.set_layer(0, (k, v));
    let mut state = CanonicalKvState::from_cache(cache);
    state.prepare(&[LayerKvGeometry {
        kv_dim: 2,
        window: Some(3),
        history: HistoryRange::Trailing(3),
    }]);
    for i in 0..128 {
        let key = vec![i as f32, -0.0];
        let value = vec![-(i as f32), f32::from_bits(0x7fc00001)];
        state.append(0, key.clone(), value.clone());
        let (k, v) = state.cache().get_layer(0).unwrap();
        assert_eq!(k.as_ptr(), kp, "K reallocated with reserved capacity");
        assert_eq!(v.as_ptr(), vp, "V reallocated with reserved capacity");
        assert_eq!(
            k.nrows(),
            i + 1,
            "the declared attention window must not evict state"
        );
        for j in 0..2 {
            assert_eq!(k[(i, j)].to_bits(), key[j].to_bits());
            assert_eq!(v[(i, j)].to_bits(), value[j].to_bits());
            assert_eq!(
                state.rows(0).to_owned_rows().0[i][j].to_bits(),
                key[j].to_bits()
            );
            assert_eq!(
                state.rows(0).to_owned_rows().1[i][j].to_bits(),
                value[j].to_bits()
            );
        }
    }
}

#[test]
fn append_accepts_an_adopted_column_major_cache() {
    use ndarray::ShapeBuilder;
    let mut cache = KvCache::with_layers(1);
    let k = ndarray::Array2::from_shape_vec((2, 2).f(), vec![1., 3., 2., 4.]).unwrap();
    cache.set_layer(0, (k.clone(), k));
    let mut state = CanonicalKvState::from_cache(cache);
    state.prepare(&[LayerKvGeometry {
        kv_dim: 2,
        window: None,
        history: HistoryRange::Full,
    }]);
    state.append(0, vec![5., 6.], vec![7., 8.]);
    assert_eq!(
        state.rows(0).to_owned_rows().0,
        &[vec![1., 2.], vec![3., 4.], vec![5., 6.]]
    );
    assert_eq!(
        state.rows(0).to_owned_rows().1,
        &[vec![1., 2.], vec![3., 4.], vec![7., 8.]]
    );
    assert_eq!(
        state.cache().get_layer(0).unwrap().0.row(2).to_vec(),
        vec![5., 6.]
    );
}

#[test]
fn latent_rows_survive_resume_and_wrong_layer_kinds_refuse() {
    use larql_vindex::format::vindex3::opplan::exec::continuation::{
        LayerContinuationGeometry, LayerLatentKvGeometry,
    };
    use larql_vindex::format::vindex3::opplan::exec::kv::ContinuationError;
    let mut state = CanonicalKvState::new();
    assert!(matches!(
        state.latent_state(1),
        Err(ContinuationError::LatentUnsupported { layer: 1, .. })
    ));
    let geometry = [
        LayerContinuationGeometry::Kv(LayerKvGeometry {
            kv_dim: 2,
            window: None,
            history: HistoryRange::Full,
        }),
        LayerContinuationGeometry::LatentKv(LayerLatentKvGeometry { width: 3 }),
    ];
    state.prepare_continuation(&geometry).unwrap();
    assert!(matches!(
        state.latent_state(0),
        Err(ContinuationError::NotLatent { layer: 0, .. })
    ));
    assert!(state.latent_state(1).unwrap().is_empty());
    state.latent_state(1).unwrap().append(vec![1.0, -0.0, 3.0]);
    state.append(0, vec![4.0, 5.0], vec![6.0, 7.0]);
    state.set_position(1);
    state.prepare_continuation(&geometry).unwrap();
    let rows = state.latent_state(1).unwrap().rows();
    assert_eq!(rows, &[vec![1.0, -0.0, 3.0]]);
    assert_eq!(rows[0][1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(state.rows(0).to_owned_rows().0, &[vec![4.0, 5.0]]);
    assert_eq!(state.position(), 1);
}

/// C1 of CONTINUATION-PLUGIN-1: the canonical provider names itself, and
/// the two built-ins are distinct authorities — the handoff that will carry
/// an identity (C4) must be able to tell them apart.
#[test]
fn canonical_states_a_valid_identity_distinct_from_row() {
    let canonical = CanonicalKvState::identity();
    canonical.validate().unwrap();
    assert_eq!(canonical.to_string(), "canonical/v1");
    assert_ne!(canonical, RowKvState::identity());
}

/// C2: the shipped registry is a fresh VALUE on every call — registering
/// into one leaves the next untouched — and it holds exactly the two
/// built-ins, each selectable against a real plan's geometry.
#[test]
fn shipped_continuations_is_a_fresh_value_holding_both_built_ins() {
    use larql_vindex::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
    use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;

    let mut first = crate::shipped_continuations();
    assert_eq!(
        first.identities(),
        [RowKvState::identity(), CanonicalKvState::identity()]
    );
    let dup = first
        .register(Box::new(crate::CanonicalFactory))
        .unwrap_err();
    assert!(dup.to_string().contains("canonical/v1"), "{dup}");
    assert_eq!(crate::shipped_continuations().len(), 2);

    let (_dir, plan, _store) = fixture();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    for identity in first.identities() {
        let selected = first
            .select(&identity, &ContinuationConfig::empty(), &geometry)
            .unwrap();
        assert_eq!(selected.authority().identity, identity);
        let mut built = selected.build();
        built.prepare_continuation(&geometry).unwrap();
        assert_eq!(built.position(), 0);
    }
}
