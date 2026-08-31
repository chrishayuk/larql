//! VI3-INF-2 gates: continuation state behind the [`KvState`] seam.
//!
//! Three claims, each pinned separately:
//!
//! - the indirection changed nothing — a caller-provided [`RowKvState`]
//!   produces logits bit-identical to the session-owned default (the
//!   decode-vs-batch gates in `tests/decode.rs` already pin the default
//!   against the batch traversal, so equality here chains all three);
//! - the provider learns its geometry **from the plan** — KV row width
//!   and sliding/full window arrive via `prepare`, not from any family
//!   registry;
//! - the state outlives the session — the caller still holds every row
//!   after the session is dropped, which is what makes continuation
//!   state a caller-side policy at all.

use super::golden::{G_HEAD_DIM, G_KV_HEADS, G_LAYERS, G_TOKENS, G_WINDOW};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{
    plan_kv_geometry, KvState, LayerKvGeometry, RowKvState,
};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::{execute_plan, prefill_plan};

/// A provider that records the calls the session makes, delegating
/// storage to [`RowKvState`] so the run still completes.
#[derive(Default)]
struct RecordingKvState {
    inner: RowKvState,
    prepared: Vec<Vec<LayerKvGeometry>>,
    append_order: Vec<usize>,
    row_lens: Vec<usize>,
}

impl KvState for RecordingKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.prepared.push(layers.to_vec());
        self.inner.prepare(layers);
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.append_order.push(layer);
        self.row_lens.push(key.len());
        assert_eq!(key.len(), value.len(), "K and V rows must share a width");
        self.inner.append(layer, key, value);
    }

    fn keys(&self, layer: usize) -> &[Vec<f32>] {
        self.inner.keys(layer)
    }

    fn values(&self, layer: usize) -> &[Vec<f32>] {
        self.inner.values(layer)
    }

    fn position(&self) -> usize {
        self.inner.position()
    }

    fn set_position(&mut self, position: usize) {
        self.inner.set_position(position);
    }

    /// A KV-only recorder: it records rows, so it says so.
    fn recurrent_state(
        &mut self,
        layer: usize,
    ) -> Result<&mut super::super::continuation::RecurrentState, super::super::kv::ContinuationError>
    {
        Err(super::super::kv::ContinuationError::RecurrentUnsupported {
            provider: "RecordingKvState",
            layer,
        })
    }

    /// Nor a latent cache — stated, not inherited.
    fn latent_state(
        &mut self,
        layer: usize,
    ) -> Result<&mut super::super::continuation::LatentKvRows, super::super::kv::ContinuationError>
    {
        Err(super::super::kv::ContinuationError::LatentUnsupported {
            provider: "RecordingKvState",
            layer,
        })
    }
}

#[test]
fn plan_geometry_names_row_width_and_window_per_layer() {
    let (_c, plan, _store) = super::decode::fixture();
    let geometry = plan_kv_geometry(&plan);
    assert_eq!(
        geometry,
        vec![
            LayerKvGeometry {
                kv_dim: G_KV_HEADS * G_HEAD_DIM,
                window: Some(G_WINDOW),
            },
            LayerKvGeometry {
                kv_dim: G_KV_HEADS * G_HEAD_DIM,
                window: None,
            },
        ],
        "the miniature's sliding+full split must be explicit in the geometry"
    );
}

#[test]
fn a_caller_owned_provider_matches_the_session_owned_default_bit_for_bit() {
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();

    let default_logits = super::decode::decode_logits(&plan, &store, &backend);

    let mut kv = RowKvState::default();
    let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
    let mut provided = None;
    for &token in G_TOKENS.iter() {
        provided = session.step(token).unwrap().logits;
    }
    assert_eq!(
        provided.as_deref(),
        Some(default_logits.as_slice()),
        "caller-owned continuation state must not change the arithmetic"
    );
}

// ── VI3-INF-3: batch prefill into the caller's provider ──
//
// Four arms, compared at the seam. A = the batch traversal
// (`execute_plan`); B = batch prefill into a caller RowKvState, then
// decode resumes the SAME provider; C = tokenwise prefill+decode over
// its own RowKvState; D = the checkpoint-driven production forward,
// already pinned ≡ A layer-by-layer by `tests/parity.rs` and the
// golden oracle, so D chains through A without re-implementation here.
//
// The decisive control is B vs C **on the state itself**: if two
// different traversals leave RowKvState bit-identical, continuation
// state is a genuinely execution-independent representation — a
// stronger claim than their next logits agreeing.

fn assert_rows_equal(a: &RowKvState, b: &RowKvState, layers: usize) {
    assert_eq!(a.position(), b.position(), "logical positions diverge");
    for layer in 0..layers {
        assert_eq!(
            a.keys(layer),
            b.keys(layer),
            "K rows diverge at layer {layer}"
        );
        assert_eq!(
            a.values(layer),
            b.values(layer),
            "V rows diverge at layer {layer}"
        );
    }
}

fn assert_prefill_matches_batch<B: PlanBackend>(backend: &B) {
    let (_c, plan, store) = super::decode::fixture();
    let batch = execute_plan(&plan, &store, &G_TOKENS, backend).unwrap();

    let mut kv = RowKvState::default();
    let prefilled = prefill_plan(&plan, &store, &G_TOKENS, backend, &mut kv).unwrap();

    assert_eq!(prefilled.logits, batch.logits, "prefill logits diverge");
    assert_eq!(
        prefilled.final_hidden, batch.final_hidden,
        "prefill final hidden diverges"
    );
    assert_eq!(kv.position(), G_TOKENS.len());
}

#[test]
fn reference_batch_prefill_matches_the_batch_traversal_bit_for_bit() {
    assert_prefill_matches_batch(&ReferenceBackend::new());
}

#[test]
fn production_batch_prefill_matches_the_batch_traversal_bit_for_bit() {
    assert_prefill_matches_batch(&ProductionBackend::new());
}

/// B vs C on the state: batch prefill and tokenwise prefill must leave
/// the provider bit-identical — rows AND logical position.
#[test]
fn batch_and_tokenwise_prefill_leave_bit_identical_state() {
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();

    let mut batch_kv = RowKvState::default();
    let batch_out = prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut batch_kv).unwrap();

    let mut step_kv = RowKvState::default();
    let mut step_logits = None;
    {
        let mut session =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut step_kv).unwrap();
        for &token in G_TOKENS.iter() {
            step_logits = session.step(token).unwrap().logits;
        }
    }

    assert_rows_equal(&batch_kv, &step_kv, G_LAYERS);
    assert_eq!(
        batch_out.logits, step_logits,
        "prefill-final logits diverge"
    );
}

/// The handoff: batch prefill populates the provider, a decode session
/// resumes the SAME provider (its start position read from the state,
/// not passed by anyone), and every continuation step's logits match a
/// session that walked the whole sequence tokenwise itself.
#[test]
fn a_prefilled_provider_resumes_decode_bit_for_bit() {
    const CONTINUATION_STEPS: usize = 8;
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();

    // Oracle: one session walks prompt + continuation entirely tokenwise.
    let mut oracle = DecodeSession::new(&plan, &store, &backend).unwrap();
    let mut logits = None;
    for &token in G_TOKENS.iter() {
        logits = oracle.step(token).unwrap().logits;
    }
    let mut oracle_logits = vec![logits.unwrap()];
    let mut oracle_ids = Vec::new();
    for _ in 0..CONTINUATION_STEPS {
        let next = argmax(oracle_logits.last().unwrap());
        oracle_ids.push(next);
        oracle_logits.push(oracle.step(next).unwrap().logits.unwrap());
    }

    // Arm B: batch prefill, then resume the same provider.
    let mut kv = RowKvState::default();
    let prefilled = prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut kv).unwrap();
    assert_eq!(
        prefilled.logits.as_ref(),
        Some(&oracle_logits[0]),
        "prefill-final logits diverge from the tokenwise oracle"
    );
    let mut resumed = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
    assert_eq!(resumed.position(), G_TOKENS.len(), "resume position");
    for (step, (&id, expected)) in oracle_ids.iter().zip(&oracle_logits[1..]).enumerate() {
        let stepped = resumed.step(id).unwrap().logits.unwrap();
        assert_eq!(&stepped, expected, "continuation step {step} diverges");
    }
    assert_eq!(resumed.position(), G_TOKENS.len() + CONTINUATION_STEPS);
}

/// Chunked prefill: extending a held state batch-style must equal one
/// whole-prompt prefill — rows, position, and final logits.
#[test]
fn chunked_prefill_extends_the_state_bit_identically() {
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();

    let mut whole = RowKvState::default();
    let whole_out = prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut whole).unwrap();

    let (head, tail) = G_TOKENS.split_at(2);
    let mut chunked = RowKvState::default();
    prefill_plan(&plan, &store, head, &backend, &mut chunked).unwrap();
    assert_eq!(chunked.position(), head.len());
    let chunked_out = prefill_plan(&plan, &store, tail, &backend, &mut chunked).unwrap();

    assert_rows_equal(&whole, &chunked, G_LAYERS);
    assert_eq!(whole_out.logits, chunked_out.logits);
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

#[test]
fn the_provider_learns_geometry_from_the_plan_and_keeps_the_rows() {
    let (_c, plan, store) = super::decode::fixture();
    let backend = ReferenceBackend::new();
    let mut kv = RecordingKvState::default();
    {
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, &mut kv).unwrap();
        for &token in G_TOKENS.iter() {
            session.step(token).unwrap();
        }
    }

    // prepare: exactly once, with the plan-derived geometry.
    assert_eq!(kv.prepared, vec![plan_kv_geometry(&plan)]);

    // appends: layers interleave within one position — 0..n per step.
    let per_step: Vec<usize> = (0..G_LAYERS).collect();
    let expected: Vec<usize> = G_TOKENS.iter().flat_map(|_| per_step.clone()).collect();
    assert_eq!(kv.append_order, expected);

    // every row has the plan's KV width.
    assert!(kv.row_lens.iter().all(|&l| l == G_KV_HEADS * G_HEAD_DIM));

    // the state survives the session: one row per consumed position.
    for layer in 0..G_LAYERS {
        assert_eq!(kv.keys(layer).len(), G_TOKENS.len());
        assert_eq!(kv.values(layer).len(), G_TOKENS.len());
    }
}

/// **Announcement, not reset** — the contract every region shares.
///
/// A provider is announced its geometry before each traversal, and a
/// prefilled state being resumed must keep what it holds: rows,
/// recurrent buffers and latent rows alike. Re-announcing is how a long
/// prompt prefills in chunks and how a decode session continues after
/// one, so a reset here would silently restart the conversation while
/// every shape still matched.
#[test]
fn re_announcing_the_geometry_keeps_every_region_it_already_holds() {
    use super::super::continuation::{
        LayerContinuationGeometry, LayerLatentKvGeometry, RecurrentBufferGeometry,
        RecurrentGeometry, StateInitialization,
    };
    use super::super::kv::{ContinuationProvider, LayerKvGeometry, RowKvState};

    let geometry = [
        LayerContinuationGeometry::Kv(LayerKvGeometry {
            kv_dim: 2,
            window: None,
        }),
        LayerContinuationGeometry::Recurrent(RecurrentGeometry::single(RecurrentBufferGeometry {
            shape: vec![2, 2],
            dtype: larql_models::inventory::report::RecurrentStateDtype::Float32,
            initialization: StateInitialization::Zeros,
        })),
        LayerContinuationGeometry::LatentKv(LayerLatentKvGeometry { width: 3 }),
    ];

    let mut provider = RowKvState::default();
    provider.prepare_continuation(&geometry).unwrap();

    // Fill one of each region, and advance the logical position.
    provider.append(0, vec![1.0, 2.0], vec![3.0, 4.0]);
    provider
        .recurrent_state(1)
        .unwrap()
        .buffer_mut(0)
        .cells_mut()[0] = 9.0;
    provider
        .latent_state(2)
        .unwrap()
        .append(vec![5.0, 6.0, 7.0]);
    provider.set_position(1);

    // The same geometry, announced again — as the next traversal does.
    provider.prepare_continuation(&geometry).unwrap();

    assert_eq!(provider.keys(0), &[vec![1.0, 2.0]], "rows survive");
    assert_eq!(
        provider.recurrent_state(1).unwrap().buffer(0).cells()[0],
        9.0,
        "recurrent buffers survive"
    );
    assert_eq!(
        provider.latent_state(2).unwrap().rows(),
        &[vec![5.0, 6.0, 7.0]],
        "latent rows survive — the region added last is not the one that resets"
    );
    assert_eq!(provider.position(), 1, "and so does the logical position");

    // The plain KV announcement takes the same path.
    provider.prepare(
        &[LayerKvGeometry {
            kv_dim: 2,
            window: None,
        }; 3],
    );
    assert_eq!(provider.keys(0).len(), 1, "still one row after re-prepare");
}

/// Every refusal says which provider, which layer, and which region —
/// the three things a caller needs to act on one.
///
/// Asserted per variant rather than through one formatted string,
/// because these messages are the seam's whole diagnostic surface and a
/// copy-paste between arms is exactly the defect that would survive a
/// looser test.
#[test]
fn each_continuation_refusal_names_provider_layer_and_region() {
    use super::super::kv::ContinuationError;

    let cases = [
        (
            ContinuationError::RecurrentUnsupported {
                provider: "P",
                layer: 3,
            },
            ["holds no recurrent state", "layer 3"],
        ),
        (
            ContinuationError::NotRecurrent {
                provider: "P",
                layer: 3,
            },
            ["not a recurrent layer", "dispatch bug"],
        ),
        (
            ContinuationError::LatentUnsupported {
                provider: "P",
                layer: 3,
            },
            ["no per-position latent cache", "layer 3"],
        ),
        (
            ContinuationError::NotLatent {
                provider: "P",
                layer: 3,
            },
            ["keeps no latent cache", "dispatch bug"],
        ),
    ];
    for (error, expected) in &cases {
        let rendered = error.to_string();
        assert!(rendered.contains('P'), "{rendered}");
        for phrase in expected {
            assert!(
                rendered.contains(phrase),
                "{error:?} rendered as {rendered}"
            );
        }
    }
    // The four render differently — a shared message would make the
    // wrong half of a hybrid look like the missing one.
    let rendered: std::collections::BTreeSet<String> =
        cases.iter().map(|(e, _)| e.to_string()).collect();
    assert_eq!(rendered.len(), cases.len());
}
