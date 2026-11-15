//! **Rollback restores logical continuation state.**
//!
//! `truncate(position)` must leave a provider indistinguishable, to every
//! later read, from one that only ever advanced to `position`. These gates
//! pin that for the row provider across the sliding-window boundary, pin
//! that a refused rewind changes nothing, and pin that a provider which
//! does not implement rollback refuses.

use super::golden::{G_LAYERS, G_TOKENS, G_VOCAB, G_WINDOW};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::continuation::{
    LatentKvRows, LayerContinuationGeometry, LayerLatentKvGeometry, RecurrentBufferGeometry,
    RecurrentGeometry, RecurrentState, StateInitialization,
};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::{
    ContinuationError, KvState, LayerKvGeometry, RowKvState,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::ComponentOpPlan;
use larql_models::inventory::report::RecurrentStateDtype;

/// Longest prefix kept before a rewind: well past the sliding window, so
/// rewinds land before, on and beyond the boundary.
const KEPT_MAX: usize = 3 * G_WINDOW;
/// Positions decoded and then discarded.
const DISCARDED: [usize; 3] = [1, G_WINDOW, 2 * G_WINDOW + 1];
/// Positions decoded after the rewind and compared with the reference.
const CONTINUED: usize = 2 * G_WINDOW + 2;
/// Width of the synthetic rows used where no model is needed.
const ROW_WIDTH: usize = 4;

/// A deterministic token stream longer than the golden one, inside the vocabulary.
fn token(index: usize) -> u32 {
    (G_TOKENS[index % G_TOKENS.len()] as usize * (index / G_TOKENS.len() + 1) % G_VOCAB) as u32
}

/// Tokens that are decoded and thrown away: deliberately not the tokens
/// the continuation uses, so a stale row would change the result.
fn discarded_token(index: usize) -> u32 {
    ((token(index) as usize + 7) % G_VOCAB) as u32
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Decode `tokens` over `kv`, returning every position's logits.
fn decode<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    kv: &mut RowKvState,
    tokens: &[u32],
) -> Vec<Vec<u32>> {
    let mut session = DecodeSession::with_kv_state(plan, store, backend, kv).unwrap();
    tokens
        .iter()
        .map(|t| bits(&session.step(*t).unwrap().logits.unwrap()))
        .collect()
}

/// One layer's held rows as bits: a key and a value per position.
type LayerBits = Vec<(Vec<u32>, Vec<u32>)>;

/// Every row a provider holds, as bits, per layer.
fn snapshot(kv: &mut RowKvState) -> (usize, Vec<LayerBits>) {
    let layers = (0..G_LAYERS)
        .map(|layer| {
            kv.prepare_layer(layer);
            let view = kv.rows(layer);
            (view.base()..view.end())
                .map(|p| (bits(view.key(p)), bits(view.value(p))))
                .collect()
        })
        .collect();
    (kv.position(), layers)
}

fn rewind_gate<B: PlanBackend>(backend: &B) {
    let (_container, plan, store) = super::decode::fixture();
    for kept in 0..=KEPT_MAX {
        let prefix: Vec<u32> = (0..kept).map(token).collect();
        let continuation: Vec<u32> = (kept..kept + CONTINUED).map(token).collect();

        // The reference never sees a discarded position.
        let mut clean = RowKvState::default();
        decode(&plan, &store, backend, &mut clean, &prefix);
        let clean_state = snapshot(&mut clean);
        let expected = decode(&plan, &store, backend, &mut clean, &continuation);

        for discarded in DISCARDED {
            let mut kv = RowKvState::default();
            decode(&plan, &store, backend, &mut kv, &prefix);
            let rejected: Vec<u32> = (kept..kept + discarded).map(discarded_token).collect();
            decode(&plan, &store, backend, &mut kv, &rejected);
            assert_eq!(kv.position(), kept + discarded);

            kv.truncate(kept).unwrap();
            // Logical state: position and every held row of every layer.
            assert_eq!(
                snapshot(&mut kv),
                clean_state,
                "kept {kept}, discarded {discarded}: state after rewind"
            );
            // And every later read agrees, across the window boundary.
            assert_eq!(
                decode(&plan, &store, backend, &mut kv, &continuation),
                expected,
                "kept {kept}, discarded {discarded}: continuation after rewind"
            );
        }
    }
}

#[test]
fn rewinding_then_advancing_equals_never_having_advanced_production() {
    rewind_gate(&ProductionBackend::new());
}

#[test]
fn rewinding_then_advancing_equals_never_having_advanced_reference() {
    rewind_gate(&ReferenceBackend::new());
}

#[test]
fn repeated_and_chained_rewinds_leave_no_stale_state() {
    let (_container, plan, store) = super::decode::fixture();
    let backend = ProductionBackend::new();
    // Advance 8, keep 3; advance 4 more, keep 2 of them; then decode on.
    let (first, keep_first, second, keep_second) = (8usize, 3usize, 4usize, 2usize);
    let committed = keep_first + keep_second;
    let kept: Vec<u32> = (0..committed).map(token).collect();
    let continuation: Vec<u32> = (committed..committed + CONTINUED).map(token).collect();

    let mut clean = RowKvState::default();
    let clean_logits = decode(&plan, &store, &backend, &mut clean, &kept);
    let expected = decode(&plan, &store, &backend, &mut clean, &continuation);

    let mut kv = RowKvState::default();
    // Block one: the first `keep_first` tokens are the real ones.
    let block_one: Vec<u32> = (0..first)
        .map(|i| {
            if i < keep_first {
                token(i)
            } else {
                discarded_token(i)
            }
        })
        .collect();
    let one = decode(&plan, &store, &backend, &mut kv, &block_one);
    assert_eq!(one[..keep_first], clean_logits[..keep_first]);
    kv.truncate(keep_first).unwrap();
    // Block two, from the committed position.
    let block_two: Vec<u32> = (0..second)
        .map(|i| {
            let position = keep_first + i;
            if i < keep_second {
                token(position)
            } else {
                discarded_token(position)
            }
        })
        .collect();
    let two = decode(&plan, &store, &backend, &mut kv, &block_two);
    assert_eq!(two[..keep_second], clean_logits[keep_first..committed]);
    kv.truncate(committed).unwrap();
    assert_eq!(snapshot(&mut kv).0, committed);
    assert_eq!(
        decode(&plan, &store, &backend, &mut kv, &continuation),
        expected
    );

    // The same block verified and discarded twice gives the same logits.
    let mut again = RowKvState::default();
    decode(&plan, &store, &backend, &mut again, &kept);
    let block: Vec<u32> = (committed..committed + first)
        .map(discarded_token)
        .collect();
    let once = decode(&plan, &store, &backend, &mut again, &block);
    again.truncate(committed).unwrap();
    let twice = decode(&plan, &store, &backend, &mut again, &block);
    assert_eq!(once, twice);
}

/// A row provider holding synthetic rows: `rows[layer]` positions per layer.
fn synthetic(rows: &[usize], position: usize) -> RowKvState {
    let mut kv = RowKvState::default();
    let geometry = LayerKvGeometry {
        kv_dim: ROW_WIDTH,
        head_dim: ROW_WIDTH,
        window: None,
        history: crate::format::vindex3::opplan::exec::kv::HistoryRange::Full,
    };
    kv.prepare(&vec![geometry; rows.len()]);
    for (layer, count) in rows.iter().enumerate() {
        for p in 0..*count {
            let value = (layer * 100 + p) as f32;
            kv.append(layer, vec![value; ROW_WIDTH], vec![-value; ROW_WIDTH]);
        }
    }
    kv.set_position(position);
    kv
}

fn synthetic_snapshot(kv: &mut RowKvState, layers: usize) -> (usize, Vec<Vec<Vec<u32>>>) {
    let rows = (0..layers)
        .map(|layer| {
            let view = kv.rows(layer);
            (view.base()..view.end())
                .map(|p| bits(view.key(p)))
                .collect()
        })
        .collect();
    (kv.position(), rows)
}

#[test]
fn a_refused_rewind_leaves_every_layer_untouched() {
    // Forward: the state continues from 4.
    let mut kv = synthetic(&[4, 4], 4);
    let before = synthetic_snapshot(&mut kv, 2);
    let error = kv.truncate(5).unwrap_err();
    assert!(matches!(
        error,
        ContinuationError::TruncateRefused { position: 5, .. }
    ));
    assert!(error.to_string().contains("cannot move forward"), "{error}");
    assert!(error.to_string().contains("state unchanged"), "{error}");
    assert_eq!(synthetic_snapshot(&mut kv, 2), before);

    // The LAST layer cannot reach the target. The first layer could have
    // been cut already if layers were rewound one at a time.
    let mut kv = synthetic(&[6, 2], 6);
    let before = synthetic_snapshot(&mut kv, 2);
    let error = kv.truncate(3).unwrap_err();
    assert!(error.to_string().contains("layer 1"), "{error}");
    assert_eq!(
        synthetic_snapshot(&mut kv, 2),
        before,
        "layer 0 was cut before layer 1 refused"
    );

    // Permitted rewinds on the same shapes, for contrast.
    let mut kv = synthetic(&[6, 6], 6);
    kv.truncate(6).unwrap();
    assert_eq!(synthetic_snapshot(&mut kv, 2).0, 6);
    kv.truncate(2).unwrap();
    let after = synthetic_snapshot(&mut kv, 2);
    assert_eq!(after.0, 2);
    assert!(after.1.iter().all(|layer| layer.len() == 2));
    kv.truncate(0).unwrap();
    assert!(synthetic_snapshot(&mut kv, 2).1.iter().all(Vec::is_empty));
}

#[test]
fn rows_held_beyond_the_position_are_discarded_by_a_rewind_to_it() {
    // What a step that failed part way leaves behind: rows ahead of the
    // position. Rewinding to the current position removes them.
    let mut kv = synthetic(&[5, 4], 3);
    kv.truncate(3).unwrap();
    let after = synthetic_snapshot(&mut kv, 2);
    assert_eq!(after.0, 3);
    assert!(after.1.iter().all(|layer| layer.len() == 3));
}

#[test]
fn recurrent_and_latent_state_refuse_a_rewind_before_anything_moves() {
    let rows = LayerKvGeometry {
        kv_dim: ROW_WIDTH,
        head_dim: ROW_WIDTH,
        window: None,
        history: crate::format::vindex3::opplan::exec::kv::HistoryRange::Full,
    };
    let recurrent = RecurrentGeometry::single(RecurrentBufferGeometry {
        shape: vec![ROW_WIDTH],
        dtype: RecurrentStateDtype::Float32,
        initialization: StateInitialization::Zeros,
    });
    for (geometry, needle) in [
        (LayerContinuationGeometry::Recurrent(recurrent), "recurrent"),
        (
            LayerContinuationGeometry::LatentKv(LayerLatentKvGeometry { width: ROW_WIDTH }),
            "latent",
        ),
    ] {
        let mut kv = RowKvState::default();
        kv.prepare_continuation(&[LayerContinuationGeometry::Kv(rows), geometry])
            .unwrap();
        for p in 0..3 {
            kv.append(0, vec![p as f32; ROW_WIDTH], vec![p as f32; ROW_WIDTH]);
        }
        kv.set_position(3);
        let error = kv.truncate(1).unwrap_err().to_string();
        assert!(error.contains(needle), "{error}");
        assert_eq!(kv.position(), 3);
        assert_eq!(kv.rows(0).end(), 3, "rows were cut before the refusal");
    }
}

/// A provider that states everything the trait requires and nothing more.
#[derive(Default)]
struct NoRollback(RowKvState);

impl KvState for NoRollback {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.0.prepare(layers)
    }
    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.0.append(layer, key, value)
    }
    fn rows(&self, layer: usize) -> super::super::kv_view::KvView<'_> {
        self.0.rows(layer)
    }
    fn prepare_layer(&mut self, layer: usize) {
        self.0.prepare_layer(layer)
    }
    fn position(&self) -> usize {
        self.0.position()
    }
    fn set_position(&mut self, position: usize) {
        self.0.set_position(position)
    }
    fn invalidate(&mut self, reason: &str) {
        self.0.invalidate(reason)
    }
    fn invalidated(&self) -> Option<&str> {
        self.0.invalidated()
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        self.0.recurrent_state(layer)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.0.latent_state(layer)
    }
}

#[test]
fn a_provider_that_does_not_implement_rollback_refuses_and_is_unchanged() {
    let mut kv = NoRollback(synthetic(&[4, 4], 4));
    for target in [0, 2, 4, 9] {
        let error = kv.truncate(target).unwrap_err();
        assert!(
            matches!(error, ContinuationError::TruncateUnsupported { position } if position == target)
        );
        assert!(
            error.to_string().contains("does not support rollback"),
            "{error}"
        );
    }
    assert_eq!(kv.position(), 4);
    assert_eq!(kv.rows(0).end(), 4);
    assert_eq!(kv.rows(1).end(), 4);
}
