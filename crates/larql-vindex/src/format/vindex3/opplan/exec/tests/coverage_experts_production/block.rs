//! **A block of positions reaches placed experts through one provider call
//! per layer.**
//!
//! Over coordinator operands with a bound provider, the block traversal
//! hands every position of a routed layer to the provider together. Each
//! position's logits must be the bits sequential local decode gives, and
//! the provider must see one multi-row call per layer and no single-row
//! call of the traversal's own.
use super::fixture::*;
use super::placed::{bits, Grid};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    kv::RowKvState,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
    routed_experts::{ExpertOutput, ExpertRow, RoutedExpertProvider, RowExpertOutput},
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Tokens decoded one at a time before the block, so it starts beyond the
/// sliding window.
const PREFIX: [u32; 5] = [3, 17, 28, 0, 11];
/// The block: wider than the window, with a repeated token.
const BLOCK: [u32; 6] = [11, 3, 17, 28, 0, 3];

/// Counts multi-row calls and the rows they carried, and answers each row
/// through the wrapped provider's single-row operation.
struct Counting {
    inner: Grid,
    calls: AtomicUsize,
    rows: AtomicUsize,
    direct: AtomicUsize,
}

impl RoutedExpertProvider for Counting {
    fn apply(
        &self,
        layer: usize,
        x: &[f32],
        selected: &[usize],
    ) -> Result<Vec<ExpertOutput>, VindexError> {
        self.direct.fetch_add(1, Ordering::SeqCst);
        self.inner.apply(layer, x, selected)
    }

    fn apply_rows(
        &self,
        layer: usize,
        rows: &[ExpertRow<'_>],
    ) -> Result<Vec<RowExpertOutput>, VindexError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.rows.fetch_add(rows.len(), Ordering::SeqCst);
        self.inner.apply_rows(layer, rows)
    }
}

#[test]
fn a_block_crosses_to_placed_experts_once_per_layer_and_matches_local_decode() {
    for topology in [vec![(0, 2, 0, 4)], vec![(0, 2, 0, 2), (0, 2, 2, 4)]] {
        let fixture = routed_fixture();
        let backend = ProductionBackend::new();
        let load =
            |slice| PreparedOperands::load(&fixture.plan, &fixture.store, &backend, slice).unwrap();
        let local = load(ExecutionSlice::Full);
        let workers = topology
            .into_iter()
            .map(|(start, end, expert_start, expert_end)| {
                load(ExecutionSlice::RoutedExperts {
                    start,
                    end,
                    expert_start,
                    expert_end,
                })
            })
            .collect();
        let provider = Arc::new(Counting {
            inner: Grid(workers),
            calls: AtomicUsize::new(0),
            rows: AtomicUsize::new(0),
            direct: AtomicUsize::new(0),
        });
        let mut coordinator = load(ExecutionSlice::RoutedExpertCoordinator);
        coordinator
            .bind_routed_expert_provider(provider.clone())
            .unwrap();

        let mut local_kv = RowKvState::default();
        let mut sequential =
            DecodeSession::over_prepared(&fixture.plan, &local, &backend, &mut local_kv).unwrap();
        let mut remote_kv = RowKvState::default();
        let mut block =
            DecodeSession::over_prepared(&fixture.plan, &coordinator, &backend, &mut remote_kv)
                .unwrap();
        for id in PREFIX {
            sequential.step(id).unwrap();
            block.step(id).unwrap();
        }
        let expected: Vec<Vec<u32>> = BLOCK
            .iter()
            .map(|id| bits(&sequential.step(*id).unwrap().logits.unwrap()))
            .collect();

        let before = provider.direct.load(Ordering::SeqCst);
        let rows = block.step_many_logits(&BLOCK).unwrap();
        let got: Vec<Vec<u32>> = rows.iter().map(|row| bits(row)).collect();
        assert_eq!(got, expected);
        // One call per routed layer, carrying the whole block; the
        // traversal itself made no single-row call.
        let layers = fixture.plan.layers.len();
        assert_eq!(provider.calls.load(Ordering::SeqCst), layers);
        assert_eq!(provider.rows.load(Ordering::SeqCst), layers * BLOCK.len());
        assert_eq!(provider.direct.load(Ordering::SeqCst), before);

        // The state the block left: the next position still agrees.
        let next = PREFIX[0];
        assert_eq!(
            bits(&block.step(next).unwrap().logits.unwrap()),
            bits(&sequential.step(next).unwrap().logits.unwrap())
        );
    }
}
