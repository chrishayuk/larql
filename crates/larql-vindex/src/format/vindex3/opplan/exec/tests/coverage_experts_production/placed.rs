use super::fixture::*;
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    kv::RowKvState,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
    routed_experts::{ExpertOutput, PreparedRoutedExperts, RoutedExpertProvider},
};
use std::sync::Arc;

struct Grid(Vec<PreparedOperands>);
impl RoutedExpertProvider for Grid {
    fn apply(
        &self,
        layer: usize,
        x: &[f32],
        selected: &[usize],
    ) -> Result<Vec<ExpertOutput>, VindexError> {
        let mut out = Vec::new();
        for worker in &self.0 {
            let ExecutionSlice::RoutedExperts {
                start,
                end,
                expert_start,
                expert_end,
            } = worker.slice()
            else {
                unreachable!()
            };
            if !(*start..*end).contains(&layer) {
                continue;
            }
            let ids: Vec<_> = selected
                .iter()
                .copied()
                .filter(|id| (*expert_start..*expert_end).contains(id))
                .collect();
            if !ids.is_empty() {
                out.extend(worker.routed_experts().unwrap().apply(
                    &ProductionBackend::new(),
                    layer,
                    &ids,
                    x,
                )?);
            }
        }
        // Completion order deliberately disagrees with production selection.
        out.sort_by_key(|r| r.expert);
        out.reverse();
        Ok(out)
    }
}
#[test]
fn owned_expert_reads_and_shuffled_outputs_preserve_continuation_bits() {
    for topology in [
        vec![(0, 2, 0, 4)],
        vec![(0, 2, 0, 2), (0, 2, 2, 4)],
        vec![(0, 1, 0, 4), (1, 2, 0, 1), (1, 2, 1, 4)],
    ] {
        let fixture = routed_fixture();
        let backend = ProductionBackend::new();
        let local = PreparedOperands::load(
            &fixture.plan,
            &fixture.store,
            &backend,
            ExecutionSlice::Full,
        )
        .unwrap();
        let mut workers = Vec::new();
        for (start, end, expert_start, expert_end) in topology {
            let slice = ExecutionSlice::RoutedExperts {
                start,
                end,
                expert_start,
                expert_end,
            };
            let ledger =
                PreparedRoutedExperts::regions_for(&fixture.plan, (&fixture.store).into(), &slice)
                    .unwrap();
            assert!(ledger.iter().all(|r| r.operand.tensor.contains("experts.")));
            let region = &ledger[0];
            let before = fixture.store.bytes_read();
            for (total, offset, len) in [
                (region.total_bytes + 1, region.offset, region.bytes),
                (region.total_bytes, region.total_bytes, 1),
                (region.total_bytes, u64::MAX, 2),
            ] {
                assert!(fixture
                    .store
                    .load_raw_range(&region.operand, total, offset, len)
                    .is_err());
            }
            assert_eq!(fixture.store.bytes_read(), before);
            let worker =
                PreparedOperands::load(&fixture.plan, &fixture.store, &backend, slice).unwrap();
            assert_eq!(
                fixture.store.bytes_read() - before,
                ledger.iter().map(|r| r.bytes).sum::<u64>()
            );
            assert_eq!(worker.routed_experts().unwrap().regions(), ledger);
            let census = worker.residency_census();
            assert_eq!(census.attention.total(), 0);
            assert_eq!(census.embedding.total(), 0);
            assert_eq!(
                census.glue.total(),
                (end - start) * (expert_end - expert_start) * (2 * INTER + HIDDEN) * 4
            );
            assert_eq!(
                census.ffn.total(),
                (end - start) * (expert_end - expert_start) * 3 * HIDDEN * INTER * 4
            );
            assert!(worker
                .routed_experts()
                .unwrap()
                .apply(&backend, end, &[expert_start], &[0.1; HIDDEN])
                .is_err());
            assert!(worker
                .routed_experts()
                .unwrap()
                .apply(&backend, start, &[expert_end], &[0.1; HIDDEN])
                .is_err());
            let a = worker
                .routed_experts()
                .unwrap()
                .apply(&backend, start, &[expert_start], &[0.1; HIDDEN])
                .unwrap();
            let b = worker
                .routed_experts()
                .unwrap()
                .apply(&backend, start, &[expert_start], &[0.1; HIDDEN])
                .unwrap();
            assert_eq!(bits(&a[0].row), bits(&b[0].row));
            workers.push(worker);
        }
        let mut coordinator = PreparedOperands::load(
            &fixture.plan,
            &fixture.store,
            &backend,
            ExecutionSlice::RoutedExpertCoordinator,
        )
        .unwrap();
        assert_eq!(coordinator.residency_census().ffn.total(), 0);
        coordinator
            .bind_routed_expert_provider(Arc::new(Grid(workers)))
            .unwrap();
        let (mut local_kv, mut remote_kv) = (RowKvState::default(), RowKvState::default());
        let mut a =
            DecodeSession::over_prepared(&fixture.plan, &local, &backend, &mut local_kv).unwrap();
        let mut b =
            DecodeSession::over_prepared(&fixture.plan, &coordinator, &backend, &mut remote_kv)
                .unwrap();
        // WINDOW=4: this crosses it repeatedly, rather than testing only prefill.
        for id in [3, 17, 28, 0, 11, 3, 17, 28, 0, 11] {
            assert_eq!(
                bits(&a.step(id).unwrap().logits.unwrap()),
                bits(&b.step(id).unwrap().logits.unwrap())
            );
        }
    }
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn selected_response_validation_rejects_incomplete_duplicate_and_nonfinite_rows() {
    use crate::format::vindex3::opplan::exec::routed_experts::reduce_selected;
    let row = |expert| ExpertOutput {
        expert,
        row: vec![1.0, 2.0],
    };
    let selected = [(2, 0.25), (1, 0.75)];
    assert!(reduce_selected(&selected, vec![row(2)], 2).is_err());
    assert!(reduce_selected(&selected, vec![row(2), row(2)], 2).is_err());
    assert!(reduce_selected(&selected, vec![row(2), row(0)], 2).is_err());
    assert!(reduce_selected(
        &selected,
        vec![
            row(2),
            ExpertOutput {
                expert: 1,
                row: vec![f32::NAN, 0.0]
            }
        ],
        2
    )
    .is_err());
    assert!(reduce_selected(
        &selected,
        vec![
            row(2),
            ExpertOutput {
                expert: 1,
                row: vec![0.0]
            }
        ],
        2
    )
    .is_err());
}

#[test]
fn reduction_uses_selection_order_even_when_arrival_order_changes_rounding() {
    use crate::format::vindex3::opplan::exec::routed_experts::reduce_selected;
    let selected = [(7, 1.0), (2, 1.0), (5, 1.0)];
    let rows = vec![
        ExpertOutput {
            expert: 5,
            row: vec![1.0],
        },
        ExpertOutput {
            expert: 7,
            row: vec![1.0e20],
        },
        ExpertOutput {
            expert: 2,
            row: vec![-1.0e20],
        },
    ];
    assert_eq!(reduce_selected(&selected, rows, 1).unwrap(), [1.0]);
}
