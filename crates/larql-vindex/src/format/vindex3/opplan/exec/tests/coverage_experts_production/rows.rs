//! The expert worker's multi-row entry point: grouped by expert, bit for
//! bit the single-row transform, and refused as a whole when any row is
//! malformed.
use super::fixture::*;
use crate::format::vindex3::opplan::exec::{
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
    routed_experts::{ExpertRow, RowsAccounting},
};

const LAYER: usize = 1;
/// Three rows over three experts: expert 1 is selected by every row.
const SELECTIONS: [[usize; TOP_K]; 3] = [[0, 1], [1, 2], [3, 1]];
const ROW_IDS: [u32; 3] = [40, 7, 19];

fn inputs() -> Vec<Vec<f32>> {
    (0..SELECTIONS.len())
        .map(|row| {
            (0..HIDDEN)
                .map(|i| ((row * HIDDEN + i) % 11) as f32 / 16.0 - 0.3)
                .collect()
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn worker(fixture: &RoutedFixture, experts: (usize, usize)) -> PreparedOperands {
    PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &ProductionBackend::new(),
        ExecutionSlice::RoutedExperts {
            start: 0,
            end: LAYERS,
            expert_start: experts.0,
            expert_end: experts.1,
        },
    )
    .unwrap()
}

#[test]
fn grouped_rows_are_the_single_row_transform_and_are_counted_as_grouping_only() {
    let fixture = routed_fixture();
    let backend = ProductionBackend::new();
    let prepared = worker(&fixture, (0, EXPERTS));
    let experts = prepared.routed_experts().unwrap();
    let inputs = inputs();
    let rows: Vec<ExpertRow<'_>> = (0..SELECTIONS.len())
        .map(|i| ExpertRow {
            id: ROW_IDS[i],
            input: &inputs[i],
            experts: &SELECTIONS[i],
        })
        .collect();

    let mut alone = std::collections::BTreeMap::new();
    for row in &rows {
        for output in experts
            .apply(&backend, LAYER, row.experts, row.input)
            .unwrap()
        {
            alone.insert((row.id, output.expert), bits(&output.row));
        }
    }

    for count in [1, rows.len()] {
        let mut accounting = RowsAccounting::default();
        let outputs = experts
            .apply_rows(&backend, LAYER, &rows[..count], Some(&mut accounting))
            .unwrap();
        let pairs = count * TOP_K;
        assert_eq!(outputs.len(), pairs);
        for output in &outputs {
            assert_eq!(
                Some(&bits(&output.output)),
                alone.get(&(output.row, output.expert)),
                "row {} expert {}",
                output.row,
                output.expert
            );
        }
        // Every row of an expert is finished before the next expert starts.
        let order: Vec<usize> = outputs.iter().map(|o| o.expert).collect();
        assert!(order.windows(2).all(|pair| pair[0] <= pair[1]), "{order:?}");
        let distinct: std::collections::BTreeSet<usize> = order.iter().copied().collect();
        assert_eq!(
            accounting,
            RowsAccounting {
                rows: count,
                pairs,
                distinct_experts: distinct.len(),
                // One transform per pair: grouping reorders the kernel's
                // calls, it does not remove any.
                transform_calls: pairs,
            }
        );
        if count == rows.len() {
            assert!(accounting.distinct_experts < accounting.pairs);
        }
        // Accounting is optional and never changes the result.
        let unaccounted = experts
            .apply_rows(&backend, LAYER, &rows[..count], None)
            .unwrap();
        assert_eq!(
            unaccounted
                .iter()
                .map(|o| bits(&o.output))
                .collect::<Vec<_>>(),
            outputs.iter().map(|o| bits(&o.output)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn one_malformed_row_refuses_the_whole_call() {
    let fixture = routed_fixture();
    let backend = ProductionBackend::new();
    let prepared = worker(&fixture, (0, EXPERTS / 2));
    let experts = prepared.routed_experts().unwrap();
    let good = vec![0.25; HIDDEN];
    let narrow = vec![0.25; HIDDEN - 1];
    let nonfinite = vec![f32::NAN; HIDDEN];
    let owned = [0usize, 1];
    let healthy = ExpertRow {
        id: 1,
        input: &good,
        experts: &owned,
    };
    assert!(experts
        .apply_rows(&backend, LAYER, &[healthy], None)
        .is_ok());

    let refusals: [(&str, Vec<ExpertRow<'_>>, usize); 8] = [
        ("no rows", vec![], LAYER),
        ("duplicate row id", vec![healthy, healthy], LAYER),
        (
            "duplicate expert in a row",
            vec![
                healthy,
                ExpertRow {
                    id: 2,
                    input: &good,
                    experts: &[1, 1],
                },
            ],
            LAYER,
        ),
        (
            "unowned expert",
            vec![
                healthy,
                ExpertRow {
                    id: 2,
                    input: &good,
                    experts: &[EXPERTS / 2],
                },
            ],
            LAYER,
        ),
        (
            "row selecting nothing",
            vec![
                healthy,
                ExpertRow {
                    id: 2,
                    input: &good,
                    experts: &[],
                },
            ],
            LAYER,
        ),
        (
            "narrow input",
            vec![
                healthy,
                ExpertRow {
                    id: 2,
                    input: &narrow,
                    experts: &owned,
                },
            ],
            LAYER,
        ),
        (
            "nonfinite input",
            vec![
                healthy,
                ExpertRow {
                    id: 2,
                    input: &nonfinite,
                    experts: &owned,
                },
            ],
            LAYER,
        ),
        ("layer outside the worker", vec![healthy], LAYERS),
    ];
    for (name, rows, layer) in refusals {
        assert!(
            experts.apply_rows(&backend, layer, &rows, None).is_err(),
            "{name} was accepted"
        );
    }
    // More experts than the layer's top-k is refused even when all are owned.
    let wide = worker(&fixture, (0, EXPERTS));
    let too_many: Vec<usize> = (0..=TOP_K).collect();
    assert!(wide
        .routed_experts()
        .unwrap()
        .apply_rows(
            &backend,
            LAYER,
            &[ExpertRow {
                id: 1,
                input: &good,
                experts: &too_many
            }],
            None
        )
        .is_err());
}
