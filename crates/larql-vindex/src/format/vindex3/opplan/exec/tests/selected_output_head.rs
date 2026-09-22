//! A gathered output head is the full head, restricted — nothing else.
//!
//! GW lenses read a carrier against a few declared vocabulary rows instead
//! of the whole head. The authority is the full readout: at every selected
//! token, the selected readout must equal it bit for bit, and each gathered
//! row must be that token's row of the resident head.

use crate::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, DENSE_HIDDEN, DENSE_VOCAB,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{PlanBackend, WeightSlice};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{
    ExecutionSlice, PreparedOperands, SelectedOutputHead,
};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::quantise::SUM_BLOCK;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// Declared rows, deliberately out of vocabulary order.
const SELECTED: [u32; 3] = [17, 3, 60];
/// A synthetic head for the representations the fixture never makes
/// resident: small matrices are widened to f32 at preparation.
const SYNTH_VOCAB: usize = 5;
const SYNTH_HIDDEN: usize = 12;
const SYNTH_BLOCK: usize = 4;
const SYNTH_ROWS: [u32; 2] = [3, 1];

struct Fixture {
    _src: tempfile::TempDir,
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn dense() -> Fixture {
    let src = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(dense_f32_model, src.path(), container.path(), "dense");
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Fixture {
        _src: src,
        _container: container,
        plan: outcome.plan.unwrap(),
        store,
    }
}

/// A carrier with no symmetry the head could hide behind.
fn carrier() -> Vec<f32> {
    (0..DENSE_HIDDEN)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) / 3.0)
        .collect()
}

fn assert_selected_is_the_full_head_restricted<B: PlanBackend>(
    fixture: &Fixture,
    backend: &B,
    representation: &str,
) {
    let ops = PreparedOperands::load(&fixture.plan, &fixture.store, backend, ExecutionSlice::Full)
        .unwrap();
    let head = ops.select_output_head(&SELECTED).unwrap();
    assert_eq!(head.token_ids(), SELECTED);
    assert_eq!(head.representation(), representation);

    let x = carrier();
    let full = ops.readout_carrier(backend, &x).unwrap();
    let selected = ops.readout_carrier_selected(backend, &x, &head).unwrap();
    assert_eq!(selected.len(), SELECTED.len());
    for (index, &token) in SELECTED.iter().enumerate() {
        assert_eq!(
            selected[index].to_bits(),
            full[token as usize].to_bits(),
            "{}: token {token}",
            backend.name()
        );
    }

    // Each gathered row, dotted with the normalised carrier, is that
    // token's logit before any multiplier or softcap the plan declares.
    let normalized = ops.normalize_carrier_for_readout(backend, &x).unwrap();
    for (index, &token) in SELECTED.iter().enumerate() {
        let row = head.row_f32(index).unwrap();
        assert_eq!(row.len(), DENSE_HIDDEN);
        let logit: f32 = row.iter().zip(&normalized).map(|(w, v)| w * v).sum();
        let error = (logit - full[token as usize]).abs();
        assert!(error < 1e-4, "row {token} is not the head's row ({error})");
    }
    assert!(
        head.row_f32(SELECTED.len()).is_err(),
        "a row past the selection"
    );
}

#[test]
fn an_f32_head_restricts_exactly_on_both_cpu_backends() {
    let fixture = dense();
    assert_selected_is_the_full_head_restricted(&fixture, &ReferenceBackend::new(), "f32");
    assert_selected_is_the_full_head_restricted(&fixture, &ProductionBackend::new(), "f32");
}

#[test]
fn a_selection_is_refused_when_it_cannot_name_distinct_rows_of_this_head() {
    let fixture = dense();
    let backend = ProductionBackend::new();
    let ops = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();
    assert!(ops.select_output_head(&[]).is_err(), "no rows");
    assert!(
        ops.select_output_head(&[DENSE_VOCAB as u32]).is_err(),
        "a row past the vocabulary"
    );
    assert!(ops.select_output_head(&[3, 3]).is_err(), "a duplicated row");

    // A layer-range image carries no head to select from or read with.
    let shard = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &backend,
        ExecutionSlice::LayerRange { start: 0, end: 1 },
    )
    .unwrap();
    assert!(shard.select_output_head(&SELECTED).is_err());
    assert!(shard
        .normalize_carrier_for_readout(&backend, &carrier())
        .is_err());
    let head = ops.select_output_head(&SELECTED).unwrap();
    assert!(ops
        .readout_carrier_selected(&backend, &carrier()[1..], &head)
        .is_err());
}

// ── Representations the fixture never makes resident ──

fn gather(head: WeightSlice<'_>) -> SelectedOutputHead {
    SelectedOutputHead::gather(head, SYNTH_VOCAB, SYNTH_HIDDEN, &SYNTH_ROWS, None, None).unwrap()
}

#[test]
fn a_bf16_head_is_gathered_and_widened_row_for_row() {
    let bits: Vec<u16> = (0..SYNTH_VOCAB * SYNTH_HIDDEN)
        .map(|i| (i as f32 * 0.25 - 3.0).to_bits().wrapping_shr(16) as u16)
        .collect();
    let head = gather(WeightSlice::Bf16(&bits));
    assert_eq!(head.representation(), "bf16");
    assert_eq!(head.token_ids(), SYNTH_ROWS);
    for (index, &token) in SYNTH_ROWS.iter().enumerate() {
        let start = token as usize * SYNTH_HIDDEN;
        let expected: Vec<f32> = bits[start..start + SYNTH_HIDDEN]
            .iter()
            .map(|&b| f32::from_bits(u32::from(b) << 16))
            .collect();
        assert_eq!(head.row_f32(index).unwrap(), expected);
    }
}

fn q8_head(with_sums: bool) -> (Vec<i8>, Vec<f32>, Vec<i16>) {
    let codes = (0..SYNTH_VOCAB * SYNTH_HIDDEN)
        .map(|i| (i % 23) as i8 - 11)
        .collect();
    let scales = (0..SYNTH_VOCAB * SYNTH_HIDDEN.div_ceil(SYNTH_BLOCK))
        .map(|i| 0.5 + i as f32)
        .collect();
    let sums = if with_sums {
        (0..SYNTH_VOCAB * SYNTH_HIDDEN.div_ceil(SUM_BLOCK))
            .map(|i| i as i16)
            .collect()
    } else {
        Vec::new()
    };
    (codes, scales, sums)
}

#[test]
fn a_q8_head_dequantises_each_gathered_row_with_its_own_block_scales() {
    for with_sums in [true, false] {
        let (codes, scales, sums) = q8_head(with_sums);
        let head = gather(WeightSlice::Q8 {
            codes: &codes,
            scales: &scales,
            sums: &sums,
            block: SYNTH_BLOCK,
        });
        assert_eq!(head.representation(), "q8");
        let per_row = SYNTH_HIDDEN.div_ceil(SYNTH_BLOCK);
        for (index, &token) in SYNTH_ROWS.iter().enumerate() {
            let row = token as usize;
            let expected: Vec<f32> = (0..SYNTH_HIDDEN)
                .map(|c| {
                    f32::from(codes[row * SYNTH_HIDDEN + c])
                        * scales[row * per_row + c / SYNTH_BLOCK]
                })
                .collect();
            assert_eq!(head.row_f32(index).unwrap(), expected, "token {token}");
        }
    }
}

#[test]
fn a_head_in_an_untested_representation_is_refused() {
    let bytes = vec![0u8; SYNTH_VOCAB * SYNTH_HIDDEN * 2];
    assert!(SelectedOutputHead::gather(
        WeightSlice::F16(&bytes),
        SYNTH_VOCAB,
        SYNTH_HIDDEN,
        &SYNTH_ROWS,
        None,
        None
    )
    .is_err());
}
