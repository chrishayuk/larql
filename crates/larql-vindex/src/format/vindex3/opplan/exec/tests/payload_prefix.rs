//! The GW payload adapter against the kernel's own V.
//!
//! `PayloadPrefix::values` recomputes one declared head's V at chosen
//! source positions from a prefix-only image. The authority it must agree
//! with is what the attention kernel itself read there — HEAD-OBS-1's
//! `source_values` — on a GQA plan where half the heads share the SECOND
//! KV group, so a slice of the wrong group cannot pass. The row slicer is
//! also driven directly on the representations the fixture never makes
//! resident (small matrices are widened to f32 at preparation).

use std::collections::BTreeMap;

use crate::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, DENSE_HEAD_DIM, DENSE_HIDDEN, DENSE_KV_HEADS,
    DENSE_LAYERS, DENSE_Q_HEADS,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{PlanBackend, WeightSlice};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{AttentionHeadRecord, StepEvent, StepObserver};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::payload_prefix::{row_range, PayloadPrefix};
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::quantise::SUM_BLOCK;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// A prompt over the dense fixture's vocabulary.
const TOKENS: [u32; 4] = [3, 17, 60, 0];
/// The source layer: the last one, so the prefix is every layer below it.
const SOURCE_LAYER: usize = DENSE_LAYERS - 1;
/// The kernel's V and the adapter's V come from different projection
/// calls over the same prepared weight; they may differ by summation order.
const V_TOLERANCE: f32 = 1e-5;
/// Q8 geometry for the direct row-slicer arms.
const Q8_ROWS: usize = 4;
const Q8_IN_DIM: usize = 40;
const Q8_BLOCK: usize = 8;

// The group, not the head, owns the rows: a fixture with a single KV group
// could not catch a slice of the wrong group.
const _: () = assert!(
    DENSE_KV_HEADS > 1,
    "a single KV group cannot catch a wrong-group slice"
);

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

/// Every head's source V rows at the source layer's last write.
#[derive(Default)]
struct SourceV {
    by_head: BTreeMap<usize, (usize, Vec<Vec<f32>>)>,
}

impl StepObserver for SourceV {
    fn event(&mut self, _: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        layer == SOURCE_LAYER && position == TOKENS.len() - 1
    }

    fn attention_head(&mut self, _layer: usize, record: AttentionHeadRecord<'_>) {
        let rows = record.source_values.iter().map(|v| v.to_vec()).collect();
        self.by_head
            .insert(record.head, (record.source_start, rows));
    }
}

fn kernel_source_values<B: PlanBackend>(
    plan: &ComponentOpPlan,
    full: &PreparedOperands,
    backend: &B,
) -> SourceV {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, full, backend, &mut kv).unwrap();
    let mut observer = SourceV::default();
    for &token in &TOKENS {
        session.step_observed(token, &mut observer).unwrap();
    }
    assert_eq!(observer.by_head.len(), DENSE_Q_HEADS, "one record per head");
    observer
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_values_match_the_kernel<B: PlanBackend>(fixture: &Fixture, backend: &B) {
    let full = PreparedOperands::load(&fixture.plan, &fixture.store, backend, ExecutionSlice::Full)
        .unwrap();
    let kernel = kernel_source_values(&fixture.plan, &full, backend);
    let prefix = PayloadPrefix::prepare(
        &fixture.plan,
        &fixture.store,
        backend,
        SOURCE_LAYER,
        &super::row_continuation(&fixture.plan),
    )
    .unwrap();
    let positions: Vec<usize> = (0..TOKENS.len()).collect();
    for head in 0..DENSE_Q_HEADS {
        let constructed = prefix
            .values(&TOKENS, &positions, head, &fixture.plan, &full, backend)
            .unwrap();
        let (source_start, rows) = &kernel.by_head[&head];
        for (p, vector) in constructed.iter().enumerate() {
            assert_eq!(vector.len(), DENSE_HEAD_DIM);
            let error = max_abs(vector, &rows[p - source_start]);
            assert!(
                error <= V_TOLERANCE,
                "{}: head {head} V at position {p} is {error} from the kernel's",
                backend.name()
            );
        }
    }
}

#[test]
fn every_heads_payload_v_is_the_v_its_kernel_read_on_both_cpu_backends() {
    // The group, not the head, owns the rows: with 8 heads over 2 KV
    // groups, heads 4..8 must read the SECOND group's V.
    let fixture = dense();
    assert_values_match_the_kernel(&fixture, &ReferenceBackend::new());
    assert_values_match_the_kernel(&fixture, &ProductionBackend::new());
}

#[test]
fn heads_in_different_groups_read_different_rows() {
    let fixture = dense();
    let backend = ReferenceBackend::new();
    let full = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();
    let prefix = PayloadPrefix::prepare(
        &fixture.plan,
        &fixture.store,
        &backend,
        SOURCE_LAYER,
        &super::row_continuation(&fixture.plan),
    )
    .unwrap();
    let group = DENSE_Q_HEADS / DENSE_KV_HEADS;
    let v = |head| {
        prefix
            .values(&TOKENS, &[0], head, &fixture.plan, &full, &backend)
            .unwrap()
    };
    assert_eq!(v(0), v(group - 1), "heads of one group share its V");
    assert_ne!(v(0), v(group), "the next group's V is its own");
}

#[test]
fn the_adapter_refuses_what_it_cannot_measure_before_reading_weights() {
    let fixture = dense();
    let backend = ReferenceBackend::new();
    let full = PreparedOperands::load(
        &fixture.plan,
        &fixture.store,
        &backend,
        ExecutionSlice::Full,
    )
    .unwrap();
    let prefix = PayloadPrefix::prepare(
        &fixture.plan,
        &fixture.store,
        &backend,
        SOURCE_LAYER,
        &super::row_continuation(&fixture.plan),
    )
    .unwrap();
    let values = |positions: &[usize], head| {
        prefix.values(&TOKENS, positions, head, &fixture.plan, &full, &backend)
    };
    assert!(values(&[], 0).is_err(), "no positions");
    assert!(
        values(&[TOKENS.len()], 0).is_err(),
        "a position past the prompt"
    );
    assert!(
        values(&[0], DENSE_Q_HEADS).is_err(),
        "a head past the layer"
    );
    assert!(prefix.carriers(&[], &backend).is_err(), "an empty prompt");

    // A prefix of every layer has no source layer above it.
    let whole = PayloadPrefix::prepare(
        &fixture.plan,
        &fixture.store,
        &backend,
        DENSE_LAYERS,
        &super::row_continuation(&fixture.plan),
    )
    .unwrap();
    assert!(whole
        .values(&TOKENS, &[0], 0, &fixture.plan, &full, &backend)
        .is_err());
    assert!(
        PayloadPrefix::prepare(
            &fixture.plan,
            &fixture.store,
            &backend,
            DENSE_LAYERS + 1,
            &super::row_continuation(&fixture.plan)
        )
        .is_err(),
        "a depth past the plan"
    );

    // V that is not the projection alone is refused.
    let mut shared_kv = fixture.plan.clone();
    shared_kv.layers[SOURCE_LAYER]
        .attention
        .softmax_mut()
        .unwrap()
        .v_from_k = true;
    assert!(prefix
        .values(&TOKENS, &[0], 0, &shared_kv, &full, &backend)
        .is_err());
}

#[test]
fn a_deeper_prefix_holds_more_resident_bytes() {
    let fixture = dense();
    let backend = ReferenceBackend::new();
    let shallow = PayloadPrefix::prepare(
        &fixture.plan,
        &fixture.store,
        &backend,
        0,
        &super::row_continuation(&fixture.plan),
    )
    .unwrap();
    let deep = PayloadPrefix::prepare(
        &fixture.plan,
        &fixture.store,
        &backend,
        SOURCE_LAYER,
        &super::row_continuation(&fixture.plan),
    )
    .unwrap();
    assert!(deep.resident_bytes() > shallow.resident_bytes());
}

// ── The row slicer, on representations the fixture never realises ──

fn q8_parts(sums: bool) -> (Vec<i8>, Vec<f32>, Vec<i16>) {
    let codes = (0..Q8_ROWS * Q8_IN_DIM).map(|i| i as i8).collect();
    let scales = (0..Q8_ROWS * Q8_IN_DIM.div_ceil(Q8_BLOCK))
        .map(|i| i as f32)
        .collect();
    let sums = if sums {
        (0..Q8_ROWS * Q8_IN_DIM.div_ceil(SUM_BLOCK))
            .map(|i| i as i16)
            .collect()
    } else {
        Vec::new()
    };
    (codes, scales, sums)
}

#[test]
fn a_q8_row_range_cuts_codes_scales_and_sums_on_row_boundaries() {
    let (codes, scales, sums) = q8_parts(true);
    let weight = WeightSlice::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &sums,
        block: Q8_BLOCK,
    };
    let rows = 1..3;
    let WeightSlice::Q8 {
        codes: cut_codes,
        scales: cut_scales,
        sums: cut_sums,
        block,
    } = row_range(weight, rows.clone(), Q8_IN_DIM).unwrap()
    else {
        panic!("a Q8 slice stays Q8");
    };
    let scale_row = Q8_IN_DIM.div_ceil(Q8_BLOCK);
    let sum_row = Q8_IN_DIM.div_ceil(SUM_BLOCK);
    assert_eq!(block, Q8_BLOCK);
    assert_eq!(
        cut_codes,
        &codes[rows.start * Q8_IN_DIM..rows.end * Q8_IN_DIM]
    );
    assert_eq!(
        cut_scales,
        &scales[rows.start * scale_row..rows.end * scale_row]
    );
    assert_eq!(cut_sums, &sums[rows.start * sum_row..rows.end * sum_row]);
}

#[test]
fn a_q8_row_range_without_sums_stays_without_sums() {
    let (codes, scales, sums) = q8_parts(false);
    let weight = WeightSlice::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &sums,
        block: Q8_BLOCK,
    };
    let WeightSlice::Q8 { sums: cut, .. } = row_range(weight, 0..1, Q8_IN_DIM).unwrap() else {
        panic!("a Q8 slice stays Q8");
    };
    assert!(cut.is_empty());
}

#[test]
fn a_bf16_row_range_is_a_view_of_the_same_rows() {
    let bits: Vec<u16> = (0..Q8_ROWS * Q8_IN_DIM).map(|i| i as u16).collect();
    let WeightSlice::Bf16(cut) = row_range(WeightSlice::Bf16(&bits), 2..4, Q8_IN_DIM).unwrap()
    else {
        panic!("a bf16 slice stays bf16");
    };
    assert_eq!(cut, &bits[2 * Q8_IN_DIM..4 * Q8_IN_DIM]);
}

#[test]
fn an_untested_representation_is_refused_not_guessed() {
    let bytes = vec![0u8; DENSE_HIDDEN * 2];
    assert!(row_range(WeightSlice::F16(&bytes), 0..1, DENSE_HIDDEN).is_err());
}
