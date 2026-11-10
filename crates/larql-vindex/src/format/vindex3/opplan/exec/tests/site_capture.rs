//! PREC-DIAG-1 E0a gates: sub-layer capture is a copy.
//!
//! 1. Capture on vs off is bit-identical on every logit, on both CPU
//!    backends, for several position sets (including one past the
//!    miniature's sliding window).
//! 2. The prepared realizations — the kernel selection — are identical
//!    with capture on and off.
//! 3. What is captured is what the executor computed: the unpositioned
//!    q/k rotate to the positioned q/k bit for bit; the residual chain
//!    re-adds bit for bit from the captured deltas; the head over the
//!    captured head input reproduces the logits bit for bit; and on the
//!    reference backend the captured residual equals the batch
//!    traversal's layer planes (the `--dump-layers` source).
//! 4. Bounds and completeness: over budget refuses before allocating; a
//!    requested slot never written, a wrong width, or a conflicting
//!    rewrite is a fault, never an omission.

use super::decode::fixture;
use super::golden::{G_HEAD_DIM, G_HIDDEN, G_KV_HEADS, G_LAYERS, G_Q_HEADS, G_TOKENS};
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::execute_prepared_streaming;
use crate::format::vindex3::opplan::exec::kernels::rope_rotate;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    AttentionHeadRecord, CarrierWriteRecord, InputSite, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::site_capture::{
    CaptureFault, CaptureRequest, CaptureSite, CapturedSites, LayerShape, SiteCapture,
    BYTES_PER_VALUE,
};
use crate::format::vindex3::opplan::exec::PlaneEvent;
use crate::format::vindex3::opplan::ComponentOpPlan;
use larql_models::config::PositionPolicy;

/// Position sets exercised: every position; the first and the first
/// position past the window (3); the last alone.
const POSITION_SETS: [&[usize]; 3] = [&[0, 1, 2, 3, 4], &[0, 3], &[4]];

fn prepared<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> PreparedOperands {
    PreparedOperands::load(plan, store, backend, ExecutionSlice::Full).unwrap()
}

fn request(plan: &ComponentOpPlan, positions: &[usize], budget: u64) -> CaptureRequest {
    CaptureRequest::for_plan(
        plan,
        0..plan.layers.len(),
        G_HIDDEN,
        positions.to_vec(),
        G_TOKENS.len(),
        budget,
    )
    .unwrap()
}

/// Step `tokens`, observed by `observer` when given; every position's logits.
fn logits<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    tokens: &[u32],
    mut observer: Option<&mut dyn StepObserver>,
) -> Vec<Vec<f32>> {
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(plan, ops, backend, &mut kv).unwrap();
    tokens
        .iter()
        .map(|&t| {
            let out = match observer.as_deref_mut() {
                Some(observer) => session.step_observed(t, observer).unwrap(),
                None => session.step(t).unwrap(),
            };
            out.logits.expect("the miniature carries an output head")
        })
        .collect()
}

fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .collect()
}

fn row_bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

/// Capture `positions` over the whole fixture sequence; the logits and the capture.
fn captured<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    positions: &[usize],
) -> (Vec<Vec<f32>>, CapturedSites) {
    let mut capture = SiteCapture::new(request(plan, positions, u64::MAX)).unwrap();
    let rows = logits(plan, ops, backend, &G_TOKENS, Some(&mut capture));
    (rows, capture.finish().expect("the capture is complete"))
}

/// One position's row of a plane.
fn row(sites: &CapturedSites, site: CaptureSite, layer: Option<usize>, position: usize) -> &[f32] {
    let plane = sites
        .planes
        .iter()
        .find(|p| p.site == site && p.layer == layer)
        .expect("the plane was requested");
    let index = sites
        .request
        .positions
        .iter()
        .position(|&p| p == position)
        .expect("the position was requested");
    &plane.values[index * plane.width..(index + 1) * plane.width]
}

fn each_backend(mut check: impl FnMut(&dyn Fn() -> Box<dyn Run>)) {
    check(&|| Box::new(ReferenceBackend::new()) as Box<dyn Run>);
    check(&|| Box::new(ProductionBackend::new()) as Box<dyn Run>);
}

/// Object-safe shim so one test body runs on both backends.
trait Run {
    fn name(&self) -> &'static str;
    fn prepare(&self, plan: &ComponentOpPlan, store: &OperandStore) -> PreparedOperands;
    fn logits(
        &self,
        plan: &ComponentOpPlan,
        ops: &PreparedOperands,
        observer: Option<&mut dyn StepObserver>,
    ) -> Vec<Vec<f32>>;
    fn head(&self, ops: &PreparedOperands, normed: &[f32]) -> Vec<f32>;
}

macro_rules! run_impl {
    ($ty:ty, $name:literal) => {
        impl Run for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn prepare(&self, plan: &ComponentOpPlan, store: &OperandStore) -> PreparedOperands {
                prepared(plan, store, self)
            }
            fn logits(
                &self,
                plan: &ComponentOpPlan,
                ops: &PreparedOperands,
                observer: Option<&mut dyn StepObserver>,
            ) -> Vec<Vec<f32>> {
                logits(plan, ops, self, &G_TOKENS, observer)
            }
            fn head(&self, ops: &PreparedOperands, normed: &[f32]) -> Vec<f32> {
                ops.head_over_normed(self, normed).unwrap().unwrap()
            }
        }
    };
}
run_impl!(ReferenceBackend, "reference");
run_impl!(ProductionBackend, "production");

// ── 1. bit identity ────────────────────────────────────────────────────

#[test]
fn capture_on_is_bit_identical_to_capture_off_on_both_backends() {
    let (_c, plan, store) = fixture();
    each_backend(|make| {
        let backend = make();
        let ops = backend.prepare(&plan, &store);
        let plain = backend.logits(&plan, &ops, None);
        for positions in POSITION_SETS {
            let mut capture = SiteCapture::new(request(&plan, positions, u64::MAX)).unwrap();
            let observed = backend.logits(&plan, &ops, Some(&mut capture));
            assert_eq!(
                bits(&plain),
                bits(&observed),
                "{}: capturing {positions:?} changed a logit",
                backend.name()
            );
            capture.finish().unwrap_or_else(|fault| {
                panic!("{}: {positions:?} incomplete: {fault}", backend.name())
            });
        }
    });
}

// ── 2. kernel selection ───────────────────────────────────────────────

#[test]
fn the_prepared_realizations_are_identical_with_capture_on_and_off() {
    let (_c, plan, store) = fixture();
    each_backend(|make| {
        let backend = make();
        let plain = backend.prepare(&plan, &store);
        let observed = backend.prepare(&plan, &store);
        let _ = backend.logits(&plan, &plain, None);
        let mut capture = SiteCapture::new(request(&plan, &[0, 3], u64::MAX)).unwrap();
        let _ = backend.logits(&plan, &observed, Some(&mut capture));
        assert!(!plain.realizations().is_empty());
        assert_eq!(
            plain.realizations(),
            observed.realizations(),
            "{}: capture changed a realization",
            backend.name()
        );
    });
}

// ── 3. the capture is what the executor computed ─────────────────────

#[test]
fn unpositioned_q_and_k_rotate_to_the_positioned_q_and_k_bit_for_bit() {
    let (_c, plan, store) = fixture();
    let mut rope_layers = 0;
    each_backend(|make| {
        let backend = make();
        let ops = backend.prepare(&plan, &store);
        let mut capture = SiteCapture::new(request(&plan, &[0, 1, 2, 3, 4], u64::MAX)).unwrap();
        let _ = backend.logits(&plan, &ops, Some(&mut capture));
        let sites = capture.finish().unwrap();
        for (layer, lp) in plan.layers.iter().enumerate() {
            let op = lp.attention.softmax().unwrap();
            let PositionPolicy::Rope { theta } = op.position else {
                continue;
            };
            rope_layers += 1;
            for &position in &sites.request.positions {
                for (un, pos) in [
                    (CaptureSite::QueryUnpositioned, CaptureSite::QueryPositioned),
                    (CaptureSite::KeyUnpositioned, CaptureSite::KeyPositioned),
                ] {
                    let mut rotated = row(&sites, un, Some(layer), position).to_vec();
                    for head in rotated.chunks_exact_mut(op.head_dim) {
                        rope_rotate(head, position, theta);
                    }
                    assert_eq!(
                        row_bits(&rotated),
                        row_bits(row(&sites, pos, Some(layer), position)),
                        "{}: layer {layer} position {position} {:?}",
                        backend.name(),
                        un
                    );
                }
            }
        }
    });
    assert!(rope_layers > 0, "the fixture exercised no rope layer");
}

#[test]
fn the_residual_chain_re_adds_bit_for_bit_from_the_captured_deltas() {
    let (_c, plan, store) = fixture();
    each_backend(|make| {
        let backend = make();
        let ops = backend.prepare(&plan, &store);
        let mut capture = SiteCapture::new(request(&plan, &[0, 1, 2, 3, 4], u64::MAX)).unwrap();
        let _ = backend.logits(&plan, &ops, Some(&mut capture));
        let sites = capture.finish().unwrap();
        for &position in &sites.request.positions {
            let mut carrier = row(&sites, CaptureSite::Embedding, None, position).to_vec();
            for layer in 0..G_LAYERS {
                for delta in [CaptureSite::AttentionDelta, CaptureSite::FfnDelta] {
                    let d = row(&sites, delta, Some(layer), position);
                    for (c, d) in carrier.iter_mut().zip(d) {
                        *c += d;
                    }
                }
                assert_eq!(
                    row_bits(&carrier),
                    row_bits(row(&sites, CaptureSite::Residual, Some(layer), position)),
                    "{}: layer {layer} position {position}",
                    backend.name()
                );
            }
        }
    });
}

#[test]
fn the_head_over_the_captured_head_input_reproduces_the_logits() {
    let (_c, plan, store) = fixture();
    each_backend(|make| {
        let backend = make();
        let ops = backend.prepare(&plan, &store);
        let mut capture = SiteCapture::new(request(&plan, &[0, 1, 2, 3, 4], u64::MAX)).unwrap();
        let rows = backend.logits(&plan, &ops, Some(&mut capture));
        let sites = capture.finish().unwrap();
        for &position in &sites.request.positions {
            let input = row(&sites, CaptureSite::HeadInput, None, position);
            assert_eq!(
                row_bits(&backend.head(&ops, input)),
                row_bits(&rows[position]),
                "{}: position {position}",
                backend.name()
            );
        }
    });
}

/// Asserted on the reference backend only: there the batch traversal
/// (what `--dump-layers` persists) and the decode step are pinned
/// bit-exact. On another backend the two paths are not claimed to
/// compute identically, and the chain test above is the check.
#[test]
fn on_the_reference_backend_the_residual_equals_the_batch_layer_planes() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let (_, sites) = captured(&plan, &ops, &backend, &[0, 1, 2, 3, 4]);
    let mut embedded = None;
    let mut layers = Vec::new();
    execute_prepared_streaming(&plan, &ops, &G_TOKENS, &backend, None, &mut |event| {
        match event {
            PlaneEvent::Embedded(plane) => embedded = Some(plane.clone()),
            PlaneEvent::Layer { trace, .. } => layers.push(trace.post_layer.clone()),
            _ => {}
        }
        Ok(())
    })
    .unwrap();
    let embedded = embedded.expect("the batch traversal embeds");
    let embedded = embedded.try_rows().unwrap();
    assert_eq!(layers.len(), G_LAYERS);
    for &position in &sites.request.positions {
        assert_eq!(
            row_bits(&embedded[position]),
            row_bits(row(&sites, CaptureSite::Embedding, None, position)),
            "embedding at {position}"
        );
        for (layer, plane) in layers.iter().enumerate() {
            assert_eq!(
                row_bits(&plane.try_rows().unwrap()[position]),
                row_bits(row(&sites, CaptureSite::Residual, Some(layer), position)),
                "layer {layer} position {position}"
            );
        }
    }
}

// ── 4. bounds and completeness ────────────────────────────────────────

#[test]
fn the_budget_is_enforced_before_allocating_and_matches_what_is_held() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let positions = [0, 3];
    let required = request(&plan, &positions, u64::MAX).required_bytes();
    // Hand-derived: per position, per layer, 7 hidden-wide sites plus
    // q (3 heads) and k (1 head) twice each, plus two once-per-step sites.
    let per_layer = 7 * G_HIDDEN + 2 * G_Q_HEADS * G_HEAD_DIM + 2 * G_KV_HEADS * G_HEAD_DIM;
    let per_position = G_LAYERS * per_layer + 2 * G_HIDDEN;
    assert_eq!(
        required,
        (positions.len() * per_position) as u64 * BYTES_PER_VALUE
    );

    match SiteCapture::new(request(&plan, &positions, required - 1)) {
        Err(fault @ CaptureFault::ResourceLimit { .. }) => {
            assert!(fault.is_resource_limit());
            assert_eq!(
                fault,
                CaptureFault::ResourceLimit {
                    required,
                    budget: required - 1
                }
            );
            assert!(fault.to_string().starts_with("resource_limit:"));
        }
        Err(other) => panic!("expected resource_limit, got {other}"),
        Ok(_) => panic!("an over-budget capture was built"),
    }

    let ops = prepared(&plan, &store, &backend);
    let mut capture = SiteCapture::new(request(&plan, &positions, required)).unwrap();
    let _ = logits(&plan, &ops, &backend, &G_TOKENS, Some(&mut capture));
    let sites = capture.finish().unwrap();
    let held: u64 = sites
        .planes
        .iter()
        .map(|p| p.values.len() as u64 * BYTES_PER_VALUE)
        .sum();
    assert_eq!(held, required);
    assert_eq!(sites.planes.len(), 2 + 11 * G_LAYERS);
}

#[test]
fn a_requested_position_never_reached_is_incomplete_not_omitted() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let mut capture = SiteCapture::new(request(&plan, &[0, 4], u64::MAX)).unwrap();
    let _ = logits(&plan, &ops, &backend, &G_TOKENS[..3], Some(&mut capture));
    match capture.finish() {
        Err(CaptureFault::Incomplete { count, first }) => {
            assert!(count > 0);
            assert_eq!(first, (CaptureSite::Embedding, None, 4));
        }
        other => panic!("expected incomplete, got {other:?}"),
    }
}

#[test]
fn malformed_requests_are_refused() {
    let (_c, plan, _store) = fixture();
    let refuse = |positions: Vec<usize>, layers: std::ops::Range<usize>| {
        CaptureRequest::for_plan(&plan, layers, G_HIDDEN, positions, G_TOKENS.len(), 1 << 20)
    };
    for (positions, layers) in [
        (vec![], 0..G_LAYERS),
        (vec![2, 1], 0..G_LAYERS),
        (vec![1, 1], 0..G_LAYERS),
        (vec![G_TOKENS.len()], 0..G_LAYERS),
        (vec![0], 0..0),
        (vec![0], 0..G_LAYERS + 1),
    ] {
        match refuse(positions.clone(), layers.clone()) {
            Err(fault @ CaptureFault::Request(_)) => {
                assert!(!fault.is_resource_limit());
                assert!(fault.to_string().contains("refused"));
            }
            other => panic!("{positions:?} {layers:?}: expected a refusal, got {other:?}"),
        }
    }
}

/// A capture with one layer and one position, driven by hand.
fn hand_capture() -> SiteCapture {
    SiteCapture::new(CaptureRequest {
        positions: vec![0],
        sequence_len: 1,
        hidden: G_HIDDEN,
        layers: vec![LayerShape {
            layer: 0,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: G_HEAD_DIM,
        }],
        budget_bytes: u64::MAX,
    })
    .unwrap()
}

fn head_record<'a>(
    head: usize,
    query: &'a [f32],
    keys: &'a [&'a [f32]],
    unpositioned: Option<(&'a [f32], &'a [f32])>,
) -> AttentionHeadRecord<'a> {
    AttentionHeadRecord {
        position: 0,
        head,
        kv_head: 0,
        source_start: 0,
        weights: &[1.0],
        sink: 0.0,
        values: query,
        gate: None,
        source_values: keys,
        query,
        source_keys: keys,
        query_unpositioned: unpositioned.map(|(q, _)| q),
        key_unpositioned: unpositioned.map(|(_, k)| k),
    }
}

#[test]
fn a_wrong_width_is_a_shape_fault() {
    let mut capture = hand_capture();
    capture.entering_carrier(0, &[0.0; G_HIDDEN + 1]);
    match capture.finish() {
        Err(fault @ CaptureFault::Shape { .. }) => {
            assert!(fault.to_string().contains("expected width"))
        }
        other => panic!("expected a shape fault, got {other:?}"),
    }
}

#[test]
fn a_shared_kv_head_rewritten_differently_is_a_conflict() {
    let mut capture = hand_capture();
    let q = [1.0; G_HEAD_DIM];
    let (k1, k2) = ([2.0; G_HEAD_DIM], [3.0; G_HEAD_DIM]);
    let (keys1, keys2): ([&[f32]; 1], [&[f32]; 1]) = ([&k1], [&k2]);
    capture.event(StepEvent::Embedded { position: 0 });
    capture.attention_head(0, head_record(0, &q, &keys1, Some((&q, &k1))));
    // The same KV head again, same values: GQA, not a fault.
    capture.attention_head(0, head_record(1, &q, &keys1, Some((&q, &k1))));
    capture.attention_head(0, head_record(1, &q, &keys2, Some((&q, &k2))));
    match capture.finish() {
        Err(fault @ CaptureFault::Conflict { .. }) => {
            assert!(fault.to_string().contains("written twice"))
        }
        other => panic!("expected a conflict, got {other:?}"),
    }
}

#[test]
fn a_head_record_without_pre_rotation_values_is_unsupported() {
    let mut capture = hand_capture();
    let q = [1.0; G_HEAD_DIM];
    let keys: [&[f32]; 1] = [&q];
    capture.attention_head(0, head_record(0, &q, &keys, None));
    match capture.finish() {
        Err(fault @ CaptureFault::Unsupported(_)) => {
            assert!(fault.to_string().contains("pre-rotation"))
        }
        other => panic!("expected unsupported, got {other:?}"),
    }
}

#[test]
fn a_head_record_missing_its_own_key_is_unsupported() {
    let mut capture = hand_capture();
    let q = [1.0; G_HEAD_DIM];
    capture.attention_head(0, head_record(0, &q, &[], Some((&q, &q))));
    assert!(matches!(
        capture.finish(),
        Err(CaptureFault::Unsupported(why)) if why.contains("no key")
    ));
}

#[test]
fn an_operand_before_the_step_position_is_unsupported() {
    let mut capture = hand_capture();
    capture.operand_input(0, InputSite::Attention, &[0.0; G_HIDDEN]);
    assert!(matches!(
        capture.finish(),
        Err(CaptureFault::Unsupported(why)) if why.contains("before the step's position")
    ));
}

#[test]
fn a_layer_scale_is_unsupported() {
    let mut capture = hand_capture();
    let row = [0.0; G_HIDDEN];
    capture.carrier_write(CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Ffn,
        position: 0,
        delta: &row,
        after: &row,
        layer_scale: Some(0.5),
    });
    assert!(matches!(
        capture.finish(),
        Err(CaptureFault::Unsupported(why)) if why.contains("layer scale")
    ));
}

#[test]
fn unrequested_positions_and_layers_are_not_captured_and_not_faults() {
    let mut capture = hand_capture();
    assert!(capture.wants_attention_heads());
    assert!(capture.wants_attention_heads_at(0, 0));
    assert!(!capture.wants_attention_heads_at(0, 1));
    assert!(!capture.wants_attention_heads_at(1, 0));
    capture.attention_output(0, 1, &[0.0; G_HIDDEN]);
    capture.attention_output(7, 0, &[0.0; G_HIDDEN]);
    // Still incomplete (nothing requested was written), and no fault.
    assert!(matches!(
        capture.finish(),
        Err(CaptureFault::Incomplete { .. })
    ));
}

#[test]
fn every_site_has_a_distinct_name_and_scope() {
    let names: std::collections::BTreeSet<_> = CaptureSite::ALL.iter().map(|s| s.name()).collect();
    assert_eq!(names.len(), CaptureSite::ALL.len());
    let once: Vec<_> = CaptureSite::ALL
        .iter()
        .filter(|s| s.once_per_step())
        .collect();
    assert_eq!(once, [&CaptureSite::Embedding, &CaptureSite::HeadInput]);
    let incomplete = CaptureFault::Incomplete {
        count: 3,
        first: (CaptureSite::Residual, Some(1), 4),
    };
    assert!(incomplete.to_string().contains("residual"));
}
