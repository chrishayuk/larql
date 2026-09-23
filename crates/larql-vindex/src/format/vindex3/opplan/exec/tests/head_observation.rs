//! V3-HEAD-OBS-1 acceptance on the golden plan
//! (`docs/v3-head-obs-1-per-head-observation.md`): HP1 parity, HP2 record
//! identity, HP3 the head-sum law through the post-norm, HP4 the source
//! split, A6 per-layer coverage on a mixed-family plan, A8 refusal before
//! the first token, and the event order.

use std::collections::BTreeMap;

use super::decode::fixture;
use super::device::LoopDevice;
use super::golden::{G_HEAD_DIM, G_KV_HEADS, G_LAYERS, G_Q_HEADS, G_TOKENS, G_WINDOW};
use super::hybrid_traversal::hybrid;
use crate::format::vindex3::opplan::exec::backend::{PlanBackend, WeightFormat};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    AttentionHeadRecord, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::observe_heads::{HeadReader, HeadStats};
use crate::format::vindex3::opplan::exec::observe_stats::FixedBasis;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::ComponentOpPlan;

type Key = (usize, SublayerSite, usize);

/// An owned copy of one head record.
#[derive(Debug, Clone, PartialEq)]
struct Head {
    layer: usize,
    position: usize,
    head: usize,
    kv_head: usize,
    source_start: usize,
    weights: Vec<f32>,
    sink: f32,
    values: Vec<f32>,
    gate: Option<Vec<f32>>,
    source_values: Vec<Vec<f32>>,
    query: Vec<f32>,
    source_keys: Vec<Vec<f32>>,
}

/// Everything a run says: writes, events with positions, logits, and —
/// when asked — every head record.
#[derive(Default)]
struct Witness {
    heads: bool,
    /// Narrow an armed capture to one `(layer, position)`.
    only: Option<(usize, usize)>,
    position: usize,
    delta: BTreeMap<Key, Vec<f32>>,
    after: BTreeMap<Key, Vec<f32>>,
    layer_scale: BTreeMap<Key, Option<f32>>,
    events: Vec<(usize, StepEvent)>,
    records: Vec<Head>,
}

impl StepObserver for Witness {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = position;
        }
        self.events.push((self.position, event));
    }

    fn wants_attention_heads(&self) -> bool {
        self.heads
    }

    fn wants_attention_heads_at(&self, layer: usize, position: usize) -> bool {
        match self.only {
            Some(address) => self.heads && address == (layer, position),
            None => self.heads,
        }
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        self.records.push(Head {
            layer,
            position: record.position,
            head: record.head,
            kv_head: record.kv_head,
            source_start: record.source_start,
            weights: record.weights.to_vec(),
            sink: record.sink,
            values: record.values.to_vec(),
            gate: record.gate.map(<[f32]>::to_vec),
            source_values: record.source_values.iter().map(|v| v.to_vec()).collect(),
            query: record.query.to_vec(),
            source_keys: record.source_keys.iter().map(|k| k.to_vec()).collect(),
        });
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        let key = (record.layer, record.site, record.position);
        self.delta.insert(key, record.delta.to_vec());
        self.after.insert(key, record.after.to_vec());
        self.layer_scale.insert(key, record.layer_scale);
    }
}

fn run<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    heads: bool,
) -> (Vec<Vec<f32>>, Witness) {
    run_narrowed(plan, store, backend, heads, None)
}

fn run_narrowed<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    heads: bool,
    only: Option<(usize, usize)>,
) -> (Vec<Vec<f32>>, Witness) {
    let mut session = DecodeSession::new(plan, store, backend).unwrap();
    let mut witness = Witness {
        heads,
        only,
        ..Witness::default()
    };
    let logits = G_TOKENS
        .iter()
        .map(|&t| {
            session
                .step_observed(t, &mut witness)
                .unwrap()
                .logits
                .unwrap()
        })
        .collect();
    (logits, witness)
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn structural(events: &[(usize, StepEvent)]) -> Vec<(usize, StepEvent)> {
    events
        .iter()
        .filter(|(_, e)| {
            !matches!(
                e,
                StepEvent::HeadsObserved { .. } | StepEvent::HeadsUncovered { .. }
            )
        })
        .cloned()
        .collect()
}

macro_rules! on_both_backends {
    (|$backend:ident| $body:block) => {{
        {
            let $backend: &ReferenceBackend = &ReferenceBackend::new();
            $body
        }
        {
            let $backend: &ProductionBackend = &ProductionBackend::new();
            $body
        }
    }};
}

// ── HP1 ─────────────────────────────────────────────────────────────

#[test]
fn hp1_heads_armed_leave_logits_writes_and_events_bit_identical() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let (plain_logits, plain) = run(&plan, &store, backend, false);
        let (head_logits, tapped) = run(&plan, &store, backend, true);
        assert_eq!(plain_logits.len(), head_logits.len());
        for (p, (a, b)) in plain_logits.iter().zip(&head_logits).enumerate() {
            assert!(
                bits_equal(a, b),
                "{}: logits differ at position {p}",
                backend.name()
            );
        }
        assert_eq!(plain.delta.len(), tapped.delta.len());
        for (key, delta) in &plain.delta {
            assert!(
                bits_equal(delta, &tapped.delta[key]),
                "delta moved at {key:?}"
            );
            assert!(
                bits_equal(&plain.after[key], &tapped.after[key]),
                "after moved at {key:?}"
            );
            assert_eq!(plain.layer_scale[key], tapped.layer_scale[key]);
        }
        assert_eq!(structural(&plain.events), structural(&tapped.events));
        assert!(plain.records.is_empty(), "nothing asked, nothing fired");
        assert_eq!(
            tapped.records.len(),
            G_TOKENS.len() * G_LAYERS * G_Q_HEADS,
            "one record per head per softmax layer per position"
        );
    });
}

/// A narrowed capture is a capture-cost filter, never an arithmetic
/// change: it fires only at the selected address, leaves the logits
/// bit-identical, and hands over exactly the records a full capture
/// fires there — including the conditioned query and key rows the GW
/// head replay reconstructs from.
#[test]
fn a_narrowed_capture_fires_only_where_asked_and_changes_nothing() {
    let (_c, plan, store) = fixture();
    let target = (G_LAYERS - 1, G_TOKENS.len() - 1);
    on_both_backends!(|backend| {
        let (plain_logits, _) = run(&plan, &store, backend, false);
        let (_, full) = run(&plan, &store, backend, true);
        let (narrow_logits, narrow) = run_narrowed(&plan, &store, backend, true, Some(target));
        for (p, (a, b)) in plain_logits.iter().zip(&narrow_logits).enumerate() {
            assert!(
                bits_equal(a, b),
                "{}: narrowing moved logits at position {p}",
                backend.name()
            );
        }
        assert_eq!(narrow.records.len(), G_Q_HEADS, "one record per head, once");
        let expected: Vec<&Head> = full
            .records
            .iter()
            .filter(|h| (h.layer, h.position) == target)
            .collect();
        assert_eq!(narrow.records.iter().collect::<Vec<_>>(), expected);
        let observed: Vec<_> = narrow
            .events
            .iter()
            .filter(|(_, e)| matches!(e, StepEvent::HeadsObserved { .. }))
            .collect();
        assert_eq!(
            observed,
            vec![&(
                target.1,
                StepEvent::HeadsObserved {
                    layer: target.0,
                    heads: G_Q_HEADS
                }
            )]
        );
        for head in &full.records {
            assert_eq!(head.query.len(), G_HEAD_DIM);
            assert_eq!(head.source_keys.len(), head.weights.len());
            assert!(head.source_keys.iter().all(|k| k.len() == G_HEAD_DIM));
        }
    });
}

// ── HP2 ─────────────────────────────────────────────────────────────

#[test]
fn hp2_every_record_says_what_the_kernel_computed() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let (_, tapped) = run(&plan, &store, backend, true);
        let group = G_Q_HEADS / G_KV_HEADS;
        let mut seen = std::collections::BTreeSet::new();
        let mut truncated = false;
        for r in &tapped.records {
            assert!(
                seen.insert((r.layer, r.position, r.head)),
                "duplicate record"
            );
            assert_eq!(r.kv_head, r.head / group);
            assert_eq!(r.values.len(), G_HEAD_DIM);
            assert!(r.values.iter().all(|v| v.is_finite()));
            // The golden plan declares an output gate: the record carries
            // its activated slice, one value per element of the head.
            let gate = r
                .gate
                .as_ref()
                .expect("the golden plan declares an output gate");
            assert_eq!(gate.len(), G_HEAD_DIM);
            assert!(
                gate.iter().all(|g| *g > 0.0 && *g < 1.0),
                "a sigmoid gate: {gate:?}"
            );
            assert_eq!(r.weights.len(), r.position + 1 - r.source_start);
            assert_eq!(r.source_values.len(), r.weights.len());
            assert!(r.weights.iter().all(|w| (0.0..=1.0).contains(w)));
            let mass: f32 = r.weights.iter().sum::<f32>() + r.sink;
            assert!((mass - 1.0).abs() < 1e-6, "mass {mass} at {r:?}");
            assert_eq!(r.sink, 0.0, "no sinks on the golden plan");
            // The window: layer 0 slides with G_WINDOW, layer 1 is full.
            if r.layer == 0 && r.position + 1 > G_WINDOW {
                assert_eq!(r.source_start, r.position + 1 - G_WINDOW);
                assert_eq!(r.weights.len(), G_WINDOW);
                truncated = true;
            } else {
                assert_eq!(r.source_start, 0);
            }
            // A3: the mixed value is the weighted sum of the source rows.
            let rebuilt: Vec<f32> = (0..G_HEAD_DIM)
                .map(|i| {
                    r.weights
                        .iter()
                        .zip(&r.source_values)
                        .map(|(w, v)| w * v[i])
                        .sum::<f32>()
                })
                .collect();
            for (a, b) in rebuilt.iter().zip(&r.values) {
                assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-6), "A3: {a} vs {b}");
            }
        }
        assert!(truncated, "the fixture crosses the sliding window");
    });
}

// ── HP3 / HP4 ───────────────────────────────────────────────────────

fn prepared<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> PreparedOperands {
    PreparedOperands::load(plan, store, backend, ExecutionSlice::Full).unwrap()
}

#[test]
fn hp3_the_head_sum_law_holds_through_the_post_norm_on_the_golden_plan() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let ops = prepared(&plan, &store, backend);
        let basis = FixedBasis::seeded(ops.hidden(), 3, 42).unwrap();
        let mut stats = HeadStats::new(&ops, &plan, backend, Some(basis), 2).retaining_children();
        let (_, plain) = run(&plan, &store, backend, false);
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, backend, &mut kv).unwrap();
        for &t in G_TOKENS.iter() {
            session.step_observed(t, &mut stats).unwrap();
        }
        assert!(stats.failure.is_none(), "{:?}", stats.failure);
        assert_eq!(
            stats.writes.len(),
            G_TOKENS.len() * G_LAYERS,
            "one decomposition per attention write"
        );
        assert_eq!(stats.records, G_TOKENS.len() * G_LAYERS * G_Q_HEADS);
        let basis = stats.basis().unwrap().rows().to_vec();
        for write in &stats.writes {
            assert!(
                write.residual <= 1e-5,
                "{}: head-sum residual {} at layer {} position {}",
                backend.name(),
                write.residual,
                write.layer,
                write.position
            );
            assert_eq!(write.rows.len(), G_Q_HEADS);
            let delta = &plain.delta[&(write.layer, SublayerSite::Attention, write.position)];
            // The reader-projected form: Σ_h ⟨r, c′_h⟩ = ⟨r, delta⟩. The bound is
            // relative to the L1 scale of the f32 terms being summed, never to the
            // cancelled sum: each stored ⟨r, c′_h⟩ carries f32 rounding proportional
            // to its own magnitude, so the identity's error scales with Σ_h |⟨r, c′_h⟩|
            // (the largest term alone is only a factor ≤ H tighter, with no analytic
            // reason). At the golden plan's first write the heads cancel 44x
            // (Σ_h |·| = 0.948 against a sum of 0.0217); bounded on |⟨r, delta⟩| this
            // read 1.07e-5 on Windows against 3.3e-6 / 7.2e-6 (production / reference)
            // on macOS — reduction order, not a defect. Against the term scale the
            // worst observed margin is 2.4e-7, and a wrong head moves the sum by ~1e-1.
            for (d, row) in basis.iter().enumerate() {
                let lhs: f64 = write.rows.iter().map(|r| f64::from(r.projection[d])).sum();
                let term_scale: f64 = write
                    .rows
                    .iter()
                    .map(|r| f64::from(r.projection[d]).abs())
                    .sum();
                let rhs: f64 = row
                    .iter()
                    .zip(delta)
                    .map(|(r, x)| f64::from(*r) * f64::from(*x))
                    .sum();
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * term_scale.max(1e-3),
                    "projection {d}: {lhs} vs {rhs} (term scale {term_scale})"
                );
            }
            for row in &write.rows {
                assert_eq!(row.sources.len(), 2.min(write.position + 1));
                assert!(
                    row.sources.windows(2).all(|w| w[0].1 >= w[1].1),
                    "sources descend"
                );
                assert!(row.norm.is_finite());
            }
            let children = write.children.as_ref().unwrap();
            assert_eq!(children.len(), G_Q_HEADS);
        }
    });
}

#[test]
fn hp4_the_source_split_sums_to_the_head_through_the_projection() {
    let (_c, plan, store) = fixture();
    on_both_backends!(|backend| {
        let ops = prepared(&plan, &store, backend);
        let (_, tapped) = run(&plan, &store, backend, true);
        for r in &tapped.records {
            let whole = ops
                .head_projection(backend, r.layer, r.head, G_HEAD_DIM, G_Q_HEADS, &r.values)
                .unwrap();
            let mut summed = vec![0.0f64; whole.len()];
            for (w, v) in r.weights.iter().zip(&r.source_values) {
                let scaled: Vec<f32> = v.iter().map(|x| w * x).collect();
                let part = ops
                    .head_projection(backend, r.layer, r.head, G_HEAD_DIM, G_Q_HEADS, &scaled)
                    .unwrap();
                for (acc, p) in summed.iter_mut().zip(&part) {
                    *acc += f64::from(*p);
                }
            }
            let err: f64 = summed
                .iter()
                .zip(&whole)
                .map(|(s, w)| (s - f64::from(*w)).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale: f64 = whole
                .iter()
                .map(|w| f64::from(*w).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                err <= 1e-5 * scale.max(1e-6),
                "HP4 at {:?}: {err} vs {scale}",
                (r.layer, r.position, r.head)
            );
        }
    });
}

#[test]
fn head_projection_refuses_what_it_cannot_represent() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let x = vec![0.5f32; G_HEAD_DIM];
    let err = ops
        .head_projection(&backend, G_LAYERS, 0, G_HEAD_DIM, G_Q_HEADS, &x)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("outside this image's executed layers"),
        "{err}"
    );
    let err = ops
        .head_projection(&backend, 0, G_Q_HEADS, G_HEAD_DIM, G_Q_HEADS, &x)
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside this layer's"), "{err}");
    let err = ops
        .head_projection(&backend, 0, 0, G_HEAD_DIM, G_Q_HEADS, &x[..G_HEAD_DIM - 1])
        .unwrap_err()
        .to_string();
    assert!(err.contains("-wide head input"), "{err}");
    assert!(ops.attention_has_heads(0).unwrap());
    assert!(
        ops.attention_output_bias(0).unwrap().is_none(),
        "the golden plan has no O bias"
    );
}

// ── Events, coverage, refusal ───────────────────────────────────────

#[test]
fn heads_observed_precedes_the_attention_write_once_per_layer_and_only_when_asked() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let (_, plain) = run(&plan, &store, &backend, false);
    assert!(!plain.events.iter().any(|(_, e)| matches!(
        e,
        StepEvent::HeadsObserved { .. } | StepEvent::HeadsUncovered { .. }
    )));
    let (_, tapped) = run(&plan, &store, &backend, true);
    let observed: Vec<&(usize, StepEvent)> = tapped
        .events
        .iter()
        .filter(|(_, e)| matches!(e, StepEvent::HeadsObserved { .. }))
        .collect();
    assert_eq!(observed.len(), G_TOKENS.len() * G_LAYERS);
    assert!(!tapped
        .events
        .iter()
        .any(|(_, e)| matches!(e, StepEvent::HeadsUncovered { .. })));
    for (i, (position, event)) in tapped.events.iter().enumerate() {
        if let StepEvent::HeadsObserved { layer, heads } = event {
            assert_eq!(*heads, G_Q_HEADS);
            let next_write = tapped.events[i + 1..]
                .iter()
                .find(|(_, e)| matches!(e, StepEvent::CarrierWrite { .. }))
                .unwrap();
            assert_eq!(
                next_write,
                &(
                    *position,
                    StepEvent::CarrierWrite {
                        layer: *layer,
                        site: SublayerSite::Attention,
                        carrier: crate::format::vindex3::opplan::exec::observe::CarrierForm::Single,
                    }
                ),
                "the heads event precedes its own attention write"
            );
        }
    }
    // Every record for a layer precedes that layer's HeadsObserved event
    // is a property of the kernel order; here: records exist for every
    // (layer, position) the event names.
    for (position, event) in &tapped.events {
        if let StepEvent::HeadsObserved { layer, .. } = event {
            assert_eq!(
                tapped
                    .records
                    .iter()
                    .filter(|r| r.layer == *layer && r.position == *position)
                    .count(),
                G_Q_HEADS
            );
        }
    }
}

#[test]
fn a6_a_mixed_family_plan_names_its_uncovered_layers_and_is_not_refused() {
    let (_c, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let softmax: Vec<usize> = plan
        .layers
        .iter()
        .enumerate()
        .filter(|(_, l)| l.attention.softmax().is_some())
        .map(|(i, _)| i)
        .collect();
    let other: Vec<usize> = (0..plan.layers.len())
        .filter(|i| !softmax.contains(i))
        .collect();
    assert!(
        !softmax.is_empty() && !other.is_empty(),
        "the fixture mixes families"
    );
    let (_, tapped) = run(&plan, &store, &backend, true);
    for &layer in &softmax {
        assert_eq!(
            tapped
                .events
                .iter()
                .filter(|(_, e)| *e
                    == StepEvent::HeadsObserved {
                        layer,
                        heads: plan.layers[layer].attention.softmax().unwrap().num_q_heads
                    })
                .count(),
            G_TOKENS.len()
        );
    }
    for &layer in &other {
        assert_eq!(
            tapped
                .events
                .iter()
                .filter(|(_, e)| *e == StepEvent::HeadsUncovered { layer })
                .count(),
            G_TOKENS.len()
        );
        assert!(!tapped.records.iter().any(|r| r.layer == layer));
    }
    // And the head-sum law still holds on the covered layers.
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 3);
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&plan, &ops, &backend, &mut kv).unwrap();
    for &t in G_TOKENS.iter() {
        session.step_observed(t, &mut stats).unwrap();
    }
    assert!(stats.failure.is_none(), "{:?}", stats.failure);
    assert_eq!(stats.writes.len(), G_TOKENS.len() * softmax.len());
    assert!(
        stats.writes.iter().all(|w| w.residual <= 1e-5),
        "{:?}",
        stats.writes.iter().map(|w| w.residual).collect::<Vec<_>>()
    );
}

#[test]
fn a8_a_backend_that_does_not_serve_heads_refuses_before_the_first_token() {
    let (_c, plan, store) = fixture();
    let device = DevicePlanBackend::new(LoopDevice, "loop-device-heads", WeightFormat::F32);
    assert!(!device.serves_attention_heads());
    let mut session = DecodeSession::new(&plan, &store, &device).unwrap();
    let mut witness = Witness {
        heads: true,
        ..Witness::default()
    };
    let err = match session.step_observed(G_TOKENS[0], &mut witness) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a backend that does not serve heads must refuse"),
    };
    assert!(err.contains("not served by the"), "{err}");
    assert_eq!(session.position(), 0, "refused before any token executed");
    assert!(witness.events.is_empty());
    // Without heads the same backend runs.
    let mut plain = Witness::default();
    session.step_observed(G_TOKENS[0], &mut plain).unwrap();
    assert_eq!(session.position(), 1);
}

/// The reader trait is what a recorder drives; drive it by hand.
#[test]
fn the_reader_decomposes_only_the_write_whose_heads_it_saw() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 1);
    let delta = vec![0.0f32; ops.hidden()];
    let ffn_write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Ffn,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(
        stats.finish_write(&ffn_write).is_none(),
        "an FFN write has no heads"
    );
    let attention_write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(
        stats.finish_write(&attention_write).is_none(),
        "no heads seen yet"
    );
    assert_eq!(HeadReader::records(&stats), 0);
    assert!(HeadReader::failure(&stats).is_none());
}

// ── The reader's refusals and degenerate branches, driven directly ──

/// A synthetic head record over borrowed buffers, for driving the reader
/// without the executor: what the kernel would hand it, spelled by hand.
fn synthetic_record<'a>(
    head: usize,
    position: usize,
    weights: &'a [f32],
    values: &'a [f32],
    source_values: &'a [&'a [f32]],
) -> AttentionHeadRecord<'a> {
    AttentionHeadRecord {
        position,
        head,
        kv_head: head / (G_Q_HEADS / G_KV_HEADS),
        source_start: 0,
        weights,
        sink: 0.0,
        values,
        gate: None,
        source_values,
        // The reader never reads keys or the query; aligned stand-ins
        // keep the record's own shape contract.
        query: values,
        source_keys: source_values,
    }
}

#[test]
fn the_reader_refuses_a_write_whose_head_count_is_not_the_layers_and_stays_failed() {
    let (_c, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 1);
    let weights = [1.0f32];
    let values = [0.5f32; G_HEAD_DIM];
    let row: &[f32] = &values;
    let sources = [row];
    // Two records for a three-head layer.
    for head in 0..2 {
        HeadReader::attention_head(
            &mut stats,
            0,
            synthetic_record(head, 0, &weights, &values, &sources),
        );
    }
    assert_eq!(HeadReader::records(&stats), 2);
    let delta = vec![0.25f32; ops.hidden()];
    let write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(stats.finish_write(&write).is_none());
    let failure = HeadReader::failure(&stats)
        .expect("the count mismatch is a failure")
        .to_string();
    assert!(failure.contains("2 head records for 3 heads"), "{failure}");
    // Failed stays failed: later records and writes are refused, the
    // failure is not overwritten, and the pending heads were cleared.
    for head in 0..G_Q_HEADS {
        HeadReader::attention_head(
            &mut stats,
            1,
            synthetic_record(head, 0, &weights, &values, &sources),
        );
    }
    let write_1 = CarrierWriteRecord {
        layer: 1,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(stats.finish_write(&write_1).is_none());
    assert!(HeadReader::failure(&stats)
        .unwrap()
        .to_string()
        .contains("2 head records for 3 heads"));
    assert_eq!(
        HeadReader::records(&stats),
        2 + G_Q_HEADS,
        "records are still counted"
    );
}

#[test]
fn the_reader_refuses_a_layer_without_softmax_heads() {
    let (_c, plan, store) = hybrid();
    let backend = ReferenceBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let other = (0..plan.layers.len())
        .find(|&i| plan.layers[i].attention.softmax().is_none())
        .expect("the hybrid fixture has a non-softmax layer");
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 1);
    let weights = [1.0f32];
    let values = [0.5f32; G_HEAD_DIM];
    let row: &[f32] = &values;
    let sources = [row];
    HeadReader::attention_head(
        &mut stats,
        other,
        synthetic_record(0, 0, &weights, &values, &sources),
    );
    let delta = vec![0.25f32; ops.hidden()];
    let write = CarrierWriteRecord {
        layer: other,
        site: SublayerSite::Attention,
        position: 0,
        delta: &delta,
        after: &delta,
        layer_scale: None,
    };
    assert!(stats.finish_write(&write).is_none());
    let failure = HeadReader::failure(&stats).unwrap().to_string();
    assert!(failure.contains("has no softmax attention"), "{failure}");
}

#[test]
fn a_zero_delta_yields_an_absolute_residual_and_the_projection_is_empty_without_a_basis() {
    let (_c, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let ops = prepared(&plan, &store, &backend);
    let mut stats = HeadStats::new(&ops, &plan, &backend, None, 2).retaining_children();
    let weights = [0.25f32, 0.75];
    let values = [0.5f32; G_HEAD_DIM];
    let row: &[f32] = &values;
    let sources = [row, row];
    for head in 0..G_Q_HEADS {
        HeadReader::attention_head(
            &mut stats,
            0,
            synthetic_record(head, 1, &weights, &values, &sources),
        );
    }
    let zero = vec![0.0f32; ops.hidden()];
    let write = CarrierWriteRecord {
        layer: 0,
        site: SublayerSite::Attention,
        position: 1,
        delta: &zero,
        after: &zero,
        layer_scale: None,
    };
    let decomposed = stats
        .finish_write(&write)
        .expect("a full set of heads decomposes");
    assert!(HeadReader::failure(&stats).is_none());
    // With a zero delta the residual is the absolute head-sum norm, not a ratio.
    let children = decomposed.children.as_ref().unwrap();
    let hidden = ops.hidden();
    let mut sum = vec![0.0f64; hidden];
    for child in children {
        for (acc, c) in sum.iter_mut().zip(child) {
            *acc += f64::from(*c);
        }
    }
    let expected: f64 = sum.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((decomposed.residual - expected).abs() <= 1e-9 * expected.max(1.0));
    assert!(
        decomposed.residual > 0.0,
        "the heads wrote something the zero delta did not"
    );
    for row in &decomposed.rows {
        assert!(row.projection.is_empty(), "no basis, no projection");
        assert_eq!(row.sources.len(), 2);
        assert_eq!(row.sources[0], (1, 0.75), "sources descend by weight");
        assert_eq!(row.sources[1], (0, 0.25));
    }
    // The pending heads were consumed: the same write again decomposes nothing.
    assert!(stats.finish_write(&write).is_none());
    assert!(stats.basis().is_none());
}
