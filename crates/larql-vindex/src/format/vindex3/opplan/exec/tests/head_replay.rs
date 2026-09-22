//! The head replay is the actual prepared W_O + norm + residual path.

use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    head_replay::{replay_attention_heads, replay_attention_mixture, replay_softmax_source_head},
    observe::{AttentionHeadRecord, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite},
    operands::OperandStore,
    prepared::{ExecutionSlice, PreparedOperands},
    production::ProductionBackend,
};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        container.path(),
        "head-replay",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, outcome.plan.unwrap(), store)
}

/// Head 0's conditioned query, key and value rows, and its natural
/// mixed value, as the kernel handed them over.
struct SourceHead(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>);

#[derive(Default)]
struct Capture {
    position: usize,
    entering: Option<Vec<f32>>,
    heads: Vec<Vec<f32>>,
    raw: Option<Vec<f32>>,
    delta: Option<Vec<f32>>,
    after: Option<Vec<f32>>,
    source: Option<SourceHead>,
}

impl StepObserver for Capture {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = position;
        }
    }

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn entering_carrier(&mut self, position: usize, values: &[f32]) {
        if position == self.position {
            self.entering = Some(values.to_vec());
        }
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        if layer == 0 && record.position == self.position {
            if record.head == 0 {
                self.source = Some(SourceHead(
                    record.query.to_vec(),
                    record.source_keys.iter().map(|k| k.to_vec()).collect(),
                    record.source_values.iter().map(|v| v.to_vec()).collect(),
                    record.values.to_vec(),
                ));
            }
            self.heads.push(record.values.to_vec());
        }
    }

    fn attention_output(&mut self, layer: usize, position: usize, values: &[f32]) {
        if layer == 0 && position == self.position {
            self.raw = Some(values.to_vec());
        }
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.layer == 0
            && record.site == SublayerSite::Attention
            && record.position == self.position
        {
            self.delta = Some(record.delta.to_vec());
            self.after = Some(record.after.to_vec());
        }
    }
}

fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let error = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm = a
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        error
    } else {
        error / norm
    }
}

#[test]
fn replay_reconstructs_the_observed_raw_norm_and_residual_path() {
    let (_dir, plan, store): (_, _, OperandStore) = fixture();
    let backend = ProductionBackend::new();
    let prepared = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let mut capture = Capture::default();
    session.step_observed(3, &mut capture).unwrap();

    let replay = replay_attention_heads(
        &plan,
        &prepared,
        &backend,
        0,
        capture.entering.as_deref().unwrap(),
        &capture.heads,
    )
    .unwrap();
    let fast = replay_attention_mixture(
        &plan,
        &prepared,
        &backend,
        0,
        capture.entering.as_deref().unwrap(),
        &capture.heads,
    )
    .unwrap();
    assert_eq!(
        replay.contributions.len(),
        plan.layers[0].attention.softmax().unwrap().num_q_heads
    );
    assert!(replay.raw_reconstruction_relative_l2 < 1e-5);
    assert!(
        relative_l2(
            capture.raw.as_deref().unwrap(),
            &replay.raw_attention_output
        ) < 1e-6
    );
    assert!(relative_l2(capture.delta.as_deref().unwrap(), &replay.applied_delta) < 1e-6);
    assert!(relative_l2(capture.after.as_deref().unwrap(), &replay.carrier_after) < 1e-6);
    assert_eq!(fast.raw_attention_output, replay.raw_attention_output);
    assert_eq!(fast.applied_delta, replay.applied_delta);
    assert_eq!(fast.carrier_after, replay.carrier_after);
    let SourceHead(query, keys, values, natural) = capture.source.unwrap();
    let op = plan.layers[0].attention.softmax().unwrap();
    let source =
        replay_softmax_source_head(&query, &keys, &values, op.score_scale, op.logit_softcapping)
            .unwrap();
    assert_eq!(source.values, natural);
}

/// Heads of the right count and width, all zero.
fn zero_heads(plan: &ComponentOpPlan, layer: usize) -> Vec<Vec<f32>> {
    let op = plan.layers[layer].attention.softmax().unwrap();
    vec![vec![0.0; op.head_dim]; op.num_q_heads]
}

/// Every inadmissible site refuses before a projection runs — the replay
/// never answers for a carrier, head set, layer or `W_O` it was not given.
#[test]
fn replay_refuses_every_inadmissible_site() {
    let (_dir, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let prepared = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let hidden = prepared.hidden();
    let carrier = vec![0.5f32; hidden];
    let heads = zero_heads(&plan, 0);
    let refuses = |plan: &ComponentOpPlan,
                   prepared: &PreparedOperands,
                   layer: usize,
                   carrier: &[f32],
                   heads: &[Vec<f32>]| {
        assert!(replay_attention_heads(plan, prepared, &backend, layer, carrier, heads).is_err());
        assert!(replay_attention_mixture(plan, prepared, &backend, layer, carrier, heads).is_err());
    };

    refuses(&plan, &prepared, 0, &carrier[1..], &heads);
    refuses(&plan, &prepared, 0, &[f32::NAN; 1].repeat(hidden), &heads);
    refuses(&plan, &prepared, plan.layers.len(), &carrier, &heads);
    refuses(&plan, &prepared, 0, &carrier, &heads[1..]);
    let mut narrow = heads.clone();
    narrow[0].pop();
    refuses(&plan, &prepared, 0, &carrier, &narrow);
    let mut poisoned = heads.clone();
    poisoned[0][0] = f32::INFINITY;
    refuses(&plan, &prepared, 0, &carrier, &poisoned);

    let mut biased = plan.clone();
    let o = biased.layers[0].attention.softmax().unwrap().o.clone();
    biased.layers[0].attention.softmax_mut().unwrap().o_bias = Some(o);
    refuses(&biased, &prepared, 0, &carrier, &heads);

    let mut misshapen = plan.clone();
    misshapen.layers[0].attention.softmax_mut().unwrap().o.shape[0] += 1;
    refuses(&misshapen, &prepared, 0, &carrier, &heads);

    // A layer-range image answers only for the layers it prepared.
    let tail = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange {
            start: 1,
            end: plan.layers.len(),
        },
    )
    .unwrap();
    refuses(&plan, &tail, 0, &carrier, &heads);
    let head = PreparedOperands::load(
        &plan,
        &store,
        &backend,
        ExecutionSlice::LayerRange { start: 0, end: 1 },
    )
    .unwrap();
    refuses(&plan, &head, 1, &carrier, &zero_heads(&plan, 1));
}

/// All-zero heads project to a zero raw output; the reconstruction error
/// is then absolute rather than a division by a zero norm.
#[test]
fn a_zero_mixture_reconstructs_exactly_without_dividing_by_zero() {
    let (_dir, plan, store) = fixture();
    let backend = ProductionBackend::new();
    let prepared = PreparedOperands::load(&plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let carrier = vec![0.5f32; prepared.hidden()];
    let replay = replay_attention_heads(
        &plan,
        &prepared,
        &backend,
        0,
        &carrier,
        &zero_heads(&plan, 0),
    )
    .unwrap();
    assert!(replay.raw_attention_output.iter().all(|&v| v == 0.0));
    assert_eq!(replay.raw_reconstruction_relative_l2, 0.0);
    assert!(replay.contribution_norms.iter().all(|&n| n == 0.0));
}

#[test]
fn the_source_head_refuses_misaligned_or_non_finite_inputs() {
    let row = vec![1.0f32, 0.0];
    let rows = vec![row.clone(), row.clone()];
    let ok = |q: &[f32], k: &[Vec<f32>], v: &[Vec<f32>], scale: f64| {
        replay_softmax_source_head(q, k, v, scale, None).is_ok()
    };
    assert!(ok(&row, &rows, &rows, 1.0));
    assert!(!ok(&[], &rows, &rows, 1.0), "an empty query");
    assert!(!ok(&row, &[], &[], 1.0), "no sources");
    assert!(
        !ok(&row, &rows, &rows[..1], 1.0),
        "keys and values misaligned"
    );
    assert!(
        !ok(&row, &[vec![1.0]], &[vec![1.0]], 1.0),
        "a narrow source"
    );
    assert!(
        !ok(&[f32::NAN, 0.0], &rows, &rows, 1.0),
        "a non-finite query"
    );
    assert!(!ok(&row, &rows, &rows, f64::INFINITY), "a non-finite scale");
}

/// The softcap is applied before the softmax, exactly as the kernel does:
/// a cap far below the raw score flattens the distribution toward uniform.
#[test]
fn the_source_head_softcap_bends_scores_before_the_softmax() {
    let query = vec![4.0f32, 0.0];
    let keys = vec![vec![4.0, 0.0], vec![0.0, 4.0]];
    let values = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let raw = replay_softmax_source_head(&query, &keys, &values, 1.0, None).unwrap();
    let capped = replay_softmax_source_head(&query, &keys, &values, 1.0, Some(0.5)).unwrap();
    let total: f32 = capped.weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-6);
    assert!(
        capped.weights[0] < raw.weights[0],
        "the cap flattens the peak"
    );
    let expected = 0.5 * (16.0f32 / 0.5).tanh();
    let reference = [expected, 0.0];
    let max = reference.iter().copied().fold(f32::MIN, f32::max);
    let exp: Vec<f32> = reference.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    assert!((capped.weights[0] - exp[0] / sum).abs() < 1e-6);
}
