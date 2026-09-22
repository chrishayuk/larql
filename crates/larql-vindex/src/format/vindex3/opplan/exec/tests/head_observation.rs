//! Per-head capture witnesses, distinct from frozen Standard carrier observation.
use super::decode::fixture;
use super::golden::G_TOKENS;
use crate::format::vindex3::opplan::exec::{
    decode::DecodeSession,
    observe::{AttentionHeadRecord, StepEvent, StepObserver},
    production::ProductionBackend,
    reference::ReferenceBackend,
};

struct CapturedHead {
    layer: usize,
    position: usize,
    head: usize,
    kv_head: usize,
    start: usize,
    weights: Vec<f32>,
    values: Vec<f32>,
    query: Vec<f32>,
    source_keys: Vec<Vec<f32>>,
    source_values: Vec<Vec<f32>>,
}
#[derive(Default)]
struct Heads {
    rows: Vec<CapturedHead>,
}
impl StepObserver for Heads {
    fn event(&mut self, _: StepEvent) {}
    fn wants_attention_heads(&self) -> bool {
        true
    }
    fn attention_head(&mut self, layer: usize, r: AttentionHeadRecord<'_>) {
        self.rows.push(CapturedHead {
            layer,
            position: r.position,
            head: r.head,
            kv_head: r.kv_head,
            start: r.source_start,
            weights: r.weights.to_vec(),
            values: r.values.to_vec(),
            query: r.query.to_vec(),
            source_keys: r.source_keys.to_vec(),
            source_values: r.source_values.to_vec(),
        });
    }
}

#[test]
fn heads_capture_preserves_logits_and_covers_gqa_and_truncated_window() {
    let (_dir, plan, store) = fixture();
    let backend = ProductionBackend::new();
    // Exercise the shared-handle forwarding as well as the production implementation.
    let reference = std::sync::Arc::new(backend);
    let mut observed = DecodeSession::new(&plan, &store, &reference).unwrap();
    let mut control = DecodeSession::new(&plan, &store, &backend).unwrap();
    let mut capture = Heads::default();
    for &token in &G_TOKENS {
        let a = observed
            .step_observed(token, &mut capture)
            .unwrap()
            .logits
            .unwrap();
        let b = control.step(token).unwrap().logits.unwrap();
        assert!(a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let expected = G_TOKENS.len()
        * plan
            .layers
            .iter()
            .map(|l| l.attention.softmax().unwrap().num_q_heads)
            .sum::<usize>();
    assert_eq!(capture.rows.len(), expected);
    let mut unique = std::collections::BTreeSet::new();
    let mut truncated = false;
    for CapturedHead {
        layer,
        position,
        head,
        kv_head,
        start,
        weights,
        values,
        query,
        source_keys,
        source_values,
    } in &capture.rows
    {
        assert!(unique.insert((*layer, *position, *head)));
        let op = plan.layers[*layer].attention.softmax().unwrap();
        assert_eq!(*kv_head, head / (op.num_q_heads / op.num_kv_heads));
        assert_eq!(values.len(), op.head_dim);
        assert_eq!(query.len(), op.head_dim);
        assert_eq!(source_keys.len(), weights.len());
        assert_eq!(source_values.len(), weights.len());
        assert!(source_keys.iter().all(|row| row.len() == op.head_dim));
        assert!(source_values.iter().all(|row| row.len() == op.head_dim));
        assert!(values.iter().all(|v| v.is_finite()));
        assert_eq!(weights.len(), position + 1 - start);
        assert!(weights
            .iter()
            .all(|v| v.is_finite() && *v >= 0. && *v <= 1.));
        assert!((weights.iter().sum::<f32>() - 1.).abs() < 1e-5);
        if *start > 0 {
            truncated = true;
            assert_eq!(weights.len(), op.window.unwrap());
        }
    }
    assert!(truncated, "fixture must cross the sliding-window boundary");
}

#[test]
fn unsupported_backend_refuses_requested_head_capture() {
    let (_dir, plan, store) = fixture();
    let backend = ReferenceBackend::new();
    let mut session = DecodeSession::new(&plan, &store, &backend).unwrap();
    let error = session
        .step_observed(G_TOKENS[0], &mut Heads::default())
        .err()
        .expect("unsupported capture must refuse");
    assert!(error.to_string().contains("per-head capture unavailable"));
}
