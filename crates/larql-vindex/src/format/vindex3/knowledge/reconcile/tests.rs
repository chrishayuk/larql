//! GW-0B's physical join: the feature sum must BE the recorded write.
//!
//! The synthetic arms use an independent forward written here — not the
//! view's own — so the proof gate is checked against a foreign reference,
//! and the per-feature law (fractions of a write that the features
//! reconstruct exactly sum to one) is checked under every post-norm kind.

use super::*;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightSlice;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::observe::{
    CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use crate::format::vindex3::opplan::exec::prepared::ExecutionSlice;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::plan_component_ops;
use ndarray::array;

/// Tolerances the proof gate is held to on exactly-reconstructible writes.
const EXACT: f64 = 1e-5;
/// The real fixture's write goes through a different summation order.
const RELATIVE_L2: f64 = 1e-4;
const RELATIVE_LINF: f64 = 1e-3;
const EPS: f64 = 1e-6;
const HIDDEN: usize = 2;
const BEFORE: [f32; HIDDEN] = [0.5, -0.25];

fn view() -> DenseFfnLayerView {
    DenseFfnLayerView {
        layer: 3,
        gate: Some(array![[1.0, 0.0], [0.0, 1.0]]),
        up: array![[2.0, 0.0], [0.0, 3.0]],
        down: array![[1.0, 2.0], [3.0, 4.0]],
        activation: Activation::Silu,
        pre_norm: None,
        post_norm: None,
        residual_scale: 1.0,
    }
}

fn norm(kind: NormType, weight: [f32; HIDDEN]) -> LoadedNorm {
    LoadedNorm {
        kind,
        eps: EPS,
        weight_offset: 0.0,
        weight: weight.to_vec(),
    }
}

/// An independent forward: the norms, gate and projections spelled out.
fn reference_norm(kind: NormType, weight: &[f32], x: &[f32]) -> Vec<f32> {
    let n = x.len() as f64;
    let (centre, spread) = match kind {
        NormType::RmsNorm => (
            0.0,
            x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n,
        ),
        NormType::LayerNorm => {
            let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            (
                mean,
                x.iter()
                    .map(|&v| (f64::from(v) - mean).powi(2))
                    .sum::<f64>()
                    / n,
            )
        }
    };
    let inverse = (spread + EPS).sqrt().recip();
    x.iter()
        .zip(weight)
        .map(|(&v, &w)| ((f64::from(v) - centre) * inverse) as f32 * w)
        .collect()
}

fn reference_write(view: &DenseFfnLayerView, before: &[f32]) -> Vec<f32> {
    let input = match &view.pre_norm {
        Some(n) => reference_norm(n.kind, &n.weight, before),
        None => before.to_vec(),
    };
    let dot =
        |row: ndarray::ArrayView1<f32>| -> f32 { row.iter().zip(&input).map(|(a, b)| a * b).sum() };
    let inner: Vec<f32> = (0..view.up.nrows())
        .map(|f| {
            let up = dot(view.up.row(f));
            match &view.gate {
                Some(gate) => activate(view.activation, dot(gate.row(f))) * up,
                None => activate(view.activation, up),
            }
        })
        .collect();
    let raw: Vec<f32> = (0..view.down.nrows())
        .map(|r| {
            view.down
                .row(r)
                .iter()
                .zip(&inner)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let normed = match &view.post_norm {
        Some(n) => reference_norm(n.kind, &n.weight, &raw),
        None => raw,
    };
    normed.iter().map(|v| v * view.residual_scale).collect()
}

/// A write the features reconstruct exactly has fractions summing to one.
fn assert_the_features_sum_to_the_write(view: &DenseFfnLayerView) {
    let observed = reference_write(view, &BEFORE);
    let result = view
        .attribute(&BEFORE, &observed, view.features(), EXACT, EXACT)
        .unwrap();
    assert_eq!(result.contributions.len(), view.features());
    let total: f64 = result
        .contributions
        .iter()
        .map(|c| c.observed_projection_fraction)
        .sum();
    assert!((total - 1.0).abs() < EXACT, "fractions sum to {total}");
    let ranked = result.contributions.windows(2);
    assert!(ranked
        .into_iter()
        .all(|w| w[0].contribution_l2 >= w[1].contribution_l2));
}

#[test]
fn complete_feature_sum_is_a_checked_physical_join() {
    let view = view();
    let a0 = activate(Activation::Silu, BEFORE[0]) * (2.0 * BEFORE[0]);
    let a1 = activate(Activation::Silu, BEFORE[1]) * (3.0 * BEFORE[1]);
    let observed = [a0 + 2.0 * a1, 3.0 * a0 + 4.0 * a1];
    let result = view.attribute(&BEFORE, &observed, 2, 1e-6, 1e-6).unwrap();
    assert_eq!(result.layer, 3);
    assert_eq!(result.features, 2);
    assert_eq!(result.contributions.len(), 2);
    assert!(result.proof.relative_l2_error <= 1e-6);
}

#[test]
fn a_write_that_is_not_the_feature_sum_is_refused() {
    let error = view()
        .attribute(&BEFORE, &[8.0, 9.0], 2, 1e-6, 1e-6)
        .unwrap_err();
    assert!(error.to_string().contains("attribution refused"));
}

#[test]
fn a_write_of_the_wrong_width_is_refused_before_any_arithmetic() {
    assert!(view()
        .attribute(&BEFORE[..1], &[0.0, 0.0], 2, 1.0, 1.0)
        .is_err());
    assert!(view().attribute(&BEFORE, &[0.0], 2, 1.0, 1.0).is_err());
}

#[test]
fn every_norm_placement_keeps_the_feature_law() {
    let weights = [1.5, 0.5];
    for kind in [NormType::RmsNorm, NormType::LayerNorm] {
        let mut pre = view();
        pre.pre_norm = Some(norm(kind, weights));
        assert_the_features_sum_to_the_write(&pre);

        let mut post = view();
        post.post_norm = Some(norm(kind, weights));
        post.residual_scale = 0.5;
        assert_the_features_sum_to_the_write(&post);
    }
}

#[test]
fn an_ungated_layer_activates_up_alone() {
    let mut ungated = view();
    ungated.gate = None;
    assert_the_features_sum_to_the_write(&ungated);
}

#[test]
fn top_k_keeps_the_proof_on_the_complete_sum() {
    let view = view();
    let observed = reference_write(&view, &BEFORE);
    let result = view.attribute(&BEFORE, &observed, 1, EXACT, EXACT).unwrap();
    assert_eq!(result.contributions.len(), 1);
    assert_eq!(result.features, 2);
}

#[test]
fn effective_q8_values_are_the_values_the_physical_join_uses() {
    let codes = [1, -2, 3, -4];
    let scales = [0.5, 2.0];
    let values = WeightSlice::Q8 {
        codes: &codes,
        scales: &scales,
        sums: &[],
        block: 2,
    }
    .decode_f32(1, 4)
    .unwrap();
    assert_eq!(values, vec![0.5, -1.0, 6.0, -8.0]);
}

// ── The prepared image of a real plan ──

struct Fixture {
    _checkpoint: tempfile::TempDir,
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
}

fn fixture() -> Fixture {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(dense_f32_model, checkpoint.path(), container.path(), "gw0b");
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    Fixture {
        _checkpoint: checkpoint,
        _container: container,
        plan: outcome.plan.unwrap(),
        store,
    }
}

/// The FFN write at layer 0, and the carrier it was written onto.
#[derive(Default)]
struct FfnWrite {
    before: Vec<f32>,
    delta: Vec<f32>,
}

impl StepObserver for FfnWrite {
    fn event(&mut self, _: StepEvent) {}

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if record.layer != 0 {
            return;
        }
        match record.site {
            SublayerSite::Attention => self.before = record.after.to_vec(),
            SublayerSite::Ffn => self.delta = record.delta.to_vec(),
        }
    }
}

#[test]
fn a_decoded_ffn_write_is_reconstructed_from_the_prepared_image() {
    let f = fixture();
    let backend = ProductionBackend::new();
    let prepared =
        PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let mut session = DecodeSession::new(&f.plan, &f.store, &backend).unwrap();
    let mut write = FfnWrite::default();
    session.step_observed(3, &mut write).unwrap();

    let view = DenseFfnLayerView::from_prepared(&f.plan, &f.store, &prepared, 0).unwrap();
    let result = view
        .attribute(&write.before, &write.delta, 4, RELATIVE_L2, RELATIVE_LINF)
        .unwrap();
    assert_eq!(result.features, view.features());
    assert_eq!(result.contributions.len(), 4);
    assert!(result.proof.relative_l2_error <= RELATIVE_L2);
}

#[test]
fn a_view_is_refused_for_a_layer_the_plan_cannot_decompose() {
    let f = fixture();
    let backend = ProductionBackend::new();
    let prepared =
        PreparedOperands::load(&f.plan, &f.store, &backend, ExecutionSlice::Full).unwrap();
    let view = |plan: &ComponentOpPlan, layer| {
        DenseFfnLayerView::from_prepared(plan, &f.store, &prepared, layer).is_err()
    };
    assert!(view(&f.plan, f.plan.layers.len()), "a layer past the plan");

    let mut no_ffn = f.plan.clone();
    no_ffn.layers[0].ffn = None;
    assert!(view(&no_ffn, 0), "a layer without a dense FFN");

    let mut no_embedding = f.plan.clone();
    no_embedding.embedding = None;
    assert!(view(&no_embedding, 0), "no declared hidden size");
}
