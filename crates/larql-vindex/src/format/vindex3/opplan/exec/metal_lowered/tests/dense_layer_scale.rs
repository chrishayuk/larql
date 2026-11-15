//! #741: a dense FFN layer that carries a `layer_scalar` executes on the
//! lowered Metal path, at parity with the reference interpreter.
//!
//! This used to be a refusal ("a layer scalar on a non-hybrid FFN ...").
//! It is now an execution test with a control: the same session with the
//! scalar stripped from the plan must miss parity by orders of magnitude,
//! so a lowering that dropped the scalar could not pass. The routed FFN
//! still refuses the scalar, by name.

use super::fixture::{dense_gemma4, fixture, metal, rel_rms, LOWERED_PARITY_REL_RMS};
use crate::format::vindex3::fixtures_routed::miniature_routed;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightFormats};
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::metal_lowered::LoweredSession;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerFfn};

/// Longer than the fixture's window, so the sliding layers drop rows.
const TOKENS: [u32; 9] = [3, 17, 28, 0, 11, 42, 5, 63, 19];
/// A dropped scalar must miss the parity gate by at least this factor.
const CONTROL_OVER_PARITY: f64 = 1e3;

/// Every layer's output plane, `[positions][hidden]`, from the lowered
/// session stepping `TOKENS` one position at a time.
fn lowered_planes(plan: &ComponentOpPlan, store: &OperandStore) -> Vec<Vec<Vec<f32>>> {
    let gpu = metal();
    let mut keep = Vec::new();
    let mut session = LoweredSession::new(
        &gpu,
        plan,
        store,
        WeightFormats::uniform(WeightFormat::F16),
        TOKENS.len(),
        &mut keep,
    )
    .expect("a dense layer scalar lowers");
    let mut planes = vec![Vec::new(); plan.layers.len()];
    for &token in &TOKENS {
        let (_, _, layers) = session.step_capturing(token).expect("step");
        for (plane, row) in planes.iter_mut().zip(layers) {
            plane.push(row);
        }
    }
    planes
}

#[test]
fn a_dense_layer_scalar_executes_at_reference_parity() {
    let f = fixture(dense_gemma4, "dense-layer-scale");
    assert!(
        f.plan
            .layers
            .iter()
            .all(|l| l.layer_scale.is_some() && matches!(l.ffn, Some(LayerFfn::Dense(_)))),
        "the fixture must carry a layer scalar on every dense layer"
    );
    let reference = execute_plan(&f.plan, &f.store, &TOKENS, &ReferenceBackend::new()).unwrap();
    let lowered = lowered_planes(&f.plan, &f.store);
    let mut worst = 0.0f64;
    for (layer, (want, got)) in reference.layers.iter().zip(&lowered).enumerate() {
        let e = rel_rms(want.post_layer.rows(), got);
        eprintln!("layer {layer}: lowered vs reference rel_rms {e:.3e}");
        assert!(
            e <= LOWERED_PARITY_REL_RMS,
            "layer {layer}: lowered vs reference rel_rms {e:.3e} > {LOWERED_PARITY_REL_RMS:.0e}"
        );
        worst = worst.max(e);
    }

    // Control: the scalar stripped from the lowered plan only.
    let mut unscaled = f.plan.clone();
    unscaled
        .layers
        .iter_mut()
        .for_each(|l| l.layer_scale = None);
    let dropped = lowered_planes(&unscaled, &f.store);
    let moved = rel_rms(reference.layers[0].post_layer.rows(), &dropped[0]);
    assert!(
        moved > worst.max(LOWERED_PARITY_REL_RMS) * CONTROL_OVER_PARITY,
        "a lowering without the scalar must miss parity: moved {moved:.3e}, parity {worst:.3e}"
    );
}

#[test]
fn a_layer_scalar_on_a_routed_ffn_is_still_refused_by_name() {
    let f = fixture(miniature_routed, "routed-layer-scale");
    let mut plan = f.plan.clone();
    let routed = plan
        .layers
        .iter()
        .position(|l| matches!(l.ffn, Some(LayerFfn::Routed(_))))
        .expect("the routed miniature carries a routed FFN");
    // Any operand stands in: the refusal fires before anything loads.
    let stand_in = plan.layers[routed]
        .pre_attention_norm
        .as_ref()
        .expect("a pre-attention norm")
        .weight
        .clone();
    plan.layers[routed].layer_scale = Some(stand_in);
    let gpu = metal();
    let mut keep = Vec::new();
    let err = match LoweredSession::new(
        &gpu,
        &plan,
        &f.store,
        WeightFormats::uniform(WeightFormat::F16),
        1,
        &mut keep,
    ) {
        Ok(_) => panic!("a layer scalar on a routed FFN must refuse"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("layer scalar on a routed FFN"), "{err}");
    assert!(err.contains(&format!("layer {routed}")), "{err}");
}
