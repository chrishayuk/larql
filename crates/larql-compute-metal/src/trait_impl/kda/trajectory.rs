//! The PHYSICAL-1 trajectory witness: score two KDA arms step by step
//! while both carry their own recurrent state.
//!
//! Shared by the synthetic gate (`super::q4_trajectory`) and the
//! real-weight example (`examples/kda_q4_trajectory_real.rs`) so that a
//! synthetic result and a Kimi or GLM result are produced by ONE
//! implementation. Two copies of a metric are two metrics.
//!
//! **The state is the primary boundary.** A quantisation error in a
//! feed-forward projection perturbs one token; the same error in a
//! recurrent operator enters the state and is carried forward, so the
//! question is whether the distance GROWS with sequence length, which no
//! single-step comparison can answer.

use super::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use crate::MetalBackend;

/// One step's agreement between a reference arm and a candidate arm.
///
/// Three output columns are carried deliberately rather than one — see
/// [`StepMetric::out_rel_l2_raw`].
#[derive(Debug, Clone, Copy)]
pub struct StepMetric {
    pub step: usize,
    /// PRIMARY. `‖s_cnd − s_ref‖₂ / ‖s_ref‖₂` over the recurrent state.
    pub state_rel_l2: f32,
    /// PRIMARY. Cosine between the two recurrent states.
    pub state_cos: f32,
    /// PRE-REGISTERED per-step relative output error, reported unchanged.
    ///
    /// **It is pathological around a small reference norm and it is kept
    /// anyway.** On the synthetic substrate the honest Q4_K arm read
    /// 18.68 here at 512 steps while its state sat at 0.012, purely
    /// because `‖out_ref‖` swings ~1000x along one trajectory. It was
    /// declared before the run, so it is reported, not replaced.
    pub out_rel_l2_raw: f32,
    /// `‖out_cnd − out_ref‖₂`, unnormalised — the quantity both relative
    /// forms divide, published so neither normalisation can hide it.
    pub out_abs_l2: f32,
    /// DIAGNOSTIC, added after the raw metric was shown to be
    /// pathological: the same difference over the trajectory's MEAN
    /// reference norm, so a step whose output passes near zero cannot
    /// manufacture a large error.
    ///
    /// **This is not an amended acceptance criterion.** Acceptance is the
    /// state columns; this exists to make the output channel readable.
    pub out_rel_l2_traj: f32,
    /// The step's own `‖out_ref‖₂`, so the pathology stays visible.
    pub out_ref_norm: f32,
}

pub fn rel_l2(reference: &[f32], candidate: &[f32]) -> f32 {
    let num: f32 = reference
        .iter()
        .zip(candidate)
        .map(|(x, y)| (x - y) * (x - y))
        .sum();
    let den: f32 = reference.iter().map(|x| x * x).sum();
    (num / den.max(f32::MIN_POSITIVE)).sqrt()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(f32::MIN_POSITIVE)
}

/// Run both arms over the same token sequence, carrying state, scoring
/// every step.
///
/// `inputs` supplies the hidden state for each step; both arms see the
/// SAME vector, so a divergence can only come from the weights.
/// `reset_each_step` is the state-reset control: with it on, no step can
/// inherit the previous step's divergence.
pub fn kda_trajectory(
    metal: &MetalBackend,
    shape: KdaShape,
    reference: KdaDeviceWeights<'_>,
    candidate: KdaDeviceWeights<'_>,
    inputs: &[Vec<f32>],
    reset_each_step: bool,
) -> Vec<StepMetric> {
    let s_ref = KdaDeviceState::zeros(metal, shape);
    let s_cnd = KdaDeviceState::zeros(metal, shape);
    let mut out = Vec::with_capacity(inputs.len());
    for (step, x) in inputs.iter().enumerate() {
        let (out_r, _) = metal
            .kda_attention_step(reference, shape, &s_ref, x)
            .expect("reference arm runs");
        let (out_c, _) = metal
            .kda_attention_step(candidate, shape, &s_cnd, x)
            .expect("candidate arm runs");
        let (st_r, _) = s_ref.read_back();
        let (st_c, _) = s_cnd.read_back();
        let abs: f32 = out_r
            .iter()
            .zip(&out_c)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        out.push(StepMetric {
            step,
            state_rel_l2: rel_l2(&st_r, &st_c),
            state_cos: cosine(&st_r, &st_c),
            out_rel_l2_raw: rel_l2(&out_r, &out_c),
            out_abs_l2: abs,
            // Filled below, once the trajectory's scale is known.
            out_rel_l2_traj: 0.0,
            out_ref_norm: out_r.iter().map(|x| x * x).sum::<f32>().sqrt(),
        });
        if reset_each_step {
            s_ref.reset();
            s_cnd.reset();
        }
    }
    let scale =
        (out.iter().map(|m| m.out_ref_norm).sum::<f32>() / out.len() as f32).max(f32::MIN_POSITIVE);
    for m in &mut out {
        m.out_rel_l2_traj = m.out_abs_l2 / scale;
    }
    out
}

/// `(worst state rel-L2, worst state cosine, worst raw output rel-L2,
/// worst trajectory-normalised output rel-L2)`.
pub fn worst(t: &[StepMetric]) -> (f32, f32, f32, f32) {
    t.iter().fold((0.0, 1.0, 0.0, 0.0), |(r, c, o, j), m| {
        (
            r.max(m.state_rel_l2),
            c.min(m.state_cos),
            o.max(m.out_rel_l2_raw),
            j.max(m.out_rel_l2_traj),
        )
    })
}

/// `state_rel@last / state_rel@early` — does divergence ACCUMULATE with
/// sequence length? A per-step bound cannot answer this, and it is the
/// whole reason the rung exists.
pub fn drift_ratio(t: &[StepMetric], early: usize) -> f32 {
    let e = t[early.min(t.len()) - 1].state_rel_l2;
    t.last().expect("non-empty").state_rel_l2 / e.max(f32::MIN_POSITIVE)
}

/// Every column, at a readable stride. Both relative output forms are
/// printed side by side on purpose.
pub fn report(name: &str, t: &[StepMetric], early: usize) {
    let (r, c, o_raw, o_traj) = worst(t);
    println!(
        "[{name}] steps={} | STATE worst_rel={r:.6} worst_cos={c:.6} drift={:.3} \
         | OUT raw={o_raw:.4} (preregistered) traj={o_traj:.4} (diagnostic)",
        t.len(),
        drift_ratio(t, early)
    );
    println!(
        "    {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "step", "state_rel", "state_cos", "out_raw", "out_abs", "|out_ref|"
    );
    for m in t
        .iter()
        .filter(|m| m.step == 0 || (m.step + 1) % 64 == 0 || m.step + 1 == t.len())
    {
        println!(
            "    {:>5}  {:>10.6}  {:>10.6}  {:>10.4}  {:>10.4}  {:>10.4}",
            m.step, m.state_rel_l2, m.state_cos, m.out_rel_l2_raw, m.out_abs_l2, m.out_ref_norm
        );
    }
}
