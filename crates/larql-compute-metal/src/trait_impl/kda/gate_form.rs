//! KDA-GATE-METAL-1 — Metal executes the FAMILY-DECLARED decay-gate
//! form, or refuses. It may never silently substitute one for the other.
//!
//! Kimi Linear and GLM-5.3-Flash compute different decay gates: Kimi
//! `-exp(A_log)·softplus(pre)`, unbounded below; GLM
//! `lower_bound·sigmoid(exp(A_log)·pre)`, floored at `exp(lower_bound)`.
//!
//! **Neither the presence nor the absence of `gate_lower_bound` selects
//! the form**, verified on the checkpoints 2026-09-07: Kimi-Linear-48B
//! mentions the key in no file at all and computes softplus, while GLM
//! declares -5.0 *and* defaults to the same -5.0 when the key is null
//! and `safe_gate` is set — so an absent bound still means clamped
//! there. Absence means opposite things in the two families. Until this rung the Metal
//! kernel hard-coded Kimi's, so a GLM layer would have run a plausible,
//! wrong recurrence whose per-step decay error (mean -0.906 → -2.528,
//! 2.8x) compounds with context — invisible to any single-position check
//! and, worse, indistinguishable from a quantisation defect at the exact
//! moment PHYSICAL-1 is trying to score one.
//!
//! The gates below are written so that each form is checked against the
//! scalar formula it claims, and so that swapping them is LOUD.

use super::trajectory::{kda_trajectory, worst};
use super::*;
use crate::MetalBackend;
use larql_models::config::KdaGateForm;

const HEADS: usize = 2;
const DIM: usize = 8;
const HIDDEN: usize = 16;
const WIDTH: usize = HEADS * DIM;
const KERNEL: usize = 4;
const BOUND: f32 = -5.0;

fn shape() -> KdaShape {
    KdaShape {
        hidden: HIDDEN,
        num_heads: HEADS,
        head_dim: DIM,
        conv_kernel: KERNEL,
    }
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn bf16(n: usize, k: usize, seed: f32) -> Vec<u8> {
    synth(n * k, seed)
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

struct W {
    qkv: Vec<u8>,
    offsets: [ExpertOffset; CONV_STREAMS],
    o: Vec<u8>,
    conv: [Vec<f32>; CONV_STREAMS],
    fa: Vec<f32>,
    fb: Vec<f32>,
    ga: Vec<f32>,
    gb: Vec<f32>,
    bp: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
    o_norm: Vec<f32>,
}

fn weights() -> W {
    let per = WIDTH * HIDDEN * 2;
    let mut qkv = Vec::new();
    for seed in [0.1f32, 1.3, 2.7] {
        qkv.extend_from_slice(&bf16(WIDTH, HIDDEN, seed));
    }
    W {
        qkv,
        offsets: [
            ExpertOffset(0),
            ExpertOffset(per as u32),
            ExpertOffset((2 * per) as u32),
        ],
        o: bf16(HIDDEN, WIDTH, 3.9),
        conv: [
            synth(WIDTH * KERNEL, 0.5),
            synth(WIDTH * KERNEL, 1.5),
            synth(WIDTH * KERNEL, 2.5),
        ],
        fa: synth(DIM * HIDDEN, 4.1),
        fb: synth(WIDTH * DIM, 5.2),
        ga: synth(DIM * HIDDEN, 6.3),
        gb: synth(WIDTH * DIM, 7.4),
        bp: synth(HEADS * HIDDEN, 8.5),
        a_log: synth(HEADS, 9.6),
        dt: synth(WIDTH, 10.7),
        o_norm: synth(DIM, 11.8).iter().map(|v| v + 1.0).collect(),
    }
}

impl W {
    fn device(&self, gate_form: KdaGateForm) -> KdaDeviceWeights<'_> {
        KdaDeviceWeights {
            qkv_bank: &self.qkv,
            qkv_offsets: &self.offsets,
            o_proj: &self.o,
            projection_encoding: ExpertEncoding::Bf16,
            q_conv1d: &self.conv[0],
            k_conv1d: &self.conv[1],
            v_conv1d: &self.conv[2],
            f_a_proj: SmallMatrix::F32(&self.fa),
            f_b_proj: SmallMatrix::F32(&self.fb),
            g_a_proj: SmallMatrix::F32(&self.ga),
            g_b_proj: SmallMatrix::F32(&self.gb),
            b_proj: SmallMatrix::F32(&self.bp),
            a_log: &self.a_log,
            dt_bias: &self.dt,
            o_norm: &self.o_norm,
            norm_eps: 1e-5,
            gate_form,
        }
    }
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device")
}

fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else {
        v.exp().ln_1p()
    }
}

/// Run one step and return `(f_low, decay)` — the kernel's own input and
/// output, so the formula is scored on the SAME `f_low` it saw rather
/// than on a re-derivation.
fn decay_planes(form: KdaGateForm) -> (Vec<f32>, Vec<f32>) {
    let m = backend();
    let w = weights();
    let state = KdaDeviceState::zeros(&m, shape());
    let p = m
        .kda_attention_step_traced(w.device(form), shape(), &state, &synth(HIDDEN, 0.3))
        .expect("step runs");
    (p.f_lowrank, p.g_decay)
}

/// Kimi's form, elementwise against the scalar formula it claims.
#[test]
fn metal_computes_softplus_when_that_is_the_declared_form() {
    let w = weights();
    let (f_low, decay) = decay_planes(KdaGateForm::Softplus);
    for i in 0..WIDTH {
        let a = w.a_log[i / DIM].exp();
        let want = -a * softplus(f_low[i] + w.dt[i]);
        assert!(
            (decay[i] - want).abs() < 1e-5,
            "channel {i}: device {} vs softplus formula {want}",
            decay[i]
        );
    }
}

/// GLM's form, elementwise against the scalar formula it claims — and
/// bounded below by its own declaration, which softplus is not.
#[test]
fn metal_computes_clamped_sigmoid_when_that_is_the_declared_form() {
    let w = weights();
    let (f_low, decay) = decay_planes(KdaGateForm::ClampedSigmoid { lower_bound: BOUND });
    for i in 0..WIDTH {
        let a = w.a_log[i / DIM].exp();
        let want = BOUND * (1.0 / (1.0 + (-(a * (f_low[i] + w.dt[i]))).exp()));
        assert!(
            (decay[i] - want).abs() < 1e-5,
            "channel {i}: device {} vs clamped-sigmoid formula {want}",
            decay[i]
        );
        assert!(
            decay[i] >= BOUND && decay[i] <= 0.0,
            "channel {i}: {} escaped the declared bound [{BOUND}, 0]",
            decay[i]
        );
    }
}

/// The two forms are not the same computation, in BOTH directions.
///
/// A one-sided control cannot tell "declared and honoured" from
/// "declared and ignored in favour of a hard-coded form": if the kernel
/// silently ran softplus always, the clamped arm would simply equal it.
#[test]
fn the_two_declared_forms_are_different_computations() {
    let (_, soft) = decay_planes(KdaGateForm::Softplus);
    let (_, clamped) = decay_planes(KdaGateForm::ClampedSigmoid { lower_bound: BOUND });
    let moved = soft
        .iter()
        .zip(&clamped)
        .filter(|(a, b)| (*a - *b).abs() > 1e-4)
        .count();
    assert_eq!(
        moved, WIDTH,
        "only {moved} of {WIDTH} channels differ between the two forms — the kernel \
         is not reading the form it was given"
    );
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (ms, mc) = (mean(&soft), mean(&clamped));
    assert!(
        ms < 0.0 && mc < 0.0 && (ms / mc - 1.0).abs() > 0.1,
        "mean decay softplus {ms} vs clamped {mc}: the forms should differ \
         materially in per-step decay, which is what compounds with context"
    );
    // Softplus is unbounded below; the clamped form is floored by its own
    // declaration. That asymmetry is the substantive difference.
    assert!(
        clamped.iter().all(|d| *d >= BOUND),
        "the clamped form escaped its bound"
    );
}

/// Serving the wrong form moves the RECURRENT TRAJECTORY, not merely one
/// gate vector — and it moves it far more than Q4_K quantisation does
/// (0.128 on real Kimi weights). This is why a wrong form would be
/// misread as a representation failure.
#[test]
fn serving_the_wrong_form_moves_the_recurrent_trajectory() {
    let m = backend();
    let w = weights();
    let xs: Vec<Vec<f32>> = (0..64)
        .map(|s| synth(HIDDEN, 0.3 + s as f32 * 0.11))
        .collect();
    let t = kda_trajectory(
        &m,
        shape(),
        w.device(KdaGateForm::Softplus),
        w.device(KdaGateForm::ClampedSigmoid { lower_bound: BOUND }),
        &xs,
        false,
    );
    let (rel, cos, _, _) = worst(&t);
    assert!(
        rel > 0.5 && cos < 0.99,
        "swapping the decay form moved the state only rel={rel} cos={cos} — a defect \
         this large must not be able to hide under a quantisation budget"
    );
}

/// An unjudged family is REFUSED, never defaulted. There is no safe
/// default: `gate_lower_bound` is declared by families that ignore it.
#[test]
fn an_undeclared_gate_form_is_refused_by_name() {
    let err = declared_gate_form(None).expect_err("None must refuse");
    assert!(matches!(err, GroupedError::KdaGateFormUndeclared));
    let text = err.to_string();
    assert!(
        text.contains("decay-gate form") && text.contains("gate_lower_bound"),
        "the refusal must name the fact and why the declaration cannot supply it: {text}"
    );
    // And a declared form passes through unchanged.
    assert_eq!(
        declared_gate_form(Some(KdaGateForm::ClampedSigmoid { lower_bound: BOUND }))
            .expect("declared passes"),
        KdaGateForm::ClampedSigmoid { lower_bound: BOUND }
    );
}
