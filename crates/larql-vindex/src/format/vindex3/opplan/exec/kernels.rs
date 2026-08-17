//! Naive f32 reference math for the plan executor (V3-G5b-2).
//!
//! Deliberately shares **nothing** with `larql-compute`'s kernels — the
//! Stage A oracle is the production forward path, and a reference that
//! called the same kernels would agree with it by construction (the
//! `runtime/fixtures/direct_moe_oracle.rs` discipline). Plain loops,
//! row-major `[out, in]` weights, no BLAS, no SIMD: semantic fidelity is
//! the only job.

use larql_models::config::{Activation, NormType};

/// `y[o] = Σ_i w[o*in + i] * x[i]` — weight stored `[out, in]` row-major.
pub fn matvec(weight: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    let mut y = vec![0.0f32; out_dim];
    for (o, y_o) in y.iter_mut().enumerate() {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for (w, v) in row.iter().zip(x) {
            acc += w * v;
        }
        *y_o = acc;
    }
    y
}

/// Gain of a parameter-free norm: the statistic alone, applied with the
/// identity. Named because `1.0` here is a judged semantic (weightless
/// normalisation), not an arbitrary starting value.
const PARAMETER_FREE_GAIN: f32 = 1.0;

/// Per-element gain: the identity when the norm is parameter-free, the
/// stored weight plus its offset otherwise.
///
/// Indexes `weight` directly instead of tolerating a missing element. A
/// weight that is neither empty nor `x`-length is a geometry bug, and
/// padding the tail would convert that bug into a plausible-looking
/// output — the norm would still return finite numbers, just wrong ones,
/// and a parity table would show drift with no obvious cause.
fn gain(weight: &[f32], weight_offset: f32, i: usize) -> f32 {
    if weight.is_empty() {
        PARAMETER_FREE_GAIN
    } else {
        weight[i] + weight_offset
    }
}

/// Normalise one vector in place-by-return with the given kind, epsilon,
/// weight and weight offset. `weight` is either empty (parameter-free
/// application, RMS statistic only) or exactly `x.len()` long; any other
/// length is a geometry bug and panics rather than being padded.
pub fn norm(kind: NormType, x: &[f32], weight: &[f32], weight_offset: f32, eps: f64) -> Vec<f32> {
    assert!(
        weight.is_empty() || weight.len() == x.len(),
        "norm weight must be empty or {} long, got {}",
        x.len(),
        weight.len()
    );
    match kind {
        NormType::RmsNorm => {
            let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            let inv = 1.0 / ((ss / x.len() as f64) + eps).sqrt();
            x.iter()
                .enumerate()
                .map(|(i, v)| ((*v as f64) * inv) as f32 * gain(weight, weight_offset, i))
                .collect()
        }
        NormType::LayerNorm => {
            let mean: f64 = x.iter().map(|v| *v as f64).sum::<f64>() / x.len() as f64;
            let var: f64 = x
                .iter()
                .map(|v| {
                    let d = *v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / x.len() as f64;
            let inv = 1.0 / (var + eps).sqrt();
            x.iter()
                .enumerate()
                .map(|(i, v)| (((*v as f64 - mean) * inv) as f32) * gain(weight, weight_offset, i))
                .collect()
        }
    }
}

/// The judged activation functions.
pub fn activate(activation: Activation, x: f32) -> f32 {
    match activation {
        Activation::Silu => x * sigmoid(x),
        Activation::Gelu => 0.5 * x * (1.0 + erf_approx(x / std::f32::consts::SQRT_2)),
        Activation::GeluTanh => {
            const SQRT_2_OVER_PI: f32 = 0.797_884_6;
            const CUBIC: f32 = 0.044_715;
            0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + CUBIC * x * x * x)).tanh())
        }
        Activation::Relu => x.max(0.0),
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Abramowitz–Stegun erf approximation — reference-grade.
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Numerically-stable softmax in place.
pub fn softmax(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
}

/// Softmax against an attention sink, in the literal form of the judged
/// semantics: the sink logit is appended to the row, the row is
/// softmaxed whole, and the sink's column is dropped — so the surviving
/// weights sum to `1 − p_sink`. Deliberately not the served path's
/// denominator-only shortcut, so the two are independent transcriptions
/// of one definition.
pub fn softmax_with_sink(scores: &mut [f32], sink: f32) {
    let mut row = Vec::with_capacity(scores.len() + 1);
    row.extend_from_slice(scores);
    row.push(sink);
    softmax(&mut row);
    scores.copy_from_slice(&row[..scores.len()]);
}

/// Rotate-half RoPE on one head slice at one position, matching the
/// production convention: pair `i` with `i + head_dim/2`,
/// `inv_freq[i] = theta^(-2i/head_dim)`.
pub fn rope_rotate(head: &mut [f32], position: usize, theta: f64) {
    let half = head.len() / 2;
    for i in 0..half {
        let inv_freq = theta.powf(-2.0 * i as f64 / head.len() as f64);
        let angle = position as f64 * inv_freq;
        let (sin_t, cos_t) = (angle.sin() as f32, angle.cos() as f32);
        let x0 = head[i];
        let x1 = head[half + i];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[half + i] = x0 * sin_t + x1 * cos_t;
    }
}

/// Rotate-half RoPE on one head slice at one position with **given**
/// per-pair inverse frequencies and an amplitude on `cos`/`sin` — the form
/// every rotary scaling reduces to (`angle = pos · inv_freq[i]`,
/// `cos·amplitude`, `sin·amplitude`). Plain rope is `theta^(-2i/d)` with
/// amplitude 1; YaRN supplies a ramped blend and an amplitude that is not.
pub fn rope_rotate_scaled(head: &mut [f32], position: usize, inv_freq: &[f64], amplitude: f32) {
    let half = head.len() / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for i in 0..half {
        let angle = position as f64 * inv_freq[i];
        let sin_t = angle.sin() as f32 * amplitude;
        let cos_t = angle.cos() as f32 * amplitude;
        let x0 = head[i];
        let x1 = head[half + i];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[half + i] = x0 * sin_t + x1 * cos_t;
    }
}

/// YaRN's per-pair inverse frequencies and attention amplitude for one
/// head of `head_dim` at base `theta` — the reference transcription of
/// HF's `_compute_yarn_parameters`, sharing nothing with the served
/// `larql-compute` rope module:
///
/// ```text
/// dim(rot)      = d · ln(L / (rot · 2π)) / (2 · ln θ)        find_correction_dim
/// low, high     = dim(β_fast), dim(β_slow)  [floor/ceil if truncate], clamped to [0, d−1]
/// ramp[i]       = clamp((i − low) / (high − low), 0, 1)     (high nudged +0.001 if == low)
/// inv_freq[i]   = extrap[i]/factor · ramp[i] + extrap[i] · (1 − ramp[i])
/// amplitude     = YarnRopeScaling::attention_amplitude (the one authority)
/// ```
pub fn yarn_frequencies(
    scaling: &larql_models::YarnRopeScaling,
    head_dim: usize,
    theta: f64,
) -> (Vec<f64>, f32) {
    let half = head_dim / 2;
    let d = head_dim as f64;
    let correction_dim = |rotations: f64| {
        (d * (scaling.original_max_position_embeddings / (rotations * std::f64::consts::TAU)).ln())
            / (2.0 * theta.ln())
    };
    let mut low = correction_dim(scaling.beta_fast);
    let mut high = correction_dim(scaling.beta_slow);
    if scaling.truncate {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let high = high.min(d - 1.0);
    let high = if high == low { high + 0.001 } else { high };
    let inv_freq = (0..half)
        .map(|i| {
            let extrapolation = theta.powf(-2.0 * i as f64 / d);
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            extrapolation / scaling.factor * ramp + extrapolation * (1.0 - ramp)
        })
        .collect();
    (inv_freq, scaling.attention_amplitude() as f32)
}

/// Tanh softcap: `cap * tanh(x / cap)`.
pub fn softcap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}
