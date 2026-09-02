//! Naive f32 reference math for the plan executor (V3-G5b-2).
//!
//! Deliberately shares **nothing** with `larql-compute`'s kernels — the
//! Stage A oracle is the production forward path, and a reference that
//! called the same kernels would agree with it by construction (the
//! `runtime/fixtures/direct_moe_oracle.rs` discipline). Plain loops,
//! row-major `[out, in]` weights, no BLAS, no SIMD: semantic fidelity is
//! the only job.

use larql_models::config::{mrope_axis_table, Activation, NormType};

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

/// A partial rotary's per-pair inverse frequencies over the whole head,
/// in the rotate-half layout [`rope_rotate_scaled`] applies (pair `i` is
/// dims `i` and `i + head_dim/2`), transcribed from HF:
///
/// * `HeadWidth` (`_compute_proportional_rope_parameters`): the first
///   `rotary_fraction · head_dim / 2` pairs at `theta^(-2i/head_dim)`, the
///   rest at zero frequency (an identity rotation) — the encoding is
///   always head-sized, so the rotated dims are the low pairs of the
///   FULL head, not a contiguous prefix.
/// * `RotaryWidth` (the plain partial rotary): the first
///   `rotary_fraction · head_dim` DIMS form their own rotate-half block
///   with pairs `(i, i + rotary_dim/2)` at `theta^(-2i/rotary_dim)`; that
///   pairing is not the full-head one, so it is applied by
///   [`rope_rotate`] over the prefix slice rather than through this table
///   (`partial_rotary_slice`).
pub fn partial_rotary_frequencies(head_dim: usize, rotary_fraction: f64, theta: f64) -> Vec<f64> {
    let half = head_dim / 2;
    let rotated_pairs = ((rotary_fraction * head_dim as f64) as usize) / 2;
    (0..half)
        .map(|i| {
            if i < rotated_pairs {
                theta.powf(-2.0 * i as f64 / head_dim as f64)
            } else {
                0.0
            }
        })
        .collect()
}

/// The prefix width a `RotaryWidth` partial rotary rotates as its own
/// rotate-half block — HF's `int(head_dim * partial_rotary_factor)`.
pub fn partial_rotary_slice(head_dim: usize, rotary_fraction: f64) -> usize {
    (head_dim as f64 * rotary_fraction) as usize
}

/// Rotate-half M-RoPE on one head at one **grid** position: the rotary
/// prefix rotates as its own block, and each frequency slot draws its
/// angle from the position axis its section assigns it.
///
/// `positions` is `(t, h, w)`. On the text path all three are the token
/// index, and the axis lookup therefore selects between equal values —
/// so this reduces, exactly, to a plain partial rotary. That degeneracy
/// is a property of the INPUT, not of this function: the assignment is
/// consulted either way, so the code an image position would take is the
/// code text takes, and only the position source differs. Bypassing the
/// lookup for text would have made the multi-axis path untested by
/// construction.
pub fn mrope_rotate(
    head: &mut [f32],
    positions: [usize; 3],
    theta: f64,
    section: [usize; 3],
    interleaved: bool,
) {
    let rotary_dim = head.len();
    let half = rotary_dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| theta.powf(-2.0 * i as f64 / rotary_dim as f64))
        .collect();
    let axes = mrope_axis_table(section, interleaved, half);
    mrope_rotate_scaled(head, positions, &axes, &inv_freq, 1.0);
}

/// [`mrope_rotate`] with **given** per-slot inverse frequencies, so the
/// served frequency planner and the reference transcription share the
/// application while keeping their own arithmetic — the same split
/// [`rope_rotate_scaled`] already provides for single-axis rotaries.
///
/// `axes[i]` is the position axis slot `i` draws from
/// ([`mrope_axis_table`]).
pub fn mrope_rotate_scaled(
    head: &mut [f32],
    positions: [usize; 3],
    axes: &[u8],
    inv_freq: &[f64],
    amplitude: f32,
) {
    let half = head.len() / 2;
    debug_assert_eq!(inv_freq.len(), half);
    debug_assert_eq!(axes.len(), half);
    for i in 0..half {
        let angle = positions[axes[i] as usize] as f64 * inv_freq[i];
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

/// Llama-3 wavelength-band frequencies, transcribed from HF's
/// `_compute_llama3_parameters`.
///
/// An independent transcription, like [`yarn_frequencies`] beside it —
/// the reference exec path exists to disagree with the production one
/// when the production one is wrong, which it cannot do by calling it.
///
/// Frequencies only. Unlike YaRN there is no amplitude term: `cos` and
/// `sin` keep unit scale, so this returns no second value. Giving it one
/// "for symmetry" would invite a caller to apply an attention-temperature
/// change that Llama-3 does not specify.
///
/// Each half-pair's wavelength `2π / inv_freq` is compared against two
/// thresholds derived from the *pre-trained* context window:
/// - shorter than `original / high_freq_factor` — rotating fast enough to
///   be unaffected by the extension, left alone;
/// - longer than `original / low_freq_factor` — divided by `factor`;
/// - between them — a smooth blend of the two.
pub fn llama3_frequencies(
    scaling: &larql_models::Llama3RopeScaling,
    head_dim: usize,
    theta: f64,
) -> Vec<f64> {
    let half = head_dim / 2;
    let d = head_dim as f64;
    let low_freq_wavelen = scaling.original_max_position_embeddings / scaling.low_freq_factor;
    let high_freq_wavelen = scaling.original_max_position_embeddings / scaling.high_freq_factor;
    (0..half)
        .map(|i| {
            let inv_freq = theta.powf(-2.0 * i as f64 / d);
            let wavelen = std::f64::consts::TAU / inv_freq;
            if wavelen < high_freq_wavelen {
                inv_freq
            } else if wavelen > low_freq_wavelen {
                inv_freq / scaling.factor
            } else {
                let smooth = (scaling.original_max_position_embeddings / wavelen
                    - scaling.low_freq_factor)
                    / (scaling.high_freq_factor - scaling.low_freq_factor);
                (1.0 - smooth) * inv_freq / scaling.factor + smooth * inv_freq
            }
        })
        .collect()
}

/// Tanh softcap: `cap * tanh(x / cap)`.
pub fn softcap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}

/// Which half of a fused query/gate projection to gather.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedHalf {
    Query,
    Gate,
}

/// Gather one half of a fused query/gate projection.
///
/// The projection emits `2 · head_dim` rows per head, laid out
/// `[q_h0 | gate_h0 | q_h1 | gate_h1 | …]` — transcribed from HF
/// `Qwen3_5Attention.forward`, which views the projection as
/// `(…, heads, 2 · head_dim)` and chunks the LAST axis.
///
/// The contiguous reading — all queries then all gates — has identical
/// dimensions and is wrong, so nothing downstream can detect the mistake
/// from shapes. `mutation_table` in `tests::output_gate_parity` carries
/// `ContiguousHalves` as the falsifier for exactly this.
pub fn gather_fused_half(
    full: &[f32],
    num_heads: usize,
    head_dim: usize,
    half: FusedHalf,
) -> Vec<f32> {
    let stride = head_dim * 2;
    let offset = match half {
        FusedHalf::Query => 0,
        FusedHalf::Gate => head_dim,
    };
    let mut out = Vec::with_capacity(num_heads * head_dim);
    for head in 0..num_heads {
        let start = head * stride + offset;
        out.extend_from_slice(&full[start..start + head_dim]);
    }
    out
}

/// A deliberate defect in the fused query/gate path, for QW-3.5C's
/// mutation table.
///
/// Lives in production code and is threaded through the SHIPPED
/// implementation — the public entry points always pass
/// [`GateMutation::None`] — following the same contract as the Gated
/// DeltaNet `Mutation`. A table that mutated a copy of the operator would
/// only prove the copy was detectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateMutation {
    /// The shipped semantics.
    None,
    /// Read the projection as `[all queries | all gates]` instead of
    /// per-head `[q_h0 | gate_h0 | …]`. **The headline falsifier**: every
    /// dimension stays valid, closure still passes, and only the values
    /// move.
    ContiguousHalves,
    /// Correct per-head stride, halves exchanged.
    SwapPerHeadQGate,
    /// `silu(g)` rather than `sigmoid(g)` — the reading the config's
    /// `output_gate_type: "swish"` would imply if it owned this gate.
    SiluGate,
    /// Skip the gate entirely.
    NoGate,
    /// Send the gate slice through the query norm, which it must not see.
    GateGetsQNorm,
    /// Rotate the gate slice, which it must not see either.
    GateGetsRoPe,
    /// Gate after the output projection instead of before it.
    GateAfterOProj,
}

impl GateMutation {
    /// Which fused half a given logical role reads under this mutation.
    pub fn half_for(self, role: FusedHalf) -> FusedHalf {
        match (self, role) {
            (Self::SwapPerHeadQGate, FusedHalf::Query) => FusedHalf::Gate,
            (Self::SwapPerHeadQGate, FusedHalf::Gate) => FusedHalf::Query,
            (_, role) => role,
        }
    }

    /// Whether the halves are read as contiguous blocks rather than
    /// per-head interleaved.
    pub fn contiguous(self) -> bool {
        matches!(self, Self::ContiguousHalves)
    }
}

/// [`gather_fused_half`] under a mutation.
///
/// The contiguous reading is what a plausible-but-wrong implementation
/// does: `full[..q_rows]` as the queries and `full[q_rows..]` as the
/// gates. Same lengths, same downstream shapes, different tensors.
pub fn gather_fused_half_mutated(
    full: &[f32],
    num_heads: usize,
    head_dim: usize,
    half: FusedHalf,
    mutation: GateMutation,
) -> Vec<f32> {
    let half = mutation.half_for(half);
    if mutation.contiguous() {
        let rows = num_heads * head_dim;
        let start = match half {
            FusedHalf::Query => 0,
            FusedHalf::Gate => rows,
        };
        return full[start..start + rows].to_vec();
    }
    gather_fused_half(full, num_heads, head_dim, half)
}

#[cfg(test)]
mod llama3_tests {
    use super::llama3_frequencies;

    fn scaling() -> larql_models::Llama3RopeScaling {
        // Llama-3.2-1B's own block.
        larql_models::Llama3RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192.0,
        }
    }

    /// The reference transcription and the production planner are two
    /// independent implementations of `_compute_llama3_parameters`, and
    /// the reference is only worth having if it can disagree. It must
    /// agree here.
    #[test]
    fn the_reference_transcription_matches_the_production_planner() {
        const HEAD_DIM: usize = 64;
        const THETA: f64 = 500000.0;
        let reference = llama3_frequencies(&scaling(), HEAD_DIM, THETA);
        let base: Vec<f64> = (0..HEAD_DIM / 2)
            .map(|i| THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64))
            .collect();
        let production = larql_compute::attention::rope::apply_llama3_inv_freq(&scaling(), &base);
        assert_eq!(reference.len(), production.len());
        for (i, (r, p)) in reference.iter().zip(&production).enumerate() {
            assert!(
                (r - p).abs() <= 1e-12 * p.abs().max(1.0),
                "dim {i}: reference {r} vs production {p}"
            );
        }
    }

    /// The control. Without it the parity test above would pass for an
    /// implementation that ignored the block entirely and returned the
    /// unscaled frequencies — which is exactly the behaviour the whole
    /// variant exists to stop.
    #[test]
    fn scaling_actually_moves_the_low_frequencies() {
        const HEAD_DIM: usize = 64;
        const THETA: f64 = 500000.0;
        let scaled = llama3_frequencies(&scaling(), HEAD_DIM, THETA);
        let plain: Vec<f64> = (0..HEAD_DIM / 2)
            .map(|i| THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64))
            .collect();
        // Fast-rotating dimensions are left alone...
        assert_eq!(scaled[0], plain[0], "the fastest band must be untouched");
        // ...and the slowest are divided by the factor. If these matched,
        // the model would be served at its pre-training context length.
        let last = scaled.len() - 1;
        assert!(
            (scaled[last] - plain[last] / 32.0).abs() < 1e-12,
            "slowest band: {} vs {}",
            scaled[last],
            plain[last] / 32.0
        );
        assert_ne!(scaled, plain);
    }
}
