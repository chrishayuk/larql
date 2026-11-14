//! PREC-DIAG-1 E0b: the RoPE cos/sin construction as a per-run choice —
//! a diagnostic intervention, off by default.
//!
//! LARQL's shipped construction ([`RopeAngles::Larql`], `kernels::rope_rotate`
//! and `kernels::rope_rotate_scaled`) computes every angle in f64:
//!
//! ```text
//! inv_freq[i] = theta.powf(-(2i) / d)        f64 pow, negative exponent
//!                                            (linear: then / factor, f64)
//! angle       = (position as f64) * inv_freq f64 multiply
//! (sin, cos)  = (angle.sin() as f32, angle.cos() as f32)
//! ```
//!
//! [`RopeAngles::HfF32`] replaces it with transformers 5.5.0's
//! construction (`modeling_rope_utils._compute_default_rope_parameters`,
//! `_compute_linear_scaling_rope_parameters`, `Gemma3RotaryEmbedding.forward`),
//! operation by operation in f32:
//!
//! ```text
//! exponent    = f32(2i) / f32(d)             f32 divide (exact for d a power of 2)
//! inv_freq[i] = 1.0f32 / f32(theta).powf(exponent)
//!                                            f32 pow, then f32 reciprocal
//! linear:       inv_freq[i] /= f32(factor)   f32 divide, in place (HF `inv_freq /= factor`)
//! angle       = inv_freq[i] * f32(position)  one f32 multiply (HF's K=1 matmul)
//! (cos, sin)  = (angle.cos(), angle.sin())   f32 trig — Rust/libm, NOT torch's
//!                                            vectorised kernels (see below)
//! amplitude   = 1.0 (HF `attention_scaling`), not applied
//! ```
//!
//! So it changes more than a type: the frequency FORMULA (`1/base^x` vs
//! `base^-x`), the DTYPE of every step, and the TRIG LIBRARY. A result
//! from this mode is "LARQL with HF's RoPE construction", not "LARQL at
//! f32 precision"; attributing an effect to one of those three needs a
//! further intervention.
//!
//! What it does not change: the rotation itself (`x0·cos − x1·sin`,
//! `x0·sin + x1·cos` in f32, the rotate-half pairing), which runs exactly
//! as the shipped kernels run it, and every other operation.
//!
//! The trig library is the residual this mode cannot close from Rust:
//! f32 `cos`/`sin`/`powf` here are the platform libm's, and torch's CPU
//! kernels may round differently. [`hf_f32_inv_freq`] and
//! [`hf_f32_cos_sin`] are public so a table comparison (PREC-DIAG-1 E1b)
//! can measure that residual bit for bit rather than assume it away.
//!
//! Scope: plain RoPE ([`PositionPolicy::Rope`]) and linear scaling
//! ([`PositionPolicy::Linear`]), the two policies Gemma 3 uses. Every other
//! rotary policy is refused in this mode, never approximated.

use crate::error::VindexError;
use larql_models::config::PositionPolicy;

/// How RoPE cos/sin are constructed for one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeAngles {
    /// LARQL's shipped construction: f64 frequencies, angles and trig,
    /// rounded to f32. The default; selecting it is bit-identical to
    /// leaving the choice unmade.
    #[default]
    Larql,
    /// transformers' construction in f32 (see the module docs).
    HfF32,
}

impl RopeAngles {
    /// The name the CLI and the run records use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Larql => "larql",
            Self::HfF32 => "hf-f32",
        }
    }
}

/// transformers' f32 inverse frequencies for a `head_dim`-wide rotary at
/// `theta`, divided by `factor` when the policy is linear.
pub fn hf_f32_inv_freq(head_dim: usize, theta: f64, factor: Option<f64>) -> Vec<f32> {
    let d = head_dim as f32;
    let base = theta as f32;
    (0..head_dim / 2)
        .map(|i| {
            let exponent = (2 * i) as f32 / d;
            let mut inv_freq = 1.0f32 / base.powf(exponent);
            if let Some(factor) = factor {
                inv_freq /= factor as f32;
            }
            inv_freq
        })
        .collect()
}

/// transformers' f32 `(cos, sin)` for one frequency at one position.
pub fn hf_f32_cos_sin(position: usize, inv_freq: f32) -> (f32, f32) {
    let angle = inv_freq * position as f32;
    (angle.cos(), angle.sin())
}

/// Rotate one head with `inv_freq` (one entry per rotate-half pair) and
/// transformers' f32 angles. The rotation arithmetic is the shipped
/// kernels' (`kernels::rope_rotate`), unchanged.
pub fn rope_rotate_hf_f32(head: &mut [f32], position: usize, inv_freq: &[f32]) {
    let half = head.len() / 2;
    debug_assert_eq!(inv_freq.len(), half);
    for i in 0..half {
        let (cos_t, sin_t) = hf_f32_cos_sin(position, inv_freq[i]);
        let x0 = head[i];
        let x1 = head[half + i];
        head[i] = x0 * cos_t - x1 * sin_t;
        head[half + i] = x0 * sin_t + x1 * cos_t;
    }
}

/// The frequencies for `policy` under [`RopeAngles::HfF32`]; `None` for a
/// policy with no rotation; a refusal for any rotary policy this mode
/// does not cover.
pub fn hf_f32_policy_inv_freq(
    policy: PositionPolicy,
    head_dim: usize,
) -> Result<Option<Vec<f32>>, VindexError> {
    match policy {
        PositionPolicy::Rope { theta } => Ok(Some(hf_f32_inv_freq(head_dim, theta, None))),
        PositionPolicy::Linear { theta, factor } => {
            Ok(Some(hf_f32_inv_freq(head_dim, theta, Some(factor))))
        }
        PositionPolicy::None => Ok(None),
        other => Err(VindexError::Parse(format!(
            "--rope-angles hf-f32 covers plain and linear RoPE only; a layer with position \
             policy {other:?} is refused, not approximated"
        ))),
    }
}

/// Apply `angles` to one position's Q and K where the backend is about
/// to rotate them, and return the policy the backend's own rotation
/// should then apply: `policy` unchanged under [`RopeAngles::Larql`] (the
/// shipped path runs exactly as before), [`PositionPolicy::None`] under
/// [`RopeAngles::HfF32`] once Q and K are rotated here.
pub fn rotate_or_defer(
    angles: RopeAngles,
    policy: PositionPolicy,
    head_dim: usize,
    position: usize,
    q: &mut [f32],
    k: &mut [f32],
) -> Result<PositionPolicy, VindexError> {
    match angles {
        RopeAngles::Larql => Ok(policy),
        RopeAngles::HfF32 => {
            if let Some(inv_freq) = hf_f32_policy_inv_freq(policy, head_dim)? {
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_hf_f32(head, position, &inv_freq);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_hf_f32(head, position, &inv_freq);
                }
            }
            Ok(PositionPolicy::None)
        }
    }
}
