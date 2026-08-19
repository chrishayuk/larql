//! Rotary Position Embeddings (RoPE) — position-dependent rotation of Q/K vectors.
//!
//! Split-half pairing: rotates (x[i], x[i + half_dim]) pairs.
//! Matches HuggingFace default and MLX traditional=False.
//!
//! Frequency-scaling families live one per module ([`llama3`], [`yarn`]) and
//! are selected by [`RopeFreqScaling`].

use ndarray::Array2;

pub mod llama3;
pub mod yarn;

pub use larql_models::YarnRopeScaling;
pub use llama3::{apply_llama3_inv_freq, Llama3RopeScaling};
pub use yarn::UNIT_AMPLITUDE;

/// Which frequency-scaling family a model's `rope_scaling` block selects.
///
/// **This is an enum rather than a bag of `Option`s on purpose.** It replaced
/// an `Option<Llama3RopeScaling>` parameter that could not express YaRN, so
/// `openai/gpt-oss-20b` was served with plain RoPE — 23 of its 32 rotary
/// dimensions at the wrong frequency and every cos/sin 34.7 % too small,
/// which the Gate-B layer diff caught as cosine 0.978 at layer 0
/// (`docs/k3-funnel.md` §4.9). A `match` over this type is exhaustive, so a
/// future family cannot be added and then silently ignored at a call site;
/// [§4.7.8]'s lesson was that only a type or a test protects against that,
/// never a comment.
///
/// [§4.7.8]: ../../../../docs/k3-funnel.md
#[derive(Debug, Clone, Copy, Default)]
pub enum RopeFreqScaling {
    /// Standard `1 / base^(2i/d)` frequencies, unit amplitude.
    #[default]
    None,
    /// Llama-3 wavelength-band adjustment. Frequencies only.
    Llama3(Llama3RopeScaling),
    /// YaRN ramp **and** attention amplitude. See [`yarn`].
    Yarn(YarnRopeScaling),
}

impl RopeFreqScaling {
    /// Resolve to `(inv_freq, amplitude)` for one head's rotary width.
    ///
    /// `amplitude` multiplies both `cos` and `sin`; every family except YaRN
    /// leaves it at [`UNIT_AMPLITUDE`].
    fn resolve(self, base_inv_freq: Vec<f64>, rotary_dim: usize, base: f64) -> (Vec<f64>, f64) {
        match self {
            RopeFreqScaling::None => (base_inv_freq, UNIT_AMPLITUDE),
            RopeFreqScaling::Llama3(s) => {
                (apply_llama3_inv_freq(&s, &base_inv_freq), UNIT_AMPLITUDE)
            }
            RopeFreqScaling::Yarn(s) => yarn::yarn_inv_freq(&s, &base_inv_freq, rotary_dim, base),
        }
    }
}

/// Everything position-independent about a layer's RoPE, precomputed.
///
/// **This exists so a kernel does not have to know about scaling families.**
/// Given `inv_freq` and `amplitude`, applying RoPE is
/// `angle = pos * inv_freq[d]; cos(angle) * amplitude`, whatever the model —
/// base, llama3 bands, YaRN's ramp, and a linear position divisor all fold
/// into these two values. The GPU kernels take them as a buffer instead of
/// recomputing `1/base^(2d/rdim)` in-shader, which is what let Metal decode
/// silently ignore *every* scaling family while Metal prefill (which routes
/// through [`apply_rope_partial_at_full`]) honoured them all. See
/// `docs/k3-funnel.md` §4.10.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeFreqPlan {
    /// Effective inverse frequency per rotary pair, length `rotary_dim / 2`.
    /// Carries the family scaling *and* the position divisor.
    ///
    /// Kept in `f64`: this is the reference tier's own series, and narrowing
    /// it for the GPU's benefit would cost precision on the path Gate B
    /// measures. [`Self::inv_freq_f32`] does the narrowing at the boundary
    /// that actually needs it.
    pub inv_freq: Vec<f64>,
    /// Scalar on `cos`/`sin`. [`UNIT_AMPLITUDE`] for every family but YaRN.
    pub amplitude: f64,
}

impl RopeFreqPlan {
    /// The frequency table as a GPU buffer payload.
    pub fn inv_freq_f32(&self) -> Vec<f32> {
        self.inv_freq.iter().map(|f| *f as f32).collect()
    }

    /// Plain `1/base^(2d/rdim)` with unit amplitude — the correct plan for an
    /// architecture whose config declares no `rope_scaling`.
    ///
    /// Spelled out rather than reached by a `None` default: a layer that
    /// *should* be scaled and silently is not is the defect this whole type
    /// exists to prevent, so "unscaled" has to be something a caller says.
    /// `rotary_dim == 0` means the full `head_dim`, matching
    /// [`super::super::pipeline::FullPipelineLayer::rotary_dim`].
    pub fn unscaled(head_dim: usize, rotary_dim: usize, rope_base: f64) -> Self {
        let fraction = if rotary_dim == 0 || rotary_dim >= head_dim {
            1.0
        } else {
            rotary_dim as f64 / head_dim as f64
        };
        rope_freq_plan(head_dim, fraction, rope_base, 1.0, RopeFreqScaling::None)
    }
}

/// HF's `proportional` partial rotary (`_compute_proportional_rope_parameters`):
/// a head-sized plan whose first `rotary_fraction · head_dim / 2` pairs
/// carry `1 / base^(2i/head_dim)` — the inverse frequencies are taken over
/// the FULL head width — and whose remaining pairs are at zero frequency,
/// an identity rotation. Unit amplitude. Gemma 4 declares it on its
/// full-attention layers over `global_head_dim`.
///
/// Distinct from [`rope_freq_plan`] at the same fraction, which is the
/// plain partial rotary: a `rotary_dim`-sized plan whose pairs are
/// `1 / base^(2i/rotary_dim)` and which is applied to the head's prefix
/// as its own rotate-half block. Same rotated dims count; different
/// pairing and different angles (`base^(2i/512)` vs `base^(2i/128)` on
/// Gemma 4's global layers) — so a family must be served with the one it
/// declares, never the other.
pub fn rope_freq_plan_proportional(
    head_dim: usize,
    rotary_fraction: f64,
    base: f64,
) -> RopeFreqPlan {
    let half = head_dim / 2;
    let rotated_pairs = ((rotary_fraction * head_dim as f64) as usize) / 2;
    RopeFreqPlan {
        inv_freq: (0..half)
            .map(|i| {
                if i < rotated_pairs {
                    1.0 / base.powf(2.0 * i as f64 / head_dim as f64)
                } else {
                    0.0
                }
            })
            .collect(),
        amplitude: UNIT_AMPLITUDE,
    }
}

/// Rotary width actually used for a head, matching
/// [`apply_rope_partial_at_full`]'s own rounding. One definition, because a
/// host and a kernel disagreeing by one dimension is invisible until it is a
/// parity failure.
pub fn rotary_width(head_dim: usize, fraction: f64) -> usize {
    ((head_dim as f64 * fraction) as usize).max(2)
}

/// Standard `1 / base^(2i/rotary_dim)` series, length `rotary_dim / 2`.
fn base_inv_freq(rotary_dim: usize, rope_base: f64) -> Vec<f64> {
    (0..rotary_dim / 2)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / rotary_dim as f64))
        .collect()
}

/// Precompute a layer's [`RopeFreqPlan`].
///
/// `position_divisor` is folded into `inv_freq` rather than kept separate:
/// `angle = (pos / divisor) * inv_freq` is exactly `pos * (inv_freq /
/// divisor)`, so a consumer needs one multiply and cannot forget the divisor
/// the way Metal decode forgot it.
pub fn rope_freq_plan(
    head_dim: usize,
    fraction: f64,
    rope_base: f64,
    position_divisor: f64,
    scaling: RopeFreqScaling,
) -> RopeFreqPlan {
    let rotary_dim = rotary_width(head_dim, fraction);
    let (inv_freq, amplitude) =
        scaling.resolve(base_inv_freq(rotary_dim, rope_base), rotary_dim, rope_base);
    let divisor = if position_divisor > 0.0 {
        position_divisor
    } else {
        1.0
    };
    RopeFreqPlan {
        inv_freq: inv_freq.iter().map(|f| f / divisor).collect(),
        amplitude,
    }
}

/// Apply full RoPE to Q or K vectors.
/// x: (seq_len, num_heads * head_dim)
pub fn apply_rope(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
) -> Array2<f32> {
    apply_rope_partial(x, num_heads, head_dim, rope_base, 1.0)
}

/// Apply RoPE with partial rotation: only the first `fraction` of each head's
/// dimensions get rotary encoding. The rest pass through unchanged.
/// fraction = 1.0 means full rotation (standard RoPE).
pub fn apply_rope_partial(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
) -> Array2<f32> {
    apply_rope_partial_at(x, num_heads, head_dim, rope_base, fraction, 0)
}

/// Apply RoPE with a positional offset — row `i` in `x` is treated as
/// token position `position_offset + i`. Use this during KV-cached
/// decode: cached K already carries RoPE for positions 0..N-1, and
/// the new token needs RoPE at position N.
pub fn apply_rope_partial_at(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
    position_offset: usize,
) -> Array2<f32> {
    apply_rope_partial_at_scaled(
        x,
        num_heads,
        head_dim,
        rope_base,
        fraction,
        position_offset,
        1.0,
    )
}

/// Same as [`apply_rope_partial_at`] but divides the position by
/// `position_divisor` before computing phase. Matches HF
/// `rope_scaling = {rope_type: linear, factor: N}` applied to a specific
/// layer-type (Gemma 3 applies it only on global / full-attention layers,
/// not on sliding-attention layers — see `Gemma3TextConfig.rope_scaling`).
pub fn apply_rope_partial_at_scaled(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
    position_offset: usize,
    position_divisor: f64,
) -> Array2<f32> {
    apply_rope_partial_at_full(
        x,
        num_heads,
        head_dim,
        rope_base,
        fraction,
        position_offset,
        position_divisor,
        RopeFreqScaling::None,
    )
}

/// Most general RoPE entry point. `scaling` selects the frequency family and,
/// for YaRN, an amplitude on `cos`/`sin`. `position_divisor` and `scaling`
/// compose independently — the divisor scales the position, `scaling` scales
/// the per-channel frequency (and possibly the amplitude).
#[allow(clippy::too_many_arguments)]
pub fn apply_rope_partial_at_full(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
    fraction: f64,
    position_offset: usize,
    position_divisor: f64,
    scaling: RopeFreqScaling,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    // Same plan the GPU kernels are handed, from the same function — the CPU
    // computing its frequencies a second way is how the two drift.
    let plan = rope_freq_plan(head_dim, fraction, rope_base, position_divisor, scaling);
    let half_rotary = plan.inv_freq.len();
    let amplitude = plan.amplitude;

    for row in 0..seq_len {
        let pos = (position_offset + row) as f64;
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half_rotary {
                let theta = pos * plan.inv_freq[i];
                // The amplitude rides on cos/sin rather than on the output, so
                // it applies to the rotated pair exactly as HF's
                // `cos = emb.cos() * attention_scaling` does.
                let cos_t = (theta.cos() * amplitude) as f32;
                let sin_t = (theta.sin() * amplitude) as f32;

                let x0 = x[[row, offset + i]];
                let x1 = x[[row, offset + half_rotary + i]];

                out[[row, offset + i]] = x0 * cos_t - x1 * sin_t;
                out[[row, offset + half_rotary + i]] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
