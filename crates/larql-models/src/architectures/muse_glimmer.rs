//! Muse-Glimmer target architecture (`muse_glimmer` / `muse_glimmer_text`).
//!
//! Every topology and scalar fact resolves through the config-driven
//! trait defaults — the checkpoint declares them and the generic parser
//! already consumes them (hybrid `layer_types`, per-layer
//! `layer_rope_theta` with NoPE zeros, `qk_scale_factor`,
//! `output_multiplier`, `post_norm_eps`). What this family adds is only
//! what no config key or tensor can state: the **judged execution
//! semantics** read from the upstream reference implementation.
//!
//! - Attention output gate: `sigmoid(gate_proj(attention_input))`
//!   multiplied into the aggregated head output before `o_proj`. The
//!   gate projection is `Linear(hidden → q_heads × head_dim)`, matching
//!   the `[q_heads × head_dim, hidden]` operand the stack ships.
//! - Parameter-free QK norm: Q and K are RMS-normalised with **no
//!   learned weights** (the stack ships no `q_norm`/`k_norm` tensors),
//!   with the declared `qk_scale_factor` applied to normalised Q — the
//!   canonical `head_dim^-0.5` stays a separate score-time multiply,
//!   which is why resolution records query and score scales separately.
//!
//! The assistant (`muse_glimmer_assistant`) is deliberately **not** this
//! family: its layers carry weighted QK norms and no gate, and nobody
//! has judged assistant-specific semantics — it stays on the generic
//! path, which its operand estate closes under.

use crate::config::{
    AttentionGateSpec, EmbeddingNorm, GateActivation, GateCombine, GatePlacement, GateSource,
    ModelArchitecture, ModelConfig, NormSpec, ParameterFreeQkNorm,
};

/// Offset of a *centred* norm: weights are stored around zero and the
/// runtime gain is `1 + w`.
const CENTRED_NORM_OFFSET: f32 = 1.0;
/// Offset of an ordinary norm: the stored weight *is* the gain.
const ABSOLUTE_NORM_OFFSET: f32 = 0.0;

pub struct MuseGlimmerArch {
    config: ModelConfig,
}

impl MuseGlimmerArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for MuseGlimmerArch {
    fn family(&self) -> &str {
        "muse_glimmer"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn attention_output_gate(&self) -> Option<AttentionGateSpec> {
        Some(AttentionGateSpec {
            source: GateSource::AttentionInput,
            activation: GateActivation::Sigmoid,
            combine: GateCombine::ElementwiseMultiply,
            placement: GatePlacement::AfterAggregationBeforeOutputProjection,
        })
    }

    fn parameter_free_qk_norm(&self) -> ParameterFreeQkNorm {
        ParameterFreeQkNorm {
            q: true,
            k: true,
            v: false,
        }
    }

    /// Layer norms are *centred*: `RMSNorm(x) * (1.0 + weight)`.
    ///
    /// All four decoder-layer norms are
    /// `MuseGlimmerTextCenteredRMSNorm`, whose weights the checkpoint
    /// stores centred on zero — layer 0's `input_layernorm` has mean
    /// +0.34 and a minimum of exactly **-1.0000**, which under `1 + w`
    /// cleanly zeroes a channel and under `w` alone would flip its sign.
    fn norm_weight_offset(&self) -> f32 {
        CENTRED_NORM_OFFSET
    }

    /// The final norm is **not** centred.
    ///
    /// `MuseGlimmerTextModel.norm` is a plain `MuseGlimmerRMSNorm`
    /// (`normed * weight`), not the centred variant the layers use —
    /// two different classes upstream, and `norm.weight` is stored
    /// absolute (mean 0.017 spanning ±4.9) to match. Declared here
    /// because a single model-scope offset would fix the four layer
    /// norms and silently break this one.
    fn final_norm_spec(&self) -> NormSpec {
        NormSpec {
            kind: self.norm_type(),
            eps: self.norm_eps() as f64,
            weight_offset: ABSOLUTE_NORM_OFFSET,
        }
    }

    /// Embedding rows are RMS-normalised **weightlessly** on lookup.
    ///
    /// `MuseGlimmerTextNormedEmbedding.forward` is
    /// `embed_norm(super().forward(ids))` with
    /// `MuseGlimmerRMSNorm(eps=rms_norm_eps, with_scale=False)`. Upstream
    /// notes it cannot be folded into the embedding matrix because the
    /// DFlash draft path needs to embed *without* it — so it is a real
    /// operation on the target's program, not a storage convention.
    ///
    /// No tensor evidences this: the norm is weightless. It was found by
    /// diffing the upstream trace, where plane 000 differed from an
    /// unnormalised lookup by a pure per-row rescale (RMS 1/16 → 1).
    ///
    /// The epsilon is resolved here, from this checkpoint's own
    /// `rms_norm_eps`, so the operation site carries a concrete value
    /// rather than inheriting one at execution time.
    fn embedding_norm(&self) -> Option<EmbeddingNorm> {
        Some(EmbeddingNorm {
            kind: self.norm_type(),
            eps: self.norm_eps() as f64,
        })
    }
}
