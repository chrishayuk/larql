//! The [`Attention`] slice of [`super::ModelArchitecture`].

use super::{score_scale_from_query_pre_attn_scalar, Position};
use crate::config::{layer_types, LayerKind};

/// The score multiplier of an attention that does not scale QK^T.
pub const UNSCALED_SCORES: f64 = 1.0;

/// Attention geometry and behaviour per layer: heads, windows, scale,
/// gates, sinks and sharing.
pub trait Attention: Position {
    /// Judged semantics of an attention output gate, when this family
    /// carries one. `None` means *no judgment exists* — never "no gate":
    /// a stack shipping a gate operand while this is `None` must fail
    /// operand closure downstream rather than run ungated.
    fn attention_output_gate(&self) -> Option<crate::config::AttentionGateSpec> {
        None
    }

    /// Judged semantics of this family's attention sinks, when it carries
    /// them. Derived from [`Self::attn_sinks_key`] rather than declared a
    /// second time: a family that names a sink tensor is served through
    /// `softmax_in_place(scores, sink)`, and this is that behaviour stated
    /// as a schema fact — the two cannot disagree. `None` means *no
    /// judgment exists*, never "no sinks": a stack shipping a
    /// `self_attn.sinks` operand while this is `None` fails operand
    /// closure downstream rather than run an ordinary softmax.
    fn attention_sinks(&self) -> Option<crate::config::AttentionSinkSpec> {
        self.attn_sinks_key(0)
            .map(|_| crate::config::AttentionSinkSpec::SoftmaxDenominator)
    }

    /// Whether this layer uses sliding window attention.
    ///
    /// Reads `config.layer_types` when the checkpoint declares one, so a model
    /// that states its interleave explicitly — `openai/gpt-oss-20b` alternates
    /// 12 sliding / 12 full at `sliding_window: 128` — is served correctly with
    /// no per-architecture code. Families that imply the interleave through a
    /// stride instead (Gemma 3) still override.
    ///
    /// See [`super::layer_types`] for why this default is not `false`.
    fn is_sliding_window_layer(&self, layer: usize) -> bool {
        // The enable flag first: a checkpoint that declares the feature
        // off means it, whatever its interleave says. Qwen2.5 ships
        // `sliding_window: 32768` AND a `use_sliding_window: false`, and
        // a family that also declared sliding `layer_types` would
        // otherwise have the window applied against its own instruction.
        if self.sliding_window_size().is_none() {
            return false;
        }
        // `max_window_layers` bounds which layers slide when the feature
        // is on: the bottom `n` use the window, the rest attend fully.
        if let Some(bound) = self.config().max_window_layers {
            if layer >= bound {
                return false;
            }
        }
        layer_types::is_sliding_from_layer_types(self.config().layer_types.as_ref(), layer)
            .unwrap_or(false)
    }

    /// The **effective** sliding window (None = full attention).
    ///
    /// One place resolves the three declarations a checkpoint can make
    /// about this feature, so nothing downstream can honour one while
    /// ignoring another:
    ///
    /// ```text
    /// sliding_window       the window, when there is one
    /// use_sliding_window   whether it applies at all
    /// max_window_layers    how far up the stack it applies
    /// ```
    ///
    /// An explicit `use_sliding_window: false` yields `None` even when a
    /// window is declared beside it. That is not a special case for Qwen
    /// — it is what the checkpoint says, and reading the size without the
    /// flag is how a declared-inactive feature becomes an active wrong
    /// answer. A family whose config states no flag is unaffected: `None`
    /// is not `Some(false)`.
    fn sliding_window_size(&self) -> Option<usize> {
        if self.config().use_sliding_window == Some(false) {
            return None;
        }
        self.config().sliding_window
    }

    /// The per-layer kind this family declares for EVERY layer of its
    /// stack, when the declaration is the `model_type` itself rather than
    /// an interleave key. A pure-SSM checkpoint (mamba2) writes no
    /// `layer_types` — there is no interleave to state — and its
    /// `model_type` is the whole-stack declaration that every layer runs
    /// the family's mixer.
    ///
    /// `None` (the default) means the family makes no such uniform claim
    /// and the declared interleave, or its absence, answers as before.
    /// This is a *declaration*, not an operator identification: which
    /// recurrence actually runs is still resolved from the declared
    /// geometry downstream, exactly as for an interleaved hybrid.
    fn declared_uniform_layer_kind(&self) -> Option<LayerKind> {
        None
    }

    /// Head dimension for a given layer. Models with different head dims for
    /// sliding vs global attention (e.g., Gemma 4) override this.
    /// Default: config.head_dim for all layers.
    fn head_dim_for_layer(&self, _layer: usize) -> usize {
        self.config().head_dim
    }

    /// Number of KV heads for a given layer. Models with different KV head counts
    /// for sliding vs global attention override this.
    /// Default: config.num_kv_heads for all layers.
    fn num_kv_heads_for_layer(&self, _layer: usize) -> usize {
        self.config().num_kv_heads
    }

    /// Number of Q heads for a given layer. Usually constant, but can vary
    /// when head_dim changes across layers (Q proj dim stays constant,
    /// but num_q_heads = q_proj_dim / head_dim).
    /// Default: config.num_q_heads for all layers.
    fn num_q_heads_for_layer(&self, _layer: usize) -> usize {
        self.config().num_q_heads
    }

    /// Whether value shares key projections at this layer (V = K).
    /// When true, the forward pass uses K in place of V.
    /// Default: false.
    fn v_shares_k(&self, _layer: usize) -> bool {
        false
    }

    /// Whether QK^T is scaled at all. `scale_attn_weights: false` (GPT-2's
    /// lineage) says it is not, and the multiplier is then exactly one; a
    /// silent config scales, which is what every class defining the key
    /// defaults to. A family whose reference never scales (GPT-Neo)
    /// overrides this.
    fn scales_attention_scores(&self) -> bool {
        self.config().scale_attn_weights != Some(false)
    }

    /// Attention scale: `attention_multiplier` when declared (Granite —
    /// replaces the standard formula outright, confirmed by every
    /// `arch.attention_multiplier()` call site on the legacy path; Granite
    /// 4.1's declared value is `1/head_dim`, not `1/sqrt(head_dim)`, so
    /// composing the two would be wrong by a factor of `sqrt(head_dim)`),
    /// else 1/sqrt(query_pre_attn_scalar) or 1/sqrt(head_dim).
    fn attention_scale(&self) -> f64 {
        if let Some(multiplier) = self.config().attention_multiplier {
            return multiplier;
        }
        if !self.scales_attention_scores() {
            return UNSCALED_SCORES;
        }
        let scalar = self
            .config()
            .query_pre_attn_scalar
            .unwrap_or(self.config().head_dim as f64);
        score_scale_from_query_pre_attn_scalar(scalar)
    }

    /// Attention scale for a specific layer. Accounts for per-layer head_dim
    /// when query_pre_attn_scalar is not set. Same `attention_multiplier`
    /// replacement as [`attention_scale`](Self::attention_scale) — no
    /// family in this registry varies `attention_multiplier` per layer, but
    /// the two must agree when it is set uniformly.
    fn attention_scale_for_layer(&self, layer: usize) -> f64 {
        if let Some(multiplier) = self.config().attention_multiplier {
            return multiplier;
        }
        if !self.scales_attention_scores() {
            return UNSCALED_SCORES;
        }
        if let Some(scalar) = self.config().query_pre_attn_scalar {
            score_scale_from_query_pre_attn_scalar(scalar)
        } else {
            score_scale_from_query_pre_attn_scalar(self.head_dim_for_layer(layer) as f64)
        }
    }

    /// Source layer for KV sharing. Returns Some(source_layer) if this layer
    /// should reuse K/V from an earlier layer instead of computing its own.
    /// Default: None (every layer computes its own K/V).
    fn kv_shared_source_layer(&self, _layer: usize) -> Option<usize> {
        None
    }

    /// Attention logit softcapping value (None = disabled).
    /// Applied before softmax: scores = tanh(scores / cap) * cap
    fn attn_logit_softcapping(&self) -> Option<f32> {
        self.config().attn_logit_softcapping.map(|v| v as f32)
    }

    /// Extra multiplier on attention scores on top of `1/sqrt(head_dim)`.
    /// Distinct from [`query_pre_attn_scalar`](Self::query_pre_attn_scalar),
    /// which replaces the denominator instead. `None` = no extra scale.
    ///
    /// Granite's `attention_multiplier` is **not** this operation despite
    /// the matching doc wording on [`ModelConfig`](super::ModelConfig) —
    /// every consumer of `arch.attention_multiplier()` on the legacy path
    /// (`larql-compute`'s attention kernels) uses it to *replace* the
    /// standard `1/sqrt(head_dim)` scale, not multiply on top of it
    /// (confirmed numerically: Granite 4.1's declared value is `1/head_dim`,
    /// not `1/sqrt(head_dim)`). [`attention_scale`](Self::attention_scale)
    /// is where that belongs.
    fn qk_scale_factor(&self) -> Option<f64> {
        self.config().qk_scale_factor
    }

    /// Whether attention projections carry biases, when the checkpoint
    /// says so. `None` = the config is silent; the loader's
    /// tensor-presence check answers.
    fn attention_bias(&self) -> Option<bool> {
        self.config().attention_bias
    }

    /// Whether the output projection — and only it — carries a bias
    /// (GPT-Neo: `q/k/v_proj` are `bias=False`, `out_proj` is biased). No
    /// config key states it; `None` = unjudged, and an output-bias
    /// operand shipped under it fails VINDEX3 operand closure unless
    /// `attention_bias` covers it.
    fn attention_output_bias(&self) -> Option<bool> {
        None
    }

    /// Whether the Q/K/V projections — not the output projection — carry
    /// biases. `None` = neither the checkpoint nor the family says; bias
    /// operands shipped under `None` fail operand closure.
    fn qkv_bias(&self) -> Option<bool> {
        self.config().qkv_bias
    }

    /// Attention score scaling factor (applied on top of 1/sqrt(head_dim)).
    fn attention_multiplier(&self) -> f32 {
        self.config()
            .attention_multiplier
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }
}
