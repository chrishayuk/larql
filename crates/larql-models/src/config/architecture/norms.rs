//! The [`Norms`] slice of [`super::ModelArchitecture`].

use super::TensorKeys;
use crate::config::{EmbeddingNorm, NormSpec, NormType, PostNormEps, QkNormScope};

/// Normalisation: kind, weight offsets, epsilons, placement and biases.
pub trait Norms: TensorKeys {
    /// The vector the QK-norm RMS statistic reduces over.
    ///
    /// Defaults to [`QkNormScope::PerHead`], which is what every architecture
    /// in the support table that carries QK norm uses except OLMoE. An
    /// architecture returning QK-norm keys must be sure this matches its
    /// reference module; the two conventions share tensor names and, for MHA
    /// models, tensor shapes.
    fn qk_norm_scope(&self) -> QkNormScope {
        QkNormScope::PerHead
    }

    /// The bias key paired with a norm *weight* key, if this architecture's
    /// norm has a bias at all.
    ///
    /// Derived from the weight key rather than written out per architecture,
    /// so an architecture that overrides its norm naming gets the matching
    /// bias for free and the two cannot drift apart. Gated on
    /// [`NormType::LayerNorm`] so RMSNorm families answer `None` rather than
    /// claiming a tensor they do not have.
    ///
    /// Uses `strip_suffix` rather than a `replace(".weight", ".bias")`:
    /// `replace` rewrites *every* occurrence, so a key that contained the
    /// substring earlier in its path would be silently mangled.
    /// Whether every weighted norm site carries an additive bias
    /// (`nn.LayerNorm`'s affine bias). `None` = unjudged, which is not "no
    /// bias": a bias operand shipped under `None` or `Some(false)` fails
    /// VINDEX3 operand closure, and `Some(true)` requires one beside every
    /// norm weight. No config key states it, so a family that has one
    /// says so.
    fn norm_bias(&self) -> Option<bool> {
        None
    }

    fn norm_bias_of(&self, weight_key: &str) -> Option<String> {
        if self.norm_type() != NormType::LayerNorm {
            return None;
        }
        weight_key
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.bias"))
    }

    /// Norm type (RMSNorm vs LayerNorm).
    fn norm_type(&self) -> NormType {
        NormType::RmsNorm
    }

    /// Weight offset added during layer normalization.
    /// Default 0.0 — saved weights are the final multiplier.
    fn norm_weight_offset(&self) -> f32 {
        0.0
    }

    /// Weight offset added during QK normalization (per-head Q/K norms).
    /// Gemma 2/3: 1.0 (weight = 1 + learned_weight at runtime), Gemma 4 and others: 0.0.
    fn qk_norm_weight_offset(&self) -> f32 {
        0.0
    }

    /// Parameter-free QK normalisation (RMS over Q/K with no learned
    /// weights). Default: none — families that normalise without weights
    /// declare it, because no tensor evidence can reveal a weightless op.
    fn parameter_free_qk_norm(&self) -> crate::config::ParameterFreeQkNorm {
        crate::config::ParameterFreeQkNorm::default()
    }

    /// Whether this model has separate pre/post norms around attention and FFN
    /// (Gemma 2/3 style with 4 norms per layer) vs standard pre-norm only.
    fn has_post_norms(&self) -> bool {
        false
    }

    /// Whether to apply parameter-free RMSNorm to V states before attention.
    /// Gemma 4 applies V-norm (normalization only, no learned scale).
    /// Default: false.
    fn has_v_norm(&self) -> bool {
        false
    }

    /// The complete normalisation applied at the *pre* sites — before
    /// attention and before the FFN.
    ///
    /// The default composes this family's model-scope answers, which is
    /// correct for the many architectures whose norm sites really are
    /// uniform. A family whose sites differ overrides the site that
    /// differs, rather than every consumer learning the exception.
    fn pre_norm_spec(&self) -> NormSpec {
        NormSpec {
            kind: self.norm_type(),
            eps: self.norm_eps() as f64,
            weight_offset: self.norm_weight_offset(),
        }
    }

    /// The complete normalisation applied at the *post* sites, when this
    /// family has judged one. `None` = unjudged; a four-norm stack in
    /// that state is refused rather than inheriting the pre-norm answer.
    fn post_norm_spec(&self) -> Option<NormSpec> {
        self.post_norm_eps().map(|eps| NormSpec {
            kind: self.norm_type(),
            eps: eps.resolve(self.norm_eps() as f64),
            weight_offset: self.norm_weight_offset(),
        })
    }

    /// The complete normalisation applied after the last block, before
    /// the head. Defaults to the pre-norm spec — true wherever a family
    /// uses one norm everywhere, and overridden where it does not.
    fn final_norm_spec(&self) -> NormSpec {
        self.pre_norm_spec()
    }

    /// The normalisation applied to embedding-table output, when this
    /// family judges one.
    ///
    /// `None` = no such operation. Weightless, so no tensor can evidence
    /// it and no gate below can infer it — a family that has one must say
    /// so, exactly like [`attention_output_gate`](Self::attention_output_gate).
    fn embedding_norm(&self) -> Option<EmbeddingNorm> {
        None
    }

    /// Which epsilon this model's post-norms use, when that has been
    /// established.
    ///
    /// `None` = unjudged: nothing has established what the post-norms use.
    /// A four-norm stack in that state is not executable, and closure
    /// refuses it rather than inheriting `norm_eps` — the two can differ by
    /// orders of magnitude (1e-8 vs 1e-5 on the same checkpoint), which is
    /// the OLMoE eps failure shape on a second axis.
    ///
    /// A checkpoint that declares `post_norm_eps` answers with
    /// [`PostNormEps::Value`]. Families whose post-norms genuinely share
    /// the pre-norm epsilon override this to say
    /// [`PostNormEps::Shared`] explicitly, because "they share" is a
    /// judgment the family makes, not a default the absence implies.
    fn post_norm_eps(&self) -> Option<PostNormEps> {
        self.config().post_norm_eps.map(PostNormEps::Value)
    }

    /// Norm epsilon for RMSNorm / LayerNorm. Reads `config.norm_eps` (parsed
    /// from `rms_norm_eps` / `layer_norm_eps` / `layer_norm_epsilon` /
    /// `norm_epsilon` in config.json by `detect::parser`) and falls back to
    /// [`Self::default_norm_eps`] only when the model's config does not
    /// specify one.
    fn norm_eps(&self) -> f32 {
        self.config()
            .norm_eps
            .map(|v| v as f32)
            .unwrap_or_else(|| self.default_norm_eps())
    }

    /// Epsilon to use when the checkpoint's config **omits** the norm-eps
    /// field — a *per-family* fact, not a crate-wide constant.
    ///
    /// `transformers`' config classes disagree, and a checkpoint that omits
    /// the field is served by whatever its own class defaults to:
    ///
    /// | default | families |
    /// |---|---|
    /// | 1e-6 | Llama, Mistral, Qwen3 (+ MoE), Gemma 2/3, GraniteMoE |
    /// | 1e-5 | **OLMoE, GPT-OSS, StarCoder2, Phi-3** |
    ///
    /// This was measured, not assumed. `allenai/OLMoE-1B-7B-0924-Instruct`
    /// ships no `rms_norm_eps`, so a single crate-wide 1e-6 fallback ran its
    /// every norm an order of magnitude tight: the Gate-B layer diff put the
    /// final residual at cosine **0.890** against the reference, and moving
    /// only this constant took it to **0.991** (`docs/k3-funnel.md` §4.8).
    ///
    /// The failure shape is [§4.7.8](../../../docs/k3-funnel.md)'s a third
    /// time — a config fact with one behavioural default that silently
    /// answers for every architecture. It is kept as a default here because
    /// the majority value is genuinely 1e-6 and a required method would make
    /// every synthetic test config declare one; the guard against silent
    /// inheritance is instead the pin test over the whole support table in
    /// `tests/test_architectures.rs`, which fails when a new family arrives
    /// undeclared.
    fn default_norm_eps(&self) -> f32 {
        crate::defaults::DEFAULT_NORM_EPS
    }

    //
    // LayerNorm is `γ·x̂ + β`; RMSNorm has no β. These default to `None` so
    // every RMSNorm architecture is unaffected, and only the `NormType::
    // LayerNorm` families (GPT-2, StarCoder2) override them.
    //
    // Until these existed, no accessor anywhere named a norm bias, so
    // extraction never wrote one and `build_pipeline_layers` hardcoded
    // `input_norm_bias: None` — while the Metal `layer_norm` shader
    // implemented `+ bias` and its no-bias variant was always the one
    // selected. The CPU dense path got away with it by mangling the weight
    // key (`".weight"` → `".bias"`), which is why raw-safetensors inference
    // was right and every vindex-backed path silently dropped the shift.
    // Same shape as the attention-sinks gap in `docs/k3-funnel.md` §4.6.

    /// Bias for the input LayerNorm. `None` for RMSNorm architectures.
    fn input_layernorm_bias_key(&self, layer: usize) -> Option<String> {
        self.norm_bias_of(&self.input_layernorm_key(layer))
    }

    /// Bias for the post-attention LayerNorm. `None` for RMSNorm.
    fn post_attention_layernorm_bias_key(&self, layer: usize) -> Option<String> {
        self.norm_bias_of(&self.post_attention_layernorm_key(layer))
    }

    /// Bias for the final LayerNorm. `None` for RMSNorm.
    fn final_norm_bias_key(&self) -> Option<String> {
        self.norm_bias_of(self.final_norm_key())
    }

    /// Per-layer norms applied inside a sub-layer, before its output
    /// projection, that the standard norm keys do not name. Empty for
    /// families that have none; BitNet b1.58 has one per branch.
    fn sub_norm_keys(&self, layer: usize) -> Vec<String> {
        let _ = layer;
        Vec::new()
    }
}
