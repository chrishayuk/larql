//! The [`ModelArchitecture`] trait — what a model *is*, with no compute.
//!
//! This file is large and deliberately not split: a Rust trait is a single
//! item, so its methods cannot live in separate modules. What *has* been moved
//! out is everything a default body needs in order to be short — selector
//! constants ([`super::rope_types`], [`super::layer_types`]), parameter structs
//! ([`super::rope`]) and the resolution rules that read them. A default here
//! should read as one config fact, one decision.
//!
//! **Default bodies that read the config are the point, not a shortcut.**
//! `docs/k3-funnel.md` §4.7.8 records the same bug three times over: a
//! behaviour that is a config fact, implemented on one architecture, with a
//! trait default silently answering for every other. A `None`-returning
//! default is safe only when the answer genuinely is "this family does not
//! have one"; where `config.json` states the answer, read it here.

use crate::validation::ConfigValidationResult;

use super::{
    layer_types, rope_types, Activation, EmbeddingNorm, ExpertFormat, ExpertGatePolicy,
    ExpertRoutingPolicy, FfnType, GateUpLayout, LayerKind, Llama3RopeScaling, ModelConfig,
    NormSpec, NormType, PositionPolicy, PostNormEps, QkNormScope, RotaryFrequencyBasis,
    YarnRopeScaling,
};

/// The multiplier that leaves a value unchanged.
///
/// Named so that lowering an absent operation to "multiply by one" reads
/// as the deliberate identity it is, and cannot be mistaken for a magic
/// constant standing in for an unread config fact.
const IDENTITY_SCALE: f32 = 1.0;

/// Attention score scale from a declared `query_pre_attn_scalar` (or any
/// other scalar the score is `1/sqrt` of, e.g. `head_dim`).
///
/// The one definition of this derivation — [`ModelArchitecture::attention_scale`]
/// and [`ModelArchitecture::attention_scale_for_layer`] both call it, and so
/// does the VINDEX3 plan's carriage check (`larql-vindex`'s
/// `plan::carriage::canonical_declared`), which needs to know that a
/// declared `256` and a carried `0.0625` are the same fact seen on either
/// side of this formula, not a dropped one. A second copy anywhere would be
/// exactly the kind of formula duplication `Activation::uses_gelu_tanh_gate_up`
/// warns about.
pub fn score_scale_from_query_pre_attn_scalar(scalar: f64) -> f64 {
    scalar.powf(-0.5)
}

/// Architecture-specific behavior. Describes how a model is structured
/// without performing any computation.
pub trait ModelArchitecture: Send + Sync {
    /// Model family name (e.g., "gemma3", "llama").
    fn family(&self) -> &str;

    /// Parsed model configuration.
    fn config(&self) -> &ModelConfig;

    /// Validate parsed architecture dimensions and cross-field invariants.
    ///
    /// Detection is intentionally permissive so callers can inspect partially
    /// specified configs. Call this before inference or extraction to fail
    /// early on inconsistent head geometry, RoPE settings, per-layer metadata,
    /// MoE routing, or other values that would otherwise surface later.
    fn validate(&self) -> ConfigValidationResult {
        crate::validation::validate_architecture(self)
    }

    // ── Tensor key patterns ──

    /// Key prefix for a layer's tensors (e.g., "layers.5.").
    fn layer_prefix(&self, layer: usize) -> String {
        format!("layers.{layer}.")
    }

    /// Prefixes to strip from raw safetensors keys.
    /// Tried in order; first match wins.
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["language_model.model.", "model."]
    }

    /// Embedding tensor key (after prefix stripping).
    fn embed_key(&self) -> &str {
        "embed_tokens.weight"
    }

    /// Whether the model has a text output projection at all — a
    /// standalone `lm_head.weight` or one tied to the embedding matrix.
    /// `true` for every text LM. `false` for models whose backbone's only
    /// product is its final hidden state (MOSS-TTS-Realtime: the output
    /// heads live in a side-loaded audio depth transformer, and
    /// `tie_word_embeddings` semantics do not apply). When `false`, the
    /// safetensors loader stores the embedding as a placeholder head that
    /// must never be sampled from; it becomes properly optional when
    /// generation output moves behind the output-adapter seam
    /// (`docs/tts-funnel.md` §3 step 4).
    fn has_lm_head(&self) -> bool {
        true
    }

    /// Whether a *missing* standalone output-head tensor should be read as
    /// tied to the embedding matrix, rather than a lost one.
    ///
    /// `true` when this architecture has no independent head at all
    /// ([`Self::has_lm_head`] false — the embedding is a
    /// never-sampled placeholder) or when the checkpoint does not declare
    /// `tie_word_embeddings: false`. `false` only when a checkpoint
    /// explicitly declares `tie_word_embeddings: false` and still has no
    /// separate head tensor — the one case that must surface as a loud
    /// missing-tensor error rather than silently serving the wrong
    /// projection (GPT-OSS, OLMoE both declare `false`).
    ///
    /// The one statement of this rule: the safetensors loader's
    /// `cfg_declares_untied` (`loading/safetensors/mod.rs`) checks the
    /// identical `tie_word_embeddings == Some(false)` fact to decide the
    /// same question for the VINDEX2 load path.
    fn output_head_reuses_embedding(&self) -> bool {
        !self.has_lm_head() || self.config().tie_word_embeddings != Some(false)
    }

    /// Learned positional-embedding tensor key, when the architecture uses
    /// absolute learned position embeddings added to the token embedding at
    /// the input (GPT-2). Architectures that use rotary or no positional
    /// signal return `None` (the default). When `Some`, the loader populates
    /// `ModelWeights::position_embed`.
    fn position_embed_key(&self) -> Option<&str> {
        None
    }

    /// Final norm weight key.
    fn final_norm_key(&self) -> &str {
        "norm.weight"
    }

    /// Attention weight keys for a layer.
    fn attn_q_key(&self, layer: usize) -> String {
        format!("{}self_attn.q_proj.weight", self.layer_prefix(layer))
    }
    fn attn_k_key(&self, layer: usize) -> String {
        format!("{}self_attn.k_proj.weight", self.layer_prefix(layer))
    }
    fn attn_v_key(&self, layer: usize) -> String {
        format!("{}self_attn.v_proj.weight", self.layer_prefix(layer))
    }
    fn attn_o_key(&self, layer: usize) -> String {
        format!("{}self_attn.o_proj.weight", self.layer_prefix(layer))
    }

    /// Tensor key for a fused Q/K/V projection, when the architecture packs
    /// all three into a single matrix (Conv1D-style: GPT-2). The loader splits
    /// the fused tensor into the per-projection keys returned by
    /// `attn_q_key` / `attn_k_key` / `attn_v_key` after loading. Default: None.
    fn fused_qkv_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Tensor key for a fused Q/K/V bias vector, paired with `fused_qkv_key`.
    /// Default: None.
    fn fused_qkv_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Attention bias keys (None if model doesn't use attention bias).
    fn attn_o_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }
    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }
    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }
    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }

    /// QK norm weight keys (None if model doesn't use QK norm).
    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }
    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }

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

    /// Attention-sink key (None if the architecture has no sinks).
    ///
    /// A sink is a learned per-head logit concatenated to the attention
    /// logits, included in the softmax, then dropped — so the weights
    /// over real keys deliberately sum to less than one. Architectures
    /// that use them (GPT-OSS) must both return this key *and* have the
    /// attention kernel apply it; returning the key alone changes
    /// nothing but the extracted bytes.
    fn attn_sinks_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// FFN bias keys (None if model doesn't use FFN bias).
    fn ffn_up_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }
    fn ffn_down_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// FFN weight keys for a layer.
    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}mlp.gate_proj.weight", self.layer_prefix(layer))
    }
    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}mlp.up_proj.weight", self.layer_prefix(layer))
    }
    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}mlp.down_proj.weight", self.layer_prefix(layer))
    }

    /// Layer norm weight keys.
    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}input_layernorm.weight", self.layer_prefix(layer))
    }
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!(
            "{}post_attention_layernorm.weight",
            self.layer_prefix(layer)
        )
    }

    // ── LayerNorm biases (declared 2026-07-31) ──
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
    fn norm_bias_of(&self, weight_key: &str) -> Option<String> {
        if self.norm_type() != NormType::LayerNorm {
            return None;
        }
        weight_key
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.bias"))
    }
    fn pre_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}pre_feedforward_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }
    fn post_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}post_feedforward_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Behavior ──

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

    /// Parameter-free QK normalisation (RMS over Q/K with no learned
    /// weights). Default: none — families that normalise without weights
    /// declare it, because no tensor evidence can reveal a weightless op.
    fn parameter_free_qk_norm(&self) -> crate::config::ParameterFreeQkNorm {
        crate::config::ParameterFreeQkNorm::default()
    }

    /// The embedding-scale *operation*, when the model has one.
    /// Gemma: sqrt(hidden_size), Granite: `embedding_multiplier`.
    ///
    /// `None` means the model declares no embedding-scale operation — a
    /// different claim from `Some(1.0)`, which means the source semantics
    /// specify a multiply by one. Collapsing the two here would destroy
    /// the distinction before inventory or vindex ever sees it, so the
    /// front end preserves absence as absence and only
    /// [`embed_scale_multiplier`](Self::embed_scale_multiplier) may turn
    /// it into a number.
    fn embed_scale(&self) -> Option<f32> {
        self.config().embedding_multiplier.map(|v| v as f32)
    }

    /// The factor the forward path multiplies embeddings by.
    ///
    /// Executing "no embedding-scale operation" and "multiply by one"
    /// coincide numerically, and execution must eventually produce a
    /// number. This is the single, named place that coincidence is
    /// allowed — a judgment about lowering, not a fallback that hides a
    /// missing fact. Semantic consumers must read
    /// [`embed_scale`](Self::embed_scale) instead.
    fn embed_scale_multiplier(&self) -> f32 {
        self.embed_scale().unwrap_or(IDENTITY_SCALE)
    }

    /// BOS token to prepend before inference when the tokenizer's
    /// `post_processor` doesn't already add one.
    ///
    /// Gemma 4's shipped `tokenizer.json` leaves BOS out of the
    /// `TemplateProcessing.single` template (unlike Gemma 2/3), so
    /// `tokenizer.encode(prompt, true)` returns tokens without BOS and
    /// the model sees a broken sequence. Architectures that need BOS
    /// return `Some(id)` here and callers prepend it if the encoding
    /// doesn't already start with it.
    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    /// Activation function for the FFN.
    ///
    /// The default reads the checkpoint's declared `hidden_act` /
    /// `hidden_activation` name when present (through the one mapping in
    /// [`Activation::from_hf_name`]) and falls back to SiLU only when the
    /// config is silent. A family override that ignores the config is
    /// asserting it knows better than the checkpoint — every current
    /// override's family also declares the matching name.
    fn activation(&self) -> Activation {
        self.config()
            .hidden_act
            .as_deref()
            .and_then(Activation::from_hf_name)
            .unwrap_or(Activation::Silu)
    }

    /// FFN type (gated vs standard).
    fn ffn_type(&self) -> FfnType {
        FfnType::Gated
    }

    /// Whether this model has separate pre/post norms around attention and FFN
    /// (Gemma 2/3 style with 4 norms per layer) vs standard pre-norm only.
    fn has_post_norms(&self) -> bool {
        false
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

    /// RoPE base frequency for a given layer.
    /// Gemma3 uses different bases for sliding vs global attention layers.
    ///
    /// Only meaningful on layers whose
    /// [`position_policy_for_layer`](Self::position_policy_for_layer) is
    /// rotary; the policy is the authority on *whether* a layer rotates.
    fn rope_base_for_layer(&self, layer: usize) -> f64 {
        let _ = layer;
        self.config().rope_base
    }

    /// Positional-encoding policy for a given layer.
    ///
    /// A checkpoint declaring `layer_rope_theta` states the policy per layer
    /// — including intentional absence (NoPE), which the HF form spells as a
    /// `0.0` sentinel honoured here and nowhere else. Without the array,
    /// every layer rotates at [`rope_base_for_layer`](Self::rope_base_for_layer).
    ///
    /// The forward path for a family that declares NoPE layers must consume
    /// this policy, not a raw theta — a zero base is degenerate
    /// (`1/0^(i/d)`), never a parameter value.
    ///
    /// A checkpoint that declares `rope_scaling.rope_type = "yarn"` states a
    /// second execution fact — scaled frequencies and an attention amplitude
    /// — for every rotating layer, and the policy carries it
    /// ([`PositionPolicy::Yarn`]) rather than letting it be read once by the
    /// forward path and dropped by everything else ([`Self::yarn_rope_scaling`]
    /// is the config read; this is where it becomes per-layer policy).
    fn position_policy_for_layer(&self, layer: usize) -> PositionPolicy {
        // A declared relative scheme decides the policy outright. Checked
        // before every rotary branch because `rope_base` carries a
        // DEFAULT: without this, a checkpoint that declares no rope key at
        // all still resolves to `Rope { theta: 10000 }`, which is a
        // rotation the author never asked for on every layer.
        if let (Some(d_rel), Some(extent)) = (self.config().d_rel, self.config().rel_extent) {
            return PositionPolicy::Relative { d_rel, extent };
        }
        // `mla_use_nope` means what it says: the MLA block applies no
        // positional rotation at all.
        //
        // Judged from Kimi Linear's own `modeling_kimi.py`, not from the
        // flag's name, because the config looks self-contradictory — it
        // declares `mla_use_nope: true` *and* `qk_rope_head_dim: 64`. The
        // reference settles it two ways:
        //
        //   1. the file contains **no rotary code whatsoever** — `q_rot`
        //      and `k_rot` are split out and concatenated straight back,
        //      unrotated;
        //   2. `self.use_nope` is read exactly once, as `assert
        //      self.use_nope` — the flag is a *precondition*, not a
        //      switch, and the class refuses to run without it.
        //
        // So `qk_rope_head_dim` is a **structural width**, not a rotary
        // subspace: it splits `q_head_dim = 128 + 64 = 192` (q_proj rows
        // 32·192 = 6144, as stored) and gives `kv_a_proj_with_mqa` its
        // extra 64 outputs, broadcast across heads as a shared unrotated K
        // component. The key name is actively misleading and only the
        // reference could settle it.
        //
        // Deliberately keyed on `Some(true)`. `false` is a combination the
        // reference does not implement — its assert fires — so this build
        // has no ground truth for it and must not answer.
        if self.config().mla_use_nope == Some(true) {
            return PositionPolicy::None;
        }
        let yarn = self.yarn_rope_scaling();
        match self
            .config()
            .layer_rope_theta
            .as_ref()
            .and_then(|thetas| thetas.get(layer))
        {
            Some(&declared) => {
                // A per-layer theta states WHICH base this layer rotates
                // at. It does not state that the layer stops being a
                // partial or multi-axis rotary, so the config's rotary
                // shape is re-applied at that theta. Without this, any
                // checkpoint declaring `layer_rope_theta` alongside
                // `partial_rotary_factor` would silently rotate the whole
                // head — the same drop this rung exists to close, one
                // branch over.
                match PositionPolicy::from_declared_theta_with_yarn(declared, yarn) {
                    PositionPolicy::Rope { theta } => self.rotary_policy(theta),
                    // NoPE has no rotary shape, and YaRN carries its own.
                    resolved => resolved,
                }
            }
            None => match yarn {
                Some(scaling) => PositionPolicy::Yarn {
                    theta: self.rope_base_for_layer(layer),
                    scaling,
                },
                None => self.rotary_policy(self.rope_base_for_layer(layer)),
            },
        }
    }

    /// The unscaled rotary policy at `theta`: plain, partial, or
    /// multi-axis, decided by what the config declares.
    ///
    /// Lives in the trait default rather than a family override because
    /// `partial_rotary_factor` and `rope_parameters.mrope_*` are config
    /// facts, not family facts — a model declaring them means the same
    /// thing whatever its `model_type`. A family whose partial rotary
    /// uses a different frequency basis (Gemma 4, HF `proportional`)
    /// overrides [`Self::position_policy_for_layer`] and never reaches
    /// here.
    ///
    /// A declared fraction of `1.0` is a full rotary and stays
    /// [`PositionPolicy::Rope`]: the fraction is then not a partial
    /// rotary fact, and manufacturing a `PartialRope` for it would put
    /// every ordinary model on the partial path.
    fn rotary_policy(&self, theta: f64) -> PositionPolicy {
        let cfg = self.config();
        let Some(rotary_fraction) = cfg.partial_rotary_factor.filter(|f| *f < 1.0) else {
            return PositionPolicy::Rope { theta };
        };
        // HF's `default` rope with `partial_rotary_factor` takes the
        // inverse frequencies over the ROTARY width — `base**(arange(0,
        // dim, 2) / dim)` with `dim = head_dim * factor`. The head-width
        // basis is a different declaration (`proportional`) and a
        // different family's override.
        let basis = RotaryFrequencyBasis::RotaryWidth;
        match (cfg.mrope_section.as_deref(), cfg.mrope_interleaved) {
            (Some([t, h, w]), Some(interleaved)) => PositionPolicy::MRope {
                theta,
                rotary_fraction,
                basis,
                section: [*t, *h, *w],
                interleaved,
            },
            // Either half alone is an incomplete declaration. Resolving
            // it as a plain partial rotary would drop the multi-axis
            // fact silently, so the partial policy is built and the
            // plan's carriage gate refuses the unpaired key — the
            // refusal belongs where refusals are reported, not here.
            _ => PositionPolicy::PartialRope {
                theta,
                rotary_fraction,
                basis,
            },
        }
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

    /// Fraction of head_dim to apply RoPE to (0.0–1.0).
    /// Models with partial rotary embedding (e.g., 0.25) override per layer.
    /// Default: 1.0 (full rotation).
    fn rotary_fraction_for_layer(&self, _layer: usize) -> f64 {
        1.0
    }

    /// Whether value shares key projections at this layer (V = K).
    /// When true, the forward pass uses K in place of V.
    /// Default: false.
    fn v_shares_k(&self, _layer: usize) -> bool {
        false
    }

    /// Whether to apply parameter-free RMSNorm to V states before attention.
    /// Gemma 4 applies V-norm (normalization only, no learned scale).
    /// Default: false.
    fn has_v_norm(&self) -> bool {
        false
    }

    /// Per-layer scalar multiplier key. When present, the layer output is
    /// multiplied by this learned scalar after the residual add.
    /// Default: None (no per-layer scalar).
    fn layer_scalar_key(&self, _layer: usize) -> Option<String> {
        None
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

    // ── Per-Layer Embeddings (PLE) ──

    /// Whether this model uses per-layer embeddings (PLE).
    /// When true, each layer adds a gated embedding lookup to the hidden state.
    fn has_per_layer_embeddings(&self) -> bool {
        self.config().per_layer_embed_dim.unwrap_or(0) > 0
    }

    /// Per-layer embedding dimension. 0 if PLE is not used.
    fn per_layer_embed_dim(&self) -> usize {
        self.config().per_layer_embed_dim.unwrap_or(0)
    }

    /// Key for the shared per-layer embedding matrix [vocab, num_layers * ple_dim].
    fn per_layer_embed_key(&self) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some("embed_tokens_per_layer.weight".to_string())
        } else {
            None
        }
    }

    /// Key for the model-level PLE projection [num_layers * ple_dim, hidden].
    fn per_layer_model_projection_key(&self) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some("per_layer_model_projection.weight".to_string())
        } else {
            None
        }
    }

    /// Key for the shared PLE projection RMSNorm weight.
    fn per_layer_projection_norm_key(&self) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some("per_layer_projection_norm.weight".to_string())
        } else {
            None
        }
    }

    /// Key for the per-layer input gate projection [ple_dim, hidden].
    fn per_layer_input_gate_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}per_layer_input_gate.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    /// Key for the per-layer output projection [hidden, ple_dim].
    fn per_layer_projection_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}per_layer_projection.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    /// Key for the post-PLE norm weight.
    fn post_per_layer_input_norm_key(&self, layer: usize) -> Option<String> {
        if self.has_per_layer_embeddings() {
            Some(format!(
                "{}post_per_layer_input_norm.weight",
                self.layer_prefix(layer)
            ))
        } else {
            None
        }
    }

    // ── Softcapping (Gemma2) ──

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

    /// Multiplier on the final hidden state before the vocabulary
    /// projection. `None` = the model declares no output-multiplier
    /// operation, which is a different claim from `Some(1.0)`.
    ///
    /// Canonical name, same reasoning as [`qk_scale_factor`](Self::qk_scale_factor):
    /// **The multiplicative factor**, already resolved. Callers multiply by
    /// it; nobody downstream needs to know which spelling produced it.
    ///
    /// The two spellings are not interchangeable, which is the trap this
    /// method exists to close:
    ///
    /// ```text
    /// output_multiplier   x  ->  logits * x        a multiplier
    /// logits_scaling      d  ->  logits / d        a DIVISOR
    /// ```
    ///
    /// Scaling does commute through the linear head, so "before the vocab
    /// projection" and "on the logits" are the same number — but only after
    /// the divisor has been inverted. Passing Granite's `logits_scaling`
    /// through as a multiplier put its logits out by `d²` (a factor of 100
    /// at `logits_scaling = 10`), which saturates every softmax built on
    /// them.
    ///
    /// That error is invisible to greedy decoding: a positive scalar cannot
    /// reorder logits, so argmax, generated ids and every oracle built on
    /// them agree exactly while the distribution is wrong. Only a
    /// probability-space measurement — KL, NLL, top-p, temperature — can
    /// see it, which is how it survived.
    ///
    /// A non-finite or zero `logits_scaling` yields `None` rather than an
    /// infinity that would silently annihilate the head.
    fn logit_scale(&self) -> Option<f64> {
        if let Some(m) = self.config().output_multiplier {
            return Some(m);
        }
        match self.config().logits_scaling {
            Some(d) if d.is_finite() && d != 0.0 => Some(1.0 / d),
            _ => None,
        }
    }

    /// Residual-stream scaling: the sublayer's own output (attention or
    /// FFN) is multiplied by this before being added into the residual
    /// stream, at both sites, with the same value (Granite's
    /// `residual_multiplier`). `None` = no such operation, distinct from
    /// `Some(1.0)`. No other family in this registry scales the residual
    /// stream, so there is no second spelling to resolve here yet.
    fn residual_scale(&self) -> Option<f32> {
        self.config().residual_multiplier.map(|v| v as f32)
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

    /// Whether attention projections carry biases, when the checkpoint
    /// says so. `None` = the config is silent; the loader's
    /// tensor-presence check answers.
    fn attention_bias(&self) -> Option<bool> {
        self.config().attention_bias
    }

    /// Declared context bound (`max_position_embeddings`).
    fn max_position_embeddings(&self) -> Option<usize> {
        self.config().max_position_embeddings
    }

    /// Target-model layers whose hidden states this artifact consumes —
    /// present only on drafter-style checkpoints (`target_layer_ids`).
    fn target_layer_ids(&self) -> Option<&[usize]> {
        self.config().target_layer_ids.as_deref()
    }

    /// Final logit softcapping value (None = disabled).
    /// Applied to output logits: logits = tanh(logits / cap) * cap
    fn final_logit_softcapping(&self) -> Option<f32> {
        self.config().final_logit_softcapping.map(|v| v as f32)
    }

    // ── Scaling multipliers (Granite-style) ──

    /// Residual stream scaling factor applied after attention and FFN additions.
    fn residual_multiplier(&self) -> f32 {
        self.config()
            .residual_multiplier
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    /// Attention score scaling factor (applied on top of 1/sqrt(head_dim)).
    fn attention_multiplier(&self) -> f32 {
        self.config()
            .attention_multiplier
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    /// Logits scaling factor applied to final logits before softmax.
    fn logits_scaling(&self) -> f32 {
        self.config()
            .logits_scaling
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }

    // ── MoE (Mixture of Experts) ──

    /// How expert weights are stored in this model.
    fn expert_format(&self) -> ExpertFormat {
        ExpertFormat::PerExpert
    }

    /// How this checkpoint's fused `gate_up` operand splits into the gate and
    /// up branches, or `None` where no fused operand exists (dense models and
    /// per-expert MoE, which store `gate_proj`/`up_proj` separately).
    ///
    /// `None` is *not* a usable default for a packed architecture: readers
    /// must refuse rather than guess, because the two layouts differ by a
    /// silent permutation of rows rather than by anything a shape check
    /// could catch. See [`GateUpLayout`] for the two families that already
    /// disagree.
    fn gate_up_layout(&self) -> Option<GateUpLayout> {
        None
    }

    /// Whether this model uses Mixture of Experts.
    ///
    /// Answered from the **declaration**, not from a family list: a
    /// checkpoint that states a routed-expert count is an MoE whether or
    /// not this build has a registry entry for it. The previous `false`
    /// default meant an unregistered MoE resolved as dense — Kimi Linear,
    /// with 256 experts per layer and top-8 routing, produced an execution
    /// surface saying `ffn: dense, intermediate_size 9216`, which is the
    /// dense-layer width of one layer out of twenty-seven.
    ///
    /// Losing an MoE this way is not a gap in a report. It is a container
    /// that would describe the wrong model.
    fn is_moe(&self) -> bool {
        self.config().num_experts.is_some_and(|experts| experts > 0)
    }

    /// Number of routed experts per layer.
    fn num_experts(&self) -> usize {
        self.config().num_experts.unwrap_or(0)
    }

    /// Number of experts activated per token.
    fn num_experts_per_token(&self) -> usize {
        self.config().num_experts_per_token.unwrap_or(0)
    }

    /// Number of shared (always-active) experts.
    fn num_shared_experts(&self) -> usize {
        self.config().num_shared_experts.unwrap_or(0)
    }

    /// Router weight key for expert selection.
    fn moe_router_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// The routing rule this architecture's MoE block uses.
    ///
    /// Override this, not [`Self::moe_router_type`] — the string form is
    /// derived from it. Returning a *typed* kind is what lets the compute
    /// layer `match` exhaustively instead of comparing strings and falling
    /// back silently on any value it has not heard of.
    fn moe_router_kind(&self) -> super::MoeRouterKind {
        // The declared scoring function decides the rule. Answering the
        // softmax default for a checkpoint that declares `sigmoid` states
        // a routing rule the model does not use — and it is not a small
        // difference: sigmoid scores are independent, so the selected
        // weights do not sum to 1.
        //
        // An unrecognised spelling keeps the default here and is caught by
        // the plan's declared-vs-resolved comparison instead, which can
        // refuse where this signature cannot.
        match self.config().router_activation.as_deref() {
            Some(super::moe_router::ROUTER_ACTIVATION_SIGMOID) => super::MoeRouterKind::Sigmoid,
            _ => super::MoeRouterKind::default(),
        }
    }

    /// Router algorithm identifier written into `MoeConfig.router_type` in a
    /// vindex. Serialisation only — dispatch on [`Self::moe_router_kind`].
    fn moe_router_type(&self) -> &str {
        self.moe_router_kind().as_str()
    }

    /// Expert FFN gate weight key.
    fn expert_ffn_gate_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    /// Expert FFN up-projection weight key.
    fn expert_ffn_up_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    /// Expert FFN down-projection weight key.
    fn expert_ffn_down_key(&self, _layer: usize, _expert_id: usize) -> Option<String> {
        None
    }

    // ── Packed expert keys (MXFP4 models) ──

    /// Packed gate+up projection blocks key (all experts fused, MXFP4).
    fn packed_gate_up_blocks_key(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Packed gate+up projection scales key.
    fn packed_gate_up_scales_key(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Packed down projection blocks key.
    fn packed_down_blocks_key(&self, _layer: usize) -> Option<String> {
        None
    }
    /// Packed down projection scales key.
    fn packed_down_scales_key(&self, _layer: usize) -> Option<String> {
        None
    }

    // ── MoE biases and gating (declared 2026-07-30) ──
    //
    // These three tensors exist per layer in the GPT-OSS checkpoint and were
    // silently dropped for the same reason the attention biases were: nothing
    // named them, so extraction never asked and the forward pass never
    // applied them. A key returned here is only half the obligation — the
    // backend has to consume it. See `docs/k3-funnel.md` §4.7.

    /// Router bias key, added to the router logits before top-k selection.
    /// `None` for routers without a bias term.
    fn moe_router_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Per-expert bias on the fused gate+up projection.
    /// Shape `[num_experts, 2 * moe_intermediate_size]`, interleaved on the
    /// last axis exactly as the weight rows are.
    fn packed_gate_up_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Per-expert bias on the expert down projection.
    /// Shape `[num_experts, hidden_size]`.
    fn packed_down_bias_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// How an expert's gate/up projection is combined into the down
    /// projection's input. Defaults to the plain gated FFN every other
    /// MoE architecture in the support table uses.
    fn expert_gate_policy(&self) -> ExpertGatePolicy {
        // Deliberately NOT derived from `swiglu_limit`.
        //
        // A declared clamp says a bound exists; it does not say the layer
        // computes [`ExpertGatePolicy::ClampedGlu`], which is a specific
        // formula — `glu = g·sigmoid(alpha·g)`, `out = (u+1)·glu`, with
        // `alpha = 1.702` — transcribed from GPT-OSS's reference. GLM-5.3-
        // Flash and Inkling-Small both declare a `swiglu_limit` too, and
        // nothing on hand says they share that activation.
        //
        // Resolving the policy from the bound alone would claim they do,
        // on the strength of one shared field name. That is the same
        // inference `layer_types` → Gated DeltaNet made, and it is wrong
        // for the same reason: a declared parameter is not evidence of the
        // operator that consumes it. An architecture that has been judged
        // against its own reference overrides this.
        ExpertGatePolicy::Gated
    }

    /// How the router's top-k weights are normalised.
    ///
    /// Read from `norm_topk_prob`, which is exactly what that field means:
    /// `true` renormalises the selected weights to sum to 1; `false` or absent
    /// keeps the raw softmax probabilities, which sum to less by however much
    /// mass the unselected experts hold.
    ///
    /// **This default reads the config on purpose.** Baking one routing order
    /// into code that serves both architectures is a rescale of the entire
    /// expert branch, and it has now been got wrong twice — `docs/k3-funnel.md`
    /// §4.7 finding 5, then again in `select_and_normalise` an hour after that
    /// finding was written up. A config-reading default makes a new MoE
    /// architecture correct on arrival instead of correct only if someone
    /// remembers to override; `QwenArch` shipped `norm_topk_prob: true` models
    /// and inherited the wrong order for exactly that reason.
    ///
    /// Override only where the order is fixed by the architecture rather than
    /// by config, as GPT-OSS's is.
    fn expert_routing_policy(&self) -> ExpertRoutingPolicy {
        if self.config().norm_topk_prob.unwrap_or(false) {
            ExpertRoutingPolicy::NormalisedOverSelected
        } else {
            ExpertRoutingPolicy::SoftmaxThenSelect
        }
    }

    /// Shared expert FFN gate weight key.
    fn shared_expert_gate_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Shared expert FFN up-projection weight key.
    fn shared_expert_up_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Shared expert FFN down-projection weight key.
    fn shared_expert_down_key(&self, _layer: usize) -> Option<String> {
        None
    }

    // ── Hybrid MoE (Gemma 4 A4B: dense MLP + expert block summed per layer) ──

    /// Whether this model has a hybrid dense-MLP + expert block per layer.
    /// Unlike pure MoE (Mixtral/DeepSeek), both branches run and their outputs are summed.
    fn is_hybrid_moe(&self) -> bool {
        false
    }

    /// Per-expert intermediate (hidden) dimension. 0 for non-MoE models.
    fn moe_intermediate_size(&self) -> usize {
        // The declared expert width. `0` was the old default and it is not
        // a width — an MoE surface carrying it states that each expert
        // projects to nothing, which no closure check can satisfy and no
        // reader can act on.
        self.config().moe_intermediate_size.unwrap_or(0)
    }

    /// Packed stacked gate+up projection key (Gemma 4 PackedBF16 format).
    /// Tensor shape: [num_experts, 2 * moe_intermediate_size, hidden_size].
    fn packed_experts_gate_up_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Packed stacked down projection key (Gemma 4 PackedBF16 format).
    /// Tensor shape: [num_experts, hidden_size, moe_intermediate_size].
    fn packed_experts_down_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Gemma 4 router learned input-scale key (`router.scale`).
    fn moe_router_scale_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Gemma 4 router per-expert output-scale key (`router.per_expert_scale`).
    fn moe_router_per_expert_scale_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Router's own RMS-norm weight, applied to the router's input *before*
    /// the router projection. HF Gemma 4's `Gemma4TextRouter.norm` is
    /// **parameter-free** (`with_scale=False`) so no tensor exists on disk —
    /// see [`moe_router_norm_parameter_free`](Self::moe_router_norm_parameter_free).
    /// This key is provided for architectures that DO ship a learned router
    /// norm weight. Return `None` to fall back to either parameter-free
    /// RMSNorm (when the flag is set) or the experts' pre-norm output.
    fn moe_router_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Whether the router applies a parameter-free RMSNorm to its input
    /// before `router_scale`/projection. Gemma 4 sets this true. When true
    /// AND `moe_router_norm_key` returns `None`, the forward pass runs
    /// `x / sqrt(mean(x²) + eps)` on the raw residual instead of reusing
    /// the experts' pre-norm (which would apply the wrong learned weight).
    fn moe_router_norm_parameter_free(&self) -> bool {
        false
    }

    /// Scalar multiplier applied to the router input after `router.norm` and
    /// after the learned `router.scale` vector. Gemma 4 uses `hidden_size^-0.5`
    /// (called `scalar_root_size` in HF). Return `None` for no scaling.
    fn moe_router_input_scalar(&self) -> Option<f32> {
        None
    }

    /// Outer post-FFN norm for hybrid MoE layers — applied to `(h1 + h2)`
    /// before the residual add, where `h1 = post_ffn_norm_1(dense)` and
    /// `h2 = post_ffn_norm_2(moe)`. HF Gemma 4 stores this at the un-suffixed
    /// key `post_feedforward_layernorm.weight`, while the dense-branch norm
    /// uses the suffixed `_1` variant (see `post_feedforward_layernorm_key`).
    /// Return `None` for architectures that either don't combine via an outer
    /// norm or reuse the dense-branch norm as the outer norm.
    fn moe_post_outer_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Post-FFN norm for dense MLP output in hybrid MoE layers.
    /// Gemma 4 A4B: `post_feedforward_layernorm_1.weight` (replaces the plain variant).
    fn moe_post_ffn1_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Pre-norm applied to the residual before feeding into the expert block.
    /// Gemma 4 A4B: `pre_feedforward_layernorm_2.weight`.
    fn moe_pre_experts_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Post-norm applied to the expert block output.
    /// Gemma 4 A4B: `post_feedforward_layernorm_2.weight`.
    fn moe_post_experts_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Whether the hybrid MoE forward applies a final RMS norm to the
    /// combined (dense + expert) output before adding to the residual.
    ///
    /// Gemma 4 26B A4B: true — matches HF `post_feedforward_layernorm(combined)`.
    /// All other models: false — use `layer_scalar * combined` instead.
    fn moe_has_combined_output_norm(&self) -> bool {
        false
    }

    // ── MLA (Multi-head Latent Attention) ──

    /// Whether this model uses MLA instead of standard GQA.
    fn uses_mla(&self) -> bool {
        false
    }

    /// MLA compressed KV dimension.
    fn kv_lora_rank(&self) -> usize {
        0
    }

    /// MLA Q compression rank.
    fn q_lora_rank(&self) -> usize {
        0
    }

    /// MLA KV down-projection key (compress).
    fn mla_kv_a_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA KV up-projection key (decompress).
    fn mla_kv_b_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA Q down-projection key (compress).
    fn mla_q_a_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA Q up-projection key (decompress).
    fn mla_q_b_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// DS-V3 MLA: non-RoPE head dim (nope). Combined qk_head_dim = nope + rope.
    fn mla_qk_nope_head_dim(&self) -> Option<usize> {
        None
    }

    /// DS-V3 MLA: RoPE head dim portion.
    fn mla_qk_rope_head_dim(&self) -> Option<usize> {
        None
    }

    /// DS-V3 MLA: V head dim (after absorption may differ from qk dims).
    fn mla_v_head_dim(&self) -> Option<usize> {
        None
    }

    /// The epsilon MLA's latent norm (`kv_a_layernorm`) runs at, when this
    /// family's own reference code fixes one.
    ///
    /// `None` — the default — means **unjudged**, never "use the layer
    /// eps". The two are genuinely different numbers: Kimi Linear
    /// constructs `kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)` with
    /// no override, so it runs at `KimiRMSNorm.__init__`'s class default
    /// `1e-6` while every other norm in the same layer runs at
    /// `config.rms_norm_eps` (`1e-5`) — a factor of ten, on the one norm
    /// standing between the compressed cache and its decompression.
    ///
    /// This is an ARCHITECTURE fact, not a config fact: no checkpoint
    /// declares it, the family's modeling code does. So it is one of the
    /// legitimate overrides — a default that answered `Some(rms_norm_eps)`
    /// here would be the silent-wrong-serving shape the trait-default rule
    /// exists to prevent, and a family whose reference has not been read
    /// must reach an executor's named refusal rather than a plausible
    /// number.
    fn mla_kv_a_norm_eps(&self) -> Option<f64> {
        None
    }

    // ── RoPE scaling ──

    /// RoPE scaling type (None, "linear", "yarn", "dynamic", "llama3").
    fn rope_scaling_type(&self) -> Option<&str> {
        self.config()
            .rope_scaling
            .as_ref()
            .map(|s| s.scaling_type.as_str())
    }

    /// RoPE scaling factor.
    fn rope_scaling_factor(&self) -> f64 {
        self.config()
            .rope_scaling
            .as_ref()
            .map_or(1.0, |s| s.factor)
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

    /// Per-layer RoPE position divisor from `rope_scaling`. Used to honour
    /// linear rope-scaling and the Gemma 3 per-layer-type structured form.
    /// Default: 1.0 (no scaling).
    ///
    /// Gemma 3 overrides this to return the linear `factor` on global
    /// layers only and 1.0 on sliding layers, matching the HF
    /// `Gemma3TextConfig.rope_scaling.full_attention` structure.
    fn rope_position_divisor_for_layer(&self, _layer: usize) -> f64 {
        1.0
    }

    /// `llama3` RoPE scaling parameters when the architecture uses them.
    /// Default: `None`. Llama 3.x overrides to return the parsed factor
    /// and frequency-band thresholds. Consumed by
    /// `larql-inference::attention::rope::apply_rope_partial_at_full`.
    fn llama3_rope_scaling(&self) -> Option<Llama3RopeScaling> {
        None
    }

    /// `yarn` RoPE scaling parameters when the checkpoint declares them.
    ///
    /// **The read lives in the trait default deliberately.** [§4.7.8] recorded
    /// three recurrences of one shape: a behaviour that is a *config fact*,
    /// read on one architecture, with a trait default silently answering for
    /// everyone else. `rope_type: "yarn"` is a config fact — GPT-OSS and
    /// DeepSeek both ship it — so an architecture must not have to opt in to
    /// being served correctly. Anything that declares YaRN in `config.json`
    /// gets YaRN, and a new family arriving with it needs no code at all.
    ///
    /// [§4.7.8]: ../../../docs/k3-funnel.md
    fn yarn_rope_scaling(&self) -> Option<YarnRopeScaling> {
        let rs = self.config().rope_scaling.as_ref()?;
        if !rs
            .scaling_type
            .eq_ignore_ascii_case(rope_types::ROPE_TYPE_YARN)
        {
            return None;
        }
        Some(YarnRopeScaling {
            factor: rs.factor,
            beta_fast: rs.yarn_beta_fast.unwrap_or(crate::defaults::YARN_BETA_FAST),
            beta_slow: rs.yarn_beta_slow.unwrap_or(crate::defaults::YARN_BETA_SLOW),
            // Required, not defaulted: this is the window the model was
            // *pre-trained* at, and YaRN's correction bounds are defined
            // against it. HF indexes it unconditionally
            // (`rope_parameters_dict["original_max_position_embeddings"]`) and
            // raises if it is absent, so there is no value to inherit. A yarn
            // block without it is malformed and resolves to `None` here —
            // pinned by `yarn_without_original_context_is_not_scaling`.
            original_max_position_embeddings: rs.llama3_original_max_position_embeddings?,
            truncate: rs.yarn_truncate.unwrap_or(crate::defaults::YARN_TRUNCATE),
            mscale: rs.yarn_mscale,
            mscale_all_dim: rs.yarn_mscale_all_dim,
        })
    }

    /// Multi-modal contract for this architecture, if any.
    ///
    /// Returns `None` for text-only models (the default). Architectures
    /// that accept images / audio override to return a stable reference
    /// to a `MultiModalProtocol` impl describing their placeholder
    /// convention, expected encoder family, token budget, and scaling
    /// rules.
    ///
    /// Phase 0: every existing impl uses the default `None`. Adding
    /// `Some(...)` returns is a Phase 1+ concern, gated on the
    /// `EmbeddingPlan` consumer landing in `larql-compute`.
    fn multimodal(&self) -> Option<&dyn crate::multimodal::MultiModalProtocol> {
        None
    }

    // ── Engine capability predicates ──

    /// Whether per-layer K/V is a pure function of `(pre-attention
    /// residual, layer weights, absolute position)` — the structural
    /// precondition for engines that persist the residual stream and
    /// rebuild K/V on demand (the markov-residual family; see
    /// `crates/larql-inference/docs/specs/markov-residual-engine.md` §4).
    ///
    /// Derived from structural metadata, not an architecture allowlist:
    /// - the norm feeding the K/V projections must be stateless at
    ///   inference time — both current [`NormType`] variants are, and a
    ///   future stateful variant falls out of the `matches!` and fails
    ///   closed;
    /// - position must enter solely through RoPE re-applied at the
    ///   row's absolute index; learned absolute position tables
    ///   ([`Self::position_embed_key`]) sit outside the residual
    ///   recompute path;
    /// - K/V must be a direct projection of the normed residual — MLA
    ///   ([`Self::uses_mla`]) routes K/V through a decode-time latent
    ///   the recompute path cannot rebuild.
    fn kv_recomputable_from_residuals(&self) -> bool {
        let norm_stateless = matches!(self.norm_type(), NormType::RmsNorm | NormType::LayerNorm);
        norm_stateless && self.position_embed_key().is_none() && !self.uses_mla()
    }
}
