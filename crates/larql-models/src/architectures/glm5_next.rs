//! GLM-5.3-Flash (`glm5_next`) — KDA linear attention interleaved with
//! MLA-NoPE + a DeepSeek Sparse Attention indexer, over a 288-expert
//! sigmoid-routed MoE, on a four-stream mHC residual topology.
//!
//! `model_type: "glm5_next"` (and its text sub-config's
//! `"glm5_next_text"`) matched no family prefix in
//! [`super::super::detect::detect_from_json`], so without this file the
//! checkpoint falls to [`super::generic::GenericArch`] and is served
//! Llama-shaped defaults for norm placement, QK norm, embedding scaling
//! and gating — none of which this checkpoint declares. That is finding 8
//! of `larql vindex3 plan`, and the root of several others.
//!
//! # Provenance
//!
//! The checkpoint ships **no modeling code**, so the reference is
//! `transformers`' own `glm5_next`, which its `config.json` names via
//! `architectures: ["Glm5NextForConditionalGeneration"]`. Every fact below
//! is read from `modeling_glm5_next.py` /
//! `configuration_glm5_next.py` at 5.16.1 (sha256 `2092bbb4…` /
//! `b62936c9…`) and cross-checked against
//! `model.safetensors.index.json` on the real 306 GiB checkpoint — not
//! inferred from the config alone.
//!
//! # What this file deliberately does NOT claim
//!
//! The DSA indexer (`self_attn.indexer.*`), the mHC stream head and the
//! MTP sub-stack are **not** declared here. They are unbuilt operators,
//! and a key method answering for one would assert an execution surface
//! this build cannot serve. They stay absent so they keep blocking.

use crate::config::{ExpertGatePolicy, KdaGateForm, ModelArchitecture, ModelConfig};

pub struct Glm5NextArch {
    config: ModelConfig,
}

impl Glm5NextArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }

    /// The layer epsilon at the width the checkpoint declared it.
    ///
    /// NOT `norm_eps()`, whose return is `f32`: routing `1e-5` through
    /// f32 and back yields `9.999999747378752e-6`, and an operator norm's
    /// epsilon should carry the value the config states rather than a
    /// round-trip of it.
    fn declared_norm_eps(&self) -> f64 {
        self.config.norm_eps.unwrap_or(GLM5_DEFAULT_NORM_EPS as f64)
    }
}

impl ModelArchitecture for Glm5NextArch {
    fn family(&self) -> &str {
        "glm5_next"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// GLM nests the decoder under `model.language_model.`, not `model.`.
    ///
    /// The trait default (`["language_model.model.", "model."]`) strips
    /// `model.` and leaves `language_model.layers.0.…`, which matches no
    /// layer prefix — so every stack tensor reads as unclassified. Listed
    /// longest-first because the default tries them in order and
    /// `"model."` would otherwise win.
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["model.language_model.", "language_model.model.", "model."]
    }

    // ── KDA ──

    /// The clamped-sigmoid branch, with the declared `gate_lower_bound`
    /// applied — the opposite of Kimi Linear, which declares the same
    /// `-5.0` and reads it nowhere.
    ///
    /// `Glm5NextTextForgetGate.forward` takes
    /// `safe_gate_lower_bound * sigmoid(exp(A_log) * g)` whenever
    /// `config.linear_lower_bound is not None`, and
    /// `Glm5NextTextConfig.__init__` fills that from
    /// `linear_attn_config.gate_lower_bound`.
    fn kda_gate_form(&self) -> Option<KdaGateForm> {
        Some(
            match (self.config.kda_gate_lower_bound, self.config.kda_safe_gate) {
                (Some(lower_bound), _) => KdaGateForm::ClampedSigmoid { lower_bound },
                // An absent bound still clamps: the reference defaults
                // `safe_gate` to `True` and then fills `-5.0`.
                (None, None | Some(true)) => KdaGateForm::ClampedSigmoid {
                    lower_bound: crate::config::GLM5_DEFAULT_GATE_LOWER_BOUND,
                },
                (None, Some(false)) => KdaGateForm::Softplus,
            },
        )
    }

    /// The clamp, then ORDINARY SwiGLU.
    ///
    /// `Glm5NextTextExperts._apply_gate` and `Glm5NextTextMLP.forward`
    /// both clamp exactly as GPT-OSS does — one-sided on the gate,
    /// symmetric on the up branch — and then compute `silu(g) * u`. The
    /// reference carries the comment *"Simple swiglu instead of alpha"*
    /// over that line.
    ///
    /// Declared here rather than derived from `swiglu_limit`, because the
    /// same declaration means [`ExpertGatePolicy::ClampedGlu`] on GPT-OSS
    /// and this on GLM. Measured on GLM layer 3's real 288-expert bank:
    /// serving the GPT-OSS form instead is relative **31.7** — roughly
    /// `1/|u|`, since `(u + 1) ≈ 1` at a residual-scale activation while
    /// `u ≈ 0.03`, and every shape closes either way.
    fn expert_gate_policy(&self) -> ExpertGatePolicy {
        ExpertGatePolicy::ClampedGated {
            limit: self
                .config
                .swiglu_limit
                .map_or(GLM5_DEFAULT_SWIGLU_LIMIT, |v| v as f32),
        }
    }

    // ── MoE router ──
    //
    // `mlp.gate.*`, not Kimi's `block_sparse_moe.gate.*` and not the
    // DeepSeek lineage's `mlp.gate.*` semantics wholesale: GLM declares
    // `topk_method: "noaux_tc"`, so selection reads the bias-corrected
    // scores and weighting reads the unbiased ones, as Kimi does.

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.gate.weight", self.layer_prefix(layer)))
    }

    fn moe_router_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.gate.e_score_correction_bias",
            self.layer_prefix(layer)
        ))
    }

    // ── Routed experts: one tensor per expert per projection ──
    //
    // Ordinary `gate_proj`/`up_proj`/`down_proj` names (no `w1/w2/w3`
    // permutation), un-fused, 288 experts on each of 43 MoE layers.

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.{expert_id}.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Shared expert: one per sparse layer ──

    fn shared_expert_gate_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.shared_experts.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_up_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.shared_experts.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_down_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.shared_experts.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── MLA, with a q-LoRA the Kimi path has never had ──
    //
    // `KimiMLAAttention.__init__` asserts `q_lora_rank is None`, so Kimi
    // ships one flat `q_proj` and both q-side key methods stay on the
    // trait default. GLM declares `q_lora_rank: 1536` and ships
    // `q_a_proj` → `q_a_layernorm` → `q_b_proj`.

    fn uses_mla(&self) -> bool {
        self.config.kv_lora_rank.is_some()
    }

    fn kv_lora_rank(&self) -> usize {
        self.config.kv_lora_rank.unwrap_or(0)
    }

    fn q_lora_rank(&self) -> usize {
        self.config.q_lora_rank.unwrap_or(0)
    }

    fn mla_qk_nope_head_dim(&self) -> Option<usize> {
        self.config.qk_nope_head_dim
    }

    /// Declared `0` on this checkpoint, and that is the fact, not a
    /// missing value: `mla_use_nope: true` and the reference splits no
    /// rotary subspace at all, so `kv_a_proj_with_mqa` emits
    /// `kv_lora_rank + 0` outputs.
    fn mla_qk_rope_head_dim(&self) -> Option<usize> {
        self.config.qk_rope_head_dim
    }

    fn mla_v_head_dim(&self) -> Option<usize> {
        self.config.v_head_dim
    }

    fn mla_q_a_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_a_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn mla_q_b_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_b_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn mla_kv_a_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.kv_a_proj_with_mqa.weight",
            self.layer_prefix(layer)
        ))
    }

    fn mla_kv_b_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.kv_b_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    /// `config.rms_norm_eps`, stated explicitly — **not** the 1e-6 class
    /// default Kimi's latent norm silently runs at.
    ///
    /// `Glm5NextTextAttention.__init__` constructs
    /// `Glm5NextTextRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)`,
    /// passing the layer epsilon through, where
    /// `KimiMLAAttention.__init__` constructs `KimiRMSNorm(kv_lora_rank)`
    /// with no override and gets `1e-6` against a layer eps of `1e-5`.
    /// Two families, one tensor name, a factor of ten between them — which
    /// is why this method exists rather than a shared default.
    fn mla_kv_a_norm_eps(&self) -> Option<f64> {
        Some(self.declared_norm_eps())
    }

    /// Also `config.rms_norm_eps` — `Glm5NextTextAttention.__init__`
    /// constructs `q_a_layernorm` and `kv_a_layernorm` the same way,
    /// both passing the layer epsilon through. Stated separately anyway,
    /// because the two being equal is a fact about THIS family and not a
    /// property of MLA.
    fn mla_q_a_norm_eps(&self) -> Option<f64> {
        Some(self.declared_norm_eps())
    }

    /// `1e-5`, from `Glm5NextTextConfig.rms_norm_eps`'s own default — not
    /// the crate-wide `1e-6` majority. Declared rather than inherited for
    /// the reason the trait's own docs give: a checkpoint that omits the
    /// field gets whatever its config class defaults to, and OLMoE's
    /// measured cosine 0.890 → 0.991 is what inheriting the wrong one
    /// costs.
    fn default_norm_eps(&self) -> f32 {
        GLM5_DEFAULT_NORM_EPS
    }
}

/// `Glm5NextTextConfig.rms_norm_eps`'s class default.
const GLM5_DEFAULT_NORM_EPS: f32 = 1e-5;

/// `Glm5NextTextConfig.swiglu_limit`'s class default.
const GLM5_DEFAULT_SWIGLU_LIMIT: f32 = 10.0;

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal parsed config, mutated per test rather than guessing
    /// `config.json` spellings — the same convention `kimi::tests` uses.
    /// Values are GLM-5.3-Flash's own, scaled down where the size is
    /// irrelevant to what is being asserted.
    fn config_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": "glm5_next_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "vocab_size": 32,
            "rms_norm_eps": 1e-5,
            "q_lora_rank": 24,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 0,
            "v_head_dim": 8,
            "mla_use_nope": true,
            "swiglu_limit": 10.0,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "scoring_func": "sigmoid",
            "linear_attn_config": {
                "num_heads": 4,
                "head_dim": 8,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": [0, 1, 2],
                "full_attn_layers": [3]
            }
        })
    }

    fn arch_from(v: serde_json::Value) -> Glm5NextArch {
        let detected = crate::detect_from_json(&v);
        Glm5NextArch::from_config(detected.config().clone())
    }

    fn arch() -> Glm5NextArch {
        arch_from(config_json())
    }

    #[test]
    fn detection_resolves_glm5_next_and_not_the_generic_fallback() {
        for spelling in ["glm5_next", "glm5_next_text"] {
            let mut v = config_json();
            v["model_type"] = serde_json::json!(spelling);
            let a = crate::detect_from_json(&v);
            assert_eq!(a.family(), "glm5_next", "{spelling} must not fall back");
        }
    }

    /// GLM nests its decoder under `model.language_model.`, and the trait
    /// default would strip `model.` and leave `language_model.layers.0.…`
    /// — a prefix that matches no layer, so every stack tensor reads
    /// unclassified.
    #[test]
    fn the_language_model_prefix_is_stripped_before_the_bare_model_one() {
        let a = arch();
        let strip = a.key_prefixes_to_strip().to_vec();
        let lm = strip.iter().position(|p| *p == "model.language_model.");
        let bare = strip.iter().position(|p| *p == "model.");
        assert!(lm.is_some() && bare.is_some(), "{strip:?}");
        assert!(
            lm < bare,
            "the longer prefix must be tried first or `model.` wins: {strip:?}"
        );
    }

    /// The rung's load-bearing fact: GLM applies the declared bound where
    /// Kimi Linear, declaring the identical `-5.0`, does not.
    #[test]
    fn the_declared_gate_bound_is_applied_on_this_family() {
        assert_eq!(
            arch().kda_gate_form(),
            Some(KdaGateForm::ClampedSigmoid { lower_bound: -5.0 })
        );
    }

    /// `Glm5NextTextConfig.__init__` fills a null bound with `-5.0`
    /// whenever `safe_gate` is not explicitly false, so an absent bound
    /// still clamps.
    #[test]
    fn an_absent_bound_still_clamps_and_an_explicit_safe_gate_false_does_not() {
        let mut v = config_json();
        v["linear_attn_config"]["gate_lower_bound"] = serde_json::Value::Null;
        assert_eq!(
            arch_from(v.clone()).kda_gate_form(),
            Some(KdaGateForm::ClampedSigmoid { lower_bound: -5.0 }),
            "an absent bound defaults to -5.0, it does not disable the clamp"
        );

        v["linear_attn_config"]["safe_gate"] = serde_json::json!(false);
        assert_eq!(
            arch_from(v).kda_gate_form(),
            Some(KdaGateForm::Softplus),
            "only an explicit `safe_gate: false` reaches the softplus branch"
        );
    }

    /// The MoE spellings, read off the real checkpoint's index — ordinary
    /// `gate_proj`/`up_proj`/`down_proj`, not Kimi's `w1`/`w2`/`w3`.
    #[test]
    fn moe_keys_name_the_real_tensors() {
        let a = arch();
        assert_eq!(
            a.moe_router_key(3).as_deref(),
            Some("layers.3.mlp.gate.weight")
        );
        assert_eq!(
            a.moe_router_bias_key(3).as_deref(),
            Some("layers.3.mlp.gate.e_score_correction_bias")
        );
        assert_eq!(
            a.expert_ffn_gate_key(3, 7).as_deref(),
            Some("layers.3.mlp.experts.7.gate_proj.weight")
        );
        assert_eq!(
            a.expert_ffn_up_key(3, 7).as_deref(),
            Some("layers.3.mlp.experts.7.up_proj.weight")
        );
        assert_eq!(
            a.expert_ffn_down_key(3, 7).as_deref(),
            Some("layers.3.mlp.experts.7.down_proj.weight")
        );
        assert_eq!(
            a.shared_expert_gate_key(3).as_deref(),
            Some("layers.3.mlp.shared_experts.gate_proj.weight")
        );
        assert_eq!(
            a.shared_expert_up_key(3).as_deref(),
            Some("layers.3.mlp.shared_experts.up_proj.weight")
        );
        assert_eq!(
            a.shared_expert_down_key(3).as_deref(),
            Some("layers.3.mlp.shared_experts.down_proj.weight")
        );
    }

    /// The q-LoRA query path Kimi Linear does not have —
    /// `KimiMLAAttention.__init__` asserts `q_lora_rank is None` and its
    /// two q-side key methods stay on the trait default.
    #[test]
    fn mla_declares_the_q_lora_triple_and_the_kv_pair() {
        let a = arch();
        assert!(a.uses_mla());
        assert_eq!(a.q_lora_rank(), 24);
        assert_eq!(a.kv_lora_rank(), 16);
        assert_eq!(
            a.mla_q_a_key(3).as_deref(),
            Some("layers.3.self_attn.q_a_proj.weight")
        );
        assert_eq!(
            a.mla_q_b_key(3).as_deref(),
            Some("layers.3.self_attn.q_b_proj.weight")
        );
        assert_eq!(
            a.mla_kv_a_key(3).as_deref(),
            Some("layers.3.self_attn.kv_a_proj_with_mqa.weight")
        );
        assert_eq!(
            a.mla_kv_b_key(3).as_deref(),
            Some("layers.3.self_attn.kv_b_proj.weight")
        );
        assert_eq!(a.mla_qk_nope_head_dim(), Some(8));
        assert_eq!(a.mla_qk_rope_head_dim(), Some(0), "NoPE: declared zero");
        assert_eq!(a.mla_v_head_dim(), Some(8));
    }

    /// Both MLA norms run at the LAYER epsilon here, where Kimi's latent
    /// norm runs at its class default `1e-6` against a layer eps of
    /// `1e-5`. Stated because the two agreeing is a fact about GLM, not a
    /// property of MLA.
    #[test]
    fn both_mla_norms_run_at_the_layer_epsilon() {
        let a = arch();
        assert_eq!(a.mla_kv_a_norm_eps(), Some(1e-5));
        assert_eq!(a.mla_q_a_norm_eps(), Some(1e-5));
    }

    /// A checkpoint omitting `rms_norm_eps` gets `Glm5NextTextConfig`'s
    /// own `1e-5`, not the crate-wide `1e-6` majority.
    #[test]
    fn an_omitted_norm_eps_defaults_to_this_familys_value() {
        let mut v = config_json();
        v.as_object_mut().expect("object").remove("rms_norm_eps");
        let a = arch_from(v);
        assert_eq!(a.default_norm_eps(), 1e-5);
        assert_eq!(a.norm_eps(), 1e-5);
    }

    /// The clamp is GPT-OSS's; the arithmetic around it is not. Serving
    /// `ClampedGlu` here measured relative 31.7 on the real bank.
    #[test]
    fn the_gate_policy_clamps_but_computes_plain_swiglu() {
        assert_eq!(
            arch().expert_gate_policy(),
            ExpertGatePolicy::ClampedGated { limit: 10.0 }
        );
        let mut v = config_json();
        v.as_object_mut().expect("object").remove("swiglu_limit");
        assert_eq!(
            arch_from(v).expert_gate_policy(),
            ExpertGatePolicy::ClampedGated { limit: 10.0 },
            "an omitted limit takes the config class default, not no clamp"
        );
    }
}
