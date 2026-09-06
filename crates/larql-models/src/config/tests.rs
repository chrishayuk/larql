//! Trait-default behaviour, pinned.
//!
//! An architecture that overrides nothing exercises every default here. The
//! per-architecture files test their own overrides; nothing else pins what a
//! family inherits by *not* overriding, and a changed default silently
//! rewrites the behaviour of every architecture that relied on it.

use super::*;

/// Every [`Activation`] variant has a canonical HF spelling, and the
/// shape vocabulary round-trips through the same two tables in both
/// directions — so the probe that answers "what FFN runs" and the parser
/// that reads `activation: swiglu` cannot drift apart.
#[test]
fn activation_names_round_trip() {
    for activation in [
        Activation::Silu,
        Activation::Gelu,
        Activation::GeluTanh,
        Activation::Relu,
    ] {
        let name = activation.hf_name().expect("every variant has a row");
        assert_eq!(Activation::from_hf_name(name), Some(activation));
        assert_eq!(
            ffn_shape_from_hf_name(name),
            Some((FfnType::Standard, activation)),
            "a plain name is the ungated shape"
        );
        let shape = ffn_shape_hf_name(FfnType::Gated, activation).unwrap();
        match activation.hf_glu_name() {
            Some(glu) => {
                assert_eq!(shape, glu);
                assert_eq!(
                    ffn_shape_from_hf_name(glu),
                    Some((FfnType::Gated, activation))
                );
            }
            None => assert_eq!(shape, format!("gated-{name}")),
        }
    }
    assert_eq!(
        ffn_shape_from_hf_name("SwiGLU"),
        Some((FfnType::Gated, Activation::Silu))
    );
    assert_eq!(
        ffn_shape_from_hf_name("hyena"),
        None,
        "an unjudged spelling is not guessed"
    );
}

/// The architecture that overrides nothing.
struct DefaultsArch(ModelConfig);

impl ModelArchitecture for DefaultsArch {
    fn family(&self) -> &str {
        "defaults-test"
    }
    fn config(&self) -> &ModelConfig {
        &self.0
    }
}

/// A minimal parsed config; tests mutate the returned fields directly
/// rather than guessing config.json spellings (parsing is
/// `detect::parser`'s test surface, not this one's).
fn base_config() -> ModelConfig {
    crate::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 32,
    }))
    .config()
    .clone()
}

#[test]
fn per_layer_ffn_width_is_parsed_verbatim_and_absent_by_default() {
    // Absent: every layer runs at `intermediate_size`.
    assert_eq!(base_config().ffn_intermediate_size_by_layer, None);
    // Declared under `text_config` (multimodal nesting) and at the top
    // level, verbatim; validation belongs to the planner.
    let nested = crate::detect_from_json(&serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 32,
            "larql_ffn_intermediate_size_by_layer": [128, 48, 128]
        }
    }))
    .config()
    .clone();
    assert_eq!(
        nested.ffn_intermediate_size_by_layer,
        Some(vec![128, 48, 128])
    );
    let flat = crate::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 32,
        "larql_ffn_intermediate_size_by_layer": [128, 96]
    }))
    .config()
    .clone();
    assert_eq!(flat.ffn_intermediate_size_by_layer, Some(vec![128, 96]));
}

#[test]
fn optional_weight_keys_default_to_absent() {
    let a = DefaultsArch(base_config());
    assert_eq!(a.position_embed_key(), None);
    assert_eq!(a.fused_qkv_key(0), None);
    assert_eq!(a.fused_qkv_bias_key(0), None);
    assert_eq!(a.attn_sinks_key(0), None);
    assert_eq!(a.ffn_up_bias_key(0), None);
    assert_eq!(a.ffn_down_bias_key(0), None);
    assert_eq!(a.layer_scalar_key(0), None);
    assert_eq!(a.kv_shared_source_layer(0), None);
}

#[test]
fn moe_defaults_describe_a_dense_model() {
    let a = DefaultsArch(base_config());
    assert!(!a.is_moe());
    assert!(!a.is_hybrid_moe());
    assert_eq!(a.num_experts(), 0);
    assert_eq!(a.num_experts_per_token(), 0);
    assert_eq!(a.num_shared_experts(), 0);
    assert_eq!(a.moe_intermediate_size(), 0);
    assert_eq!(a.moe_router_type(), "top_k_softmax");
    assert_eq!(a.expert_format(), ExpertFormat::PerExpert);
    assert_eq!(a.expert_gate_policy(), ExpertGatePolicy::Gated);
    assert_eq!(a.moe_router_key(0), None);
    assert_eq!(a.moe_router_bias_key(0), None);
    assert_eq!(a.moe_router_scale_key(0), None);
    assert_eq!(a.moe_router_per_expert_scale_key(0), None);
    assert_eq!(a.moe_router_norm_key(0), None);
    assert!(!a.moe_router_norm_parameter_free());
    assert_eq!(a.moe_router_input_scalar(), None);
    assert_eq!(a.expert_ffn_gate_key(0, 0), None);
    assert_eq!(a.expert_ffn_up_key(0, 0), None);
    assert_eq!(a.expert_ffn_down_key(0, 0), None);
    assert_eq!(a.shared_expert_gate_key(0), None);
    assert_eq!(a.shared_expert_up_key(0), None);
    assert_eq!(a.shared_expert_down_key(0), None);
    assert_eq!(a.packed_gate_up_blocks_key(0), None);
    assert_eq!(a.packed_gate_up_scales_key(0), None);
    assert_eq!(a.packed_gate_up_bias_key(0), None);
    assert_eq!(a.packed_down_blocks_key(0), None);
    assert_eq!(a.packed_down_scales_key(0), None);
    assert_eq!(a.packed_down_bias_key(0), None);
    assert_eq!(a.packed_experts_gate_up_key(0), None);
    assert_eq!(a.packed_experts_down_key(0), None);
    assert_eq!(a.moe_post_outer_norm_key(0), None);
    assert_eq!(a.moe_post_ffn1_norm_key(0), None);
    assert_eq!(a.moe_pre_experts_norm_key(0), None);
    assert_eq!(a.moe_post_experts_norm_key(0), None);
    assert!(!a.moe_has_combined_output_norm());
}

/// The routing default reads `norm_topk_prob` from the config on purpose
/// (see the method's doc for the two times a baked-in order went wrong):
/// a new MoE architecture must be correct on arrival.
#[test]
fn expert_routing_policy_reads_norm_topk_prob() {
    let mut cfg = base_config();
    cfg.norm_topk_prob = None;
    assert_eq!(
        DefaultsArch(cfg.clone()).expert_routing_policy(),
        ExpertRoutingPolicy::SoftmaxThenSelect
    );
    cfg.norm_topk_prob = Some(false);
    assert_eq!(
        DefaultsArch(cfg.clone()).expert_routing_policy(),
        ExpertRoutingPolicy::SoftmaxThenSelect
    );
    cfg.norm_topk_prob = Some(true);
    assert_eq!(
        DefaultsArch(cfg).expert_routing_policy(),
        ExpertRoutingPolicy::NormalisedOverSelected
    );
}

#[test]
fn mla_defaults_describe_standard_gqa() {
    let a = DefaultsArch(base_config());
    assert!(!a.uses_mla());
    assert_eq!(a.kv_lora_rank(), 0);
    assert_eq!(a.q_lora_rank(), 0);
    assert_eq!(a.mla_kv_a_key(0), None);
    assert_eq!(a.mla_kv_b_key(0), None);
    assert_eq!(a.mla_q_a_key(0), None);
    assert_eq!(a.mla_q_b_key(0), None);
    assert_eq!(a.mla_qk_nope_head_dim(), None);
    assert_eq!(a.mla_qk_rope_head_dim(), None);
    assert_eq!(a.mla_v_head_dim(), None);
}

#[test]
fn scaling_multipliers_default_to_identity() {
    let mut cfg = base_config();
    cfg.residual_multiplier = None;
    cfg.attention_multiplier = None;
    cfg.logits_scaling = None;
    let a = DefaultsArch(cfg.clone());
    assert_eq!(a.residual_multiplier(), 1.0);
    assert_eq!(a.attention_multiplier(), 1.0);
    assert_eq!(a.logits_scaling(), 1.0);

    cfg.residual_multiplier = Some(0.5);
    cfg.attention_multiplier = Some(0.25);
    cfg.logits_scaling = Some(8.0);
    let a = DefaultsArch(cfg);
    assert_eq!(a.residual_multiplier(), 0.5);
    assert_eq!(a.attention_multiplier(), 0.25);
    assert_eq!(a.logits_scaling(), 8.0);
}

#[test]
fn attention_scale_prefers_query_pre_attn_scalar_over_head_dim() {
    let mut cfg = base_config();
    cfg.query_pre_attn_scalar = None;
    let a = DefaultsArch(cfg.clone());
    let by_head_dim = (cfg.head_dim as f64).powf(-0.5);
    assert_eq!(a.attention_scale(), by_head_dim);
    assert_eq!(a.attention_scale_for_layer(0), by_head_dim);

    cfg.query_pre_attn_scalar = Some(256.0);
    let a = DefaultsArch(cfg);
    assert_eq!(a.attention_scale(), 256.0f64.powf(-0.5));
    assert_eq!(a.attention_scale_for_layer(1), 256.0f64.powf(-0.5));
}

/// PLE keys are all-or-nothing on `per_layer_embed_dim`: a partial key
/// set would load half a mechanism.
#[test]
fn ple_keys_are_all_present_or_all_absent() {
    let mut cfg = base_config();
    cfg.per_layer_embed_dim = None;
    let off = DefaultsArch(cfg.clone());
    assert!(!off.has_per_layer_embeddings());
    assert_eq!(off.per_layer_embed_dim(), 0);
    assert_eq!(off.per_layer_embed_key(), None);
    assert_eq!(off.per_layer_model_projection_key(), None);
    assert_eq!(off.per_layer_projection_norm_key(), None);
    assert_eq!(off.per_layer_input_gate_key(0), None);
    assert_eq!(off.per_layer_projection_key(0), None);
    assert_eq!(off.post_per_layer_input_norm_key(0), None);

    cfg.per_layer_embed_dim = Some(8);
    let on = DefaultsArch(cfg);
    assert!(on.has_per_layer_embeddings());
    assert_eq!(on.per_layer_embed_dim(), 8);
    assert_eq!(
        on.per_layer_embed_key().as_deref(),
        Some("embed_tokens_per_layer.weight")
    );
    assert_eq!(
        on.per_layer_model_projection_key().as_deref(),
        Some("per_layer_model_projection.weight")
    );
    assert_eq!(
        on.per_layer_projection_norm_key().as_deref(),
        Some("per_layer_projection_norm.weight")
    );
    let prefix = on.layer_prefix(1);
    assert_eq!(
        on.per_layer_input_gate_key(1),
        Some(format!("{prefix}per_layer_input_gate.weight"))
    );
    assert_eq!(
        on.per_layer_projection_key(1),
        Some(format!("{prefix}per_layer_projection.weight"))
    );
    assert_eq!(
        on.post_per_layer_input_norm_key(1),
        Some(format!("{prefix}post_per_layer_input_norm.weight"))
    );
}

#[test]
fn softcapping_reads_config_and_defaults_off() {
    let mut cfg = base_config();
    cfg.attn_logit_softcapping = None;
    cfg.final_logit_softcapping = None;
    let a = DefaultsArch(cfg.clone());
    assert_eq!(a.attn_logit_softcapping(), None);
    assert_eq!(a.final_logit_softcapping(), None);

    cfg.attn_logit_softcapping = Some(50.0);
    cfg.final_logit_softcapping = Some(30.0);
    let a = DefaultsArch(cfg);
    assert_eq!(a.attn_logit_softcapping(), Some(50.0));
    assert_eq!(a.final_logit_softcapping(), Some(30.0));
}

#[test]
fn rope_scaling_defaults_and_config_read() {
    let mut cfg = base_config();
    cfg.rope_scaling = None;
    let a = DefaultsArch(cfg.clone());
    assert_eq!(a.rope_scaling_type(), None);
    assert_eq!(a.rope_scaling_factor(), 1.0);
    assert_eq!(a.rope_position_divisor_for_layer(0), 1.0);
    assert!(a.llama3_rope_scaling().is_none());

    cfg.rope_scaling = Some(RopeScaling {
        scaling_type: "linear".to_string(),
        factor: 8.0,
        llama3_low_freq_factor: None,
        llama3_high_freq_factor: None,
        llama3_original_max_position_embeddings: None,
        yarn_beta_fast: None,
        yarn_beta_slow: None,
        yarn_truncate: None,
        yarn_mscale: None,
        yarn_mscale_all_dim: None,
        gemma3_global_only: false,
    });
    let a = DefaultsArch(cfg);
    assert_eq!(a.rope_scaling_type(), Some("linear"));
    assert_eq!(a.rope_scaling_factor(), 8.0);
}

/// `openai/gpt-oss-20b`'s block verbatim. The scaling type is the only
/// selector, so this doubles as the pin that an architecture needs **no**
/// override to be served under YaRN — the §4.7.8 mechanism fix.
fn yarn_scaling() -> RopeScaling {
    RopeScaling {
        scaling_type: "yarn".to_string(),
        factor: 32.0,
        llama3_low_freq_factor: None,
        llama3_high_freq_factor: None,
        llama3_original_max_position_embeddings: Some(4096.0),
        yarn_beta_fast: Some(32.0),
        yarn_beta_slow: Some(1.0),
        yarn_truncate: Some(false),
        yarn_mscale: None,
        yarn_mscale_all_dim: None,
        gemma3_global_only: false,
    }
}

#[test]
fn yarn_is_read_from_config_without_an_architecture_override() {
    let mut cfg = base_config();
    cfg.rope_scaling = Some(yarn_scaling());
    let s = DefaultsArch(cfg)
        .yarn_rope_scaling()
        .expect("a yarn config must resolve on the trait default alone");
    assert_eq!(s.factor, 32.0);
    assert_eq!(s.beta_fast, 32.0);
    assert_eq!(s.beta_slow, 1.0);
    assert_eq!(s.original_max_position_embeddings, 4096.0);
    assert!(!s.truncate, "gpt-oss ships truncate: false");
}

/// A non-yarn block must not be answered by the yarn accessor — otherwise
/// every linear-scaled model would silently acquire an amplitude.
#[test]
fn non_yarn_scaling_types_do_not_resolve_as_yarn() {
    for ty in ["linear", "llama3", "dynamic", "default"] {
        let mut cfg = base_config();
        cfg.rope_scaling = Some(RopeScaling {
            scaling_type: ty.to_string(),
            ..yarn_scaling()
        });
        assert!(
            DefaultsArch(cfg).yarn_rope_scaling().is_none(),
            "rope_type {ty} must not resolve as yarn"
        );
    }
}

#[test]
fn absent_rope_scaling_is_not_yarn() {
    let mut cfg = base_config();
    cfg.rope_scaling = None;
    assert!(DefaultsArch(cfg).yarn_rope_scaling().is_none());
}

/// `original_max_position_embeddings` is the window YaRN's correction
/// bounds are defined against; HF indexes it unconditionally and raises
/// when absent. Guessing one would silently serve a *different* ramp, so
/// a block without it resolves to `None` rather than to a plausible
/// number.
#[test]
fn yarn_without_original_context_is_not_scaling() {
    let mut cfg = base_config();
    cfg.rope_scaling = Some(RopeScaling {
        llama3_original_max_position_embeddings: None,
        ..yarn_scaling()
    });
    assert!(DefaultsArch(cfg).yarn_rope_scaling().is_none());
}

/// Absent band bounds fall back to the paper's defaults, matching HF's
/// `beta_fast or 32` / `beta_slow or 1`, and absent `truncate` to HF's
/// `true`. A partial yarn block is still fully determined.
#[test]
fn partial_yarn_block_falls_back_to_hf_class_defaults() {
    let mut cfg = base_config();
    cfg.rope_scaling = Some(RopeScaling {
        yarn_beta_fast: None,
        yarn_beta_slow: None,
        yarn_truncate: None,
        ..yarn_scaling()
    });
    let s = DefaultsArch(cfg).yarn_rope_scaling().unwrap();
    assert_eq!(s.beta_fast, crate::defaults::YARN_BETA_FAST);
    assert_eq!(s.beta_slow, crate::defaults::YARN_BETA_SLOW);
    assert_eq!(s.truncate, crate::defaults::YARN_TRUNCATE);
}

/// DeepSeek's paired mscales must survive parsing to the compute layer;
/// dropping them there is what would make the amplitude 1.35 instead of
/// 1.0 for that family.
#[test]
fn paired_mscales_reach_the_scaling_struct() {
    let mut cfg = base_config();
    cfg.rope_scaling = Some(RopeScaling {
        yarn_mscale: Some(1.0),
        yarn_mscale_all_dim: Some(1.0),
        ..yarn_scaling()
    });
    let s = DefaultsArch(cfg).yarn_rope_scaling().unwrap();
    assert_eq!(s.mscale, Some(1.0));
    assert_eq!(s.mscale_all_dim, Some(1.0));
}

#[test]
fn norm_eps_reads_config_with_crate_fallback() {
    let mut cfg = base_config();
    cfg.norm_eps = Some(1e-5);
    assert_eq!(DefaultsArch(cfg.clone()).norm_eps(), 1e-5);
    cfg.norm_eps = None;
    assert_eq!(
        DefaultsArch(cfg).norm_eps(),
        crate::defaults::DEFAULT_NORM_EPS
    );
}

#[test]
fn misc_defaults() {
    let a = DefaultsArch(base_config());
    assert!(a.multimodal().is_none());
    assert_eq!(a.sliding_window_size(), None);
    assert!(!a.is_sliding_window_layer(0));
}

/// The markov-residual precondition: stateless norm, no learned position
/// table, no MLA — all true of the default architecture surface.
#[test]
fn kv_recomputable_from_residuals_by_default() {
    assert!(DefaultsArch(base_config()).kv_recomputable_from_residuals());
}

// ── Activation::uses_gelu_tanh_gate_up — the ONE gate/up kernel
//    mapping (2026-07-30 review §4 dedupe) ─────────────────────────

#[test]
fn gelu_family_maps_to_gelu_tanh_kernel() {
    assert!(Activation::GeluTanh.uses_gelu_tanh_gate_up());
    // Exact GELU is served by the tanh approximation — documented.
    assert!(Activation::Gelu.uses_gelu_tanh_gate_up());
}

#[test]
fn silu_maps_to_silu_kernel() {
    assert!(!Activation::Silu.uses_gelu_tanh_gate_up());
}

/// A variant with no gate/up kernel must fail LOUDLY, never
/// silently compute SiLU numerics. Together with the helper's
/// wildcard-free match (a new `Activation` variant is a compile
/// error there), this pins the review requirement that a
/// hypothetical new activation cannot silently land in the SiLU arm.
#[test]
#[should_panic(expected = "no gate/up FFN kernel")]
fn relu_panics_instead_of_silently_running_silu() {
    Activation::Relu.uses_gelu_tanh_gate_up();
}

// ── ActivationDeclaration and the refusing kernel-family accessor
//    (K3-ACT-1) ──────────────────────────────────────────────────────
//
// The defect these close: `activation()` used to answer SiLU to BOTH "no
// declaration" and "a declaration this build cannot read", so a
// checkpoint declaring `situ` or `relu2` was executed as SwiGLU. Each
// test below names one of the four states, so a future collapse of two
// of them fails here.

/// An architecture over the default config, declaring one activation.
fn arch_declaring(hidden_act: Option<&str>) -> DefaultsArch {
    let mut config = base_config();
    config.hidden_act = hidden_act.map(str::to_string);
    DefaultsArch(config)
}

#[test]
fn a_silent_config_takes_the_documented_silu_default() {
    let arch = arch_declaring(None);
    assert_eq!(arch.activation_declaration(), ActivationDeclaration::Absent);
    assert_eq!(arch.activation(), Activation::Silu);
    assert!(
        !arch.gate_up_is_gelu_tanh(),
        "silence is the one state the SiLU default is for"
    );
}

#[test]
fn a_judged_name_maps_through_the_one_table() {
    let arch = arch_declaring(Some("gelu_pytorch_tanh"));
    assert_eq!(
        arch.activation_declaration(),
        ActivationDeclaration::Nonlinearity(Activation::GeluTanh)
    );
    assert!(arch.gate_up_is_gelu_tanh());
}

#[test]
fn situ_names_a_gate_policy_not_a_nonlinearity() {
    let arch = arch_declaring(Some("situ"));
    assert_eq!(
        arch.activation_declaration(),
        ActivationDeclaration::NamesGatePolicy(SITU_NAME)
    );
    assert!(
        matches!(arch.expert_gate_policy(), ExpertGatePolicy::SituGlu { .. }),
        "the name must resolve to the policy, not to a nonlinearity"
    );
}

#[test]
fn an_unjudged_name_is_kept_verbatim_and_not_mapped() {
    // BitNet's spelling. `from_hf_name` correctly refuses it; the point
    // of the enum is that the refusal survives to the caller instead of
    // being flattened into the silent default.
    assert_eq!(
        arch_declaring(Some("relu2")).activation_declaration(),
        ActivationDeclaration::Unjudged("relu2".to_string())
    );
    assert_eq!(Activation::from_hf_name("relu2"), None);
}

/// **The class this rung closes.** A declared name this build has never
/// judged must not select a gate/up kernel at all.
#[test]
#[should_panic(expected = "never judged")]
fn an_unjudged_activation_refuses_the_gate_up_kernel_by_name() {
    arch_declaring(Some("relu2")).gate_up_is_gelu_tanh();
}

/// A combine that is not plain gating has no gate/up kernel on these
/// paths either — the bool cannot express a third family, so it refuses
/// rather than answering for one of the two it can.
#[test]
#[should_panic(expected = "SituGlu")]
fn a_situ_policy_refuses_the_gate_up_kernel_by_name() {
    arch_declaring(Some("situ")).gate_up_is_gelu_tanh();
}

/// The HF spelling of the combine a component computes — the function
/// that decides whether a correctly-carried FFN *reports* as carried.
///
/// All three arms, because each one answers a different question the plan
/// asks: a SiTU policy owns the whole combine and answers with its own
/// name; a plain policy answers with its nonlinearity's; and ClampedGlu
/// answers with NOTHING, which is why GPT-OSS's row keeps its existing
/// behaviour rather than acquiring a resolution no HF word supports.
#[test]
fn the_combine_name_is_the_policys_when_the_policy_owns_the_combine() {
    assert_eq!(
        hf_combine_name(
            ExpertGatePolicy::SituGlu {
                beta: 4.0,
                linear_beta: Some(25.0)
            },
            Activation::Silu
        )
        .as_deref(),
        Some(SITU_NAME),
        "a SiTU FFN answers `situ` whatever inert activation sits beside it"
    );
    assert_eq!(
        hf_combine_name(
            ExpertGatePolicy::SituGlu {
                beta: 1.0,
                linear_beta: None
            },
            Activation::GeluTanh
        )
        .as_deref(),
        Some(SITU_NAME),
        "and the parameters do not change the NAME of the combine"
    );
    assert_eq!(
        hf_combine_name(ExpertGatePolicy::Gated, Activation::GeluTanh).as_deref(),
        Activation::GeluTanh.hf_name(),
        "plain gating answers with the gate's nonlinearity"
    );
    assert_eq!(
        hf_combine_name(ExpertGatePolicy::Gated, Activation::Silu).as_deref(),
        Some("silu")
    );
    assert_eq!(
        hf_combine_name(
            ExpertGatePolicy::ClampedGlu {
                limit: 7.0,
                alpha: 1.702
            },
            Activation::Silu
        ),
        None,
        "no HF word names GPT-OSS's clamped GLU, so there is nothing to answer with — \
         the arm that keeps that row on its existing code path"
    );
}

/// A judged declaration names itself with its CANONICAL spelling, not the
/// alias the checkpoint happened to use.
///
/// The other three arms are covered beside the refusals; this one is the
/// success path, which is the arm a gate is most often missing
/// (`feedback_untested_gate_success_path`).
#[test]
fn a_judged_declaration_names_itself_canonically() {
    assert_eq!(
        arch_declaring(Some("gelu_pytorch_tanh"))
            .activation_declaration()
            .declared_name(),
        Activation::GeluTanh.hf_name(),
    );
    assert_eq!(
        arch_declaring(Some("swish"))
            .activation_declaration()
            .declared_name(),
        Some("silu"),
        "an alias resolves to the variant's own first spelling"
    );
}

/// **A family override of `activation()` decides the kernel, not the
/// config the family ignored.**
///
/// StarCoder2 declares no `hidden_act` and overrides `activation()` to
/// tanh-GELU. The first version of `gate_up_is_gelu_tanh` answered from
/// the DECLARATION, so it read `Absent`, fell to SiLU, and quietly
/// replaced a family's own judgment — the same shape of defect this rung
/// removes, one level up, and it was caught by the walk-vs-dense parity
/// test rather than by design. The declaration decides whether to REFUSE;
/// `activation()` supplies the answer.
#[test]
fn a_family_override_decides_the_gate_up_kernel() {
    let arch = crate::detect_from_json(&serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "vocab_size": 32,
    }));
    assert_eq!(
        arch.activation_declaration(),
        ActivationDeclaration::Absent,
        "the fixture must declare nothing, or this proves the wrong thing"
    );
    assert_eq!(arch.activation(), Activation::GeluTanh, "the family's own");
    assert!(
        arch.gate_up_is_gelu_tanh(),
        "the kernel family must follow the override, not the absent declaration"
    );
}

/// The declared name survives into the refusal, so a reader is told WHICH
/// declaration was refused rather than that some declaration was.
#[test]
fn a_declaration_can_name_itself_for_a_refusal_message() {
    assert_eq!(
        arch_declaring(Some("relu2"))
            .activation_declaration()
            .declared_name(),
        Some("relu2")
    );
    assert_eq!(
        arch_declaring(Some("situ"))
            .activation_declaration()
            .declared_name(),
        Some("situ")
    );
    assert_eq!(
        arch_declaring(None)
            .activation_declaration()
            .declared_name(),
        None
    );
}

// ── tie_word_embeddings ──────────────────────────────────────────────
//
// Parsed as a *check* on the loader's tie-on-absence behaviour, not as a
// shortcut. See `loading/safetensors.rs`.

#[test]
fn tie_word_embeddings_is_read_from_the_config() {
    let cfg = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 16, "tie_word_embeddings": false,
    }));
    assert_eq!(cfg.config().tie_word_embeddings, Some(false));
}

#[test]
fn tie_word_embeddings_true_is_distinguished_from_absent() {
    let with_true = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 16, "tie_word_embeddings": true,
    }));
    assert_eq!(with_true.config().tie_word_embeddings, Some(true));

    let absent = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 16,
    }));
    assert_eq!(
        absent.config().tie_word_embeddings,
        None,
        "absent must stay None — it is not a claim that the model is tied"
    );
}

/// GPT-OSS declares `false` at the top level, not inside `text_config`.
#[test]
fn tie_word_embeddings_is_read_from_the_outer_config_too() {
    let cfg = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 16,
        },
    }));
    assert_eq!(cfg.config().tie_word_embeddings, Some(false));
}

/// `Shared` and `Value` are different claims, and `resolve` is the only
/// place either becomes a number.
#[test]
fn post_norm_eps_resolves_shared_and_distinct_differently() {
    use crate::config::PostNormEps;
    const PRE: f64 = 1e-5;
    // Sharing takes the pre-norm epsilon it is handed — never one it
    // read from somewhere else.
    assert_eq!(PostNormEps::Shared.resolve(PRE), PRE);
    assert_eq!(PostNormEps::Shared.resolve(1e-12), 1e-12);
    // A declared value ignores the pre-norm epsilon entirely.
    assert_eq!(PostNormEps::Value(1e-8).resolve(PRE), 1e-8);
    assert_ne!(PostNormEps::Value(1e-8).resolve(PRE), PRE);
}

/// The four-norm Gemma families state the sharing judgment rather than
/// leaving it to be inherited — the state VINDEX3 refuses.
#[test]
fn four_norm_gemma_families_declare_shared_post_norm_eps() {
    use crate::config::PostNormEps;
    for family in ["gemma2", "gemma3"] {
        let arch = crate::detect::detect_from_json(&serde_json::json!({
            "model_type": family,
            "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 16,
        }));
        assert!(arch.has_post_norms(), "{family} is a four-norm family");
        assert_eq!(
            arch.post_norm_eps(),
            Some(PostNormEps::Shared),
            "{family} must declare sharing, not leave it unjudged"
        );
    }
}
