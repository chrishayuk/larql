use super::*;

/// Inventory guard for the lossy projection this struct performs.
///
/// `larql_models::ModelConfig` is the parsed checkpoint. Anything in
/// it that reaches the forward pass must either appear here or be
/// listed below with the reason it does not need to. The list is the
/// point: `rope_scaling` was missing for as long as this file
/// existed and nothing failed, because nothing was counting.
///
/// If this test fails you added a field to `ModelConfig`. Decide
/// which bucket it is in — do not just add it to the exempt list to
/// get green.
#[test]
fn model_config_persists_every_forward_affecting_field() {
    // Carried elsewhere in `VindexConfig`, not in `model_config`.
    const CARRIED_AT_TOP_LEVEL: &[&str] = &[
        "num_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        // Round-trips as `VindexConfig.embed_scale`; duplicating it
        // here would give one number two sources of truth.
        "embedding_multiplier",
    ];
    // Carried inside the nested `moe` object.
    const CARRIED_IN_MOE: &[&str] = &[
        "num_experts",
        "num_experts_per_token",
        "num_shared_experts",
        "shared_expert_intermediate_size",
        "enable_moe_block",
        "top_k_experts",
        "moe_intermediate_size",
        // K3-LATENTMOE-1: both reach the surface through
        // `MoeExecution.routed_expert_form` — the width as the
        // variant that selects the latent branch, the flag as the
        // norm nested inside it. Carried as a FORM rather than two
        // fields, so a norm without a width cannot be expressed.
        "routed_expert_hidden_size",
        "latent_moe_use_norm",
    ];
    // Read only to RESOLVE another field that IS carried, and
    // deliberately not persisted itself: persisting both the input and
    // the conclusion would give one fact two sources of truth, and a
    // reader would have to know which one the executor honours.
    const RESOLVED_INTO_ANOTHER_FIELD: &[&str] = &[
        // `linear_attn_config.safe_gate` is an input to
        // `ModelArchitecture::kda_gate_form`, whose CONCLUSION —
        // `ExecutionSurface.kda_gate_form`, and `KdaOp.gate_form` —
        // is what the container carries and the executor reads. The
        // form is the forward-affecting fact; `safe_gate` is one of
        // two config values the family combines to reach it.
        "kda_safe_gate",
    ];
    // Genuinely not persisted yet. Each entry is a known gap, not an
    // exemption: no vindex-served model can use these today.
    const KNOWN_GAPS: &[&str] = &[
        // Hyper-connections. The residual topology is REPRESENTED on
        // the execution surface (wave 16) and explicitly not
        // executable, so no vindex-served model can use it — and
        // persisting it in the model config would claim a serving
        // path that refuses. It moves here when wave 17 lowers it.
        "hc_streams",
        "hc_sinkhorn_iters",
        "hc_eps",
        // Attention residuals, and the same status for a sharper
        // reason: the VINDEX3 execution surface carries the declared
        // period (K3-ATTNRES-1) and the executor refuses the topology
        // by name at preparation, so no vindex-served model can use
        // it. Persisting the period in the legacy model config would
        // claim a serving path that refuses.
        "attn_res_block_size",
        // The two K3 attention output gates (K3-REP-GATE-1): the KDA
        // gate's FORM and MLA's gate reach VINDEX3 through the
        // execution surface, never through this legacy round-trip.
        "use_full_rank_gate",
        "mla_use_output_gate",
        // Multi-head latent attention (DeepSeek V2/V3). No MLA model
        // is served from a vindex yet; serving one without these
        // would silently rebuild the wrong attention geometry.
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        // GLM-5.2's DSA indexer, which sits on top of MLA. It is
        // registered extraction-tier only: no compute path implements
        // its top-k token selection, so no vindex-served model can use
        // these, and a served DSA model would already miss the MLA
        // geometry above.
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        // Vision tower presence. The multimodal path loads its own
        // config rather than reconstructing from the vindex.
        "has_vision_config",
        // Multimodal protocol + adapter geometry + drafter interface.
        // These describe cross-component structure — which token ids
        // stand in for other modalities, what the adapter projects,
        // which target layers a drafter taps. Their home is the
        // VINDEX3 system graph (format::vindex3, G2c), which carries
        // components and interface edges explicitly; duplicating them
        // in the per-model config would put the system topology in
        // two places. No vindex-served model consumes them today.
        "image_token_id",
        "video_token_id",
        "out_hidden_size",
        "projector_hidden_size",
        "projector_hidden_act",
        "target_layer_ids",
        "draft_block_size",
        "mask_token_id",
        // FFN/MLP bias terms. Same status as `attention_bias` had
        // before it got a real field, except this legacy path's
        // Q4K writer has no bias-tensor handling for the FFN
        // projections at all yet — every checkpoint on hand (Granite
        // 4.1 3B/8B/30B included) declares `false`, so no vindex-served
        // model needs this today.
        "mlp_bias",
        // GPT-2's lineage. `declares_rotary` is a parse-time fact whose
        // only consumer is the position policy of a family with a learned
        // table, which a reload without it answers as GPT-2's truth (no
        // rotation). The four attention switches were ignored by this
        // path before they were read at all; VINDEX3 carries
        // `scale_attn_weights` through the score scale and refuses the
        // other three unless they are `false`.
        "declares_rotary",
        "scale_attn_weights",
        "scale_attn_by_inverse_layer_idx",
        "reorder_and_upcast_attn",
        "add_cross_attention",
        // Hybrid linear-attention + multi-token-prediction geometry
        // (Qwen3.5/Kimi-Linear-style — R2/Kimi-Linear rung,
        // docs/k3-funnel.md). No forward pass anywhere in this crate,
        // VINDEX1/2 or VINDEX3, executes a linear-attention layer or
        // an MTP head yet — there is no served model this gap could
        // silently mis-serve. `larql vindex3 plan` reports every one
        // of these `unrepresented` (see
        // `format::vindex3::plan::semantics::EXECUTION_SEMANTIC_KEYS`)
        // rather than answering for it, which is what this list
        // exists to force a decision about.
        // The interleave itself, in the index-set spelling. Same
        // status as the geometry beside it — VINDEX1/2 cannot execute
        // a recurrent layer, so a container that dropped it could not
        // mis-serve one. It reaches VINDEX3 through the resolved
        // per-layer table (`LayerPolicy::declared_span`), not through
        // this legacy round-trip.
        "linear_attn_interleave",
        // The hybrid Mamba2Attn estate (OuteAI): the conv-QKV
        // attention geometry and the mamba_ssm lineage's MLP
        // declaration (`mlp_intermediate_size: 0` = no MLP blocks,
        // plus its padding/bias parameters). Same status as the
        // linear-attention geometry above — VINDEX1/2 cannot execute
        // a hybrid layer, so a container dropping these could not
        // mis-serve one. They reach VINDEX3 through
        // `ExecutionSurface.conv_qkv` and the per-layer operator
        // table.
        "conv_qkv_attn",
        "conv_qkv_provenance",
        "attn_causal",
        "pad_vocab_size_multiple",
        "fused_add_norm",
        "mlp_intermediate_size",
        "mlp_padding_size",
        "use_mlp_bias",
        // The MTP sub-stack's interleave. VINDEX1/2 has no MTP object
        // at all, so a container dropping it could not mis-serve one.
        "mtp_interleave",
        // The relative-position scheme. VINDEX1/2's forward rotates or
        // does not; it has no relative term, so a container dropping
        // these could not serve one wrongly — it could not serve one
        // at all. VINDEX3 carries it on the per-layer position policy.
        // Router scoring function, carried verbatim beside the typed
        // kind. VINDEX1/2 dispatches on the typed kind.
        "router_activation",
        // Declared MoE facts VINDEX1/2 has no field for. Its MoE
        // config carries expert count, top-k and the router type; a
        // branch scale, expert grouping, a dense prefix or a cadence
        // period would each change the forward and none has a home
        // here. Read so the plan can judge them, not so this path can
        // serve them.
        "routed_scaling_factor",
        "expert_groups",
        "topk_group",
        "use_grouped_topk",
        "moe_layer_freq",
        "first_k_dense_replace",
        "mla_use_nope",
        "model_max_length",
        "d_rel",
        "rel_extent",
        // KDA's geometry and decay clamp. Same status: VINDEX1/2
        // cannot execute a recurrence, so a container dropping these
        // could not mis-serve one. They reach VINDEX3 through the
        // execution surface, not through this legacy round-trip.
        "kda_geometry",
        "kda_gate_lower_bound",
        "kda_use_full_rank_gate",
        "mla_use_output_gate",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "mamba_ssm_dtype",
        "attn_output_gate",
        "output_gate_type",
        "mtp_num_hidden_layers",
        "mtp_use_dedicated_embeddings",
        "mrope_interleaved",
        "mrope_section",
    ];

    let src = include_str!("../../../../larql-models/src/config/model_config.rs");
    let start = src
        .find("pub struct ModelConfig")
        .expect("ModelConfig struct not found — did the file move?");
    let body = &src[start..src[start..].find("\n}").unwrap() + start];
    let model_fields: Vec<&str> = body
        .lines()
        .filter_map(|l| l.trim().strip_prefix("pub "))
        .filter_map(|l| l.split(':').next())
        .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
        .collect();
    assert!(
        model_fields.len() > 30,
        "parsed only {} ModelConfig fields — the scraper broke, which \
         would make this guard silently vacuous",
        model_fields.len()
    );

    // Scrape this struct from source rather than serialising an
    // instance: every optional field carries
    // `skip_serializing_if = "Option::is_none"`, so a `None`-valued
    // instance serialises to almost nothing and the guard would
    // report the entire struct as missing.
    let own = include_str!("../model.rs");
    let vstart = own
        .find("pub struct VindexModelConfig")
        .expect("VindexModelConfig struct not found");
    let vbody = &own[vstart..own[vstart..].find("\n}").unwrap() + vstart];
    let persisted: Vec<&str> = vbody
        .lines()
        .filter_map(|l| l.trim().strip_prefix("pub "))
        .filter_map(|l| l.split(':').next())
        .filter(|n| !n.is_empty())
        .collect();
    assert!(
        persisted.len() > 20,
        "parsed only {} VindexModelConfig fields — the scraper broke",
        persisted.len()
    );

    let mut unaccounted = Vec::new();
    for f in &model_fields {
        let known = persisted.contains(f)
            || CARRIED_AT_TOP_LEVEL.contains(f)
            || CARRIED_IN_MOE.contains(f)
            || RESOLVED_INTO_ANOTHER_FIELD.contains(f)
            || KNOWN_GAPS.contains(f)
            // `model_type` / geometry share names across both structs.
            || ["model_type", "head_dim", "num_q_heads", "num_kv_heads",
                "rope_base", "sliding_window", "norm_eps"].contains(f);
        if !known {
            unaccounted.push(*f);
        }
    }
    assert!(
        unaccounted.is_empty(),
        "ModelConfig fields with no home in the vindex round-trip: {unaccounted:?}. \
         A checkpoint declaring one of these is served without it — the defect is \
         invisible to CPU-vs-Metal parity because both arms read the same index.json."
    );
}

fn minimal_model_config() -> VindexModelConfig {
    VindexModelConfig {
        model_type: "gemma3".into(),
        head_dim: 256,
        num_q_heads: 8,
        num_kv_heads: 4,
        rope_base: 10000.0,
        sliding_window: None,
        use_sliding_window: None,
        max_window_layers: None,
        position_embedding_type: None,
        no_rope_layers: None,
        no_rope_layer_interval: None,
        rope_interleaved: None,
        use_mrope: None,
        ffn_shape_name: None,
        is_llama_config: None,
        moe: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        use_double_wide_mlp: None,
        vocab_size_per_layer_input: None,
        layer_rope_theta: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        norm_eps: None,
        ..Default::default()
    }
}

#[test]
fn model_config_serde_round_trip() {
    let cfg = minimal_model_config();
    let j = serde_json::to_string(&cfg).unwrap();
    let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
    assert_eq!(back.model_type, "gemma3");
    assert_eq!(back.head_dim, 256);
    assert_eq!(back.num_q_heads, 8);
    assert_eq!(back.num_kv_heads, 4);
}

#[test]
fn optional_fields_absent_in_json_when_none() {
    let cfg = minimal_model_config();
    let j = serde_json::to_string(&cfg).unwrap();
    assert!(
        !j.contains("global_head_dim"),
        "None optional should be omitted"
    );
    assert!(
        !j.contains("sliding_window_pattern"),
        "None optional should be omitted"
    );
}

#[test]
fn model_config_with_softcap_round_trips() {
    let mut cfg = minimal_model_config();
    cfg.final_logit_softcapping = Some(30.0);
    let j = serde_json::to_string(&cfg).unwrap();
    let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
    assert_eq!(back.final_logit_softcapping, Some(30.0));
}

#[test]
fn model_config_with_moe() {
    let mut cfg = minimal_model_config();
    cfg.moe = Some(MoeConfig {
        num_experts: 8,
        top_k: 2,
        shared_expert: false,
        shared_expert_intermediate_size: None,
        router_type: "top_k_softmax".into(),
        moe_intermediate_size: Some(2048),
        hybrid: false,
    });
    let j = serde_json::to_string(&cfg).unwrap();
    let back: VindexModelConfig = serde_json::from_str(&j).unwrap();
    let moe = back.moe.unwrap();
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);
}

#[test]
fn moe_config_default_router_type_via_serde() {
    let json = r#"{"num_experts":4,"top_k":1,"shared_expert":false}"#;
    let moe: MoeConfig = serde_json::from_str(json).unwrap();
    assert_eq!(moe.router_type, "top_k_softmax");
    assert!(!moe.hybrid);
}

#[test]
fn moe_shared_expert_default_false() {
    let json = r#"{"num_experts":4,"top_k":2,"router_type":"custom"}"#;
    let moe: MoeConfig = serde_json::from_str(json).unwrap();
    assert!(!moe.shared_expert);
    assert!(!moe.hybrid);
}

#[test]
fn granite_scalars_round_trip_through_from_arch() {
    // Granite 4.1 3B exact config. The four scalars must survive
    // arch detect → from_arch → JSON → deserialize so the vindex
    // load path can hand them back to the forward pass.
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "granite",
        "hidden_size": 2560,
        "num_hidden_layers": 40,
        "intermediate_size": 8192,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "attention_multiplier": 0.015625,
        "embedding_multiplier": 12.0,
        "logits_scaling": 10.0,
        "residual_multiplier": 0.22,
    }));
    let vc = VindexModelConfig::from_arch(&*arch);
    assert_eq!(vc.attention_multiplier, Some(0.015625));
    assert_eq!(vc.residual_multiplier, Some(0.22));
    assert_eq!(vc.logits_scaling, Some(10.0));
    assert_eq!(vc.norm_eps, Some(1e-05));

    let json = serde_json::to_string(&vc).unwrap();
    // All four must serialise (regression: an earlier vindex format
    // dropped them silently, so Granite 4.1 vindexes loaded with
    // multipliers defaulted to 1.0 and the model emitted garbage).
    assert!(json.contains("\"attention_multiplier\":0.015625"), "{json}");
    assert!(json.contains("\"residual_multiplier\":0.22"), "{json}");
    assert!(json.contains("\"logits_scaling\":10.0"), "{json}");
    // `serde_json::to_string` emits this f64 as `0.00001`, not
    // `1e-5`; numeric equality (not text equality) is what matters.
    assert!(json.contains("\"norm_eps\":0.00001"), "{json}");

    let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.attention_multiplier, Some(0.015625));
    assert_eq!(back.residual_multiplier, Some(0.22));
    assert_eq!(back.logits_scaling, Some(10.0));
    assert_eq!(back.norm_eps, Some(1e-05));
}

#[test]
fn moe_arch_populates_moe_field_via_from_arch() {
    // Mixtral exercises the `if arch.is_moe()` Some-branch in
    // from_arch — the Granite/Llama tests above only hit the
    // None-branch.
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }));
    let vc = VindexModelConfig::from_arch(&*arch);
    let moe = vc.moe.expect("MoE arch must populate moe field");
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);
}

#[test]
fn gemma4_a4b_hybrid_moe_populates_intermediate_size_and_hybrid() {
    // Gemma 4 A4B is hybrid MoE with a distinct
    // moe_intermediate_size. Hits the from_arch branches:
    //   - `Some(arch.moe_intermediate_size())` (the > 0 path)
    //   - `hybrid: arch.is_hybrid_moe()` returning true
    //   - `router_type: arch.moe_router_type().into()` for the
    //     non-default gemma4 router
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": 2048,
        "num_hidden_layers": 30,
        "intermediate_size": 8192,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "enable_moe_block": true,
    }));
    let vc = VindexModelConfig::from_arch(&*arch);
    {
        let moe = vc.moe.as_ref().expect("Gemma 4 A4B must be MoE");
        assert_eq!(moe.num_experts, 128);
        assert_eq!(moe.top_k, 8);
        assert_eq!(moe.moe_intermediate_size, Some(768));
        assert!(moe.hybrid, "Gemma 4 A4B is hybrid MoE");
    }

    // Serialise and check hybrid + moe_intermediate_size land in JSON.
    let json = serde_json::to_string(&vc).unwrap();
    assert!(json.contains("\"hybrid\":true"), "{json}");
    assert!(json.contains("\"moe_intermediate_size\":768"), "{json}");

    let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
    let back_moe = back.moe.unwrap();
    assert!(back_moe.hybrid);
    assert_eq!(back_moe.moe_intermediate_size, Some(768));
}

#[test]
fn moe_config_with_shared_expert_round_trips() {
    // shared_expert=true exercises the non-default branch of the
    // bool field; existing tests only hit shared_expert=false.
    let moe = MoeConfig {
        num_experts: 64,
        top_k: 6,
        shared_expert: true,
        shared_expert_intermediate_size: None,
        router_type: "top_k_softmax".into(),
        moe_intermediate_size: None,
        hybrid: false,
    };
    let json = serde_json::to_string(&moe).unwrap();
    assert!(json.contains("\"shared_expert\":true"), "{json}");
    let back: MoeConfig = serde_json::from_str(&json).unwrap();
    assert!(back.shared_expert);
}

#[test]
fn norm_eps_field_round_trips_independent_of_granite_scalars() {
    // norm_eps lives under skip_serializing_if; cover the Some-branch
    // standalone (no Granite multipliers).
    let mut cfg = minimal_model_config();
    cfg.norm_eps = Some(1e-6);
    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("\"norm_eps\""), "{json}");
    let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.norm_eps, Some(1e-6));
}

#[test]
fn gemma4_per_layer_attn_geometry_round_trips() {
    // Gemma 4 sets the optional per-layer attention fields
    // (global_head_dim, sliding_window_pattern, partial_rotary_factor,
    // layer_types). These fields exist in VindexModelConfig but
    // the granite/llama tests don't exercise them — populate them
    // directly via the struct so the serde derive macros and the
    // skip_serializing_if branches all get coverage.
    let mut cfg = minimal_model_config();
    cfg.global_head_dim = Some(512);
    cfg.num_global_kv_heads = Some(2);
    cfg.partial_rotary_factor = Some(0.25);
    cfg.sliding_window_pattern = Some(6);
    cfg.layer_types = Some(vec!["sliding_attention".into(), "full_attention".into()]);
    cfg.num_kv_shared_layers = Some(2);
    cfg.per_layer_embed_dim = Some(256);
    cfg.use_double_wide_mlp = Some(true);
    cfg.vocab_size_per_layer_input = Some(262144);
    cfg.rope_local_base = Some(10_000.0);
    cfg.query_pre_attn_scalar = Some(1.0);
    cfg.final_logit_softcapping = Some(30.0);

    let json = serde_json::to_string(&cfg).unwrap();
    assert!(json.contains("global_head_dim"));
    assert!(json.contains("sliding_window_pattern"));
    assert!(json.contains("layer_types"));

    let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.global_head_dim, Some(512));
    assert_eq!(back.num_global_kv_heads, Some(2));
    assert_eq!(back.partial_rotary_factor, Some(0.25));
    assert_eq!(back.sliding_window_pattern, Some(6));
    assert_eq!(back.layer_types.as_ref().map(|v| v.len()), Some(2));
    assert_eq!(back.num_kv_shared_layers, Some(2));
    assert_eq!(back.per_layer_embed_dim, Some(256));
    assert_eq!(back.use_double_wide_mlp, Some(true));
    assert_eq!(back.vocab_size_per_layer_input, Some(262144));
    assert_eq!(back.rope_local_base, Some(10_000.0));
    assert_eq!(back.query_pre_attn_scalar, Some(1.0));
    assert_eq!(back.final_logit_softcapping, Some(30.0));
}

/// The G2b declared scalars must survive the vindex round trip — same
/// contract as every other forward-affecting field.
#[test]
fn declared_scaling_scalars_round_trip() {
    let mut cfg = minimal_model_config();
    cfg.qk_scale_factor = Some(3.87);
    cfg.output_multiplier = Some(0.196);
    cfg.post_norm_eps = Some(1e-8);
    cfg.attention_bias = Some(false);
    cfg.hidden_act = Some("silu".to_string());
    cfg.activation_situ_beta = Some(4.0);
    cfg.activation_situ_linear_beta = Some(25.0);
    cfg.max_position_embeddings = Some(131072);

    let json = serde_json::to_string(&cfg).unwrap();
    let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.qk_scale_factor, Some(3.87));
    assert_eq!(back.output_multiplier, Some(0.196));
    assert_eq!(back.post_norm_eps, Some(1e-8));
    assert_eq!(back.attention_bias, Some(false));
    assert_eq!(back.hidden_act.as_deref(), Some("silu"));
    assert_eq!(back.activation_situ_beta, Some(4.0));
    assert_eq!(back.activation_situ_linear_beta, Some(25.0));
    assert_eq!(back.max_position_embeddings, Some(131072));
}

/// The per-layer position policy must survive the whole round trip:
/// checkpoint → arch → vindex `model_config` → reconstructed arch. A
/// NoPE layer (declared `layer_rope_theta[i] == 0`) that comes back
/// rotary is the served-with-full-rotation defect this field exists to
/// prevent — invisible to CPU-vs-Metal parity because both arms read
/// the same `index.json`.
#[test]
fn nope_position_policy_survives_the_vindex_round_trip() {
    use larql_models::config::PositionPolicy;
    let source = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "some_hybrid_nope_model",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0]
    }));
    assert_eq!(
        source.position_policy_for_layer(3),
        PositionPolicy::None,
        "precondition: the source arch resolves the sentinel"
    );

    let persisted = VindexModelConfig::from_arch(source.as_ref());
    let json = serde_json::to_string(&persisted).unwrap();
    let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(
        back.layer_rope_theta,
        Some(vec![500000.0, 500000.0, 500000.0, 0.0]),
        "array must persist verbatim, sentinel included"
    );
}

#[test]
fn granite_scalars_absent_for_non_granite_arch() {
    // Llama and Mistral don't carry these multipliers; verify the
    // serialised JSON omits the fields entirely so existing vindexes
    // on those arches are byte-stable after a round trip.
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }));
    let vc = VindexModelConfig::from_arch(&*arch);
    assert!(vc.attention_multiplier.is_none());
    assert!(vc.residual_multiplier.is_none());
    assert!(vc.logits_scaling.is_none());
    let json = serde_json::to_string(&vc).unwrap();
    assert!(!json.contains("attention_multiplier"));
    assert!(!json.contains("residual_multiplier"));
    assert!(!json.contains("logits_scaling"));
}
