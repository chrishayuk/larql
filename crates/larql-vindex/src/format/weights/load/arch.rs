//! Shared architecture-reconstruction helper.
//!
//! Both the f32 and Q4_K loaders rebuild a `serde_json::Value` from
//! `VindexConfig` + `VindexModelConfig` and feed it to
//! `larql_models::detect_from_json`. The two paths used to carry
//! identical ~50-line copies of the field-by-field setup; this module
//! is the single source of truth.
//!
//! New per-layer-geometry fields (Gemma 4 family, future architectures)
//! get added here once and flow through both loaders automatically.

use crate::config::{VindexConfig, VindexModelConfig};

/// Build the architecture-detection JSON blob from the on-disk config.
///
/// Includes core architecture fields, Gemma 4 per-layer geometry, and
/// MoE config when present. The result is the input to
/// `larql_models::detect_from_json`.
pub(super) fn build_arch_json(
    config: &VindexConfig,
    model_cfg: &VindexModelConfig,
) -> serde_json::Value {
    let mut arch_obj = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    let obj = arch_obj.as_object_mut().expect("json! built an object");

    // Gemma 4 per-layer geometry — pass through whatever the vindex
    // recorded; `detect_from_json` decides which fields to consume.
    if let Some(v) = model_cfg.global_head_dim {
        obj.insert("global_head_dim".into(), v.into());
    }
    if let Some(v) = model_cfg.num_global_kv_heads {
        obj.insert("num_global_key_value_heads".into(), v.into());
    }
    if let Some(v) = model_cfg.partial_rotary_factor {
        obj.insert("partial_rotary_factor".into(), v.into());
    }
    if let Some(v) = model_cfg.sliding_window_pattern {
        obj.insert("sliding_window_pattern".into(), v.into());
    }
    if let Some(ref v) = model_cfg.layer_types {
        obj.insert(
            "layer_types".into(),
            serde_json::to_value(v).unwrap_or_default(),
        );
    }
    if model_cfg.attention_k_eq_v {
        obj.insert("attention_k_eq_v".into(), true.into());
    }
    if let Some(v) = model_cfg.num_kv_shared_layers {
        obj.insert("num_kv_shared_layers".into(), v.into());
    }
    if let Some(v) = model_cfg.per_layer_embed_dim {
        obj.insert("hidden_size_per_layer_input".into(), v.into());
    }
    if let Some(v) = model_cfg.rope_local_base {
        obj.insert("rope_local_base_freq".into(), v.into());
    }
    if let Some(v) = model_cfg.query_pre_attn_scalar {
        obj.insert("query_pre_attn_scalar".into(), v.into());
    }
    if let Some(v) = model_cfg.final_logit_softcapping {
        obj.insert("final_logit_softcapping".into(), v.into());
    }

    // Granite-family scaling multipliers. Re-emit the same field names
    // the safetensors detect path reads (`detect/parser.rs`) so
    // `detect_from_json` rebuilds an arch with the right
    // `embedding_multiplier`, `attention_multiplier`,
    // `residual_multiplier`, and `logits_scaling` — all four are
    // consumed in the forward path (`attention/{gpu,decode,block}.rs`,
    // `vindex/kquant_forward/*`, `predict/*`, `forward/{embed,layer,
    // vocab_proj}.rs`). Without these the reconstructed arch defaults
    // them to 1.0 and the model emits gibberish — symptom from
    // `larql run` on a Granite vindex before the fields were added here.
    //
    // `embedding_multiplier` is also stored at the top level of
    // `VindexConfig.embed_scale` (legacy capture site that predates the
    // model_cfg surface). The top-level value is the source of truth at
    // build time; we forward it back through the safetensors-style field
    // name `embedding_multiplier` so `detect_from_json` populates
    // `ModelConfig.embedding_multiplier`, and through it
    // `arch.embed_scale()`. Only emit when non-1.0 so non-Granite
    // vindexes don't suddenly start declaring a multiplier they never
    // had.
    if (config.embed_scale - 1.0).abs() > f32::EPSILON {
        obj.insert(
            "embedding_multiplier".into(),
            (config.embed_scale as f64).into(),
        );
    }
    if let Some(v) = model_cfg.attention_multiplier {
        obj.insert("attention_multiplier".into(), v.into());
    }
    if let Some(v) = model_cfg.residual_multiplier {
        obj.insert("residual_multiplier".into(), v.into());
    }
    if let Some(v) = model_cfg.logits_scaling {
        obj.insert("logits_scaling".into(), v.into());
    }
    if let Some(v) = model_cfg.norm_eps {
        obj.insert("rms_norm_eps".into(), v.into());
    }

    // MoE — Mixtral, Gemma 4 26B A4B, DeepSeek-V*, etc.
    if let Some(ref moe) = model_cfg.moe {
        obj.insert("num_experts".into(), moe.num_experts.into());
        obj.insert("top_k_experts".into(), moe.top_k.into());
        if let Some(v) = moe.moe_intermediate_size {
            obj.insert("moe_intermediate_size".into(), v.into());
        }
        if moe.hybrid {
            obj.insert("enable_moe_block".into(), true.into());
        }
    }

    arch_obj
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::QuantFormat;

    fn minimal_model_cfg() -> VindexModelConfig {
        VindexModelConfig {
            model_type: "test_arch".into(),
            head_dim: 64,
            num_q_heads: 8,
            num_kv_heads: 8,
            rope_base: 10_000.0,
            sliding_window: None,
            moe: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
        }
    }

    fn minimal_config(model_cfg: VindexModelConfig) -> VindexConfig {
        VindexConfig {
            version: 2,
            model: "test/model".into(),
            family: "test".into(),
            num_layers: 4,
            hidden_size: 512,
            intermediate_size: 2048,
            vocab_size: 32_000,
            embed_scale: 1.0,
            layers: Vec::new(),
            down_top_k: 5,
            has_model_weights: true,
            source: None,
            checksums: None,
            extract_level: crate::ExtractLevel::All,
            dtype: crate::config::dtype::StorageDtype::F32,
            quant: QuantFormat::None,
            layer_bands: crate::LayerBands::for_family("test", 4),
            model_config: Some(model_cfg),
            fp4: None,
            ffn_layout: None,
        }
    }

    #[test]
    fn build_arch_json_emits_core_fields() {
        let model_cfg = minimal_model_cfg();
        let config = minimal_config(model_cfg.clone());

        let v = build_arch_json(&config, &model_cfg);
        let obj = v.as_object().unwrap();

        assert_eq!(obj["model_type"], "test_arch");
        assert_eq!(obj["hidden_size"], 512);
        assert_eq!(obj["num_hidden_layers"], 4);
        assert_eq!(obj["intermediate_size"], 2048);
        assert_eq!(obj["head_dim"], 64);
        assert_eq!(obj["num_attention_heads"], 8);
        assert_eq!(obj["num_key_value_heads"], 8);
        assert_eq!(obj["vocab_size"], 32_000);
        // Optional fields absent when unset.
        assert!(!obj.contains_key("global_head_dim"));
        assert!(!obj.contains_key("num_experts"));
        assert!(!obj.contains_key("attention_k_eq_v"));
    }

    #[test]
    fn build_arch_json_passes_through_gemma4_fields() {
        let mut model_cfg = minimal_model_cfg();
        model_cfg.global_head_dim = Some(256);
        model_cfg.num_global_kv_heads = Some(2);
        model_cfg.partial_rotary_factor = Some(0.5);
        model_cfg.sliding_window_pattern = Some(6);
        model_cfg.layer_types = Some(vec!["sliding".into(), "global".into()]);
        model_cfg.attention_k_eq_v = true;
        model_cfg.num_kv_shared_layers = Some(8);
        model_cfg.per_layer_embed_dim = Some(256);
        model_cfg.rope_local_base = Some(10_000.0);
        model_cfg.query_pre_attn_scalar = Some(256.0);
        model_cfg.final_logit_softcapping = Some(30.0);
        let config = minimal_config(model_cfg.clone());

        let v = build_arch_json(&config, &model_cfg);
        let obj = v.as_object().unwrap();

        assert_eq!(obj["global_head_dim"], 256);
        assert_eq!(obj["num_global_key_value_heads"], 2);
        assert_eq!(obj["partial_rotary_factor"].as_f64().unwrap(), 0.5);
        assert_eq!(obj["sliding_window_pattern"], 6);
        assert_eq!(
            obj["layer_types"].as_array().unwrap(),
            &[
                serde_json::Value::String("sliding".into()),
                serde_json::Value::String("global".into()),
            ]
        );
        assert_eq!(obj["attention_k_eq_v"], true);
        assert_eq!(obj["num_kv_shared_layers"], 8);
        assert_eq!(obj["hidden_size_per_layer_input"], 256);
        assert_eq!(obj["rope_local_base_freq"].as_f64().unwrap(), 10_000.0);
        assert_eq!(obj["query_pre_attn_scalar"].as_f64().unwrap(), 256.0);
        assert_eq!(obj["final_logit_softcapping"].as_f64().unwrap(), 30.0);
    }

    #[test]
    fn build_arch_json_omits_attention_k_eq_v_when_false() {
        // Pin the contract: false → field absent (default behaviour).
        // detect_from_json treats absent as false too.
        let mut model_cfg = minimal_model_cfg();
        model_cfg.attention_k_eq_v = false;
        let config = minimal_config(model_cfg.clone());

        let v = build_arch_json(&config, &model_cfg);
        assert!(!v.as_object().unwrap().contains_key("attention_k_eq_v"));
    }

    #[test]
    fn build_arch_json_emits_moe_block_when_present() {
        let mut model_cfg = minimal_model_cfg();
        model_cfg.moe = Some(crate::MoeConfig {
            num_experts: 8,
            top_k: 2,
            shared_expert: false,
            router_type: "softmax".into(),
            moe_intermediate_size: Some(1024),
            hybrid: false,
        });
        let config = minimal_config(model_cfg.clone());

        let v = build_arch_json(&config, &model_cfg);
        let obj = v.as_object().unwrap();
        assert_eq!(obj["num_experts"], 8);
        assert_eq!(obj["top_k_experts"], 2);
        assert_eq!(obj["moe_intermediate_size"], 1024);
        // hybrid=false → enable_moe_block field absent
        assert!(!obj.contains_key("enable_moe_block"));
    }

    #[test]
    fn build_arch_json_emits_hybrid_moe_flag() {
        let mut model_cfg = minimal_model_cfg();
        model_cfg.moe = Some(crate::MoeConfig {
            num_experts: 128,
            top_k: 8,
            shared_expert: true,
            router_type: "topk".into(),
            moe_intermediate_size: None, // no override
            hybrid: true,
        });
        let config = minimal_config(model_cfg.clone());

        let v = build_arch_json(&config, &model_cfg);
        let obj = v.as_object().unwrap();
        assert_eq!(obj["num_experts"], 128);
        assert_eq!(obj["top_k_experts"], 8);
        assert_eq!(obj["enable_moe_block"], true);
        // moe_intermediate_size omitted when None
        assert!(!obj.contains_key("moe_intermediate_size"));
    }

    #[test]
    fn build_arch_json_no_moe_block_when_absent() {
        let model_cfg = minimal_model_cfg(); // moe = None
        let config = minimal_config(model_cfg.clone());

        let v = build_arch_json(&config, &model_cfg);
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("num_experts"));
        assert!(!obj.contains_key("top_k_experts"));
        assert!(!obj.contains_key("moe_intermediate_size"));
        assert!(!obj.contains_key("enable_moe_block"));
    }

    #[test]
    fn build_arch_json_round_trips_granite_scalars() {
        // Regression pin: Granite 4.1 3B vindexes built before the
        // multiplier fields were added round-trip with `attention_multiplier`
        // / `residual_multiplier` / `logits_scaling` defaulted to 1.0 in
        // the reconstructed arch, and the model emits garbage. Hand the
        // helper a Granite-shaped model_cfg, run it through
        // `detect_from_json`, and assert the trait getters see the
        // training-time values.
        let mut model_cfg = minimal_model_cfg();
        model_cfg.model_type = "granite".into();
        model_cfg.head_dim = 64;
        model_cfg.num_q_heads = 40;
        model_cfg.num_kv_heads = 8;
        model_cfg.attention_multiplier = Some(0.015625);
        model_cfg.residual_multiplier = Some(0.22);
        model_cfg.logits_scaling = Some(10.0);
        model_cfg.norm_eps = Some(1e-05);
        let mut config = minimal_config(model_cfg.clone());
        config.hidden_size = 2560;
        config.num_layers = 40;
        config.intermediate_size = 8192;
        config.vocab_size = 100_352;
        config.embed_scale = 12.0;

        let v = build_arch_json(&config, &model_cfg);
        // Re-emitted with the field names `detect/parser.rs` reads.
        assert_eq!(v["attention_multiplier"].as_f64().unwrap(), 0.015625);
        assert_eq!(v["residual_multiplier"].as_f64().unwrap(), 0.22);
        assert_eq!(v["logits_scaling"].as_f64().unwrap(), 10.0);
        assert_eq!(v["rms_norm_eps"].as_f64().unwrap(), 1e-05);
        // `embed_scale` is captured at the top level of `VindexConfig`
        // but the reconstructed arch reads `embedding_multiplier` (the
        // safetensors-style key). Forwarding 12.0 here gives the Metal
        // forward path the right ×12 scaling at embedding lookup.
        assert_eq!(v["embedding_multiplier"].as_f64().unwrap(), 12.0);

        // Round-trip through detect_from_json: the trait getters must
        // see the same values, since the forward path reads through
        // those getters (see `attention/gpu.rs`, `vocab_proj.rs`,
        // `vindex/kquant_forward/cached.rs`).
        let arch = larql_models::detect_from_json(&v);
        assert_eq!(arch.family(), "granite");
        assert_eq!(arch.embed_scale(), 12.0);
        assert_eq!(arch.attention_multiplier(), 0.015625);
        assert_eq!(arch.residual_multiplier(), 0.22);
        assert_eq!(arch.logits_scaling(), 10.0);
        assert_eq!(arch.norm_eps(), 1e-05);
    }

    #[test]
    fn build_arch_json_omits_granite_scalars_when_absent() {
        // Legacy vindex / non-Granite arch: no scalar fields recorded.
        // The reconstructed arch's trait getters must return the
        // default 1.0 / default norm_eps, and the JSON must not carry
        // the field names (forward-compat: an older vindex shouldn't
        // suddenly start declaring a multiplier of 0).
        let model_cfg = minimal_model_cfg(); // all None
        let config = minimal_config(model_cfg.clone());
        let v = build_arch_json(&config, &model_cfg);
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("attention_multiplier"));
        assert!(!obj.contains_key("residual_multiplier"));
        assert!(!obj.contains_key("logits_scaling"));
        assert!(!obj.contains_key("rms_norm_eps"));
    }

    #[test]
    fn build_arch_json_round_trips_through_detect_from_json() {
        // The whole reason this helper exists is to feed
        // detect_from_json. Pin that the integration still works.
        let mut model_cfg = minimal_model_cfg();
        model_cfg.model_type = "llama".into();
        let config = minimal_config(model_cfg.clone());

        let v = build_arch_json(&config, &model_cfg);
        let arch = larql_models::detect_from_json(&v);
        let cfg = arch.config();
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.num_layers, 4);
        assert_eq!(cfg.num_q_heads, 8);
    }
}
