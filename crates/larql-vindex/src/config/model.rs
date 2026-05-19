//! Model-architecture config carried in `index.json` so the
//! architecture can be reconstructed without the original
//! `config.json`.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// MoE configuration (None for dense models).
    #[serde(default)]
    pub moe: Option<MoeConfig>,

    // ── Gemma 4 per-layer attention geometry ──
    // All optional for backward compatibility with existing vindexes.
    /// Head dimension for global (full) attention layers. If None, all layers use head_dim.
    /// Gemma 4: 512 for global layers, head_dim (256) for sliding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers. If None, all layers use num_kv_heads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_kv_heads: Option<usize>,
    /// Fraction of head_dim to apply RoPE to (0.0–1.0). If None, full rotation.
    /// Gemma 4 global layers: 0.25.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Sliding window pattern: every Nth layer is full attention.
    /// Gemma 4: 6 (layers 5, 11, 17, ... are full).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window_pattern: Option<usize>,
    /// Explicit per-layer type array (e.g., ["sliding_attention", "full_attention", ...]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Whether value projection shares key projection (K=V).
    #[serde(default)]
    pub attention_k_eq_v: bool,
    /// Number of layers at the end that share KV from earlier layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<usize>,
    /// Per-layer embedding dimension (PLE). 0 or None = no PLE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_layer_embed_dim: Option<usize>,
    /// RoPE base for local/sliding window layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base: Option<f64>,
    /// Query pre-attention scalar (overrides 1/sqrt(head_dim)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    /// Final-logit tanh softcap (Gemma 2/3/4: 30.0). Applied to logits
    /// immediately before softmax in `logits_to_predictions`. Omitting it
    /// leaves logits uncapped — on E2B this peaked the softmax on the
    /// wrong token (observed: "Paris" → "hyperparameters").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f64>,

    // ── Granite-family scaling multipliers ──
    // None on every other arch. Captured at vindex-build time so the
    // reconstructed `ModelArchitecture` knows about them at load time;
    // without these the vindex Metal forward path silently runs with
    // all three at 1.0 and Granite emits gibberish (the safetensors
    // detect path picks them up from config.json directly, which is why
    // `shannon verify` was clean while `larql run` on a Granite vindex
    // was not). `embedding_multiplier` is already captured at the top
    // level of `VindexConfig` as `embed_scale`.
    /// Attention score multiplier (Granite 4.1: 1/64 on 3B, 1/128 on
    /// 8B/30B). Applied on top of 1/sqrt(head_dim) — see
    /// [`larql_models::ModelArchitecture::attention_multiplier`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_multiplier: Option<f64>,
    /// Residual-stream scaling factor applied after attention and FFN
    /// additions (Granite 4.1: 0.22 on 3B/8B, 0.175 on 30B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_multiplier: Option<f64>,
    /// Logits scaling factor — final logits are divided by this before
    /// softmax (Granite 4.1: 10 on 3B, 16 on 8B/30B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits_scaling: Option<f64>,
    /// RMS-norm / LayerNorm epsilon parsed from `rms_norm_eps` (or
    /// `layer_norm_eps`). Llama 3, Mistral, Gemma 3, and Granite 4.1 all
    /// ship 1e-5; older default was 1e-6. Captured here so the vindex
    /// load path doesn't silently fall back to the arch-class default —
    /// same regression mode that broke the safetensors path before the
    /// fix in `docs/diagnoses/shannon-cross-engine-divergence.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_eps: Option<f64>,
}

/// MoE (Mixture of Experts) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Number of experts per layer.
    pub num_experts: usize,
    /// Number of experts selected per token (top-K routing).
    pub top_k: usize,
    /// Whether there's a shared expert always active (DeepSeek V2/V3).
    #[serde(default)]
    pub shared_expert: bool,
    /// Router type (e.g., "top_k_softmax", "gemma4_top_k_softmax").
    #[serde(default = "default_router_type")]
    pub router_type: String,
    /// Per-expert intermediate (hidden) dimension.
    /// Differs from the dense FFN intermediate_size in hybrid models (Gemma 4 A4B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,
    /// Hybrid MoE: dense MLP and expert block coexist in each layer, outputs summed.
    /// True for Gemma 4 A4B. False for pure MoE (Mixtral, DeepSeek).
    #[serde(default)]
    pub hybrid: bool,
}

fn default_router_type() -> String {
    "top_k_softmax".to_string()
}

impl VindexModelConfig {
    /// Build the serialisable vindex architecture config from the detected
    /// model architecture. Keeping this mapping in one place prevents vector
    /// imports, f32 writers, and Q4K writers from drifting.
    pub fn from_arch(arch: &dyn larql_models::ModelArchitecture) -> Self {
        let cfg = arch.config();
        Self {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: if arch.is_moe() {
                Some(MoeConfig {
                    num_experts: arch.num_experts(),
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    router_type: arch.moe_router_type().into(),
                    moe_intermediate_size: if arch.moe_intermediate_size() > 0 {
                        Some(arch.moe_intermediate_size())
                    } else {
                        None
                    },
                    hybrid: arch.is_hybrid_moe(),
                })
            } else {
                None
            },
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            rope_local_base: cfg.rope_local_base,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            final_logit_softcapping: cfg.final_logit_softcapping,
            attention_multiplier: cfg.attention_multiplier,
            residual_multiplier: cfg.residual_multiplier,
            logits_scaling: cfg.logits_scaling,
            norm_eps: cfg.norm_eps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_model_config() -> VindexModelConfig {
        VindexModelConfig {
            model_type: "gemma3".into(),
            head_dim: 256,
            num_q_heads: 8,
            num_kv_heads: 4,
            rope_base: 10000.0,
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
        assert_eq!(back.rope_local_base, Some(10_000.0));
        assert_eq!(back.query_pre_attn_scalar, Some(1.0));
        assert_eq!(back.final_logit_softcapping, Some(30.0));
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
}
