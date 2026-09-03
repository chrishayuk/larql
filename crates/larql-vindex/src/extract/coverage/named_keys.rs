//! Every tensor key an architecture is capable of naming.
//!
//! This is one half of the coverage audit: the *recognised* set. A source
//! tensor whose name is not in here was never reachable by extraction,
//! because extraction only ever asks the architecture for keys.
//!
//! ## Why naming is the right thing to measure
//!
//! Every silent-drop bug found in this codebase has been a naming failure,
//! not a plumbing failure:
//!
//! - GPT-OSS declared "attention has biases, sinks" in a doc comment while
//!   every accessor returned `None`, so extraction dropped 5 of 11 attention
//!   tensors per layer (`docs/k3-funnel.md` §4.6.1);
//! - the same module named 5 of its 8 MLP tensors, silently dropping the
//!   router bias and both expert biases (§4.7).
//!
//! In both cases the extractor was working correctly — it asked for every key
//! it was told about. Nobody told it about those.
//!
//! ## Failure direction
//!
//! This enumeration is hand-maintained against [`ModelArchitecture`]'s
//! accessors, so a newly-added accessor that is not wired in here makes its
//! tensors report as **unrecognised**. That is deliberate and is the safe
//! direction: the audit gets noisier, never quieter. It cannot mint a false
//! "this tensor is covered" — only a false "look at this", which a human
//! resolves in one line. The `every_key_accessor_is_enumerated` test pins the
//! count so the omission is caught at the source rather than in a report.

use std::collections::HashSet;

use larql_models::ModelArchitecture;

/// Number of `*_key` accessors on [`ModelArchitecture`] that name a tensor.
///
/// Pinned so that adding an accessor without teaching [`collect`] about it
/// fails a test here rather than surfacing as spurious "unrecognised" tensors
/// on somebody's extraction run.
#[cfg(test)]
pub(crate) const KEY_ACCESSOR_COUNT: usize = 63;

/// Output projection, resolved by `WeightSource::lm_head()` rather than by a
/// key accessor. Mirrors the name the safetensors loader looks up.
const LM_HEAD_KEY: &str = "lm_head.weight";

/// Every tensor key `arch` can produce for a model of this shape.
///
/// `num_layers` and the architecture's own `num_experts()` bound the
/// per-layer and per-expert accessors. Keys are returned normalised the way
/// the architecture emits them; the caller is responsible for applying the
/// same prefix-stripping it applies to source names.
pub fn collect(arch: &dyn ModelArchitecture, num_layers: usize) -> HashSet<String> {
    let mut keys = HashSet::new();

    // ── Model-level (no index) ──
    keys.insert(arch.embed_key().to_string());
    keys.insert(arch.final_norm_key().to_string());
    let push_opt = |k: Option<String>, set: &mut HashSet<String>| {
        if let Some(k) = k {
            set.insert(k);
        }
    };
    if let Some(k) = arch.position_embed_key() {
        keys.insert(k.to_string());
    }
    // `lm_head` is reachable through `WeightSource::lm_head()` rather than a
    // `*_key` accessor, so the accessor sweep below never names it. It is
    // genuinely extracted (`write_kquant::lm_head`), and omitting it here
    // would report every checkpoint that ships an untied output projection
    // as having an unrecognised tensor. Models with tied embeddings simply
    // do not carry this name, so adding it costs nothing there.
    keys.insert(LM_HEAD_KEY.to_string());
    push_opt(arch.final_norm_bias_key(), &mut keys);
    push_opt(arch.per_layer_embed_key(), &mut keys);
    push_opt(arch.per_layer_model_projection_key(), &mut keys);
    push_opt(arch.per_layer_projection_norm_key(), &mut keys);

    for layer in 0..num_layers {
        // Always-named per-layer tensors.
        for k in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
            arch.input_layernorm_key(layer),
            arch.post_attention_layernorm_key(layer),
        ] {
            keys.insert(k);
        }

        // Optionally-named per-layer tensors.
        for k in [
            arch.fused_qkv_key(layer),
            arch.fused_qkv_bias_key(layer),
            arch.attn_q_bias_key(layer),
            arch.attn_k_bias_key(layer),
            arch.attn_v_bias_key(layer),
            arch.attn_o_bias_key(layer),
            arch.attn_q_norm_key(layer),
            arch.attn_k_norm_key(layer),
            arch.attn_sinks_key(layer),
            arch.input_layernorm_bias_key(layer),
            arch.post_attention_layernorm_bias_key(layer),
            arch.ffn_up_bias_key(layer),
            arch.ffn_down_bias_key(layer),
            arch.pre_feedforward_layernorm_key(layer),
            arch.post_feedforward_layernorm_key(layer),
            arch.layer_scalar_key(layer),
            arch.per_layer_input_gate_key(layer),
            arch.per_layer_projection_key(layer),
            arch.post_per_layer_input_norm_key(layer),
            arch.moe_router_key(layer),
            arch.moe_router_bias_key(layer),
            arch.moe_router_scale_key(layer),
            arch.moe_router_per_expert_scale_key(layer),
            arch.moe_router_norm_key(layer),
            arch.moe_post_outer_norm_key(layer),
            arch.moe_pre_experts_norm_key(layer),
            arch.moe_post_experts_norm_key(layer),
            arch.moe_post_ffn1_norm_key(layer),
            arch.packed_gate_up_blocks_key(layer),
            arch.packed_gate_up_scales_key(layer),
            arch.packed_gate_up_bias_key(layer),
            arch.packed_down_blocks_key(layer),
            arch.packed_down_scales_key(layer),
            arch.packed_down_bias_key(layer),
            arch.packed_experts_gate_up_key(layer),
            arch.packed_experts_down_key(layer),
            arch.shared_expert_gate_key(layer),
            arch.shared_expert_up_key(layer),
            arch.shared_expert_down_key(layer),
            arch.shared_expert_branch_gate_key(layer),
            arch.mla_kv_a_key(layer),
            arch.mla_kv_b_key(layer),
            arch.mla_q_a_key(layer),
            arch.mla_q_b_key(layer),
        ] {
            push_opt(k, &mut keys);
        }

        // Per-expert tensors. `num_experts()` is 0 for dense models, so this
        // loop is empty there rather than needing an `is_moe()` guard.
        for expert in 0..arch.num_experts() {
            for k in [
                arch.expert_ffn_gate_key(layer, expert),
                arch.expert_ffn_up_key(layer, expert),
                arch.expert_ffn_down_key(layer, expert),
            ] {
                push_opt(k, &mut keys);
            }
        }
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::detect_from_json;

    fn arch(json: serde_json::Value) -> Box<dyn ModelArchitecture> {
        detect_from_json(&json)
    }

    fn gpt_oss() -> Box<dyn ModelArchitecture> {
        arch(serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 2,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "num_local_experts": 4,
            "num_experts_per_tok": 4,
        }))
    }

    /// The count is pinned so that adding a `*_key` accessor to the trait
    /// without adding it to [`collect`] fails here — at the source — rather
    /// than as spurious "unrecognised" tensors on an extraction run.
    ///
    /// If this fails: add the new accessor to `collect`, then bump the
    /// constant. Do not bump it alone.
    #[test]
    fn every_key_accessor_is_enumerated() {
        // The trait's own file, not the `config` module root — a trait is one
        // item, so every `*_key` accessor is declared here or nowhere.
        let src = include_str!("../../../../larql-models/src/config/architecture.rs");
        let declared = src
            .lines()
            .filter(|l| {
                let l = l.trim_start();
                l.starts_with("fn ") && l.contains("_key(&self")
            })
            .count();
        assert_eq!(
            declared, KEY_ACCESSOR_COUNT,
            "ModelArchitecture has {declared} key accessors but this module \
             is pinned at {KEY_ACCESSOR_COUNT}. Add the new accessor to \
             `collect()` before bumping the constant — otherwise its tensors \
             will report as unrecognised."
        );
    }

    #[test]
    fn names_the_gpt_oss_attention_surface_including_biases_and_sinks() {
        let a = gpt_oss();
        let keys = collect(&*a, 2);
        // The 11-tensor attention surface from `docs/k3-funnel.md` §4.6.1.
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
            "self_attn.sinks",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            let want = format!("layers.1.{suffix}");
            assert!(keys.contains(&want), "missing {want}");
        }
    }

    #[test]
    fn names_the_gpt_oss_mlp_surface_including_all_three_biases() {
        let a = gpt_oss();
        let keys = collect(&*a, 2);
        for suffix in [
            "mlp.router.weight",
            "mlp.router.bias",
            "mlp.experts.gate_up_proj_blocks",
            "mlp.experts.gate_up_proj_scales",
            "mlp.experts.gate_up_proj_bias",
            "mlp.experts.down_proj_blocks",
            "mlp.experts.down_proj_scales",
            "mlp.experts.down_proj_bias",
        ] {
            let want = format!("layers.0.{suffix}");
            assert!(keys.contains(&want), "missing {want}");
        }
    }

    /// Per-expert keys must be enumerated for every expert, not just expert 0
    /// — an off-by-one here would leave most of a 128-expert bank unnamed.
    #[test]
    fn enumerates_every_expert_not_just_the_first() {
        let a = gpt_oss();
        let keys = collect(&*a, 1);
        for expert in 0..a.num_experts() {
            let want = format!("layers.0.block_sparse_moe.experts.{expert}.w1.weight");
            assert!(keys.contains(&want), "missing expert {expert}: {want}");
        }
    }

    #[test]
    fn covers_every_layer_not_just_the_first() {
        let a = gpt_oss();
        let keys = collect(&*a, 3);
        for layer in 0..3 {
            assert!(keys.contains(&format!("layers.{layer}.self_attn.q_proj.weight")));
        }
        assert!(!keys.contains("layers.3.self_attn.q_proj.weight"));
    }

    /// A dense architecture must not emit expert keys — `num_experts()` is 0,
    /// so the per-expert loop is empty rather than needing an `is_moe` guard.
    #[test]
    fn a_dense_architecture_names_no_expert_keys() {
        let a = arch(serde_json::json!({
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 32,
        }));
        let keys = collect(&*a, 2);
        assert!(!keys.iter().any(|k| k.contains("experts")));
        assert!(keys.contains(&a.ffn_gate_key(0)));
    }

    #[test]
    fn includes_the_model_level_tensors() {
        let a = gpt_oss();
        let keys = collect(&*a, 1);
        assert!(keys.contains(a.embed_key()));
        assert!(keys.contains(a.final_norm_key()));
    }
}
