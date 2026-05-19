//! Stage 6 of the build pipeline — assemble + write `index.json`.

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::build_helpers::chrono_now;
use crate::format::filenames::*;

use super::BuildContext;

impl<'a> BuildContext<'a> {
    /// Stage 6 — assemble + write `index.json`. If the extract level
    /// requires it, also write the model weights and re-emit the index
    /// with `has_model_weights = true`. Final pass adds provenance +
    /// checksums.
    pub(super) fn write_index_json(
        &mut self,
        model_name: &str,
        extract_level: crate::ExtractLevel,
    ) -> Result<(), VindexError> {
        let family = self.weights.arch.family().to_string();
        let mut config = VindexConfig {
            version: 2,
            model: model_name.to_string(),
            family: family.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            embed_scale: self.embed_scale,
            layers: std::mem::take(&mut self.layer_infos),
            down_top_k: self.down_top_k,
            has_model_weights: false,
            source: None,
            checksums: None,
            extract_level,
            dtype: self.dtype,
            quant: crate::QuantFormat::None,
            layer_bands: crate::LayerBands::for_family(&family, self.num_layers),
            model_config: {
                let cfg = self.weights.arch.config();
                Some(VindexModelConfig {
                    model_type: cfg.model_type.clone(),
                    head_dim: self.weights.head_dim,
                    num_q_heads: self.weights.num_q_heads,
                    num_kv_heads: self.weights.num_kv_heads,
                    rope_base: self.weights.rope_base,
                    sliding_window: cfg.sliding_window,
                    moe: if self.is_moe {
                        let a = &*self.weights.arch;
                        Some(crate::MoeConfig {
                            num_experts: self.n_experts,
                            top_k: a.num_experts_per_token(),
                            shared_expert: a.num_shared_experts() > 0,
                            router_type: a.moe_router_type().to_string(),
                            moe_intermediate_size: if a.moe_intermediate_size() > 0 {
                                Some(a.moe_intermediate_size())
                            } else {
                                None
                            },
                            hybrid: a.is_hybrid_moe(),
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
                })
            },
            fp4: None,
            ffn_layout: None,
        };

        // Preliminary write — `write_model_weights` reads the index.
        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;

        if extract_level != crate::ExtractLevel::Browse {
            crate::format::weights::write_model_weights(
                self.weights,
                self.output_dir,
                self.callbacks,
            )?;
            config.has_model_weights = true;
        }

        // Final pass — provenance + checksums.
        config.source = Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
            // v1 provenance — populated once the extractor learns to
            // fetch the upstream commit hash + safetensors digests.
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        });
        config.checksums = crate::format::checksums::compute_checksums(self.output_dir).ok();

        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;
        Ok(())
    }
}
