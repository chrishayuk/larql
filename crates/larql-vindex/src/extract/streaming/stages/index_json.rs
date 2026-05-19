//! Stage 5 — assemble + write `index.json` (preliminary; checksums
//! added later in `finalize`).

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::streaming::context::StreamingContext;
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 5 — assemble + write `index.json` (preliminary; checksums
    /// added later in `finalize`).
    pub(in crate::extract::streaming) fn write_index_json(&mut self) -> Result<(), VindexError> {
        let cfg = self.arch.config();
        let family = self.arch.family().to_string();
        let layer_infos = std::mem::take(&mut self.layer_infos);
        let config = VindexConfig {
            version: 2,
            model: self.model_name.to_string(),
            family: family.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            embed_scale: self.embed_scale,
            layers: layer_infos,
            down_top_k: self.down_top_k,
            has_model_weights: false,
            source: Some(crate::VindexSource {
                huggingface_repo: Some(self.model_name.to_string()),
                huggingface_revision: None,
                safetensors_sha256: None,
                extracted_at: crate::extract::build_helpers::chrono_now(),
                larql_version: env!("CARGO_PKG_VERSION").to_string(),
                // v1 provenance — populated once the extractor learns
                // to fetch the upstream commit hash + safetensors
                // digests.
                base_model_sha: None,
                extractor_sha: None,
                base_safetensors_sha256: None,
            }),
            checksums: None,
            extract_level: self.extract_level,
            dtype: self.dtype,
            quant: self.quant,
            layer_bands: crate::LayerBands::for_family(&family, self.num_layers),
            model_config: Some(VindexModelConfig {
                model_type: cfg.model_type.clone(),
                head_dim: cfg.head_dim,
                num_q_heads: cfg.num_q_heads,
                num_kv_heads: cfg.num_kv_heads,
                rope_base: cfg.rope_base,
                sliding_window: cfg.sliding_window,
                moe: if self.is_moe {
                    Some(crate::MoeConfig {
                        num_experts: self.n_experts,
                        top_k: self.arch.num_experts_per_token(),
                        shared_expert: self.arch.num_shared_experts() > 0,
                        router_type: self.arch.moe_router_type().to_string(),
                        moe_intermediate_size: if self.arch.moe_intermediate_size() > 0 {
                            Some(self.arch.moe_intermediate_size())
                        } else {
                            None
                        },
                        hybrid: self.arch.is_hybrid_moe(),
                    })
                } else {
                    None
                },
                // Per-layer geometry (Gemma 4)
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
            }),
            fp4: None,
            ffn_layout: None,
        };

        // Write preliminary index.json (needed by write_model_weights which reads dtype from it).
        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;
        Ok(())
    }
}
