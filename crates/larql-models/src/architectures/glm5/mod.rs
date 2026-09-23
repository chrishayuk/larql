//! GLM-5.2 (`glm_moe_dsa`) architecture — MLA + MoE, with a DSA
//! sparse-attention indexer represented in config only.
//!
//! HF `zai-org/GLM-5.2`, `architectures: ["GlmMoeDsaForCausalLM"]`,
//! `model_type: "glm_moe_dsa"`.
//!
//! ## Scope — detection/extraction-tier only, same discipline as `deepseek_v4.rs`
//!
//! - **MLA + MoE shape is scaffolded from the real `config.json`** (fetched
//!   from the HF hub 2026-08-14) and tensor keys confirmed against the real
//!   `model.safetensors.index.json` weight map (see "Tensor keys" below).
//!   This half of the architecture is naming-identical to `deepseek.rs`
//!   (V3): same `model.` prefix, same `self_attn.{q,kv}_{a,b}_proj*` MLA
//!   naming, same `mlp.{gate,experts.{id},shared_experts}.*` MoE naming —
//!   reused via the base trait defaults and [`crate::tensor_keys::moe_experts`]
//!   rather than re-declared.
//! - **The DSA sparse-attention indexer is REPRESENTED IN CONFIG ONLY.**
//!   `index_topk` / `index_n_heads` / `index_head_dim` and the four
//!   `self_attn.indexer.*` tensor keys are exposed as inert facts via the
//!   `dsa_*` trait methods. **No compute implementation of the sparse
//!   top-k token-selection kernel exists anywhere in `larql` for it** —
//!   nothing downstream reads these values or these tensors. Implementing
//!   the indexer's forward pass (score the small side-attention, select
//!   top-k cached K/V entries, mask the real MLA attention accordingly)
//!   is unstarted work, out of scope here.
//! - This architecture is **detection/extraction-tier, not verified for
//!   inference** — identical scope discipline to `deepseek_v4.rs`: gate
//!   vectors, embeddings, and tensor-key metadata only.
//!
//! ## Known gap: no per-layer dense/MoE toggle
//!
//! The real config declares `first_k_dense_replace: 3` and
//! `mlp_layer_types` confirms layers 0-2 use a dense FFN
//! (`mlp.{gate,up,down}_proj.weight`) while layers 3-77 are MoE
//! (`mlp.gate.weight` + `mlp.experts.{id}.*` + `mlp.shared_experts.*`).
//! [`ModelArchitecture::is_moe`] is a whole-model flag with no `layer`
//! parameter, so it cannot express "MoE from layer 3 onward". This impl
//! does not paper over that: `is_moe()` returns `true` for the whole
//! model, which is imprecise for layers 0-2 exactly the same way it
//! already is for `deepseek.rs`'s DeepSeek-V3 (whose real checkpoints
//! carry the identical `first_k_dense_replace` shape and get no special
//! handling here either). No `is_moe_layer(layer)` hook exists on the
//! trait; one is deliberately not added speculatively — see the parent
//! programme notes for this gap.
//!
//! ## Tensor keys — confirmed from HF, not guessed
//!
//! Read directly off `zai-org/GLM-5.2`'s `model.safetensors.index.json`
//! `weight_map` (fetched 2026-08-14): the embedding, one dense attention
//! block (layer 0, pre-MoE), one MoE block (a layer past
//! `first_k_dense_replace`), and the indexer. All of the following are
//! **real, confirmed weight names**, not inferred from convention:
//!
//! - `model.embed_tokens.weight`, `lm_head.weight`
//! - `model.layers.0.self_attn.{q_a,q_b,kv_b,o}_proj.weight`,
//!   `model.layers.0.self_attn.kv_a_proj_with_mqa.weight`
//! - `model.layers.0.self_attn.indexer.{wq_b,wk,k_norm,weights_proj}.weight`
//! - `model.layers.0.mlp.{gate,up,down}_proj.weight` (dense FFN, layers 0-2)
//! - `model.layers.N.mlp.gate.weight`,
//!   `model.layers.N.mlp.gate.e_score_correction_bias` (N >= 3, MoE layers)
//! - `model.layers.N.mlp.experts.{id}.{gate,up,down}_proj.weight`
//! - `model.layers.N.mlp.shared_experts.{gate,up,down}_proj.weight`
//!
//! Not confirmed / not represented: `q_a_layernorm.weight` and
//! `kv_a_layernorm.weight` (the RMSNorm applied to the compressed MLA
//! latent between the `_a` and `_b` projections) are visible in the real
//! weight map but the trait has no accessor slot for them —
//! `deepseek.rs` has the identical gap for DeepSeek-V3, so this is not
//! new to GLM-5.2.

use crate::config::{ModelArchitecture, ModelConfig};
use crate::tensor_keys::moe_experts;

/// GLM-5.2 (`glm_moe_dsa`): MLA + MoE + DSA-indexer-in-config-only.
pub struct GlmMoeDsaArch {
    config: ModelConfig,
}

impl GlmMoeDsaArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for GlmMoeDsaArch {
    fn family(&self) -> &str {
        "glm_moe_dsa"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── MoE — standard HF per-expert layout, identical naming to `deepseek.rs` ──

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(256)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(8)
    }

    fn num_shared_experts(&self) -> usize {
        self.config.num_shared_experts.unwrap_or(1)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        moe_experts::router(&self.layer_prefix(layer))
    }

    fn moe_router_bias_key(&self, layer: usize) -> Option<String> {
        // Confirmed real key: `mlp.gate.e_score_correction_bias` — the
        // `noaux_tc` routing bias GLM-5.2 shares with DeepSeek-V3's router.
        Some(format!(
            "{}mlp.gate.e_score_correction_bias",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        moe_experts::gate_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        moe_experts::up_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        moe_experts::down_proj(&self.layer_prefix(layer), expert_id)
    }

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

    // ── MLA — same shape and naming as `deepseek.rs` (V3) ──

    fn uses_mla(&self) -> bool {
        self.config.kv_lora_rank.is_some()
    }

    fn kv_lora_rank(&self) -> usize {
        self.config.kv_lora_rank.unwrap_or(512)
    }

    fn q_lora_rank(&self) -> usize {
        self.config.q_lora_rank.unwrap_or(2048)
    }

    fn mla_qk_nope_head_dim(&self) -> Option<usize> {
        self.config.qk_nope_head_dim
    }

    fn mla_qk_rope_head_dim(&self) -> Option<usize> {
        self.config.qk_rope_head_dim
    }

    fn mla_v_head_dim(&self) -> Option<usize> {
        self.config.v_head_dim
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

    // ── DSA sparse-attention indexer — config/key facts only, no compute ──

    fn dsa_index_topk(&self) -> Option<usize> {
        self.config.index_topk
    }

    fn dsa_index_n_heads(&self) -> Option<usize> {
        self.config.index_n_heads
    }

    fn dsa_index_head_dim(&self) -> Option<usize> {
        self.config.index_head_dim
    }

    fn dsa_indexer_wq_b_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.indexer.wq_b.weight",
            self.layer_prefix(layer)
        ))
    }

    fn dsa_indexer_wk_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.indexer.wk.weight",
            self.layer_prefix(layer)
        ))
    }

    fn dsa_indexer_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.indexer.k_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn dsa_indexer_weights_proj_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.indexer.weights_proj.weight",
            self.layer_prefix(layer)
        ))
    }
}

#[cfg(test)]
mod tests;
