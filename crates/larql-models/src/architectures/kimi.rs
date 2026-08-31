//! Kimi Linear — hybrid KDA/MLA attention + sigmoid-routed MoE with a
//! bias-corrected selection and a shared expert.
//!
//! `model_type: "kimi_linear"` matches no family prefix in
//! [`super::super::detect::detect_from_json`], so without this file it
//! falls to [`super::generic::GenericArch`] — which answers every MoE key
//! method `None` (P3d-d: `vindex3 ops` reported 20,126 unclassified/missing
//! MoE operand defects on the real container). This overrides only the MoE
//! key methods; everything else — attention geometry, `mla_use_nope`
//! position policy, the recurrence topology `LinearAttnInterleave` reads —
//! stays on the trait's config-driven defaults, unchanged from
//! `GenericArch`'s behaviour.
//!
//! Every key below is read from `model.safetensors.index.json` on the real
//! `Kimi-Linear-48B-A3B-Instruct` checkpoint, not guessed from
//! `modeling_kimi.py` alone:
//! `block_sparse_moe.{gate.weight, gate.e_score_correction_bias,
//! shared_experts.{gate_proj,up_proj,down_proj}.weight,
//! experts.{id}.{w1,w2,w3}.weight}`. The `w1`/`w2`/`w3` → gate/down/up
//! mapping is [`KimiBlockSparseMLP.__init__`]'s own comment in the
//! checkpoint's `modeling_kimi.py` (`w1` gate, `w2` down, `w3` up) — the
//! same spelling Mixtral uses, but Mixtral ships no router bias and no
//! shared expert.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct KimiLinearArch {
    config: ModelConfig,
}

impl KimiLinearArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for KimiLinearArch {
    fn family(&self) -> &str {
        "kimi_linear"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── MoE router ──

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.gate.weight",
            self.layer_prefix(layer)
        ))
    }

    /// `e_score_correction_bias` — added to the sigmoid scores that CHOOSE
    /// the top-k expert ids; the WEIGHTS applied to those experts are
    /// gathered from the unbiased scores (`KimiMoEGate.forward`:
    /// `scores_for_choice = scores + bias` picks `topk_idx`, then
    /// `topk_weight = scores.gather(1, topk_idx)` reads the un-biased
    /// tensor). Selection and weighting read different tensors — the
    /// config alone does not say this, only the reference forward does.
    fn moe_router_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.gate.e_score_correction_bias",
            self.layer_prefix(layer)
        ))
    }

    // ── Routed experts: `w1`/`w2`/`w3`, not gate_proj/up_proj/down_proj ──

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.experts.{expert_id}.w1.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.experts.{expert_id}.w3.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.experts.{expert_id}.w2.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Shared expert: always active, standard gate/up/down naming ──
    //
    // `KimiMLP` (unlike `KimiBlockSparseMLP`) uses the ordinary
    // `gate_proj`/`up_proj`/`down_proj` names — the w1/w2/w3 permutation is
    // specific to the per-expert routed block.

    fn shared_expert_gate_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.shared_experts.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_up_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.shared_experts.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn shared_expert_down_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}block_sparse_moe.shared_experts.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── MLA (Multi-Latent Attention): every full-attention layer ──
    //
    // `KimiMLAAttention.__init__` — `assert self.q_lora_rank is None`, so
    // no `q_a_proj`/`q_b_proj` exists and `mla_q_a_key`/`mla_q_b_key` stay
    // on the trait default (`None`); only K/V are low-rank compressed.
    // Confirmed against the real checkpoint: `kv_a_proj_with_mqa
    // [576, 2304]` = `[kv_lora_rank(512)+qk_rope_head_dim(64), hidden]`,
    // `kv_b_proj [8192, 512]` = `[num_heads(32)·(nope(128)+v(128)),
    // kv_lora_rank]`, `q_proj [6144, 2304]` =
    // `[num_heads·(nope+rope)=32·192, hidden]`.

    fn uses_mla(&self) -> bool {
        true
    }

    fn kv_lora_rank(&self) -> usize {
        self.config().kv_lora_rank.unwrap_or(0)
    }

    fn mla_qk_nope_head_dim(&self) -> Option<usize> {
        self.config().qk_nope_head_dim
    }

    fn mla_qk_rope_head_dim(&self) -> Option<usize> {
        self.config().qk_rope_head_dim
    }

    fn mla_v_head_dim(&self) -> Option<usize> {
        self.config().v_head_dim
    }

    /// `KimiMLAAttention.__init__` builds the latent norm as
    /// `self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)` —
    /// `modeling_kimi.py:365`, no `eps` argument — so it runs at
    /// `KimiRMSNorm.__init__`'s own default, `eps=1e-6`
    /// (`modeling_kimi.py:225`). The config's `rms_norm_eps` is `1e-5`
    /// and governs every OTHER norm in the layer; this is the one that
    /// does not read it.
    ///
    /// Recorded here, carried through the graph, and consumed by the
    /// operator — this is the fact the ontology drill found the container
    /// could not carry (F6), living as `MLA_KV_A_NORM_EPS` inside a
    /// family-shaped executor where no deleted checkpoint could restore
    /// it.
    fn mla_kv_a_norm_eps(&self) -> Option<f64> {
        Some(1e-6)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal parsed config — mutated per test rather than guessing
    /// `config.json` spellings, same convention as `config::tests`.
    fn base_config() -> ModelConfig {
        crate::detect_from_json(&serde_json::json!({
            "model_type": "kimi_linear",
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

    fn arch() -> KimiLinearArch {
        KimiLinearArch::from_config(base_config())
    }

    #[test]
    fn family_is_kimi_linear_not_generic() {
        assert_eq!(arch().family(), "kimi_linear");
    }

    #[test]
    fn router_and_bias_keys_use_the_gate_component() {
        let a = arch();
        assert_eq!(
            a.moe_router_key(3).as_deref(),
            Some("layers.3.block_sparse_moe.gate.weight")
        );
        assert_eq!(
            a.moe_router_bias_key(3).as_deref(),
            Some("layers.3.block_sparse_moe.gate.e_score_correction_bias")
        );
    }

    /// The mapping checked against `modeling_kimi.py`'s own comment:
    /// `w1` gate, `w2` down, `w3` up — not alphabetic order.
    #[test]
    fn expert_keys_map_w1_w2_w3_to_gate_down_up() {
        let a = arch();
        assert_eq!(
            a.expert_ffn_gate_key(5, 12).as_deref(),
            Some("layers.5.block_sparse_moe.experts.12.w1.weight")
        );
        assert_eq!(
            a.expert_ffn_up_key(5, 12).as_deref(),
            Some("layers.5.block_sparse_moe.experts.12.w3.weight")
        );
        assert_eq!(
            a.expert_ffn_down_key(5, 12).as_deref(),
            Some("layers.5.block_sparse_moe.experts.12.w2.weight")
        );
    }

    #[test]
    fn shared_expert_keys_use_standard_gate_up_down_naming() {
        let a = arch();
        assert_eq!(
            a.shared_expert_gate_key(0).as_deref(),
            Some("layers.0.block_sparse_moe.shared_experts.gate_proj.weight")
        );
        assert_eq!(
            a.shared_expert_up_key(0).as_deref(),
            Some("layers.0.block_sparse_moe.shared_experts.up_proj.weight")
        );
        assert_eq!(
            a.shared_expert_down_key(0).as_deref(),
            Some("layers.0.block_sparse_moe.shared_experts.down_proj.weight")
        );
    }

    /// Everything not overridden above stays on the trait's config-driven
    /// defaults — the same behaviour Kimi got from `GenericArch` before
    /// this file existed. `is_moe`/`num_experts`/`num_shared_experts` read
    /// straight off the declaration (Kimi's field names are not aliased),
    /// so a plain `ModelConfig` with the routed-MoE fields set proves it
    /// without a JSON fixture.
    #[test]
    fn unoverridden_facts_stay_on_the_trait_defaults() {
        let mut config = base_config();
        config.num_experts = Some(256);
        config.num_experts_per_token = Some(8);
        config.num_shared_experts = Some(1);
        config.moe_intermediate_size = Some(1024);
        let a = KimiLinearArch::from_config(config);
        assert!(a.is_moe());
        assert_eq!(a.num_experts(), 256);
        assert_eq!(a.num_experts_per_token(), 8);
        assert_eq!(a.num_shared_experts(), 1);
        assert_eq!(a.moe_intermediate_size(), 1024);
    }

    /// `uses_mla` is unconditionally `true` — the checkpoint's full-
    /// attention layers are MLA whatever else the config declares. The
    /// four geometry facts pass through from the real config keys, not
    /// aliased spellings.
    #[test]
    fn mla_is_always_declared_and_reads_the_real_geometry_keys() {
        let mut config = base_config();
        config.kv_lora_rank = Some(512);
        config.qk_nope_head_dim = Some(128);
        config.qk_rope_head_dim = Some(64);
        config.v_head_dim = Some(128);
        let a = KimiLinearArch::from_config(config);
        assert!(a.uses_mla());
        assert_eq!(a.kv_lora_rank(), 512);
        assert_eq!(a.mla_qk_nope_head_dim(), Some(128));
        assert_eq!(a.mla_qk_rope_head_dim(), Some(64));
        assert_eq!(a.mla_v_head_dim(), Some(128));
    }

    /// No `q_lora_rank` on this checkpoint (`assert self.q_lora_rank is
    /// None` in `modeling_kimi.py`) — the compressed-Q key methods stay
    /// on the trait default rather than inventing tensors that do not
    /// exist.
    #[test]
    fn no_query_compression_keys_are_declared() {
        let a = arch();
        assert_eq!(a.mla_q_a_key(0), None);
        assert_eq!(a.mla_q_b_key(0), None);
    }

    #[test]
    fn mla_kv_keys_name_the_real_tensors() {
        let a = arch();
        assert_eq!(
            a.mla_kv_a_key(3).as_deref(),
            Some("layers.3.self_attn.kv_a_proj_with_mqa.weight")
        );
        assert_eq!(
            a.mla_kv_b_key(3).as_deref(),
            Some("layers.3.self_attn.kv_b_proj.weight")
        );
    }
}
