//! Model-architecture config carried in `index.json` so the
//! architecture can be reconstructed without the original
//! `config.json`.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup.
//!
//! ## This struct is a lossy projection, and the loss is load-bearing
//!
//! Every field a checkpoint declares that reaches the forward pass has
//! to appear here or the served model silently differs from the
//! checkpoint. That is not hypothetical: `rope_scaling` was absent
//! until 2026-08-06, so `gemma-3-4b-it` — whose `config.json` says
//! `{"factor": 8.0, "rope_type": "linear"}` — was served with a
//! position divisor of 1.0 on its five global layers instead of 8.0.
//! CPU and Metal read the same `index.json`, so both were wrong in the
//! same way and the CPU-vs-Metal parity suite stayed green. A parity
//! gate cannot see a defect in the config both of its arms share.
//!
//! `model_config_persists_every_forward_affecting_field` pins the
//! inventory. When you add a field to `larql_models::ModelConfig`, that
//! test tells you to either persist it here or record why it does not
//! need persisting. `embedding_multiplier` is the standing example of
//! the second case: it round-trips through the top-level
//! `VindexConfig.embed_scale` instead, and duplicating it here would
//! create a second source of truth for one number.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct VindexModelConfig {
    pub model_type: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// The window's explicit enable flag, persisted separately from the
    /// window itself because they are separate declarations: Qwen2.5
    /// ships a 32768 window beside `use_sliding_window: false`. Dropping
    /// the flag on the way into a vindex would serve the window against
    /// the checkpoint's instruction.
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    /// How far up the stack an enabled window applies.
    #[serde(default)]
    pub max_window_layers: Option<usize>,
    /// The declared positional scheme, verbatim.
    ///
    /// Persisted because on `granitemoehybrid` it is what turns rotation
    /// on at all — a container that dropped it would rebuild the model as
    /// a NoPE model, or as a rotating one, depending only on which side of
    /// the boundary the question was asked.
    #[serde(default)]
    pub position_embedding_type: Option<String>,
    /// The per-layer rotary schedule, in the checkpoint's own polarity:
    /// `1` rotates, `0` is NoPE.
    ///
    /// Persisted because it decides which layers encode position at all.
    /// Dropping it would rebuild SmolLM3 as a model that rotates
    /// everywhere — fluent, and wrong on 9 of 36 layers.
    #[serde(default)]
    pub no_rope_layers: Option<Vec<i64>>,
    /// The interval fallback, persisted so a container that carried no
    /// explicit mask still knows the schedule it was compiled under.
    #[serde(default)]
    pub no_rope_layer_interval: Option<usize>,
    /// The declared rotary pairing, persisted so the container records
    /// what the checkpoint claimed rather than what this build does.
    #[serde(default)]
    pub rope_interleaved: Option<bool>,
    /// The declared multi-axis flag, persisted for the same reason.
    #[serde(default)]
    pub use_mrope: Option<bool>,
    /// Falcon's one-word FFN shape (`activation`), persisted so the
    /// container records the claim the boundary checked.
    #[serde(default)]
    pub ffn_shape_name: Option<String>,
    /// The declared `is_llama_config` flag, persisted for the same reason.
    #[serde(default)]
    pub is_llama_config: Option<bool>,
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
    /// Gemma 3n/4-E: double-wide MLP on the KV-shared layers, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_double_wide_mlp: Option<bool>,
    /// Gemma 3n/4-E: the per-layer-input embedding vocabulary, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vocab_size_per_layer_input: Option<u64>,
    /// RoPE base for local/sliding window layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_local_base: Option<f64>,
    /// Per-layer declared rope theta, verbatim from `layer_rope_theta` —
    /// `0.0` entries are the upstream NoPE sentinel, interpreted only by
    /// `ModelArchitecture::position_policy_for_layer` after the round-trip.
    /// Dropping this served every NoPE layer with full-strength rotation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_rope_theta: Option<Vec<f64>>,
    /// Query pre-attention scalar (overrides 1/sqrt(head_dim)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_pre_attn_scalar: Option<f64>,
    /// Extra attention-score multiplier on top of 1/sqrt(head_dim)
    /// (`qk_scale_factor`). Distinct from `query_pre_attn_scalar`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qk_scale_factor: Option<f64>,
    /// Multiplier on the final hidden state before the vocab projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_multiplier: Option<f64>,
    /// Post-norm epsilon when it differs from `norm_eps` (1e-8 vs 1e-5 on
    /// the same checkpoint is a real shape).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_norm_eps: Option<f64>,
    /// Whether attention projections carry biases, when declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<bool>,
    /// FFN activation name, verbatim (`hidden_act`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_act: Option<String>,
    /// Declared context bound (`max_position_embeddings`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_position_embeddings: Option<usize>,
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

    // ── Fields that were dropped until 2026-08-06 ──
    // Each is read by the forward pass and each was absent from this
    // struct, so no vindex-served model ever saw it. All are
    // `#[serde(default)]`, so vindexes written before this lands still
    // load — they just keep answering `None`, which is what they
    // already did. Re-extract to pick the values up.
    /// RoPE scaling block, in the `config.json` shape
    /// (`larql_models::RopeScaling::to_config_json`). Carried as raw
    /// JSON rather than a typed mirror so the detector's parser stays
    /// the single definition of how each family is read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<serde_json::Value>,
    /// Gemma 2 attention-logit softcapping. Note `final_logit_softcapping`
    /// was already persisted and this one was not — the pair splits
    /// across the same model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_logit_softcapping: Option<f64>,
    /// GPT-OSS clamp on both halves of the fused gate/up projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swiglu_limit: Option<f64>,
    /// OLMoE / Mixtral router top-k renormalisation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_topk_prob: Option<bool>,
    /// Whether `lm_head` is tied to the embedding matrix. ROADMAP H5a
    /// made an untied-but-missing `lm_head` a hard error in
    /// `larql-models`; without this field that fix could not reach a
    /// vindex-served model, which always answered `None` (= "absent, no
    /// claim either way").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tie_word_embeddings: Option<bool>,
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
    /// That branch's intermediate width, where the judgment declares one.
    /// Carried beside the boolean rather than derived from
    /// [`Self::moe_intermediate_size`]: the DeepSeek/Kimi lineage sizes
    /// the branch as one wider FFN at `moe_intermediate_size * count`
    /// while Qwen sizes it from its own key, and on Qwen1.5-MoE the two
    /// answers differ fourfold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_intermediate_size: Option<usize>,
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
            use_sliding_window: cfg.use_sliding_window,
            max_window_layers: cfg.max_window_layers,
            position_embedding_type: cfg.position_embedding_type.clone(),
            no_rope_layers: cfg.no_rope_layers.clone(),
            no_rope_layer_interval: cfg.no_rope_layer_interval,
            rope_interleaved: cfg.rope_interleaved,
            use_mrope: cfg.use_mrope,
            ffn_shape_name: cfg.ffn_shape_name.clone(),
            is_llama_config: cfg.is_llama_config,
            moe: if arch.is_moe() {
                Some(MoeConfig {
                    num_experts: arch.num_experts(),
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    shared_expert_intermediate_size: arch.shared_expert_intermediate_size(),
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
            use_double_wide_mlp: cfg.use_double_wide_mlp,
            vocab_size_per_layer_input: cfg.vocab_size_per_layer_input,
            rope_local_base: cfg.rope_local_base,
            layer_rope_theta: cfg.layer_rope_theta.clone(),
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            qk_scale_factor: cfg.qk_scale_factor,
            output_multiplier: cfg.output_multiplier,
            post_norm_eps: cfg.post_norm_eps,
            attention_bias: cfg.attention_bias,
            hidden_act: cfg.hidden_act.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            final_logit_softcapping: cfg.final_logit_softcapping,
            attention_multiplier: cfg.attention_multiplier,
            residual_multiplier: cfg.residual_multiplier,
            logits_scaling: cfg.logits_scaling,
            norm_eps: cfg.norm_eps,
            rope_scaling: cfg.rope_scaling.as_ref().map(|rs| rs.to_config_json()),
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            swiglu_limit: cfg.swiglu_limit,
            norm_topk_prob: cfg.norm_topk_prob,
            tie_word_embeddings: cfg.tie_word_embeddings,
        }
    }
}

#[cfg(test)]
mod tests {
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
        ];
        // Genuinely not persisted yet. Each entry is a known gap, not an
        // exemption: no vindex-served model can use these today.
        const KNOWN_GAPS: &[&str] = &[
            // Multi-head latent attention (DeepSeek V2/V3). No MLA model
            // is served from a vindex yet; serving one without these
            // would silently rebuild the wrong attention geometry.
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
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

        let src = include_str!("../../../larql-models/src/config/model_config.rs");
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
        let own = include_str!("model.rs");
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
        cfg.max_position_embeddings = Some(131072);

        let json = serde_json::to_string(&cfg).unwrap();
        let back: VindexModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.qk_scale_factor, Some(3.87));
        assert_eq!(back.output_multiplier, Some(0.196));
        assert_eq!(back.post_norm_eps, Some(1e-8));
        assert_eq!(back.attention_bias, Some(false));
        assert_eq!(back.hidden_act.as_deref(), Some("silu"));
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
}
