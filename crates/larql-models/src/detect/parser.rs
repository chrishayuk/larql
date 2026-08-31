//! Parse a `config.json` JSON value into [`ModelConfig`].
//!
//! Handles both top-level and nested `text_config` (multimodal) layouts.
//! Optional fields with widely-accepted architecture-class defaults
//! (head_dim for Gemma, num_kv_heads, rope_theta) fall through to those
//! defaults; required topology fields (see [`super::config_io`]) are
//! validated by the caller before this runs.

use crate::config::{ModelConfig, RopeScaling};

use super::config_io::{
    CONFIG_KEY_HIDDEN_SIZE_ALIASES, CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES,
    CONFIG_KEY_NUM_ATTENTION_HEADS_ALIASES, CONFIG_KEY_NUM_HIDDEN_LAYERS_ALIASES,
    CONFIG_KEY_TEXT_CONFIG,
};

// ── RoPE base defaults ───────────────────────────────────────────────────────
// Shared with `architectures/gemma{3,4}.rs` and `config.rs` via `defaults`,
// so the loader fallback and the per-arch fallback agree.
use crate::defaults::{ROPE_BASE_DEFAULT, ROPE_BASE_GEMMA};

// ── Architecture-class defaults for attention-shape fields ──────────────────
// These are NOT topology guesses — they're the values transformers uses
// when an HF config omits the field for the corresponding model class.
// They only surface from the in-memory `detect_from_json` path; the disk
// path enforces presence of topology fields in
// `config_io::require_config_fields` so no on-disk model silently picks
// up an architecture-class default it shouldn't.

/// Transformers default for `num_attention_heads` when the config omits it.
const DEFAULT_NUM_ATTENTION_HEADS: u64 = 8;

/// Transformers default for `num_key_value_heads` when the config omits it.
const DEFAULT_NUM_KV_HEADS: u64 = 4;

/// Gemma-family default `head_dim` when the config omits it. Other archs
/// derive `head_dim = hidden_size / num_attention_heads`.
const DEFAULT_HEAD_DIM_GEMMA: usize = 256;

/// Family-prefix that triggers Gemma-specific defaults (RoPE base and
/// `head_dim` fallback). Comes from HF `model_type` naming
/// (`gemma`, `gemma2`, `gemma3`, `gemma3_text`, `gemma4`, ...).
const MODEL_TYPE_PREFIX_GEMMA: &str = "gemma";

// ── Config field name aliases ────────────────────────────────────────────────
// Different model families use different JSON keys for the same concept.
// Ordering is priority: first match wins.

/// Total routed expert count: DeepSeek, Qwen MoE, Mixtral variants.
const NUM_EXPERTS_KEYS: &[&str] = &["n_routed_experts", "num_local_experts", "num_experts"];

/// Experts activated per token: llama.cpp / HF spelling variants.
const NUM_EXPERTS_PER_TOK_KEYS: &[&str] = &["num_experts_per_tok", "num_experts_per_token"];

/// Shared-expert count. DeepSeek-lineage checkpoints write
/// `n_shared_experts`; Kimi Linear writes `num_shared_experts`. One fact,
/// and reading only the first spelling silently drops the always-on branch.
const NUM_SHARED_EXPERTS_KEYS: &[&str] = &["n_shared_experts", "num_shared_experts"];

/// Whether the router renormalises its selected top-k probabilities.
/// `norm_topk_prob` in the DeepSeek lineage, `moe_renormalize` on Kimi
/// Linear. The two settings differ by a rescale of the whole expert
/// branch, so a default here is a quiet numerical change.
const NORM_TOPK_PROB_KEYS: &[&str] = &["norm_topk_prob", "moe_renormalize"];

/// Router scoring function: `scoring_func` (DeepSeek, GLM-5.3-Flash) or
/// `moe_router_activation_func` (Kimi Linear).
const ROUTER_ACTIVATION_KEYS: &[&str] = &["scoring_func", "moe_router_activation_func"];

/// Expert-group count: `n_group` (DeepSeek, GLM-5.3-Flash) or
/// `num_expert_group` (Kimi Linear).
const EXPERT_GROUP_KEYS: &[&str] = &["n_group", "num_expert_group"];

/// Return the first `u64` found under any of `keys` in `config`.
fn field_u64(config: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| config[k].as_u64())
}

/// Read a topology field by alias list as `usize`, preferring `text_config`
/// (multimodal nesting) and falling back to the top-level object. The first
/// alias to resolve wins. Returns 0 when no alias is present; the configured
/// field validators reject 0 at the next layer, so the magic-number guess
/// defaults (e.g. 2048) don't leak in and masquerade as a real model topology.
///
/// Alias lists live in `config_io.rs` so the loader's `require_config_fields`
/// validator and this parser agree on what names are acceptable for each
/// canonical field — see [`super::config_io::CONFIG_KEY_HIDDEN_SIZE_ALIASES`]
/// (GPT-2's `n_embd` etc.).
fn topology_field(
    config: &serde_json::Value,
    text_config: &serde_json::Value,
    aliases: &[&str],
) -> usize {
    super::config_io::read_aliased_u64(config, text_config, aliases).unwrap_or(0) as usize
}

/// Parse [`ModelConfig`] from a `config.json` JSON value.
pub(super) fn parse_model_config(config: &serde_json::Value) -> ModelConfig {
    let text_config = config.get(CONFIG_KEY_TEXT_CONFIG).unwrap_or(config);

    // Detect model_type from text_config or top level. The mamba_ssm
    // package writes no `model_type` at all — its configs identify the
    // model by the layer class named in `ssm_cfg.layer`, and "Mamba2"
    // there is the same family fact transformers spells `model_type:
    // "mamba2"`. A judged spelling, exact match only: any other class
    // name stays undeclared rather than acquiring a family.
    let model_type = text_config["model_type"]
        .as_str()
        .or_else(|| config["model_type"].as_str())
        .or_else(|| {
            (text_config["ssm_cfg"]["layer"].as_str() == Some("Mamba2")).then_some("mamba2")
        })
        .unwrap_or("")
        .to_string();

    // Pick defaults based on model type.
    let is_gemma = model_type.starts_with(MODEL_TYPE_PREFIX_GEMMA);
    let rope_default = if is_gemma {
        ROPE_BASE_GEMMA
    } else {
        ROPE_BASE_DEFAULT
    };

    // Required topology fields. On the disk path `detect_architecture`
    // already errored when any of these are absent, so a zero here only
    // surfaces from `detect_from_json` callers who pass partial JSON
    // (test ergonomics); the validator catches the zero downstream
    // rather than letting a magic-number default impersonate a real
    // topology and panic deep inside extract.
    let num_layers = topology_field(config, text_config, CONFIG_KEY_NUM_HIDDEN_LAYERS_ALIASES);
    let hidden_size = topology_field(config, text_config, CONFIG_KEY_HIDDEN_SIZE_ALIASES);
    let mut intermediate_size =
        topology_field(config, text_config, CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES);
    // GPT-2 doesn't ship `n_inner` and HF computes intermediate_size as
    // `4 * n_embd` at the model boundary. Reproduce that here so the
    // validator (which has already accepted the missing field via the
    // gpt2-specific alias rule) doesn't surface a 0.
    if intermediate_size == 0 && model_type == "gpt2" && hidden_size > 0 {
        intermediate_size = 4 * hidden_size;
    }
    // The Mamba2 mixer's declared geometry, all fields or none. Read
    // before the attention-shape fields because it changes what their
    // absence means (below).
    let mamba2_read = crate::config::Mamba2Geometry::read_with_provenance(text_config);
    let mamba2_geometry = mamba2_read.as_ref().map(|(g, _)| *g);
    let mamba2_provenance = mamba2_read.map(|(_, p)| p);
    // The hybrid stack's conv-QKV attention block, all fields or none.
    // Its presence changes what "attention-shaped" means below: a hybrid
    // declares real attention heads beside the mixer geometry.
    let conv_qkv_read = crate::config::ConvQkvAttnGeometry::read_with_provenance(text_config);
    let conv_qkv_attn = conv_qkv_read.as_ref().map(|(g, _)| *g);
    let conv_qkv_provenance = conv_qkv_read.map(|(_, p)| p);
    // `attn_cfg.causal` — the block's masking, declared. Our operator is
    // causal by construction; a declared `false` must block downstream,
    // never silently run causal anyway.
    let attn_causal = text_config["attn_cfg"]["causal"].as_bool();
    // Gemma HF configs commonly omit num_attention_heads, head_dim, and
    // num_key_value_heads — they're architecture-class defaults from
    // transformers. See the `DEFAULT_*` constants for the values used.
    //
    // The defaults are attention-class facts, so they apply only to a
    // config that is attention-shaped. A checkpoint declaring a complete
    // Mamba2 mixer geometry and no attention-head key has NO attention
    // heads — transformers' own Mamba2Config carries no
    // `num_attention_heads` at all — and fabricating 8/4 here is how a
    // pure-SSM stack was once reported as a 48-layer softmax tower with
    // invented head geometry (ontology drill F1, observed live on
    // mamba2-780m). Zero is the parser's ordinary "absent" sentinel, and
    // the architecture's own validation judges what absence means.
    let attention_free_ssm = mamba2_geometry.is_some() && conv_qkv_attn.is_none();
    let default_head_dim: usize = if is_gemma { DEFAULT_HEAD_DIM_GEMMA } else { 0 };
    let num_q_heads = super::config_io::read_aliased_u64(
        config,
        text_config,
        CONFIG_KEY_NUM_ATTENTION_HEADS_ALIASES,
    )
    .unwrap_or(if attention_free_ssm {
        0
    } else {
        DEFAULT_NUM_ATTENTION_HEADS
    }) as usize;
    // head_dim: explicit config value, Gemma class default, or compute
    // from hidden/heads (the conventional MHA invariant). On a Mamba2
    // declaration the explicit value is the MIXER head width — the same
    // number `Mamba2Geometry` carries — not a softmax head's.
    // A hybrid declares its attention head width apart from the mixer's
    // (`attention_head_dim`), and it is NOT `hidden_size / num_heads` —
    // 16 · 128 = 2048 ≠ 1024 on OuteAI Mamba2Attn — so the derivation
    // below must not answer for it.
    let head_dim = text_config["head_dim"]
        .as_u64()
        .map(|v| v as usize)
        .or(conv_qkv_attn.map(|a| a.head_dim))
        .unwrap_or(if default_head_dim > 0 {
            default_head_dim
        } else {
            hidden_size.checked_div(num_q_heads).unwrap_or(0)
        });
    let num_kv_heads =
        text_config["num_key_value_heads"]
            .as_u64()
            .unwrap_or(if attention_free_ssm {
                0
            } else {
                DEFAULT_NUM_KV_HEADS
            }) as usize;
    // RoPE base, in declaration-specificity order:
    //  1. rope_parameters.full_attention.rope_theta — Gemma 4's structured
    //     per-layer-type form;
    //  2. rope_parameters.rope_theta — the transformers-5.x flat form
    //     (`rope_parameters: {rope_theta: N, rope_type: "default"}`), which
    //     replaces the legacy top-level field in new checkpoints;
    //  3. rope_theta at the top level — the legacy flat form;
    //  4. the architecture-class default.
    //
    // Form 2 was silently skipped until the Muse-Glimmer inventory caught the
    // fallthrough: a checkpoint declaring θ=500000 in the flat 5.x form
    // resolved to the 10000 default — the §4.7.8 shape on a brand-new key
    // spelling. Any transformers-5.x checkpoint hits this, not one family.
    let rope_params = text_config.get("rope_parameters");
    let rope_base = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa["rope_theta"].as_f64())
        .or_else(|| rope_params.and_then(|rp| rp["rope_theta"].as_f64()))
        .or_else(|| text_config["rope_theta"].as_f64())
        .unwrap_or(rope_default);
    // Per-layer declared theta array (`layer_rope_theta`), kept verbatim —
    // including `0.0` NoPE sentinels. The sentinel is interpreted exactly
    // once, in `ModelArchitecture::position_policy_for_layer`.
    let layer_rope_theta = text_config.get("layer_rope_theta").and_then(|lt| {
        lt.as_array()
            .map(|arr| arr.iter().filter_map(serde_json::Value::as_f64).collect())
    });
    // Local RoPE base for sliding window layers: check rope_parameters.sliding_attention,
    // then rope_local_base_freq.
    let rope_local_base = rope_params
        .and_then(|rp| rp.get("sliding_attention"))
        .and_then(|sa| sa["rope_theta"].as_f64())
        .or_else(|| text_config["rope_local_base_freq"].as_f64());
    let vocab_size = text_config["vocab_size"].as_u64().map(|v| v as usize);
    let sliding_window = text_config["sliding_window"].as_u64().map(|v| v as usize);
    // Read from the *outer* config too: some families declare it at the top
    // level next to `architectures` rather than inside `text_config`.
    // The mamba_ssm lineage (OuteAI Mamba2Attn) spells the same fact
    // `tie_embedding_weights` and declares no canonical spelling beside
    // it, so this is a read, not an alias-table entry.
    // Three spellings of one fact: transformers' `tie_word_embeddings`,
    // OuteAI's `tie_embedding_weights`, mamba_ssm's own `tie_embeddings`.
    let tie_word_embeddings = text_config
        .get("tie_word_embeddings")
        .or_else(|| config.get("tie_word_embeddings"))
        .or_else(|| text_config.get("tie_embedding_weights"))
        .or_else(|| text_config.get("tie_embeddings"))
        .and_then(|v| v.as_bool());

    // MoE fields
    let num_experts = field_u64(text_config, NUM_EXPERTS_KEYS).map(|v| v as usize);
    let num_experts_per_token =
        field_u64(text_config, NUM_EXPERTS_PER_TOK_KEYS).map(|v| v as usize);
    let num_shared_experts = field_u64(text_config, NUM_SHARED_EXPERTS_KEYS).map(|v| v as usize);
    // Gemma 4 A4B hybrid MoE fields
    let enable_moe_block = text_config["enable_moe_block"].as_bool().unwrap_or(false);
    let top_k_experts = text_config["top_k_experts"].as_u64().map(|v| v as usize);
    let moe_intermediate_size = text_config["moe_intermediate_size"]
        .as_u64()
        .map(|v| v as usize);
    // GPT-OSS clamps both halves of the fused gate/up projection at ±this
    // value before the GLU. Read rather than hardcoded: it is a published
    // config field and a future checkpoint may pick a different bound.
    let swiglu_limit = text_config["swiglu_limit"].as_f64();
    // Whether the router renormalises its selected top-k probabilities.
    // Read rather than assumed: the same architecture ships both settings, and
    // the two differ by a rescale of the whole expert branch.
    let norm_topk_prob = NORM_TOPK_PROB_KEYS
        .iter()
        .find_map(|key| text_config[*key].as_bool());
    let router_activation = ROUTER_ACTIVATION_KEYS
        .iter()
        .find_map(|key| text_config[*key].as_str().map(str::to_string));
    // Declared MoE facts carried verbatim so the plan can judge them.
    // Reading them is not endorsing them: a key nothing reads grades
    // "read by nothing in any registered parser" and blocks with no
    // account of WHY, where a key that is read faces its carriage rule and
    // blocks — or clears — for a stated reason.
    let routed_scaling_factor = text_config["routed_scaling_factor"].as_f64();
    let expert_groups = field_u64(text_config, EXPERT_GROUP_KEYS).map(|v| v as usize);
    let topk_group = text_config["topk_group"].as_u64().map(|v| v as usize);
    let use_grouped_topk = text_config["use_grouped_topk"].as_bool();
    let moe_layer_freq = text_config["moe_layer_freq"].as_u64().map(|v| v as usize);
    let first_k_dense_replace = text_config["first_k_dense_replace"]
        .as_u64()
        .map(|v| v as usize);
    let mla_use_nope = text_config["mla_use_nope"].as_bool();
    let model_max_length = text_config["model_max_length"].as_u64().map(|v| v as usize);

    // MLA fields
    let kv_lora_rank = text_config["kv_lora_rank"].as_u64().map(|v| v as usize);
    let q_lora_rank = text_config["q_lora_rank"].as_u64().map(|v| v as usize);
    let qk_nope_head_dim = text_config["qk_nope_head_dim"].as_u64().map(|v| v as usize);
    let qk_rope_head_dim = text_config["qk_rope_head_dim"].as_u64().map(|v| v as usize);
    let v_head_dim = text_config["v_head_dim"].as_u64().map(|v| v as usize);

    // RoPE scaling. Four shapes appear in the wild:
    //
    // 1. Flat with `factor` (Llama 2-style linear, simple `rope_type=linear`).
    // 2. `rope_type=llama3` with the four wavelength-band fields below.
    // 3. Gemma 3 structured per-layer-type:
    //      `{full_attention: {rope_type: linear, factor: N, ...},
    //        sliding_attention: {rope_type: default, ...}}`
    //    In that shape, only the `full_attention` slot carries a non-default
    //    scaling — sliding layers use plain RoPE — so we lift its `rope_type`
    //    + `factor` and mark `gemma3_global_only = true`.
    // 4. Missing entirely (older Llama, Mistral) → `None`.
    //
    // And two *homes* for any of those shapes: the legacy `rope_scaling`
    // key, and transformers-5.x's `rope_parameters`, which carries theta AND
    // scaling in one block (`{rope_theta, rope_type: "yarn", factor, …}`).
    // The theta read above already prefers `rope_parameters`; the scaling
    // read must too, or a 5.x checkpoint's YaRN block is dropped at parse
    // while its theta is honoured — the §4.7.8 shape again, caught by the
    // VINDEX3 carriage test on a Glimmer-shaped fixture. A `rope_parameters`
    // block that declares no scaling (`rope_type: "default"`, no `factor`)
    // parses to `None` and the legacy key is consulted.
    let parse_rope_scaling = |rs: &serde_json::Value| -> Option<RopeScaling> {
        // Gemma 3 per-layer-type form.
        if let Some(full) = rs.get("full_attention") {
            let scaling_type = full
                .get("rope_type")
                .or_else(|| full.get("type"))
                .and_then(|v| v.as_str())?
                .to_string();
            let factor = full.get("factor")?.as_f64()?;
            return Some(RopeScaling {
                scaling_type,
                factor,
                llama3_low_freq_factor: None,
                llama3_high_freq_factor: None,
                llama3_original_max_position_embeddings: None,
                yarn_beta_fast: None,
                yarn_beta_slow: None,
                yarn_truncate: None,
                yarn_mscale: None,
                yarn_mscale_all_dim: None,
                gemma3_global_only: true,
            });
        }
        // Flat form (Llama, Mistral, Gemma 1/2, GPT-OSS, DeepSeek, etc.).
        let scaling_type = rs
            .get("type")
            .or_else(|| rs.get("rope_type"))
            .and_then(|v| v.as_str())?
            .to_string();
        let factor = rs.get("factor")?.as_f64()?;
        let llama3_low = rs.get("low_freq_factor").and_then(|v| v.as_f64());
        let llama3_high = rs.get("high_freq_factor").and_then(|v| v.as_f64());
        let llama3_old_ctx = rs
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_f64());
        // YaRN band bounds. Absent means "use the paper's defaults" (32 / 1),
        // which is what `_compute_yarn_parameters` falls back to — so `None`
        // here is a real value downstream, not a missing one. `truncate`
        // decides whether the correction range is rounded outward to integer
        // dimensions; HF defaults it to true and GPT-OSS ships false.
        let yarn_beta_fast = rs.get("beta_fast").and_then(|v| v.as_f64());
        let yarn_beta_slow = rs.get("beta_slow").and_then(|v| v.as_f64());
        let yarn_truncate = rs.get("truncate").and_then(|v| v.as_bool());
        // DeepSeek's two extra amplitude knobs. They must be parsed even
        // though no R1 checkpoint uses them: when *both* are present HF
        // computes the attention factor as a *ratio* that typically collapses
        // to 1.0, where the single-argument form would give 1.35. Reading
        // yarn without reading these would newly apply a wrong amplitude to
        // every DeepSeek layer.
        let yarn_mscale = rs.get("mscale").and_then(|v| v.as_f64());
        let yarn_mscale_all_dim = rs.get("mscale_all_dim").and_then(|v| v.as_f64());
        Some(RopeScaling {
            scaling_type,
            factor,
            llama3_low_freq_factor: llama3_low,
            llama3_high_freq_factor: llama3_high,
            llama3_original_max_position_embeddings: llama3_old_ctx,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_truncate,
            yarn_mscale,
            yarn_mscale_all_dim,
            gemma3_global_only: false,
        })
    };
    let rope_scaling = rope_params
        .and_then(parse_rope_scaling)
        .or_else(|| text_config.get("rope_scaling").and_then(parse_rope_scaling));

    // RMS-norm / LayerNorm epsilon. Field-name aliases across families:
    //  - `rms_norm_eps`           — Llama, Mistral, Gemma
    //  - `layer_norm_eps`         — BERT-family
    //  - `layer_norm_epsilon`     — GPT-2
    //  - `norm_epsilon`           — StarCoder2
    // Most modern archs ship 1e-5; older ones used 1e-6. None → arch default.
    let norm_eps = text_config["rms_norm_eps"]
        .as_f64()
        .or_else(|| text_config["layer_norm_eps"].as_f64())
        .or_else(|| text_config["layer_norm_epsilon"].as_f64())
        .or_else(|| text_config["norm_epsilon"].as_f64());

    // Softcapping and attention scale
    let attn_logit_softcapping = text_config["attn_logit_softcapping"].as_f64();
    let final_logit_softcapping = text_config["final_logit_softcapping"].as_f64();
    let query_pre_attn_scalar = text_config["query_pre_attn_scalar"].as_f64();

    // Granite-style scaling multipliers
    let embedding_multiplier = text_config["embedding_multiplier"].as_f64();
    let residual_multiplier = text_config["residual_multiplier"].as_f64();
    let attention_multiplier = text_config["attention_multiplier"].as_f64();
    let logits_scaling = text_config["logits_scaling"].as_f64();

    // Per-layer attention geometry (Gemma 4 style)
    let global_head_dim = text_config["global_head_dim"].as_u64().map(|v| v as usize);
    let num_global_kv_heads = text_config["num_global_key_value_heads"]
        .as_u64()
        .map(|v| v as usize);
    // Partial rotary factor: check rope_parameters.full_attention first (Gemma 4),
    // then top-level partial_rotary_factor.
    let partial_rotary_factor = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa["partial_rotary_factor"].as_f64())
        .or_else(|| text_config["partial_rotary_factor"].as_f64());
    // Sliding window pattern: explicit sliding_window_pattern field, or infer later.
    let sliding_window_pattern = text_config["sliding_window_pattern"]
        .as_u64()
        .map(|v| v as usize);
    // Explicit per-layer type array (Gemma 4: ["sliding_attention", "full_attention", ...])
    let layer_types = text_config.get("layer_types").and_then(|lt| {
        lt.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
    });
    // K=V sharing flag
    let attention_k_eq_v = text_config["attention_k_eq_v"].as_bool().unwrap_or(false);
    // KV sharing across layers
    let num_kv_shared_layers = text_config["num_kv_shared_layers"]
        .as_u64()
        .map(|v| v as usize)
        .filter(|&v| v > 0);

    // Per-layer embedding dimension (PLE)
    let per_layer_embed_dim = text_config["hidden_size_per_layer_input"]
        .as_u64()
        .map(|v| v as usize)
        .filter(|&v| v > 0);

    // The rest of the PLE family, verbatim (see `ModelConfig`).
    let use_double_wide_mlp = text_config["use_double_wide_mlp"].as_bool();
    let vocab_size_per_layer_input = text_config["vocab_size_per_layer_input"].as_u64();

    let has_vision_config = config.get("vision_config").is_some();

    // Attention/output scaling + norm shape. Declared per checkpoint;
    // families that don't declare them get `None` and their own defaults.
    let qk_scale_factor = text_config["qk_scale_factor"].as_f64();
    let output_multiplier = text_config["output_multiplier"].as_f64();
    let post_norm_eps = text_config["post_norm_eps"].as_f64();
    let attention_bias = text_config["attention_bias"].as_bool();
    let mlp_bias = text_config["mlp_bias"].as_bool();
    // Both HF spellings; verbatim — the Activation mapping (and its failure
    // on unrecognised names) lives on the architecture trait.
    let hidden_act = text_config["hidden_act"]
        .as_str()
        .or_else(|| text_config["hidden_activation"].as_str())
        .map(str::to_string);
    let max_position_embeddings = text_config["max_position_embeddings"]
        .as_u64()
        .map(|v| v as usize);

    // Multimodal protocol + adapter geometry — root-level HF fields.
    let image_token_id = config["image_token_id"].as_u64();
    let video_token_id = config["video_token_id"].as_u64();
    let out_hidden_size = config["out_hidden_size"].as_u64().map(|v| v as usize);
    let projector_hidden_size = config["projector_hidden_size"].as_u64().map(|v| v as usize);
    let projector_hidden_act = config["projector_hidden_act"].as_str().map(str::to_string);

    // Drafter interface declaration. `block_size` is read only alongside
    // `target_layer_ids`: the pair is one declaration (a DFlash-style
    // hidden-state consumer); a bare `block_size` elsewhere is a different
    // concept and stays unconsumed rather than misread.
    let target_layer_ids: Option<Vec<usize>> = text_config.get("target_layer_ids").and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .collect()
        })
    });
    let draft_block_size = target_layer_ids
        .as_ref()
        .and_then(|_| text_config["block_size"].as_u64().map(|v| v as usize));
    let mask_token_id = text_config["mask_token_id"].as_u64();

    // Hybrid linear-attention + multi-token-prediction (declared,
    // R2/Kimi-Linear-rung prep). Read verbatim — no semantics judged here.
    let linear_conv_kernel_dim = text_config["linear_conv_kernel_dim"]
        .as_u64()
        .map(|v| v as usize);
    let linear_key_head_dim = text_config["linear_key_head_dim"]
        .as_u64()
        .map(|v| v as usize);
    let linear_value_head_dim = text_config["linear_value_head_dim"]
        .as_u64()
        .map(|v| v as usize);
    let linear_num_key_heads = text_config["linear_num_key_heads"]
        .as_u64()
        .map(|v| v as usize);
    let linear_num_value_heads = text_config["linear_num_value_heads"]
        .as_u64()
        .map(|v| v as usize);
    // The same fact `layer_types` states, in the index-set spelling.
    // `text_config` falls back to the root config, so one read covers
    // GLM-5.3-Flash (nested) and Kimi Linear (flat) alike.
    // The declared interleave, in whichever spelling this checkpoint uses.
    // `text_config` falls back to the root config, so one read covers a
    // nested layout (GLM-5.3-Flash) and a flat one (Kimi Linear, Inkling)
    // alike. The window travels with it: a sliding kind whose size was
    // never declared must not acquire one downstream.
    // Inkling-Small spells the window `sliding_window_size`; the
    // families before it spell it `sliding_window`. Both are the same
    // fact, and reading only one leaves a sliding layer with no size —
    // which is the number a KV planner needs most.
    let declared_window = sliding_window.or_else(|| {
        text_config["sliding_window_size"]
            .as_u64()
            .map(|v| v as usize)
    });
    let linear_attn_interleave = crate::config::read_declared_interleave(
        text_config,
        crate::config::InterleaveScope::DecoderStack,
        num_layers,
        declared_window,
    );
    // The MTP sub-stack indexes its own layer space — Inkling-Small
    // declares `local_layer_ids` for both, and resolving the second
    // against the first's layer count would be wrong.
    let mtp_layers = text_config["mtp_config"]["num_nextn_predict_layers"]
        .as_u64()
        .or_else(|| config["mtp_config"]["num_nextn_predict_layers"].as_u64())
        .unwrap_or(0) as usize;
    let mtp_interleave = if mtp_layers > 0 {
        crate::config::read_declared_interleave(
            if text_config.get("mtp_config").is_some() {
                text_config
            } else {
                config
            },
            crate::config::InterleaveScope::MtpStack,
            mtp_layers,
            declared_window,
        )
    } else {
        crate::config::DeclaredInterleave::Absent
    };
    let kda_geometry = crate::config::KdaGeometry::read(&text_config["linear_attn_config"]);
    let kda_gate_lower_bound = text_config["linear_attn_config"]["gate_lower_bound"]
        .as_f64()
        .map(|v| v as f32);
    let d_rel = text_config["d_rel"].as_u64().map(|v| v as usize);
    let rel_extent = text_config["rel_extent"].as_u64().map(|v| v as usize);
    let mamba_ssm_dtype = text_config["mamba_ssm_dtype"].as_str().map(str::to_string);
    // The mamba_ssm lineage declares its MLP estate apart from
    // `intermediate_size`: `mlp_intermediate_size` is the gated MLP's
    // width and ZERO is a declaration — no MLP blocks exist anywhere in
    // the stack (OuteAI Mamba2Attn ships none). The padding multiple and
    // bias flag parameterise that same (possibly absent) MLP.
    // `d_intermediate` is mamba_ssm's own spelling of the same width;
    // OuteAI renamed it `mlp_intermediate_size`. Zero declares NO MLP
    // blocks in either spelling.
    let mlp_intermediate_size = text_config["mlp_intermediate_size"]
        .as_u64()
        .or_else(|| text_config["d_intermediate"].as_u64())
        .map(|v| v as usize);
    let mlp_padding_size = text_config["mlp_padding_size"].as_u64().map(|v| v as usize);
    let use_mlp_bias = text_config["use_mlp_bias"].as_bool();
    // mamba_ssm rounds the embedding rows up to a multiple; the declared
    // vocab and the tensor's row count differ by exactly this padding.
    let pad_vocab_size_multiple = text_config["pad_vocab_size_multiple"]
        .as_u64()
        .map(|v| v as usize);
    // Whether the reference runtime fuses residual-add with the norm —
    // a kernel-schedule fact of the same operation, carried verbatim.
    let fused_add_norm = text_config["fused_add_norm"].as_bool();
    let residual_in_fp32 = text_config["residual_in_fp32"].as_bool();
    let attn_output_gate = text_config["attn_output_gate"].as_bool();
    let output_gate_type = text_config["output_gate_type"].as_str().map(str::to_string);
    let mtp_num_hidden_layers = text_config["mtp_num_hidden_layers"]
        .as_u64()
        .map(|v| v as usize);
    let mtp_use_dedicated_embeddings = text_config["mtp_use_dedicated_embeddings"].as_bool();
    // mRoPE sectioning (Qwen-VL-style multi-axis position encoding),
    // declared under the same `rope_parameters` block the flat-form
    // rope_theta/rope_type/partial_rotary_factor already read.
    let mrope_interleaved = rope_params.and_then(|rp| rp["mrope_interleaved"].as_bool());
    let mrope_section = rope_params.and_then(|rp| {
        rp["mrope_section"].as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
    });

    ModelConfig {
        model_type,
        norm_eps,
        num_layers,
        hidden_size,
        intermediate_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        vocab_size,
        rope_base,
        rope_local_base,
        layer_rope_theta,
        sliding_window,
        num_experts,
        num_experts_per_token,
        num_shared_experts,
        kv_lora_rank,
        q_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_scaling,
        attn_logit_softcapping,
        final_logit_softcapping,
        query_pre_attn_scalar,
        embedding_multiplier,
        residual_multiplier,
        attention_multiplier,
        logits_scaling,
        global_head_dim,
        num_global_kv_heads,
        partial_rotary_factor,
        sliding_window_pattern,
        layer_types,
        attention_k_eq_v,
        per_layer_embed_dim,
        num_kv_shared_layers,
        enable_moe_block,
        top_k_experts,
        moe_intermediate_size,
        swiglu_limit,
        norm_topk_prob,
        router_activation,
        routed_scaling_factor,
        expert_groups,
        topk_group,
        use_grouped_topk,
        moe_layer_freq,
        first_k_dense_replace,
        mla_use_nope,
        model_max_length,
        has_vision_config,
        tie_word_embeddings,
        qk_scale_factor,
        output_multiplier,
        post_norm_eps,
        attention_bias,
        mlp_bias,
        hidden_act,
        max_position_embeddings,
        image_token_id,
        video_token_id,
        out_hidden_size,
        projector_hidden_size,
        projector_hidden_act,
        target_layer_ids,
        draft_block_size,
        mask_token_id,
        use_double_wide_mlp,
        vocab_size_per_layer_input,
        linear_conv_kernel_dim,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_attn_interleave,
        mtp_interleave,
        kda_geometry,
        kda_gate_lower_bound,
        d_rel,
        rel_extent,
        mamba_ssm_dtype,
        mamba2_geometry,
        mamba2_provenance,
        conv_qkv_attn,
        conv_qkv_provenance,
        attn_causal,
        pad_vocab_size_multiple,
        fused_add_norm,
        mlp_intermediate_size,
        mlp_padding_size,
        use_mlp_bias,
        residual_in_fp32,
        attn_output_gate,
        output_gate_type,
        mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings,
        mrope_interleaved,
        mrope_section,
    }
}
