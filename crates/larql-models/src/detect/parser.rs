//! Parse a `config.json` JSON value into [`ModelConfig`].
//!
//! Handles both top-level and nested `text_config` (multimodal) layouts.
//! Optional fields fall through to the family's own class defaults
//! ([`super::registry::ConfigDefaults`]); required topology fields (see
//! [`super::config_io`]) are validated by the caller before this runs.

use crate::config::ModelConfig;

use super::config_io::{
    CONFIG_KEY_FFN_INTERMEDIATE_SIZE_BY_LAYER, CONFIG_KEY_HIDDEN_SIZE_ALIASES,
    CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES, CONFIG_KEY_NUM_ATTENTION_HEADS_ALIASES,
    CONFIG_KEY_NUM_HIDDEN_LAYERS_ALIASES, CONFIG_KEY_TEXT_CONFIG,
};

// ── Omitted-field defaults ─────────────────────────────────────────────────
// What an omitted field means is a per-family fact (each family's
// `transformers` config class), so it comes from the family's registry row:
// see `registry::ConfigDefaults`. They surface mostly from the in-memory
// `detect_from_json` path; the disk path enforces presence of topology
// fields in `config_io::require_config_fields`.
use super::registry::{find_architecture, ConfigDefaults, IntermediateSize};

mod attention_layers;
mod keys;
mod rope_scaling;

use keys::*;
use rope_scaling::parse_rope_scaling;

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

    // What an omitted field means for this family.
    let defaults =
        find_architecture(&model_type).map_or(ConfigDefaults::STANDARD, |e| e.config_defaults);

    // Required topology fields. On the disk path `detect_architecture`
    // already errored when any of these are absent, so a zero here only
    // surfaces from `detect_from_json` callers who pass partial JSON
    // (test ergonomics); the validator catches the zero downstream
    // rather than letting a magic-number default impersonate a real
    // topology and panic deep inside extract.
    let family_field =
        |key: Option<&str>| key.map_or(0, |k| topology_field(config, text_config, &[k]));
    let num_layers = match topology_field(config, text_config, CONFIG_KEY_NUM_HIDDEN_LAYERS_ALIASES)
    {
        0 => family_field(defaults.layer_count_key),
        n => n,
    };
    let hidden_size = topology_field(config, text_config, CONFIG_KEY_HIDDEN_SIZE_ALIASES);
    let mut intermediate_size =
        topology_field(config, text_config, CONFIG_KEY_INTERMEDIATE_SIZE_ALIASES);
    // A family may derive an undeclared FFN width from the hidden size
    // (GPT-2 ships no `n_inner`; transformers computes `4 * n_embd`).
    if let IntermediateSize::HiddenMultiple(multiple) = defaults.intermediate_size {
        if intermediate_size == 0 {
            intermediate_size = multiple * hidden_size;
        }
    }
    // A derived static-shard checkpoint declares each layer's dense-FFN
    // width; kept verbatim (the planner validates length and range, and
    // refuses there rather than here, so the reason lands beside the
    // tensors it disagrees with). Absent = uniform, bit-identical to a
    // checkpoint that never carried the key.
    let ffn_intermediate_size_by_layer = text_config
        .get(CONFIG_KEY_FFN_INTERMEDIATE_SIZE_BY_LAYER)
        .or_else(|| config.get(CONFIG_KEY_FFN_INTERMEDIATE_SIZE_BY_LAYER))
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|k| k as usize))
                .collect::<Vec<_>>()
        });
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
    // Some configs omit num_attention_heads, head_dim, or
    // num_key_value_heads, relying on their transformers class defaults;
    // the family's `ConfigDefaults` supplies those.
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
    let num_q_heads = super::config_io::read_aliased_u64(
        config,
        text_config,
        CONFIG_KEY_NUM_ATTENTION_HEADS_ALIASES,
    )
    .or_else(|| {
        defaults
            .head_count_key
            .and_then(|k| super::config_io::read_aliased_u64(config, text_config, &[k]))
    })
    .map(|v| v as usize)
    .unwrap_or(if attention_free_ssm {
        0
    } else {
        defaults.num_attention_heads.unwrap_or(0)
    });
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
        .unwrap_or_else(|| {
            defaults
                .head_dim
                .unwrap_or_else(|| hidden_size.checked_div(num_q_heads).unwrap_or(0))
        });
    let num_kv_heads = text_config["num_key_value_heads"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(if attention_free_ssm {
            0
        } else {
            defaults.num_key_value_heads.unwrap_or(num_q_heads)
        });
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
    // Whether ANY rotary key is declared, before the family default fills
    // `rope_base`. A learned-absolute family must not be graded rotary off
    // a default nobody declared.
    let declares_rotary = rope_params.is_some_and(|v| !v.is_null())
        || ["rope_theta", "layer_rope_theta", "partial_rotary_factor"]
            .iter()
            .any(|k| text_config.get(*k).is_some_and(|v| !v.is_null()))
        || text_config
            .get("rope_scaling")
            .is_some_and(serde_json::Value::is_object);
    let rope_base = rope_params
        .and_then(|rp| rp.get("full_attention"))
        .and_then(|fa| fa["rope_theta"].as_f64())
        .or_else(|| rope_params.and_then(|rp| rp["rope_theta"].as_f64()))
        .or_else(|| text_config["rope_theta"].as_f64())
        .unwrap_or(defaults.rope_theta);
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
    // GPT-Neo spells the trailing window `window_size`.
    let sliding_window = text_config["sliding_window"]
        .as_u64()
        .or_else(|| text_config["window_size"].as_u64())
        .map(|v| v as usize);
    // The window's ENABLE flag and its layer bound, read here so the
    // effective policy can be resolved in one place. A checkpoint that
    // declares a window and then disables it means what it says: Qwen2.5
    // ships `sliding_window: 32768` with `use_sliding_window: false`.
    let use_sliding_window = text_config["use_sliding_window"].as_bool();
    let max_window_layers = text_config["max_window_layers"]
        .as_u64()
        .map(|v| v as usize);
    // The declared positional scheme, verbatim. Read here rather than
    // interpreted here: `granitemoehybrid` uses it to *enable* rotation
    // at all, while the BERT lineage uses the same leaf for
    // `absolute` / `relative_key`. Only the family knows which vocabulary
    // it is speaking, so the string travels and the architecture decides.
    let position_embedding_type = text_config["position_embedding_type"]
        .as_str()
        .map(str::to_string);
    // The per-layer rotary schedule, verbatim in the checkpoint's own
    // polarity — `1` rotates, `0` is NoPE, despite the key's name. The
    // inversion is honoured once, in `PositionPolicy::rope_enabled_by_flag`.
    let no_rope_layers = text_config["no_rope_layers"].as_array().map(|mask| {
        mask.iter()
            .map(|v| v.as_i64().unwrap_or_default())
            .collect::<Vec<i64>>()
    });
    let no_rope_layer_interval = text_config["no_rope_layer_interval"]
        .as_u64()
        .map(|v| v as usize);
    // Two declarations no reference implementation reads. Read here so
    // that agreement is CHECKED rather than assumed — an unread flag that
    // happens to match is one value away from a silent wrong answer.
    let rope_interleaved = text_config["rope_interleaved"].as_bool();
    let use_mrope = text_config["use_mrope"].as_bool();
    // Two more declarations no reference reads, stored so they are
    // checked (see `ModelConfig::ffn_shape_name` / `is_llama_config`).
    let ffn_shape_name = text_config["activation"].as_str().map(str::to_string);
    let is_llama_config = text_config["is_llama_config"].as_bool();
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
    let shared_expert_intermediate_size =
        field_u64(text_config, SHARED_EXPERT_INTERMEDIATE_SIZE_KEYS).map(|v| v as usize);
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
    // Kimi-K3's latent routed branch. PRESENCE selects the form, so this
    // stays an Option all the way through: `0` is a declared (degenerate)
    // width that closure refuses by name, and `null` is the same answer
    // as absent, matching the reference's `is not None`.
    let routed_expert_hidden_size = text_config["routed_expert_hidden_size"]
        .as_u64()
        .map(|v| v as usize);
    // Truthiness, not presence — the reference reads this with a
    // `getattr(..., False)` consumed by a plain `if`. Kept as an Option
    // so the plan can still report whether the checkpoint said anything.
    let latent_moe_use_norm = text_config["latent_moe_use_norm"].as_bool();
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

    // DSA (DeepSeek Sparse Attention) indexer fields — GLM-5.2. Config
    // facts only; nothing downstream reads these yet.
    let index_topk = text_config["index_topk"].as_u64().map(|v| v as usize);
    let index_n_heads = text_config["index_n_heads"].as_u64().map(|v| v as usize);
    let index_head_dim = text_config["index_head_dim"].as_u64().map(|v| v as usize);

    // RoPE scaling: see `rope_scaling::parse_rope_scaling` for the shapes
    // and the two homes (`rope_parameters`, legacy `rope_scaling`).
    let rope_scaling = rope_params
        .and_then(parse_rope_scaling)
        .or_else(|| text_config.get("rope_scaling").and_then(parse_rope_scaling));

    // RMS-norm / LayerNorm epsilon. Field-name aliases across families:
    //  - `rms_norm_eps`           — Llama, Mistral, Gemma
    //  - `layer_norm_eps`         — BERT-family
    //  - `layer_norm_epsilon`     — GPT-2
    //  - `norm_epsilon`           — StarCoder2
    //  - `norm_eps`               — LFM2 (which also ships a separate
    //                               `block_norm_eps` for its FFN blocks;
    //                               only the stack epsilon is this fact)
    // Most modern archs ship 1e-5; older ones used 1e-6. None → arch default.
    let norm_eps = text_config["rms_norm_eps"]
        .as_f64()
        .or_else(|| text_config["layer_norm_eps"].as_f64())
        .or_else(|| text_config["layer_norm_epsilon"].as_f64())
        .or_else(|| text_config["norm_epsilon"].as_f64())
        .or_else(|| text_config["norm_eps"].as_f64());

    // Hyper-connections (DeepSeek-V4). Read as declared or not at all —
    // a defaulted stream count would silently make a four-stream model a
    // one-stream one.
    let hc_streams = text_config["hc_mult"].as_u64().map(|v| v as usize);
    let hc_sinkhorn_iters = text_config["hc_sinkhorn_iters"]
        .as_u64()
        .map(|v| v as usize);
    let hc_eps = text_config["hc_eps"].as_f64();

    // Attention residuals (Kimi-K3). Same rule as the stream count above
    // and the same reason: a defaulted block size would snapshot at
    // different layers than the checkpoint declares, which computes a
    // different model rather than failing.
    let attn_res_block_size = text_config["attn_res_block_size"]
        .as_u64()
        .map(|v| v as usize);

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
    // Partial rotary factor, in the reference's precedence
    // (`PreTrainedConfig::standardize_rope_params`, transformers 5.x):
    //  1. partial_rotary_factor at the top level — the legacy flat form.
    //     The reference copies it INTO `rope_parameters`, overwriting
    //     whatever the block declared, so when both are present this one
    //     is the fraction the model runs with;
    //  2. rope_parameters.full_attention.partial_rotary_factor — Gemma 4's
    //     per-layer-type form;
    //  3. rope_parameters.partial_rotary_factor — the 5.x flat form, and
    //     the ONLY spelling every Qwen3.5 checkpoint uses.
    //
    // Form 3 was not read until wave 8 of the conformance sweep. Qwen3.8
    // writes forms 1 and 3 together, so it resolved; Qwen3.5 writes form 3
    // alone, lost the fraction, and would have rotated all 256 head dims
    // where the checkpoint asks for 64 — the same fallthrough shape as
    // `rope_theta` form 2 above, on the next key. VINDEX3 refused those
    // checkpoints outright; the engine path did not.
    let partial_rotary_factor = text_config["partial_rotary_factor"]
        .as_f64()
        .or_else(|| {
            rope_params
                .and_then(|rp| rp.get("full_attention"))
                .and_then(|fa| fa["partial_rotary_factor"].as_f64())
        })
        .or_else(|| rope_params.and_then(|rp| rp["partial_rotary_factor"].as_f64()));
    // Sliding window pattern: explicit sliding_window_pattern field, or infer later.
    let sliding_window_pattern = text_config["sliding_window_pattern"]
        .as_u64()
        .map(|v| v as usize);
    // Explicit per-layer type array (Gemma 4: ["sliding_attention", "full_attention", ...])
    let layer_types = text_config
        .get("layer_types")
        .and_then(|lt| {
            lt.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
        })
        // GPT-Neo's spelling of the same per-layer fact, in its own
        // vocabulary (`global` / `local`). `attention_types` is the
        // authority — `GPTNeoConfig` derives `attention_layers` from it —
        // so it is read first and `attention_layers` only when it is
        // absent; the planner holds the two to each other.
        .or_else(|| attention_layers::gpt_neo_layer_types(text_config));
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
    let qkv_bias = text_config["qkv_bias"].as_bool();
    let scale_attn_weights = text_config["scale_attn_weights"].as_bool();
    let scale_attn_by_inverse_layer_idx = text_config["scale_attn_by_inverse_layer_idx"].as_bool();
    let reorder_and_upcast_attn = text_config["reorder_and_upcast_attn"].as_bool();
    let add_cross_attention = text_config["add_cross_attention"].as_bool();
    let mlp_bias = text_config["mlp_bias"].as_bool();
    // Both HF spellings; verbatim — the Activation mapping (and its failure
    // on unrecognised names) lives on the architecture trait.
    // GPT-2's lineage spells it `activation_function` (`gelu_new`).
    let hidden_act = text_config["hidden_act"]
        .as_str()
        .or_else(|| text_config["hidden_activation"].as_str())
        .or_else(|| text_config["activation_function"].as_str())
        .map(str::to_string);
    // SiTU-GLU's two softcaps. Read verbatim beside the name that selects
    // them; whether they are the parameters of an activation this
    // checkpoint actually uses is judged on the architecture, not here.
    let activation_situ_beta = text_config["activation_situ_beta"].as_f64();
    let activation_situ_linear_beta = text_config["activation_situ_linear_beta"].as_f64();
    // GPT-2's lineage spells the serving bound `n_positions` (the `wpe`
    // table's row count).
    let max_position_embeddings = text_config["max_position_embeddings"]
        .as_u64()
        .or_else(|| text_config["n_positions"].as_u64())
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
    // `safe_gate` is read so the gate-form rule is CHECKED rather than
    // assumed: GLM-5.3-Flash declares no `safe_gate` and its reference
    // defaults it to `True`, so the absent case still clamps — but a
    // checkpoint that says `false` must reach the softplus branch, and
    // that can only happen if the key is carried.
    let kda_safe_gate = text_config["linear_attn_config"]["safe_gate"].as_bool();
    let kda_use_full_rank_gate = text_config["linear_attn_config"]["use_full_rank_gate"].as_bool();
    let mla_use_output_gate = text_config["mla_use_output_gate"].as_bool();
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
        ffn_intermediate_size_by_layer,
        head_dim,
        num_q_heads,
        num_kv_heads,
        vocab_size,
        rope_base,
        rope_local_base,
        layer_rope_theta,
        sliding_window,
        use_sliding_window,
        max_window_layers,
        position_embedding_type,
        no_rope_layers,
        no_rope_layer_interval,
        rope_interleaved,
        use_mrope,
        ffn_shape_name,
        is_llama_config,
        num_experts,
        num_experts_per_token,
        num_shared_experts,
        shared_expert_intermediate_size,
        kv_lora_rank,
        q_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        index_topk,
        index_n_heads,
        index_head_dim,
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
        routed_expert_hidden_size,
        latent_moe_use_norm,
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
        qkv_bias,
        declares_rotary,
        scale_attn_weights,
        scale_attn_by_inverse_layer_idx,
        reorder_and_upcast_attn,
        add_cross_attention,
        mlp_bias,
        hidden_act,
        activation_situ_beta,
        activation_situ_linear_beta,
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
        kda_safe_gate,
        kda_use_full_rank_gate,
        mla_use_output_gate,
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
        hc_streams,
        hc_sinkhorn_iters,
        hc_eps,
        attn_res_block_size,
        attn_output_gate,
        output_gate_type,
        mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings,
        mrope_interleaved,
        mrope_section,
    }
}
