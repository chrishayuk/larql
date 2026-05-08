//! Validation for parsed model architecture configs.

use crate::config::{ModelArchitecture, ModelConfig};

pub const FIELD_NUM_LAYERS: &str = "num_layers";
pub const FIELD_HIDDEN_SIZE: &str = "hidden_size";
pub const FIELD_INTERMEDIATE_SIZE: &str = "intermediate_size";
pub const FIELD_HEAD_DIM: &str = "head_dim";
pub const FIELD_NUM_Q_HEADS: &str = "num_q_heads";
pub const FIELD_NUM_KV_HEADS: &str = "num_kv_heads";
pub const FIELD_VOCAB_SIZE: &str = "vocab_size";
pub const FIELD_ROPE_BASE: &str = "rope_base";
pub const FIELD_ROPE_LOCAL_BASE: &str = "rope_local_base";
pub const FIELD_SLIDING_WINDOW: &str = "sliding_window";
pub const FIELD_NUM_EXPERTS: &str = "num_experts";
pub const FIELD_NUM_EXPERTS_PER_TOKEN: &str = "num_experts_per_token";
pub const FIELD_NUM_SHARED_EXPERTS: &str = "num_shared_experts";
pub const FIELD_TOP_K_EXPERTS: &str = "top_k_experts";
pub const FIELD_MOE_INTERMEDIATE_SIZE: &str = "moe_intermediate_size";
pub const FIELD_KV_LORA_RANK: &str = "kv_lora_rank";
pub const FIELD_Q_LORA_RANK: &str = "q_lora_rank";
pub const FIELD_ROPE_SCALING_TYPE: &str = "rope_scaling.type";
pub const FIELD_ROPE_SCALING_FACTOR: &str = "rope_scaling.factor";
pub const FIELD_ATTN_LOGIT_SOFTCAPPING: &str = "attn_logit_softcapping";
pub const FIELD_FINAL_LOGIT_SOFTCAPPING: &str = "final_logit_softcapping";
pub const FIELD_QUERY_PRE_ATTN_SCALAR: &str = "query_pre_attn_scalar";
pub const FIELD_EMBEDDING_MULTIPLIER: &str = "embedding_multiplier";
pub const FIELD_RESIDUAL_MULTIPLIER: &str = "residual_multiplier";
pub const FIELD_ATTENTION_MULTIPLIER: &str = "attention_multiplier";
pub const FIELD_LOGITS_SCALING: &str = "logits_scaling";
pub const FIELD_GLOBAL_HEAD_DIM: &str = "global_head_dim";
pub const FIELD_NUM_GLOBAL_KV_HEADS: &str = "num_global_kv_heads";
pub const FIELD_PARTIAL_ROTARY_FACTOR: &str = "partial_rotary_factor";
pub const FIELD_SLIDING_WINDOW_PATTERN: &str = "sliding_window_pattern";
pub const FIELD_LAYER_TYPES: &str = "layer_types";
pub const FIELD_PER_LAYER_EMBED_DIM: &str = "per_layer_embed_dim";
pub const FIELD_NUM_KV_SHARED_LAYERS: &str = "num_kv_shared_layers";
pub const FIELD_HEAD_DIM_FOR_LAYER: &str = "head_dim_for_layer";
pub const FIELD_NUM_Q_HEADS_FOR_LAYER: &str = "num_q_heads_for_layer";
pub const FIELD_NUM_KV_HEADS_FOR_LAYER: &str = "num_kv_heads_for_layer";
pub const FIELD_ROTARY_FRACTION_FOR_LAYER: &str = "rotary_fraction_for_layer";
pub const FIELD_ROPE_BASE_FOR_LAYER: &str = "rope_base_for_layer";

const MESSAGE_MUST_BE_POSITIVE: &str = "must be greater than 0";
const MESSAGE_MUST_BE_POSITIVE_FINITE: &str = "must be finite and greater than 0";
const MESSAGE_MUST_BE_FRACTION: &str = "must be finite and in the range (0, 1]";
const MESSAGE_MUST_NOT_BE_EMPTY: &str = "must not be empty";

/// One configuration invariant violation found by [`ModelArchitecture::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigValidationError {
    /// Stable field identifier, suitable for matching in tests or caller diagnostics.
    pub field: &'static str,
    /// Human-readable explanation of the invalid value.
    pub message: String,
}

impl ConfigValidationError {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

/// Result type returned by [`ModelArchitecture::validate`].
pub type ConfigValidationResult = Result<(), Vec<ConfigValidationError>>;

pub(crate) fn validate_architecture<A: ModelArchitecture + ?Sized>(
    arch: &A,
) -> ConfigValidationResult {
    let cfg = arch.config();
    let mut errors = Vec::new();

    validate_positive_usize(&mut errors, FIELD_NUM_LAYERS, cfg.num_layers);
    validate_positive_usize(&mut errors, FIELD_HIDDEN_SIZE, cfg.hidden_size);
    validate_positive_usize(&mut errors, FIELD_INTERMEDIATE_SIZE, cfg.intermediate_size);
    validate_positive_usize(&mut errors, FIELD_HEAD_DIM, cfg.head_dim);
    validate_positive_usize(&mut errors, FIELD_NUM_Q_HEADS, cfg.num_q_heads);
    validate_positive_usize(&mut errors, FIELD_NUM_KV_HEADS, cfg.num_kv_heads);
    validate_optional_positive_usize(&mut errors, FIELD_VOCAB_SIZE, cfg.vocab_size);
    validate_optional_positive_usize(&mut errors, FIELD_SLIDING_WINDOW, cfg.sliding_window);
    validate_optional_positive_usize(&mut errors, FIELD_GLOBAL_HEAD_DIM, cfg.global_head_dim);
    validate_optional_positive_usize(
        &mut errors,
        FIELD_NUM_GLOBAL_KV_HEADS,
        cfg.num_global_kv_heads,
    );
    validate_optional_positive_usize(
        &mut errors,
        FIELD_PER_LAYER_EMBED_DIM,
        cfg.per_layer_embed_dim,
    );
    validate_optional_positive_usize(
        &mut errors,
        FIELD_NUM_KV_SHARED_LAYERS,
        cfg.num_kv_shared_layers,
    );
    validate_optional_positive_usize(&mut errors, FIELD_KV_LORA_RANK, cfg.kv_lora_rank);
    validate_optional_positive_usize(&mut errors, FIELD_Q_LORA_RANK, cfg.q_lora_rank);

    validate_positive_f64(&mut errors, FIELD_ROPE_BASE, cfg.rope_base);
    validate_optional_positive_f64(&mut errors, FIELD_ROPE_LOCAL_BASE, cfg.rope_local_base);
    validate_optional_positive_f64(
        &mut errors,
        FIELD_QUERY_PRE_ATTN_SCALAR,
        cfg.query_pre_attn_scalar,
    );
    validate_optional_positive_f64(
        &mut errors,
        FIELD_ATTN_LOGIT_SOFTCAPPING,
        cfg.attn_logit_softcapping,
    );
    validate_optional_positive_f64(
        &mut errors,
        FIELD_FINAL_LOGIT_SOFTCAPPING,
        cfg.final_logit_softcapping,
    );
    validate_optional_positive_f64(
        &mut errors,
        FIELD_EMBEDDING_MULTIPLIER,
        cfg.embedding_multiplier,
    );
    validate_optional_positive_f64(
        &mut errors,
        FIELD_RESIDUAL_MULTIPLIER,
        cfg.residual_multiplier,
    );
    validate_optional_positive_f64(
        &mut errors,
        FIELD_ATTENTION_MULTIPLIER,
        cfg.attention_multiplier,
    );
    validate_optional_positive_f64(&mut errors, FIELD_LOGITS_SCALING, cfg.logits_scaling);

    validate_hidden_head_dim(cfg, &mut errors);
    validate_attention_heads(
        &mut errors,
        FIELD_NUM_Q_HEADS,
        cfg.num_q_heads,
        cfg.num_kv_heads,
    );

    if let Some(num_global_kv_heads) = cfg.num_global_kv_heads {
        validate_attention_heads(
            &mut errors,
            FIELD_NUM_GLOBAL_KV_HEADS,
            cfg.num_q_heads,
            num_global_kv_heads,
        );
    }

    if let Some(pattern) = cfg.sliding_window_pattern {
        validate_positive_usize(&mut errors, FIELD_SLIDING_WINDOW_PATTERN, pattern);
    }

    if let Some(partial_rotary_factor) = cfg.partial_rotary_factor {
        validate_fraction(
            &mut errors,
            FIELD_PARTIAL_ROTARY_FACTOR,
            partial_rotary_factor,
        );
    }

    validate_rope_scaling(cfg, &mut errors);
    validate_layer_metadata(cfg, &mut errors);
    validate_moe_config(arch, cfg, &mut errors);
    validate_per_layer_overrides(arch, cfg, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn validate_positive_usize(
    errors: &mut Vec<ConfigValidationError>,
    field: &'static str,
    value: usize,
) {
    if value == 0 {
        errors.push(ConfigValidationError::new(field, MESSAGE_MUST_BE_POSITIVE));
    }
}

fn validate_optional_positive_usize(
    errors: &mut Vec<ConfigValidationError>,
    field: &'static str,
    value: Option<usize>,
) {
    if let Some(value) = value {
        validate_positive_usize(errors, field, value);
    }
}

fn validate_positive_f64(errors: &mut Vec<ConfigValidationError>, field: &'static str, value: f64) {
    if !value.is_finite() || value <= 0.0 {
        errors.push(ConfigValidationError::new(
            field,
            MESSAGE_MUST_BE_POSITIVE_FINITE,
        ));
    }
}

fn validate_optional_positive_f64(
    errors: &mut Vec<ConfigValidationError>,
    field: &'static str,
    value: Option<f64>,
) {
    if let Some(value) = value {
        validate_positive_f64(errors, field, value);
    }
}

fn validate_fraction(errors: &mut Vec<ConfigValidationError>, field: &'static str, value: f64) {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        errors.push(ConfigValidationError::new(field, MESSAGE_MUST_BE_FRACTION));
    }
}

fn validate_hidden_head_dim(_cfg: &ModelConfig, _errors: &mut Vec<ConfigValidationError>) {
    // Intentionally no-op. Gemma-4 (and any model with separate inner attention
    // dim) decouples head_dim from hidden_size: Q/K/V projections expand to
    // num_heads * head_dim, attention runs at that dim, and the output
    // projection re-projects back to hidden_size. So `hidden_size % head_dim
    // == 0` is NOT a real invariant of multi-head attention, only an
    // accidental property of older designs (Gemma-3, Llama, etc.) where
    // num_heads * head_dim happened to equal hidden_size. Enforcing it here
    // blocked extraction of e.g. google/gemma-4-26B-A4B-it (hidden 2816,
    // sliding head_dim 256, global head_dim 512 — neither divides 2816).
    // Per-layer geometry sanity (head_dim > 0, num_heads > 0, kv|q heads)
    // is still validated in `validate_layer`.
}

fn validate_attention_heads(
    errors: &mut Vec<ConfigValidationError>,
    field: &'static str,
    num_q_heads: usize,
    num_kv_heads: usize,
) {
    if num_q_heads == 0 || num_kv_heads == 0 {
        return;
    }

    if num_kv_heads > num_q_heads {
        errors.push(ConfigValidationError::new(
            field,
            format!("num_kv_heads ({num_kv_heads}) must not exceed num_q_heads ({num_q_heads})"),
        ));
        return;
    }

    if !num_q_heads.is_multiple_of(num_kv_heads) {
        errors.push(ConfigValidationError::new(
            field,
            format!(
                "num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            ),
        ));
    }
}

fn validate_rope_scaling(cfg: &ModelConfig, errors: &mut Vec<ConfigValidationError>) {
    if let Some(rope_scaling) = &cfg.rope_scaling {
        if rope_scaling.scaling_type.trim().is_empty() {
            errors.push(ConfigValidationError::new(
                FIELD_ROPE_SCALING_TYPE,
                MESSAGE_MUST_NOT_BE_EMPTY,
            ));
        }
        validate_positive_f64(errors, FIELD_ROPE_SCALING_FACTOR, rope_scaling.factor);
    }
}

fn validate_layer_metadata(cfg: &ModelConfig, errors: &mut Vec<ConfigValidationError>) {
    if let Some(layer_types) = &cfg.layer_types {
        if layer_types.len() != cfg.num_layers {
            errors.push(ConfigValidationError::new(
                FIELD_LAYER_TYPES,
                format!(
                    "contains {} entries but num_layers is {}",
                    layer_types.len(),
                    cfg.num_layers
                ),
            ));
        }
        if let Some(index) = layer_types
            .iter()
            .position(|layer_type| layer_type.is_empty())
        {
            errors.push(ConfigValidationError::new(
                FIELD_LAYER_TYPES,
                format!("entry {index} must not be empty"),
            ));
        }
    }

    if let Some(num_shared) = cfg.num_kv_shared_layers {
        if cfg.num_layers > 0 && num_shared >= cfg.num_layers {
            errors.push(ConfigValidationError::new(
                FIELD_NUM_KV_SHARED_LAYERS,
                format!(
                    "must be less than num_layers ({}) but was {}",
                    cfg.num_layers, num_shared
                ),
            ));
        }
    }
}

fn validate_moe_config<A: ModelArchitecture + ?Sized>(
    arch: &A,
    cfg: &ModelConfig,
    errors: &mut Vec<ConfigValidationError>,
) {
    validate_optional_positive_usize(errors, FIELD_NUM_EXPERTS, cfg.num_experts);
    validate_optional_positive_usize(
        errors,
        FIELD_NUM_EXPERTS_PER_TOKEN,
        cfg.num_experts_per_token,
    );
    validate_optional_positive_usize(errors, FIELD_NUM_SHARED_EXPERTS, cfg.num_shared_experts);
    validate_optional_positive_usize(errors, FIELD_TOP_K_EXPERTS, cfg.top_k_experts);
    validate_optional_positive_usize(
        errors,
        FIELD_MOE_INTERMEDIATE_SIZE,
        cfg.moe_intermediate_size,
    );

    if cfg.num_experts.unwrap_or(0) > 0
        && cfg.num_experts_per_token.is_none()
        && cfg.top_k_experts.is_none()
    {
        errors.push(ConfigValidationError::new(
            FIELD_NUM_EXPERTS_PER_TOKEN,
            "must be set when num_experts is set",
        ));
    }

    if arch.is_moe() || arch.is_hybrid_moe() {
        let num_experts = arch.num_experts();
        let num_experts_per_token = arch.num_experts_per_token();

        validate_positive_usize(errors, FIELD_NUM_EXPERTS, num_experts);
        validate_positive_usize(errors, FIELD_NUM_EXPERTS_PER_TOKEN, num_experts_per_token);

        if num_experts > 0 && num_experts_per_token > num_experts {
            errors.push(ConfigValidationError::new(
                FIELD_NUM_EXPERTS_PER_TOKEN,
                format!(
                    "experts per token ({num_experts_per_token}) must not exceed num_experts ({num_experts})"
                ),
            ));
        }
    }

    if arch.is_hybrid_moe() {
        validate_positive_usize(
            errors,
            FIELD_MOE_INTERMEDIATE_SIZE,
            arch.moe_intermediate_size(),
        );
    }
}

fn validate_per_layer_overrides<A: ModelArchitecture + ?Sized>(
    arch: &A,
    cfg: &ModelConfig,
    errors: &mut Vec<ConfigValidationError>,
) {
    if cfg.num_layers == 0 {
        return;
    }

    for layer in 0..cfg.num_layers {
        if !validate_one_layer(arch, cfg, layer, errors) {
            break;
        }
    }
}

fn validate_one_layer<A: ModelArchitecture + ?Sized>(
    arch: &A,
    _cfg: &ModelConfig,
    layer: usize,
    errors: &mut Vec<ConfigValidationError>,
) -> bool {
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q_heads = arch.num_q_heads_for_layer(layer);
    let num_kv_heads = arch.num_kv_heads_for_layer(layer);
    let rotary_fraction = arch.rotary_fraction_for_layer(layer);
    let rope_base = arch.rope_base_for_layer(layer);

    if head_dim == 0 {
        errors.push(ConfigValidationError::new(
            FIELD_HEAD_DIM_FOR_LAYER,
            format!("layer {layer} returned 0"),
        ));
        return false;
    }
    // Note: we intentionally do NOT enforce `hidden_size % head_dim == 0`
    // here — see `validate_hidden_head_dim` above for why. Gemma-4
    // dual-head_dim layers (sliding 256 / global 512) coexist with
    // hidden_size 2816 by design.
    if num_q_heads == 0 {
        errors.push(ConfigValidationError::new(
            FIELD_NUM_Q_HEADS_FOR_LAYER,
            format!("layer {layer} returned 0"),
        ));
        return false;
    }
    if num_kv_heads == 0 {
        errors.push(ConfigValidationError::new(
            FIELD_NUM_KV_HEADS_FOR_LAYER,
            format!("layer {layer} returned 0"),
        ));
        return false;
    }
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        errors.push(ConfigValidationError::new(
            FIELD_NUM_KV_HEADS_FOR_LAYER,
            format!(
                "layer {layer} num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            ),
        ));
        return false;
    }
    if !rotary_fraction.is_finite() || rotary_fraction <= 0.0 || rotary_fraction > 1.0 {
        errors.push(ConfigValidationError::new(
            FIELD_ROTARY_FRACTION_FOR_LAYER,
            format!("layer {layer} returned {rotary_fraction}, expected (0, 1]"),
        ));
        return false;
    }
    if !rope_base.is_finite() || rope_base <= 0.0 {
        errors.push(ConfigValidationError::new(
            FIELD_ROPE_BASE_FOR_LAYER,
            format!("layer {layer} returned {rope_base}, expected > 0"),
        ));
        return false;
    }

    true
}
