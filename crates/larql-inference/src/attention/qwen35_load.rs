//! GGUF → `Qwen35Weights` bridge (Phase C.4d-bridge of
//! `inference-qwen35-deltanet`).
//!
//! Translates a loaded `ModelWeights` (HashMap of normalized tensor
//! keys → `ArcArray2<f32>` / `Vec<f32>`) into the typed
//! `Qwen35Weights` struct expected by `qwen35_forward_step`.
//!
//! Layer dispatch goes through `arch.is_linear_attention_layer(layer)`
//! so the same loader handles both Qwen 3.6 27B (dense) and
//! 35B-A3B (MoE) once an MoE-aware forward arrives. For now this
//! handles only the dense FFN suffix (Phase D will add the MoE
//! routed-expert path).
//!
//! All tensor look-ups are O(1) HashMap probes; the function does
//! `Arc` bumps and one `Arc::from(Vec<f32>)` per 1-D tensor (which
//! moves the Vec without copying its elements). No data copies on
//! the 2-D weights — `ArcArray2::clone()` only bumps the refcount.

use std::sync::Arc;

use larql_models::{ModelArchitecture, ModelWeights};

use super::deltanet_block::DeltaNetLayerWeights;
use super::qwen35_block::{Qwen35AttentionLayerWeights, Qwen35LayerWeights};
use super::qwen35_forward::{Qwen35FullLayerWeights, Qwen35Weights};

/// Bridge errors — every variant names the missing tensor key so
/// the user can grep the GGUF for "did this load?" diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum Qwen35LoadError {
    #[error("missing 2-D tensor: {0}")]
    MissingTensor(String),
    #[error("missing 1-D tensor (in `vectors` map): {0}")]
    MissingVector(String),
    #[error(
        "architecture method returned None for required key on layer {layer}: {method}; \
         is this layer really {kind}?"
    )]
    NoKey {
        layer: usize,
        method: &'static str,
        kind: &'static str,
    },
    #[error(
        "architecture is not hybrid Qwen 3.6 — {field} is 0. Did you pass a \
         Qwen35Arch or Qwen35MoeArch?"
    )]
    NotHybrid { field: &'static str },
}

/// Build a typed `Qwen35Weights` from a loaded `ModelWeights`.
///
/// Walks 0..num_layers, dispatching each to either the DeltaNet
/// branch (linear attention) or the full-attention branch via
/// `arch.is_linear_attention_layer(layer)`. Each branch pulls the
/// per-layer tensors using the architecture's key methods, falling
/// back to the trait defaults for the FFN trio + norms that follow
/// the standard `layers.N.<x>.weight` convention.
///
/// Returns `Qwen35Weights` with shared Arc-owned tensors that point
/// into the original `ModelWeights` storage (no copies).
pub fn load_qwen35_weights(
    weights: &ModelWeights,
    arch: &dyn ModelArchitecture,
) -> Result<Qwen35Weights, Qwen35LoadError> {
    if arch.full_attention_interval() == 0 {
        return Err(Qwen35LoadError::NotHybrid {
            field: "full_attention_interval",
        });
    }
    if arch.ssm_state_size() == 0 {
        return Err(Qwen35LoadError::NotHybrid {
            field: "ssm_state_size",
        });
    }

    let n_layers = weights.num_layers;
    let mut layers: Vec<Qwen35FullLayerWeights> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let block = if arch.is_linear_attention_layer(layer) {
            Qwen35LayerWeights::Linear(load_deltanet_layer(weights, arch, layer)?)
        } else {
            Qwen35LayerWeights::Attention(load_attention_layer(weights, arch, layer)?)
        };
        let attn_post_norm = get_vec(weights, &arch.post_attention_layernorm_key(layer))?;
        let ffn_gate = get_tensor(weights, &arch.ffn_gate_key(layer))?;
        let ffn_up = get_tensor(weights, &arch.ffn_up_key(layer))?;
        let ffn_down = get_tensor(weights, &arch.ffn_down_key(layer))?;
        layers.push(Qwen35FullLayerWeights {
            block,
            attn_post_norm,
            ffn_gate,
            ffn_up,
            ffn_down,
        });
    }

    let final_norm = get_vec(weights, arch.final_norm_key())?;

    Ok(Qwen35Weights {
        embed: weights.embed.clone(),
        layers,
        final_norm,
        lm_head: weights.lm_head.clone(),
        ffn_dim: arch.config().intermediate_size,
    })
}

fn load_deltanet_layer(
    weights: &ModelWeights,
    arch: &dyn ModelArchitecture,
    layer: usize,
) -> Result<DeltaNetLayerWeights, Qwen35LoadError> {
    let attn_norm = get_vec(weights, &arch.input_layernorm_key(layer))?;
    let attn_qkv_key = require_key(arch.attn_qkv_key(layer), layer, "attn_qkv_key", "linear")?;
    let attn_qkv = get_tensor(weights, &attn_qkv_key)?;
    let attn_gate_key = require_key(arch.attn_gate_key(layer), layer, "attn_gate_key", "linear")?;
    let attn_gate = get_tensor(weights, &attn_gate_key)?;
    let ssm_conv1d_key = require_key(
        arch.ssm_conv1d_key(layer),
        layer,
        "ssm_conv1d_key",
        "linear",
    )?;
    let ssm_conv1d = get_tensor(weights, &ssm_conv1d_key)?;
    let ssm_dt_key = require_key(arch.ssm_dt_key(layer), layer, "ssm_dt_key", "linear")?;
    let ssm_dt = get_vec(weights, &ssm_dt_key)?;
    let ssm_a_key = require_key(arch.ssm_a_key(layer), layer, "ssm_a_key", "linear")?;
    let ssm_a = get_vec(weights, &ssm_a_key)?;
    let ssm_beta_key = require_key(arch.ssm_beta_key(layer), layer, "ssm_beta_key", "linear")?;
    let ssm_beta = get_tensor(weights, &ssm_beta_key)?;
    let ssm_alpha_key = require_key(arch.ssm_alpha_key(layer), layer, "ssm_alpha_key", "linear")?;
    let ssm_alpha = get_tensor(weights, &ssm_alpha_key)?;
    let ssm_norm_key = require_key(arch.ssm_norm_key(layer), layer, "ssm_norm_key", "linear")?;
    let ssm_norm = get_vec(weights, &ssm_norm_key)?;
    let ssm_out_key = require_key(arch.ssm_out_key(layer), layer, "ssm_out_key", "linear")?;
    let ssm_out = get_tensor(weights, &ssm_out_key)?;

    Ok(DeltaNetLayerWeights {
        attn_norm,
        attn_qkv,
        attn_gate,
        ssm_conv1d,
        ssm_dt,
        ssm_a,
        ssm_beta,
        ssm_alpha,
        ssm_norm,
        ssm_out,
    })
}

fn load_attention_layer(
    weights: &ModelWeights,
    arch: &dyn ModelArchitecture,
    layer: usize,
) -> Result<Qwen35AttentionLayerWeights, Qwen35LoadError> {
    let attn_norm = get_vec(weights, &arch.input_layernorm_key(layer))?;
    let attn_q = get_tensor(weights, &arch.attn_q_key(layer))?;
    let attn_k = get_tensor(weights, &arch.attn_k_key(layer))?;
    let attn_v = get_tensor(weights, &arch.attn_v_key(layer))?;
    let attn_output = get_tensor(weights, &arch.attn_o_key(layer))?;
    let q_norm_key = require_key(
        arch.attn_q_per_head_norm_key(layer),
        layer,
        "attn_q_per_head_norm_key",
        "full-attn",
    )?;
    let attn_q_norm = get_vec(weights, &q_norm_key)?;
    let k_norm_key = require_key(
        arch.attn_k_per_head_norm_key(layer),
        layer,
        "attn_k_per_head_norm_key",
        "full-attn",
    )?;
    let attn_k_norm = get_vec(weights, &k_norm_key)?;

    Ok(Qwen35AttentionLayerWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_q_norm,
        attn_k_norm,
        attn_output,
    })
}

fn get_tensor(
    weights: &ModelWeights,
    key: &str,
) -> Result<ndarray::ArcArray2<f32>, Qwen35LoadError> {
    weights
        .tensors
        .get(key)
        .cloned()
        .ok_or_else(|| Qwen35LoadError::MissingTensor(key.to_string()))
}

fn get_vec(weights: &ModelWeights, key: &str) -> Result<Arc<[f32]>, Qwen35LoadError> {
    weights
        .vectors
        .get(key)
        .map(|v| Arc::from(v.as_slice()))
        .ok_or_else(|| Qwen35LoadError::MissingVector(key.to_string()))
}

fn require_key(
    opt: Option<String>,
    layer: usize,
    method: &'static str,
    kind: &'static str,
) -> Result<String, Qwen35LoadError> {
    opt.ok_or(Qwen35LoadError::NoKey {
        layer,
        method,
        kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::config::{ModelArchitecture as _, ModelConfig};
    use larql_models::{Qwen35Arch, WeightArray};
    use ndarray::Array2;
    use std::collections::HashMap;

    /// Build a tiny synthetic Qwen3.6 config — 4 layers,
    /// `full_attention_interval=2` so layers 1 and 3 are full-attention
    /// and 0, 2 are linear.
    fn qwen35_tiny_config() -> ModelConfig {
        ModelConfig {
            model_type: "qwen35".into(),
            num_layers: 4,
            hidden_size: 8,
            intermediate_size: 16,
            head_dim: 4,
            num_q_heads: 2,
            num_kv_heads: 1,
            vocab_size: Some(32),
            rope_base: 10_000.0,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            attention_multiplier: None,
            logits_scaling: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
            full_attention_interval: Some(2),
            ssm_state_size: Some(2),
            ssm_inner_size: Some(4),
            ssm_group_count: Some(1),
            ssm_dt_rank: Some(2),
            ssm_conv_kernel: Some(2),
            rope_dimension_sections: None,
        }
    }

    fn non_hybrid_config() -> ModelConfig {
        let mut cfg = qwen35_tiny_config();
        cfg.full_attention_interval = None;
        cfg.ssm_state_size = None;
        cfg
    }

    fn make_2d(rows: usize, cols: usize, fill: f32) -> WeightArray {
        Array2::from_elem((rows, cols), fill).into_shared()
    }

    fn populate_tensors(
        tensors: &mut HashMap<String, WeightArray>,
        vectors: &mut HashMap<String, Vec<f32>>,
        arch: &Qwen35Arch,
    ) {
        let hidden = arch.config().hidden_size;
        let ffn_dim = arch.config().intermediate_size;
        let head_dim = arch.config().head_dim;
        let q_dim = arch.config().num_q_heads * head_dim;
        let kv_dim = arch.config().num_kv_heads * head_dim;
        let fused_q_dim = 2 * q_dim;

        let n_v_heads = arch.ssm_dt_rank();
        let head_v_dim = arch.ssm_state_size();
        let n_k_heads = arch.ssm_group_count();
        let key_dim = head_v_dim * n_k_heads;
        let value_dim = head_v_dim * n_v_heads;
        let conv_dim = 2 * key_dim + value_dim;

        vectors.insert("norm.weight".into(), vec![1.0; hidden]);

        for layer in 0..arch.config().num_layers {
            let prefix = format!("layers.{layer}.");
            vectors.insert(format!("{prefix}input_layernorm.weight"), vec![1.0; hidden]);
            vectors.insert(
                format!("{prefix}post_attention_layernorm.weight"),
                vec![1.0; hidden],
            );
            tensors.insert(
                format!("{prefix}mlp.gate_proj.weight"),
                make_2d(ffn_dim, hidden, 0.1),
            );
            tensors.insert(
                format!("{prefix}mlp.up_proj.weight"),
                make_2d(ffn_dim, hidden, 0.1),
            );
            tensors.insert(
                format!("{prefix}mlp.down_proj.weight"),
                make_2d(hidden, ffn_dim, 0.1),
            );

            if arch.is_linear_attention_layer(layer) {
                tensors.insert(
                    format!("{prefix}attn_qkv.weight"),
                    make_2d(conv_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}attn_gate.weight"),
                    make_2d(value_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}ssm_conv1d.weight"),
                    make_2d(arch.ssm_conv_kernel(), conv_dim, 0.5),
                );
                vectors.insert(format!("{prefix}ssm_dt.bias"), vec![0.0; n_v_heads]);
                vectors.insert(format!("{prefix}ssm_a"), vec![-1.0; n_v_heads]);
                tensors.insert(
                    format!("{prefix}ssm_beta.weight"),
                    make_2d(n_v_heads, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}ssm_alpha.weight"),
                    make_2d(n_v_heads, hidden, 0.1),
                );
                vectors.insert(format!("{prefix}ssm_norm.weight"), vec![1.0; head_v_dim]);
                tensors.insert(
                    format!("{prefix}ssm_out.weight"),
                    make_2d(hidden, value_dim, 0.5),
                );
            } else {
                tensors.insert(
                    format!("{prefix}self_attn.q_proj.weight"),
                    make_2d(fused_q_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}self_attn.k_proj.weight"),
                    make_2d(kv_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}self_attn.v_proj.weight"),
                    make_2d(kv_dim, hidden, 0.1),
                );
                tensors.insert(
                    format!("{prefix}self_attn.o_proj.weight"),
                    make_2d(hidden, q_dim, 0.5),
                );
                vectors.insert(
                    format!("{prefix}self_attn.q_norm.weight"),
                    vec![1.0; head_dim],
                );
                vectors.insert(
                    format!("{prefix}self_attn.k_norm.weight"),
                    vec![1.0; head_dim],
                );
            }
        }
    }

    fn build_model_weights(cfg: ModelConfig) -> (ModelWeights, Qwen35Arch) {
        let arch = Qwen35Arch::from_config(cfg.clone());
        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();
        populate_tensors(&mut tensors, &mut vectors, &arch);

        let vocab = cfg.vocab_size.unwrap();
        let hidden = cfg.hidden_size;
        let embed = make_2d(vocab, hidden, 0.5);
        let lm_head = make_2d(vocab, hidden, 0.5);

        let mw = ModelWeights {
            tensors,
            vectors,
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            embed,
            lm_head,
            arch: Box::new(Qwen35Arch::from_config(cfg.clone())),
            num_layers: cfg.num_layers,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            vocab_size: vocab,
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
        };
        (mw, arch)
    }

    #[test]
    fn bridge_loads_hybrid_layer_kinds_correctly() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load_qwen35_weights");
        assert_eq!(w.layers.len(), cfg.num_layers);
        // Layer 0,2 linear; 1,3 full-attn.
        for (i, layer) in w.layers.iter().enumerate() {
            let is_linear = (i + 1) % 2 != 0;
            match (&layer.block, is_linear) {
                (Qwen35LayerWeights::Linear(_), true) => {}
                (Qwen35LayerWeights::Attention(_), false) => {}
                _ => panic!("layer {} dispatched to wrong branch", i),
            }
        }
    }

    #[test]
    fn bridge_propagates_ffn_dim_and_embed_lm_head() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");
        assert_eq!(w.ffn_dim, cfg.intermediate_size);
        assert_eq!(w.embed.shape(), &[cfg.vocab_size.unwrap(), cfg.hidden_size]);
        assert_eq!(
            w.lm_head.shape(),
            &[cfg.vocab_size.unwrap(), cfg.hidden_size]
        );
        assert_eq!(w.final_norm.len(), cfg.hidden_size);
    }

    #[test]
    fn bridge_missing_tensor_errors_with_key_name() {
        let cfg = qwen35_tiny_config();
        let (mut mw, arch) = build_model_weights(cfg);
        // Drop a known tensor from layer 1 (full-attn): self_attn.q_proj.weight
        mw.tensors.remove("layers.1.self_attn.q_proj.weight");
        match load_qwen35_weights(&mw, &arch) {
            Err(Qwen35LoadError::MissingTensor(k)) => {
                assert!(k.contains("layers.1.self_attn.q_proj"), "got {k}")
            }
            Err(other) => panic!("expected MissingTensor, got {other:?}"),
            Ok(_) => panic!("expected MissingTensor, got Ok"),
        }
    }

    #[test]
    fn bridge_missing_vector_errors_with_key_name() {
        let cfg = qwen35_tiny_config();
        let (mut mw, arch) = build_model_weights(cfg);
        mw.vectors.remove("layers.0.ssm_a");
        match load_qwen35_weights(&mw, &arch) {
            Err(Qwen35LoadError::MissingVector(k)) => {
                assert_eq!(k, "layers.0.ssm_a")
            }
            Err(other) => panic!("expected MissingVector, got {other:?}"),
            Ok(_) => panic!("expected MissingVector, got Ok"),
        }
    }

    #[test]
    fn bridge_rejects_non_hybrid_arch() {
        // A Qwen35Arch built without full_attention_interval / ssm_state_size
        // should be flagged before any tensor lookup happens.
        let cfg = non_hybrid_config();
        let arch = Qwen35Arch::from_config(cfg.clone());
        let mw = ModelWeights {
            tensors: HashMap::new(),
            vectors: HashMap::new(),
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            embed: make_2d(1, 1, 0.0),
            lm_head: make_2d(1, 1, 0.0),
            arch: Box::new(Qwen35Arch::from_config(cfg)),
            num_layers: 0,
            hidden_size: 0,
            intermediate_size: 0,
            vocab_size: 0,
            head_dim: 0,
            num_q_heads: 0,
            num_kv_heads: 0,
            rope_base: 0.0,
        };
        match load_qwen35_weights(&mw, &arch) {
            Err(Qwen35LoadError::NotHybrid { field }) => {
                assert_eq!(field, "full_attention_interval")
            }
            Err(other) => panic!("expected NotHybrid, got {other:?}"),
            Ok(_) => panic!("expected NotHybrid, got Ok"),
        }
    }
}
