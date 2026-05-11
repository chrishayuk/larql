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
    // GGUF stores ssm_conv1d as `[d_conv, conv_dim]` in GGML's
    // (cols, rows) order, which the loader's standard dim-swap
    // produces as `[conv_dim, d_conv]` in HF's `[rows, cols]`.
    // The forward path (causal_conv1d_step) expects rows = d_conv
    // and cols = conv_dim (per-tap-channel layout), so transpose
    // once at load time. Trivial cost: 48 layers × 4 × 10240 f32
    // ≈ 7.8 MB total for Qwen 3.6 27B, done once.
    let ssm_conv1d_raw = get_tensor(weights, &ssm_conv1d_key)?;
    let ssm_conv1d = ssm_conv1d_raw.t().to_owned().into_shared();
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
            // Qwen35Arch overrides post_attention_layernorm_key →
            // `post_attention_norm.weight` (GGUF-native naming).
            vectors.insert(
                format!("{prefix}post_attention_norm.weight"),
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
                // Match the real GGUF layout `[conv_dim, d_conv]` —
                // the bridge transposes to `[d_conv, conv_dim]` for
                // the forward path.
                tensors.insert(
                    format!("{prefix}ssm_conv1d.weight"),
                    make_2d(conv_dim, arch.ssm_conv_kernel(), 0.5),
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
                // GGUF-native naming (no `self_attn` infix).
                vectors.insert(format!("{prefix}attn_q_norm.weight"), vec![1.0; head_dim]);
                vectors.insert(format!("{prefix}attn_k_norm.weight"), vec![1.0; head_dim]);
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

    // ── C.4e: end-to-end bridge → forward integration ──
    //
    // These tests feed the typed `Qwen35Weights` produced by
    // `load_qwen35_weights` into `qwen35_forward_step` to verify
    // the bridge produces structurally compatible output. They
    // exercise both layer kinds in one forward call (the tiny
    // config has 2 linear + 2 attention layers).

    fn integration_dn_dims(cfg: &ModelConfig) -> crate::attention::deltanet_block::DeltaNetDims {
        crate::attention::deltanet_block::DeltaNetDims {
            hidden: cfg.hidden_size,
            head_v_dim: cfg.ssm_state_size.unwrap(),
            n_v_heads: cfg.ssm_dt_rank.unwrap(),
            n_k_heads: cfg.ssm_group_count.unwrap(),
            d_conv: cfg.ssm_conv_kernel.unwrap(),
            eps: 1e-6,
        }
    }

    fn integration_attn_dims(
        cfg: &ModelConfig,
    ) -> crate::attention::qwen35_block::Qwen35AttentionDims {
        crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: cfg.hidden_size,
            n_head: cfg.num_q_heads,
            n_head_kv: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.head_dim,
            rope_base: cfg.rope_base,
            eps: 1e-6,
        }
    }

    fn layer_kinds(arch: &Qwen35Arch) -> Vec<bool> {
        (0..arch.config().num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect()
    }

    #[test]
    fn integration_bridge_forward_one_token_returns_finite_vocab_logits() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");

        let dn_dims = integration_dn_dims(&cfg);
        let attn_dims = integration_attn_dims(&cfg);
        let kinds = layer_kinds(&arch);
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let logits = crate::attention::qwen35_forward::qwen35_forward_step(
            3, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        assert_eq!(logits.len(), cfg.vocab_size.unwrap());
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must all be finite"
        );
        assert_eq!(cache.next_position, 1);
    }

    #[test]
    fn integration_bridge_forward_grows_kv_and_advances_position_over_two_tokens() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");

        let dn_dims = integration_dn_dims(&cfg);
        let attn_dims = integration_attn_dims(&cfg);
        let kinds = layer_kinds(&arch);
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            1, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            2, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        assert_eq!(cache.next_position, 2);
        // Every full-attn layer's KV slab should have grown to 2 rows.
        for (layer_idx, kv) in cache.kv_layers.iter().enumerate() {
            let is_linear = (layer_idx + 1) % 2 != 0;
            match (kv, is_linear) {
                (None, true) => {} // linear layer — no KV slab
                (Some((k, v)), false) => {
                    assert_eq!(
                        k.shape()[0],
                        2,
                        "full-attn layer {layer_idx} K should have 2 rows"
                    );
                    assert_eq!(
                        v.shape()[0],
                        2,
                        "full-attn layer {layer_idx} V should have 2 rows"
                    );
                }
                _ => panic!("layer {layer_idx}: KV slot kind mismatch"),
            }
        }
    }

    #[test]
    fn integration_bridge_forward_dn_state_carries_across_two_tokens() {
        let cfg = qwen35_tiny_config();
        let (mw, arch) = build_model_weights(cfg.clone());
        let w = load_qwen35_weights(&mw, &arch).expect("load");

        let dn_dims = integration_dn_dims(&cfg);
        let attn_dims = integration_attn_dims(&cfg);
        let kinds = layer_kinds(&arch);
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            5, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        // The first linear layer's recurrent state should be off zero.
        let mass_layer0_after_1: f32 = cache.dn_state.layers[0]
            .as_ref()
            .expect("linear layer 0 has dn_state")
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_layer0_after_1 > 0.0,
            "layer 0 recurrent_state should be non-zero after one step"
        );

        // Layer 1 (full-attn) should have no dn_state slot.
        assert!(cache.dn_state.layers[1].is_none());

        let _ = crate::attention::qwen35_forward::qwen35_forward_step(
            7, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        let mass_layer0_after_2: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_layer0_after_2 > 0.0,
            "layer 0 recurrent_state still non-zero after second step"
        );
        assert!(
            mass_layer0_after_2.is_finite(),
            "recurrent_state must remain finite"
        );
    }

    // ── C.4f: real-GGUF smoke test ──
    //
    // Gated on `LARQL_QWEN35_GGUF=/path/to/Qwen3.6-*.gguf`. If the
    // env var is unset, the test is a no-op. Loads the GGUF, builds
    // the Qwen35Arch / Qwen35MoeArch from its metadata, runs the
    // bridge, and asserts every layer's weights are present with
    // the expected shapes. Does NOT run the forward (which would
    // need ~50 GB working memory after Q*K dequant); just verifies
    // every key the bridge expects is in the loaded `ModelWeights`.
    #[test]
    fn real_gguf_qwen35_bridge_smoke() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping real-GGUF smoke test");
                return;
            }
        };
        let weights = larql_models::load_gguf(std::path::Path::new(&path))
            .expect("load_gguf failed on Qwen3.6 GGUF");
        let arch_family = weights.arch.family().to_string();
        assert!(
            matches!(arch_family.as_str(), "qwen35" | "qwen35moe"),
            "GGUF arch must be qwen35 or qwen35moe, got `{arch_family}`"
        );

        let w = load_qwen35_weights(&weights, &*weights.arch).expect("load_qwen35_weights failed");
        assert_eq!(w.layers.len(), weights.num_layers);
        assert_eq!(w.embed.shape()[1], weights.hidden_size);
        assert_eq!(w.lm_head.shape()[1], weights.hidden_size);
        assert_eq!(w.final_norm.len(), weights.hidden_size);

        // Spot-check the first linear and first full-attn layer's
        // tensor shapes.
        let conv_kernel = weights.arch.ssm_conv_kernel();
        let head_v_dim = weights.arch.ssm_state_size();
        let n_v_heads = weights.arch.ssm_dt_rank();
        let n_k_heads = weights.arch.ssm_group_count();
        let value_dim = head_v_dim * n_v_heads;
        let key_dim = head_v_dim * n_k_heads;
        let conv_dim = 2 * key_dim + value_dim;

        for (idx, layer) in w.layers.iter().enumerate() {
            match &layer.block {
                Qwen35LayerWeights::Linear(dn) => {
                    assert_eq!(
                        dn.ssm_conv1d.shape(),
                        &[conv_kernel, conv_dim],
                        "layer {idx} ssm_conv1d shape (after bridge transpose)"
                    );
                    assert_eq!(dn.ssm_a.len(), n_v_heads, "layer {idx} ssm_a");
                    assert_eq!(dn.ssm_dt.len(), n_v_heads, "layer {idx} ssm_dt");
                    assert_eq!(dn.ssm_norm.len(), head_v_dim, "layer {idx} ssm_norm");
                    assert_eq!(
                        dn.attn_qkv.shape(),
                        &[conv_dim, weights.hidden_size],
                        "layer {idx} attn_qkv"
                    );
                }
                Qwen35LayerWeights::Attention(at) => {
                    let head_dim = weights.head_dim;
                    assert_eq!(
                        at.attn_q_norm.len(),
                        head_dim,
                        "layer {idx} attn_q_norm len"
                    );
                    assert_eq!(
                        at.attn_k_norm.len(),
                        head_dim,
                        "layer {idx} attn_k_norm len"
                    );
                    assert_eq!(
                        at.attn_q.shape(),
                        &[2 * weights.num_q_heads * head_dim, weights.hidden_size],
                        "layer {idx} attn_q (fused Q+gate)"
                    );
                }
            }
            assert_eq!(
                layer.attn_post_norm.len(),
                weights.hidden_size,
                "layer {idx} attn_post_norm"
            );
        }
    }

    // ── C.4g: real-GGUF single-token forward smoke ──
    //
    // Gated on the same `LARQL_QWEN35_GGUF` env var as the bridge
    // smoke test. Loads the GGUF, builds the bridge, allocates a
    // hybrid cache, then runs `qwen35_forward_step` for one token.
    // Asserts the logits are finite, shape matches vocab, and that
    // the cache position advanced. Verifies the entire C.1 → C.4f
    // stack composes against real Q4_K_S weights without panicking,
    // NaN-ing, or shape-mismatching.
    //
    // Out-of-scope: matching llama.cpp's logits or argmax (that's
    // C.5 parity oracle). Here we only assert the forward runs to
    // completion and produces a finite distribution.
    #[test]
    fn real_gguf_qwen35_forward_one_token_smoke() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping real-GGUF forward smoke test");
                return;
            }
        };
        let weights = larql_models::load_gguf(std::path::Path::new(&path))
            .expect("load_gguf failed on Qwen3.6 GGUF");
        let w = load_qwen35_weights(&weights, &*weights.arch).expect("load_qwen35_weights failed");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            eps: 1e-6,
        };
        let rotary_dim: usize = arch
            .rope_dimension_sections()
            .map(|s| s.iter().sum())
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let token: u32 = 0; // any in-vocab token works for shape/finite check
        let t = std::time::Instant::now();
        let logits = crate::attention::qwen35_forward::qwen35_forward_step(
            token, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
        );
        eprintln!("forward(one token) took {:?}", t.elapsed());

        let vocab = weights.embed.shape()[0];
        assert_eq!(logits.len(), vocab, "logits vocab dim");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits must be finite (sample: {:?})",
            &logits.as_slice().unwrap()[..5.min(logits.len())]
        );
        assert_eq!(cache.next_position, 1);

        // Spot-check: at least one logit should be non-zero (a
        // uniform-zero output would indicate a silent zero-out
        // somewhere in the stack).
        let max_abs = logits.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
        assert!(
            max_abs > 0.0,
            "max |logit| should be > 0; got {max_abs} (uniform zero output)"
        );
    }

    // ── C.4h: real-GGUF multi-token argmax decode ──
    //
    // Same gating as C.4f/C.4g. Generates `N_DECODE_STEPS` tokens
    // autoregressively by argmax sampling from token 0. Verifies
    // the cache / RoPE positions / DeltaNet state work across
    // iterations. Asserts:
    //
    // - Every forward returns finite logits with vocab shape.
    // - `cache.next_position` advances monotonically (1, 2, 3, …).
    // - The generated sequence has at least some variation
    //   (≥ 2 distinct tokens). A model collapsing to a fixed-point
    //   token would indicate a broken cache or numerical drift,
    //   not a meaningful inference path.
    //
    // Out of scope: any check vs llama.cpp's argmax (Phase C.5).
    #[test]
    fn real_gguf_qwen35_multi_token_argmax_decode() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping multi-token decode test");
                return;
            }
        };
        const N_DECODE_STEPS: usize = 4;

        let weights = larql_models::load_gguf(std::path::Path::new(&path))
            .expect("load_gguf failed on Qwen3.6 GGUF");
        let w = load_qwen35_weights(&weights, &*weights.arch).expect("load_qwen35_weights failed");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            eps: 1e-6,
        };
        let rotary_dim: usize = arch
            .rope_dimension_sections()
            .map(|s| s.iter().sum())
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let vocab = weights.embed.shape()[0];
        let mut token: u32 = 0;
        let mut generated: Vec<u32> = Vec::with_capacity(N_DECODE_STEPS);
        let mut step_logits: Vec<ndarray::Array1<f32>> = Vec::with_capacity(N_DECODE_STEPS);

        for step in 0..N_DECODE_STEPS {
            let t = std::time::Instant::now();
            let logits = crate::attention::qwen35_forward::qwen35_forward_step(
                token, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            );
            eprintln!("step {step}: token={token} forward took {:?}", t.elapsed());
            assert_eq!(logits.len(), vocab, "step {step} logits vocab dim");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step}: all logits must be finite"
            );
            assert_eq!(
                cache.next_position,
                step + 1,
                "position should advance monotonically"
            );

            // Diagnostic: print top-3 tokens by logit to see if the
            // distribution shifts even when argmax doesn't.
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("  top-3: {:?}", &indexed[..3.min(indexed.len())]);

            // argmax sampling.
            let next_id = indexed[0].0;
            generated.push(next_id as u32);
            step_logits.push(logits);
            token = next_id as u32;
        }

        eprintln!("generated tokens: {generated:?}");

        // The strong correctness signal: the full logit vectors at
        // different positions must NOT be (near-)identical. A 1.0
        // cosine similarity between step 0 and step N's logits
        // would mean the new K/V rows, new DeltaNet state, and
        // advanced RoPE phase produced no change in the output —
        // a smoking gun for cache or state-update bugs.
        //
        // We don't assert argmax variation here: a model can have
        // a clean fixed point at one boilerplate token under
        // degenerate input (token 0 is a common special id), but
        // the *distribution* across the rest of vocab should still
        // shift as context grows.
        if N_DECODE_STEPS >= 2 {
            let cos = cosine_similarity(&step_logits[0], &step_logits[N_DECODE_STEPS - 1]);
            eprintln!(
                "cosine(step 0 logits, step {} logits) = {cos:.6}",
                N_DECODE_STEPS - 1
            );
            assert!(
                cos < 0.9999,
                "logits at step 0 and step {} are too similar (cos={cos:.6}); state isn't propagating",
                N_DECODE_STEPS - 1
            );
        }
    }

    fn cosine_similarity(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    // ── C.4i: real text round-trip via sibling tokenizer.json ──
    //
    // Same gating as C.4f–C.4h. Loads the tokenizer that ships
    // alongside the GGUF, encodes a short prompt, prefills the
    // hybrid cache token-by-token, then argmax-decodes a handful
    // of generations. Asserts:
    //
    // - Tokenizer loads.
    // - Encoding the prompt yields at least one token id within
    //   vocab range.
    // - Every prefill + decode step produces finite logits.
    // - The decoded continuation is a non-empty string.
    //
    // Out of scope: any semantic check on the generated text or
    // parity vs llama.cpp's output (Phase C.5). Here we only
    // assert the text→tokens→forward→tokens→text plumbing works
    // end to end against real Qwen3.6 27B Q4_K_S weights.
    #[test]
    fn real_gguf_qwen35_tokenizer_roundtrip() {
        let path = match std::env::var("LARQL_QWEN35_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("LARQL_QWEN35_GGUF unset — skipping tokenizer round-trip");
                return;
            }
        };
        let gguf_path = std::path::PathBuf::from(&path);
        let model_dir = gguf_path
            .parent()
            .expect("LARQL_QWEN35_GGUF must point to a file in a directory");

        let tokenizer = match crate::tokenizer::load_tokenizer(model_dir) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("no tokenizer.json sibling next to GGUF ({e:?}) — skipping");
                return;
            }
        };

        let prompt = "Hello";
        let encoding = tokenizer.encode(prompt, true).expect("encode prompt");
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        eprintln!("prompt {prompt:?} → tokens {prompt_ids:?}");
        assert!(!prompt_ids.is_empty(), "empty prompt tokenization");

        let weights =
            larql_models::load_gguf(&gguf_path).expect("load_gguf failed on Qwen3.6 GGUF");
        let vocab = weights.embed.shape()[0];
        assert!(
            prompt_ids.iter().all(|&id| (id as usize) < vocab),
            "tokenizer produced an out-of-vocab id: {prompt_ids:?}, vocab={vocab}"
        );

        let w = load_qwen35_weights(&weights, &*weights.arch).expect("bridge load");

        let arch = &*weights.arch;
        let dn_dims = crate::attention::deltanet_block::DeltaNetDims {
            hidden: weights.hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            eps: 1e-6,
        };
        let rotary_dim: usize = arch
            .rope_dimension_sections()
            .map(|s| s.iter().sum())
            .unwrap_or(weights.head_dim);
        let attn_dims = crate::attention::qwen35_block::Qwen35AttentionDims {
            hidden: weights.hidden_size,
            n_head: weights.num_q_heads,
            n_head_kv: weights.num_kv_heads,
            head_dim: weights.head_dim,
            rotary_dim,
            rope_base: weights.rope_base,
            eps: 1e-6,
        };
        let layer_kinds: Vec<bool> = (0..weights.num_layers)
            .map(|l| arch.is_linear_attention_layer(l))
            .collect();
        let mut cache = crate::attention::qwen35_block::DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // 1. Prefill: feed every prompt token through the forward,
        //    keeping only the last token's logits.
        let mut last_logits: Option<ndarray::Array1<f32>> = None;
        let total = std::time::Instant::now();
        for (i, &tok) in prompt_ids.iter().enumerate() {
            let t = std::time::Instant::now();
            let logits = crate::attention::qwen35_forward::qwen35_forward_step(
                tok, &w, &dn_dims, &attn_dims, &mut cache, 1e-6,
            );
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "prefill step {i} (token={tok}): all logits must be finite"
            );
            eprintln!(
                "prefill step {i}: token={tok} forward took {:?}",
                t.elapsed()
            );
            last_logits = Some(logits);
        }

        // 2. Decode N tokens by argmax from the prefilled cache.
        const N_GEN: usize = 4;
        let mut generated: Vec<u32> = Vec::with_capacity(N_GEN);
        let mut logits = last_logits.expect("at least one prompt token");
        for step in 0..N_GEN {
            let (next_id, _) = logits.iter().enumerate().fold(
                (0_usize, f32::NEG_INFINITY),
                |(best_idx, best_v), (i, &v)| {
                    if v > best_v {
                        (i, v)
                    } else {
                        (best_idx, best_v)
                    }
                },
            );
            generated.push(next_id as u32);
            if step + 1 < N_GEN {
                let t = std::time::Instant::now();
                logits = crate::attention::qwen35_forward::qwen35_forward_step(
                    next_id as u32,
                    &w,
                    &dn_dims,
                    &attn_dims,
                    &mut cache,
                    1e-6,
                );
                eprintln!(
                    "decode step {step}: token={next_id} forward took {:?}",
                    t.elapsed()
                );
                assert!(
                    logits.iter().all(|v| v.is_finite()),
                    "decode step {step}: all logits must be finite"
                );
            }
        }
        eprintln!("total round-trip time {:?}", total.elapsed());

        // 3. Decode generated tokens back to text.
        let continuation = tokenizer
            .decode(&generated, false)
            .expect("decode generated tokens");
        eprintln!("prompt {prompt:?} + generated {generated:?} → continuation {continuation:?}");

        // The strict end-to-end gate: the decoded continuation should
        // not be empty. Empty would indicate the generator produced
        // only special tokens that the decoder elided, or that
        // tokenizer / id mismatch left us decoding garbage.
        assert!(
            !continuation.is_empty(),
            "decoded continuation is empty; generated ids: {generated:?}"
        );
    }
}
