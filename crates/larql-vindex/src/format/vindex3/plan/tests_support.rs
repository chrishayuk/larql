//! Shared fixtures: Glimmer-shaped inventories built through the real
//! inventory pipeline (no hand-written report structs), so plan tests
//! exercise the same shapes `inspect-hf` produces.

use std::io::Write;
use std::path::Path;

use larql_models::config::{LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_LINEAR_ATTENTION};
use larql_models::inventory::{build_inventory, ArchitectureInventory};

/// Number of layers in the fixture target model.
pub const FIXTURE_LAYERS: usize = 8;

/// Write a config + one shard (with real payload bytes) and build its
/// inventory. Payloads are a deterministic per-offset pattern so encode
/// tests can compare bytes end to end, not just counts.
fn inventory_from(
    dir: &Path,
    config: &serde_json::Value,
    header: &serde_json::Value,
) -> ArchitectureInventory {
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let header_bytes = serde_json::to_vec(header).unwrap();
    let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    let payload_len = header
        .as_object()
        .unwrap()
        .values()
        .filter_map(|d| d["data_offsets"].as_array())
        .filter_map(|offs| offs.get(1)?.as_u64())
        .max()
        .unwrap_or(0);
    file.write_all(&payload_pattern(payload_len)).unwrap();
    build_inventory(dir).unwrap()
}

/// Deterministic payload bytes for fixture shards: `f(i) = (i * 31 + 7) mod 251`.
pub fn payload_pattern(len: u64) -> Vec<u8> {
    (0..len)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) % 251) as u8)
        .collect()
}

/// Build an artifact from an explicit tensor list `(name, shape)`,
/// offsets sequential — for closure-variant fixtures whose whole point
/// is a precise operand estate.
pub fn custom_artifact(
    dir: &Path,
    config: &serde_json::Value,
    tensors: &[(&str, &[usize])],
) -> ArchitectureInventory {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in tensors {
        push_tensor(&mut header, &mut offset, name, shape);
    }
    inventory_from(dir, config, &serde_json::Value::Object(header))
}

/// Append one BF16 tensor at the running offset.
fn push_tensor(
    header: &mut serde_json::Map<String, serde_json::Value>,
    offset: &mut u64,
    name: &str,
    shape: &[usize],
) {
    let len = 2 * shape.iter().product::<usize>() as u64;
    header.insert(
        name.to_string(),
        serde_json::json!({
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [*offset, *offset + len],
        }),
    );
    *offset += len;
}

/// A Glimmer-shaped target: unknown family, hybrid attention, per-layer
/// rope array with NoPE zeros on the global layers, dangerous unconsumed
/// scalars, a vision subtree, mixed tensor stack.
pub fn glimmer_shaped_target(dir: &Path) -> ArchitectureInventory {
    glimmer_shaped_target_with(dir, |_| {})
}

/// The same fixture with `mutate` applied to its config first.
///
/// Exists so a test can *withdraw* a declaration — the only way to check
/// that an absent semantic fact stays absent instead of reappearing as a
/// plausible identity.
pub fn glimmer_shaped_target_with(
    dir: &Path,
    mutate: impl FnOnce(&mut serde_json::Value),
) -> ArchitectureInventory {
    let layer_types: Vec<&str> = (0..FIXTURE_LAYERS)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let layer_rope_theta: Vec<f64> = (0..FIXTURE_LAYERS)
        .map(|i| if i % 4 == 3 { 0.0 } else { 500000.0 })
        .collect();
    let mut config = serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 64,
            "num_hidden_layers": FIXTURE_LAYERS,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "sliding_window": 16,
            "rms_norm_eps": 1e-5,
            "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
            "layer_types": layer_types,
            "layer_rope_theta": layer_rope_theta,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "post_norm_eps": 1e-8
        },
        // The vision tower declares its own interleave and rope base.
        // Both were absent from this fixture until the carriage gate
        // found them dropped on the real checkpoint — the fixture could
        // not see a hole it did not declare, so the nested-component
        // policy gap passed every test while failing the actual model.
        "vision_config": {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-6,
            "layer_types": [
                "window_attention",
                "window_attention",
                "window_attention",
                "full_attention"
            ],
            "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" }
        }
    });
    mutate(&mut config);
    // Full operand estate, mirroring the real four-norm Glimmer layer
    // anatomy, attention gate included (its semantics are judged on the
    // registered family): 12 tensors per layer.
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    push_tensor(
        &mut header,
        &mut offset,
        "model.language_model.embed_tokens.weight",
        &[128, 64],
    );
    push_tensor(
        &mut header,
        &mut offset,
        "model.vision_tower.layers.0.attn.qkv.weight",
        &[32, 32],
    );
    push_tensor(&mut header, &mut offset, "lm_head.weight", &[128, 64]);
    push_tensor(
        &mut header,
        &mut offset,
        "model.language_model.norm.weight",
        &[64],
    );
    for layer in 0..FIXTURE_LAYERS {
        let stack = format!("model.language_model.layers.{layer}");
        // 8 q-heads * head_dim 8 = 64 rows; 2 kv-heads * 8 = 16 rows.
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.q_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.k_proj.weight"),
            &[16, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.v_proj.weight"),
            &[16, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.o_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.self_attn.gate_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.mlp.gate_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.mlp.up_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("{stack}.mlp.down_proj.weight"),
            &[64, 256],
        );
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            push_tensor(
                &mut header,
                &mut offset,
                &format!("{stack}.{norm}.weight"),
                &[64],
            );
        }
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}

/// Gemma 4 26B-A4B in miniature — the V3-F0 witness-3 family, one
/// declared fact per hostile semantic: per-layer head geometry
/// (`head_dim` 8 / 2 KV heads on sliding layers, `global_head_dim` 24 /
/// 1 KV head on the full layer — 24 not 16, so kv_rows 24 ≠ 16 and no
/// product collides), per-layer-type rope (`proportional` + partial
/// rotary on full, plain on sliding), K≡V on the full layer (no `v_proj`
/// there), a hybrid dense+routed FFN with router scales, five FFN norms
/// and `layer_scalar`, tied softcapped head, the PLE/double-wide/KV-shared
/// knobs declared OFF, the multimodal interface at the root, and a vision
/// tower without `layer_types`. Tensor names and shapes follow the real
/// checkpoint's `model.language_model.…` spelling.
pub const GEMMA4_FIXTURE_LAYERS: usize = 4;
pub const GEMMA4_FULL_LAYER: usize = 3;
pub const GEMMA4_HIDDEN: usize = 64;
pub const GEMMA4_Q_HEADS: usize = 8;
pub const GEMMA4_HEAD_DIM: usize = 8;
pub const GEMMA4_KV_HEADS: usize = 2;
pub const GEMMA4_GLOBAL_HEAD_DIM: usize = 24;
pub const GEMMA4_GLOBAL_KV_HEADS: usize = 1;
pub const GEMMA4_INTER: usize = 128;
pub const GEMMA4_EXPERTS: usize = 4;
pub const GEMMA4_TOP_K: usize = 2;
pub const GEMMA4_MOE_INTER: usize = 32;
pub const GEMMA4_VOCAB: usize = 128;
pub const GEMMA4_FULL_THETA: f64 = 1_000_000.0;
pub const GEMMA4_SLIDING_THETA: f64 = 10_000.0;
pub const GEMMA4_PARTIAL_ROTARY: f64 = 0.25;

pub fn gemma4_shaped_target(dir: &Path) -> ArchitectureInventory {
    gemma4_shaped_target_with(dir, |_| {}, |_| {})
}

/// The same fixture with `mutate_config` applied to its config and
/// `mutate_tensors` to its `(name, shape)` list before writing.
pub fn gemma4_shaped_target_with(
    dir: &Path,
    mutate_config: impl FnOnce(&mut serde_json::Value),
    mutate_tensors: impl FnOnce(&mut Vec<(String, Vec<usize>)>),
) -> ArchitectureInventory {
    let layer_types: Vec<&str> = (0..GEMMA4_FIXTURE_LAYERS)
        .map(|i| {
            if i == GEMMA4_FULL_LAYER {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    // Built in three pieces: one `json!` of this depth trips the macro's
    // recursion limit.
    let text_config = serde_json::json!({
        "model_type": "gemma4_text",
        "hidden_size": GEMMA4_HIDDEN,
            "num_hidden_layers": GEMMA4_FIXTURE_LAYERS,
            "intermediate_size": GEMMA4_INTER,
            "num_attention_heads": GEMMA4_Q_HEADS,
            "num_key_value_heads": GEMMA4_KV_HEADS,
            "head_dim": GEMMA4_HEAD_DIM,
            "global_head_dim": GEMMA4_GLOBAL_HEAD_DIM,
            "num_global_key_value_heads": GEMMA4_GLOBAL_KV_HEADS,
            "attention_k_eq_v": true,
            "attention_bias": false,
            "enable_moe_block": true,
            "num_experts": GEMMA4_EXPERTS,
            "top_k_experts": GEMMA4_TOP_K,
            "moe_intermediate_size": GEMMA4_MOE_INTER,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": GEMMA4_VOCAB,
            "use_double_wide_mlp": false,
            "num_kv_shared_layers": 0,
            "use_bidirectional_attention": "vision",
            "vocab_size": GEMMA4_VOCAB,
            "sliding_window": 16,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": GEMMA4_PARTIAL_ROTARY,
                    "rope_theta": GEMMA4_FULL_THETA,
                    "rope_type": "proportional"
                },
                "sliding_attention": { "rope_theta": GEMMA4_SLIDING_THETA, "rope_type": "default" }
            },
        "layer_types": layer_types,
        "tie_word_embeddings": true
    });
    let vision_config = serde_json::json!({
        "model_type": "gemma4_vision",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "intermediate_size": 64,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "rope_theta": 100.0, "rope_type": "default" },
            "attention_bias": false,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 64,
            "default_output_length": 4,
            "standardize": true,
            "use_clipped_linears": false,
            "id2label": { "0": "LABEL_0" },
            "label2id": { "LABEL_0": 0 },
            "problem_type": null,
            "return_dict": true,
            "output_attentions": false,
            "output_hidden_states": false,
        "is_encoder_decoder": false,
        "chunk_size_feed_forward": 0
    });
    let mut config = serde_json::json!({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "gemma4",
        "audio_config": null,
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "boi_token_id": 255999,
        "eoa_token_id": 258883,
        "eoa_token_index": 258883,
        "eoi_token_id": 258882,
        "image_token_id": 258880,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 4,
        "tie_word_embeddings": true,
        "text_config": text_config,
        "vision_config": vision_config
    });
    mutate_config(&mut config);

    let h = GEMMA4_HIDDEN;
    let mut tensors: Vec<(String, Vec<usize>)> = vec![
        (
            "model.language_model.embed_tokens.weight".into(),
            vec![GEMMA4_VOCAB, h],
        ),
        ("model.language_model.norm.weight".into(), vec![h]),
        (
            "model.embed_vision.embedding_projection.weight".into(),
            vec![h, 32],
        ),
        (
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight".into(),
            vec![32, 32],
        ),
    ];
    for layer in 0..GEMMA4_FIXTURE_LAYERS {
        let stack = format!("model.language_model.layers.{layer}");
        let full = layer == GEMMA4_FULL_LAYER;
        let (head_dim, kv_heads) = if full {
            (GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_KV_HEADS)
        } else {
            (GEMMA4_HEAD_DIM, GEMMA4_KV_HEADS)
        };
        let q_rows = GEMMA4_Q_HEADS * head_dim;
        let kv_rows = kv_heads * head_dim;
        tensors.push((format!("{stack}.self_attn.q_proj.weight"), vec![q_rows, h]));
        tensors.push((format!("{stack}.self_attn.k_proj.weight"), vec![kv_rows, h]));
        if !full {
            tensors.push((format!("{stack}.self_attn.v_proj.weight"), vec![kv_rows, h]));
        }
        tensors.push((format!("{stack}.self_attn.o_proj.weight"), vec![h, q_rows]));
        tensors.push((format!("{stack}.self_attn.q_norm.weight"), vec![head_dim]));
        tensors.push((format!("{stack}.self_attn.k_norm.weight"), vec![head_dim]));
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ] {
            tensors.push((format!("{stack}.{norm}.weight"), vec![h]));
        }
        tensors.push((format!("{stack}.layer_scalar"), vec![1]));
        tensors.push((
            format!("{stack}.mlp.gate_proj.weight"),
            vec![GEMMA4_INTER, h],
        ));
        tensors.push((format!("{stack}.mlp.up_proj.weight"), vec![GEMMA4_INTER, h]));
        tensors.push((
            format!("{stack}.mlp.down_proj.weight"),
            vec![h, GEMMA4_INTER],
        ));
        tensors.push((
            format!("{stack}.router.proj.weight"),
            vec![GEMMA4_EXPERTS, h],
        ));
        tensors.push((format!("{stack}.router.scale"), vec![h]));
        tensors.push((
            format!("{stack}.router.per_expert_scale"),
            vec![GEMMA4_EXPERTS],
        ));
        tensors.push((
            format!("{stack}.experts.gate_up_proj"),
            vec![GEMMA4_EXPERTS, 2 * GEMMA4_MOE_INTER, h],
        ));
        tensors.push((
            format!("{stack}.experts.down_proj"),
            vec![GEMMA4_EXPERTS, h, GEMMA4_MOE_INTER],
        ));
    }
    mutate_tensors(&mut tensors);
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in &tensors {
        push_tensor(&mut header, &mut offset, name, shape);
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}

/// A drafter-shaped artifact declaring `target_layer_ids` taps into a
/// deeper producer.
pub fn drafter_shaped(dir: &Path) -> ArchitectureInventory {
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerAssistantModel"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer_assistant",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "sliding_window": 16,
        "block_size": 4,
        "mask_token_id": 99,
        "target_layer_ids": [1, 3, 5]
    });
    // The first four tensors keep their historical offsets — encode tests
    // slice the source pattern at [8320, 32896) for the projector. The
    // appended estate mirrors the real assistant's two-norm + QK-norm
    // layer anatomy (11 tensors per layer).
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    push_tensor(
        &mut header,
        &mut offset,
        "layers.0.self_attn.q_proj.weight",
        &[64, 64],
    );
    push_tensor(&mut header, &mut offset, "norm.weight", &[64]);
    push_tensor(&mut header, &mut offset, "encoder.fc.weight", &[192, 64]);
    push_tensor(
        &mut header,
        &mut offset,
        "encoder.output_norm_enc.weight",
        &[64],
    );
    for layer in 0..2 {
        if layer > 0 {
            push_tensor(
                &mut header,
                &mut offset,
                &format!("layers.{layer}.self_attn.q_proj.weight"),
                &[64, 64],
            );
        }
        // 8 q-heads * head_dim 8 = 64 rows; 4 kv-heads * 8 = 32 rows.
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.k_proj.weight"),
            &[32, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.v_proj.weight"),
            &[32, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.o_proj.weight"),
            &[64, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.q_norm.weight"),
            &[8],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.self_attn.k_norm.weight"),
            &[8],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.input_layernorm.weight"),
            &[64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.post_attention_layernorm.weight"),
            &[64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.gate_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.up_proj.weight"),
            &[256, 64],
        );
        push_tensor(
            &mut header,
            &mut offset,
            &format!("layers.{layer}.mlp.down_proj.weight"),
            &[64, 256],
        );
    }
    inventory_from(dir, &config, &serde_json::Value::Object(header))
}

/// A fully-known dense model: recognised family, no unconsumed keys beyond
/// metadata, uniform attention.
/// [`known_dense`] with the caller's config, for gates that turn on one
/// declared key rather than on the shape.
pub fn known_dense_with_config(dir: &Path, config: serde_json::Value) -> ArchitectureInventory {
    let header = serde_json::json!({
        "model.embed_tokens.weight":
            {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 16384]}
    });
    inventory_from(dir, &config, &header)
}

pub fn known_dense(dir: &Path) -> ArchitectureInventory {
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    let header = serde_json::json!({
        "model.embed_tokens.weight":
            {"dtype": "BF16", "shape": [128, 64], "data_offsets": [0, 16384]}
    });
    inventory_from(dir, &config, &header)
}

/// Period of the hybrid interleave: one `full_attention` layer in every
/// [`HYBRID_FULL_ATTENTION_INTERVAL`], the rest recurrent. The 3:1 Qwen3.8
/// / Kimi-Linear cadence.
pub const HYBRID_FULL_ATTENTION_INTERVAL: usize = 4;

/// Depthwise causal conv width over the fused q|k|v channels.
pub const HYBRID_CONV_KERNEL: usize = 4;
/// Dk — key-side head width.
pub const HYBRID_KEY_HEAD_DIM: usize = 16;
/// Dv — value-side head width.
pub const HYBRID_VALUE_HEAD_DIM: usize = 16;
/// Hk — key-side head count.
pub const HYBRID_KEY_HEADS: usize = 2;
/// Hv — value-side head count. Larger than [`HYBRID_KEY_HEADS`], as on
/// Qwen3.8, so a fixture cannot pass by folding the two sides together.
pub const HYBRID_VALUE_HEADS: usize = 4;
/// Precision the recurrence keeps its state at.
pub const HYBRID_STATE_DTYPE: &str = "float32";

/// Whether layer `i` is the full-attention layer of the hybrid cadence.
fn is_full_attention_layer(i: usize) -> bool {
    i % HYBRID_FULL_ATTENTION_INTERVAL == HYBRID_FULL_ATTENTION_INTERVAL - 1
}

/// Write the 3:1 `linear_attention` / `full_attention` cadence into a
/// config, the way Qwen3.8 and Kimi Linear write it.
///
/// Declares *that* the stack is hybrid and nothing about which recurrence
/// it runs — [`declare_gated_delta_geometry`] is the separate fact that
/// identifies the operator.
pub fn declare_hybrid_cadence(config: &mut serde_json::Value) {
    let layer_types: Vec<&str> = (0..FIXTURE_LAYERS)
        .map(|i| {
            if is_full_attention_layer(i) {
                LAYER_TYPE_FULL_ATTENTION
            } else {
                LAYER_TYPE_LINEAR_ATTENTION
            }
        })
        .collect();
    config["text_config"]["layer_types"] = serde_json::json!(layer_types);
    config["text_config"]["full_attention_interval"] =
        serde_json::json!(HYBRID_FULL_ATTENTION_INTERVAL);
}

/// Declare the geometry that *identifies* the recurrence as Gated
/// DeltaNet.
///
/// Kept separate from [`declare_hybrid_cadence`] because the two are
/// independent facts and the discriminator is precisely whether this one
/// is present: a cadence names a recurrence, and only the geometry names
/// *which* recurrence. `LinearAttentionTopology::from_config` refuses a
/// partial declaration, so these are declared together or not at all.
pub fn declare_gated_delta_geometry(config: &mut serde_json::Value) {
    let text = &mut config["text_config"];
    text["linear_conv_kernel_dim"] = serde_json::json!(HYBRID_CONV_KERNEL);
    text["linear_key_head_dim"] = serde_json::json!(HYBRID_KEY_HEAD_DIM);
    text["linear_value_head_dim"] = serde_json::json!(HYBRID_VALUE_HEAD_DIM);
    text["linear_num_key_heads"] = serde_json::json!(HYBRID_KEY_HEADS);
    text["linear_num_value_heads"] = serde_json::json!(HYBRID_VALUE_HEADS);
    text["mamba_ssm_dtype"] = serde_json::json!(HYBRID_STATE_DTYPE);
}
