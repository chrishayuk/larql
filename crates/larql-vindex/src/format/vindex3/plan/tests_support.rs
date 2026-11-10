//! Shared fixtures: Glimmer-shaped inventories built through the real
//! inventory pipeline (no hand-written report structs), so plan tests
//! exercise the same shapes `inspect-hf` produces.

use std::io::Write;
use std::path::Path;

use larql_models::inventory::{build_inventory, ArchitectureInventory};

mod flat_gemma3;
mod known_dense;
mod shaped_targets;
pub use flat_gemma3::*;
pub use known_dense::*;
pub use shaped_targets::*;

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
