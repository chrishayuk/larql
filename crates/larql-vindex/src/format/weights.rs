//! Model weights serialization to/from .vindex directories.
//!
//! Split format (v2): separate files per component, no duplication.
//!   attn_weights.bin  — Q, K, V, O per layer
//!   up_weights.bin    — FFN up projections (gate is in gate_vectors.bin)
//!   down_weights.bin  — FFN down projections
//!   norms.bin         — all LayerNorm/RMSNorm vectors
//!   lm_head.bin       — output projection
//!
//! Both the build path (full ModelWeights in RAM) and the streaming path
//! (mmap'd safetensors) write through the same `write_model_weights` function
//! via the `WeightSource` trait.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;
use crate::format::load::load_vindex_config;
use crate::index::core::IndexLoadCallbacks;

use larql_models::ModelWeights;

#[derive(Serialize, Deserialize)]
struct WeightEntry {
    key: String,
    kind: String,
    shape: Vec<usize>,
    offset: u64,
    length: u64,
    #[serde(default)]
    file: String,
}

// ── WeightSource trait ──

/// Abstraction over where model weights come from.
///
/// Implemented by `ModelWeights` (build path — everything in RAM)
/// and `StreamingWeights` (streaming path — mmap'd safetensors on demand).
pub trait WeightSource {
    /// Get a 2D weight tensor by normalized key. Returns (data, rows, cols).
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)>;

    /// Get a 1D vector (norm weights, biases) by normalized key.
    fn get_vector(&self, key: &str) -> Option<Vec<f32>>;

    /// Architecture handle for key generation.
    fn arch(&self) -> &dyn larql_models::ModelArchitecture;

    /// Number of layers.
    fn num_layers(&self) -> usize;

    /// LM head matrix. Returns (data, rows, cols).
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)>;

    /// All 1D vector names (for norms).
    fn vector_names(&self) -> Vec<String>;
}

// ── ModelWeights implementation ──

impl WeightSource for ModelWeights {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let t = self.tensors.get(key)?;
        Some((t.as_slice()?.to_vec(), t.shape()[0], t.shape()[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        self.vectors.get(key).cloned()
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        let h = &self.lm_head;
        Some((h.as_slice()?.to_vec(), h.shape()[0], h.shape()[1]))
    }

    fn vector_names(&self) -> Vec<String> {
        self.vectors.keys().cloned().collect()
    }
}

// ── Streaming implementation ──

/// Weight source backed by mmap'd safetensors files.
/// Tensors are deserialized on demand — peak memory is one tensor at a time.
pub struct StreamingWeights<'a> {
    pub shard_mmaps: &'a [&'a [u8]],
    pub tensor_index: &'a HashMap<String, (usize, String)>,
    pub arch: &'a dyn larql_models::ModelArchitecture,
    pub num_layers: usize,
    pub mlx_affine_group_size: Option<usize>,
}

impl<'a> StreamingWeights<'a> {
    fn read_tensor_raw(&self, key: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let (shard_idx, tensor_name) = self.tensor_index.get(key)?;
        let st = safetensors::SafeTensors::deserialize(self.shard_mmaps[*shard_idx]).ok()?;
        let view = st.tensor(tensor_name).ok()?;
        let mut shape = view.shape().to_vec();

        let data = match view.dtype() {
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
            safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
            safetensors::Dtype::U32 if shape.len() == 2 => {
                let group_size = self.mlx_affine_group_size?;
                let stem = key.strip_suffix(".weight")?;

                let (scales_shard, scales_name) =
                    self.tensor_index.get(&format!("{stem}.scales"))?;
                let scales_st =
                    safetensors::SafeTensors::deserialize(self.shard_mmaps[*scales_shard]).ok()?;
                let scales_view = scales_st.tensor(scales_name).ok()?;
                let scales = tensor_view_to_f32(&scales_view).ok()?;

                let biases = if let Some((bias_shard, bias_name)) =
                    self.tensor_index.get(&format!("{stem}.biases"))
                {
                    let bias_st =
                        safetensors::SafeTensors::deserialize(self.shard_mmaps[*bias_shard])
                            .ok()?;
                    let bias_view = bias_st.tensor(bias_name).ok()?;
                    Some(tensor_view_to_f32(&bias_view).ok()?)
                } else {
                    None
                };

                let (data, cols) = crate::format::quant::mlx_affine::dequantize_u32_matrix_bytes(
                    view.data(),
                    shape[0],
                    shape[1],
                    &scales,
                    biases.as_deref(),
                    group_size,
                )
                .ok()?;
                shape[1] = cols;
                data
            }
            _ => return None,
        };
        Some((data, shape))
    }
}

fn tensor_view_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, VindexError> {
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        safetensors::Dtype::F16 => Ok(crate::format::quant::half::decode_f16(view.data())),
        safetensors::Dtype::BF16 => Ok(crate::format::quant::half::decode_bf16(view.data())),
        other => Err(VindexError::Parse(format!(
            "unsupported tensor dtype: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingWeights;
    use crate::format::weights::WeightSource;
    use std::collections::HashMap;

    fn pack_codes(codes: &[u32], bits: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut acc = 0u64;
        let mut acc_bits = 0usize;

        for &code in codes {
            acc |= (code as u64) << acc_bits;
            acc_bits += bits;

            while acc_bits >= 32 {
                out.push((acc & 0xFFFF_FFFF) as u32);
                acc >>= 32;
                acc_bits -= 32;
            }
        }

        if acc_bits > 0 {
            out.push(acc as u32);
        }

        out
    }

    #[test]
    fn streaming_weights_dequantize_mlx_affine_u32() {
        let bits = 4usize;
        let group_size = 64usize;
        let codes: Vec<u32> = (0..group_size).map(|i| (i as u32) & 0xF).collect();
        let packed = pack_codes(&codes, bits);
        let packed_bytes: Vec<u8> = packed.iter().flat_map(|w| w.to_le_bytes()).collect();
        let scales = vec![0.5f32];
        let biases = vec![-2.0f32];
        let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bias_bytes: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();

        let tensors = [
            (
                "embed_tokens.weight".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::U32,
                    vec![1, packed.len()],
                    &packed_bytes,
                )
                .unwrap(),
            ),
            (
                "embed_tokens.scales".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 1],
                    &scale_bytes,
                )
                .unwrap(),
            ),
            (
                "embed_tokens.biases".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![1, 1],
                    &bias_bytes,
                )
                .unwrap(),
            ),
        ];
        let serialized = safetensors::tensor::serialize(tensors, &None).unwrap();

        let mut tensor_index = HashMap::new();
        tensor_index.insert(
            "embed_tokens.weight".to_string(),
            (0usize, "embed_tokens.weight".to_string()),
        );
        tensor_index.insert(
            "embed_tokens.scales".to_string(),
            (0usize, "embed_tokens.scales".to_string()),
        );
        tensor_index.insert(
            "embed_tokens.biases".to_string(),
            (0usize, "embed_tokens.biases".to_string()),
        );

        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 1,
                "hidden_size": 64,
                "intermediate_size": 64,
                "head_dim": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "vocab_size": 1
            }
        }));

        let shards = [serialized.as_slice()];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &tensor_index,
            arch: &*arch,
            num_layers: 1,
            mlx_affine_group_size: Some(group_size),
        };

        let (data, rows, cols) = source.get_tensor("embed_tokens.weight").unwrap();
        assert_eq!((rows, cols), (1, group_size));

        for (idx, value) in data.iter().enumerate() {
            let expected = -2.0 + 0.5 * codes[idx] as f32;
            assert!(
                (value - expected).abs() < 1e-6,
                "mismatch at {idx}: {value} vs {expected}"
            );
        }
    }
}

impl<'a> WeightSource for StreamingWeights<'a> {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 2 {
            return None;
        }
        Some((data, shape[0], shape[1]))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        let (data, shape) = self.read_tensor_raw(key)?;
        if shape.len() != 1 {
            return None;
        }
        Some(data)
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        // Try common lm_head key names
        for key in &["lm_head.weight", "output.weight"] {
            if let Some(t) = self.get_tensor(key) {
                return Some(t);
            }
        }
        None
    }

    fn vector_names(&self) -> Vec<String> {
        // Return all 1D tensor keys (norms, biases)
        let mut names = Vec::new();
        for key in self.tensor_index.keys() {
            if key.contains("layernorm") || key.contains("norm") || key.contains("bias") {
                names.push(key.clone());
            }
        }
        names.sort();
        names
    }
}

// ── Write model weights (generic over source) ──

/// Write model weights to split component files.
///
/// Works with any `WeightSource`: ModelWeights (build path) or
/// StreamingWeights (streaming path from mmap'd safetensors).
pub fn write_model_weights(
    source: &dyn WeightSource,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    callbacks.on_stage("model_weights");
    let start = std::time::Instant::now();

    let dtype = load_vindex_config(dir)
        .map(|c| c.dtype)
        .unwrap_or(crate::config::dtype::StorageDtype::F32);

    let arch = source.arch();
    let num_layers = source.num_layers();
    let mut entries: Vec<WeightEntry> = Vec::new();

    // ── Attention weights ──
    let attn_path = dir.join("attn_weights.bin");
    let mut attn_file = BufWriter::new(std::fs::File::create(&attn_path)?);
    let mut attn_offset: u64 = 0;

    for layer in 0..num_layers {
        callbacks.on_layer_start("attn_weights", layer, num_layers);
        for key in &[
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            if let Some((data, rows, cols)) = source.get_tensor(key) {
                let len = write_floats(&mut attn_file, &data, dtype)?;
                entries.push(WeightEntry {
                    key: key.clone(),
                    kind: "tensor".into(),
                    shape: vec![rows, cols],
                    offset: attn_offset,
                    length: len,
                    file: "attn_weights.bin".into(),
                });
                attn_offset += len;
            }
        }

        // QK norms (1D vectors, stored alongside attention)
        for key in [arch.attn_q_norm_key(layer), arch.attn_k_norm_key(layer)]
            .iter()
            .flatten()
        {
            if let Some(data) = source.get_vector(key) {
                let bytes = crate::config::dtype::encode_floats(&data, dtype);
                attn_file.write_all(&bytes)?;
                entries.push(WeightEntry {
                    key: key.clone(),
                    kind: "vector".into(),
                    shape: vec![data.len()],
                    offset: attn_offset,
                    length: bytes.len() as u64,
                    file: "attn_weights.bin".into(),
                });
                attn_offset += bytes.len() as u64;
            }
        }

        callbacks.on_layer_done("attn_weights", layer, 0.0);
    }
    attn_file.flush()?;

    // ── FFN up + down weights (gate is in gate_vectors.bin) ──
    let up_path = dir.join("up_weights.bin");
    let mut up_file = BufWriter::new(std::fs::File::create(&up_path)?);
    let mut up_offset: u64 = 0;

    let down_path = dir.join("down_weights.bin");
    let mut down_file = BufWriter::new(std::fs::File::create(&down_path)?);
    let mut down_offset: u64 = 0;

    for layer in 0..num_layers {
        callbacks.on_layer_start("up/down_weights", layer, num_layers);

        if arch.is_moe() {
            for expert in 0..arch.num_experts() {
                if let Some(key) = arch.expert_ffn_up_key(layer, expert) {
                    if let Some((data, rows, cols)) = source.get_tensor(&key) {
                        let len = write_floats(&mut up_file, &data, dtype)?;
                        entries.push(WeightEntry {
                            key,
                            kind: "tensor".into(),
                            shape: vec![rows, cols],
                            offset: up_offset,
                            length: len,
                            file: "up_weights.bin".into(),
                        });
                        up_offset += len;
                    }
                }
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    if let Some((data, rows, cols)) = source.get_tensor(&key) {
                        let len = write_floats(&mut down_file, &data, dtype)?;
                        entries.push(WeightEntry {
                            key,
                            kind: "tensor".into(),
                            shape: vec![rows, cols],
                            offset: down_offset,
                            length: len,
                            file: "down_weights.bin".into(),
                        });
                        down_offset += len;
                    }
                }
            }
            if let Some(key) = arch.moe_router_key(layer) {
                if let Some((data, rows, cols)) = source.get_tensor(&key) {
                    let len = write_floats(&mut up_file, &data, dtype)?;
                    entries.push(WeightEntry {
                        key,
                        kind: "tensor".into(),
                        shape: vec![rows, cols],
                        offset: up_offset,
                        length: len,
                        file: "up_weights.bin".into(),
                    });
                    up_offset += len;
                }
            }
        } else {
            let up_key = arch.ffn_up_key(layer);
            if let Some((data, rows, cols)) = source.get_tensor(&up_key) {
                let len = write_floats(&mut up_file, &data, dtype)?;
                entries.push(WeightEntry {
                    key: up_key,
                    kind: "tensor".into(),
                    shape: vec![rows, cols],
                    offset: up_offset,
                    length: len,
                    file: "up_weights.bin".into(),
                });
                up_offset += len;
            }

            let down_key = arch.ffn_down_key(layer);
            if let Some((data, rows, cols)) = source.get_tensor(&down_key) {
                let len = write_floats(&mut down_file, &data, dtype)?;
                entries.push(WeightEntry {
                    key: down_key,
                    kind: "tensor".into(),
                    shape: vec![rows, cols],
                    offset: down_offset,
                    length: len,
                    file: "down_weights.bin".into(),
                });
                down_offset += len;
            }
        }

        callbacks.on_layer_done("up/down_weights", layer, 0.0);
    }
    up_file.flush()?;
    down_file.flush()?;

    // ── Norms ──
    let norms_path = dir.join("norms.bin");
    let mut norms_file = BufWriter::new(std::fs::File::create(&norms_path)?);
    let mut norms_offset: u64 = 0;

    // Per-layer norms
    for layer in 0..num_layers {
        let norm_keys: Vec<String> = [
            Some(arch.input_layernorm_key(layer)),
            Some(arch.post_attention_layernorm_key(layer)),
            arch.pre_feedforward_layernorm_key(layer),
            arch.post_feedforward_layernorm_key(layer),
        ]
        .into_iter()
        .flatten()
        .collect();

        for key in norm_keys {
            if let Some(data) = source.get_vector(&key) {
                let bytes = crate::config::dtype::encode_floats(&data, dtype);
                norms_file.write_all(&bytes)?;
                entries.push(WeightEntry {
                    key,
                    kind: "vector".into(),
                    shape: vec![data.len()],
                    offset: norms_offset,
                    length: bytes.len() as u64,
                    file: "norms.bin".into(),
                });
                norms_offset += bytes.len() as u64;
            }
        }
    }

    // Final norm (model.norm.weight)
    if let Some(data) = source.get_vector("norm.weight") {
        let bytes = crate::config::dtype::encode_floats(&data, dtype);
        norms_file.write_all(&bytes)?;
        entries.push(WeightEntry {
            key: "norm.weight".into(),
            kind: "vector".into(),
            shape: vec![data.len()],
            offset: norms_offset,
            length: bytes.len() as u64,
            file: "norms.bin".into(),
        });
    }
    norms_file.flush()?;

    // ── LM Head ──
    if let Some((data, rows, cols)) = source.lm_head() {
        let lm_bytes = crate::config::dtype::encode_floats(&data, dtype);
        std::fs::write(dir.join("lm_head.bin"), &lm_bytes)?;
        entries.push(WeightEntry {
            key: "lm_head.weight".into(),
            kind: "tensor".into(),
            shape: vec![rows, cols],
            offset: 0,
            length: lm_bytes.len() as u64,
            file: "lm_head.bin".into(),
        });
    }

    // ── Manifest ──
    let manifest_json =
        serde_json::to_string_pretty(&entries).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(dir.join("weight_manifest.json"), manifest_json)?;

    // ── Update index.json ──
    let config_path = dir.join("index.json");
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig =
        serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;

    config.has_model_weights = true;

    let cfg = arch.config();
    config.model_config = Some(VindexModelConfig {
        model_type: cfg.model_type.clone(),
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        sliding_window: cfg.sliding_window,
        moe: if arch.is_moe() {
            Some(crate::MoeConfig {
                num_experts: arch.num_experts(),
                top_k: arch.num_experts_per_token(),
                shared_expert: arch.num_shared_experts() > 0,
                router_type: "top_k_softmax".into(),
            })
        } else {
            None
        },
        // Per-layer geometry (Gemma 4)
        global_head_dim: cfg.global_head_dim,
        num_global_kv_heads: cfg.num_global_kv_heads,
        partial_rotary_factor: cfg.partial_rotary_factor,
        sliding_window_pattern: cfg.sliding_window_pattern,
        layer_types: cfg.layer_types.clone(),
        attention_k_eq_v: cfg.attention_k_eq_v,
        num_kv_shared_layers: cfg.num_kv_shared_layers,
        per_layer_embed_dim: cfg.per_layer_embed_dim,
        rope_local_base: cfg.rope_local_base,
        query_pre_attn_scalar: cfg.query_pre_attn_scalar,
    });

    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;

    callbacks.on_stage_done("model_weights", start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

fn write_floats(
    w: &mut impl Write,
    data: &[f32],
    dtype: crate::config::dtype::StorageDtype,
) -> Result<u64, VindexError> {
    let bytes = crate::config::dtype::encode_floats(data, dtype);
    w.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}

/// Load a full ModelWeights from a vindex directory.
///
/// Tries split files (v2) first, falls back to model_weights.bin (v1).
pub fn load_model_weights(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, VindexError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(VindexError::Parse(
            "vindex does not contain model weights. Rebuild with: larql extract-index <model> -o <vindex> --level all".into(),
        ));
    }

    let model_cfg = config
        .model_config
        .as_ref()
        .ok_or_else(|| VindexError::Parse("vindex missing model_config in index.json".into()))?;

    // Reconstruct full architecture config — includes per-layer geometry for Gemma 4.
    let mut arch_obj = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    // Pass through Gemma 4 per-layer geometry fields (if present in vindex config).
    let obj = arch_obj.as_object_mut().unwrap();
    if let Some(v) = model_cfg.global_head_dim {
        obj.insert("global_head_dim".into(), v.into());
    }
    if let Some(v) = model_cfg.num_global_kv_heads {
        obj.insert("num_global_key_value_heads".into(), v.into());
    }
    if let Some(v) = model_cfg.partial_rotary_factor {
        obj.insert("partial_rotary_factor".into(), v.into());
    }
    if let Some(v) = model_cfg.sliding_window_pattern {
        obj.insert("sliding_window_pattern".into(), v.into());
    }
    if let Some(ref v) = model_cfg.layer_types {
        obj.insert(
            "layer_types".into(),
            serde_json::to_value(v).unwrap_or_default(),
        );
    }
    if model_cfg.attention_k_eq_v {
        obj.insert("attention_k_eq_v".into(), true.into());
    }
    if let Some(v) = model_cfg.num_kv_shared_layers {
        obj.insert("num_kv_shared_layers".into(), v.into());
    }
    if let Some(v) = model_cfg.per_layer_embed_dim {
        obj.insert("hidden_size_per_layer_input".into(), v.into());
    }
    if let Some(v) = model_cfg.rope_local_base {
        obj.insert("rope_local_base_freq".into(), v.into());
    }
    if let Some(v) = model_cfg.query_pre_attn_scalar {
        obj.insert("query_pre_attn_scalar".into(), v.into());
    }
    let arch = larql_models::detect_from_json(&arch_obj);

    callbacks.on_file_start(
        "embeddings",
        &dir.join("embeddings.bin").display().to_string(),
    );
    let embed_file = std::fs::File::open(dir.join("embeddings.bin"))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    // Detect actual dtype from file size (may differ from index.json global dtype)
    let expected_embed_f32 = config.vocab_size * config.hidden_size * 4;
    let embed_dtype = if embed_mmap.len() == expected_embed_f32 {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, embed_dtype);
    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);

    let manifest_path = dir.join("weight_manifest.json");
    if !manifest_path.exists() {
        return Err(VindexError::Parse("weight_manifest.json not found".into()));
    }

    callbacks.on_file_start("model_weights", "weight_manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let entries: Vec<WeightEntry> =
        serde_json::from_str(&manifest_text).map_err(|e| VindexError::Parse(e.to_string()))?;

    let mut mmap_cache: HashMap<String, memmap2::Mmap> = HashMap::new();
    let mut tensors: HashMap<String, larql_models::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_loaded: Option<larql_models::WeightArray> = None;

    for entry in &entries {
        let filename = if entry.file.is_empty() {
            "model_weights.bin".to_string()
        } else {
            entry.file.clone()
        };

        if !mmap_cache.contains_key(&filename) {
            let fpath = dir.join(&filename);
            if fpath.exists() {
                if let Ok(f) = std::fs::File::open(&fpath) {
                    if let Ok(m) = unsafe { memmap2::Mmap::map(&f) } {
                        mmap_cache.insert(filename.clone(), m);
                    }
                }
            }
        }
        let data = match mmap_cache.get(&filename) {
            Some(m) => m.as_ref(),
            None => continue,
        };
        if data.is_empty() {
            continue;
        }

        let byte_offset = entry.offset as usize;
        let byte_count = entry.length as usize;
        if byte_offset + byte_count > data.len() {
            continue;
        }
        let raw_bytes = &data[byte_offset..byte_offset + byte_count];
        // Detect actual dtype from byte count vs expected shape.
        // Gate vector conversion may have changed index.json dtype to f32
        // while weight files remain f16.
        let expected_floats: usize = entry.shape.iter().product();
        let actual_dtype = if byte_count == expected_floats * 4 {
            crate::config::dtype::StorageDtype::F32
        } else if byte_count == expected_floats * 2 {
            crate::config::dtype::StorageDtype::F16
        } else {
            config.dtype // fallback to global
        };
        let floats = crate::config::dtype::decode_floats(raw_bytes, actual_dtype);

        match entry.kind.as_str() {
            "tensor" => {
                let arr = Array2::from_shape_vec((entry.shape[0], entry.shape[1]), floats)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                if entry.key == "lm_head.weight" {
                    lm_head_loaded = Some(arr.into_shared());
                } else {
                    tensors.insert(entry.key.clone(), arr.into_shared());
                }
            }
            "vector" => {
                vectors.insert(entry.key.clone(), floats);
            }
            _ => {}
        }
    }

    // Gate vectors from gate_vectors.bin
    let gate_file = std::fs::File::open(dir.join("gate_vectors.bin"))?;
    let gate_mmap = unsafe { memmap2::Mmap::map(&gate_file)? };
    let gate_floats = crate::config::dtype::decode_floats(&gate_mmap, config.dtype);
    let bpf = crate::config::dtype::bytes_per_float(config.dtype);
    for info in &config.layers {
        let float_offset = info.offset as usize / bpf;
        let float_count = info.num_features * config.hidden_size;
        if float_offset + float_count <= gate_floats.len() {
            let gate_data = &gate_floats[float_offset..float_offset + float_count];
            let gate_matrix =
                Array2::from_shape_vec((info.num_features, config.hidden_size), gate_data.to_vec())
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
            tensors.insert(arch.ffn_gate_key(info.layer), gate_matrix.into_shared());
        }
    }

    callbacks.on_file_done("model_weights", entries.len(), 0.0);

    let cfg = arch.config();
    let embed = embed.into_shared();
    let lm_head = lm_head_loaded.unwrap_or_else(|| embed.clone());

    Ok(ModelWeights {
        tensors,
        vectors,
        embed,
        lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: config.vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Find the tokenizer path near a model or vindex directory.
pub fn find_tokenizer_path(dir: &Path) -> Option<std::path::PathBuf> {
    let p = dir.join("tokenizer.json");
    if p.exists() {
        return Some(p);
    }
    if let Some(parent) = dir.parent() {
        let p = parent.join("tokenizer.json");
        if p.exists() {
            return Some(p);
        }
    }
    None
}
