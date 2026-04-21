//! Model loading — safetensors, MLX, GGUF → ModelWeights.
//!
//! Handles dtype conversion (f16, bf16 → f32), HuggingFace cache resolution,
//! and architecture detection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::detect::ModelError;
use crate::weights::ModelWeights;

/// Load model weights from a directory or file.
///
/// Auto-detects the format:
/// - Directory with `.safetensors` files → safetensors loading
/// - Directory with `.gguf` file → GGUF loading (dequantized to f32)
/// - Single `.gguf` file → GGUF loading
///
/// Detects architecture from config.json (safetensors) or GGUF metadata.
pub fn load_model_dir(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    let path = path.as_ref();

    // Single GGUF file
    if path.is_file() {
        if path.extension().is_some_and(|ext| ext == "gguf") {
            return super::gguf::load_gguf(path);
        }
        return Err(ModelError::NotADirectory(path.to_path_buf()));
    }

    if !path.is_dir() {
        return Err(ModelError::NotADirectory(path.to_path_buf()));
    }

    // Check for GGUF files in directory
    let gguf_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "gguf"))
        .collect();

    if !gguf_files.is_empty() {
        // Use the first (or largest) GGUF file
        let gguf_path = gguf_files
            .into_iter()
            .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .unwrap();
        return super::gguf::load_gguf(&gguf_path);
    }

    // Safetensors loading (also handles MLX format — same files, sometimes in weights/ subdir)
    let arch = crate::detect_architecture(path).map_err(|e| ModelError::Parse(e.to_string()))?;
    let prefixes = arch.key_prefixes_to_strip();
    let mlx_affine_group_size = mlx_affine_group_size_from_config(path);

    let mut st_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();

    // MLX models sometimes put weights in a weights/ subdirectory
    if st_files.is_empty() {
        let weights_dir = path.join("weights");
        if weights_dir.is_dir() {
            st_files = std::fs::read_dir(&weights_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
                .collect();
        }
    }
    st_files.sort();

    if st_files.is_empty() {
        return Err(ModelError::NoSafetensors(path.to_path_buf()));
    }

    let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for st_path in &st_files {
        let file = std::fs::File::open(st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| ModelError::Parse(e.to_string()))?;

        // Check for MXFP4 packed expert tensors (GPT-OSS format)
        let tensor_names: Vec<String> = st.names().iter().map(|n| n.to_string()).collect();
        let is_packed_mxfp4 = arch.expert_format() == crate::ExpertFormat::PackedMxfp4;

        if is_packed_mxfp4 {
            // MXFP4 path: dequantize packed expert blocks+scales into per-expert tensors
            dequantize_mxfp4_experts(&st, &tensor_names, prefixes, &mut tensors, &mut vectors)?;
            // Also load normal float tensors (router, norms, attn, embeddings)
            for (name, view) in st.tensors() {
                let key = normalize_key(&name, prefixes);
                let shape = view.shape();
                if name.ends_with("_blocks") || name.ends_with("_scales") {
                    continue;
                }
                let data = match tensor_to_f32(&view) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                match shape.len() {
                    2 => {
                        let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?;
                        tensors.insert(key, arr.into_shared());
                    }
                    1 => {
                        vectors.insert(key, data);
                    }
                    _ => {}
                }
            }
        } else {
            // Standard float path
            for (name, view) in st.tensors() {
                if name.ends_with(".scales") || name.ends_with(".biases") {
                    continue;
                }
                let key = normalize_key(&name, prefixes);
                let shape = view.shape();
                let data = match view.dtype() {
                    safetensors::Dtype::U32 if shape.len() == 2 => {
                        match dequantize_mlx_affine_tensor(&st, &name, &view, mlx_affine_group_size)
                        {
                            Ok((data, _, cols)) => {
                                // Replace packed width with dequantized width.
                                let mut dequant_shape = shape.to_vec();
                                dequant_shape[1] = cols;
                                match dequant_shape.len() {
                                    2 => {
                                        let arr = Array2::from_shape_vec(
                                            (dequant_shape[0], dequant_shape[1]),
                                            data,
                                        )
                                        .map_err(|e| ModelError::Parse(e.to_string()))?;
                                        tensors.insert(key, arr.into_shared());
                                        continue;
                                    }
                                    _ => continue,
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                    _ => match tensor_to_f32(&view) {
                        Ok(d) => d,
                        Err(_) => continue,
                    },
                };
                match shape.len() {
                    2 => {
                        let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?;
                        tensors.insert(key, arr.into_shared());
                    }
                    1 => {
                        vectors.insert(key, data);
                    }
                    // 0D scalar tensors (e.g., layer_scalar) → store as 1-element vector
                    0 => {
                        vectors.insert(key, data);
                    }
                    _ => {}
                }
            }
        }
    }

    let embed_key = arch.embed_key();
    let embed = tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();

    let lm_head = tensors
        .get("lm_head.weight")
        .cloned()
        .unwrap_or_else(|| embed.clone());

    let vocab_size = lm_head.shape()[0];
    let cfg = arch.config();

    Ok(ModelWeights {
        tensors,
        vectors,
        embed,
        lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Read the global MLX affine quantization group size from config.json.
pub fn mlx_affine_group_size_from_config(model_dir: &Path) -> Option<usize> {
    let config_path = model_dir.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;

    json.get("quantization")
        .or_else(|| json.get("quantization_config"))
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
}

/// Resolve a HuggingFace model ID or path to a local directory or GGUF file.
pub fn resolve_model_path(model: &str) -> Result<PathBuf, ModelError> {
    let path = PathBuf::from(model);
    if path.is_dir() {
        return Ok(path);
    }
    // Single GGUF file
    if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
        return Ok(path);
    }

    // Try HuggingFace cache
    let cache_name = format!("models--{}", model.replace('/', "--"));
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    let hf_cache = home.join(format!(".cache/huggingface/hub/{cache_name}/snapshots"));

    if hf_cache.is_dir() {
        // Find the snapshot that has actual model files (safetensors or config.json+weights)
        let mut best: Option<PathBuf> = None;
        if let Ok(entries) = std::fs::read_dir(&hf_cache) {
            for entry in entries.flatten() {
                let p = entry.path();
                if !p.is_dir() {
                    continue;
                }
                // Prefer snapshot with safetensors files
                let has_st = std::fs::read_dir(&p)
                    .ok()
                    .map(|rd| {
                        rd.flatten()
                            .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                    })
                    .unwrap_or(false);
                if has_st {
                    return Ok(p);
                }
                // Fallback: any snapshot with config.json
                if p.join("config.json").exists() {
                    best = Some(p);
                }
            }
        }
        if let Some(p) = best {
            return Ok(p);
        }
    }

    Err(ModelError::NotADirectory(path))
}

/// Normalize a tensor key by stripping known prefixes.
pub fn normalize_key_pub(key: &str, prefixes: &[&str]) -> String {
    normalize_key(key, prefixes)
}

/// Dequantize MXFP4 packed expert tensors into per-expert standard weight matrices.
///
/// GPT-OSS stores experts as:
///   layers.{L}.mlp.experts.gate_up_proj_blocks: [experts, 2*hidden, groups, 16] U8
///   layers.{L}.mlp.experts.gate_up_proj_scales: [experts, 2*hidden, groups] U8
///   layers.{L}.mlp.experts.down_proj_blocks: [experts, hidden, groups, 16] U8
///   layers.{L}.mlp.experts.down_proj_scales: [experts, hidden, groups] U8
///
/// We dequantize and split into per-expert Mixtral-style keys:
///   layers.{L}.block_sparse_moe.experts.{E}.w1.weight (gate)
///   layers.{L}.block_sparse_moe.experts.{E}.w3.weight (up)
///   layers.{L}.block_sparse_moe.experts.{E}.w2.weight (down)
fn dequantize_mxfp4_experts(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    tensors: &mut HashMap<String, crate::WeightArray>,
    _vectors: &mut HashMap<String, Vec<f32>>,
) -> Result<(), ModelError> {
    // Find all gate_up_proj_blocks tensors (one per layer)
    for name in tensor_names {
        if !name.ends_with(".gate_up_proj_blocks") {
            continue;
        }

        let scales_name = name.replace("_blocks", "_scales");
        let down_blocks_name = name.replace("gate_up_proj_blocks", "down_proj_blocks");
        let down_scales_name = name.replace("gate_up_proj_blocks", "down_proj_scales");

        // Get tensor views
        let blocks_view = st
            .tensor(name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 blocks: {e}")))?;
        let scales_view = st
            .tensor(&scales_name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 scales: {e}")))?;

        let shape = blocks_view.shape();
        if shape.len() != 4 {
            continue;
        }

        let num_experts = shape[0];
        let out_features = shape[1]; // 2*hidden for gate_up, hidden for down
        let groups = shape[2];
        let in_features = groups * 32; // 16 bytes * 2 nibbles per group
        let _hidden = in_features; // = hidden_size

        // Dequantize gate_up (fused: first half = gate, second half = up)
        let expert_data = crate::quant::mxfp4::dequantize_all_experts(
            blocks_view.data(),
            scales_view.data(),
            num_experts,
            out_features,
            groups,
        );

        // Extract layer number from key
        let base_key = normalize_key(name, prefixes);
        let layer_prefix = base_key.split(".mlp.").next().unwrap_or("");

        let half = out_features / 2; // gate vs up split

        for (e, data) in expert_data.iter().enumerate() {
            // Split fused gate_up: rows [0..half] = gate (w1), rows [half..] = up (w3)
            let gate_data: Vec<f32> = data[..half * in_features].to_vec();
            let up_data: Vec<f32> = data[half * in_features..].to_vec();

            let gate_key = format!("{layer_prefix}.block_sparse_moe.experts.{e}.w1.weight");
            let up_key = format!("{layer_prefix}.block_sparse_moe.experts.{e}.w3.weight");

            tensors.insert(
                gate_key,
                Array2::from_shape_vec((half, in_features), gate_data)
                    .map_err(|e| ModelError::Parse(e.to_string()))?
                    .into_shared(),
            );
            tensors.insert(
                up_key,
                Array2::from_shape_vec((half, in_features), up_data)
                    .map_err(|e| ModelError::Parse(e.to_string()))?
                    .into_shared(),
            );
        }

        // Dequantize down projection
        if let (Ok(db), Ok(ds)) = (st.tensor(&down_blocks_name), st.tensor(&down_scales_name)) {
            let down_shape = db.shape();
            if down_shape.len() == 4 {
                let down_out = down_shape[1];
                let down_groups = down_shape[2];
                let down_in = down_groups * 32;

                let down_experts = crate::quant::mxfp4::dequantize_all_experts(
                    db.data(),
                    ds.data(),
                    num_experts,
                    down_out,
                    down_groups,
                );

                for (e, data) in down_experts.iter().enumerate() {
                    let down_key = format!("{layer_prefix}.block_sparse_moe.experts.{e}.w2.weight");
                    tensors.insert(
                        down_key,
                        Array2::from_shape_vec((down_out, down_in), data.clone())
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
            }
        }

        // Also remap router: mlp.router.weight → block_sparse_moe.gate.weight
        let router_name = name.replace("experts.gate_up_proj_blocks", "router.weight");
        if let Ok(router_view) = st.tensor(&router_name) {
            if let Ok(data) = tensor_to_f32(&router_view) {
                let s = router_view.shape();
                if s.len() == 2 {
                    let router_key = format!("{layer_prefix}.block_sparse_moe.gate.weight");
                    tensors.insert(
                        router_key,
                        Array2::from_shape_vec((s[0], s[1]), data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
            }
        }
    }

    Ok(())
}

fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

fn dequantize_mlx_affine_tensor(
    st: &safetensors::SafeTensors,
    name: &str,
    view: &safetensors::tensor::TensorView<'_>,
    group_size: Option<usize>,
) -> Result<(Vec<f32>, usize, usize), ModelError> {
    let group_size = group_size.ok_or_else(|| {
        ModelError::Parse(format!(
            "missing MLX affine group_size for quantized tensor: {name}"
        ))
    })?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Err(ModelError::UnsupportedDtype(format!("{:?}", view.dtype())));
    }

    let stem = name
        .strip_suffix(".weight")
        .ok_or_else(|| ModelError::UnsupportedDtype(format!("{:?}", view.dtype())))?;
    let scales_name = format!("{stem}.scales");
    let biases_name = format!("{stem}.biases");

    let scales_view = st
        .tensor(&scales_name)
        .map_err(|e| ModelError::Parse(format!("MLX affine scales {scales_name}: {e}")))?;
    let biases_view = st.tensor(&biases_name).ok();

    let scales = tensor_to_f32(&scales_view)?;
    let biases = match biases_view {
        Some(biases_view) => Some(tensor_to_f32(&biases_view)?),
        None => None,
    };

    let (data, cols) = crate::quant::mlx_affine::dequantize_u32_matrix_bytes(
        view.data(),
        shape[0],
        shape[1],
        &scales,
        biases.as_deref(),
        group_size,
    )
    .map_err(ModelError::Parse)?;

    Ok((data, shape[0], cols))
}

fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, ModelError> {
    use crate::quant::half;
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        safetensors::Dtype::F16 => Ok(half::decode_f16(view.data())),
        safetensors::Dtype::BF16 => Ok(half::decode_bf16(view.data())),
        other => Err(ModelError::UnsupportedDtype(format!("{other:?}"))),
    }
}
