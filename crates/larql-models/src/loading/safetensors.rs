//! Model loading — safetensors, MLX, GGUF → ModelWeights.
//!
//! Handles dtype conversion (f16, bf16 → f32), HuggingFace cache resolution,
//! and architecture detection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::detect::{detect_architecture_validated, ModelError};
use crate::weights::{ModelWeights, PACKED_EXPERTS_DOWN_PROJ, PACKED_EXPERTS_GATE_UP_PROJ};

const SAFETENSORS_EXT: &str = "safetensors";
const GGUF_EXT: &str = "gguf";
const CONFIG_JSON: &str = "config.json";
const WEIGHTS_DIR: &str = "weights";
const MODEL_PREFIX: &str = "models--";
const SNAPSHOTS_DIR: &str = "snapshots";

const MXFP4_GATE_UP_BLOCKS_SUFFIX: &str = ".gate_up_proj_blocks";
const MXFP4_BLOCKS_SUFFIX: &str = "_blocks";
const MXFP4_SCALES_SUFFIX: &str = "_scales";
const MXFP4_GATE_UP_BLOCKS: &str = "gate_up_proj_blocks";
const MXFP4_EXPERTS_GATE_UP_BLOCKS: &str = "experts.gate_up_proj_blocks";
const MXFP4_DOWN_BLOCKS: &str = "down_proj_blocks";
const MXFP4_DOWN_SCALES: &str = "down_proj_scales";
const MXFP4_ROUTER_WEIGHT: &str = "router.weight";

const BLOCK_SPARSE_EXPERTS_PREFIX: &str = "block_sparse_moe.experts";
const BLOCK_SPARSE_ROUTER_WEIGHT: &str = "block_sparse_moe.gate.weight";
const MIXTRAL_GATE_PROJ: &str = "w1";
const MIXTRAL_DOWN_PROJ: &str = "w2";
const MIXTRAL_UP_PROJ: &str = "w3";
const FP8_SCALE_INV_SUFFIX: &str = "_scale_inv";
const KIMI_FP8_BLOCK_SIZE: usize = 128;

/// Returns true when `key` names a FFN weight tensor (gate/up/down projection
/// or packed expert block). Used by `load_model_dir_walk_only` to skip
/// decoding these entirely — critical for large models where decoding them
/// into f32 heap would blow RAM before they can be dropped.
pub fn is_ffn_tensor(key: &str) -> bool {
    crate::weights::FFN_TENSOR_PATTERNS
        .iter()
        .any(|p| key.contains(p))
}

/// Load model weights from a directory or file, never reading FFN tensors.
///
/// Equivalent to `load_model_dir` + `drop_ffn_weights` but without the heap
/// spike: FFN tensors are skipped at deserialisation time, so peak RSS
/// tracks only the retained (attention / embed / lm_head / norms) weights.
/// Use this with vindex-backed FFN (walk-only inference).
pub fn load_model_dir_walk_only(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered(path, is_ffn_tensor)
}

/// Validated variant of [`load_model_dir_walk_only`].
pub fn load_model_dir_walk_only_validated(
    path: impl AsRef<Path>,
) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, is_ffn_tensor, true)
}

/// Load model weights from a directory or file.
///
/// Auto-detects the format:
/// - Directory with `.safetensors` files → safetensors loading
/// - Directory with `.gguf` file → GGUF loading (dequantized to f32)
/// - Single `.gguf` file → GGUF loading
///
/// Detects architecture from config.json (safetensors) or GGUF metadata.
pub fn load_model_dir(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered(path, |_| false)
}

/// Validated variant of [`load_model_dir`].
///
/// Architecture detection stays permissive in `load_model_dir`; use this when
/// inference or extraction should fail fast on inconsistent config values.
pub fn load_model_dir_validated(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, |_| false, true)
}

/// Same as `load_model_dir` but `skip_key` returning true causes a tensor to
/// be dropped before decode — its bytes are never read from the mmap and no
/// f32 heap allocation occurs for it.
pub fn load_model_dir_filtered(
    path: impl AsRef<Path>,
    skip_key: impl Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, skip_key, false)
}

/// Validated variant of [`load_model_dir_filtered`].
pub fn load_model_dir_filtered_validated(
    path: impl AsRef<Path>,
    skip_key: impl Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered_with_validation(path, skip_key, true)
}

fn load_model_dir_filtered_with_validation(
    path: impl AsRef<Path>,
    skip_key: impl Fn(&str) -> bool,
    validate_config: bool,
) -> Result<ModelWeights, ModelError> {
    let path = path.as_ref();

    // Single GGUF file
    if path.is_file() {
        if path.extension().is_some_and(|ext| ext == GGUF_EXT) {
            return super::gguf::load_gguf_filtered_with_validation(
                path,
                &skip_key,
                validate_config,
            );
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
        .filter(|p| p.extension().is_some_and(|ext| ext == GGUF_EXT))
        .collect();

    if !gguf_files.is_empty() {
        // Use the first (or largest) GGUF file
        let gguf_path = gguf_files
            .into_iter()
            .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .unwrap();
        return super::gguf::load_gguf_filtered_with_validation(
            &gguf_path,
            &skip_key,
            validate_config,
        );
    }

    // Safetensors loading (also handles MLX format — same files, sometimes in weights/ subdir)
    let arch = if validate_config {
        detect_architecture_validated(path)?
    } else {
        crate::detect_architecture(path)?
    };
    let prefixes = arch.key_prefixes_to_strip();

    let mut st_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == SAFETENSORS_EXT))
        .collect();

    // MLX models sometimes put weights in a weights/ subdirectory
    if st_files.is_empty() {
        let weights_dir = path.join(WEIGHTS_DIR);
        if weights_dir.is_dir() {
            st_files = std::fs::read_dir(&weights_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == SAFETENSORS_EXT))
                .collect();
        }
    }
    st_files.sort();

    if st_files.is_empty() {
        return Err(ModelError::NoSafetensors(path.to_path_buf()));
    }

    let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();
    let mut packed_mmaps: HashMap<String, memmap2::Mmap> = HashMap::new();
    let mut packed_byte_ranges: HashMap<String, (String, usize, usize)> = HashMap::new();
    let mut skipped_tensors: Vec<(String, String)> = Vec::new();

    let expert_format = arch.expert_format();
    let is_packed_mxfp4 = expert_format == crate::ExpertFormat::PackedMxfp4;
    let is_packed_bf16 = expert_format == crate::ExpertFormat::PackedBF16;

    // Keys that must be preserved as raw bytes rather than converted to f32.
    // For PackedBF16 (Gemma 4 26B A4B): experts.gate_up_proj and experts.down_proj
    // are 3D tensors [num_experts, out_dim, in_dim] in BF16. Converting them to f32
    // would double their memory footprint; the compute path dequantizes per-expert on demand.
    let should_keep_raw = |key: &str| -> bool {
        is_packed_bf16
            && (key.contains(PACKED_EXPERTS_GATE_UP_PROJ) || key.contains(PACKED_EXPERTS_DOWN_PROJ))
    };

    for st_path in &st_files {
        let file = std::fs::File::open(st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let (header_len, metadata) = safetensors::SafeTensors::read_metadata(&mmap)
            .map_err(|e| ModelError::Parse(e.to_string()))?;
        let data_base = header_len
            .checked_add(8)
            .ok_or_else(|| ModelError::Parse("safetensors data offset overflow".to_string()))?;
        let file_key = st_path.to_string_lossy().into_owned();
        let mut retain_mmap = false;

        {
            let st = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| ModelError::Parse(e.to_string()))?;

            // Check for MXFP4 packed expert tensors (GPT-OSS format)
            let tensor_names: Vec<String> = st.names().iter().map(|n| n.to_string()).collect();

            if is_packed_mxfp4 {
                // MXFP4 path: dequantize packed expert blocks+scales into per-expert tensors
                load_mxfp4_expert_tensors(&st, &tensor_names, prefixes, &skip_key, &mut tensors)?;
                // Also load normal float tensors (router, norms, attn, embeddings)
                for (name, view) in st.tensors() {
                    let key = normalize_key(&name, prefixes);
                    let shape = view.shape();
                    if name.ends_with(MXFP4_BLOCKS_SUFFIX) || name.ends_with(MXFP4_SCALES_SUFFIX) {
                        continue;
                    }
                    if skip_key(&key) {
                        continue;
                    }
                    if is_fp8_scale_tensor(&key) {
                        continue;
                    }
                    let data = match tensor_to_f32_maybe_scaled(&st, &name, &view) {
                        Ok(d) => d,
                        Err(ModelError::UnsupportedDtype(ref dtype)) => {
                            skipped_tensors.push((key, dtype.clone()));
                            continue;
                        }
                        Err(e) => return Err(e),
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
                for (name, view) in st.tensors() {
                    let key = normalize_key(&name, prefixes);
                    let shape = view.shape();
                    if skip_key(&key) {
                        continue;
                    }

                    // PackedBF16 expert tensors: preserve mmap byte ranges,
                    // skip f32 conversion, and avoid cloning multi-GB tensors.
                    if should_keep_raw(&key) {
                        let info = metadata.info(&name).ok_or_else(|| {
                            ModelError::Parse(format!("missing safetensors metadata for {name}"))
                        })?;
                        let offset =
                            data_base.checked_add(info.data_offsets.0).ok_or_else(|| {
                                ModelError::Parse(format!("tensor {name}: data offset overflow"))
                            })?;
                        let length = info
                            .data_offsets
                            .1
                            .checked_sub(info.data_offsets.0)
                            .ok_or_else(|| {
                                ModelError::Parse(format!("tensor {name}: invalid data offsets"))
                            })?;
                        packed_byte_ranges.insert(key, (file_key.clone(), offset, length));
                        retain_mmap = true;
                        continue;
                    }

                    if is_fp8_scale_tensor(&key) {
                        continue;
                    }
                    let data = match tensor_to_f32_maybe_scaled(&st, &name, &view) {
                        Ok(d) => d,
                        Err(ModelError::UnsupportedDtype(ref dtype)) => {
                            skipped_tensors.push((key, dtype.clone()));
                            continue;
                        }
                        Err(e) => return Err(e),
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

        if retain_mmap {
            packed_mmaps.insert(file_key, mmap);
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
        raw_bytes,
        skipped_tensors,
        packed_mmaps,
        packed_byte_ranges,
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

/// Return the HuggingFace hub cache directory, respecting env-var overrides.
///
/// Priority (matches Python `huggingface_hub`):
/// 1. `HF_HUB_CACHE` — exact cache dir
/// 2. `HF_HOME` — HF home; hub cache = `$HF_HOME/hub`
/// 3. `HOME` (Unix) / `USERPROFILE` (Windows) — `~/.cache/huggingface/hub`
fn hf_hub_cache() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join(".cache").join("huggingface").join("hub")
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

    // Try HuggingFace cache — resolve location using the same env-var priority
    // as the Python huggingface_hub library: HF_HUB_CACHE > HF_HOME > home dir.
    let cache_name = format!("{MODEL_PREFIX}{}", model.replace('/', "--"));
    let hf_cache = hf_hub_cache().join(&cache_name).join(SNAPSHOTS_DIR);

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
                        rd.flatten().any(|e| {
                            e.path()
                                .extension()
                                .is_some_and(|ext| ext == SAFETENSORS_EXT)
                        })
                    })
                    .unwrap_or(false);
                if has_st {
                    return Ok(p);
                }
                // Fallback: any snapshot with config.json
                if p.join(CONFIG_JSON).exists() {
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

/// Load GPT-OSS MXFP4 packed expert tensors from a safetensors file into the
/// weights map, using per-expert Mixtral-style key names.
///
/// GPT-OSS stores experts as:
///   layers.{L}.mlp.experts.gate_up_proj_blocks: [experts, 2*hidden, groups, 16] U8
///   layers.{L}.mlp.experts.gate_up_proj_scales: [experts, 2*hidden, groups] U8
///   layers.{L}.mlp.experts.down_proj_blocks: [experts, hidden, groups, 16] U8
///   layers.{L}.mlp.experts.down_proj_scales: [experts, hidden, groups] U8
///
/// Dequantization and gate/up splitting are handled by `quant::mxfp4`.
/// Output keys follow Mixtral conventions:
///   layers.{L}.block_sparse_moe.experts.{E}.w1.weight (gate)
///   layers.{L}.block_sparse_moe.experts.{E}.w3.weight (up)
///   layers.{L}.block_sparse_moe.experts.{E}.w2.weight (down)
fn load_mxfp4_expert_tensors(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    skip_key: &impl Fn(&str) -> bool,
    tensors: &mut HashMap<String, crate::WeightArray>,
) -> Result<(), ModelError> {
    for name in tensor_names {
        if !name.ends_with(MXFP4_GATE_UP_BLOCKS_SUFFIX) {
            continue;
        }

        let scales_name = name.replace(MXFP4_BLOCKS_SUFFIX, MXFP4_SCALES_SUFFIX);
        let down_blocks_name = name.replace(MXFP4_GATE_UP_BLOCKS, MXFP4_DOWN_BLOCKS);
        let down_scales_name = name.replace(MXFP4_GATE_UP_BLOCKS, MXFP4_DOWN_SCALES);

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
        let out_features = shape[1]; // = 2 * hidden (gate + up fused)
        let groups = shape[2];
        let in_features = groups * 32;
        let half = out_features / 2;

        let base_key = normalize_key(name, prefixes);
        let layer_prefix = base_key.split(".mlp.").next().unwrap_or("");
        let should_load_gate_up = (0..num_experts).any(|e| {
            !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_GATE_PROJ))
                || !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_UP_PROJ))
        });

        // Dequantize and split fused gate_up → separate gate (w1) and up (w3).
        if should_load_gate_up {
            let (gate_experts, up_experts) = crate::quant::mxfp4::split_gate_up_experts(
                blocks_view.data(),
                scales_view.data(),
                num_experts,
                out_features,
                groups,
            )?;

            for (e, (gate_data, up_data)) in gate_experts.into_iter().zip(up_experts).enumerate() {
                let gate_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_GATE_PROJ);
                if !skip_key(&gate_key) {
                    tensors.insert(
                        gate_key,
                        Array2::from_shape_vec((half, in_features), gate_data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
                let up_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_UP_PROJ);
                if !skip_key(&up_key) {
                    tensors.insert(
                        up_key,
                        Array2::from_shape_vec((half, in_features), up_data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
            }
        }

        // Dequantize down projection.
        if let (Ok(db), Ok(ds)) = (st.tensor(&down_blocks_name), st.tensor(&down_scales_name)) {
            let down_shape = db.shape();
            if down_shape.len() == 4 {
                let down_out = down_shape[1];
                let down_groups = down_shape[2];
                let down_in = down_groups * 32;
                let should_load_down = (0..num_experts)
                    .any(|e| !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_DOWN_PROJ)));
                if should_load_down {
                    let down_experts = crate::quant::mxfp4::dequantize_all_experts(
                        db.data(),
                        ds.data(),
                        num_experts,
                        down_out,
                        down_groups,
                    )?;
                    for (e, data) in down_experts.into_iter().enumerate() {
                        let down_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_DOWN_PROJ);
                        if !skip_key(&down_key) {
                            tensors.insert(
                                down_key,
                                Array2::from_shape_vec((down_out, down_in), data)
                                    .map_err(|e| ModelError::Parse(e.to_string()))?
                                    .into_shared(),
                            );
                        }
                    }
                }
            }
        }

        // Remap router: mlp.router.weight → block_sparse_moe.gate.weight
        let router_name = name.replace(MXFP4_EXPERTS_GATE_UP_BLOCKS, MXFP4_ROUTER_WEIGHT);
        if let Ok(router_view) = st.tensor(&router_name) {
            if let Ok(data) = tensor_to_f32(&router_view) {
                let s = router_view.shape();
                if s.len() == 2 {
                    let router_key = format!("{layer_prefix}.{BLOCK_SPARSE_ROUTER_WEIGHT}");
                    if !skip_key(&router_key) {
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
    }

    Ok(())
}

fn mxfp4_expert_key(layer_prefix: &str, expert_id: usize, projection: &str) -> String {
    format!("{layer_prefix}.{BLOCK_SPARSE_EXPERTS_PREFIX}.{expert_id}.{projection}.weight")
}

pub(crate) fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
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

fn tensor_to_f32_maybe_scaled(
    st: &safetensors::SafeTensors<'_>,
    name: &str,
    view: &safetensors::tensor::TensorView<'_>,
) -> Result<Vec<f32>, ModelError> {
    if view.dtype() == safetensors::Dtype::F8_E4M3 {
        let scale_name = format!("{name}{FP8_SCALE_INV_SUFFIX}");
        let scale_view = st.tensor(&scale_name).map_err(|_| {
            ModelError::MissingTensor(format!(
                "{scale_name} (required to dequantize FP8 tensor {name})"
            ))
        })?;
        let scales = tensor_to_f32(&scale_view)?;
        return crate::quant::fp8::decode_e4m3_scaled_blocks(
            view.data(),
            view.shape(),
            &scales,
            scale_view.shape(),
            KIMI_FP8_BLOCK_SIZE,
        )
        .map_err(ModelError::Parse);
    }

    tensor_to_f32(view)
}

fn is_fp8_scale_tensor(key: &str) -> bool {
    key.ends_with(FP8_SCALE_INV_SUFFIX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::TempDir;

    // Tests that mutate HOME must not run concurrently.
    static HOME_LOCK: Mutex<()> = Mutex::new(());

    // ── is_ffn_tensor ──────────────────────────────────────────────────────

    #[test]
    fn is_ffn_tensor_gate_proj() {
        assert!(is_ffn_tensor("layers.0.mlp.gate_proj.weight"));
        assert!(is_ffn_tensor("layers.31.mlp.up_proj.weight"));
        assert!(is_ffn_tensor("layers.0.mlp.down_proj.weight"));
    }

    #[test]
    fn is_ffn_tensor_ffn_variants() {
        assert!(is_ffn_tensor("layers.0.ffn_gate"));
        assert!(is_ffn_tensor("layers.0.ffn_up"));
        assert!(is_ffn_tensor("layers.0.ffn_down"));
    }

    #[test]
    fn is_ffn_tensor_moe_experts() {
        assert!(is_ffn_tensor("layers.0.mlp.experts.0.gate_proj.weight"));
        assert!(is_ffn_tensor(
            "layers.0.block_sparse_moe.experts.1.w1.weight"
        ));
    }

    #[test]
    fn is_ffn_tensor_packed_keys() {
        assert!(is_ffn_tensor("packed_gate_up_blocks"));
        assert!(is_ffn_tensor("packed_down_blocks"));
    }

    #[test]
    fn is_ffn_tensor_rejects_non_ffn() {
        assert!(!is_ffn_tensor("layers.0.self_attn.q_proj.weight"));
        assert!(!is_ffn_tensor("layers.0.input_layernorm.weight"));
        assert!(!is_ffn_tensor("embed_tokens.weight"));
        assert!(!is_ffn_tensor("norm.weight"));
        assert!(!is_ffn_tensor("lm_head.weight"));
    }

    #[test]
    fn is_ffn_tensor_empty_key() {
        assert!(!is_ffn_tensor(""));
    }

    // ── normalize_key ──────────────────────────────────────────────────────

    #[test]
    fn normalize_key_strips_first_matching_prefix() {
        let prefixes = &["model.language_model.", "model."];
        // Longer prefix matches first
        assert_eq!(
            normalize_key(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                prefixes
            ),
            "layers.0.mlp.gate_proj.weight"
        );
    }

    #[test]
    fn normalize_key_falls_through_to_shorter_prefix() {
        let prefixes = &["model.language_model.", "model."];
        assert_eq!(normalize_key("model.norm.weight", prefixes), "norm.weight");
    }

    #[test]
    fn normalize_key_no_match_passthrough() {
        let prefixes = &["model."];
        assert_eq!(
            normalize_key("embed_tokens.weight", prefixes),
            "embed_tokens.weight"
        );
    }

    #[test]
    fn normalize_key_empty_prefixes() {
        assert_eq!(normalize_key("layers.0.weight", &[]), "layers.0.weight");
    }

    // ── resolve_model_path ─────────────────────────────────────────────────

    #[test]
    fn resolve_model_path_existing_dir() {
        let dir = TempDir::new().unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(result, dir.path());
    }

    #[test]
    fn resolve_model_path_existing_gguf_file() {
        let dir = TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        fs::write(&gguf, b"").unwrap();
        let result = resolve_model_path(gguf.to_str().unwrap()).unwrap();
        assert_eq!(result, gguf);
    }

    #[test]
    fn resolve_model_path_nonexistent_returns_error() {
        // Use a temp dir that we immediately drop, so the path is guaranteed
        // not to exist on any OS — no hardcoded Unix-style paths.
        let dir = TempDir::new().unwrap();
        let gone = dir.path().join("subdir_that_was_never_created");
        drop(dir);
        let result = resolve_model_path(gone.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn resolve_model_path_hf_cache_with_safetensors() {
        let _lock = HOME_LOCK.lock().unwrap();
        let home = TempDir::new().unwrap();
        let snapshot = home
            .path()
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("models--org--name")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model.safetensors"), b"").unwrap();
        std::env::set_var("HOME", home.path().to_str().unwrap());
        let result = resolve_model_path("org/name").unwrap();
        std::env::remove_var("HOME");
        assert_eq!(result, snapshot);
    }

    #[test]
    fn resolve_model_path_hf_cache_fallback_config_json() {
        let _lock = HOME_LOCK.lock().unwrap();
        let home = TempDir::new().unwrap();
        let snapshot = home
            .path()
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("models--org--model")
            .join("snapshots")
            .join("def456");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("config.json"), b"{}").unwrap();
        std::env::set_var("HOME", home.path().to_str().unwrap());
        let result = resolve_model_path("org/model").unwrap();
        std::env::remove_var("HOME");
        assert_eq!(result, snapshot);
    }
}
