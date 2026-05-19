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
                    let data = match tensor_to_f32(&view) {
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
                // Per-expert MXFP4 detection (DeepSeek-V4 family): each expert's
                // gate/up/down is stored as a separate (.weight=I8, .scale=F8_E8M0)
                // pair, distinct from GPT-OSS's fused `gate_up_proj_blocks` +
                // `scales` tensors handled by `load_mxfp4_expert_tensors` above.
                // The detector returns the set of consumed tensor names so the
                // main loop below skips them. No-op for non-V4 architectures.
                let v4_dequantized_keys =
                    dequantize_per_expert_mxfp4(&st, &tensor_names, prefixes, &mut tensors)?;

                for (name, view) in st.tensors() {
                    let key = normalize_key(&name, prefixes);
                    let shape = view.shape();
                    if skip_key(&key) {
                        continue;
                    }

                    // Skip tensors consumed by the V4 dequantizer (both .weight
                    // and the companion .scale).
                    if v4_dequantized_keys.contains(&name) {
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

                    let data = match tensor_to_f32(&view) {
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
    let position_embed = arch
        .position_embed_key()
        .and_then(|key| tensors.get(key).cloned());

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
        position_embed,
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

/// Per-expert MXFP4 dequantization (DeepSeek-V4 family).
///
/// DeepSeek-V4 stores expert weights one (.weight, .scale) pair per
/// (expert, projection) — `layers.X.ffn.experts.E.w1.weight` (I8 packed FP4) +
/// `layers.X.ffn.experts.E.w1.scale` (F8_E8M0 scales), ditto w2/w3. This is
/// distinct from GPT-OSS's fused `experts.gate_up_proj_blocks` layout that
/// `load_mxfp4_expert_tensors` handles.
///
/// Detects the format by scanning for `*.experts.<digit>.w[123].weight` tensors
/// with `I8` dtype. For each match, looks up the companion `.scale` (`F8_E8M0`)
/// and dequantizes via `quant::mxfp4::dequantize_expert`.
///
/// Returns the set of tensor names that were consumed (both `.weight` and
/// `.scale`) so the main loading loop can skip them.
fn dequantize_per_expert_mxfp4(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    tensors: &mut HashMap<String, crate::WeightArray>,
) -> Result<std::collections::HashSet<String>, ModelError> {
    use std::collections::HashSet;
    let mut consumed: HashSet<String> = HashSet::new();

    // Match V4-style per-expert weights: any tensor name containing
    // ".experts.<int>.w<1|2|3>.weight" — broad enough to catch both the
    // full `model.layers.X.ffn.experts.E.wY.weight` (HF default) and any
    // shortened variant (`layers.X.ffn.experts.E.wY.weight`).
    let is_v4_expert_weight = |name: &str| -> bool {
        if !name.ends_with(".w1.weight")
            && !name.ends_with(".w2.weight")
            && !name.ends_with(".w3.weight")
        {
            return false;
        }
        // Must have ".experts.<digit>" before the .wN.weight suffix
        if let Some(idx) = name.rfind(".experts.") {
            let after = &name[idx + ".experts.".len()..];
            if let Some(dot) = after.find('.') {
                return after[..dot].chars().all(|c| c.is_ascii_digit());
            }
        }
        false
    };

    for name in tensor_names {
        if !is_v4_expert_weight(name) {
            continue;
        }

        let weight_view = match st.tensor(name) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // V4 packed FP4 weights are stored as I8 (signed) per the safetensors header.
        if weight_view.dtype() != safetensors::Dtype::I8 {
            continue;
        }

        let scale_name = name.replacen(".weight", ".scale", 1);
        let scale_view = match st.tensor(&scale_name) {
            Ok(v) => v,
            Err(_) => continue, // No scale companion → not MXFP4, leave to main loop.
        };
        if scale_view.dtype() != safetensors::Dtype::F8_E8M0 {
            continue;
        }

        // Shape sanity. weight: (out_features, packed_in/2). scale: (out_features, groups).
        let w_shape = weight_view.shape();
        let s_shape = scale_view.shape();
        if w_shape.len() != 2 || s_shape.len() != 2 {
            continue;
        }
        if w_shape[0] != s_shape[0] {
            continue;
        }

        let out_features = w_shape[0];
        let groups = s_shape[1];
        let in_features = groups * 32;

        // Assert layout consistency: weight cols × 2 (nibbles per byte) == groups × 32.
        if w_shape[1] * 2 != in_features {
            continue;
        }

        let unpacked = crate::quant::mxfp4::dequantize_expert(
            weight_view.data(),
            scale_view.data(),
            out_features,
            groups,
        )?;

        let key = normalize_key(name, prefixes);
        let arr = Array2::from_shape_vec((out_features, in_features), unpacked)
            .map_err(|e| ModelError::Parse(e.to_string()))?;
        tensors.insert(key, arr.into_shared());

        consumed.insert(name.clone());
        consumed.insert(scale_name);
    }

    Ok(consumed)
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

        // ── FP8 / I8 — used by DeepSeek-V4 (MXFP4 experts), GPT-OSS, etc. ──
        // Decoded bit-pattern → f32 in isolation. MXFP4 unpacking proper (where
        // an I8 packed-nibble weight is paired with its F8_E8M0 scale companion)
        // happens at the FFN tensor loading layer — `tensor_to_f32` sees one
        // tensor at a time and can't look at companions.
        safetensors::Dtype::F8_E4M3 => Ok(decode_f8_e4m3(view.data())),
        safetensors::Dtype::F8_E5M2 => Ok(decode_f8_e5m2(view.data())),
        safetensors::Dtype::F8_E8M0 => Ok(decode_f8_e8m0(view.data())),
        safetensors::Dtype::I8 => Ok(view.data().iter().map(|&b| (b as i8) as f32).collect()),

        other => Err(ModelError::UnsupportedDtype(format!("{other:?}"))),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FP8 / E8M0 decoders — bit-pattern → f32. Operate per-byte on the raw view.
// Standard Open Compute Project encodings; verified against the F8_E*M* table
// in the safetensors crate (≥ 0.7).
// ────────────────────────────────────────────────────────────────────────────

/// FP8 E4M3 (FN, finite-only): 1 sign + 4 exponent + 3 mantissa bits, bias 7.
/// NaN encoded at 0x7F / 0xFF (Open Compute convention).
#[inline]
fn decode_f8_e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            let sign = (b >> 7) & 1;
            let exp_bits = (b >> 3) & 0x0F;
            let mant_bits = b & 0x07;
            let v = if exp_bits == 0 {
                (mant_bits as f32) / 8.0 * 2f32.powi(1 - 7)
            } else if exp_bits == 0x0F && mant_bits == 0x07 {
                f32::NAN
            } else {
                let m = 1.0 + (mant_bits as f32) / 8.0;
                m * 2f32.powi(exp_bits as i32 - 7)
            };
            if sign == 1 {
                -v
            } else {
                v
            }
        })
        .collect()
}

/// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits, bias 15.
#[inline]
fn decode_f8_e5m2(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            let sign = (b >> 7) & 1;
            let exp_bits = (b >> 2) & 0x1F;
            let mant_bits = b & 0x03;
            let v = if exp_bits == 0 {
                (mant_bits as f32) / 4.0 * 2f32.powi(1 - 15)
            } else if exp_bits == 0x1F {
                if mant_bits == 0 {
                    f32::INFINITY
                } else {
                    f32::NAN
                }
            } else {
                let m = 1.0 + (mant_bits as f32) / 4.0;
                m * 2f32.powi(exp_bits as i32 - 15)
            };
            if sign == 1 {
                -v
            } else {
                v
            }
        })
        .collect()
}

/// FP8 E8M0 (Open Compute Microscaling MX format scale): 8 exponent bits, no
/// sign or mantissa. Value = 2^(byte - 127). Byte 0xFF reserved as NaN.
#[inline]
fn decode_f8_e8m0(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .map(|&b| {
            if b == 0xFF {
                f32::NAN
            } else {
                2f32.powi(b as i32 - 127)
            }
        })
        .collect()
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
    fn hf_hub_cache_honours_hf_hub_cache_env() {
        let _lock = HOME_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let prior_hub = std::env::var_os("HF_HUB_CACHE");
        let prior_home_var = std::env::var_os("HF_HOME");
        std::env::set_var("HF_HUB_CACHE", tmp.path());
        // HF_HUB_CACHE wins over HF_HOME — clear it to avoid ambiguity.
        std::env::remove_var("HF_HOME");
        let got = hf_hub_cache();
        match prior_hub {
            Some(v) => std::env::set_var("HF_HUB_CACHE", v),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }
        if let Some(v) = prior_home_var {
            std::env::set_var("HF_HOME", v);
        }
        assert_eq!(got, tmp.path());
    }

    #[test]
    fn hf_hub_cache_honours_hf_home_env_when_hub_cache_unset() {
        let _lock = HOME_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let prior_hub = std::env::var_os("HF_HUB_CACHE");
        let prior_home_var = std::env::var_os("HF_HOME");
        // HF_HUB_CACHE must be unset for the HF_HOME branch to fire.
        std::env::remove_var("HF_HUB_CACHE");
        std::env::set_var("HF_HOME", tmp.path());
        let got = hf_hub_cache();
        match prior_hub {
            Some(v) => std::env::set_var("HF_HUB_CACHE", v),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }
        match prior_home_var {
            Some(v) => std::env::set_var("HF_HOME", v),
            None => std::env::remove_var("HF_HOME"),
        }
        assert_eq!(got, tmp.path().join("hub"));
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

    // ── FP8 decoders (per-byte bit-pattern → f32) ─────────────────────────

    #[test]
    fn decode_f8_e4m3_zero_byte_is_zero() {
        // sign=0, exp=0, mant=0 → 0.0 (subnormal at +0).
        assert_eq!(decode_f8_e4m3(&[0x00]), vec![0.0]);
    }

    #[test]
    fn decode_f8_e4m3_negative_zero() {
        // sign=1, exp=0, mant=0 → -0.0.
        let v = decode_f8_e4m3(&[0x80])[0];
        assert_eq!(v, 0.0); // -0.0 == 0.0 in IEEE comparison
        assert!(v.is_sign_negative());
    }

    #[test]
    fn decode_f8_e4m3_one_is_unit() {
        // 0x38 = 0011 1000 = sign=0, exp=0b0111=7, mant=0 → 1.0 × 2^(7-7) = 1.0.
        assert_eq!(decode_f8_e4m3(&[0x38]), vec![1.0]);
    }

    #[test]
    fn decode_f8_e4m3_nan_byte_is_nan() {
        // 0x7F = sign=0, exp=0x0F, mant=0x07 → NaN by OCP convention.
        assert!(decode_f8_e4m3(&[0x7F])[0].is_nan());
        // 0xFF = sign=1, exp=0x0F, mant=0x07 → NaN.
        assert!(decode_f8_e4m3(&[0xFF])[0].is_nan());
    }

    #[test]
    fn decode_f8_e4m3_subnormal_path() {
        // exp=0, mant=4 → (4/8) × 2^(1-7) = 0.5 × 2^-6 = 1/128.
        let v = decode_f8_e4m3(&[0x04])[0];
        assert!((v - (1.0 / 128.0)).abs() < 1e-7);
    }

    #[test]
    fn decode_f8_e4m3_handles_multiple_bytes() {
        // Vec output length = input length.
        let out = decode_f8_e4m3(&[0x00, 0x38, 0x80]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
    }

    #[test]
    fn decode_f8_e5m2_zero_byte_is_zero() {
        assert_eq!(decode_f8_e5m2(&[0x00]), vec![0.0]);
    }

    #[test]
    fn decode_f8_e5m2_one_is_unit() {
        // bias=15 → exp byte for 1.0 is 15 → mant bits = 0 → 0b01111000 = 0x3C.
        assert_eq!(decode_f8_e5m2(&[0x3C]), vec![1.0]);
    }

    #[test]
    fn decode_f8_e5m2_inf_byte() {
        // exp=0x1F, mant=0 → ±∞.
        assert!(decode_f8_e5m2(&[0x7C])[0].is_infinite());
        let neg = decode_f8_e5m2(&[0xFC])[0];
        assert!(neg.is_infinite() && neg.is_sign_negative());
    }

    #[test]
    fn decode_f8_e5m2_nan_byte() {
        // exp=0x1F, mant!=0 → NaN.
        assert!(decode_f8_e5m2(&[0x7D])[0].is_nan());
    }

    #[test]
    fn decode_f8_e5m2_subnormal_path() {
        // exp=0, mant=2 → (2/4) × 2^(1-15) = 0.5 × 2^-14.
        let v = decode_f8_e5m2(&[0x02])[0];
        assert!((v - (0.5_f32 * (-14_i32).exp2_f())).abs() < 1e-10);
    }

    // f32 doesn't have an exp2 method on i32 directly; small helper.
    trait ExpHelper {
        fn exp2_f(self) -> f32;
    }
    impl ExpHelper for i32 {
        fn exp2_f(self) -> f32 {
            2f32.powi(self)
        }
    }

    #[test]
    fn decode_f8_e8m0_one_is_byte_127() {
        // E8M0 has no sign or mantissa: value = 2^(byte - 127). byte=127 → 1.0.
        assert_eq!(decode_f8_e8m0(&[127]), vec![1.0]);
    }

    #[test]
    fn decode_f8_e8m0_byte_128_is_two() {
        // 2^(128-127) = 2.
        assert_eq!(decode_f8_e8m0(&[128]), vec![2.0]);
    }

    #[test]
    fn decode_f8_e8m0_byte_zero_is_smallest() {
        // 2^(-127) is a very small positive — never NaN.
        let v = decode_f8_e8m0(&[0])[0];
        assert!(v > 0.0);
        assert!(!v.is_nan());
    }

    #[test]
    fn decode_f8_e8m0_byte_ff_is_nan() {
        assert!(decode_f8_e8m0(&[0xFF])[0].is_nan());
    }

    // ── tensor_to_f32 dispatcher ──────────────────────────────────────────

    fn make_view(
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> safetensors::tensor::TensorView<'static> {
        // Leak the bytes for 'static lifetime — fine in tests.
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        safetensors::tensor::TensorView::new(dtype, shape, leaked).unwrap()
    }

    #[test]
    fn tensor_to_f32_dispatches_f32() {
        let view = make_view(
            safetensors::Dtype::F32,
            vec![2],
            vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40],
        );
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_f8_e4m3() {
        let view = make_view(safetensors::Dtype::F8_E4M3, vec![1], vec![0x38]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_f8_e5m2() {
        let view = make_view(safetensors::Dtype::F8_E5M2, vec![1], vec![0x3C]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_f8_e8m0() {
        let view = make_view(safetensors::Dtype::F8_E8M0, vec![1], vec![127]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![1.0]);
    }

    #[test]
    fn tensor_to_f32_dispatches_i8() {
        // I8 dispatches via `(b as i8) as f32` — sign-extend.
        let view = make_view(safetensors::Dtype::I8, vec![3], vec![0, 1, 0xFF]);
        assert_eq!(tensor_to_f32(&view).unwrap(), vec![0.0, 1.0, -1.0]);
    }

    #[test]
    fn tensor_to_f32_unsupported_dtype_returns_error() {
        // I64 is not in the allow-list (used for token-id metadata, not weights).
        let view = make_view(safetensors::Dtype::I64, vec![1], vec![0u8; 8]);
        let err = tensor_to_f32(&view).expect_err("I64 must be unsupported");
        assert!(matches!(err, ModelError::UnsupportedDtype(_)));
    }
}
