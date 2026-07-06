//! GGUF tensor loading, config building, and entry points.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use crate::detect::{detect_from_json_validated, ModelError};
use crate::weights::ModelWeights;

use super::constants::*;
use super::orient::{
    orient_attention_tensors, orient_embedding, orient_ffn_tensors, split_fused_qkv,
};
use super::types::{GgufFile, GgufValue};

/// Sentinel suffix appended to a BitNet I2_S tensor's key under which
/// its per-tensor scale f32 is stashed in `ModelWeights::raw_bytes`
/// during a keep-quant load.  The packed-trit bytes live under the
/// bare key; the 4-byte little-endian scale lives under
/// `"{key}{I2S_SCALE_SUFFIX}"`.  The NUL byte guarantees the sentinel
/// can never collide with a real GGUF tensor name.
pub const I2S_SCALE_SUFFIX: &str = "\0i2s_scale";

impl GgufFile {
    /// Load all tensors, dequantizing to f32.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors(
        &self,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        self.load_tensors_filtered(&|_| false)
    }

    /// Load tensors, skipping normalized keys before reading/dequantizing tensor data.
    ///
    /// `skip_key` sees keys after GGUF-to-HF normalization but before architecture-specific
    /// prefix stripping. GGUF keys do not carry the HF wrapper prefixes, so this is enough for
    /// the current GGUF path and lets walk-only loading avoid FFN dequantization.
    ///
    /// Multi-shard models: tensors are read from `self.shards[info.shard_idx]`,
    /// which is mmap'd lazily on first use within this call. Shards that
    /// contain no surviving tensors after `skip_key` are not mmap'd at all.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered(
        &self,
        skip_key: &dyn Fn(&str) -> bool,
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
        ),
        ModelError,
    > {
        // Single source of truth: the keep-quant variant with no retained
        // types does exactly this (lazy per-shard mmap, normalize, bounds-check,
        // dequantize, dim-swap) and simply returns an empty raw-bytes map.
        let (tensors, vectors, _raw) = self.load_tensors_filtered_keep_quant(skip_key, &[])?;
        Ok((tensors, vectors))
    }

    /// As [`Self::load_tensors_filtered`] but also returns the raw
    /// pre-dequant bytes for tensors whose ggml type matches
    /// `keep_raw_for_types`.  Used by the `--keep-quant` convert
    /// path so I2_S BitLinear tensors can be written verbatim to
    /// the vindex (see `larql_vindex::extract::bitnet_writer`).
    ///
    /// The returned tensors map still contains the dequantised
    /// f32 view (for shape inspection downstream); the raw bytes
    /// are an additional sidecar.
    #[allow(clippy::type_complexity)]
    pub fn load_tensors_filtered_keep_quant(
        &self,
        skip_key: &dyn Fn(&str) -> bool,
        keep_raw_for_types: &[u32],
    ) -> Result<
        (
            HashMap<String, crate::WeightArray>,
            HashMap<String, Vec<f32>>,
            HashMap<String, Vec<u8>>,
        ),
        ModelError,
    > {
        let mut shard_mmaps: Vec<Option<memmap2::Mmap>> =
            (0..self.shards.len()).map(|_| None).collect();

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();
        let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

        let arch = self
            .metadata
            .get(GGUF_GENERAL_ARCHITECTURE)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        for info in &self.tensor_infos {
            let key = normalize_gguf_key_for_arch(&info.name, &arch);
            if skip_key(&key) {
                continue;
            }

            let shard = &self.shards[info.shard_idx];
            if shard_mmaps[info.shard_idx].is_none() {
                let f = std::fs::File::open(&shard.path)?;
                let m = unsafe { memmap2::Mmap::map(&f)? };
                shard_mmaps[info.shard_idx] = Some(m);
            }
            let mmap = shard_mmaps[info.shard_idx]
                .as_ref()
                .expect("mmap initialised above");

            let abs_offset = shard.data_offset.checked_add(info.offset).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: data_offset {} + tensor offset {} overflows u64",
                    info.name, shard.data_offset, info.offset,
                ))
            })?;
            let n_elements: u64 = info.dims.iter().product();

            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| {
                ModelError::Parse(format!(
                    "tensor {}: absolute offset {} exceeds usize on this platform",
                    info.name, abs_offset,
                ))
            })?;
            let end = abs_offset_usize.checked_add(data_size).ok_or_else(|| {
                ModelError::Parse(format!(
                    "tensor {}: offset {} + size {} overflows usize",
                    info.name, abs_offset_usize, data_size,
                ))
            })?;
            if end > mmap.len() {
                return Err(ModelError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > shard {} file {})",
                    info.name,
                    abs_offset,
                    data_size,
                    info.shard_idx,
                    mmap.len()
                )));
            }

            let raw = &mmap[abs_offset_usize..end];
            if keep_raw_for_types.contains(&info.tensor_type) {
                raw_bytes.insert(key.clone(), raw.to_vec());
                // BitNet I2_S stores a single per-tensor scale f32
                // (= max|W| at quant time) immediately AFTER the n/4
                // packed-trit bytes (microsoft/BitNet
                // ggml-bitnet-mad.cpp::quantize_i2_s writes
                // `scale_ptr[0]` at byte offset n/4).  tensor_data_size
                // returns only n/4, so the scale lives in the
                // 32-byte-aligned padding at [end, end+4).  Capture it
                // under a sentinel key so the keep-quant writer can set
                // BitLinearWeight.channel_scales without re-reading the
                // GGUF.  Reconstruction convention: W = trit * scale.
                if info.tensor_type == crate::quant::ggml::TYPE_I2_S {
                    if let Some(se) = end.checked_add(4) {
                        if se <= mmap.len() {
                            let sb = &mmap[end..se];
                            let scale = f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
                            raw_bytes.insert(
                                format!("{key}{I2S_SCALE_SUFFIX}"),
                                scale.to_le_bytes().to_vec(),
                            );
                        }
                    }
                }
            }
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            match info.n_dims {
                2 => {
                    let ne0 = info.dims[0] as usize;
                    let ne1 = info.dims[1] as usize;
                    let arr = Array2::from_shape_vec((ne1, ne0), floats)
                        .map_err(|e| ModelError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr.into_shared());
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {}
            }
        }

        Ok((tensors, vectors, raw_bytes))
    }

    /// Build a config.json-equivalent from GGUF metadata for architecture detection.
    pub fn to_config_json(&self) -> serde_json::Value {
        let get_str = |k: &str| {
            self.metadata
                .get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        let _get_u32 = |k: &str| self.metadata.get(k).and_then(|v| v.as_u32()).unwrap_or(0);

        // GGUF uses "general.architecture" and "{arch}.*" keys
        let arch = get_str(GGUF_GENERAL_ARCHITECTURE);
        let prefix = format!("{arch}.");

        let get_arch_u32 = |suffix: &str| {
            let key = format!("{prefix}{suffix}");
            if let Some(v) = self.metadata.get(&key) {
                // Try scalar first, then array max (handles Gemma 4 variable FFN sizes)
                if let Some(val) = v.as_u32() {
                    return val;
                }
                if let GgufValue::Array(arr) = v {
                    return arr.iter().filter_map(|x| x.as_u32()).max().unwrap_or(0);
                }
            }
            0
        };
        let get_arch_u32_opt = |suffix: &str| {
            let key = format!("{prefix}{suffix}");
            self.metadata.get(&key).and_then(|v| v.as_u32())
        };
        let get_arch_f64 = |suffix: &str| {
            self.metadata
                .get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_f64())
        };

        // Map GGUF architecture names to HF model_type
        let model_type = match arch.as_str() {
            "llama" => "llama",
            "gemma" | "gemma2" | "gemma3" | "gemma4" => &arch,
            "qwen" | "qwen2" => "qwen2",
            "mistral" => "mistral",
            "mixtral" => "mixtral",
            "phi" | "phi2" | "phi3" => "phi",
            "gpt2" => "gpt2",
            "deepseek" | "deepseek2" => "deepseek_v2",
            "deepseek_v4" | "deepseekv4" => "deepseek_v4",
            other => other,
        };

        let hidden_size = get_arch_u32(GGUF_EMBEDDING_LENGTH);
        let num_heads = get_arch_u32(GGUF_ATTENTION_HEAD_COUNT);
        let num_kv_heads = get_arch_u32(GGUF_ATTENTION_HEAD_COUNT_KV);
        let head_dim = if arch == "gemma4" && num_heads > 0 {
            // Gemma 4 GGUF metadata reports the global key length; known
            // exports use 256 for the per-head dimension that the runtime
            // architecture needs as its base layer head_dim.
            GEMMA4_GGUF_HEAD_DIM
        } else {
            let key_length = get_arch_u32(GGUF_ATTENTION_KEY_LENGTH);
            if key_length > 0 {
                key_length
            } else {
                hidden_size.checked_div(num_heads).unwrap_or(0)
            }
        };
        let num_kv_heads = if num_kv_heads > 0 {
            num_kv_heads
        } else {
            num_heads
        };

        // intermediate_size: prefer the global `feed_forward_length`. For
        // MoE-only models (DeepSeek-V4 family) the global key is omitted,
        // so we fall back to the per-expert size. The HF config exposes
        // `intermediate_size` as a single number even on MoE archs because
        // per-expert and per-layer FFNs share that dim in every
        // llama.cpp-supported architecture.
        let intermediate_size = {
            let global = get_arch_u32(GGUF_FEED_FORWARD_LENGTH);
            if global > 0 {
                global
            } else {
                get_arch_u32(GGUF_EXPERT_FEED_FORWARD_LENGTH)
            }
        };
        let mut config = serde_json::json!({
            HF_MODEL_TYPE: model_type,
            HF_HIDDEN_SIZE: hidden_size,
            HF_NUM_HIDDEN_LAYERS: get_arch_u32(GGUF_BLOCK_COUNT),
            HF_INTERMEDIATE_SIZE: intermediate_size,
            HF_NUM_ATTENTION_HEADS: num_heads,
            HF_NUM_KEY_VALUE_HEADS: num_kv_heads,
            HF_HEAD_DIM: head_dim,
        });

        if let Some(rope_base) = get_arch_f64(GGUF_ROPE_FREQ_BASE) {
            config[HF_ROPE_THETA] = serde_json::json!(rope_base);
        }
        if let Some(vocab_size) = get_arch_u32_opt(GGUF_VOCAB_SIZE).filter(|&v| v > 0) {
            config[HF_VOCAB_SIZE] = serde_json::json!(vocab_size);
        }

        // ── Gemma 4 per-layer attention geometry ────────────────────────────
        // Gemma 4 GGUFs describe heterogeneous attention (sliding layers vs
        // global layers) with per-layer arrays and `*_swa` twin keys; the
        // flat mapping above collapses them to one number and drops the
        // rest. Re-emit the flat HF-style keys the safetensors detect path
        // reads (`detect/parser.rs`) so the reconstructed arch can route
        // per layer. Without these, gemma-4 models whose global layers have
        // no attn_v tensor (12B, 31B: `attention_k_eq_v`) index past the
        // end of the V collection at inference time.
        if arch == "gemma4" {
            // head_count_kv is a per-layer array (e.g. 8 on sliding layers,
            // 1 on global layers for the 12B). Layer 0 is always sliding;
            // the first differing value is the global-layer count.
            if let Some(GgufValue::Array(arr)) = self
                .metadata
                .get(&format!("{prefix}{GGUF_ATTENTION_HEAD_COUNT_KV}"))
            {
                let vals: Vec<u32> = arr.iter().filter_map(|x| x.as_u32()).collect();
                if let Some(&sliding) = vals.first() {
                    config[HF_NUM_KEY_VALUE_HEADS] = serde_json::json!(sliding);
                    if let Some(&global) = vals.iter().find(|&&v| v != sliding) {
                        config["num_global_key_value_heads"] = serde_json::json!(global);
                    }
                }
            }
            // key_length is the global-layer head width; key_length_swa the
            // sliding-layer width (the base head_dim the arch expects).
            let key_len = get_arch_u32(GGUF_ATTENTION_KEY_LENGTH);
            let key_len_swa = get_arch_u32("attention.key_length_swa");
            if key_len_swa > 0 {
                config[HF_HEAD_DIM] = serde_json::json!(key_len_swa);
            }
            if key_len > 0 && key_len != key_len_swa {
                config["global_head_dim"] = serde_json::json!(key_len);
                // Global layers rotate a quarter of their head dims —
                // constant across every Gemma 4 HF config (12B/26B/31B);
                // the GGUF carries no equivalent key.
                config["partial_rotary_factor"] = serde_json::json!(0.25);
            }
            // Dual RoPE bases: `rope.freq_base` is the global-layer clock
            // (already emitted as rope_theta above), `rope.freq_base_swa`
            // the sliding-layer one.
            if let Some(swa) = get_arch_f64("rope.freq_base_swa") {
                config["rope_local_base_freq"] = serde_json::json!(swa);
            }
            if let Some(sw) = get_arch_u32_opt("attention.sliding_window").filter(|&v| v > 0) {
                config["sliding_window"] = serde_json::json!(sw);
            }
            // Layers with no attn_v tensor reuse K as V (`attention_k_eq_v`
            // in the HF config). The GGUF metadata has no flag for this;
            // detect it from the tensor inventory, as llama.cpp does.
            let n_blocks = get_arch_u32(GGUF_BLOCK_COUNT) as usize;
            let n_v = self
                .tensor_infos
                .iter()
                .filter(|t| t.name().ends_with(".attn_v.weight"))
                .count();
            if n_blocks > 0 && n_v > 0 && n_v < n_blocks {
                config["attention_k_eq_v"] = serde_json::json!(true);
            }
        }

        // ── MLA fields (DeepSeek-V2/V3 family, e.g. Kimi K2) ─────────────────
        // The HF config exposes `q_lora_rank` / `kv_lora_rank` /
        // `qk_nope_head_dim` / `qk_rope_head_dim` / `v_head_dim`. llama.cpp
        // emits the equivalent fields under the `{arch}.attention.*` and
        // `{arch}.rope.dimension_count` namespace; we surface them here so
        // the existing parser → `ModelConfig` path picks them up and MLA
        // absorption (PR #96) fires for GGUF-sourced inputs.
        //
        // For per-head dims we prefer the `_mla` variants when present —
        // those carry the pre-absorption (DeepSeek-V3 standard) split that
        // `mla_absorb::absorb()` operates on. The non-`_mla` keys can hold
        // post-absorption / "effective" widths (576/512 on Kimi K2.6) which
        // are too large to feed back into the absorption math.
        if let Some(q_lora) = get_arch_u32_opt(GGUF_ATTENTION_Q_LORA_RANK).filter(|&v| v > 0) {
            config["q_lora_rank"] = serde_json::json!(q_lora);
        }
        if let Some(kv_lora) = get_arch_u32_opt(GGUF_ATTENTION_KV_LORA_RANK).filter(|&v| v > 0) {
            config["kv_lora_rank"] = serde_json::json!(kv_lora);
        }
        let qk_rope = get_arch_u32_opt(GGUF_ROPE_DIMENSION_COUNT).filter(|&v| v > 0);
        if let Some(rope) = qk_rope {
            config["qk_rope_head_dim"] = serde_json::json!(rope);
        }
        // qk_head_dim total: prefer key_length_mla, fall back to key_length.
        let key_length_mla = get_arch_u32_opt(GGUF_ATTENTION_KEY_LENGTH_MLA).filter(|&v| v > 0);
        let key_length = get_arch_u32_opt(GGUF_ATTENTION_KEY_LENGTH).filter(|&v| v > 0);
        let qk_head_dim = key_length_mla.or(key_length);
        if let (Some(qk_total), Some(rope)) = (qk_head_dim, qk_rope) {
            if qk_total > rope {
                config["qk_nope_head_dim"] = serde_json::json!(qk_total - rope);
            }
        }
        // v_head_dim: prefer value_length_mla, fall back to value_length.
        let v_head = get_arch_u32_opt(GGUF_ATTENTION_VALUE_LENGTH_MLA)
            .filter(|&v| v > 0)
            .or_else(|| get_arch_u32_opt(GGUF_ATTENTION_VALUE_LENGTH).filter(|&v| v > 0));
        if let Some(v) = v_head {
            config["v_head_dim"] = serde_json::json!(v);
        }

        config
    }
}

/// Load a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered(path, &|_| false)
}

/// Load a GGUF file into ModelWeights, retaining the original
/// pre-dequant bytes for tensors of the listed types.
///
/// Used by `larql convert gguf-to-vindex --keep-quant` so the
/// BitNet 1.58 I2_S BitLinear bytes survive into
/// `ModelWeights::raw_bytes` (rather than being dropped after
/// dequantization).  See BUG-infer-deadlock §5.4.
///
/// `keep_types` should be the list of GGML type IDs whose bytes
/// you want preserved.  For BitNet pass `&[36]` (TYPE_I2_S); for
/// future TQ1_0/TQ2_0 native paths pass `&[34, 35]`.
pub fn load_gguf_keep_quant(path: &Path, keep_types: &[u32]) -> Result<ModelWeights, ModelError> {
    load_gguf_keep_quant_filtered(path, &|_| false, keep_types)
}

pub(crate) fn load_gguf_keep_quant_filtered(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    keep_types: &[u32],
) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation_and_keep(path, skip_key, false, keep_types)
}

/// Load and validate a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf_validated(path: &Path) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation(path, &|_| false, true)
}

/// Load a GGUF file into ModelWeights with optional architecture validation.
pub(crate) fn load_gguf_filtered(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    load_gguf_filtered_with_validation_and_keep(path, skip_key, false, &[])
}

/// Same as [`load_gguf_filtered_with_validation`] but also retains
/// raw pre-dequant bytes for tensors whose GGML type appears in
/// `keep_types`.
pub(crate) fn load_gguf_filtered_with_validation_and_keep(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    validate_config: bool,
    keep_types: &[u32],
) -> Result<ModelWeights, ModelError> {
    let gguf = GgufFile::open(path)?;

    let config_json = gguf.to_config_json();
    let arch = if validate_config {
        detect_from_json_validated(&config_json)?
    } else {
        crate::detect_from_json(&config_json)
    };
    let prefixes = arch.key_prefixes_to_strip();

    let (mut tensors, mut vectors, mut raw_keep) =
        gguf.load_tensors_filtered_keep_quant(skip_key, keep_types)?;

    let mut normalized_tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = crate::loading::safetensors::normalize_key(&k, prefixes);
        normalized_tensors.insert(key, v);
    }
    // Re-key the raw_bytes map through the same normalisation so
    // downstream consumers can look up by the canonical name.
    let mut normalized_raw: HashMap<String, Vec<u8>> = HashMap::new();
    for (k, v) in raw_keep.drain() {
        let key = crate::loading::safetensors::normalize_key(&k, prefixes);
        normalized_raw.insert(key, v);
    }

    orient_ffn_tensors(&mut normalized_tensors, &*arch);
    orient_attention_tensors(&mut normalized_tensors, &*arch);
    split_fused_qkv(&mut normalized_tensors, &mut vectors, &*arch);

    let embed_key = arch.embed_key();
    let embed_raw = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();
    let cfg = arch.config();
    let tokenizer_vocab_size = read_tokenizer_vocab_size(path);
    let configured_vocab_size = cfg.vocab_size.filter(|&v| v > 0);
    let expected_vocab_size = configured_vocab_size.or(tokenizer_vocab_size);
    let embed = orient_embedding(embed_raw, cfg.hidden_size, expected_vocab_size);

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get(GGUF_OUTPUT_WEIGHT))
        .cloned()
        .unwrap_or_else(|| embed.clone());
    let position_embed = arch
        .position_embed_key()
        .and_then(|key| normalized_tensors.get(key).cloned());

    let vocab_size = expected_vocab_size
        .or_else(|| (embed.shape()[0] > 0).then_some(embed.shape()[0]))
        .unwrap_or(DEFAULT_GGUF_VOCAB_SIZE);

    let cfg_clone = cfg.clone();
    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
        raw_bytes: normalized_raw,
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed,
        num_layers: cfg_clone.num_layers,
        hidden_size: cfg_clone.hidden_size,
        intermediate_size: cfg_clone.intermediate_size,
        vocab_size,
        head_dim: cfg_clone.head_dim,
        num_q_heads: cfg_clone.num_q_heads,
        num_kv_heads: cfg_clone.num_kv_heads,
        rope_base: cfg_clone.rope_base,
        arch,
    })
}

/// Load a GGUF file into ModelWeights with optional architecture validation.
pub(crate) fn load_gguf_filtered_with_validation(
    path: &Path,
    skip_key: &dyn Fn(&str) -> bool,
    validate_config: bool,
) -> Result<ModelWeights, ModelError> {
    // Identical to the keep-quant path with no retained types — that variant
    // runs the same detect/load/orient/embed pipeline and leaves `raw_bytes`
    // empty.  Single source of truth for the GGUF load orchestration.
    load_gguf_filtered_with_validation_and_keep(path, skip_key, validate_config, &[])
}

pub(super) fn read_tokenizer_vocab_size(path: &Path) -> Option<usize> {
    let parent = path.parent()?;
    let tok_path = parent.join(TOKENIZER_JSON);
    let data = std::fs::read_to_string(tok_path).ok()?;
    let json = serde_json::from_str::<serde_json::Value>(&data).ok()?;
    json[TOKENIZER_MODEL][TOKENIZER_VOCAB]
        .as_object()
        .map(|v| v.len())
        .filter(|&v| v > 0)
}

pub(super) fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    crate::quant::ggml::tensor_data_size(tensor_type, n_elements)
}

pub(super) fn dequantize(
    data: &[u8],
    tensor_type: u32,
    n_elements: usize,
) -> Result<Vec<f32>, ModelError> {
    crate::quant::ggml::dequantize(data, tensor_type, n_elements)
}

/// Normalize GGUF tensor key names to match HuggingFace conventions.
pub fn normalize_gguf_key(name: &str) -> String {
    // GGUF uses "blk.N.attn_q.weight" format
    // HF uses "model.layers.N.self_attn.q_proj.weight" format
    // We normalize to the HF style since that's what ModelArchitecture expects

    GGUF_TO_HF_KEY_REPLACEMENTS
        .iter()
        .fold(name.to_string(), |acc, (from, to)| acc.replace(from, to))
}

/// As [`normalize_gguf_key`], but arch-aware: gemma 2/3/4 GGUFs use a
/// four-norm layer layout (attn_norm / post_attention_norm / ffn_norm /
/// post_ffw_norm) plus QK-norms and, on Gemma 4, a per-layer output
/// scalar. The generic table maps `ffn_norm.` to the llama-style
/// post-attention slot — actively wrong for gemma — and drops the rest
/// on the floor, so gemma-specific replacements run first. Gemma 1 has
/// the llama two-norm layout and stays on the generic path.
pub fn normalize_gguf_key_for_arch(name: &str, arch: &str) -> String {
    let gemma_layout =
        matches!(arch, "gemma2" | "gemma3") || arch.starts_with("gemma4");
    let name = if gemma_layout {
        GGUF_TO_HF_KEY_REPLACEMENTS_GEMMA
            .iter()
            .fold(name.to_string(), |acc, (from, to)| acc.replace(from, to))
    } else {
        name.to_string()
    };
    GGUF_TO_HF_KEY_REPLACEMENTS
        .iter()
        .fold(name, |acc, (from, to)| acc.replace(from, to))
}

#[cfg(test)]
mod tests {
    use super::super::constants::*;
    use super::super::types::ShardInfo;
    use super::*;

    #[test]
    fn test_normalize_gguf_key() {
        assert_eq!(
            normalize_gguf_key("blk.0.attn_q.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.15.ffn_gate.weight"),
            "layers.15.mlp.gate_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("token_embd.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(normalize_gguf_key("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_normalize_gguf_key_gemma_layout() {
        assert_eq!(
            normalize_gguf_key_for_arch("blk.0.attn_q_norm.weight", "gemma4"),
            "layers.0.self_attn.q_norm.weight"
        );
        assert_eq!(
            normalize_gguf_key_for_arch("blk.0.attn_k_norm.weight", "gemma4_unified"),
            "layers.0.self_attn.k_norm.weight"
        );
        assert_eq!(
            normalize_gguf_key_for_arch("blk.0.post_attention_norm.weight", "gemma2"),
            "layers.0.post_attention_layernorm.weight"
        );
        // gemma's ffn_norm is the PRE-feedforward norm...
        assert_eq!(
            normalize_gguf_key_for_arch("blk.3.ffn_norm.weight", "gemma3"),
            "layers.3.pre_feedforward_layernorm.weight"
        );
        // ...while llama's ffn_norm keeps the generic post-attention mapping.
        assert_eq!(
            normalize_gguf_key_for_arch("blk.3.ffn_norm.weight", "llama"),
            "layers.3.post_attention_layernorm.weight"
        );
        // Gemma 1 is llama-layout too.
        assert_eq!(
            normalize_gguf_key_for_arch("blk.3.ffn_norm.weight", "gemma"),
            "layers.3.post_attention_layernorm.weight"
        );
        assert_eq!(
            normalize_gguf_key_for_arch("blk.47.post_ffw_norm.weight", "gemma4"),
            "layers.47.post_feedforward_layernorm.weight"
        );
        assert_eq!(
            normalize_gguf_key_for_arch("blk.5.layer_output_scale.weight", "gemma4"),
            "layers.5.layer_scalar"
        );
        // Mapped projections are untouched by the gemma pre-pass.
        assert_eq!(
            normalize_gguf_key_for_arch("blk.0.attn_q.weight", "gemma4"),
            "layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn test_load_tensors_swaps_gguf_2d_dims_to_rows_cols() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let mut file = std::fs::File::create(&path).unwrap();

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // version
        file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

        // Tensor info: ggml dims order is [cols, rows].
        let name = b"blk.0.ffn_down.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
        file.write_all(&4u64.to_le_bytes()).unwrap(); // cols
        file.write_all(&2u64.to_le_bytes()).unwrap(); // rows
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor data offset

        // Pad tensor data start to 32-byte boundary.
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();

        // Raw row-major data for a logical [2, 4] matrix.
        for v in 1u32..=8 {
            file.write_all(&(v as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let (tensors, _) = gguf.load_tensors().unwrap();
        let down = tensors.get("layers.0.mlp.down_proj.weight").unwrap();

        assert_eq!(down.shape(), &[2, 4]);
        assert_eq!(down[[0, 0]], 1.0);
        assert_eq!(down[[0, 1]], 2.0);
        assert_eq!(down[[0, 2]], 3.0);
        assert_eq!(down[[0, 3]], 4.0);
        assert_eq!(down[[1, 0]], 5.0);
        assert_eq!(down[[1, 1]], 6.0);
        assert_eq!(down[[1, 2]], 7.0);
        assert_eq!(down[[1, 3]], 8.0);
    }

    // ── keep-quant (BitNet I2_S) loader coverage ───────────────────
    //
    // Exercise the `--keep-quant` path (`load_gguf_keep_quant` /
    // `GgufFile::load_tensors_filtered_keep_quant`) that preserves the
    // raw pre-dequant I2_S BitLinear bytes plus the trailing per-tensor
    // scale f32 (microsoft/BitNet layout — see `dequantize_i2_s`).

    fn kq_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }
    fn kq_meta_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        kq_str(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF string type
        kq_str(buf, val);
    }
    fn kq_meta_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        kq_str(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // GGUF uint32 type
        buf.extend_from_slice(&val.to_le_bytes());
    }
    fn kq_meta_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
        kq_str(buf, key);
        buf.extend_from_slice(&6u32.to_le_bytes()); // GGUF float32 type
        buf.extend_from_slice(&val.to_le_bytes());
    }
    fn kq_tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, offset: u64) {
        kq_str(buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&ty.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
    }

    /// 32 bytes of I2_S packed trits — one full 128-element block.
    const KQ_I2S_PACKED: [u8; 32] = [
        0b00_01_10_00,
        0x55,
        0xAA,
        0x0F,
        0xF0,
        0x33,
        0xCC,
        0x12,
        0x00,
        0x55,
        0xAA,
        0x0F,
        0xF0,
        0x33,
        0xCC,
        0x12,
        0x00,
        0x55,
        0xAA,
        0x0F,
        0xF0,
        0x33,
        0xCC,
        0x12,
        0x00,
        0x55,
        0xAA,
        0x0F,
        0xF0,
        0x33,
        0xCC,
        0x12,
    ];

    /// Build a complete minimal llama GGUF: token_embd / output /
    /// output_norm (F32) plus one I2_S "BitLinear" tensor
    /// `blk.0.bitlinear.weight` (128 trits → 32 packed bytes). The
    /// name is deliberately *not* an attn/ffn projection so the
    /// orientation passes leave it untouched. When `scale` is `Some`,
    /// the per-tensor scale f32 is appended immediately after the
    /// packed trits (where `load_tensors_filtered_keep_quant` looks
    /// for it).
    fn build_i2s_gguf(scale: Option<f32>) -> Vec<u8> {
        build_i2s_gguf_opts(scale, true, true)
    }

    /// As [`build_i2s_gguf`] but lets a test omit `token_embd` (to drive the
    /// missing-embed error) or `output` (to drive the tied-lm_head fallback).
    fn build_i2s_gguf_opts(
        scale: Option<f32>,
        include_embed: bool,
        include_output: bool,
    ) -> Vec<u8> {
        const HIDDEN: u64 = 4;
        const VOCAB: u64 = 8;
        let f32t = crate::quant::ggml::TYPE_F32;
        let i2st = crate::quant::ggml::TYPE_I2_S;
        let align32 = |n: u64| n.div_ceil(32) * 32;

        // (name, dims, ggml_type, data) — the I2_S tensor's trailing scale
        // is part of its `data` here (it lives in the alignment slack after
        // the 32 declared bytes; the I2_S tensor is always last).
        let embed = vec![0u8; (HIDDEN * VOCAB * 4) as usize];
        let norm = vec![0u8; (HIDDEN * 4) as usize];
        let mut i2s = KQ_I2S_PACKED.to_vec();
        if let Some(s) = scale {
            i2s.extend_from_slice(&s.to_le_bytes());
        }
        let mut tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = Vec::new();
        if include_embed {
            tensors.push((
                "token_embd.weight",
                vec![HIDDEN, VOCAB],
                f32t,
                embed.clone(),
            ));
        }
        if include_output {
            tensors.push(("output.weight", vec![HIDDEN, VOCAB], f32t, embed));
        }
        tensors.push(("output_norm.weight", vec![HIDDEN], f32t, norm));
        tensors.push(("blk.0.bitlinear.weight", vec![32, 4], i2st, i2s));

        // Lay out 32-aligned data-section-relative offsets.
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut running = 0u64;
        for (_, _, _, data) in &tensors {
            offsets.push(running);
            running = align32(running + data.len() as u64);
        }

        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes()); // n_metadata
        kq_meta_str(&mut buf, "general.architecture", "llama");
        kq_meta_u32(&mut buf, "llama.embedding_length", HIDDEN as u32);
        kq_meta_u32(&mut buf, "llama.block_count", 1);
        kq_meta_u32(&mut buf, "llama.feed_forward_length", 16);
        kq_meta_u32(&mut buf, "llama.attention.head_count", 2);
        kq_meta_u32(&mut buf, "llama.attention.head_count_kv", 2);
        kq_meta_u32(&mut buf, "llama.attention.key_length", 2);
        kq_meta_f32(&mut buf, "llama.rope.freq_base", 10000.0);
        for ((name, dims, ty, _), &off) in tensors.iter().zip(&offsets) {
            kq_tensor_info(&mut buf, name, dims, *ty, off);
        }
        // Pad to the 32-byte data-section boundary, then write each tensor
        // at its declared offset.
        let pad = (32 - (buf.len() % 32)) % 32;
        buf.resize(buf.len() + pad, 0);
        let data_start = buf.len();
        for ((_, _, _, data), &off) in tensors.iter().zip(&offsets) {
            buf.resize(data_start + off as usize, 0);
            buf.extend_from_slice(data);
        }
        buf
    }

    fn write_tmp_gguf(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bitnet.gguf");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(bytes)
            .unwrap();
        (dir, path)
    }

    #[test]
    fn keep_quant_captures_i2s_bytes_and_trailing_scale() {
        let scale = 0.527_f32;
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(scale)));

        let weights =
            load_gguf_keep_quant(&path, &[crate::quant::ggml::TYPE_I2_S]).expect("keep-quant load");

        // The raw I2_S bytes survive into ModelWeights::raw_bytes under
        // the normalized tensor key, byte-for-byte.
        let key = crate::loading::safetensors::normalize_key(
            &normalize_gguf_key("blk.0.bitlinear.weight"),
            weights.arch.key_prefixes_to_strip(),
        );
        let raw = weights.raw_bytes.get(&key).unwrap_or_else(|| {
            panic!(
                "missing raw I2_S bytes for {key}; have {:?}",
                weights.raw_bytes.keys().collect::<Vec<_>>()
            )
        });
        assert_eq!(
            raw.as_slice(),
            &KQ_I2S_PACKED,
            "raw trits preserved verbatim"
        );

        // The trailing per-tensor scale is captured under the sentinel key.
        let scale_bytes = weights
            .raw_bytes
            .get(&format!("{key}{}", crate::I2S_SCALE_SUFFIX))
            .expect("missing captured I2_S scale");
        let got = f32::from_le_bytes(scale_bytes.as_slice().try_into().unwrap());
        assert!((got - scale).abs() < 1e-6, "scale {got} != {scale}");
    }

    #[test]
    fn keep_quant_method_captures_scale_and_filters_by_type() {
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(1.5)));
        let gguf = GgufFile::open(&path).unwrap();
        let (_tensors, _vectors, raw) = gguf
            .load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_I2_S])
            .unwrap();

        // Only the I2_S tensor (and its scale) are retained — the F32
        // embed/output/norm tensors are not in `keep_raw_for_types`.
        let i2s_key = normalize_gguf_key("blk.0.bitlinear.weight");
        assert!(raw.contains_key(&i2s_key));
        assert!(raw.contains_key(&format!("{i2s_key}{}", crate::I2S_SCALE_SUFFIX)));
        assert!(
            !raw.contains_key("embed_tokens.weight"),
            "F32 tensors must not be retained when only I2_S is requested"
        );
    }

    #[test]
    fn keep_quant_non_i2s_type_retained_without_scale() {
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(2.0)));
        let gguf = GgufFile::open(&path).unwrap();
        // Request F32 retention: bytes are kept, but no scale sentinel
        // is written (scale capture is I2_S-specific).
        let (_t, _v, raw) = gguf
            .load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_F32])
            .unwrap();
        assert!(
            raw.contains_key("embed_tokens.weight"),
            "F32 bytes retained"
        );
        assert!(
            raw.keys().all(|k| !k.ends_with(crate::I2S_SCALE_SUFFIX)),
            "no scale sentinel for non-I2_S types"
        );
    }

    #[test]
    fn keep_quant_method_honours_skip_key() {
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(1.0)));
        let gguf = GgufFile::open(&path).unwrap();
        let (_t, _v, raw) = gguf
            .load_tensors_filtered_keep_quant(
                &|k| k.contains("bitlinear"),
                &[crate::quant::ggml::TYPE_I2_S],
            )
            .unwrap();
        // The only keep-eligible tensor was skipped, so nothing is retained.
        assert!(
            raw.is_empty(),
            "skipped tensor must not be retained: {raw:?}"
        );
    }

    #[test]
    fn keep_quant_scale_absent_when_no_trailing_room() {
        // No trailing scale f32 → the [end, end+4) window runs past EOF
        // and the scale is simply not captured (the raw trits still are).
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(None));
        let gguf = GgufFile::open(&path).unwrap();
        let (_t, _v, raw) = gguf
            .load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_I2_S])
            .unwrap();
        let i2s_key = normalize_gguf_key("blk.0.bitlinear.weight");
        assert!(raw.contains_key(&i2s_key), "raw trits still captured");
        assert!(
            !raw.contains_key(&format!("{i2s_key}{}", crate::I2S_SCALE_SUFFIX)),
            "scale must be absent when there is no trailing f32"
        );
    }

    #[test]
    fn load_gguf_keep_quant_with_empty_keep_types_matches_plain_load() {
        // keep_types = [] → no raw bytes retained; otherwise identical
        // to the plain load path.
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(Some(1.0)));
        let weights = load_gguf_keep_quant(&path, &[]).unwrap();
        assert!(
            weights.raw_bytes.is_empty(),
            "no raw bytes when keep_types is empty"
        );
        assert_eq!(weights.embed.shape(), &[8, 4]);
    }

    #[test]
    fn load_gguf_resolves_embed_lm_head_and_vocab() {
        // Drives the unified orchestrator over the GGUF path (load_gguf now
        // routes through the keep-quant orchestrator with no retained types):
        // embed orientation, the present-`output.weight` lm_head branch, and
        // vocab-size resolution from the embedding shape.
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(None));
        let weights = load_gguf(&path).unwrap();
        assert_eq!(weights.embed.shape(), &[8, 4]);
        assert_eq!(weights.lm_head.shape(), &[8, 4]);
        assert_eq!(weights.vocab_size, 8);
        assert!(weights.raw_bytes.is_empty());
    }

    #[test]
    fn load_gguf_lm_head_falls_back_to_embed_when_output_absent() {
        // No `output.weight` → lm_head ties to the embedding.
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf_opts(None, true, false));
        let weights = load_gguf(&path).unwrap();
        assert_eq!(weights.lm_head.shape(), weights.embed.shape());
    }

    #[test]
    fn load_gguf_missing_embed_is_missing_tensor_error() {
        // No `token_embd.weight` → the embed-key lookup fails.
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf_opts(None, false, true));
        match load_gguf(&path) {
            Ok(_) => panic!("expected MissingTensor error for embed-less GGUF"),
            Err(e) => assert!(
                matches!(e, ModelError::MissingTensor(_)),
                "expected MissingTensor, got {e:?}"
            ),
        }
    }

    #[test]
    fn load_gguf_validated_routes_through_unified_pipeline() {
        // load_gguf_validated shares the same orchestrator (validate=true);
        // a well-formed llama GGUF passes architecture validation.
        let (_dir, path) = write_tmp_gguf(&build_i2s_gguf(None));
        let weights = load_gguf_validated(&path).unwrap();
        assert_eq!(weights.embed.shape(), &[8, 4]);
    }

    #[test]
    fn keep_quant_oob_tensor_offset_errors() {
        // Corrupt the last tensor-info offset so the declared data runs past
        // EOF — exercises the bounds-check error arm in the loader loop.
        let mut bytes = build_i2s_gguf(Some(1.0));
        // The data section is well under 1 MiB; an offset of 1 MiB is past EOF
        // for every tensor.  Rewrite the final tensor info's offset field.
        // Tensor-info offset is the last u64 of the I2_S tensor info, which is
        // the last tensor-info written; locate it by scanning for the packed
        // marker is fragile, so instead rebuild with a deliberately bad file:
        // truncate the data section to force the bounds check.
        bytes.truncate(bytes.len() - 40);
        let (_dir, path) = write_tmp_gguf(&bytes);
        let err = GgufFile::open(&path)
            .and_then(|g| {
                g.load_tensors_filtered_keep_quant(&|_| false, &[crate::quant::ggml::TYPE_I2_S])
                    .map(|_| ())
            })
            .unwrap_err();
        assert!(
            matches!(err, ModelError::Parse(_)),
            "expected a Parse (out-of-bounds) error, got {err:?}"
        );
    }

    #[test]
    fn test_gemma4_gguf_to_config_json_maps_arch_and_overrides_head_dim() {
        // Synthesize GGUF metadata matching gemma-4-e2b's shape.
        // Exercises: (a) gemma4 name pass-through, (b) head_dim=256 override,
        // (c) array metadata (per-layer variable FFN sizes → take max).
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("gemma4".to_string()),
        );
        metadata.insert("gemma4.embedding_length".to_string(), GgufValue::U32(1536));
        metadata.insert("gemma4.block_count".to_string(), GgufValue::U32(35));
        metadata.insert("gemma4.attention.head_count".to_string(), GgufValue::U32(8));
        metadata.insert(
            "gemma4.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        // Gemma 4 reports attention.key_length=512 (global head_dim), not the
        // per-head 256 we want. Loader must override to 256 for arch="gemma4".
        metadata.insert(
            "gemma4.attention.key_length".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert("gemma4.vocab_size".to_string(), GgufValue::U32(262144));
        // Per-layer variable FFN — some layers 6144, some 12288. Must take max.
        metadata.insert(
            "gemma4.feed_forward_length".to_string(),
            GgufValue::Array(vec![
                GgufValue::U32(6144),
                GgufValue::U32(12288),
                GgufValue::U32(6144),
            ]),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        assert_eq!(cfg["model_type"], "gemma4");
        assert_eq!(cfg["hidden_size"], 1536);
        assert_eq!(cfg["num_hidden_layers"], 35);
        // head_dim override: 256 despite attention.key_length=512
        assert_eq!(cfg["head_dim"], 256);
        // intermediate_size: max of the per-layer FFN array (12288), not 6144
        assert_eq!(cfg["intermediate_size"], 12288);
        assert_eq!(cfg["num_attention_heads"], 8);
        assert_eq!(cfg["num_key_value_heads"], 1);
        assert_eq!(cfg["vocab_size"], 262144);
    }

    #[test]
    fn test_gguf_to_config_json_omits_absent_rope_base_for_arch_default() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );
        metadata.insert(
            "llama.attention.key_length".to_string(),
            GgufValue::U32(128),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        assert!(cfg.get(HF_ROPE_THETA).is_none());
        let arch = crate::detect_from_json_validated(&cfg).unwrap();
        assert_eq!(arch.config().rope_base, 10_000.0);
    }

    #[test]
    fn test_kimi_k2_gguf_to_config_json_extracts_mla_fields() {
        // Synthesize GGUF metadata matching Kimi K2.6's unsloth Q8_K_XL shape.
        // Verifies the MLA fields surface into the HF-style config that the
        // parser → ModelConfig path consumes, so that PR #96's MLA absorption
        // fires for GGUF-sourced DeepSeek-V2/V3/Kimi-K2 models. Closes #67.
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(7168),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(61));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(64),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        metadata.insert(
            "deepseek2.feed_forward_length".to_string(),
            GgufValue::U32(18432),
        );
        metadata.insert("deepseek2.vocab_size".to_string(), GgufValue::U32(163840));
        // MLA-specific keys emitted by llama.cpp for DeepSeek-V2/V3 family.
        // `_mla` carries the pre-absorption per-head split that PR #96 needs.
        metadata.insert(
            "deepseek2.attention.q_lora_rank".to_string(),
            GgufValue::U32(1536),
        );
        metadata.insert(
            "deepseek2.attention.kv_lora_rank".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(576),
        );
        metadata.insert(
            "deepseek2.attention.value_length".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert(
            "deepseek2.attention.key_length_mla".to_string(),
            GgufValue::U32(192),
        );
        metadata.insert(
            "deepseek2.attention.value_length_mla".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.rope.dimension_count".to_string(),
            GgufValue::U32(64),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        // Model type maps deepseek2 → deepseek_v2 (existing logic).
        assert_eq!(cfg["model_type"], "deepseek_v2");
        // MLA fields populated from GGUF metadata.
        assert_eq!(cfg["q_lora_rank"], 1536);
        assert_eq!(cfg["kv_lora_rank"], 512);
        assert_eq!(cfg["qk_rope_head_dim"], 64);
        // qk_nope_head_dim = key_length_mla - rope.dimension_count = 192-64 = 128
        // (prefers _mla variant over the absorbed key_length=576).
        assert_eq!(cfg["qk_nope_head_dim"], 128);
        // v_head_dim prefers the _mla variant (128 pre-absorption, not 512).
        assert_eq!(cfg["v_head_dim"], 128);

        // Architecture-detection path picks the fields up into ModelConfig.
        let arch = crate::detect_from_json(&cfg);
        assert_eq!(arch.mla_qk_nope_head_dim(), Some(128));
        assert_eq!(arch.mla_qk_rope_head_dim(), Some(64));
        assert_eq!(arch.mla_v_head_dim(), Some(128));
        assert_eq!(arch.q_lora_rank(), 1536);
        assert_eq!(arch.kv_lora_rank(), 512);
        assert!(arch.uses_mla());
    }

    #[test]
    fn test_gguf_mla_falls_back_to_non_mla_key_length_when_mla_keys_absent() {
        // Some DeepSeek-V2 GGUFs may not emit the `_mla` variants. The
        // loader must fall back to attention.key_length / value_length so
        // the pre-absorption split is still computed.
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(5120),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(27));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.feed_forward_length".to_string(),
            GgufValue::U32(12288),
        );
        metadata.insert(
            "deepseek2.attention.q_lora_rank".to_string(),
            GgufValue::U32(1536),
        );
        metadata.insert(
            "deepseek2.attention.kv_lora_rank".to_string(),
            GgufValue::U32(512),
        );
        // Only non-`_mla` variants present.
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(192),
        );
        metadata.insert(
            "deepseek2.attention.value_length".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.rope.dimension_count".to_string(),
            GgufValue::U32(64),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["qk_nope_head_dim"], 128); // 192 - 64
        assert_eq!(cfg["qk_rope_head_dim"], 64);
        assert_eq!(cfg["v_head_dim"], 128);
    }

    #[test]
    fn test_gguf_mla_fields_absent_for_non_mla_architectures() {
        // Llama / Qwen / Mistral GGUFs do not emit MLA keys. The config
        // builder must leave the optional MLA fields out so `uses_mla()`
        // stays false and the streaming path keeps its existing behaviour.
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );
        metadata.insert(
            "llama.attention.key_length".to_string(),
            GgufValue::U32(128),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();

        assert!(cfg.get("q_lora_rank").is_none());
        assert!(cfg.get("kv_lora_rank").is_none());
        assert!(cfg.get("qk_nope_head_dim").is_none());
        assert!(cfg.get("v_head_dim").is_none());
        assert!(cfg.get("qk_rope_head_dim").is_none());
    }

    #[test]
    fn load_tensors_filtered_skips_key() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("skip.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
                                                      // Tensor 0: kept
        let n0 = b"blk.0.attn_q.weight";
        file.write_all(&(n0.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n0).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        // Tensor 1: skipped by key
        let n1 = b"blk.0.ffn_gate.weight";
        file.write_all(&(n1.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n1).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&16u64.to_le_bytes()).unwrap();
        // Data
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for i in 0..8 {
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let skip: &dyn Fn(&str) -> bool = &|k| k.contains("gate_proj");
        let (tensors, _) = gguf.load_tensors_filtered(skip).unwrap();
        assert_eq!(tensors.len(), 1);
        assert!(tensors.contains_key("layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn load_tensors_handles_1d_and_higher_dim() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dims.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap(); // 2 tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
                                                      // Tensor 0: 1D norm vector (4 elements)
        let n0 = b"blk.0.attn_norm.weight";
        file.write_all(&(n0.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n0).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap(); // 1D
        file.write_all(&4u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        // Tensor 1: 3D tensor (should be skipped)
        let n1 = b"blk.0.expert.weight";
        file.write_all(&(n1.len() as u64).to_le_bytes()).unwrap();
        file.write_all(n1).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // 3D
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&16u64.to_le_bytes()).unwrap();
        // Data
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for i in 0..12 {
            file.write_all(&(i as f32).to_le_bytes()).unwrap();
        }
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let (tensors, vectors) = gguf.load_tensors().unwrap();
        // 1D → vectors map
        assert_eq!(vectors.len(), 1);
        assert!(vectors.contains_key("layers.0.input_layernorm.weight"));
        assert_eq!(vectors["layers.0.input_layernorm.weight"].len(), 4);
        // 3D → skipped (not in tensors or vectors)
        assert!(tensors.is_empty());
    }

    #[test]
    fn to_config_json_head_dim_falls_back_to_hidden_div_heads() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );
        // No attention.key_length → head_dim = hidden / heads = 128

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["head_dim"], 128);
    }

    #[test]
    fn to_config_json_kv_heads_defaults_to_heads_when_absent() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert("llama.embedding_length".to_string(), GgufValue::U32(4096));
        metadata.insert("llama.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert("llama.attention.head_count".to_string(), GgufValue::U32(32));
        // No head_count_kv → defaults to head_count

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["num_key_value_heads"], 32);
    }

    #[test]
    fn to_config_json_maps_deepseek_v4_arch() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek_v4".to_string()),
        );
        metadata.insert(
            "deepseek_v4.embedding_length".to_string(),
            GgufValue::U32(4096),
        );
        metadata.insert("deepseek_v4.block_count".to_string(), GgufValue::U32(32));
        metadata.insert(
            "deepseek_v4.feed_forward_length".to_string(),
            GgufValue::U32(11008),
        );
        metadata.insert(
            "deepseek_v4.attention.head_count".to_string(),
            GgufValue::U32(32),
        );
        metadata.insert(
            "deepseek_v4.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["model_type"], "deepseek_v4");
    }

    #[test]
    fn to_config_json_maps_unknown_arch_passthrough() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("futurearch".to_string()),
        );
        metadata.insert(
            "futurearch.embedding_length".to_string(),
            GgufValue::U32(512),
        );
        metadata.insert("futurearch.block_count".to_string(), GgufValue::U32(6));
        metadata.insert(
            "futurearch.feed_forward_length".to_string(),
            GgufValue::U32(2048),
        );
        metadata.insert(
            "futurearch.attention.head_count".to_string(),
            GgufValue::U32(8),
        );
        metadata.insert(
            "futurearch.attention.head_count_kv".to_string(),
            GgufValue::U32(8),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["model_type"], "futurearch");
    }

    #[test]
    fn test_gguf_to_config_json_falls_back_to_expert_feed_forward_length_on_moe() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(4096),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(43));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(64),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(1),
        );
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(128),
        );
        metadata.insert(
            "deepseek2.expert_feed_forward_length".to_string(),
            GgufValue::U32(2048),
        );
        metadata.insert("deepseek2.vocab_size".to_string(), GgufValue::U32(129280));

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["intermediate_size"], 2048);
        crate::detect_from_json_validated(&cfg)
            .expect("MoE-only GGUF config should pass validation after fallback");
    }

    #[test]
    fn test_gguf_to_config_json_prefers_global_feed_forward_length_when_both_present() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("deepseek2".to_string()),
        );
        metadata.insert(
            "deepseek2.embedding_length".to_string(),
            GgufValue::U32(2048),
        );
        metadata.insert("deepseek2.block_count".to_string(), GgufValue::U32(27));
        metadata.insert(
            "deepseek2.attention.head_count".to_string(),
            GgufValue::U32(16),
        );
        metadata.insert(
            "deepseek2.attention.head_count_kv".to_string(),
            GgufValue::U32(16),
        );
        metadata.insert(
            "deepseek2.attention.key_length".to_string(),
            GgufValue::U32(192),
        );
        metadata.insert(
            "deepseek2.feed_forward_length".to_string(),
            GgufValue::U32(10944),
        );
        metadata.insert(
            "deepseek2.expert_feed_forward_length".to_string(),
            GgufValue::U32(1408),
        );

        let gguf = GgufFile {
            metadata,
            tensor_infos: Vec::new(),
            data_offset: 0,
            path: std::path::PathBuf::from("<no-file>"),
            shards: vec![ShardInfo {
                path: std::path::PathBuf::from("<no-file>"),
                data_offset: 0,
            }],
        };
        let cfg = gguf.to_config_json();
        assert_eq!(cfg["intermediate_size"], 10944);
    }

    /// Build a minimal GGUF file with one 2-D F32 tensor, but truncate the
    /// tensor data region so that `offset + size > file len`. Loader must
    /// reject this cleanly, not panic on a slice OOB.
    #[test]
    fn test_load_tensors_rejects_truncated_tensor_data() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.gguf");
        let mut file = std::fs::File::create(&path).unwrap();

        // Header
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap(); // version
        file.write_all(&1u64.to_le_bytes()).unwrap(); // n_tensors
        file.write_all(&0u64.to_le_bytes()).unwrap(); // n_metadata

        // Tensor info: declares 2x4 F32 (32 bytes of data) at tensor offset 0.
        let name = b"blk.0.ffn_down.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();

        // Pad to 32-byte boundary, then write only 16 bytes of tensor data
        // (half of the declared 32). Loader must detect the shortfall.
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        file.write_all(&[0u8; 16]).unwrap();
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        match gguf.load_tensors() {
            Err(ModelError::Parse(msg)) => {
                assert!(
                    msg.contains("out of bounds") || msg.contains("too short"),
                    "unexpected error: {msg}"
                );
            }
            Err(other) => panic!("expected Parse error, got {other:?}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }

    // Dequant tests are in format::quant::ggml::tests

    #[test]
    fn read_tokenizer_vocab_size_reads_vocab_object_length() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        let tokenizer_json = serde_json::json!({
            TOKENIZER_MODEL: {
                TOKENIZER_VOCAB: {
                    "<unk>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "a": 3,
                    "b": 4,
                }
            }
        });
        std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), Some(5));
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_tokenizer_json_absent() {
        let dir = tempfile::TempDir::new().unwrap();
        // model.gguf path with no tokenizer.json next to it.
        assert_eq!(
            read_tokenizer_vocab_size(&dir.path().join("model.gguf")),
            None
        );
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_vocab_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        // Empty vocab object — filtered out by `.filter(|&v| v > 0)`.
        let tokenizer_json = serde_json::json!({
            TOKENIZER_MODEL: {
                TOKENIZER_VOCAB: {}
            }
        });
        std::fs::write(dir.path().join(TOKENIZER_JSON), tokenizer_json.to_string()).unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), None);
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_on_malformed_json() {
        let dir = tempfile::TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        std::fs::write(dir.path().join(TOKENIZER_JSON), b"not-json").unwrap();
        assert_eq!(read_tokenizer_vocab_size(&gguf), None);
    }

    #[test]
    fn read_tokenizer_vocab_size_returns_none_when_path_has_no_parent() {
        assert_eq!(read_tokenizer_vocab_size(std::path::Path::new("")), None);
    }
}
