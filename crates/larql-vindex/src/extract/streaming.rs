//! Streaming vindex extraction — build from safetensors without loading the full model.
//!
//! Instead of loading all weights into ModelWeights (which requires the entire model
//! in RAM), this module mmaps safetensors files and processes one layer at a time.
//! Peak memory = 1 layer's tensors + embeddings, not the full model.
//!
//! For a 120B MoE model: ~120 GB as ModelWeights vs ~2 GB streaming.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::config::dtype::StorageDtype;
use crate::config::{VindexConfig, VindexLayerInfo, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;

/// Mmap'd safetensors file — kept alive for the duration of extraction.
struct MmapShard {
    _file: std::fs::File,
    mmap: memmap2::Mmap,
}

/// Build a vindex by streaming from safetensors files (no full model load).
///
/// Peak memory: embeddings + 1 layer of gate/down weights at a time.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex_streaming(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    std::fs::create_dir_all(output_dir)?;

    // Detect architecture
    let arch = larql_models::detect_architecture(model_dir)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let prefixes = arch.key_prefixes_to_strip();
    let cfg = arch.config();
    let mlx_affine_group_size =
        larql_models::loading::safetensors::mlx_affine_group_size_from_config(model_dir);

    let num_layers = cfg.num_layers;
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg.intermediate_size;
    let embed_scale = arch.embed_scale();
    let is_moe = arch.is_moe();
    let n_experts = arch.num_experts();

    // Mmap all safetensors files
    let mut st_files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    if st_files.is_empty() {
        let weights_dir = model_dir.join("weights");
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
        return Err(VindexError::NoSafetensors(model_dir.to_path_buf()));
    }

    callbacks.on_stage("loading");
    eprintln!(
        "  Streaming mode: {} safetensors shards (mmap'd, not loaded)",
        st_files.len()
    );

    // (shards vec was for an earlier design — tensor_index + shard_mmaps is the actual approach)

    // SAFETY: We need to hold both the mmap and the SafeTensors that borrows from it.
    // We use a two-phase approach: first mmap all files, then deserialize.
    // The mmaps are kept alive in `shard_mmaps` for the lifetime of the function.
    let shard_mmaps: Vec<MmapShard> = st_files
        .iter()
        .map(|path| {
            let file = std::fs::File::open(path).unwrap();
            let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
            MmapShard { _file: file, mmap }
        })
        .collect();

    // Build a tensor index: key → (shard_idx, tensor_name)
    // We need to find which shard contains each tensor.
    let mut tensor_index: HashMap<String, (usize, String)> = HashMap::new();
    for (shard_idx, shard) in shard_mmaps.iter().enumerate() {
        let st = safetensors::SafeTensors::deserialize(&shard.mmap)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        for name in st.names() {
            let key = normalize_key(name, prefixes);
            tensor_index.insert(key.clone(), (shard_idx, name.to_string()));
        }
    }

    callbacks.on_stage_done("loading", 0.0);

    // ── 1. Gate vectors (streaming, one layer at a time) ──
    callbacks.on_stage("gate_vectors");
    let gate_path = output_dir.join("gate_vectors.bin");
    let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
    let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
    let mut offset: u64 = 0;

    // Check expert format from the architecture
    let expert_format = arch.expert_format();

    for layer in 0..num_layers {
        callbacks.on_layer_start("gate", layer, num_layers);
        let start = std::time::Instant::now();

        if expert_format == larql_models::ExpertFormat::PackedMxfp4 {
            // MXFP4 packed experts: dequantize gate_up_proj_blocks per layer
            // The fused tensor is [num_experts, 2*intermediate, groups, 16]
            // First half of output features = gate, second half = up
            let blocks_key = arch.packed_gate_up_blocks_key(layer).unwrap_or_default();
            let scales_key = arch.packed_gate_up_scales_key(layer).unwrap_or_default();

            if let (Some(blocks_info), Some(scales_info)) =
                (tensor_index.get(&blocks_key), tensor_index.get(&scales_key))
            {
                let blocks_st =
                    safetensors::SafeTensors::deserialize(&shard_mmaps[blocks_info.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;
                let scales_st =
                    safetensors::SafeTensors::deserialize(&shard_mmaps[scales_info.0].mmap)
                        .map_err(|e| VindexError::Parse(e.to_string()))?;

                let blocks_view = blocks_st
                    .tensor(&blocks_info.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let scales_view = scales_st
                    .tensor(&scales_info.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;

                let shape = blocks_view.shape();
                let n_exp = shape[0];
                let out_features = shape[1]; // 2 * intermediate (fused gate+up)
                let groups = shape[2];
                let in_features = groups * 32;
                let half = out_features / 2; // gate portion

                let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                    blocks_view.data(),
                    scales_view.data(),
                    n_exp,
                    out_features,
                    groups,
                );

                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;

                for expert_data in &experts {
                    // Extract gate portion (first half rows)
                    let gate_data = &expert_data[..half * in_features];
                    layer_bytes += write_floats(&mut gate_file, gate_data, dtype)?;
                    total_features += half;
                }

                if total_features > 0 {
                    layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features: total_features,
                        offset,
                        length: layer_bytes,
                        num_experts: Some(n_exp),
                        num_features_per_expert: Some(half),
                    });
                    offset += layer_bytes;
                }
            }
        } else if is_moe && n_experts > 0 {
            // Standard MoE (Mixtral): per-expert gate tensors
            let mut total_features = 0usize;
            let mut layer_bytes = 0u64;
            let mut features_per_expert = 0usize;

            for expert in 0..n_experts {
                let gate_key = match arch.expert_ffn_gate_key(layer, expert) {
                    Some(k) => normalize_key(&k, prefixes),
                    None => continue,
                };

                if let Some(tensor) = get_tensor_f32(
                    &shard_mmaps,
                    &tensor_index,
                    &gate_key,
                    mlx_affine_group_size,
                )? {
                    features_per_expert = tensor.shape()[0];
                    total_features += features_per_expert;
                    let data = tensor.as_slice().unwrap();
                    layer_bytes += write_floats(&mut gate_file, data, dtype)?;
                }
            }

            if total_features > 0 {
                layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features: total_features,
                    offset,
                    length: layer_bytes,
                    num_experts: Some(n_experts),
                    num_features_per_expert: Some(features_per_expert),
                });
                offset += layer_bytes;
            }
        } else {
            // Dense: single gate matrix per layer
            let gate_key = normalize_key(&arch.ffn_gate_key(layer), prefixes);
            if let Some(tensor) = get_tensor_f32(
                &shard_mmaps,
                &tensor_index,
                &gate_key,
                mlx_affine_group_size,
            )? {
                let num_features = tensor.shape()[0];
                let data = tensor.as_slice().unwrap();
                let length = write_floats(&mut gate_file, data, dtype)?;
                layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features,
                    offset,
                    length,
                    num_experts: None,
                    num_features_per_expert: None,
                });
                offset += length;
            }
        }

        callbacks.on_layer_done("gate", layer, start.elapsed().as_secs_f64() * 1000.0);
    }
    gate_file.flush()?;
    callbacks.on_stage_done("gate_vectors", 0.0);

    // ── 1b. Router weights (MoE models only) ──
    if is_moe {
        callbacks.on_stage("router_weights");
        let router_path = output_dir.join("router_weights.bin");
        let mut router_file = BufWriter::new(std::fs::File::create(&router_path)?);

        for layer in 0..num_layers {
            let router_key = arch
                .moe_router_key(layer)
                .map(|k| normalize_key(&k, prefixes))
                .unwrap_or_default();

            if let Some(tensor) = get_tensor_f32(
                &shard_mmaps,
                &tensor_index,
                &router_key,
                mlx_affine_group_size,
            )? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, dtype);
                router_file.write_all(&bytes)?;
            }

            // Also try router bias
            let bias_key = router_key.replace(".weight", ".bias");
            if let Some(tensor) = get_tensor_f32(
                &shard_mmaps,
                &tensor_index,
                &bias_key,
                mlx_affine_group_size,
            )? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, dtype);
                // Write bias after weight for each layer
                router_file.write_all(&bytes)?;
            }
        }
        router_file.flush()?;
        callbacks.on_stage_done("router_weights", 0.0);
    }

    // ── 2. Embeddings ──
    callbacks.on_stage("embeddings");
    let embed_key = normalize_key(arch.embed_key(), prefixes);
    let embed = get_tensor_f32(
        &shard_mmaps,
        &tensor_index,
        &embed_key,
        mlx_affine_group_size,
    )?
    .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?;
    let vocab_size = embed.shape()[0];
    let embed_data = embed.as_slice().unwrap();
    let embed_bytes = crate::config::dtype::encode_floats(embed_data, dtype);
    std::fs::write(output_dir.join("embeddings.bin"), &embed_bytes)?;
    callbacks.on_stage_done("embeddings", 0.0);

    // ── 3. Down meta (streaming) ──
    callbacks.on_stage("down_meta");
    let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> = vec![None; num_layers];

    // Build whole-word vocab once
    let (_ww_ids, _ww_embed) =
        super::build::build_whole_word_vocab(tokenizer, &embed, vocab_size, hidden_size);

    for (layer, layer_down_meta) in all_down_meta.iter_mut().enumerate().take(num_layers) {
        callbacks.on_layer_start("down", layer, num_layers);
        let start = std::time::Instant::now();

        // Get down matrices for this layer
        let down_matrices: Vec<Array2<f32>> = if expert_format
            == larql_models::ExpertFormat::PackedMxfp4
        {
            // MXFP4: dequantize down_proj_blocks
            let blocks_key = arch.packed_down_blocks_key(layer).unwrap_or_default();
            let scales_key = arch.packed_down_scales_key(layer).unwrap_or_default();
            if let (Some(bi), Some(si)) =
                (tensor_index.get(&blocks_key), tensor_index.get(&scales_key))
            {
                let bst = safetensors::SafeTensors::deserialize(&shard_mmaps[bi.0].mmap)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let sst = safetensors::SafeTensors::deserialize(&shard_mmaps[si.0].mmap)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let bv = bst
                    .tensor(&bi.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let sv = sst
                    .tensor(&si.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let shape = bv.shape();
                let n_exp = shape[0];
                let out_features = shape[1];
                let groups = shape[2];
                let in_features = groups * 32;
                let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                    bv.data(),
                    sv.data(),
                    n_exp,
                    out_features,
                    groups,
                );
                experts
                    .into_iter()
                    .map(|data| Array2::from_shape_vec((out_features, in_features), data).unwrap())
                    .collect()
            } else {
                callbacks.on_layer_done("down", layer, 0.0);
                continue;
            }
        } else if is_moe && n_experts > 0 {
            let mut mats = Vec::new();
            for expert in 0..n_experts {
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    let nk = normalize_key(&key, prefixes);
                    if let Some(t) =
                        get_tensor_f32(&shard_mmaps, &tensor_index, &nk, mlx_affine_group_size)?
                    {
                        mats.push(t);
                    }
                }
            }
            mats
        } else {
            let down_key = normalize_key(&arch.ffn_down_key(layer), prefixes);
            match get_tensor_f32(
                &shard_mmaps,
                &tensor_index,
                &down_key,
                mlx_affine_group_size,
            )? {
                Some(t) => vec![t],
                None => {
                    callbacks.on_layer_done("down", layer, 0.0);
                    continue;
                }
            }
        };

        if down_matrices.is_empty() {
            callbacks.on_layer_done("down", layer, 0.0);
            continue;
        }

        let mut feature_offset = 0usize;
        for w_down in &down_matrices {
            let num_features = w_down.shape()[1];
            let batch_size = 1024;

            for batch_start in (0..num_features).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_features);
                callbacks.on_feature_progress(
                    "down",
                    layer,
                    feature_offset + batch_start,
                    down_matrices.iter().map(|m| m.shape()[1]).sum(),
                );

                let w_chunk = w_down
                    .slice(ndarray::s![.., batch_start..batch_end])
                    .to_owned();
                let cpu = larql_compute::CpuBackend;
                use larql_compute::ComputeBackend;
                let chunk_logits = cpu.matmul(embed.view(), w_chunk.view());

                for feat in batch_start..batch_end {
                    let col = chunk_logits.column(feat - batch_start);
                    let mut scores: Vec<(usize, f32)> = col.iter().copied().enumerate().collect();
                    let k = down_top_k.min(scores.len());
                    if k > 0 && k < scores.len() {
                        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                    }
                    scores.truncate(k);
                    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    let top_k_entries: Vec<larql_models::TopKEntry> = scores
                        .into_iter()
                        .filter_map(|(idx, logit)| {
                            tokenizer
                                .decode(&[idx as u32], true)
                                .ok()
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .map(|token| larql_models::TopKEntry {
                                    token,
                                    token_id: idx as u32,
                                    logit,
                                })
                        })
                        .collect();

                    let (top_token, top_token_id, c_score) =
                        if let Some(first) = top_k_entries.first() {
                            (first.token.clone(), first.token_id, first.logit)
                        } else {
                            (String::new(), 0, 0.0)
                        };

                    let feat_idx = feature_offset + feat;
                    if layer_down_meta.is_none() {
                        *layer_down_meta = Some(Vec::new());
                    }
                    if let Some(ref mut metas) = layer_down_meta {
                        while metas.len() <= feat_idx {
                            metas.push(None);
                        }
                        metas[feat_idx] = Some(crate::FeatureMeta {
                            top_token,
                            top_token_id,
                            c_score,
                            top_k: top_k_entries,
                        });
                    }
                }
            }
            feature_offset += num_features;
        }

        callbacks.on_layer_done("down", layer, start.elapsed().as_secs_f64() * 1000.0);
    }

    crate::format::down_meta::write_binary(output_dir, &all_down_meta, down_top_k)?;
    callbacks.on_stage_done("down_meta", 0.0);

    // ── 4. Tokenizer ──
    callbacks.on_stage("tokenizer");
    let tokenizer_json = tokenizer
        .to_string(true)
        .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
    std::fs::write(output_dir.join("tokenizer.json"), tokenizer_json)?;
    callbacks.on_stage_done("tokenizer", 0.0);

    // ── 5. Config ──
    let family = arch.family().to_string();
    let config = VindexConfig {
        version: 2,
        model: model_name.to_string(),
        family: family.clone(),
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        embed_scale,
        layers: layer_infos,
        down_top_k,
        has_model_weights: false,
        source: Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: super::build::chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        checksums: None,
        extract_level,
        dtype,
        layer_bands: crate::LayerBands::for_family(&family, num_layers),
        model_config: Some(VindexModelConfig {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: if is_moe {
                Some(crate::MoeConfig {
                    num_experts: n_experts,
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    router_type: "top_k_softmax".to_string(),
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
        }),
    };

    // Write preliminary index.json (needed by write_model_weights which reads dtype from it)
    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("index.json"), config_json)?;

    // ── 6. Model weights (if extract level requires them) ──
    if extract_level != crate::ExtractLevel::Browse {
        let shard_refs: Vec<&[u8]> = shard_mmaps.iter().map(|s| s.mmap.as_ref()).collect();
        let streaming_source = crate::format::weights::StreamingWeights {
            shard_mmaps: &shard_refs,
            tensor_index: &tensor_index,
            arch: &*arch,
            num_layers,
            mlx_affine_group_size,
        };
        crate::format::weights::write_model_weights(&streaming_source, output_dir, callbacks)?;
        // write_model_weights updates index.json with has_model_weights=true
    }

    // Final checksums
    let config_text = std::fs::read_to_string(output_dir.join("index.json"))?;
    let mut config: VindexConfig =
        serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;
    config.checksums = crate::format::checksums::compute_checksums(output_dir).ok();
    let config_json =
        serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("index.json"), config_json)?;

    Ok(())
}

/// Get a 2D tensor from mmap'd safetensors, dequantizing to f32.
fn get_tensor_f32(
    shards: &[MmapShard],
    index: &HashMap<String, (usize, String)>,
    key: &str,
    mlx_affine_group_size: Option<usize>,
) -> Result<Option<Array2<f32>>, VindexError> {
    let (shard_idx, tensor_name) = match index.get(key) {
        Some(v) => v,
        None => return Ok(None),
    };

    let st = safetensors::SafeTensors::deserialize(&shards[*shard_idx].mmap)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let view = st
        .tensor(tensor_name)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Ok(None);
    }

    let data = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
        safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
        safetensors::Dtype::U32 => {
            let Some(group_size) = mlx_affine_group_size else {
                return Ok(None);
            };
            let Some((data, cols)) =
                dequantize_mlx_affine_tensor(shards, index, key, shape, view.data(), group_size)?
            else {
                return Ok(None);
            };
            let arr = Array2::from_shape_vec((shape[0], cols), data)
                .map_err(|e| VindexError::Parse(e.to_string()))?;
            return Ok(Some(arr));
        }
        _ => return Ok(None), // skip non-float
    };

    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    Ok(Some(arr))
}

fn dequantize_mlx_affine_tensor(
    shards: &[MmapShard],
    index: &HashMap<String, (usize, String)>,
    key: &str,
    weight_shape: &[usize],
    packed: &[u8],
    group_size: usize,
) -> Result<Option<(Vec<f32>, usize)>, VindexError> {
    let Some(stem) = key.strip_suffix(".weight") else {
        return Ok(None);
    };

    let scales_key = format!("{stem}.scales");
    let biases_key = format!("{stem}.biases");

    let (scales_shard, scales_name) = match index.get(&scales_key) {
        Some(v) => v,
        None => return Ok(None),
    };
    let scales_st = safetensors::SafeTensors::deserialize(&shards[*scales_shard].mmap)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let scales_view = scales_st
        .tensor(scales_name)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let scales = tensor_view_to_f32(&scales_view)?;

    let biases = if let Some((bias_shard, bias_name)) = index.get(&biases_key) {
        let biases_st = safetensors::SafeTensors::deserialize(&shards[*bias_shard].mmap)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        let biases_view = biases_st
            .tensor(bias_name)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        Some(tensor_view_to_f32(&biases_view)?)
    } else {
        None
    };

    let (data, cols) = crate::format::quant::mlx_affine::dequantize_u32_matrix_bytes(
        packed,
        weight_shape[0],
        weight_shape[1],
        &scales,
        biases.as_deref(),
        group_size,
    )
    .map_err(VindexError::Parse)?;

    Ok(Some((data, cols)))
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

fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

fn write_floats(w: &mut impl Write, data: &[f32], dtype: StorageDtype) -> Result<u64, VindexError> {
    let bytes = crate::config::dtype::encode_floats(data, dtype);
    w.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}
