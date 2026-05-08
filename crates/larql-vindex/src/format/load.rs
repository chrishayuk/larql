//! Binary loading path for .vindex directories.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Seek};
use std::path::Path;

use ndarray::Array2;

use crate::config::VindexConfig;
use crate::error::VindexError;
use crate::format::filenames::{
    DOWN_META_BIN, EMBEDDINGS_BIN, GATE_VECTORS_BIN, GGUF_DOWN_META_MANIFEST_JSON,
    GGUF_EMBEDDINGS_MANIFEST_JSON, GGUF_GATE_MANIFEST_JSON, INDEX_JSON, INTERLEAVED_Q4K_BIN,
    INTERLEAVED_Q4K_MANIFEST_JSON, LM_HEAD_BIN, LM_HEAD_Q4_BIN, TOKENIZER_JSON,
};
use crate::index::{IndexLoadCallbacks, VectorIndex};

impl VectorIndex {
    /// Load a VectorIndex from a .vindex directory.
    ///
    /// Reads gate_vectors.bin (mmap'd), down_meta.jsonl, and index.json.
    /// The embeddings and tokenizer are loaded separately via `load_vindex_embeddings`.
    pub fn load_vindex(
        dir: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        Self::load_vindex_with_range(dir, callbacks, None)
    }

    /// Load a VectorIndex restricted to a layer range `(start, end)` where
    /// `start` is inclusive and `end` is exclusive.
    ///
    /// Use this on layer-sharded servers to avoid allocating or touching mmap
    /// pages for layers outside the owned range. The full vindex files are
    /// still mmap'd (cheap — virtual address space only), but:
    /// - `synthesize_gate_from_q4k` only dequantizes owned layers, so the
    ///   anonymous allocation shrinks proportionally.
    /// - `is_layer_owned(layer)` returns false for out-of-range layers,
    ///   letting callers reject requests before touching any pages.
    pub fn load_vindex_with_range(
        dir: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
        layer_range: Option<(usize, usize)>,
    ) -> Result<Self, VindexError> {
        // Read config
        let config_path = dir.join(INDEX_JSON);
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: VindexConfig =
            serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;

        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;

        // Load gate vectors from binary. If `gate_vectors.bin` is
        // missing but `interleaved_q4k.bin` is present, synthesize an
        // anonymous mmap by dequantizing the Q4K gate slices at f16 —
        // that's dedup #2 in action (a Q4K vindex extracted with
        // `--drop-gate-vectors` carries gate weights only once, Q4K).
        let gate_path = dir.join(GATE_VECTORS_BIN);
        let interleaved_q4k_path = dir.join(INTERLEAVED_Q4K_BIN);

        let (gate_mmap, gate_slices, gate_dtype) = if gate_path.exists() {
            callbacks.on_file_start("gate_vectors", &gate_path.display().to_string());
            let start = std::time::Instant::now();
            let gate_file = std::fs::File::open(&gate_path)?;
            // Demand-paged: gate_vectors are large and only a fraction of
            // pages are touched per token (HNSW path) or scanned sequentially
            // once per query (linear path). MADV_WILLNEED would prefault the
            // entire file into RAM at load time, inflating RSS by ~13 GB on
            // 31B before any inference runs.
            let gate_mmap = unsafe { crate::mmap_util::mmap_demand_paged(&gate_file)? };
            let bpf = crate::config::dtype::bytes_per_float(config.dtype);

            let mut gate_slices: Vec<crate::index::core::GateLayerSlice> = vec![
                crate::index::core::GateLayerSlice { float_offset: 0, num_features: 0 };
                num_layers
            ];
            let mut total_gate = 0;
            for info in &config.layers {
                gate_slices[info.layer] = crate::index::core::GateLayerSlice {
                    float_offset: info.offset as usize / bpf,
                    num_features: info.num_features,
                };
                total_gate += info.num_features;
            }
            callbacks.on_file_done(
                "gate_vectors",
                total_gate,
                start.elapsed().as_secs_f64() * 1000.0,
            );
            (gate_mmap, gate_slices, config.dtype)
        } else if interleaved_q4k_path.exists() {
            callbacks.on_file_start(
                "gate_vectors (synth from Q4K)",
                &interleaved_q4k_path.display().to_string(),
            );
            let start = std::time::Instant::now();
            let (gate_mmap, gate_slices) =
                synthesize_gate_from_q4k(dir, &config, hidden_size, layer_range)?;
            let total: usize = gate_slices.iter().map(|s| s.num_features).sum();
            callbacks.on_file_done(
                "gate_vectors (synth from Q4K)",
                total,
                start.elapsed().as_secs_f64() * 1000.0,
            );
            (
                gate_mmap,
                gate_slices,
                crate::config::dtype::StorageDtype::F16,
            )
        } else {
            // Neither gate_vectors.bin nor interleaved_q4k.bin present.
            // This is the attention-only client-side slice (produced by
            // `larql slice --preset client`): the client runs attention
            // locally and delegates gate-KNN + FFN to the remote server
            // via `--ffn URL`, so it genuinely does not need gate data.
            // Hand back an empty gate mmap + all-zero slices. `gate_knn`
            // returns an empty result on this index, which is the correct
            // behaviour for an attention-only client — nothing calls it.
            callbacks.on_file_start(
                "gate_vectors (absent — client-only slice)",
                &dir.display().to_string(),
            );
            let empty = memmap2::MmapMut::map_anon(0)?.make_read_only()?;
            let gate_slices: Vec<crate::index::core::GateLayerSlice> = vec![
                crate::index::core::GateLayerSlice { float_offset: 0, num_features: 0 };
                num_layers
            ];
            callbacks.on_file_done("gate_vectors (absent — client-only slice)", 0, 0.0);
            (empty, gate_slices, crate::config::dtype::StorageDtype::F16)
        };

        // Load down metadata — mmap binary (zero heap), fall back to JSONL (legacy)
        let start = std::time::Instant::now();

        let down_meta_mmap = if crate::format::down_meta::has_binary(dir) {
            match load_vindex_tokenizer(dir) {
                Ok(tokenizer) => {
                    callbacks
                        .on_file_start("down_meta", &dir.join(DOWN_META_BIN).display().to_string());
                    let tok = std::sync::Arc::new(tokenizer);
                    match crate::format::down_meta::mmap_binary(dir, tok) {
                        Ok(dm) => {
                            let count = dm.total_features();
                            callbacks.on_file_done(
                                "down_meta",
                                count,
                                start.elapsed().as_secs_f64() * 1000.0,
                            );
                            Some(dm)
                        }
                        Err(_) => None,
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        let mut index = VectorIndex::new_mmap(
            gate_mmap,
            gate_slices,
            gate_dtype,
            down_meta_mmap,
            num_layers,
            hidden_size,
        );

        // Propagate `vocab_size` from index.json. Previously this only got
        // set inside the embeddings-as-tied-lm_head adoption block below,
        // so a vindex with `lm_head_q4.bin` but no `lm_head.bin` ended up
        // with `vocab_size = 0` — silently disabling the Q4 lm_head path
        // (4× slower fallback to the f32 BLAS gemv).
        if config.vocab_size > 0 {
            index.vocab_size = config.vocab_size;
        }

        let gguf_down_meta_manifest_path = dir.join(GGUF_DOWN_META_MANIFEST_JSON);
        if gguf_down_meta_manifest_path.exists() {
            let manifest_text = std::fs::read_to_string(&gguf_down_meta_manifest_path)?;
            let manifest: crate::index::storage::metadata_store::GgufDownMetaManifest =
                serde_json::from_str(&manifest_text)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
            index.metadata.gguf_down_meta_manifest = Some(std::sync::Arc::new(manifest));
        }
        let gguf_gate_manifest_path = dir.join(GGUF_GATE_MANIFEST_JSON);
        if gguf_gate_manifest_path.exists() {
            let manifest_text = std::fs::read_to_string(&gguf_gate_manifest_path)?;
            let manifest: crate::index::storage::metadata_store::GgufGateManifest =
                serde_json::from_str(&manifest_text)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
            index.metadata.gguf_gate_manifest = Some(std::sync::Arc::new(manifest));
        }

        // Opportunistically wire up FFN payload mmaps so walk_ffn_sparse can
        // find up/down data without callers needing to know which flavour
        // is on disk. Each load_* returns Err(_) if its file isn't present;
        // those errors are non-fatal here.
        if let Some(range) = layer_range {
            index.set_layer_range(range);
        }

        let _ = index.load_interleaved_q4k(dir);
        let _ = index.load_interleaved_q4(dir);
        let _ = index.load_interleaved(dir);
        let _ = index.load_up_features(dir);
        let _ = index.load_down_features(dir);
        // W2: feature-major Q4_K down. Optional file; when present the
        // CPU sparse walk skips the `q4k_ffn_layer` cache for component=2.
        let _ = index.load_down_features_q4k(dir);
        // Opt-in FP4/FP8 storage (exp 26): present iff `index.json.fp4`
        // is set. Non-fatal if absent or malformed — other FFN mmaps
        // already loaded remain authoritative.
        let _ = index.load_fp4_storage(dir, &config);

        // Engine observability: emit the walk-kernel backend summary
        // to stderr when `LARQL_VINDEX_DESCRIBE=1`. Lets users spot
        // silent fallbacks (e.g. FP4 vindex wired as "weights fallback"
        // would have prevented the exp 26 Q2 bug if this had existed).
        if std::env::var("LARQL_VINDEX_DESCRIBE").ok().as_deref() == Some("1") {
            eprintln!(
                "[larql-vindex] {} → walk backend: {}",
                dir.display(),
                index.describe_ffn_backend(),
            );
        }
        // Opportunistically adopt the f16 `embeddings.bin` as an f16 view
        // of the LM head — but ONLY when the vindex has no separate lm_head
        // file. `embeddings.bin` IS the lm_head for tied-embedding models
        // (Gemma 2/3/4, Llama with `tie_word_embeddings=true`). For untied
        // models the two matrices differ, so adopting embed here would
        // make `lm_head_knn_backend` return wrong logits.
        //
        // Gate: file is f16-sized AND neither `lm_head.bin` nor
        // `lm_head_q4.bin` is present in the vindex directory. The
        // untied models that ship those files are always extracted with
        // one of them, so presence is a reliable untied-signal.
        let has_separate_lm_head =
            dir.join(LM_HEAD_BIN).exists() || dir.join(LM_HEAD_Q4_BIN).exists();
        if !has_separate_lm_head {
            if let Ok(f) = std::fs::File::open(dir.join(EMBEDDINGS_BIN)) {
                if let Ok(mmap) = unsafe { memmap2::Mmap::map(&f) } {
                    let expected_f16 = config.vocab_size * config.hidden_size * 2;
                    if mmap.len() >= expected_f16 && mmap.len() < expected_f16 * 2 {
                        if index.vocab_size == 0 {
                            index.vocab_size = config.vocab_size;
                        }
                        index.set_lm_head_f16_mmap(std::sync::Arc::new(mmap));
                        index.synthesize_lm_head_q4();
                    }
                }
            }
        }

        Ok(index)
    }
}

/// Dequantize gate slices from `interleaved_q4k.bin` into an anonymous
/// f16 mmap shaped like a real `gate_vectors.bin` file. Used when a
/// Q4K vindex was extracted with `--drop-gate-vectors`.
///
/// Layout matches `gate_vectors.bin` so the rest of the gate-mmap
/// accessors (`gate_vectors_at`, `gate_knn`, …) work unchanged.
fn synthesize_gate_from_q4k(
    dir: &Path,
    config: &VindexConfig,
    hidden_size: usize,
    layer_range: Option<(usize, usize)>,
) -> Result<(memmap2::Mmap, Vec<crate::index::core::GateLayerSlice>), VindexError> {
    let interleaved_path = dir.join(INTERLEAVED_Q4K_BIN);
    let manifest_path = dir.join(INTERLEAVED_Q4K_MANIFEST_JSON);
    if !manifest_path.exists() {
        return Err(VindexError::Parse(format!(
            "interleaved_q4k_manifest.json missing alongside {}",
            interleaved_path.display()
        )));
    }
    // Open the Q4K file and the manifest.
    let iq4_file = std::fs::File::open(&interleaved_path)?;
    let iq4_mmap = unsafe { crate::mmap_util::mmap_optimized(&iq4_file)? };
    let manifest_json: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

    let num_layers = config.num_layers;
    // Allocate one anon MmapMut sized for owned layers only (f16, 2 bytes/float).
    // When layer_range is set, unowned layers get a zero GateLayerSlice and are
    // never accessed (is_layer_owned guard in callers). This shrinks the
    // allocation proportionally — a 1/3-shard uses 1/3 the anon memory.
    let is_owned = |layer: usize| -> bool {
        match layer_range {
            None => true,
            Some((start, end)) => layer >= start && layer < end,
        }
    };
    let mut byte_offset: u64 = 0;
    let mut gate_slices = vec![
        crate::index::core::GateLayerSlice {
            float_offset: 0,
            num_features: 0
        };
        num_layers
    ];
    for info in &config.layers {
        if !is_owned(info.layer) {
            continue;
        }
        gate_slices[info.layer] = crate::index::core::GateLayerSlice {
            // Offset measured in floats (f16 → bpf=2).
            float_offset: (byte_offset as usize) / 2,
            num_features: info.num_features,
        };
        byte_offset += (info.num_features as u64) * (hidden_size as u64) * 2;
    }
    let total_bytes = byte_offset as usize;

    let mut anon = memmap2::MmapMut::map_anon(total_bytes)
        .map_err(|e| VindexError::Parse(format!("anon mmap: {e}")))?;

    for info in &config.layers {
        if !is_owned(info.layer) {
            continue;
        }
        // Manifest entries per layer are [gate, up, down] in order.
        let base = info.layer * 3;
        let gate_entry = manifest_json.get(base).ok_or_else(|| {
            VindexError::Parse(format!(
                "q4k manifest missing gate entry for layer {}",
                info.layer
            ))
        })?;
        let offset = gate_entry["offset"].as_u64().unwrap_or(0) as usize;
        let length = gate_entry["length"].as_u64().unwrap_or(0) as usize;
        let format = gate_entry["format"].as_str().ok_or_else(|| {
            VindexError::Parse(format!(
                "interleaved_q4k_manifest gate entry at layer {} missing `format`",
                info.layer
            ))
        })?;
        // Route through the registry so a future Q6_K (or other K-quant)
        // gate slice would dequantise the same way without another
        // string-compare here.
        let format_info = crate::quant::registry::lookup(format).ok_or_else(|| {
            VindexError::Parse(format!(
                "interleaved_q4k_manifest layer {}: unknown format tag {format:?}",
                info.layer
            ))
        })?;
        let end = offset.checked_add(length).ok_or_else(|| {
            VindexError::Parse(format!(
                "interleaved_q4k_manifest layer {}: offset+length overflow ({offset}+{length})",
                info.layer
            ))
        })?;
        if end > iq4_mmap.len() {
            return Err(VindexError::Parse(format!(
                "interleaved_q4k_manifest layer {}: gate slice {offset}..{end} exceeds mmap length {}",
                info.layer,
                iq4_mmap.len()
            )));
        }
        let q_bytes = &iq4_mmap[offset..end];
        let n = info.num_features * hidden_size;
        let padded = n.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
            * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        let gate_f32 = (format_info.dequantize)(q_bytes, padded)
            .map_err(|e| VindexError::Parse(format!("dequantize layer {}: {e}", info.layer)))?;
        let gate_f16_bytes = larql_models::quant::half::encode_f16(&gate_f32[..n]);

        // Copy into the anon mmap at the right byte offset.
        let slot_byte_offset = gate_slices[info.layer].float_offset * 2;
        let dst = &mut anon[slot_byte_offset..slot_byte_offset + gate_f16_bytes.len()];
        dst.copy_from_slice(&gate_f16_bytes);
    }

    let mmap = anon
        .make_read_only()
        .map_err(|e| VindexError::Parse(format!("make_read_only: {e}")))?;
    Ok((mmap, gate_slices))
}

/// Load embeddings from a .vindex directory.
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), VindexError> {
    let config_text = std::fs::read_to_string(dir.join(INDEX_JSON))?;
    let config: VindexConfig =
        serde_json::from_str(&config_text).map_err(|e| VindexError::Parse(e.to_string()))?;

    let embed_file = std::fs::File::open(dir.join(EMBEDDINGS_BIN))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    // Detect actual dtype from file size (may differ from index.json global dtype
    // if gate vectors were converted to f32 but embeddings remain f16).
    let expected_f32 = config.vocab_size * config.hidden_size * 4;
    let actual_dtype = if embed_mmap.len() == expected_f32 {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, actual_dtype);

    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    Ok((embed, config.embed_scale))
}

#[derive(serde::Deserialize)]
struct GgufEmbeddingsManifestForLoad {
    source_file: String,
    tensor_type: u32,
    vocab_size: usize,
    hidden_size: usize,
    tensor_offset: u64,
    data_offset: u64,
}

pub fn load_vindex_embedding_rows(
    dir: &Path,
    token_ids: &[u32],
) -> Result<(Array2<f32>, f32), VindexError> {
    let config = load_vindex_config(dir)?;
    if dir.join(EMBEDDINGS_BIN).exists() {
        let (embed, scale) = load_vindex_embeddings(dir)?;
        let mut rows = Vec::with_capacity(token_ids.len() * config.hidden_size);
        for &token_id in token_ids {
            let idx = token_id as usize;
            if idx >= embed.nrows() {
                return Err(VindexError::Parse(format!(
                    "token id {idx} out of embedding vocab range {}",
                    embed.nrows()
                )));
            }
            rows.extend(embed.row(idx).iter().copied());
        }
        return Array2::from_shape_vec((token_ids.len(), config.hidden_size), rows)
            .map(|rows| (rows, scale))
            .map_err(|e| VindexError::Parse(e.to_string()));
    }

    let manifest_path = dir.join(GGUF_EMBEDDINGS_MANIFEST_JSON);
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let manifest: GgufEmbeddingsManifestForLoad =
        serde_json::from_str(&manifest_text).map_err(|e| VindexError::Parse(e.to_string()))?;
    if manifest.hidden_size != config.hidden_size || manifest.vocab_size != config.vocab_size {
        return Err(VindexError::Parse(format!(
            "GGUF embeddings manifest shape mismatch: manifest vocab={} hidden={} index vocab={} hidden={}",
            manifest.vocab_size, manifest.hidden_size, config.vocab_size, config.hidden_size
        )));
    }
    let mut rows = Vec::with_capacity(token_ids.len() * manifest.hidden_size);
    for &token_id in token_ids {
        let idx = token_id as usize;
        if idx >= manifest.vocab_size {
            return Err(VindexError::Parse(format!(
                "token id {idx} out of embedding vocab range {}",
                manifest.vocab_size
            )));
        }
        let element_offset = idx
            .checked_mul(manifest.hidden_size)
            .ok_or_else(|| VindexError::Parse("GGUF embedding row offset overflow".into()))?;
        let byte_offset = gguf_type_byte_offset(manifest.tensor_type, element_offset)?;
        let byte_len = gguf_type_byte_len(manifest.tensor_type, manifest.hidden_size)?;
        let absolute = manifest
            .data_offset
            .checked_add(manifest.tensor_offset)
            .and_then(|base| base.checked_add(byte_offset as u64))
            .ok_or_else(|| VindexError::Parse("GGUF embedding byte offset overflow".into()))?;
        let mut file = std::fs::File::open(&manifest.source_file)?;
        use std::io::{Read, Seek};
        file.seek(std::io::SeekFrom::Start(absolute))?;
        let mut raw = vec![0u8; byte_len];
        file.read_exact(&mut raw)?;
        let row =
            larql_models::quant::ggml::dequantize(&raw, manifest.tensor_type, manifest.hidden_size)
                .map_err(|e| VindexError::Parse(e.to_string()))?;
        rows.extend(row);
    }
    Array2::from_shape_vec((token_ids.len(), manifest.hidden_size), rows)
        .map(|rows| (rows, config.embed_scale))
        .map_err(|e| VindexError::Parse(e.to_string()))
}

pub fn load_vindex_gguf_feature_meta(
    dir: &Path,
    layer: usize,
    feature: usize,
    candidate_token_ids: &[u32],
    top_k: usize,
) -> Result<crate::FeatureMeta, VindexError> {
    let manifest_path = dir.join(GGUF_DOWN_META_MANIFEST_JSON);
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let manifest: crate::index::storage::metadata_store::GgufDownMetaManifest =
        serde_json::from_str(&manifest_text).map_err(|e| VindexError::Parse(e.to_string()))?;
    let entry = manifest.layer(layer).ok_or_else(|| {
        VindexError::Parse(format!("GGUF down-meta manifest missing layer {layer}"))
    })?;
    let down = read_gguf_down_feature_vector(entry, feature)?;
    let (embeddings, _scale) = load_vindex_embedding_rows(dir, candidate_token_ids)?;
    let mut scores: Vec<larql_models::TopKEntry> = embeddings
        .outer_iter()
        .enumerate()
        .map(|(row_idx, emb)| {
            let logit = emb.iter().zip(down.iter()).map(|(a, b)| a * b).sum::<f32>();
            let token_id = candidate_token_ids[row_idx];
            larql_models::TopKEntry {
                token: format!("T{token_id}"),
                token_id,
                logit,
            }
        })
        .collect();
    scores.sort_by(|a, b| {
        b.logit
            .partial_cmp(&a.logit)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores.truncate(top_k.min(scores.len()));
    let Some(first) = scores.first() else {
        return Err(VindexError::Parse(
            "GGUF feature meta requires at least one candidate token".into(),
        ));
    };
    Ok(crate::FeatureMeta {
        top_token: first.token.clone(),
        top_token_id: first.token_id,
        c_score: first.logit,
        top_k: scores,
    })
}

fn read_gguf_down_feature_vector(
    entry: &crate::index::storage::metadata_store::GgufDownMetaLayerManifest,
    feature: usize,
) -> Result<Vec<f32>, VindexError> {
    if entry.cols == 0 || entry.rows == 0 || entry.experts == 0 || feature >= entry.features {
        return Err(VindexError::Parse(format!(
            "GGUF down-meta feature {feature} out of range for layer {} (features={})",
            entry.layer, entry.features
        )));
    }
    let expert = feature / entry.cols;
    let local_feature = feature % entry.cols;
    if expert >= entry.experts {
        return Err(VindexError::Parse(format!(
            "GGUF down-meta feature {feature} maps to expert {expert}, but layer {} has {} experts",
            entry.layer, entry.experts
        )));
    }
    let expert_elements = entry
        .rows
        .checked_mul(entry.cols)
        .ok_or_else(|| VindexError::Parse("GGUF down expert element count overflow".into()))?;
    let expert_bytes = gguf_type_byte_len(entry.tensor_type, expert_elements)?;
    let expert_byte_offset = expert
        .checked_mul(expert_bytes)
        .ok_or_else(|| VindexError::Parse("GGUF down expert byte offset overflow".into()))?;
    let absolute = entry
        .data_offset
        .checked_add(entry.tensor_offset)
        .and_then(|base| base.checked_add(expert_byte_offset as u64))
        .ok_or_else(|| VindexError::Parse("GGUF down expert absolute offset overflow".into()))?;
    let mut file = std::fs::File::open(&entry.source_file)?;
    file.seek(std::io::SeekFrom::Start(absolute))?;
    let mut raw = vec![0u8; expert_bytes];
    file.read_exact(&mut raw)?;
    let expert_matrix =
        larql_models::quant::ggml::dequantize(&raw, entry.tensor_type, expert_elements)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
    let mut down = Vec::with_capacity(entry.rows);
    for row in 0..entry.rows {
        let idx = row
            .checked_mul(entry.cols)
            .and_then(|base| base.checked_add(local_feature))
            .ok_or_else(|| VindexError::Parse("GGUF down column index overflow".into()))?;
        down.push(*expert_matrix.get(idx).ok_or_else(|| {
            VindexError::Parse(format!(
                "GGUF down expert matrix missing row {row} feature {local_feature}"
            ))
        })?);
    }
    Ok(down)
}

fn gguf_type_byte_offset(tensor_type: u32, element_offset: usize) -> Result<usize, VindexError> {
    use larql_models::quant::ggml;
    match tensor_type {
        ggml::TYPE_F32 => element_offset
            .checked_mul(4)
            .ok_or_else(|| VindexError::Parse("F32 byte offset overflow".into())),
        ggml::TYPE_F16 | ggml::TYPE_BF16 => element_offset
            .checked_mul(2)
            .ok_or_else(|| VindexError::Parse("F16/BF16 byte offset overflow".into())),
        ggml::TYPE_Q4_K => {
            if !element_offset.is_multiple_of(ggml::K_QUANT_BLOCK_ELEMS) {
                return Err(VindexError::Parse(format!(
                    "Q4_K embedding row offset {element_offset} is not block-aligned"
                )));
            }
            Ok(element_offset / ggml::K_QUANT_BLOCK_ELEMS * ggml::Q4_K_BLOCK_BYTES)
        }
        ggml::TYPE_Q6_K => {
            if !element_offset.is_multiple_of(ggml::K_QUANT_BLOCK_ELEMS) {
                return Err(VindexError::Parse(format!(
                    "Q6_K embedding row offset {element_offset} is not block-aligned"
                )));
            }
            Ok(element_offset / ggml::K_QUANT_BLOCK_ELEMS * ggml::Q6_K_BLOCK_BYTES)
        }
        other => Err(VindexError::Parse(format!(
            "GGUF embedding row loading unsupported for tensor type {}",
            ggml::type_name(other)
        ))),
    }
}

fn gguf_type_byte_len(tensor_type: u32, n_elements: usize) -> Result<usize, VindexError> {
    larql_models::quant::ggml::tensor_data_size(tensor_type, n_elements)
        .map_err(|e| VindexError::Parse(e.to_string()))
}

/// Load tokenizer from a .vindex directory.
pub fn load_vindex_tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer, VindexError> {
    let path = dir.join(TOKENIZER_JSON);
    tokenizers::Tokenizer::from_file(&path).map_err(|e| VindexError::Parse(e.to_string()))
}

/// Load the vindex config.
pub fn load_vindex_config(dir: &Path) -> Result<VindexConfig, VindexError> {
    let text = std::fs::read_to_string(dir.join(INDEX_JSON))?;
    serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))
}

/// Load feature labels from down_meta.jsonl — fast hash lookup, no vocab projection.
///
/// Returns a map: (layer, feature) → top_token string.
/// Also works with the gate vectors NDJSON from vector-extract (has same fields).
pub fn load_feature_labels(path: &Path) -> Result<HashMap<(usize, usize), String>, VindexError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::with_capacity(1 << 20, file);
    let mut labels: HashMap<(usize, usize), String> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

        if obj.get("_header").is_some() {
            continue;
        }

        // Support both compact (l/f/t) and full (layer/feature/top_token) formats
        let layer = obj
            .get("l")
            .or_else(|| obj.get("layer"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let feature = obj
            .get("f")
            .or_else(|| obj.get("feature"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let token = obj
            .get("t")
            .or_else(|| obj.get("top_token"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        labels.insert((layer, feature), token);
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── helpers ─────────────────────────────────────────────────────────

    /// Write a minimal valid index.json into `dir`.
    fn write_minimal_index_json(dir: &std::path::Path, num_layers: usize, hidden: usize) {
        let json = serde_json::json!({
            "version": 2,
            "model": "test/unit",
            "family": "llama",
            "num_layers": num_layers,
            "hidden_size": hidden,
            "intermediate_size": 4,
            "vocab_size": 16,
            "embed_scale": 1.0,
            "layers": [],
            "down_top_k": 5,
            "has_model_weights": false,
            "extract_level": "browse",
            "dtype": "f32",
            "quant": "none"
        });
        std::fs::write(dir.join("index.json"), json.to_string()).unwrap();
    }

    // ── load_vindex_config ──────────────────────────────────────────────

    #[test]
    fn load_vindex_config_parses_valid_json() {
        let dir = TempDir::new().unwrap();
        write_minimal_index_json(dir.path(), 2, 8);
        let cfg = load_vindex_config(dir.path()).unwrap();
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.hidden_size, 8);
        assert_eq!(cfg.model, "test/unit");
        assert_eq!(cfg.family, "llama");
    }

    #[test]
    fn load_vindex_config_missing_file_errors() {
        let dir = TempDir::new().unwrap();
        let result = load_vindex_config(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn load_vindex_config_malformed_json_errors() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("index.json"), b"{not valid json}").unwrap();
        let result = load_vindex_config(dir.path());
        assert!(result.is_err());
    }

    // ── load_feature_labels ─────────────────────────────────────────────

    #[test]
    fn load_feature_labels_compact_format() {
        let dir = TempDir::new().unwrap();
        let jsonl = r#"{"l":0,"f":0,"t":"Paris"}
{"l":0,"f":1,"t":"French"}
{"l":1,"f":0,"t":"Berlin"}
"#;
        let path = dir.path().join("down_meta.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        let labels = load_feature_labels(&path).unwrap();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[&(0, 0)], "Paris");
        assert_eq!(labels[&(0, 1)], "French");
        assert_eq!(labels[&(1, 0)], "Berlin");
    }

    #[test]
    fn load_feature_labels_full_format() {
        let dir = TempDir::new().unwrap();
        let jsonl = r#"{"layer":2,"feature":5,"top_token":"Spain"}
"#;
        let path = dir.path().join("down_meta.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        let labels = load_feature_labels(&path).unwrap();
        assert_eq!(labels[&(2, 5)], "Spain");
    }

    #[test]
    fn load_feature_labels_skips_header_lines() {
        let dir = TempDir::new().unwrap();
        let jsonl = r#"{"_header":true,"version":1}
{"l":0,"f":0,"t":"Rome"}
"#;
        let path = dir.path().join("down_meta.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        let labels = load_feature_labels(&path).unwrap();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[&(0, 0)], "Rome");
    }

    #[test]
    fn load_feature_labels_skips_blank_lines() {
        let dir = TempDir::new().unwrap();
        let jsonl = "  \n{\"l\":0,\"f\":0,\"t\":\"Tokyo\"}\n\n";
        let path = dir.path().join("down_meta.jsonl");
        std::fs::write(&path, jsonl).unwrap();
        let labels = load_feature_labels(&path).unwrap();
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn load_feature_labels_missing_file_errors() {
        let result = load_feature_labels(std::path::Path::new("/no/such/file.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn load_feature_labels_empty_file_returns_empty_map() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, b"").unwrap();
        let labels = load_feature_labels(&path).unwrap();
        assert!(labels.is_empty());
    }

    // ── VectorIndex::load_vindex — minimal fixture ──────────────────────

    /// Write a zero-byte gate_vectors.bin and a matching index.json
    /// for a model with no features (all-zero slices). This lets us test
    /// `load_vindex` without running the full extract pipeline.
    fn write_minimal_loadable_vindex(dir: &std::path::Path, num_layers: usize, hidden: usize) {
        // Empty gate_vectors.bin (0 features per layer → 0 bytes)
        std::fs::write(dir.join("gate_vectors.bin"), b"").unwrap();
        let json = serde_json::json!({
            "version": 2,
            "model": "test/unit",
            "family": "llama",
            "num_layers": num_layers,
            "hidden_size": hidden,
            "intermediate_size": 4,
            "vocab_size": 16,
            "embed_scale": 1.0,
            "layers": [],   // no layers → gate_slices all-zero
            "down_top_k": 5,
            "has_model_weights": false,
            "extract_level": "browse",
            "dtype": "f32",
            "quant": "none"
        });
        std::fs::write(dir.join("index.json"), json.to_string()).unwrap();
    }

    #[test]
    fn load_vindex_missing_dir_errors() {
        let mut cb = crate::index::SilentLoadCallbacks;
        let result = VectorIndex::load_vindex(std::path::Path::new("/nonexistent/vindex"), &mut cb);
        assert!(result.is_err());
    }

    #[test]
    fn load_vindex_missing_index_json_errors() {
        let dir = TempDir::new().unwrap();
        // No index.json written
        let mut cb = crate::index::SilentLoadCallbacks;
        let result = VectorIndex::load_vindex(dir.path(), &mut cb);
        assert!(result.is_err());
    }

    #[test]
    fn load_vindex_minimal_fixture_succeeds() {
        let dir = TempDir::new().unwrap();
        write_minimal_loadable_vindex(dir.path(), 3, 8);
        let mut cb = crate::index::SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
        assert_eq!(index.num_layers, 3);
        assert_eq!(index.hidden_size, 8);
    }

    #[test]
    fn load_vindex_with_range_sets_layer_range() {
        let dir = TempDir::new().unwrap();
        write_minimal_loadable_vindex(dir.path(), 4, 8);
        let mut cb = crate::index::SilentLoadCallbacks;
        let index = VectorIndex::load_vindex_with_range(dir.path(), &mut cb, Some((1, 3))).unwrap();
        assert!(index.is_layer_owned(1));
        assert!(index.is_layer_owned(2));
        assert!(!index.is_layer_owned(0));
        assert!(!index.is_layer_owned(3));
    }

    #[test]
    fn load_vindex_embedding_rows_reads_gguf_manifest_selected_tokens() {
        let dir = TempDir::new().unwrap();
        let source = dir.path().join("tiny-embeddings.gguf.payload");
        let mut bytes = vec![0u8; 96];
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&source, bytes).unwrap();
        let index_json = serde_json::json!({
            "version": 2,
            "model": "test/kimi-gguf",
            "family": "deepseek2",
            "num_layers": 1,
            "hidden_size": 2,
            "intermediate_size": 2,
            "vocab_size": 3,
            "embed_scale": 0.5,
            "layers": [],
            "down_top_k": 5,
            "has_model_weights": false,
            "extract_level": "browse",
            "dtype": "f32",
            "quant": "none"
        });
        std::fs::write(dir.path().join("index.json"), index_json.to_string()).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "architecture": "deepseek2",
            "tensor": "token_embd.weight",
            "source_file": source.display().to_string(),
            "shard_idx": 0,
            "tensor_type": 0,
            "dims": [2, 3],
            "vocab_size": 3,
            "hidden_size": 2,
            "estimated_dense_bytes": 24,
            "dense_budget_bytes": 8,
            "dtype": "f32",
            "tensor_offset": 32,
            "data_offset": 64
        });
        std::fs::write(
            dir.path().join("gguf_embeddings_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let (rows, scale) = load_vindex_embedding_rows(dir.path(), &[2, 0]).unwrap();

        assert_eq!(scale, 0.5);
        assert_eq!(rows.shape(), &[2, 2]);
        assert_eq!(rows.row(0).to_vec(), vec![5.0, 6.0]);
        assert_eq!(rows.row(1).to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn load_vindex_uses_gguf_gate_manifest_for_bounded_gate_knn() {
        let dir = TempDir::new().unwrap();
        let source = dir.path().join("tiny-gates.gguf.payload");
        let mut bytes = vec![0u8; 96];
        for value in [1.0f32, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&source, bytes).unwrap();
        let json = serde_json::json!({
            "version": 2,
            "model": "test/kimi-gguf",
            "family": "deepseek2",
            "num_layers": 3,
            "hidden_size": 2,
            "intermediate_size": 2,
            "vocab_size": 3,
            "embed_scale": 1.0,
            "layers": [{
                "layer": 1,
                "num_features": 4,
                "offset": 0,
                "length": 0,
                "num_experts": 2,
                "num_features_per_expert": 2
            }],
            "down_top_k": 5,
            "has_model_weights": false,
            "extract_level": "browse",
            "dtype": "f32",
            "quant": "none"
        });
        std::fs::write(dir.path().join("index.json"), json.to_string()).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "architecture": "deepseek2",
            "split_count": 1,
            "estimated_dense_bytes": 32,
            "dense_budget_bytes": 8,
            "layers": [{
                "layer": 1,
                "tensor": "blk.1.ffn_gate_exps.weight",
                "source_file": source.display().to_string(),
                "shard_idx": 0,
                "tensor_type": 0,
                "dims": [2, 2, 2],
                "rows": 2,
                "cols": 2,
                "experts": 2,
                "features": 4,
                "tensor_offset": 32,
                "data_offset": 64
            }]
        });
        std::fs::write(
            dir.path().join("gguf_gate_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let mut cb = crate::index::SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
        let residual = ndarray::Array1::from_vec(vec![1.0, 0.0]);
        let hits = index.gate_knn(1, &residual, 2);

        assert_eq!(index.loaded_layers(), vec![1]);
        assert_eq!(hits, vec![(2, 3.0), (0, 1.0)]);
    }

    #[test]
    fn load_vindex_gguf_feature_meta_projects_selected_down_feature() {
        let dir = TempDir::new().unwrap();
        let embed_source = dir.path().join("tiny-embeddings.gguf.payload");
        let mut embed_bytes = vec![0u8; 32];
        for value in [1.0f32, 0.0, 0.0, 2.0, 0.0, 3.0] {
            embed_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&embed_source, embed_bytes).unwrap();

        let down_source = dir.path().join("tiny-down.gguf.payload");
        let mut down_bytes = vec![0u8; 96];
        // One expert, conventional down matrix rows=hidden, cols=features:
        // [[1, 0],
        //  [0, 1]]
        for value in [1.0f32, 0.0, 0.0, 1.0] {
            down_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&down_source, down_bytes).unwrap();

        let index_json = serde_json::json!({
            "version": 2,
            "model": "test/kimi-gguf",
            "family": "deepseek2",
            "num_layers": 2,
            "hidden_size": 2,
            "intermediate_size": 2,
            "vocab_size": 3,
            "embed_scale": 1.0,
            "layers": [{
                "layer": 1,
                "num_features": 2,
                "offset": 0,
                "length": 0,
                "num_experts": 1,
                "num_features_per_expert": 2
            }],
            "down_top_k": 2,
            "has_model_weights": false,
            "extract_level": "browse",
            "dtype": "f32",
            "quant": "none"
        });
        std::fs::write(dir.path().join("index.json"), index_json.to_string()).unwrap();
        let embed_manifest = serde_json::json!({
            "version": 1,
            "architecture": "deepseek2",
            "tensor": "token_embd.weight",
            "source_file": embed_source.display().to_string(),
            "shard_idx": 0,
            "tensor_type": 0,
            "dims": [2, 3],
            "vocab_size": 3,
            "hidden_size": 2,
            "estimated_dense_bytes": 24,
            "dense_budget_bytes": 8,
            "dtype": "f32",
            "tensor_offset": 8,
            "data_offset": 24
        });
        std::fs::write(
            dir.path().join("gguf_embeddings_manifest.json"),
            serde_json::to_string_pretty(&embed_manifest).unwrap(),
        )
        .unwrap();
        let down_manifest = serde_json::json!({
            "version": 1,
            "architecture": "deepseek2",
            "split_count": 1,
            "top_k": 2,
            "estimated_dot_ops": 12,
            "dense_dot_ops_budget": 4,
            "layers": [{
                "layer": 1,
                "tensor": "blk.1.ffn_down_exps.weight",
                "source_file": down_source.display().to_string(),
                "shard_idx": 0,
                "tensor_type": 0,
                "dims": [2, 2, 1],
                "rows": 2,
                "cols": 2,
                "experts": 1,
                "features": 2,
                "tensor_offset": 32,
                "data_offset": 64
            }]
        });
        std::fs::write(
            dir.path().join("gguf_down_meta_manifest.json"),
            serde_json::to_string_pretty(&down_manifest).unwrap(),
        )
        .unwrap();

        let meta = load_vindex_gguf_feature_meta(dir.path(), 1, 1, &[0, 1, 2], 2).unwrap();

        assert_eq!(meta.top_token, "T2");
        assert_eq!(meta.top_token_id, 2);
        assert_eq!(meta.top_k[0].token, "T2");
        assert_eq!(meta.top_k[0].logit, 3.0);
        assert_eq!(meta.top_k[1].token, "T1");
        assert_eq!(meta.top_k[1].logit, 2.0);
    }

    #[test]
    fn load_vindex_uses_gguf_down_meta_manifest_for_feature_scans() {
        let dir = TempDir::new().unwrap();
        let json = serde_json::json!({
            "version": 2,
            "model": "test/kimi-gguf",
            "family": "deepseek2",
            "num_layers": 3,
            "hidden_size": 2,
            "intermediate_size": 2,
            "vocab_size": 3,
            "embed_scale": 1.0,
            "layers": [{
                "layer": 1,
                "num_features": 4,
                "offset": 0,
                "length": 0,
                "num_experts": 2,
                "num_features_per_expert": 2
            }],
            "down_top_k": 5,
            "has_model_weights": false,
            "extract_level": "browse",
            "dtype": "f32",
            "quant": "none"
        });
        std::fs::write(dir.path().join("index.json"), json.to_string()).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "architecture": "deepseek2",
            "split_count": 1,
            "top_k": 5,
            "estimated_dot_ops": 999,
            "dense_dot_ops_budget": 10,
            "layers": [{
                "layer": 1,
                "tensor": "blk.1.ffn_down_exps.weight",
                "source_file": "/tmp/kimi-00001.gguf",
                "shard_idx": 0,
                "tensor_type": 14,
                "dims": [2, 2, 2],
                "rows": 2,
                "cols": 2,
                "experts": 2,
                "features": 4,
                "tensor_offset": 128,
                "data_offset": 64
            }]
        });
        std::fs::write(
            dir.path().join("gguf_down_meta_manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let mut cb = crate::index::SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();

        assert_eq!(index.loaded_layers(), vec![1]);
        assert_eq!(index.num_features(1), 4);
        assert_eq!(index.total_down_meta(), 4);
        let meta = index.feature_meta(1, 3).unwrap();
        assert_eq!(meta.top_token, "gguf:blk.1.ffn_down_exps.weight:E1:F1");
        assert_eq!(meta.top_k[0].token, meta.top_token);
    }
}
