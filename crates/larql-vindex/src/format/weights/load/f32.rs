//! f32 weight loader — reconstructs `ModelWeights` from the split
//! `attn_weights.bin` / `up_weights.bin` / `down_weights.bin` /
//! `norms.bin` / `lm_head.bin` files.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use larql_models::ModelWeights;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::format::load::load_vindex_config;
use crate::index::core::IndexLoadCallbacks;

use super::super::write_f32::{kind, WeightEntry};
use super::LoadWeightsOptions;

/// Load `ModelWeights` from a vindex directory, skipping component
/// tensors per [`LoadWeightsOptions`].
pub fn load_model_weights_with_opts(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
    opts: LoadWeightsOptions,
) -> Result<ModelWeights, VindexError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(VindexError::Parse(
            "vindex does not contain model weights. Rebuild with: larql extract-index <model> -o <vindex> --level all".into(),
        ));
    }

    // `load_model_weights` only knows how to reconstruct the full float
    // `ModelWeights` struct. A Q4_K vindex stores weights in
    // `attn_weights_q4k.bin` + `interleaved_kquant.bin` + per-tensor manifests
    // and must be accessed via `VectorIndex::load_attn_kquant` +
    // `VectorIndex::load_interleaved_kquant` (which return raw quantised
    // bytes that compute dequantises on the fly). Surface a clear error
    // instead of producing a confusing "attn_weights.bin not found".
    if config.quant != crate::QuantFormat::None {
        return Err(VindexError::Parse(format!(
            "vindex is quantised ({}). `load_model_weights` only handles float weights. \
             Call `VectorIndex::load_attn_kquant` + `load_interleaved_kquant` on the loaded \
             VectorIndex instead.",
            config.quant,
        )));
    }

    let model_cfg = config
        .model_config
        .as_ref()
        .ok_or_else(|| VindexError::Parse("vindex missing model_config in index.json".into()))?;

    // Reconstruct full architecture (shared with the Q4_K loader — see
    // `super::arch::build_arch_json`).
    let arch_obj = super::arch::build_arch_json(&config, model_cfg);
    let arch = larql_models::detect_from_json(&arch_obj);

    // Embeddings — skippable for FFN-service servers that only handle
    // residual-vector requests and never see token IDs.
    let embed = if opts.skip_embed {
        super::embeddings::empty_embeddings(callbacks)
    } else {
        super::embeddings::load_embeddings(dir, &config, callbacks)?
    };

    let manifest_path = dir.join(WEIGHT_MANIFEST_JSON);
    if !manifest_path.exists() {
        return Err(VindexError::Parse("weight_manifest.json not found".into()));
    }

    callbacks.on_file_start("model_weights", WEIGHT_MANIFEST_JSON);
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let entries: Vec<WeightEntry> =
        serde_json::from_str(&manifest_text).map_err(|e| VindexError::Parse(e.to_string()))?;

    let mut mmap_cache: HashMap<String, memmap2::Mmap> = HashMap::new();
    let mut tensors: HashMap<String, larql_models::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_loaded: Option<larql_models::WeightArray> = None;

    for entry in &entries {
        // Pre-load filter: skip entries we don't need — never mmap or
        // decode, so peak RSS reflects only what the caller wanted.
        if opts.should_skip(&entry.key) {
            continue;
        }

        let filename = if entry.file.is_empty() {
            crate::format::filenames::MODEL_WEIGHTS_BIN.to_string()
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
            kind::TENSOR => {
                let arr = Array2::from_shape_vec((entry.shape[0], entry.shape[1]), floats)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                if entry.key == "lm_head.weight" {
                    lm_head_loaded = Some(arr.into_shared());
                } else {
                    tensors.insert(entry.key.clone(), arr.into_shared());
                }
            }
            kind::TENSOR_F16 => {
                // Gemma 4 PLE sidecars are always written as f16 (see
                // `weights::ple_sidecar`). The byte-count vs expected
                // floats already picked F16 in `actual_dtype` above, so
                // `floats` is already decoded — same handling as TENSOR
                // from here, just routed via a distinct manifest kind.
                if entry.shape.len() != 2 {
                    continue;
                }
                let arr = Array2::from_shape_vec((entry.shape[0], entry.shape[1]), floats)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                tensors.insert(entry.key.clone(), arr.into_shared());
            }
            kind::VECTOR => {
                vectors.insert(entry.key.clone(), floats);
            }
            _ => {}
        }
    }

    // ── Gemma-4 PLE invariant ──
    //
    // The PLE forward path (crates/larql-inference/src/forward/ple.rs)
    // looks tensors up unconditionally once `arch.has_per_layer_embeddings()`
    // is true. Pre-#49, the writer dropped them on --quant none and the
    // forward path silently fell through to `return h.clone()`, producing
    // garbage tokens with no diagnostic. Validate at load time instead so
    // a stale (pre-fix) vindex fails loudly with actionable guidance.
    if arch.has_per_layer_embeddings() {
        let mut missing: Vec<String> = Vec::new();
        let require_tensor = |key: &str, missing: &mut Vec<String>| {
            if !tensors.contains_key(key) {
                missing.push(key.to_string());
            }
        };
        require_tensor("per_layer_model_projection.weight", &mut missing);
        if let Some(k) = arch.per_layer_embed_key() {
            require_tensor(&k, &mut missing);
        }
        for layer in 0..config.num_layers {
            if let Some(k) = arch.per_layer_input_gate_key(layer) {
                require_tensor(&k, &mut missing);
            }
            if let Some(k) = arch.per_layer_projection_key(layer) {
                require_tensor(&k, &mut missing);
            }
            if let Some(k) = arch.post_per_layer_input_norm_key(layer) {
                if !vectors.contains_key(&k) {
                    missing.push(k);
                }
            }
        }
        if let Some(k) = arch.per_layer_projection_norm_key() {
            if !vectors.contains_key(&k) {
                missing.push(k);
            }
        }
        // `layer_scalar_key` returns Some only on Gemma-4; treat it as a
        // per-layer requirement on those models. Out-of-the-box Llama /
        // Gemma-2/3 leave this None, so the loop below is a no-op there.
        if arch.layer_scalar_key(0).is_some() {
            for layer in 0..config.num_layers {
                if let Some(k) = arch.layer_scalar_key(layer) {
                    if !vectors.contains_key(&k) {
                        missing.push(k);
                    }
                }
            }
        }
        if !missing.is_empty() {
            let sample = missing
                .iter()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            return Err(VindexError::Parse(format!(
                "vindex is missing Gemma-4 PLE sidecar tensors ({} entries, e.g. {sample}). \
                 Rebuild with current larql — older extracts dropped these on --quant none / f16 \
                 and silently produced garbage INFER (chrishayuk/larql#49).",
                missing.len(),
            )));
        }
    }

    // Gate vectors from gate_vectors.bin — only when running in non-Q4 mode.
    //
    // In Q4 vindexes (quant=q4k) the forward pass reads FFN weights straight
    // from the Q4-packed `interleaved_kquant.bin` mmap via
    // `VectorIndex::interleaved_kquant_layer_data`, so expanding `gate_vectors.bin`
    // into an f32 HashMap just to have an unused copy wastes ~27 GB of heap at
    // 31B scale and prevents the model from loading on a 96 GB machine.
    // gate_vectors → FFN gate tensors. Skip when the caller doesn't
    // want FFN weights (saves ~3-14 GB heap for a 4B/31B client).
    if config.quant == crate::config::types::QuantFormat::None && !opts.skip_ffn {
        let gate_file = std::fs::File::open(dir.join(GATE_VECTORS_BIN))?;
        let gate_mmap = unsafe { memmap2::Mmap::map(&gate_file)? };
        let gate_floats = crate::config::dtype::decode_floats(&gate_mmap, config.dtype);
        let bpf = crate::config::dtype::bytes_per_float(config.dtype);
        for info in &config.layers {
            let float_offset = info.offset as usize / bpf;
            let float_count = info.num_features * config.hidden_size;
            if float_offset + float_count <= gate_floats.len() {
                let gate_data = &gate_floats[float_offset..float_offset + float_count];
                let gate_matrix = Array2::from_shape_vec(
                    (info.num_features, config.hidden_size),
                    gate_data.to_vec(),
                )
                .map_err(|e| VindexError::Parse(e.to_string()))?;
                tensors.insert(arch.ffn_gate_key(info.layer), gate_matrix.into_shared());
            }
        }
    }

    if config.extract_level.writes_ffn() && !opts.skip_ffn && !arch.is_moe() {
        let mut missing = Vec::new();
        for layer in 0..config.num_layers {
            for key in [arch.ffn_up_key(layer), arch.ffn_down_key(layer)] {
                if !tensors.contains_key(&key) {
                    missing.push(key);
                }
            }
        }
        if !missing.is_empty() {
            let sample = missing.into_iter().take(4).collect::<Vec<_>>().join(", ");
            return Err(VindexError::Parse(format!(
                "vindex is missing dense FFN tensors ({sample}); compact FFN vindexes are \
                 supported by sparse WalkFfn paths, not load_model_weights/MEMIT"
            )));
        }
    }

    // lm_head from lm_head_q4.bin (dequantise to f32) when the quantised
    // variant is present — the forward path expects an f32 lm_head for the
    // final logits projection. Falls through to embed-tied derivation below
    // if the file is absent (or dequantisation fails).
    if lm_head_loaded.is_none() && !opts.skip_lm_head {
        let lm_q4_path = crate::format::filenames::resolve_lm_head_kquant(dir).bin;
        if lm_q4_path.exists() {
            if let Some(model_cfg) = config.model_config.as_ref() {
                // lm_head shape is (vocab_size, hidden_size) — same as embed.
                let _ = model_cfg; // shape comes from config.vocab_size / hidden_size.
            }
            let bytes = std::fs::read(&lm_q4_path)?;
            let num_floats = config.vocab_size * config.hidden_size;
            let padded_floats = num_floats.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
                * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
            if let Ok(floats) = larql_models::quant::ggml::dequantize_q4_k(&bytes, padded_floats) {
                if floats.len() >= num_floats {
                    if let Ok(arr) = Array2::from_shape_vec(
                        (config.vocab_size, config.hidden_size),
                        floats[..num_floats].to_vec(),
                    ) {
                        lm_head_loaded = Some(arr.into_shared());
                    }
                }
            }
        }
    }

    callbacks.on_file_done("model_weights", entries.len(), 0.0);

    let cfg = arch.config();
    let embed = embed.into_shared();
    // Embed-tied fallback: models like Gemma share embed ↔ lm_head
    // weights. When the caller asked to skip lm_head we don't want to
    // clone embed into it — use an empty placeholder instead.
    let lm_head = if opts.skip_lm_head {
        lm_head_loaded.unwrap_or_else(|| Array2::<f32>::zeros((0, 0)).into_shared())
    } else {
        lm_head_loaded.unwrap_or_else(|| embed.clone())
    };

    Ok(ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
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
