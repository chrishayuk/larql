//! Binary loading path for .vindex directories.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array2;

use crate::config::VindexConfig;
use crate::error::VindexError;
use crate::format::filenames::{
    has_kquant_lm_head, resolve_interleaved_kquant, DOWN_META_BIN, DOWN_META_JSONL, EMBEDDINGS_BIN,
    GATE_VECTORS_BIN, INDEX_JSON, LM_HEAD_BIN, TOKENIZER_JSON,
};
use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER;
use crate::index::{IndexLoadCallbacks, VectorIndex};

impl VectorIndex {
    /// Load a VectorIndex from a .vindex directory.
    ///
    /// Reads gate_vectors.bin (mmap'd), down_meta.bin or legacy down_meta.jsonl,
    /// and index.json.
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
        // missing but `interleaved_kquant.bin` is present, synthesize an
        // anonymous mmap by dequantizing the Q4K gate slices at f16 —
        // that's dedup #2 in action (a Q4K vindex extracted with
        // `--drop-gate-vectors` carries gate weights only once, Q4K).
        let gate_path = dir.join(GATE_VECTORS_BIN);
        let interleaved_kquant_path = resolve_interleaved_kquant(dir).bin;

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
        } else if interleaved_kquant_path.exists() {
            callbacks.on_file_start(
                "gate_vectors (synth from k-quant)",
                &interleaved_kquant_path.display().to_string(),
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
            // Neither gate_vectors.bin nor interleaved_kquant.bin present.
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

        // Load down metadata — mmap binary (zero heap), fall back to JSONL
        // only when the binary file is absent. A present-but-invalid binary
        // is a corrupt vindex and should fail loudly.
        let start = std::time::Instant::now();

        let has_binary_down_meta = crate::format::down_meta::has_binary(dir);
        let down_meta_mmap = if has_binary_down_meta {
            let tokenizer = load_vindex_tokenizer(dir)?;
            callbacks.on_file_start("down_meta", &dir.join(DOWN_META_BIN).display().to_string());
            let tok = std::sync::Arc::new(tokenizer);
            let dm = crate::format::down_meta::mmap_binary(dir, tok)?;
            let count = dm.total_features();
            callbacks.on_file_done("down_meta", count, start.elapsed().as_secs_f64() * 1000.0);
            Some(dm)
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

        if !has_binary_down_meta {
            let legacy_path = dir.join(DOWN_META_JSONL);
            if legacy_path.exists() {
                index.load_down_meta(&legacy_path, callbacks)?;
            }
        }

        // Opportunistically wire up FFN payload mmaps so walk_ffn_sparse can
        // find up/down data without callers needing to know which flavour
        // is on disk. Each load_* returns Err(_) if its file isn't present;
        // those errors are non-fatal here.
        if let Some(range) = layer_range {
            index.set_layer_range(range);
        }

        let _ = index.load_interleaved_kquant(dir);
        let _ = index.load_interleaved_q4(dir);
        let _ = index.load_interleaved(dir);
        let _ = index.load_up_features(dir);
        let _ = index.load_down_features(dir);
        // W2: feature-major Q4_K down. Optional file; when present the
        // CPU sparse walk skips the `kquant_ffn_layer` cache for component=2.
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
        let has_separate_lm_head = dir.join(LM_HEAD_BIN).exists() || has_kquant_lm_head(dir);
        if !has_separate_lm_head {
            if let Ok(f) = std::fs::File::open(dir.join(EMBEDDINGS_BIN)) {
                if let Ok(mmap) = unsafe { memmap2::Mmap::map(&f) } {
                    let expected_f16 = config.vocab_size * config.hidden_size * 2;
                    if mmap.len() >= expected_f16 && mmap.len() < expected_f16 * 2 {
                        if index.vocab_size == 0 {
                            index.vocab_size = config.vocab_size;
                        }
                        index.set_lm_head_f16_mmap(std::sync::Arc::new(mmap));
                        index.synthesize_lm_head_kquant();
                    }
                }
            }
        }

        Ok(index)
    }
}

/// Dequantize gate slices from `interleaved_kquant.bin` into an anonymous
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
    let resolved = resolve_interleaved_kquant(dir);
    let interleaved_path = resolved.bin;
    let manifest_path = resolved
        .manifest
        .expect("interleaved kquant resolver always pairs a manifest");
    if !manifest_path.exists() {
        return Err(VindexError::Parse(format!(
            "interleaved k-quant manifest missing alongside {}",
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
        let base = info.layer * FFN_COMPONENTS_PER_LAYER;
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
                "interleaved_kquant_manifest gate entry at layer {} missing `format`",
                info.layer
            ))
        })?;
        // Route through the registry so a future Q6_K (or other K-quant)
        // gate slice would dequantise the same way without another
        // string-compare here.
        let format_info = crate::quant::registry::lookup(format).ok_or_else(|| {
            VindexError::Parse(format!(
                "interleaved_kquant_manifest layer {}: unknown format tag {format:?}",
                info.layer
            ))
        })?;
        let end = offset.checked_add(length).ok_or_else(|| {
            VindexError::Parse(format!(
                "interleaved_kquant_manifest layer {}: offset+length overflow ({offset}+{length})",
                info.layer
            ))
        })?;
        if end > iq4_mmap.len() {
            return Err(VindexError::Parse(format!(
                "interleaved_kquant_manifest layer {}: gate slice {offset}..{end} exceeds mmap length {}",
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
        let path = dir.path().join(DOWN_META_JSONL);
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
        let path = dir.path().join(DOWN_META_JSONL);
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
        let path = dir.path().join(DOWN_META_JSONL);
        std::fs::write(&path, jsonl).unwrap();
        let labels = load_feature_labels(&path).unwrap();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[&(0, 0)], "Rome");
    }

    #[test]
    fn load_feature_labels_skips_blank_lines() {
        let dir = TempDir::new().unwrap();
        let jsonl = "  \n{\"l\":0,\"f\":0,\"t\":\"Tokyo\"}\n\n";
        let path = dir.path().join(DOWN_META_JSONL);
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
}
