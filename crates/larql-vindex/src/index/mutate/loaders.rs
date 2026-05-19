//! NDJSON loaders for `VectorIndex` — read gate vectors and down
//! metadata from `ffn_gate.vectors.jsonl` / `ffn_down.meta.jsonl`.
//!
//! These are the heap-mode constructors. The mmap-mode entry point
//! `VectorIndex::new_mmap` lives in `super::core` next to `new`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use larql_models::TopKEntry;
use ndarray::Array2;

use crate::error::VindexError;

use crate::index::core::VectorIndex;
use crate::index::types::*;

impl VectorIndex {
    pub fn load_gates(
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        callbacks.on_file_start("ffn_gate", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);

        // First pass: collect all records to determine dimensions
        let mut records: Vec<(usize, usize, Vec<f32>, FeatureMeta)> = Vec::new();
        let mut hidden_size = 0;
        let mut max_layer = 0;
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                if let Some(dim) = obj.get("dimension").and_then(|v| v.as_u64()) {
                    hidden_size = dim as usize;
                }
                continue;
            }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let vector: Vec<f32> = obj["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            if hidden_size == 0 {
                hidden_size = vector.len();
            }

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer > max_layer {
                max_layer = layer;
            }

            records.push((layer, feature, vector, meta));

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let num_layers = max_layer + 1;

        // Group by layer, find max feature per layer
        let mut layer_sizes: HashMap<usize, usize> = HashMap::new();
        for &(layer, feature, _, _) in &records {
            let entry = layer_sizes.entry(layer).or_insert(0);
            if feature + 1 > *entry {
                *entry = feature + 1;
            }
        }

        // Build per-layer matrices
        let mut gate_vectors: Vec<Option<Array2<f32>>> = vec![None; num_layers];
        let mut gate_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = vec![None; num_layers];

        // Pre-allocate
        for (&layer, &num_features) in &layer_sizes {
            gate_vectors[layer] = Some(Array2::zeros((num_features, hidden_size)));
            gate_meta[layer] = Some(vec![None; num_features]);
        }

        // Fill
        for (layer, feature, vector, meta) in records {
            if let Some(ref mut matrix) = gate_vectors[layer] {
                for (j, &val) in vector.iter().enumerate() {
                    matrix[[feature, j]] = val;
                }
            }
            if let Some(ref mut metas) = gate_meta[layer] {
                metas[feature] = Some(meta);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_gate", count, elapsed_ms);

        let mut v = VectorIndex::empty(num_layers, hidden_size);
        v.gate.gate_vectors = gate_vectors;
        v.metadata.down_meta = gate_meta;
        Ok(v)
    }

    /// Load down-projection token metadata from an NDJSON file (ffn_down.vectors.jsonl).
    ///
    /// Only loads the metadata (top_token, top_k, c_score), NOT the full vectors.
    /// This replaces any gate-file metadata with the down-projection metadata,
    /// which tells you what each feature *outputs* rather than what it *responds to*.
    pub fn load_down_meta(
        &mut self,
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<usize, VindexError> {
        callbacks.on_file_start("ffn_down", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);
        let mut count = 0;

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

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer < self.num_layers {
                // Ensure layer slot exists
                while self.metadata.down_meta.len() <= layer {
                    self.metadata.down_meta.push(None);
                }
                if self.metadata.down_meta[layer].is_none() {
                    self.metadata.down_meta[layer] = Some(Vec::new());
                }
                if let Some(ref mut metas) = self.metadata.down_meta[layer] {
                    while metas.len() <= feature {
                        metas.push(None);
                    }
                    metas[feature] = Some(meta);
                }
            }

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_down", count, elapsed_ms);

        Ok(count)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::index::types::SilentLoadCallbacks;

    fn write_ndjson(
        dir: &std::path::Path,
        name: &str,
        lines: &[serde_json::Value],
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let body: String = lines
            .iter()
            .map(|v| serde_json::to_string(v).unwrap() + "\n")
            .collect();
        std::fs::write(&path, body).unwrap();
        path
    }

    fn header(dim: usize) -> serde_json::Value {
        serde_json::json!({"_header": true, "dimension": dim})
    }

    fn gate_record(
        layer: usize,
        feature: usize,
        vector: &[f32],
        top_token: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "layer": layer,
            "feature": feature,
            "vector": vector,
            "top_token": top_token,
            "top_token_id": 1,
            "c_score": 0.9,
            "top_k": [
                {"token": top_token, "token_id": 1, "logit": 5.0},
                {"token": "alt", "token_id": 2, "logit": 1.0}
            ]
        })
    }

    fn down_record(layer: usize, feature: usize, top_token: &str, score: f32) -> serde_json::Value {
        serde_json::json!({
            "layer": layer,
            "feature": feature,
            "top_token": top_token,
            "top_token_id": 7,
            "c_score": score,
            "top_k": [{"token": top_token, "token_id": 7, "logit": score}]
        })
    }

    // ── load_gates ──

    #[test]
    fn load_gates_errors_when_file_missing() {
        let mut cb = SilentLoadCallbacks;
        let result =
            VectorIndex::load_gates(std::path::Path::new("/tmp/_no_such_gates.jsonl"), &mut cb);
        assert!(result.is_err(), "missing file must error");
    }

    #[test]
    fn load_gates_errors_on_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.jsonl");
        std::fs::write(&path, "not json {\n").unwrap();
        let mut cb = SilentLoadCallbacks;
        let result = VectorIndex::load_gates(&path, &mut cb);
        assert!(result.is_err(), "invalid json must error");
    }

    #[test]
    fn load_gates_skips_blank_lines_and_reads_dimension_from_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gates.jsonl");
        let body = format!(
            "{}\n\n{}\n{}\n",
            serde_json::to_string(&header(4)).unwrap(),
            serde_json::to_string(&gate_record(0, 0, &[1.0, 2.0, 3.0, 4.0], "Paris")).unwrap(),
            serde_json::to_string(&gate_record(1, 2, &[5.0, 6.0, 7.0, 8.0], "French")).unwrap(),
        );
        std::fs::write(&path, body).unwrap();

        let mut cb = SilentLoadCallbacks;
        let v = VectorIndex::load_gates(&path, &mut cb).unwrap();
        assert_eq!(v.num_layers, 2);
        assert_eq!(v.hidden_size, 4);

        let layer0 = v.gate.gate_vectors[0].as_ref().expect("layer 0 present");
        assert_eq!(layer0.shape(), &[1, 4]);
        assert_eq!(layer0[[0, 0]], 1.0);
        assert_eq!(layer0[[0, 3]], 4.0);

        let layer1 = v.gate.gate_vectors[1].as_ref().expect("layer 1 present");
        assert_eq!(layer1.shape(), &[3, 4]);
        assert_eq!(layer1[[2, 0]], 5.0);
        assert_eq!(layer1[[2, 3]], 8.0);
        assert_eq!(layer1[[0, 0]], 0.0);

        let meta1 = v.metadata.down_meta[1].as_ref().unwrap();
        assert!(meta1[0].is_none());
        assert!(meta1[1].is_none());
        let meta_filled = meta1[2].as_ref().unwrap();
        assert_eq!(meta_filled.top_token, "French");
        assert_eq!(meta_filled.top_token_id, 1);
        assert_eq!(meta_filled.top_k.len(), 2);
    }

    #[test]
    fn load_gates_infers_dimension_from_first_record_when_no_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("noheader.jsonl");
        let body = format!(
            "{}\n",
            serde_json::to_string(&gate_record(0, 0, &[1.0, 2.0, 3.0], "x")).unwrap()
        );
        std::fs::write(&path, body).unwrap();

        let mut cb = SilentLoadCallbacks;
        let v = VectorIndex::load_gates(&path, &mut cb).unwrap();
        assert_eq!(v.hidden_size, 3);
    }

    #[test]
    fn load_gates_handles_missing_top_k_field() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("no_topk.jsonl");
        let rec = serde_json::json!({
            "layer": 0,
            "feature": 0,
            "vector": [1.0, 2.0],
            "top_token": "x",
            "top_token_id": 1,
            "c_score": 0.5
        });
        std::fs::write(&path, serde_json::to_string(&rec).unwrap() + "\n").unwrap();
        let mut cb = SilentLoadCallbacks;
        let v = VectorIndex::load_gates(&path, &mut cb).unwrap();
        let meta = v.metadata.down_meta[0].as_ref().unwrap()[0]
            .as_ref()
            .unwrap();
        assert!(meta.top_k.is_empty());
    }

    // ── load_down_meta ──

    #[test]
    fn load_down_meta_errors_when_file_missing() {
        let mut v = VectorIndex::empty(2, 4);
        let mut cb = SilentLoadCallbacks;
        let err = v
            .load_down_meta(std::path::Path::new("/tmp/_no_such_down.jsonl"), &mut cb)
            .expect_err("missing file errors");
        let _ = err;
    }

    #[test]
    fn load_down_meta_attaches_records_to_existing_layers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_ndjson(
            tmp.path(),
            "down.jsonl",
            &[
                header(4),
                down_record(0, 0, "Paris", 0.95),
                down_record(0, 2, "Berlin", 0.80),
                down_record(1, 1, "French", 0.70),
            ],
        );

        let mut v = VectorIndex::empty(2, 4);
        let mut cb = SilentLoadCallbacks;
        let count = v.load_down_meta(&path, &mut cb).unwrap();
        assert_eq!(count, 3);

        let l0 = v.metadata.down_meta[0].as_ref().unwrap();
        assert_eq!(l0[0].as_ref().unwrap().top_token, "Paris");
        assert!(l0[1].is_none());
        assert_eq!(l0[2].as_ref().unwrap().top_token, "Berlin");

        let l1 = v.metadata.down_meta[1].as_ref().unwrap();
        assert_eq!(l1[1].as_ref().unwrap().top_token, "French");
    }

    #[test]
    fn load_down_meta_skips_layers_above_num_layers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = write_ndjson(
            tmp.path(),
            "oob.jsonl",
            &[down_record(99, 0, "x", 0.5), down_record(0, 0, "y", 0.6)],
        );

        let mut v = VectorIndex::empty(2, 4);
        let mut cb = SilentLoadCallbacks;
        let count = v.load_down_meta(&path, &mut cb).unwrap();
        assert_eq!(count, 2);
        assert!(v.metadata.down_meta[0].as_ref().unwrap()[0].is_some());
    }

    #[test]
    fn load_down_meta_skips_blank_lines_and_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("with_blank.jsonl");
        let body = format!(
            "{}\n\n  \n{}\n",
            serde_json::to_string(&header(4)).unwrap(),
            serde_json::to_string(&down_record(0, 0, "ok", 0.5)).unwrap(),
        );
        std::fs::write(&path, body).unwrap();

        let mut v = VectorIndex::empty(1, 4);
        let mut cb = SilentLoadCallbacks;
        let count = v.load_down_meta(&path, &mut cb).unwrap();
        assert_eq!(count, 1);
        assert!(v.metadata.down_meta[0].as_ref().unwrap()[0].is_some());
    }

    #[test]
    fn load_down_meta_errors_on_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad_down.jsonl");
        std::fs::write(&path, "garbage{\n").unwrap();
        let mut v = VectorIndex::empty(1, 4);
        let mut cb = SilentLoadCallbacks;
        assert!(v.load_down_meta(&path, &mut cb).is_err());
    }
}
