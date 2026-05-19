//! Attention weight loaders + per-layer accessors.
//!
//! Loads the per-layer Q / K / V / O projection weights in Q8, Q4_K, or
//! Q4_0 format from `attn_weights_*.bin` files plus their JSON
//! manifests. Mirrors the FFN walk plumbing in `super::walk`; lives in
//! its own file so attention storage isn't tangled with FFN storage.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::mmap_util::mmap_optimized;

use crate::index::core::VectorIndex;
use crate::index::storage::vindex_storage::VindexStorage;

/// Number of attention projection tensors recorded per layer in every
/// `attn_weights_*.bin` manifest: Q, K, V, O — in that order.
pub(crate) const ATTN_TENSORS_PER_LAYER: usize = 4;

impl VectorIndex {
    /// Load Q8 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q8(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(ATTN_WEIGHTS_Q8_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q8.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = Arc::new(unsafe { mmap_optimized(&file)? });

        let manifest_path = dir.join(ATTN_WEIGHTS_Q8_MANIFEST_JSON);
        let manifest = if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            Some(
                json.iter()
                    .map(|e| {
                        let offset = e["q8_offset"].as_u64().unwrap_or(0) as usize;
                        let vals_len = e["q8_vals_len"].as_u64().unwrap_or(0) as usize;
                        let scales_len = e["q8_scales_len"].as_u64().unwrap_or(0) as usize;
                        (offset, vals_len, scales_len)
                    })
                    .collect(),
            )
        } else {
            None
        };

        Arc::make_mut(&mut self.storage).set_attn_q8(mmap, manifest);
        Ok(())
    }

    /// Get per-layer Q8 attention slices: (q_vals, q_scales, k_vals, k_scales, v_vals, v_scales, o_vals, o_scales)
    ///
    /// Forwarded through [`VectorIndex::storage`] (step 4 of the
    /// `VindexStorage` migration). Public signature unchanged so
    /// existing callers don't move; the returned `&[u8]` / `&[f32]`
    /// borrow from the storage façade's `Bytes` (zero-copy).
    pub fn attn_q8_layer_data(&self, layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
        let arr = self.storage.attn_q8_layer_data(layer)?;
        let mut out = [(&[] as &[u8], &[] as &[f32]); ATTN_TENSORS_PER_LAYER];
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (vals_view, scales_view) = arr[i];
            let vals = vals_view.as_slice();
            let scales_bytes = scales_view.as_slice();
            // Same `slice::from_raw_parts` reinterpretation today's
            // accessor used; preserves the alignment-and-padding
            // contract enforced by the writer.
            let scales = unsafe {
                std::slice::from_raw_parts(
                    scales_bytes.as_ptr() as *const f32,
                    scales_bytes.len() / 4,
                )
            };
            out[i] = (vals, scales);
        }
        Some(out)
    }

    /// Load Q4_K/Q6_K attention weights for Ollama-compatible GPU pipeline.
    pub fn load_attn_kquant(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let resolved = resolve_attn_weights_kquant(dir);
        let path = resolved.bin;
        if !path.exists() {
            return Err(VindexError::Parse(format!(
                "attention k-quant weights not found (looked for {} and legacy {})",
                ATTN_WEIGHTS_KQUANT_BIN, LEGACY_ATTN_WEIGHTS_Q4K_BIN
            )));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = Arc::new(unsafe { mmap_optimized(&file)? });

        let manifest_path = resolved
            .manifest
            .expect("attn weights resolver always pairs a manifest");
        let manifest = if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            // Each entry: {key, shape, format, offset, length}.
            //
            // Format is required. We used to default to `"Q4_K"` here
            // when the field was missing, which silently masked
            // malformed manifests — see ROADMAP P0 "Replace
            // unwrap_or(Q4_K) silent fallbacks".
            let entries: Vec<(usize, usize, String)> = json
                .iter()
                .map(|e| {
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let tag = e["format"].as_str().ok_or_else(|| {
                        VindexError::Parse(
                            "attn kquant manifest entry missing `format` field".into(),
                        )
                    })?;
                    let qfmt = crate::quant::registry::lookup(tag).ok_or_else(|| {
                        VindexError::Parse(format!(
                            "attn kquant manifest: unknown format tag {tag:?} \
                             — quant::registry has no entry"
                        ))
                    })?;

                    // Stride sanity check — catches stale vindexes built
                    // with the legacy 148-byte block_q4_K layout against
                    // the current 144-byte GGUF kernels (the read drifts
                    // 4 bytes per superblock, producing all-NaN output).
                    let key = e["key"].as_str().unwrap_or("<no-key>");
                    let shape: Vec<usize> = e["shape"]
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect()
                        })
                        .unwrap_or_default();
                    if let Some(expected) = qfmt.expected_bytes(&shape) {
                        if expected != length {
                            return Err(VindexError::Parse(format!(
                                "attn kquant manifest: tensor {key:?} ({tag}, shape {shape:?}) \
                                 has length {length} but format expects {expected} \
                                 ({} bytes/block × {}). \
                                 Likely cause: vindex built with legacy 148-byte block_q4_K layout — \
                                 rebuild the vindex with current code (`larql kquant <model>` or equivalent).",
                                qfmt.bytes_per_block,
                                length / qfmt.bytes_per_block.max(1),
                            )));
                        }
                    }
                    Ok((offset, length, tag.to_string()))
                })
                .collect::<Result<Vec<_>, VindexError>>()?;
            Some(entries)
        } else {
            None
        };
        Arc::make_mut(&mut self.storage).set_attn_kquant(mmap, manifest);
        Ok(())
    }

    /// Get per-layer Q4_K/Q6_K attention slices: (data, format) for Q, K, V, O.
    ///
    /// Forwarded through [`VectorIndex::storage`] (step 4 of the
    /// `VindexStorage` migration). Public signature unchanged.
    pub fn attn_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let arr = self.storage.attn_kquant_layer_data(layer)?;
        let mut out: [(&[u8], &str); ATTN_TENSORS_PER_LAYER] = [(&[], ""); ATTN_TENSORS_PER_LAYER];
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (view, fmt) = arr[i];
            out[i] = (view.as_slice(), fmt);
        }
        Some(out)
    }

    /// Load Q4 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(ATTN_WEIGHTS_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = Arc::new(unsafe { mmap_optimized(&file)? });

        // Load manifest with per-matrix offsets
        let manifest_path = dir.join(ATTN_WEIGHTS_Q4_MANIFEST_JSON);
        let manifest = if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            Some(
                json.iter()
                    .map(|e| {
                        let offset = e["q4_offset"].as_u64().unwrap_or(0) as usize;
                        let length = e["q4_length"].as_u64().unwrap_or(0) as usize;
                        (offset, length)
                    })
                    .collect(),
            )
        } else {
            None
        };
        Arc::make_mut(&mut self.storage).set_attn_q4(mmap, manifest);
        Ok(())
    }

    /// Get raw Q4 attention weight bytes (all layers packed).
    ///
    /// Forwarded through [`VectorIndex::storage`]. The borrow lifetime
    /// is tied to `&self` because `MmapStorage` keeps the
    /// `bytes::Bytes` whole-buffer handle alive.
    pub fn attn_q4_data(&self) -> Option<&[u8]> {
        self.storage.attn_q4_whole_buffer_view().map(|b| b.as_ref())
    }

    /// Get per-layer Q4 attention weight slices (Q, K, V, O) using the manifest.
    /// Returns None if manifest or Q4 attn data is not loaded.
    ///
    /// Forwarded through [`VectorIndex::storage`] (step 4 of the
    /// `VindexStorage` migration).
    #[allow(clippy::type_complexity)]
    pub fn attn_q4_layer_slices(&self, layer: usize) -> Option<(&[u8], &[u8], &[u8], &[u8])> {
        let arr = self.storage.attn_q4_layer_slices(layer)?;
        Some((
            arr[0].as_slice(),
            arr[1].as_slice(),
            arr[2].as_slice(),
            arr[3].as_slice(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal vindex directory with the given attn_weights_q4k.bin
    /// payload + manifest. Returns a `tempfile::TempDir` whose path can be
    /// passed straight to `load_attn_kquant`.
    fn make_vindex_with_attn_kquant(
        payload: &[u8],
        manifest: serde_json::Value,
    ) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(ATTN_WEIGHTS_KQUANT_BIN), payload).unwrap();
        std::fs::write(
            tmp.path().join(ATTN_WEIGHTS_KQUANT_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        tmp
    }

    fn empty_vindex() -> VectorIndex {
        // Layer count and hidden size don't matter for the load_attn_kquant
        // path — both are read from the manifest, not from the index.
        VectorIndex::empty(1, 2560)
    }

    /// Q4_K shape `[2048, 2560]` at the canonical Q4_K_BLOCK_BYTES stride
    /// must load cleanly.
    #[test]
    fn load_attn_kquant_accepts_correct_144_byte_stride() {
        use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};
        let len = 2048 * (2560 / Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES; // 2_949_120
        let payload = vec![0u8; len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": len,
            }
        ]);
        let tmp = make_vindex_with_attn_kquant(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_kquant(tmp.path())
            .expect("clean stride must load");
    }

    /// Regression: an attn_weights_q4k.bin written with the legacy
    /// 148-byte block_q4_K layout must be rejected at load time. The
    /// kernel reads 144-byte GGUF strides; without this check, every
    /// row's read window drifts by 4 bytes per superblock and the GPU
    /// prefill silently produces all-NaN.
    #[test]
    fn load_attn_kquant_rejects_legacy_148_byte_stride() {
        use crate::quant::registry::LEGACY_BLOCK_Q4_K_STRIDE;
        use larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        // 3_031_040 — what 8-Apr vindexes have.
        let bad_len = 2048 * (2560 / K_QUANT_BLOCK_ELEMS) * LEGACY_BLOCK_Q4_K_STRIDE;
        let payload = vec![0u8; bad_len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": bad_len,
            }
        ]);
        let tmp = make_vindex_with_attn_kquant(&payload, manifest);
        let mut idx = empty_vindex();
        let err = idx
            .load_attn_kquant(tmp.path())
            .expect_err("legacy 148-byte stride must be rejected");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("rebuild the vindex"),
            "error must guide the user to rebuild — got: {msg}"
        );
        assert!(
            msg.contains("3031040") || msg.contains("2949120"),
            "error must include both lengths so the user can see the drift — got: {msg}"
        );
    }

    /// A length that's neither 144 × n nor 148 × n still gets rejected
    /// (anything that's not the canonical stride is an error).
    #[test]
    fn load_attn_kquant_rejects_arbitrary_wrong_length() {
        let weird_len = 2_949_120 + 17; // off-by-17 — definitely not aligned
        let payload = vec![0u8; weird_len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": weird_len,
            }
        ]);
        let tmp = make_vindex_with_attn_kquant(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_kquant(tmp.path())
            .expect_err("non-canonical stride must be rejected");
    }

    /// Q6_K stride (210 bytes per 256-element block) must also validate
    /// — V projections in Gemma 3 4B are Q6_K and would suffer the same
    /// silent-drift class of bug.
    #[test]
    fn load_attn_kquant_validates_q6k_v_projection() {
        use larql_models::quant::ggml::{K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};
        let q4k_len = 1024 * (2560 / K_QUANT_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES; // K proj: 1024 × 1440
        let q6k_len = 1024 * (2560 / K_QUANT_BLOCK_ELEMS) * Q6_K_BLOCK_BYTES; // V proj: 1024 × 2100
        let total = q4k_len + q6k_len;
        let payload = vec![0u8; total];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.k_proj.weight",
                "shape": [1024, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": q4k_len,
            },
            {
                "key": "layers.0.self_attn.v_proj.weight",
                "shape": [1024, 2560],
                "format": "Q6_K",
                "offset": q4k_len,
                "length": q6k_len,
            }
        ]);
        let tmp = make_vindex_with_attn_kquant(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_kquant(tmp.path())
            .expect("matched Q4_K + Q6_K strides must load");
    }

    /// A Q6_K manifest entry recorded with a Q4_K-sized length (210 vs
    /// 144 confusion at write time) must be rejected.
    #[test]
    fn load_attn_kquant_rejects_q6k_with_q4k_stride() {
        use larql_models::quant::ggml::{K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES};
        let wrong_len = 1024 * (2560 / K_QUANT_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES; // Q4_K stride for Q6_K tensor
        let payload = vec![0u8; wrong_len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.v_proj.weight",
                "shape": [1024, 2560],
                "format": "Q6_K",
                "offset": 0,
                "length": wrong_len,
            }
        ]);
        let tmp = make_vindex_with_attn_kquant(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_kquant(tmp.path())
            .expect_err("Q6_K tensor with Q4_K length must be rejected");
    }

    /// A stale or corrupt Q4_K manifest entry whose `offset + length`
    /// runs past the mmap end must produce `None` from
    /// `attn_kquant_layer_data`, not a slice-bounds panic. Mirrors the
    /// defensive behavior already in `interleaved_kquant_layer_data`.
    #[test]
    fn attn_kquant_layer_data_returns_none_on_out_of_bounds_manifest() {
        use larql_models::quant::ggml::{K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES};
        // Load a real, valid Q4_K vindex first…
        let len = 2048 * (2560 / K_QUANT_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES;
        let payload = vec![0u8; len];
        let manifest = serde_json::json!([
            {
                "key": "layers.0.self_attn.q_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": len,
            },
            {
                "key": "layers.0.self_attn.k_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": len,
            },
            {
                "key": "layers.0.self_attn.v_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": len,
            },
            {
                "key": "layers.0.self_attn.o_proj.weight",
                "shape": [2048, 2560],
                "format": "Q4_K",
                "offset": 0,
                "length": len,
            },
        ]);
        let tmp = make_vindex_with_attn_kquant(&payload, manifest);
        let mut idx = empty_vindex();
        idx.load_attn_kquant(tmp.path()).expect("clean load");

        // …then corrupt the manifest so the V entry's slice would walk
        // off the end of the mmap. The bounds check should turn this
        // into `None` rather than a panic. Direct mutation of the
        // storage's pub(crate) manifest field is a test-only pattern
        // (production goes through `set_attn_kquant`).
        let storage = std::sync::Arc::make_mut(&mut idx.storage);
        let m = storage.attn_kquant_manifest.as_mut().expect("manifest");
        m[2] = (len, 1, "Q4_K".to_string()); // offset = len → end = len + 1 > mmap.len()
        assert!(idx.attn_kquant_layer_data(0).is_none());
    }

    /// `attn_q8_layer_data` must reject a manifest entry where
    /// `offset + vals_len + scales_len` overflows the mmap.
    #[test]
    fn attn_q8_layer_data_returns_none_on_out_of_bounds_manifest() {
        let mmap_len = 1024;
        let payload = vec![0u8; mmap_len];
        let manifest = serde_json::json!([
            { "q8_offset": 0,   "q8_vals_len": 64, "q8_scales_len": 16 },
            { "q8_offset": 100, "q8_vals_len": 64, "q8_scales_len": 16 },
            { "q8_offset": 200, "q8_vals_len": 64, "q8_scales_len": 16 },
            { "q8_offset": mmap_len - 32, "q8_vals_len": 64, "q8_scales_len": 16 },
            // Total = mmap_len + 48 > mmap_len.
        ]);
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(ATTN_WEIGHTS_Q8_BIN), &payload).unwrap();
        std::fs::write(
            tmp.path().join(ATTN_WEIGHTS_Q8_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let mut idx = empty_vindex();
        idx.load_attn_q8(tmp.path())
            .expect("Q8 has no load-time len check");
        assert!(idx.attn_q8_layer_data(0).is_none());
    }

    /// `attn_q4_layer_slices` must reject a manifest entry whose
    /// `offset + length` runs past the mmap.
    #[test]
    fn attn_q4_layer_slices_returns_none_on_out_of_bounds_manifest() {
        let mmap_len = 1024;
        let payload = vec![0u8; mmap_len];
        let manifest = serde_json::json!([
            { "q4_offset": 0,   "q4_length": 128 },
            { "q4_offset": 128, "q4_length": 128 },
            { "q4_offset": 256, "q4_length": 128 },
            { "q4_offset": mmap_len, "q4_length": 1 }, // end = mmap_len + 1
        ]);
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(ATTN_WEIGHTS_Q4_BIN), &payload).unwrap();
        std::fs::write(
            tmp.path().join(ATTN_WEIGHTS_Q4_MANIFEST_JSON),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let mut idx = empty_vindex();
        idx.load_attn_q4(tmp.path())
            .expect("Q4 has no load-time len check");
        assert!(idx.attn_q4_layer_slices(0).is_none());
    }
}
