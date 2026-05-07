//! `VectorIndex` metadata + gate-vector accessors and one-time setup.
//!
//! Pulls the read-only getters and `warmup` out of `gate.rs` so the
//! KNN-dispatch file stays focused on hot-path search code.
//!
//! - `feature_meta`, `down_meta_at`, `loaded_layers`, `num_features`,
//!   `num_features_at`, `total_gate_vectors`, `total_down_meta`:
//!   metadata readers (heap + mmap aware).
//! - `gate_vector`, `gate_vectors_at`, `gate_vectors_flat`:
//!   raw gate-matrix accessors (heap + mmap, single-row + bulk).
//! - `warmup`: pre-decode f16 mmap to f32 once so per-query KNN avoids
//!   re-decoding on every dispatch.

use ndarray::Array2;

use crate::index::core::VectorIndex;
use crate::index::types::*;

impl VectorIndex {
    /// Look up metadata for a specific feature.
    /// Checks heap first (mutation overrides), then mmap (production read path).
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        // Heap path first — catches mutation overrides (INSERT/UPDATE)
        if let Some(meta) = self
            .metadata
            .down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .and_then(|metas| metas.get(feature))
            .and_then(|m| m.clone())
        {
            return Some(meta);
        }
        // Mmap path (production — zero heap, no mutations)
        if let Some(ref dm) = self.metadata.down_meta_mmap {
            return dm.feature_meta(layer, feature);
        }
        None
    }

    /// Human-readable description of what the walk kernel will actually
    /// do on this vindex. Use to sanity-check a loaded vindex — if the
    /// description says "weights fallback" or "dense (legacy)", the
    /// vindex is not being used for FFN storage and that is probably
    /// not what the caller expected.
    ///
    /// Emitted by [`crate::format::load::load_vindex`] at load time
    /// when `LARQL_VINDEX_DESCRIBE=1` and by the CLI `--describe`
    /// flag. Also useful from tests to assert the expected storage
    /// backend is attached.
    pub fn describe_ffn_backend(&self) -> String {
        // Mirror the walk_ffn routing priority order (see
        // larql-inference::vindex::walk_ffn/mod.rs routing table).
        let mut parts = Vec::new();
        if let Some(fp4) = self.ffn.fp4_storage.as_ref() {
            let g = fp4.manifest.projections.gate.precision;
            let u = fp4.manifest.projections.up.precision;
            let d = fp4.manifest.projections.down.precision;
            parts.push(format!("FP4 sparse (gate={g}, up={u}, down={d})"));
        }
        if self.ffn.interleaved_q4k_mmap.is_some() {
            parts.push("Q4K interleaved".into());
        }
        if self.ffn.interleaved_q4_mmap.is_some() {
            parts.push("Q4_0 interleaved".into());
        }
        if self.ffn.interleaved_mmap.is_some() {
            parts.push("f32 interleaved".into());
        }
        if self.ffn.up_features_mmap.is_some() && self.ffn.down_features_mmap.is_some() {
            parts.push("full mmap (up+down f32)".into());
        }
        if self.gate.gate_mmap_bytes.is_some() {
            parts.push(format!("gate KNN ({:?} mmap)", self.gate.gate_mmap_dtype));
        }
        if parts.is_empty() {
            "weights fallback (safetensors — vindex not wired)".into()
        } else {
            parts.join(", ")
        }
    }

    /// Number of features indexed at a layer.
    ///
    /// Check order: legacy gate mmap slices → legacy heap gate vectors
    /// → FP4 storage's per-layer feature counts (exp 26). The FP4
    /// fallback fires when an FP4-only vindex has no legacy
    /// `gate_vectors.bin` mapped — without this, the walk kernel
    /// sees `num_features == 0` and falls through to the safetensors
    /// weights path, silently bypassing the vindex entirely.
    pub fn num_features(&self, layer: usize) -> usize {
        if self.gate.gate_mmap_bytes.is_some() {
            let n = self
                .gate
                .gate_mmap_slices
                .get(layer)
                .map(|s| s.num_features)
                .unwrap_or(0);
            if n > 0 {
                return n;
            }
        }
        if let Some(n) = self
            .gate
            .gate_vectors
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|m| m.shape()[0])
        {
            if n > 0 {
                return n;
            }
        }
        // FP4 storage fallback — layer_features is populated from
        // `index.json.layers[]` at load time.
        if let Some(ref fp4) = self.ffn.fp4_storage {
            if let Some(&n) = fp4.layer_features.get(layer) {
                return n;
            }
        }
        0
    }

    /// Total gate vectors loaded across all layers.
    pub fn total_gate_vectors(&self) -> usize {
        if self.gate.gate_mmap_bytes.is_some() {
            return self
                .gate
                .gate_mmap_slices
                .iter()
                .map(|s| s.num_features)
                .sum();
        }
        self.gate
            .gate_vectors
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .sum()
    }

    /// Total down metadata entries loaded across all layers.
    pub fn total_down_meta(&self) -> usize {
        if let Some(ref dm) = self.metadata.down_meta_mmap {
            return dm.total_features();
        }
        self.metadata
            .down_meta
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|metas| metas.iter().filter(|m| m.is_some()).count())
            .sum()
    }

    /// Layers that have gate vectors loaded.
    pub fn loaded_layers(&self) -> Vec<usize> {
        if self.gate.gate_mmap_bytes.is_some() {
            return self
                .gate
                .gate_mmap_slices
                .iter()
                .enumerate()
                .filter(|(_, s)| s.num_features > 0)
                .map(|(i, _)| i)
                .collect();
        }
        self.gate
            .gate_vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|_| i))
            .collect()
    }

    /// Access down metadata for a specific layer.
    pub fn down_meta_at(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        self.metadata
            .down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|v| v.as_slice())
    }

    /// Access gate vectors matrix for a specific layer (heap mode only).
    /// Returns None in mmap mode — use gate_knn() directly instead.
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&Array2<f32>> {
        self.gate.gate_vectors.get(layer).and_then(|v| v.as_ref())
    }

    /// Extract a single gate vector for a feature. Works in both heap and mmap mode.
    /// Returns the raw f32 vector (hidden_size elements).
    pub fn gate_vector(&self, layer: usize, feature: usize) -> Option<Vec<f32>> {
        // Heap path
        if let Some(Some(matrix)) = self.gate.gate_vectors.get(layer) {
            if feature < matrix.shape()[0] {
                return Some(matrix.row(feature).to_vec());
            }
            return None;
        }
        // Mmap path
        if let Some(ref mmap) = self.gate.gate_mmap_bytes {
            if let Some(slice) = self.gate.gate_mmap_slices.get(layer) {
                if feature >= slice.num_features {
                    return None;
                }
                let bpf = crate::config::dtype::bytes_per_float(self.gate.gate_mmap_dtype);
                let byte_offset = (slice.float_offset + feature * self.hidden_size) * bpf;
                let byte_count = self.hidden_size * bpf;
                if byte_offset + byte_count > mmap.len() {
                    return None;
                }
                let raw = &mmap[byte_offset..byte_offset + byte_count];
                return Some(crate::config::dtype::decode_floats(
                    raw,
                    self.gate.gate_mmap_dtype,
                ));
            }
        }
        None
    }

    /// Extract all gate vectors at a layer as flat f32 data.
    /// Returns (flat_data, num_features, hidden_size). Works in both heap and mmap mode.
    /// Use for bulk operations (SVD, PCA, numpy export).
    pub fn gate_vectors_flat(&self, layer: usize) -> Option<(Vec<f32>, usize, usize)> {
        // Heap path
        if let Some(Some(matrix)) = self.gate.gate_vectors.get(layer) {
            let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
            if let Some(data) = matrix.as_slice() {
                return Some((data.to_vec(), rows, cols));
            }
            // Non-contiguous — copy row by row
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                data.extend(matrix.row(r).iter());
            }
            return Some((data, rows, cols));
        }
        // Mmap path
        if let Some(ref mmap) = self.gate.gate_mmap_bytes {
            if let Some(slice) = self.gate.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 {
                    return None;
                }
                let bpf = crate::config::dtype::bytes_per_float(self.gate.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                if byte_offset + byte_count > mmap.len() {
                    return None;
                }
                let raw = &mmap[byte_offset..byte_offset + byte_count];
                let data = crate::config::dtype::decode_floats(raw, self.gate.gate_mmap_dtype);
                return Some((data, slice.num_features, self.hidden_size));
            }
        }
        None
    }

    /// Number of features at a layer (works in both heap and mmap mode).
    pub fn num_features_at(&self, layer: usize) -> usize {
        if self.gate.gate_mmap_bytes.is_some() {
            self.gate
                .gate_mmap_slices
                .get(layer)
                .map(|s| s.num_features)
                .unwrap_or(0)
        } else {
            self.num_features(layer)
        }
    }

    /// Release (ask the kernel to evict) resident pages for every mmap'd
    /// file this index holds. Best-effort: calls `madvise(MADV_DONTNEED)`
    /// on each mapping. On Linux this immediately drops clean pages from
    /// RSS; on Darwin MADV_DONTNEED is advisory and the kernel may delay.
    ///
    /// Use when serving as a long-lived FFN endpoint with a hard RSS
    /// cap — the next request will re-fault whatever pages it needs.
    /// Layer sharding (`--layers`) is the preferred route because it
    /// prevents out-of-shard pages from ever being touched; this method
    /// is for single-shard-holds-everything topologies that still want
    /// to bound RSS between requests.
    pub fn release_mmap_pages(&self) {
        use memmap2::UncheckedAdvice;
        // Linux: MADV_DONTNEED immediately drops clean pages from RSS.
        // Darwin: MADV_DONTNEED is advisory for shared file-backed mmap;
        // the kernel may defer release until memory pressure. Layer
        // sharding (`--layers`) is the strict bound on macOS; this call
        // is the strict bound on Linux.
        //
        // Safety: `unchecked_advise` requires no live references into the
        // mmap during the call. The server calls this from the walk-ffn
        // handler AFTER the per-request borrow of `patched` (and any
        // derived byte slices) has dropped — the handler closure builds
        // its own read-lock on `patched`, and the earlier request
        // closure has returned before this function runs.
        let advise = |m: &memmap2::Mmap| unsafe {
            let _ = m.unchecked_advise(UncheckedAdvice::DontNeed);
        };
        if let Some(ref m) = self.gate.gate_mmap_bytes {
            advise(m);
        }
        if let Some(ref m) = self.ffn.down_features_mmap {
            advise(m);
        }
        if let Some(ref m) = self.ffn.up_features_mmap {
            advise(m);
        }
        if let Some(ref m) = self.projections.lm_head_mmap {
            advise(m);
        }
        if let Some(ref m) = self.projections.lm_head_f16_mmap {
            advise(m);
        }
        if let Some(ref m) = self.ffn.interleaved_mmap {
            advise(m);
        }
        if let Some(ref m) = self.ffn.interleaved_q4_mmap {
            advise(m);
        }
        if let Some(ref m) = self.ffn.interleaved_q4k_mmap {
            advise(m);
        }
        if let Some(ref m) = self.gate.gate_q4_mmap {
            advise(m);
        }
        if let Some(ref m) = self.projections.lm_head_q4_mmap {
            advise(m);
        }
        if let Some(ref m) = self.projections.attn_q4k_mmap {
            advise(m);
        }
        if let Some(ref m) = self.projections.attn_q4_mmap {
            advise(m);
        }
        if let Some(ref m) = self.projections.attn_q8_mmap {
            advise(m);
        }
    }

    /// Pre-decode f16 gate vectors to f32 for lock-free access.
    /// For f32 vindexes this is a no-op — the mmap path is already zero-copy.
    pub fn warmup(&self) {
        if self.gate.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            return;
        }

        let Some(ref mmap) = self.gate.gate_mmap_bytes else {
            return;
        };
        let mut warmed = self.gate.warmed_gates.write().unwrap();
        if warmed.len() < self.num_layers {
            warmed.resize_with(self.num_layers, || None);
        }
        for layer in 0..self.num_layers {
            if warmed[layer].is_some() {
                continue;
            }
            if let Some(slice) = self.gate.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 {
                    continue;
                }
                let bpf = crate::config::dtype::bytes_per_float(self.gate.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                let byte_end = byte_offset + byte_count;
                if byte_end > mmap.len() {
                    continue;
                }
                let raw = &mmap[byte_offset..byte_end];
                warmed[layer] = Some(larql_models::quant::half::decode_f16(raw));
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
// release_mmap_pages smoke test
//
// RSS assertions are intentionally avoided: MADV_DONTNEED is advisory
// on macOS, racy on Linux under memory pressure, and flaky in CI. The
// contract we can meaningfully assert is that the method doesn't
// panic and leaves the index usable for subsequent queries.
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod release_mmap_pages_tests {
    use crate::config::dtype::StorageDtype;
    use crate::index::core::VectorIndex;
    use crate::index::types::GateLayerSlice;
    use ndarray::{Array1, Array2};

    #[test]
    fn release_mmap_pages_no_panic_on_heap_only_index() {
        // Heap-only index: no mmaps at all — release_mmap_pages must no-op.
        let hidden = 4;
        let gate0 = Array2::<f32>::zeros((2, hidden));
        let idx = VectorIndex::new(vec![Some(gate0)], vec![None], 1, hidden);
        assert!(!idx.is_mmap(), "heap-only index sanity check");
        // Must not panic — there are literally no mmaps to advise.
        idx.release_mmap_pages();
    }

    #[test]
    fn release_mmap_pages_no_panic_with_f16_gate_mmap() {
        // f16 mmap-backed index — exercises the `gate_mmap_bytes` arm
        // of `release_mmap_pages` on a valid mapping.
        let num_features = 2;
        let hidden = 4;
        let floats = num_features * hidden;
        let bytes = floats * 2;
        let mut anon = memmap2::MmapMut::map_anon(bytes).unwrap();
        let data = vec![1.0f32; floats];
        let encoded = larql_models::quant::half::encode_f16(&data);
        anon[..bytes].copy_from_slice(&encoded);
        let mmap = anon.make_read_only().unwrap();
        let slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features,
        }];
        let idx = VectorIndex::new_mmap(mmap, slices, StorageDtype::F16, None, 1, hidden);
        assert!(idx.is_mmap(), "mmap-backed index sanity check");

        // Baseline query to force at least one page fault + cache decode.
        let q = Array1::from_vec(vec![1.0f32; hidden]);
        let _ = idx.gate_knn(0, &q, 1);

        // Must not panic — the mmap is live and held by Arc.
        idx.release_mmap_pages();

        // And the index must stay usable afterwards — `gate_knn` will
        // re-fault whatever pages the kernel actually evicted.
        let hits = idx.gate_knn(0, &q, 1);
        assert!(
            !hits.is_empty(),
            "gate_knn must still work after page release"
        );
    }
}
