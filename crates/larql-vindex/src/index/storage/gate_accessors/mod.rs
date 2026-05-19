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
use crate::index::storage::vindex_storage::VindexStorage;
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
        if self.storage.has_interleaved_kquant() {
            parts.push("Q4K interleaved".into());
        }
        if self.storage.has_interleaved_q4() {
            parts.push("Q4_0 interleaved".into());
        }
        if self.storage.has_interleaved_f32() {
            parts.push("f32 interleaved".into());
        }
        if self.storage.has_up_features() && self.storage.has_down_features() {
            parts.push("full mmap (up+down f32)".into());
        }
        if self.storage.has_gate_vectors() {
            parts.push(format!("gate KNN ({:?} mmap)", self.storage.gate_dtype()));
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
    /// → FP4 storage's per-layer feature counts (exp 26) → Q4_K/Q6_K
    /// FFN intermediate width derived from the gate manifest entry.
    ///
    /// The FP4 fallback fires when an FP4-only vindex has no legacy
    /// `gate_vectors.bin` mapped. The Q4K fallback fires for
    /// quantised-FFN-only vindexes (Gemma 3 4B Q4K and similar) where
    /// gate vectors live inside `interleaved_kquant.bin` and no separate
    /// `gate_vectors.bin` is emitted. Without these fallbacks the walk
    /// kernel sees `num_features == 0` and falls through to the
    /// safetensors weights path — silently bypassing the vindex and
    /// regressing FFN throughput ~100× on Gemma 3 4B Q4K.
    pub fn num_features(&self, layer: usize) -> usize {
        if self.storage.has_gate_vectors() {
            let n = self
                .storage
                .gate_layer_slice(layer)
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
                if n > 0 {
                    return n;
                }
            }
        }
        // Q4_K/Q6_K FFN width fallback — derive intermediate dim from
        // the gate component's manifest byte length. Same data the
        // matmul kernel consults, so this can't drift out of sync.
        if let Some(n) = self.kquant_ffn_intermediate_width(layer) {
            return n;
        }
        0
    }

    /// FFN intermediate width derived from the layer's gate Q4_K/Q6_K
    /// manifest entry. Returns `None` when no Q4K FFN bytes are mapped
    /// for this layer, the format tag is unknown to the registry, or
    /// the recorded byte length isn't a whole number of rows.
    ///
    /// Surface used as a `num_features` fallback for FFN-only vindexes
    /// (no gate_vectors.bin, no FP4 storage). Reading the same manifest
    /// the matmul kernel reads keeps the width authoritative — there's
    /// no shape duplication that could drift.
    pub(crate) fn kquant_ffn_intermediate_width(&self, layer: usize) -> Option<usize> {
        let slices = self.interleaved_kquant_layer_data(layer)?;
        let (gate_bytes, gate_fmt) = slices[0];
        let info = crate::quant::registry::lookup(gate_fmt)?;
        let bytes_per_row = info.bytes_per_row(self.hidden_size)?;
        let len = gate_bytes.len();
        if !len.is_multiple_of(bytes_per_row) {
            return None;
        }
        Some(len / bytes_per_row)
    }

    /// Total gate vectors loaded across all layers.
    pub fn total_gate_vectors(&self) -> usize {
        if self.storage.has_gate_vectors() {
            return self
                .storage
                .gate_layer_slices()
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
        if self.storage.has_gate_vectors() {
            return self
                .storage
                .gate_layer_slices()
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
        if let Some(view) = self.storage.gate_layer_view(layer) {
            if feature >= view.slice.num_features {
                return None;
            }
            let bpf = crate::config::dtype::bytes_per_float(view.dtype);
            let byte_offset = (view.slice.float_offset + feature * self.hidden_size) * bpf;
            let byte_count = self.hidden_size * bpf;
            let mmap: &[u8] = view.bytes.as_ref();
            if byte_offset + byte_count > mmap.len() {
                return None;
            }
            let raw = &mmap[byte_offset..byte_offset + byte_count];
            return Some(crate::config::dtype::decode_floats(raw, view.dtype));
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
        if let Some(view) = self.storage.gate_layer_view(layer) {
            if view.slice.num_features == 0 {
                return None;
            }
            let bpf = crate::config::dtype::bytes_per_float(view.dtype);
            let byte_offset = view.slice.float_offset * bpf;
            let byte_count = view.slice.num_features * self.hidden_size * bpf;
            let mmap: &[u8] = view.bytes.as_ref();
            if byte_offset + byte_count > mmap.len() {
                return None;
            }
            let raw = &mmap[byte_offset..byte_offset + byte_count];
            let data = crate::config::dtype::decode_floats(raw, view.dtype);
            return Some((data, view.slice.num_features, self.hidden_size));
        }
        None
    }

    /// Number of features at a layer (works in both heap and mmap mode).
    pub fn num_features_at(&self, layer: usize) -> usize {
        if self.storage.has_gate_vectors() {
            self.storage
                .gate_layer_slice(layer)
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
        // Linux: MADV_DONTNEED immediately drops clean pages from RSS.
        // Darwin: MADV_DONTNEED is advisory for shared file-backed
        // mmap; the kernel may defer release until memory pressure.
        //
        // Every mmap-backed file in the vindex is registered with
        // `MmapStorage::mmap_handles` (the setters track them as they
        // install bytes). One call covers all of them.
        // Heap-backed entries (synth lm_head) aren't registered, so
        // iterating is safe.
        //
        // Safety: `unchecked_advise` requires no live references into
        // the mmap during the call. The server calls this from the
        // walk-ffn handler AFTER the per-request borrow of `patched`
        // (and any derived byte slices) has dropped.
        self.storage.release_pages();
    }

    /// Pre-decode f16 gate vectors to f32 for lock-free access.
    /// For f32 vindexes this is a no-op — the mmap path is already zero-copy.
    pub fn warmup(&self) {
        if self.storage.gate_dtype() == crate::config::dtype::StorageDtype::F32 {
            return;
        }

        let Some(bytes) = self.storage.gate_bytes_view() else {
            return;
        };
        let mmap: &[u8] = bytes.as_ref();
        let dtype = self.storage.gate_dtype();
        let mut warmed = self.gate.warmed_gates.write().unwrap();
        if warmed.len() < self.num_layers {
            warmed.resize_with(self.num_layers, || None);
        }
        for layer in 0..self.num_layers {
            if warmed[layer].is_some() {
                continue;
            }
            if let Some(slice) = self.storage.gate_layer_slice(layer) {
                if slice.num_features == 0 {
                    continue;
                }
                let bpf = crate::config::dtype::bytes_per_float(dtype);
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
mod release_pages_tests;

#[cfg(test)]
mod accessor_tests;
