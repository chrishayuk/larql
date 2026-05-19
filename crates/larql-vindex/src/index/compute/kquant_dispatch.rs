//! Q4_K / Q6_K codec dispatch — fused decode + dot / scaled-add /
//! decode-into-buffer for FFN compute on quantised weights.
//!
//! Storage-side accessors (the mmap loaders, manifest parsing, cache
//! management) live in `crate::index::storage::ffn_store`. This module
//! reads `interleaved_kquant_layer_data` slices and routes them through
//! the registry (`crate::quant::registry`) — there are no inline
//! 144 / 210 byte-stride literals here.

use rayon::prelude::*;

use crate::index::core::VectorIndex;

impl VectorIndex {
    /// Direct Q4K/Q6K matmul — Y = X @ W.T, where W is the FFN matrix
    /// stored as Q4K/Q6K bytes in the vindex. Decodes and FMAs fused,
    /// parallelised across W rows. Zero extra RAM (no f32 cache).
    ///
    /// `x` is `[x_rows, w_cols]` row-major. `component` selects the layer's
    /// gate (0) / up (1) / down (2) Q4K slice. On return the output is
    /// `[x_rows, w_rows]` row-major where `w_rows` equals the slice's
    /// shape-0 (intermediate for gate/up, hidden for down).
    ///
    /// Dispatches to the backend's `q4k_matvec` / `q6k_matvec` when a
    /// compute backend is provided (Metal on Apple Silicon, CPU-SIMD
    /// otherwise) — one submission per X row. Falls back to the rayon
    /// + CPU-NEON scalar path when no backend is attached.
    pub fn kquant_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        if component > 2 {
            return None;
        }
        let slices = self.interleaved_kquant_layer_data(layer)?;
        let (bytes, format) = slices[component];

        let intermediate = self.num_features(layer);
        let hidden = self.hidden_size;
        let (w_rows, w_cols) = match component {
            0 | 1 => (intermediate, hidden),
            2 => (hidden, intermediate),
            _ => return None,
        };
        if x.len() != x_rows * w_cols {
            return None;
        }
        if w_cols % 256 != 0 {
            return None;
        }

        // Backend per-row dispatch is *slower* than CPU-NEON here because
        // each q4k_matvec call pays a Metal submission (~15 ms). With x_rows
        // × layers × 3 components we'd spend all our time in dispatch.
        // A batched Metal shader (one submission per layer) would fix this,
        // but we don't have it wired yet — keep the hook for future use.
        let _ = backend;

        // Format dispatch via the registry — one lookup, no inline 144/210
        // magic, no silent `_ => 0.0` arm scattered in the hot loop.
        let info = crate::quant::registry::lookup(format)?;
        let row_dot = info.row_dot?;
        let bytes_per_w_row = info.bytes_per_row(w_cols)?;

        // CPU fallback: rayon over W rows, NEON per-row dot.
        let mut y_t = vec![0.0f32; w_rows * x_rows];
        y_t.par_chunks_mut(x_rows)
            .enumerate()
            .for_each(|(j, slot)| {
                let w_row_start = j * bytes_per_w_row;
                let w_row = &bytes[w_row_start..w_row_start + bytes_per_w_row];
                for i in 0..x_rows {
                    let x_row = &x[i * w_cols..(i + 1) * w_cols];
                    slot[i] = row_dot(w_row, x_row).unwrap_or(0.0);
                }
            });
        let mut y = vec![0.0f32; x_rows * w_rows];
        for j in 0..w_rows {
            let src_base = j * x_rows;
            for i in 0..x_rows {
                y[i * w_rows + j] = y_t[src_base + i];
            }
        }
        Some(y)
    }

    /// Fused Q4K/Q6K decode + dot with `x` for one feature. Returns `None`
    /// if the row isn't available. This is ~2× faster than the
    /// `kquant_ffn_row_into` → BLAS sdot sequence because it skips the Vec
    /// allocation, the intermediate copy, and keeps the decoded data in
    /// registers.
    #[inline]
    pub fn kquant_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        if component > 2 || x.len() != self.hidden_size {
            return None;
        }
        let slices = self.interleaved_kquant_layer_data(layer)?;
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) {
            return None;
        }
        let info = crate::quant::registry::lookup(format)?;
        let row_dot = info.row_dot?;
        let bytes_per_row = info.bytes_per_row(hidden)?;
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() {
            return None;
        }
        row_dot(&bytes[start..end], x).ok()
    }

    /// Fused Q4K/Q6K decode + scaled-add into `out` for one feature of
    /// the gate (component 0) or up (component 1) leg.
    ///
    /// **Down (component 2) is rejected.** Down is stored
    /// `[hidden, intermediate]` on disk, so `feat`-th row is hidden-dim
    /// wide — not a single feature's down vector. Calling with
    /// `component == 2` here would silently produce wrong values
    /// (correct stride, wrong meaning). Callers wanting one feature's
    /// down vector must go through `kquant_ffn_row_scaled_add_via_cache`,
    /// which transposes the layer first. See ROADMAP W2.
    #[inline]
    pub fn kquant_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if component >= 2 || out.len() != self.hidden_size {
            return false;
        }
        let Some(slices) = self.interleaved_kquant_layer_data(layer) else {
            return false;
        };
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) {
            return false;
        }
        let Some(info) = crate::quant::registry::lookup(format) else {
            return false;
        };
        let Some(scaled_add) = info.row_scaled_add else {
            return false;
        };
        let Some(bytes_per_row) = info.bytes_per_row(hidden) else {
            return false;
        };
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() {
            return false;
        }
        scaled_add(&bytes[start..end], alpha, out).is_ok()
    }

    /// Fused Q4_K/Q6_K decode + `out += alpha * down[feat]` reading
    /// from `down_features_q4k.bin` — the W2 feature-major down path.
    ///
    /// When the vindex was extracted with `feature_major_down=true`,
    /// down lives in feature-major orientation on disk and a single
    /// row is one feature's down vector (`hidden`-dim wide). This
    /// skips the `kquant_ffn_layer` cache entirely — no whole-layer
    /// dequant, no transpose, no Mutex contention, no ~840 MB RSS
    /// ceiling on Gemma 4B.
    ///
    /// Returns `false` when `down_features_q4k.bin` isn't loaded —
    /// caller falls back to `kquant_ffn_row_scaled_add_via_cache`.
    #[inline]
    pub fn kquant_down_feature_scaled_add(
        &self,
        layer: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        let hidden = self.hidden_size;
        if out.len() != hidden {
            return false;
        }
        let Some((bytes, format, padded_width)) = self.down_features_q4k_layer_data(layer) else {
            return false;
        };
        if feat >= self.num_features(layer) {
            return false;
        }
        let Some(info) = crate::quant::registry::lookup(format) else {
            return false;
        };
        let Some(bytes_per_row) = info.bytes_per_row(padded_width) else {
            return false;
        };
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() {
            return false;
        }

        if padded_width == hidden {
            // Production fast path: row width matches hidden, fused
            // scaled-add writes straight into `out`.
            let Some(scaled_add) = info.row_scaled_add else {
                return false;
            };
            return scaled_add(&bytes[start..end], alpha, out).is_ok();
        }
        // Padded path: dequant the full padded row, accumulate the
        // first `hidden` floats. Used by synthetic fixtures with
        // `hidden % 256 != 0`; production hits the fast path above.
        let Ok(decoded) = (info.dequantize)(&bytes[start..end], padded_width) else {
            return false;
        };
        for (h, slot) in out.iter_mut().enumerate() {
            *slot += alpha * decoded[h];
        }
        true
    }

    /// Decode one row of a Q4K/Q6K FFN matrix directly into `out` without
    /// caching. `component`: 0=gate, 1=up, 2=down; `feat` is the feature
    /// (row) index; `out` must have length `hidden_size`. Returns `false`
    /// when the vindex has no Q4K data or shape is invalid.
    ///
    /// Row-level decode is the small-memory path for very large models
    /// (~30B+) where caching entire dequantised layers blows the RAM
    /// budget. Cost is ~50–70μs per row for hidden≈5376; at K=100 on a
    /// 60-layer model that's ~60 × 100 × 2 decodes × 60μs ≈ 720ms per
    /// forward pass.
    pub fn kquant_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        if component > 2 || out.len() != self.hidden_size {
            return false;
        }
        let Some(slices) = self.interleaved_kquant_layer_data(layer) else {
            return false;
        };
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) {
            return false;
        }

        let Some(info) = crate::quant::registry::lookup(format) else {
            return false;
        };
        let Some(bytes_per_row) = info.bytes_per_row(hidden) else {
            return false;
        };
        let start = feat * bytes_per_row;
        let end = start + bytes_per_row;
        if end > bytes.len() {
            return false;
        }
        match (info.dequantize)(&bytes[start..end], hidden) {
            Ok(v) => {
                out.copy_from_slice(&v[..hidden]);
                true
            }
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::index::core::VectorIndex;

    /// Locks in the W2 footgun fix: `kquant_ffn_row_scaled_add` rejects
    /// `component == 2` (down) up-front. Down on disk is
    /// `[hidden, intermediate]` so `feat`-th row is hidden-dim wide,
    /// not a single feature's down vector — calling this function
    /// with `component == 2` would have silently produced wrong
    /// values. The dispatch in `ffn_row_scaled_add` routes
    /// `component == 2` to either `kquant_down_feature_scaled_add` (W2)
    /// or `kquant_ffn_row_scaled_add_via_cache` (legacy); this raw entry
    /// point must refuse the coordinate explicitly.
    #[test]
    fn q4k_ffn_row_scaled_add_rejects_component_2() {
        let index = VectorIndex::empty(1, 256);
        let mut out = vec![0.0f32; 256];
        for component in [2usize, 3, 4, 99] {
            let ok = index.kquant_ffn_row_scaled_add(0, component, 0, 1.0, &mut out);
            assert!(!ok, "component {component} must be rejected");
        }
    }

    /// Mismatched output buffer size is rejected up-front — the
    /// scaled-add API contract is `out.len() == hidden_size`.
    #[test]
    fn q4k_ffn_row_scaled_add_rejects_wrong_out_len() {
        let index = VectorIndex::empty(1, 256);
        let mut bad = vec![0.0f32; 128]; // half-width
        let ok = index.kquant_ffn_row_scaled_add(0, 0, 0, 1.0, &mut bad);
        assert!(!ok, "out.len() != hidden_size must be rejected");
    }

    /// `kquant_down_feature_scaled_add` returns `false` when no feature-major
    /// down file is loaded — caller's responsibility to fall back to the
    /// cache path. The dispatch in `ffn_row_scaled_add` does exactly that.
    #[test]
    fn q4k_down_feature_scaled_add_returns_false_when_unloaded() {
        let index = VectorIndex::empty(1, 256);
        let mut out = vec![0.0f32; 256];
        assert!(!index.kquant_down_feature_scaled_add(0, 0, 1.0, &mut out));
    }

    // ── kquant_matmul_transb early returns ────────────────────────────

    #[test]
    fn q4k_matmul_transb_rejects_invalid_component() {
        let index = VectorIndex::empty(1, 256);
        let x = vec![0.0_f32; 256];
        for component in [3usize, 4, 99] {
            assert!(index
                .kquant_matmul_transb(0, component, &x, 1, None)
                .is_none());
        }
    }

    #[test]
    fn q4k_matmul_transb_returns_none_when_no_q4k_data() {
        // No interleaved_kquant mmap → interleaved_kquant_layer_data returns
        // None → matmul bails out.
        let index = VectorIndex::empty(1, 256);
        let x = vec![0.0_f32; 256];
        assert!(index.kquant_matmul_transb(0, 0, &x, 1, None).is_none());
    }

    // ── kquant_ffn_row_dot early returns ──────────────────────────────

    #[test]
    fn q4k_ffn_row_dot_rejects_invalid_component() {
        let index = VectorIndex::empty(1, 256);
        let x = vec![0.0_f32; 256];
        for component in [3usize, 99] {
            assert!(index.kquant_ffn_row_dot(0, component, 0, &x).is_none());
        }
    }

    #[test]
    fn q4k_ffn_row_dot_rejects_wrong_x_len() {
        let index = VectorIndex::empty(1, 256);
        let bad = vec![0.0_f32; 128]; // hidden_size is 256
        assert!(index.kquant_ffn_row_dot(0, 0, 0, &bad).is_none());
    }

    #[test]
    fn q4k_ffn_row_dot_returns_none_when_no_q4k_data() {
        let index = VectorIndex::empty(1, 256);
        let x = vec![0.0_f32; 256];
        assert!(index.kquant_ffn_row_dot(0, 0, 0, &x).is_none());
    }

    // ── kquant_ffn_row_scaled_add: existing tests covered c=2 and out.len()
    //
    // Add: no q4k data → false.

    #[test]
    fn q4k_ffn_row_scaled_add_returns_false_when_no_q4k_data() {
        let index = VectorIndex::empty(1, 256);
        let mut out = vec![0.0_f32; 256];
        // component=0, valid out.len() — fails on the `interleaved_kquant_layer_data
        // → None` guard, not on the out.len()/component checks.
        assert!(!index.kquant_ffn_row_scaled_add(0, 0, 0, 1.0, &mut out));
    }

    // ── kquant_down_feature_scaled_add early returns ──────────────────

    #[test]
    fn q4k_down_feature_scaled_add_rejects_wrong_out_len() {
        let index = VectorIndex::empty(1, 256);
        let mut bad = vec![0.0_f32; 128];
        assert!(!index.kquant_down_feature_scaled_add(0, 0, 1.0, &mut bad));
    }

    // ── kquant_ffn_row_into early returns ─────────────────────────────

    #[test]
    fn q4k_ffn_row_into_rejects_invalid_component() {
        let index = VectorIndex::empty(1, 256);
        let mut out = vec![0.0_f32; 256];
        for component in [3usize, 99] {
            assert!(!index.kquant_ffn_row_into(0, component, 0, &mut out));
        }
    }

    #[test]
    fn q4k_ffn_row_into_rejects_wrong_out_len() {
        let index = VectorIndex::empty(1, 256);
        let mut bad = vec![0.0_f32; 128];
        assert!(!index.kquant_ffn_row_into(0, 0, 0, &mut bad));
    }

    #[test]
    fn q4k_ffn_row_into_returns_false_when_no_q4k_data() {
        let index = VectorIndex::empty(1, 256);
        let mut out = vec![0.0_f32; 256];
        assert!(!index.kquant_ffn_row_into(0, 0, 0, &mut out));
    }
}
