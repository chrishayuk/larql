//! `QuantizedFfnAccess` — Q4_0 / Q4_K / Q6_K FFN row access.

/// Q4_0/Q4_K/Q6_K FFN storage access.
pub trait QuantizedFfnAccess: Send + Sync {
    fn has_interleaved_q4(&self) -> bool {
        false
    }
    fn interleaved_q4_gate(&self, _layer: usize) -> Option<ndarray::Array2<f32>> {
        None
    }
    fn interleaved_q4_up(&self, _layer: usize) -> Option<ndarray::Array2<f32>> {
        None
    }
    fn interleaved_q4_down(&self, _layer: usize) -> Option<ndarray::Array2<f32>> {
        None
    }
    fn prefetch_interleaved_q4_layer(&self, _layer: usize) {}
    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        None
    }
    fn has_interleaved_kquant(&self) -> bool {
        false
    }
    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        None
    }
    /// Issue MADV_WILLNEED for the next layer's Q4_K/Q6_K FFN data so
    /// pages are streamed in while the current layer computes. No-op
    /// default for non-mmap implementations.
    fn prefetch_interleaved_kquant_layer(&self, _layer: usize) {}
    /// Per-layer FFN Q4_K/Q6_K slices — [gate, up, down] with format tags.
    /// `None` when the FFN manifest wasn't emitted (older vindexes).
    fn interleaved_kquant_layer_data(&self, _layer: usize) -> Option<[(&[u8], &str); 3]> {
        None
    }

    /// Whether feature-major Q4_K-encoded down vectors
    /// (`down_features_q4k.bin`) are loaded. When true,
    /// `kquant_down_feature_scaled_add` can serve component=2 row decode
    /// without going through the `kquant_ffn_layer` cache.
    fn has_down_features_kquant(&self) -> bool {
        false
    }

    /// W2: feature-major down decode. Returns `true` on success and
    /// writes `out += alpha * down[layer][feat]`. Returns `false` when
    /// the file isn't loaded; caller falls back to the cache path.
    fn kquant_down_feature_scaled_add(
        &self,
        _layer: usize,
        _feat: usize,
        _alpha: f32,
        _out: &mut [f32],
    ) -> bool {
        false
    }

    /// Dequantised Q4K/Q6K FFN matrix for `(layer, component)` where
    /// `component` is 0=gate, 1=up, 2=down. Lazily decoded and cached.
    /// Returns `None` when the vindex has no Q4K interleaved data.
    fn kquant_ffn_layer(
        &self,
        _layer: usize,
        _component: usize,
    ) -> Option<std::sync::Arc<Vec<f32>>> {
        None
    }

    /// Decode one row of a Q4K FFN matrix without caching. Small-memory
    /// alternative to `kquant_ffn_layer`. See `VectorIndex::kquant_ffn_row_into`.
    fn kquant_ffn_row_into(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _out: &mut [f32],
    ) -> bool {
        false
    }

    /// Fused Q4K/Q6K decode + dot — returns `dot(dequant(row), x)` without
    /// materialising the decoded row. See `VectorIndex::kquant_ffn_row_dot`.
    fn kquant_ffn_row_dot(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _x: &[f32],
    ) -> Option<f32> {
        None
    }

    /// Cache-based fused scaled-add for the down leg. Required because
    /// down is stored `[hidden, intermediate]` on disk — there is no
    /// per-row decode that gives a single feature's down vector
    /// without first transposing the layer (which is what
    /// `kquant_ffn_layer` does and caches). See ROADMAP W2.
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _alpha: f32,
        _out: &mut [f32],
    ) -> bool {
        false
    }

    /// Fused Q4K/Q6K decode + scaled-add — `out += alpha * dequant(row)`
    /// without materialising the decoded row.
    fn kquant_ffn_row_scaled_add(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _alpha: f32,
        _out: &mut [f32],
    ) -> bool {
        false
    }

    /// Direct Q4K/Q6K matmul — `Y = X @ W.T` against the layer's Q4K bytes.
    /// See `VectorIndex::kquant_matmul_transb`. `x` is `[x_rows, w_cols]`.
    /// `backend` (when provided) routes through Metal/CPU-SIMD kernels.
    fn kquant_matmul_transb(
        &self,
        _layer: usize,
        _component: usize,
        _x: &[f32],
        _x_rows: usize,
        _backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoOpQuant;
    impl QuantizedFfnAccess for NoOpQuant {}

    #[test]
    fn flag_defaults_are_false() {
        let n = NoOpQuant;
        assert!(!n.has_interleaved_q4());
        assert!(!n.has_interleaved_kquant());
        assert!(!n.has_down_features_kquant());
    }

    #[test]
    fn matrix_defaults_are_none() {
        let n = NoOpQuant;
        assert!(n.interleaved_q4_gate(0).is_none());
        assert!(n.interleaved_q4_up(0).is_none());
        assert!(n.interleaved_q4_down(0).is_none());
        assert!(n.interleaved_q4_mmap_ref().is_none());
        assert!(n.interleaved_kquant_mmap_ref().is_none());
        assert!(n.interleaved_kquant_layer_data(0).is_none());
        assert!(n.kquant_ffn_layer(0, 0).is_none());
    }

    #[test]
    fn row_dispatch_defaults_are_falsy() {
        let n = NoOpQuant;
        let mut out = [0.0_f32; 4];
        let x = [1.0_f32; 4];
        assert!(n.kquant_ffn_row_dot(0, 0, 0, &x).is_none());
        assert!(!n.kquant_ffn_row_into(0, 0, 0, &mut out));
        assert!(!n.kquant_ffn_row_scaled_add(0, 0, 0, 1.0, &mut out));
        assert!(!n.kquant_ffn_row_scaled_add_via_cache(0, 0, 0, 1.0, &mut out));
        assert!(!n.kquant_down_feature_scaled_add(0, 0, 1.0, &mut out));
        assert!(n.kquant_matmul_transb(0, 0, &x, 1, None).is_none());
    }

    #[test]
    fn prefetch_defaults_are_no_ops() {
        // Both prefetch hints must be safe to call on a non-mmap impl.
        let n = NoOpQuant;
        n.prefetch_interleaved_q4_layer(0);
        n.prefetch_interleaved_kquant_layer(0);
    }
}
