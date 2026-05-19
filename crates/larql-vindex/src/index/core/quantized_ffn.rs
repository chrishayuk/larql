//! `impl QuantizedFfnAccess for VectorIndex`.
//!
//! Delegation shim for the Q4-family FFN paths: legacy interleaved Q4,
//! interleaved Q4_K, the dequant cache, and the per-row matmul / dot /
//! scaled-add helpers. Two methods (`interleaved_q4_mmap_ref`,
//! `interleaved_kquant_mmap_ref`) are real implementations because they
//! reach directly into the `FfnStore` mmap handles.

use super::VectorIndex;
use crate::index::types::QuantizedFfnAccess;

impl QuantizedFfnAccess for VectorIndex {
    fn has_interleaved_q4(&self) -> bool {
        self.has_interleaved_q4()
    }

    fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_gate(layer)
    }

    fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_up(layer)
    }

    fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_down(layer)
    }

    fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        self.prefetch_interleaved_q4_layer(layer)
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.storage
            .interleaved_q4_whole_buffer_view()
            .map(|b| b.as_ref())
    }

    fn has_interleaved_kquant(&self) -> bool {
        self.has_interleaved_kquant()
    }

    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        self.storage
            .interleaved_kquant_whole_buffer_view()
            .map(|b| b.as_ref())
    }

    fn prefetch_interleaved_kquant_layer(&self, layer: usize) {
        self.prefetch_interleaved_kquant_layer(layer)
    }

    fn interleaved_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        VectorIndex::interleaved_kquant_layer_data(self, layer)
    }

    fn kquant_ffn_layer(&self, layer: usize, component: usize) -> Option<std::sync::Arc<Vec<f32>>> {
        VectorIndex::kquant_ffn_layer(self, layer, component)
    }

    fn kquant_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::kquant_ffn_row_into(self, layer, component, feat, out)
    }

    fn kquant_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        VectorIndex::kquant_ffn_row_dot(self, layer, component, feat, x)
    }

    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::kquant_ffn_row_scaled_add_via_cache(self, layer, component, feat, alpha, out)
    }

    fn has_down_features_kquant(&self) -> bool {
        VectorIndex::has_down_features_kquant(self)
    }

    fn kquant_down_feature_scaled_add(
        &self,
        layer: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::kquant_down_feature_scaled_add(self, layer, feat, alpha, out)
    }

    fn kquant_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::kquant_ffn_row_scaled_add(self, layer, component, feat, alpha, out)
    }

    fn kquant_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        VectorIndex::kquant_matmul_transb(self, layer, component, x, x_rows, backend)
    }
}

#[cfg(test)]
mod tests {
    //! Trait-impl shim coverage. Inherent Q4_K methods are exercised by
    //! the dedicated `q4k_*` integration tests + benches; here we just
    //! confirm each trait line runs and falls through to the inherent
    //! method's "no Q4_K data" guard.
    use super::*;

    fn fresh() -> VectorIndex {
        VectorIndex::empty(2, 8)
    }

    #[test]
    fn flag_methods_false_on_empty() {
        let v = fresh();
        assert!(!<VectorIndex as QuantizedFfnAccess>::has_interleaved_q4(&v));
        assert!(!<VectorIndex as QuantizedFfnAccess>::has_interleaved_kquant(&v));
        assert!(!<VectorIndex as QuantizedFfnAccess>::has_down_features_kquant(&v));
    }

    #[test]
    fn matrix_methods_none_on_empty() {
        let v = fresh();
        assert!(<VectorIndex as QuantizedFfnAccess>::interleaved_q4_gate(&v, 0).is_none());
        assert!(<VectorIndex as QuantizedFfnAccess>::interleaved_q4_up(&v, 0).is_none());
        assert!(<VectorIndex as QuantizedFfnAccess>::interleaved_q4_down(&v, 0).is_none());
    }

    #[test]
    fn mmap_refs_none_on_empty() {
        let v = fresh();
        assert!(<VectorIndex as QuantizedFfnAccess>::interleaved_q4_mmap_ref(&v).is_none());
        assert!(<VectorIndex as QuantizedFfnAccess>::interleaved_kquant_mmap_ref(&v).is_none());
        assert!(
            <VectorIndex as QuantizedFfnAccess>::interleaved_kquant_layer_data(&v, 0).is_none()
        );
    }

    #[test]
    fn q4k_ffn_layer_none_on_empty() {
        let v = fresh();
        assert!(<VectorIndex as QuantizedFfnAccess>::kquant_ffn_layer(&v, 0, 0).is_none());
    }

    #[test]
    fn row_dispatch_methods_falsy_on_empty() {
        let v = fresh();
        let x = [1.0_f32; 8];
        let mut out = [0.0_f32; 8];
        assert!(<VectorIndex as QuantizedFfnAccess>::kquant_ffn_row_dot(&v, 0, 0, 0, &x).is_none());
        assert!(!<VectorIndex as QuantizedFfnAccess>::kquant_ffn_row_into(
            &v, 0, 0, 0, &mut out
        ));
        assert!(
            !<VectorIndex as QuantizedFfnAccess>::kquant_ffn_row_scaled_add_via_cache(
                &v, 0, 0, 0, 1.0, &mut out
            )
        );
        assert!(
            !<VectorIndex as QuantizedFfnAccess>::kquant_ffn_row_scaled_add(
                &v, 0, 0, 0, 1.0, &mut out
            )
        );
        assert!(
            !<VectorIndex as QuantizedFfnAccess>::kquant_down_feature_scaled_add(
                &v, 0, 0, 1.0, &mut out
            )
        );
    }

    #[test]
    fn q4k_matmul_transb_none_on_empty() {
        let v = fresh();
        let x = [1.0_f32; 8];
        let backend = larql_compute::CpuBackend;
        assert!(<VectorIndex as QuantizedFfnAccess>::kquant_matmul_transb(
            &v,
            0,
            0,
            &x,
            1,
            Some(&backend)
        )
        .is_none());
        // Backend=None path also covered.
        assert!(
            <VectorIndex as QuantizedFfnAccess>::kquant_matmul_transb(&v, 0, 0, &x, 1, None)
                .is_none()
        );
    }

    #[test]
    fn prefetch_methods_safe_on_empty() {
        let v = fresh();
        <VectorIndex as QuantizedFfnAccess>::prefetch_interleaved_q4_layer(&v, 0);
        <VectorIndex as QuantizedFfnAccess>::prefetch_interleaved_kquant_layer(&v, 0);
    }
}
