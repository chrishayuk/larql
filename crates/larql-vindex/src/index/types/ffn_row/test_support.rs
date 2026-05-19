//! `Stub` fixture — a configurable stand-in for the four FFN access
//! traits. Each backend bool toggles which `has_*` flag returns true
//! and which row methods route through, so tests can pin individual
//! arms of the dispatch cascade.

use ndarray::Array2;

use super::super::{Fp4FfnAccess, NativeFfnAccess, QuantizedFfnAccess};

/// Configurable stub. Each backend bool toggles which `has_*` flag
/// returns true and which row methods route through.
#[derive(Default)]
pub(super) struct Stub {
    // Native f32 mmap matrices, keyed by (component → matrix).
    pub(super) gate: Option<Array2<f32>>,
    pub(super) up: Option<Array2<f32>>,
    pub(super) down: Option<Array2<f32>>,
    // Alternate native arms (the dispatch chain tries
    // `interleaved_*` first, then falls through to `*_layer_matrix`).
    pub(super) up_layer: Option<Array2<f32>>,
    pub(super) down_layer: Option<Array2<f32>>,
    // Per-feature down vectors (preferred over interleaved_down in
    // the component=2 chain).
    pub(super) down_feature: Option<Vec<f32>>,
    // FP4 — when set, `fp4_ffn_row_*` returns predetermined sentinels
    // so dispatch routing is observable.
    pub(super) fp4_dot: Option<f32>,
    pub(super) fp4_scaled_add_returns: bool,
    pub(super) fp4_into_returns: bool,
    // Q4_K — when set, `q4k_ffn_row_*` returns predetermined sentinels.
    pub(super) q4k_dot: Option<f32>,
    pub(super) q4k_scaled_add_returns: bool,
    pub(super) q4k_into_returns: bool,
    pub(super) q4k_down_feature_returns: bool,
}

impl NativeFfnAccess for Stub {
    fn has_interleaved(&self) -> bool {
        self.gate.is_some() || self.up.is_some() || self.down.is_some()
    }
    fn has_full_mmap_ffn(&self) -> bool {
        self.up_layer.is_some() || self.down_layer.is_some()
    }
    fn has_down_features(&self) -> bool {
        self.down_feature.is_some()
    }
    fn interleaved_gate(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.gate.as_ref().map(|m| m.view())
    }
    fn interleaved_up(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.up.as_ref().map(|m| m.view())
    }
    fn interleaved_down(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.down.as_ref().map(|m| m.view())
    }
    fn up_layer_matrix(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.up_layer.as_ref().map(|m| m.view())
    }
    fn down_layer_matrix(&self, _: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.down_layer.as_ref().map(|m| m.view())
    }
    fn down_feature_vector(&self, _: usize, _: usize) -> Option<&[f32]> {
        self.down_feature.as_deref()
    }
}

impl QuantizedFfnAccess for Stub {
    fn has_interleaved_kquant(&self) -> bool {
        self.q4k_dot.is_some() || self.q4k_scaled_add_returns || self.q4k_into_returns
    }
    fn kquant_ffn_row_dot(&self, _: usize, _: usize, _: usize, _: &[f32]) -> Option<f32> {
        self.q4k_dot
    }
    fn kquant_ffn_row_scaled_add(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: f32,
        _: &mut [f32],
    ) -> bool {
        self.q4k_scaled_add_returns
    }
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: f32,
        _: &mut [f32],
    ) -> bool {
        self.q4k_scaled_add_returns
    }
    fn kquant_down_feature_scaled_add(&self, _: usize, _: usize, _: f32, _: &mut [f32]) -> bool {
        self.q4k_down_feature_returns
    }
    fn kquant_ffn_row_into(&self, _: usize, _: usize, _: usize, _: &mut [f32]) -> bool {
        self.q4k_into_returns
    }
}

impl Fp4FfnAccess for Stub {
    fn has_fp4_storage(&self) -> bool {
        self.fp4_dot.is_some() || self.fp4_scaled_add_returns || self.fp4_into_returns
    }
    fn fp4_ffn_row_dot(&self, _: usize, _: usize, _: usize, _: &[f32]) -> Option<f32> {
        self.fp4_dot
    }
    fn fp4_ffn_row_scaled_add(&self, _: usize, _: usize, _: usize, _: f32, _: &mut [f32]) -> bool {
        self.fp4_scaled_add_returns
    }
    fn fp4_ffn_row_into(&self, _: usize, _: usize, _: usize, _: &mut [f32]) -> bool {
        self.fp4_into_returns
    }
}

pub(super) fn one_row(values: &[f32]) -> Array2<f32> {
    Array2::from_shape_vec((1, values.len()), values.to_vec()).unwrap()
}
