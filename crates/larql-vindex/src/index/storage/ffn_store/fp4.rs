//! FP4 / FP8 FFN storage (exp 26) — load + dispatch the row-level
//! decode functions. Wraps the actual codec in `index/storage/fp4_store.rs`;
//! this module is the `VectorIndex`-facing API surface so the rest of
//! the crate can route through `ffn_row_*` without knowing whether the
//! backing storage is FP4, Q4_K, or f32.
//!
//! Carved out of `ffn_store.rs` in the 2026-04-25 modularity pass.

use crate::error::VindexError;
use crate::index::core::VectorIndex;

impl VectorIndex {
    /// Load FP4/FP8 FFN storage from `dir` per `config.fp4`. No-op when
    /// the manifest is absent (vindexes extracted before exp 26 don't
    /// have one). Returns an error only on filesystem issues or
    /// malformed manifests (e.g. file sizes that don't match the
    /// per-layer feature counts).
    pub fn load_fp4_storage(
        &mut self,
        dir: &std::path::Path,
        config: &crate::config::types::VindexConfig,
    ) -> Result<(), VindexError> {
        let Some(ref manifest) = config.fp4 else {
            return Ok(());
        };
        let layer_features: Vec<usize> = config.layers.iter().map(|l| l.num_features).collect();
        let storage = super::super::fp4_store::Fp4Storage::load(
            dir,
            manifest.clone(),
            layer_features,
            config.hidden_size,
        )?;
        self.ffn.fp4_storage = Some(std::sync::Arc::new(storage));
        Ok(())
    }

    /// Whether FP4/FP8 FFN storage is attached.
    pub fn has_fp4_storage(&self) -> bool {
        self.ffn.fp4_storage.is_some()
    }

    /// Fused dequant + dot for one FFN feature when FP4/FP8 storage is
    /// attached. `component` is 0=gate, 1=up, 2=down. Returns `None`
    /// if no FP4 storage is attached, if the projection is stored in
    /// f16/f32 (caller falls back to the legacy path), or if the
    /// coordinates are out of range.
    #[inline]
    pub fn fp4_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        let fp4 = self.ffn.fp4_storage.as_ref()?;
        fp4.row_dot(layer, component, feat, x)
    }

    /// Fused dequant + scaled-add for the FP4/FP8 path.
    #[inline]
    pub fn fp4_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        let Some(fp4) = self.ffn.fp4_storage.as_ref() else {
            return false;
        };
        fp4.row_scaled_add(layer, component, feat, alpha, out)
    }

    /// Dequantise one FFN feature into the caller's buffer (FP4/FP8 path).
    /// Counterpart of `kquant_ffn_row_into`.
    #[inline]
    pub fn fp4_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        let Some(fp4) = self.ffn.fp4_storage.as_ref() else {
            return false;
        };
        fp4.dequant_row_into(layer, component, feat, out)
    }
}
