//! Tests for the unified `GateIndex::ffn_row_dot` / `ffn_row_scaled_add`
//! / `ffn_row_into` dispatch priority: FP4 → native f32 → Q4K → None.
//!
//! Uses a minimal `Mock` impl of `GateIndex` that records which backend
//! each call dispatched into, so we can assert the priority chain
//! without constructing a real `VectorIndex` or loading mmap fixtures.
//!
//! The module is gated with `#[cfg(test)]` at its declaration in
//! `index/mod.rs`; no file-level cfg needed.

use ndarray::{Array1, Array2, ArrayView2};
use std::sync::Mutex;

use super::types::{
    FeatureMeta, FfnRowAccess, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides,
    QuantizedFfnAccess,
};

/// Test-only GateIndex implementation. Each backend flag controls
/// whether that layer fires; `last` tracks the dispatch trail.
struct Mock {
    fp4_on: bool,
    native_up: Option<Array2<f32>>,
    native_down: Option<Array2<f32>>,
    q4k_on: bool,
    last: Mutex<&'static str>,
    fp4_dot_return: Option<f32>,
    q4k_dot_return: Option<f32>,
}

impl Default for Mock {
    fn default() -> Self {
        Self {
            fp4_on: false,
            native_up: None,
            native_down: None,
            q4k_on: false,
            last: Mutex::new("none"),
            fp4_dot_return: None,
            q4k_dot_return: None,
        }
    }
}

impl Mock {
    fn mark(&self, label: &'static str) {
        *self.last.lock().unwrap() = label;
    }
    fn last(&self) -> &'static str {
        *self.last.lock().unwrap()
    }
}

impl GateLookup for Mock {
    fn gate_knn(&self, _layer: usize, _residual: &Array1<f32>, _top_k: usize) -> Vec<(usize, f32)> {
        vec![]
    }
    fn feature_meta(&self, _layer: usize, _feature: usize) -> Option<FeatureMeta> {
        None
    }
    fn num_features(&self, _layer: usize) -> usize {
        8
    }
}

impl PatchOverrides for Mock {}

impl Fp4FfnAccess for Mock {
    fn has_fp4_storage(&self) -> bool {
        self.fp4_on
    }
    fn fp4_ffn_row_dot(&self, _layer: usize, _c: usize, _f: usize, _x: &[f32]) -> Option<f32> {
        if !self.fp4_on {
            return None;
        }
        self.mark("fp4");
        self.fp4_dot_return
    }
    fn fp4_ffn_row_scaled_add(
        &self,
        _layer: usize,
        _c: usize,
        _f: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if !self.fp4_on {
            return false;
        }
        self.mark("fp4");
        for v in out.iter_mut() {
            *v += alpha * 1.0;
        }
        true
    }
    fn fp4_ffn_row_into(&self, _layer: usize, _c: usize, _f: usize, out: &mut [f32]) -> bool {
        if !self.fp4_on {
            return false;
        }
        self.mark("fp4");
        out.fill(42.0);
        true
    }
}

impl NativeFfnAccess for Mock {
    fn up_layer_matrix(&self, _layer: usize) -> Option<ArrayView2<'_, f32>> {
        self.native_up.as_ref().map(|m| m.view())
    }
    fn down_layer_matrix(&self, _layer: usize) -> Option<ArrayView2<'_, f32>> {
        self.native_down.as_ref().map(|m| m.view())
    }
    fn down_feature_vector(&self, _layer: usize, feat: usize) -> Option<&[f32]> {
        self.native_down
            .as_ref()
            .filter(|m| feat < m.nrows())
            .and_then(|m| m.row(feat).to_slice())
    }
}

impl QuantizedFfnAccess for Mock {
    fn has_interleaved_kquant(&self) -> bool {
        self.q4k_on
    }
    fn kquant_ffn_row_dot(&self, _layer: usize, _c: usize, _f: usize, _x: &[f32]) -> Option<f32> {
        if !self.q4k_on {
            return None;
        }
        self.mark("q4k");
        self.q4k_dot_return
    }
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        _layer: usize,
        _c: usize,
        _f: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if !self.q4k_on {
            return false;
        }
        self.mark("q4k_via_cache");
        for v in out.iter_mut() {
            *v += alpha * 2.0;
        }
        true
    }
    fn kquant_ffn_row_scaled_add(
        &self,
        _layer: usize,
        _c: usize,
        _f: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if !self.q4k_on {
            return false;
        }
        self.mark("q4k_direct");
        for v in out.iter_mut() {
            *v += alpha * 3.0;
        }
        true
    }
    fn kquant_ffn_row_into(&self, _layer: usize, _c: usize, _f: usize, out: &mut [f32]) -> bool {
        if !self.q4k_on {
            return false;
        }
        self.mark("q4k");
        out.fill(99.0);
        true
    }
}

mod tests {
    use super::*;

    fn make_native_row(rows: usize, cols: usize, fill: f32) -> Array2<f32> {
        Array2::from_elem((rows, cols), fill)
    }

    // ── ffn_row_dot ────────────────────────────────────────────────────────

    #[test]
    fn ffn_row_dot_priority_fp4_wins_over_native_and_q4k() {
        let m = Mock {
            fp4_on: true,
            fp4_dot_return: Some(1.23),
            native_up: Some(make_native_row(8, 4, 99.0)),
            q4k_on: true,
            q4k_dot_return: Some(4.56),
            ..Default::default()
        };
        let x = vec![0.1f32; 4];
        assert_eq!(m.ffn_row_dot(0, 1, 0, &x), Some(1.23));
        assert_eq!(m.last(), "fp4");
    }

    #[test]
    fn ffn_row_dot_falls_through_fp4_none_to_native() {
        let m = Mock {
            fp4_on: true,
            fp4_dot_return: None, // FP4 loaded but projection precision is f16/f32
            native_up: Some(make_native_row(8, 4, 2.0)),
            ..Default::default()
        };
        let x = vec![1.0f32; 4];
        let dot = m.ffn_row_dot(0, 1, 0, &x).unwrap();
        assert!((dot - 8.0).abs() < 1e-5, "native dot = 4 × 2.0 × 1.0 = 8");
    }

    #[test]
    fn ffn_row_dot_falls_through_to_q4k_when_no_native() {
        let m = Mock {
            q4k_on: true,
            q4k_dot_return: Some(7.0),
            ..Default::default()
        };
        let x = vec![0.5f32; 4];
        assert_eq!(m.ffn_row_dot(0, 1, 0, &x), Some(7.0));
        assert_eq!(m.last(), "q4k");
    }

    #[test]
    fn ffn_row_dot_returns_none_when_no_backend_covers() {
        let m = Mock::default();
        let x = vec![0.0f32; 4];
        assert!(m.ffn_row_dot(0, 1, 0, &x).is_none());
    }

    #[test]
    fn ffn_row_dot_respects_component_for_native() {
        let m = Mock {
            native_up: Some(make_native_row(8, 4, 1.0)),
            ..Default::default()
        };
        let x = vec![1.0; 4];
        assert_eq!(m.ffn_row_dot(0, 1, 0, &x), Some(4.0));
        assert!(
            m.ffn_row_dot(0, 2, 0, &x).is_none(),
            "down projection unset — no backend covers it"
        );
    }

    #[test]
    fn ffn_row_dot_bounds_fallthrough_in_native() {
        let m = Mock {
            native_up: Some(make_native_row(4, 4, 1.0)),
            ..Default::default()
        };
        let x = vec![1.0; 4];
        // feat 10 is out of range for the 4-row native matrix.
        assert!(m.ffn_row_dot(0, 1, 10, &x).is_none());
    }

    #[test]
    fn ffn_row_dot_shape_mismatch_fallthrough_to_q4k() {
        // Native has hidden=4, caller passes x of length 5. The unified
        // method's ncols check rejects native and falls through to Q4K.
        let m = Mock {
            native_up: Some(make_native_row(8, 4, 1.0)),
            q4k_on: true,
            q4k_dot_return: Some(42.0),
            ..Default::default()
        };
        let x = vec![1.0; 5];
        assert_eq!(m.ffn_row_dot(0, 1, 0, &x), Some(42.0));
        assert_eq!(m.last(), "q4k");
    }

    // ── ffn_row_scaled_add ─────────────────────────────────────────────────

    #[test]
    fn ffn_row_scaled_add_priority_fp4_wins() {
        let m = Mock {
            fp4_on: true,
            native_down: Some(make_native_row(8, 4, 99.0)),
            q4k_on: true,
            ..Default::default()
        };
        let mut out = vec![0.0f32; 4];
        assert!(m.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
        // fp4 stub adds alpha × 1.0.
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert_eq!(m.last(), "fp4");
    }

    #[test]
    fn ffn_row_scaled_add_falls_through_to_native_down() {
        let m = Mock {
            native_down: Some(make_native_row(8, 4, 2.5)),
            ..Default::default()
        };
        let mut out = vec![0.0f32; 4];
        assert!(m.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
        assert!(out.iter().all(|&v| (v - 2.5).abs() < 1e-6));
    }

    #[test]
    fn ffn_row_scaled_add_down_uses_q4k_via_cache() {
        // No FP4, no native. For component 2 (down), the unified method
        // must route Q4K to the via-cache variant (which handles
        // transposed-down storage efficiently).
        let m = Mock {
            q4k_on: true,
            ..Default::default()
        };
        let mut out = vec![0.0f32; 4];
        assert!(m.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert_eq!(m.last(), "q4k_via_cache");
    }

    #[test]
    fn ffn_row_scaled_add_gate_up_uses_direct_q4k() {
        // Components 0 / 1 use the non-via-cache Q4K variant.
        let m = Mock {
            q4k_on: true,
            ..Default::default()
        };
        let mut out = vec![0.0f32; 4];
        assert!(m.ffn_row_scaled_add(0, 1, 0, 1.0, &mut out));
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert_eq!(m.last(), "q4k_direct");
    }

    #[test]
    fn ffn_row_scaled_add_returns_false_when_no_backend() {
        let m = Mock::default();
        let mut out = vec![0.0f32; 4];
        assert!(!m.ffn_row_scaled_add(0, 2, 0, 1.0, &mut out));
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // ── ffn_row_into ───────────────────────────────────────────────────────

    #[test]
    fn ffn_row_into_priority_fp4_wins() {
        let m = Mock {
            fp4_on: true,
            native_up: Some(make_native_row(8, 4, 99.0)),
            ..Default::default()
        };
        let mut out = vec![0.0f32; 4];
        assert!(m.ffn_row_into(0, 1, 0, &mut out));
        assert!(out.iter().all(|&v| v == 42.0));
        assert_eq!(m.last(), "fp4");
    }

    #[test]
    fn ffn_row_into_falls_through_to_native() {
        let m = Mock {
            native_up: Some(make_native_row(8, 4, 7.5)),
            ..Default::default()
        };
        let mut out = vec![0.0f32; 4];
        assert!(m.ffn_row_into(0, 1, 0, &mut out));
        assert!(out.iter().all(|&v| v == 7.5));
    }

    #[test]
    fn ffn_row_into_falls_through_to_q4k() {
        let m = Mock {
            q4k_on: true,
            ..Default::default()
        };
        let mut out = vec![0.0f32; 4];
        assert!(m.ffn_row_into(0, 1, 0, &mut out));
        assert!(out.iter().all(|&v| v == 99.0));
        assert_eq!(m.last(), "q4k");
    }
}
