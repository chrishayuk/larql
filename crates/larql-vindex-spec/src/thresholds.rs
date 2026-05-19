//! Per-quant reconstruction thresholds.
//!
//! These live in the *spec* crate (not the manifest) so that bumping
//! a threshold is a spec-crate version bump — downstream tooling pins
//! it via Cargo. The validator pulls cosine_min / max_diff from here
//! at runtime; the manifest only declares the quant + dtype combo.
//!
//! The threshold matrix is conditioned on `(QuantFormat, StorageDtype)`.
//! When `QuantFormat == Q4K` the quant dominates loss and the dtype is
//! ignored; when `QuantFormat == None` the storage dtype drives the
//! tightness.
//!
//! FP4 storage is configured in the `extra["fp4"]` loader fields and
//! isn't validated by this crate in v1 — the FP4 compliance gate
//! already lives in `larql-vindex` and runs at extract time.

use crate::{QuantFormat, StorageDtype};

/// Validation thresholds for one (quant, dtype) combination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// Minimum cosine similarity between reconstructed and reference
    /// activations, computed per sampled layer.
    pub cosine_min: f32,

    /// Maximum element-wise absolute difference between reconstructed
    /// and reference activations, computed per sampled layer.
    pub max_diff: f32,
}

/// Threshold lookup for v1. Q4K dominates when present; otherwise the
/// storage dtype drives the bound.
pub fn thresholds_for(quant: QuantFormat, dtype: StorageDtype) -> Thresholds {
    match (quant, dtype) {
        (QuantFormat::Q4K, _) => Thresholds {
            cosine_min: 0.995,
            max_diff: 0.05,
        },
        (QuantFormat::None, StorageDtype::F16) => Thresholds {
            cosine_min: 0.9999,
            max_diff: 0.01,
        },
        (QuantFormat::None, StorageDtype::F32) => Thresholds {
            cosine_min: 0.999_99,
            max_diff: 0.001,
        },
    }
}

/// Deterministic sampled-layer pattern: `[0, L/4, L/2, 3L/4, L-1]`.
/// Five reads per validation regardless of model depth. Returns at
/// most `num_layers` distinct indices (collapses for very shallow
/// models).
pub fn sampled_layers(num_layers: u32) -> Vec<u32> {
    if num_layers == 0 {
        return Vec::new();
    }
    let last = num_layers - 1;
    let mut out = vec![
        0,
        num_layers / 4,
        num_layers / 2,
        (3 * num_layers) / 4,
        last,
    ];
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4k_strictness_independent_of_dtype() {
        let q4k_f16 = thresholds_for(QuantFormat::Q4K, StorageDtype::F16);
        let q4k_f32 = thresholds_for(QuantFormat::Q4K, StorageDtype::F32);
        assert_eq!(q4k_f16, q4k_f32);
    }

    #[test]
    fn f32_storage_is_strictest() {
        let f32 = thresholds_for(QuantFormat::None, StorageDtype::F32);
        let f16 = thresholds_for(QuantFormat::None, StorageDtype::F16);
        let q4k = thresholds_for(QuantFormat::Q4K, StorageDtype::F16);
        assert!(f32.cosine_min > f16.cosine_min);
        assert!(f16.cosine_min > q4k.cosine_min);
        assert!(f32.max_diff < f16.max_diff);
        assert!(f16.max_diff < q4k.max_diff);
    }

    #[test]
    fn sampled_layers_picks_five_for_typical_depth() {
        assert_eq!(sampled_layers(34), vec![0, 8, 17, 25, 33]);
    }

    #[test]
    fn sampled_layers_dedupes_shallow_models() {
        // 4-layer model: indices [0, 1, 2, 3, 3] → dedup → [0,1,2,3]
        assert_eq!(sampled_layers(4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn sampled_layers_empty_for_zero() {
        assert!(sampled_layers(0).is_empty());
    }
}
