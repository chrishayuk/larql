//! `FfnRowAccess` — unified FFN row dispatch over native + Q4 + FP4
//! backends, plus the `GateIndex` compatibility composition.
//!
//! These two traits are grouped together because the blanket impls
//! cascade: every type that implements the three storage traits gets
//! `FfnRowAccess` for free, and every type that adds `GateLookup` +
//! `PatchOverrides` on top gets `GateIndex`. The dispatch logic in
//! `ffn_row_*` is the load-bearing default that keeps walk-kernel
//! callers storage-agnostic.
//!
//! Tests + the `Stub` fixture live in sibling modules
//! (`tests`, `test_support`) so the trait file stays under the soft
//! 600-LOC threshold (round-6 split, 2026-05-10).

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

use super::StorageBucket;
use super::{Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess};
use crate::index::storage::ffn_store::FFN_DOWN;

/// Unified FFN row operations over native, Q4K/Q6K, and FP4/FP8 storage.
pub trait FfnRowAccess: NativeFfnAccess + QuantizedFfnAccess + Fp4FfnAccess {
    // ── Unified FFN row access ─────────────────────────────────────────────
    //
    // One entry point per operation; the walk kernel calls these and
    // doesn't have to care about storage format. Default impls below
    // dispatch through the priority chain:
    //   1. FP4/FP8 (exp 26) — tried first when `has_fp4_storage()` is true
    //   2. Native f32 mmap  — interleaved / up_features / down_features
    //   3. Q4K interleaved  — `q4k_ffn_row_*` with via-cache for down
    //
    // Each step returns early on success. If every backend declines,
    // returns `None` / `false`.
    //
    // Overriding these in a concrete impl is rarely correct — the default
    // logic is the contract. Override the *specific* backend methods
    // (`fp4_ffn_row_dot`, `kquant_ffn_row_dot`, etc.) instead.

    /// Unified fused dequant + dot. `component`: 0=gate, 1=up, 2=down.
    /// Returns the dot product `row(layer, component, feat) · x` from
    /// whichever backend is loaded, or `None` if no backend covers this
    /// coordinate.
    fn ffn_row_dot(&self, layer: usize, component: usize, feat: usize, x: &[f32]) -> Option<f32> {
        // 1. FP4/FP8 backend (if loaded). fp4_ffn_row_dot returns None
        //    when the projection's precision tag is f16/f32 (caller
        //    falls through to native).
        if self.has_fp4_storage() {
            if let Some(dot) = self.fp4_ffn_row_dot(layer, component, feat, x) {
                return Some(dot);
            }
        }
        // 2. Native f32 mmap.
        let x_view = ndarray::ArrayView1::from(x);
        match component {
            0 => {
                if let Some(m) = self.interleaved_gate(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
            }
            1 => {
                if let Some(m) = self.interleaved_up(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
                if let Some(m) = self.up_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
            }
            2 => {
                if let Some(row) = self.down_feature_vector(layer, feat) {
                    if row.len() == x.len() {
                        return Some(ndarray::ArrayView1::from(row).dot(&x_view));
                    }
                }
                if let Some(m) = self.interleaved_down(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
                if let Some(m) = self.down_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == x.len() {
                        return Some(m.row(feat).dot(&x_view));
                    }
                }
            }
            _ => {}
        }
        // 3. Q4K fallback.
        if self.has_interleaved_kquant() {
            return self.kquant_ffn_row_dot(layer, component, feat, x);
        }
        None
    }

    /// Unified fused dequant + scaled-add: `out[i] += alpha * row[i]`.
    /// Returns `true` on success, `false` if no backend covers the
    /// coordinate (or shapes don't match).
    fn ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if self.has_fp4_storage() && self.fp4_ffn_row_scaled_add(layer, component, feat, alpha, out)
        {
            return true;
        }
        let mut out_view = ndarray::ArrayViewMut1::from(&mut out[..]);
        match component {
            0 => {
                if let Some(m) = self.interleaved_gate(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
            }
            1 => {
                if let Some(m) = self.interleaved_up(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
                if let Some(m) = self.up_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
            }
            2 => {
                if let Some(row) = self.down_feature_vector(layer, feat) {
                    if row.len() == out_view.len() {
                        out_view.scaled_add(alpha, &ndarray::ArrayView1::from(row));
                        return true;
                    }
                }
                if let Some(m) = self.interleaved_down(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
                if let Some(m) = self.down_layer_matrix(layer) {
                    if feat < m.nrows() && m.ncols() == out_view.len() {
                        out_view.scaled_add(alpha, &m.row(feat));
                        return true;
                    }
                }
            }
            _ => return false,
        }
        if self.has_interleaved_kquant() {
            if component == FFN_DOWN {
                // W2: prefer the feature-major down file when present —
                // a single row decode beats the whole-layer dequant +
                // transpose path. Fall back to the cache for vindexes
                // extracted before the feature-major down emit landed.
                if self.kquant_down_feature_scaled_add(layer, feat, alpha, out) {
                    return true;
                }
                return self
                    .kquant_ffn_row_scaled_add_via_cache(layer, component, feat, alpha, out);
            }
            return self.kquant_ffn_row_scaled_add(layer, component, feat, alpha, out);
        }
        false
    }

    /// Unified decode-into-buffer. `out.len()` must equal the row width.
    fn ffn_row_into(&self, layer: usize, component: usize, feat: usize, out: &mut [f32]) -> bool {
        if self.has_fp4_storage() && self.fp4_ffn_row_into(layer, component, feat, out) {
            return true;
        }
        let copy_row = |row: ndarray::ArrayView1<'_, f32>, out: &mut [f32]| -> bool {
            if row.len() != out.len() {
                return false;
            }
            for (i, &v) in row.iter().enumerate() {
                out[i] = v;
            }
            true
        };
        match component {
            0 => {
                if let Some(m) = self.interleaved_gate(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
            }
            1 => {
                if let Some(m) = self.interleaved_up(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
                if let Some(m) = self.up_layer_matrix(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
            }
            2 => {
                if let Some(row) = self.down_feature_vector(layer, feat) {
                    return copy_row(ndarray::ArrayView1::from(row), out);
                }
                if let Some(m) = self.interleaved_down(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
                if let Some(m) = self.down_layer_matrix(layer) {
                    if feat < m.nrows() {
                        return copy_row(m.row(feat), out);
                    }
                }
            }
            _ => return false,
        }
        if self.has_interleaved_kquant() {
            return self.kquant_ffn_row_into(layer, component, feat, out);
        }
        false
    }

    /// Bucket the index's primary FFN storage falls into. Encapsulates the
    /// `has_*`-flag logic so audits and tooling (e.g. `walk_path_audit`)
    /// don't scatter flag-checks across their bucketing logic.
    ///
    /// Priority mirrors `ffn_row_dot`'s dispatch chain (FP4 first, then
    /// native f32, then Q4K), so the bucket reflects what data the
    /// unified row dispatch will *actually* walk on a mixed-format vindex
    /// — not just which flags happen to be set.
    ///
    /// New storage formats should update this default impl so downstream
    /// consumers automatically pick up the right bucket. Override only
    /// when an implementer wants to pin the bucket explicitly (rare).
    fn primary_storage_bucket(&self) -> StorageBucket {
        if self.has_fp4_storage() {
            StorageBucket::Fp4
        } else if self.has_interleaved() || self.has_full_mmap_ffn() || self.has_down_features() {
            // Native f32 mmap available; ffn_row_* dispatch prefers it
            // over Q4K, so sparse on a mixed (f32 + Q4K) vindex walks
            // f32 features and lands in the Exact bucket.
            StorageBucket::Exact
        } else if self.has_interleaved_kquant() || self.has_interleaved_q4() {
            StorageBucket::Quantized
        } else {
            StorageBucket::Exact
        }
    }
}

impl<T> FfnRowAccess for T where T: NativeFfnAccess + QuantizedFfnAccess + Fp4FfnAccess + ?Sized {}

/// Compatibility trait for consumers that need the whole vindex surface.
///
/// New code should prefer the narrower traits above (`GateLookup`,
/// `PatchOverrides`, `NativeFfnAccess`, `QuantizedFfnAccess`,
/// `Fp4FfnAccess`, or `FfnRowAccess`) when it does not need the full
/// combined API.
pub trait GateIndex: GateLookup + PatchOverrides + FfnRowAccess {}

impl<T> GateIndex for T where T: GateLookup + PatchOverrides + FfnRowAccess + ?Sized {}
