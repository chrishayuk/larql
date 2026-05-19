//! Shared walk-path helpers.

use crate::vindex::walk_config::WalkFfnConfig;

/// True when the user asked for full-K (K ≥ feature count) — the signal
/// that we should route the walk through batched gemm rather than a
/// per-feature loop. Treats `usize::MAX` (set by `::dense` / `--k full`)
/// as full-K; also caches the check when top-K happens to exceed the
/// layer's feature count.
#[inline]
pub(super) fn hits_len_ge_intermediate(
    config: &WalkFfnConfig,
    layer: usize,
    intermediate: usize,
) -> bool {
    match config.k_for(layer) {
        Some(k) => k >= (intermediate * 8) / 10,
        None => true,
    }
}

/// Dispatch-trace entry: records which walk path fired for a given
/// `(forward_call, layer)`. Enabled via `WalkFfn::with_dispatch_trace()`.
///
/// Each walk path function calls `ctx.trace_path(layer, "name")` on
/// exit. Tests assert the expected sequence; the Q2 debugging flow
/// uses the trace to identify which path consumed a given vindex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchEntry {
    pub layer: usize,
    pub path: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_config(layers: usize) -> WalkFfnConfig {
        WalkFfnConfig::dense(layers)
    }

    fn sparse_config(layers: usize, k: usize) -> WalkFfnConfig {
        WalkFfnConfig::sparse(layers, k)
    }

    #[test]
    fn hits_len_full_k_returns_true() {
        // dense() sets k_per_layer to None — should always pick the
        // full-K path.
        let cfg = dense_config(2);
        assert!(hits_len_ge_intermediate(&cfg, 0, 1024));
        assert!(hits_len_ge_intermediate(&cfg, 1, 32));
    }

    #[test]
    fn hits_len_sparse_below_80_percent_returns_false() {
        // k=10, intermediate=100 → threshold 80 → 10 < 80.
        let cfg = sparse_config(1, 10);
        assert!(!hits_len_ge_intermediate(&cfg, 0, 100));
    }

    #[test]
    fn hits_len_sparse_at_80_percent_returns_true() {
        // k=80, intermediate=100 → threshold 80 → 80 >= 80.
        let cfg = sparse_config(1, 80);
        assert!(hits_len_ge_intermediate(&cfg, 0, 100));
    }

    #[test]
    fn hits_len_sparse_above_intermediate_returns_true() {
        // k=200 > intermediate=100 → full-K equivalent.
        let cfg = sparse_config(1, 200);
        assert!(hits_len_ge_intermediate(&cfg, 0, 100));
    }

    #[test]
    fn dispatch_entry_equality_is_field_wise() {
        let a = DispatchEntry {
            layer: 3,
            path: "interleaved_q4",
        };
        let b = DispatchEntry {
            layer: 3,
            path: "interleaved_q4",
        };
        let c = DispatchEntry {
            layer: 3,
            path: "sparse",
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
