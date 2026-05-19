//! WalkFfnConfig — per-layer K schedule for the unified walk kernel.
//!
//! `None` selects the dense-equivalent mmap path for that layer
//! (interleaved / q4 / full_mmap — chosen internally based on what
//! the vindex exposes). `Some(k)` selects the sparse walk path
//! (gate KNN → top-K up dot products → GEGLU → K down accumulations).

#[derive(Debug, Clone)]
pub struct WalkFfnConfig {
    /// Per-layer K. None = dense walk (all features). Some(k) = top-K sparse.
    pub k_per_layer: Vec<Option<usize>>,
    /// Skip features whose |activation| falls below this threshold.
    /// 0.0 preserves dense equivalence.
    pub activation_floor: f32,
}

impl WalkFfnConfig {
    /// Dense walk for every layer. Produces the same math as the classic
    /// `gate @ up @ down` matmul pipeline, routed through mmap'd vectors.
    pub fn dense(num_layers: usize) -> Self {
        Self {
            k_per_layer: vec![None; num_layers],
            activation_floor: 0.0,
        }
    }

    /// Uniform sparse walk at K per layer.
    pub fn sparse(num_layers: usize, k: usize) -> Self {
        Self {
            k_per_layer: vec![Some(k); num_layers],
            activation_floor: 0.0,
        }
    }

    /// Dense for `0..sparse_from`, sparse-K from `sparse_from..num_layers`.
    /// Matches the "dense early, sparse late" split used in hybrid configs.
    pub fn hybrid(num_layers: usize, sparse_from: usize, k: usize) -> Self {
        let mut k_per_layer = vec![None; num_layers];
        for slot in &mut k_per_layer[sparse_from.min(num_layers)..] {
            *slot = Some(k);
        }
        Self {
            k_per_layer,
            activation_floor: 0.0,
        }
    }

    /// Set the activation magnitude floor. Default 0.0 (no skip).
    pub fn with_floor(mut self, floor: f32) -> Self {
        self.activation_floor = floor;
        self
    }

    /// K for a layer. Out-of-range layers fall through to the last entry
    /// (or None if the config is empty) — mirrors `LayerFfnRouter::get`.
    pub fn k_for(&self, layer: usize) -> Option<usize> {
        if self.k_per_layer.is_empty() {
            return None;
        }
        let idx = layer.min(self.k_per_layer.len() - 1);
        self.k_per_layer[idx]
    }

    /// True when this layer should take the sparse walk path.
    pub fn is_sparse(&self, layer: usize) -> bool {
        self.k_for(layer).is_some()
    }

    pub fn num_layers(&self) -> usize {
        self.k_per_layer.len()
    }
}

impl Default for WalkFfnConfig {
    /// Empty config — all layers resolve to dense (None). Callers
    /// should prefer the named constructors when num_layers is known.
    fn default() -> Self {
        Self {
            k_per_layer: Vec::new(),
            activation_floor: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_sets_none_for_every_layer() {
        let cfg = WalkFfnConfig::dense(4);
        assert_eq!(cfg.num_layers(), 4);
        for l in 0..4 {
            assert_eq!(cfg.k_for(l), None);
            assert!(!cfg.is_sparse(l));
        }
        assert_eq!(cfg.activation_floor, 0.0);
    }

    #[test]
    fn sparse_sets_uniform_k_for_every_layer() {
        let cfg = WalkFfnConfig::sparse(3, 64);
        for l in 0..3 {
            assert_eq!(cfg.k_for(l), Some(64));
            assert!(cfg.is_sparse(l));
        }
    }

    #[test]
    fn hybrid_splits_at_sparse_from_index() {
        let cfg = WalkFfnConfig::hybrid(6, 3, 16);
        assert_eq!(cfg.k_for(0), None);
        assert_eq!(cfg.k_for(2), None);
        assert_eq!(cfg.k_for(3), Some(16));
        assert_eq!(cfg.k_for(5), Some(16));
    }

    #[test]
    fn hybrid_clamps_sparse_from_to_num_layers() {
        // sparse_from > num_layers must not panic — clamps so nothing
        // is sparse.
        let cfg = WalkFfnConfig::hybrid(4, 99, 8);
        for l in 0..4 {
            assert_eq!(cfg.k_for(l), None);
        }
    }

    #[test]
    fn with_floor_sets_activation_floor() {
        let cfg = WalkFfnConfig::dense(2).with_floor(0.01);
        assert_eq!(cfg.activation_floor, 0.01);
    }

    #[test]
    fn k_for_clamps_out_of_range_to_last_entry() {
        let cfg = WalkFfnConfig::hybrid(4, 2, 32);
        // Layer 99 clamps to last (index 3) — sparse.
        assert_eq!(cfg.k_for(99), Some(32));
    }

    #[test]
    fn k_for_empty_config_returns_none() {
        let cfg = WalkFfnConfig::default();
        assert_eq!(cfg.num_layers(), 0);
        assert_eq!(cfg.k_for(0), None);
        assert_eq!(cfg.k_for(99), None);
        assert!(!cfg.is_sparse(0));
    }

    #[test]
    fn default_matches_empty_dense() {
        let d = WalkFfnConfig::default();
        let e = WalkFfnConfig::dense(0);
        assert_eq!(d.num_layers(), e.num_layers());
        assert_eq!(d.activation_floor, e.activation_floor);
    }
}
