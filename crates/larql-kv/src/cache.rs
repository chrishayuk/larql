//! Per-layer K/V tensor cache — the canonical engine-side state shape.
//!
//! Used by `StandardEngine`, `NoCacheEngine`, and the legacy generation
//! loops in [`crate::generation`] as the substrate for incremental decode:
//! each engine extends the cache during prefill and appends one new K/V
//! row per decode step.
//!
//! Memory: O(num_layers × window × kv_dim × 4 bytes) when bounded,
//! O(num_layers × seq_len × kv_dim × 4 bytes) when unbounded.
//!
//! Lifted out of `larql-inference::attention::decode` in 2026-05-16:
//! see `docs/specs/kv-engine-unification.md` for the migration rationale.

use larql_inference::attention::SharedKV;

/// Per-layer K/V cache. Can grow unbounded or be clamped to a fixed
/// sliding window (Markov-residual-bounded strategy — keep the last W
/// positions' K/V, evict older). When bounded, attention becomes
/// "look at the last W tokens" — identical to StreamingLLM / sliding
/// window approaches.
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    /// One entry per layer. `None` for layers that reuse another
    /// layer's K/V (Gemma 4 cross-layer sharing).
    pub layers: Vec<Option<SharedKV>>,
    /// When `Some(W)`, each layer's K/V is clipped to the last W
    /// positions after every append — the "bounded" part of the
    /// Markov Residual Bounded strategy. `None` = unbounded growth.
    pub max_window: Option<usize>,
    /// Absolute token position of the NEXT token to be appended.
    /// Used for RoPE: a new token's K needs RoPE at its true absolute
    /// position, not its row index in the clipped cache. Starts at 0
    /// and increments per append (not per eviction).
    pub next_position: usize,
}

impl KvCache {
    /// Unbounded cache — grows with every decode step.
    pub fn with_layers(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            max_window: None,
            next_position: 0,
        }
    }

    /// Bounded (Markov-residual-bounded) — keeps only the last
    /// `window` positions per layer. Memory stays O(window).
    pub fn with_window(num_layers: usize, window: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            max_window: if window == 0 { None } else { Some(window) },
            next_position: 0,
        }
    }

    /// Number of cached positions for a given layer. Returns 0 if the
    /// layer has no cache yet.
    pub fn cached_len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .and_then(|opt| opt.as_ref())
            .map(|(k, _)| k.shape()[0])
            .unwrap_or(0)
    }

    /// Apply the window bound to a layer's cache: if the cache has more
    /// than `max_window` rows, drop the oldest rows (keeping the tail).
    /// No-op when unbounded or under the limit.
    pub fn clip_layer(&mut self, layer: usize) {
        let window = match self.max_window {
            Some(w) => w,
            None => return,
        };
        let Some(Some((k, v))) = self.layers.get_mut(layer) else {
            return;
        };
        let rows = k.shape()[0];
        if rows <= window {
            return;
        }
        let start = rows - window;
        let k_slice = k.slice(ndarray::s![start..rows, ..]).to_owned();
        let v_slice = v.slice(ndarray::s![start..rows, ..]).to_owned();
        *k = k_slice;
        *v = v_slice;
    }

    // ── KV surgery ──────────────────────────────────────────────────────────
    //
    // Lazarus's `prefill_inject` and `kv_inject_test` need to lift K/V from
    // one cache into another. The fields are pub so callers could reach in,
    // but these methods give a stable, documented API and handle the
    // `Vec<Option<_>>` indexing in one place.

    /// Read K/V for a layer (post-RoPE K, post-V-norm V). `None` if the
    /// layer index is out of range or that layer's cache is empty (e.g.
    /// before prefill, or when the layer reuses another layer's K/V).
    pub fn get_layer(&self, layer: usize) -> Option<&SharedKV> {
        self.layers.get(layer).and_then(|opt| opt.as_ref())
    }

    /// Overwrite K/V for a layer with the supplied tensors. `K` and `V`
    /// must have the same row count. Caller is responsible for the rows
    /// being post-RoPE / post-V-norm — surgery happens at the same stage
    /// the forward pass writes.
    pub fn set_layer(&mut self, layer: usize, kv: SharedKV) {
        if layer >= self.layers.len() {
            return;
        }
        debug_assert_eq!(
            kv.0.shape()[0],
            kv.1.shape()[0],
            "K and V must have the same row count"
        );
        self.layers[layer] = Some(kv);
    }

    /// Clear a layer's cache. Subsequent decode at that layer will start
    /// fresh — i.e. attend only to new K/V.
    pub fn clear_layer(&mut self, layer: usize) {
        if let Some(slot) = self.layers.get_mut(layer) {
            *slot = None;
        }
    }

    /// Lift `other`'s entire K/V for `layer` into `self`. No-op if either
    /// side's layer is empty or out of range. Implements lazarus
    /// `kv_inject_test` (full-layer transplant).
    pub fn clone_layer_from(&mut self, other: &KvCache, layer: usize) {
        let Some((k, v)) = other.get_layer(layer) else {
            return;
        };
        self.set_layer(layer, (k.clone(), v.clone()));
    }

    /// Lift positions `[start..end]` of `other`'s `layer` K/V into `self`.
    /// Replaces `self`'s entire layer cache with the slice (it does not
    /// merge — concatenation/merge is the caller's job because each
    /// engine has its own append semantics).
    ///
    /// `start` is clamped to the donor's cache length; `end` is clamped
    /// to one past the last cached position. No-op if the resulting
    /// slice is empty or the donor's layer is missing.
    ///
    /// Implements lazarus `prefill_inject` (partial position transplant).
    pub fn clone_layer_position_range(
        &mut self,
        other: &KvCache,
        layer: usize,
        start: usize,
        end: usize,
    ) {
        let Some((k, v)) = other.get_layer(layer) else {
            return;
        };
        let cached = k.shape()[0];
        let s = start.min(cached);
        let e = end.min(cached);
        if s >= e {
            return;
        }
        let k_slice = k.slice(ndarray::s![s..e, ..]).to_owned();
        let v_slice = v.slice(ndarray::s![s..e, ..]).to_owned();
        self.set_layer(layer, (k_slice, v_slice));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn fill_kv(layer_rows: usize, kv_dim: usize, fill: f32) -> SharedKV {
        let k = Array2::from_elem((layer_rows, kv_dim), fill);
        let v = Array2::from_elem((layer_rows, kv_dim), fill);
        (k, v)
    }

    #[test]
    fn kv_cache_starts_empty() {
        let cache = KvCache::with_layers(4);
        assert_eq!(cache.cached_len(0), 0);
        assert_eq!(cache.next_position, 0);
    }

    #[test]
    fn kv_cache_with_window_clips() {
        let kv_dim = 4usize;
        let mut cache = KvCache::with_window(1, 2);
        for step in 0..3usize {
            let k = Array2::from_elem((1, kv_dim), step as f32);
            let v = Array2::from_elem((1, kv_dim), step as f32);
            let prior = cache.layers[0].take();
            let new_kv = if let Some((pk, pv)) = prior {
                let mut nk = Array2::zeros((pk.shape()[0] + 1, kv_dim));
                nk.slice_mut(ndarray::s![..pk.shape()[0], ..]).assign(&pk);
                nk.slice_mut(ndarray::s![pk.shape()[0].., ..]).assign(&k);
                let mut nv = Array2::zeros((pv.shape()[0] + 1, kv_dim));
                nv.slice_mut(ndarray::s![..pv.shape()[0], ..]).assign(&pv);
                nv.slice_mut(ndarray::s![pv.shape()[0].., ..]).assign(&v);
                (nk, nv)
            } else {
                (k, v)
            };
            cache.layers[0] = Some(new_kv);
            cache.clip_layer(0);
        }
        assert!(cache.cached_len(0) <= 2, "window=2 should cap at 2 entries");
    }

    #[test]
    fn get_layer_returns_none_when_empty() {
        let cache = KvCache::with_layers(2);
        assert!(cache.get_layer(0).is_none());
        assert!(cache.get_layer(99).is_none(), "out-of-range is None");
    }

    #[test]
    fn set_layer_then_get_layer_round_trips() {
        let mut cache = KvCache::with_layers(2);
        cache.set_layer(1, fill_kv(3, 4, 7.0));
        let (k, v) = cache.get_layer(1).expect("layer 1 set");
        assert_eq!(k.shape(), &[3, 4]);
        assert_eq!(v.shape(), &[3, 4]);
        assert_eq!(k[[0, 0]], 7.0);
        assert!(cache.get_layer(0).is_none());
    }

    #[test]
    fn set_layer_out_of_range_is_noop() {
        let mut cache = KvCache::with_layers(2);
        cache.set_layer(99, fill_kv(1, 4, 1.0));
        assert_eq!(cache.layers.len(), 2);
    }

    #[test]
    fn clear_layer_removes_kv() {
        let mut cache = KvCache::with_layers(2);
        cache.set_layer(0, fill_kv(2, 4, 1.0));
        assert!(cache.get_layer(0).is_some());
        cache.clear_layer(0);
        assert!(cache.get_layer(0).is_none());
    }

    #[test]
    fn clone_layer_from_copies_donor_kv() {
        let mut donor = KvCache::with_layers(2);
        donor.set_layer(1, fill_kv(4, 6, 2.5));

        let mut recipient = KvCache::with_layers(2);
        recipient.clone_layer_from(&donor, 1);

        let (k, v) = recipient.get_layer(1).unwrap();
        assert_eq!(k.shape(), &[4, 6]);
        assert_eq!(v[[0, 0]], 2.5);
    }

    #[test]
    fn clone_layer_from_missing_donor_layer_is_noop() {
        let donor = KvCache::with_layers(2);
        let mut recipient = KvCache::with_layers(2);
        recipient.set_layer(0, fill_kv(1, 4, 9.0));
        recipient.clone_layer_from(&donor, 0);
        assert_eq!(recipient.get_layer(0).unwrap().0[[0, 0]], 9.0);
    }

    #[test]
    fn clone_layer_position_range_slices_donor() {
        let mut donor = KvCache::with_layers(1);
        let kv_dim = 3usize;
        let k = Array2::from_shape_fn((5, kv_dim), |(r, _)| r as f32);
        let v = Array2::from_shape_fn((5, kv_dim), |(r, _)| r as f32);
        donor.set_layer(0, (k, v));

        let mut recipient = KvCache::with_layers(1);
        recipient.clone_layer_position_range(&donor, 0, 1, 4);
        let (rk, _) = recipient.get_layer(0).unwrap();
        assert_eq!(rk.shape(), &[3, kv_dim]);
        assert_eq!(rk[[0, 0]], 1.0, "first sliced row is donor row 1");
        assert_eq!(rk[[2, 0]], 3.0, "last sliced row is donor row 3");
    }

    #[test]
    fn clone_layer_position_range_clamps_to_donor_length() {
        let mut donor = KvCache::with_layers(1);
        donor.set_layer(0, fill_kv(2, 3, 1.0));
        let mut recipient = KvCache::with_layers(1);
        recipient.clone_layer_position_range(&donor, 0, 0, 99);
        let (rk, _) = recipient.get_layer(0).unwrap();
        assert_eq!(rk.shape(), &[2, 3]);
    }

    #[test]
    fn clone_layer_position_range_empty_slice_is_noop() {
        let mut donor = KvCache::with_layers(1);
        donor.set_layer(0, fill_kv(2, 3, 1.0));
        let mut recipient = KvCache::with_layers(1);
        recipient.clone_layer_position_range(&donor, 0, 5, 5);
        assert!(recipient.get_layer(0).is_none(), "empty range -> no write");
    }
}
