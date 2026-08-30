//! VI3-KV-1: the canonical [`KvCache`] behind the VINDEX3 [`KvState`]
//! contract — deliberately unambitious.
//!
//! One target, nothing else: the ordinary canonical `larql-kv` cache
//! satisfies the V3 continuation-state contract with **no change in
//! semantics**. No windowing (that is VI3-KV-2 — the executor's span
//! logic owns position exclusion, so wiring `LayerKvGeometry::window`
//! into [`KvCache::max_window`] here would *change* semantics, not
//! adapt them). No quantisation, no residual state, no residency
//! optimisation.
//!
//! The two contracts already say the same thing: [`KvCache`] stores
//! "post-RoPE K, post-V-norm V" per layer with `next_position` as the
//! absolute append count ("increments per append, not per eviction"),
//! and the V3 step contract appends exactly the post-norm, post-rope
//! rows the backend returned, with the provider owning the logical
//! continuation position. This module is the proof that the overlap
//! is real: the KV-1 gates demand [`RowKvState`] and
//! [`CanonicalKvState`] stay **bit-identical** through prefill,
//! resume, and decode — chaining `V3 batch == V3 tokenwise ==
//! RowKvState == larql-kv canonical cache` onto the existing
//! executor ≡ production-forward parity.
//!
//! Geometry comes from the executable plan and nowhere else: the
//! adapter records what [`KvState::prepare`] announces —
//! `larql-kv` needs no `ModelArchitecture` inference to know a
//! VINDEX3 model's continuation geometry (row width, sliding/full
//! split). That closure is an explicit gate, not an incidental fact.
//!
//! Adapter shape: the cache's matrices are the **storage authority**.
//! Today's `KvState` read contract serves `&[Vec<f32>]` rows (it
//! mirrors `AttentionStepCall`, a recorded debt), so the adapter keeps
//! per-layer row views — materialised *from* the matrices after every
//! write, never alongside them, so the served bits are the stored
//! bits by construction.

use larql_vindex::format::vindex3::opplan::exec::continuation::{
    LayerContinuationGeometry, RecurrentState,
};
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, KvState, LayerKvGeometry,
};
use ndarray::Array2;

use crate::cache::KvCache;

#[cfg(test)]
mod tests;

/// Per-layer row views, derived from the cache's matrices.
#[derive(Default, Clone)]
struct LayerRows {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl LayerRows {
    /// Rebuild both views from one layer's matrices.
    fn from_matrices(kv: Option<&(Array2<f32>, Array2<f32>)>) -> Self {
        match kv {
            Some((k, v)) => Self {
                keys: k.rows().into_iter().map(|row| row.to_vec()).collect(),
                values: v.rows().into_iter().map(|row| row.to_vec()).collect(),
            },
            None => Self::default(),
        }
    }
}

/// The canonical [`KvCache`] as a VINDEX3 continuation-state provider.
///
/// Owns the cache; [`cache`](Self::cache) and
/// [`into_cache`](Self::into_cache) hand the state to the existing KV
/// machinery (engines, surgery, injection), and
/// [`from_cache`](Self::from_cache) brings it back — the same
/// conversation state crossing the `larql-kv` boundary intact.
pub struct CanonicalKvState {
    cache: KvCache,
    /// Per layer, ABSOLUTE index: `Some` where the plan declares KV rows,
    /// `None` on a recurrent layer (whose state lives in
    /// [`Self::recurrent`] instead). Absolute indexing is load-bearing —
    /// filtering to a KV subset is exactly how a hybrid stack's layer 7
    /// would read layer 3's rows.
    geometry: Vec<Option<LayerKvGeometry>>,
    rows: Vec<LayerRows>,
    /// SERVE-HYBRID: the recurrent layers' durable buffers, absolute
    /// index, allocated from the plan's declared geometry at
    /// [`prepare_continuation`](KvState::prepare_continuation). A
    /// KV-only program allocates none and behaves exactly as before.
    recurrent: Vec<Option<RecurrentState>>,
}

impl Default for CanonicalKvState {
    fn default() -> Self {
        Self::new()
    }
}

impl CanonicalKvState {
    /// An empty provider; the first `prepare` sizes it from the plan.
    pub fn new() -> Self {
        Self {
            cache: KvCache::with_layers(0),
            geometry: Vec::new(),
            rows: Vec::new(),
            recurrent: Vec::new(),
        }
    }

    /// Adopt an existing canonical cache (engine-held, injected,
    /// transplanted) as V3 continuation state. Row views are rebuilt
    /// from the cache's matrices — the matrices stay the authority —
    /// and the next `prepare` validates the plan's geometry against
    /// them. The cache must be unwindowed: a clipped cache has evicted
    /// rows this contract requires (windowing is VI3-KV-2).
    pub fn from_cache(cache: KvCache) -> Self {
        assert!(
            cache.max_window.is_none(),
            "canonical V3 continuation state requires an unwindowed cache"
        );
        let rows = cache
            .layers
            .iter()
            .map(|kv| LayerRows::from_matrices(kv.as_ref()))
            .collect();
        Self {
            cache,
            geometry: Vec::new(),
            rows,
            recurrent: Vec::new(),
        }
    }

    /// The held cache — the storage authority, readable by everything
    /// that already speaks [`KvCache`].
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    /// Surrender the cache to the existing KV machinery.
    pub fn into_cache(self) -> KvCache {
        self.cache
    }

    /// The plan-declared geometry this provider was prepared with —
    /// row width and sliding/full window per layer, read from the
    /// executable program, never from `ModelArchitecture`.
    pub fn geometry(&self) -> Vec<LayerKvGeometry> {
        self.geometry.iter().flatten().cloned().collect()
    }
}

impl KvState for CanonicalKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        let layers_opt: Vec<Option<LayerKvGeometry>> = layers.iter().cloned().map(Some).collect();
        if self.geometry.is_empty() {
            if self.cache.layers.is_empty() {
                self.cache = KvCache::with_layers(layers.len());
                self.rows = vec![LayerRows::default(); layers.len()];
            } else {
                // The from_cache path: the held state must be state
                // for a program of this shape.
                assert_eq!(
                    self.cache.layers.len(),
                    layers.len(),
                    "adopted cache holds {} layers but the plan declares {}",
                    self.cache.layers.len(),
                    layers.len()
                );
                for (layer, geometry) in layers.iter().enumerate() {
                    if let Some((k, _)) = self.cache.get_layer(layer) {
                        assert_eq!(
                            k.shape()[1],
                            geometry.kv_dim,
                            "adopted cache rows at layer {layer} are {} wide; the plan says {}",
                            k.shape()[1],
                            geometry.kv_dim
                        );
                    }
                }
            }
            self.geometry = layers_opt;
        } else {
            // A resumed provider: same program, same geometry.
            assert_eq!(
                self.geometry, layers_opt,
                "resumed KV state was prepared for a different program geometry"
            );
        }
    }

    /// The FULL geometry, KV and recurrent alike (SERVE-HYBRID).
    ///
    /// Overrides the KV-only default, which filters to the KV subset and
    /// therefore breaks absolute layer indexing the moment any layer is
    /// recurrent. A KV-only program delegates to [`KvState::prepare`],
    /// which already handles fresh, adopted and resumed caches alike; a
    /// program with recurrence keeps every layer in its absolute slot —
    /// KV layers size the cache, recurrent layers allocate their
    /// declared buffers beside it, and a resumed provider keeps both,
    /// because that persistence IS the continuation.
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        if self.recurrent.is_empty() {
            self.recurrent = layers
                .iter()
                .map(|g| g.recurrent().map(RecurrentState::zeros))
                .collect();
        } else {
            assert_eq!(
                self.recurrent.len(),
                layers.len(),
                "resumed state holds {} layers but the plan declares {}",
                self.recurrent.len(),
                layers.len()
            );
        }
        let all_kv: Option<Vec<LayerKvGeometry>> = layers.iter().map(|g| g.kv().cloned()).collect();
        if let Some(kv) = all_kv {
            self.prepare(&kv);
            return Ok(());
        }
        let kv_opt: Vec<Option<LayerKvGeometry>> = layers.iter().map(|g| g.kv().cloned()).collect();
        if self.geometry.is_empty() {
            assert!(
                self.cache.layers.is_empty(),
                "an adopted canonical cache carries KV-only history; preparing it for a \
                 program with recurrent layers would claim state the cache never held"
            );
            self.cache = KvCache::with_layers(layers.len());
            self.rows = vec![LayerRows::default(); layers.len()];
            self.geometry = kv_opt;
        } else {
            assert_eq!(
                self.geometry, kv_opt,
                "resumed KV state was prepared for a different program geometry"
            );
        }
        Ok(())
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let kv_dim = self.geometry[layer]
            .as_ref()
            .unwrap_or_else(|| panic!("layer {layer} is recurrent; it appends no KV row"))
            .kv_dim;
        assert_eq!(
            key.len(),
            kv_dim,
            "K row at layer {layer} is {} wide; the plan says {kv_dim}",
            key.len()
        );
        assert_eq!(
            value.len(),
            kv_dim,
            "V row at layer {layer} is {} wide; the plan says {kv_dim}",
            value.len()
        );

        // Write into the matrices first…
        let (k, v) = match self.cache.get_layer(layer) {
            Some((k, v)) => {
                let mut k_new = Array2::zeros((k.shape()[0] + 1, kv_dim));
                k_new.slice_mut(ndarray::s![..k.shape()[0], ..]).assign(k);
                k_new
                    .row_mut(k.shape()[0])
                    .assign(&ndarray::ArrayView1::from(key.as_slice()));
                let mut v_new = Array2::zeros((v.shape()[0] + 1, kv_dim));
                v_new.slice_mut(ndarray::s![..v.shape()[0], ..]).assign(v);
                v_new
                    .row_mut(v.shape()[0])
                    .assign(&ndarray::ArrayView1::from(value.as_slice()));
                (k_new, v_new)
            }
            None => (
                Array2::from_shape_vec((1, kv_dim), key).expect("row width asserted above"),
                Array2::from_shape_vec((1, kv_dim), value).expect("row width asserted above"),
            ),
        };
        self.cache.set_layer(layer, (k, v));

        // …then materialise the served view from what the cache now
        // holds, so the view provably carries the stored bits.
        let (k, v) = self.cache.get_layer(layer).expect("layer was just written");
        let last = k.shape()[0] - 1;
        self.rows[layer].keys.push(k.row(last).to_vec());
        self.rows[layer].values.push(v.row(last).to_vec());
    }

    fn keys(&self, layer: usize) -> &[Vec<f32>] {
        &self.rows[layer].keys
    }

    fn values(&self, layer: usize) -> &[Vec<f32>] {
        &self.rows[layer].values
    }

    fn position(&self) -> usize {
        self.cache.next_position
    }

    fn set_position(&mut self, position: usize) {
        self.cache.next_position = position;
    }

    /// The layer's durable recurrent buffers (SERVE-HYBRID) — allocated
    /// at `prepare_continuation` from the plan's declared geometry, kept
    /// across prefills and steps.
    ///
    /// Still fail-closed in both remaining directions: a provider that
    /// was never announced the full geometry refuses rather than
    /// conjuring buffers, and a KV layer asked for recurrent state is a
    /// dispatch defect named as such — never "empty state".
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        match self.recurrent.get_mut(layer) {
            Some(Some(state)) => Ok(state),
            Some(None) => Err(ContinuationError::NotRecurrent {
                provider: "CanonicalKvState",
                layer,
            }),
            None => Err(ContinuationError::RecurrentUnsupported {
                provider: "CanonicalKvState",
                layer,
            }),
        }
    }
}
