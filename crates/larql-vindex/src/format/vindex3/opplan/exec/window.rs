//! A continuation state that is physically bounded on sliding layers.
//!
//! STATE-3. Behind a sliding window, past positions are **provably
//! unreachable**: the span logic starts every query at
//! `position + 1 - window`, so a row that has fallen behind the window
//! of every future query can never be read again. The STATE-2 gates
//! established that excluding such rows is bit-for-bit free; this
//! provider stops *holding* them. That is the liveness model doing real
//! work — garbage collection of provably dead KV, under the `Exact`
//! contract, with no approximation anywhere.
//!
//! The store stays position-aligned: freed rows keep their slot but
//! their heap payload is released (`Vec::new()` husks). Alignment is
//! what lets the executor's `keys[p]` indexing stand unchanged, and it
//! makes an erroneous read of a freed row a loud out-of-bounds panic
//! rather than silent attention over zeros. The husk metadata still
//! grows with the sequence — ~24 bytes per position per layer against
//! `4·kv_dim` payload bytes per row — so what this provider bounds is
//! the PAYLOAD; a flat representation that also bounds the metadata is
//! a later storage rung, tied to the backend row contract (see the kv
//! module notes).
//!
//! The freeing rule is conservative by construction: after the row for
//! position `p` is appended the store holds `p + 1` rows, every future
//! query sits at `q > p` with span start `q + 1 - window`, so indices
//! below `len - window` are dead. The provider trusts the geometry it
//! was prepared with: a plan that pairs a `window` with a FULL span
//! would read freed rows and panic — the geometry's `window` is only
//! populated from sliding-span attention ops today, and teaching
//! [`LayerKvGeometry`] to carry the span explicitly is the integration
//! rung's business.
//!
//! KV-only, deliberately: a recurrent layer's state is already bounded
//! by construction, and this provider refuses hybrids loudly rather
//! than pretending to hold buffers it does not manage.

use super::continuation::RecurrentState;
use super::kv::{ContinuationError, ContinuationProvider, LayerKvGeometry};

/// One layer's rows with the window shadow physically freed.
struct BoundedLayerRows {
    /// `Some(w)` bounds the layer at `w` live rows; `None` retains all.
    window: Option<usize>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// Index of the oldest row whose payload is still held. Everything
    /// below it is a freed husk.
    first_live: usize,
}

impl BoundedLayerRows {
    fn new(geometry: &LayerKvGeometry) -> Self {
        if let Some(window) = geometry.window {
            assert!(window > 0, "a zero-width attention window is not a layer");
        }
        Self {
            window: geometry.window,
            keys: Vec::new(),
            values: Vec::new(),
            first_live: 0,
        }
    }

    fn append(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.keys.push(key);
        self.values.push(value);
        if let Some(window) = self.window {
            // Rows below `len - window` are behind every future query's
            // span start and can never be read again.
            let dead_below = self.keys.len().saturating_sub(window);
            while self.first_live < dead_below {
                self.keys[self.first_live] = Vec::new();
                self.values[self.first_live] = Vec::new();
                self.first_live += 1;
            }
        }
    }

    fn resident_rows(&self) -> usize {
        self.keys.len() - self.first_live
    }

    fn resident_payload_bytes(&self) -> usize {
        self.keys[self.first_live..]
            .iter()
            .zip(&self.values[self.first_live..])
            .map(|(k, v)| (k.len() + v.len()) * std::mem::size_of::<f32>())
            .sum()
    }
}

/// The window-shadow provider: [`RowKvState`](super::kv::RowKvState)
/// semantics with sliding layers' payload bounded at their window.
///
/// Same rows wherever a row can still be read — the gates pin decode
/// and prefill over this provider bit-identical to the full store —
/// and `O(window)` payload residency on every layer that declares one.
#[derive(Default)]
pub struct WindowKvState {
    layers: Vec<BoundedLayerRows>,
    position: usize,
}

impl WindowKvState {
    /// Live rows currently held for `layer`.
    pub fn resident_rows(&self, layer: usize) -> usize {
        self.layers[layer].resident_rows()
    }

    /// Payload bytes currently held across all layers — the number a
    /// residency claim must quote, measured rather than derived.
    pub fn resident_payload_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(BoundedLayerRows::resident_payload_bytes)
            .sum()
    }
}

impl ContinuationProvider for WindowKvState {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        if self.layers.is_empty() {
            self.layers = layers.iter().map(BoundedLayerRows::new).collect();
        } else {
            // Same resume contract as the plain row state: an
            // announcement, not a reset — and a different shape is a
            // wrong-conversation bug worth failing loudly on.
            assert_eq!(
                self.layers.len(),
                layers.len(),
                "resumed bounded KV state holds {} layers but the plan declares {}",
                self.layers.len(),
                layers.len()
            );
        }
    }

    fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        self.layers[layer].append(key, value);
    }

    fn keys(&self, layer: usize) -> &[Vec<f32>] {
        &self.layers[layer].keys
    }

    fn values(&self, layer: usize) -> &[Vec<f32>] {
        &self.layers[layer].values
    }

    fn position(&self) -> usize {
        self.position
    }

    fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        Err(ContinuationError::RecurrentUnsupported {
            provider: "WindowKvState",
            layer,
        })
    }
}
