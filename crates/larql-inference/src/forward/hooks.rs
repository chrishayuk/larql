//! Mid-forward hook system — read and write the residual stream during a
//! forward pass.
//!
//! Lazarus-style mechanistic interp tools (capture, ablate, patch, steer,
//! probe, DLA) all collapse to one primitive: an in-process callback that
//! fires at well-defined points inside each transformer layer and may
//! optionally mutate the residual.
//!
//! The trait has five callbacks, each defaulting to a no-op so impls only
//! override what they need:
//!
//! - [`LayerHook::on_pre_layer`] — read residual entering the layer.
//! - [`LayerHook::on_post_attention`] — **read or write** post-attention
//!   residual, before FFN.
//! - [`LayerHook::on_attention_weights`] — read per-head attention.
//! - [`LayerHook::on_ffn_activation`] — read FFN gate activation.
//! - [`LayerHook::on_post_layer`] — **read or write** the residual exiting
//!   the layer.
//!
//! The two `&mut` callbacks are what unlock the entire intervention surface.
//! Ablation, steering, patching, and subspace surgery are all just
//! [`LayerHook`] impls over those points.
//!
//! Plumbing: `run_layer_with_capture` and `trace_forward_full_hooked` accept
//! a `&mut dyn LayerHook`. The existing zero-hook signatures stay as thin
//! wrappers passing [`NoopHook`], so call-sites that don't care pay no cost.

use crate::attention::AttentionWeights;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Mid-forward callbacks. All defaults are no-ops; impls override only the
/// callbacks they need.
///
/// `on_post_attention` and `on_post_layer` take `&mut Array2<f32>` so a hook
/// can mutate the residual in place. The other three callbacks are
/// read-only.
#[allow(unused_variables)]
pub trait LayerHook {
    /// Fires before attention runs at `layer`. `h` is the residual entering
    /// the layer (post-norm has not yet been applied).
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {}

    /// Fires after attention, before FFN. The hook may mutate `h` in place
    /// — that is the insertion point for activation patching and
    /// pre-FFN steering.
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {}

    /// Fires when attention weights have been captured. Read-only.
    /// Only called on layers where `capture_attention=true` was requested.
    fn on_attention_weights(&mut self, layer: usize, weights: &AttentionWeights) {}

    /// Fires when an FFN gate activation has been captured. Read-only.
    /// Only called on layers where `capture_activation=true` was requested.
    /// Shape is `(seq_len, ffn_dim)`.
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {}

    /// Fires after the full layer (attention + FFN + PLE + scalar). The
    /// hook may mutate `h` — that is the insertion point for residual-stream
    /// ablation, steering, and any "edit before the next layer sees it"
    /// transform.
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {}
}

/// Hook that does nothing. Used as the default when callers don't care.
pub struct NoopHook;
impl LayerHook for NoopHook {}

/// Captures pre-layer / post-attention / post-layer residuals (and optionally
/// FFN activations + attention weights) at the requested layers. Replaces
/// the file-output pattern of the legacy `LARQL_CPU_DUMP_LAYERS` env var.
///
/// Use [`RecordHook::for_layers`] to construct, then read the public maps
/// after the forward pass returns.
pub struct RecordHook {
    /// Layers to record. Other layers are skipped (zero overhead).
    pub layers: HashSet<usize>,
    /// `(seq_len, hidden)` residual entering each captured layer.
    pub pre_layer: HashMap<usize, Array2<f32>>,
    /// `(seq_len, hidden)` residual after attention at each captured layer.
    pub post_attention: HashMap<usize, Array2<f32>>,
    /// `(seq_len, hidden)` residual after the full layer.
    pub post_layer: HashMap<usize, Array2<f32>>,
    /// `(seq_len, ffn_dim)` FFN gate activation. Only populated when the
    /// outer trace was asked to capture FFN activations.
    pub ffn_activation: HashMap<usize, Array2<f32>>,
    /// Per-head attention weights for the last token position. Only
    /// populated when the outer trace was asked to capture attention.
    pub attention_weights: HashMap<usize, Vec<Vec<f32>>>,
}

impl RecordHook {
    /// Build a recorder that captures the listed layers.
    pub fn for_layers<I: IntoIterator<Item = usize>>(layers: I) -> Self {
        Self {
            layers: layers.into_iter().collect(),
            pre_layer: HashMap::new(),
            post_attention: HashMap::new(),
            post_layer: HashMap::new(),
            ffn_activation: HashMap::new(),
            attention_weights: HashMap::new(),
        }
    }
}

impl LayerHook for RecordHook {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {
        if self.layers.contains(&layer) {
            self.pre_layer.insert(layer, h.clone());
        }
    }
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {
        if self.layers.contains(&layer) {
            self.post_attention.insert(layer, h.clone());
        }
    }
    fn on_attention_weights(&mut self, layer: usize, weights: &AttentionWeights) {
        if self.layers.contains(&layer) {
            self.attention_weights.insert(layer, weights.heads.clone());
        }
    }
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {
        if self.layers.contains(&layer) {
            self.ffn_activation.insert(layer, gate.clone());
        }
    }
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        if self.layers.contains(&layer) {
            self.post_layer.insert(layer, h.clone());
        }
    }
}

/// Zeros rows of the post-layer residual at requested layers.
///
/// `positions == None` zeros every row at that layer (full-layer ablation).
/// `positions == Some(vec)` zeros only the listed token positions.
///
/// Implements lazarus's `ablate_layers` and per-position residual ablation.
pub struct ZeroAblateHook {
    pub layers: HashMap<usize, Option<Vec<usize>>>,
}

impl ZeroAblateHook {
    pub fn for_layers<I: IntoIterator<Item = usize>>(layers: I) -> Self {
        Self {
            layers: layers.into_iter().map(|l| (l, None)).collect(),
        }
    }
}

impl LayerHook for ZeroAblateHook {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        let Some(positions) = self.layers.get(&layer) else {
            return;
        };
        match positions {
            None => h.fill(0.0),
            Some(ps) => {
                let n_rows = h.nrows();
                for &p in ps {
                    if p < n_rows {
                        h.row_mut(p).fill(0.0);
                    }
                }
            }
        }
    }
}

/// Adds `alpha * v` to the last-token row of the post-layer residual at
/// requested layers. Implements lazarus's `steer_and_generate`.
///
/// Use a separate `SteerHook` per (layer, vector) pair, or compose them in
/// [`CompositeHook`].
pub struct SteerHook {
    /// Layer → (steering vector of shape `(hidden,)`, scalar gain).
    pub steers: HashMap<usize, (Array1<f32>, f32)>,
}

impl SteerHook {
    pub fn new() -> Self {
        Self {
            steers: HashMap::new(),
        }
    }

    pub fn add(mut self, layer: usize, vector: Array1<f32>, alpha: f32) -> Self {
        self.steers.insert(layer, (vector, alpha));
        self
    }
}

impl Default for SteerHook {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerHook for SteerHook {
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        let Some((v, alpha)) = self.steers.get(&layer) else {
            return;
        };
        if h.nrows() == 0 || v.len() != h.ncols() {
            return;
        }
        let last = h.nrows() - 1;
        let mut row = h.row_mut(last);
        for (i, val) in row.iter_mut().enumerate() {
            *val += *alpha * v[i];
        }
    }
}

/// Runs an arbitrary collection of hooks in order. Useful for combining
/// (e.g.) a `RecordHook` with a `SteerHook` so you can both intervene and
/// measure in one pass.
pub struct CompositeHook<'a> {
    pub hooks: Vec<&'a mut dyn LayerHook>,
}

impl<'a> CompositeHook<'a> {
    pub fn new(hooks: Vec<&'a mut dyn LayerHook>) -> Self {
        Self { hooks }
    }
}

impl LayerHook for CompositeHook<'_> {
    fn on_pre_layer(&mut self, layer: usize, h: &Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_pre_layer(layer, h);
        }
    }
    fn on_post_attention(&mut self, layer: usize, h: &mut Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_post_attention(layer, h);
        }
    }
    fn on_attention_weights(&mut self, layer: usize, weights: &AttentionWeights) {
        for hook in self.hooks.iter_mut() {
            hook.on_attention_weights(layer, weights);
        }
    }
    fn on_ffn_activation(&mut self, layer: usize, gate: &Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_ffn_activation(layer, gate);
        }
    }
    fn on_post_layer(&mut self, layer: usize, h: &mut Array2<f32>) {
        for hook in self.hooks.iter_mut() {
            hook.on_post_layer(layer, h);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn noop_hook_compiles_and_does_nothing() {
        let mut h: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let mut hook = NoopHook;
        let original = h.clone();
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, original);
    }

    #[test]
    fn record_hook_captures_only_requested_layers() {
        let mut hook = RecordHook::for_layers([1, 3]);
        let mut h: Array2<f32> = array![[1.0, 2.0]];

        hook.on_pre_layer(0, &h); // not in set
        hook.on_pre_layer(1, &h); // in set
        hook.on_post_layer(2, &mut h); // not in set
        hook.on_post_layer(3, &mut h); // in set

        assert!(!hook.pre_layer.contains_key(&0));
        assert!(hook.pre_layer.contains_key(&1));
        assert!(!hook.post_layer.contains_key(&2));
        assert!(hook.post_layer.contains_key(&3));
    }

    #[test]
    fn record_hook_clones_residual_so_later_writes_dont_pollute() {
        let mut hook = RecordHook::for_layers([0]);
        let mut h: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        hook.on_pre_layer(0, &h);
        h[[0, 0]] = 999.0;
        let recorded = hook.pre_layer.get(&0).unwrap();
        assert_eq!(recorded[[0, 0]], 1.0, "RecordHook must snapshot, not alias");
    }

    #[test]
    fn zero_ablate_full_layer() {
        let mut hook = ZeroAblateHook::for_layers([2]);
        let mut h: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, array![[1.0, 2.0], [3.0, 4.0]], "wrong layer untouched");
        hook.on_post_layer(2, &mut h);
        assert_eq!(h, array![[0.0, 0.0], [0.0, 0.0]], "target layer zeroed");
    }

    #[test]
    fn zero_ablate_specific_positions() {
        let mut hook = ZeroAblateHook {
            layers: [(1, Some(vec![1, 3]))].into_iter().collect(),
        };
        let mut h: Array2<f32> = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        hook.on_post_layer(1, &mut h);
        assert_eq!(h.row(0).to_vec(), vec![1.0, 1.0], "pos 0 untouched");
        assert_eq!(h.row(1).to_vec(), vec![0.0, 0.0], "pos 1 zeroed");
        assert_eq!(h.row(2).to_vec(), vec![3.0, 3.0], "pos 2 untouched");
        assert_eq!(h.row(3).to_vec(), vec![0.0, 0.0], "pos 3 zeroed");
    }

    #[test]
    fn zero_ablate_out_of_range_position_is_noop() {
        let mut hook = ZeroAblateHook {
            layers: [(0, Some(vec![99]))].into_iter().collect(),
        };
        let mut h: Array2<f32> = array![[1.0, 2.0]];
        let original = h.clone();
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, original);
    }

    #[test]
    fn steer_adds_alpha_v_to_last_row() {
        let mut hook = SteerHook::new().add(0, array![10.0, 20.0], 0.5);
        let mut h: Array2<f32> = array![[1.0, 1.0], [2.0, 2.0]];
        hook.on_post_layer(0, &mut h);
        assert_eq!(h.row(0).to_vec(), vec![1.0, 1.0], "non-last row untouched");
        assert_eq!(
            h.row(1).to_vec(),
            vec![2.0 + 0.5 * 10.0, 2.0 + 0.5 * 20.0],
            "last row += alpha * v"
        );
    }

    #[test]
    fn steer_silently_skips_on_dim_mismatch() {
        let mut hook = SteerHook::new().add(0, array![1.0, 2.0, 3.0], 1.0);
        let mut h: Array2<f32> = array![[1.0, 1.0]];
        let original = h.clone();
        hook.on_post_layer(0, &mut h);
        assert_eq!(h, original, "wrong-dim vector must not corrupt residual");
    }

    #[test]
    fn composite_runs_hooks_in_order() {
        // Steer then record: recorded value must include the steer.
        let mut steer = SteerHook::new().add(0, array![1.0, 1.0], 1.0);
        let mut record = RecordHook::for_layers([0]);
        let mut comp = CompositeHook::new(vec![&mut steer, &mut record]);
        let mut h: Array2<f32> = array![[5.0, 5.0]];
        comp.on_post_layer(0, &mut h);
        let recorded = record.post_layer.get(&0).unwrap();
        assert_eq!(recorded.row(0).to_vec(), vec![6.0, 6.0]);
    }
}
