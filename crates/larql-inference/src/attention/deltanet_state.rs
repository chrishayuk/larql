//! Gated DeltaNet per-layer state cache (Phase C.1 of
//! `inference-qwen35-deltanet`).
//!
//! Linear-attention layers don't maintain a KV cache. Instead each
//! layer carries forward two state tensors across tokens:
//!
//! - `conv_state` of shape `[d_conv - 1, conv_dim]` — the previous
//!   `d_conv - 1` tokens' QKV-projected values, used as the sliding
//!   window for the causal depthwise Conv1D.
//! - `recurrent_state` of shape `[head_v_dim, head_v_dim, n_v_heads]`
//!   — the matrix-valued recurrent state that the delta-rule update
//!   modifies each token.
//!
//! On Qwen 3.6 27B: `conv_state = [3, 10240]` (~120 KiB fp32),
//! `recurrent_state = [128, 128, 48]` (~3 MiB fp32). With 48 linear
//! layers that's ~144 MiB fp32 / ~72 MiB fp16 per active sequence.
//!
//! Full-attention layers (the every-4th layer in Qwen 3.6) still use
//! the existing `KvCache`; the wrapper `DeltaNetHybridCache` keeps
//! both kinds of state side by side so a single forward pass can
//! dispatch per layer.
//!
//! This module is pure data structures + an autoregressive Conv1D
//! helper. The DeltaNet recurrence math itself lands in Phase C.2.

use ndarray::{Array1, Array2, Array3};

/// State of a single Gated DeltaNet linear-attention layer.
///
/// Fields are public so the forward kernel (Phase C.2) can mutate
/// them in-place without taking a method call per token. Callers that
/// don't need direct access should use the `step_conv` /
/// `update_recurrent` helpers on `DeltaNetStateCache`.
#[derive(Clone, Debug)]
pub struct DeltaNetLayerState {
    /// Causal-Conv1D ring buffer: rows 0..d_conv-1 hold the last
    /// `d_conv - 1` tokens (oldest first). Each row is `conv_dim`
    /// wide.
    pub conv_state: Array2<f32>,
    /// Per-head matrix-valued recurrent state. Index order:
    /// `[head_v_dim_row, head_v_dim_col, n_v_heads]`. The delta-rule
    /// update is `S ← S + k ⊗ d^T` where `S[r, c, h]` is the rank-1
    /// outer product contribution for head h.
    pub recurrent_state: Array3<f32>,
}

impl DeltaNetLayerState {
    /// Allocate a zero-initialised state for a single linear layer.
    ///
    /// - `d_conv`: causal Conv1D kernel size (Qwen 3.6: 4).
    /// - `conv_dim`: `2 * key_dim + value_dim` (Qwen 3.6: 10240).
    /// - `head_v_dim`: per-head dim (`ssm_state_size`, Qwen 3.6: 128).
    /// - `n_v_heads`: number of V heads (`ssm_dt_rank`, Qwen 3.6: 48).
    pub fn allocate(d_conv: usize, conv_dim: usize, head_v_dim: usize, n_v_heads: usize) -> Self {
        Self {
            conv_state: Array2::zeros((d_conv.saturating_sub(1), conv_dim)),
            recurrent_state: Array3::zeros((head_v_dim, head_v_dim, n_v_heads)),
        }
    }

    /// Reset all state to zero — typically called between sequences
    /// so a new generation doesn't leak the previous sequence's
    /// recurrent state.
    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.recurrent_state.fill(0.0);
    }
}

/// Per-sequence Gated DeltaNet state cache. One entry per **linear-
/// attention** layer; full-attention layers have a `None` entry
/// (they store K/V in the parallel `KvCache`).
#[derive(Clone, Debug)]
pub struct DeltaNetStateCache {
    /// `layers[i]` is `Some(state)` iff layer `i` is a linear-
    /// attention layer (per `Qwen35Arch::is_linear_attention_layer`).
    pub layers: Vec<Option<DeltaNetLayerState>>,
}

impl DeltaNetStateCache {
    /// Allocate state for all linear layers. `linear_layer_mask[i]`
    /// is `true` for each linear-attention layer; `false` slots get
    /// `None`. Same `conv_dim` / `head_v_dim` / `n_v_heads` /
    /// `d_conv` for every layer (Qwen 3.6 is uniform across
    /// linear layers).
    pub fn allocate(
        linear_layer_mask: &[bool],
        d_conv: usize,
        conv_dim: usize,
        head_v_dim: usize,
        n_v_heads: usize,
    ) -> Self {
        let layers = linear_layer_mask
            .iter()
            .map(|&is_linear| {
                if is_linear {
                    Some(DeltaNetLayerState::allocate(
                        d_conv, conv_dim, head_v_dim, n_v_heads,
                    ))
                } else {
                    None
                }
            })
            .collect();
        Self { layers }
    }

    /// Number of linear-attention layers in this cache (i.e. how
    /// many `Some(...)` entries `layers` has).
    pub fn num_linear_layers(&self) -> usize {
        self.layers.iter().filter(|s| s.is_some()).count()
    }

    /// Reset every linear-layer state. Full-attention slots stay
    /// `None`.
    pub fn reset(&mut self) {
        for slot in self.layers.iter_mut().flatten() {
            slot.reset();
        }
    }
}

/// Causal depthwise Conv1D with autoregressive state update.
///
/// Computes `out[c] = sum_k weight[k, c] * window[k, c]` where
/// `window` is `[state[0, c], state[1, c], ..., state[d_conv-2, c],
/// new[c]]` for each channel `c`. After producing `out`, the state
/// is shifted so the new token replaces the oldest:
/// `state[0..d_conv-2] = state[1..d_conv-1]`, `state[d_conv-2] = new`.
///
/// Shapes:
/// - `weight`: `[d_conv, conv_dim]` — per-channel kernel taps.
/// - `state`: `[d_conv - 1, conv_dim]` — mutated in place.
/// - `new`: `[conv_dim]` — the current token's QKV-mixed input.
/// - returns: `Array1<f32>` of length `conv_dim`.
///
/// `d_conv = 1` is degenerate: state is empty, output is `weight[0]
/// .* new` element-wise.
pub fn causal_conv1d_step(
    weight: &Array2<f32>,
    state: &mut Array2<f32>,
    new: &Array1<f32>,
) -> Array1<f32> {
    let d_conv = weight.shape()[0];
    let conv_dim = weight.shape()[1];
    debug_assert_eq!(state.shape(), [d_conv.saturating_sub(1), conv_dim]);
    debug_assert_eq!(new.len(), conv_dim);

    let mut out = Array1::<f32>::zeros(conv_dim);
    let state_rows = d_conv.saturating_sub(1);
    // Contribute from historical state rows.
    for k in 0..state_rows {
        let w_row = weight.row(k);
        let s_row = state.row(k);
        for c in 0..conv_dim {
            out[c] += w_row[c] * s_row[c];
        }
    }
    // Contribute from the new token (last kernel tap).
    let w_last = weight.row(d_conv - 1);
    for c in 0..conv_dim {
        out[c] += w_last[c] * new[c];
    }

    // Shift state: drop the oldest row, append the new one at the
    // end. For `d_conv = 1` (no state) this is a no-op.
    if state_rows > 0 {
        // Copy rows 1..state_rows into rows 0..state_rows-1.
        for k in 0..state_rows.saturating_sub(1) {
            let (mut older, newer) =
                state.multi_slice_mut((ndarray::s![k, ..], ndarray::s![k + 1, ..]));
            older.assign(&newer);
        }
        // Last state row becomes `new`.
        state.row_mut(state_rows - 1).assign(new);
    }

    out
}

/// L2-normalise each head's vector independently (`x[:, h] /=
/// sqrt(sum(x[:, h]^2) + eps)`). The result is row-major per the
/// reshape convention used in the DeltaNet recurrence: input shape
/// `[head_dim, n_heads]`.
///
/// `eps` defaults to `1e-6` to match llama.cpp; callers that want a
/// different epsilon can use `l2_normalize_per_head_eps`.
pub fn l2_normalize_per_head(x: &Array2<f32>) -> Array2<f32> {
    l2_normalize_per_head_eps(x, 1e-6)
}

/// L2-normalise with explicit epsilon.
pub fn l2_normalize_per_head_eps(x: &Array2<f32>, eps: f32) -> Array2<f32> {
    let (head_dim, n_heads) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::<f32>::zeros((head_dim, n_heads));
    for h in 0..n_heads {
        let mut sum_sq = 0.0_f32;
        for d in 0..head_dim {
            let v = x[[d, h]];
            sum_sq += v * v;
        }
        let inv = 1.0 / (sum_sq + eps).sqrt();
        for d in 0..head_dim {
            out[[d, h]] = x[[d, h]] * inv;
        }
    }
    out
}

/// Element-wise softplus: `softplus(x) = log(1 + exp(x))`. Uses a
/// numerically-stable formulation that avoids overflow when `x > 0`.
/// Matches the formulation llama.cpp's `ggml_softplus` uses.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        // exp(x) overflows; log1p(exp(x)) ≈ x within 1e-9.
        x
    } else if x < -20.0 {
        // exp(x) underflows; log1p ≈ exp(x).
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Element-wise sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn deltanet_layer_state_allocate_zero_filled() {
        let s = DeltaNetLayerState::allocate(4, 10, 8, 3);
        assert_eq!(s.conv_state.shape(), &[3, 10]);
        assert_eq!(s.recurrent_state.shape(), &[8, 8, 3]);
        assert!(s.conv_state.iter().all(|&v| v == 0.0));
        assert!(s.recurrent_state.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn deltanet_state_cache_only_linear_slots_are_some() {
        // Qwen 3.6 27B mask: layer (i+1) % 4 == 0 → full attn
        // (None); else linear (Some).
        let mask: Vec<bool> = (0..8).map(|i| (i + 1) % 4 != 0).collect();
        let cache = DeltaNetStateCache::allocate(&mask, 4, 10, 8, 3);
        assert_eq!(cache.layers.len(), 8);
        assert_eq!(cache.num_linear_layers(), 6); // 3 of every 4 are linear
                                                  // Layers 3 and 7 (i+1 multiples of 4) are full-attn → None.
        assert!(cache.layers[3].is_none());
        assert!(cache.layers[7].is_none());
        // The rest are Some.
        for &i in &[0, 1, 2, 4, 5, 6] {
            assert!(cache.layers[i].is_some(), "layer {i} should be linear");
        }
    }

    #[test]
    fn deltanet_state_cache_reset_zeroes_only_linear_layers() {
        let mask = vec![true, false, true];
        let mut cache = DeltaNetStateCache::allocate(&mask, 4, 5, 4, 2);
        // Dirty the live linear layers.
        for slot in cache.layers.iter_mut().flatten() {
            slot.conv_state.fill(7.0);
            slot.recurrent_state.fill(11.0);
        }
        cache.reset();
        for slot in cache.layers.iter().flatten() {
            assert!(slot.conv_state.iter().all(|&v| v == 0.0));
            assert!(slot.recurrent_state.iter().all(|&v| v == 0.0));
        }
        // Layer 1 (None) is still None.
        assert!(cache.layers[1].is_none());
    }

    #[test]
    fn causal_conv1d_step_d_conv_1_is_pointwise() {
        // d_conv=1: no state; out = weight[0] .* new.
        let weight = array![[2.0, 3.0, 4.0]];
        let mut state = Array2::<f32>::zeros((0, 3));
        let new = array![1.0, 1.0, 1.0];
        let out = causal_conv1d_step(&weight, &mut state, &new);
        assert_eq!(out.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn causal_conv1d_step_d_conv_2_uses_one_step_history() {
        // d_conv=2, conv_dim=2.
        //   weight = [[w00, w01], [w10, w11]]
        // First call (state = 0): out = w00*0 + w10*new = w10*new.
        // Second call (state = [new1]): out = w00*new1 + w10*new2.
        let weight = array![[0.5, 1.5], [2.0, 3.0]];
        let mut state = Array2::<f32>::zeros((1, 2));
        let new1 = array![1.0, 2.0];
        let out1 = causal_conv1d_step(&weight, &mut state, &new1);
        // out1 = 2.0*1.0, 3.0*2.0
        assert_eq!(out1.to_vec(), vec![2.0, 6.0]);
        // After call 1, state[0] = new1 = [1.0, 2.0]
        assert_eq!(state.row(0).to_vec(), vec![1.0, 2.0]);
        let new2 = array![10.0, 20.0];
        let out2 = causal_conv1d_step(&weight, &mut state, &new2);
        // out2[c] = w00*state[0,c] + w10*new2[c]
        //   c=0: 0.5*1 + 2.0*10 = 20.5
        //   c=1: 1.5*2 + 3.0*20 = 63.0
        assert_eq!(out2.to_vec(), vec![20.5, 63.0]);
        // After call 2, state[0] = new2.
        assert_eq!(state.row(0).to_vec(), vec![10.0, 20.0]);
    }

    #[test]
    fn causal_conv1d_step_d_conv_4_shifts_correctly() {
        // d_conv=4, conv_dim=1. Window after N steps = [s0, s1, s2, new].
        // After 1 step: state = [0, 0, new1].
        // After 2 steps: state = [0, new1, new2].
        // After 4 steps: state = [new1, new2, new3], output uses all four.
        let weight = array![[1.0], [10.0], [100.0], [1000.0]];
        let mut state = Array2::<f32>::zeros((3, 1));
        for (i, v) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            let new = array![*v];
            let out = causal_conv1d_step(&weight, &mut state, &new);
            // Pre-step state: rows[0..3]; new at row 3 (slot 0 of
            // the kernel). Output = w0*s0 + w1*s1 + w2*s2 + w3*new.
            let expected = match i {
                0 => 1000.0 * 1.0,                                  // state was [0,0,0]
                1 => 100.0 * 1.0 + 1000.0 * 2.0,                    // state was [0, 0, 1]
                2 => 10.0 * 1.0 + 100.0 * 2.0 + 1000.0 * 3.0,       // [0, 1, 2]
                3 => 1.0 + 10.0 * 2.0 + 100.0 * 3.0 + 1000.0 * 4.0, // [1, 2, 3]
                _ => unreachable!(),
            };
            assert!(
                (out[0] - expected).abs() < 1e-5,
                "step {i}: got {} expected {}",
                out[0],
                expected
            );
        }
        // Final state should be [2, 3, 4].
        assert_eq!(state.column(0).to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn l2_normalize_per_head_unit_vector_returns_self() {
        // head_dim=1, n_heads=2. Head 0's lone element = 3.0 →
        // normalises to 1.0. Head 1's lone element is 0.0; with the
        // default eps the normalisation safely returns 0.0 instead
        // of NaN.
        let x = array![[3.0_f32, 0.0]];
        let out = l2_normalize_per_head(&x);
        assert!((out[[0, 0]] - 1.0).abs() < 1e-3);
        assert!(out[[0, 1]].abs() < 1e-3);
    }

    #[test]
    fn l2_normalize_per_head_multi_dim() {
        // head_dim=2, n_heads=2.
        let x = array![[3.0_f32, 4.0], [4.0, 3.0]];
        let out = l2_normalize_per_head_eps(&x, 0.0);
        // head 0: column [3, 4], norm = 5 → [0.6, 0.8].
        // head 1: column [4, 3], norm = 5 → [0.8, 0.6].
        assert!((out[[0, 0]] - 0.6).abs() < 1e-5);
        assert!((out[[1, 0]] - 0.8).abs() < 1e-5);
        assert!((out[[0, 1]] - 0.8).abs() < 1e-5);
        assert!((out[[1, 1]] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn softplus_is_stable_and_monotone() {
        // Spot-check the saturation regions + the analytic value.
        assert_eq!(softplus(50.0), 50.0); // saturates to identity
        assert!(softplus(-50.0) < 1e-20); // saturates to exp(x)
                                          // Analytic: softplus(0) = ln(2) ≈ 0.693147
        assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 1e-6);
        // Analytic: softplus(1) = ln(1+e) ≈ 1.31326
        assert!((softplus(1.0) - (1.0 + 1.0_f32.exp()).ln()).abs() < 1e-6);
        // Monotone:
        let mut prev = softplus(-3.0);
        for i in -2..10 {
            let cur = softplus(i as f32);
            assert!(cur >= prev);
            prev = cur;
        }
    }

    #[test]
    fn sigmoid_is_stable_at_extremes() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        // Numerically stable on the negative branch:
        // direct 1/(1+exp(-(-100))) = 1/(1+exp(100)) ≈ 1e-44 (underflows).
        // Our branch handles it.
        assert!(sigmoid(-100.0).abs() < 1e-30);
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-30);
    }
}
