//! Gated DeltaNet recurrence (Phase C.2 of
//! `inference-qwen35-deltanet`).
//!
//! Single-token forward step that mutates the per-layer recurrent
//! state and returns the pre-out-projection output. Operates on
//! already-projected, already-normed `(q, k, v)` plus per-head
//! scalars `log_g` (pre-exp decay) and `beta` (post-sigmoid delta-
//! rule learning rate).
//!
//! See `openspec/changes/inference-qwen35-deltanet/design.md` §4 for
//! the math. Reference: llama.cpp `src/models/delta-net-base.cpp`
//! function `build_delta_net_autoregressive`.

use ndarray::{Array1, Array2, Array3};

/// Single-token Gated DeltaNet recurrence step.
///
/// Mutates `state` in place; returns the output `[S_v, H_v]`
/// (pre-RMSNorm, pre-Z-gate, pre-out-projection — caller applies
/// those).
///
/// Inputs:
/// - `q`: `[S_k, H_k]` — Q already L2-normalised per head.
/// - `k`: `[S_k, H_k]` — K already L2-normalised per head.
/// - `v`: `[S_v, H_v]` — V (no normalisation; raw projection
///   output).
/// - `log_g`: `[H_v]` — pre-exp scalar log-decay per V head. The
///   recurrence exps it internally (`g_h = exp(log_g[h])`); decay
///   is `state ← state * g`.
/// - `beta`: `[H_v]` — post-sigmoid delta-rule learning rate per V
///   head. Caller applies sigmoid before passing in.
/// - `state`: `[S_v, S_v, H_v]` — matrix-valued recurrent state,
///   mutated in place.
///
/// Constraints:
/// - `S_k == S_v` (both equal to `ssm_state_size`, Qwen 3.6: 128).
/// - `H_k` divides `H_v`. K head `kh` is shared across V heads
///   `kh * (H_v / H_k) .. (kh + 1) * (H_v / H_k)` (repeat-interleave).
/// - Per design.md §4, Q is scaled by `1 / sqrt(S_k)` inside this
///   function.
pub fn delta_net_step(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    log_g: &Array1<f32>,
    beta: &Array1<f32>,
    state: &mut Array3<f32>,
) -> Array2<f32> {
    let s_v = v.shape()[0];
    let h_v = v.shape()[1];
    let s_k = k.shape()[0];
    let h_k = k.shape()[1];

    debug_assert_eq!(s_v, s_k, "S_k must equal S_v");
    debug_assert_eq!(q.shape(), [s_k, h_k]);
    debug_assert_eq!(state.shape(), [s_v, s_v, h_v]);
    debug_assert_eq!(log_g.len(), h_v);
    debug_assert_eq!(beta.len(), h_v);
    debug_assert!(h_k > 0 && h_v.is_multiple_of(h_k), "H_k must divide H_v");

    let repeat_factor = h_v / h_k;
    let scale_q = 1.0 / (s_k as f32).sqrt();

    let mut output = Array2::<f32>::zeros((s_v, h_v));

    // Working buffers, allocated once per call.
    let mut sk = vec![0.0_f32; s_v];
    let mut d = vec![0.0_f32; s_v];

    for h in 0..h_v {
        let kh = h / repeat_factor;
        let g_h = log_g[h].exp();
        let b_h = beta[h];

        // 1. Decay state per head: S ← S * g.
        for r in 0..s_v {
            for c in 0..s_v {
                state[[r, c, h]] *= g_h;
            }
        }

        // 2. sk[c] = sum_r state[r, c, h] * k[r, kh]  (S_v × S_k matvec).
        //    Reuse the buffer to avoid allocating per head.
        for entry in sk.iter_mut() {
            *entry = 0.0;
        }
        for r in 0..s_k {
            let k_val = k[[r, kh]];
            if k_val == 0.0 {
                continue;
            }
            for c in 0..s_v {
                sk[c] += state[[r, c, h]] * k_val;
            }
        }

        // 3. d = (v - sk) * b.
        for c in 0..s_v {
            d[c] = (v[[c, h]] - sk[c]) * b_h;
        }

        // 4. S ← S + k ⊗ d^T  (rank-1 update; state[r,c,h] += k[r,kh] * d[c]).
        for r in 0..s_k {
            let k_val = k[[r, kh]];
            if k_val == 0.0 {
                continue;
            }
            for c in 0..s_v {
                state[[r, c, h]] += k_val * d[c];
            }
        }

        // 5. o[c, h] = sum_r state[r, c, h] * q[r, kh] * scale_q.
        for c in 0..s_v {
            let mut acc = 0.0_f32;
            for r in 0..s_k {
                acc += state[[r, c, h]] * q[[r, kh]];
            }
            output[[c, h]] = acc * scale_q;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    /// Trivial case: H_k = H_v = 1, S_k = S_v = 2. State starts at
    /// zero; one step with non-trivial b and k. After the step:
    /// - state should be rank-1: S = k ⊗ (v * b).
    /// - output should be S^T @ q / sqrt(S_k) = k * (v*b)^T q.
    #[test]
    fn delta_net_step_initial_state_rank1_update() {
        let s = 2;
        let h_k = 1;
        let h_v = 1;
        // q, k are L2-normalised per head. Use k = [1, 0] (unit
        // basis vector) so the rank-1 update is exact.
        let q = array![[1.0_f32], [0.0]];
        let k = array![[1.0_f32], [0.0]];
        // v[:, 0] = [3.0, 4.0].
        let v = array![[3.0_f32], [4.0]];
        // Identity decay: log_g = 0 → g = 1.
        let log_g = Array1::from(vec![0.0_f32]);
        // beta = 1.
        let beta = Array1::from(vec![1.0_f32]);
        let mut state = Array3::<f32>::zeros((s, s, 1));

        let out = delta_net_step(&q, &k, &v, &log_g, &beta, &mut state);

        // After the step:
        //   sk = state^T @ k = 0 (state was zero).
        //   d = (v - 0) * 1 = [3, 4].
        //   state[r, c, 0] = k[r, 0] * d[c]
        //     = (1, 0)^T * (3, 4)
        //     = [[3, 4], [0, 0]]
        assert!((state[[0, 0, 0]] - 3.0).abs() < 1e-6);
        assert!((state[[0, 1, 0]] - 4.0).abs() < 1e-6);
        assert!(state[[1, 0, 0]].abs() < 1e-6);
        assert!(state[[1, 1, 0]].abs() < 1e-6);

        // Output:
        //   o[c, 0] = sum_r state[r, c, 0] * q[r, 0] / sqrt(s)
        //   sqrt(2) ≈ 1.41421
        //   o[0] = (3 * 1 + 0 * 0) / sqrt(2) = 3 / sqrt(2) ≈ 2.12132
        //   o[1] = (4 * 1 + 0 * 0) / sqrt(2) = 4 / sqrt(2) ≈ 2.82843
        assert!((out[[0, 0]] - 3.0 / (2.0_f32).sqrt()).abs() < 1e-5);
        assert!((out[[1, 0]] - 4.0 / (2.0_f32).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn delta_net_step_zero_beta_leaves_state_unchanged() {
        // beta = 0 → d = 0 → no rank-1 update. State stays whatever
        // decay leaves it.
        let s = 2;
        let q = array![[1.0_f32], [0.0]];
        let k = array![[1.0_f32], [0.0]];
        let v = array![[3.0_f32], [4.0]];
        let log_g = Array1::from(vec![0.0_f32]); // g = 1
        let beta = Array1::from(vec![0.0_f32]);
        let mut state = Array3::<f32>::zeros((s, s, 1));
        // Seed state with a known pattern.
        state[[0, 0, 0]] = 5.0;
        state[[1, 1, 0]] = 7.0;

        let _ = delta_net_step(&q, &k, &v, &log_g, &beta, &mut state);

        // After: state * g (=1) + 0 = state. Unchanged.
        assert!((state[[0, 0, 0]] - 5.0).abs() < 1e-6);
        assert!((state[[1, 1, 0]] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn delta_net_step_decay_halves_state() {
        // log_g = ln(0.5) ⇒ g = 0.5. State pre-update: [5, 7].
        // After decay: [2.5, 3.5]. beta = 0 → no further update.
        let s = 2;
        let q = array![[1.0_f32], [0.0]];
        let k = array![[1.0_f32], [0.0]];
        let v = array![[3.0_f32], [4.0]];
        let log_g = Array1::from(vec![(0.5_f32).ln()]);
        let beta = Array1::from(vec![0.0_f32]);
        let mut state = Array3::<f32>::zeros((s, s, 1));
        state[[0, 0, 0]] = 5.0;
        state[[1, 1, 0]] = 7.0;

        let _ = delta_net_step(&q, &k, &v, &log_g, &beta, &mut state);

        assert!((state[[0, 0, 0]] - 2.5).abs() < 1e-6);
        assert!((state[[1, 1, 0]] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn delta_net_step_two_calls_compound() {
        // Verify state carries forward across two calls.
        // Step 1: k1 = [1, 0], v1 = [3, 0] → state = k1 ⊗ (v1 * 1).
        // Step 2: k2 = [0, 1], v2 = [0, 5] → state += k2 ⊗ d2.
        //         sk2 = state^T @ k2 = [0, 0] (state was [[3,0],[0,0]]).
        //         d2  = (v2 - sk2) * 1 = [0, 5].
        //         state += k2 ⊗ d2 = [[0,0], [0,5]].
        //         final state = [[3,0], [0,5]].
        let s = 2;
        let q = array![[0.0_f32], [0.0]]; // o doesn't matter here
        let mut state = Array3::<f32>::zeros((s, s, 1));
        let log_g = Array1::from(vec![0.0_f32]);
        let beta = Array1::from(vec![1.0_f32]);

        // Step 1.
        let k1 = array![[1.0_f32], [0.0]];
        let v1 = array![[3.0_f32], [0.0]];
        let _ = delta_net_step(&q, &k1, &v1, &log_g, &beta, &mut state);
        assert!((state[[0, 0, 0]] - 3.0).abs() < 1e-6);
        assert!(state[[1, 1, 0]].abs() < 1e-6);

        // Step 2.
        let k2 = array![[0.0_f32], [1.0]];
        let v2 = array![[0.0_f32], [5.0]];
        let _ = delta_net_step(&q, &k2, &v2, &log_g, &beta, &mut state);
        assert!((state[[0, 0, 0]] - 3.0).abs() < 1e-6);
        assert!((state[[1, 1, 0]] - 5.0).abs() < 1e-6);
        assert!(state[[0, 1, 0]].abs() < 1e-6);
        assert!(state[[1, 0, 0]].abs() < 1e-6);
    }

    #[test]
    fn delta_net_step_k_broadcast_3x_to_v_heads() {
        // H_k = 1, H_v = 3 → repeat factor 3. All three V heads see
        // the same K. With distinct v per head, the rank-1 updates
        // populate distinct state slabs but share the K factor.
        let s = 2;
        let q = array![[1.0_f32], [0.0]];
        let k = array![[1.0_f32], [0.0]];
        // v shape [S_v, H_v] = [2, 3]. Each column is one V head.
        let v = array![[1.0_f32, 10.0, 100.0], [2.0, 20.0, 200.0]];
        let log_g = Array1::from(vec![0.0_f32, 0.0, 0.0]);
        let beta = Array1::from(vec![1.0_f32, 1.0, 1.0]);
        let mut state = Array3::<f32>::zeros((s, s, 3));

        let _ = delta_net_step(&q, &k, &v, &log_g, &beta, &mut state);

        // Each head h's update is k ⊗ v[:, h] (since state was 0).
        // K = [1, 0], so state[r, c, h] = k[r] * v[c, h]:
        //   state[0, 0, 0] = 1 * 1 = 1
        //   state[0, 1, 0] = 1 * 2 = 2
        //   state[0, 0, 1] = 1 * 10
        //   state[0, 1, 1] = 1 * 20
        //   etc.
        assert!((state[[0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((state[[0, 1, 0]] - 2.0).abs() < 1e-6);
        assert!((state[[0, 0, 1]] - 10.0).abs() < 1e-6);
        assert!((state[[0, 1, 1]] - 20.0).abs() < 1e-6);
        assert!((state[[0, 0, 2]] - 100.0).abs() < 1e-6);
        assert!((state[[0, 1, 2]] - 200.0).abs() < 1e-6);
        // Row 1 (where k = 0) gets no update.
        assert!(state[[1, 0, 0]].abs() < 1e-6);
        assert!(state[[1, 0, 1]].abs() < 1e-6);
        assert!(state[[1, 0, 2]].abs() < 1e-6);
    }

    #[test]
    fn delta_net_step_delta_rule_corrects_seeded_state() {
        // Verify the delta-rule "self-correction" property: if the
        // state already perfectly predicts v at this k, the delta
        // (v - sk) is zero and the state doesn't change.
        //
        // Seed: state = k ⊗ v (so S^T @ k = v).
        // Step with the same k, v: sk = v, d = (v - v) * b = 0,
        // state += 0 ⇒ unchanged (modulo decay).
        let s = 2;
        let h_k = 1;
        let h_v = 1;
        let q = array![[1.0_f32], [0.0]];
        let k = array![[1.0_f32], [0.0]];
        let v = array![[3.0_f32], [4.0]];
        let log_g = Array1::from(vec![0.0_f32]); // g = 1, no decay
        let beta = Array1::from(vec![1.0_f32]);
        let mut state = Array3::<f32>::zeros((s, s, h_v));
        let _ = h_k;
        // Seed: state[r, c, 0] = k[r, 0] * v[c, 0].
        state[[0, 0, 0]] = 1.0 * 3.0;
        state[[0, 1, 0]] = 1.0 * 4.0;
        state[[1, 0, 0]] = 0.0;
        state[[1, 1, 0]] = 0.0;

        let _ = delta_net_step(&q, &k, &v, &log_g, &beta, &mut state);

        // State should remain the seeded rank-1.
        assert!((state[[0, 0, 0]] - 3.0).abs() < 1e-5);
        assert!((state[[0, 1, 0]] - 4.0).abs() < 1e-5);
        assert!(state[[1, 0, 0]].abs() < 1e-5);
        assert!(state[[1, 1, 0]].abs() < 1e-5);
    }

    #[test]
    fn delta_net_step_q_scale_is_inverse_sqrt_s_k() {
        // Output amplitude must include the 1/sqrt(S_k) factor.
        // With S_k = 4, identity K=Q=basis, v = [1; 1; 1; 1] * 1.0,
        // beta = 1, g = 1, state starts zero:
        //   d = v - 0 = v. state = k ⊗ v.
        //   o[c] = sum_r state[r, c, 0] * q[r, kh] / sqrt(S_k)
        //        = state[0, c, 0] * 1 / sqrt(4) = v[c] / 2.
        let s = 4;
        let h_k = 1;
        let h_v = 1;
        // q = e_0 (first basis), k = e_0.
        let mut q = Array2::<f32>::zeros((s, h_k));
        q[[0, 0]] = 1.0;
        let k = q.clone();
        let v = Array2::from_elem((s, h_v), 1.0_f32);
        let log_g = Array1::from(vec![0.0_f32]);
        let beta = Array1::from(vec![1.0_f32]);
        let mut state = Array3::<f32>::zeros((s, s, h_v));

        let out = delta_net_step(&q, &k, &v, &log_g, &beta, &mut state);

        // Expected o[c, 0] = 1 / sqrt(4) = 0.5 for every c.
        for c in 0..s {
            assert!(
                (out[[c, 0]] - 0.5).abs() < 1e-5,
                "out[{c}, 0] = {}",
                out[[c, 0]]
            );
        }
    }
}
