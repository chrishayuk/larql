//! Tests for fused online-softmax attention.
//!
//! Validates that the fused GQA attention kernel produces correct results
//! across: single token, causal mask, multi-head, GQA, softcap, and
//! attention weight capture. Also tests against a naive reference
//! implementation to verify numerical equivalence.

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

use larql_inference::attention::{gqa_attention, gqa_attention_with_weights};
use ndarray::Array2;

/// Deterministic matrix for tests.
fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Naive reference attention (materialized scores) for validation.
/// C = softmax(causal_mask(Q @ K^T * scale)) @ V
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let scale_f32 = scale as f32;

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            // Compute all scores for this query position
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[[qi, q_off + d]] * k[[ki, kv_off + d]];
                }
                dot *= scale_f32;
                if let Some(cap) = softcap {
                    dot = (dot / cap).tanh() * cap;
                }
                scores[ki] = dot;
            }

            // Standard softmax with f64 accumulation
            let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            let mut exp_scores = vec![0.0f64; seq_len];
            for j in 0..seq_len {
                let e = ((scores[j] - max_val) as f64).exp();
                exp_scores[j] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;

            // Weighted V sum
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for ki in 0..=qi {
                    acc += exp_scores[ki] * v[[ki, kv_off + d]] as f64;
                }
                out[[qi, q_off + d]] = (acc * inv_sum) as f32;
            }
        }
    }
    out
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── Basic correctness ──

mod basic {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn single_token() {
        // Single token: attention weight = 1.0, output = V
        let q = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = q.clone();
        let v = Array2::from_shape_vec((1, 4), vec![0.5, 0.5, 0.5, 0.5]).unwrap();
        let out = gqa_attention(&q, &k, &v, 1, 4, 1, 0.5, 1);
        for j in 0..4 {
            assert!(
                (out[[0, j]] - 0.5).abs() < 1e-5,
                "single token output[{j}] = {}, expected 0.5",
                out[[0, j]]
            );
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn causal_mask_two_tokens() {
        // Token 0 only sees itself, token 1 sees both
        let q = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = q.clone();
        let v = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let out = gqa_attention(&q, &k, &v, 1, 2, 1, 0.5_f64.sqrt(), 2);
        assert_eq!(out.shape(), &[2, 2]);
        // Token 0: only attends to itself, output = V[0] = [1, 0]
        assert!((out[[0, 0]] - 1.0).abs() < 1e-4);
        assert!((out[[0, 1]] - 0.0).abs() < 1e-4);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn output_shape() {
        let seq = 5;
        let head_dim = 8;
        let num_heads = 3;
        let q = synth_matrix(seq, num_heads * head_dim, 1);
        let k = synth_matrix(seq, num_heads * head_dim, 2);
        let v = synth_matrix(seq, num_heads * head_dim, 3);
        let out = gqa_attention(
            &q,
            &k,
            &v,
            num_heads,
            head_dim,
            1,
            1.0 / (head_dim as f64).sqrt(),
            seq,
        );
        assert_eq!(out.shape(), &[seq, num_heads * head_dim]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn uniform_attention_averages_v() {
        // When all Q·K scores are equal, attention averages V rows
        let seq = 3;
        let head_dim = 2;
        let q = Array2::zeros((seq, head_dim));
        let k = Array2::zeros((seq, head_dim));
        // V rows: [1,0], [0,1], [2,2]
        let v =
            Array2::from_shape_vec((seq, head_dim), vec![1.0, 0.0, 0.0, 1.0, 2.0, 2.0]).unwrap();
        let out = gqa_attention(&q, &k, &v, 1, head_dim, 1, 1.0, seq);

        // Token 0: only sees V[0] = [1, 0]
        assert!((out[[0, 0]] - 1.0).abs() < 1e-4);
        assert!((out[[0, 1]] - 0.0).abs() < 1e-4);

        // Token 1: uniform over V[0], V[1] = avg([1,0], [0,1]) = [0.5, 0.5]
        assert!((out[[1, 0]] - 0.5).abs() < 1e-4);
        assert!((out[[1, 1]] - 0.5).abs() < 1e-4);

        // Token 2: uniform over V[0], V[1], V[2] = avg = [1.0, 1.0]
        assert!((out[[2, 0]] - 1.0).abs() < 1e-4);
        assert!((out[[2, 1]] - 1.0).abs() < 1e-4);
    }
}

// ── Numerical agreement with reference ──

mod reference_agreement {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn single_head_small() {
        let seq = 4;
        let head_dim = 8;
        let q = synth_matrix(seq, head_dim, 10);
        let k = synth_matrix(seq, head_dim, 11);
        let v = synth_matrix(seq, head_dim, 12);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let fused = gqa_attention(&q, &k, &v, 1, head_dim, 1, scale, seq);
        let naive = reference_attention(&q, &k, &v, 1, head_dim, 1, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-5, "single head diff = {diff}");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn multi_head() {
        let seq = 6;
        let head_dim = 16;
        let num_heads = 4;
        let q = synth_matrix(seq, num_heads * head_dim, 20);
        let k = synth_matrix(seq, num_heads * head_dim, 21);
        let v = synth_matrix(seq, num_heads * head_dim, 22);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let fused = gqa_attention(&q, &k, &v, num_heads, head_dim, 1, scale, seq);
        let naive = reference_attention(&q, &k, &v, num_heads, head_dim, 1, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-4, "multi-head diff = {diff}");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn gqa_2x_ratio() {
        // 4 Q heads sharing 2 KV heads (reps=2)
        let seq = 6;
        let head_dim = 8;
        let num_q = 4;
        let num_kv = 2;
        let reps = num_q / num_kv;
        let q = synth_matrix(seq, num_q * head_dim, 30);
        let k = synth_matrix(seq, num_kv * head_dim, 31);
        let v = synth_matrix(seq, num_kv * head_dim, 32);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let fused = gqa_attention(&q, &k, &v, num_q, head_dim, reps, scale, seq);
        let naive = reference_attention(&q, &k, &v, num_q, head_dim, reps, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-4, "GQA 2x diff = {diff}");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn gqa_gemma3_dimensions() {
        // Gemma-3 4B: 10 Q heads, 2 KV heads, head_dim=256, reps=5
        let seq = 6;
        let head_dim = 32; // scaled down for test speed
        let num_q = 10;
        let num_kv = 2;
        let reps = num_q / num_kv;
        let q = synth_matrix(seq, num_q * head_dim, 40);
        let k = synth_matrix(seq, num_kv * head_dim, 41);
        let v = synth_matrix(seq, num_kv * head_dim, 42);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let fused = gqa_attention(&q, &k, &v, num_q, head_dim, reps, scale, seq);
        let naive = reference_attention(&q, &k, &v, num_q, head_dim, reps, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-4, "Gemma3-like GQA diff = {diff}");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn with_softcap() {
        let seq = 4;
        let head_dim = 8;
        let q = synth_matrix(seq, head_dim, 50);
        let k = synth_matrix(seq, head_dim, 51);
        let v = synth_matrix(seq, head_dim, 52);
        let scale = 1.0 / (head_dim as f64).sqrt();
        let softcap = Some(50.0f32);

        let (fused, _) =
            gqa_attention_with_weights(&q, &k, &v, 1, head_dim, 1, scale, seq, false, softcap);
        let naive = reference_attention(&q, &k, &v, 1, head_dim, 1, scale, seq, softcap);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-5, "softcap diff = {diff}");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn longer_sequence() {
        let seq = 24;
        let head_dim = 16;
        let num_heads = 2;
        let q = synth_matrix(seq, num_heads * head_dim, 60);
        let k = synth_matrix(seq, num_heads * head_dim, 61);
        let v = synth_matrix(seq, num_heads * head_dim, 62);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let fused = gqa_attention(&q, &k, &v, num_heads, head_dim, 1, scale, seq);
        let naive = reference_attention(&q, &k, &v, num_heads, head_dim, 1, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-3, "seq=24 diff = {diff}");
    }
}

// ── Attention weight capture ──

mod capture {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn capture_returns_weights() {
        let seq = 4;
        let head_dim = 8;
        let num_heads = 2;
        let q = synth_matrix(seq, num_heads * head_dim, 70);
        let k = synth_matrix(seq, num_heads * head_dim, 71);
        let v = synth_matrix(seq, num_heads * head_dim, 72);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let (_, weights) =
            gqa_attention_with_weights(&q, &k, &v, num_heads, head_dim, 1, scale, seq, true, None);

        let weights = weights.expect("should capture weights");
        assert_eq!(weights.heads.len(), num_heads);
        for head_weights in &weights.heads {
            assert_eq!(head_weights.len(), seq);
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn captured_weights_sum_to_one() {
        let seq = 6;
        let head_dim = 8;
        let q = synth_matrix(seq, head_dim, 80);
        let k = synth_matrix(seq, head_dim, 81);
        let v = synth_matrix(seq, head_dim, 82);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let (_, weights) =
            gqa_attention_with_weights(&q, &k, &v, 1, head_dim, 1, scale, seq, true, None);

        let w = &weights.unwrap().heads[0];
        let sum: f32 = w.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "attention weights sum = {sum}, expected 1.0"
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn captured_weights_causal() {
        // Last token's weights: positions > last_pos should be 0
        let seq = 4;
        let head_dim = 4;
        let q = synth_matrix(seq, head_dim, 90);
        let k = synth_matrix(seq, head_dim, 91);
        let v = synth_matrix(seq, head_dim, 92);

        let (_, weights) =
            gqa_attention_with_weights(&q, &k, &v, 1, head_dim, 1, 0.5, seq, true, None);

        let w = &weights.unwrap().heads[0];
        // All weights should be non-negative (softmax output)
        for &w_val in w {
            assert!(w_val >= 0.0, "negative attention weight: {w_val}");
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn no_capture_returns_none() {
        let q = synth_matrix(3, 4, 100);
        let k = synth_matrix(3, 4, 101);
        let v = synth_matrix(3, 4, 102);

        let (_, weights) = gqa_attention_with_weights(&q, &k, &v, 1, 4, 1, 0.5, 3, false, None);
        assert!(weights.is_none());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn capture_does_not_change_output() {
        let seq = 4;
        let head_dim = 8;
        let q = synth_matrix(seq, head_dim, 110);
        let k = synth_matrix(seq, head_dim, 111);
        let v = synth_matrix(seq, head_dim, 112);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let (out_no_cap, _) =
            gqa_attention_with_weights(&q, &k, &v, 1, head_dim, 1, scale, seq, false, None);
        let (out_cap, _) =
            gqa_attention_with_weights(&q, &k, &v, 1, head_dim, 1, scale, seq, true, None);

        let diff = max_diff(&out_no_cap, &out_cap);
        assert!(diff < 1e-6, "capture changed output: diff = {diff}");
    }
}

// ── Edge cases ──

mod edge_cases {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn single_token_single_dim() {
        let q = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let k = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let v = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let out = gqa_attention(&q, &k, &v, 1, 1, 1, 1.0, 1);
        assert!((out[[0, 0]] - 3.0).abs() < 1e-5);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn large_head_dim() {
        let seq = 3;
        let head_dim = 256; // Gemma-3 head_dim
        let q = synth_matrix(seq, head_dim, 120);
        let k = synth_matrix(seq, head_dim, 121);
        let v = synth_matrix(seq, head_dim, 122);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let fused = gqa_attention(&q, &k, &v, 1, head_dim, 1, scale, seq);
        let naive = reference_attention(&q, &k, &v, 1, head_dim, 1, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-4, "large head_dim diff = {diff}");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn custom_scale() {
        // Granite-style: custom attention_multiplier instead of 1/sqrt(head_dim)
        let seq = 3;
        let head_dim = 8;
        let q = synth_matrix(seq, head_dim, 130);
        let k = synth_matrix(seq, head_dim, 131);
        let v = synth_matrix(seq, head_dim, 132);
        let scale = 0.125; // Custom scale

        let fused = gqa_attention(&q, &k, &v, 1, head_dim, 1, scale, seq);
        let naive = reference_attention(&q, &k, &v, 1, head_dim, 1, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-5, "custom scale diff = {diff}");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn large_head_dim_512() {
        // Gemma 4 global attention: head_dim=512
        let seq = 3;
        let head_dim = 512;
        let q = synth_matrix(seq, head_dim, 140);
        let k = synth_matrix(seq, head_dim, 141);
        let v = synth_matrix(seq, head_dim, 142);
        let scale = 1.0 / (head_dim as f64).sqrt();

        let fused = gqa_attention(&q, &k, &v, 1, head_dim, 1, scale, seq);
        let naive = reference_attention(&q, &k, &v, 1, head_dim, 1, scale, seq, None);

        let diff = max_diff(&fused, &naive);
        assert!(diff < 1e-3, "head_dim=512 diff = {diff}");
    }
}

// ── RoPE tests ──

mod rope_tests {
    use larql_inference::attention::{apply_rope, apply_rope_partial};
    use ndarray::Array2;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn partial_rope_fraction_1_matches_full() {
        // apply_rope_partial with fraction=1.0 should match apply_rope exactly
        let seq = 4;
        let heads = 2;
        let head_dim = 64;
        let dim = heads * head_dim;
        let base = 10000.0;

        let x = Array2::from_shape_fn((seq, dim), |(i, j)| ((i * dim + j) as f32 * 0.01).sin());

        let full = apply_rope(&x, heads, head_dim, base);
        let partial = apply_rope_partial(&x, heads, head_dim, base, 1.0);

        for i in 0..seq {
            for j in 0..dim {
                assert!(
                    (full[[i, j]] - partial[[i, j]]).abs() < 1e-6,
                    "mismatch at [{i},{j}]: full={}, partial={}",
                    full[[i, j]],
                    partial[[i, j]]
                );
            }
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn partial_rope_preserves_non_rotated_dims() {
        // With fraction=0.25 (Gemma 4 style), dims beyond rotary_dim should be unchanged
        let seq = 4;
        let heads = 1;
        let head_dim = 64;
        let dim = heads * head_dim;
        let base = 1000000.0;
        let fraction = 0.25;

        let x = Array2::from_shape_fn((seq, dim), |(i, j)| ((i * dim + j) as f32 * 0.01).sin());

        let result = apply_rope_partial(&x, heads, head_dim, base, fraction);

        let rotary_dim = (head_dim as f64 * fraction) as usize; // 16
                                                                // Dims [rotary_dim..head_dim] should be untouched
        for pos in 0..seq {
            for d in rotary_dim..head_dim {
                assert_eq!(
                    result[[pos, d]],
                    x[[pos, d]],
                    "dim {d} at pos {pos} was modified: {} -> {}",
                    x[[pos, d]],
                    result[[pos, d]]
                );
            }
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn partial_rope_rotates_correct_dims() {
        // Rotated dims should differ from input (at pos > 0)
        let seq = 4;
        let heads = 1;
        let head_dim = 64;
        let dim = heads * head_dim;
        let base = 10000.0;
        let fraction = 0.25;

        let x = Array2::from_shape_fn((seq, dim), |(_, j)| (j as f32 + 1.0) * 0.1);

        let result = apply_rope_partial(&x, heads, head_dim, base, fraction);

        let rotary_dim = (head_dim as f64 * fraction) as usize;
        // At pos=0, RoPE is identity (angle=0, cos=1, sin=0)
        for d in 0..rotary_dim {
            assert!(
                (result[[0, d]] - x[[0, d]]).abs() < 1e-6,
                "pos=0 should be identity"
            );
        }
        // At pos > 0, rotated dims should differ
        let mut any_changed = false;
        for d in 0..rotary_dim {
            if (result[[1, d]] - x[[1, d]]).abs() > 1e-6 {
                any_changed = true;
            }
        }
        assert!(any_changed, "no dims were rotated at pos=1");
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn partial_rope_multi_head() {
        // Verify per-head rotation works correctly with partial RoPE
        let seq = 2;
        let heads = 4;
        let head_dim = 32;
        let dim = heads * head_dim;
        let base = 10000.0;
        let fraction = 0.5;

        let x = Array2::from_shape_fn((seq, dim), |(i, j)| ((i * dim + j) as f32 * 0.01).sin());

        let result = apply_rope_partial(&x, heads, head_dim, base, fraction);

        let rotary_dim = (head_dim as f64 * fraction) as usize;
        // Each head's non-rotated dims should be unchanged
        for h in 0..heads {
            let offset = h * head_dim;
            for pos in 0..seq {
                for d in rotary_dim..head_dim {
                    assert_eq!(
                        result[[pos, offset + d]],
                        x[[pos, offset + d]],
                        "head {h} dim {d} at pos {pos} was modified"
                    );
                }
            }
        }
    }
}
