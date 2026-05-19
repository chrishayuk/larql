//! Grouped-Query Attention (GQA) — causal attention with BLAS-fused dot products.
//!
//! Memory-efficient: O(seq) per position, never materializes full [seq, seq] matrix.
//! Uses BLAS gemv for both Q·K scores and softmax·V accumulation.

use super::{AttentionAllWeights, AttentionWeights};
use ndarray::Array2;

/// GQA with causal masking (no weight capture).
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let (out, _) =
        gqa_attention_with_weights(q, k, v, num_q, head_dim, reps, scale, seq_len, false, None);
    out
}

/// GQA that optionally captures per-head attention weights for the last token.
/// `softcap`: if Some(cap), apply tanh(scores/cap)*cap before softmax.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
) -> (Array2<f32>, Option<AttentionWeights>) {
    let (out, last, _) = gqa_attention_capture(
        q, k, v, num_q, head_dim, reps, scale, seq_len, capture, false, softcap,
    );
    (out, last)
}

/// GQA that captures every query-position attention distribution.
///
/// Diagnostic/capture tooling uses this for relation-state probes. Production
/// inference should use [`gqa_attention`] or [`gqa_attention_with_weights`].
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_all_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
) -> (Array2<f32>, AttentionAllWeights) {
    let (out, _, all) = gqa_attention_capture(
        q, k, v, num_q, head_dim, reps, scale, seq_len, false, true, softcap,
    );
    (
        out,
        all.expect("all-position attention capture requested but missing"),
    )
}

/// Capture every query-position attention distribution using only the first
/// `qk_rank` dimensions of each Q/K head. This is a diagnostic surface for
/// reduced-QK address probes; it does not compute a V-weighted output.
#[allow(clippy::too_many_arguments)]
pub fn gqa_reduced_qk_all_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    qk_rank: usize,
) -> AttentionAllWeights {
    let rank = qk_rank.clamp(1, head_dim);
    let mut captured_all_heads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_q);
    let scale_f32 = scale as f32;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        let mut captured_positions: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1;
            let q_row = q.slice(ndarray::s![qi, q_off..q_off + rank]);
            let k_block = k.slice(ndarray::s![0..causal_len, kv_off..kv_off + rank]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..causal_len {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            let max_val = scores_buf[..causal_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for score in scores_buf.iter_mut().take(causal_len) {
                let e = ((*score - max_val) as f64).exp();
                *score = e as f32;
                sum += e;
            }
            let inv_sum = (1.0 / sum) as f32;
            for score in scores_buf.iter_mut().take(causal_len) {
                *score *= inv_sum;
            }

            let mut captured = vec![0.0f32; seq_len];
            captured[..causal_len].copy_from_slice(&scores_buf[..causal_len]);
            captured_positions.push(captured);
        }
        captured_all_heads.push(captured_positions);
    }

    AttentionAllWeights {
        heads: captured_all_heads,
    }
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_capture(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture_last: bool,
    capture_all: bool,
    softcap: Option<f32>,
) -> (
    Array2<f32>,
    Option<AttentionWeights>,
    Option<AttentionAllWeights>,
) {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let mut captured_heads: Vec<Vec<f32>> = if capture_last {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };
    let mut captured_all_heads: Vec<Vec<Vec<f32>>> = if capture_all {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };

    let scale_f32 = scale as f32;
    let last_pos = seq_len - 1;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        let mut captured_positions: Vec<Vec<f32>> = if capture_all {
            Vec::with_capacity(seq_len)
        } else {
            Vec::new()
        };
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1;

            let q_row = q.slice(ndarray::s![qi, q_off..q_off + head_dim]);
            let k_block = k.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..causal_len {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            let max_val = scores_buf[..causal_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for score in scores_buf.iter_mut().take(causal_len) {
                let e = ((*score - max_val) as f64).exp();
                *score = e as f32;
                sum += e;
            }
            let inv_sum = (1.0 / sum) as f32;
            for score in scores_buf.iter_mut().take(causal_len) {
                *score *= inv_sum;
            }

            if capture_last && qi == last_pos {
                let mut captured = vec![0.0f32; seq_len];
                captured[..causal_len].copy_from_slice(&scores_buf[..causal_len]);
                captured_heads.push(captured);
            }
            if capture_all {
                let mut captured = vec![0.0f32; seq_len];
                captured[..causal_len].copy_from_slice(&scores_buf[..causal_len]);
                captured_positions.push(captured);
            }

            let v_block = v.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let scores_view = ndarray::ArrayView1::from(&scores_buf[..causal_len]);
            let weighted_v = v_block.t().dot(&scores_view);

            for d in 0..head_dim {
                out[[qi, q_off + d]] = weighted_v[d];
            }
        }
        if capture_all {
            captured_all_heads.push(captured_positions);
        }
    }

    let weights = if capture_last {
        Some(AttentionWeights {
            heads: captured_heads,
        })
    } else {
        None
    };

    let all_weights = if capture_all {
        Some(AttentionAllWeights {
            heads: captured_all_heads,
        })
    } else {
        None
    };

    (out, weights, all_weights)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn ones(rows: usize, cols: usize) -> Array2<f32> {
        Array2::ones((rows, cols))
    }

    fn small(rows: usize, cols: usize, scale: f32) -> Array2<f32> {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * scale).collect();
        Array2::from_shape_vec((rows, cols), data).unwrap()
    }

    // seq=4, num_q=2, head_dim=4, num_kv=1, reps=2
    fn run(seq: usize) -> Array2<f32> {
        let hd = 4usize;
        let nq = 2usize;
        let nkv = 1usize;
        let q = small(seq, nq * hd, 0.01);
        let k = small(seq, nkv * hd, 0.01);
        let v = small(seq, nkv * hd, 0.01);
        gqa_attention(&q, &k, &v, nq, hd, nq / nkv, 1.0 / (hd as f64).sqrt(), seq)
    }

    #[test]
    fn gqa_output_shape() {
        let out = run(3);
        assert_eq!(out.shape(), &[3, 2 * 4]); // [seq, num_q * head_dim]
    }

    #[test]
    fn gqa_output_finite() {
        let out = run(4);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "gqa output has non-finite values"
        );
    }

    #[test]
    fn gqa_single_token() {
        let out = run(1);
        assert_eq!(out.shape(), &[1, 8]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gqa_causal_last_token_attends_all() {
        // Last token can attend to all positions.
        // With uniform Q/K, attention should be distributed (not focused).
        let seq = 4usize;
        let hd = 4usize;
        let nq = 1usize;
        let q = ones(seq, hd);
        let k = ones(seq, hd);
        let v = small(seq, hd, 1.0); // distinct values
        let out = gqa_attention(&q, &k, &v, nq, hd, 1, 1.0 / (hd as f64).sqrt(), seq);
        // Last row should be a weighted average of V rows (all weights equal → mean)
        let expected_last: Vec<f32> =
            v.rows().into_iter().fold(vec![0.0f32; hd], |mut acc, row| {
                for (a, v) in acc.iter_mut().zip(row.iter()) {
                    *a += v / seq as f32;
                }
                acc
            });
        let got_last: Vec<f32> = out.row(seq - 1).to_vec();
        for (e, g) in expected_last.iter().zip(got_last.iter()) {
            assert!(
                (e - g).abs() < 0.01,
                "last token mean-attn mismatch: {e} vs {g}"
            );
        }
    }

    #[test]
    fn gqa_with_weights_captures_softmax() {
        let seq = 3usize;
        let hd = 4usize;
        let q = small(seq, hd, 0.1);
        let k = small(seq, hd, 0.1);
        let v = small(seq, hd, 0.1);
        let (out, weights) = gqa_attention_with_weights(
            &q,
            &k,
            &v,
            1,
            hd,
            1,
            1.0 / (hd as f64).sqrt(),
            seq,
            true,
            None,
        );
        assert!(out.iter().all(|v| v.is_finite()));
        let w = weights.expect("weights should be captured");
        // Attention weights for last position should sum to ~1
        let sum: f32 = w.heads[0].iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "attention weights should sum to 1, got {sum}"
        );
    }

    // ── GQA reps > 1: multiple Q-heads per KV-head ───────────────────────────

    #[test]
    fn gqa_reps_2_output_shape() {
        // num_q=4, num_kv=2, reps=2 — 2 Q-heads share each KV-head
        let seq = 3usize;
        let hd = 4usize;
        let num_q = 4usize;
        let num_kv = 2usize;
        let reps = num_q / num_kv;
        let q = small(seq, num_q * hd, 0.01);
        let k = small(seq, num_kv * hd, 0.01);
        let v = small(seq, num_kv * hd, 0.01);
        let out = gqa_attention(&q, &k, &v, num_q, hd, reps, 1.0 / (hd as f64).sqrt(), seq);
        assert_eq!(
            out.shape(),
            &[seq, num_q * hd],
            "output should be [seq, num_q * head_dim]"
        );
    }

    #[test]
    fn gqa_reps_2_output_is_finite() {
        let seq = 4usize;
        let hd = 8usize;
        let num_q = 4usize;
        let num_kv = 2usize;
        let q = small(seq, num_q * hd, 0.01);
        let k = small(seq, num_kv * hd, 0.01);
        let v = small(seq, num_kv * hd, 0.01);
        let out = gqa_attention(
            &q,
            &k,
            &v,
            num_q,
            hd,
            num_q / num_kv,
            1.0 / (hd as f64).sqrt(),
            seq,
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "reps=2 GQA output has non-finite values"
        );
    }

    #[test]
    fn gqa_softcap_keeps_output_finite_and_within_cap() {
        // softcap=Some(cap) routes through `(s/cap).tanh() * cap` — must
        // produce finite output AND attention weights still summing to 1.
        let seq = 4usize;
        let hd = 4usize;
        let q = small(seq, hd, 0.5); // large enough to push raw scores >> cap
        let k = small(seq, hd, 0.5);
        let v = small(seq, hd, 0.1);
        let cap = 5.0f32;
        let (out, weights) = gqa_attention_with_weights(
            &q,
            &k,
            &v,
            1,
            hd,
            1,
            1.0 / (hd as f64).sqrt(),
            seq,
            true,
            Some(cap),
        );
        assert!(out.iter().all(|v| v.is_finite()));
        let w = weights.unwrap();
        // Last-position weights must still be a valid distribution.
        let sum: f32 = w.heads[0].iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "softcap weights should sum to 1, got {sum}"
        );
    }

    #[test]
    fn gqa_softcap_differs_from_no_cap_when_cap_binds() {
        // With small softcap and large logits, output should differ from
        // the no-cap version — confirms the softcap branch is actually
        // running rather than being a no-op.
        let seq = 3usize;
        let hd = 4usize;
        let q = small(seq, hd, 1.0); // big logits
        let k = small(seq, hd, 1.0);
        let v = small(seq, hd, 1.0);
        let scale = 1.0 / (hd as f64).sqrt();
        let out_nocap = gqa_attention(&q, &k, &v, 1, hd, 1, scale, seq);
        let (out_cap, _) =
            gqa_attention_with_weights(&q, &k, &v, 1, hd, 1, scale, seq, false, Some(0.5));
        let mut max_diff = 0.0f32;
        for (a, b) in out_nocap.iter().zip(out_cap.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff > 1e-3,
            "softcap should change output when cap binds, max_diff={max_diff}"
        );
    }

    // ── gqa_attention_with_all_weights — every-position capture ────────

    #[test]
    fn gqa_with_all_weights_captures_every_position() {
        let seq = 3usize;
        let hd = 4usize;
        let num_q = 2usize;
        let q = small(seq, num_q * hd, 0.1);
        let k = small(seq, hd, 0.1);
        let v = small(seq, hd, 0.1);
        let (out, all) = gqa_attention_with_all_weights(
            &q,
            &k,
            &v,
            num_q,
            hd,
            num_q, // reps so 1 KV head
            1.0 / (hd as f64).sqrt(),
            seq,
            None,
        );
        assert_eq!(out.shape(), &[seq, num_q * hd]);
        // Every Q-head x every Q-position must have a captured distribution.
        assert_eq!(all.heads.len(), num_q);
        for head in &all.heads {
            assert_eq!(head.len(), seq);
            for (qi, dist) in head.iter().enumerate() {
                assert_eq!(dist.len(), seq);
                // Causal: positions > qi are zero; positions ≤ qi sum to 1.
                let causal_sum: f32 = dist[..=qi].iter().sum();
                assert!(
                    (causal_sum - 1.0).abs() < 1e-3,
                    "qi={qi} causal weights should sum to 1, got {causal_sum}"
                );
                for &future in &dist[qi + 1..] {
                    assert_eq!(future, 0.0, "qi={qi} non-causal weight non-zero");
                }
            }
        }
    }

    #[test]
    fn gqa_with_all_weights_softcap_keeps_distribution_valid() {
        let seq = 2usize;
        let hd = 4usize;
        let q = small(seq, hd, 1.0);
        let k = small(seq, hd, 1.0);
        let v = small(seq, hd, 1.0);
        let (_, all) = gqa_attention_with_all_weights(
            &q,
            &k,
            &v,
            1,
            hd,
            1,
            1.0 / (hd as f64).sqrt(),
            seq,
            Some(0.5),
        );
        for head in &all.heads {
            for (qi, dist) in head.iter().enumerate() {
                let causal_sum: f32 = dist[..=qi].iter().sum();
                assert!((causal_sum - 1.0).abs() < 1e-3);
            }
        }
    }

    // ── gqa_reduced_qk_all_weights — diagnostic reduced-rank capture ────

    #[test]
    fn reduced_qk_all_weights_returns_per_head_per_position_distributions() {
        let seq = 3usize;
        let hd = 8usize;
        let num_q = 2usize;
        let q = small(seq, num_q * hd, 0.1);
        let k = small(seq, hd, 0.1);
        let all = gqa_reduced_qk_all_weights(
            &q,
            &k,
            num_q,
            hd,
            num_q, // reps
            1.0 / (hd as f64).sqrt(),
            seq,
            None,
            4, // qk_rank — half of head_dim
        );
        assert_eq!(all.heads.len(), num_q);
        for head in &all.heads {
            assert_eq!(head.len(), seq);
            for (qi, dist) in head.iter().enumerate() {
                let causal_sum: f32 = dist[..=qi].iter().sum();
                assert!((causal_sum - 1.0).abs() < 1e-3);
                for &future in &dist[qi + 1..] {
                    assert_eq!(future, 0.0);
                }
            }
        }
    }

    #[test]
    fn reduced_qk_clamps_rank_above_head_dim() {
        // qk_rank > head_dim should clamp to head_dim and behave like
        // the full-rank capture.
        let seq = 2usize;
        let hd = 4usize;
        let q = small(seq, hd, 0.1);
        let k = small(seq, hd, 0.1);
        let scale = 1.0 / (hd as f64).sqrt();
        let all_full = gqa_reduced_qk_all_weights(&q, &k, 1, hd, 1, scale, seq, None, hd);
        let all_clamped = gqa_reduced_qk_all_weights(&q, &k, 1, hd, 1, scale, seq, None, hd * 4);
        for (a, b) in all_full.heads[0].iter().zip(all_clamped.heads[0].iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-6,
                    "qk_rank > head_dim must clamp: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn reduced_qk_clamps_rank_zero_to_one() {
        // qk_rank=0 clamps to 1 — a 1-dim QK projection still produces
        // a valid distribution.
        let seq = 3usize;
        let hd = 4usize;
        let q = small(seq, hd, 0.1);
        let k = small(seq, hd, 0.1);
        let all =
            gqa_reduced_qk_all_weights(&q, &k, 1, hd, 1, 1.0 / (hd as f64).sqrt(), seq, None, 0);
        for (qi, dist) in all.heads[0].iter().enumerate() {
            let causal_sum: f32 = dist[..=qi].iter().sum();
            assert!(
                (causal_sum - 1.0).abs() < 1e-3,
                "qi={qi} dist must still be valid distribution"
            );
        }
    }

    #[test]
    fn reduced_qk_softcap_branch() {
        // Hit the `Some(cap)` arm in the reduced-QK function.
        let seq = 2usize;
        let hd = 4usize;
        let q = small(seq, hd, 1.0);
        let k = small(seq, hd, 1.0);
        let all = gqa_reduced_qk_all_weights(
            &q,
            &k,
            1,
            hd,
            1,
            1.0 / (hd as f64).sqrt(),
            seq,
            Some(0.5),
            hd,
        );
        for (qi, dist) in all.heads[0].iter().enumerate() {
            let causal_sum: f32 = dist[..=qi].iter().sum();
            assert!((causal_sum - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn gqa_reps_2_head_pairs_share_kv() {
        // Q-heads 0,1 use KV-head 0; Q-heads 2,3 use KV-head 1.
        // With Q equal to each other within a pair, output should also match.
        let seq = 2usize;
        let hd = 4usize;
        let num_q = 4usize;
        let num_kv = 2usize;
        let reps = num_q / num_kv;
        // Q rows: heads 0 and 1 are identical; heads 2 and 3 are identical but different from 0/1
        let mut q_data = vec![0.0f32; seq * num_q * hd];
        for s in 0..seq {
            for d in 0..hd {
                q_data[s * num_q * hd + d] = 0.1; // head 0
                q_data[s * num_q * hd + hd + d] = 0.1; // head 1 (same as 0)
                q_data[s * num_q * hd + 2 * hd + d] = 0.5; // head 2
                q_data[s * num_q * hd + 3 * hd + d] = 0.5; // head 3 (same as 2)
            }
        }
        let q = Array2::from_shape_vec((seq, num_q * hd), q_data).unwrap();
        let k = small(seq, num_kv * hd, 0.1);
        let v = small(seq, num_kv * hd, 0.1);
        let out = gqa_attention(&q, &k, &v, num_q, hd, reps, 1.0 / (hd as f64).sqrt(), seq);
        // heads 0 and 1 should produce identical output rows (same Q, same KV)
        let h0: Vec<f32> = out.row(0).iter().take(hd).copied().collect();
        let h1: Vec<f32> = out.row(0).iter().skip(hd).take(hd).copied().collect();
        for (a, b) in h0.iter().zip(h1.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "heads 0 and 1 should produce same output: {a} vs {b}"
            );
        }
    }
}
