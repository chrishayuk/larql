//! Direct embedding (`W_E`) and unembedding (`W_U`) primitives.
//!
//! The matrices themselves are public on [`ModelWeights`] (`weights.embed`,
//! `weights.lm_head`), but mech-interp tools want a few canned operations
//! on top of them:
//!
//! - [`embedding_row`] / [`embedding_row_scaled`] — read one token's
//!   embedding row from `W_E`, with or without the architecture's
//!   `embed_scale` (so the result matches what the forward pass actually
//!   inserts into the residual).
//! - [`unembedding_row`] — read one token's row from `W_U` (i.e. the
//!   direction the unembed projects onto when scoring that token).
//! - [`embedding_neighbors`] — top-k tokens by cosine similarity to a
//!   query vector, scored against `W_E`. Replaces lazarus's
//!   `embedding_neighbors`.
//! - [`project_through_unembed`] — raw `W_U @ vec` followed by top-k
//!   over logits. **No final norm, no softcap, no scaling.** This is
//!   pure DLA; for the full lens (with norm/softcap/scale) use
//!   [`super::lens::logit_lens_topk`].

use crate::model::ModelWeights;
use ndarray::{ArrayView1, ArrayView2};

/// Raw row of `W_E` for `token_id`. Returns `None` if the id is out of
/// range. Does **not** apply the architecture's `embed_scale` — this is
/// the matrix as stored. Use [`embedding_row_scaled`] if you want what
/// the forward pass actually inserts.
pub fn embedding_row(weights: &ModelWeights, token_id: u32) -> Option<Vec<f32>> {
    let idx = token_id as usize;
    if idx >= weights.embed.nrows() {
        return None;
    }
    Some(weights.embed.row(idx).to_vec())
}

/// Same as [`embedding_row`] but multiplied by `arch.embed_scale()` —
/// matches the residual the forward pass writes for this token.
pub fn embedding_row_scaled(weights: &ModelWeights, token_id: u32) -> Option<Vec<f32>> {
    let mut row = embedding_row(weights, token_id)?;
    let scale = weights.arch.embed_scale();
    if scale != 1.0 {
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
    Some(row)
}

/// Raw row of `W_U` (the unembedding / `lm_head` matrix) for `token_id`.
/// This is the direction whose dot product with the final residual gives
/// the raw logit for that token (before any norm/softcap/scaling).
pub fn unembedding_row(weights: &ModelWeights, token_id: u32) -> Option<Vec<f32>> {
    let idx = token_id as usize;
    if idx >= weights.lm_head.nrows() {
        return None;
    }
    Some(weights.lm_head.row(idx).to_vec())
}

/// Top-k tokens by **cosine similarity** to `query` against the embedding
/// matrix `W_E`. Returns `(token_id, cosine)` pairs in descending order.
///
/// Used for "what tokens does this vector look like?" — lazarus's
/// `embedding_neighbors`. Cosine, not raw dot-product, so different-norm
/// vectors are comparable.
///
/// Returns empty on dimension mismatch or `k == 0`.
pub fn embedding_neighbors(weights: &ModelWeights, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    if query.len() != weights.hidden_size || k == 0 {
        return Vec::new();
    }
    let q_view = ArrayView1::from(query);
    let q_norm = vec_norm(q_view);
    if q_norm == 0.0 {
        return Vec::new();
    }
    cosine_topk_against_matrix(weights.embed.view(), q_view, q_norm, k)
}

/// Raw unembedding projection: returns top-k `(token_id, logit)` pairs
/// from `lm_head @ vec`. **No final norm, no softcap, no logits-scale,
/// no softmax.** This is the direct-logit-attribution primitive — apply
/// it to a head's output, an FFN's contribution, or any direction you
/// want to read out as a vocabulary distribution without the model's
/// usual final-stage normalisation.
///
/// For the full logit-lens (norm + softcap + softmax) use
/// [`super::lens::logit_lens_topk`].
pub fn project_through_unembed(weights: &ModelWeights, vec: &[f32], k: usize) -> Vec<(u32, f32)> {
    if vec.len() != weights.hidden_size || k == 0 {
        return Vec::new();
    }
    let v = ArrayView1::from(vec);
    let mut scored: Vec<(usize, f32)> = (0..weights.lm_head.nrows())
        .map(|i| {
            let row = weights.lm_head.row(i);
            let dot: f32 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            (i, dot)
        })
        .collect();
    let n = scored.len();
    let take = k.min(n);
    let pivot = take.min(n - 1);
    scored.select_nth_unstable_by(pivot, cmp_desc_nan_last);
    scored.truncate(take);
    scored.sort_unstable_by(cmp_desc_nan_last);
    scored.into_iter().map(|(i, s)| (i as u32, s)).collect()
}

// ── internals ───────────────────────────────────────────────────────────────

fn vec_norm(v: ArrayView1<f32>) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_topk_against_matrix(
    matrix: ArrayView2<f32>,
    query: ArrayView1<f32>,
    query_norm: f32,
    k: usize,
) -> Vec<(u32, f32)> {
    let n = matrix.nrows();
    let mut scored: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let row = matrix.row(i);
            let dot: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            let r_norm = vec_norm(row);
            let denom = r_norm * query_norm;
            let cos = if denom > 0.0 { dot / denom } else { 0.0 };
            (i, cos)
        })
        .collect();
    let take = k.min(n);
    if take == 0 {
        return Vec::new();
    }
    let pivot = take.min(n - 1);
    scored.select_nth_unstable_by(pivot, cmp_desc_nan_last);
    scored.truncate(take);
    scored.sort_unstable_by(cmp_desc_nan_last);
    scored.into_iter().map(|(i, s)| (i as u32, s)).collect()
}

fn cmp_desc_nan_last(a: &(usize, f32), b: &(usize, f32)) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::model::ModelWeights;
    use crate::test_utils::make_test_weights;
    use std::sync::OnceLock;

    fn shared_weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    // ── embedding_row ──────────────────────────────────────────────────────

    #[test]
    fn embedding_row_shape() {
        let weights = shared_weights();
        let row = embedding_row(weights, 0).expect("token 0");
        assert_eq!(row.len(), weights.hidden_size);
        assert!(row.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn embedding_row_out_of_range_returns_none() {
        let weights = shared_weights();
        assert!(embedding_row(weights, u32::MAX).is_none());
    }

    #[test]
    fn embedding_row_scaled_matches_forward_path() {
        // Scaled row should equal what embed_tokens_pub writes for that token.
        let weights = shared_weights();
        let from_helper = embedding_row_scaled(weights, 2).expect("token 2");
        let from_forward = super::super::embed::embed_tokens_pub(weights, &[2u32]);
        for (a, b) in from_helper.iter().zip(from_forward.row(0).iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "scaled row diverged from forward path"
            );
        }
    }

    // ── unembedding_row ────────────────────────────────────────────────────

    #[test]
    fn unembedding_row_shape() {
        let weights = shared_weights();
        let row = unembedding_row(weights, 0).expect("token 0");
        assert_eq!(row.len(), weights.hidden_size);
    }

    #[test]
    fn unembedding_row_out_of_range_returns_none() {
        let weights = shared_weights();
        assert!(unembedding_row(weights, u32::MAX).is_none());
    }

    // ── embedding_neighbors ────────────────────────────────────────────────

    #[test]
    fn embedding_neighbors_self_is_top_with_unit_cosine() {
        // Querying with token N's own embedding should put N at the top
        // with cosine ≈ 1.0.
        let weights = shared_weights();
        let q = embedding_row(weights, 3).unwrap();
        let neighbors = embedding_neighbors(weights, &q, 3);
        assert!(!neighbors.is_empty());
        assert_eq!(neighbors[0].0, 3, "self should be top neighbor");
        assert!(
            (neighbors[0].1 - 1.0).abs() < 1e-4,
            "self-cosine should be ~1.0, got {}",
            neighbors[0].1
        );
    }

    #[test]
    fn embedding_neighbors_descending() {
        let weights = shared_weights();
        let q = embedding_row(weights, 0).unwrap();
        let neighbors = embedding_neighbors(weights, &q, 5);
        for w in neighbors.windows(2) {
            assert!(w[0].1 >= w[1].1, "must be descending");
        }
    }

    #[test]
    fn embedding_neighbors_dim_mismatch_returns_empty() {
        let weights = shared_weights();
        assert!(embedding_neighbors(weights, &[0.0; 1], 5).is_empty());
    }

    #[test]
    fn embedding_neighbors_zero_query_returns_empty() {
        let weights = shared_weights();
        let zero = vec![0.0; weights.hidden_size];
        assert!(embedding_neighbors(weights, &zero, 5).is_empty());
    }

    // ── project_through_unembed ────────────────────────────────────────────

    #[test]
    fn project_through_unembed_returns_descending_topk() {
        let weights = shared_weights();
        let vec: Vec<f32> = (0..weights.hidden_size)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        let result = project_through_unembed(weights, &vec, 5);
        assert_eq!(result.len(), 5);
        for w in result.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn project_through_unembed_matches_manual_dot() {
        let weights = shared_weights();
        let vec: Vec<f32> = (0..weights.hidden_size)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let result = project_through_unembed(weights, &vec, weights.vocab_size);
        // Verify a couple of entries by manual dot product.
        for &(token_id, score) in result.iter().take(3) {
            let row = weights.lm_head.row(token_id as usize);
            let manual: f32 = row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            assert!(
                (manual - score).abs() < 1e-4,
                "token {token_id}: manual {manual} vs reported {score}"
            );
        }
    }

    #[test]
    fn project_through_unembed_dim_mismatch_returns_empty() {
        let weights = shared_weights();
        assert!(project_through_unembed(weights, &[0.0; 1], 5).is_empty());
    }
}
