//! LM-head top-K helpers and constrained-decode token sampling.

use crate::model::ModelWeights;
use larql_compute::prelude::*;

const ENV_LM_HEAD_SKIP_Q4K: &str = "LARQL_LM_HEAD_SKIP_Q4K";

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LmHeadPolicy {
    pub skip_q4k: bool,
}

impl LmHeadPolicy {
    pub(crate) fn from_env() -> Self {
        Self {
            skip_q4k: env_bool(ENV_LM_HEAD_SKIP_Q4K),
        }
    }
}

fn env_bool(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("on") | Ok("yes")
    )
}

/// Top-K logits lookup that transparently handles models with tied
/// input/output embeddings (Gemma 2/3/4) whose vindex has no dedicated
/// `lm_head.bin` / `lm_head_q4.bin`.
///
/// Resolution order:
/// 1. Vindex-native KNN (`lm_head_knn_backend`) — fastest, used when the
///    vindex was built with a separate lm_head.
/// 2. CPU gemv against `weights.lm_head` — the loader fills this from
///    `embed.clone()` for tied-embedding models, so it's always populated
///    even when no lm_head file is present.
///
/// The second path is O(vocab * hidden) floats through the CPU, but that's
/// a one-shot matvec per generated token — negligible compared to the
/// per-layer attention + FFN. It lets every model generate tokens through
/// the Metal pipeline regardless of how its vindex was packaged.
pub fn lm_head_topk(
    index: &larql_vindex::VectorIndex,
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
) -> Vec<(u32, f32)> {
    lm_head_topk_with_policy(
        index,
        weights,
        query,
        top_k,
        backend,
        &LmHeadPolicy::from_env(),
    )
}

pub(crate) fn lm_head_topk_with_policy(
    index: &larql_vindex::VectorIndex,
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
    policy: &LmHeadPolicy,
) -> Vec<(u32, f32)> {
    // Default route: `lm_head_knn_backend` — Metal `q4k_matvec` first
    // (1.85 ms/tok on Gemma 3 4B, was 2.95 ms via the stride-32 workaround
    // before the 2026-05-02 dispatch-geometry fix), f16 GEMV fallback for
    // vindexes lacking Q4_K lm_head bytes, f32 BLAS as last resort.
    //
    // `LARQL_LM_HEAD_SKIP_Q4K=1` routes through `_skip_q4k` instead
    // (stride-32 Q4_K → f16 → f32) for diagnostic A/B against the Q4_K
    // path. See `crates/larql-compute/PERFORMANCE.md` "Decision: lm_head
    // dispatch order" for the full root-cause history.
    if policy.skip_q4k && backend.supports(Capability::F32Gemv) {
        // Diagnostic path: skip the Q4_K accelerated matvec and use stride-32
        // Q4_K (or f16 GEMV / f32 BLAS) instead. Useful for verifying
        // top-1 stability against a known-stable reduction tree, or for
        // vindexes where the Q4_K lm_head bytes aren't populated.
        let hits = index.lm_head_knn_backend_skip_q4k(query, top_k, backend);
        let all_zero = !hits.is_empty() && hits.iter().all(|(_, s)| *s == 0.0 || s.is_nan());
        if !hits.is_empty() && !all_zero {
            return hits;
        }
        return backend_lm_head_topk(weights, query, top_k, backend);
    }
    let hits = index.lm_head_knn_backend(query, top_k, backend);
    // Workaround for the prefill→decode boundary: on the first decode
    // step, the Metal `q4k_matvec` / `f16_gemv` for lm_head occasionally
    // returns an all-zeros score vector even though the query is healthy
    // (verified rms ≈ 4, max_abs ≈ 60). The underlying cause appears to
    // be a GPU command-buffer ordering edge case after the first
    // `decode_token_with_moe` call. Falling back to the CPU/backend
    // gemv path produces correct scores immediately.
    let all_zero = !hits.is_empty() && hits.iter().all(|(_, s)| *s == 0.0 || s.is_nan());
    if !hits.is_empty() && !all_zero {
        return hits;
    }
    backend_lm_head_topk(weights, query, top_k, backend)
}

/// LM-head top-K via the active ComputeBackend.
///
/// Performs a single gemv `scores[vocab] = lm_head[vocab, hidden] · query[hidden]`
/// by dispatching `matmul_transb(query[1, hidden], lm_head[vocab, hidden])`.
/// On Metal this is a GPU f32 gemv (under Apple Silicon unified memory the
/// 2.68 GB `weights.lm_head` is shared, not copied). On CPU it's the
/// BLAS fallback via the same trait method. Either way this replaces the
/// previous unconditional CPU `ndarray::dot`, which was ~26 ms/tok on
/// Gemma 3 4B — the dominant cost of real-vindex decode.
pub(super) fn backend_lm_head_topk(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
    backend: &dyn ComputeBackend,
) -> Vec<(u32, f32)> {
    let lm = &weights.lm_head;
    if lm.is_empty() || query.is_empty() {
        return Vec::new();
    }
    let vocab = lm.shape()[0];
    let hidden = lm.shape()[1];
    if hidden != query.len() {
        return Vec::new();
    }

    let query_slice = match query.as_slice() {
        Some(s) => s,
        None => &query.to_vec(),
    };

    // Fast path for top-1 (greedy decode): GPU gemv + GPU argmax
    // reads back only 8 KB partial results instead of 1 MB, saving ~0.33ms.
    if top_k == 1 {
        if let Some((idx, score)) = backend.f32_gemv_topk1(lm.view(), query_slice) {
            return vec![(idx, score)];
        }
    }

    // General path: GPU gemv → full Vec<f32> → CPU top-k.
    let scores_vec: Vec<f32> = if let Some(s) = backend.f32_gemv(lm.view(), query_slice) {
        s
    } else {
        let q_row = match query.view().into_shape_with_order((1, hidden)) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        backend.matmul_transb(q_row, lm.view()).row(0).to_vec()
    };

    // Fast path for greedy decode (top_k=1): a single linear scan avoids
    // allocating the full 262K×8=2MB indexed Vec and the select_nth pass.
    if top_k == 1 {
        let best = scores_vec
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, s)| s.is_finite())
            .fold(None::<(usize, f32)>, |acc, (i, s)| {
                Some(match acc {
                    None => (i, s),
                    Some((bi, bs)) => {
                        if s > bs {
                            (i, s)
                        } else {
                            (bi, bs)
                        }
                    }
                })
            });
        let _ = vocab;
        return match best {
            Some((i, s)) => vec![(i as u32, s)],
            None => vec![],
        };
    }

    // Min-heap of size k: O(k) space, O(N log k) time.
    // Avoids allocating the full 262K×8=2MB indexed Vec.
    let k = top_k.min(vocab);
    if k == 0 {
        // top_k=0 means "no results requested" — return empty without
        // touching the heap (else `heap[0]` indexes out of bounds in
        // the per-score loop).
        return Vec::new();
    }
    let _ = vocab;
    let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);

    // sift-down to maintain min-heap property (smallest score at index 0).
    fn sift_down(h: &mut [(f32, u32)], mut i: usize) {
        let n = h.len();
        loop {
            let mut smallest = i;
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            if l < n && h[l].0 < h[smallest].0 {
                smallest = l;
            }
            if r < n && h[r].0 < h[smallest].0 {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            h.swap(i, smallest);
            i = smallest;
        }
    }

    for (i, &s) in scores_vec.iter().enumerate() {
        if !s.is_finite() {
            continue;
        }
        if heap.len() < k {
            heap.push((s, i as u32));
            if heap.len() == k {
                // Build min-heap in O(k)
                for j in (0..k / 2).rev() {
                    sift_down(&mut heap, j);
                }
            }
        } else if s > heap[0].0 {
            heap[0] = (s, i as u32);
            sift_down(&mut heap, 0);
        }
    }
    // If we gathered fewer than k finite values, still heapify.
    if heap.len() < k && heap.len() > 1 {
        for j in (0..heap.len() / 2).rev() {
            sift_down(&mut heap, j);
        }
    }

    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    heap.into_iter().map(|(s, i)| (i, s)).collect()
}

/// Kept for the `LARQL_METAL_COMPARE_CPU=1` diagnostic mode which wants a
/// known-good CPU reference. Not used in the hot path.
#[allow(dead_code)]
pub(super) fn cpu_lm_head_topk(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    top_k: usize,
) -> Vec<(u32, f32)> {
    backend_lm_head_topk(weights, query, top_k, &larql_compute::CpuBackend)
}

/// Dense LM-head: full `Vec<f32>` of vocabulary scores. Required for
/// constrained decoding — the sparse vindex KNN can't apply an arbitrary
/// vocabulary mask because masked-out tokens might fall outside the top-K.
/// Same compute kernel as [`backend_lm_head_topk`], just no truncation.
pub(super) fn backend_lm_head_scores(
    weights: &ModelWeights,
    query: &ndarray::Array1<f32>,
    backend: &dyn ComputeBackend,
) -> Vec<f32> {
    let lm = &weights.lm_head;
    if lm.is_empty() || query.is_empty() {
        return Vec::new();
    }
    let hidden = lm.shape()[1];
    if hidden != query.len() {
        return Vec::new();
    }
    let query_slice = match query.as_slice() {
        Some(s) => s,
        None => &query.to_vec(),
    };
    if let Some(s) = backend.f32_gemv(lm.view(), query_slice) {
        s
    } else {
        let q_row = match query.view().into_shape_with_order((1, hidden)) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        backend.matmul_transb(q_row, lm.view()).row(0).to_vec()
    }
}

/// Sampling-under-mask variant. Runs the dense LM head, applies the
/// mask, then defers token selection to the caller-supplied
/// [`Sampler`]. Repetition penalties on the sampler are applied as
/// usual via the `generated` history.
///
/// Returns `(id, raw_post_mask_score)` so callers that record per-token
/// probability still get the masked logit for the picked id (even
/// though the multinomial draw used the softmaxed distribution).
pub(super) fn pick_next_token_masked_sampled<M>(
    weights: &ModelWeights,
    h_1d: &ndarray::Array1<f32>,
    generated: &[u32],
    backend: &dyn ComputeBackend,
    mask_fn: &mut M,
    sampler: &mut super::sampling::Sampler,
) -> Option<(u32, f32)>
where
    M: FnMut(&[u32], &mut Vec<f32>),
{
    let mut logits = backend_lm_head_scores(weights, h_1d, backend);
    if logits.is_empty() {
        return None;
    }
    mask_fn(generated, &mut logits);
    let id = sampler.sample_with_history(&logits, generated)?;
    let score = *logits.get(id as usize)?;
    Some((id, score))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::test_utils::TestFixtures;
    use ndarray::Array1;

    fn fx() -> TestFixtures {
        TestFixtures::build()
    }

    #[test]
    fn env_bool_recognises_truthy_values() {
        std::env::remove_var("LARQL_TEST_LMHEAD_ENV_BOOL");
        assert!(!env_bool("LARQL_TEST_LMHEAD_ENV_BOOL"));
        for &v in &["1", "true", "on", "yes"] {
            std::env::set_var("LARQL_TEST_LMHEAD_ENV_BOOL", v);
            assert!(
                env_bool("LARQL_TEST_LMHEAD_ENV_BOOL"),
                "value {v:?} should be truthy"
            );
        }
        // Falsy: anything else.
        std::env::set_var("LARQL_TEST_LMHEAD_ENV_BOOL", "no");
        assert!(!env_bool("LARQL_TEST_LMHEAD_ENV_BOOL"));
        std::env::remove_var("LARQL_TEST_LMHEAD_ENV_BOOL");
    }

    #[test]
    fn lm_head_policy_default_is_off() {
        let p = LmHeadPolicy::default();
        assert!(!p.skip_q4k);
    }

    #[test]
    fn backend_lm_head_topk_returns_at_most_top_k() {
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.1);
        let hits = backend_lm_head_topk(&f.weights, &q, 5, &larql_compute::CpuBackend);
        assert!(hits.len() <= 5);
        assert!(hits.iter().all(|(_, s)| s.is_finite()));
    }

    #[test]
    fn backend_lm_head_topk_handles_empty_lm_head() {
        // Force an empty lm_head.
        let mut f = fx();
        f.weights.lm_head = ndarray::Array2::<f32>::zeros((0, f.weights.hidden_size)).into_shared();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.1);
        let hits = backend_lm_head_topk(&f.weights, &q, 5, &larql_compute::CpuBackend);
        assert!(hits.is_empty());
    }

    #[test]
    fn backend_lm_head_topk_handles_empty_query() {
        let f = fx();
        let q = Array1::<f32>::zeros(0);
        let hits = backend_lm_head_topk(&f.weights, &q, 5, &larql_compute::CpuBackend);
        assert!(hits.is_empty());
    }

    #[test]
    fn backend_lm_head_topk_handles_dim_mismatch() {
        let f = fx();
        let q = Array1::<f32>::zeros(f.weights.hidden_size + 1);
        let hits = backend_lm_head_topk(&f.weights, &q, 5, &larql_compute::CpuBackend);
        assert!(hits.is_empty());
    }

    #[test]
    fn cpu_lm_head_topk_matches_backend_with_cpu() {
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.05);
        let cpu = cpu_lm_head_topk(&f.weights, &q, 4);
        let direct = backend_lm_head_topk(&f.weights, &q, 4, &larql_compute::CpuBackend);
        assert_eq!(cpu.len(), direct.len());
        for ((a_t, a_s), (b_t, b_s)) in cpu.iter().zip(direct.iter()) {
            assert_eq!(a_t, b_t);
            assert!((a_s - b_s).abs() < 1e-5);
        }
    }

    #[test]
    fn backend_lm_head_scores_returns_full_vocab_vector() {
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.1);
        let scores = backend_lm_head_scores(&f.weights, &q, &larql_compute::CpuBackend);
        assert_eq!(scores.len(), f.weights.vocab_size);
        assert!(scores.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn backend_lm_head_scores_handles_dim_mismatch() {
        let f = fx();
        let q = Array1::<f32>::zeros(f.weights.hidden_size + 1);
        let scores = backend_lm_head_scores(&f.weights, &q, &larql_compute::CpuBackend);
        assert!(scores.is_empty());
    }

    #[test]
    fn backend_lm_head_scores_handles_empty_lm_head() {
        let mut f = fx();
        f.weights.lm_head = ndarray::Array2::<f32>::zeros((0, f.weights.hidden_size)).into_shared();
        let q = Array1::<f32>::zeros(f.weights.hidden_size);
        assert!(backend_lm_head_scores(&f.weights, &q, &larql_compute::CpuBackend).is_empty());
    }

    #[test]
    fn pick_next_token_masked_sampled_returns_id_and_score() {
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.05);
        let mut sampler =
            super::super::sampling::Sampler::new(super::super::sampling::SamplingConfig::greedy());
        let mut mask = |_generated: &[u32], _logits: &mut Vec<f32>| {
            // No-op mask
        };
        let pick = pick_next_token_masked_sampled(
            &f.weights,
            &q,
            &[],
            &larql_compute::CpuBackend,
            &mut mask,
            &mut sampler,
        );
        let (id, score) = pick.expect("greedy pick on full vocab must succeed");
        assert!((id as usize) < f.weights.vocab_size);
        assert!(score.is_finite());
    }

    #[test]
    fn pick_next_token_masked_sampled_returns_none_when_lm_head_empty() {
        let mut f = fx();
        f.weights.lm_head = ndarray::Array2::<f32>::zeros((0, f.weights.hidden_size)).into_shared();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.0);
        let mut sampler =
            super::super::sampling::Sampler::new(super::super::sampling::SamplingConfig::greedy());
        let mut mask = |_g: &[u32], _l: &mut Vec<f32>| {};
        assert!(pick_next_token_masked_sampled(
            &f.weights,
            &q,
            &[],
            &larql_compute::CpuBackend,
            &mut mask,
            &mut sampler,
        )
        .is_none());
    }

    #[test]
    fn lm_head_topk_uses_policy_from_env_default() {
        // Public `lm_head_topk` consults LmHeadPolicy::from_env() — when
        // the env var is unset the default skip_q4k=false routes through
        // the standard path. Synthetic vindex has no lm_head data, so
        // `lm_head_knn_backend` returns empty hits → fallback to
        // `backend_lm_head_topk` against weights.lm_head.
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.05);
        let hits = lm_head_topk(&f.index, &f.weights, &q, 4, &larql_compute::CpuBackend);
        // Either populated by backend_lm_head_topk fallback or empty (both valid).
        assert!(hits.len() <= 4);
        for (id, _) in &hits {
            assert!((*id as usize) < f.weights.vocab_size);
        }
    }

    #[test]
    fn lm_head_topk_with_policy_skip_q4k_routes_through_diagnostic_path() {
        // skip_q4k=true + a backend that supports F32Gemv (CpuBackend
        // does) routes through the `lm_head_knn_backend_skip_q4k` arm
        // and falls back to backend_lm_head_topk on empty hits.
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.05);
        let policy = LmHeadPolicy { skip_q4k: true };
        let hits = lm_head_topk_with_policy(
            &f.index,
            &f.weights,
            &q,
            4,
            &larql_compute::CpuBackend,
            &policy,
        );
        assert!(hits.len() <= 4);
    }

    #[test]
    fn backend_lm_head_topk_top_k_1_fast_path_returns_argmax() {
        // top_k=1 routes through the f32_gemv_topk1 fast path on backends
        // that support it; CpuBackend may or may not — falls through to
        // the linear scan otherwise. Either way the result is a single
        // top-scoring (id, score) pair.
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.05);
        let hits = backend_lm_head_topk(&f.weights, &q, 1, &larql_compute::CpuBackend);
        assert_eq!(hits.len(), 1);
        let (id, score) = hits[0];
        assert!((id as usize) < f.weights.vocab_size);
        assert!(score.is_finite());
    }

    #[test]
    fn backend_lm_head_topk_top_k_zero_returns_empty() {
        // Regression: top_k=0 used to panic with `index out of bounds: 0`
        // in the heap path because the per-score loop accessed `heap[0]`
        // before checking `k > 0`. Fixed by early-return on `k == 0`.
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.1);
        let hits = backend_lm_head_topk(&f.weights, &q, 0, &larql_compute::CpuBackend);
        assert!(hits.is_empty());
    }

    #[test]
    fn pick_next_token_masked_sampled_invokes_mask_fn() {
        let f = fx();
        let q = Array1::<f32>::from_elem(f.weights.hidden_size, 0.05);
        let mut sampler =
            super::super::sampling::Sampler::new(super::super::sampling::SamplingConfig::greedy());
        // Mask all but token 7 to NEG_INFINITY → greedy pick must be 7.
        let target_id = 7u32;
        let mut mask = |_g: &[u32], logits: &mut Vec<f32>| {
            for (i, l) in logits.iter_mut().enumerate() {
                if i as u32 != target_id {
                    *l = f32::NEG_INFINITY;
                }
            }
        };
        let (id, _) = pick_next_token_masked_sampled(
            &f.weights,
            &q,
            &[],
            &larql_compute::CpuBackend,
            &mut mask,
            &mut sampler,
        )
        .expect("greedy pick under tight mask");
        assert_eq!(id, target_id);
    }
}
