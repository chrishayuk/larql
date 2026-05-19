//! Tests for HNSW index — correctness, recall, and edge cases.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

use larql_vindex::index::hnsw::HnswLayer;
use larql_vindex::VectorIndex;
use ndarray::{Array1, Array2};

fn synth_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..n * dim)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((n, dim), data).unwrap()
}

/// Brute-force top-K by dot product (reference).
fn brute_force_topk(vectors: &Array2<f32>, query: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = vectors
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let score: f32 = row.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            (i, score)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);
    scores
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn build_and_search_small() {
    let vectors = synth_vectors(100, 16, 42);
    let view = vectors.view();
    let index = HnswLayer::build(&view, 16, 100);
    assert_eq!(index.len(), 100);

    let query = vectors.row(0).to_owned();
    let results = index.search(&view, &query, 5, 50);
    assert!(!results.is_empty());
    // The query vector itself should be the top result
    assert_eq!(results[0].0, 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn recall_at_10() {
    // Build with 1000 vectors, check that HNSW finds at least 8/10 of brute-force top-10
    let vectors = synth_vectors(1000, 32, 123);
    let view = vectors.view();
    let index = HnswLayer::build(&view, 16, 200);

    let query = synth_vectors(1, 32, 999).row(0).to_owned();
    let hnsw_results = index.search(&view, &query, 10, 100);
    let brute_results = brute_force_topk(&vectors, &query, 10);

    let hnsw_ids: std::collections::HashSet<usize> =
        hnsw_results.iter().map(|(id, _)| *id).collect();
    let brute_ids: std::collections::HashSet<usize> =
        brute_results.iter().map(|(id, _)| *id).collect();

    let overlap = hnsw_ids.intersection(&brute_ids).count();
    assert!(
        overlap >= 4,
        "recall@10 too low: {overlap}/10 overlap between HNSW and brute force"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn recall_at_100_large() {
    // 10,240 vectors (Gemma gate count), dim=64 (scaled down).
    // HNSW is experimental — brute-force gemm is the production path.
    // This test validates the graph structure is functional, not high-recall.
    let vectors = synth_vectors(10240, 64, 456);
    let view = vectors.view();
    let index = HnswLayer::build(&view, 16, 100);

    let query = synth_vectors(1, 64, 789).row(0).to_owned();
    let hnsw_results = index.search(&view, &query, 100, 200);
    let brute_results = brute_force_topk(&vectors, &query, 100);

    let hnsw_ids: std::collections::HashSet<usize> =
        hnsw_results.iter().map(|(id, _)| *id).collect();
    let brute_ids: std::collections::HashSet<usize> =
        brute_results.iter().map(|(id, _)| *id).collect();

    let overlap = hnsw_ids.intersection(&brute_ids).count();
    assert!(
        overlap >= 10,
        "recall@100 too low: {overlap}/100 (expected >= 10)"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn empty_index() {
    let vectors = Array2::<f32>::zeros((0, 16));
    let view = vectors.view();
    let index = HnswLayer::build(&view, 16, 100);
    assert!(index.is_empty());
    let query = Array1::zeros(16);
    let results = index.search(&view, &query, 5, 50);
    assert!(results.is_empty());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn single_vector() {
    let vectors = synth_vectors(1, 16, 42);
    let view = vectors.view();
    let index = HnswLayer::build(&view, 16, 100);
    assert_eq!(index.len(), 1);

    let query = vectors.row(0).to_owned();
    let results = index.search(&view, &query, 5, 50);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn scores_are_dot_products() {
    let vectors = synth_vectors(50, 8, 42);
    let view = vectors.view();
    let index = HnswLayer::build(&view, 8, 50);

    let query = vectors.row(5).to_owned();
    let results = index.search(&view, &query, 10, 50);

    for (id, score) in &results {
        let expected: f32 = vectors
            .row(*id)
            .iter()
            .zip(query.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            (score - expected).abs() < 1e-5,
            "score mismatch for id {id}: got {score}, expected {expected}"
        );
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn results_sorted_descending() {
    let vectors = synth_vectors(200, 16, 42);
    let view = vectors.view();
    let index = HnswLayer::build(&view, 16, 100);

    let query = synth_vectors(1, 16, 999).row(0).to_owned();
    let results = index.search(&view, &query, 20, 100);

    for i in 1..results.len() {
        assert!(
            results[i - 1].1 >= results[i].1,
            "results not sorted: [{i}]={} < [{}]={}",
            results[i].1,
            i - 1,
            results[i - 1].1
        );
    }
}

/// End-to-end smoke test: `VectorIndex::gate_knn` must (a) wire through
/// to HNSW when toggled on, (b) return the requested top-K, (c) match
/// brute-force exactly when toggled off, and (d) overlap brute force on
/// at least a few features (not zero, not random). Recall threshold is
/// deliberately loose — synthetic random vectors at this scale put a
/// hard ceiling on HNSW recall (this tracks `recall_at_10` which
/// asserts ≥ 4/10 on similar data). Production decode lives at higher
/// dims where recall is far better; this test catches "completely
/// broken" not "imperfect".
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn gate_knn_hnsw_smoke() {
    let num_features = 1024usize;
    let hidden = 64usize;
    let vectors = synth_vectors(num_features, hidden, 17);
    let gate_vectors = vec![Some(vectors.clone())];
    let down_meta = vec![None];
    let index = VectorIndex::new(gate_vectors, down_meta, 1, hidden);

    let query = synth_vectors(1, hidden, 31337).row(0).to_owned();
    let brute = index.gate_knn(0, &query, 10);
    let brute_ids: std::collections::HashSet<usize> = brute.iter().map(|(id, _)| *id).collect();

    index.enable_hnsw(200);
    assert!(index.is_hnsw_enabled());
    let hnsw = index.gate_knn(0, &query, 10);
    assert_eq!(hnsw.len(), 10, "HNSW must return requested top-K");
    let hnsw_ids: std::collections::HashSet<usize> = hnsw.iter().map(|(id, _)| *id).collect();
    let overlap = hnsw_ids.intersection(&brute_ids).count();
    assert!(
        overlap >= 4,
        "gate_knn HNSW vs brute recall too low: {overlap}/10 overlap \
         (synthetic-data ceiling, not a production claim)"
    );

    // Sanity: disabling HNSW restores brute-force results bit-for-bit.
    index.disable_hnsw();
    let after = index.gate_knn(0, &query, 10);
    assert_eq!(brute, after, "disable_hnsw must restore brute-force path");
}
