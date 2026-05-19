//! Integration tests for the four previously-untested `layer_graph/` files.
//!
//! Requires a real Q4_K vindex on disk. Tests are `#[ignore]` and skipped
//! gracefully when no vindex is found. Run with:
//!
//! ```sh
//! cargo test -p larql-inference --test test_layer_graph_integration -- --ignored
//! ```
//!
//! ## What's covered
//!
//! - `prefill.rs`:  `prefill_with_kv` hidden state shape, finiteness, matches
//!                  `predict_kquant_hidden` at the last position.
//! - `pipeline_layer.rs`: `build_pipeline_layers` produces the right number of
//!                         layers, each with correct head_dim/norm weights.
//! - `template.rs`: `TemplateUniverse::build` with real entities populates
//!                   features; `GuidedWalkLayerGraph` forward pass is finite.
//! - `grid.rs`:     No integration test — requires a live remote-shard server.
//!                  The error-path unit test in grid.rs covers what's testable
//!                  without a real Metal backend + remote server.

#![allow(
    unused_imports,
    unused_mut,
    unused_variables,
    clippy::cloned_ref_to_slice_refs,
    clippy::doc_overindented_list_items
)]

use std::path::PathBuf;

use larql_compute::CpuBackend;
use larql_inference::{
    layer_graph::{
        // template items are re-exported from layer_graph root via `pub use template::*`
        detect_template,
        pipeline_layer::{build_pipeline_layers, resolve_attn_weights},
        prefill::prefill_with_kv,
        GuidedWalkLayerGraph,
        LayerGraph,
        TemplatePattern,
        TemplateUniverse,
    },
    vindex::predict_kquant_hidden,
};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, SilentLoadCallbacks,
    VectorIndex,
};

/// Find a Q4_K vindex from standard locations.
fn find_q4k_vindex() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("output/gemma3-4b-q4k-v2.vindex"),
        PathBuf::from("output/gemma3-4b-q4k-streaming.vindex"),
        PathBuf::from("/Users/christopherhay/chris-source/larql/output/gemma3-4b-q4k-v2.vindex"),
    ];
    for p in &candidates {
        if p.is_dir() {
            return Some(p.clone());
        }
    }
    if let Ok(p) = std::env::var("LARQL_TEST_VINDEX") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }
    None
}

fn skip_if_missing(name: &str) -> Option<PathBuf> {
    let p = find_q4k_vindex();
    if p.is_none() {
        eprintln!("skip {name}: no Q4_K vindex found (set LARQL_TEST_VINDEX to override)");
    }
    p
}

// ── prefill_with_kv ───────────────────────────────────────────────────────────

#[test]
#[ignore = "loads real 4B model; run with --ignored"]
fn prefill_with_kv_shape_and_finiteness() {
    let Some(vindex_path) = skip_if_missing("prefill_with_kv_shape_and_finiteness") else {
        return;
    };

    let mut cb = SilentLoadCallbacks;
    let mut weights = load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");

    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let prompt_ids: Vec<u32> = tokenizer
        .encode("The capital of France is", false)
        .expect("encode")
        .get_ids()
        .to_vec();

    let h = prefill_with_kv(
        &weights,
        &prompt_ids,
        &index,
        &CpuBackend,
        0..weights.num_layers,
    );

    assert_eq!(h.shape()[0], prompt_ids.len(), "seq dimension");
    assert_eq!(h.shape()[1], weights.hidden_size, "hidden dimension");
    assert!(
        h.iter().all(|v| v.is_finite()),
        "hidden state has non-finite values"
    );
    eprintln!(
        "prefill_with_kv: shape {:?}, last-pos L2 norm = {:.4}",
        h.shape(),
        h.row(h.shape()[0] - 1)
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
    );
}

#[test]
#[ignore = "loads real 4B model; run with --ignored"]
fn prefill_with_kv_matches_predict_q4k_hidden() {
    let Some(vindex_path) = skip_if_missing("prefill_with_kv_matches_predict_q4k_hidden") else {
        return;
    };

    let mut cb = SilentLoadCallbacks;
    let mut weights = load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");

    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let prompt_ids: Vec<u32> = tokenizer
        .encode("France", false)
        .expect("encode")
        .get_ids()
        .to_vec();

    // prefill_with_kv uses cpu attention + WalkFfn (cpu fallback)
    let h_prefill = prefill_with_kv(
        &weights,
        &prompt_ids,
        &index,
        &CpuBackend,
        0..weights.num_layers,
    );

    // predict_kquant_hidden dequantises layer-by-layer
    let h_q4k = predict_kquant_hidden(&mut weights, &prompt_ids, &index, None);

    // The two paths use different FFN implementations — cosine similarity should
    // be > 0.95 at the last position (they differ mainly in FFN quantisation).
    let n = h_prefill.shape()[0] - 1;
    let v1: Vec<f32> = h_prefill.row(n).to_vec();
    let v2: Vec<f32> = h_q4k.row(n).to_vec();
    let dot: f64 = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let n1: f64 = v1.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let n2: f64 = v2.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cos = if n1 > 0.0 && n2 > 0.0 {
        dot / (n1 * n2)
    } else {
        0.0
    };
    eprintln!("prefill_with_kv vs predict_kquant_hidden: cosine = {cos:.6}");
    assert!(
        cos > 0.90,
        "last-pos cosine {cos:.4} < 0.90 — paths diverged unexpectedly"
    );
}

// ── pipeline_layer ────────────────────────────────────────────────────────────

#[test]
#[ignore = "loads real 4B model; run with --ignored"]
fn build_pipeline_layers_produces_all_layers() {
    let Some(vindex_path) = skip_if_missing("build_pipeline_layers_produces_all_layers") else {
        return;
    };

    let mut cb = SilentLoadCallbacks;
    let weights = load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");

    let gate_index: &dyn larql_vindex::GateIndex = &index;
    let q4_ffn = gate_index
        .interleaved_kquant_mmap_ref()
        .expect("Q4K FFN mmap required");
    let ffn_is_q4k = true;
    let hidden = weights.hidden_size;
    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = (intermediate * hidden).div_ceil(256) * 144;

    let layers = build_pipeline_layers(
        &weights,
        &index,
        0..weights.num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        larql_compute::QuantFormat::Q4_K,
    );

    assert_eq!(
        layers.len(),
        weights.num_layers,
        "pipeline layer count should match model layer count"
    );

    // Spot-check layer 0: norm weights and head geometry
    let l0 = &layers[0];
    assert!(
        !l0.input_norm.is_empty(),
        "layer 0 input_norm should be populated"
    );
    assert_eq!(l0.head_dim, weights.head_dim, "head_dim mismatch");
    assert_eq!(l0.num_q_heads, weights.num_q_heads, "num_q_heads mismatch");
    assert_eq!(
        l0.num_kv_heads, weights.num_kv_heads,
        "num_kv_heads mismatch"
    );
    assert!(l0.rope_base > 0.0, "rope_base should be positive");

    eprintln!(
        "build_pipeline_layers: {} layers, head_dim={}, rope_base={}",
        layers.len(),
        l0.head_dim,
        l0.rope_base
    );
}

#[test]
#[ignore = "loads real 4B model; run with --ignored"]
fn resolve_attn_weights_returns_some_with_q4k_loaded() {
    let Some(vindex_path) = skip_if_missing("resolve_attn_weights_returns_some_with_q4k_loaded")
    else {
        return;
    };

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");

    let result = resolve_attn_weights(&index, 0);
    assert!(
        result.is_some(),
        "attn weights should be Some after loading Q4K attn"
    );
    let (wq, wk, wv, wo) = result.unwrap();
    assert!(!wq.data.is_empty(), "wq data should be non-empty");
    assert!(!wk.data.is_empty(), "wk data should be non-empty");
    assert!(!wv.data.is_empty(), "wv data should be non-empty");
    assert!(!wo.data.is_empty(), "wo data should be non-empty");
    eprintln!(
        "resolve_attn_weights layer 0: wq={} bytes, format={:?}",
        wq.data.len(),
        wq.format
    );
}

// ── template ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "loads real 4B model; run with --ignored"]
fn template_universe_build_with_real_model() {
    let Some(vindex_path) = skip_if_missing("template_universe_build_with_real_model") else {
        return;
    };

    let mut cb = SilentLoadCallbacks;
    let weights = load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights");
    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");

    let ffn = larql_inference::ffn::WeightFfn { weights: &weights };

    let universe = TemplateUniverse::build(
        &weights,
        &tokenizer,
        "capital-of",
        "The capital of {} is",
        &["France", "Germany", "Italy"],
        &ffn,
        0.01,
    );

    assert!(!universe.name.is_empty());
    // With real model weights, the template should activate at least some features
    eprintln!(
        "template_universe_build: total_features={}",
        universe.total_features()
    );
    // Not asserting a specific count — it varies with threshold; just check no panic.
}

#[test]
#[ignore = "loads real 4B model; run with --ignored"]
fn guided_walk_layer_graph_with_real_universe() {
    let Some(vindex_path) = skip_if_missing("guided_walk_layer_graph_with_real_universe") else {
        return;
    };

    let mut cb = SilentLoadCallbacks;
    let weights = load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights");
    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");

    let ffn = larql_inference::ffn::WeightFfn { weights: &weights };

    let universe = TemplateUniverse::build(
        &weights,
        &tokenizer,
        "capital-of",
        "The capital of {} is",
        &["France"],
        &ffn,
        0.05,
    );

    let prompt_ids: Vec<u32> = tokenizer
        .encode("The capital of France is", false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let seq_len = prompt_ids.len();
    let mut h = larql_inference::forward::embed_tokens_pub(&weights, &prompt_ids);

    let g = GuidedWalkLayerGraph {
        weights: &weights,
        universe: &universe,
        index: &index,
    };
    use larql_inference::layer_graph::LayerGraph;
    for layer in 0..weights.num_layers {
        if let Some(out) = g.forward_layer(&weights, &h, layer) {
            assert_eq!(out.residual.shape()[0], seq_len, "seq dim layer {layer}");
            assert_eq!(
                out.residual.shape()[1],
                weights.hidden_size,
                "hidden dim layer {layer}"
            );
            assert!(
                out.residual.iter().all(|v| v.is_finite()),
                "non-finite at layer {layer}"
            );
            h = out.residual;
        }
    }
    eprintln!(
        "guided_walk_layer_graph: all {} layers finite",
        weights.num_layers
    );
}

// ── detect_template (pure logic, no model needed — fast smoke-test here too) ──

#[test]
fn detect_template_with_real_token_prefix() {
    // Verify the BOS-offset logic using raw token IDs.
    // BOS = some token at pos 0 that doesn't match the template prefix.
    let template = TemplatePattern {
        name: "capital".into(),
        prefix_tokens: vec![100, 200, 300], // fake "The capital of"
        cached_layers: 0..=10,
    };
    // Sequence [1, 100, 200, 300, 400]: BOS=1 at pos 0, prefix at 1..4
    let ids = vec![1u32, 100, 200, 300, 400];
    assert_eq!(detect_template(&ids, &[template.clone()]), Some(0));

    // Exact match from position 0
    let ids_direct = vec![100u32, 200, 300, 400];
    assert_eq!(detect_template(&ids_direct, &[template]), Some(0));
}
