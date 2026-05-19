//! Integration tests for walker modules using the shared mock-model fixture.
//!
//! Covers the public API of `WeightWalker`, `AttentionWalker`, and
//! `VectorExtractor` as exercised through `larql_vindex::walker::*`. The
//! mock model is a Gemma3-style 2-layer/hidden=8 fixture built by
//! `larql_vindex::walker::test_fixture::create_mock_model`.

#![cfg(not(target_arch = "wasm32"))]
use larql_vindex::walker::test_fixture::create_mock_model;

// ── Weight Walker ────────────────────────────────────────────────────────

#[test]
fn weight_walker_loads() {
    let dir = std::env::temp_dir().join("larql_test_weight_load");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let walker =
        larql_vindex::walker::weight_walker::WeightWalker::load(dir.to_str().unwrap()).unwrap();
    assert_eq!(walker.num_layers(), 2);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn weight_walker_extracts_edges() {
    let dir = std::env::temp_dir().join("larql_test_weight_edges");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let walker =
        larql_vindex::walker::weight_walker::WeightWalker::load(dir.to_str().unwrap()).unwrap();
    let config = larql_vindex::walker::weight_walker::WalkConfig {
        top_k: 3,
        min_score: 0.0,
    };
    let mut graph = larql_core::Graph::new();
    let mut callbacks = larql_vindex::walker::weight_walker::SilentWalkCallbacks;

    let result = walker
        .walk_layer(0, &config, &mut graph, &mut callbacks)
        .unwrap();

    assert_eq!(result.layer, 0);
    assert_eq!(result.features_scanned, 4); // intermediate_size
    assert!(result.edges_found > 0);
    assert!(graph.edge_count() > 0);

    for edge in graph.edges() {
        assert_eq!(edge.source, larql_core::SourceType::Parametric);
        let meta = edge.metadata.as_ref().unwrap();
        assert!(meta.contains_key("layer"));
        assert!(meta.contains_key("feature"));
        assert!(meta.contains_key("c_in"));
        assert!(meta.contains_key("c_out"));
        assert!(meta.contains_key("selectivity"));
        assert!(edge.confidence >= 0.0);
        assert!(edge.confidence <= 1.0);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn weight_walker_walks_all_layers() {
    let dir = std::env::temp_dir().join("larql_test_weight_all");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let config = larql_vindex::walker::weight_walker::WalkConfig {
        top_k: 2,
        min_score: 0.0,
    };
    let mut graph = larql_core::Graph::new();
    let mut callbacks = larql_vindex::walker::weight_walker::SilentWalkCallbacks;

    let results = larql_vindex::walk_model(
        dir.to_str().unwrap(),
        None,
        &config,
        &mut graph,
        &mut callbacks,
    )
    .unwrap();

    assert_eq!(results.len(), 2);
    assert!(graph.edge_count() > 0);

    let count_layer = |layer: u64| {
        graph
            .edges()
            .iter()
            .filter(|e| {
                e.metadata
                    .as_ref()
                    .and_then(|m| m.get("layer"))
                    .and_then(|v| v.as_u64())
                    == Some(layer)
            })
            .count()
    };
    assert!(count_layer(0) > 0);
    assert!(count_layer(1) > 0);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn weight_walker_layer_stats_invariants() {
    let dir = std::env::temp_dir().join("larql_test_weight_stats");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let walker =
        larql_vindex::walker::weight_walker::WeightWalker::load(dir.to_str().unwrap()).unwrap();
    let config = larql_vindex::walker::weight_walker::WalkConfig {
        top_k: 3,
        min_score: 0.0,
    };
    let mut graph = larql_core::Graph::new();
    let mut callbacks = larql_vindex::walker::weight_walker::SilentWalkCallbacks;

    let result = walker
        .walk_layer(0, &config, &mut graph, &mut callbacks)
        .unwrap();

    let s = &result.stats;
    assert!(s.mean_confidence >= 0.0 && s.mean_confidence <= 1.0);
    assert!(s.max_confidence <= 1.0);
    assert!(s.min_confidence >= 0.0);
    assert!(s.max_confidence >= s.min_confidence);
    assert!(s.mean_selectivity >= 0.0);
    assert!(s.mean_c_in >= 0.0);
    assert!(s.mean_c_out >= 0.0);

    std::fs::remove_dir_all(&dir).ok();
}

// ── Attention Walker ─────────────────────────────────────────────────────

#[test]
fn attention_walker_loads() {
    let dir = std::env::temp_dir().join("larql_test_attn_load");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let walker =
        larql_vindex::walker::attention_walker::AttentionWalker::load(dir.to_str().unwrap())
            .unwrap();
    assert_eq!(walker.num_layers(), 2);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn attention_walker_extracts_edges() {
    let dir = std::env::temp_dir().join("larql_test_attn_edges");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let walker =
        larql_vindex::walker::attention_walker::AttentionWalker::load(dir.to_str().unwrap())
            .unwrap();
    let config = larql_vindex::walker::weight_walker::WalkConfig {
        top_k: 2,
        min_score: 0.0,
    };
    let mut graph = larql_core::Graph::new();
    let mut callbacks = larql_vindex::walker::weight_walker::SilentWalkCallbacks;

    let result = walker
        .walk_layer(0, &config, &mut graph, &mut callbacks)
        .unwrap();

    assert_eq!(result.layer, 0);
    assert_eq!(result.heads_walked, 2);
    assert!(result.edges_found > 0);

    for edge in graph.edges() {
        let meta = edge.metadata.as_ref().unwrap();
        assert!(meta.contains_key("layer"));
        assert!(meta.contains_key("head"));
        assert_eq!(meta["circuit"], "OV");
    }

    std::fs::remove_dir_all(&dir).ok();
}

// ── Vector Extractor ─────────────────────────────────────────────────────

#[test]
fn vector_extractor_ffn_down() {
    let dir = std::env::temp_dir().join("larql_test_vec_down");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let extractor =
        larql_vindex::walker::vector_extractor::VectorExtractor::load(dir.to_str().unwrap())
            .unwrap();
    assert_eq!(extractor.num_layers(), 2);
    assert_eq!(extractor.hidden_size(), 8);

    let output_dir = dir.join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    let config = larql_vindex::walker::vector_extractor::ExtractConfig {
        components: vec!["ffn_down".to_string()],
        layers: Some(vec![0]),
        top_k: 3,
    };
    let mut callbacks = larql_vindex::walker::vector_extractor::SilentExtractCallbacks;

    let summary = extractor
        .extract_all(&config, &output_dir, false, &mut callbacks)
        .unwrap();

    assert_eq!(summary.total_vectors, 4);
    assert_eq!(summary.components.len(), 1);
    assert_eq!(summary.components[0].component, "ffn_down");
    assert!(output_dir.join("ffn_down.vectors.jsonl").exists());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn vector_extractor_embeddings() {
    let dir = std::env::temp_dir().join("larql_test_vec_embed");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let extractor =
        larql_vindex::walker::vector_extractor::VectorExtractor::load(dir.to_str().unwrap())
            .unwrap();

    let output_dir = dir.join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    let config = larql_vindex::walker::vector_extractor::ExtractConfig {
        components: vec!["embeddings".to_string()],
        layers: None,
        top_k: 3,
    };
    let mut callbacks = larql_vindex::walker::vector_extractor::SilentExtractCallbacks;

    let summary = extractor
        .extract_all(&config, &output_dir, false, &mut callbacks)
        .unwrap();

    assert_eq!(summary.total_vectors, 16);

    std::fs::remove_dir_all(&dir).ok();
}
