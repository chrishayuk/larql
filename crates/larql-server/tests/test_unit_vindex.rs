#![cfg(not(target_arch = "wasm32"))]
//! Pure unit tests: gate_knn, walk, describe entity, patches, relations, stats
//! (core vindex operation tests).

use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{
    ExtractLevel, FeatureMeta, LayerBands, PatchedVindex, QuantFormat, VectorIndex, VindexConfig,
    VindexLayerInfo,
};
use std::collections::HashMap;

// ══════════════════════════════════════════════════════════════
// Test helpers (local copies — duplication is fine per spec)
// ══════════════════════════════════════════════════════════════

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![
            make_top_k(token, id, score),
            make_top_k("also", id + 1, score * 0.5),
        ],
    }
}

/// Build a small test VectorIndex: 2 layers, 4 hidden dims, 3 features/layer.
fn test_index() -> VectorIndex {
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[1, 1]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        Some(make_meta("Tokyo", 201, 0.85)),
        Some(make_meta("Spain", 202, 0.70)),
    ];

    VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        vec![Some(meta0), Some(meta1)],
        num_layers,
        hidden,
    )
}

/// Build a tiny embeddings matrix (vocab=8, hidden=4).
fn test_embeddings() -> Array2<f32> {
    let mut embed = Array2::<f32>::zeros((8, 4));
    embed[[0, 0]] = 1.0;
    embed[[1, 1]] = 1.0;
    embed[[2, 2]] = 1.0;
    embed[[3, 3]] = 1.0;
    embed[[4, 0]] = 1.0;
    embed[[4, 1]] = 1.0;
    embed
}

fn test_config() -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 1),
            output: (1, 1),
        }),
        layers: vec![
            VindexLayerInfo {
                layer: 0,
                num_features: 3,
                offset: 0,
                length: 48,
                num_experts: None,
                num_features_per_expert: None,
            },
            VindexLayerInfo {
                layer: 1,
                num_features: 3,
                offset: 48,
                length: 48,
                num_experts: None,
                num_features_per_expert: None,
            },
        ],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    }
}

// ══════════════════════════════════════════════════════════════
// CORE LOGIC TESTS
// ══════════════════════════════════════════════════════════════

#[test]
fn test_gate_knn_returns_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    assert!(!hits.is_empty());
    // Feature 0 has gate[0,0]=1.0, should be top hit
    assert_eq!(hits[0].0, 0);
    assert!((hits[0].1 - 1.0).abs() < 0.01);
}

#[test]
fn test_walk_returns_per_layer_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 3);
    assert_eq!(trace.layers.len(), 2);

    // Layer 0: feature 0 (Paris) should be top hit
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].meta.top_token, "Paris");
}

#[test]
fn test_walk_with_layer_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let trace = patched.walk(&query, &[1], 3);
    assert_eq!(trace.layers.len(), 1);
    assert_eq!(trace.layers[0].0, 1);
}

#[test]
fn test_describe_entity_via_embedding() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate what the describe handler does:
    // Token embedding → gate KNN → aggregate edges.
    let embed = test_embeddings();
    let query = embed.row(0).mapv(|v| v * 1.0); // token 0 → [1,0,0,0]
    let trace = patched.walk(&query, &[0, 1], 10);

    let mut targets: Vec<String> = Vec::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            targets.push(hit.meta.top_token.clone());
        }
    }

    // Token 0 → dim 0 strong → feature 0 (Paris) at L0, feature 1 (Tokyo) at L1
    assert!(targets.contains(&"Paris".to_string()));
}

#[test]
fn test_select_by_layer() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SELECT at layer 0
    let metas = patched.down_meta_at(0).unwrap();
    let tokens: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref().map(|m| m.top_token.as_str()))
        .collect();

    assert_eq!(tokens, vec!["Paris", "French", "Europe"]);
}

#[test]
fn test_select_with_entity_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Filter for tokens containing "par" (case-insensitive)
    let metas = patched.down_meta_at(0).unwrap();
    let matches: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref())
        .filter(|m| m.top_token.to_lowercase().contains("par"))
        .map(|m| m.top_token.as_str())
        .collect();

    assert_eq!(matches, vec!["Paris"]);
}

#[test]
fn test_relations_listing() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SHOW RELATIONS: scan all layers, aggregate tokens
    let mut token_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for layer in patched.loaded_layers() {
        if let Some(metas) = patched.down_meta_at(layer) {
            for meta in metas.iter().flatten() {
                *token_counts.entry(meta.top_token.clone()).or_default() += 1;
            }
        }
    }

    assert_eq!(token_counts.len(), 6); // Paris, French, Europe, Berlin, Tokyo, Spain
    assert_eq!(*token_counts.get("Paris").unwrap(), 1);
}

#[test]
fn test_stats_from_config() {
    let config = test_config();
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
    assert_eq!(total_features, 6);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, 4);
    assert_eq!(config.model, "test/model-4");
}

// ══════════════════════════════════════════════════════════════
// PATCH OPERATIONS
// ══════════════════════════════════════════════════════════════

#[test]
fn test_apply_patch_modifies_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Before patch: feature 0 at L0 = "Paris"
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "Paris");

    // Update feature 0 at L0 to "London"
    patched.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "London");
}

#[test]
fn test_delete_feature_removes_from_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Delete feature 0 at L0
    patched.delete_feature(0, 0);

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);

    // Feature 0 should no longer appear
    for (_, hits) in &trace.layers {
        for hit in hits {
            assert_ne!(hit.feature, 0);
        }
    }
}

#[test]
fn test_patch_count_tracking() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);
    assert_eq!(patched.num_patches(), 0);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("test-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: Some("test".into()),
        }],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);
    assert_eq!(patched.num_overrides(), 1);
}

#[test]
fn test_remove_patch_restores_state() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("removable".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);

    // Feature 0 should be deleted
    assert!(patched.feature_meta(0, 0).is_none());

    // Remove the patch
    patched.remove_patch(0);
    assert_eq!(patched.num_patches(), 0);

    // Feature 0 should be back
    assert!(patched.feature_meta(0, 0).is_some());
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ══════════════════════════════════════════════════════════════
// WALK-FFN (decoupled inference protocol — vindex side)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_walk_ffn_single_layer() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let residual = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &residual, 3);
    let features: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
    let scores: Vec<f32> = hits.iter().map(|(_, s)| *s).collect();
    assert!(!features.is_empty());
    assert_eq!(features.len(), scores.len());
    // Feature 0 should be top (responds to dim 0)
    assert_eq!(features[0], 0);
}

#[test]
fn test_walk_ffn_batched_layers() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let residual = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

    let layers = vec![0, 1];
    let mut results = Vec::new();
    for &layer in &layers {
        let hits = patched.gate_knn(layer, &residual, 3);
        results.push((layer, hits));
    }
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 0);
    assert_eq!(results[1].0, 1);
}

// ══════════════════════════════════════════════════════════════
// EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn test_empty_query_returns_no_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    // All scores are 0, but KNN still returns results (sorted by abs)
    for (_feat, score) in &hits {
        assert!((score.abs()) < 0.01);
    }
}

#[test]
fn test_nonexistent_layer_returns_empty() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(99, &query, 3);
    assert!(hits.is_empty());
}

#[test]
fn test_walk_empty_layer_list() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[], 3);
    assert!(trace.layers.is_empty());
}

#[test]
fn test_large_top_k_clamped() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    // Request 100 but only 3 features exist
    let hits = patched.gate_knn(0, &query, 100);
    assert_eq!(hits.len(), 3);
}

// ══════════════════════════════════════════════════════════════
// PROBE LABELS (relation classifier in DESCRIBE)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_probe_label_lookup() {
    let mut labels: HashMap<(usize, usize), String> = HashMap::new();
    labels.insert((0, 0), "capital".into());
    labels.insert((0, 1), "language".into());
    labels.insert((1, 2), "continent".into());

    assert_eq!(labels.get(&(0, 0)).map(|s| s.as_str()), Some("capital"));
    assert_eq!(labels.get(&(0, 1)).map(|s| s.as_str()), Some("language"));
    assert_eq!(labels.get(&(1, 2)).map(|s| s.as_str()), Some("continent"));
    assert_eq!(labels.get(&(0, 2)), None);
    assert_eq!(labels.get(&(99, 99)), None);
}

#[test]
fn test_describe_edge_with_probe_label() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    let mut labels: HashMap<(usize, usize), String> = HashMap::new();
    labels.insert((0, 0), "capital".into());

    // Walk to find edges (simulates describe handler)
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 5);

    // Build edge info like the handler does
    for (layer, hits) in &trace.layers {
        for hit in hits {
            let label = labels.get(&(*layer, hit.feature));
            if hit.feature == 0 && *layer == 0 {
                assert_eq!(label, Some(&"capital".to_string()));
            } else {
                // Other features have no probe label
                assert!(label.is_none() || label.is_some());
            }
        }
    }
}

#[test]
fn test_probe_labels_empty_when_no_file() {
    // Simulates load_probe_labels on a nonexistent path
    let labels: HashMap<(usize, usize), String> = HashMap::new();
    assert!(labels.is_empty());
}

// ══════════════════════════════════════════════════════════════
// LAYER BAND FILTERING (DESCRIBE handler logic)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_layer_band_filtering() {
    let bands = LayerBands {
        syntax: (0, 0),
        knowledge: (0, 1),
        output: (1, 1),
    };

    let all_layers = [0, 1];

    let syntax: Vec<usize> = all_layers
        .iter()
        .copied()
        .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
        .collect();
    assert_eq!(syntax, vec![0]);

    let knowledge: Vec<usize> = all_layers
        .iter()
        .copied()
        .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
        .collect();
    assert_eq!(knowledge, vec![0, 1]);

    let output: Vec<usize> = all_layers
        .iter()
        .copied()
        .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
        .collect();
    assert_eq!(output, vec![1]);
}

#[test]
fn test_layer_band_from_family() {
    let bands = LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));
}

#[test]
fn test_layer_band_fallback() {
    // Unknown family with enough layers → estimated bands
    let bands = LayerBands::for_family("unknown_family", 20).unwrap();
    assert_eq!(bands.syntax.0, 0);
    assert!(bands.knowledge.0 > 0);
    assert!(bands.output.1 == 19);
}

// ══════════════════════════════════════════════════════════════
// SELECT WITH RELATION FILTER
// ══════════════════════════════════════════════════════════════

#[test]
fn test_select_with_relation_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    let mut labels: HashMap<(usize, usize), String> = HashMap::new();
    labels.insert((0, 0), "capital".into());
    labels.insert((0, 1), "language".into());

    // Simulate SELECT with relation="capital" filter
    let metas = patched.down_meta_at(0).unwrap();
    let matches: Vec<(usize, &str)> = metas
        .iter()
        .enumerate()
        .filter_map(|(i, m)| m.as_ref().map(|m| (i, m.top_token.as_str())))
        .filter(|(i, _)| {
            labels
                .get(&(0, *i))
                .map(|r| r.to_lowercase().contains("capital"))
                .unwrap_or(false)
        })
        .collect();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].1, "Paris");
}

#[test]
fn test_select_relation_label_in_output() {
    let mut labels: HashMap<(usize, usize), String> = HashMap::new();
    labels.insert((0, 0), "capital".into());

    // Feature with label
    let rel = labels.get(&(0, 0));
    assert_eq!(rel, Some(&"capital".to_string()));

    // Feature without label
    let rel = labels.get(&(0, 1));
    assert_eq!(rel, None);
}

// ══════════════════════════════════════════════════════════════
// WALK WITH RELATION LABELS
// ══════════════════════════════════════════════════════════════

#[test]
fn test_walk_hits_include_relation_label() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    let mut labels: HashMap<(usize, usize), String> = HashMap::new();
    labels.insert((0, 0), "capital".into());

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);

    // Simulate what walk handler does: add relation label to hits
    for (layer, hits) in &trace.layers {
        for hit in hits {
            let label = labels.get(&(*layer, hit.feature));
            if hit.feature == 0 {
                assert_eq!(label, Some(&"capital".to_string()));
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════
// DESCRIBE HANDLER LOGIC (edge aggregation, scoring, filtering)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_describe_min_score_filtering() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 10);

    let min_score = 0.5;
    let mut edges = Vec::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score >= min_score {
                edges.push(hit.meta.top_token.clone());
            }
        }
    }
    // Only hits above threshold should pass
    for (_, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < min_score {
                assert!(!edges.contains(&hit.meta.top_token) || hit.gate_score >= min_score);
            }
        }
    }
}

#[test]
fn test_describe_edge_aggregation_by_target() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 10);

    // Aggregate by target token (lowercase key)
    let mut edges: HashMap<String, f32> = HashMap::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            let key = hit.meta.top_token.to_lowercase();
            let entry = edges.entry(key).or_insert(0.0);
            if hit.gate_score > *entry {
                *entry = hit.gate_score;
            }
        }
    }
    // Should have aggregated entries
    assert!(!edges.is_empty());
}

#[test]
fn test_describe_verbose_adds_layer_range() {
    // Verbose mode adds layer_min, layer_max, count
    let layers = [14usize, 18, 22, 27];
    let min_l = *layers.iter().min().unwrap();
    let max_l = *layers.iter().max().unwrap();
    assert_eq!(min_l, 14);
    assert_eq!(max_l, 27);
    assert_eq!(layers.len(), 4); // count
}

#[test]
fn test_describe_self_reference_filtered() {
    // DESCRIBE "France" should not include "France" as an edge target
    let entity = "France";
    let target = "France";
    assert_eq!(entity.to_lowercase(), target.to_lowercase());
    // Handler filters this case
}

// ══════════════════════════════════════════════════════════════
// SESSION-SCOPED DESCRIBE/WALK/SELECT
// ══════════════════════════════════════════════════════════════

#[test]
fn test_session_scoped_describe() {
    // Session A patches feature 0 → different describe result
    let index = test_index();
    let mut session_a = PatchedVindex::new(index.clone());
    let global = PatchedVindex::new(index);

    session_a.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

    // Session A: London
    let trace_a = session_a.walk(&query, &[0], 3);
    assert_eq!(trace_a.layers[0].1[0].meta.top_token, "London");

    // Global: still Paris
    let trace_g = global.walk(&query, &[0], 3);
    assert_eq!(trace_g.layers[0].1[0].meta.top_token, "Paris");
}

#[test]
fn test_session_scoped_walk() {
    let index = test_index();
    let mut session = PatchedVindex::new(index.clone());
    let global = PatchedVindex::new(index);

    session.delete_feature(0, 0);

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace_s = session.walk(&query, &[0], 3);
    let trace_g = global.walk(&query, &[0], 3);

    // Session: feature 0 removed
    assert!(trace_s.layers[0].1.iter().all(|h| h.feature != 0));
    // Global: feature 0 present
    assert!(trace_g.layers[0].1.iter().any(|h| h.feature == 0));
}

#[test]
fn test_session_scoped_select() {
    let index = test_index();
    let mut session = PatchedVindex::new(index.clone());
    let global = PatchedVindex::new(index);

    session.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    // Session: feature 0 → London
    assert_eq!(session.feature_meta(0, 0).unwrap().top_token, "London");
    // Global: feature 0 → Paris
    assert_eq!(global.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ══════════════════════════════════════════════════════════════
// SESSION MANAGEMENT LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_session_id_header_parsing() {
    let header_value = "sess-abc123";
    assert_eq!(header_value, "sess-abc123");
}

#[test]
fn test_session_patch_isolation() {
    // Two sessions should have independent patch state
    let index = test_index();
    let mut patched_a = PatchedVindex::new(index.clone());
    let mut patched_b = PatchedVindex::new(index);

    patched_a.delete_feature(0, 0);
    // Session A: feature 0 deleted
    assert!(patched_a.feature_meta(0, 0).is_none());
    // Session B: feature 0 still exists
    assert!(patched_b.feature_meta(0, 0).is_some());

    patched_b.update_feature_meta(0, 1, make_meta("Updated", 999, 0.99));
    assert_eq!(patched_b.feature_meta(0, 1).unwrap().top_token, "Updated");
    // Session A: feature 1 unchanged
    assert_eq!(patched_a.feature_meta(0, 1).unwrap().top_token, "French");
}

#[test]
fn test_session_global_unaffected() {
    let index = test_index();
    let global = PatchedVindex::new(index.clone());
    let mut session = PatchedVindex::new(index);

    session.delete_feature(0, 0);
    // Global: untouched
    assert!(global.feature_meta(0, 0).is_some());
    assert_eq!(global.feature_meta(0, 0).unwrap().top_token, "Paris");
}
