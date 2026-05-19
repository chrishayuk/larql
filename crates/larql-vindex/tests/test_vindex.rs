//! Tests for the larql-vindex crate.

use larql_vindex::format::filenames::*;
use larql_vindex::{
    FeatureMeta, GateIndex, PatchOverrides, VectorIndex, VindexConfig, VindexLayerInfo,
};
use ndarray::{ArcArray2, Array1, Array2};

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
        top_k: vec![make_top_k(token, id, score)],
    }
}

/// Build a small in-memory VectorIndex for testing.
fn test_index() -> VectorIndex {
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    // Layer 0: 3 features × 4 hidden
    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0; // feature 0 responds to dim 0
    gate0[[1, 1]] = 1.0; // feature 1 responds to dim 1
    gate0[[2, 2]] = 1.0; // feature 2 responds to dim 2

    // Layer 1: 3 features × 4 hidden
    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[1, 1]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let gate_vectors = vec![Some(gate0), Some(gate1)];

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None, // feature 1 has no metadata
        Some(make_meta("Spain", 202, 0.70)),
    ];

    let down_meta = vec![Some(meta0), Some(meta1)];

    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

// ══════════════════════════════════════════════════════════════
// CONSTRUCTION
// ══════════════════════════════════════════════════════════════

#[test]
fn new_index_has_correct_dimensions() {
    let idx = test_index();
    assert_eq!(idx.num_layers, 2);
    assert_eq!(idx.hidden_size, 4);
}

#[test]
fn loaded_layers() {
    let idx = test_index();
    assert_eq!(idx.loaded_layers(), vec![0, 1]);
}

#[test]
fn num_features_per_layer() {
    let idx = test_index();
    assert_eq!(idx.num_features(0), 3);
    assert_eq!(idx.num_features(1), 3);
    assert_eq!(idx.num_features(99), 0); // out of range
}

#[test]
fn total_counts() {
    let idx = test_index();
    assert_eq!(idx.total_gate_vectors(), 6); // 3 + 3
    assert_eq!(idx.total_down_meta(), 5); // 3 + 2 (one None)
}

// ══════════════════════════════════════════════════════════════
// FEATURE LOOKUP
// ══════════════════════════════════════════════════════════════

#[test]
fn feature_meta_lookup() {
    let idx = test_index();
    let meta = idx.feature_meta(0, 0).unwrap();
    assert_eq!(meta.top_token, "Paris");
    assert_eq!(meta.top_token_id, 100);
    assert!((meta.c_score - 0.95).abs() < 0.01);
}

#[test]
fn feature_meta_none_for_missing() {
    let idx = test_index();
    assert!(idx.feature_meta(1, 1).is_none()); // explicitly None
    assert!(idx.feature_meta(99, 0).is_none()); // out of range layer
    assert!(idx.feature_meta(0, 99).is_none()); // out of range feature
}

#[test]
fn down_meta_at_returns_slice() {
    let idx = test_index();
    let metas = idx.down_meta_at(0).unwrap();
    assert_eq!(metas.len(), 3);
    assert!(metas[0].is_some());
    assert!(metas[1].is_some());
    assert!(metas[2].is_some());

    let metas1 = idx.down_meta_at(1).unwrap();
    assert!(metas1[1].is_none()); // the gap
}

// ══════════════════════════════════════════════════════════════
// GATE KNN
// ══════════════════════════════════════════════════════════════

#[test]
fn gate_knn_finds_best_match() {
    let idx = test_index();

    // Query along dim 0 → should match feature 0 at layer 0
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0); // feature 0
    assert!((hits[0].1 - 1.0).abs() < 0.01); // dot product = 1.0
}

#[test]
fn gate_knn_top_k_ordering() {
    let idx = test_index();

    // Query with components in dim 0 and dim 1
    let query = Array1::from_vec(vec![0.8, 0.6, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 3);

    assert_eq!(hits.len(), 3);
    // Feature 0 (dim 0): dot = 0.8
    // Feature 1 (dim 1): dot = 0.6
    // Feature 2 (dim 2): dot = 0.0
    assert_eq!(hits[0].0, 0); // highest
    assert_eq!(hits[1].0, 1);
}

#[test]
fn gate_knn_empty_for_missing_layer() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(99, &query, 5);
    assert!(hits.is_empty());
}

// ══════════════════════════════════════════════════════════════
// WALK
// ══════════════════════════════════════════════════════════════

#[test]
fn walk_across_layers() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = idx.walk(&query, &[0, 1], 2);

    assert_eq!(trace.layers.len(), 2);

    // Layer 0: feature 0 fires (dim 0 = 1.0)
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].feature, 0);
    assert_eq!(hits[0].meta.top_token, "Paris");

    // Layer 1: feature 1 fires (dim 0 contributes 0.5)
    let (layer1, hits1) = &trace.layers[1];
    assert_eq!(*layer1, 1);
    assert!(!hits1.is_empty());
}

#[test]
fn walk_skips_features_without_meta() {
    let idx = test_index();
    // Query that activates feature 1 at layer 1 (which has no metadata)
    let query = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let trace = idx.walk(&query, &[1], 3);

    // Feature 1 at layer 1 has None metadata — should be filtered out
    let (_, hits) = &trace.layers[0];
    for hit in hits {
        assert_ne!(hit.feature, 1); // feature 1 should not appear
    }
}

// ══════════════════════════════════════════════════════════════
// MUTATION
// ══════════════════════════════════════════════════════════════

#[test]
fn set_feature_meta() {
    let mut idx = test_index();
    assert!(idx.feature_meta(1, 1).is_none());

    let meta = make_meta("London", 300, 0.85);
    idx.set_feature_meta(1, 1, meta);

    let loaded = idx.feature_meta(1, 1).unwrap();
    assert_eq!(loaded.top_token, "London");
    assert_eq!(loaded.top_token_id, 300);
}

#[test]
fn delete_feature_meta() {
    let mut idx = test_index();
    assert!(idx.feature_meta(0, 0).is_some());

    idx.delete_feature_meta(0, 0);
    assert!(idx.feature_meta(0, 0).is_none());
}

#[test]
fn find_free_feature() {
    let mut idx = test_index();

    // Layer 0: all 3 features have metadata → returns weakest (lowest c_score)
    // Scores: Paris=0.95, French=0.88, Europe=0.75 → weakest is Europe at F2
    let slot = idx.find_free_feature(0).unwrap();
    assert_eq!(slot, 2); // Europe has lowest c_score

    // Layer 1: feature 1 is None → returns empty slot first
    assert_eq!(idx.find_free_feature(1), Some(1));

    // Delete one in layer 0 → returns the now-empty slot
    idx.delete_feature_meta(0, 2);
    assert_eq!(idx.find_free_feature(0), Some(2));
}

#[test]
fn set_gate_vector() {
    let mut idx = test_index();
    let new_vec = Array1::from_vec(vec![0.0, 0.0, 0.0, 9.9]);
    idx.set_gate_vector(0, 1, &new_vec);

    // Query along dim 3 should now match feature 1 at layer 0
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = idx.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, 1); // feature 1
    assert!((hits[0].1 - 9.9).abs() < 0.01);
}

#[test]
fn mutation_does_not_affect_other_features() {
    let mut idx = test_index();

    // Mutate feature 0
    idx.set_feature_meta(0, 0, make_meta("Modified", 999, 0.5));

    // Feature 1 should be unchanged
    let meta1 = idx.feature_meta(0, 1).unwrap();
    assert_eq!(meta1.top_token, "French");
}

// ══════════════════════════════════════════════════════════════
// DOWN VECTOR OVERRIDES (used by COMPILE INTO VINDEX baker)
// ══════════════════════════════════════════════════════════════

#[test]
fn down_overrides_starts_empty() {
    let idx = test_index();
    assert!(idx.down_overrides().is_empty());
}

#[test]
fn set_down_vector_records_override() {
    let mut idx = test_index();
    let v = vec![0.1, 0.2, 0.3, 0.4];
    idx.set_down_vector(0, 1, v.clone());

    assert_eq!(idx.down_overrides().len(), 1);
    let stored = idx.down_overrides().get(&(0, 1)).unwrap();
    assert_eq!(stored, &v);

    // Trait method exposes the same data.
    let via_trait = idx.down_override(0, 1).unwrap();
    assert_eq!(via_trait, v.as_slice());

    // Inherent singular accessor returns the same slice.
    let via_inherent = idx.down_override_at(0, 1).unwrap();
    assert_eq!(via_inherent, v.as_slice());

    // Missing key returns None on both forms.
    assert!(idx.down_override_at(0, 2).is_none());
    assert!(idx.down_override(0, 2).is_none());

    // has_overrides_at reflects the layer.
    assert!(idx.has_overrides_at(0));
    assert!(!idx.has_overrides_at(1));
}

#[test]
fn patched_vindex_down_override_at_forwards_to_base() {
    let mut idx = test_index();
    let v = vec![1.0, 2.0, 3.0, 4.0];
    idx.set_down_vector(1, 0, v.clone());
    let patched = larql_vindex::PatchedVindex::new(idx);

    // Singular forwarder mirrors the inherent base accessor.
    let via_patched = patched.down_override_at(1, 0).unwrap();
    assert_eq!(via_patched, v.as_slice());
    assert!(patched.down_override_at(0, 0).is_none());

    // The plural collection accessor still returns everything.
    assert_eq!(patched.down_overrides().len(), 1);
}

#[test]
fn down_overrides_overwrite_in_place() {
    let mut idx = test_index();
    idx.set_down_vector(0, 1, vec![1.0; 4]);
    idx.set_down_vector(0, 1, vec![9.0; 4]);
    let stored = idx.down_overrides().get(&(0, 1)).unwrap();
    assert_eq!(stored, &vec![9.0; 4]);
    assert_eq!(idx.down_overrides().len(), 1);
}

#[test]
fn patched_vindex_exposes_base_down_overrides() {
    let mut idx = test_index();
    idx.set_down_vector(1, 0, vec![5.0, 5.0, 5.0, 5.0]);
    let patched = larql_vindex::PatchedVindex::new(idx);
    assert_eq!(patched.down_overrides().len(), 1);
    assert!(patched.down_overrides().contains_key(&(1, 0)));
}

#[test]
fn patched_vindex_overrides_gate_at_returns_inserted_gate() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Before insert: nothing.
    assert!(patched.overrides_gate_at(0, 0).is_none());

    let gate = vec![0.5, 0.5, 0.5, 0.5];
    let meta = make_meta("Inserted", 42, 0.99);
    patched.insert_feature(0, 0, gate.clone(), meta);

    let read = patched.overrides_gate_at(0, 0).unwrap();
    assert_eq!(read, gate.as_slice());

    // Other slots remain absent.
    assert!(patched.overrides_gate_at(0, 1).is_none());
    assert!(patched.overrides_gate_at(1, 0).is_none());
}

// ══════════════════════════════════════════════════════════════
// SAVE / LOAD ROUND-TRIP
// ══════════════════════════════════════════════════════════════

#[test]
fn save_and_load_down_meta_round_trip() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_down_meta_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Save gate vectors + down_meta + config (needed for load_vindex)
    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    let count = idx.save_down_meta(&dir).unwrap();
    assert_eq!(count, 5); // 3 + 2 (one None skipped)

    let config = VindexConfig {
        version: 2,
        model: "test".into(),
        family: "test".into(),
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: layer_infos,
        down_top_k: 1,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Write a minimal tokenizer (needed for binary down_meta loading)
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    // Load it back via the proper load path
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let idx2 = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

    // Verify content — binary down_meta stores token IDs, not strings.
    // With an empty tokenizer vocab, strings decode to empty or token IDs.
    // Check that the data round-trips (token_id and c_score preserved).
    let meta = idx2.feature_meta(0, 0).unwrap();
    assert_eq!(meta.top_token_id, 100);
    assert!((meta.c_score - 0.95).abs() < 0.01);

    let meta1 = idx2.feature_meta(1, 0).unwrap();
    assert_eq!(meta1.top_token_id, 200);

    // Feature 1 at layer 1 should still be None
    assert!(idx2.feature_meta(1, 1).is_none());

    // Gate vectors should also round-trip
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx2.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, 0); // feature 0

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_and_load_gate_vectors_round_trip() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_gate_rt");
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    assert_eq!(layer_infos.len(), 2);
    assert_eq!(layer_infos[0].layer, 0);
    assert_eq!(layer_infos[0].num_features, 3);
    assert_eq!(layer_infos[1].layer, 1);

    // Verify file exists with expected size
    let gate_path = dir.join("gate_vectors.bin");
    assert!(gate_path.exists());
    let file_size = std::fs::metadata(&gate_path).unwrap().len();
    // 2 layers × 3 features × 4 hidden × 4 bytes = 96 bytes
    assert_eq!(file_size, 96);

    // Clean up
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_config_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_config_rt");
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-model".into(),
        family: "test".into(),
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
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
        down_top_k: 10,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();

    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(loaded.model, "test-model");
    assert_eq!(loaded.num_layers, 2);
    assert_eq!(loaded.hidden_size, 4);
    assert_eq!(loaded.layers.len(), 2);
    assert_eq!(loaded.layers[0].num_features, 3);

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// BINARY DOWN_META
// ══════════════════════════════════════════════════════════════

#[test]
fn binary_down_meta_write_read_round_trip() {
    let _idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_binary_dm");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write binary format
    let count = larql_vindex::down_meta::write_binary(
        &dir,
        &[
            Some(vec![
                Some(make_meta("Paris", 100, 0.95)),
                Some(make_meta("French", 101, 0.88)),
                None,
            ]),
            Some(vec![
                Some(make_meta("Berlin", 200, 0.90)),
                None,
                Some(make_meta("Spain", 202, 0.70)),
            ]),
        ],
        1, // top_k = 1
    )
    .unwrap();
    assert_eq!(count, 4); // 2 + 2 (Nones don't count)

    // Verify file exists and is much smaller than JSONL would be
    let bin_path = dir.join("down_meta.bin");
    assert!(bin_path.exists());
    let bin_size = std::fs::metadata(&bin_path).unwrap().len();
    // Header (16) + 2 layers × (4 bytes layer header + 3 features × (4+4+1×8) bytes)
    assert!(bin_size > 0);
    assert!(bin_size < 200); // should be very small for 6 features

    // Read back — needs a tokenizer for string resolution
    // Create a minimal tokenizer that maps IDs to strings
    // Since we can't easily create a real tokenizer in tests,
    // verify the raw binary structure is correct
    let data = std::fs::read(&bin_path).unwrap();
    // Check magic
    assert_eq!(
        u32::from_le_bytes([data[0], data[1], data[2], data[3]]),
        0x444D4554
    );
    // Check version
    assert_eq!(u32::from_le_bytes([data[4], data[5], data[6], data[7]]), 1);
    // Check num_layers
    assert_eq!(
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
        2
    );
    // Check top_k
    assert_eq!(
        u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
        1
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_down_meta_writes_binary() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_dm_binary");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let count = idx.save_down_meta(&dir).unwrap();
    assert_eq!(count, 5); // 3 + 2

    // Binary file should exist (JSONL no longer written)
    assert!(dir.join("down_meta.bin").exists());
    assert!(!dir.join("down_meta.jsonl").exists());

    let bin_size = std::fs::metadata(dir.join("down_meta.bin")).unwrap().len();
    assert!(bin_size > 0);

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// ERROR HANDLING
// ══════════════════════════════════════════════════════════════

#[test]
fn load_nonexistent_vindex_errors() {
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let result =
        VectorIndex::load_vindex(std::path::Path::new("/nonexistent/fake.vindex"), &mut cb);
    assert!(result.is_err());
}

#[test]
fn load_nonexistent_config_errors() {
    let result = larql_vindex::load_vindex_config(std::path::Path::new("/nonexistent/fake.vindex"));
    assert!(result.is_err());
}

// ══════════════════════════════════════════════════════════════
// LAYER BANDS
// ══════════════════════════════════════════════════════════════

#[test]
fn layer_bands_gemma3_4b() {
    let bands = larql_vindex::LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));
}

#[test]
fn layer_bands_gemma2_9b() {
    let bands = larql_vindex::LayerBands::for_family("gemma2", 42).unwrap();
    assert_eq!(bands.syntax, (0, 16));
    assert_eq!(bands.knowledge, (17, 34));
    assert_eq!(bands.output, (35, 41));
}

#[test]
fn layer_bands_llama3_70b() {
    let bands = larql_vindex::LayerBands::for_family("llama", 80).unwrap();
    assert_eq!(bands.syntax, (0, 31));
    assert_eq!(bands.knowledge, (32, 63));
    assert_eq!(bands.output, (64, 79));
}

#[test]
fn layer_bands_llama3_8b() {
    let bands = larql_vindex::LayerBands::for_family("llama", 32).unwrap();
    assert_eq!(bands.syntax, (0, 12));
    assert_eq!(bands.knowledge, (13, 25));
    assert_eq!(bands.output, (26, 31));
}

#[test]
fn layer_bands_mixtral() {
    let bands = larql_vindex::LayerBands::for_family("mixtral", 32).unwrap();
    assert_eq!(bands.syntax, (0, 12));
    assert_eq!(bands.knowledge, (13, 25));
    assert_eq!(bands.output, (26, 31));
}

#[test]
fn layer_bands_gpt2_small() {
    let bands = larql_vindex::LayerBands::for_family("gpt2", 12).unwrap();
    assert_eq!(bands.syntax, (0, 4));
    assert_eq!(bands.knowledge, (5, 9));
    assert_eq!(bands.output, (10, 11));
}

#[test]
fn layer_bands_unknown_family_fallback() {
    // Unknown family with enough layers → falls back to heuristic
    let bands = larql_vindex::LayerBands::for_family("unknown_model", 40).unwrap();
    assert_eq!(bands.syntax.0, 0);
    assert!(bands.knowledge.0 > bands.syntax.1);
    assert!(bands.output.0 > bands.knowledge.1);
    assert_eq!(bands.output.1, 39);
}

#[test]
fn layer_bands_tiny_model_returns_none() {
    // Too few layers to band meaningfully
    assert!(larql_vindex::LayerBands::for_family("test", 2).is_none());
    assert!(larql_vindex::LayerBands::for_family("test", 4).is_none());
}

#[test]
fn layer_bands_band_for_layer() {
    let bands = larql_vindex::LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.band_for_layer(0), "syntax");
    assert_eq!(bands.band_for_layer(13), "syntax");
    assert_eq!(bands.band_for_layer(14), "knowledge");
    assert_eq!(bands.band_for_layer(27), "knowledge");
    assert_eq!(bands.band_for_layer(28), "output");
    assert_eq!(bands.band_for_layer(33), "output");
}

#[test]
fn v1_config_loads_with_defaults() {
    // Simulate a v1 index.json that lacks new fields
    let dir = std::env::temp_dir().join("larql_test_v1_compat");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let v1_json = r#"{
        "version": 1,
        "model": "old-model",
        "family": "test",
        "num_layers": 32,
        "hidden_size": 4,
        "intermediate_size": 3,
        "vocab_size": 100,
        "embed_scale": 1.0,
        "layers": [],
        "down_top_k": 10
    }"#;
    std::fs::write(dir.join("index.json"), v1_json).unwrap();

    let config = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(config.version, 1);
    assert_eq!(config.model, "old-model");
    // New fields should have sensible defaults
    assert_eq!(config.extract_level, larql_vindex::ExtractLevel::Browse);
    assert!(config.layer_bands.is_none());
    assert!(config.source.is_none());
    assert!(config.checksums.is_none());
    assert!(!config.has_model_weights);
    assert!(config.model_config.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_config_full_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_v2_full_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write a dummy gate_vectors.bin so checksums have something to hash
    std::fs::write(dir.join("gate_vectors.bin"), b"test data").unwrap();

    let checksums = larql_vindex::checksums::compute_checksums(&dir).ok();

    let config = VindexConfig {
        version: 2,
        model: "google/gemma-3-4b-it".into(),
        family: "gemma3".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some("google/gemma-3-4b-it".into()),
            huggingface_revision: Some("abc123".into()),
            safetensors_sha256: None,
            extracted_at: "2026-04-01T12:00:00Z".into(),
            larql_version: "0.1.0".into(),
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        }),
        checksums,
        num_layers: 34,
        hidden_size: 2560,
        intermediate_size: 10240,
        vocab_size: 262144,
        embed_scale: 50.596,
        extract_level: larql_vindex::ExtractLevel::Inference,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: Some(larql_vindex::LayerBands {
            syntax: (0, 13),
            knowledge: (14, 27),
            output: (28, 33),
        }),
        layers: vec![],
        down_top_k: 10,
        has_model_weights: true,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "gemma3".into(),
            head_dim: 256,
            num_q_heads: 8,
            num_kv_heads: 4,
            rope_base: 10000.0,
            sliding_window: Some(1024),
            moe: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
        }),
        fp4: None,
        ffn_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    // Verify all v2 fields round-trip
    assert_eq!(loaded.version, 2);
    assert_eq!(loaded.model, "google/gemma-3-4b-it");
    assert_eq!(loaded.extract_level, larql_vindex::ExtractLevel::Inference);
    assert!(loaded.has_model_weights);

    let source = loaded.source.unwrap();
    assert_eq!(
        source.huggingface_repo.as_deref(),
        Some("google/gemma-3-4b-it")
    );
    assert_eq!(source.huggingface_revision.as_deref(), Some("abc123"));
    assert_eq!(source.larql_version, "0.1.0");

    let bands = loaded.layer_bands.unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));

    let mc = loaded.model_config.unwrap();
    assert_eq!(mc.model_type, "gemma3");
    assert_eq!(mc.head_dim, 256);
    assert_eq!(mc.sliding_window, Some(1024));
    assert!(mc.moe.is_none());

    assert!(loaded.checksums.is_some());
    let cs = loaded.checksums.unwrap();
    assert!(cs.contains_key("gate_vectors.bin"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_config_with_moe() {
    let dir = std::env::temp_dir().join("larql_test_v2_moe");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "mistralai/Mixtral-8x7B".into(),
        family: "mixtral".into(),
        source: None,
        checksums: None,
        num_layers: 32,
        hidden_size: 4096,
        intermediate_size: 14336,
        vocab_size: 32000,
        embed_scale: 64.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: Some(larql_vindex::LayerBands::for_family("mixtral", 32).unwrap()),
        layers: vec![],
        down_top_k: 10,
        has_model_weights: false,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "mixtral".into(),
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            rope_base: 1000000.0,
            sliding_window: None,
            moe: Some(larql_vindex::MoeConfig {
                num_experts: 8,
                top_k: 2,
                shared_expert: false,
                router_type: "top_k_softmax".into(),
                moe_intermediate_size: None,
                hybrid: false,
            }),
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
        }),
        fp4: None,
        ffn_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let mc = loaded.model_config.unwrap();
    let moe = mc.moe.unwrap();
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);
    assert!(!moe.shared_expert);
    assert_eq!(moe.router_type, "top_k_softmax");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn moe_index_gate_knn_across_experts() {
    // Simulate a MoE layer: 2 experts × 3 features = 6 total features
    // Expert 0 features respond to dims 0,1,2
    // Expert 1 features respond to dim 3
    let hidden = 4;
    let features_per_expert = 3;
    let num_experts = 2;

    // Concatenate expert gate matrices (as build_vindex would)
    let mut gate0 = Array2::<f32>::zeros((num_experts * features_per_expert, hidden));
    // Expert 0
    gate0[[0, 0]] = 10.0; // E0F0 responds to dim 0
    gate0[[1, 1]] = 10.0; // E0F1 responds to dim 1
    gate0[[2, 2]] = 10.0; // E0F2 responds to dim 2
                          // Expert 1
    gate0[[3, 3]] = 10.0; // E1F0 responds to dim 3
    gate0[[4, 0]] = 5.0;
    gate0[[4, 3]] = 5.0; // E1F1 mixed
    gate0[[5, 1]] = 3.0; // E1F2 weak dim 1

    let gate_vectors = vec![Some(gate0)];

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),  // E0F0
        Some(make_meta("Berlin", 101, 0.92)), // E0F1
        Some(make_meta("Tokyo", 102, 0.88)),  // E0F2
        Some(make_meta("London", 103, 0.90)), // E1F0
        Some(make_meta("Rome", 104, 0.85)),   // E1F1
        Some(make_meta("Madrid", 105, 0.80)), // E1F2
    ];
    let down_meta = vec![Some(meta0)];

    let idx = VectorIndex::new(gate_vectors, down_meta, 1, hidden);
    assert_eq!(idx.num_features(0), 6); // 2 experts × 3 features

    // Query dim 0 → should match E0F0 (Paris) strongest
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 2);
    assert_eq!(hits[0].0, 0); // E0F0 = Paris
    assert_eq!(hits[1].0, 4); // E1F1 = Rome (has dim 0 component)

    // Query dim 3 → should match E1F0 (London) strongest
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = idx.gate_knn(0, &query, 2);
    assert_eq!(hits[0].0, 3); // E1F0 = London
    assert_eq!(hits[1].0, 4); // E1F1 = Rome (has dim 3 component)

    // Walk should find features across experts
    let query = Array1::from_vec(vec![0.5, 0.0, 0.0, 0.5]);
    let trace = idx.walk(&query, &[0], 3);
    let (_, hits) = &trace.layers[0];
    // Both E0F0 (Paris, dim0) and E1F0 (London, dim3) should appear
    let tokens: Vec<&str> = hits.iter().map(|h| h.meta.top_token.as_str()).collect();
    assert!(tokens.contains(&"Paris") || tokens.contains(&"London"));
}

#[test]
fn moe_layer_info_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_moe_layer_info");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-moe".into(),
        family: "mixtral".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: larql_vindex::LayerBands::for_family("mixtral", 32),
        layers: vec![VindexLayerInfo {
            layer: 0,
            num_features: 24, // 8 experts × 3 features
            offset: 0,
            length: 384,
            num_experts: Some(8),
            num_features_per_expert: Some(3),
        }],
        down_top_k: 10,
        has_model_weights: false,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "mixtral".into(),
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            rope_base: 1000000.0,
            sliding_window: None,
            moe: Some(larql_vindex::MoeConfig {
                num_experts: 8,
                top_k: 2,
                shared_expert: false,
                router_type: "top_k_softmax".into(),
                moe_intermediate_size: None,
                hybrid: false,
            }),
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: None,
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            num_kv_shared_layers: None,
            per_layer_embed_dim: None,
            rope_local_base: None,
            query_pre_attn_scalar: None,
            final_logit_softcapping: None,
            attention_multiplier: None,
            residual_multiplier: None,
            logits_scaling: None,
            norm_eps: None,
        }),
        fp4: None,
        ffn_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    // Verify MoE layer info round-trips
    assert_eq!(loaded.layers[0].num_experts, Some(8));
    assert_eq!(loaded.layers[0].num_features_per_expert, Some(3));
    assert_eq!(loaded.layers[0].num_features, 24);

    // Verify MoE config round-trips
    let moe = loaded.model_config.unwrap().moe.unwrap();
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn layer_bands_config_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_bands_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-bands".into(),
        family: "test".into(),
        num_layers: 34,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: vec![],
        down_top_k: 10,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: Some(larql_vindex::LayerBands {
            syntax: (0, 13),
            knowledge: (14, 27),
            output: (28, 33),
        }),
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let bands = loaded.layer_bands.unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// CHECKSUM VERIFICATION
// ══════════════════════════════════════════════════════════════

#[test]
fn checksum_compute_and_verify() {
    let dir = std::env::temp_dir().join("larql_test_checksums");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write some test data
    std::fs::write(dir.join("gate_vectors.bin"), b"test gate data").unwrap();
    std::fs::write(dir.join("embeddings.bin"), b"test embed data").unwrap();
    std::fs::write(dir.join("down_meta.bin"), b"test down data").unwrap();

    // Compute checksums
    let checksums = larql_vindex::checksums::compute_checksums(&dir).unwrap();
    assert_eq!(checksums.len(), 3); // 3 files present
    assert!(checksums.contains_key("gate_vectors.bin"));
    assert!(checksums.contains_key("embeddings.bin"));
    assert!(checksums.contains_key("down_meta.bin"));

    // Verify — should all pass
    let results = larql_vindex::checksums::verify_checksums(&dir, &checksums).unwrap();
    assert!(results.iter().all(|(_, ok)| *ok));

    // Corrupt a file
    std::fs::write(dir.join("gate_vectors.bin"), b"corrupted!").unwrap();
    let results = larql_vindex::checksums::verify_checksums(&dir, &checksums).unwrap();
    let gate_result = results
        .iter()
        .find(|(f, _)| f == "gate_vectors.bin")
        .unwrap();
    assert!(!gate_result.1); // should fail

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checksum_individual_file() {
    let dir = std::env::temp_dir().join("larql_test_sha256");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("test.bin"), b"hello world").unwrap();
    let hash = larql_vindex::checksums::sha256_file(&dir.join("test.bin")).unwrap();
    // SHA256 of "hello world" is known
    assert_eq!(
        hash,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// EXTRACT LEVEL
// ══════════════════════════════════════════════════════════════

#[test]
fn extract_level_serialization() {
    assert_eq!(format!("{}", larql_vindex::ExtractLevel::Browse), "browse");
    assert_eq!(
        format!("{}", larql_vindex::ExtractLevel::Inference),
        "inference"
    );
    assert_eq!(format!("{}", larql_vindex::ExtractLevel::All), "all");

    // serde round-trip
    let json = serde_json::to_string(&larql_vindex::ExtractLevel::Inference).unwrap();
    assert_eq!(json, "\"inference\"");
    let parsed: larql_vindex::ExtractLevel = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, larql_vindex::ExtractLevel::Inference);
}

#[test]
fn extract_level_default_is_browse() {
    let level: larql_vindex::ExtractLevel = Default::default();
    assert_eq!(level, larql_vindex::ExtractLevel::Browse);
}

// ══════════════════════════════════════════════════════════════
// DESCRIBE TYPES
// ══════════════════════════════════════════════════════════════

#[test]
fn label_source_display() {
    assert_eq!(format!("{}", larql_vindex::LabelSource::Probe), "probe");
    assert_eq!(format!("{}", larql_vindex::LabelSource::Cluster), "cluster");
    assert_eq!(format!("{}", larql_vindex::LabelSource::Pattern), "pattern");
    assert_eq!(format!("{}", larql_vindex::LabelSource::None), "");
}

#[test]
fn describe_edge_construction() {
    let edge = larql_vindex::DescribeEdge {
        relation: Some("capital".into()),
        source: larql_vindex::LabelSource::Probe,
        target: "Paris".into(),
        gate_score: 1436.9,
        layer_min: 27,
        layer_max: 27,
        count: 1,
        also_tokens: vec![],
    };
    assert_eq!(edge.relation.as_deref(), Some("capital"));
    assert_eq!(edge.source, larql_vindex::LabelSource::Probe);
    assert_eq!(edge.target, "Paris");
}

// ══════════════════════════════════════════════════════════════
// SOURCE PROVENANCE
// ══════════════════════════════════════════════════════════════

#[test]
fn source_provenance_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_provenance");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test/provenance".into(),
        family: "test".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some("google/gemma-3-4b-it".into()),
            huggingface_revision: Some("abc123def456".into()),
            safetensors_sha256: Some("deadbeef".into()),
            extracted_at: "2026-04-01T12:00:00Z".into(),
            larql_version: "0.1.0".into(),
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        }),
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::All,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: vec![],
        down_top_k: 10,
        has_model_weights: true,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let src = loaded.source.unwrap();
    assert_eq!(
        src.huggingface_repo.as_deref(),
        Some("google/gemma-3-4b-it")
    );
    assert_eq!(src.huggingface_revision.as_deref(), Some("abc123def456"));
    assert_eq!(src.safetensors_sha256.as_deref(), Some("deadbeef"));
    assert_eq!(src.extracted_at, "2026-04-01T12:00:00Z");
    assert_eq!(src.larql_version, "0.1.0");
    assert_eq!(loaded.extract_level, larql_vindex::ExtractLevel::All);

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// PATCHES
// ══════════════════════════════════════════════════════════════

#[test]
fn patch_save_and_load_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_patch_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "google/gemma-3-4b-it".into(),
        base_checksum: Some("abc123".into()),
        created_at: "2026-04-01T12:00:00Z".into(),
        description: Some("Test patch".into()),
        author: Some("test".into()),
        tags: vec!["test".into()],
        operations: vec![
            larql_vindex::PatchOp::Insert {
                layer: 26,
                feature: 8821,
                relation: Some("lives-in".into()),
                entity: "John Coyle".into(),
                target: "Colchester".into(),
                confidence: Some(0.85),
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                    top_token: "Colchester".into(),
                    top_token_id: 42,
                    c_score: 4.2,
                }),
            },
            larql_vindex::PatchOp::Delete {
                layer: 24,
                feature: 1337,
                reason: Some("hallucinated".into()),
            },
        ],
    };

    let path = dir.join("test.vlp");
    patch.save(&path).unwrap();

    // Verify file exists and is valid JSON
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("John Coyle"));
    assert!(text.contains("hallucinated"));

    // Load back
    let loaded = larql_vindex::VindexPatch::load(&path).unwrap();
    assert_eq!(loaded.version, 1);
    assert_eq!(loaded.base_model, "google/gemma-3-4b-it");
    assert_eq!(loaded.operations.len(), 2);

    let (ins, _upd, del) = loaded.counts();
    assert_eq!(ins, 1);
    assert_eq!(del, 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn patched_vindex_overrides_base() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Base has Paris at (0, 0)
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");

    // Apply patch that overrides (0, 0) to London
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "London".into(),
                top_token_id: 300,
                c_score: 0.99,
            }),
        }],
    };
    patched.apply_patch(patch);

    // Now (0, 0) should return London
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "London");
    // Other features unchanged
    assert_eq!(patched.feature_meta(0, 1).unwrap().top_token, "French");
    assert_eq!(patched.num_patches(), 1);
}

#[test]
fn patched_vindex_delete_hides_feature() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    assert!(patched.feature_meta(0, 2).is_some()); // Europe

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 2,
            reason: Some("test delete".into()),
        }],
    };
    patched.apply_patch(patch);

    assert!(patched.feature_meta(0, 2).is_none()); // deleted
}

#[test]
fn patched_vindex_bake_down() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Apply insert + delete
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Update {
                layer: 0,
                feature: 0,
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                    top_token: "London".into(),
                    top_token_id: 300,
                    c_score: 0.99,
                }),
            },
            larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 2,
                reason: None,
            },
        ],
    };
    patched.apply_patch(patch);

    // Bake down to a new clean index
    let baked = patched.bake_down();

    // Verify baked result
    assert_eq!(baked.feature_meta(0, 0).unwrap().top_token, "London");
    assert_eq!(baked.feature_meta(0, 1).unwrap().top_token, "French");
    assert!(baked.feature_meta(0, 2).is_none()); // deleted
}

#[test]
fn base64_gate_vector_round_trip() {
    let vec = vec![1.0f32, 2.0, 3.0, -4.5];
    let encoded = larql_vindex::patch::core::encode_gate_vector(&vec);
    let decoded = larql_vindex::patch::core::decode_gate_vector(&encoded).unwrap();
    assert_eq!(vec, decoded);
}

#[test]
fn patched_vindex_remove_patch() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "London".into(),
                top_token_id: 300,
                c_score: 0.99,
            }),
        }],
    };
    patched.apply_patch(patch);
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "London");

    // Remove the patch — should revert to base
    patched.remove_patch(0);
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");
    assert_eq!(patched.num_patches(), 0);
}

// ══════════════════════════════════════════════════════════════
// WEIGHTS (split file write/read)
// ══════════════════════════════════════════════════════════════

#[test]
fn weight_manifest_round_trip() {
    // Verify weight_manifest.json is valid JSON after write
    let dir = std::env::temp_dir().join("larql_test_weight_manifest");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write a minimal index.json first (write_model_weights reads it)
    let config = VindexConfig {
        version: 2,
        model: "test".into(),
        family: "test".into(),
        source: None,
        checksums: None,
        num_layers: 0,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 4,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: vec![],
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Verify config round-trips with dtype
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(loaded.dtype, larql_vindex::StorageDtype::F32);

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// DTYPE
// ══════════════════════════════════════════════════════════════

#[test]
fn dtype_config_f16_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_dtype_f16");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-f16".into(),
        family: "test".into(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F16,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: vec![],
        down_top_k: 10,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(loaded.dtype, larql_vindex::StorageDtype::F16);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dtype_display() {
    assert_eq!(format!("{}", larql_vindex::StorageDtype::F32), "f32");
    assert_eq!(format!("{}", larql_vindex::StorageDtype::F16), "f16");
}

#[test]
fn dtype_serde_round_trip() {
    let json = serde_json::to_string(&larql_vindex::StorageDtype::F16).unwrap();
    assert_eq!(json, "\"f16\"");
    let parsed: larql_vindex::StorageDtype = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, larql_vindex::StorageDtype::F16);
}

#[test]
fn dtype_bytes_per_float() {
    assert_eq!(
        larql_vindex::config::dtype::bytes_per_float(larql_vindex::StorageDtype::F32),
        4
    );
    assert_eq!(
        larql_vindex::config::dtype::bytes_per_float(larql_vindex::StorageDtype::F16),
        2
    );
}

// ══════════════════════════════════════════════════════════════
// LOADER (HF cache resolution)
// ══════════════════════════════════════════════════════════════

#[test]
fn resolve_model_path_local_dir() {
    // An existing directory should resolve to itself
    let dir = std::env::temp_dir();
    let result = larql_models::resolve_model_path(dir.to_str().unwrap());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), dir);
}

#[test]
fn resolve_model_path_nonexistent() {
    let result = larql_models::resolve_model_path("/nonexistent/path/to/model");
    assert!(result.is_err());
}

// ══════════════════════════════════════════════════════════════
// PATCH EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn patch_empty_operations() {
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![],
    };
    assert_eq!(patch.len(), 0);
    let (i, u, d) = patch.counts();
    assert_eq!((i, u, d), (0, 0, 0));
}

#[test]
fn patch_multiple_patches_stack() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Patch 1: update F0
    let p1 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "London".into(),
                top_token_id: 300,
                c_score: 0.99,
            }),
        }],
    };
    patched.apply_patch(p1);

    // Patch 2: update F1
    let p2 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 1,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "Munich".into(),
                top_token_id: 301,
                c_score: 0.95,
            }),
        }],
    };
    patched.apply_patch(p2);

    assert_eq!(patched.num_patches(), 2);
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "London");
    assert_eq!(patched.feature_meta(0, 1).unwrap().top_token, "Munich");
    assert_eq!(patched.feature_meta(0, 2).unwrap().top_token, "Europe"); // unchanged
}

#[test]
fn patched_vindex_later_patch_overrides_earlier() {
    let idx = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(idx);

    // Both patches modify F0
    let p1 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "London".into(),
                top_token_id: 300,
                c_score: 0.99,
            }),
        }],
    };
    let p2 = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "Tokyo".into(),
                top_token_id: 400,
                c_score: 0.88,
            }),
        }],
    };
    patched.apply_patch(p1);
    patched.apply_patch(p2);

    // P2 wins
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Tokyo");
}

// ══════════════════════════════════════════════════════════════
// FULL VINDEX LIFECYCLE
// ══════════════════════════════════════════════════════════════

#[test]
fn full_lifecycle_build_query_mutate_save_reload() {
    // Build → query → mutate → save → reload → verify
    let hidden = 4;
    let mut g0 = Array2::<f32>::zeros((4, hidden));
    g0[[0, 0]] = 10.0; // Paris
    g0[[1, 1]] = 10.0; // Berlin
    g0[[2, 2]] = 10.0; // Tokyo
                       // F3 is empty (free slot)
    let gate_vectors = vec![Some(g0)];

    let meta = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("Berlin", 101, 0.92)),
        Some(make_meta("Tokyo", 102, 0.88)),
        None,
    ];
    let down_meta = vec![Some(meta)];

    let mut idx = VectorIndex::new(gate_vectors, down_meta, 1, hidden);

    // Query
    let q = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(idx.gate_knn(0, &q, 1)[0].0, 0); // Paris

    // Mutate
    let slot = idx.find_free_feature(0).unwrap();
    assert_eq!(slot, 3);
    idx.set_gate_vector(0, slot, &Array1::from_vec(vec![0.0, 0.0, 0.0, 10.0]));
    idx.set_feature_meta(0, slot, make_meta("Canberra", 103, 0.85));

    // Save
    let dir = std::env::temp_dir().join("larql_test_lifecycle");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    idx.save_down_meta(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "lifecycle-test".into(),
        family: "test".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: hidden,
        intermediate_size: 4,
        vocab_size: 200,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: layer_infos,
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Write tokenizer for binary down_meta loading
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    // Reload
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

    // Verify — token IDs and scores round-trip through binary
    assert_eq!(loaded.total_gate_vectors(), 4);
    assert_eq!(loaded.total_down_meta(), 4);
    assert_eq!(loaded.feature_meta(0, 0).unwrap().top_token_id, 100);
    assert_eq!(loaded.feature_meta(0, 3).unwrap().top_token_id, 103);

    // KNN should find Canberra for dim 3
    let q2 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(loaded.gate_knn(0, &q2, 1)[0].0, 3);

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// EXTRACT PIPELINE (synthetic model)
// ══════════════════════════════════════════════════════════════

fn make_synthetic_model() -> larql_models::ModelWeights {
    use std::collections::HashMap;

    let num_layers = 2;
    let hidden = 8;
    let intermediate = 4;
    let vocab_size = 16;

    let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    // Per layer: gate, up, down, attn Q/K/V/O, norms
    for layer in 0..num_layers {
        // FFN gate (intermediate × hidden)
        let mut gate = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            gate[[i, i % hidden]] = 1.0 + layer as f32;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.gate_proj.weight"),
            gate.into_shared(),
        );

        // FFN up (intermediate × hidden)
        let mut up = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate {
            up[[i, (i + 1) % hidden]] = 0.5;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.up_proj.weight"),
            up.into_shared(),
        );

        // FFN down (hidden × intermediate)
        let mut down = ndarray::Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate {
            down[[i % hidden, i]] = 0.3;
        }
        tensors.insert(
            format!("layers.{layer}.mlp.down_proj.weight"),
            down.into_shared(),
        );

        // Attention Q/K/V/O (hidden × hidden)
        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut attn = ndarray::Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden {
                attn[[i, i]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.self_attn.{suffix}.weight"),
                attn.into_shared(),
            );
        }

        // Norms
        vectors.insert(
            format!("layers.{layer}.input_layernorm.weight"),
            vec![1.0; hidden],
        );
        vectors.insert(
            format!("layers.{layer}.post_attention_layernorm.weight"),
            vec![1.0; hidden],
        );
    }

    // Final norm
    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    // Embeddings (vocab × hidden)
    let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size {
        embed[[i, i % hidden]] = 1.0;
    }

    let embed = embed.into_shared();
    let lm_head = embed.clone();

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "head_dim": hidden,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
    }));

    larql_models::ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        head_dim: hidden,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: 10000.0,
        arch,
    }
}

#[test]
fn extract_synthetic_model_f32() {
    let dir = std::env::temp_dir().join("larql_test_extract_f32");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let weights = make_synthetic_model();

    // Write tokenizer (minimal — just needs to exist)
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    // Build with extract level All
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizers::Tokenizer::from_bytes(tok_json).unwrap(),
        "test/synthetic",
        &dir,
        5,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .unwrap();

    // Verify files exist
    assert!(dir.join("gate_vectors.bin").exists());
    assert!(dir.join("embeddings.bin").exists());
    assert!(dir.join("down_meta.bin").exists());
    assert!(
        dir.join("down_meta.bin").exists(),
        "binary down_meta should be written during extract"
    );
    assert!(dir.join("index.json").exists());
    assert!(dir.join("attn_weights.bin").exists());
    assert!(dir.join("up_weights.bin").exists());
    assert!(dir.join("down_weights.bin").exists());
    assert!(dir.join("norms.bin").exists());
    assert!(dir.join(LM_HEAD_BIN).exists());
    assert!(dir.join("weight_manifest.json").exists());

    // Binary down_meta should be non-empty (JSONL no longer written)
    let bin_size = std::fs::metadata(dir.join("down_meta.bin")).unwrap().len();
    assert!(bin_size > 0, "binary down_meta should be non-empty");
    assert!(
        !dir.join("down_meta.jsonl").exists(),
        "JSONL should not be written during extract"
    );

    // Verify config
    let config = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(config.version, 2);
    assert_eq!(config.model, "test/synthetic");
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, 8);
    assert_eq!(config.intermediate_size, 4);
    assert!(config.has_model_weights);
    assert_eq!(config.dtype, larql_vindex::StorageDtype::F32);
    assert!(config.source.is_some());
    // layer_bands may be None for tiny models (< 8 layers)

    // Load and query
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let index = larql_vindex::VectorIndex::load_vindex(&dir, &mut lcb).unwrap();
    assert_eq!(index.num_layers, 2);
    assert_eq!(index.total_gate_vectors(), 8); // 2 layers × 4 features

    // KNN should work
    let query = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = index.gate_knn(0, &query, 2);
    assert!(!hits.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn extract_synthetic_model_f16() {
    let dir = std::env::temp_dir().join("larql_test_extract_f16");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let weights = make_synthetic_model();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizers::Tokenizer::from_bytes(tok_json).unwrap(),
        "test/synthetic-f16",
        &dir,
        5,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F16,
        &mut cb,
    )
    .unwrap();

    // Verify both down_meta formats written
    assert!(
        dir.join("down_meta.bin").exists(),
        "binary down_meta should be written during f16 extract"
    );
    assert!(dir.join("down_meta.bin").exists());

    // Verify f16 files are smaller
    let gate_size = std::fs::metadata(dir.join("gate_vectors.bin"))
        .unwrap()
        .len();
    // 2 layers × 4 features × 8 hidden × 2 bytes = 128 bytes (f16)
    // vs 256 bytes (f32)
    assert_eq!(gate_size, 128);

    let embed_size = std::fs::metadata(dir.join("embeddings.bin")).unwrap().len();
    // 16 vocab × 8 hidden × 2 bytes = 256 bytes (f16)
    assert_eq!(embed_size, 256);

    // Config should record f16
    let config = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(config.dtype, larql_vindex::StorageDtype::F16);

    // Load should decode f16 → f32 transparently
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let index = larql_vindex::VectorIndex::load_vindex(&dir, &mut lcb).unwrap();
    assert_eq!(index.total_gate_vectors(), 8);

    // KNN should still work (f16 precision is sufficient)
    let query = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = index.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, 0); // feature 0 responds to dim 0

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn extract_then_load_weights_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_weight_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let weights = make_synthetic_model();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizers::Tokenizer::from_bytes(tok_json).unwrap(),
        "test/weight-rt",
        &dir,
        5,
        larql_vindex::ExtractLevel::All,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .unwrap();

    // Load weights back
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let loaded = larql_vindex::load_model_weights(&dir, &mut lcb).unwrap();

    // Verify dimensions match
    assert_eq!(loaded.num_layers, 2);
    assert_eq!(loaded.hidden_size, 8);
    assert_eq!(loaded.intermediate_size, 4);

    // Verify gate vectors round-tripped (loaded from gate_vectors.bin)
    let gate_key = loaded.arch.ffn_gate_key(0);
    let gate = loaded.tensors.get(&gate_key).unwrap();
    assert_eq!(gate.shape(), &[4, 8]);
    assert!((gate[[0, 0]] - 1.0).abs() < 0.01); // layer 0, feature 0, dim 0

    // Verify up/down from split files
    let up_key = loaded.arch.ffn_up_key(0);
    assert!(loaded.tensors.contains_key(&up_key));
    let down_key = loaded.arch.ffn_down_key(0);
    assert!(loaded.tensors.contains_key(&down_key));

    // Verify attention weights
    let q_key = loaded.arch.attn_q_key(0);
    assert!(loaded.tensors.contains_key(&q_key));

    // Verify norms
    assert!(!loaded.vectors.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn extract_mutate_reload_verifies_mutation() {
    let dir = std::env::temp_dir().join("larql_test_extract_mutate");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let weights = make_synthetic_model();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizers::Tokenizer::from_bytes(tok_json).unwrap(),
        "test/mutate",
        &dir,
        5,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .unwrap();

    // Load, mutate, save
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(&dir, &mut lcb).unwrap();

    // Insert a new feature
    let gate_vec = ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.0]);
    let slot = index.find_free_feature(0).unwrap();
    index.set_gate_vector(0, slot, &gate_vec);
    index.set_feature_meta(0, slot, make_meta("INSERTED", 999, 0.99));

    // Save back — save_gate_vectors writes a new file, which is safe because
    // set_gate_vector promoted layer 0 to heap. But other layers still mmap the
    // old file. Drop the index after save to release the mmap cleanly.
    index.save_gate_vectors(&dir).unwrap();
    index.save_down_meta(&dir).unwrap();
    drop(index);

    // Reload and verify mutation persisted (binary format round-trip)
    let index2 = larql_vindex::VectorIndex::load_vindex(&dir, &mut lcb).unwrap();
    let meta = index2.feature_meta(0, slot).unwrap();
    assert_eq!(meta.top_token_id, 999);
    assert!((meta.c_score - 0.99).abs() < 0.01);

    // KNN should find the inserted feature for dim 7
    let query = ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let hits = index2.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, slot);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn extract_with_patches_bake_down() {
    let dir = std::env::temp_dir().join("larql_test_extract_patch");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let weights = make_synthetic_model();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex(
        &weights,
        &tokenizers::Tokenizer::from_bytes(tok_json).unwrap(),
        "test/patch",
        &dir,
        5,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        &mut cb,
    )
    .unwrap();

    // Load base
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let base = larql_vindex::VectorIndex::load_vindex(&dir, &mut lcb).unwrap();

    // Create and apply a patch
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test/patch".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some("test patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "PATCHED".into(),
                top_token_id: 888,
                c_score: 5.0,
            }),
        }],
    };

    let mut patched = larql_vindex::PatchedVindex::new(base);
    patched.apply_patch(patch);

    // Verify patch applied
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "PATCHED");

    // Bake down to new index
    let baked = patched.bake_down();
    assert_eq!(baked.feature_meta(0, 0).unwrap().top_token, "PATCHED");
    assert_eq!(baked.total_gate_vectors(), 8);

    let _ = std::fs::remove_dir_all(&dir);
}

// ═══════════════════════════════════════════════════════════════
// GGUF tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn gguf_key_normalization() {
    let key = larql_models::loading::gguf::normalize_gguf_key("blk.5.attn_q.weight");
    assert_eq!(key, "layers.5.self_attn.q_proj.weight");

    let key = larql_models::loading::gguf::normalize_gguf_key("blk.0.ffn_gate.weight");
    assert_eq!(key, "layers.0.mlp.gate_proj.weight");

    let key = larql_models::loading::gguf::normalize_gguf_key("token_embd.weight");
    assert_eq!(key, "embed_tokens.weight");

    let key = larql_models::loading::gguf::normalize_gguf_key("output.weight");
    assert_eq!(key, "lm_head.weight");
}

#[test]
fn gguf_config_from_metadata() {
    use larql_models::loading::gguf::{GgufFile, GgufValue};
    let gguf = GgufFile {
        metadata: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "general.architecture".into(),
                GgufValue::String("llama".into()),
            );
            m.insert("llama.embedding_length".into(), GgufValue::U32(4096));
            m.insert("llama.block_count".into(), GgufValue::U32(32));
            m.insert("llama.feed_forward_length".into(), GgufValue::U32(11008));
            m.insert("llama.attention.head_count".into(), GgufValue::U32(32));
            m.insert("llama.attention.head_count_kv".into(), GgufValue::U32(32));
            m.insert("llama.attention.key_length".into(), GgufValue::U32(128));
            m.insert("llama.rope.freq_base".into(), GgufValue::F32(10000.0));
            m
        },
        tensor_infos: vec![],
        data_offset: 0,
        path: std::path::PathBuf::new(),
    };
    let config = gguf.to_config_json();
    assert_eq!(config["model_type"], "llama");
    assert_eq!(config["hidden_size"], 4096);
    assert_eq!(config["num_hidden_layers"], 32);
    assert_eq!(config["intermediate_size"], 11008);
}

// ═══════════════════════════════════════════════════════════════
// PatchedVindex insert/delete/gate_knn tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn patched_vindex_insert_feature() {
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    patched.insert_feature(
        0,
        2,
        vec![0.0, 0.0, 0.0, 1.0],
        make_meta("Canberra", 99, 0.8),
    );
    assert_eq!(patched.feature_meta(0, 2).unwrap().top_token, "Canberra");
    assert_eq!(patched.num_overrides(), 1);
    // Base unchanged
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");
}

#[test]
fn patched_vindex_delete_feature() {
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    patched.delete_feature(0, 0);
    assert!(patched.feature_meta(0, 0).is_none());
    // Other features at layer 0 remain
    assert_eq!(patched.feature_meta(0, 1).unwrap().top_token, "French");
}

#[test]
fn patched_vindex_gate_knn_includes_inserts() {
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    patched.insert_feature(
        0,
        2,
        vec![0.0, 0.0, 0.0, 100.0],
        make_meta("Inserted", 55, 5.0),
    );
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = patched.gate_knn(0, &query, 5);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0, 2); // inserted feature should dominate
}

#[test]
fn patched_vindex_gate_knn_excludes_deletes() {
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    patched.delete_feature(0, 0); // delete Paris
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 5);
    assert!(hits.iter().all(|(f, _)| *f != 0)); // Paris should not appear
}

#[test]
fn patched_vindex_bake_down_preserves() {
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    patched.insert_feature(0, 2, vec![0.0, 0.0, 0.0, 1.0], make_meta("New", 77, 3.0));
    patched.delete_feature(1, 0);

    let baked = patched.bake_down();
    assert_eq!(baked.feature_meta(0, 2).unwrap().top_token, "New");
    assert!(baked.feature_meta(1, 0).is_none());
    assert_eq!(baked.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ═══════════════════════════════════════════════════════════════
// Vindexfile parse + build test
// ═══════════════════════════════════════════════════════════════

#[test]
fn vindexfile_parse_and_build() {
    let base_dir = std::env::temp_dir().join("larql_test_vindexfile_base");
    let _ = std::fs::remove_dir_all(&base_dir);
    std::fs::create_dir_all(&base_dir).unwrap();

    // Save a base vindex (with tokenizer for binary down_meta loading)
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(base_dir.join("tokenizer.json"), tok_json).unwrap();

    let index = test_index();
    let mut config = VindexConfig {
        version: 2,
        model: "test/vindexfile".into(),
        family: "llama".into(),
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 10,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        has_model_weights: false,
        layer_bands: None,
        layers: vec![],
        down_top_k: 5,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    index.save_vindex(&base_dir, &mut config).unwrap();

    // Create a patch
    let patch_dir = std::env::temp_dir().join("larql_test_vindexfile_patch");
    let _ = std::fs::remove_dir_all(&patch_dir);
    std::fs::create_dir_all(&patch_dir).unwrap();

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test/vindexfile".into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some("test".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Update {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                top_token: "PATCHED".into(),
                top_token_id: 999,
                c_score: 9.0,
            }),
        }],
    };
    let patch_path = patch_dir.join("test.vlp");
    patch.save(&patch_path).unwrap();

    // Build from Vindexfile
    let vf_content = format!(
        "FROM {}\nPATCH {}\n",
        base_dir.display(),
        patch_path.display()
    );
    let vf = larql_vindex::vindexfile::parse_vindexfile_str(&vf_content).unwrap();
    let result = larql_vindex::build_from_vindexfile(&vf, None, &std::env::temp_dir()).unwrap();

    // Patched feature should have the update applied
    assert_eq!(result.index.feature_meta(0, 0).unwrap().top_token_id, 999);
    assert!((result.index.feature_meta(0, 0).unwrap().c_score - 9.0).abs() < 0.01);
    // Unpatched feature should remain (token_id preserved through binary round-trip)
    assert_eq!(result.index.feature_meta(0, 1).unwrap().top_token_id, 101);
    assert_eq!(result.layers.len(), 2);

    let _ = std::fs::remove_dir_all(&base_dir);
    let _ = std::fs::remove_dir_all(&patch_dir);
}

// ═══════════════════════════════════════════════════════════════
// HuggingFace path tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn hf_path_detection() {
    assert!(larql_vindex::is_hf_path(
        "hf://chrishayuk/gemma-3-4b-it-vindex"
    ));
    assert!(larql_vindex::is_hf_path("hf://user/repo@v2.0"));
    assert!(!larql_vindex::is_hf_path("./local.vindex"));
    assert!(!larql_vindex::is_hf_path("/absolute/path"));
    assert!(!larql_vindex::is_hf_path("google/gemma-3-4b-it"));
}

#[test]
fn hf_path_with_revision() {
    let path = "hf://chrishayuk/gemma-3-4b-it-vindex@v2.0";
    assert!(larql_vindex::is_hf_path(path));
    let stripped = path.strip_prefix("hf://").unwrap();
    let (repo, rev) = stripped.split_once('@').unwrap();
    assert_eq!(repo, "chrishayuk/gemma-3-4b-it-vindex");
    assert_eq!(rev, "v2.0");
}

#[test]
fn hf_resolve_invalid_path_fails() {
    // Non-hf:// path should fail
    let result = larql_vindex::resolve_hf_vindex("./not-an-hf-path");
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════
// Streaming extraction test
// ═══════════════════════════════════════════════════════════════

#[test]
fn streaming_extract_from_safetensors() {
    // Create a minimal safetensors model directory with synthetic weights
    let model_dir = std::env::temp_dir().join("larql_test_streaming_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
    std::fs::create_dir_all(&model_dir).unwrap();

    // Write config.json
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rope_theta": 10000.0,
        "vocab_size": 16,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    // Write a minimal safetensors file with gate + down + embed tensors
    let mut tensors: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    // Embeddings: 16 × 8
    let embed: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    tensors.insert("model.embed_tokens.weight".into(), embed);
    metadata.push(("model.embed_tokens.weight".into(), vec![16, 8]));

    // Per-layer: gate (4×8), down (8×4)
    for layer in 0..2 {
        let gate: Vec<f32> = (0..32).map(|i| (i as f32 + layer as f32) * 0.1).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.gate_proj.weight"), gate);
        metadata.push((
            format!("model.layers.{layer}.mlp.gate_proj.weight"),
            vec![4, 8],
        ));

        let down: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();
        tensors.insert(format!("model.layers.{layer}.mlp.down_proj.weight"), down);
        metadata.push((
            format!("model.layers.{layer}.mlp.down_proj.weight"),
            vec![8, 4],
        ));
    }

    // Build safetensors file
    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    // Write tokenizer
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();

    // Run streaming extraction
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let mut cb = larql_vindex::SilentBuildCallbacks;

    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/streaming",
        &output_dir,
        5,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        larql_vindex::QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // Verify output
    assert!(output_dir.join("gate_vectors.bin").exists());
    assert!(output_dir.join("embeddings.bin").exists());
    assert!(output_dir.join("down_meta.bin").exists());
    assert!(output_dir.join("index.json").exists());
    assert!(output_dir.join("tokenizer.json").exists());

    let config = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.model, "test/streaming");

    // Load and verify KNN works
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut lcb).unwrap();
    assert_eq!(index.total_gate_vectors(), 8); // 2 layers × 4 features
    assert!(index.is_mmap()); // should use mmap mode

    let query = ndarray::Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let hits = index.gate_knn(0, &query, 2);
    assert!(!hits.is_empty());

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

// ─── streaming_extract with QuantFormat::Q4K ────────────────────
//
// End-to-end coverage for `write_model_weights_kquant`:
//   - Manifest shape: attn has 4 entries per layer, FFN has 3;
//     V and down carry Q6_K, everything else Q4_K.
//   - Offsets tile start-to-end with no gaps.
//   - `config.quant = Q4K` and `has_model_weights = true` land in
//     `index.json` so loaders can dispatch without sniffing files.
//   - The non-Q4 `attn_weights.bin` / `interleaved.bin` are absent.
#[test]
fn streaming_extract_q4k_from_safetensors() {
    use larql_vindex::QuantFormat;
    use std::collections::HashMap;

    let model_dir = std::env::temp_dir().join("larql_test_streaming_q4k_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_q4k_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
    std::fs::create_dir_all(&model_dir).unwrap();

    // Small llama config — dims chosen so each tensor pads to exactly
    // one 256-element Q4_K/Q6_K super-block (256 elems = 2×128 or 8×32
    // or 16×16). Hidden=8 keeps padding overhead visible; the padder
    // zero-fills to the next 256-multiple.
    let hidden = 8usize;
    let intermediate = 4usize;
    let num_layers = 2usize;
    let vocab = 16usize;

    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "rope_theta": 10000.0,
        "vocab_size": vocab,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    let push = |tensors: &mut HashMap<String, Vec<f32>>,
                metadata: &mut Vec<(String, Vec<usize>)>,
                name: &str,
                shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    push(
        &mut tensors,
        &mut metadata,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.norm.weight",
        vec![hidden],
    );

    for layer in 0..num_layers {
        let lp = format!("model.layers.{layer}");
        // Attention: Q/K/V/O all [hidden, hidden]
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        // FFN: gate [inter, hidden], up [inter, hidden], down [hidden, inter]
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        // Norms
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    // Run with QuantFormat::Q4K — also verifies the Browse-level auto-
    // promotion to "all" that the streaming extractor applies when
    // quant != None.
    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/streaming-q4k",
        &output_dir,
        5,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // ── File layout ──
    //
    // Writers emit the new kquant-canonical filenames. Legacy q4k-named
    // files must NOT be written (they're read-only back-compat fallbacks).
    assert!(output_dir.join("attn_weights_kquant.bin").exists());
    assert!(output_dir
        .join("attn_weights_kquant_manifest.json")
        .exists());
    assert!(output_dir.join("interleaved_kquant.bin").exists());
    assert!(output_dir.join("interleaved_kquant_manifest.json").exists());
    assert!(output_dir.join("norms.bin").exists());
    assert!(output_dir.join("weight_manifest.json").exists());
    assert!(output_dir.join("index.json").exists());
    assert!(
        !output_dir.join("attn_weights_q4k.bin").exists(),
        "writer must NOT emit the legacy q4k filename"
    );
    assert!(
        !output_dir.join("interleaved_q4k.bin").exists(),
        "writer must NOT emit the legacy q4k filename"
    );

    // k-quant path writes its own filenames; the non-quantised names should be absent.
    assert!(
        !output_dir.join("attn_weights.bin").exists(),
        "k-quant path should not emit attn_weights.bin"
    );

    // ── Config schema ──
    let cfg = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(cfg.num_layers, num_layers);
    assert_eq!(cfg.quant, QuantFormat::Q4K, "config.quant must be Q4K");
    assert!(
        cfg.has_model_weights,
        "config.has_model_weights must flip true"
    );

    // ── attn manifest ──
    let attn_manifest_json =
        std::fs::read_to_string(output_dir.join("attn_weights_kquant_manifest.json")).unwrap();
    let attn_entries: Vec<serde_json::Value> = serde_json::from_str(&attn_manifest_json).unwrap();

    // 4 tensors (Q, K, V, O) × num_layers
    assert_eq!(
        attn_entries.len(),
        num_layers * 4,
        "attn manifest should have 4N entries (Q/K/V/O per layer)"
    );

    // Per-layer slot order: Q=Q4_K, K=Q4_K, V=Q6_K, O=Q4_K.
    // Offsets must chain start-to-end with no gaps.
    let mut expected_offset: u64 = 0;
    for (i, entry) in attn_entries.iter().enumerate() {
        let slot = i % 4;
        let format = entry["format"].as_str().unwrap();
        let expected_format = if slot == 2 { "Q6_K" } else { "Q4_K" };
        assert_eq!(
            format, expected_format,
            "entry {i} slot {slot}: expected {expected_format}, got {format}"
        );
        let offset = entry["offset"].as_u64().unwrap();
        assert_eq!(offset, expected_offset, "offsets must tile with no gaps");
        let length = entry["length"].as_u64().unwrap();
        assert!(length > 0, "each entry must carry bytes");
        expected_offset += length;
    }

    // ── interleaved (FFN) manifest ──
    let ff_manifest_json =
        std::fs::read_to_string(output_dir.join("interleaved_kquant_manifest.json")).unwrap();
    let ff_entries: Vec<serde_json::Value> = serde_json::from_str(&ff_manifest_json).unwrap();

    // 3 tensors (gate, up, down) × num_layers
    assert_eq!(
        ff_entries.len(),
        num_layers * 3,
        "FFN manifest should have 3N entries (gate/up/down per layer)"
    );

    // Per-layer slot order: gate=Q4_K, up=Q4_K, down=Q6_K.
    let mut expected_offset: u64 = 0;
    for (i, entry) in ff_entries.iter().enumerate() {
        let slot = i % 3;
        let format = entry["format"].as_str().unwrap();
        let expected_format = if slot == 2 { "Q6_K" } else { "Q4_K" };
        assert_eq!(
            format, expected_format,
            "FFN entry {i} slot {slot}: expected {expected_format}, got {format}"
        );
        let offset = entry["offset"].as_u64().unwrap();
        assert_eq!(
            offset, expected_offset,
            "FFN offsets must tile with no gaps"
        );
        expected_offset += entry["length"].as_u64().unwrap();
    }

    // ── manifest byte counts match file sizes ──
    let attn_bytes = std::fs::metadata(output_dir.join("attn_weights_kquant.bin"))
        .unwrap()
        .len();
    let attn_manifest_total: u64 = attn_entries
        .iter()
        .map(|e| e["length"].as_u64().unwrap())
        .sum();
    assert_eq!(
        attn_bytes, attn_manifest_total,
        "attn_weights_kquant.bin size must equal sum of manifest lengths"
    );

    let ff_bytes = std::fs::metadata(output_dir.join("interleaved_kquant.bin"))
        .unwrap()
        .len();
    let ff_manifest_total: u64 = ff_entries
        .iter()
        .map(|e| e["length"].as_u64().unwrap())
        .sum();
    assert_eq!(
        ff_bytes, ff_manifest_total,
        "interleaved_kquant.bin size must equal sum of manifest lengths"
    );

    // ── load_model_weights on a Q4K vindex must surface a clear error ──
    // The float-weight loader can't reconstruct a ModelWeights struct
    // from Q4_K/Q6_K blocks; callers must go through
    // `VectorIndex::load_attn_kquant` / `load_interleaved_kquant` instead.
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    match larql_vindex::load_model_weights(&output_dir, &mut lcb) {
        Ok(_) => panic!("load_model_weights on a Q4K vindex must error"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("quantised") && msg.contains("load_attn_kquant"),
                "expected quant-dispatch error, got: {msg}"
            );
        }
    }

    // ── VectorIndex::load_attn_kquant + load_interleaved_kquant must read
    //     back what the writer emitted ──
    let mut index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut lcb).unwrap();
    index.load_attn_kquant(&output_dir).unwrap();
    index.load_interleaved_kquant(&output_dir).unwrap();
    assert!(
        index.has_interleaved_kquant(),
        "interleaved Q4K should be loaded"
    );
    // Layer 0 attn slices: [Q/Q4_K, K/Q4_K, V/Q6_K, O/Q4_K]
    let slices = index.attn_kquant_layer_data(0).expect("layer 0 attn data");
    assert_eq!(slices[0].1, "Q4_K", "Q slot format");
    assert_eq!(slices[1].1, "Q4_K", "K slot format");
    assert_eq!(slices[2].1, "Q6_K", "V slot format");
    assert_eq!(slices[3].1, "Q4_K", "O slot format");

    // ── Write-side correctness: dequantize the bytes the writer emitted
    //     and confirm they round-trip back to the source within block
    //     error tolerance. Proves the writer's manifest → data
    //     correspondence is correct (not just a shape assertion).
    //
    // Source data for every tensor: (0..n).map(|i| i as f32 * 0.01).
    // Q/K/V/O are hidden×hidden = 64 elems each, zero-padded to 256.
    //
    // Block-level error on a 64-value-then-192-zero-padded 256-value
    // super-block: ~0.02 for Q4_K and ~0.006 for Q6_K on this linear
    // ramp. Use 0.03 / 0.01 as ceilings — loose enough for the
    // quantiser's block allocation on this padding-heavy synthetic
    // case, tight enough to catch a manifest that points at the wrong
    // bytes (which would produce garbage orders of magnitude worse).
    let expected: Vec<f32> = (0..(hidden * hidden)).map(|i| (i as f32) * 0.01).collect();

    // The writer's `pad_rows_to_256` zero-extends each row from `hidden`
    // to 256 cols before quantising, so the dequantised output is a
    // [hidden × 256] padded matrix, not a flat copy of `expected`.
    // Map (row, col) of the original to the padded layout for comparison.
    let padded_cols = 256;
    let padded_at = |row: usize, col: usize| -> usize { row * padded_cols + col };

    let q_dequant =
        larql_models::quant::ggml::dequantize_q4_k(slices[0].0, hidden * padded_cols).unwrap();
    for row in 0..hidden {
        for col in 0..hidden {
            let i = row * hidden + col;
            let v = expected[i];
            let got = q_dequant[padded_at(row, col)];
            assert!(
                (got - v).abs() < 0.03,
                "Q[r{row} c{col}] round-trip diverged: got {got}, expected {v}",
            );
        }
        // Per-row zero pad: cols [hidden..256] should dequantise near zero
        // (within block error — the row's value range sets the scale).
        for col in hidden..padded_cols {
            let got = q_dequant[padded_at(row, col)];
            assert!(
                got.abs() < 0.05,
                "Q padding[r{row} c{col}] expected ~0, got {got}",
            );
        }
    }

    let v_dequant =
        larql_models::quant::ggml::dequantize_q6_k(slices[2].0, hidden * padded_cols).unwrap();
    for row in 0..hidden {
        for col in 0..hidden {
            let i = row * hidden + col;
            let v = expected[i];
            let got = v_dequant[padded_at(row, col)];
            assert!(
                (got - v).abs() < 0.01,
                "V[r{row} c{col}] round-trip diverged (Q6_K): got {got}, expected {v}",
            );
        }
    }

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

#[test]
fn quant_block_format_serde_roundtrip() {
    // The manifest format strings are load-bearing — llama.cpp / Ollama
    // expect the literal "Q4_K" and "Q6_K" on the wire. The enum uses
    // #[serde(rename)] to keep those strings; a future refactor must
    // not drift to e.g. "Q4K" without also updating every reader.
    use larql_vindex::format::weights::write_kquant::QuantBlockFormat;
    let q4 = serde_json::to_string(&QuantBlockFormat::Q4K).unwrap();
    let q6 = serde_json::to_string(&QuantBlockFormat::Q6K).unwrap();
    assert_eq!(q4, "\"Q4_K\"");
    assert_eq!(q6, "\"Q6_K\"");

    let parsed: QuantBlockFormat = serde_json::from_str("\"Q4_K\"").unwrap();
    assert_eq!(parsed, QuantBlockFormat::Q4K);
    let parsed: QuantBlockFormat = serde_json::from_str("\"Q6_K\"").unwrap();
    assert_eq!(parsed, QuantBlockFormat::Q6K);
}

// ═══════════════════════════════════════════════════════════════
// GateIndex trait tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn gate_index_trait_on_vector_index() {
    let index = test_index();
    let gi: &dyn GateIndex = &index;

    // num_features
    assert_eq!(gi.num_features(0), 3);
    assert_eq!(gi.num_features(1), 3);

    // feature_meta
    assert_eq!(gi.feature_meta(0, 0).unwrap().top_token, "Paris");
    assert_eq!(gi.feature_meta(1, 0).unwrap().top_token, "Berlin");
    assert!(gi.feature_meta(0, 99).is_none());

    // gate_knn
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = gi.gate_knn(0, &query, 3);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0, 0); // feature 0 responds to dim 0
}

#[test]
fn gate_index_trait_on_patched_vindex() {
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    // Insert a strong feature that should dominate KNN
    patched.insert_feature(
        0,
        2,
        vec![0.0, 0.0, 0.0, 100.0],
        make_meta("Inserted", 55, 5.0),
    );
    // Delete feature 0 (Paris)
    patched.delete_feature(0, 0);

    let gi: &dyn GateIndex = &patched;

    // feature_meta sees the insert
    assert_eq!(gi.feature_meta(0, 2).unwrap().top_token, "Inserted");
    // feature_meta sees the delete
    assert!(gi.feature_meta(0, 0).is_none());
    // base features still visible
    assert_eq!(gi.feature_meta(0, 1).unwrap().top_token, "French");

    // gate_knn sees the inserted feature
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = gi.gate_knn(0, &query, 5);
    assert_eq!(hits[0].0, 2); // inserted feature dominates
                              // gate_knn excludes the deleted feature
    assert!(hits.iter().all(|(f, _)| *f != 0));
}

#[test]
fn gate_index_patched_walk_sees_mutations() {
    // This is the core test: walk (used by WALK and DESCRIBE) sees patches.
    let index = test_index();
    let mut patched = larql_vindex::PatchedVindex::new(index);

    // Before mutation: walk should find "Paris" at layer 0
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace_before = patched.walk(&query, &[0], 5);
    let layer0_before = &trace_before.layers[0].1;
    assert!(layer0_before.iter().any(|h| h.meta.top_token == "Paris"));

    // Insert a dominating feature
    patched.insert_feature(
        0,
        2,
        vec![100.0, 0.0, 0.0, 0.0],
        make_meta("NewCity", 77, 9.0),
    );
    // Delete Paris
    patched.delete_feature(0, 0);

    // After mutation: walk should find "NewCity", not "Paris"
    let trace_after = patched.walk(&query, &[0], 5);
    let layer0_after = &trace_after.layers[0].1;
    assert!(layer0_after.iter().any(|h| h.meta.top_token == "NewCity"));
    assert!(!layer0_after.iter().any(|h| h.meta.top_token == "Paris"));
}

#[test]
fn gate_index_dynamic_dispatch_matches_direct() {
    // Verify trait dispatch produces identical results to direct method calls
    let index = test_index();
    let gi: &dyn GateIndex = &index;

    let query = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);

    let direct = index.gate_knn(0, &query, 3);
    let via_trait = gi.gate_knn(0, &query, 3);

    assert_eq!(direct.len(), via_trait.len());
    for (d, t) in direct.iter().zip(via_trait.iter()) {
        assert_eq!(d.0, t.0);
        assert!((d.1 - t.1).abs() < 1e-6);
    }
}

// ══════════════════════════════════════════════════════════════
// GATE WALK (BLAS gemv path)
// ══════════════════════════════════════════════════════════════

#[test]
fn gate_walk_matches_gate_knn() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.5, 0.0, 0.0]);

    let knn = idx.gate_knn(0, &query, 3);
    let walk = idx.gate_walk(0, &query, 3).unwrap();

    // Same features, same order
    assert_eq!(knn.len(), walk.len());
    for (k, w) in knn.iter().zip(walk.iter()) {
        assert_eq!(k.0, w.0, "feature index mismatch");
        assert!(
            (k.1 - w.1).abs() < 1e-5,
            "score mismatch: {} vs {}",
            k.1,
            w.1
        );
    }
}

#[test]
fn gate_walk_returns_none_for_empty_layer() {
    let idx = VectorIndex::new(vec![None], vec![None], 1, 4);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    assert!(idx.gate_walk(0, &query, 5).is_none());
}

// ══════════════════════════════════════════════════════════════
// Q4 GATE KNN
// ══════════════════════════════════════════════════════════════

#[test]
fn gate_knn_q4_produces_results() {
    let hidden = 256;
    let features = 64;
    let gate: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let gate_arr = Array2::from_shape_vec((features, hidden), gate.clone()).unwrap();
    let idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);

    let q4_data = larql_compute::cpu::q4::quantize_q4_0(&gate);
    let backend = larql_compute::cpu_backend();
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.01).sin());

    // Simulate Q4 scoring path (same logic as gate_knn_q4)
    let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(query.as_slice().unwrap());
    let scores = backend
        .q4_matvec(&q4_data, &q8_x, &q8_scales, features, hidden)
        .unwrap();
    assert_eq!(scores.len(), features);
    assert!(
        scores.iter().any(|&v| v.abs() > 0.01),
        "Q4 should produce nonzero scores"
    );

    // f32 KNN for comparison
    let f32_hits = idx.gate_knn(0, &query, 5);
    assert_eq!(f32_hits.len(), 5);

    // Q4 top-1 should usually match f32 top-1 (same dominant feature)
    let mut q4_indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    q4_indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    assert_eq!(
        q4_indexed[0].0, f32_hits[0].0,
        "Q4 top-1 should match f32 top-1"
    );
}

#[test]
fn gate_knn_q4_method_works() {
    use larql_compute::cpu::q4::quantize_q4_0;

    let hidden = 256;
    let features = 64;
    let gate_f32: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&gate_f32);
    let gate_arr = Array2::from_shape_vec((features, hidden), gate_f32).unwrap();

    // Build index with gate vectors and manually set Q4 data
    let mut idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);

    // Save Q4 to temp file, then load
    let dir = std::env::temp_dir().join("larql_test_q4_gate");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("gate_vectors_q4.bin"), &q4_data).unwrap();
    idx.load_gate_vectors_q4(&dir).unwrap();
    assert!(idx.has_gate_q4());

    // Now call gate_knn_q4
    let backend = larql_compute::cpu_backend();
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.01).sin());
    let hits = idx.gate_knn_q4(0, &query, 5, backend.as_ref()).unwrap();
    assert_eq!(hits.len(), 5);
    assert!(
        hits[0].1.abs() > hits[4].1.abs(),
        "results should be sorted by abs score"
    );

    // Compare with f32 KNN
    let f32_hits = idx.gate_knn(0, &query, 5);
    assert_eq!(hits[0].0, f32_hits[0].0, "Q4 top-1 should match f32 top-1");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gate_q4_data_returns_correct_bytes() {
    use larql_compute::cpu::q4::quantize_q4_0;

    let hidden = 256;
    let features = 32;
    let gate_f32: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&gate_f32);
    let gate_arr = Array2::from_shape_vec((features, hidden), gate_f32).unwrap();

    let mut idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);

    let dir = std::env::temp_dir().join("larql_test_q4_data");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("gate_vectors_q4.bin"), &q4_data).unwrap();
    idx.load_gate_vectors_q4(&dir).unwrap();

    let loaded = idx.gate_q4_data(0).unwrap();
    assert_eq!(loaded.len(), q4_data.len());
    assert_eq!(loaded, q4_data.as_slice());

    // Out of range returns None
    assert!(idx.gate_q4_data(99).is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// LM HEAD KNN
// ══════════════════════════════════════════════════════════════

#[test]
fn lm_head_knn_returns_top_k() {
    let hidden = 4;
    let vocab = 8;

    // Build a small lm_head: [vocab, hidden]
    let mut lm_head = vec![0.0f32; vocab * hidden];
    // Token 0 responds to dim 0
    lm_head[0] = 10.0;
    // Token 3 responds to dim 1
    lm_head[3 * hidden + 1] = 5.0;
    // Token 7 responds to dim 2
    lm_head[7 * hidden + 2] = 3.0;

    // Write to temp file
    let dir = std::env::temp_dir().join("larql_test_lm_head");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let lm_bytes: Vec<u8> = lm_head.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(dir.join(LM_HEAD_BIN), &lm_bytes).unwrap();

    let mut idx = VectorIndex::new(vec![None], vec![None], 1, hidden);
    idx.load_lm_head(&dir).unwrap();
    assert!(idx.has_lm_head());

    // Query aligned with dim 0 → token 0 should win
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.lm_head_knn(&query, 3);
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].0, 0, "token 0 should be top-1 for dim 0 query");
    assert!(
        hits[0].1 > hits[1].1,
        "results should be sorted by score desc"
    );

    // Query aligned with dim 1 → token 3 should win
    let query = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let hits = idx.lm_head_knn(&query, 3);
    assert_eq!(hits[0].0, 3, "token 3 should be top-1 for dim 1 query");

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// HNSW INTEGRATION
// ══════════════════════════════════════════════════════════════

#[test]
fn hnsw_enable_disable() {
    let idx = test_index();
    assert!(!idx.is_hnsw_enabled());

    idx.enable_hnsw(100);
    assert!(idx.is_hnsw_enabled());

    idx.disable_hnsw();
    assert!(!idx.is_hnsw_enabled());
}

#[test]
fn hnsw_knn_produces_valid_results() {
    let hidden = 128;
    let features = 500;
    let gate = Array2::from_shape_fn((features, hidden), |(r, c)| {
        let s = (r * hidden + c) as u64;
        let h = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (h >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0
    });
    let idx = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.1).sin());

    // HNSW via VectorIndex integration
    idx.enable_hnsw(100);
    let hnsw = idx.gate_knn(0, &query, 10);
    idx.disable_hnsw();

    // HNSW should return valid, non-empty results with features in range
    assert_eq!(hnsw.len(), 10, "HNSW should return requested top-K");
    for (feat, score) in &hnsw {
        assert!(*feat < features, "feature index out of range");
        assert!(score.is_finite(), "score should be finite");
    }
    // Results should be sorted by absolute score descending
    for w in hnsw.windows(2) {
        assert!(
            w[0].1.abs() >= w[1].1.abs(),
            "results should be sorted by |score| desc"
        );
    }
}

// ══════════════════════════════════════════════════════════════
// ADAPTIVE RESIDENCY
// ══════════════════════════════════════════════════════════════

#[test]
fn residency_pin_and_evict() {
    use larql_vindex::{LayerState, ResidencyManager};

    let mut rm = ResidencyManager::new(10, 4, 256, vec![32, 32, 32, 32]);
    assert_eq!(rm.num_pinned(), 0);
    assert_eq!(rm.state(0), LayerState::Cold);

    rm.mark_q4_available();
    assert_eq!(rm.state(0), LayerState::MmapQ4);

    // Pin layer 0 (32 features × 256 hidden / 32 * 18 = 4608 bytes)
    let fake_q4 = vec![0u8; 4608];
    assert!(rm.pin_layer(0, &fake_q4));
    assert_eq!(rm.state(0), LayerState::Pinned);
    assert_eq!(rm.num_pinned(), 1);
    assert!(rm.pinned_q4(0).is_some());

    // Pin again is a no-op
    assert!(rm.pin_layer(0, &fake_q4));
    assert_eq!(rm.num_pinned(), 1);

    // Evict
    rm.evict_layer(0);
    assert_eq!(rm.state(0), LayerState::MmapQ4);
    assert_eq!(rm.num_pinned(), 0);
    assert!(rm.pinned_q4(0).is_none());
}

#[test]
fn residency_budget_enforcement() {
    use larql_vindex::ResidencyManager;

    // Budget: 5 KB = 5120 bytes. Each layer's Q4 = 4608 bytes. Can fit 1, not 2.
    let mut rm = ResidencyManager::new(0, 2, 256, vec![32, 32]);
    // 0 MB budget — nothing should pin
    let data = vec![0u8; 4608];
    assert!(!rm.pin_layer(0, &data));
    assert_eq!(rm.num_pinned(), 0);

    // Use raw bytes to test budget: 4608 bytes per layer, budget just under 2 layers
    // We need a budget in MB that fits 1 layer but not 2.
    // 4608 * 2 = 9216 bytes. Create a manager and pin with exact byte checks.
    let _rm2 = ResidencyManager::new(1, 2, 256, vec![32, 32]); // 1 MB budget
                                                               // 1 MB >> 9216 bytes, so both will fit. Instead test with large layers.
                                                               // Use features=4096 so each layer is 4096*256/32*18 = 589,824 bytes = 0.56 MB
    let big_features = 4096;
    let big_data = vec![0u8; big_features * 256 / 32 * 18]; // ~576 KB
    let mut rm3 = ResidencyManager::new(1, 3, 256, vec![big_features; 3]); // 1 MB budget
    assert!(rm3.pin_layer(0, &big_data)); // ~576 KB, fits
    assert!(!rm3.pin_layer(1, &big_data)); // ~1152 KB total, exceeds 1 MB
    assert_eq!(rm3.num_pinned(), 1);
}

#[test]
fn residency_auto_pin_fills_budget() {
    use larql_vindex::ResidencyManager;

    let layers = 8;
    let features = 32;
    let hidden = 256;
    let layer_features = vec![features; layers];
    let q4_per_layer = features * hidden / 32 * 18; // 4608 bytes

    // Budget for 4 layers
    let budget_mb = 1; // 1 MB >> 4608 * 8 = 36 KB, so all fit
    let mut rm = ResidencyManager::new(budget_mb, layers, hidden, layer_features);
    rm.mark_q4_available();

    // Record accesses — layers 2, 5 are hot
    for _ in 0..100 {
        rm.record_access(2);
    }
    for _ in 0..50 {
        rm.record_access(5);
    }

    let pinned = rm.auto_pin(|_| Some(vec![0u8; q4_per_layer]));
    assert_eq!(pinned, layers); // budget fits all

    // Hottest layers should be pinned
    assert!(rm.pinned_q4(2).is_some());
    assert!(rm.pinned_q4(5).is_some());
}

#[test]
fn residency_pin_range() {
    use larql_vindex::ResidencyManager;

    let layers = 10;
    let features = 32;
    let hidden = 256;
    let q4_per_layer = features * hidden / 32 * 18;

    let mut rm = ResidencyManager::new(1, layers, hidden, vec![features; layers]);
    rm.mark_q4_available();

    // Pin knowledge band L3-L7
    let pinned = rm.pin_range(3, 8, |_| Some(vec![0u8; q4_per_layer]));
    assert_eq!(pinned, 5);
    assert!(rm.pinned_q4(3).is_some());
    assert!(rm.pinned_q4(7).is_some());
    assert!(rm.pinned_q4(2).is_none()); // not in range
    assert!(rm.pinned_q4(8).is_none()); // not in range
}

#[test]
fn residency_summary() {
    use larql_vindex::ResidencyManager;

    let mut rm = ResidencyManager::new(1, 4, 256, vec![32; 4]);
    rm.mark_q4_available();
    rm.pin_layer(0, &vec![0u8; 4608]);

    let s = rm.summary();
    assert!(s.contains("1 pinned"));
    assert!(s.contains("3 mmap"));
    assert!(s.contains("0 cold"));
}

#[test]
fn adaptive_gate_knn_uses_pinned() {
    use larql_compute::cpu::q4::quantize_q4_0;
    use larql_vindex::ResidencyManager;

    let hidden = 256;
    let features = 64;
    let gate_f32: Vec<f32> = (0..features * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4_data = quantize_q4_0(&gate_f32);
    let gate_arr = Array2::from_shape_vec((features, hidden), gate_f32).unwrap();

    let idx = VectorIndex::new(vec![Some(gate_arr)], vec![None], 1, hidden);
    let backend = larql_compute::cpu_backend();
    let query = Array1::from_shape_fn(hidden, |i| (i as f32 * 0.01).sin());

    let mut rm = ResidencyManager::new(10, 1, hidden, vec![features]);
    rm.mark_q4_available();
    rm.pin_layer(0, &q4_data);

    // Adaptive dispatch should use pinned path
    let hits = idx.gate_knn_adaptive(0, &query, 5, &mut rm, backend.as_ref());
    assert_eq!(hits.len(), 5);

    // Should match f32 brute-force top-1
    let f32_hits = idx.gate_knn(0, &query, 5);
    assert_eq!(
        hits[0].0, f32_hits[0].0,
        "pinned Q4 top-1 should match f32 top-1"
    );
}

// ── Shared Gemma-4 PLE fixture ──────────────────────────────────
//
// Writes a Gemma-4-shaped HuggingFace model dir on disk: config.json
// with `hidden_size_per_layer_input` set (the knob
// `has_per_layer_embeddings()` keys off), tokenizer.json stub, and a
// safetensors with every tensor the extractor expects — including the
// six PLE tensors per layer plus the three globals AND the per-layer
// `layer_scalar` (Gemma-4-only, 0-D scalar surfaced as a 1-element
// vector). Shared between the Q4_K and the `--quant none` regression
// tests so they exercise the exact same fixture.
//
// Returns the populated model dir path; caller decides where to write
// the vindex output and what quant to extract with.
#[allow(clippy::type_complexity)]
fn write_gemma4_ple_fixture(
    model_dir: &std::path::Path,
    num_layers: usize,
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    ple_dim: usize,
) {
    use std::collections::HashMap;

    std::fs::create_dir_all(model_dir).unwrap();

    let config = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": hidden,
            "hidden_size_per_layer_input": ple_dim,
            "vocab_size": vocab,
            // Gemma 4 ships with a final-logit tanh softcap of 30.0. This
            // must survive extract → load; without it predict_kquant peaks
            // on the wrong token on E2B.
            "final_logit_softcapping": 30.0,
        }
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();

    let push = |tensors: &mut HashMap<String, Vec<f32>>,
                metadata: &mut Vec<(String, Vec<usize>)>,
                name: &str,
                shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    // Core Gemma 4 tensors (with the multimodal `model.language_model.` prefix
    // the arch strips on load). Attn/FFN dims kept small but 256-aligned.
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.embed_tokens.weight",
        vec![vocab, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.norm.weight",
        vec![hidden],
    );

    for layer in 0..num_layers {
        let lp = format!("model.language_model.layers.{layer}");
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![intermediate, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, intermediate],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_norm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_norm.weight"),
            vec![hidden],
        );
        // Gemma-4 per-layer scalar multiplier (`layer_scalar`). Real
        // models ship it as a 0-D scalar; we use a 1-element vector
        // because that's how `WeightSource::get_vector` surfaces it.
        // Omitting it on the writer side silently broke Gemma-4
        // inference (same family of bug as #49 for PLE).
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.layer_scalar"),
            vec![1],
        );

        // ── PLE per-layer tensors (the regression surface) ──
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.per_layer_input_gate.weight"),
            vec![ple_dim, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.per_layer_projection.weight"),
            vec![hidden, ple_dim],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_per_layer_input_norm.weight"),
            vec![hidden],
        );
    }

    // ── PLE global tensors ──
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.per_layer_model_projection.weight",
        vec![ple_dim * num_layers, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.embed_tokens_per_layer.weight",
        vec![vocab, ple_dim * num_layers],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.language_model.per_layer_projection_norm.weight",
        vec![ple_dim],
    );

    // Serialise as f32 safetensors.
    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
}

// ─── PLE tensors survive Q4_K extract → load round-trip ─────────
//
// Regression test for the Gemma 4 E2B "predict returns garbage on
// Q4K vindex" bug: the extractor used to drop the six Per-Layer
// Embedding tensors, so `precompute_per_layer_inputs` silently
// returned an empty Vec and PLE was never applied. Extraction now
// writes `ple_weights.bin` (Q4_K-packed tensors) plus the two small
// PLE norms into norms.bin. This test builds a Gemma 4-shaped
// synthetic safetensors, runs the real extract pipeline, loads via
// `load_model_weights_kquant`, and asserts every PLE tensor is back in
// `weights.tensors` / `weights.vectors` with the right shape.
#[test]
fn streaming_extract_q4k_carries_ple_tensors() {
    use larql_vindex::QuantFormat;

    let model_dir = std::env::temp_dir().join("larql_test_streaming_q4k_ple_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_q4k_ple_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);

    // E2B-shaped config at a test-friendly scale. `hidden_size_per_layer_input`
    // is the knob `has_per_layer_embeddings()` keys off, so it must be present
    // AND non-zero for the extractor to hit the PLE path. Gemma 4 uses the
    // text_config wrapper; detect_from_json handles that.
    let hidden = 256usize; // multiple of 256 so Q/K/V/O skip the padder
    let intermediate = 256usize;
    let num_layers = 2usize;
    let vocab = 256usize;
    let ple_dim = 256usize;

    write_gemma4_ple_fixture(&model_dir, num_layers, hidden, intermediate, vocab, ple_dim);
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/streaming-q4k-ple",
        &output_dir,
        5,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // ── ple_weights.bin must exist and the manifest must list all 3
    //     global + (2 per-layer) PLE tensor entries as `tensor_q4k`. ──
    assert!(
        output_dir.join("ple_weights.bin").exists(),
        "Q4 extract should emit ple_weights.bin when the arch has PLE"
    );

    let manifest_json = std::fs::read_to_string(output_dir.join("weight_manifest.json")).unwrap();
    let manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest_json).unwrap();
    // PLE tensors are stored as f16 (not Q4_K) — Q4_K's per-super-block
    // calibration zeros out the non-outlier cells of embedding-style
    // tensors, compounding to garbage across Gemma 4 E2B's 35 layers.
    let ple_tensor_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "tensor_f16")
        .filter_map(|e| e["key"].as_str())
        .collect();

    // 2 global tensors (per_layer_model_projection, embed_tokens_per_layer)
    // + 2 per-layer tensors × num_layers. per_layer_projection_norm is a
    // vector and belongs in norms.bin, not here.
    assert_eq!(
        ple_tensor_keys.len(),
        2 + 2 * num_layers,
        "expected {} PLE tensor_f16 entries, got: {:?}",
        2 + 2 * num_layers,
        ple_tensor_keys
    );
    assert!(
        ple_tensor_keys.contains(&"per_layer_model_projection.weight"),
        "global model projection missing from manifest"
    );
    assert!(
        ple_tensor_keys.contains(&"embed_tokens_per_layer.weight"),
        "global per-layer embed missing from manifest"
    );

    // ── post_per_layer_input_norm + per_layer_projection_norm must land
    //     in norms.bin as vector entries. ──
    let ple_vector_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "vector")
        .filter_map(|e| e["key"].as_str())
        .filter(|k| k.contains("per_layer"))
        .collect();
    assert!(
        ple_vector_keys.contains(&"per_layer_projection_norm.weight"),
        "global PLE norm missing from norms.bin manifest: {ple_vector_keys:?}"
    );
    for layer in 0..num_layers {
        let k = format!("layers.{layer}.post_per_layer_input_norm.weight");
        assert!(
            ple_vector_keys.iter().any(|v| *v == k),
            "layer {layer} post-PLE norm missing: {ple_vector_keys:?}"
        );
    }

    // ── Load back and verify the dequantised PLE tensors surface in
    //     weights.tensors with the expected shapes. ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(&output_dir, &mut lcb).unwrap();

    let proj = weights
        .tensors
        .get("per_layer_model_projection.weight")
        .expect("per_layer_model_projection missing after load");
    assert_eq!(proj.shape(), &[ple_dim * num_layers, hidden]);

    let embed_ple = weights
        .tensors
        .get("embed_tokens_per_layer.weight")
        .expect("embed_tokens_per_layer missing after load");
    assert_eq!(embed_ple.shape(), &[vocab, ple_dim * num_layers]);

    for layer in 0..num_layers {
        let gate_key = format!("layers.{layer}.per_layer_input_gate.weight");
        let proj_key = format!("layers.{layer}.per_layer_projection.weight");
        let gate = weights
            .tensors
            .get(&gate_key)
            .unwrap_or_else(|| panic!("{gate_key} missing"));
        assert_eq!(gate.shape(), &[ple_dim, hidden]);
        let proj = weights
            .tensors
            .get(&proj_key)
            .unwrap_or_else(|| panic!("{proj_key} missing"));
        assert_eq!(proj.shape(), &[hidden, ple_dim]);
    }

    // Norms land in weights.vectors (f32 raw).
    assert!(
        weights
            .vectors
            .contains_key("per_layer_projection_norm.weight"),
        "global PLE norm missing from loaded weights.vectors"
    );

    // Gemma-4 layer_scalar: a per-layer 0-D scalar surfaced as a 1-element
    // vector. The forward path multiplies h by this value after FFN; omitting
    // it silently produced garbage on the 31B model. Previously uncovered
    // even on the Q4_K path — same failure mode shape as the PLE drop in #49,
    // so we pin it here too.
    for layer in 0..num_layers {
        let key = format!("layers.{layer}.layer_scalar");
        assert!(
            weights.vectors.contains_key(&key),
            "layer {layer} layer_scalar missing from loaded weights.vectors"
        );
    }

    // final_logit_softcapping must survive the round-trip. Missing it
    // lets predict_kquant peak the softmax on the wrong token.
    let cfg = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(
        cfg.model_config
            .as_ref()
            .and_then(|m| m.final_logit_softcapping),
        Some(30.0),
        "final_logit_softcapping dropped from vindex model_config"
    );
    assert_eq!(
        weights.arch.final_logit_softcapping(),
        Some(30.0),
        "loaded arch must surface the softcap via final_logit_softcapping()"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

// ─── PLE tensors survive --quant none extract → load round-trip ─────
//
// The companion regression test to `streaming_extract_q4k_carries_ple_tensors`
// for the `--quant none` / `--quant f16` extract path. Q4_K already wrote
// the PLE sidecar correctly; the float writer silently dropped every PLE
// tensor (the bug from chrishayuk/larql#49). After the fix, both writers
// route through `weights::ple_sidecar::write_ple_weights` and the float
// loader validates the invariant on load, refusing to surface a vindex that
// would silently produce garbage INFER output. ExtractLevel must be
// Inference (not Browse) because non-Q4 extracts only write model weights
// when the level demands attn — see `maybe_write_model_weights`.
#[test]
fn streaming_extract_noquant_carries_ple_tensors() {
    use larql_vindex::QuantFormat;

    let model_dir = std::env::temp_dir().join("larql_test_streaming_noquant_ple_model");
    let output_dir = std::env::temp_dir().join("larql_test_streaming_noquant_ple_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);

    let hidden = 256usize;
    let intermediate = 256usize;
    let num_layers = 2usize;
    let vocab = 256usize;
    let ple_dim = 256usize;

    write_gemma4_ple_fixture(&model_dir, num_layers, hidden, intermediate, vocab, ple_dim);
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/streaming-noquant-ple",
        &output_dir,
        5,
        // Inference (not Browse): non-Q4 only writes model weights when
        // the level includes attn.
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // ── ple_weights.bin must exist and the manifest must list all 3
    //     global + (2 per-layer) PLE tensor_f16 entries. Identical
    //     layout to the Q4_K case because both writers call into the
    //     shared `weights::ple_sidecar`. ──
    assert!(
        output_dir.join("ple_weights.bin").exists(),
        "non-Q4 extract should emit ple_weights.bin when the arch has PLE (#49)"
    );

    let manifest_json = std::fs::read_to_string(output_dir.join("weight_manifest.json")).unwrap();
    let manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest_json).unwrap();
    let ple_tensor_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "tensor_f16")
        .filter_map(|e| e["key"].as_str())
        .collect();

    assert_eq!(
        ple_tensor_keys.len(),
        2 + 2 * num_layers,
        "expected {} PLE tensor_f16 entries, got: {:?}",
        2 + 2 * num_layers,
        ple_tensor_keys
    );
    assert!(
        ple_tensor_keys.contains(&"per_layer_model_projection.weight"),
        "global model projection missing from manifest"
    );
    assert!(
        ple_tensor_keys.contains(&"embed_tokens_per_layer.weight"),
        "global per-layer embed missing from manifest"
    );

    // ── PLE norms (per-layer + global) must land in norms.bin as
    //     vector entries. ──
    let ple_vector_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "vector")
        .filter_map(|e| e["key"].as_str())
        .filter(|k| k.contains("per_layer"))
        .collect();
    assert!(
        ple_vector_keys.contains(&"per_layer_projection_norm.weight"),
        "global PLE norm missing from norms.bin manifest: {ple_vector_keys:?}"
    );
    for layer in 0..num_layers {
        let k = format!("layers.{layer}.post_per_layer_input_norm.weight");
        assert!(
            ple_vector_keys.iter().any(|v| *v == k),
            "layer {layer} post-PLE norm missing: {ple_vector_keys:?}"
        );
    }

    // ── layer_scalar (Gemma-4 only) must also land in norms.bin. ──
    let scalar_keys: Vec<&str> = manifest
        .iter()
        .filter(|e| e["kind"] == "vector")
        .filter_map(|e| e["key"].as_str())
        .filter(|k| k.contains("layer_scalar"))
        .collect();
    assert_eq!(
        scalar_keys.len(),
        num_layers,
        "expected one layer_scalar entry per layer, got: {scalar_keys:?}"
    );

    // ── Load back via the float loader and verify every PLE tensor +
    //     vector is hydrated. The loader now validates this invariant
    //     and would refuse to return Ok if anything is missing. ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights(&output_dir, &mut lcb).unwrap();

    let proj = weights
        .tensors
        .get("per_layer_model_projection.weight")
        .expect("per_layer_model_projection missing after load");
    assert_eq!(proj.shape(), &[ple_dim * num_layers, hidden]);

    let embed_ple = weights
        .tensors
        .get("embed_tokens_per_layer.weight")
        .expect("embed_tokens_per_layer missing after load");
    assert_eq!(embed_ple.shape(), &[vocab, ple_dim * num_layers]);

    for layer in 0..num_layers {
        let gate_key = format!("layers.{layer}.per_layer_input_gate.weight");
        let proj_key = format!("layers.{layer}.per_layer_projection.weight");
        let gate = weights
            .tensors
            .get(&gate_key)
            .unwrap_or_else(|| panic!("{gate_key} missing"));
        assert_eq!(gate.shape(), &[ple_dim, hidden]);
        let proj = weights
            .tensors
            .get(&proj_key)
            .unwrap_or_else(|| panic!("{proj_key} missing"));
        assert_eq!(proj.shape(), &[hidden, ple_dim]);

        let norm_key = format!("layers.{layer}.post_per_layer_input_norm.weight");
        assert!(
            weights.vectors.contains_key(&norm_key),
            "{norm_key} missing from weights.vectors"
        );

        let scalar_key = format!("layers.{layer}.layer_scalar");
        assert!(
            weights.vectors.contains_key(&scalar_key),
            "{scalar_key} missing from weights.vectors"
        );
    }

    assert!(
        weights
            .vectors
            .contains_key("per_layer_projection_norm.weight"),
        "global PLE norm missing from loaded weights.vectors"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

// ─── load_model_weights rejects PLE-arch vindexes with missing sidecars ──
//
// Pre-#49, dropping PLE tensors on the writer side was paired with a
// silent "missing tensor → return empty" in the forward path, so an
// affected vindex loaded fine but produced garbage INFER. The fix moves
// the failure into the load path: any PLE-active vindex whose
// `weights.tensors` / `weights.vectors` don't carry the required PLE
// keys is rejected with an actionable rebuild hint. This test stages
// that failure by writing the vindex normally, then nuking the manifest
// entries for the PLE tensors before calling `load_model_weights` —
// exercises the error branch in `load/f32.rs` end-to-end.
#[test]
fn load_model_weights_rejects_ple_arch_with_missing_sidecars() {
    use larql_vindex::QuantFormat;

    let model_dir = std::env::temp_dir().join("larql_test_ple_missing_sidecar_model");
    let output_dir = std::env::temp_dir().join("larql_test_ple_missing_sidecar_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);

    let hidden = 256usize;
    let intermediate = 256usize;
    let num_layers = 2usize;
    let vocab = 256usize;
    let ple_dim = 256usize;

    write_gemma4_ple_fixture(&model_dir, num_layers, hidden, intermediate, vocab, ple_dim);
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/ple-missing-sidecar",
        &output_dir,
        5,
        larql_vindex::ExtractLevel::Inference,
        larql_vindex::StorageDtype::F32,
        QuantFormat::None,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // Simulate a stale (pre-fix) vindex: drop the manifest entries that
    // would otherwise hydrate the PLE tensors at load time, then re-write
    // the manifest. The bytes in `ple_weights.bin` / `norms.bin` are left
    // alone — only the manifest table changes, which is exactly what the
    // pre-fix writer did (it never wrote the entries in the first place).
    let manifest_path = output_dir.join("weight_manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).unwrap();
    let mut entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_text).unwrap();
    entries.retain(|e| {
        let key = e["key"].as_str().unwrap_or("");
        !key.contains("per_layer")
            && !key.contains("embed_tokens_per_layer")
            && !key.contains("layer_scalar")
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&entries).unwrap(),
    )
    .unwrap();

    let mut lcb = larql_vindex::SilentLoadCallbacks;
    // `ModelWeights` doesn't implement Debug, so use a match here
    // instead of `Result::expect_err` (which requires T: Debug).
    let err = match larql_vindex::load_model_weights(&output_dir, &mut lcb) {
        Ok(_) => panic!("load must reject PLE-arch vindex with missing sidecars"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("Gemma-4 PLE sidecar"),
        "error must call out the PLE sidecars — got: {msg}"
    );
    assert!(
        msg.contains("chrishayuk/larql#49"),
        "error must point at the issue for the rebuild hint — got: {msg}"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}

// ─── Variable per-layer intermediate size (Gemma 4 E2B double-wide MLP) ──
//
// E2B's `use_double_wide_mlp=True` gives half the layers a 2× intermediate
// dimension (6144 → 12288 on the real model). `predict_kquant` previously
// hardcoded `weights.intermediate_size` for every layer's FFN dequant,
// so the wide layers' weights were read at half-size and the forward
// pass computed garbage. Fix: read per-layer feature count from the
// vindex via `VectorIndex::num_features(layer)`. This test locks the
// invariant that num_features matches the real per-layer shape so the
// fix stays honest.
#[test]
fn streaming_extract_preserves_per_layer_intermediate_for_variable_ffn() {
    use larql_vindex::QuantFormat;
    use std::collections::HashMap;

    let model_dir = std::env::temp_dir().join("larql_test_variable_ffn_model");
    let output_dir = std::env::temp_dir().join("larql_test_variable_ffn_output");
    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
    std::fs::create_dir_all(&model_dir).unwrap();

    let hidden = 256usize;
    let num_layers = 4usize;
    let vocab = 256usize;
    // Layers 0,1 narrow (256), layers 2,3 double-wide (512). Matches the
    // E2B pattern: the last half of the stack doubles the FFN width.
    let intermediates = [256usize, 256, 512, 512];
    let max_intermediate = *intermediates.iter().max().unwrap();

    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": max_intermediate,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": hidden,
        "vocab_size": vocab,
    });
    std::fs::write(
        model_dir.join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut metadata: Vec<(String, Vec<usize>)> = Vec::new();
    let push = |tensors: &mut HashMap<String, Vec<f32>>,
                metadata: &mut Vec<(String, Vec<usize>)>,
                name: &str,
                shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        tensors.insert(name.into(), data);
        metadata.push((name.into(), shape));
    };

    push(
        &mut tensors,
        &mut metadata,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    );
    push(
        &mut tensors,
        &mut metadata,
        "model.norm.weight",
        vec![hidden],
    );

    for (layer, &inter) in intermediates.iter().enumerate() {
        let lp = format!("model.layers.{layer}");
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.q_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.k_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.v_proj.weight"),
            vec![hidden, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.self_attn.o_proj.weight"),
            vec![hidden, hidden],
        );
        // Per-layer FFN width.
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.gate_proj.weight"),
            vec![inter, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.up_proj.weight"),
            vec![inter, hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.mlp.down_proj.weight"),
            vec![hidden, inter],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.input_layernorm.weight"),
            vec![hidden],
        );
        push(
            &mut tensors,
            &mut metadata,
            &format!("{lp}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }

    let tensor_bytes: Vec<(String, Vec<u8>, Vec<usize>)> = metadata
        .iter()
        .map(|(name, shape)| {
            let data = &tensors[name];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes, shape.clone())
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensor_bytes
        .iter()
        .map(|(name, bytes, shape)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), &serialized).unwrap();

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json).unwrap();
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let mut cb = larql_vindex::SilentBuildCallbacks;
    larql_vindex::build_vindex_streaming(
        &model_dir,
        &tokenizer,
        "test/variable-ffn",
        &output_dir,
        5,
        larql_vindex::ExtractLevel::Browse,
        larql_vindex::StorageDtype::F32,
        QuantFormat::Q4K,
        larql_vindex::WriteWeightsOptions::default(),
        larql_vindex::KquantWriteOptions::default(),
        false,
        &mut cb,
    )
    .unwrap();

    // ── Per-layer num_features in index.json ──
    let cfg = larql_vindex::load_vindex_config(&output_dir).unwrap();
    assert_eq!(cfg.layers.len(), num_layers);
    for (layer, li) in cfg.layers.iter().enumerate() {
        assert_eq!(
            li.num_features, intermediates[layer],
            "layer {layer} num_features must equal source FFN intermediate"
        );
    }

    // ── VectorIndex::num_features(layer) — the accessor predict_kquant calls ──
    let mut lcb = larql_vindex::SilentLoadCallbacks;
    let index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut lcb).unwrap();
    for (layer, &inter) in intermediates.iter().enumerate().take(num_layers) {
        assert_eq!(
            index.num_features(layer),
            inter,
            "VectorIndex::num_features(layer={layer}) wrong"
        );
    }

    // ── FFN manifest shape — the raw Q4K bytes must match the per-layer
    //     intermediate, NOT the model-wide max. Earlier predict_kquant bug:
    //     dequantising with the wrong width silently produced half-width
    //     weights on wide layers, so this assertion is the invariant. ──
    let ff_manifest_json =
        std::fs::read_to_string(output_dir.join("interleaved_kquant_manifest.json")).unwrap();
    let ff_entries: Vec<serde_json::Value> = serde_json::from_str(&ff_manifest_json).unwrap();
    for (layer, &inter) in intermediates.iter().enumerate() {
        let base = layer * 3; // gate, up, down per layer
        let gate_shape: Vec<usize> = ff_entries[base]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let up_shape: Vec<usize> = ff_entries[base + 1]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let down_shape: Vec<usize> = ff_entries[base + 2]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(gate_shape, vec![inter, hidden], "layer {layer} gate shape");
        assert_eq!(up_shape, vec![inter, hidden], "layer {layer} up shape");
        assert_eq!(down_shape, vec![hidden, inter], "layer {layer} down shape");
    }

    let _ = std::fs::remove_dir_all(&model_dir);
    let _ = std::fs::remove_dir_all(&output_dir);
}
