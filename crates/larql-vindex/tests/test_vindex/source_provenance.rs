//! SOURCE PROVENANCE
//! PATCHES
//! WEIGHTS (split file write/read)
//! DTYPE
//! LOADER (HF cache resolution)

use super::*;

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
        bitnet_layout: None,
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
    patched.try_apply_patch(patch).expect("patch applies");

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
    patched.try_apply_patch(patch).expect("patch applies");

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
    patched.try_apply_patch(patch).expect("patch applies");

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
    patched.try_apply_patch(patch).expect("patch applies");
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "London");

    // Remove the patch — should revert to base
    patched.remove_patch(0);
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");
    assert_eq!(patched.num_patches(), 0);
}

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
        bitnet_layout: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Verify config round-trips with dtype
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(loaded.dtype, larql_vindex::StorageDtype::F32);

    let _ = std::fs::remove_dir_all(&dir);
}

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
        bitnet_layout: None,
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
