//! EXTRACT PIPELINE (synthetic model)
//! GGUF tests
//! PatchedVindex insert/delete/gate_knn tests

use super::*;

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
    patched.try_apply_patch(patch).expect("patch applies");

    // Verify patch applied
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "PATCHED");

    // Bake down to new index
    let baked = patched.bake_down();
    assert_eq!(baked.feature_meta(0, 0).unwrap().top_token, "PATCHED");
    assert_eq!(baked.total_gate_vectors(), 8);

    let _ = std::fs::remove_dir_all(&dir);
}

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
    use larql_models::loading::gguf::{GgufFile, GgufValue, ShardInfo};
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
        shards: vec![ShardInfo {
            path: std::path::PathBuf::new(),
            data_offset: 0,
        }],
    };
    let config = gguf.to_config_json();
    assert_eq!(config["model_type"], "llama");
    assert_eq!(config["hidden_size"], 4096);
    assert_eq!(config["num_hidden_layers"], 32);
    assert_eq!(config["intermediate_size"], 11008);
}

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
