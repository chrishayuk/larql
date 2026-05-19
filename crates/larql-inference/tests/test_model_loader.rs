#![cfg(not(target_arch = "wasm32"))]

//! Inference-side tests: model loader, forward pass, decoy residual capture.
//!
//! Uses the shared mock-model fixture from `larql-vindex` so we don't have
//! to re-implement safetensors writing here.

use larql_vindex::walker::test_fixture::create_mock_model;

// ── Safetensors Loader ───────────────────────────────────────────────────

#[test]
fn loader_loads_all_tensors() {
    let dir = std::env::temp_dir().join("larql_test_loader");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let weights = larql_inference::model::load_model_dir(&dir).unwrap();

    assert_eq!(weights.num_layers, 2);
    assert_eq!(weights.hidden_size, 8);
    assert_eq!(weights.intermediate_size, 4);
    assert_eq!(weights.vocab_size, 16);
    assert_eq!(weights.head_dim, 4);
    assert_eq!(weights.num_q_heads, 2);
    assert_eq!(weights.num_kv_heads, 2);
    assert_eq!(weights.embed.shape(), &[16, 8]);

    assert!(weights
        .tensors
        .contains_key("layers.0.mlp.gate_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.1.mlp.down_proj.weight"));
    assert!(weights
        .tensors
        .contains_key("layers.0.self_attn.q_proj.weight"));

    assert!(weights
        .vectors
        .contains_key("layers.0.input_layernorm.weight"));
    assert!(weights.vectors.contains_key("norm.weight"));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn loader_missing_directory() {
    let result = larql_inference::model::load_model_dir("/nonexistent/path/model");
    assert!(result.is_err());
}

// ── Forward Pass ─────────────────────────────────────────────────────────

#[test]
fn forward_captures_residuals() {
    let dir = std::env::temp_dir().join("larql_test_forward");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let weights = larql_inference::model::load_model_dir(&dir).unwrap();
    let token_ids = vec![4u32, 5, 0];

    let residuals = larql_inference::forward::capture_residuals(&weights, &token_ids, &[0, 1]);

    assert_eq!(residuals.len(), 2);
    assert_eq!(residuals[0].0, 0);
    assert_eq!(residuals[1].0, 1);
    assert_eq!(residuals[0].1.len(), 8);
    assert_eq!(residuals[1].1.len(), 8);
    assert_ne!(residuals[0].1, residuals[1].1);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn forward_single_token() {
    let dir = std::env::temp_dir().join("larql_test_forward_single");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let weights = larql_inference::model::load_model_dir(&dir).unwrap();
    let residuals = larql_inference::forward::capture_residuals(&weights, &[0], &[1]);

    assert_eq!(residuals.len(), 1);
    assert_eq!(residuals[0].1.len(), 8);

    std::fs::remove_dir_all(&dir).ok();
}

// ── Decoy Residual Capture ───────────────────────────────────────────────

#[test]
fn capture_decoy_residuals_returns_vector_per_prompt() {
    let dir = std::env::temp_dir().join("larql_test_capture_decoys");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let weights = larql_inference::model::load_model_dir(&dir).unwrap();
    let prompts = vec![vec![4u32, 5], vec![0u32, 1, 2], vec![3u32]];
    let residuals = larql_inference::capture_decoy_residuals(&weights, &prompts, 1);

    assert_eq!(residuals.len(), 3);
    for (i, r) in residuals.iter().enumerate() {
        assert_eq!(r.len(), 8, "prompt {i} residual must be hidden_size");
    }
    assert_ne!(residuals[0], residuals[1]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn capture_decoy_residuals_empty_input() {
    let dir = std::env::temp_dir().join("larql_test_capture_decoys_empty");
    let _ = std::fs::remove_dir_all(&dir);
    create_mock_model(&dir);

    let weights = larql_inference::model::load_model_dir(&dir).unwrap();
    let residuals: Vec<_> = larql_inference::capture_decoy_residuals(&weights, &[], 0);
    assert!(residuals.is_empty());

    std::fs::remove_dir_all(&dir).ok();
}
