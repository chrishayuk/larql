//! End-to-end **numerical validation** for the `/v1/attention/prefill`
//! endpoint against the same `larql_inference` runner the server uses.
//!
//! Builds a fully-functional synthetic `ModelWeights` via
//! `larql_inference::engines::test_utils::make_test_weights` (no
//! disk I/O), wires it into a `LoadedModel` with `infer_disabled =
//! false` and the weights cell pre-populated, drives a real prefill
//! through the axum router, and compares every per-layer residual
//! to the same forward pass run directly in-process.
//!
//! The numerical contract on these synthetic weights is **bit-exact**
//! (the request/response wire path is the only delta), which catches
//! JSON serialisation drift, layer-order bugs, and any future
//! refactor that changes which layers populate which response slots.
//!
//! Coverage hooks:
//!   - server-attention-service / "Prefill of 1024 tokens populates
//!     the KV cache" — covered structurally (we use seq_len=4 here
//!     to keep the test fast; the layer-iteration logic is the same).
//!   - server-attention-service / "Prefill returns layers × seq_len
//!     × hidden_dim residuals"
//!   - server-attention-service / "decode advances seqlen"

mod common;
use common::*;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use axum::body::Body;
use axum::http::StatusCode;
use larql_inference::engines::test_utils::{
    make_test_weights, make_test_weights_with, SyntheticDims,
};
use larql_inference::ndarray::Array2;
use larql_inference::{run_layer_with_ffn_capturing_h_post_attn, ModelWeights};
use larql_server::attention_session::AttentionSessionMap;
use larql_server::cache::DescribeCache;
use larql_server::ffn_l2_cache::FfnL2Cache;
use larql_server::routes::single_model_router;
use larql_server::session::SessionManager;
use larql_server::state::{AppState, LoadedModel};
use larql_server::tokenizer_cache::TokenizerCache;
use larql_vindex::{
    ndarray::Array2 as VArray2, ExtractLevel, LayerBands, PatchedVindex, QuantFormat, VectorIndex,
    VindexConfig,
};

const MODEL_ID: &str = "tiny";

fn build_loaded_model(weights: ModelWeights) -> LoadedModel {
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;
    let intermediate = weights.intermediate_size;
    let vocab = weights.vocab_size;

    // Minimal vindex matching the synthetic weights' layer count.
    let gate = VArray2::<f32>::zeros((1, hidden));
    let gate_vectors: Vec<Option<VArray2<f32>>> =
        (0..num_layers).map(|_| Some(gate.clone())).collect();
    let down_meta = vec![None; num_layers];
    let index = VectorIndex::new(gate_vectors, down_meta, num_layers, hidden);
    let patched = PatchedVindex::new(index);

    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();

    let config = VindexConfig {
        version: 2,
        model: MODEL_ID.into(),
        family: "tiny".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size: vocab,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Inference,
        dtype: larql_vindex::StorageDtype::default(),
        quant: QuantFormat::None,
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 0),
            output: (0, num_layers.saturating_sub(1)),
        }),
        layers: (0..num_layers)
            .map(|l| larql_vindex::VindexLayerInfo {
                layer: l,
                num_features: 1,
                offset: 0,
                length: (hidden * 4) as u64,
                num_experts: None,
                num_features_per_expert: None,
            })
            .collect(),
        down_top_k: 1,
        has_model_weights: true,
        model_config: None,
        ffn_layout: None,
        fp4: None,
    };

    let lock = OnceLock::new();
    lock.set(std::sync::RwLock::new(weights)).ok();

    LoadedModel {
        id: MODEL_ID.into(),
        path: PathBuf::from("/nonexistent"),
        config,
        patched: tokio::sync::RwLock::new(patched),
        embeddings: Array2::zeros((vocab, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: false,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: lock,
        probe_labels: HashMap::new(),
        ffn_l2_cache: FfnL2Cache::new(1),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        tokenizer_cache: Arc::new(TokenizerCache::new(0, 0)),
    }
}

fn build_state(model: LoadedModel) -> Arc<AppState> {
    Arc::new(AppState {
        models: vec![Arc::new(model)],
        started_at: std::time::Instant::now(),
        requests_served: AtomicU64::new(0),
        api_key: None,
        sessions: SessionManager::new(60),
        describe_cache: DescribeCache::new(0),
        attention_sessions: Arc::new(AttentionSessionMap::new(60, 16)),
    })
}

fn deterministic_embeddings(seq_len: usize, hidden: usize) -> Vec<Vec<f32>> {
    (0..seq_len)
        .map(|i| {
            (0..hidden)
                .map(|j| ((i * 37 + j * 13) as f32 * 0.001).sin() * 0.1)
                .collect()
        })
        .collect()
}

/// Run prefill via the server (oneshot HTTP) and via direct in-process
/// inference; return both per-layer residuals plus the session id.
async fn run_via_server_and_locally(
    seq_len: usize,
) -> (Vec<Vec<Vec<f32>>>, Vec<Array2<f32>>, String, Arc<AppState>) {
    run_via_server_and_locally_with(seq_len, make_test_weights, make_test_weights).await
}

/// Generic core. Takes two factory closures so the same test body
/// can validate against either tiny or Gemma-shaped synthetics.
/// Both factories MUST produce identical weights (we call them
/// twice — once for the server's `LoadedModel`, once for the
/// in-process reference). `make_test_weights` is deterministic by
/// LCG seed, so two calls return the same values.
async fn run_via_server_and_locally_with<F, G>(
    seq_len: usize,
    server_weights_factory: F,
    ref_weights_factory: G,
) -> (Vec<Vec<Vec<f32>>>, Vec<Array2<f32>>, String, Arc<AppState>)
where
    F: FnOnce() -> ModelWeights,
    G: FnOnce() -> ModelWeights,
{
    let weights = server_weights_factory();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    let ref_weights = ref_weights_factory();

    let model = build_loaded_model(weights);
    let st = build_state(model);

    // Create session via HTTP.
    let app = single_model_router(st.clone());
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({"model_id": MODEL_ID, "kv_format": "fp32"}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "session create failed");
    let body = body_json(resp.into_body()).await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let token_embeddings = deterministic_embeddings(seq_len, hidden);

    // Server-side prefill — opt into per-layer capture (FP32 model only).
    let app2 = single_model_router(st.clone());
    let resp = post_json(
        app2,
        "/v1/attention/prefill?capture=layer-residuals",
        serde_json::json!({
            "session_id": sid,
            "token_embeddings": token_embeddings,
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "prefill failed: {:?}", resp);
    let body = body_json(resp.into_body()).await;
    let server_residuals: Vec<Vec<Vec<f32>>> =
        serde_json::from_value(body["post_attention_residuals"].clone())
            .expect("layer-residuals capture must produce a populated array on FP32 models");

    // Local reference: same forward pass, no transport.
    let mut flat = Vec::with_capacity(seq_len * hidden);
    for row in &token_embeddings {
        flat.extend_from_slice(row);
    }
    let h0 = Array2::from_shape_vec((seq_len, hidden), flat).unwrap();
    let ffn = larql_inference::ffn::WeightFfn {
        weights: &ref_weights,
    };
    let mut h = h0;
    let mut local_residuals = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let (h_next, h_post_attn, _, _) = run_layer_with_ffn_capturing_h_post_attn(
            &ref_weights,
            &h,
            layer,
            &ffn,
            false,
            None,
            None,
        )
        .expect("layer must run");
        local_residuals.push(h_post_attn);
        h = h_next;
    }

    (server_residuals, local_residuals, sid, st)
}

async fn body_bytes(body: Body) -> Vec<u8> {
    axum::body::to_bytes(body, usize::MAX)
        .await
        .unwrap()
        .to_vec()
}

// ── tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn prefill_server_residuals_match_local_reference_layer_by_layer() {
    let (server, local, _sid, _st) = run_via_server_and_locally(4).await;

    assert_eq!(
        server.len(),
        local.len(),
        "layer count mismatch: server={} local={}",
        server.len(),
        local.len()
    );

    for (layer, (s, l)) in server.iter().zip(local.iter()).enumerate() {
        assert_eq!(s.len(), l.nrows(), "layer {layer}: seq_len mismatch");
        for (row_idx, server_row) in s.iter().enumerate() {
            let local_row = l.row(row_idx);
            assert_eq!(
                server_row.len(),
                local_row.len(),
                "layer {layer} row {row_idx}: hidden_dim mismatch"
            );
            // Both paths run the same compiled function on the same
            // f32 inputs. JSON round-trips f32 exactly — assert on
            // bit-equality.
            for (j, (&sv, &lv)) in server_row.iter().zip(local_row.iter()).enumerate() {
                let abs_diff = (sv - lv).abs();
                assert!(
                    abs_diff <= 1e-5,
                    "layer {layer} row {row_idx} col {j}: server={sv} local={lv} diff={abs_diff}"
                );
            }
        }
    }
}

#[tokio::test]
async fn prefill_response_shape_matches_layers_seq_hidden() {
    // The shape harness uses `?capture=layer-residuals`, so the
    // server's `post_attention_residuals` array is populated with
    // [layers][seq][hidden] = [2][4][16] for make_test_weights.
    let (server, _local, _sid, _st) = run_via_server_and_locally(4).await;
    assert_eq!(server.len(), 2);
    assert_eq!(server[0].len(), 4);
    assert_eq!(server[0][0].len(), 16);
}

#[tokio::test]
async fn prefill_default_returns_final_hidden_only() {
    // Without `?capture=layer-residuals`, the response SHALL contain
    // `final_hidden: [seq_len][hidden_dim]` and SHALL omit
    // `post_attention_residuals` entirely.
    let weights = make_test_weights();
    let model = build_loaded_model(weights);
    let st = build_state(model);

    let app = single_model_router(st.clone());
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({"model_id": MODEL_ID, "kv_format": "fp32"}),
    )
    .await;
    let body = body_json(resp.into_body()).await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/attention/prefill",
        serde_json::json!({
            "session_id": sid,
            "token_embeddings": deterministic_embeddings(4, 16),
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;

    let final_hidden = body["final_hidden"]
        .as_array()
        .expect("final_hidden present");
    assert_eq!(final_hidden.len(), 4, "seq_len=4");
    assert_eq!(
        final_hidden[0].as_array().expect("row is array").len(),
        16,
        "hidden_dim=16",
    );

    // post_attention_residuals omitted by default — the field is
    // serde-skipped when None, so it should be absent (not null).
    assert!(
        !body
            .as_object()
            .unwrap()
            .contains_key("post_attention_residuals"),
        "post_attention_residuals must be absent without capture flag, got body={body}",
    );
}

#[tokio::test]
async fn session_seq_len_advances_after_prefill() {
    let (_server, _local, sid, st) = run_via_server_and_locally(7).await;
    let app = single_model_router(st);
    let resp = get(app, &format!("/v1/attention/session/{sid}")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["seq_len"], 7);
    assert_eq!(body["prefilled"], true);
}

/// Same validation contract as the tiny-shape test, but at
/// Gemma-3-4B head dimensions (hidden=2560, num_q=8, num_kv=4,
/// head_dim=320, intermediate=10240, 4 layers). Confirms the
/// runner's numerical correctness scales to realistic dims, not
/// just unit-test sizes. Marked `#[ignore]` so `cargo test`
/// doesn't pay the ~3s weight-build cost on every run; opt in
/// via `make attention-validate-gemma`.
#[tokio::test]
#[ignore]
async fn prefill_gemma_shaped_residuals_match_local_reference() {
    let factory = || make_test_weights_with(SyntheticDims::gemma_3_4b_4layer());
    let (server, local, _sid, _st) = run_via_server_and_locally_with(2, factory, factory).await;

    assert_eq!(server.len(), local.len(), "layer count mismatch");
    for (layer, (s, l)) in server.iter().zip(local.iter()).enumerate() {
        for (row_idx, server_row) in s.iter().enumerate() {
            let local_row = l.row(row_idx);
            for (j, (&sv, &lv)) in server_row.iter().zip(local_row.iter()).enumerate() {
                let abs_diff = (sv - lv).abs();
                assert!(
                    abs_diff <= 1e-3,
                    "layer {layer} row {row_idx} col {j}: server={sv} local={lv} diff={abs_diff}"
                );
            }
        }
    }
}

/// Quantised-model guard. Strip the FP32 attention tensors from a
/// synthetic `make_test_weights` and confirm the prefill route
/// fails loud (503 `quantized_model_unsupported`) instead of
/// silently returning zero residuals — the bug we hit running the
/// real Q4_K Gemma 3 4B vindex through this route.
///
/// Decode is checked symmetrically.
fn build_state_with_q4k_like_weights() -> Arc<AppState> {
    // Strip every `attn_q_proj.weight` from `weights.tensors` —
    // `check_fp32_attention_loaded` asks the arch for the per-layer
    // attn_q key and looks it up there. Removing those keys mimics a
    // Q4_K-quantised model whose attention weights live in
    // `packed_mmaps` rather than the FP32 tensor map.
    let mut weights = make_test_weights();
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        weights.tensors.remove(&arch.attn_q_key(layer));
    }
    let model = build_loaded_model(weights);
    build_state(model)
}

#[tokio::test]
async fn prefill_returns_503_quantized_model_unsupported() {
    let st = build_state_with_q4k_like_weights();
    let app = single_model_router(st.clone());

    // Create a session against the q4k-like model.
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({"model_id": MODEL_ID, "kv_format": "fp32"}),
    )
    .await;
    let body = body_json(resp.into_body()).await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    // Prefill should now refuse loudly.
    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/attention/prefill",
        serde_json::json!({
            "session_id": sid,
            "token_embeddings": [[0.0, 0.1, 0.2, 0.3]],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "quantized_model_unsupported");
    assert!(
        body["detail"].as_str().unwrap_or("").contains("FP32-only"),
        "detail should mention FP32-only; got {:?}",
        body["detail"]
    );
}

#[tokio::test]
async fn decode_returns_503_quantized_model_unsupported() {
    let st = build_state_with_q4k_like_weights();
    let app = single_model_router(st.clone());

    // Create a session.
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({"model_id": MODEL_ID, "kv_format": "fp32"}),
    )
    .await;
    let body = body_json(resp.into_body()).await;
    let sid = body["session_id"].as_str().unwrap().to_string();

    // Force-mark prefilled so the decode handler reaches the q4k guard
    // (which lives after the prefilled check).
    {
        let entry = st
            .attention_sessions
            .get(&larql_server::attention_session::SessionId(sid.clone()))
            .expect("session present");
        let mut g = entry.write().await;
        g.prefilled = true;
        g.seq_len = 1;
    }

    let app2 = single_model_router(st);
    let resp = post_json(
        app2,
        "/v1/attention/decode",
        serde_json::json!({
            "session_id": sid,
            "query_token_embedding": [0.0, 0.1, 0.2, 0.3],
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = body_json(resp.into_body()).await;
    assert_eq!(body["error"], "quantized_model_unsupported");
}

#[tokio::test]
async fn snapshot_after_prefill_round_trips_through_restore() {
    let (server, _local, sid, st) = run_via_server_and_locally(3).await;

    // Snapshot.
    let app = single_model_router(st.clone());
    let resp = post_json(
        app,
        "/v1/kv-cache/snapshot",
        serde_json::json!({"session_id": sid}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let snap_b64 = body["snapshot"].as_str().unwrap().to_string();

    // Restore into a fresh session.
    let app = single_model_router(st);
    let resp = post_json(
        app,
        "/v1/attention/session",
        serde_json::json!({
            "model_id": MODEL_ID,
            "restore_from_snapshot": snap_b64,
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    let new_sid = body["session_id"].as_str().unwrap();
    assert_ne!(new_sid, &sid);
    // Suppress unused.
    let _ = server;
}

// Suppress unused-helper warnings — the serialised-bytes form is a
// hook for future binary-shape tests that may diff bytemuck::cast_slice
// against the JSON arm.
#[allow(dead_code)]
async fn _suppress(b: Body) -> Vec<u8> {
    body_bytes(b).await
}
