#![cfg(not(target_arch = "wasm32"))]
//! Tests for the gRPC service handlers.
//!
//! The handlers are called directly as async trait methods — no network
//! socket required. A test AppState with an in-memory VectorIndex is
//! sufficient for all non-inference paths.

mod common;
use common::*;

use larql_server::grpc::proto::vindex_service_server::VindexService;
use larql_server::grpc::proto::*;
use larql_server::grpc::VindexGrpcService;
use tonic::Request;

fn svc(models: Vec<std::sync::Arc<larql_server::state::LoadedModel>>) -> VindexGrpcService {
    VindexGrpcService {
        state: state(models),
    }
}

fn svc_functional() -> VindexGrpcService {
    svc(vec![model_functional("test")])
}

// ══════════════════════════════════════════════════════════════
// health
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_health_returns_ok_status() {
    let resp = svc_functional()
        .health(Request::new(HealthRequest {}))
        .await
        .unwrap();
    assert_eq!(resp.get_ref().status, "ok");
}

#[tokio::test]
async fn grpc_health_returns_uptime() {
    let resp = svc_functional()
        .health(Request::new(HealthRequest {}))
        .await
        .unwrap();
    assert!(resp.get_ref().uptime_seconds < 60);
}

#[tokio::test]
async fn grpc_health_bumps_request_counter() {
    let st = state(vec![model_functional("test")]);
    let svc = VindexGrpcService { state: st.clone() };
    svc.health(Request::new(HealthRequest {})).await.unwrap();
    assert_eq!(
        st.requests_served
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

// ══════════════════════════════════════════════════════════════
// get_stats
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_get_stats_returns_model_info() {
    let resp = svc_functional()
        .get_stats(Request::new(StatsRequest {}))
        .await
        .unwrap();
    let stats = resp.get_ref();
    assert_eq!(stats.model, "test/model-4");
    assert_eq!(stats.family, "test");
    assert_eq!(stats.layers, 1);
    assert_eq!(stats.hidden_size, 4);
}

#[tokio::test]
async fn grpc_get_stats_no_model_returns_not_found() {
    let st = state(vec![]);
    let svc = VindexGrpcService { state: st };
    let err = svc
        .get_stats(Request::new(StatsRequest {}))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn grpc_get_stats_has_layer_bands() {
    let resp = svc_functional()
        .get_stats(Request::new(StatsRequest {}))
        .await
        .unwrap();
    assert!(resp.get_ref().layer_bands.is_some());
}

// ══════════════════════════════════════════════════════════════
// describe
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_describe_empty_tokenizer_returns_empty_edges() {
    // Empty BPE tokenizer → empty token ids → early-return path.
    let svc = svc(vec![model("test")]);
    let resp = svc
        .describe(Request::new(DescribeRequest {
            entity: "France".into(),
            band: String::new(),
            limit: 0,
            min_score: 0.0,
            verbose: false,
        }))
        .await
        .unwrap();
    assert_eq!(resp.get_ref().entity, "France");
    assert!(resp.get_ref().edges.is_empty());
}

#[tokio::test]
async fn grpc_describe_functional_returns_edges() {
    // Functional tokenizer: France→0 → embedding[0]=[1,0,0,0] → hits feature 0 (Paris).
    // Use min_score=0.1 (positive) so the gRPC handler doesn't fall back to default 5.0.
    let svc = svc_functional();
    let resp = svc
        .describe(Request::new(DescribeRequest {
            entity: "France".into(),
            band: String::new(),
            limit: 10,
            min_score: 0.1,
            verbose: false,
        }))
        .await
        .unwrap();
    assert_eq!(resp.get_ref().entity, "France");
    assert!(!resp.get_ref().edges.is_empty());
}

#[tokio::test]
async fn grpc_describe_top_edge_is_paris() {
    let svc = svc_functional();
    let resp = svc
        .describe(Request::new(DescribeRequest {
            entity: "France".into(),
            band: String::new(),
            limit: 10,
            min_score: 0.1,
            verbose: false,
        }))
        .await
        .unwrap();
    let edges = &resp.get_ref().edges;
    assert!(edges.iter().any(|e| e.target == "Paris"));
}

#[tokio::test]
async fn grpc_describe_no_model_returns_not_found() {
    let svc = svc(vec![]);
    let err = svc
        .describe(Request::new(DescribeRequest {
            entity: "France".into(),
            band: String::new(),
            limit: 0,
            min_score: 0.0,
            verbose: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ══════════════════════════════════════════════════════════════
// walk
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_walk_functional_returns_hits() {
    let svc = svc_functional();
    let resp = svc
        .walk(Request::new(WalkRequest {
            prompt: "France".into(),
            top: 5,
            layers: String::new(),
        }))
        .await
        .unwrap();
    assert_eq!(resp.get_ref().prompt, "France");
    assert!(!resp.get_ref().hits.is_empty());
}

#[tokio::test]
async fn grpc_walk_top_hit_is_paris() {
    let svc = svc_functional();
    let resp = svc
        .walk(Request::new(WalkRequest {
            prompt: "France".into(),
            top: 5,
            layers: String::new(),
        }))
        .await
        .unwrap();
    let hits = &resp.get_ref().hits;
    assert_eq!(hits[0].target, "Paris");
}

#[tokio::test]
async fn grpc_walk_empty_prompt_returns_invalid_arg() {
    let svc = svc_functional();
    let err = svc
        .walk(Request::new(WalkRequest {
            prompt: String::new(),
            top: 5,
            layers: String::new(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn grpc_walk_no_model_returns_not_found() {
    let svc = svc(vec![]);
    let err = svc
        .walk(Request::new(WalkRequest {
            prompt: "hello".into(),
            top: 5,
            layers: String::new(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ══════════════════════════════════════════════════════════════
// select
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_select_all_returns_features() {
    let svc = svc_functional();
    let resp = svc
        .select(Request::new(SelectRequest {
            entity: String::new(),
            layer: 0,
            limit: 20,
            min_confidence: 0.0,
            relation: String::new(),
            order_by: String::new(),
            order: String::new(),
        }))
        .await
        .unwrap();
    assert!(!resp.get_ref().edges.is_empty());
}

#[tokio::test]
async fn grpc_select_with_entity_filter() {
    let svc = svc_functional();
    let resp = svc
        .select(Request::new(SelectRequest {
            entity: "Paris".into(),
            layer: 0,
            limit: 20,
            min_confidence: 0.0,
            relation: String::new(),
            order_by: String::new(),
            order: String::new(),
        }))
        .await
        .unwrap();
    for edge in &resp.get_ref().edges {
        assert!(edge.target.to_lowercase().contains("paris"));
    }
}

#[tokio::test]
async fn grpc_select_no_model_returns_not_found() {
    let svc = svc(vec![]);
    let err = svc
        .select(Request::new(SelectRequest {
            entity: String::new(),
            layer: 0,
            limit: 20,
            min_confidence: 0.0,
            relation: String::new(),
            order_by: String::new(),
            order: String::new(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ══════════════════════════════════════════════════════════════
// infer
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_infer_disabled_returns_unavailable() {
    // model_functional has infer_disabled=true (default).
    let svc = svc_functional();
    let err = svc
        .infer(Request::new(InferRequest {
            prompt: "France".into(),
            top: 5,
            mode: String::new(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::Unavailable);
}

#[tokio::test]
async fn grpc_infer_no_model_returns_not_found() {
    let svc = svc(vec![]);
    let err = svc
        .infer(Request::new(InferRequest {
            prompt: "France".into(),
            top: 5,
            mode: String::new(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ══════════════════════════════════════════════════════════════
// get_relations
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_get_relations_returns_list() {
    let svc = svc_functional();
    let resp = svc
        .get_relations(Request::new(RelationsRequest {
            source: String::new(),
        }))
        .await
        .unwrap();
    // Relations are derived from feature meta top_tokens. The test index has 3 features.
    assert!(resp.get_ref().total > 0);
}

#[tokio::test]
async fn grpc_get_relations_no_model_returns_not_found() {
    let svc = svc(vec![]);
    let err = svc
        .get_relations(Request::new(RelationsRequest {
            source: String::new(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ══════════════════════════════════════════════════════════════
// walk_ffn (features-only, no weights needed)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_walk_ffn_features_only_returns_results() {
    let svc = svc_functional();
    let residual = vec![1.0f32, 0.0, 0.0, 0.0];
    let resp = svc
        .walk_ffn(Request::new(WalkFfnRequest {
            layer: 0,
            layers: vec![],
            residual,
            seq_len: 1,
            top_k: 5,
            full_output: false,
        }))
        .await
        .unwrap();
    let results = &resp.get_ref().results;
    assert_eq!(results.len(), 1);
    assert!(!results[0].features.is_empty());
    assert_eq!(results[0].features[0], 0); // feature 0 = Paris, matches [1,0,0,0]
}

#[tokio::test]
async fn grpc_walk_ffn_wrong_residual_size_returns_invalid_arg() {
    let svc = svc_functional();
    let err = svc
        .walk_ffn(Request::new(WalkFfnRequest {
            layer: 0,
            layers: vec![],
            residual: vec![1.0, 0.0], // too short (hidden=4, expected 4)
            seq_len: 1,
            top_k: 5,
            full_output: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn grpc_walk_ffn_no_model_returns_not_found() {
    let svc = svc(vec![]);
    let err = svc
        .walk_ffn(Request::new(WalkFfnRequest {
            layer: 0,
            layers: vec![],
            residual: vec![1.0, 0.0, 0.0, 0.0],
            seq_len: 1,
            top_k: 5,
            full_output: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn grpc_walk_ffn_multi_layer_batch_returns_all() {
    let svc = svc_functional();
    // layers=[0,0] → two results (same layer twice is valid).
    let resp = svc
        .walk_ffn(Request::new(WalkFfnRequest {
            layer: 0,
            layers: vec![0, 0],
            residual: vec![1.0f32, 0.0, 0.0, 0.0],
            seq_len: 1,
            top_k: 3,
            full_output: false,
        }))
        .await
        .unwrap();
    assert_eq!(resp.get_ref().results.len(), 2);
}

// ══════════════════════════════════════════════════════════════
// stream_describe (spawns background task, returns stream)
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn grpc_stream_describe_returns_stream() {
    let svc = svc_functional();
    let resp = svc
        .stream_describe(Request::new(DescribeRequest {
            entity: "France".into(),
            band: String::new(),
            limit: 10,
            min_score: 0.1,
            verbose: false,
        }))
        .await
        .unwrap();
    // Stream is returned immediately; consuming it is async.
    // Just verify we get a response with a stream.
    let _stream = resp.into_inner();
}

#[tokio::test]
async fn grpc_stream_describe_no_model_returns_not_found() {
    let svc = svc(vec![]);
    let err = svc
        .stream_describe(Request::new(DescribeRequest {
            entity: "France".into(),
            band: String::new(),
            limit: 10,
            min_score: 0.1,
            verbose: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn grpc_stream_describe_collects_events() {
    use tokio_stream::StreamExt;

    let svc = svc_functional();
    let resp = svc
        .stream_describe(Request::new(DescribeRequest {
            entity: "France".into(),
            band: String::new(),
            limit: 10,
            min_score: 0.1,
            verbose: false,
        }))
        .await
        .unwrap();

    let mut stream = resp.into_inner();
    let mut events = vec![];
    // Allow the background task time to send events, then collect.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    while let Ok(Some(ev)) =
        tokio::time::timeout(std::time::Duration::from_millis(50), stream.next()).await
    {
        if let Ok(e) = ev {
            events.push(e);
        }
    }
    // Should receive at least one event (the done marker or a layer event).
    assert!(!events.is_empty());
}
