//! WS /v1/stream — WebSocket streaming for layer-by-layer DESCRIBE.
//!
//! Client sends JSON messages, server streams results back layer by layer.
//!
//! Protocol:
//!   → {"type": "describe", "entity": "France"}
//!   ← {"type": "layer", "layer": 14, "edges": [...]}
//!   ← {"type": "layer", "layer": 15, "edges": [...]}
//!   ← {"type": "done", "total_edges": 6, "latency_ms": 12.3}

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;

use crate::band_utils::{
    filter_layers_by_band, get_layer_bands, INFER_MODE_DENSE, PROBE_RELATION_SOURCE,
};
use crate::state::{elapsed_ms, AppState};

// WebSocket message type strings (outbound protocol contract).
const WS_TYPE_ERROR: &str = "error";
const WS_TYPE_LAYER: &str = "layer";
const WS_TYPE_DONE: &str = "done";
const WS_TYPE_PREDICTION: &str = "prediction";
const WS_TYPE_INFER_DONE: &str = "infer_done";

// Inbound message type strings.
const WS_CMD_DESCRIBE: &str = "describe";
const WS_CMD_INFER: &str = "infer";

fn ws_error(message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({"type": WS_TYPE_ERROR, "message": message.into()})
}

/// Send a JSON value over the WebSocket as a text frame. Returns the
/// underlying `axum::Error` if the peer has disconnected; callers
/// typically use [`send_msg_or_return`] to short-circuit cleanly.
async fn send_msg(socket: &mut WebSocket, value: &serde_json::Value) -> Result<(), axum::Error> {
    socket.send(Message::Text(value.to_string().into())).await
}

/// Convenience: send + return on send failure (peer disconnected).
/// Centralises the disconnect-handling pattern that otherwise repeats
/// at every send site. Used inside loops where one bad write means
/// the whole stream is over.
async fn send_msg_or_return(socket: &mut WebSocket, value: &serde_json::Value) -> bool {
    send_msg(socket, value).await.is_ok()
}

/// Send an error message, ignoring failures. The error is the last
/// thing we'd send before returning anyway, so a closed socket here
/// is fine.
async fn send_error(socket: &mut WebSocket, message: impl Into<String>) {
    let _ = send_msg(socket, &ws_error(message)).await;
}

fn ws_layer(layer: usize, edges: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "type": WS_TYPE_LAYER,
        "layer": layer,
        "edges": edges,
    })
}

fn ws_done(entity: impl Into<String>, total_edges: usize, latency_ms: f64) -> serde_json::Value {
    serde_json::json!({
        "type": WS_TYPE_DONE,
        "entity": entity.into(),
        "total_edges": total_edges,
        "latency_ms": latency_ms,
    })
}

fn ws_empty_done() -> serde_json::Value {
    serde_json::json!({"type": WS_TYPE_DONE, "total_edges": 0, "latency_ms": 0})
}

fn ws_prediction(rank: usize, token: &str, prob: f64) -> serde_json::Value {
    serde_json::json!({
        "type": WS_TYPE_PREDICTION,
        "rank": rank,
        "token": token,
        "probability": (prob * 10000.0).round() / 10000.0,
    })
}

fn ws_infer_done(
    prompt: impl Into<String>,
    mode: impl Into<String>,
    predictions: usize,
    latency_ms: f64,
) -> serde_json::Value {
    serde_json::json!({
        "type": WS_TYPE_INFER_DONE,
        "prompt": prompt.into(),
        "mode": mode.into(),
        "predictions": predictions,
        "latency_ms": latency_ms,
    })
}

#[utoipa::path(
    get,
    path = "/v1/stream",
    tag = "admin",
    responses(
        (status = 101, description = "\
WebSocket upgrade. After upgrade:\n\n\
**Client → server** (text frames, JSON):\n\
- `{\"type\":\"describe\", \"entity\":\"France\", \"band\":\"all\"}` — streams layer-by-layer describe.\n\
- `{\"type\":\"infer\", \"prompt\":\"The capital of France is\", \"top\":5, \"mode\":\"walk\"}` — \
streams top-K predictions one at a time.\n\n\
**Server → client** (text frames, JSON):\n\
- `{\"type\":\"layer\", \"layer\":N, \"edges\":[...]}` — per-layer describe output.\n\
- `{\"type\":\"done\", \"entity\":..., \"total_edges\":N, \"latency_ms\":M}` — describe finished.\n\
- `{\"type\":\"prediction\", \"rank\":I, \"token\":..., \"probability\":P}` — inference result.\n\
- `{\"type\":\"infer_done\", \"prompt\":..., \"mode\":..., \"predictions\":N, \"latency_ms\":M}` — inference finished.\n\
- `{\"type\":\"error\", \"message\":...}` — protocol or runtime error.\n"),
    ),
)]
pub async fn handle_stream(State(state): State<Arc<AppState>>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    while let Some(Ok(msg)) = socket.recv().await {
        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => break,
            _ => continue,
        };

        let request: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                send_error(&mut socket, e.to_string()).await;
                continue;
            }
        };

        let msg_type = request["type"].as_str().unwrap_or("");
        match msg_type {
            WS_CMD_DESCRIBE => {
                handle_stream_describe(&mut socket, &state, &request).await;
            }
            WS_CMD_INFER => {
                handle_stream_infer(&mut socket, &state, &request).await;
            }
            _ => {
                send_error(
                    &mut socket,
                    format!("unknown message type: {msg_type}. Supported: describe, infer"),
                )
                .await;
            }
        }
    }
}

async fn handle_stream_describe(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
    request: &serde_json::Value,
) {
    for msg in stream_describe_messages(state, request).await {
        if !send_msg_or_return(socket, &msg).await {
            return;
        }
    }
}

async fn stream_describe_messages(
    state: &AppState,
    request: &serde_json::Value,
) -> Vec<serde_json::Value> {
    let entity = match request["entity"].as_str() {
        Some(e) => e.to_string(),
        None => return vec![ws_error("missing entity")],
    };

    let model = match state.model(None) {
        Some(m) => Arc::clone(m),
        None => return vec![ws_error("no model loaded")],
    };

    let band = request["band"].as_str().unwrap_or("all");

    // Run the describe in a blocking task and stream results layer by layer.
    let start = std::time::Instant::now();

    let encoding = match model.tokenizer.encode(entity.as_str(), false) {
        Ok(e) => e,
        Err(e) => return vec![ws_error(e.to_string())],
    };
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return vec![ws_empty_done()];
    }

    let hidden = model.embeddings.shape()[1];
    let query = if token_ids.len() == 1 {
        model
            .embeddings
            .row(token_ids[0] as usize)
            .mapv(|v| v * model.embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            avg += &model
                .embeddings
                .row(tok as usize)
                .mapv(|v| v * model.embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };

    let bands = get_layer_bands(&model);

    let patched = model.patched.read().await;
    let all_layers = patched.loaded_layers();

    let scan_layers = filter_layers_by_band(all_layers, band, &bands);

    let entity_lower = entity.to_lowercase();
    let mut total_edges = 0;
    let mut messages = Vec::new();

    // Stream layer by layer.
    for &layer in &scan_layers {
        let hits = patched.gate_knn(layer, &query, 20);
        let mut edges = Vec::new();

        for (feature, gate_score) in &hits {
            if *gate_score < 5.0 {
                continue;
            }
            if let Some(meta) = patched.feature_meta(layer, *feature) {
                let tok = meta.top_token.trim();
                if tok.is_empty() || tok.len() < 2 || tok.to_lowercase() == entity_lower {
                    continue;
                }
                let mut edge = serde_json::json!({
                    "target": tok,
                    "gate_score": (*gate_score * 10.0).round() / 10.0,
                    "feature": feature,
                });
                if let Some(label) = model.probe_labels.get(&(layer, *feature)) {
                    edge["relation"] = serde_json::json!(label);
                    edge["source"] = serde_json::json!(PROBE_RELATION_SOURCE);
                }
                edges.push(edge);
            }
        }

        total_edges += edges.len();

        messages.push(ws_layer(layer, edges));
    }

    messages.push(ws_done(entity, total_edges, elapsed_ms(start)));
    messages
}

/// Handle streaming INFER: run forward pass and stream top-K predictions.
///
/// Protocol:
///   → {"type": "infer", "prompt": "The capital of France is", "top": 5, "mode": "walk"}
///   ← {"type": "prediction", "rank": 1, "token": "Paris", "probability": 0.9791}
///   ← {"type": "prediction", "rank": 2, "token": "the", "probability": 0.0042}
///   ← {"type": "infer_done", "prompt": "...", "mode": "walk", "latency_ms": 210}
async fn handle_stream_infer(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
    request: &serde_json::Value,
) {
    let prompt = match request["prompt"].as_str() {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => {
            send_error(socket, "missing or empty prompt").await;
            return;
        }
    };

    let model = match state.model(None) {
        Some(m) => Arc::clone(m),
        None => {
            send_error(socket, "no model loaded").await;
            return;
        }
    };

    if model.infer_disabled {
        send_error(socket, "inference disabled (--no-infer)").await;
        return;
    }

    // Validate access first; hold the guard only inside the sync
    // prediction block below so it doesn't cross any await
    // (`std::sync::RwLockReadGuard` is `!Send`). Map straight to a
    // String so the Result doesn't keep the guard alive past `?`.
    let weights_check: Result<(), String> = model.get_or_load_weights().map(|_| ());
    if let Err(e) = weights_check {
        send_error(socket, e).await;
        return;
    }

    let top_k = request["top"].as_u64().unwrap_or(5) as usize;
    let mode = request["mode"]
        .as_str()
        .unwrap_or(crate::band_utils::INFER_MODE_WALK);

    let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
        Ok(e) => e,
        Err(e) => {
            send_error(socket, e.to_string()).await;
            return;
        }
    };
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        send_error(socket, "empty prompt after tokenization").await;
        return;
    }

    let start = std::time::Instant::now();

    let predictions = {
        // Re-acquire the read guard for this sync compute block; drop
        // before re-entering the await ladder.
        let weights_guard = model.get_or_load_weights().expect("re-acquire weights");
        let weights: &larql_inference::ModelWeights = &weights_guard;
        if mode == INFER_MODE_DENSE {
            larql_inference::predict(weights, &model.tokenizer, &token_ids, top_k).predictions
        } else {
            let patched = model.patched.blocking_read();
            let r = larql_inference::infer_patched(
                weights,
                &model.tokenizer,
                &*patched,
                Some(&patched.knn_store),
                &token_ids,
                top_k,
            );
            r.predictions
        }
    };

    // Stream each prediction.
    for (rank, (token, prob)) in predictions.iter().enumerate() {
        let msg = ws_prediction(rank + 1, token, *prob);
        if !send_msg_or_return(socket, &msg).await {
            return;
        }
    }

    let done_msg = ws_infer_done(prompt, mode, predictions.len(), elapsed_ms(start));
    let _ = send_msg(socket, &done_msg).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::AtomicU64;

    use larql_vindex::ndarray::Array2;
    use larql_vindex::{
        ExtractLevel, FeatureMeta, LayerBands, PatchedVindex, QuantFormat, VectorIndex,
        VindexConfig, VindexLayerInfo,
    };
    use tokio::sync::RwLock;

    use crate::cache::DescribeCache;
    use crate::ffn_l2_cache::FfnL2Cache;
    use crate::session::SessionManager;
    use crate::state::LoadedModel;

    #[test]
    fn websocket_error_shape_is_stable() {
        let msg = ws_error("bad input");
        assert_eq!(msg["type"], WS_TYPE_ERROR);
        assert_eq!(msg["message"], "bad input");
    }

    // The send helpers need a live WebSocket to exercise; they're
    // covered transitively by the integration suite (test_http_*),
    // which exercises the WS upgrade path. The intent of the refactor
    // is purely a deduplication of the
    // `socket.send(Message::Text(value.to_string().into())).await`
    // pattern that previously appeared at 8 sites.

    #[test]
    fn websocket_layer_shape_includes_edges() {
        let msg = ws_layer(
            7,
            vec![serde_json::json!({
                "target": "Paris",
                "gate_score": 9.1,
                "feature": 3,
            })],
        );
        assert_eq!(msg["type"], WS_TYPE_LAYER);
        assert_eq!(msg["layer"], 7);
        assert_eq!(msg["edges"][0]["target"], "Paris");
    }

    #[test]
    fn websocket_done_shapes_are_stable() {
        let empty = ws_empty_done();
        assert_eq!(empty["type"], WS_TYPE_DONE);
        assert_eq!(empty["total_edges"], 0);

        let done = ws_done("France", 2, 1.25);
        assert_eq!(done["type"], WS_TYPE_DONE);
        assert_eq!(done["entity"], "France");
        assert_eq!(done["total_edges"], 2);
        assert_eq!(done["latency_ms"], 1.25);
    }

    #[test]
    fn websocket_prediction_rounds_probability() {
        let msg = ws_prediction(2, "Paris", 0.123456);
        assert_eq!(msg["type"], WS_TYPE_PREDICTION);
        assert_eq!(msg["rank"], 2);
        assert_eq!(msg["token"], "Paris");
        assert_eq!(msg["probability"], 0.1235);
    }

    #[test]
    fn websocket_infer_done_shape_is_stable() {
        let msg = ws_infer_done("prompt", "walk", 3, 4.5);
        assert_eq!(msg["type"], WS_TYPE_INFER_DONE);
        assert_eq!(msg["prompt"], "prompt");
        assert_eq!(msg["mode"], "walk");
        assert_eq!(msg["predictions"], 3);
        assert_eq!(msg["latency_ms"], 4.5);
    }

    fn functional_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"France":0,"Germany":1,"capital":2,"UNK":7},"unk_token":"UNK"}}"#;
        larql_vindex::tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap()
    }

    fn test_model(labels: HashMap<(usize, usize), String>) -> Arc<LoadedModel> {
        let mut gate = Array2::<f32>::zeros((3, 4));
        gate[[0, 0]] = 10.0;
        gate[[1, 1]] = 10.0;
        gate[[2, 2]] = 1.0;
        let meta = vec![
            Some(FeatureMeta {
                top_token: "Paris".into(),
                top_token_id: 10,
                c_score: 0.9,
                top_k: vec![],
            }),
            Some(FeatureMeta {
                top_token: "French".into(),
                top_token_id: 11,
                c_score: 0.8,
                top_k: vec![],
            }),
            Some(FeatureMeta {
                top_token: "x".into(),
                top_token_id: 12,
                c_score: 0.1,
                top_k: vec![],
            }),
        ];
        let mut embeddings = Array2::<f32>::zeros((8, 4));
        embeddings[[0, 0]] = 1.0;
        embeddings[[1, 1]] = 1.0;
        let config = VindexConfig {
            version: 2,
            model: "test/model".into(),
            family: "test".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 4,
            intermediate_size: 3,
            vocab_size: 8,
            embed_scale: 1.0,
            extract_level: ExtractLevel::Browse,
            dtype: larql_vindex::StorageDtype::default(),
            quant: QuantFormat::None,
            layer_bands: Some(LayerBands {
                syntax: (0, 0),
                knowledge: (0, 0),
                output: (0, 0),
            }),
            layers: vec![VindexLayerInfo {
                layer: 0,
                num_features: 3,
                offset: 0,
                length: 48,
                num_experts: None,
                num_features_per_expert: None,
            }],
            down_top_k: 5,
            has_model_weights: false,
            model_config: None,
            fp4: None,
            ffn_layout: None,
        };
        Arc::new(LoadedModel {
            id: "model".into(),
            path: std::path::PathBuf::from("/nonexistent"),
            config,
            patched: RwLock::new(PatchedVindex::new(VectorIndex::new(
                vec![Some(gate)],
                vec![Some(meta)],
                1,
                4,
            ))),
            embeddings,
            embed_scale: 1.0,
            tokenizer: functional_tokenizer(),
            infer_disabled: true,
            ffn_only: false,
            embed_only: false,
            embed_store: None,
            release_mmap_after_request: false,
            weights: std::sync::OnceLock::new(),
            probe_labels: labels,
            ffn_l2_cache: FfnL2Cache::new(1),
            expert_filter: None,
            unit_filter: None,
            moe_remote: None,
            #[cfg(feature = "metal-experts")]
            metal_backend: std::sync::OnceLock::new(),
            #[cfg(feature = "metal-experts")]
            moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn test_state(models: Vec<Arc<LoadedModel>>) -> Arc<AppState> {
        Arc::new(AppState {
            models,
            started_at: std::time::Instant::now(),
            requests_served: AtomicU64::new(0),
            api_key: None,
            sessions: SessionManager::new(3600),
            describe_cache: DescribeCache::new(0),
        })
    }

    #[tokio::test]
    async fn stream_describe_messages_reports_missing_entity() {
        let state = test_state(vec![test_model(HashMap::new())]);
        let messages = stream_describe_messages(&state, &serde_json::json!({})).await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["type"], WS_TYPE_ERROR);
        assert_eq!(messages[0]["message"], "missing entity");
    }

    #[tokio::test]
    async fn stream_describe_messages_reports_no_model() {
        let state = test_state(vec![]);
        let messages =
            stream_describe_messages(&state, &serde_json::json!({"entity": "France"})).await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["type"], WS_TYPE_ERROR);
        assert_eq!(messages[0]["message"], "no model loaded");
    }

    #[tokio::test]
    async fn stream_describe_messages_builds_layer_and_done_messages() {
        let mut labels = HashMap::new();
        labels.insert((0, 0), "capital".into());
        let state = test_state(vec![test_model(labels)]);
        let messages =
            stream_describe_messages(&state, &serde_json::json!({"entity": "France"})).await;

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["type"], WS_TYPE_LAYER);
        assert_eq!(messages[0]["layer"], 0);
        assert_eq!(messages[0]["edges"][0]["target"], "Paris");
        assert_eq!(messages[0]["edges"][0]["relation"], "capital");
        assert_eq!(messages[0]["edges"][0]["source"], PROBE_RELATION_SOURCE);
        assert_eq!(messages[1]["type"], WS_TYPE_DONE);
        assert_eq!(messages[1]["entity"], "France");
        assert_eq!(messages[1]["total_edges"], 1);
    }
}
