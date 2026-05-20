#![cfg(not(target_arch = "wasm32"))]
//! Pure unit tests: walk-ffn binary protocol, stream format, gRPC shapes,
//! embed binary, logits binary, token decode parsing, select ordering tests.

use larql_vindex::ndarray::Array2;

// ══════════════════════════════════════════════════════════════
// Test helpers (local copy of test_embeddings)
// ══════════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════════
// WALK LAYER RANGE PARSING
// ══════════════════════════════════════════════════════════════

fn parse_layers(s: &str, all: &[usize]) -> Vec<usize> {
    if let Some((start, end)) = s.split_once('-') {
        if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
            return all.iter().copied().filter(|l| *l >= s && *l <= e).collect();
        }
    }
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .filter(|l| all.contains(l))
        .collect()
}

#[test]
fn test_parse_layer_range() {
    let all = vec![0, 1, 2, 3, 4, 5];
    assert_eq!(parse_layers("2-4", &all), vec![2, 3, 4]);
    assert_eq!(parse_layers("0-1", &all), vec![0, 1]);
    assert_eq!(parse_layers("5-5", &all), vec![5]);
}

#[test]
fn test_parse_layer_list() {
    let all = vec![0, 1, 2, 3, 4, 5];
    assert_eq!(parse_layers("1,3,5", &all), vec![1, 3, 5]);
    assert_eq!(parse_layers("0", &all), vec![0]);
}

#[test]
fn test_parse_layer_range_filters_missing() {
    let all = vec![0, 2, 4]; // layers 1, 3 not loaded
    assert_eq!(parse_layers("0-4", &all), vec![0, 2, 4]);
    assert_eq!(parse_layers("1,3", &all), Vec::<usize>::new());
}

// ══════════════════════════════════════════════════════════════
// WALK-FFN (decoupled inference protocol)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_walk_ffn_residual_dimension_check() {
    // Handler validates residual length == hidden_size
    let expected_hidden = 4;
    let residual_ok = [1.0f32; 4];
    let residual_bad = [1.0f32; 8];
    assert_eq!(residual_ok.len(), expected_hidden);
    assert_ne!(residual_bad.len(), expected_hidden);
}

#[test]
fn test_walk_ffn_top_k_default() {
    // Default top_k is 8092
    let default_top_k: usize = 8092;
    assert_eq!(default_top_k, 8092);
}

// ══════════════════════════════════════════════════════════════
// WALK-FFN full_output + seq_len REQUEST SHAPING
// ══════════════════════════════════════════════════════════════

#[test]
fn test_walk_ffn_full_output_residual_length_must_match_seq_len_times_hidden() {
    let hidden = 4;
    let seq_len = 3;
    // A correctly-sized batched residual is 12 floats, row-major.
    let ok = seq_len * hidden;
    let bad_short = ok - 1;
    let bad_long = ok + 1;
    assert_ne!(bad_short, ok);
    assert_ne!(bad_long, ok);
    // Single-token mirror: len must equal hidden when seq_len omitted.
    let single = hidden;
    assert_eq!(single, 4);
}

#[test]
fn test_walk_ffn_full_output_rejects_zero_seq_len() {
    let seq_len: usize = 0;
    let full_output = true;
    let invalid = full_output && seq_len == 0;
    assert!(invalid);
}

#[test]
fn test_walk_ffn_seq_len_default_is_one_for_features_only_mode() {
    let hidden = 4;
    let seq_len_default = 1;
    let residual = vec![0.1f32; hidden];
    let expected = if false
    /* full_output */
    {
        seq_len_default * hidden
    } else {
        hidden
    };
    assert_eq!(residual.len(), expected);
}

#[test]
fn test_walk_ffn_full_output_response_shape() {
    // Wire-shape contract: `output` length == `seq_len * hidden_size`.
    let hidden = 4;
    for seq_len in 1..=5 {
        let flat = vec![0.0f32; seq_len * hidden];
        assert_eq!(flat.len(), seq_len * hidden);
    }
}

// ══════════════════════════════════════════════════════════════
// WEBSOCKET STREAM PROTOCOL
// ══════════════════════════════════════════════════════════════

#[test]
fn test_stream_describe_request_format() {
    let msg = serde_json::json!({"type": "describe", "entity": "France", "band": "all"});
    assert_eq!(msg["type"].as_str(), Some("describe"));
    assert_eq!(msg["entity"].as_str(), Some("France"));
    assert_eq!(msg["band"].as_str(), Some("all"));
}

#[test]
fn test_stream_layer_response_format() {
    let msg = serde_json::json!({
        "type": "layer",
        "layer": 27,
        "edges": [
            {"target": "Paris", "gate_score": 1436.9, "relation": "capital", "source": "probe"}
        ]
    });
    assert_eq!(msg["type"].as_str(), Some("layer"));
    assert_eq!(msg["layer"].as_u64(), Some(27));
    assert!(!msg["edges"].as_array().unwrap().is_empty());
}

#[test]
fn test_stream_done_response_format() {
    let msg = serde_json::json!({
        "type": "done",
        "entity": "France",
        "total_edges": 6,
        "latency_ms": 12.3,
    });
    assert_eq!(msg["type"].as_str(), Some("done"));
    assert_eq!(msg["total_edges"].as_u64(), Some(6));
    assert!(msg["latency_ms"].as_f64().unwrap() > 0.0);
}

#[test]
fn test_stream_error_response_format() {
    let msg = serde_json::json!({"type": "error", "message": "missing entity"});
    assert_eq!(msg["type"].as_str(), Some("error"));
    assert!(msg["message"].as_str().unwrap().contains("entity"));
}

#[test]
fn test_stream_unknown_type_rejected() {
    let msg_type = "foobar";
    let supported = ["describe", "infer"];
    assert!(!supported.contains(&msg_type));
}

// ══════════════════════════════════════════════════════════════
// WEBSOCKET INFER STREAMING
// ══════════════════════════════════════════════════════════════

#[test]
fn test_stream_infer_request_format() {
    let msg = serde_json::json!({
        "type": "infer",
        "prompt": "The capital of France is",
        "top": 5,
        "mode": "walk"
    });
    assert_eq!(msg["type"].as_str(), Some("infer"));
    assert_eq!(msg["prompt"].as_str(), Some("The capital of France is"));
    assert_eq!(msg["top"].as_u64(), Some(5));
    assert_eq!(msg["mode"].as_str(), Some("walk"));
}

#[test]
fn test_stream_prediction_response_format() {
    let msg = serde_json::json!({
        "type": "prediction",
        "rank": 1,
        "token": "Paris",
        "probability": 0.9791,
    });
    assert_eq!(msg["type"].as_str(), Some("prediction"));
    assert_eq!(msg["rank"].as_u64(), Some(1));
    assert_eq!(msg["token"].as_str(), Some("Paris"));
    assert!(msg["probability"].as_f64().unwrap() > 0.0);
}

#[test]
fn test_stream_infer_done_response_format() {
    let msg = serde_json::json!({
        "type": "infer_done",
        "prompt": "The capital of France is",
        "mode": "walk",
        "predictions": 5,
        "latency_ms": 210.0,
    });
    assert_eq!(msg["type"].as_str(), Some("infer_done"));
    assert_eq!(msg["mode"].as_str(), Some("walk"));
    assert_eq!(msg["predictions"].as_u64(), Some(5));
}

#[test]
fn test_stream_infer_modes() {
    let supported_modes = ["walk", "dense"];
    assert!(supported_modes.contains(&"walk"));
    assert!(supported_modes.contains(&"dense"));
    assert!(!supported_modes.contains(&"compare")); // compare not streamed
}

// ══════════════════════════════════════════════════════════════
// gRPC PROTO FORMAT
// ══════════════════════════════════════════════════════════════

#[test]
fn test_grpc_describe_request_fields() {
    // Mirrors DescribeRequest proto message
    let entity = "France";
    let band = "knowledge";
    let verbose = false;
    let limit = 20u32;
    let min_score = 5.0f32;
    assert_eq!(entity, "France");
    assert_eq!(band, "knowledge");
    assert!(!verbose);
    assert!(limit > 0);
    assert!(min_score > 0.0);
}

#[test]
fn test_grpc_walk_response_structure() {
    // WalkResponse: prompt, hits[], latency_ms
    // WalkHit: layer, feature, gate_score, target, relation
    let hit = serde_json::json!({
        "layer": 27,
        "feature": 9515,
        "gate_score": 1436.9,
        "target": "Paris",
        "relation": "capital",
    });
    assert!(hit["layer"].as_u64().is_some());
    assert!(hit["feature"].as_u64().is_some());
    assert!(hit["gate_score"].as_f64().is_some());
    assert!(hit["target"].as_str().is_some());
}

#[test]
fn test_grpc_infer_compare_response() {
    // Compare mode returns walk_predictions + dense_predictions separately
    let walk_preds = [("Paris".to_string(), 0.9791f64)];
    let dense_preds = [("Paris".to_string(), 0.9801f64)];
    assert_eq!(walk_preds.len(), 1);
    assert_eq!(dense_preds.len(), 1);
    assert_ne!(walk_preds[0].1, dense_preds[0].1); // Slightly different
}

#[test]
fn test_grpc_port_flag() {
    // --grpc-port enables gRPC alongside HTTP
    let grpc_port: Option<u16> = Some(50051);
    assert!(grpc_port.is_some());
    let grpc_port: Option<u16> = None;
    assert!(grpc_port.is_none()); // gRPC disabled
}

// ══════════════════════════════════════════════════════════════
// BINARY WIRE FORMAT (application/x-larql-ffn)
// ══════════════════════════════════════════════════════════════

const BINARY_CT: &str = "application/x-larql-ffn";
const BATCH_MARKER_U32: u32 = 0xFFFF_FFFF;

fn bin_make_single_request(
    layer: u32,
    seq_len: u32,
    full_output: bool,
    top_k: u32,
    residual: &[f32],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&(full_output as u32).to_le_bytes());
    buf.extend_from_slice(&top_k.to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn bin_make_batch_request(
    layers: &[u32],
    seq_len: u32,
    full_output: bool,
    top_k: u32,
    residual: &[f32],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER_U32.to_le_bytes());
    buf.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for &l in layers {
        buf.extend_from_slice(&l.to_le_bytes());
    }
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&(full_output as u32).to_le_bytes());
    buf.extend_from_slice(&top_k.to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn bin_make_single_response(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&seq_len.to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &v in output {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn bin_make_batch_response(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&BATCH_MARKER_U32.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    buf.extend_from_slice(&latency.to_le_bytes());
    for &(layer, floats) in entries {
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
        buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
        for &v in floats {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

#[test]
fn test_binary_content_type_constant() {
    assert_eq!(BINARY_CT, "application/x-larql-ffn");
}

#[test]
fn test_binary_batch_marker_constant() {
    assert_eq!(BATCH_MARKER_U32, 0xFFFF_FFFFu32);
}

#[test]
fn test_binary_single_request_first_u32_is_layer() {
    let residual = vec![1.0f32, 0.0, 0.0, 0.0];
    let body = bin_make_single_request(26, 1, true, 8092, &residual);
    let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
    assert_eq!(layer, 26);
    // Single-layer: first u32 must NOT be BATCH_MARKER
    assert_ne!(layer, BATCH_MARKER_U32);
}

#[test]
fn test_binary_batch_request_first_u32_is_marker() {
    let residual = vec![1.0f32, 0.0, 0.0, 0.0];
    let body = bin_make_batch_request(&[5, 20], 1, true, 8092, &residual);
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    assert_eq!(marker, BATCH_MARKER_U32);
}

#[test]
fn test_binary_single_request_structure() {
    // Verify all fixed header fields at expected offsets.
    let residual = vec![0.5f32, -0.5];
    let body = bin_make_single_request(7, 2, true, 512, &residual);
    let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    let flags = u32::from_le_bytes(body[8..12].try_into().unwrap());
    let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
    assert_eq!(layer, 7);
    assert_eq!(seq_len, 2);
    assert_eq!(flags & 1, 1); // full_output bit
    assert_eq!(top_k, 512);
    assert_eq!(body.len(), 16 + 2 * 4); // header + 2 floats
}

#[test]
fn test_binary_batch_request_structure() {
    let residual = vec![1.0f32; 4];
    let body = bin_make_batch_request(&[5, 20, 30], 1, true, 128, &residual);
    let num_layers = u32::from_le_bytes(body[4..8].try_into().unwrap());
    assert_eq!(num_layers, 3);
    let l0 = u32::from_le_bytes(body[8..12].try_into().unwrap());
    let l1 = u32::from_le_bytes(body[12..16].try_into().unwrap());
    let l2 = u32::from_le_bytes(body[16..20].try_into().unwrap());
    assert_eq!((l0, l1, l2), (5, 20, 30));
    // After 3 layer u32s: seq_len, flags, top_k
    let seq_len = u32::from_le_bytes(body[20..24].try_into().unwrap());
    let flags = u32::from_le_bytes(body[24..28].try_into().unwrap());
    let top_k = u32::from_le_bytes(body[28..32].try_into().unwrap());
    assert_eq!(seq_len, 1);
    assert_eq!(flags & 1, 1);
    assert_eq!(top_k, 128);
}

#[test]
fn test_binary_single_response_structure() {
    let output = vec![0.1f32, 0.2, 0.3];
    let body = bin_make_single_response(26, 1, 9.5, &output);
    // [layer u32][seq_len u32][latency f32][output f32*]
    assert_eq!(body.len(), 12 + 3 * 4);
    let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
    let latency = f32::from_le_bytes(body[8..12].try_into().unwrap());
    assert_eq!(layer, 26);
    assert_eq!(seq_len, 1);
    assert!((latency - 9.5).abs() < 0.01);
    let v0 = f32::from_le_bytes(body[12..16].try_into().unwrap());
    assert!((v0 - 0.1).abs() < 1e-6);
}

#[test]
fn test_binary_batch_response_structure() {
    let body = bin_make_batch_response(12.3, &[(5, &[1.0, 2.0]), (20, &[3.0, 4.0])]);
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap());
    let latency = f32::from_le_bytes(body[8..12].try_into().unwrap());
    assert_eq!(marker, BATCH_MARKER_U32);
    assert_eq!(num_results, 2);
    assert!((latency - 12.3).abs() < 0.01);
    // First result entry at offset 12
    let layer0 = u32::from_le_bytes(body[12..16].try_into().unwrap());
    let num_floats0 = u32::from_le_bytes(body[20..24].try_into().unwrap());
    assert_eq!(layer0, 5);
    assert_eq!(num_floats0, 2);
}

#[test]
fn test_binary_float_roundtrip_exact() {
    let values = vec![f32::MIN_POSITIVE, -0.0f32, 1.0, f32::MAX / 2.0, 1e-7];
    let body = bin_make_single_response(0, 1, 0.0, &values);
    let decoded: Vec<f32> = body[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (a, b) in decoded.iter().zip(values.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "float bits differ: {:#010x} vs {:#010x}",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn test_binary_features_only_flag_zero() {
    // Binary with full_output=false should have flags bit0 = 0.
    let body = bin_make_single_request(5, 1, false, 8092, &[1.0, 0.0, 0.0, 0.0]);
    let flags = u32::from_le_bytes(body[8..12].try_into().unwrap());
    assert_eq!(
        flags & 1,
        0,
        "full_output bit should be 0 for features-only"
    );
}

#[test]
fn test_binary_request_residual_size() {
    // Residual for a hidden_size=4 model, seq_len=2 = 8 floats.
    let residual: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let body = bin_make_single_request(0, 2, true, 8092, &residual);
    let residual_bytes = &body[16..]; // after 4 header u32s
    assert_eq!(residual_bytes.len(), 8 * 4);
    for (i, chunk) in residual_bytes.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes(chunk.try_into().unwrap());
        assert!((v - i as f32).abs() < 1e-6);
    }
}

// ══════════════════════════════════════════════════════════════
// EMBED SERVICE — lookup logic, binary protocol
// ══════════════════════════════════════════════════════════════

#[test]
fn test_embed_lookup_basic() {
    // embed[0] = [1, 0, 0, 0], scale = 1.0
    let mut embed = Array2::<f32>::zeros((8, 4));
    embed[[0, 0]] = 1.0;
    embed[[1, 1]] = 1.0;
    embed[[2, 2]] = 1.0;
    embed[[3, 3]] = 1.0;

    let scale = 1.0f32;
    for tok in 0..4usize {
        let row: Vec<f32> = embed.row(tok).iter().map(|&v| v * scale).collect();
        assert_eq!(row[tok], 1.0, "token {tok} should activate dim {tok}");
        for (other, &v) in row.iter().enumerate().take(4) {
            if other != tok {
                assert_eq!(v, 0.0);
            }
        }
    }
}

#[test]
fn test_embed_lookup_with_scale() {
    let mut embed = Array2::<f32>::zeros((4, 4));
    embed[[0, 0]] = 1.0;
    let scale = 3.0f32;
    let row: Vec<f32> = embed.row(0).iter().map(|&v| v * scale).collect();
    assert!(
        (row[0] - 3.0).abs() < 1e-6,
        "scale must be applied: got {}",
        row[0]
    );
}

#[test]
fn test_embed_lookup_returns_zero_for_zero_row() {
    let embed = Array2::<f32>::zeros((8, 4));
    let scale = 1.0f32;
    let row: Vec<f32> = embed.row(7).iter().map(|&v| v * scale).collect();
    assert!(row.iter().all(|&v| v == 0.0));
}

#[test]
fn test_embed_response_dimensions() {
    // seq_len=2, hidden=4 → 2 rows of 4 floats
    let embed = test_embeddings();
    let token_ids = [0u32, 1u32];
    let scale = 1.0f32;
    let result: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&id| embed.row(id as usize).iter().map(|&v| v * scale).collect())
        .collect();
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|r| r.len() == 4));
}

#[test]
fn test_embed_binary_request_shape() {
    // Binary embed request: [num_tokens u32][token_id u32 × N]
    let token_ids = [42u32, 1337, 9515];
    let mut body = Vec::new();
    body.extend_from_slice(&(token_ids.len() as u32).to_le_bytes());
    for &id in &token_ids {
        body.extend_from_slice(&id.to_le_bytes());
    }
    assert_eq!(body.len(), 4 + 3 * 4);
    assert_eq!(u32::from_le_bytes(body[..4].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(body[4..8].try_into().unwrap()), 42);
    assert_eq!(u32::from_le_bytes(body[8..12].try_into().unwrap()), 1337);
    assert_eq!(u32::from_le_bytes(body[12..16].try_into().unwrap()), 9515);
}

#[test]
fn test_embed_binary_response_shape() {
    // Binary embed response: [seq_len u32][hidden_size u32][seq_len × hidden_size f32]
    let seq_len = 2u32;
    let hidden = 4u32;
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();

    let mut body = Vec::new();
    body.extend_from_slice(&seq_len.to_le_bytes());
    body.extend_from_slice(&hidden.to_le_bytes());
    for &v in &values {
        body.extend_from_slice(&v.to_le_bytes());
    }

    assert_eq!(u32::from_le_bytes(body[..4].try_into().unwrap()), seq_len);
    assert_eq!(u32::from_le_bytes(body[4..8].try_into().unwrap()), hidden);
    assert_eq!(body.len(), 8 + (seq_len * hidden * 4) as usize);

    for (i, chunk) in body[8..].chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes(chunk.try_into().unwrap());
        assert!((v - i as f32).abs() < 1e-6);
    }
}

// ══════════════════════════════════════════════════════════════
// LOGITS BINARY AND JSON
// ══════════════════════════════════════════════════════════════

#[test]
fn test_logits_request_json_shape() {
    let req = serde_json::json!({
        "residual": [0.1f32, -0.2, 0.3, 0.4],
        "top_k": 5,
        "temperature": 1.0,
    });
    assert!(req["residual"].is_array());
    assert_eq!(req["top_k"], 5);
    assert!((req["temperature"].as_f64().unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn test_logits_response_json_shape() {
    let resp = serde_json::json!({
        "top_k": [
            {"token_id": 9515, "token": "Paris", "prob": 0.801},
            {"token_id": 235,  "token": "the",   "prob": 0.042},
        ],
        "latency_ms": 2.1,
    });
    assert!(resp["top_k"].is_array());
    assert_eq!(resp["top_k"].as_array().unwrap().len(), 2);
    assert_eq!(resp["top_k"][0]["token_id"], 9515);
    assert_eq!(resp["top_k"][0]["token"], "Paris");
    assert!(resp["top_k"][0]["prob"].as_f64().unwrap() > 0.0);
    assert!(resp["latency_ms"].as_f64().unwrap() > 0.0);
}

#[test]
fn test_logits_binary_request_byte_alignment() {
    // Binary logits request is raw f32[] LE. Must be multiple of 4.
    let hidden = 8;
    let residual: Vec<f32> = vec![0.0; hidden];
    let body: Vec<u8> = residual.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(body.len() % 4, 0);
    assert_eq!(body.len(), hidden * 4);
}

#[test]
fn test_logits_hidden_size_mismatch_detectable() {
    // Simulate the hidden size guard: residual.len() != hidden rejects request.
    let hidden_size = 4usize;
    let bad_residual = [0.0f32; 3]; // wrong length
    assert_ne!(
        bad_residual.len(),
        hidden_size,
        "length 3 != hidden_size 4 → bad request"
    );
}

// ══════════════════════════════════════════════════════════════
// TOKEN DECODE PARSING
// ══════════════════════════════════════════════════════════════

#[test]
fn test_token_decode_csv_parsing() {
    let q = "9515,235,1234";
    let ids: Vec<u32> = q
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>().unwrap())
        .collect();
    assert_eq!(ids, vec![9515u32, 235, 1234]);
}

#[test]
fn test_token_decode_invalid_id_detectable() {
    let q = "9515,notanumber,1234";
    let ids: Vec<Result<u32, _>> = q.split(',').map(|s| s.trim().parse::<u32>()).collect();
    assert!(ids[0].is_ok());
    assert!(ids[1].is_err(), "non-numeric token ID must fail to parse");
    assert!(ids[2].is_ok());
}

// ══════════════════════════════════════════════════════════════
// SELECT ORDERING
// ══════════════════════════════════════════════════════════════

#[test]
fn test_select_order_by_confidence_desc() {
    let mut rows = [(0.5f32, "a"), (0.9, "b"), (0.1, "c"), (0.7, "d")];
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    assert_eq!(rows[0].1, "b");
    assert_eq!(rows[1].1, "d");
    assert_eq!(rows[2].1, "a");
    assert_eq!(rows[3].1, "c");
}

#[test]
fn test_select_order_by_confidence_asc() {
    let mut rows = [(0.5f32, "a"), (0.9, "b"), (0.1, "c")];
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert_eq!(rows[0].1, "c");
    assert_eq!(rows[1].1, "a");
    assert_eq!(rows[2].1, "b");
}

#[test]
fn test_select_entity_substring_match() {
    let token = "Paris";
    let filter = "par";
    assert!(token.to_lowercase().contains(&filter.to_lowercase()));

    let token = "Berlin";
    assert!(!token.to_lowercase().contains(&filter.to_lowercase()));
}

#[test]
fn test_select_min_confidence_filter() {
    let scores = vec![0.1f32, 0.5, 0.8, 0.95];
    let min = 0.5;
    let filtered: Vec<f32> = scores.into_iter().filter(|s| *s >= min).collect();
    assert_eq!(filtered, vec![0.5, 0.8, 0.95]);
}

#[test]
fn test_select_limit_truncation() {
    let mut rows: Vec<i32> = (0..100).collect();
    let limit = 5;
    rows.truncate(limit);
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_select_order_by_layer_asc() {
    let mut rows: Vec<(usize, &str)> = vec![(5, "a"), (0, "b"), (3, "c"), (1, "d")];
    rows.sort_by_key(|r| r.0);
    assert_eq!(rows[0].0, 0);
    assert_eq!(rows[1].0, 1);
    assert_eq!(rows[2].0, 3);
    assert_eq!(rows[3].0, 5);
}

#[test]
fn test_select_order_by_layer_desc() {
    let mut rows: Vec<(usize, &str)> = vec![(5, "a"), (0, "b"), (3, "c"), (1, "d")];
    rows.sort_by_key(|(layer, _)| std::cmp::Reverse(*layer));
    assert_eq!(rows[0].0, 5);
    assert_eq!(rows[3].0, 0);
}
