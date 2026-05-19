use serde::{Deserialize, Serialize};

// ── Binary wire format ────────────────────────────────────────────────────────
//
// Content-Type: application/x-larql-expert
//
// Request:  [N u32][hidden u32] + N × [layer u32][expert_id u32][f32 × hidden]
// Response: [N u32][hidden u32][latency_ms f32] + N × [layer u32][expert_id u32][f32 × hidden]
//
// All integers and floats are little-endian.  This is ~6× smaller than JSON
// for typical 2816-float payloads and avoids serde_json float formatting.

pub const EXPERT_BINARY_CONTENT_TYPE: &str = "application/x-larql-expert";

/// Decoded layer-batch request: `(layer, residual, expert_ids, expert_weights)`.
pub type DecodedLayerBatchRequest = (usize, Vec<f32>, Vec<u32>, Vec<f32>);

/// HTTP path served by the single-expert / batched-expert binary
/// endpoint matched to [`EXPERT_BINARY_CONTENT_TYPE`].
pub const EXPERT_BATCH_PATH: &str = "/v1/expert/batch";

/// Content type for the `/v1/experts/layer-batch` endpoint — the layer-batched
/// MoE wire format that ships one residual + K (expert_id, weight) pairs and
/// receives back ONE weighted-sum vector.  Eliminates the K-1 redundant
/// residual copies on the wire (~78 KB per call at Gemma 4 26B-A4B sizes)
/// and the K-1 redundant `pre_experts_norm` + Q8_K quantisations on the
/// server (~10-20 µs per layer of CPU work).
pub const LAYER_BATCH_CONTENT_TYPE: &str = "application/x-larql-experts-layer";

/// HTTP path for the f32 layer-batch endpoint.
pub const LAYER_BATCH_PATH: &str = "/v1/experts/layer-batch";

/// f16 variant of the layer-batch wire format.  Halves the per-call wire
/// bytes (residual + weighted-sum response): 11 KB → 5.5 KB at hidden=2816.
/// Quantisation is `f32 → IEEE-754 half`, ~3 decimal digits of precision —
/// well within MoE activation noise (Q8_K already adds ~0.4% per-element
/// quant error on the activation in the SDOT path; f16 wire adds another
/// ~0.05% which is negligible).  Mathematically identical when both sides
/// dequantise to f32 before compute.
pub const LAYER_BATCH_F16_CONTENT_TYPE: &str = "application/x-larql-experts-layer-f16";

/// HTTP path for the f16 layer-batch endpoint.
pub const LAYER_BATCH_F16_PATH: &str = "/v1/experts/layer-batch-f16";

fn checked_mul(a: usize, b: usize) -> Option<usize> {
    a.checked_mul(b)
}

fn checked_add(a: usize, b: usize) -> Option<usize> {
    a.checked_add(b)
}

// ── Layer-batch wire format ───────────────────────────────────────────────────
//
// Content-Type: application/x-larql-experts-layer
//
// Request:  [layer u32][hidden u32][K u32]
//           + hidden × f32  (residual, sent ONCE)
//           + K × [expert_id u32, weight f32]
//
// Response: [hidden u32][latency_ms f32]
//           + hidden × f32  (router-weighted sum across the K experts)
//
// Server-side fast path: the response is the result of
// `run_experts_cpu_batch(layer, residual, expert_ids, expert_weights)` — the
// server applies pre_experts_norm once, quantises h_norm to Q8_K once, and
// fans out the K expert kernels with the shared activation.

/// Encode a layer-batch request.
pub fn encode_layer_batch_request(
    layer: usize,
    residual: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
) -> Vec<u8> {
    let hidden = residual.len();
    let k = expert_ids.len();
    debug_assert_eq!(k, expert_weights.len());
    let mut buf = Vec::with_capacity(12 + hidden * 4 + k * 8);
    buf.extend_from_slice(&(layer as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&(k as u32).to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for (i, &eid) in expert_ids.iter().enumerate() {
        buf.extend_from_slice(&eid.to_le_bytes());
        buf.extend_from_slice(&expert_weights[i].to_le_bytes());
    }
    buf
}

/// Decode a layer-batch request from raw bytes.  Returns
/// `(layer, residual, expert_ids, expert_weights)` or `None` on truncation.
pub fn decode_layer_batch_request(bytes: &[u8]) -> Option<DecodedLayerBatchRequest> {
    if bytes.len() < 12 {
        return None;
    }
    let layer = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let k = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let residual_bytes = checked_mul(hidden, 4)?;
    let expert_bytes = checked_mul(k, 8)?;
    let want = checked_add(checked_add(12, residual_bytes)?, expert_bytes)?;
    if bytes.len() < want {
        return None;
    }
    let mut pos = 12usize;
    let residual: Vec<f32> = bytes[pos..pos + residual_bytes]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    pos += residual_bytes;
    let mut expert_ids = Vec::with_capacity(k);
    let mut expert_weights = Vec::with_capacity(k);
    for _ in 0..k {
        let eid = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?);
        let w = f32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?);
        expert_ids.push(eid);
        expert_weights.push(w);
        pos += 8;
    }
    Some((layer, residual, expert_ids, expert_weights))
}

/// Encode a layer-batch response (one weighted-sum vector).
pub fn encode_layer_batch_response(weighted_sum: &[f32], latency_ms: f32) -> Vec<u8> {
    let hidden = weighted_sum.len();
    let mut buf = Vec::with_capacity(8 + hidden * 4);
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&latency_ms.to_le_bytes());
    for &v in weighted_sum {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Decode a layer-batch response.  Returns the weighted-sum vector or `None`
/// on truncation.  Discards the latency_ms field (informational only).
pub fn decode_layer_batch_response(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() < 8 {
        return None;
    }
    let hidden = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let payload_bytes = checked_mul(hidden, 4)?;
    let want = checked_add(8, payload_bytes)?;
    if bytes.len() < want {
        return None;
    }
    Some(
        bytes[8..want]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}

// ── f16 wire helpers ──────────────────────────────────────────────────────────
// IEEE-754 binary16 conversion via the `half` crate (already a workspace dep).

#[inline(always)]
pub(super) fn f32_to_f16_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[inline(always)]
pub(super) fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Encode a layer-batch request with f16 residual.  Same shape as the f32
/// version but residual bytes are 2 per element (vs 4).  Header layout
/// `[layer u32][hidden u32][K u32]` is unchanged so the server can size
/// the read slice correctly.
pub fn encode_layer_batch_request_f16(
    layer: usize,
    residual: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
) -> Vec<u8> {
    let hidden = residual.len();
    let k = expert_ids.len();
    debug_assert_eq!(k, expert_weights.len());
    let mut buf = Vec::with_capacity(12 + hidden * 2 + k * 8);
    buf.extend_from_slice(&(layer as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&(k as u32).to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    for (i, &eid) in expert_ids.iter().enumerate() {
        buf.extend_from_slice(&eid.to_le_bytes());
        // Weights stay f32 — only K of them, and they're routing
        // probabilities (small dynamic range, but full f32 precision keeps
        // the renormalised sum exactly 1.0).
        buf.extend_from_slice(&expert_weights[i].to_le_bytes());
    }
    buf
}

/// Decode an f16 layer-batch request.  Reconstructs `residual` to f32 on
/// the server before passing into `run_experts_cpu_batch`.
pub fn decode_layer_batch_request_f16(bytes: &[u8]) -> Option<DecodedLayerBatchRequest> {
    if bytes.len() < 12 {
        return None;
    }
    let layer = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let k = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let residual_bytes = checked_mul(hidden, 2)?;
    let expert_bytes = checked_mul(k, 8)?;
    let want = checked_add(checked_add(12, residual_bytes)?, expert_bytes)?;
    if bytes.len() < want {
        return None;
    }
    let mut pos = 12usize;
    let residual: Vec<f32> = bytes[pos..pos + residual_bytes]
        .chunks_exact(2)
        .map(|b| f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    pos += residual_bytes;
    let mut expert_ids = Vec::with_capacity(k);
    let mut expert_weights = Vec::with_capacity(k);
    for _ in 0..k {
        let eid = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?);
        let w = f32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?);
        expert_ids.push(eid);
        expert_weights.push(w);
        pos += 8;
    }
    Some((layer, residual, expert_ids, expert_weights))
}

/// Encode the f16 layer-batch response (weighted-sum vector packed as f16).
pub fn encode_layer_batch_response_f16(weighted_sum: &[f32], latency_ms: f32) -> Vec<u8> {
    let hidden = weighted_sum.len();
    let mut buf = Vec::with_capacity(8 + hidden * 2);
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&latency_ms.to_le_bytes());
    for &v in weighted_sum {
        buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    buf
}

/// Decode the f16 layer-batch response back to f32 for client-side
/// accumulation.
pub fn decode_layer_batch_response_f16(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() < 8 {
        return None;
    }
    let hidden = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let payload_bytes = checked_mul(hidden, 2)?;
    let want = checked_add(8, payload_bytes)?;
    if bytes.len() < want {
        return None;
    }
    Some(
        bytes[8..want]
            .chunks_exact(2)
            .map(|b| f16_bits_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
    )
}

/// Encode a batch of expert requests as binary.
pub fn encode_expert_request(items: &[ExpertCallItem]) -> Vec<u8> {
    let n = items.len();
    let hidden = items.first().map(|r| r.residual.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(8 + n * (8 + hidden * 4));
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    for item in items {
        buf.extend_from_slice(&(item.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(item.expert_id as u32).to_le_bytes());
        for &v in &item.residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

/// Decode a binary expert response. Returns None on truncation.
pub fn decode_expert_response(bytes: &[u8]) -> Option<Vec<ExpertResultItem>> {
    if bytes.len() < 12 {
        return None;
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    // bytes[8..12] = latency_ms f32 (informational, skip)
    let mut pos = 12usize;
    let payload_bytes = checked_mul(hidden, 4)?;
    let item_bytes = checked_add(8, payload_bytes)?;
    let want = checked_add(12, checked_mul(n, item_bytes)?)?;
    if bytes.len() < want {
        return None;
    }
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        let expert_id = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        pos += 8;
        let output: Vec<f32> = bytes[pos..pos + payload_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += payload_bytes;
        results.push(ExpertResultItem {
            layer,
            expert_id,
            output,
        });
    }
    Some(results)
}

/// Decode a binary expert request from the server side.
pub fn decode_expert_request(bytes: &[u8]) -> Option<Vec<ExpertCallItem>> {
    if bytes.len() < 8 {
        return None;
    }
    let n = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let hidden = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let mut pos = 8usize;
    let payload_bytes = checked_mul(hidden, 4)?;
    let item_bytes = checked_add(8, payload_bytes)?;
    let want = checked_add(8, checked_mul(n, item_bytes)?)?;
    if bytes.len() < want {
        return None;
    }
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        let expert_id = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
        pos += 8;
        let residual: Vec<f32> = bytes[pos..pos + payload_bytes]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        pos += payload_bytes;
        items.push(ExpertCallItem {
            layer,
            expert_id,
            residual,
        });
    }
    Some(items)
}

/// Encode a batch of expert results as binary (server-side response).
pub fn encode_expert_response(items: &[ExpertResultItem], latency_ms: f32) -> Vec<u8> {
    let n = items.len();
    let hidden = items.first().map(|r| r.output.len()).unwrap_or(0);
    let mut buf = Vec::with_capacity(12 + n * (8 + hidden * 4));
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    buf.extend_from_slice(&(hidden as u32).to_le_bytes());
    buf.extend_from_slice(&latency_ms.to_le_bytes());
    for item in items {
        buf.extend_from_slice(&(item.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(item.expert_id as u32).to_le_bytes());
        for &v in &item.output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
pub struct ExpertCallItem {
    pub layer: usize,
    pub expert_id: usize,
    pub residual: Vec<f32>,
}

#[derive(Deserialize)]
pub struct ExpertResultItem {
    pub layer: usize,
    pub expert_id: usize,
    pub output: Vec<f32>,
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn layer_batch_request_roundtrips() {
        let residual = vec![1.0, -2.0, 3.5];
        let expert_ids = vec![7, 9];
        let expert_weights = vec![0.25, 0.75];
        let body = encode_layer_batch_request(3, &residual, &expert_ids, &expert_weights);

        let (layer, decoded_residual, decoded_ids, decoded_weights) =
            decode_layer_batch_request(&body).unwrap();
        assert_eq!(layer, 3);
        assert_eq!(decoded_residual, residual);
        assert_eq!(decoded_ids, expert_ids);
        assert_eq!(decoded_weights, expert_weights);
    }

    #[test]
    fn layer_batch_request_rejects_impossible_k_before_allocating() {
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_layer_batch_request(&body).is_none());
    }

    #[test]
    fn layer_batch_response_rejects_impossible_hidden_before_allocating() {
        let mut body = Vec::new();
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        assert!(decode_layer_batch_response(&body).is_none());
        assert!(decode_layer_batch_response_f16(&body).is_none());
    }

    #[test]
    fn expert_request_roundtrips() {
        let items = vec![
            ExpertCallItem {
                layer: 1,
                expert_id: 2,
                residual: vec![0.1, 0.2],
            },
            ExpertCallItem {
                layer: 3,
                expert_id: 4,
                residual: vec![0.3, 0.4],
            },
        ];
        let body = encode_expert_request(&items);
        let decoded = decode_expert_request(&body).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].layer, 1);
        assert_eq!(decoded[1].expert_id, 4);
        assert_eq!(decoded[1].residual, vec![0.3, 0.4]);
    }

    #[test]
    fn expert_response_rejects_overflowing_item_layout_before_allocating() {
        let mut body = Vec::new();
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        assert!(decode_expert_response(&body).is_none());
    }

    #[test]
    fn layer_batch_response_round_trips() {
        let weighted = vec![0.1f32, -0.5, 1.25, 3.0];
        let body = encode_layer_batch_response(&weighted, 7.5);
        let decoded = decode_layer_batch_response(&body).unwrap();
        assert_eq!(decoded, weighted);
    }

    #[test]
    fn decode_layer_batch_response_returns_none_on_truncation() {
        // Header says hidden=4 (16 bytes), but only 8 bytes after header.
        let mut body = Vec::new();
        body.extend_from_slice(&4u32.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        body.extend_from_slice(&[0u8; 8]); // need 16, have 8
        assert!(decode_layer_batch_response(&body).is_none());
    }

    #[test]
    fn decode_layer_batch_response_returns_none_on_short_header() {
        // < 8 bytes total — truncation before hidden+latency header.
        assert!(decode_layer_batch_response(&[0u8; 4]).is_none());
    }

    #[test]
    fn decode_layer_batch_request_returns_none_on_truncation() {
        // Header advertises hidden=2 (8 bytes residual), K=1 (8 bytes for
        // expert id+weight) → 12 + 8 + 8 = 28 bytes. Provide only 16.
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes()); // layer
        body.extend_from_slice(&2u32.to_le_bytes()); // hidden
        body.extend_from_slice(&1u32.to_le_bytes()); // K
        body.extend_from_slice(&[0u8; 4]); // partial residual
        assert!(decode_layer_batch_request(&body).is_none());
    }

    #[test]
    fn decode_layer_batch_request_returns_none_on_short_header() {
        // < 12 bytes total (no full layer/hidden/K header).
        assert!(decode_layer_batch_request(&[0u8; 8]).is_none());
    }

    #[test]
    fn expert_response_round_trips() {
        // Drives `encode_expert_response` and the happy path of
        // `decode_expert_response` (the existing test only covers the
        // overflow-guard early return).
        let items = vec![
            ExpertResultItem {
                layer: 5,
                expert_id: 11,
                output: vec![1.0, 2.0, 3.0],
            },
            ExpertResultItem {
                layer: 6,
                expert_id: 12,
                output: vec![4.0, 5.0, 6.0],
            },
        ];
        let body = encode_expert_response(&items, 0.5);
        let decoded = decode_expert_response(&body).expect("decode should succeed");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].layer, 5);
        assert_eq!(decoded[0].expert_id, 11);
        assert_eq!(decoded[0].output, vec![1.0, 2.0, 3.0]);
        assert_eq!(decoded[1].output, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn decode_expert_response_short_header_returns_none() {
        // < 12 bytes (no full n+hidden+latency header).
        assert!(decode_expert_response(&[0u8; 8]).is_none());
    }

    #[test]
    fn decode_expert_request_short_header_returns_none() {
        // < 8 bytes header.
        assert!(decode_expert_request(&[0u8; 4]).is_none());
    }

    #[test]
    fn decode_expert_request_returns_none_on_truncated_payload() {
        // Header n=1, hidden=2: per item = 8 + 8 = 16 bytes. Body has 0.
        let mut body = Vec::new();
        body.extend_from_slice(&1u32.to_le_bytes()); // n
        body.extend_from_slice(&2u32.to_le_bytes()); // hidden
                                                     // Missing the 16-byte item.
        assert!(decode_expert_request(&body).is_none());
    }

    #[test]
    fn decode_layer_batch_request_f16_short_header_returns_none() {
        assert!(decode_layer_batch_request_f16(&[0u8; 8]).is_none());
    }

    #[test]
    fn decode_layer_batch_request_f16_truncated_payload_returns_none() {
        // hidden=4 (8 bytes f16), K=1 (8 bytes); need 12 + 16 = 28 total.
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&4u32.to_le_bytes());
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&[0u8; 4]); // partial
        assert!(decode_layer_batch_request_f16(&body).is_none());
    }

    #[test]
    fn encode_layer_batch_response_writes_expected_byte_count() {
        let v = vec![1.0f32, 2.0, 3.0];
        let body = encode_layer_batch_response(&v, 0.0);
        // 4 (hidden) + 4 (latency) + 3*4 (f32 payload) = 20 bytes.
        assert_eq!(body.len(), 4 + 4 + 3 * 4);
    }

    #[test]
    fn encode_expert_response_handles_empty_batch() {
        // n=0 still emits the full 12-byte header.
        let body = encode_expert_response(&[], 1.0);
        assert_eq!(body.len(), 12);
        let decoded = decode_expert_response(&body).expect("empty batch round-trips");
        assert!(decoded.is_empty());
    }

    #[test]
    fn encode_expert_request_handles_empty_batch() {
        let body = encode_expert_request(&[]);
        assert_eq!(body.len(), 8); // 4 + 4 header only
        let decoded = decode_expert_request(&body).expect("empty batch decode");
        assert!(decoded.is_empty());
    }
}
