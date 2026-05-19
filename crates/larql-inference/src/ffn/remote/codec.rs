//! Binary wire codec for the LARQL FFN remote protocol.
//!
//! See the `super` module doc for the full binary frame layout.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(super) const BINARY_CT: &str = "application/x-larql-ffn";
pub(super) const BATCH_MARKER: u32 = 0xFFFF_FFFF;

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("{what}: byte length overflow"))
}

fn checked_end(offset: usize, len: usize, total: usize, what: &str) -> Result<usize, String> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| format!("{what}: byte range overflow"))?;
    if end > total {
        return Err(format!(
            "{what}: truncated: need {len}, have {}",
            total.saturating_sub(offset)
        ));
    }
    Ok(end)
}

fn read_u32(body: &[u8], offset: usize, what: &str) -> Result<u32, String> {
    let end = checked_end(offset, 4, body.len(), what)?;
    Ok(u32::from_le_bytes(body[offset..end].try_into().unwrap()))
}

fn validate_batch_result_count(body: &[u8], num_results: usize, what: &str) -> Result<(), String> {
    let max_results_with_headers = body.len().saturating_sub(12) / 12;
    if num_results > max_results_with_headers {
        return Err(format!(
            "{what}: declared {num_results} results but only {max_results_with_headers} headers fit"
        ));
    }
    Ok(())
}

// ── Wire types (JSON fallback) ────────────────────────────────────────────────

#[derive(Serialize)]
#[allow(dead_code)]
pub(super) struct WalkFfnHttpRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<Vec<usize>>,
    pub residual: Vec<f32>,
    pub seq_len: usize,
    pub full_output: bool,
}

#[derive(Deserialize)]
pub(super) struct WalkFfnSingleResponse {
    #[allow(dead_code)]
    pub layer: usize,
    pub output: Vec<f32>,
    #[allow(dead_code)]
    pub seq_len: usize,
}

// ── Latency profiling result ──────────────────────────────────────────────────

/// Breakdown returned by [`super::http::RemoteWalkBackend::probe_latency`].
#[derive(Debug, Clone)]
pub struct RemoteLatencyStats {
    /// Wall-clock round-trip (client-measured), averaged over `samples` calls.
    pub total_ms: f64,
    /// FFN compute time reported by the server in the binary response header.
    pub server_ms: f64,
    /// `total_ms - server_ms`: HTTP framing + TCP + serialization overhead.
    pub overhead_ms: f64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub samples: usize,
}

impl std::fmt::Display for RemoteLatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "layers={} hidden={} samples={}\n  total    {:7.2} ms\n  server   {:7.2} ms  (FFN compute)\n  overhead {:7.2} ms  (HTTP + TCP + framing)",
            self.num_layers, self.hidden_size, self.samples,
            self.total_ms, self.server_ms, self.overhead_ms,
        )
    }
}

// ── Binary codec ──────────────────────────────────────────────────────────────

/// Encode a request as binary.
/// `layer` and `layers` are mutually exclusive; pass `None` for the unused one.
pub fn encode_binary_request(
    layer: Option<usize>,
    layers: Option<&[usize]>,
    residual: &[f32],
    seq_len: usize,
    full_output: bool,
    top_k: usize,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + residual.len() * 4);

    if let Some(ls) = layers {
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(ls.len() as u32).to_le_bytes());
        for &l in ls {
            buf.extend_from_slice(&(l as u32).to_le_bytes());
        }
    } else {
        let l = layer.unwrap_or(0) as u32;
        buf.extend_from_slice(&l.to_le_bytes());
    }

    buf.extend_from_slice(&(seq_len as u32).to_le_bytes());
    buf.extend_from_slice(&(full_output as u32).to_le_bytes());
    buf.extend_from_slice(&(top_k as u32).to_le_bytes());
    for &v in residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Decode a binary single-layer full_output response.
/// Returns `(layer, output_floats)`.
pub fn decode_binary_single(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    if body.len() < 12 {
        return Err(format!("binary response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer response but got batch marker".into());
    }
    let layer = marker as usize;
    // bytes 4-7: seq_len (ignored here — caller validates against expected shape)
    // bytes 8-11: latency f32
    if !(body.len() - 12).is_multiple_of(4) {
        return Err("binary response: output byte length is not a multiple of f32".into());
    }
    let floats: Vec<f32> = body[12..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch full_output response.
/// Returns a map from layer → output floats.
pub fn decode_binary_batch(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < 12 {
        return Err(format!(
            "binary batch response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());

    // Single-layer response — accept it as a batch of 1.
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single(body)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }

    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "binary batch")?;
    // bytes 8-11: latency f32 (skip)
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);

    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "binary batch result header")?;
        let layer = read_u32(body, offset, "binary batch layer")? as usize;
        // offset+4: seq_len (skip)
        let num_floats = read_u32(body, offset + 8, "binary batch output length")? as usize;
        offset += 12;
        let bytes_needed = checked_mul(num_floats, 4, "binary batch output")?;
        let end = checked_end(offset, bytes_needed, body.len(), "binary batch output")?;
        let floats: Vec<f32> = body[offset..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset = end;
        out.insert(layer, floats);
    }
    Ok(out)
}

/// f16 content-type constant (ADR-0009).
pub(crate) const F16_CT: &str = "application/x-larql-ffn-f16";

/// Decode a binary single-layer f16 response into f32 output.
pub fn decode_binary_single_f16(body: &[u8]) -> Result<(usize, Vec<f32>), String> {
    use half::f16;
    if body.len() < 12 {
        return Err(format!(
            "f16 binary response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer f16 response but got batch marker".into());
    }
    let layer = marker as usize;
    if !(body.len() - 12).is_multiple_of(2) {
        return Err("f16 binary response: output byte length is not a multiple of f16".into());
    }
    let floats: Vec<f32> = body[12..]
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
        .collect();
    Ok((layer, floats))
}

/// Decode a binary batch f16 response into f32 outputs.
pub fn decode_binary_batch_f16(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    use half::f16;
    if body.len() < 12 {
        return Err(format!(
            "f16 batch response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single_f16(body)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }
    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "f16 batch")?;
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);
    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "f16 batch result header")?;
        let layer = read_u32(body, offset, "f16 batch layer")? as usize;
        let num_floats = read_u32(body, offset + 8, "f16 batch output length")? as usize;
        offset += 12;
        let bytes_needed = checked_mul(num_floats, 2, "f16 batch output")?;
        let end = checked_end(offset, bytes_needed, body.len(), "f16 batch output")?;
        let floats: Vec<f32> = body[offset..end]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        offset = end;
        out.insert(layer, floats);
    }
    Ok(out)
}

/// i8 content-type constant (ADR-0009).
pub(crate) const I8_CT: &str = "application/x-larql-ffn-i8";

/// Decode one position from an i8 per-position block.
/// Format: `[scale f32 LE][zero_point f32 LE (ignored)][data i8[hidden_size]]`
fn decode_i8_position(
    body: &[u8],
    offset: usize,
    hidden: usize,
) -> Result<(Vec<f32>, usize), String> {
    let needed = 8usize
        .checked_add(hidden)
        .ok_or_else(|| "i8: position byte length overflow".to_string())?;
    let end = checked_end(offset, needed, body.len(), "i8 position")?;
    let scale = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
    // zero_point at offset+4 is always 0.0 (symmetric), skip it
    let floats: Vec<f32> = body[offset + 8..offset + 8 + hidden]
        .iter()
        .map(|&b| (b as i8) as f32 * scale)
        .collect();
    Ok((floats, end))
}

/// Decode a binary single-layer i8 response into f32 output.
pub(crate) fn decode_binary_single_i8(
    body: &[u8],
    hidden_size: usize,
) -> Result<(usize, Vec<f32>), String> {
    if body.len() < 12 {
        return Err(format!(
            "i8 binary response too short: {} bytes",
            body.len()
        ));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker == BATCH_MARKER {
        return Err("expected single-layer i8 response but got batch marker".into());
    }
    let layer = marker as usize;
    let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    let seq_len = seq_len.max(1);
    let mut offset = 12usize;
    let total_floats = checked_mul(seq_len, hidden_size, "i8 single output")?;
    let mut all_floats = Vec::with_capacity(total_floats);
    for _ in 0..seq_len {
        let (pos_floats, next_offset) = decode_i8_position(body, offset, hidden_size)?;
        all_floats.extend(pos_floats);
        offset = next_offset;
    }
    Ok((layer, all_floats))
}

/// Decode a binary batch i8 response into f32 outputs.
pub(crate) fn decode_binary_batch_i8(
    body: &[u8],
    hidden_size: usize,
) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < 12 {
        return Err(format!("i8 batch response too short: {} bytes", body.len()));
    }
    let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
    if marker != BATCH_MARKER {
        let (layer, floats) = decode_binary_single_i8(body, hidden_size)?;
        let mut m = HashMap::new();
        m.insert(layer, floats);
        return Ok(m);
    }
    let num_results = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
    validate_batch_result_count(body, num_results, "i8 batch")?;
    let mut offset = 12usize;
    let mut out = HashMap::with_capacity(num_results);
    for _ in 0..num_results {
        checked_end(offset, 12, body.len(), "i8 batch result header")?;
        let layer = read_u32(body, offset, "i8 batch layer")? as usize;
        let seq_len = read_u32(body, offset + 4, "i8 batch sequence length")? as usize;
        let seq_len = seq_len.max(1);
        let num_floats = read_u32(body, offset + 8, "i8 batch output length")? as usize;
        let expected_floats = checked_mul(seq_len, hidden_size, "i8 batch output")?;
        if num_floats != expected_floats {
            return Err(format!(
                "i8 batch: layer {layer} declared {num_floats} floats, expected {expected_floats}"
            ));
        }
        offset += 12;
        let mut all_floats = Vec::with_capacity(num_floats);
        for _ in 0..seq_len {
            let (pos_floats, next_offset) = decode_i8_position(body, offset, hidden_size)?;
            all_floats.extend(pos_floats);
            offset = next_offset;
        }
        out.insert(layer, all_floats);
    }
    Ok(out)
}

/// Extract the `latency_ms` f32 embedded at bytes 8-11 of a binary response.
/// Returns 0.0 if the body is too short or the value is non-finite.
pub(super) fn extract_response_latency_ms(body: &[u8]) -> f64 {
    if body.len() < 12 {
        return 0.0;
    }
    // Both single-layer and batch responses have latency_ms at offset 8.
    let v = f32::from_le_bytes(body[8..12].try_into().unwrap());
    if v.is_finite() {
        v as f64
    } else {
        0.0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    // ── JSON serialisation ────────────────────────────────────────────────────

    #[test]
    fn request_serializes_with_seq_len_and_full_output() {
        let req = WalkFfnHttpRequest {
            layer: Some(3),
            layers: None,
            residual: vec![0.1, -0.2, 0.3, 0.4],
            seq_len: 2,
            full_output: true,
        };
        let v: serde_json::Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["layer"], 3);
        assert_eq!(v["seq_len"], 2);
        assert_eq!(v["full_output"], true);
        assert!(
            v.get("layers").is_none() || v["layers"].is_null(),
            "layers should not appear when None, got: {v}"
        );
        assert_eq!(v["residual"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn response_deserializes_hidden_vector() {
        let json = serde_json::json!({
            "layer": 5,
            "output": [0.1, 0.2, 0.3, 0.4, 0.5],
            "seq_len": 1,
            "latency_ms": 2.5,
        });
        let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.layer, 5);
        assert_eq!(parsed.output.len(), 5);
        assert_eq!(parsed.seq_len, 1);
    }

    #[test]
    fn response_deserializes_multi_token_output() {
        let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let json = serde_json::json!({
            "layer": 0,
            "output": flat,
            "seq_len": 3,
        });
        let parsed: WalkFfnSingleResponse = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.output.len(), 12);
        assert_eq!(parsed.seq_len, 3);
    }

    // ── encode_binary_request ─────────────────────────────────────────────────

    #[test]
    fn encode_single_layer_header() {
        let residual = vec![1.0f32, 2.0, 3.0, 4.0];
        let body = encode_binary_request(Some(7), None, &residual, 1, true, 256);
        // First u32 = layer index
        let layer = u32::from_le_bytes(body[0..4].try_into().unwrap());
        assert_eq!(layer, 7);
        let seq_len = u32::from_le_bytes(body[4..8].try_into().unwrap());
        assert_eq!(seq_len, 1);
        let flags = u32::from_le_bytes(body[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1); // full_output
        let top_k = u32::from_le_bytes(body[12..16].try_into().unwrap());
        assert_eq!(top_k, 256);
        assert_eq!(body.len(), 16 + 4 * 4);
    }

    #[test]
    fn encode_batch_header() {
        let residual = vec![0.5f32; 4];
        let body = encode_binary_request(None, Some(&[5, 20, 30]), &residual, 1, true, 512);
        let marker = u32::from_le_bytes(body[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num_layers = u32::from_le_bytes(body[4..8].try_into().unwrap());
        assert_eq!(num_layers, 3);
        let l0 = u32::from_le_bytes(body[8..12].try_into().unwrap());
        let l1 = u32::from_le_bytes(body[12..16].try_into().unwrap());
        let l2 = u32::from_le_bytes(body[16..20].try_into().unwrap());
        assert_eq!((l0, l1, l2), (5, 20, 30));
    }

    #[test]
    fn encode_residual_values_preserved() {
        let residual = vec![-1.5f32, 0.0, 3.25];
        let body = encode_binary_request(Some(0), None, &residual, 1, true, 8092);
        let offset = 16; // 4 header u32s × 4 bytes
        let v0 = f32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
        let v1 = f32::from_le_bytes(body[offset + 4..offset + 8].try_into().unwrap());
        let v2 = f32::from_le_bytes(body[offset + 8..offset + 12].try_into().unwrap());
        assert_eq!(v0.to_bits(), (-1.5f32).to_bits());
        assert_eq!(v1.to_bits(), 0.0f32.to_bits());
        assert!((v2 - 3.25f32).abs() < 1e-5);
    }

    // ── decode_binary_single ──────────────────────────────────────────────────

    fn make_single_response(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &v in output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn make_batch_response(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
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
    fn decode_single_response_correct() {
        let output = vec![1.0f32, -2.0, 3.5];
        let body = make_single_response(5, 1, 7.3, &output);
        let (layer, floats) = decode_binary_single(&body).unwrap();
        assert_eq!(layer, 5);
        assert_eq!(floats.len(), 3);
        assert!((floats[0] - 1.0).abs() < 1e-6);
        assert!((floats[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn decode_single_response_rejects_batch_marker() {
        let body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
        let result = decode_binary_single(&body);
        assert!(result.is_err());
    }

    #[test]
    fn decode_single_response_too_short() {
        let result = decode_binary_single(&[0u8; 8]);
        assert!(result.is_err());
    }

    // ── decode_binary_batch ───────────────────────────────────────────────────

    #[test]
    fn decode_batch_response_correct() {
        let body = make_batch_response(15.0, &[(5, &[1.0, 2.0]), (20, &[3.0, 4.0])]);
        let map = decode_binary_batch(&body).unwrap();
        assert_eq!(map.len(), 2);
        let v5 = map.get(&5).unwrap();
        assert_eq!(v5.len(), 2);
        assert!((v5[0] - 1.0).abs() < 1e-6);
        let v20 = map.get(&20).unwrap();
        assert!((v20[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn decode_batch_accepts_single_response() {
        // A server returning single-layer response to a same-shard batch.
        let output = vec![7.0f32, 8.0];
        let body = make_single_response(10, 1, 5.0, &output);
        let map = decode_binary_batch(&body).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&10));
    }

    #[test]
    fn decode_batch_truncated_returns_error() {
        let mut body = make_batch_response(1.0, &[(5, &[1.0, 2.0])]);
        body.truncate(body.len() - 4); // cut off last float
        let result = decode_binary_batch(&body);
        assert!(result.is_err());
    }

    #[test]
    fn decode_single_rejects_partial_float_payload() {
        let mut body = make_single_response(5, 1, 7.3, &[1.0]);
        body.push(0);
        let result = decode_binary_single(&body);
        assert!(result.is_err());
    }

    #[test]
    fn decode_batch_rejects_impossible_result_count_before_allocating() {
        let mut body = Vec::new();
        body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        let result = decode_binary_batch(&body);
        assert!(result.is_err());
    }

    #[test]
    fn decode_batch_rejects_impossible_output_length_before_allocating() {
        let mut body = Vec::new();
        body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        let result = decode_binary_batch(&body);
        assert!(result.is_err());
    }

    #[test]
    fn decode_batch_i8_rejects_inconsistent_output_shape() {
        let mut body = Vec::new();
        body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        body.extend_from_slice(&5u32.to_le_bytes());
        body.extend_from_slice(&2u32.to_le_bytes());
        body.extend_from_slice(&3u32.to_le_bytes());
        let result = decode_binary_batch_i8(&body, 2);
        assert!(result.is_err());
    }

    // ── decode_binary_single_f16 + decode_binary_batch_f16 ─────────────

    fn make_single_response_f16(layer: u32, seq_len: u32, latency: f32, output: &[f32]) -> Vec<u8> {
        use half::f16;
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &v in output {
            buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        buf
    }

    fn make_batch_response_f16(latency: f32, entries: &[(u32, &[f32])]) -> Vec<u8> {
        use half::f16;
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &(layer, floats) in entries {
            buf.extend_from_slice(&layer.to_le_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // seq_len
            buf.extend_from_slice(&(floats.len() as u32).to_le_bytes());
            for &v in floats {
                buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn decode_single_f16_round_trip_within_quant_noise() {
        let body = make_single_response_f16(7, 1, 1.0, &[0.5, -0.25, 1.5, -2.5]);
        let (layer, floats) = decode_binary_single_f16(&body).unwrap();
        assert_eq!(layer, 7);
        assert_eq!(floats.len(), 4);
        // f16 round-trip is exact for these clean fractions.
        assert!((floats[0] - 0.5).abs() < 1e-6);
        assert!((floats[3] - (-2.5)).abs() < 1e-6);
    }

    #[test]
    fn decode_single_f16_too_short_errors() {
        assert!(decode_binary_single_f16(&[0u8; 8]).is_err());
    }

    #[test]
    fn decode_single_f16_rejects_batch_marker() {
        let body = make_batch_response_f16(1.0, &[(0, &[1.0])]);
        assert!(decode_binary_single_f16(&body).is_err());
    }

    #[test]
    fn decode_single_f16_rejects_odd_payload_length() {
        let mut body = make_single_response_f16(0, 1, 0.0, &[1.0]);
        body.push(0u8); // odd byte tail
        assert!(decode_binary_single_f16(&body).is_err());
    }

    #[test]
    fn decode_batch_f16_round_trip_two_entries() {
        let body = make_batch_response_f16(2.0, &[(3, &[1.0, 2.0]), (11, &[-1.0, 0.5])]);
        let map = decode_binary_batch_f16(&body).unwrap();
        assert_eq!(map.len(), 2);
        let v3 = map.get(&3).unwrap();
        assert!((v3[0] - 1.0).abs() < 1e-6 && (v3[1] - 2.0).abs() < 1e-6);
        let v11 = map.get(&11).unwrap();
        assert!((v11[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn decode_batch_f16_falls_through_to_single_when_no_marker() {
        let body = make_single_response_f16(5, 1, 1.0, &[1.0, 2.0, 3.0]);
        let map = decode_binary_batch_f16(&body).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&5));
    }

    #[test]
    fn decode_batch_f16_too_short_errors() {
        assert!(decode_binary_batch_f16(&[0u8; 4]).is_err());
    }

    #[test]
    fn decode_batch_f16_rejects_impossible_result_count() {
        let mut body = Vec::new();
        body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        assert!(decode_binary_batch_f16(&body).is_err());
    }

    // ── decode_binary_single_i8 + decode_binary_batch_i8 ───────────────

    fn make_single_response_i8(
        layer: u32,
        seq_len: u32,
        latency: f32,
        positions: &[(f32, &[i8])],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&layer.to_le_bytes());
        buf.extend_from_slice(&seq_len.to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &(scale, data) in positions {
            buf.extend_from_slice(&scale.to_le_bytes());
            buf.extend_from_slice(&0.0f32.to_le_bytes()); // zero_point ignored
            for &b in data {
                buf.push(b as u8);
            }
        }
        buf
    }

    #[allow(clippy::type_complexity)]
    fn make_batch_response_i8(latency: f32, entries: &[(u32, u32, &[(f32, &[i8])])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        buf.extend_from_slice(&latency.to_le_bytes());
        for &(layer, seq_len, positions) in entries {
            // num_floats per the codec contract (`seq_len * hidden_size`)
            let hidden = positions.first().map(|(_, d)| d.len()).unwrap_or(0);
            let num_floats = seq_len as usize * hidden;
            buf.extend_from_slice(&layer.to_le_bytes());
            buf.extend_from_slice(&seq_len.to_le_bytes());
            buf.extend_from_slice(&(num_floats as u32).to_le_bytes());
            for &(scale, data) in positions {
                buf.extend_from_slice(&scale.to_le_bytes());
                buf.extend_from_slice(&0.0f32.to_le_bytes());
                for &b in data {
                    buf.push(b as u8);
                }
            }
        }
        buf
    }

    #[test]
    fn decode_single_i8_round_trip_one_position() {
        let hidden = 4;
        let body = make_single_response_i8(5, 1, 1.0, &[(0.5, &[2i8, -4, 8, -8])]);
        let (layer, floats) = decode_binary_single_i8(&body, hidden).unwrap();
        assert_eq!(layer, 5);
        assert_eq!(floats, vec![1.0f32, -2.0, 4.0, -4.0]);
    }

    #[test]
    fn decode_single_i8_round_trip_multi_position() {
        let hidden = 2;
        let body = make_single_response_i8(
            0,
            3,
            0.0,
            &[(1.0, &[10i8, 20]), (0.25, &[-4i8, 8]), (2.0, &[1i8, -1])],
        );
        let (_, floats) = decode_binary_single_i8(&body, hidden).unwrap();
        assert_eq!(floats, vec![10.0, 20.0, -1.0, 2.0, 2.0, -2.0]);
    }

    #[test]
    fn decode_single_i8_rejects_batch_marker() {
        let body = make_batch_response_i8(1.0, &[(0, 1, &[(1.0, &[1i8])])]);
        assert!(decode_binary_single_i8(&body, 1).is_err());
    }

    #[test]
    fn decode_single_i8_too_short_errors() {
        assert!(decode_binary_single_i8(&[0u8; 8], 4).is_err());
    }

    #[test]
    fn decode_single_i8_zero_seq_len_treated_as_one() {
        // Codec promotes seq_len 0 → 1 to keep the per-position loop alive.
        let body = make_single_response_i8(2, 0, 0.0, &[(1.0, &[5i8])]);
        let (layer, floats) = decode_binary_single_i8(&body, 1).unwrap();
        assert_eq!(layer, 2);
        assert_eq!(floats, vec![5.0]);
    }

    #[test]
    fn decode_single_i8_truncated_payload_errors() {
        // Position needs 8 + hidden bytes; cut the payload short.
        let mut body = make_single_response_i8(0, 1, 0.0, &[(1.0, &[1i8, 2, 3, 4])]);
        body.truncate(body.len() - 1);
        assert!(decode_binary_single_i8(&body, 4).is_err());
    }

    #[test]
    fn decode_batch_i8_round_trip_two_layers() {
        let hidden = 2;
        let body = make_batch_response_i8(
            3.0,
            &[
                (10, 1, &[(1.0, &[7i8, -7])]),
                (20, 2, &[(0.5, &[10i8, -10]), (0.5, &[20i8, -20])]),
            ],
        );
        let map = decode_binary_batch_i8(&body, hidden).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&10).unwrap(), &vec![7.0, -7.0]);
        assert_eq!(map.get(&20).unwrap(), &vec![5.0, -5.0, 10.0, -10.0]);
    }

    #[test]
    fn decode_batch_i8_falls_through_to_single_when_no_marker() {
        let body = make_single_response_i8(9, 1, 0.0, &[(1.0, &[3i8, -3])]);
        let map = decode_binary_batch_i8(&body, 2).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&9));
    }

    #[test]
    fn decode_batch_i8_too_short_errors() {
        assert!(decode_binary_batch_i8(&[0u8; 4], 1).is_err());
    }

    #[test]
    fn decode_batch_i8_rejects_impossible_result_count() {
        let mut body = Vec::new();
        body.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        body.extend_from_slice(&u32::MAX.to_le_bytes());
        body.extend_from_slice(&0.0f32.to_le_bytes());
        assert!(decode_binary_batch_i8(&body, 2).is_err());
    }

    // ── extract_response_latency_ms ────────────────────────────────────

    #[test]
    fn extract_latency_returns_zero_for_short_body() {
        assert_eq!(extract_response_latency_ms(&[]), 0.0);
        assert_eq!(extract_response_latency_ms(&[0u8; 11]), 0.0);
    }

    #[test]
    fn extract_latency_reads_offset_8_as_f32() {
        // Body: layer(4) + seq_len(4) + latency(4)=8.5
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&8.5f32.to_le_bytes());
        assert!((extract_response_latency_ms(&body) - 8.5).abs() < 1e-6);
    }

    #[test]
    fn extract_latency_returns_zero_for_non_finite() {
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_le_bytes());
        body.extend_from_slice(&1u32.to_le_bytes());
        body.extend_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(extract_response_latency_ms(&body), 0.0);
    }

    // ── Wire string consts ─────────────────────────────────────────────

    #[test]
    fn binary_content_type_consts_pin_wire_strings() {
        assert_eq!(BINARY_CT, "application/x-larql-ffn");
        assert_eq!(F16_CT, "application/x-larql-ffn-f16");
        assert_eq!(I8_CT, "application/x-larql-ffn-i8");
        assert_eq!(BATCH_MARKER, 0xFFFF_FFFFu32);
    }

    #[test]
    fn binary_request_response_roundtrip() {
        // Encode a single-layer request, then simulate what the server echoes.
        let residual = vec![0.1f32, 0.2, 0.3, 0.4];
        let req = encode_binary_request(Some(5), None, &residual, 1, true, 8092);
        // Simulate server extracting the layer.
        let layer = u32::from_le_bytes(req[0..4].try_into().unwrap());
        assert_eq!(layer, 5);

        // Simulate server response.
        let output = vec![0.9f32, 0.8, 0.7, 0.6];
        let resp = make_single_response(layer, 1, 8.5, &output);
        let (resp_layer, floats) = decode_binary_single(&resp).unwrap();
        assert_eq!(resp_layer as u32, layer);
        assert_eq!(floats, output);
    }
}
