//! Binary + JSON codec for the walk-ffn wire protocol.
//!
//! - [`decode_binary_request`] parses the packed `application/x-larql-ffn`
//!   body (single-layer or batch via `BATCH_MARKER`) into a
//!   [`WalkFfnRequest`].
//! - [`encode_binary_output`] / [`encode_binary_output_f16`] /
//!   [`encode_binary_output_i8`] encode the [`FfnOutput`] back out;
//!   the three variants are negotiated by `Accept` header per ADR-0009.
//! - [`encode_json_full_output`] is the JSON-shape equivalent used
//!   when the request came in as JSON.

use crate::error::ServerError;

use super::types::{FfnOutput, WalkFfnRequest, BATCH_MARKER};

/// Decode a binary-format request body into a [`WalkFfnRequest`].
pub(crate) fn decode_binary_request(body: &[u8]) -> Result<WalkFfnRequest, ServerError> {
    if body.len() < 16 {
        return Err(ServerError::BadRequest(
            "binary: body too short (need ≥ 16 bytes)".into(),
        ));
    }

    let first = u32::from_le_bytes(body[0..4].try_into().unwrap());

    let (layer, layers, header_end) = if first == BATCH_MARKER {
        if body.len() < 8 {
            return Err(ServerError::BadRequest(
                "binary batch: truncated num_layers".into(),
            ));
        }
        let n = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
        let layers_end = 8 + n * 4;
        if body.len() < layers_end {
            return Err(ServerError::BadRequest(format!(
                "binary batch: body too short for {n} layer indices"
            )));
        }
        let layers: Vec<usize> = (0..n)
            .map(|i| u32::from_le_bytes(body[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize)
            .collect();
        (None, Some(layers), layers_end)
    } else {
        (Some(first as usize), None, 4)
    };

    if body.len() < header_end + 12 {
        return Err(ServerError::BadRequest(
            "binary: truncated fixed header (seq_len/flags/top_k)".into(),
        ));
    }
    let seq_len = u32::from_le_bytes(body[header_end..header_end + 4].try_into().unwrap()) as usize;
    let flags = u32::from_le_bytes(body[header_end + 4..header_end + 8].try_into().unwrap());
    let top_k =
        u32::from_le_bytes(body[header_end + 8..header_end + 12].try_into().unwrap()) as usize;
    let full_output = (flags & 1) != 0;

    let residual_bytes = &body[header_end + 12..];
    if !residual_bytes.len().is_multiple_of(4) {
        return Err(ServerError::BadRequest(
            "binary: residual byte length is not a multiple of 4".into(),
        ));
    }
    let residual: Vec<f32> = residual_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Ok(WalkFfnRequest {
        layer,
        layers,
        residual,
        seq_len,
        top_k,
        full_output,
        moe_layer: false,
    })
}

/// Encode an [`FfnOutput`] as the binary response format.
pub(crate) fn encode_binary_output(out: &FfnOutput) -> Vec<u8> {
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let mut buf = Vec::with_capacity(12 + entry.output.len() * 4);
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for &v in &entry.output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    } else {
        let num = out.entries.len();
        let mut buf = Vec::with_capacity(12 + num * 12);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for &v in &entry.output {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] using f16 values for the residual/output arrays.
///
/// Wire layout: identical to the f32 format except every float in the output
/// arrays is a `u16` LE (IEEE 754 half-precision). Header fields (layer,
/// seq_len, latency_ms) remain f32/u32 LE. See ADR-0009.
pub(crate) fn encode_binary_output_f16(out: &FfnOutput) -> Vec<u8> {
    use half::f16;
    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let mut buf = Vec::with_capacity(12 + entry.output.len() * 2);
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for &v in &entry.output {
            buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        buf
    } else {
        let num = out.entries.len();
        let total_floats: usize = out.entries.iter().map(|e| e.output.len()).sum();
        let mut buf = Vec::with_capacity(12 + num * 12 + total_floats * 2);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for &v in &entry.output {
                buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] using i8 symmetric quantisation (ADR-0009).
///
/// Per position: `[scale f32 LE][zero_point f32 LE][data i8[hidden_size]]`.
/// `scale = max(|x|) / 127.0`, `zero_point = 0.0` (symmetric).
/// Header fields (layer, seq_len, latency_ms) remain f32/u32 LE.
pub(crate) fn encode_binary_output_i8(out: &FfnOutput) -> Vec<u8> {
    fn quantise_position(vals: &[f32], buf: &mut Vec<u8>) {
        let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        buf.extend_from_slice(&scale.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes()); // zero_point = 0
        for &v in vals {
            let q = (v / scale).clamp(-127.0, 127.0).round() as i8;
            buf.push(q as u8);
        }
    }

    if out.entries.len() == 1 {
        let entry = &out.entries[0];
        let seq = out.seq_len.max(1);
        let hidden = entry.output.len() / seq;
        let mut buf = Vec::with_capacity(12 + seq * (8 + hidden));
        buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
        buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for pos in 0..seq {
            quantise_position(&entry.output[pos * hidden..(pos + 1) * hidden], &mut buf);
        }
        buf
    } else {
        let num = out.entries.len();
        let mut buf = Vec::with_capacity(12 + num * 16);
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&(num as u32).to_le_bytes());
        buf.extend_from_slice(&(out.latency_ms as f32).to_le_bytes());
        for entry in &out.entries {
            let seq = out.seq_len.max(1);
            let hidden = entry.output.len() / seq;
            buf.extend_from_slice(&(entry.layer as u32).to_le_bytes());
            buf.extend_from_slice(&(out.seq_len as u32).to_le_bytes());
            buf.extend_from_slice(&(entry.output.len() as u32).to_le_bytes());
            for pos in 0..seq {
                quantise_position(&entry.output[pos * hidden..(pos + 1) * hidden], &mut buf);
            }
        }
        buf
    }
}

/// Encode an [`FfnOutput`] as the existing JSON response format (unchanged wire
/// contract for JSON clients).
pub(crate) fn encode_json_full_output(out: &FfnOutput) -> serde_json::Value {
    let latency_rounded = (out.latency_ms * 10.0).round() / 10.0;
    if out.entries.len() == 1 {
        let e = &out.entries[0];
        serde_json::json!({
            "layer": e.layer,
            "output": e.output,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    } else {
        let results: Vec<serde_json::Value> = out
            .entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "layer": e.layer,
                    "output": e.output,
                    "seq_len": out.seq_len,
                })
            })
            .collect();
        serde_json::json!({
            "results": results,
            "seq_len": out.seq_len,
            "latency_ms": latency_rounded,
        })
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests — covers decode + every encoder variant
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::types::{FfnEntry, FfnOutput};
    use super::*;

    // ── decode_binary_request ─────────────────────────────────────────────────

    fn make_single_binary(
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

    fn make_batch_binary(
        layers: &[u32],
        seq_len: u32,
        full_output: bool,
        top_k: u32,
        residual: &[f32],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
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

    #[test]
    fn decode_single_layer_request() {
        let body = make_single_binary(5, 1, true, 8, &[1.0, 2.0, 3.0, 4.0]);
        let req = decode_binary_request(&body).unwrap();
        assert_eq!(req.layer, Some(5));
        assert!(req.layers.is_none());
        assert_eq!(req.seq_len, 1);
        assert_eq!(req.top_k, 8);
        assert!(req.full_output);
        assert_eq!(req.residual, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn decode_batch_request() {
        let body = make_batch_binary(&[0, 1, 2], 1, true, 16, &[1.0; 4]);
        let req = decode_binary_request(&body).unwrap();
        assert!(req.layer.is_none());
        assert_eq!(req.layers, Some(vec![0, 1, 2]));
        assert_eq!(req.top_k, 16);
    }

    #[test]
    fn decode_features_only_binary() {
        let body = make_single_binary(0, 1, false, 8, &[1.0, 2.0, 3.0, 4.0]);
        let req = decode_binary_request(&body).unwrap();
        assert!(!req.full_output);
    }

    #[test]
    fn decode_binary_truncated_body() {
        let body = vec![0u8; 8];
        assert!(decode_binary_request(&body).is_err());
    }

    #[test]
    fn decode_binary_empty_body() {
        assert!(decode_binary_request(&[]).is_err());
    }

    #[test]
    fn decode_binary_batch_truncated_layers() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&BATCH_MARKER.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // claim 4 layers
        buf.extend_from_slice(&0u32.to_le_bytes()); // only 1
        buf.extend_from_slice(&[0u8; 4]);
        assert!(decode_binary_request(&buf).is_err());
    }

    #[test]
    fn decode_binary_odd_residual_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.push(0u8); // 1-byte residual — not a multiple of 4
        assert!(decode_binary_request(&buf).is_err());
    }

    // ── encode_binary_output (f32) ────────────────────────────────────────────

    #[test]
    fn encode_single_entry_output() {
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 5,
                output: vec![1.0f32, -2.0, 3.5],
            }],
            seq_len: 1,
            latency_ms: 7.3,
        };
        let bytes = encode_binary_output(&out);
        assert_eq!(bytes.len(), 4 + 4 + 4 + 3 * 4);
        let layer = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let seq_len = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let latency = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(layer, 5);
        assert_eq!(seq_len, 1);
        assert!((latency - 7.3f32).abs() < 0.01);
        let v0 = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert!((v0 - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn encode_batch_output() {
        let out = FfnOutput {
            entries: vec![
                FfnEntry {
                    layer: 5,
                    output: vec![1.0f32, 2.0],
                },
                FfnEntry {
                    layer: 20,
                    output: vec![3.0f32, 4.0],
                },
            ],
            seq_len: 1,
            latency_ms: 15.0,
        };
        let bytes = encode_binary_output(&out);
        let marker = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num_results = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(num_results, 2);
        let latency = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert!((latency - 15.0f32).abs() < 0.01);
        let layer0 = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(layer0, 5);
        let num_floats0 = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!(num_floats0, 2);
    }

    #[test]
    fn binary_roundtrip_float_preservation() {
        let original_output = vec![0.12345f32, -9.87654, 1e-7, f32::MAX / 2.0];
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 10,
                output: original_output.clone(),
            }],
            seq_len: 1,
            latency_ms: 1.0,
        };
        let bytes = encode_binary_output(&out);
        // Skip 12-byte header; decode float values.
        let decoded: Vec<f32> = bytes[12..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, original_output);
    }

    // ── encode_json_full_output ──────────────────────────────────────────────

    #[test]
    fn json_single_layer_format() {
        let out = FfnOutput {
            entries: vec![FfnEntry {
                layer: 7,
                output: vec![1.0f32, 2.0, 3.0],
            }],
            seq_len: 1,
            latency_ms: 4.2,
        };
        let v = encode_json_full_output(&out);
        assert!(v.get("layer").is_some());
        assert!(v.get("output").is_some());
        assert!(v.get("results").is_none());
        assert_eq!(v["layer"].as_u64(), Some(7));
    }

    #[test]
    fn json_batch_format() {
        let out = FfnOutput {
            entries: vec![
                FfnEntry {
                    layer: 0,
                    output: vec![1.0f32],
                },
                FfnEntry {
                    layer: 1,
                    output: vec![2.0f32],
                },
            ],
            seq_len: 2,
            latency_ms: 20.0,
        };
        let v = encode_json_full_output(&out);
        assert!(v.get("results").is_some());
        let results = v["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["layer"].as_u64(), Some(0));
    }

    // ── encode_binary_output_f16 / _i8 ────────────────────────────────────────

    fn make_single_output(layer: u32, vals: Vec<f32>) -> FfnOutput {
        FfnOutput {
            entries: vec![FfnEntry {
                layer: layer as usize,
                output: vals,
            }],
            seq_len: 1,
            latency_ms: 5.0,
        }
    }

    fn make_batch_output(entries: Vec<(u32, Vec<f32>)>) -> FfnOutput {
        FfnOutput {
            entries: entries
                .into_iter()
                .map(|(l, v)| FfnEntry {
                    layer: l as usize,
                    output: v,
                })
                .collect(),
            seq_len: 1,
            latency_ms: 8.0,
        }
    }

    #[test]
    fn encode_f16_single_entry_halves_payload_size() {
        let out = make_single_output(3, vec![1.0, -2.0, 3.5, 4.0]);
        let bytes = encode_binary_output_f16(&out);
        assert_eq!(bytes.len(), 4 + 4 + 4 + 4 * 2);
        let layer = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(layer, 3);
    }

    #[test]
    fn encode_f16_batch_uses_marker_header() {
        let out = make_batch_output(vec![(0, vec![1.0, 2.0]), (1, vec![3.0, 4.0])]);
        let bytes = encode_binary_output_f16(&out);
        let marker = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(num, 2);
    }

    #[test]
    fn encode_i8_single_entry_symmetric_quantisation() {
        let out = make_single_output(7, vec![1.0, -1.0, 0.5, -0.5]);
        let bytes = encode_binary_output_i8(&out);
        assert_eq!(bytes.len(), 4 + 4 + 4 + 4 + 4 + 4);
        let zero = f32::from_le_bytes(bytes[12 + 4..12 + 8].try_into().unwrap());
        assert_eq!(zero, 0.0, "symmetric quantisation: zero_point=0");
    }

    #[test]
    fn encode_i8_batch_marker_then_per_entry_quantisation() {
        let out = make_batch_output(vec![(0, vec![2.0, -2.0]), (1, vec![1.0, -1.0])]);
        let bytes = encode_binary_output_i8(&out);
        let marker = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(marker, BATCH_MARKER);
        let num = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(num, 2);
    }

    #[test]
    fn encode_i8_zero_input_uses_unit_scale() {
        let out = make_single_output(0, vec![0.0; 4]);
        let bytes = encode_binary_output_i8(&out);
        let scale = f32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(scale, 1.0);
        for &b in &bytes[20..24] {
            assert_eq!(b as i8, 0);
        }
    }
}
