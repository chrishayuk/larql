//! Binary wire codec for the Q8K-prenormed dense-FFN batch protocol.
//!
//! # Motivation
//!
//! The standard `/v1/walk-ffn` endpoint sends `h_post_attn` as f32 (21 KB per
//! layer at hidden=5376). By pre-applying the FFN input norm on the client and
//! quantising to Q8_K, upload shrinks ~3.7×: the server can skip `rms_norm`
//! and run the NEON `q4k_q8k_gate_up_into` kernel.
//!
//! # Wire layout
//!
//! ## Request — N entries packed sequentially
//! Each entry:
//! ```text
//! Offset  Size             Field
//! 0       4                layer_idx (u32 LE)
//! 4       4                n_blocks  (u32 LE, = hidden / 256)
//! 8       n_blocks × 256   qs        (i8[])
//! 8+B     n_blocks × 4     d         (f32[] LE, per-block scales)
//! 8+B+D   n_blocks × 8 × 2 sums     (i16[] LE, 8 sub-block sums per block)
//! ```
//! where `B = n_blocks * 256`, `D = n_blocks * 4`.
//!
//! The request begins with a 4-byte `num_entries` u32 header.
//!
//! ## Response — N entries packed sequentially
//! Response begins with 4-byte `num_entries` u32 header.  Each entry:
//! ```text
//! 0       4                layer_idx (u32 LE)
//! 4       4                hidden    (u32 LE, = output vec length)
//! 8       hidden × 4       output    (f32[] LE)
//! ```
//!
//! Content-Type: `application/x-larql-ffn-q8k-batch`

use std::collections::HashMap;

use larql_compute::cpu::ops::q4k_q8k_dot::Q8KActivation;

/// Content-type for the Q8K dense-FFN batch protocol.
pub const Q8K_BATCH_CT: &str = "application/x-larql-ffn-q8k-batch";

use crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS as ELEMS_PER_BLOCK;
const SUBBLOCKS_PER_BLOCK: usize = 8;

// ── Encode (client → server) ──────────────────────────────────────────────────

/// Encode a batch of `(layer_idx, Q8KActivation)` pairs for the Q8K wire protocol.
///
/// Output is the full request body — starts with `num_entries: u32 LE` followed
/// by one packed entry per layer.
pub fn encode_q8k_batch_request(layers: &[(usize, &Q8KActivation)]) -> Vec<u8> {
    let n = layers.len();
    // Rough capacity estimate: header + n * (4+4 + 256*n_blocks + 4*n_blocks + 16*n_blocks)
    let mut buf = Vec::with_capacity(4 + n * 8);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for &(layer_idx, q8k) in layers {
        let n_blocks = q8k.n_blocks();
        buf.extend_from_slice(&(layer_idx as u32).to_le_bytes());
        buf.extend_from_slice(&(n_blocks as u32).to_le_bytes());
        // qs: n_blocks * 256 i8 values (one byte each)
        buf.extend(q8k.qs.iter().map(|&v| v as u8));
        // d: n_blocks f32 values
        for &v in &q8k.d {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // sums: n_blocks * 8 i16 values
        for &v in &q8k.sums {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

// ── Decode (client ← server) ─────────────────────────────────────────────────

/// Decode a Q8K batch response body into a `HashMap<layer_idx → output_floats>`.
pub fn decode_q8k_batch_response(body: &[u8]) -> Result<HashMap<usize, Vec<f32>>, String> {
    if body.len() < 4 {
        return Err(format!(
            "q8k batch response too short: {} bytes",
            body.len()
        ));
    }
    let num_entries = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
    // Each response entry needs at least 8 bytes (layer_idx + hidden);
    // guard the with_capacity allocation against a garbage 4-byte
    // body that decodes `num_entries = u32::MAX`.
    let max_possible_entries = body.len().saturating_sub(4) / 8;
    if num_entries > max_possible_entries {
        return Err(format!(
            "q8k batch response: num_entries={num_entries} exceeds capacity for body of {} bytes",
            body.len()
        ));
    }
    let mut offset = 4usize;
    let mut out = HashMap::with_capacity(num_entries);
    for i in 0..num_entries {
        if body.len() < offset + 8 {
            return Err(format!("q8k batch response: truncated entry header {i}"));
        }
        let layer_idx = u32::from_le_bytes(body[offset..offset + 4].try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(body[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let floats_bytes = hidden * 4;
        if body.len() < offset + floats_bytes {
            return Err(format!(
                "q8k batch response: truncated output for layer {layer_idx}: \
                 need {floats_bytes} bytes, have {}",
                body.len().saturating_sub(offset)
            ));
        }
        let floats: Vec<f32> = body[offset..offset + floats_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset += floats_bytes;
        out.insert(layer_idx, floats);
    }
    Ok(out)
}

// ── Decode (server receives request) ─────────────────────────────────────────

/// A decoded Q8K request entry as received by the server.
pub struct Q8KRequestEntry {
    pub layer_idx: usize,
    pub q8k: Q8KActivation,
}

/// Decode a Q8K batch request body into a `Vec<Q8KRequestEntry>`.
///
/// The server calls this to reconstruct the per-layer Q8K activations from the
/// binary body sent by the client.
pub fn decode_q8k_batch_request(body: &[u8]) -> Result<Vec<Q8KRequestEntry>, String> {
    if body.len() < 4 {
        return Err(format!("q8k batch request too short: {} bytes", body.len()));
    }
    let num_entries = u32::from_le_bytes(body[0..4].try_into().unwrap()) as usize;
    // Each entry occupies at least 8 bytes (layer_idx u32 + n_blocks u32),
    // so a body with `num_entries > (body.len() - 4) / 8` is malformed.
    // Without this guard a 4-byte garbage body of `[0xFF; 4]` decodes
    // `num_entries = u32::MAX` and the `Vec::with_capacity` below tries
    // to reserve ~320 GiB, aborting the process — see PR 104 CI
    // `walk_ffn_q8k_garbage_body_with_q4k_returns_400`.
    let max_possible_entries = body.len().saturating_sub(4) / 8;
    if num_entries > max_possible_entries {
        return Err(format!(
            "q8k batch request: num_entries={num_entries} exceeds capacity for body of {} bytes",
            body.len()
        ));
    }
    let mut offset = 4usize;
    let mut entries = Vec::with_capacity(num_entries);
    for i in 0..num_entries {
        if body.len() < offset + 8 {
            return Err(format!("q8k batch request: truncated entry header {i}"));
        }
        let layer_idx = u32::from_le_bytes(body[offset..offset + 4].try_into().unwrap()) as usize;
        let n_blocks =
            u32::from_le_bytes(body[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        // qs: n_blocks * 256 bytes
        let qs_bytes = n_blocks * ELEMS_PER_BLOCK;
        if body.len() < offset + qs_bytes {
            return Err(format!(
                "q8k batch request: truncated qs for entry {i} (layer {layer_idx})"
            ));
        }
        let qs: Vec<i8> = body[offset..offset + qs_bytes]
            .iter()
            .map(|&b| b as i8)
            .collect();
        offset += qs_bytes;

        // d: n_blocks f32
        let d_bytes = n_blocks * 4;
        if body.len() < offset + d_bytes {
            return Err(format!(
                "q8k batch request: truncated d for entry {i} (layer {layer_idx})"
            ));
        }
        let d: Vec<f32> = body[offset..offset + d_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset += d_bytes;

        // sums: n_blocks * 8 i16
        let sums_bytes = n_blocks * SUBBLOCKS_PER_BLOCK * 2;
        if body.len() < offset + sums_bytes {
            return Err(format!(
                "q8k batch request: truncated sums for entry {i} (layer {layer_idx})"
            ));
        }
        let sums: Vec<i16> = body[offset..offset + sums_bytes]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        offset += sums_bytes;

        entries.push(Q8KRequestEntry {
            layer_idx,
            q8k: Q8KActivation { qs, d, sums },
        });
    }
    Ok(entries)
}

/// Encode a Q8K batch response from a slice of `(layer_idx, output_floats)` pairs.
///
/// The server calls this to build the response body.
pub fn encode_q8k_batch_response(entries: &[(usize, &[f32])]) -> Vec<u8> {
    let n = entries.len();
    let mut buf = Vec::with_capacity(4 + n * 8);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for &(layer_idx, output) in entries {
        buf.extend_from_slice(&(layer_idx as u32).to_le_bytes());
        buf.extend_from_slice(&(output.len() as u32).to_le_bytes());
        for &v in output {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;

    #[test]
    fn request_roundtrip_single_block() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let q8k = quantize_x_to_q8k(&x);
        let layers = vec![(7usize, &q8k)];
        let body = encode_q8k_batch_request(&layers);

        let decoded = decode_q8k_batch_request(&body).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].layer_idx, 7);
        assert_eq!(decoded[0].q8k.qs, q8k.qs);
        assert_eq!(decoded[0].q8k.d, q8k.d);
        assert_eq!(decoded[0].q8k.sums, q8k.sums);
    }

    #[test]
    fn request_roundtrip_multi_block_multi_layer() {
        // Two layers, each 2 blocks (hidden=512).
        let x: Vec<f32> = (0..512).map(|i| (i as f32 * 0.007).cos() * 2.0).collect();
        let q0 = quantize_x_to_q8k(&x);
        let q1 = quantize_x_to_q8k(&x.iter().map(|v| v * -0.5).collect::<Vec<_>>());
        let layers = vec![(0usize, &q0), (1usize, &q1)];
        let body = encode_q8k_batch_request(&layers);

        let decoded = decode_q8k_batch_request(&body).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].layer_idx, 0);
        assert_eq!(decoded[1].layer_idx, 1);
        assert_eq!(decoded[0].q8k.d, q0.d);
        assert_eq!(decoded[1].q8k.sums, q1.sums);
    }

    #[test]
    fn response_roundtrip() {
        let out0 = vec![1.0f32, 2.0, -3.5];
        let out1 = vec![-0.5f32, 0.0, 7.0];
        let entries: Vec<(usize, &[f32])> = vec![(5usize, &out0), (10usize, &out1)];
        let body = encode_q8k_batch_response(&entries);
        let map = decode_q8k_batch_response(&body).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&5], out0);
        assert_eq!(map[&10], out1);
    }

    #[test]
    fn decode_request_truncated_returns_error() {
        let result = decode_q8k_batch_request(&[0u8; 3]);
        assert!(result.is_err());
    }

    #[test]
    fn decode_response_truncated_returns_error() {
        let result = decode_q8k_batch_response(&[0u8; 3]);
        assert!(result.is_err());
    }

    #[test]
    fn empty_batch_roundtrip() {
        let body = encode_q8k_batch_request(&[]);
        let decoded = decode_q8k_batch_request(&body).unwrap();
        assert!(decoded.is_empty());

        let body2 = encode_q8k_batch_response(&[]);
        let map = decode_q8k_batch_response(&body2).unwrap();
        assert!(map.is_empty());
    }
}
