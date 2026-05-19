//! Binary wire format for `POST /v1/experts/multi-layer-batch`.
//!
//! Collapses 30 per-layer HTTP requests into one per shard, eliminating the
//! per-request HTTPS overhead (~20 ms × 30 = 600 ms in the predispatch path).
//! The server processes tasks sequentially so rayon runs at full utilisation
//! (no oversubscription); the client parallelises across shards only.
//!
//! Request layout (little-endian):
//!   u32  num_tasks
//!   for each task:
//!     u32  layer
//!     u32  hidden            (residual length = h_post_attn size)
//!     u32  num_experts
//!     f32[hidden]  residual
//!     u32[n]       expert_ids
//!     f32[n]       weights
//!
//! Response layout:
//!   u32  num_results
//!   for each result:
//!     u32  layer
//!     u32  hidden
//!     f32[hidden]  h2         (raw weighted sum; caller applies post-experts norm)

pub const MULTI_LAYER_BATCH_CONTENT_TYPE: &str = "application/x-larql-experts-multi-layer";

/// HTTP path served by the multi-layer batch endpoint.
pub const MULTI_LAYER_BATCH_PATH: &str = "/v1/experts/multi-layer-batch";

/// Q8K-prenormed variant: client sends `h_norm` pre-quantised to Q8_K
/// (already computed during routing — zero extra client compute).  Server
/// skips `pre_experts_norm` + `quantize_h_norm_for_q4k` and calls the
/// matvec directly.  4× smaller upload than the f32 residual path.
///
/// Request layout — same header as f32, but residual field replaced:
///   u32  num_tasks
///   for each task:
///     u32  layer
///     u32  hidden              (= n_blocks × 256)
///     u32  num_experts
///     i8[hidden]  q8k_qs       (quantised activation)
///     f32[n_blocks]  q8k_d     (per-super-block scales)
///     i16[n_blocks × 8]  q8k_sums  (precomputed sub-block sums)
///     u32[num_experts]  expert_ids
///     f32[num_experts]  weights
pub const MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE: &str = "application/x-larql-experts-multi-layer-q8k";

/// HTTP path served by the Q8K-prenormed multi-layer batch endpoint.
pub const MULTI_LAYER_BATCH_Q8K_PATH: &str = "/v1/experts/multi-layer-batch-q8k";

pub struct MultiLayerTask {
    pub layer: usize,
    pub residual: Vec<f32>,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Q8K-prenormed task: carries already-quantised h_norm so the server skips
/// normalisation and directly calls `q4k_q8k_matvec_into`.
pub struct MultiLayerTaskQ8K {
    pub layer: usize,
    pub hidden: usize,
    /// Flat i8 activation: `qs[block * 256 .. (block+1) * 256]` per block.
    pub qs: Vec<i8>,
    /// Per-super-block f32 scale: `d[block]`.
    pub d: Vec<f32>,
    /// Per-sub-block i16 sums: `sums[block * 8 + sb]`.
    pub sums: Vec<i16>,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

pub struct MultiLayerResult {
    pub layer: usize,
    pub h2: Vec<f32>,
}

pub fn encode_multi_layer_request(tasks: &[MultiLayerTask]) -> Vec<u8> {
    let cap = 4 + tasks
        .iter()
        .map(|t| 12 + t.residual.len() * 4 + t.expert_ids.len() * 8)
        .sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, tasks.len() as u32);
    for t in tasks {
        push_u32(&mut buf, t.layer as u32);
        push_u32(&mut buf, t.residual.len() as u32);
        push_u32(&mut buf, t.expert_ids.len() as u32);
        for &v in &t.residual {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &e in &t.expert_ids {
            push_u32(&mut buf, e);
        }
        for &w in &t.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    buf
}

pub fn decode_multi_layer_request(bytes: &[u8]) -> Option<Vec<MultiLayerTask>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    let mut tasks = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let ne = read_u32(bytes, &mut pos)? as usize;
        let residual = read_f32_slice(bytes, &mut pos, hidden)?;
        let mut expert_ids = Vec::with_capacity(ne);
        for _ in 0..ne {
            expert_ids.push(read_u32(bytes, &mut pos)?);
        }
        let mut weights = Vec::with_capacity(ne);
        for _ in 0..ne {
            weights.push(read_f32(bytes, &mut pos)?);
        }
        tasks.push(MultiLayerTask {
            layer,
            residual,
            expert_ids,
            weights,
        });
    }
    Some(tasks)
}

pub fn encode_multi_layer_response(results: &[MultiLayerResult]) -> Vec<u8> {
    let cap = 4 + results.iter().map(|r| 8 + r.h2.len() * 4).sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, results.len() as u32);
    for r in results {
        push_u32(&mut buf, r.layer as u32);
        push_u32(&mut buf, r.h2.len() as u32);
        for &v in &r.h2 {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

pub fn decode_multi_layer_response(bytes: &[u8]) -> Option<Vec<MultiLayerResult>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let h2 = read_f32_slice(bytes, &mut pos, hidden)?;
        results.push(MultiLayerResult { layer, h2 });
    }
    Some(results)
}

// ── Q8K-prenormed wire ────────────────────────────────────────────────────────

use crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS as ELEMS_PER_Q8K_BLOCK;
const SUMS_PER_Q8K_BLOCK: usize = 8;

pub fn encode_multi_layer_request_q8k(tasks: &[MultiLayerTaskQ8K]) -> Vec<u8> {
    let cap = 4 + tasks
        .iter()
        .map(|t| {
            let nb = t.hidden / ELEMS_PER_Q8K_BLOCK;
            12 // layer + hidden + num_experts
            + t.hidden  // qs (i8)
            + nb * 4    // d (f32)
            + nb * SUMS_PER_Q8K_BLOCK * 2  // sums (i16)
            + t.expert_ids.len() * 8 // expert_ids + weights
        })
        .sum::<usize>();
    let mut buf = Vec::with_capacity(cap);
    push_u32(&mut buf, tasks.len() as u32);
    for t in tasks {
        let nb = t.hidden / ELEMS_PER_Q8K_BLOCK;
        push_u32(&mut buf, t.layer as u32);
        push_u32(&mut buf, t.hidden as u32);
        push_u32(&mut buf, t.expert_ids.len() as u32);
        // Q8K activation
        for &q in &t.qs {
            buf.push(q as u8);
        }
        for &v in &t.d {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &s in &t.sums {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        debug_assert_eq!(t.qs.len(), t.hidden, "qs length mismatch");
        debug_assert_eq!(t.d.len(), nb, "d length mismatch");
        debug_assert_eq!(
            t.sums.len(),
            nb * SUMS_PER_Q8K_BLOCK,
            "sums length mismatch"
        );
        // Expert routing
        for &e in &t.expert_ids {
            push_u32(&mut buf, e);
        }
        for &w in &t.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    buf
}

pub fn decode_multi_layer_request_q8k(bytes: &[u8]) -> Option<Vec<MultiLayerTaskQ8K>> {
    let mut pos = 0;
    let n = read_u32(bytes, &mut pos)? as usize;
    let mut tasks = Vec::with_capacity(n);
    for _ in 0..n {
        let layer = read_u32(bytes, &mut pos)? as usize;
        let hidden = read_u32(bytes, &mut pos)? as usize;
        let ne = read_u32(bytes, &mut pos)? as usize;
        let nb = hidden / ELEMS_PER_Q8K_BLOCK;
        // Q8K activation
        let qs = read_i8_slice(bytes, &mut pos, hidden)?;
        let d = read_f32_slice(bytes, &mut pos, nb)?;
        let sums = read_i16_slice(bytes, &mut pos, nb * SUMS_PER_Q8K_BLOCK)?;
        // Expert routing
        let mut expert_ids = Vec::with_capacity(ne);
        for _ in 0..ne {
            expert_ids.push(read_u32(bytes, &mut pos)?);
        }
        let mut weights = Vec::with_capacity(ne);
        for _ in 0..ne {
            weights.push(read_f32(bytes, &mut pos)?);
        }
        tasks.push(MultiLayerTaskQ8K {
            layer,
            hidden,
            qs,
            d,
            sums,
            expert_ids,
            weights,
        });
    }
    Some(tasks)
}

fn read_i8_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<i8>> {
    let end = pos.checked_add(n)?;
    if end > bytes.len() {
        return None;
    }
    let v: Vec<i8> = bytes[*pos..end].iter().map(|&b| b as i8).collect();
    *pos = end;
    Some(v)
}

fn read_i16_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<i16>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let end = pos.checked_add(2)?;
        if end > bytes.len() {
            return None;
        }
        let val = i16::from_le_bytes(bytes[*pos..end].try_into().unwrap());
        *pos = end;
        v.push(val);
    }
    Some(v)
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let end = pos.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let v = u32::from_le_bytes(bytes[*pos..end].try_into().unwrap());
    *pos = end;
    Some(v)
}

fn read_f32(bytes: &[u8], pos: &mut usize) -> Option<f32> {
    let end = pos.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let v = f32::from_le_bytes(bytes[*pos..end].try_into().unwrap());
    *pos = end;
    Some(v)
}

fn read_f32_slice(bytes: &[u8], pos: &mut usize, n: usize) -> Option<Vec<f32>> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(read_f32(bytes, pos)?);
    }
    Some(v)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn request_round_trip() {
        let tasks = vec![
            MultiLayerTask {
                layer: 0,
                residual: vec![1.0, 2.0, 3.0],
                expert_ids: vec![5, 17],
                weights: vec![0.6, 0.4],
            },
            MultiLayerTask {
                layer: 7,
                residual: vec![0.5, -1.0, 2.5],
                expert_ids: vec![42],
                weights: vec![1.0],
            },
        ];
        let encoded = encode_multi_layer_request(&tasks);
        let decoded = decode_multi_layer_request(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].layer, 0);
        assert_eq!(decoded[0].residual, vec![1.0, 2.0, 3.0]);
        assert_eq!(decoded[0].expert_ids, vec![5, 17]);
        assert_eq!(decoded[0].weights, vec![0.6, 0.4]);
        assert_eq!(decoded[1].layer, 7);
        assert_eq!(decoded[1].expert_ids, vec![42]);
    }

    #[test]
    fn response_round_trip() {
        let results = vec![
            MultiLayerResult {
                layer: 3,
                h2: vec![0.1, 0.2, 0.3],
            },
            MultiLayerResult {
                layer: 15,
                h2: vec![-1.0, 0.0, 1.0],
            },
        ];
        let encoded = encode_multi_layer_response(&results);
        let decoded = decode_multi_layer_response(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].layer, 3);
        assert_eq!(decoded[0].h2, vec![0.1, 0.2, 0.3]);
        assert_eq!(decoded[1].layer, 15);
    }

    #[test]
    fn handles_truncation() {
        assert!(decode_multi_layer_request(&[]).is_none());
        assert!(decode_multi_layer_request(&[0, 0, 0, 1]).is_none()); // claims 1 task but no body
        assert!(decode_multi_layer_response(&[]).is_none());
    }

    #[test]
    fn empty_request_round_trips_to_zero_tasks() {
        let encoded = encode_multi_layer_request(&[]);
        // Just the [u32 num_tasks=0] header.
        assert_eq!(encoded.len(), 4);
        let decoded = decode_multi_layer_request(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn empty_response_round_trips_to_zero_results() {
        let encoded = encode_multi_layer_response(&[]);
        assert_eq!(encoded.len(), 4);
        let decoded = decode_multi_layer_response(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn request_with_zero_experts_round_trips() {
        // Skip-vote pattern: a layer that routed nothing — encoder must
        // still emit the header so the layer index isn't lost.
        let tasks = vec![MultiLayerTask {
            layer: 9,
            residual: vec![0.0, 0.0, 0.0, 0.0],
            expert_ids: vec![],
            weights: vec![],
        }];
        let encoded = encode_multi_layer_request(&tasks);
        let decoded = decode_multi_layer_request(&encoded).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].layer, 9);
        assert_eq!(decoded[0].residual.len(), 4);
        assert!(decoded[0].expert_ids.is_empty());
        assert!(decoded[0].weights.is_empty());
    }

    #[test]
    fn truncated_response_returns_none() {
        let encoded = encode_multi_layer_response(&[MultiLayerResult {
            layer: 1,
            h2: vec![1.0; 4],
        }]);
        // Drop the last byte → truncated f32; decoder must reject.
        assert!(decode_multi_layer_response(&encoded[..encoded.len() - 1]).is_none());
    }

    #[test]
    fn truncated_request_returns_none_at_each_field() {
        let encoded = encode_multi_layer_request(&[MultiLayerTask {
            layer: 0,
            residual: vec![1.0, 2.0],
            expert_ids: vec![0],
            weights: vec![1.0],
        }]);
        for cut in 1..encoded.len() {
            // Every prefix shorter than the full encoding must be rejected
            // — there's no valid framing to recover from.
            assert!(
                decode_multi_layer_request(&encoded[..cut]).is_none(),
                "decode succeeded on prefix len={cut}, expected None"
            );
        }
    }

    // ── Q8K-prenormed wire ──────────────────────────────────────────────

    fn make_q8k_task(layer: usize, hidden: usize, ne: usize) -> MultiLayerTaskQ8K {
        let nb = hidden / ELEMS_PER_Q8K_BLOCK;
        MultiLayerTaskQ8K {
            layer,
            hidden,
            qs: (0..hidden)
                .map(|i| ((i % 256) as i32 - 128) as i8)
                .collect(),
            d: (0..nb).map(|i| 0.01 * (i as f32 + 1.0)).collect(),
            sums: (0..nb * SUMS_PER_Q8K_BLOCK)
                .map(|i| (i as i16) - 64)
                .collect(),
            expert_ids: (0..ne).map(|i| (i as u32) * 17).collect(),
            weights: (0..ne)
                .map(|i| 1.0 / ne.max(1) as f32 * (i as f32 + 1.0))
                .collect(),
        }
    }

    #[test]
    fn q8k_request_round_trip_single_block() {
        let tasks = vec![make_q8k_task(3, ELEMS_PER_Q8K_BLOCK, 4)];
        let encoded = encode_multi_layer_request_q8k(&tasks);
        let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
        assert_eq!(decoded.len(), 1);
        let t = &decoded[0];
        assert_eq!(t.layer, 3);
        assert_eq!(t.hidden, ELEMS_PER_Q8K_BLOCK);
        assert_eq!(t.qs, tasks[0].qs);
        assert_eq!(t.d, tasks[0].d);
        assert_eq!(t.sums, tasks[0].sums);
        assert_eq!(t.expert_ids, tasks[0].expert_ids);
        assert_eq!(t.weights, tasks[0].weights);
    }

    #[test]
    fn q8k_request_round_trip_multi_block_multi_task() {
        // Two tasks, different hidden sizes → both nb counts must be
        // independently respected by the decoder.
        let tasks = vec![
            make_q8k_task(0, ELEMS_PER_Q8K_BLOCK, 2),
            make_q8k_task(11, ELEMS_PER_Q8K_BLOCK * 3, 8),
        ];
        let encoded = encode_multi_layer_request_q8k(&tasks);
        let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        for (orig, got) in tasks.iter().zip(decoded.iter()) {
            assert_eq!(orig.layer, got.layer);
            assert_eq!(orig.hidden, got.hidden);
            assert_eq!(orig.qs, got.qs);
            assert_eq!(orig.d, got.d);
            assert_eq!(orig.sums, got.sums);
            assert_eq!(orig.expert_ids, got.expert_ids);
            assert_eq!(orig.weights, got.weights);
        }
    }

    #[test]
    fn q8k_request_with_zero_experts_round_trips() {
        let tasks = vec![make_q8k_task(2, ELEMS_PER_Q8K_BLOCK, 0)];
        let encoded = encode_multi_layer_request_q8k(&tasks);
        let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
        assert_eq!(decoded.len(), 1);
        assert!(decoded[0].expert_ids.is_empty());
        assert!(decoded[0].weights.is_empty());
        // Activation payload still present.
        assert_eq!(decoded[0].qs.len(), ELEMS_PER_Q8K_BLOCK);
    }

    #[test]
    fn empty_q8k_request_round_trips() {
        let encoded = encode_multi_layer_request_q8k(&[]);
        assert_eq!(encoded.len(), 4);
        let decoded = decode_multi_layer_request_q8k(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn truncated_q8k_request_returns_none_at_each_field() {
        let encoded = encode_multi_layer_request_q8k(&[make_q8k_task(0, ELEMS_PER_Q8K_BLOCK, 1)]);
        for cut in 1..encoded.len() {
            assert!(
                decode_multi_layer_request_q8k(&encoded[..cut]).is_none(),
                "Q8K decode succeeded on prefix len={cut}, expected None"
            );
        }
    }

    #[test]
    fn read_i8_slice_handles_signed_bytes() {
        // i8 round-trip via u8 byte storage: 0xff (255) must surface as -1.
        let bytes = [0u8, 0x7f, 0x80, 0xff];
        let mut pos = 0;
        let v = read_i8_slice(&bytes, &mut pos, 4).unwrap();
        assert_eq!(v, vec![0i8, 127, -128, -1]);
        assert_eq!(pos, 4);
    }

    #[test]
    fn read_i16_slice_handles_negative_values() {
        // Three i16 little-endian: 0, 32767, -1.
        let bytes = [0x00, 0x00, 0xff, 0x7f, 0xff, 0xff];
        let mut pos = 0;
        let v = read_i16_slice(&bytes, &mut pos, 3).unwrap();
        assert_eq!(v, vec![0i16, 32767, -1]);
        assert_eq!(pos, 6);
    }

    #[test]
    fn read_helpers_reject_overruns() {
        let bytes = [0u8; 4];
        let mut pos = 0;
        // Asking for one past the end is None; pos unchanged.
        assert!(read_u32(&bytes, &mut 1).is_none());
        assert!(read_f32(&bytes, &mut 1).is_none());
        assert!(read_f32_slice(&bytes, &mut pos, 2).is_none());
        assert!(read_i8_slice(&bytes, &mut pos, 5).is_none());
        assert!(read_i16_slice(&bytes, &mut pos, 3).is_none());
    }

    #[test]
    fn content_type_and_path_consts_pin_wire_strings() {
        // Renaming any of these breaks deployed clients/servers.
        assert_eq!(
            MULTI_LAYER_BATCH_CONTENT_TYPE,
            "application/x-larql-experts-multi-layer"
        );
        assert_eq!(MULTI_LAYER_BATCH_PATH, "/v1/experts/multi-layer-batch");
        assert_eq!(
            MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
            "application/x-larql-experts-multi-layer-q8k"
        );
        assert_eq!(
            MULTI_LAYER_BATCH_Q8K_PATH,
            "/v1/experts/multi-layer-batch-q8k"
        );
    }
}
