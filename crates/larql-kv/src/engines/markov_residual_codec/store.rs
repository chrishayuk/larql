//! `RsStoreCodec` — `RsStore` with a codec-encoded cold tier.

use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use crate::engines::markov_residual_codec::codec::{decode_block, encode_block, ColdResidualCodec};

/// Per-layer encoded cold residuals.
#[derive(Debug, Clone)]
pub struct EncodedColdLayer {
    /// Number of cold positions stored.
    pub n_positions: usize,
    /// Hidden size (constant per layer).
    pub hidden_size: usize,
    /// Encoded payload bytes for `n_positions × hidden_size` elements.
    pub payload: Vec<u8>,
}

impl EncodedColdLayer {
    pub fn empty(hidden_size: usize) -> Self {
        Self {
            n_positions: 0,
            hidden_size,
            payload: Vec::new(),
        }
    }

    /// Append `block` (which must have the same `hidden_size`) to the existing
    /// encoded payload. The codec is applied to `block` once on append.
    pub fn append(&mut self, codec: ColdResidualCodec, block: &Array2<f32>) {
        let cols = block.shape()[1];
        let rows = block.shape()[0];
        assert_eq!(
            cols, self.hidden_size,
            "EncodedColdLayer hidden_size mismatch (have {}, got {cols})",
            self.hidden_size
        );
        if rows == 0 {
            return;
        }
        let block_bytes = encode_block(codec, block);
        self.payload.extend_from_slice(&block_bytes);
        self.n_positions += rows;
    }

    /// Decode the layer back to a 2-D `f32` block.
    pub fn decode(&self, codec: ColdResidualCodec) -> Array2<f32> {
        decode_block(codec, &self.payload, self.n_positions, self.hidden_size)
    }
}

/// `RsStoreCodec` — per-layer hot residuals (f32) + per-layer codec-encoded
/// cold residuals. Mirrors `RsStore` from the `markov_residual` engine, with
/// the cold tier swapped for a byte-packed representation.
pub struct RsStoreCodec {
    pub stored: Vec<Array2<f32>>,
    pub cold_encoded: Option<Vec<EncodedColdLayer>>,
    pub cold_kv: Option<Vec<SharedKV>>,
    pub cold_abs_start: usize,
    pub next_position: usize,
    pub max_window: Option<usize>,
    pub codec: ColdResidualCodec,
}

impl RsStoreCodec {
    pub fn memory_bytes(&self) -> usize {
        let hot: usize = self.stored.iter().map(|s| s.len() * 4).sum();
        let cold_enc: usize = self
            .cold_encoded
            .as_ref()
            .map(|layers| layers.iter().map(|l| l.payload.len()).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        hot + cold_enc + cold_kv
    }

    pub fn cold_bytes(&self) -> usize {
        let cold_enc: usize = self
            .cold_encoded
            .as_ref()
            .map(|layers| layers.iter().map(|l| l.payload.len()).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        cold_enc + cold_kv
    }

    pub fn window_tokens(&self) -> usize {
        self.stored.first().map_or(0, |s| s.shape()[0])
    }

    /// Clip the hot tier for `layer` against `max_window`. Returns the
    /// overflow as an `f32` block (the caller is responsible for encoding it
    /// onto the cold tier).
    pub(crate) fn clip_layer_overflow(&mut self, layer: usize) -> Array2<f32> {
        let window = match self.max_window {
            Some(w) => w,
            None => return Array2::zeros((0, self.stored[layer].shape()[1])),
        };
        let s_arr = &self.stored[layer];
        let rows = s_arr.shape()[0];
        let cols = s_arr.shape()[1];
        if rows <= window {
            return Array2::zeros((0, cols));
        }
        let start = rows - window;
        let overflow = s_arr.slice(s![..start, ..]).to_owned();
        self.stored[layer] = s_arr.slice(s![start.., ..]).to_owned();
        overflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(num_layers: usize, seq_len: usize, hidden: usize) -> RsStoreCodec {
        let stored = (0..num_layers)
            .map(|_| Array2::from_elem((seq_len, hidden), 1.0f32))
            .collect();
        RsStoreCodec {
            stored,
            cold_encoded: None,
            cold_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: None,
            codec: ColdResidualCodec::Bf16,
        }
    }

    #[test]
    fn encoded_layer_empty_starts_at_zero() {
        let l = EncodedColdLayer::empty(16);
        assert_eq!(l.n_positions, 0);
        assert_eq!(l.hidden_size, 16);
        assert!(l.payload.is_empty());
    }

    #[test]
    fn append_block_grows_payload_and_count() {
        let mut l = EncodedColdLayer::empty(4);
        let block = Array2::<f32>::ones((3, 4));
        l.append(ColdResidualCodec::Bf16, &block);
        assert_eq!(l.n_positions, 3);
        // bf16 = 2 bytes per element.
        assert_eq!(l.payload.len(), 3 * 4 * 2);
    }

    #[test]
    fn append_then_decode_roundtrips() {
        let mut l = EncodedColdLayer::empty(2);
        let block = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        l.append(ColdResidualCodec::Bf16, &block);
        let dec = l.decode(ColdResidualCodec::Bf16);
        assert_eq!(dec.shape(), &[2, 2]);
        for (orig, got) in block.iter().zip(dec.iter()) {
            assert!((orig - got).abs() < 0.1);
        }
    }

    #[test]
    fn append_empty_block_is_noop() {
        let mut l = EncodedColdLayer::empty(4);
        let block: Array2<f32> = Array2::zeros((0, 4));
        l.append(ColdResidualCodec::Bf16, &block);
        assert_eq!(l.n_positions, 0);
        assert!(l.payload.is_empty());
    }

    #[test]
    #[should_panic(expected = "hidden_size mismatch")]
    fn append_wrong_hidden_size_panics() {
        let mut l = EncodedColdLayer::empty(4);
        let block: Array2<f32> = Array2::zeros((1, 5)); // wrong hidden
        l.append(ColdResidualCodec::Bf16, &block);
    }

    // ── RsStoreCodec ──────────────────────────────────────────────────────────

    #[test]
    fn memory_bytes_hot_only() {
        let s = make_store(2, 3, 8);
        assert_eq!(s.memory_bytes(), 2 * 3 * 8 * 4);
    }

    #[test]
    fn cold_bytes_zero_when_no_cold() {
        let s = make_store(1, 3, 8);
        assert_eq!(s.cold_bytes(), 0);
    }

    #[test]
    fn window_tokens_matches_stored() {
        let s = make_store(2, 5, 4);
        assert_eq!(s.window_tokens(), 5);
    }

    #[test]
    fn window_tokens_zero_for_empty_store() {
        let s = make_store(0, 0, 4);
        assert_eq!(s.window_tokens(), 0);
    }

    #[test]
    fn clip_layer_no_window_returns_empty_overflow() {
        let mut s = make_store(1, 10, 4);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 0);
        assert_eq!(s.stored[0].shape()[0], 10);
    }

    #[test]
    fn clip_layer_within_window_returns_empty() {
        let mut s = make_store(1, 4, 4);
        s.max_window = Some(8);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 0);
        assert_eq!(s.stored[0].shape()[0], 4);
    }

    #[test]
    fn clip_layer_excess_overflow_moves() {
        let mut s = make_store(1, 10, 4);
        s.max_window = Some(3);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 7);
        assert_eq!(s.stored[0].shape()[0], 3);
    }

    #[test]
    fn memory_includes_cold_payloads_and_kv() {
        let mut s = make_store(1, 0, 4);
        s.cold_encoded = Some(vec![EncodedColdLayer {
            n_positions: 3,
            hidden_size: 4,
            payload: vec![0u8; 24], // 3 × 4 × 2
        }]);
        let cold_only = s.memory_bytes();
        assert_eq!(cold_only, 24);
        assert_eq!(s.cold_bytes(), 24);
    }
}
