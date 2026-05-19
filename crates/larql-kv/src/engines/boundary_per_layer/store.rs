//! `RsStorePerLayer` — `RsStore` with per-layer codec choice.
//!
//! For v0.1 the codec for every layer is restricted to `Bf16` (per
//! [`super::policy::BoundaryLayerPolicy`]), so the encoding logic reduces
//! to `MarkovResidualCodecEngine`'s. The infrastructure is laid out for
//! per-layer mixing once additional codecs gain calibration support.

use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use crate::engines::markov_residual_codec::codec::{decode_block, encode_block, ColdResidualCodec};

/// Per-layer encoded cold residuals. Carries its own codec so each layer
/// can be decoded independently of the others.
#[derive(Debug, Clone)]
pub struct PerLayerEncodedColdLayer {
    pub codec: ColdResidualCodec,
    pub n_positions: usize,
    pub hidden_size: usize,
    pub payload: Vec<u8>,
}

impl PerLayerEncodedColdLayer {
    pub fn empty(codec: ColdResidualCodec, hidden_size: usize) -> Self {
        Self {
            codec,
            n_positions: 0,
            hidden_size,
            payload: Vec::new(),
        }
    }

    pub fn append(&mut self, block: &Array2<f32>) {
        let rows = block.shape()[0];
        let cols = block.shape()[1];
        assert_eq!(
            cols, self.hidden_size,
            "PerLayerEncodedColdLayer hidden_size mismatch (have {}, got {cols})",
            self.hidden_size
        );
        if rows == 0 {
            return;
        }
        let block_bytes = encode_block(self.codec, block);
        self.payload.extend_from_slice(&block_bytes);
        self.n_positions += rows;
    }

    pub fn decode(&self) -> Array2<f32> {
        decode_block(
            self.codec,
            &self.payload,
            self.n_positions,
            self.hidden_size,
        )
    }
}

/// `RsStorePerLayer` — hot residuals (f32) + per-layer cold encodings.
pub struct RsStorePerLayer {
    pub stored: Vec<Array2<f32>>,
    pub cold_encoded: Option<Vec<PerLayerEncodedColdLayer>>,
    pub cold_kv: Option<Vec<SharedKV>>,
    pub cold_abs_start: usize,
    pub next_position: usize,
    pub max_window: Option<usize>,
    /// Per-layer codec choice; `policy_codecs.len()` matches `weights.num_layers`.
    pub policy_codecs: Vec<ColdResidualCodec>,
}

impl RsStorePerLayer {
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

    #[test]
    fn empty_encoded_layer_starts_at_zero() {
        let l = PerLayerEncodedColdLayer::empty(ColdResidualCodec::Bf16, 8);
        assert_eq!(l.n_positions, 0);
        assert_eq!(l.hidden_size, 8);
        assert!(l.payload.is_empty());
        assert_eq!(l.codec, ColdResidualCodec::Bf16);
    }

    #[test]
    fn append_grows_payload_and_count() {
        let mut l = PerLayerEncodedColdLayer::empty(ColdResidualCodec::Bf16, 4);
        let block = Array2::<f32>::ones((2, 4));
        l.append(&block);
        assert_eq!(l.n_positions, 2);
        assert_eq!(l.payload.len(), 2 * 4 * 2);
    }

    #[test]
    fn append_then_decode_roundtrips() {
        let mut l = PerLayerEncodedColdLayer::empty(ColdResidualCodec::Bf16, 2);
        let block = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        l.append(&block);
        let dec = l.decode();
        for (orig, got) in block.iter().zip(dec.iter()) {
            assert!((orig - got).abs() < 0.1);
        }
    }

    #[test]
    fn append_empty_block_is_noop() {
        let mut l = PerLayerEncodedColdLayer::empty(ColdResidualCodec::Bf16, 4);
        let block: Array2<f32> = Array2::zeros((0, 4));
        l.append(&block);
        assert_eq!(l.n_positions, 0);
    }

    #[test]
    #[should_panic(expected = "hidden_size mismatch")]
    fn append_wrong_hidden_size_panics() {
        let mut l = PerLayerEncodedColdLayer::empty(ColdResidualCodec::Bf16, 4);
        let block: Array2<f32> = Array2::zeros((1, 5));
        l.append(&block);
    }

    // ── RsStorePerLayer ──────────────────────────────────────────────────────

    fn make_store(num_layers: usize, seq_len: usize, hidden: usize) -> RsStorePerLayer {
        let stored = (0..num_layers)
            .map(|_| Array2::from_elem((seq_len, hidden), 1.0f32))
            .collect();
        RsStorePerLayer {
            stored,
            cold_encoded: None,
            cold_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: None,
            policy_codecs: vec![ColdResidualCodec::Bf16; num_layers],
        }
    }

    #[test]
    fn memory_bytes_hot_only() {
        let s = make_store(2, 3, 8);
        assert_eq!(s.memory_bytes(), 2 * 3 * 8 * 4);
        assert_eq!(s.cold_bytes(), 0);
    }

    #[test]
    fn window_tokens_matches_stored() {
        let s = make_store(2, 5, 4);
        assert_eq!(s.window_tokens(), 5);
    }

    #[test]
    fn window_tokens_zero_for_empty() {
        let s = make_store(0, 0, 4);
        assert_eq!(s.window_tokens(), 0);
    }

    #[test]
    fn clip_overflow_zero_window_returns_empty() {
        let mut s = make_store(1, 10, 4);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 0);
        assert_eq!(s.stored[0].shape()[0], 10);
    }

    #[test]
    fn clip_overflow_within_window_returns_empty() {
        let mut s = make_store(1, 4, 4);
        s.max_window = Some(8);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 0);
    }

    #[test]
    fn clip_overflow_excess_moves() {
        let mut s = make_store(1, 10, 4);
        s.max_window = Some(3);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 7);
        assert_eq!(s.stored[0].shape()[0], 3);
    }

    #[test]
    fn cold_bytes_includes_payload() {
        let mut s = make_store(1, 0, 4);
        s.cold_encoded = Some(vec![PerLayerEncodedColdLayer {
            codec: ColdResidualCodec::Bf16,
            n_positions: 2,
            hidden_size: 4,
            payload: vec![0u8; 16],
        }]);
        assert_eq!(s.cold_bytes(), 16);
    }

    #[test]
    fn memory_bytes_with_cold_kv_populated_uses_kv_branch() {
        // The `cold_kv` closure inside `memory_bytes` is only exercised when
        // cold_kv is `Some`; the existing tests leave it `None`. We construct
        // a minimal `SharedKV` pair (empty ndarrays) so the branch's `.map`
        // closure fires.
        use ndarray::Array2;
        let mut s = make_store(2, 0, 4);
        let k1 = Array2::<f32>::zeros((3, 2));
        let v1 = Array2::<f32>::zeros((3, 2));
        let k2 = Array2::<f32>::zeros((1, 2));
        let v2 = Array2::<f32>::zeros((1, 2));
        s.cold_kv = Some(vec![(k1, v1), (k2, v2)]);
        // Layer 0: (3*2 + 3*2) * 4 = 48 bytes. Layer 1: (1*2 + 1*2) * 4 = 16.
        assert_eq!(s.cold_bytes(), 48 + 16);
        // memory_bytes folds hot + cold_enc (0) + cold_kv (64).
        assert_eq!(s.memory_bytes(), 64);
    }

    #[test]
    fn memory_bytes_with_both_cold_payload_and_kv() {
        // Exercises both `.map` closures inside `memory_bytes`.
        let mut s = make_store(1, 0, 4);
        s.cold_encoded = Some(vec![PerLayerEncodedColdLayer {
            codec: ColdResidualCodec::Bf16,
            n_positions: 2,
            hidden_size: 4,
            payload: vec![0u8; 16],
        }]);
        let k = ndarray::Array2::<f32>::zeros((2, 4));
        let v = ndarray::Array2::<f32>::zeros((2, 4));
        s.cold_kv = Some(vec![(k, v)]);
        assert_eq!(s.memory_bytes(), 16 + (2 * 4 + 2 * 4) * 4);
        assert_eq!(s.cold_bytes(), 16 + (2 * 4 + 2 * 4) * 4);
    }
}
