//! Cold-tier codec choice.
//!
//! `Bf16` is the only v0.1 default — it is the one codec in `larql-boundary`
//! whose §4.7 calibration is robust without a per-layer sweep. The other
//! codecs in the enum are reserved for §3 of the spec ("present in the
//! configuration surface for users who can tolerate the weaker contract or
//! who have run their own per-layer calibration").
//!
//! Construction of the engine with a non-`Bf16` codec **must** require an
//! explicit acknowledgement that the contract is weaker — implementations
//! enforce this through the `CodecAck` parameter in [`engine`].

use larql_boundary::codec::bf16 as codec_bf16;
use ndarray::Array2;

/// Codec selection for the cold residual tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdResidualCodec {
    /// `f32` → `bf16` → `f32` per element. 2× smaller than f32. The only
    /// v0.1 default.
    Bf16,
}

impl ColdResidualCodec {
    /// Bytes per element in the encoded payload.
    pub const fn bytes_per_elem(&self) -> usize {
        match self {
            Self::Bf16 => 2,
        }
    }

    /// Human-readable label.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
        }
    }
}

/// Encode a 2-D residual block to the codec's wire form.
///
/// The encoding is per-row: for codecs that carry per-row state (e.g.
/// `Int8Clip3Sigma` in the broader `larql-boundary` family) this matters.
/// For `Bf16` it is a trivial whole-buffer transformation.
pub fn encode_block(codec: ColdResidualCodec, residuals: &Array2<f32>) -> Vec<u8> {
    match codec {
        ColdResidualCodec::Bf16 => {
            let slice = residuals
                .as_slice()
                .expect("residual array must be standard-layout for encode");
            codec_bf16::encode(slice)
        }
    }
}

/// Decode the codec's wire form back to an `[n_positions, hidden_size]` block.
///
/// # Panics
/// Panics if `payload.len()` does not match `n_positions × hidden_size ×
/// codec.bytes_per_elem()`.
pub fn decode_block(
    codec: ColdResidualCodec,
    payload: &[u8],
    n_positions: usize,
    hidden_size: usize,
) -> Array2<f32> {
    let expected = n_positions * hidden_size * codec.bytes_per_elem();
    assert_eq!(
        payload.len(),
        expected,
        "payload length {} does not match {n_positions} × {hidden_size} × {} = {expected}",
        payload.len(),
        codec.bytes_per_elem(),
    );
    let flat: Vec<f32> = match codec {
        ColdResidualCodec::Bf16 => codec_bf16::decode(payload),
    };
    Array2::from_shape_vec((n_positions, hidden_size), flat)
        .expect("decoded payload must reshape to [n_positions, hidden_size]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_bytes_per_elem_is_two() {
        assert_eq!(ColdResidualCodec::Bf16.bytes_per_elem(), 2);
    }

    #[test]
    fn bf16_label_matches() {
        assert_eq!(ColdResidualCodec::Bf16.label(), "bf16");
    }

    #[test]
    fn encode_then_decode_roundtrips_to_bf16_precision() {
        let a = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, -3.0, 0.5, 10.0, 0.0, -1.5, 100.0, 0.125, 0.25, 0.5, 1.0,
            ],
        )
        .unwrap();
        let bytes = encode_block(ColdResidualCodec::Bf16, &a);
        assert_eq!(bytes.len(), 3 * 4 * 2);
        let dec = decode_block(ColdResidualCodec::Bf16, &bytes, 3, 4);
        for (orig, got) in a.iter().zip(dec.iter()) {
            assert!(
                (orig - got).abs() <= orig.abs() * 0.01 + 1e-3,
                "bf16 roundtrip drift: orig={orig} got={got}"
            );
        }
    }

    #[test]
    fn empty_block_roundtrips() {
        let a: Array2<f32> = Array2::zeros((0, 4));
        let bytes = encode_block(ColdResidualCodec::Bf16, &a);
        assert!(bytes.is_empty());
        let dec = decode_block(ColdResidualCodec::Bf16, &bytes, 0, 4);
        assert_eq!(dec.shape(), &[0, 4]);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn decode_panics_on_wrong_length() {
        let bytes = vec![0u8; 6]; // 3 elements at bf16, but caller asks for 2×4=8
        let _ = decode_block(ColdResidualCodec::Bf16, &bytes, 2, 4);
    }
}
