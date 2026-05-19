//! f16/bf16 ↔ f32 conversion.

/// Convert f16 bits to f32.
///
/// Subnormals are reconstructed as `m * 2^-24` where `m` is the 10-bit
/// mantissa (no implicit leading 1). The previous normalisation formula
/// `127 - 15 + 1 - e` produced values exactly 2× too small for every
/// subnormal path — fine when all scales were normal floats (legacy quant
/// settings), catastrophic once k-quant super-block scales were forced
/// into f16 subnormal range by the corrected Q4_K/Q6_K scale formulas.
/// The right formula is `114 - e`: for `e = shifts + 1`, we need f32
/// biased exponent `127 + (-14 - shifts)` = `114 - e`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        return f32::from_bits(sign | ((114 - e) << 23) | ((m & 0x3FF) << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | (0xFF << 23) | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Convert bf16 bits to f32.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to f16 bits.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let exp16 = exp - 127 + 15;
    if exp16 >= 31 {
        return sign | 0x7C00;
    }
    if exp16 <= 0 {
        return sign;
    }
    sign | ((exp16 as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert f32 to bf16 bits.
pub fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// Decode f16 byte slice to f32 vec.
pub fn decode_f16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// Decode bf16 byte slice to f32 vec.
pub fn decode_bf16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

/// Encode f32 slice to f16 bytes.
pub fn encode_f16(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

/// Encode f32 slice to bf16 bytes.
pub fn encode_bf16(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 100.0, 2.71] {
            let bits = f32_to_f16(v);
            let back = f16_to_f32(bits);
            assert!(
                (v - back).abs() < 0.01 * v.abs().max(0.001),
                "{v} → {bits} → {back}"
            );
        }
    }

    #[test]
    fn bf16_round_trip() {
        for &v in &[0.0f32, 1.0, -1.0, 0.5, 100.0, -42.0] {
            let bits = f32_to_bf16(v);
            let back = bf16_to_f32(bits);
            assert!(
                (v - back).abs() < 0.01 * v.abs().max(0.001),
                "{v} → {bits} → {back}"
            );
        }
    }

    #[test]
    fn f16_special_values() {
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0); // negative zero
        assert!(f16_to_f32(0x7C00).is_infinite()); // +inf
        assert!(f16_to_f32(0xFC00).is_infinite()); // -inf
        assert!(f16_to_f32(0x7E00).is_nan()); // nan
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bf16_known_values() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
    }

    #[test]
    fn f16_encode_decode_round_trip() {
        let data = vec![1.0f32, -2.0, 0.0, 0.5, 100.0];
        let encoded = encode_f16(&data);
        assert_eq!(encoded.len(), data.len() * 2);
        let decoded = decode_f16(&encoded);
        assert_eq!(decoded.len(), data.len());
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01 * orig.abs().max(0.001));
        }
    }

    #[test]
    fn bf16_encode_decode_round_trip() {
        let data = vec![1.0f32, -2.0, 0.0, 0.5, 100.0];
        let encoded = encode_bf16(&data);
        assert_eq!(encoded.len(), data.len() * 2);
        let decoded = decode_bf16(&encoded);
        assert_eq!(decoded.len(), data.len());
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.01 * orig.abs().max(0.001));
        }
    }

    #[test]
    fn f16_subnormal_round_trip() {
        // Exercise the `exp == 0 && mant != 0` denormal branch in
        // f16_to_f32 (lines 22-29). 0x0001 is the smallest positive
        // subnormal: 2^-24 ≈ 5.96e-8.
        let bits: u16 = 0x0001;
        let decoded = f16_to_f32(bits);
        assert!(decoded > 0.0, "smallest subnormal must decode positive");
        assert!(decoded < 1e-6, "subnormal magnitude should be tiny");
        // A larger subnormal should still be in (0, 2^-14).
        let mid_sub = f16_to_f32(0x03FF);
        assert!(mid_sub > 0.0 && mid_sub < 6.1e-5);
    }

    #[test]
    fn f16_overflow_to_infinity_and_nan_paths() {
        // f32 magnitude beyond f16 max overflows to ±inf — hits the
        // `exp16 >= 31` branch (line 54) in f32_to_f16.
        let big = 1.0e20f32;
        let bits_pos = f32_to_f16(big);
        assert_eq!(bits_pos, 0x7C00, "+inf encoding");
        let bits_neg = f32_to_f16(-big);
        assert_eq!(bits_neg, 0xFC00, "-inf encoding");
        // f32 NaN with non-zero mantissa hits the line-50 NaN branch
        // in f32_to_f16 (sign | 0x7C00 | 0x0200).
        let nan_bits = f32_to_f16(f32::NAN);
        let mantissa = nan_bits & 0x03FF;
        assert!(
            nan_bits & 0x7C00 == 0x7C00 && mantissa != 0,
            "NaN must round-trip with non-zero mantissa, got bits={nan_bits:#06x}"
        );
    }
}
