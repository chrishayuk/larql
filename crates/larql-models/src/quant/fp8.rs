//! FP8 E4M3 ↔ f32 conversion.
//!
//! FP8 E4M3 per the OCP FP8 specification v1.0:
//! 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
//! Range ≈ ±448, min positive normal 2⁻⁶, min positive subnormal 2⁻⁹.
//! `0x7F` and `0xFF` are NaN; there is no Inf.
//!
//! Used by the LARQL FP4 vindex format (exp 26) as both the
//! per-sub-block scale format and the per-block scale format.

/// Convert one E4M3 byte to f32.
///
/// Uses a 256-entry precomputed lookup table for speed; the table is
/// materialised once at program start via `Lazy`.
#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    E4M3_TABLE.with(|t| t[byte as usize])
}

thread_local! {
    static E4M3_TABLE: [f32; 256] = build_e4m3_table();
}

fn build_e4m3_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for i in 0..256u32 {
        t[i as usize] = e4m3_bits_to_f32_compute(i as u8);
    }
    t
}

fn e4m3_bits_to_f32_compute(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    // NaN encoding: exp = 1111, mant = 111 (both signs).
    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }

    let mag = if exp == 0 {
        // Subnormal: value = mant / 8 × 2⁻⁶.
        (mant as f32) * (1.0 / 8.0) * (2.0_f32).powi(-6)
    } else {
        // Normal: value = (1 + mant/8) × 2^(exp - 7).
        let frac = 1.0 + (mant as f32) / 8.0;
        frac * (2.0_f32).powi(exp as i32 - 7)
    };

    if sign == 1 {
        -mag
    } else {
        mag
    }
}

/// Convert f32 to E4M3 byte with round-to-nearest-even.
///
/// Saturates to ±448 on overflow (no Inf in E4M3). NaN inputs produce
/// the canonical E4M3 NaN (`0x7F` for positive, `0xFF` for negative).
#[inline]
pub fn f32_to_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return if value.is_sign_negative() { 0xFF } else { 0x7F };
    }

    let sign_bit: u8 = if value.is_sign_negative() { 0x80 } else { 0x00 };
    let mag = value.abs();

    if mag == 0.0 {
        return sign_bit;
    }

    // E4M3 max (normal, exp=14, mant=6): (1 + 6/8) × 2^7 = 1.75 × 128 = 224?
    // Actually OCP spec: max = 448 = 1.75 × 256 (exp=15 would be reserved for
    // NaN in standard IEEE, but E4M3 uses exp=15,mant<7 as normals).
    // So max = (1 + 7/8) × 2^8 = 1.875 × 256 = 480? No — mantissa 111 combined
    // with exp 1111 is NaN, so max normal is mantissa 110, exp 1111 =
    // 1.75 × 256 = 448. Confirmed.
    const E4M3_MAX: f32 = 448.0;
    if mag >= E4M3_MAX {
        // Saturate. Max normal is 0x7E (+448) / 0xFE (-448).
        return sign_bit | 0x7E;
    }

    // Decompose mag = 2^e × (1 + m) for normal, or = 2^-6 × m/8 for subnormal.
    let bits = mag.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;

    if f32_exp < -9 {
        // Below E4M3's smallest subnormal — flush to zero.
        return sign_bit;
    }

    if f32_exp < -6 {
        // Subnormal in E4M3. Value = 2^-6 × (mant/8).
        // So mant/8 = mag × 2^6, i.e. mant = mag × 2^9.
        let scaled = mag * (2.0_f32).powi(9);
        let rounded = round_ties_to_even(scaled);
        let m = rounded.clamp(0.0, 7.0) as u32;
        return sign_bit | (m as u8);
    }

    // Normal in E4M3. exp_e4m3 = f32_exp + 7, mant_e4m3 = (f32_mantissa >> 20).
    // With round-to-nearest-even on the dropped bits.
    let e4m3_exp = (f32_exp + 7) as u32;
    if e4m3_exp > 15 {
        // Shouldn't happen because we saturated earlier, but guard.
        return sign_bit | 0x7E;
    }

    // f32 mantissa stored as 23 bits of fraction; E4M3 keeps 3 bits.
    // Shift right by 20, apply round-to-nearest-even on bits 19..0.
    let f32_mant_full = bits & 0x007F_FFFF;
    let keep = f32_mant_full >> 20; // 3 bits
    let rem = f32_mant_full & 0x000F_FFFF; // 20 bits
    let half = 0x0008_0000;
    let rounded_up = rem > half || (rem == half && (keep & 1) == 1);

    let (mut e, mut m) = (e4m3_exp, keep);
    if rounded_up {
        m += 1;
        if m == 8 {
            m = 0;
            e += 1;
        }
    }

    if e >= 15 && m >= 7 {
        // Would land in NaN; saturate to max normal instead.
        return sign_bit | 0x7E;
    }
    if e > 15 {
        return sign_bit | 0x7E;
    }

    sign_bit | ((e as u8) << 3) | (m as u8)
}

fn round_ties_to_even(x: f32) -> f32 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        // Exact half — round to even integer.
        if (r as i32) % 2 != 0 {
            r - r.signum()
        } else {
            r
        }
    } else {
        r
    }
}

/// Encode a slice of f32 values to E4M3 bytes.
pub fn encode_e4m3(data: &[f32]) -> Vec<u8> {
    data.iter().map(|&v| f32_to_e4m3(v)).collect()
}

/// Decode an E4M3 byte slice to f32.
pub fn decode_e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| e4m3_to_f32(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_canonical_values() {
        // Zero.
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x80).to_bits(), (-0.0f32).to_bits());

        // Smallest positive subnormal: 2^-9 = 1/512 ≈ 0.001953125.
        assert!((e4m3_to_f32(0x01) - 1.0 / 512.0).abs() < 1e-7);

        // Smallest positive normal: 2^-6 = 1/64.
        assert!((e4m3_to_f32(0x08) - 1.0 / 64.0).abs() < 1e-7);

        // Max normal: 1.75 × 2^8 = 448.
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0xFE), -448.0);

        // NaN.
        assert!(e4m3_to_f32(0x7F).is_nan());
        assert!(e4m3_to_f32(0xFF).is_nan());
    }

    #[test]
    fn e4m3_round_trip_representable() {
        // Every representable E4M3 value should round-trip exactly.
        for byte in 0..=255u8 {
            let f = e4m3_to_f32(byte);
            if f.is_nan() {
                continue;
            }
            let back = f32_to_e4m3(f);
            // ±0 ambiguity: both 0x00 and 0x80 map to 0.0.
            if f == 0.0 {
                assert!(back == 0x00 || back == 0x80, "zero roundtrip got {back:#x}");
                continue;
            }
            assert_eq!(back, byte, "roundtrip {byte:#x} → {f} → {back:#x}");
        }
    }

    #[test]
    fn e4m3_saturation() {
        // Values above max normal saturate rather than overflow.
        assert_eq!(f32_to_e4m3(1000.0), 0x7E);
        assert_eq!(f32_to_e4m3(-1000.0), 0xFE);
        assert_eq!(f32_to_e4m3(448.0), 0x7E);
        assert_eq!(f32_to_e4m3(-448.0), 0xFE);
    }

    #[test]
    fn e4m3_tiny_flush_to_zero() {
        assert_eq!(f32_to_e4m3(1e-10), 0x00);
        assert_eq!(f32_to_e4m3(-1e-10), 0x80);
    }

    #[test]
    fn e4m3_rounding_to_nearest() {
        // 1.0 is exactly representable.
        assert_eq!(f32_to_e4m3(1.0), 0x38); // exp=7, mant=0 → (1+0)×2^0 = 1
                                            // Between 1.0 and 1.125 (next representable): expect rounding.
        let midpoint = 1.0625; // halfway
        let b = f32_to_e4m3(midpoint);
        let f_back = e4m3_to_f32(b);
        // Round-to-nearest-even picks 1.0 (mantissa 0, even) over 1.125 (mantissa 1, odd).
        assert_eq!(f_back, 1.0);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    /// E4M3 has subnormals for exponent=0. These represent values
    /// `m/8 × 2⁻⁶` for m ∈ [0, 7], i.e. `{0, 2⁻⁹, 2·2⁻⁹, …, 7·2⁻⁹}`.
    #[test]
    fn e4m3_subnormal_sweep() {
        // All 7 non-zero subnormals should decode to m/8 × 2⁻⁶.
        for m in 1..=7u8 {
            let expected = (m as f32 / 8.0) * (2.0_f32).powi(-6);
            let decoded = e4m3_to_f32(m);
            assert!(
                (decoded - expected).abs() < 1e-12,
                "m={m}: expected {expected}, got {decoded}"
            );
        }
        // Negative subnormals mirror.
        for m in 1..=7u8 {
            let expected = -(m as f32 / 8.0) * (2.0_f32).powi(-6);
            let decoded = e4m3_to_f32(0x80 | m);
            assert!((decoded - expected).abs() < 1e-12);
        }
    }

    /// Boundary between subnormal and smallest normal: 0x07 is the
    /// largest subnormal, 0x08 is 2⁻⁶ (smallest normal). The gap here
    /// is smaller than subsequent gaps because subnormals are uniformly
    /// spaced while normals are exponentially spaced.
    #[test]
    fn e4m3_subnormal_normal_boundary() {
        let largest_subnormal = e4m3_to_f32(0x07);
        let smallest_normal = e4m3_to_f32(0x08);
        assert!(
            smallest_normal > largest_subnormal,
            "normal must be larger than largest subnormal"
        );
        // Gap between 0x07 and 0x08 is 2⁻⁹ (same step as subnormals).
        let gap = smallest_normal - largest_subnormal;
        let expected_gap = (2.0_f32).powi(-9);
        assert!((gap - expected_gap).abs() < 1e-12);
    }

    /// Values that would require rounding up past max normal (448)
    /// must saturate to max rather than produce NaN (which is a
    /// separate bit pattern).
    #[test]
    fn e4m3_saturates_short_of_nan() {
        // Just below 448.0.
        let b = f32_to_e4m3(448.0 - 1.0);
        assert_ne!(b, 0x7F, "must not be NaN");
        assert!(!e4m3_to_f32(b).is_nan());
        // Way above 448.0 — saturates to max normal (0x7E), not NaN.
        assert_eq!(f32_to_e4m3(1e20), 0x7E);
        assert_eq!(f32_to_e4m3(-1e20), 0xFE);
        assert!(!e4m3_to_f32(f32_to_e4m3(1e20)).is_nan());
    }

    /// `+Inf` / `-Inf` also saturate, not NaN.
    #[test]
    fn e4m3_infinity_saturates() {
        assert_eq!(f32_to_e4m3(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_e4m3(f32::NEG_INFINITY), 0xFE);
    }

    /// Negative NaN should map to a NaN pattern (0xFF), not a normal.
    #[test]
    fn e4m3_negative_nan_preserved() {
        let neg_nan = f32::from_bits(f32::NAN.to_bits() | 0x8000_0000);
        assert_eq!(f32_to_e4m3(neg_nan), 0xFF);
        assert!(e4m3_to_f32(0xFF).is_nan());
    }

    /// Bulk round-trip: a sweep over the f32 representable range
    /// intersecting E4M3's representable set. Within the per-value
    /// precision bound (roughly 2⁻³ × value), round-trip error should
    /// be modest.
    #[test]
    fn e4m3_bulk_representable_round_trip() {
        let values = [
            0.0, 0.01, 0.1, 0.5, 1.0, 2.5, 10.0, 100.0, 400.0, -0.1, -1.0, -100.0,
        ];
        for &v in &values {
            let back = e4m3_to_f32(f32_to_e4m3(v));
            let bound = v.abs().max(1.0 / 512.0) * 0.125; // 3-bit mantissa
            assert!(
                (v - back).abs() <= bound,
                "v={v}: back={back}, err={} > bound {bound}",
                (v - back).abs()
            );
        }
    }

    #[test]
    fn encode_decode_e4m3_slice_round_trip() {
        // Exercise the encode_e4m3 / decode_e4m3 convenience wrappers
        // (file lines 156-158 / 161-163) which the scalar tests above
        // bypass — they call the per-value helpers directly.
        let data = vec![0.0f32, 1.0, -1.0, 2.5, -100.0, 0.125];
        let bytes = encode_e4m3(&data);
        assert_eq!(bytes.len(), data.len(), "one byte per value");
        let decoded = decode_e4m3(&bytes);
        assert_eq!(decoded.len(), data.len());
        for (orig, dec) in data.iter().zip(decoded.iter()) {
            let bound = orig.abs().max(1.0 / 512.0) * 0.125;
            assert!(
                (orig - dec).abs() <= bound,
                "round-trip {orig} → {dec} exceeded E4M3 precision bound {bound}",
            );
        }
    }
}
