//! The oracle is the decoder, and the comparison is exact.
//!
//! A tolerance would hide the thing this kernel exists to guarantee. Its
//! whole claim is that executing the stored blocks is the SAME
//! arithmetic as decoding them and multiplying — so any difference at
//! all is a different program, not rounding, and the assertions say
//! `assert_eq!` rather than "within epsilon".

use super::{q8_0_gemv, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
use larql_models::quant::ggml::dequantize_q8_0;

/// Deterministic weights spanning several magnitudes, so blocks get
/// different scales and a single shared scale cannot pass by luck.
fn weights(n: usize, k: usize) -> Vec<f32> {
    (0..n * k)
        .map(|i| {
            let t = (i % 37) as f32 / 37.0 - 0.5;
            t * (1.0 + (i / k) as f32 * 0.25)
        })
        .collect()
}

/// Encode to Q8_0 the way ggml defines it: per 32 elements, `d =
/// amax/127`, codes are `round(w/d)`.
fn encode_q8_0(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() / Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES);
    for block in w.chunks(Q8_0_BLOCK_ELEMS) {
        let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax / 127.0;
        let dh = half_bits(d);
        // Round-trip the scale through f16 so the codes are chosen
        // against the scale that will actually be stored.
        let ds = larql_models::quant::half::f16_to_f32(dh);
        out.extend_from_slice(&dh.to_le_bytes());
        for &v in block {
            let q = if ds == 0.0 { 0.0 } else { (v / ds).round() };
            out.push((q.clamp(-127.0, 127.0) as i8) as u8);
        }
    }
    out
}

/// f32 -> f16 bits, round-to-nearest-even, for the normal range this
/// test exercises.
fn half_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut m = (mant >> 13) as u16;
    // Round to nearest even on the dropped bits.
    let dropped = mant & 0x1fff;
    if dropped > 0x1000 || (dropped == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m
}

fn decoded_gemv(w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|r| {
            let mut acc = 0.0f32;
            for c in 0..k {
                acc += w[r * k + c] * x[c];
            }
            acc
        })
        .collect()
}

fn activations(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 11) as f32 - 5.0) / 7.0).collect()
}

/// The oracle: decoding the pack and multiplying must give exactly what
/// the kernel gives. Not "within a tolerance" — the kernel is the same
/// arithmetic in the same association, so any difference is a different
/// program.
#[test]
fn the_kernel_equals_dequantise_then_multiply_bit_for_bit() {
    // 70 rows span three parallel work units, so the schedule is
    // exercised as well as the arithmetic.
    for (n, k) in [(1, 32), (3, 64), (8, 128), (5, 256), (70, 64)] {
        let w = weights(n, k);
        let packed = encode_q8_0(&w);
        let decoded = dequantize_q8_0(&packed, n * k).expect("decode");
        let x = activations(k);
        let got = q8_0_gemv(&packed, &x, n, k).expect("kernel");
        let want = decoded_gemv(&decoded, &x, n, k);
        assert_eq!(got, want, "[{n},{k}] kernel diverged from the decoder");
    }
}

/// The kernel must be reading the stored codes, not reconstructing the
/// original floats: quantisation is lossy, so an exact match against the
/// SOURCE weights would mean this is not testing a quantised path.
#[test]
fn the_result_is_lossy_against_the_source_weights() {
    let (n, k) = (4, 128);
    let w = weights(n, k);
    let packed = encode_q8_0(&w);
    let x = activations(k);
    let got = q8_0_gemv(&packed, &x, n, k).expect("kernel");
    let exact = decoded_gemv(&w, &x, n, k);
    assert_ne!(
        got, exact,
        "kernel matched the unquantised weights exactly — it is not reading codes"
    );
    // …but it must still be close, or it is reading them wrongly.
    for (g, e) in got.iter().zip(&exact) {
        assert!(
            (g - e).abs() < 0.05 * e.abs().max(1.0),
            "kernel {g} is not a Q8_0 approximation of {e}"
        );
    }
}

/// Geometry that does not describe the pack returns `None` rather than a
/// plausible vector read across a row boundary.
#[test]
fn a_geometry_that_does_not_describe_the_pack_is_refused() {
    let (n, k) = (2, 64);
    let packed = encode_q8_0(&weights(n, k));
    let x = activations(k);
    assert!(q8_0_gemv(&packed, &x, n, k).is_some(), "the control case");

    // k off the block grid.
    assert!(q8_0_gemv(&packed, &activations(48), n, 48).is_none());
    // x the wrong length for k.
    assert!(q8_0_gemv(&packed, &activations(32), n, k).is_none());
    // a stream short by one block.
    let short = &packed[..packed.len() - Q8_0_BLOCK_BYTES];
    assert!(q8_0_gemv(short, &x, n, k).is_none());
    // more rows than the stream holds.
    assert!(q8_0_gemv(&packed, &x, n + 1, k).is_none());
    // zero k.
    assert!(q8_0_gemv(&packed, &[], n, 0).is_none());
}

/// A block whose weights are all zero has a zero scale, which must not
/// become a NaN through a division that never happens.
#[test]
fn an_all_zero_block_contributes_nothing() {
    let (n, k) = (1, 64);
    let mut w = weights(n, k);
    for v in w.iter_mut().take(Q8_0_BLOCK_ELEMS) {
        *v = 0.0;
    }
    let packed = encode_q8_0(&w);
    let x = activations(k);
    let got = q8_0_gemv(&packed, &x, n, k).expect("kernel");
    assert!(got[0].is_finite(), "a zero-scale block produced {}", got[0]);
    let decoded = dequantize_q8_0(&packed, n * k).expect("decode");
    assert_eq!(got, decoded_gemv(&decoded, &x, n, k));
}

/// One block per scale, with codes covering the full range. `half_bits` above
/// flushes subnormals, so these are written directly.
fn blocks_with_scales(scales: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(scales.len() * Q8_0_BLOCK_BYTES);
    for (b, d) in scales.iter().enumerate() {
        out.extend_from_slice(&d.to_le_bytes());
        for j in 0..Q8_0_BLOCK_ELEMS {
            let code = -127 + ((b * 13 + j * 8) % 255) as i32;
            out.push((code as i8) as u8);
        }
    }
    out
}

/// f16 subnormals, from the smallest up to and across the normal boundary, in
/// both signs. An encoder that rounds to nearest stores these for small blocks.
const SUBNORMAL_SCALES: [u16; 8] = [
    0x0001, 0x0002, 0x00e3, 0x03fc, 0x03ff, 0x0400, 0x8001, 0x83ff,
];

/// Below the normal range an f16 is `m * 2^-24`, and the smallest normal, `0x0400`,
/// continues the same formula with `m = 1024`. Each is normal in f32, so the decode
/// has an exact value that needs no f16 code to state.
#[test]
fn a_subnormal_scale_decodes_to_its_exact_value() {
    let packed = blocks_with_scales(&SUBNORMAL_SCALES);
    let decoded =
        dequantize_q8_0(&packed, SUBNORMAL_SCALES.len() * Q8_0_BLOCK_ELEMS).expect("decode");
    for (b, d) in SUBNORMAL_SCALES.iter().enumerate() {
        let magnitude = f64::from(d & 0x7fff) * 2f64.powi(-24);
        let scale = if d & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        };
        for j in 0..Q8_0_BLOCK_ELEMS {
            let code = packed[b * Q8_0_BLOCK_BYTES + 2 + j] as i8;
            let want = (f64::from(code) * scale) as f32;
            assert_eq!(
                decoded[b * Q8_0_BLOCK_ELEMS + j].to_bits(),
                want.to_bits(),
                "scale {d:#06x} element {j}"
            );
        }
    }
}

/// The kernel on subnormal scales: exactly the decoder's arithmetic, and not the
/// zeros a flushed scale would give.
#[test]
fn the_kernel_executes_subnormal_scales_exactly() {
    let k = SUBNORMAL_SCALES.len() * Q8_0_BLOCK_ELEMS;
    let packed = blocks_with_scales(&SUBNORMAL_SCALES);
    let decoded = dequantize_q8_0(&packed, k).expect("decode");
    let x = activations(k);
    let got = q8_0_gemv(&packed, &x, 1, k).expect("kernel");
    assert_eq!(got, decoded_gemv(&decoded, &x, 1, k));
    assert_ne!(got[0], 0.0, "a flushed scale would give exactly this");
    let flushed = blocks_with_scales(&[0x0000; 8]);
    assert_eq!(
        q8_0_gemv(&flushed, &x, 1, k).expect("kernel")[0],
        0.0,
        "the control"
    );
}

/// The encoder and the kernel together, on the input this path exists for: a
/// block too small for a normal f16 scale.
#[test]
fn a_block_with_a_subnormal_scale_survives_encoding_and_execution() {
    let k = Q8_0_BLOCK_ELEMS;
    let w: Vec<f32> = (0..k)
        .map(|i| if i == 0 { 0.0077 } else { 0.001 })
        .collect();
    let packed = larql_models::quant::ggml::quantize_q8_0(&w);
    assert_eq!(
        u16::from_le_bytes([packed[0], packed[1]]) & 0x7c00,
        0,
        "a subnormal scale"
    );
    let x = vec![1.0f32; k];
    let got = q8_0_gemv(&packed, &x, 1, k).expect("kernel")[0];
    let exact: f32 = w.iter().sum();
    // Rounding moves each element by at most half a step, so the sum by at most
    // 32 half-steps. A flushed scale would give 0 and miss this by far more.
    let half_step = 0.0077 / 127.0 / 2.0;
    assert!(
        (got - exact).abs() <= 32.0 * half_step,
        "{got} against {exact}"
    );
}
