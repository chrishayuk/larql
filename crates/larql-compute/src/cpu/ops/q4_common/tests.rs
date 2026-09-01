use super::*;

use super::matvec_f32::dot_256_f32;
#[cfg(target_arch = "aarch64")]
use super::matvec_f32::{dot_256_f32_neon, dot_256_f32_scalar};

/// Reference implementation kept here as the correctness oracle for
/// the bit-manipulation `f16_to_f32`.  Mirrors the previous (slow)
/// version that used `2.0f32.powi(...)`.  The new fast path must
/// match this for all 65536 possible f16 inputs except canonical NaN
/// payload preservation (handled in the test).
fn f16_to_f32_powi_reference(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Exhaustive bit-exact parity for all 65536 f16 inputs.  The fast
/// bit-manipulation `f16_to_f32` must produce the same f32 bits as
/// the powi-based reference for every finite (non-NaN) input.  NaN
/// payloads differ by design (reference collapses to canonical NaN,
/// fast path preserves payload — both are valid IEEE NaNs and the
/// distinction is unobservable in Q4_K decode because real-world
/// Q4_K headers never contain NaNs).
#[test]
fn f16_to_f32_bit_exact_for_all_inputs() {
    let mut diffs = 0usize;
    for bits in 0u16..=u16::MAX {
        let new = f16_to_f32(bits);
        let old = f16_to_f32_powi_reference(bits);
        if new.is_nan() && old.is_nan() {
            continue; // both NaN — different payloads OK
        }
        if new.to_bits() != old.to_bits() {
            if diffs < 5 {
                eprintln!(
                    "diff at bits=0x{bits:04x}: new={} ({:#x}) old={} ({:#x})",
                    new,
                    new.to_bits(),
                    old,
                    old.to_bits()
                );
            }
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "{diffs} f16 inputs decode to different f32 bits");
}

// ── f16 subnormal regression battery (2026-06-12). The subnormal
// branch decoded 2× too large while the exhaustive test silently
// verified a test-local `f16_to_f32` that shadowed the production fn.
// Assertions below call through `super::` so a future shadow cannot
// re-mask the production path. ──

#[test]
fn f16_to_f32_subnormal_pinned_values() {
    // IEEE 754 half subnormals: value = mant × 2^-24 exactly.
    assert_eq!(
        super::f16_to_f32(0x0001),
        2f32.powi(-24),
        "smallest subnormal"
    );
    assert_eq!(
        super::f16_to_f32(0x03fe),
        1022.0 * 2f32.powi(-24),
        "the field case — the gemma3-4b L32 K-scale that exposed the 2× bug"
    );
    assert_eq!(
        super::f16_to_f32(0x03ff),
        1023.0 * 2f32.powi(-24),
        "largest subnormal"
    );
    assert_eq!(super::f16_to_f32(0x0400), 2f32.powi(-14), "smallest normal");
    assert_eq!(
        super::f16_to_f32(0x8001),
        -(2f32.powi(-24)),
        "negative subnormal"
    );
}

#[test]
fn f16_to_f32_strictly_monotonic_across_subnormal_boundary() {
    // The 2× bug made f16(0x03ff) ≈ 1.22e-4 > f16(0x0400) = 6.1e-5 — a
    // monotonicity violation at the subnormal/normal seam. Walk the
    // positive seam region and require strict increase.
    let mut prev = super::f16_to_f32(0x0000);
    for bits in 0x0001u16..=0x0410 {
        let v = super::f16_to_f32(bits);
        assert!(
            v > prev,
            "f16 decode must be strictly increasing: bits={bits:#06x} gives {v:e}, prev {prev:e}"
        );
        prev = v;
    }
}

/// Deterministic pseudo-random data at a chosen magnitude. Magnitude
/// ~4e-4 drives the per-super-block `d`/`dmin` f16 scales into the
/// subnormal range (< 2^-14), the regime the 2× bug corrupted.
fn seeded_data(n: usize, magnitude: f32, mut seed: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * magnitude
        })
        .collect()
}

/// True if any Q4_K super-block in `bytes` carries a subnormal f16
/// `d` or `dmin` (exp bits zero, mantissa nonzero).
fn q4k_has_subnormal_scale(bytes: &[u8]) -> bool {
    bytes.chunks_exact(144).any(|b| {
        let d = u16::from_le_bytes([b[0], b[1]]);
        let dmin = u16::from_le_bytes([b[2], b[3]]);
        let sub = |v: u16| (v >> 10) & 0x1F == 0 && (v & 0x3FF) != 0;
        sub(d) || sub(dmin)
    })
}

/// Cross-crate seam test: same bytes, q4_common decoder vs the
/// larql-models decoder (which backs the vindex registry and the
/// staged/dequant path). These disagreed on every subnormal-scale
/// block until 2026-06-12 — same bytes, silently different weights.
#[test]
fn q4k_decode_matches_models_reference_incl_subnormal_scales() {
    for (name, magnitude) in [("normal", 1.0f32), ("subnormal-scale", 4.0e-4)] {
        let data = seeded_data(1024, magnitude, 0xA11C1);
        let bytes = quantize_q4_k(&data);
        if magnitude < 1e-3 {
            assert!(
                q4k_has_subnormal_scale(&bytes),
                "fixture drift: {name} case no longer produces subnormal f16 scales"
            );
        }
        let ours = dequantize_q4_k(&bytes, 1024);
        let reference =
            larql_models::quant::ggml::dequantize_q4_k(&bytes, 1024).expect("models decode");
        for (i, (a, b)) in ours.iter().zip(reference.iter()).enumerate() {
            let tol = 1e-5 * a.abs().max(b.abs()).max(1e-30);
            assert!(
                (a - b).abs() <= tol,
                "{name}: decoders disagree at elem {i}: q4_common {a:e} vs models {b:e}"
            );
        }
    }
}

/// Q6_K twin — its `d` is also an f16 scale, and the int8 Q6K matvec
/// reads it through the shared (previously buggy) `f16_to_f32`.
/// Reference decode comes from larql-models (independent f16 impl).
#[test]
fn q6k_int8_matvec_matches_models_reference_incl_tiny_scales() {
    use crate::cpu::ops::q4k_q8k_dot::{
        q6k_q8k_matvec_into, quantize_x_to_q8k_into, Q8KActivation,
    };
    let (rows, cols) = (2usize, 256usize);
    for (name, magnitude) in [("normal", 1.0f32), ("tiny-scale", 4.0e-4)] {
        let data = seeded_data(rows * cols, magnitude, 0xA11C2);
        let bytes = quantize_q6_k(&data);
        let x = seeded_data(cols, 1.0, 0xA11C5);
        let reference =
            larql_models::quant::ggml::dequantize_q6_k(&bytes, rows * cols).expect("models decode");
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                reference[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            })
            .collect();
        let denom: f32 = expected.iter().map(|v| v.abs()).fold(1e-12, f32::max);
        let mut x_q8k = Q8KActivation::with_capacity(cols);
        quantize_x_to_q8k_into(&mut x_q8k, &x);
        let mut out = vec![0.0f32; rows];
        q6k_q8k_matvec_into(&mut out, &x_q8k, &bytes, rows, cols).expect("valid shape");
        for (r, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 2e-2 * denom,
                "{name}: Q6K int8 matvec row {r}: {got:e} vs models reference {want:e}"
            );
        }
    }
}

/// Both Q4_K matvec kernels against the dequant·dot reference on the
/// same bytes, including subnormal-scale blocks. Pre-fix, affected
/// blocks contributed 2× — far outside either tolerance.
#[test]
fn q4k_matvecs_match_dequant_dot_incl_subnormal_scales() {
    use crate::cpu::ops::q4k_q8k_dot::{
        q4k_q8k_matvec_into, quantize_x_to_q8k_into, Q8KActivation,
    };
    let (rows, cols) = (4usize, 256usize);
    for (name, magnitude) in [("normal", 1.0f32), ("subnormal-scale", 4.0e-4)] {
        let data = seeded_data(rows * cols, magnitude, 0xA11C3);
        let bytes = quantize_q4_k(&data);
        if magnitude < 1e-3 {
            assert!(q4k_has_subnormal_scale(&bytes), "fixture drift ({name})");
        }
        let x = seeded_data(cols, 1.0, 0xA11C4);
        let deq = dequantize_q4_k(&bytes, rows * cols);
        let expected: Vec<f32> = (0..rows)
            .map(|r| {
                deq[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            })
            .collect();
        let denom: f32 = expected.iter().map(|v| v.abs()).fold(1e-12, f32::max);

        // f32-activation kernel: decode-identical, tight tolerance.
        let mut out_f32 = vec![0.0f32; rows];
        q4k_matvec_into(&mut out_f32, &x, &bytes, rows, cols).expect("valid shape");
        for (r, (got, want)) in out_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * denom,
                "{name}: f32-act matvec row {r}: {got:e} vs {want:e}"
            );
        }

        // int8-activation kernel: Q8_K rounding allowed, 2× is not.
        let mut x_q8k = Q8KActivation::with_capacity(cols);
        quantize_x_to_q8k_into(&mut x_q8k, &x);
        let mut out_i8 = vec![0.0f32; rows];
        q4k_q8k_matvec_into(&mut out_i8, &x_q8k, &bytes, rows, cols).expect("valid shape");
        for (r, (got, want)) in out_i8.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 2e-2 * denom,
                "{name}: int8 matvec row {r}: {got:e} vs {want:e}"
            );
        }
    }
}

#[test]
fn q8_quantize_round_trip() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let (q8, scales) = quantize_to_q8(&x);
    assert_eq!(q8.len(), 64);
    assert_eq!(scales.len(), 2); // 64 / 32
    assert!(scales.iter().all(|&s| s >= 0.0));
}

#[test]
fn q8_zero_input() {
    let x = vec![0.0f32; 32];
    let (q8, scales) = quantize_to_q8(&x);
    assert!(q8.iter().all(|&v| v == 0));
    assert!(scales[0] == 0.0);
}

// ── quantize_q4_0 tests ──

#[test]
fn q4_output_size() {
    // 64 floats = 2 blocks of 32, each block → 18 bytes (2 f16 scale + 16 nibbles)
    let data = vec![1.0f32; 64];
    let q4 = quantize_q4_0(&data);
    assert_eq!(q4.len(), 2 * 18);

    let data = vec![1.0f32; 256];
    let q4 = quantize_q4_0(&data);
    assert_eq!(q4.len(), 8 * 18);
}

#[test]
fn q4_zero_input() {
    let data = vec![0.0f32; 32];
    let q4 = quantize_q4_0(&data);
    assert_eq!(q4.len(), 18);
    // Scale should be zero (f16 zero = 0x0000)
    assert_eq!(q4[0], 0);
    assert_eq!(q4[1], 0);
    // All nibbles should encode 8 (zero quantized = 0 + bias 8)
    for &b in &q4[2..18] {
        assert_eq!(b, 0x88, "zero input should quantize to bias value 0x88");
    }
}

#[test]
fn q4_round_trip_accuracy() {
    // Quantize then dequantize, check values are close
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
    let q4 = quantize_q4_0(&data);

    // Dequantize: read f16 scale, unpack nibbles, multiply
    let scale_bits = u16::from_le_bytes([q4[0], q4[1]]);
    let scale = f16_to_f32(scale_bits);

    // ggml planar layout: low nibbles are elements 0..16, high 16..32.
    let mut decoded = [0.0f32; 32];
    for j in 0..16 {
        let byte = q4[2 + j];
        decoded[j] = ((byte & 0x0F) as i32 - 8) as f32 * scale;
        decoded[j + 16] = ((byte >> 4) as i32 - 8) as f32 * scale;
    }

    // Check approximate reconstruction (Q4 is lossy, but should be close)
    let max_err: f32 = data
        .iter()
        .zip(decoded.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 2.0,
        "Q4 round-trip max error {max_err} exceeds 2.0"
    );
}

/// `q4k_matvec_into` must produce numerically identical output to
/// the reference `dequantize_q4_k(...) → matmul_vec(...)` path.  Same
/// f32 weights, same arithmetic — just decoded streaming.  We use a
/// designed Q4_K-quantised input where the round-trip error is
/// already inside the quantizer, so the matvec output should match
/// within float-rounding noise (1e-3 on small magnitudes).
#[test]
fn q4k_matvec_matches_dequant_then_matmul() {
    // 4 rows × 256 cols (one super-block per row).
    let rows = 4;
    let cols = 256;
    let n_elem = rows * cols;

    // Designed weights: gradient ramp so the per-sub-block scale/min
    // varies, exercises every code path in q4k_matvec_into.
    let weights: Vec<f32> = (0..n_elem)
        .map(|i| ((i as f32 / n_elem as f32) - 0.5) * 1.0)
        .collect();
    let q4k = quantize_q4_k(&weights);
    assert_eq!(q4k.len(), rows * 144);

    // Reference: dequantize → row-major sgemv (manual, so this test
    // doesn't reach into the moe::math BLAS path).
    let dequant = dequantize_q4_k(&q4k, n_elem);
    assert_eq!(dequant.len(), n_elem);

    let x: Vec<f32> = (0..cols).map(|j| (j as f32 * 0.01).sin()).collect();
    let mut reference = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += dequant[r * cols + c] * x[c];
        }
        reference[r] = acc;
    }

    let mut got = vec![0.0f32; rows];
    q4k_matvec_into(&mut got, &x, &q4k, rows, cols).expect("valid shape");

    let max_diff: f32 = reference
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    // Both paths use the same nibble + scale arithmetic — differ only
    // in summation order.  f32 fp accumulation reorders are bounded
    // by ~ulp(max_intermediate); for 256-element sums of ~1.0 magnitudes
    // that's well under 1e-3.
    assert!(
        max_diff < 1e-3,
        "q4k_matvec_into diverges from dequant→matmul reference: \
         max_diff={max_diff}, reference={reference:?}, got={got:?}"
    );
}

/// Multi-block path: cols = 2 × 256 forces the per-row inner loop to
/// iterate `n_blocks > 1`.  Catches off-by-one in row-stride arithmetic
/// (`row_bytes = n_blocks * 144`) that the single-block test wouldn't
/// notice.
#[test]
fn q4k_matvec_multi_block_matches_dequant() {
    let rows = 3;
    let cols = 512; // 2 super-blocks per row
    let n_elem = rows * cols;
    let weights: Vec<f32> = (0..n_elem).map(|i| (i as f32 * 0.003).cos()).collect();
    let q4k = quantize_q4_k(&weights);
    assert_eq!(q4k.len(), rows * 2 * 144);

    let dequant = dequantize_q4_k(&q4k, n_elem);
    let x: Vec<f32> = (0..cols)
        .map(|j| ((j as f32) * 0.013).sin() * 0.7)
        .collect();
    let mut reference = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..cols {
            reference[r] += dequant[r * cols + c] * x[c];
        }
    }
    let mut got = vec![0.0f32; rows];
    q4k_matvec_into(&mut got, &x, &q4k, rows, cols).expect("valid shape");
    let max_diff: f32 = reference
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max_diff < 5e-3, "multi-block diverged: max_diff={max_diff}");
}

/// Amortised matmul must match dequantise→matmul for multiple rows AND
/// multiple sequence positions across several super-blocks. Exercises
/// the row-stride, the per-seq accumulation, and the [rows,seq] →
/// [seq,rows] transpose.
#[test]
fn q4k_matmul_matches_dequant_then_matmul() {
    let rows = 5;
    let hidden = 512; // 2 super-blocks per row
    let seq = 4;
    let n_elem = rows * hidden;
    let weights: Vec<f32> = (0..n_elem)
        .map(|i| (i as f32 * 0.0007).sin() * 0.9)
        .collect();
    let q4k = quantize_q4_k(&weights);
    let dequant = dequantize_q4_k(&q4k, n_elem);

    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| (i as f32 * 0.013).cos() * 0.5)
        .collect();

    // Reference: dequant → row-major matmul, out[s, r].
    let mut reference = vec![0.0f32; seq * rows];
    for s in 0..seq {
        for r in 0..rows {
            let mut acc = 0.0f32;
            for k in 0..hidden {
                acc += dequant[r * hidden + k] * x[s * hidden + k];
            }
            reference[s * rows + r] = acc;
        }
    }

    let mut got = vec![0.0f32; seq * rows];
    q4k_matmul_into(&mut got, &x, &q4k, rows, hidden, seq);

    let max_diff: f32 = reference
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max_diff < 5e-3,
        "q4k_matmul diverged from dequant→matmul: max_diff={max_diff}"
    );
}

/// Each sequence row of the matmul must equal the single-vector
/// `q4k_matvec` for that activation. The two kernels share decode
/// arithmetic, so they must agree row-for-row — catches transpose /
/// offset bugs that a dequant reference could mask if both paths were
/// wrong the same way.
#[test]
fn q4k_matmul_rows_match_q4k_matvec() {
    let rows = 6;
    let hidden = 256;
    let seq = 3;
    let weights: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i as f32 * 0.002) - 1.0) * 0.3)
        .collect();
    let q4k = quantize_q4_k(&weights);
    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| (i as f32 * 0.017).sin())
        .collect();

    let mut mm = vec![0.0f32; seq * rows];
    q4k_matmul_into(&mut mm, &x, &q4k, rows, hidden, seq);

    for s in 0..seq {
        let mut mv = vec![0.0f32; rows];
        q4k_matvec_into(
            &mut mv,
            &x[s * hidden..(s + 1) * hidden],
            &q4k,
            rows,
            hidden,
        )
        .expect("valid shape");
        for r in 0..rows {
            let diff = (mm[s * rows + r] - mv[r]).abs();
            assert!(
                diff < 1e-4,
                "matmul row s={s} r={r} != matvec: {} vs {}",
                mm[s * rows + r],
                mv[r]
            );
        }
    }
}

/// Defensive shape guard: `hidden` not a multiple of 256 → zeroed
/// output (mirrors `q4k_matvec_into`).
#[test]
fn q4k_matmul_rejects_non_multiple_of_256() {
    let mut out = vec![1.0f32; 2 * 3]; // seq=2, rows=3, pre-filled to detect zeroing
    let x = vec![0.5f32; 2 * 100];
    let w = vec![0u8; 3 * 144];
    q4k_matmul_into(&mut out, &x, &w, 3, 100, 2);
    assert_eq!(out, vec![0.0f32; 6]);
}

/// `dot_256_f32` (NEON on aarch64) must match the scalar reference and a
/// plain sequential dot.
#[test]
fn dot_256_f32_matches_reference() {
    let wf: [f32; 256] = std::array::from_fn(|i| (i as f32 * 0.013).sin() * 0.5);
    let xs: Vec<f32> = (0..256).map(|i| (i as f32 * 0.021).cos() * 0.7).collect();
    let reference: f64 = (0..256).map(|i| (wf[i] * xs[i]) as f64).sum();
    let got = dot_256_f32(&wf, &xs);
    assert!(
        (got as f64 - reference).abs() < 1e-3,
        "dot_256_f32 diverged: {got} vs {reference}"
    );
    #[cfg(target_arch = "aarch64")]
    {
        let neon = unsafe { dot_256_f32_neon(&wf, &xs) };
        let scalar = dot_256_f32_scalar(&wf, &xs);
        assert!(
            (neon - scalar).abs() < 1e-4,
            "neon {neon} != scalar {scalar}"
        );
    }
}

/// Multi-chunk (rows > CHUNK_ROWS = 32, spans multiple parallel work
/// units) and seq = 1 — the two q4k_matmul paths the earlier tests missed.
#[test]
fn q4k_matmul_multi_chunk_and_seq1() {
    for (rows, seq) in [(40usize, 3usize), (4, 1)] {
        let hidden = 512;
        let n_elem = rows * hidden;
        let weights: Vec<f32> = (0..n_elem)
            .map(|i| (i as f32 * 0.0005).sin() * 0.6)
            .collect();
        let q4k = quantize_q4_k(&weights);
        let dequant = dequantize_q4_k(&q4k, n_elem);
        let x: Vec<f32> = (0..seq * hidden)
            .map(|i| (i as f32 * 0.011).cos() * 0.4)
            .collect();
        let mut reference = vec![0.0f32; seq * rows];
        for s in 0..seq {
            for r in 0..rows {
                let mut acc = 0.0f32;
                for k in 0..hidden {
                    acc += dequant[r * hidden + k] * x[s * hidden + k];
                }
                reference[s * rows + r] = acc;
            }
        }
        let mut got = vec![0.0f32; seq * rows];
        q4k_matmul_into(&mut got, &x, &q4k, rows, hidden, seq);
        let max: f32 = reference
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(max < 5e-3, "rows={rows} seq={seq} diverged: {max}");
    }
}

/// Direct `q6k_matmul_into` kernel parity vs dequantise → matmul.
#[test]
fn q6k_matmul_matches_dequant_then_matmul() {
    let rows = 5;
    let hidden = 512;
    let seq = 3;
    let n_elem = rows * hidden;
    let weights: Vec<f32> = (0..n_elem)
        .map(|i| (i as f32 * 0.0007).sin() * 0.5)
        .collect();
    let q6k = quantize_q6_k(&weights);
    let dq = crate::kquant_forward::dequant::dequantize_matrix(&q6k, "Q6_K", rows, hidden);
    let dq = dq.as_slice().expect("contiguous");
    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| (i as f32 * 0.009).cos() * 0.5)
        .collect();
    let mut reference = vec![0.0f32; seq * rows];
    for s in 0..seq {
        for r in 0..rows {
            let mut acc = 0.0f32;
            for k in 0..hidden {
                acc += dq[r * hidden + k] * x[s * hidden + k];
            }
            reference[s * rows + r] = acc;
        }
    }
    let mut got = vec![0.0f32; seq * rows];
    q6k_matmul_into(&mut got, &x, &q6k, rows, hidden, seq);
    let max: f32 = reference
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max < 5e-3, "q6k_matmul diverged: {max}");
}

/// A malformed `cols` (not a multiple of 256) is refused by name rather
/// than read past the buffer, and the output is left exactly as handed
/// in — the old contract zeroed it, which a caller could mistake for a
/// computed result.
#[test]
fn q4k_matvec_rejects_non_multiple_of_256() {
    let mut out = vec![1.0f32; 4]; // pre-fill to detect any write
    let x = vec![0.5f32; 100];
    let w = vec![0u8; 4 * 144];
    let err = q4k_matvec_into(&mut out, &x, &w, 4, 100).expect_err("cols off a super-block");
    assert_eq!(out, vec![1.0f32; 4], "refusal must not touch the output");
    assert_eq!(err.cols, 100);
    assert!(
        err.to_string().contains("whole number of super-blocks"),
        "{err}"
    );
}

/// `q4k_matvec_into`: zero dims are the empty product, and a short weight
/// slab is refused by name with the output untouched.
#[test]
fn q4k_matvec_zero_dims_are_empty_and_short_weights_are_refused() {
    let mut out = vec![1.0f32; 3];
    q4k_matvec_into(&mut out, &[], &[], 3, 0).expect("zero dims are the empty product");
    assert_eq!(out, vec![0.0f32; 3]);

    let mut out = vec![1.0f32; 2];
    let x = vec![0.5f32; 256];
    let short_w = vec![0u8; 144];
    let err =
        q4k_matvec_into(&mut out, &x, &short_w, 2, 256).expect_err("short slab must be refused");
    assert_eq!(out, vec![1.0f32; 2], "refusal must not touch the output");
    assert_eq!(err.kernel, "q4k_matvec_into");
    assert_eq!((err.weight_bytes, err.needed_bytes), (144, 288));

    // The dual kernel refuses the same way, naming the half that is short.
    let (mut a, mut b) = (vec![1.0f32; 2], vec![1.0f32; 2]);
    let full = vec![0u8; 288];
    let err = q4k_dual_matvec_into(&mut a, &mut b, &x, &full, &short_w, 2, 256)
        .expect_err("short b slab must be refused");
    assert_eq!((a, b), (vec![1.0f32; 2], vec![1.0f32; 2]));
    assert_eq!(err.kernel, "q4k_dual_matvec_into (b)");
}

#[test]
fn dequantize_q4k_rejects_misaligned_or_truncated_input() {
    assert!(dequantize_q4_k(&[0u8; 144], 255).is_empty());
    assert!(dequantize_q4_k(&[0u8; 143], 256).is_empty());
}

#[test]
#[should_panic(expected = "multiple of 32")]
fn q4_rejects_non_aligned() {
    let data = vec![1.0f32; 33];
    let _ = quantize_q4_0(&data);
}

#[test]
fn q4_matvec_uses_quantized_data() {
    // End-to-end: quantize a matrix, run matvec, verify nonzero output
    let hidden = 256;
    let rows = 64;
    let matrix: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let q4 = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let mut scores = vec![0.0f32; rows];
    unsafe {
        q4_0_matvec_c(
            q4.as_ptr(),
            q8_x.as_ptr(),
            q8_scales.as_ptr(),
            scores.as_mut_ptr(),
            rows,
            hidden,
        );
    }
    assert!(
        scores.iter().any(|&v| v.abs() > 0.01),
        "Q4 matvec should produce nonzero"
    );
}

/// Test alias — dispatches to the canonical module-scope implementation.
fn dequantize_q4_k_llama(data: &[u8], n_elements: usize) -> Vec<f32> {
    super::dequantize_q4_k(data, n_elements)
}

#[test]
fn q4_k_round_trip_is_gguf_format() {
    // One super-block of a smooth [-1, 1] ramp — the worst case for
    // block-level scales. Verifies (a) the output is the 144-byte
    // llama.cpp layout and (b) quantise+dequantise agree to within Q4
    // quantisation noise.
    let data: Vec<f32> = (0..256).map(|i| (i as f32 / 255.0) * 2.0 - 1.0).collect();
    let bytes = quantize_q4_k(&data);
    assert_eq!(
        bytes.len(),
        144,
        "Q4_K super-block must be 144 bytes (GGUF), got {}",
        bytes.len()
    );
    let decoded = dequantize_q4_k_llama(&bytes, 256);
    let max_err = data
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Q4 over a 2.0 range → nibble step ≈ 0.13; allow 2× for the
    // per-sub-block scale/min quantisation bias.
    assert!(
        max_err < 0.12,
        "Q4_K GGUF round-trip max error {max_err} > 0.12 — \
         packing likely drifted from llama.cpp's get_scale_min_k4"
    );
}

// ── quantize_q6_k tests ──

#[test]
fn q6_k_output_size() {
    let data = vec![0.5f32; 256];
    let q6k = quantize_q6_k(&data);
    assert_eq!(q6k.len(), 210, "Q6_K super-block must be 210 bytes");

    let data2 = vec![0.5f32; 512];
    let q6k2 = quantize_q6_k(&data2);
    assert_eq!(q6k2.len(), 420, "two Q6_K super-blocks must be 420 bytes");
}

#[test]
fn q6_k_round_trip_via_matvec() {
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let q6k = quantize_q6_k(&weights);
    assert_eq!(q6k.len(), rows * 210);
    let result = super::super::q6k_matvec::dispatch(&q6k, &x, rows, hidden);
    assert_eq!(result.len(), rows);
    assert!(
        result.iter().any(|v| v.abs() > 1e-4),
        "Q6_K matvec should produce nonzero output"
    );
}

// ── q4k_to_q4kf / quantize_q4_kf tests ──

#[test]
fn q4kf_output_size() {
    let data = vec![0.5f32; 256];
    let q4kf = quantize_q4_kf(&data);
    assert_eq!(q4kf.len(), 160, "Q4_KF super-block must be 160 bytes");
}

#[test]
fn q4k_to_q4kf_converts_format() {
    let hidden = 256usize;
    let rows = 2usize;
    let weights: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let q4k = quantize_q4_k(&weights);
    let q4kf = q4k_to_q4kf(&q4k, rows, hidden);
    // Q4_KF is 160 bytes per 256-element super-block vs Q4_K's 144 bytes
    assert_eq!(q4kf.len(), rows * 160);
    assert_eq!(q4k.len(), rows * 144);
}

#[test]
fn q4k_to_q4kf_multi_superblock_rows() {
    let hidden = 512usize;
    let rows = 3usize;
    let weights: Vec<f32> = (0..rows * hidden)
        .map(|i| (i as f32 * 0.004).cos() * 0.25)
        .collect();
    let q4k = quantize_q4_k(&weights);
    let q4kf = q4k_to_q4kf(&q4k, rows, hidden);

    assert_eq!(q4k.len(), rows * 2 * 144);
    assert_eq!(q4kf.len(), rows * 2 * 160);
    assert!(
        q4kf.iter().any(|v| *v != 0),
        "converted Q4_KF should retain nonzero scales or nibbles"
    );
}

// ── f32_to_f16 edge cases ──

#[test]
fn f32_to_f16_normal_round_trip() {
    // 1.0, -1.0, 0.5: all representable exactly in f16
    for &val in &[1.0f32, -1.0, 0.5, -0.5, 2.0] {
        let bits = super::f32_to_f16(val);
        let back = f16_to_f32(bits);
        assert!(
            (back - val).abs() < 1e-3,
            "round-trip failed for {val}: got {back}"
        );
    }
}

#[test]
fn f32_to_f16_infinity() {
    let inf_bits = super::f32_to_f16(f32::INFINITY);
    let back = f16_to_f32(inf_bits);
    assert!(
        back.is_infinite() && back > 0.0,
        "expected +inf, got {back}"
    );

    let neg_inf_bits = super::f32_to_f16(f32::NEG_INFINITY);
    let neg_back = f16_to_f32(neg_inf_bits);
    assert!(
        neg_back.is_infinite() && neg_back < 0.0,
        "expected -inf, got {neg_back}"
    );
}

#[test]
fn f32_to_f16_large_value_clamps_to_infinity() {
    // 1e30 is beyond f16 max (~65504) → should return f16 infinity
    let bits = super::f32_to_f16(1e30f32);
    let back = f16_to_f32(bits);
    assert!(
        back.is_infinite(),
        "1e30 → f16 should be infinity, got {back}"
    );
}

#[test]
fn f32_to_f16_subnormal_range() {
    // 1e-10 is below f16 normal range (min normal ≈ 6.1e-5) → subnormal or zero f16
    let bits = super::f32_to_f16(1e-10f32);
    let back = f16_to_f32(bits);
    // Should be small (subnormal or zero), not a normal f16 value
    assert!(
        back.abs() < 1e-4,
        "1e-10 → f16 back-conversion {back} should be very small"
    );
}

#[test]
fn f32_to_f16_denormal_f32_input() {
    // f32 denormal (exp == 0) → f32_to_f16 should return signed zero
    let denormal = f32::from_bits(1u32); // smallest positive f32 denormal
    let bits = super::f32_to_f16(denormal);
    // exp == 0 path returns sign as u16, which for positive is 0
    assert_eq!(bits, 0, "f32 denormal should encode as f16 zero");
}

#[test]
fn q4_k_round_trip_matches_larql_models_decoder() {
    // Cross-check against the authoritative decoder in larql-models.
    // Guards against silent drift between the quantizer here and the
    // dequantizer every caller actually uses (kquant_forward.rs, vindex
    // weight load, etc.). 3 super-blocks, a mix of positive/negative.
    let data: Vec<f32> = (0..256 * 3)
        .map(|i| ((i as f32 - 383.0) / 127.0).sin())
        .collect();
    let bytes = quantize_q4_k(&data);
    assert_eq!(bytes.len(), 144 * 3);

    let decoded =
        larql_models::quant::ggml::dequantize_q4_k(&bytes, 256 * 3).expect("dequantize_q4_k");
    assert_eq!(decoded.len(), 256 * 3);

    let max_err = data
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.15,
        "cross-crate Q4_K round-trip max error {max_err} > 0.15 — \
         quantize_q4_k in larql-compute disagrees with \
         larql_models::quant::ggml::dequantize_q4_k (PR #24 llama.cpp format)"
    );
}

#[test]
fn f32_to_f16_valid_f16_subnormal() {
    // 1e-7 maps to new_exp ≈ -9 → shift = 10 → total_shift = 23 < 24
    // so it encodes as a nonzero f16 subnormal rather than clamping to zero.
    let bits = super::f32_to_f16(1e-7f32);
    let back = f16_to_f32(bits);
    // Must be a small positive subnormal, not zero.
    assert!(
        back > 0.0,
        "1e-7 should encode as nonzero f16 subnormal, got {back}"
    );
    assert!(
        back < 1e-4,
        "1e-7 encoded as f16 subnormal should still be small, got {back}"
    );
}

#[test]
fn quantize_q4k_all_zero_covers_d_zero_branch() {
    // All-zero data → global_max_range = 0 → d = 0 branch; global_min = 0 → dmin = 0 branch.
    // Also exercises f16_to_f32(0) in the decoder (mant==0, sign==0 path).
    let data = vec![0.0f32; 256];
    let q4k = quantize_q4_k(&data);
    assert_eq!(q4k.len(), 144);
    // Decoding should also produce all zeros.
    let decoded = dequantize_q4_k_llama(&q4k, 256);
    assert!(
        decoded.iter().all(|&v| v == 0.0),
        "all-zero encode/decode should stay zero"
    );
}

#[test]
fn quantize_q4k_all_positive_covers_dmin_zero() {
    // All-positive data → global_min = 0 → dmin = 0 branch (no negative offset needed).
    let data = vec![1.0f32; 256];
    let q4k = quantize_q4_k(&data);
    assert_eq!(q4k.len(), 144);
    // dmin bytes should encode f16 zero.
    let dmin_bits = u16::from_le_bytes([q4k[2], q4k[3]]);
    assert_eq!(
        dmin_bits, 0,
        "all-positive data should produce dmin=0 (f16 zero)"
    );
}

#[test]
fn quantize_q6k_all_zero_covers_d_zero_branch() {
    // All-zero data → d = 0 branch; all sub-block scales = 0.
    let data = vec![0.0f32; 256];
    let q6k = quantize_q6_k(&data);
    assert_eq!(q6k.len(), 210);
    // f16 super-block scale at bytes [208..210] should be zero.
    let d_bits = u16::from_le_bytes([q6k[208], q6k[209]]);
    assert_eq!(d_bits, 0, "all-zero data should produce d=0 (f16 zero)");
}

#[test]
#[should_panic(expected = "multiple of 256")]
fn quantize_q6k_rejects_non_aligned() {
    let _ = quantize_q6_k(&vec![1.0f32; 255]);
}
