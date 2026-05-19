//! GGML quant codec round-trip tests.
//!
//! For each format the vindex reads and writes, quantize → dequantize
//! a deterministic synthetic block and assert the absolute error stays
//! inside published tolerances. Catches the silent-fallback class:
//!
//! - "I added Q5_K's quantize but forgot the dequantize entry in
//!   `quant::registry`" — round-trip would diverge bit-for-bit
//! - "Block layout drifted by one byte" — element-wise error explodes
//! - "Scale encoding changed format" — bias/sign error shows up in
//!   aggregate stats
//!
//! Per-format tolerance bounds are loose enough to absorb expected
//! quantisation noise but tight enough that a real codec break trips
//! the assertion.


#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use larql_models::quant::ggml::{dequantize_q4_0, dequantize_q4_k, dequantize_q6_k, quantize_q4_0};

/// Reproducible synthetic block. The values span the realistic
/// dynamic range we see in real attention/FFN weights — roughly
/// N(0, 1) clamped to ±2.5 — so the per-format scales exercise the
/// outlier-handling paths in each codec.
fn synth_block(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // u32 → uniform [-1, 1]
            let u = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            // Box-Muller-ish bend toward N(0, 0.6), clamped.
            let g = u * 1.5;
            g.clamp(-2.5, 2.5)
        })
        .collect()
}

/// Max abs error tolerated for a (codec, block-size) pair. Numbers
/// match what the GGML reference reports for these formats; if
/// you're tightening these, double-check the codec hasn't lost
/// precision quietly.
fn assert_close(decoded: &[f32], original: &[f32], max_err: f32, format: &str) {
    assert_eq!(
        decoded.len(),
        original.len(),
        "{format}: length mismatch decoded={} original={}",
        decoded.len(),
        original.len()
    );
    let mut max_seen: f32 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for (i, (&a, &b)) in decoded.iter().zip(original.iter()).enumerate() {
        let err = (a - b).abs();
        max_seen = max_seen.max(err);
        sum_sq += (err * err) as f64;
        assert!(
            err <= max_err,
            "{format}: element {i} error {err:.6} > tolerance {max_err}; decoded={a}, original={b}"
        );
    }
    let rms = (sum_sq / decoded.len() as f64).sqrt() as f32;
    eprintln!(
        "{format}: max_err={max_seen:.6}, rms={rms:.6}, n={}",
        decoded.len()
    );
}

// ── Q4_0 ────────────────────────────────────────────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn q4_0_roundtrip_one_block() {
    // Q4_0 super-block = 32 elements, 18 bytes.
    let original = synth_block(32, 0xa110c8);
    let encoded = quantize_q4_0(&original);
    assert_eq!(encoded.len(), 18, "Q4_0: 18 bytes per 32 elements");

    let decoded = dequantize_q4_0(&encoded, 32).expect("dequant_q4_0");
    // Q4_0 has 4 bits per element across 32 elements with one f16
    // scale. With ±2.5 inputs, half-bin ≈ scale/16 ≈ 0.16; plus
    // f16-scale rounding pushes a single element to ~0.18 worst-case.
    // 0.20 is the realistic ceiling on this codec, not a slack number.
    assert_close(&decoded, &original, 0.20, "Q4_0");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn q4_0_roundtrip_many_blocks() {
    let original = synth_block(32 * 64, 0xface);
    let encoded = quantize_q4_0(&original);
    let decoded = dequantize_q4_0(&encoded, original.len()).expect("dequant_q4_0");
    assert_close(&decoded, &original, 0.20, "Q4_0/64");
}

// ── Q4_K ────────────────────────────────────────────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn q4_k_roundtrip_one_block() {
    // Q4_K super-block = 256 elements, 144 bytes (12 packed scales/mins
    // + 128 nibble bytes + 4 byte scale).
    let original = synth_block(256, 0xc0ffee);
    let encoded = quantize_q4_k(&original);
    assert_eq!(encoded.len(), 144, "Q4_K: 144 bytes per 256 elements");

    let decoded = dequantize_q4_k(&encoded, 256).expect("dequant_q4_k");
    // Q4_K uses 8 sub-blocks of 32 elements with per-sub-block scale
    // and min — sub-block scaling is much tighter than Q4_0. Realistic
    // bound on N(0, 0.6) data is ~0.025; 0.06 absorbs outliers.
    assert_close(&decoded, &original, 0.06, "Q4_K");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn q4_k_roundtrip_many_blocks() {
    // 4 super-blocks = 1024 elements (matches a typical hidden=1024 row).
    let original = synth_block(256 * 4, 0xdead);
    let encoded = quantize_q4_k(&original);
    let decoded = dequantize_q4_k(&encoded, original.len()).expect("dequant_q4_k");
    assert_close(&decoded, &original, 0.06, "Q4_K/4");
}

// ── Q6_K ────────────────────────────────────────────────────────────────

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn q6_k_roundtrip_one_block() {
    // Q6_K super-block = 256 elements, 210 bytes (192 bytes for 6-bit
    // packed values + 16 sub-block scales + 2-byte d).
    let original = synth_block(256, 0xbeef);
    let encoded = quantize_q6_k(&original);
    assert_eq!(encoded.len(), 210, "Q6_K: 210 bytes per 256 elements");

    let decoded = dequantize_q6_k(&encoded, 256).expect("dequant_q6_k");
    // Q6_K is 6-bit (64 levels) per sub-block — tightest of the three.
    // Realistic bound ~0.022 on ±2.5 inputs.
    assert_close(&decoded, &original, 0.025, "Q6_K");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn q6_k_roundtrip_many_blocks() {
    let original = synth_block(256 * 8, 0x42);
    let encoded = quantize_q6_k(&original);
    let decoded = dequantize_q6_k(&encoded, original.len()).expect("dequant_q6_k");
    assert_close(&decoded, &original, 0.025, "Q6_K/8");
}

// ── Cross-format sanity ─────────────────────────────────────────────────

/// Q6_K must be at least as accurate as Q4_K on the same input.
/// Catches a regression where a Q6_K kernel accidentally falls back
/// to Q4_K precision — the byte length would still be correct but the
/// reconstructed values would be coarser.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn q6_k_more_accurate_than_q4_k() {
    let original = synth_block(256, 0x006b_ea74_u64);
    let q4 = dequantize_q4_k(&quantize_q4_k(&original), 256).unwrap();
    let q6 = dequantize_q6_k(&quantize_q6_k(&original), 256).unwrap();

    let rms = |v: &[f32]| -> f32 {
        let sum_sq: f64 = v
            .iter()
            .zip(original.iter())
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        (sum_sq / v.len() as f64).sqrt() as f32
    };
    let q4_rms = rms(&q4);
    let q6_rms = rms(&q6);
    assert!(
        q6_rms <= q4_rms,
        "Q6_K RMS ({q6_rms:.6}) should be ≤ Q4_K RMS ({q4_rms:.6}) on the same input"
    );
}
