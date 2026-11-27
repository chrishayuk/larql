//! Prefill must agree with decode on a Q4_0 FFN.
//!
//! The two paths encode the FFN independently. Decode (`decode/encode_ffn`)
//! runs a Q4_0 down projection as `q4_f32_matvec` over the f32 activation.
//! Prefill (`ops/full_pipeline` → `stages/ffn`) quantises the activation to
//! Q8 and runs `q4_matvec`. For a single token on an empty cache both compute
//! the same layer, so their outputs may differ only by that Q8 rounding.
//!
//! Every other prefill test in this suite checks only that the output is
//! finite and non-zero. That is how prefill ran the Q4_0 down projection on
//! the Q8 copy of the FFN *input* (`hidden` wide) instead of the activation
//! (`inter` wide), from 9d340b68d until 5f4e9ed38, without any test noticing.
//! The output was finite whenever the bytes read past the staging buffer
//! happened to be benign. This test puts a reference on that path.

#[allow(unused_imports)]
use crate::common::*;

use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
use larql_compute::FfnType;

/// Largest |prefill - decode| over the output, relative to decode's largest
/// magnitude.
fn prefill_vs_decode(ffn_type: FfnType) -> f32 {
    let metal = larql_compute_metal::MetalBackend::new().expect(
        "Metal backend must build: the shader library failed to compile or no device exists",
    );
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = || {
        let mut l = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
        l.ffn_type = ffn_type;
        l
    };
    let x = synth_input(HIDDEN, 0.9);

    let prefill = metal
        .prefill_kquant(&[layer()], &x, HIDDEN, INTER, 1, false, 0.0)
        .expect("prefill_kquant returns Some");

    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let decode = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer()],
        &x,
        HIDDEN,
        INTER,
        Q_DIM,
        KV_DIM,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        10_000.0,
    );

    assert_eq!(prefill.len(), HIDDEN);
    assert_eq!(decode.len(), HIDDEN);
    assert!(
        decode.iter().all(|v| v.is_finite()),
        "decode output is not finite"
    );
    let scale = decode.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(scale > 0.0, "decode output is all-zero");
    let diff = prefill
        .iter()
        .zip(&decode)
        .map(|(p, d)| (p - d).abs())
        .fold(
            0.0f32,
            |a, b| if b.is_nan() { f32::INFINITY } else { a.max(b) },
        );
    eprintln!("{ffn_type:?}: max |prefill - decode| = {diff} (decode max |v| = {scale})");
    diff / scale
}

/// Q8-quantising an activation costs at most half a step (1/254 of the block
/// maximum) per element, so 1% of the output's magnitude is a generous bound.
/// The wrong operand misses by order 1 or more.
const TOLERANCE: f32 = 1e-2;

#[test]
fn prefill_matches_decode_on_a_q4_0_gated_ffn() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rel = prefill_vs_decode(FfnType::Gated);
    assert!(
        rel < TOLERANCE,
        "gated: prefill differs from decode by {rel} of its scale"
    );
}

#[test]
fn prefill_matches_decode_on_a_q4_0_standard_ffn() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let rel = prefill_vs_decode(FfnType::Standard);
    assert!(
        rel < TOLERANCE,
        "standard: prefill differs from decode by {rel} of its scale"
    );
}
