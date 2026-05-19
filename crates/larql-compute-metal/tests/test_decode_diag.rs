//! Cover `decode/diag.rs::log_decode_entry`.
//!
//! The diag block in `decode_token_with_moe_split_fn` is gated on a
//! process-static `CALL_COUNT < 3` (see `decode/mod.rs`). Inside any
//! single test binary, the function permanently early-returns after
//! three decode calls — so test ordering inside the main synthetic
//! decode binary (`test_metal_decode_synthetic.rs`) caps coverage at
//! the 85% mark documented in `coverage-policy.json`.
//!
//! This file is its own integration-test binary: it gets a fresh
//! process with `CALL_COUNT == 0`, runs `decode_token` with
//! `DECODE_DEBUG=1` as its very first call, and so executes the
//! diagnostic eprintln body that `test_metal_decode_synthetic.rs`
//! can't reach.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::{
    Activation, DecodeBackend, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight,
};

const HIDDEN: usize = 256;
const INTER: usize = 512;
const HEAD_DIM: usize = 64;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;

fn synth_weight_f32(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.001 + seed).sin() + 0.2 * ((i >> 8) as f32).cos()) * 0.3)
        .collect()
}

fn synth_input(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.01 + seed).sin()) * 0.5)
        .collect()
}

#[test]
fn decode_token_with_decode_debug_env_first_call_executes_log_body() {
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    let prior = std::env::var_os("DECODE_DEBUG");
    unsafe { std::env::set_var("DECODE_DEBUG", "1") };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    let layer = FullPipelineLayer {
        wq: QuantWeight {
            data: &wq,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: &wk,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: &wv,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wo: QuantWeight {
            data: &wo,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: &gate,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        up: QuantWeight {
            data: &up,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        down: QuantWeight {
            data: &down,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        input_norm: &norm_w,
        post_attn_norm: &norm_w,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 0.0,
        has_post_norms: false,
        activation: Activation::Silu,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        attn_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q_HEADS,
        num_kv_heads: NUM_KV_HEADS,
        rope_base: 10_000.0,
        rotary_dim: 0,
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        q_norm_weight: None,
        k_norm_weight: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        kv_shared_source: None,
        residual_multiplier: 1.0,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    };
    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer],
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
    assert_eq!(out.len(), HIDDEN);
    <larql_compute_metal::MetalBackend as DecodeBackend>::reset_kv_cache(&metal);

    match prior {
        Some(v) => unsafe { std::env::set_var("DECODE_DEBUG", v) },
        None => unsafe { std::env::remove_var("DECODE_DEBUG") },
    }
}
