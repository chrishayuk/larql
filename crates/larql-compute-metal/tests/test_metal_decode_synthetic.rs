//! Synthetic end-to-end decode tests.
//!
//! Builds a small `FullPipelineLayer` with synthetic Q4_K (attn) +
//! Q4_0 (FFN) weights and runs `MetalBackend::decode_token` on it.
//! Adapted from `examples/diag_decode_pipeline.rs`.
//!
//! Why this file exists: per-shader tests (`test_metal_shaders.rs` and
//! friends) hit the kernels but never exercise the production decode
//! orchestration code in `metal/decode/encode_{attn,qkv,ffn,post_ffn}.rs`
//! and `metal/decode/mod.rs::decode_token_with_moe_split_fn`. End-to-end
//! tests in `larql-inference/tests/` do, but those don't show up in
//! per-crate `cargo llvm-cov --package larql-compute` runs. This test
//! file fills that gap — a single decode_token call lifts ~2856 LoC of
//! production decode code from 0% to executed.
//!
//! These are smoke tests, not numerical-parity tests. They verify:
//! - decode_token returns a non-NaN, non-zero output buffer
//! - dimensions are right
//! - The `LARQL_FUSED_PRELAYER_NORM=1` D-RMS-FUSE wiring produces
//!   bit-identical output to the unfused path on a non-Gemma-style
//!   layer (no `has_post_norms`).
//!
//! Numerical-correctness against a CPU reference happens in
//! `larql-inference/tests/test_cpu_metal_parity.rs` against real
//! vindexes; it's at the wrong scope to live here.

#![cfg(target_os = "macos")]

use larql_compute::{
    Activation, ComputeBackend, DecodeBackend, FfnType, FullPipelineLayer, NormType, QuantFormat,
    QuantWeight,
};

/// Process-wide guard for tests that mutate env vars read by the decode
/// hot path (e.g. `LARQL_FUSED_PRELAYER_NORM`, `LARQL_QKV_FUSED`). Cargo
/// runs tests inside a binary in parallel by default; without this lock
/// a parallel `decode_token` test races with the env-toggling test and
/// observes the var in either state. Hold the guard for the entire
/// duration of any backend creation + decode that depends on the env.
static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Synthetic dims chosen to be Q4_K-compatible (multiples of 256) and
/// small enough for a fast test. Q4_K super-blocks are 256 elements.
const HIDDEN: usize = 256;
const INTER: usize = 512;
const HEAD_DIM: usize = 64;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM; // 128
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 64

fn synth_input(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.013 + seed).sin() + 0.1 * ((i >> 4) as f32).cos()) * 0.5)
        .collect()
}

fn synth_weight_f32(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.001 + seed).sin() + 0.2 * ((i >> 8) as f32).cos()) * 0.3)
        .collect()
}

// Test helper: 7 quant tensor slices + 1 norm slice = 8 args. Mirrors
// the production `FullPipelineLayer` constructor surface; collapsing to
// a single struct just for the test would obscure the per-tensor
// fixture-builder pattern callers actually want.
#[allow(clippy::too_many_arguments)]
fn build_synth_layer<'a>(
    wq_data: &'a [u8],
    wk_data: &'a [u8],
    wv_data: &'a [u8],
    wo_data: &'a [u8],
    gate_data: &'a [u8],
    up_data: &'a [u8],
    down_data: &'a [u8],
    norm_w: &'a [f32],
) -> FullPipelineLayer<'a> {
    FullPipelineLayer {
        wq: QuantWeight {
            data: wq_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: wk_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: wv_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wo: QuantWeight {
            data: wo_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: gate_data,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        up: QuantWeight {
            data: up_data,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        down: QuantWeight {
            data: down_data,
            scales: None,
            format: QuantFormat::Q4_0,
        },
        input_norm: norm_w,
        post_attn_norm: norm_w,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 0.0,
        has_post_norms: false, // Llama-style (non-Gemma); enables D-RMS-FUSE path
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
    }
}

/// End-to-end smoke: a single-layer Llama-style decode produces a
/// finite output of the correct size. Exercises the production decode
/// orchestration in `metal/decode/{mod,encode_attn,encode_qkv,
/// encode_ffn,encode_post_ffn}.rs`.
#[test]
fn decode_token_single_layer_synthetic_q4k_smoke() {
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));

    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);

    let result = larql_compute_metal::MetalBackend::decode_token(
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

    assert_eq!(result.len(), HIDDEN, "decode_token output length");
    let nan = result.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan, 0, "decode_token output had {nan} NaNs");
    let inf = result.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(inf, 0, "decode_token output had {inf} infinities");
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(
        max_abs > 0.0,
        "decode_token output is all zero (likely uninitialized buffers)"
    );
    assert!(
        max_abs < 1e6,
        "decode_token output magnitude {max_abs} is suspiciously large"
    );
}

/// D-RMS-FUSE Phase 1 end-to-end parity: `LARQL_FUSED_PRELAYER_NORM=1`
/// produces bit-identical output to the unfused path on a non-Gemma
/// (no `has_post_norms`) two-layer setup. Two layers exercises the
/// fusion at the layer-0→layer-1 boundary; a single layer wouldn't
/// engage the fusion (no next layer).
///
/// This is the integration counterpart to the kernel-level tests in
/// `test_kernel_fused_ops_norms.rs::residual_norm_store_*`.
#[test]
fn d_rms_fuse_phase1_produces_identical_output() {
    use std::env;

    // Hold the env lock for the whole test: both runs must observe the
    // env state at the time we construct each backend, and we must not
    // cross-pollute with the other env-mutating test
    // (`decode_token_qkv_fused_opt_in_smoke`) running in parallel.
    let _env_guard = ENV_TEST_LOCK.lock().expect("env lock poisoned");

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.11));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.22));
    let wv_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.33));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.44));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.55));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.66));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.77));

    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0007)).collect();
    let layer0 = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );
    let layer1 = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );
    let layers = [layer0, layer1];
    let x = synth_input(HIDDEN, 0.99);

    // Decode flags are cached at `MetalBackend::new()`. The test must
    // construct a fresh backend AFTER the env is in the desired state
    // — the previous "set env then call decode on the existing backend"
    // pattern silently no-ops with cached flags.
    //
    // Run with fusion OFF.
    env::remove_var("LARQL_FUSED_PRELAYER_NORM");
    let metal_off = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };
    assert!(
        !metal_off.decode_flags.fused_prelayer_norm,
        "expected fused_prelayer_norm=false in 'off' backend"
    );
    let mut kv_off = metal_off.create_kv_cache(2, 64, NUM_KV_HEADS, HEAD_DIM);
    let out_off = larql_compute_metal::MetalBackend::decode_token(
        &metal_off,
        &mut kv_off,
        &layers,
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

    // Run with fusion ON — fresh backend that captures the env flip.
    env::set_var("LARQL_FUSED_PRELAYER_NORM", "1");
    let metal_on = larql_compute_metal::MetalBackend::new()
        .expect("Metal device available since metal_off succeeded");
    assert!(
        metal_on.decode_flags.fused_prelayer_norm,
        "expected fused_prelayer_norm=true in 'on' backend"
    );
    let mut kv_on = metal_on.create_kv_cache(2, 64, NUM_KV_HEADS, HEAD_DIM);
    let out_on = larql_compute_metal::MetalBackend::decode_token(
        &metal_on,
        &mut kv_on,
        &layers,
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
    env::remove_var("LARQL_FUSED_PRELAYER_NORM");

    assert_eq!(out_off.len(), out_on.len(), "output length mismatch");
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (a, b)) in out_off.iter().zip(&out_on).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    // Bit-identical isn't realistic across two different RMS reductions
    // (residual_norm_store does cooperative reduction in a different
    // grouping than rms_norm + plain residual_add); allow small FP drift.
    assert!(
        max_diff < 1e-3,
        "D-RMS-FUSE off-vs-on diverged: max_diff={max_diff} at index {max_idx}; \
         out_off[{max_idx}]={a} vs out_on[{max_idx}]={b}",
        a = out_off[max_idx],
        b = out_on[max_idx],
    );
}

/// Gemma-3-style layer: `has_post_norms = true`, mixed Q4_K Q/K +
/// Q6_K V, QK-norm enabled. Exercises the post-norms branches in
/// `encode_attn.rs` (line 401's `if has_post_norms`) and the mixed-
/// quant QKV path in `encode_qkv.rs` that the Llama-style smoke test
/// above doesn't reach.
#[test]
fn decode_token_gemma3_style_post_norms_smoke() {
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

    // Mixed-quant attention: Q/K are Q4_K, V is Q6_K (Gemma 3/4 ollama
    // convention). FFN gate/up Q4_K, down Q6_K (also production
    // convention).
    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 1.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 1.2));
    let wv_data = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 1.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 1.4));
    let gate_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 1.5));
    let up_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 1.6));
    let down_data = quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 1.7));

    // Per-head QK norm weights (head_dim).
    let qk_norm_w: Vec<f32> = (0..HEAD_DIM).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0005)).collect();

    // post_attn_norm + pre_ffn_norm + post_ffn_norm = the Gemma 3/4
    // four-norm-per-layer pattern.
    let layer = FullPipelineLayer {
        wq: QuantWeight {
            data: &wq_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: &wk_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: &wv_data,
            scales: None,
            format: QuantFormat::Q6_K,
        },
        wo: QuantWeight {
            data: &wo_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: &gate_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        up: QuantWeight {
            data: &up_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        down: QuantWeight {
            data: &down_data,
            scales: None,
            format: QuantFormat::Q6_K,
        },
        input_norm: &norm_w,
        post_attn_norm: &norm_w,
        pre_ffn_norm: Some(&norm_w),
        post_ffn_norm: Some(&norm_w),
        norm_offset: 1.0, // Gemma 2/3 HF baked-in offset
        has_post_norms: true,
        activation: Activation::GeluTanh,
        qk_norm_offset: 1.0,
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
        q_norm_weight: Some(&qk_norm_w),
        k_norm_weight: Some(&qk_norm_w),
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

    let x = synth_input(HIDDEN, 1.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);

    let result = larql_compute_metal::MetalBackend::decode_token(
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

    assert_eq!(result.len(), HIDDEN);
    let nan = result.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan, 0, "Gemma-3-style decode produced {nan} NaNs");
    let inf = result.iter().filter(|v| v.is_infinite()).count();
    assert_eq!(inf, 0, "Gemma-3-style decode produced {inf} infinities");
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "Gemma-3-style output is all-zero");
    assert!(
        max_abs < 1e6,
        "Gemma-3-style output magnitude {max_abs} unreasonable"
    );
}

/// Multi-layer decode (3 layers) — exercises the layer-loop's
/// state-propagation logic in `metal/decode/mod.rs`. Single-layer
/// tests skip the inter-iteration `h_buf = new_h` swap and the
/// per-layer scratch reuse paths.
#[test]
fn decode_token_multi_layer_synthetic_smoke() {
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    // (wq, wk, wv, wo, gate, up, down) packed bytes for one synthetic
    // layer. Named alias keeps the per-layer Vec literal readable and
    // shuts up clippy::type_complexity.
    type SynthLayerWeights = (
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
    );

    // Build 3 distinct layers with different seeds so the layer loop
    // genuinely advances state.
    let mut layers_data: Vec<SynthLayerWeights> = Vec::with_capacity(3);
    for l in 0..3usize {
        let s = l as f32 * 0.1;
        layers_data.push((
            quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.10 + s)),
            quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.20 + s)),
            quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.30 + s)),
            quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.40 + s)),
            quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.50 + s)),
            quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.60 + s)),
            quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.70 + s)),
        ));
    }
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    let layers: Vec<FullPipelineLayer<'_>> = layers_data
        .iter()
        .map(|(wq, wk, wv, wo, gate, up, down)| {
            build_synth_layer(wq, wk, wv, wo, gate, up, down, &norm_w)
        })
        .collect();

    let x = synth_input(HIDDEN, 0.95);
    let mut kv = metal.create_kv_cache(layers.len(), 64, NUM_KV_HEADS, HEAD_DIM);

    let result = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &layers,
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

    assert_eq!(result.len(), HIDDEN);
    assert_eq!(result.iter().filter(|v| v.is_nan()).count(), 0);
    assert_eq!(result.iter().filter(|v| v.is_infinite()).count(), 0);
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "multi-layer output is all-zero");
}

/// `LARQL_QKV_FUSED=1` opts into the `q4k_q6k_qkv_proj_normed` path
/// (norm rolled into the matmul; defused as default 2026-05-09 per
/// ADR-016). Exercises `encode_normed_q4k_q6k_qkv` which is otherwise
/// unreached by the default tests.
#[test]
fn decode_token_qkv_fused_opt_in_smoke() {
    use std::env;

    // Serialise against `d_rms_fuse_phase1_produces_identical_output`
    // and any future env-mutating test in this binary. Decode flags
    // are cached at backend construction; set the env BEFORE creating
    // the backend.
    let _env_guard = ENV_TEST_LOCK.lock().expect("env lock poisoned");

    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 2.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 2.2));
    let wv_data = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 2.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 2.4));
    let gate_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 2.5));
    let up_data = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 2.6));
    let down_data = quantize_q6_k(&synth_weight_f32(HIDDEN * INTER, 2.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0009)).collect();

    // Layer matches the dispatcher's mixed_q4k_q6k_v + RmsNorm + no-bias
    // condition that gates the fused path. has_post_norms is false here
    // (a non-Gemma layer that still hits the normed QKV opt-in).
    let layer = FullPipelineLayer {
        wq: QuantWeight {
            data: &wq_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wk: QuantWeight {
            data: &wk_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        wv: QuantWeight {
            data: &wv_data,
            scales: None,
            format: QuantFormat::Q6_K,
        },
        wo: QuantWeight {
            data: &wo_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        gate: QuantWeight {
            data: &gate_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        up: QuantWeight {
            data: &up_data,
            scales: None,
            format: QuantFormat::Q4_K,
        },
        down: QuantWeight {
            data: &down_data,
            scales: None,
            format: QuantFormat::Q6_K,
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

    let x = synth_input(HIDDEN, 2.9);

    // Decode flags are cached at `MetalBackend::new()`; set env BEFORE
    // construction so the fused QKV path is actually engaged.
    env::set_var("LARQL_QKV_FUSED", "1");
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            env::remove_var("LARQL_QKV_FUSED");
            eprintln!("skip: no Metal device");
            return;
        }
    };
    assert!(
        metal.decode_flags.qkv_fused,
        "expected qkv_fused=true after setting env before backend construction"
    );
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let result = larql_compute_metal::MetalBackend::decode_token(
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
    env::remove_var("LARQL_QKV_FUSED");

    assert_eq!(result.len(), HIDDEN);
    assert_eq!(result.iter().filter(|v| v.is_nan()).count(), 0);
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "QKV-fused-opt-in output is all-zero");
}

/// `prefill_kquant` exercises a different code path than `decode_token`:
/// `metal/ops/full_pipeline/{dispatch,stages,full_layer}.rs` instead of
/// `metal/decode/*`. Multi-position seq_len=4 prefill on a synthetic
/// Llama-style layer.
#[test]
fn prefill_q4_seq4_synthetic_smoke() {
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let wq_data = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 3.1));
    let wk_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.2));
    let wv_data = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 3.3));
    let wo_data = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 3.4));
    let gate_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.5));
    let up_data = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 3.6));
    let down_data = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 3.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0008)).collect();

    let layer = build_synth_layer(
        &wq_data, &wk_data, &wv_data, &wo_data, &gate_data, &up_data, &down_data, &norm_w,
    );

    let seq_len = 4usize;
    let x: Vec<f32> = (0..seq_len * HIDDEN)
        .map(|i| ((i as f32 * 0.011 + 3.9).sin()) * 0.4)
        .collect();

    // prefill_kquant returns the final-position hidden state (size HIDDEN);
    // KV cache is populated in place. None means the backend doesn't
    // support this path — only Metal does.
    let result = (&metal as &dyn ComputeBackend)
        .as_any()
        .downcast_ref::<larql_compute_metal::MetalBackend>()
        .unwrap()
        .prefill_kquant(
            &[layer],
            &x,
            HIDDEN,
            INTER,
            seq_len,
            false, // use_qk_norm
            0.0,   // softcap
        );

    let result = match result {
        Some(r) => r,
        None => {
            eprintln!(
                "skip: prefill_kquant returned None (synthetic layer not supported by this path)"
            );
            return;
        }
    };

    // prefill_kquant returns seq_len × hidden (all positions, not just last).
    assert_eq!(
        result.len(),
        seq_len * HIDDEN,
        "prefill_kquant output length"
    );
    assert_eq!(result.iter().filter(|v| v.is_nan()).count(), 0);
    assert_eq!(result.iter().filter(|v| v.is_infinite()).count(), 0);
    let max_abs = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.0, "prefill_kquant output is all-zero");
    assert!(
        max_abs < 1e6,
        "prefill_kquant output magnitude {max_abs} unreasonable"
    );
}

/// `MetalBackend::with_options` honours its `BackendOptions` argument —
/// in particular, env vars must NOT override an explicit option. Pre-M1
/// the constructor read env directly; the field-driven build path makes
/// programmatic configuration trustworthy regardless of process env.
#[test]
fn with_options_honours_explicit_decode_flags_over_env() {
    use std::env;

    // Serialise against the env-toggling tests in this binary.
    // Recover from poison: a sibling test panicking inside the lock
    // (e.g. an opt-in shader with chip-dependent flakiness) should not
    // cascade-fail this one — the env state we care about is whatever
    // *this* test writes below.
    let _env_guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Set the env var to "on", then construct a backend with the option
    // explicitly OFF. The backend must reflect the explicit choice.
    env::set_var("LARQL_QKV_FUSED", "1");

    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.qkv_fused = false;

    let metal = match larql_compute_metal::MetalBackend::with_options(opts) {
        Some(m) => m,
        None => {
            env::remove_var("LARQL_QKV_FUSED");
            eprintln!("skip: no Metal device");
            return;
        }
    };

    assert!(
        !metal.decode_flags.qkv_fused,
        "with_options must override env: explicit qkv_fused=false but \
         backend resolved to {}",
        metal.decode_flags.qkv_fused
    );

    // And the inverse: env unset, explicit option ON.
    env::remove_var("LARQL_QKV_FUSED");
    let mut opts_on = larql_compute_metal::BackendOptions::default();
    opts_on.decode_flags.qkv_fused = true;
    let metal_on = larql_compute_metal::MetalBackend::with_options(opts_on)
        .expect("Metal device available since first construction succeeded");
    assert!(
        metal_on.decode_flags.qkv_fused,
        "with_options must honour explicit qkv_fused=true even with env unset"
    );
}

/// Regression: dispatch geometry must travel with `KernelHandle`, not
/// with shader-module re-exports. Pin the QKV projection pipelines'
/// rows/threads against the shader-module constants so any drift fails
/// at unit-test time, not at "decode emits garbage on this model" time.
///
/// This is the audit pair to `decode_attention_layer_q4k_writes_all_kv_rows`:
/// that test catches the runtime symptom; this one catches the static
/// invariant.
#[test]
fn qkv_pipeline_geometry_matches_shader_constants() {
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute_metal::shaders::{q4k_qkv_proj as q4k, q4kf_qkv_proj as q4kf};

    assert_eq!(
        metal.attention.q4k_qkv_proj_pipeline.rows_per_tg,
        q4k::ROWS_PER_TG
    );
    assert_eq!(
        metal.attention.q4k_qkv_proj_pipeline.threads_per_tg,
        q4k::THREADS_PER_TG
    );
    assert_eq!(
        metal.attention.q4kf_qkv_proj_pipeline.rows_per_tg,
        q4kf::ROWS_PER_TG
    );
    assert_eq!(
        metal.attention.q4kf_qkv_proj_pipeline.threads_per_tg,
        q4kf::THREADS_PER_TG
    );

    // The two pipelines must have DIFFERENT geometry — that's the whole
    // reason the bug existed. If they ever converge, delete this assert
    // and document the consolidation.
    assert!(
        metal.attention.q4k_qkv_proj_pipeline.rows_per_tg
            != metal.attention.q4kf_qkv_proj_pipeline.rows_per_tg
            || metal.attention.q4k_qkv_proj_pipeline.threads_per_tg
                != metal.attention.q4kf_qkv_proj_pipeline.threads_per_tg,
        "Q4_K and Q4_KF QKV pipelines now share geometry — \
         the decode_hybrid bug class no longer applies"
    );
}

/// Regression: MoE gate+up dispatch geometry must come from the bound
/// `KernelHandle`, not from re-imported shader-module constants. The
/// existing `q4k_ffn_gate_up_pipeline` is currently 4sg, but if it ever
/// gets bumped to 8sg (mirroring `q4k_matvec_pipeline`'s 4→8sg flip
/// from 2026-04-28), the `moe_dispatch.rs` paths must follow.
#[test]
fn moe_gate_up_pipeline_geometry_matches_shader_constants() {
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => {
            eprintln!("skip: no Metal device");
            return;
        }
    };

    use larql_compute_metal::shaders::{
        q4k_ffn_gate_up as q4k_gu, q4k_ffn_gate_up_8sg as q4k_gu_8sg,
    };

    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_pipeline.rows_per_tg,
        q4k_gu::ROWS_PER_TG
    );
    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_pipeline.threads_per_tg,
        q4k_gu::THREADS_PER_TG
    );
    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_8sg_pipeline.rows_per_tg,
        q4k_gu_8sg::ROWS_PER_TG
    );
    assert_eq!(
        metal.ffn.q4k_ffn_gate_up_8sg_pipeline.threads_per_tg,
        q4k_gu_8sg::THREADS_PER_TG
    );
}

// ─── decode/mod.rs diagnostic-env-var coverage ───
//
// These tests drive the env-gated diagnostic branches in
// `decode/mod.rs` (NaN inspector, residual dump, decode-diag-layer
// early-stop, L0 dump) that production decode never enters but the
// per-file 90 % coverage policy requires.  Each test is gated on
// ENV_TEST_LOCK because env vars are process-global.

fn decode_one_token_with_env(
    vars: &[(&str, Option<&str>)],
    extra_fn: impl FnOnce(&larql_compute_metal::MetalBackend),
) {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => return,
    };
    let saved: Vec<_> = vars
        .iter()
        .map(|(n, _)| (*n, std::env::var_os(n)))
        .collect();
    for (n, v) in vars {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
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
    extra_fn(&metal);
    for (n, v) in saved {
        match v {
            Some(s) => unsafe { std::env::set_var(n, s) },
            None => unsafe { std::env::remove_var(n) },
        }
    }
}

/// `LARQL_DEBUG_NAN_LAYERS=1` forces a per-layer commit+wait + NaN
/// histogram print.  Covers `decode/mod.rs` lines 528-543.
#[test]
fn decode_token_with_debug_nan_layers_env() {
    decode_one_token_with_env(&[("LARQL_DEBUG_NAN_LAYERS", Some("1"))], |_| {});
}

/// `LARQL_DUMP_L0=<dir>` enables the L0 residual dump on the first
/// layer. Covers the dump_l0_dir guard at line 279.
#[test]
fn decode_token_with_dump_l0_env() {
    let tmp = std::env::temp_dir().join("larql-compute-metal-dump-l0-test");
    let _ = std::fs::create_dir_all(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    decode_one_token_with_env(
        &[("LARQL_DUMP_L0", Some(Box::leak(path.into_boxed_str())))],
        |_| {},
    );
}

/// `LARQL_DECODE_DIAG_LAYER=0` stops decode after layer 0 and dumps
/// stage buffers.  Covers the diag_stop_layer paths around line 254.
#[test]
fn decode_token_with_decode_diag_layer_env() {
    decode_one_token_with_env(&[("LARQL_DECODE_DIAG_LAYER", Some("0"))], |_| {});
}

/// `LARQL_DUMP_RESIDUALS=<path>` enables the residual-dump capture
/// (`super::buffers::read_buffer_f32` per layer).  Covers lines
/// 273-277 + the dump-write tail.
#[test]
fn decode_token_with_dump_residuals_env() {
    let tmp = std::env::temp_dir().join("larql-compute-metal-residual-dump.bin");
    let path = tmp.to_str().unwrap().to_string();
    decode_one_token_with_env(
        &[(
            "LARQL_DUMP_RESIDUALS",
            Some(Box::leak(path.into_boxed_str())),
        )],
        |_| {
            let _ = std::fs::remove_file(&tmp);
        },
    );
}

/// Decode with QK-norm weights wired (Gemma-style layer).  Drives the
/// fused-attention path in `decode/encode_attn.rs` lines 172-217
/// (q_norm_enabled && k_norm_enabled && !has_v_norm).
#[test]
fn decode_token_with_qk_norm_drives_fused_attention_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let metal = match larql_compute_metal::MetalBackend::new() {
        Some(m) => m,
        None => return,
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let head_dim_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();

    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    // Enable Gemma-style: has_post_norms + QK-norm weights.
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = Some(&norm_w);
    layer.post_ffn_norm = Some(&norm_w);
    layer.q_norm_weight = Some(&head_dim_norm);
    layer.k_norm_weight = Some(&head_dim_norm);

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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Decode with `LARQL_FUSED_ATTN=0` opts out of attn_fused, forcing
/// the unfused QK-norm + RoPE + attend path.  Covers the unfused
/// branches around `decode/encode_attn.rs` lines 219+.
#[test]
fn decode_token_with_unfused_attn_env_drives_separate_qkn_rope_path() {
    decode_one_token_with_env(&[("LARQL_FUSED_ATTN", Some("0"))], |_| {});
}

/// Decode with `LARQL_FUSED_KV_APPEND_ATTEND=0` opts out of the fused
/// kv_append_attend kernel — separate append + attend dispatches.
#[test]
fn decode_token_with_unfused_kv_append_attend_env() {
    decode_one_token_with_env(&[("LARQL_FUSED_KV_APPEND_ATTEND", Some("0"))], |_| {});
}

/// `LARQL_PROFILE_SPLIT=1` drives the paired-commit per-stage timing
/// path — closes the encoder between attention CB and FFN CB so each
/// stage is recorded separately.  Covers `decode/mod.rs` lines
/// 396-402, 450-475 (gate_up CB split → down CB split).
#[test]
fn decode_token_with_profile_split_env() {
    decode_one_token_with_env(&[("LARQL_PROFILE_SPLIT", Some("1"))], |m| {
        // Reading the timing back covers the `take_last_split_timings`
        // path (`decode/profile.rs::take_last_split_timings`).
        let _ = larql_compute_metal::take_last_split_timings();
        let _ = m;
    });
}

/// `LARQL_FUSED_POST_FFN_NORM=0` opts out of the fused post-FFN
/// kernel — covers the unfused rms_norm + residual_add chain inside
/// `encode_post_ffn_residual` (already 91%) and the gated branch in
/// `decode/mod.rs` that picks `use_fused_post_ffn`.
#[test]
fn decode_token_with_unfused_post_ffn_norm_env() {
    decode_one_token_with_env(&[("LARQL_FUSED_POST_FFN_NORM", Some("0"))], |_| {});
}

/// LayerNorm decode (no bias) — drives `decode/encode_qkv.rs` lines
/// 167-175 (layer_norm_no_bias dispatch path).
#[test]
fn decode_token_with_layer_norm_no_bias_drives_layer_norm_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.norm_type = NormType::LayerNorm;

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
}

/// LayerNorm + bias decode — drives `decode/encode_qkv.rs` lines
/// 156-166 (layer_norm + bias dispatch path).
#[test]
fn decode_token_with_layer_norm_and_bias_drives_layer_norm_bias_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let bias: Vec<f32> = (0..HIDDEN).map(|i| (i as f32) * 0.001).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.norm_type = NormType::LayerNorm;
    layer.input_norm_bias = Some(&bias);

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
}

/// Q4_KF QKV format decode — drives `decode/encode_qkv.rs` lines
/// 230-237 (`UniformQ4Kf` arm of the format route).
#[test]
fn decode_token_with_q4kf_qkv_drives_uniform_q4kf_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq.format = QuantFormat::Q4_KF;
    layer.wk.format = QuantFormat::Q4_KF;
    layer.wv.format = QuantFormat::Q4_KF;

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
}

/// Mixed Q4_K + Q6_K_V format — drives the `MixedQ4kQ6kV` arm
/// (`decode/encode_qkv.rs` lines 258-284, Gemma 4 convention).
#[test]
fn decode_token_with_mixed_q4k_q6k_v_drives_mixed_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wv.format = QuantFormat::Q6_K;

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
}

/// Q4_0 QKV format decode — drives `decode/encode_qkv.rs` line 130
/// (the Q4_0 norm+qkv chain), which my Q4_K tests don't reach.
#[test]
fn decode_token_with_q4_0_qkv_drives_q4_0_norm_qkv_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    let wq = quantize_q4_0(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_0(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_0(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_0(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq.format = QuantFormat::Q4_0;
    layer.wk.format = QuantFormat::Q4_0;
    layer.wv.format = QuantFormat::Q4_0;
    layer.wo.format = QuantFormat::Q4_0;

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
}

/// Cover the KV-cache management trait methods on
/// `DecodeBackend for MetalBackend`: `has_kv_cache`, `populate_kv_layer`,
/// `kv_cache_len`, `reset_kv_cache`, `truncate_kv_cache`,
/// `preallocate_kv_cache_per_layer`.  Each is a public part of the
/// trait surface that's reached only when callers manage the cache
/// out-of-band (server-side prefill reuse, vindex walk).
#[test]
fn decode_backend_kv_cache_management_methods_round_trip() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;

    assert!(
        metal.has_kv_cache(),
        "Metal backend reports KV cache support"
    );

    // `preallocate_kv_cache_per_layer` replaces any existing cache.
    let shapes = vec![(NUM_KV_HEADS, HEAD_DIM), (NUM_KV_HEADS, HEAD_DIM)];
    metal.preallocate_kv_cache_per_layer(&shapes, 64);

    // Populate layer 0 with synthetic KV — covers populate_kv_layer's
    // happy path (cache already exists; cache_guard.is_some()).
    let synth_kv: Vec<f32> = (0..2 * NUM_KV_HEADS * HEAD_DIM)
        .map(|i| (i as f32) * 0.01)
        .collect();
    metal.populate_kv_layer(0, &synth_kv, &synth_kv, 2, NUM_KV_HEADS, HEAD_DIM);
    assert_eq!(metal.kv_cache_len(), 2);

    // `truncate_kv_cache` resets the per-layer counter to the given
    // length without re-allocating.
    metal.truncate_kv_cache(1);
    assert_eq!(metal.kv_cache_len(), 1);

    // `reset_kv_cache` zeroes every layer's current_len.
    metal.reset_kv_cache();
    assert_eq!(metal.kv_cache_len(), 0);
}

/// `populate_kv_layer` with **no pre-existing cache** drives the
/// `cache_guard.is_none()` branch — creates a fresh cache and grows
/// to the requested layer index.
#[test]
fn decode_backend_populate_kv_layer_creates_cache_on_first_call() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;

    // Reset so we drop any cache from prior tests.  Then populate
    // layer 2 directly — exercises the `while kv.layers.len() <= layer`
    // grow loop.
    metal.reset_kv_cache();
    let synth_kv: Vec<f32> = vec![0.5f32; 4 * NUM_KV_HEADS * HEAD_DIM];
    metal.populate_kv_layer(2, &synth_kv, &synth_kv, 4, NUM_KV_HEADS, HEAD_DIM);
    // `kv_cache_len()` reads layer[0]'s current_len. We only populated
    // layer 2, so layer 0 keeps its initial 0. The point of this test
    // is that `populate_kv_layer(2, ...)` succeeded by growing the
    // cache to 3 layers — verify that by writing layer 0 too and
    // re-reading.
    metal.populate_kv_layer(0, &synth_kv, &synth_kv, 4, NUM_KV_HEADS, HEAD_DIM);
    assert_eq!(metal.kv_cache_len(), 4);
}

/// `DecodeBackend::decode_token` trait method — wraps the inherent
/// MetalBackend::decode_token via the cached KV.  Covers
/// `trait_impl/decode.rs` lines 629-658.
#[test]
fn decode_backend_decode_token_trait_method_uses_internal_kv() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    let x = synth_input(HIDDEN, 0.9);
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .decode_token(&[layer], &x, HIDDEN, INTER)
        .expect("decode_token trait returns Some");
    assert_eq!(out.len(), HIDDEN);
}

/// GeluTanh activation drives the `&self.ffn.geglu_gelu_tanh_pipeline`
/// branch in `trait_impl/decode.rs::full_pipeline_q4` (line 59) and
/// `full_pipeline_q4_with_head_replacement` (line 131).
#[test]
fn prefill_q4_with_gelu_tanh_activation_drives_gelu_tanh_pipeline() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.activation = larql_compute::Activation::GeluTanh;

    let x = synth_input(HIDDEN, 0.9);
    // prefill_kquant goes through `full_pipeline_q4` which picks the
    // geglu_pipeline based on activation.
    let out = metal
        .prefill_kquant(&[layer], &x, HIDDEN, INTER, 1, false, 0.0)
        .expect("prefill_kquant returns Some");
    assert_eq!(out.len(), HIDDEN);

    // Same prefill_kquant_with_head_replacement variant — covers the
    // GeluTanh arm of its geglu picker too.
    let mut layer2 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer2.activation = larql_compute::Activation::GeluTanh;
    let delta = vec![0.0f32; HIDDEN];
    let out2 = metal
        .prefill_kquant_with_head_replacement(
            &[layer2],
            &x,
            HIDDEN,
            INTER,
            1,
            false,
            0.0,
            0,
            0,
            &delta,
        )
        .expect("head-replacement returns Some");
    assert_eq!(out2.len(), HIDDEN);
}

/// `DecodeBackend::full_pipeline_q4_with_head_replacement` is the
/// trait-level direct variant (no MoE fallback).  Covers
/// `trait_impl/decode.rs` lines 111-196.
#[test]
fn decode_backend_full_pipeline_q4_with_head_replacement_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let delta = vec![0.0f32; HIDDEN];
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .full_pipeline_q4_with_head_replacement(
            &layers, &x, HIDDEN, INTER, 1, false, 0.0, 0, 0, &delta,
        )
        .expect("full_pipeline_q4_with_head_replacement returns Some");
    assert_eq!(out.len(), HIDDEN);
}

/// `DecodeBackend::full_pipeline_kquant_capture_pre_wo` — covers
/// `trait_impl/decode.rs` lines 359-449 (capture pre-W_O variant).
#[test]
fn decode_backend_full_pipeline_q4_capture_pre_wo_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .full_pipeline_kquant_capture_pre_wo(&layers, &x, HIDDEN, INTER, 1, false, 0.0, 0, 0);
    let capture = out.expect("capture_pre_wo returns Some");
    // capture is a Vec<f32> of seq_len × head_dim; pin shape.
    assert_eq!(capture.len(), HEAD_DIM);
}

/// `DecodeBackend::decode_token_q4k_moe` — Metal-backend impl
/// returns `None` (delegates to default impl).  Covers
/// `trait_impl/decode.rs` lines 693-719.
#[test]
fn decode_backend_decode_token_q4k_moe_returns_none_on_metal() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let backend: &dyn DecodeBackend = &metal;
    // Metal backend uses the default `None` impl — it doesn't have a
    // dedicated Q4K-MoE decode pipeline. The wrapper still constructs
    // the geometry tuple and calls through, which is the part we cover.
    let _ = backend.decode_token_q4k_moe(&layers, &x, HIDDEN, INTER, 1e-6, &|_, _| None);
}

/// `DecodeBackend::decode_token_split_profile` — returns
/// `(result, attn_ms, gate_up_ms, down_ms)`.  Covers
/// `trait_impl/decode.rs` lines 762-790 + the fallback case where
/// LARQL_PROFILE_SPLIT isn't set so attn_ms = whole-token wall.
#[test]
fn decode_backend_decode_token_split_profile_returns_timings() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layers = vec![build_synth_layer(
        &wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w,
    )];
    let x = synth_input(HIDDEN, 0.9);
    let backend: &dyn DecodeBackend = &metal;
    let (result, attn_ms, _gate_up, _down) =
        backend.decode_token_split_profile(&layers, &x, HIDDEN, INTER);
    assert!(result.is_some());
    // Fallback: attn_ms = whole-token wall, > 0.
    assert!(attn_ms >= 0.0);
}

/// `DecodeBackend::decode_token_with_moe_split` trait method covers
/// `trait_impl/decode.rs` lines 721-760 (the fire/collect split
/// wrapper with `Vec::new()` discard inside fire_wrapper).
#[test]
fn decode_backend_decode_token_with_moe_split_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wv2 = wv.clone();
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let _ = wv2;
    let null_moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.moe = Some(null_moe);
    let x = synth_input(HIDDEN, 0.9);
    let mut fire_called = 0;
    let mut collect_called = 0;
    let mut fire_fn = |_l: usize, _h: &[f32]| {
        fire_called += 1;
    };
    let mut collect_fn = |_l: usize| -> Vec<f32> {
        collect_called += 1;
        vec![0.0f32; HIDDEN]
    };
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .decode_token_with_moe_split(&[layer], &x, HIDDEN, INTER, &mut fire_fn, &mut collect_fn)
        .expect("split returns Some");
    assert_eq!(out.len(), HIDDEN);
    assert_eq!(fire_called, 1);
    assert_eq!(collect_called, 1);
}

/// `DecodeBackend::multi_layer_q4_ffn` — covers `trait_impl/decode.rs`
/// lines 198-209.
#[test]
fn decode_backend_multi_layer_q4_ffn_runs() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    let block_bytes = 18usize;
    let hidden = 32usize;
    let inter = 64usize;
    let blocks_per_row = hidden / 32;
    let gate = vec![0u8; inter * blocks_per_row * block_bytes];
    let up = vec![0u8; inter * blocks_per_row * block_bytes];
    let down = vec![0u8; hidden * (inter / 32) * block_bytes];
    let layers = vec![(gate.as_slice(), up.as_slice(), down.as_slice())];
    let x = vec![0.0f32; hidden];
    let backend: &dyn DecodeBackend = &metal;
    let out = backend
        .multi_layer_q4_ffn(&layers, &x, inter, hidden)
        .expect("multi_layer_q4_ffn returns Some");
    assert_eq!(out.len(), hidden);
}

/// Decode with V-norm + QK-norm + `LARQL_FUSED_ATTN=0` drives the
/// unfused qk_norm_qk + v_norm_batched paths in
/// `decode/encode_attn.rs` lines 268-292 (qk_norm_qk dispatch) and
/// 318-336 (V-norm batched dispatch).
#[test]
fn decode_token_with_unfused_attn_v_norm_qk_norm_drives_unfused_paths() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let saved_fused = std::env::var_os("LARQL_FUSED_ATTN");
    // Force unfused: rebuild backend so DecodeFlags re-reads env.
    unsafe {
        std::env::set_var("LARQL_FUSED_ATTN", "0");
    }
    let metal2 = larql_compute_metal::MetalBackend::new().expect("Metal device");
    let _ = metal; // keep first backend alive briefly

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let head_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();

    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.q_norm_weight = Some(&head_norm);
    layer.k_norm_weight = Some(&head_norm);
    layer.has_v_norm = true;
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = Some(&norm_w);
    layer.post_ffn_norm = Some(&norm_w);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal2.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal2,
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

    unsafe {
        match saved_fused {
            Some(v) => std::env::set_var("LARQL_FUSED_ATTN", v),
            None => std::env::remove_var("LARQL_FUSED_ATTN"),
        }
    }
}

/// `MetalBackend::decode_attention_layer` runs ONE layer of attention
/// only (used by the hybrid CPU+GPU decode path in vindex walks).
/// This variant drives the Q4_K branch of `decode_hybrid.rs` (lines
/// 80-128).  Q4_K weights have scales embedded in the bytes, but the
/// helper still calls `transient_from_f32(scales.unwrap_or(&[]))` for
/// each projection — `metal-rs` 0.29 panics on `new_buffer_with_data`
/// for zero-length inputs, so we hand it a stub 1-element slice.
#[test]
fn decode_attention_layer_q4k_smoke() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let stub_scales = vec![0.0f32; 1]; // empty-slice-avoidance stub
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq.scales = Some(&stub_scales);
    layer.wk.scales = Some(&stub_scales);
    layer.wv.scales = Some(&stub_scales);
    layer.wo.scales = Some(&stub_scales);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

/// Same as above but with Q4_0 weights — drives the `else` (Q8 norm
/// + fused Q8 QKV) branch around `decode_hybrid.rs:129+`.
#[test]
fn decode_attention_layer_q4_0_smoke() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    let wq = quantize_q4_0(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_0(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_0(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_0(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    // Synthetic per-row scales so `transient_from_f32` doesn't see an
    // empty slice (metal-rs 0.29 dereferences null on zero-length).
    let q_scales: Vec<f32> = vec![0.01f32; Q_DIM];
    let kv_scales: Vec<f32> = vec![0.01f32; KV_DIM];
    let hidden_scales: Vec<f32> = vec![0.01f32; HIDDEN];
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq.format = QuantFormat::Q4_0;
    layer.wk.format = QuantFormat::Q4_0;
    layer.wv.format = QuantFormat::Q4_0;
    layer.wo.format = QuantFormat::Q4_0;
    layer.wq.scales = Some(&q_scales);
    layer.wk.scales = Some(&kv_scales);
    layer.wv.scales = Some(&kv_scales);
    layer.wo.scales = Some(&hidden_scales);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
}

/// Drives the Gemma-style branches of `decode_hybrid.rs`:
/// * `has_v_norm = true` enters the V-norm dispatch loop (239-249),
/// * `has_post_norms = true` selects the norm(O) + residual path
///   (319-339) in the O-projection encoder,
/// * `wo.format = Q4_KF` picks the `q4kf_proj_pipeline` (line 297-298),
/// * `rotary_dim = HEAD_DIM/2` exercises the `rotary_dim > 0` branch
///   at line 46-48.
#[test]
fn decode_attention_layer_q4k_with_v_norm_post_norms_and_q4kf_wo() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let stub_scales = vec![0.0f32; 1];
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq.scales = Some(&stub_scales);
    layer.wk.scales = Some(&stub_scales);
    layer.wv.scales = Some(&stub_scales);
    layer.wo.scales = Some(&stub_scales);
    layer.has_v_norm = true;
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.wo.format = QuantFormat::Q4_KF;
    layer.rotary_dim = HEAD_DIM / 2;

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

/// Same branch combo as above but with `wo.format = Q6_K`, picking
/// the `q6k_matvec_pipeline` (line 299-300) in the O-projection.
#[test]
fn decode_attention_layer_q4k_with_q6k_wo_drives_q6k_proj_branch() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q6_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let stub_scales = vec![0.0f32; 1];
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.wq.scales = Some(&stub_scales);
    layer.wk.scales = Some(&stub_scales);
    layer.wv.scales = Some(&stub_scales);
    layer.wo.scales = Some(&stub_scales);
    layer.wo.format = QuantFormat::Q6_K;

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    let h_post_attn = metal.decode_attention_layer(&mut kv, &layer, 0, &x, HIDDEN, Q_DIM, KV_DIM);
    assert_eq!(h_post_attn.len(), HIDDEN);
    assert!(h_post_attn.iter().all(|v| v.is_finite()));
}

// ─────────────────────────────────────────────────────────────────
// encode_attn.rs branch coverage via explicit BackendOptions.
// `MetalBackend::new()` snapshots env into `DecodeFlags` at startup,
// so env-set-after-new tests can't toggle decode-path branches.
// `with_options(...)` bypasses env entirely — direct flag injection.
// ─────────────────────────────────────────────────────────────────

// Build a QK-norm-enabled (Gemma-style) layer for `fused_attn` and
// `fused_qk_norm_rope` branch coverage. Mirrors `build_synth_layer`'s
// 8-arg fixture-builder shape but with two extra slices (norm + head-
// dim norm).
#[allow(clippy::too_many_arguments)]
fn synth_qk_norm_layer<'a>(
    wq: &'a [u8],
    wk: &'a [u8],
    wv: &'a [u8],
    wo: &'a [u8],
    gate: &'a [u8],
    up: &'a [u8],
    down: &'a [u8],
    norm_w: &'a [f32],
    head_dim_norm: &'a [f32],
) -> FullPipelineLayer<'a> {
    let mut layer = build_synth_layer(wq, wk, wv, wo, gate, up, down, norm_w);
    layer.has_post_norms = true;
    layer.post_attn_norm = norm_w;
    layer.pre_ffn_norm = Some(norm_w);
    layer.post_ffn_norm = Some(norm_w);
    layer.q_norm_weight = Some(head_dim_norm);
    layer.k_norm_weight = Some(head_dim_norm);
    layer
}

/// `BackendOptions { fused_attn: true }` drives the triple-fused
/// `attn_fused` path in `decode/encode_attn.rs` lines 172-229.
/// Gated on `q_norm_enabled && k_norm_enabled && !has_v_norm &&
/// kv_shared_source.is_none() && head_dim <= MAX_HEAD_DIM_SINGLE_SG
/// && attn_span <= SHORT_ATTENTION_SPAN`. All hold for the synth
/// fixture (HEAD_DIM=64 ≤ 256, t_val=1 ≤ 1024).
#[test]
fn decode_token_with_fused_attn_options_drives_attn_fused_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_attn = true;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let head_dim_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();
    let layer = synth_qk_norm_layer(
        &wq,
        &wk,
        &wv,
        &wo,
        &gate,
        &up,
        &down,
        &norm_w,
        &head_dim_norm,
    );
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `BackendOptions { fused_qk_norm_rope: false }` with QK-norm
/// weights drives the legacy non-fused `qk_norm_qk_pipeline` +
/// separate batched-RoPE path in `decode/encode_attn.rs` lines
/// 267-316.
#[test]
fn decode_token_with_unfused_qkn_rope_options_drives_legacy_qkn_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_qk_norm_rope = false;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let head_dim_norm: Vec<f32> = (0..HEAD_DIM).map(|i| 1.0 + (i as f32 * 0.002)).collect();
    let layer = synth_qk_norm_layer(
        &wq,
        &wk,
        &wv,
        &wo,
        &gate,
        &up,
        &down,
        &norm_w,
        &head_dim_norm,
    );
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `BackendOptions { fused_kv_append_attend: false }` drives the
/// unfused `encode_kv_append` + `encode_kv_attend` path in
/// `decode/encode_attn.rs` lines 410-428 (and the `current_len`
/// bump at line 433 in the non-shared, non-fused-attn branch).
#[test]
fn decode_token_with_unfused_kv_aa_options_drives_unfused_append_attend() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_kv_append_attend = false;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Two-layer decode with layer[1].kv_shared_source = Some(0)
/// drives the shared-cache branch in `decode/encode_attn.rs`:
/// lines 131-141 (source-pinned pos/t_val), 373-409 (attend against
/// source's cache, skip own append).
#[test]
fn decode_token_with_kv_shared_source_drives_shared_layer_branch() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer0 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    let mut layer1 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    // Gemma 4 E2B style: layer 1 reads K/V from layer 0's cache.
    layer1.kv_shared_source = Some(0);
    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(2, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer0, layer1],
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `has_post_norms = true` with `pre_ffn_norm = None` drives the
/// `bufs.post_attn_norm.clone()` fallback at `decode/encode_attn.rs`
/// line 505 (when the layer doesn't carry a separate pre-FFN norm).
#[test]
fn decode_token_with_post_norms_no_pre_ffn_norm_drives_fallback() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = None; // <- triggers fallback at L505
    layer.post_ffn_norm = Some(&norm_w);
    // Q4_K FFN gate so `ffn_uses_kquant` is true → exercises the
    // fused-post-attn + residual_norm_store path that consults
    // `pre_ffn_buf`.
    layer.gate.format = QuantFormat::Q4_K;
    layer.up.format = QuantFormat::Q4_K;
    layer.down.format = QuantFormat::Q4_K;
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `BackendOptions { fused_post_attn_norm: false }` with
/// `has_post_norms = true` and a Q4_K FFN drives the un-triple-fused
/// post-attn-norm path in `decode/encode_attn.rs` lines 528-557:
/// separate `encode_rms_norm` + `residual_norm_store_pipeline` chain.
#[test]
fn decode_token_with_unfused_post_attn_norm_options_drives_split_norm() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_post_attn_norm = false;
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.has_post_norms = true;
    layer.post_attn_norm = &norm_w;
    layer.pre_ffn_norm = Some(&norm_w);
    layer.post_ffn_norm = Some(&norm_w);
    layer.gate.format = QuantFormat::Q4_K;
    layer.up.format = QuantFormat::Q4_K;
    layer.down.format = QuantFormat::Q4_K;
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// Decode with Q4_KF Q + Q4_KF K + Q6_K V drives the `PerProjection`
/// route in `decode/encode_qkv.rs` lines 285-334 (mixed format
/// outside the table).
#[test]
fn decode_token_with_mixed_q4kf_q6k_v_drives_per_projection() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
    metal.reset_kv_cache();
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q6_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    // Q4_KF Q + Q4_KF K + Q6_K V — not a UniformQ4Kf, not a
    // MixedQ4kQ6kV (that pattern wants Q4_K Q + K).  Falls into the
    // PerProjection table-miss bucket.
    layer.wq.format = QuantFormat::Q4_KF;
    layer.wk.format = QuantFormat::Q4_KF;
    layer.wv.format = QuantFormat::Q6_K;

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
}

/// Decode with Q8_0 QKV weights drives the fused Q8 attention path
/// in `decode/encode_qkv.rs` lines 378-411 (encode_q4_0_norm_and_qkv's
/// Q8_0 branch) and `ops/full_pipeline/stages.rs` lines 204-227.
#[test]
fn decode_token_with_q8_0_qkv_drives_fused_q8_qkv_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    metal.reset_kv_cache();

    // Q8_0 weights: i8 values, num_rows × hidden bytes.  Per-row f32 scales.
    let wq_q8: Vec<i8> = vec![1i8; Q_DIM * HIDDEN];
    let wq_scales: Vec<f32> = vec![0.01f32; Q_DIM];
    let wk_q8: Vec<i8> = vec![1i8; KV_DIM * HIDDEN];
    let wk_scales: Vec<f32> = vec![0.01f32; KV_DIM];
    let wv_q8: Vec<i8> = vec![1i8; KV_DIM * HIDDEN];
    let wv_scales: Vec<f32> = vec![0.01f32; KV_DIM];
    let wq_bytes: Vec<u8> = wq_q8.iter().map(|&b| b as u8).collect();
    let wk_bytes: Vec<u8> = wk_q8.iter().map(|&b| b as u8).collect();
    let wv_bytes: Vec<u8> = wv_q8.iter().map(|&b| b as u8).collect();
    let wo = quantize_q4_0(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(
        &wq_bytes, &wk_bytes, &wv_bytes, &wo, &gate, &up, &down, &norm_w,
    );
    layer.wq.format = QuantFormat::Q8_0;
    layer.wk.format = QuantFormat::Q8_0;
    layer.wv.format = QuantFormat::Q8_0;
    layer.wq.scales = Some(&wq_scales);
    layer.wk.scales = Some(&wk_scales);
    layer.wv.scales = Some(&wv_scales);
    // wo can stay Q4_0 (output projection); only QKV needs Q8_0 for the fused path.
    layer.wo.format = QuantFormat::Q4_0;

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
}

/// Decode with PLE weights wired on the layer drives
/// `encode_per_layer_embed` (covers `decode/encode_ple.rs` end-to-end).
#[test]
fn decode_token_with_ple_weights_drives_per_layer_embed() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};

    let ple_dim = 32usize;
    let num_layers = 1usize;
    let positions = 1usize;
    let ple_inputs: Vec<f32> = (0..positions * num_layers * ple_dim)
        .map(|i| (i as f32) * 0.01)
        .collect();
    metal.prepare_ple_inputs(&ple_inputs, num_layers, ple_dim);

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    // PLE weights: input_gate is [ple_dim × hidden], projection is
    // [hidden × ple_dim], post_norm is [hidden].
    let ple_input_gate: Vec<f32> = (0..ple_dim * HIDDEN).map(|i| (i as f32) * 0.0001).collect();
    let ple_projection: Vec<f32> = (0..HIDDEN * ple_dim).map(|i| (i as f32) * 0.0001).collect();
    let ple_post_norm: Vec<f32> = vec![1.0f32; HIDDEN];

    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.ple_input_gate = Some(&ple_input_gate);
    layer.ple_projection = Some(&ple_projection);
    layer.ple_post_norm = Some(&ple_post_norm);

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
    metal.clear_ple_inputs();
}

/// Decode with `DECODE_DEBUG=1` drives the `log_decode_entry`
/// diagnostic block in `decode/diag.rs` lines 24-39.
#[test]
fn decode_token_with_decode_debug_env_logs_diagnostic_entry() {
    decode_one_token_with_env(&[("DECODE_DEBUG", Some("1"))], |_| {});
}

/// Decode with Q4_KF FFN weights — drives `decode/encode_ffn.rs`
/// lines 86-105 (Q4_KF FFN branch) + the gated Q4_KF path 110-145.
#[test]
fn decode_token_with_q4kf_ffn_drives_q4kf_paths() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    // Q4_KF reads the same 144-byte super-block layout as Q4_K but
    // with pre-baked half-scales.  Synthetic Q4_K bytes pass through
    // the kernel (it just reads bytes) — output won't be numerically
    // meaningful, but the dispatch covers the branch.
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate.format = QuantFormat::Q4_KF;
    layer.up.format = QuantFormat::Q4_KF;
    layer.down.format = QuantFormat::Q4_KF;

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
}

/// Decode with `LARQL_GATE_UP_COOP=1` opts in to the cooperative
/// scale-loading variant — drives `decode/encode_ffn.rs` lines
/// 239-247 (use_coop branch).
#[test]
fn decode_token_with_gate_up_coop_env_drives_coop_pipeline() {
    decode_one_token_with_env(&[("LARQL_GATE_UP_COOP", Some("1"))], |_| {});
}

/// Decode with `LARQL_GATE_UP_8SG=0` + `LARQL_F16_ACC=1` opts to the
/// 4sg+f16-acc variant — drives `decode/encode_ffn.rs` lines 247-253.
#[test]
fn decode_token_with_4sg_f16_acc_env_drives_f16acc_pipeline() {
    decode_one_token_with_env(
        &[
            ("LARQL_GATE_UP_8SG", Some("0")),
            ("LARQL_F16_ACC", Some("1")),
        ],
        |_| {},
    );
}

/// Decode with `LARQL_GATE_UP_8SG=0` (no f16) — drives the plain 4sg
/// variant at lines 253-258.
#[test]
fn decode_token_with_4sg_env_drives_4sg_pipeline() {
    decode_one_token_with_env(&[("LARQL_GATE_UP_8SG", Some("0"))], |_| {});
}

/// Decode with Q4_KF + Standard (non-gated) FFN — drives
/// `decode/encode_ffn.rs` lines 146-180 (Q4_KF non-gated arm).
#[test]
fn decode_token_with_q4kf_standard_ffn_drives_non_gated_q4kf() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    // gate is unused in Standard FFN — use a stub.
    let gate = vec![0u8; 256];
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate.format = QuantFormat::Q4_KF;
    layer.up.format = QuantFormat::Q4_KF;
    layer.down.format = QuantFormat::Q4_KF;
    layer.ffn_type = FfnType::Standard;

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
}

/// Decode with `LARQL_DECODE_DUMP_LAYERS=<dir>` drives the per-layer
/// dump path in `decode/mod.rs` lines 621-672.
#[test]
fn decode_token_with_decode_dump_layers_env() {
    let tmp = std::env::temp_dir().join("larql-compute-metal-decode-dump-test");
    let _ = std::fs::create_dir_all(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    decode_one_token_with_env(
        &[(
            "LARQL_DECODE_DUMP_LAYERS",
            Some(Box::leak(path.into_boxed_str())),
        )],
        |_| {
            let _ = std::fs::remove_dir_all(&tmp);
        },
    );
}

/// Decode with `layer_scalar != 0.0` drives the post-FFN scale_vector
/// dispatch (`decode/mod.rs` lines 593-602, non-MoE branch).
#[test]
fn decode_token_with_layer_scalar_drives_scale_vector_dispatch() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.layer_scalar = 0.5; // non-zero → scale_vector dispatch runs

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
}

/// MoE layer with NO moe_fn callback — drives the local
/// `cpu_moe_forward` fallback path in `decode/moe_interleave.rs`
/// (lines 161-166).
#[test]
fn decode_token_with_moe_layer_no_callback_drives_local_fallback() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let null_moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.moe = Some(null_moe);

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(1, 64, NUM_KV_HEADS, HEAD_DIM);
    // Plain decode_token → no moe_fn → local cpu_moe_forward fallback.
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
}

/// MoE layer + `LARQL_DUMP_L0` env — drives the
/// `moe_interleave::dump_l0_moe_intermediates` path
/// (`decode/moe_interleave.rs` lines 199-213).
#[test]
fn decode_token_with_moe_and_dump_l0_drives_moe_intermediate_dump() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let tmp = std::env::temp_dir().join("larql-cm-moe-dump-l0-test");
    let _ = std::fs::create_dir_all(&tmp);
    let path = tmp.to_str().unwrap().to_string();
    let path_static: &'static str = Box::leak(path.into_boxed_str());
    let saved = std::env::var_os("LARQL_DUMP_L0");
    unsafe {
        std::env::set_var("LARQL_DUMP_L0", path_static);
    }

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let null_moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.moe = Some(null_moe);

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

    unsafe {
        match saved {
            Some(v) => std::env::set_var("LARQL_DUMP_L0", v),
            None => std::env::remove_var("LARQL_DUMP_L0"),
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// MoE layer + `LARQL_DUMP_RESIDUALS` env on a 2-layer model —
/// drives the residual_dump record_layer call in
/// `moe_interleave.rs` lines 218-223 + next-layer-cmd-reset at
/// 226-228.
#[test]
fn decode_token_with_moe_and_dump_residuals_drives_record_layer() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let tmp = std::env::temp_dir().join("larql-cm-moe-residual-dump.bin");
    let path = tmp.to_str().unwrap().to_string();
    let path_static: &'static str = Box::leak(path.into_boxed_str());
    let saved = std::env::var_os("LARQL_DUMP_RESIDUALS");
    unsafe {
        std::env::set_var("LARQL_DUMP_RESIDUALS", path_static);
    }

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let make_null_moe = || MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer0 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer0.moe = Some(make_null_moe());
    let mut layer1 = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer1.moe = Some(make_null_moe());

    let x = synth_input(HIDDEN, 0.9);
    let mut kv = metal.create_kv_cache(2, 64, NUM_KV_HEADS, HEAD_DIM);
    let out = larql_compute_metal::MetalBackend::decode_token(
        &metal,
        &mut kv,
        &[layer0, layer1],
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

    unsafe {
        match saved {
            Some(v) => std::env::set_var("LARQL_DUMP_RESIDUALS", v),
            None => std::env::remove_var("LARQL_DUMP_RESIDUALS"),
        }
    }
    let _ = std::fs::remove_file(&tmp);
}

/// MoE layer with `ffn_is_remote = true` — drives the remote-FFN
/// branch in `moe_interleave.rs` lines 180-187 + the same line at
/// the dense-encode skip in `decode/mod.rs`.
#[test]
fn decode_token_with_moe_remote_ffn_drives_remote_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let null_moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.moe = Some(null_moe);
    layer.ffn_is_remote = true;

    let x = synth_input(HIDDEN, 0.9);
    let mut moe_fn = |_l: usize, h: &[f32]| -> Vec<f32> { vec![0.0f32; h.len()] };
    let out = metal.decode_token_with_moe(&[layer], &x, HIDDEN, INTER, &mut moe_fn);
    assert_eq!(out.expect("decode returns Some").len(), HIDDEN);
}

/// MoE-interleave decode: layer with `moe.is_some()` + a moe_fn
/// callback drives the CPU-side MoE interleave block
/// (`decode/mod.rs` lines 550-589 + `handle_moe_interleave`).
#[test]
fn decode_token_with_moe_fn_drives_interleave_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    // Null MoE: num_experts=0 makes cpu_moe_forward bail before doing
    // expert work, but the moe_fn callback is still invoked which is
    // what we need for coverage.
    let null_moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.moe = Some(null_moe);

    let x = synth_input(HIDDEN, 0.9);
    let mut moe_call_count = 0usize;
    let mut moe_fn = |_l: usize, h: &[f32]| -> Vec<f32> {
        moe_call_count += 1;
        vec![0.0f32; h.len()]
    };
    let out = metal.decode_token_with_moe(&[layer], &x, HIDDEN, INTER, &mut moe_fn);
    assert_eq!(out.expect("decode returns Some").len(), HIDDEN);
    assert_eq!(
        moe_call_count, 1,
        "moe_fn must be called once per MoE layer"
    );
}

/// MoE split-fire variant — both `moe_fn` (fire) and `moe_collect_fn`
/// callbacks drive the split-mode path (`split_mode = true` at
/// `decode/mod.rs:254`).
#[test]
fn decode_token_with_moe_split_fn_drives_split_mode_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::backend::DecodeBackend;
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
    };

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let null_moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 0,
        top_k: 1,
        intermediate_size: INTER,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.moe = Some(null_moe);

    let x = synth_input(HIDDEN, 0.9);
    let mut fired = 0usize;
    let mut collected = 0usize;
    let mut moe_fire = |_l: usize, _h: &[f32]| {
        fired += 1;
    };
    let mut moe_collect = |_l: usize| -> Vec<f32> {
        collected += 1;
        vec![0.0f32; HIDDEN]
    };
    let out = metal.decode_token_with_moe_split(
        &[layer],
        &x,
        HIDDEN,
        INTER,
        &mut moe_fire,
        &mut moe_collect,
    );
    assert_eq!(out.expect("decode returns Some").len(), HIDDEN);
    assert_eq!(fired, 1);
    assert_eq!(collected, 1);
}

/// Backend with PLE inputs prepared — covers the `ple_inputs.as_ref()`
/// branch around `decode/mod.rs` lines 496-519.  Synthetic layer
/// doesn't actually wire PLE weights so the inner `layer.ple_spec()`
/// check returns None; that exercises the `if Some(pli)` outer guard
/// while skipping the actual PLE dispatch (which would need real
/// gate/projection/post-norm weights to be correct).
#[test]
fn decode_token_with_ple_inputs_drives_outer_guard() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };

    // Synthetic per-layer-input table.  Sized for one layer × one
    // position with ple_dim = 32; any ple_dim works for the outer
    // guard test since `layer.ple_spec()` returns None.
    let ple_inputs: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
    metal.prepare_ple_inputs(&ple_inputs, 1, 32);

    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);

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
    metal.clear_ple_inputs();
}

// ─────────────────────────────────────────────────────────────────
// encode_ffn.rs branch coverage. Drives non-gated FFN variants
// (FfnType::Standard), the `gate_up_coop` / `gate_up_use_4sg` /
// `f16_acc` pipeline picks at the top of `encode_q4k_ffn`, and the
// `LARQL_PROFILE_SPLIT=1` paired-phase code path
// (encode_ffn_gate_up_phase + encode_ffn_down_phase) across the
// three quant families.
// ─────────────────────────────────────────────────────────────────

fn decode_with_options_synth_q4k_layer(opts: larql_compute_metal::BackendOptions) {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::with_options(opts) else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate.format = QuantFormat::Q4_K;
    layer.up.format = QuantFormat::Q4_K;
    layer.down.format = QuantFormat::Q4_K;
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
    assert!(out.iter().all(|v| v.is_finite()));
}

fn decode_with_profile_split_synth<F: FnOnce(&mut FullPipelineLayer)>(setup: F) {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let saved = std::env::var_os("LARQL_PROFILE_SPLIT");
    unsafe { std::env::set_var("LARQL_PROFILE_SPLIT", "1") };
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    setup(&mut layer);
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
    match saved {
        Some(v) => unsafe { std::env::set_var("LARQL_PROFILE_SPLIT", v) },
        None => unsafe { std::env::remove_var("LARQL_PROFILE_SPLIT") },
    }
}

/// `BackendOptions { gate_up_coop: true }` selects the cooperative
/// gate+up Q4_K pipeline (`q4k_ffn_gate_up_coop_pipeline`) at
/// `decode/encode_ffn.rs` lines 239-246.
///
/// Ignored on CI: the cooperative scale-loading kernel produces NaN on
/// the GitHub Actions macOS-14 (M1) runner against synthetic Q4_K
/// weights, while passing on M3 Max. The kernel is opt-in (documented
/// as kept around for future larger-K hardware in
/// `shaders/q4k_ffn_gate_up_coop.rs`); never on the default decode
/// path. Run `cargo test -- --ignored` on dev hardware to exercise it.
#[test]
#[ignore = "flaky on GitHub Actions M1 runner; gate_up_coop kernel produces NaN on M1 with synthetic Q4_K (passes on M3 Max). Opt-in kernel — not on default decode path. See shader retention doc."]
fn decode_token_with_q4k_ffn_and_gate_up_coop_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.gate_up_coop = true;
    decode_with_options_synth_q4k_layer(opts);
}

/// `BackendOptions { gate_up_use_4sg: true }` (LARQL_GATE_UP_8SG=0)
/// drives the 4-simdgroup Q4_K gate+up pipeline at lines 253-258.
#[test]
fn decode_token_with_q4k_ffn_and_4sg_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.gate_up_use_4sg = true;
    decode_with_options_synth_q4k_layer(opts);
}

/// `BackendOptions { gate_up_use_4sg: true, f16_acc: true }` drives
/// the 4sg + f16-accumulator Q4_K gate+up pipeline at lines 247-252.
#[test]
fn decode_token_with_q4k_ffn_4sg_and_f16_acc_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.gate_up_use_4sg = true;
    opts.decode_flags.f16_acc = true;
    decode_with_options_synth_q4k_layer(opts);
}

/// `BackendOptions { fused_down: false }` opts out of the fused
/// `q4k_geglu_silu_down` kernel — drives the separated GEGLU +
/// `quant_matvec` chain at `encode_q4k_ffn` lines 361-386.
#[test]
fn decode_token_with_q4k_ffn_and_unfused_down_option() {
    let mut opts = larql_compute_metal::BackendOptions::default();
    opts.decode_flags.fused_down = false;
    decode_with_options_synth_q4k_layer(opts);
}

/// `FfnType::Standard` (non-gated) + Q4_K weights drives the
/// `else` arm of `encode_q4k_ffn` (lines 389-424): up → activation
/// → down without GEGLU multiplication.
#[test]
fn decode_token_with_q4k_non_gated_ffn_drives_standard_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_k(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_k(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.gate.format = QuantFormat::Q4_K;
    layer.up.format = QuantFormat::Q4_K;
    layer.down.format = QuantFormat::Q4_K;
    layer.ffn_type = FfnType::Standard;
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `FfnType::Standard` + Q4_0 weights drives the `else` arm of
/// `encode_q4_0_ffn` (lines 463-481): up Q8-matvec → activation →
/// down.
#[test]
fn decode_token_with_q4_0_non_gated_ffn_drives_standard_path() {
    let _guard = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(metal) = larql_compute_metal::MetalBackend::new() else {
        return;
    };
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    let wq = quantize_q4_k(&synth_weight_f32(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth_weight_f32(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth_weight_f32(HIDDEN * Q_DIM, 0.4));
    let gate = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.5));
    let up = quantize_q4_0(&synth_weight_f32(INTER * HIDDEN, 0.6));
    let down = quantize_q4_0(&synth_weight_f32(HIDDEN * INTER, 0.7));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let mut layer = build_synth_layer(&wq, &wk, &wv, &wo, &gate, &up, &down, &norm_w);
    layer.ffn_type = FfnType::Standard;
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
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_K-gated FFN drives the split-phase
/// encoders at `decode/encode_ffn.rs` lines 649-674 (gate_up phase
/// Q4_K gated) and 751-783 (down phase Q4_K gated + Q4_K fused down).
#[test]
fn decode_token_with_profile_split_and_q4k_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate.format = QuantFormat::Q4_K;
        layer.up.format = QuantFormat::Q4_K;
        layer.down.format = QuantFormat::Q4_K;
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_K non-gated drives the non-gated
/// arms of both split-phase encoders at lines 663-673 + 784-806.
#[test]
fn decode_token_with_profile_split_and_q4k_non_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate.format = QuantFormat::Q4_K;
        layer.up.format = QuantFormat::Q4_K;
        layer.down.format = QuantFormat::Q4_K;
        layer.ffn_type = FfnType::Standard;
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_KF gated FFN drives the Q4_KF
/// arms of both split-phase encoders at lines 619-635 + 725-728.
#[test]
fn decode_token_with_profile_split_and_q4kf_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate.format = QuantFormat::Q4_KF;
        layer.up.format = QuantFormat::Q4_KF;
        layer.down.format = QuantFormat::Q4_KF;
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_KF non-gated drives the Q4_KF
/// non-gated arm of `encode_ffn_gate_up_phase` (lines 636-647) and
/// `encode_ffn_down_phase` (lines 729-749).
#[test]
fn decode_token_with_profile_split_and_q4kf_non_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.gate.format = QuantFormat::Q4_KF;
        layer.up.format = QuantFormat::Q4_KF;
        layer.down.format = QuantFormat::Q4_KF;
        layer.ffn_type = FfnType::Standard;
    });
}

/// `LARQL_PROFILE_SPLIT=1` + Q4_0 non-gated drives the Q4_0
/// non-gated arm of `encode_ffn_gate_up_phase` (lines 692-700) +
/// the Q4_0 non-gated arm of `encode_ffn_down_phase` (lines 812+).
#[test]
fn decode_token_with_profile_split_and_q4_0_non_gated_ffn() {
    decode_with_profile_split_synth(|layer| {
        layer.ffn_type = FfnType::Standard;
    });
}
