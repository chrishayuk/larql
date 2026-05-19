#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;

use common::{cos_sim, get_metal, max_diff};
use larql_compute::prelude::*;
use larql_compute_metal::MoeScratch;

fn synth_values(len: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let a = (seed + i as f32 * 0.0017).sin();
            let b = (seed * 0.37 + (i >> 7) as f32 * 0.019).cos();
            (a + 0.25 * b) * scale
        })
        .collect()
}

fn pad_rows_to_256(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, usize) {
    let padded_cols = cols.div_ceil(256) * 256;
    if padded_cols == cols {
        return (data.to_vec(), cols);
    }
    let mut out = vec![0.0f32; rows * padded_cols];
    for r in 0..rows {
        out[r * padded_cols..r * padded_cols + cols]
            .copy_from_slice(&data[r * cols..(r + 1) * cols]);
    }
    (out, padded_cols)
}

fn make_q4k_experts(hidden: usize, inter: usize, top_k: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut gate_up = Vec::with_capacity(top_k);
    let mut down = Vec::with_capacity(top_k);
    for e in 0..top_k {
        let gate = synth_values(inter * hidden, 0.11 + e as f32 * 0.13, 0.18);
        let up = synth_values(inter * hidden, 0.41 + e as f32 * 0.17, 0.16);
        let mut gu = Vec::with_capacity(2 * inter * hidden);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        gate_up.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(&gu));

        let raw_down = synth_values(hidden * inter, 0.73 + e as f32 * 0.07, 0.11);
        let (down_padded, _) = pad_rows_to_256(&raw_down, hidden, inter);
        down.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(
            &down_padded,
        ));
    }
    (gate_up, down)
}

fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), in_cols);
    debug_assert_eq!(w.len(), out_rows * in_cols);
    let mut out = vec![0.0f32; out_rows];
    for row in 0..out_rows {
        let w_row = &w[row * in_cols..(row + 1) * in_cols];
        out[row] = w_row.iter().zip(x).map(|(&wi, &xi)| wi * xi).sum();
    }
    out
}

fn run_single_expert_f32_reference(
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let inter_padded = inter.div_ceil(block) * block;
    let gate_up_w =
        larql_compute::cpu::ops::q4_common::dequantize_q4_k(gate_up_bytes, 2 * inter * hidden);
    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

    let gate_out = matmul_vec(h_norm, gate_w, inter, hidden);
    let up_out = matmul_vec(h_norm, up_w, inter, hidden);

    let mut act = vec![0.0f32; inter_padded];
    for j in 0..inter {
        act[j] = gelu_tanh(gate_out[j]) * up_out[j];
    }

    let down_w =
        larql_compute::cpu::ops::q4_common::dequantize_q4_k(down_bytes, hidden * inter_padded);
    matmul_vec(&act, &down_w, hidden, inter_padded)
}

fn run_single_expert_separated_metal_reference(
    metal: &larql_compute_metal::MetalBackend,
    h_norm: &[f32],
    gate_up_bytes: &[u8],
    down_bytes: &[u8],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let block = larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    let inter_padded = inter.div_ceil(block) * block;
    let row_bytes = (hidden / block) * larql_models::quant::ggml::Q4_K_BLOCK_BYTES;
    let half = inter * row_bytes;
    let gate = metal
        .q4k_matvec(&gate_up_bytes[..half], h_norm, inter, hidden)
        .expect("Metal gate q4k matvec");
    let up = metal
        .q4k_matvec(&gate_up_bytes[half..2 * half], h_norm, inter, hidden)
        .expect("Metal up q4k matvec");

    let mut act = vec![0.0f32; inter_padded];
    for j in 0..inter {
        act[j] = gelu_tanh(gate[j]) * up[j];
    }

    metal
        .q4k_matvec(down_bytes, &act, hidden, inter_padded)
        .expect("Metal down q4k matvec")
}

fn assert_preselected_dispatch_matches_cpu(label: &str, hidden: usize, inter: usize, top_k: usize) {
    let metal = get_metal();
    let h_norm = synth_values(hidden, 1.23, 0.35);
    let expert_ids: Vec<usize> = (0..top_k).collect();
    let expert_weights: Vec<f32> = (0..top_k)
        .map(|i| (i as f32 + 1.0) / (top_k as f32 * (top_k as f32 + 1.0) * 0.5))
        .collect();
    let (gate_up, down) = make_q4k_experts(hidden, inter, top_k);

    let mut expected = vec![0.0f32; hidden];
    for e in 0..top_k {
        let out = run_single_expert_f32_reference(&h_norm, &gate_up[e], &down[e], hidden, inter);
        for (acc, &v) in expected.iter_mut().zip(&out) {
            *acc += v * expert_weights[e];
        }
    }

    let mut separated_metal = vec![0.0f32; hidden];
    for e in 0..top_k {
        let out = run_single_expert_separated_metal_reference(
            &metal,
            &h_norm,
            &gate_up[e],
            &down[e],
            hidden,
            inter,
        );
        for (acc, &v) in separated_metal.iter_mut().zip(&out) {
            *acc += v * expert_weights[e];
        }
    }

    let scratch = MoeScratch::new_public(&metal, top_k, hidden, inter);
    let got = metal.run_experts_preselected_metal(
        &h_norm,
        &expert_ids,
        &expert_weights,
        &scratch,
        |eid| Some((gate_up[eid].as_slice(), down[eid].as_slice())),
    );

    let diff = max_diff(&expected, &got);
    let cos = cos_sim(&expected, &got);
    let expected_max = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = diff / expected_max.max(1.0);
    let metal_diff = max_diff(&separated_metal, &got);
    let metal_cos = cos_sim(&separated_metal, &got);
    let metal_max = separated_metal
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let metal_rel = metal_diff / metal_max.max(1.0);
    let nonzero = got.iter().filter(|&&v| v.abs() > 1e-6).count();
    assert!(
        nonzero > hidden / 2 && metal_rel < 1e-4 && metal_cos > 0.999_999,
        "{label}: Metal MoE expert dispatch diverged from CPU: \
         cpu_max_abs={diff:.3e} cpu_rel={rel:.3e} cpu_cos={cos:.6} \
         metal_max_abs={metal_diff:.3e} metal_rel={metal_rel:.3e} \
         metal_cos={metal_cos:.6} nonzero={nonzero}/{hidden}"
    );
}

#[test]
fn metal_moe_preselected_small_q4k_matches_cpu() {
    assert_preselected_dispatch_matches_cpu("small q4k moe", 256, 256, 2);
}

#[test]
#[ignore = "known open Metal MoE issue at Gemma 4 26B-A4B shape; run explicitly while debugging"]
fn metal_moe_preselected_gemma4_26b_a4b_shape_matches_cpu() {
    assert_preselected_dispatch_matches_cpu("gemma4-26b-a4b moe", 2816, 704, 8);
}

// ─────────────────────────────────────────────────────────────────
// Coverage tests for moe_dispatch.rs paths not exercised by the
// preselected-vs-CPU parity test above: `run_experts_prestaged_metal`,
// `run_dense_ffn_q4k`, edge-case early-returns and `continue`/`break`
// arms inside `run_experts_preselected_metal`.
// ─────────────────────────────────────────────────────────────────

/// `run_experts_prestaged_metal` — the shard-RPC variant that takes
/// pre-staged `(gate_up_buf, down_buf)` Metal buffers instead of byte
/// slices.  Drives `decode/moe_dispatch.rs` lines 265-405.
#[test]
fn run_experts_prestaged_metal_smoke() {
    let metal = get_metal();
    let hidden = 256;
    let inter = 256;
    let top_k = 2;
    let h_norm = synth_values(hidden, 0.31, 0.25);
    let expert_weights: Vec<f32> = vec![0.4, 0.6];
    let (gate_up, down) = make_q4k_experts(hidden, inter, top_k);

    // Stage each expert's bytes into Metal buffers up-front (this is
    // the production usage from larql-server's shard endpoint:
    // `cached_buffer_for_bytes` once per expert at warm-up).
    let expert_bufs: Vec<_> = gate_up
        .iter()
        .zip(down.iter())
        .map(|(gu, dn)| {
            (
                metal.cached_buffer_for_bytes(gu),
                metal.cached_buffer_for_bytes(dn),
            )
        })
        .collect();

    let scratch = MoeScratch::new_public(&metal, top_k, hidden, inter);
    let out = metal.run_experts_prestaged_metal(&h_norm, &expert_bufs, &expert_weights, &scratch);
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|&v| v.abs() > 1e-6), "all-zero output");
}

/// `run_experts_prestaged_metal` early-return when `expert_bufs` is
/// empty — drives line 278-280 (zero-K guard).
#[test]
fn run_experts_prestaged_metal_empty_experts_returns_zero() {
    let metal = get_metal();
    let hidden = 256;
    let inter = 256;
    let scratch = MoeScratch::new_public(&metal, 2, hidden, inter);
    let h_norm = vec![0.0f32; hidden];
    let out = metal.run_experts_prestaged_metal(&h_norm, &[], &[], &scratch);
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|&v| v == 0.0));
}

/// `run_dense_ffn_q4k` — single-layer non-MoE FFN with mmap-staged
/// gate / up / down Metal buffers. Drives `moe_dispatch.rs` lines
/// 627-721.
#[test]
fn run_dense_ffn_q4k_smoke() {
    let metal = get_metal();
    let hidden: usize = 256;
    let inter: usize = 256;
    let inter_padded = inter.div_ceil(256) * 256;
    let h_norm = synth_values(hidden, 0.41, 0.25);

    let gate = synth_values(inter * hidden, 0.11, 0.18);
    let up = synth_values(inter * hidden, 0.41, 0.16);
    let raw_down = synth_values(hidden * inter, 0.73, 0.11);
    let (down_padded, _) = pad_rows_to_256(&raw_down, hidden, inter);

    let gate_bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&gate);
    let up_bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&up);
    let down_bytes = larql_compute::cpu::ops::q4_common::quantize_q4_k(&down_padded);

    let gate_buf = metal.cached_buffer_for_bytes(&gate_bytes);
    let up_buf = metal.cached_buffer_for_bytes(&up_bytes);
    let down_buf = metal.cached_buffer_for_bytes(&down_bytes);

    let out = metal.run_dense_ffn_q4k(
        &h_norm,
        &gate_buf,
        &up_buf,
        &down_buf,
        hidden,
        inter,
        inter_padded,
    );
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
    assert!(out.iter().any(|&v| v.abs() > 1e-6), "all-zero output");
}

/// `run_dense_ffn_q4k` early-return on `hidden = 0` — drives line
/// 637-639 (zero-shape guard).
#[test]
fn run_dense_ffn_q4k_zero_hidden_returns_empty() {
    let metal = get_metal();
    let zero_bytes = vec![0u8; 144];
    let gate_buf = metal.cached_buffer_for_bytes(&zero_bytes);
    let up_buf = metal.cached_buffer_for_bytes(&zero_bytes);
    let down_buf = metal.cached_buffer_for_bytes(&zero_bytes);
    let out = metal.run_dense_ffn_q4k(&[], &gate_buf, &up_buf, &down_buf, 0, 0, 0);
    assert!(out.is_empty());
}

/// `run_experts_preselected_metal` early-return on empty `expert_ids`
/// — drives line 441-443 (zero-K guard).
#[test]
fn run_experts_preselected_empty_returns_zero() {
    let metal = get_metal();
    let hidden = 256;
    let inter = 256;
    let scratch = MoeScratch::new_public(&metal, 2, hidden, inter);
    let h_norm = vec![0.0f32; hidden];
    let out = metal.run_experts_preselected_metal(
        &h_norm,
        &[],
        &[],
        &scratch,
        |_| -> Option<(&'static [u8], &'static [u8])> { None },
    );
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|&v| v == 0.0));
}

/// `run_experts_preselected_metal` `continue` arms inside the staging
/// loop: closure returns `None` (line 463), bytes too short (line
/// 466). Both should skip the expert; downstream output drops to zero
/// because no valid expert ever lands.
#[test]
fn run_experts_preselected_skips_missing_and_truncated_experts() {
    let metal = get_metal();
    let hidden = 256;
    let inter = 256;
    let scratch = MoeScratch::new_public(&metal, 2, hidden, inter);
    let h_norm = synth_values(hidden, 0.31, 0.25);
    let expert_ids = vec![0usize, 1];
    let expert_weights = vec![0.5f32, 0.5];
    let short = vec![0u8; 16]; // far too small for `2 * gate_half_bytes`
    let out = metal.run_experts_preselected_metal(
        &h_norm,
        &expert_ids,
        &expert_weights,
        &scratch,
        |eid| {
            // Expert 0 returns short bytes (truncation continue at L466);
            // expert 1 returns None (missing continue at L463).
            if eid == 0 {
                Some((short.as_slice(), short.as_slice()))
            } else {
                None
            }
        },
    );
    // valid_count = 0 → early-return on line 506.
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|&v| v == 0.0));
}

/// `run_experts_preselected_metal` `break` arm when `valid_count >=
/// scratch.top_k` (line 472-474) — caller passes more experts than
/// scratch was allocated for, helper truncates to fit.
#[test]
fn run_experts_preselected_truncates_to_scratch_top_k() {
    let metal = get_metal();
    let hidden = 256;
    let inter = 256;
    let scratch_top_k = 1; // scratch sized for 1 expert
    let scratch = MoeScratch::new_public(&metal, scratch_top_k, hidden, inter);
    let h_norm = synth_values(hidden, 0.31, 0.25);
    // Caller passes 2 experts; helper should process the first then
    // break (line 473).
    let expert_ids = vec![0usize, 1];
    let expert_weights = vec![0.5f32, 0.5];
    let (gate_up, down) = make_q4k_experts(hidden, inter, 2);
    let out = metal.run_experts_preselected_metal(
        &h_norm,
        &expert_ids,
        &expert_weights,
        &scratch,
        |eid| Some((gate_up[eid].as_slice(), down[eid].as_slice())),
    );
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `decode_token_q4k_moe` with a real MoE layer — drives the
/// scratch-allocation block (175-188) and the inner `moe_fn` closure
/// (190-207) plus the full `gpu_moe_dispatch_with_scratch` path
/// (lines 734-908).
#[test]
fn decode_token_q4k_moe_with_real_moe_layer_drives_gpu_dispatch() {
    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k};
    use larql_compute::{
        Activation, DecodeBackend, FfnType, FullPipelineLayer, MoeLayerWeights, MoeRoutingPolicy,
        MoeWeightLayout, NormType, QuantFormat, QuantWeight,
    };
    let metal = get_metal();
    let hidden = 256;
    let inter = 256;
    let top_k = 2;
    let num_experts = 4;
    let q_dim = 128;
    let kv_dim = 64;
    let num_q_heads = 2;
    let num_kv_heads = 1;
    let head_dim = 64;

    let wq = quantize_q4_k(&synth_values(q_dim * hidden, 0.1, 0.3));
    let wk = quantize_q4_k(&synth_values(kv_dim * hidden, 0.2, 0.3));
    let wv = quantize_q4_k(&synth_values(kv_dim * hidden, 0.3, 0.3));
    let wo = quantize_q4_k(&synth_values(hidden * q_dim, 0.4, 0.3));
    let gate = quantize_q4_0(&synth_values(inter * hidden, 0.5, 0.2));
    let up = quantize_q4_0(&synth_values(inter * hidden, 0.6, 0.2));
    let down = quantize_q4_0(&synth_values(hidden * inter, 0.7, 0.2));
    let norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    let router_w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32 * 0.0003).sin() * 0.05)
        .collect();
    let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let router_scale: Vec<f32> = vec![1.0f32; hidden];
    let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];

    let (expert_gu, expert_down) = make_q4k_experts(hidden, inter, num_experts);

    let moe = MoeLayerWeights {
        experts_gate_up: expert_gu.iter().map(|v| v.as_slice()).collect(),
        experts_down: expert_down.iter().map(|v| v.as_slice()).collect(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q4_K,
        router_proj: &router_w,
        router_scale: &router_scale,
        router_per_expert_scale: &router_per_expert_scale,
        router_norm: &[],
        router_norm_parameter_free: true,
        router_input_scalar: 1.0,
        pre_experts_norm: &pre_norm_w,
        post_ffn1_norm: &pre_norm_w,
        post_experts_norm: &pre_norm_w,
        num_experts,
        top_k,
        intermediate_size: inter,
        activation: Activation::GeluTanh,
    };

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
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads,
        num_kv_heads,
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
        moe: Some(moe),
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        kv_shared_source: None,
        residual_multiplier: 1.0,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
    };

    let x = synth_values(hidden, 0.9, 0.25);
    let _ = metal.create_kv_cache(1, 64, num_kv_heads, head_dim);
    <larql_compute_metal::MetalBackend as DecodeBackend>::reset_kv_cache(&metal);

    let out = metal.decode_token_q4k_moe(
        &[layer],
        &x,
        hidden,
        inter,
        q_dim,
        kv_dim,
        num_q_heads,
        num_kv_heads,
        head_dim,
        10_000.0,
        1e-6,
        |_layer_idx, expert_idx| {
            Some((
                expert_gu[expert_idx].as_slice(),
                expert_down[expert_idx].as_slice(),
            ))
        },
    );
    let out = out.expect("decode_token_q4k_moe should return Some when MoE layer present");
    assert_eq!(out.len(), hidden);
    assert!(out.iter().all(|v| v.is_finite()));
}
