//! Tests for [`super`].
//!
//! Split out of `moe_interleave.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;
use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;
use larql_compute::pipeline::FullPipelineLayer;
use larql_compute::{
    Activation, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat,
};

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

fn synth(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.013).sin() * 0.2)
        .collect()
}

fn pad_rows_to_256(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let padded_cols = cols.div_ceil(256) * 256;
    if padded_cols == cols {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; rows * padded_cols];
    for r in 0..rows {
        out[r * padded_cols..r * padded_cols + cols]
            .copy_from_slice(&data[r * cols..(r + 1) * cols]);
    }
    out
}

/// Same layout `tests/test_kernel_moe_expert_dispatch.rs` uses for
/// Q4_K experts: fused `[gate | up]` halves, block-padded down rows.
fn make_q4k_experts(hidden: usize, inter: usize, n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut gate_up = Vec::with_capacity(n);
    let mut down = Vec::with_capacity(n);
    for e in 0..n {
        let gate = synth(inter * hidden, 0.11 + e as f32 * 0.13);
        let up = synth(inter * hidden, 0.41 + e as f32 * 0.17);
        let mut gu = Vec::with_capacity(2 * inter * hidden);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        gate_up.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(&gu));

        let raw_down = synth(hidden * inter, 0.73 + e as f32 * 0.07);
        let down_padded = pad_rows_to_256(&raw_down, hidden, inter);
        down.push(larql_compute::cpu::ops::q4_common::quantize_q4_k(
            &down_padded,
        ));
    }
    (gate_up, down)
}

/// Every `try_inline_zero_copy_moe` precondition satisfied: pure-MoE
/// layer (no dense FFN branch, via `FullPipelineLayer::default()`'s
/// empty `up`/`down` weights), identity-combine routing policy
/// (`top_k_softmax`'s `post_expert_norm: None`, `layer_scalar: 0.0`,
/// no combined-output norm), no diagnostic captures, and every
/// expert's bytes pre-registered as a zero-copy region. This is the
/// merged-CB fast path `handle_moe_interleave` takes when the
/// backend's expert scratch is live — never reached by the
/// staged-path tests in `moe_dispatch.rs`/the integration suite,
/// which all use the hybrid (dense+MoE) or default routing-policy
/// shape instead.
#[test]
fn try_inline_zero_copy_moe_encodes_experts_and_combine_on_registered_region() {
    let m = backend();
    let hidden = 256usize;
    let inter = 256usize;
    let top_k = 2usize;
    let num_experts = 4usize;

    let (expert_gu, expert_down) = make_q4k_experts(hidden, inter, num_experts);

    // Lay every expert out contiguously in one page-aligned anonymous
    // mmap, exactly the production `register_weight_region` contract.
    let total: usize = expert_gu
        .iter()
        .zip(expert_down.iter())
        .map(|(g, d)| g.len() + d.len())
        .sum();
    let mut region = memmap2::MmapMut::map_anon(total).expect("anon mmap");
    let mut offsets = Vec::with_capacity(num_experts);
    let mut cursor = 0usize;
    for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
        region[cursor..cursor + g.len()].copy_from_slice(g);
        let g_off = cursor;
        cursor += g.len();
        region[cursor..cursor + d.len()].copy_from_slice(d);
        offsets.push((g_off, g.len(), cursor, d.len()));
        cursor += d.len();
    }
    let region = region.make_read_only().expect("read-only mmap");
    assert!(
        m.bufs.register_region(&region[..]),
        "page-aligned anon mmap must register"
    );

    // `moe.experts_gate_up`/`experts_down` MUST be slices into the
    // registered `region`, not the original `expert_gu`/`expert_down`
    // vectors those bytes were copied from — those still live at a
    // different, unregistered address. Passing the originals here was
    // the actual bug this test spent several CI round-trips finding:
    // every precondition matched, but `resolve_selected_experts`
    // still failed because `moe`'s own byte slices didn't point into
    // the region `register_region` was called on, so `resolve_region`
    // correctly reported no match for either selected expert.
    let experts_gate_up: Vec<&[u8]> = offsets
        .iter()
        .map(|&(g_off, g_len, _, _)| &region[g_off..g_off + g_len])
        .collect();
    let experts_down: Vec<&[u8]> = offsets
        .iter()
        .map(|&(_, _, d_off, d_len)| &region[d_off..d_off + d_len])
        .collect();

    let router_w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32 * 0.0003).sin() * 0.05)
        .collect();
    let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let router_scale: Vec<f32> = vec![1.0f32; hidden];
    let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
    let moe = MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        // `top_k_softmax`, NOT the crate default (`gemma4_hybrid`):
        // the default's `post_expert_norm: RmsNorm` fails this
        // function's identity-combine precondition outright.
        routing_policy: MoeRoutingPolicy::top_k_softmax(),
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
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
    };

    let scratch = MoeScratch::new_public(&m, top_k, hidden, inter);
    // `FullPipelineLayer::default()` has empty `up`/`down` weights
    // (`has_dense_ffn() == false`), `layer_scalar: 0.0`,
    // `moe_combined_output_norm: false`, `ffn_is_remote: false` —
    // every non-MoE precondition this function checks.
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter,
        inter_padded: inter,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);

    let h_post_attn_data = synth(hidden, 0.9);
    let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
    let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
    // Unused by this precondition/path combination — one shared dummy
    // buffer is enough for every field `try_inline_zero_copy_moe`
    // never reads.
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &h_post_attn_buf,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &new_h_buf,
    };

    let mut cmd = m.queue.new_command_buffer().to_owned();
    let mut enc = cmd.new_compute_command_encoder().to_owned();
    let mut encoder_ended = true;
    // `try_inline_zero_copy_moe` REPLACES `*enc`/`*cmd` in place on the
    // fast-path hit — it assumes the caller already ended/committed
    // the incoming encoder (exactly what `handle_moe_interleave` does
    // right before calling it). Skipping this crashes the whole test
    // binary: Metal fatally asserts on dropping a command encoder
    // that was never `end_encoding()`'d.
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:213",
    )
    .expect("command buffer completed");
    let took_zero_copy_path = m.try_inline_zero_copy_moe(
        &layer,
        &ctx,
        &bufs,
        &ictx,
        &h_post_attn_data,
        &mut cmd,
        &mut enc,
        &mut encoder_ended,
    );
    assert!(
        took_zero_copy_path,
        "every precondition was satisfied; the merged-CB fast path must fire"
    );
    assert!(!encoder_ended);

    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:232",
    )
    .expect("command buffer completed");

    let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite combine output"
    );
    assert!(
        out.iter().any(|&v| v.abs() > 1e-6),
        "combine wrote an all-zero buffer — vacuous dispatch"
    );
}

/// The test above puts every expert in ONE registered region, so
/// `encode_experts_zero_copy`'s `single_base` check is always true and
/// only the grouped-kernel arms run. Registering each expert in its
/// OWN region instead forces `single_base` to false for both the
/// gate/up and down dispatches regardless of which two experts the
/// router selects, driving the per-expert (non-grouped) fused Q4_K
/// kernel and per-expert down-matvec fallback arms — the other half
/// of that function's dispatch-shape branching. Also sets non-empty
/// `experts_gate_up_bias`/`experts_down_bias` to drive the bias-staging
/// block here and the `has_bias` combine arm in
/// `encode_experts_and_combine_zero_copy`, neither of which the
/// bias-free test above reaches. Non-empty biases force
/// `gate_rule: ClampedGlu` too — `biased_gated_servable` requires
/// either ClampedGlu or both bias arrays empty, since a `Gated`
/// layer with expert biases has no kernel — which additionally
/// covers the ClampedGlu activation arm the first test never takes.
#[test]
fn try_inline_zero_copy_moe_uses_non_grouped_dispatch_across_separate_regions() {
    let m = backend();
    let hidden = 256usize;
    let inter = 256usize;
    let top_k = 2usize;
    let num_experts = 4usize;

    let (expert_gu, expert_down) = make_q4k_experts(hidden, inter, num_experts);

    // One page-aligned anonymous mmap PER expert — `resolve_region`
    // returns the same Metal buffer for any two sub-slices of the same
    // registered region, so this is what actually forces
    // `single_base` to observe distinct base buffers.
    let mut regions = Vec::with_capacity(num_experts);
    for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
        let mut region = memmap2::MmapMut::map_anon(g.len() + d.len()).expect("anon mmap");
        region[..g.len()].copy_from_slice(g);
        region[g.len()..].copy_from_slice(d);
        let region = region.make_read_only().expect("read-only mmap");
        assert!(
            m.bufs.register_region(&region[..]),
            "page-aligned anon mmap must register"
        );
        regions.push(region);
    }
    let experts_gate_up: Vec<&[u8]> = regions
        .iter()
        .zip(expert_gu.iter())
        .map(|(region, g)| &region[..g.len()])
        .collect();
    let experts_down: Vec<&[u8]> = regions
        .iter()
        .zip(expert_gu.iter())
        .map(|(region, g)| &region[g.len()..])
        .collect();

    let router_w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32 * 0.0003).sin() * 0.05)
        .collect();
    let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let router_scale: Vec<f32> = vec![1.0f32; hidden];
    let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
    // Non-empty so `expert_mlp(..).gate_up_bias`/`down_bias` are
    // non-empty too — `ExpertMlp::expert_mlp` slices these per-expert
    // at strides `2 * inter` and `hidden` respectively.
    let experts_gate_up_bias = vec![0.1f32; num_experts * 2 * inter];
    let experts_down_bias = vec![0.05f32; num_experts * hidden];
    let moe = MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        routing_policy: MoeRoutingPolicy::top_k_softmax(),
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
        router_bias: &[],
        experts_gate_up_bias: &experts_gate_up_bias,
        experts_down_bias: &experts_down_bias,
        // `biased_gated_servable` requires EITHER ClampedGlu OR both
        // bias arrays empty — "a Gated layer with expert biases has
        // no kernel" (see try_inline_zero_copy_moe's own comment).
        // Non-empty biases with `Gated` here made the function bail
        // at that check on the first attempt; this is also the
        // combination that drives the ClampedGlu activation arm
        // (limit/alpha values match tests/test_moe_clamped_glu_q6k.rs).
        gate_rule: MoeGateRule::ClampedGlu {
            limit: 7.0,
            alpha: 1.702,
        },
    };

    let scratch = MoeScratch::new_public(&m, top_k, hidden, inter);
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter,
        inter_padded: inter,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);

    let h_post_attn_data = synth(hidden, 0.4);
    let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
    let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &h_post_attn_buf,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &new_h_buf,
    };

    let mut cmd = m.queue.new_command_buffer().to_owned();
    let mut enc = cmd.new_compute_command_encoder().to_owned();
    let mut encoder_ended = true;
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:388",
    )
    .expect("command buffer completed");
    let took_zero_copy_path = m.try_inline_zero_copy_moe(
        &layer,
        &ctx,
        &bufs,
        &ictx,
        &h_post_attn_data,
        &mut cmd,
        &mut enc,
        &mut encoder_ended,
    );
    assert!(
        took_zero_copy_path,
        "every precondition was satisfied; the merged-CB fast path must fire"
    );
    assert!(!encoder_ended);

    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:407",
    )
    .expect("command buffer completed");

    let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite combine output"
    );
    assert!(
        out.iter().any(|&v| v.abs() > 1e-6),
        "combine wrote an all-zero buffer — vacuous dispatch"
    );
}

/// Both tests above use `expert_data_format: QuantFormat::Q4_K` — the
/// two Q6_K arms in `encode_experts_zero_copy` (grouped and
/// non-grouped matvec) are entirely untested by either. Single shared
/// region (`single_base` true) drives the Q6_K grouped kernel arm,
/// same shape `tests/test_kernel_moe_expert_dispatch.rs`'s
/// `zero_copy_grouped_q6k_dispatch_matches_staged_path` already
/// proves numerically — this test only needs the fast path to fire
/// and produce a non-vacuous result, not bit-exact parity.
#[test]
fn try_inline_zero_copy_moe_uses_q6k_grouped_dispatch() {
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;

    let m = backend();
    let hidden = 256usize;
    let inter = 256usize;
    let top_k = 2usize;
    let num_experts = 4usize;

    let mut expert_gu: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut expert_down: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let gate = synth(inter * hidden, 0.21 + e as f32 * 0.13);
        let up = synth(inter * hidden, 0.51 + e as f32 * 0.17);
        let mut gu = Vec::with_capacity(2 * inter * hidden);
        gu.extend_from_slice(&gate);
        gu.extend_from_slice(&up);
        expert_gu.push(quantize_q6_k(&gu));
        let raw_down = synth(hidden * inter, 0.83 + e as f32 * 0.07);
        let down_padded = pad_rows_to_256(&raw_down, hidden, inter);
        expert_down.push(quantize_q6_k(&down_padded));
    }

    let total: usize = expert_gu
        .iter()
        .zip(expert_down.iter())
        .map(|(g, d)| g.len() + d.len())
        .sum();
    let mut region = memmap2::MmapMut::map_anon(total).expect("anon mmap");
    let mut offsets = Vec::with_capacity(num_experts);
    let mut cursor = 0usize;
    for (g, d) in expert_gu.iter().zip(expert_down.iter()) {
        region[cursor..cursor + g.len()].copy_from_slice(g);
        let g_off = cursor;
        cursor += g.len();
        region[cursor..cursor + d.len()].copy_from_slice(d);
        offsets.push((g_off, g.len(), cursor, d.len()));
        cursor += d.len();
    }
    let region = region.make_read_only().expect("read-only mmap");
    assert!(
        m.bufs.register_region(&region[..]),
        "page-aligned anon mmap must register"
    );
    let experts_gate_up: Vec<&[u8]> = offsets
        .iter()
        .map(|&(g_off, g_len, _, _)| &region[g_off..g_off + g_len])
        .collect();
    let experts_down: Vec<&[u8]> = offsets
        .iter()
        .map(|&(_, _, d_off, d_len)| &region[d_off..d_off + d_len])
        .collect();

    let router_w: Vec<f32> = (0..num_experts * hidden)
        .map(|i| (i as f32 * 0.0004).cos() * 0.05)
        .collect();
    let pre_norm_w: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let router_scale: Vec<f32> = vec![1.0f32; hidden];
    let router_per_expert_scale: Vec<f32> = vec![1.0f32; num_experts];
    let moe = MoeLayerWeights {
        expert_scales: larql_compute::MoeExpertScales::Inline,
        fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up,
        experts_down,
        routing_policy: MoeRoutingPolicy::top_k_softmax(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::Q6_K,
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
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
    };

    let scratch = MoeScratch::new_public_with_format(
        &m,
        top_k,
        hidden,
        inter,
        QuantFormat::Q6_K,
        moe.gate_up_cols(hidden),
    );
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter,
        inter_padded: inter,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);

    let h_post_attn_data = synth(hidden, 0.6);
    let h_post_attn_buf = m.bufs.transient_from_f32(&h_post_attn_data);
    let new_h_buf = m.bufs.transient_from_f32(&vec![0.0f32; hidden]);
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &h_post_attn_buf,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &new_h_buf,
    };

    let mut cmd = m.queue.new_command_buffer().to_owned();
    let mut enc = cmd.new_compute_command_encoder().to_owned();
    let mut encoder_ended = true;
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:564",
    )
    .expect("command buffer completed");
    let took_zero_copy_path = m.try_inline_zero_copy_moe(
        &layer,
        &ctx,
        &bufs,
        &ictx,
        &h_post_attn_data,
        &mut cmd,
        &mut enc,
        &mut encoder_ended,
    );
    assert!(
        took_zero_copy_path,
        "every precondition was satisfied; the merged-CB fast path must fire"
    );
    assert!(!encoder_ended);

    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:583",
    )
    .expect("command buffer completed");

    let out = unsafe { std::slice::from_raw_parts(new_h_buf.contents() as *const f32, hidden) };
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite combine output"
    );
    assert!(
        out.iter().any(|&v| v.abs() > 1e-6),
        "combine wrote an all-zero buffer — vacuous dispatch"
    );
}

/// `layer.moe.is_none()` is the first precondition check — must
/// bail out before touching the command buffer/encoder at all.
#[test]
fn try_inline_zero_copy_moe_returns_false_without_moe_layer() {
    let m = backend();
    // `MoeScratch::new` debug-asserts `weight_cols.is_multiple_of(block)`
    // (Q4_K block = 256 elements) unconditionally, before this test's
    // early-return path is ever reached — must be a block multiple even
    // though the actual dispatch never runs.
    let hidden = 256usize;
    let layer = FullPipelineLayer {
        moe: None,
        ..Default::default()
    };
    let ctx = MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden,
        inter: hidden,
        inter_padded: hidden,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    };
    let scratch = MoeScratch::new_public(&m, 1, hidden, hidden);
    let ictx = InlineMoeCtx::new(&scratch, 1e-6);
    let h_post_attn_data = vec![0.0f32; hidden];
    let dummy = m.bufs.transient_from_f32(&[0.0f32; 4]);
    let bufs = MoeInterleaveBufs {
        gate_w: &dummy,
        up_w: &dummy,
        down_w: &dummy,
        h_post_attn: &dummy,
        ffn_norm_out: &dummy,
        ffn_q8: &dummy,
        ffn_q8s: &dummy,
        gate_out_scratch: &dummy,
        up_out: &dummy,
        act_buf: &dummy,
        down_out: &dummy,
        normed_scratch: &dummy,
        new_h: &dummy,
    };
    let mut cmd = m.queue.new_command_buffer().to_owned();
    let mut enc = cmd.new_compute_command_encoder().to_owned();
    let mut encoder_ended = true;
    // `try_inline_zero_copy_moe` REPLACES `*enc`/`*cmd` in place on the
    // fast-path hit — it assumes the caller already ended/committed
    // the incoming encoder (exactly what `handle_moe_interleave` does
    // right before calling it). Skipping this crashes the whole test
    // binary: Metal fatally asserts on dropping a command encoder
    // that was never `end_encoding()`'d.
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        &cmd,
        "crates/larql-compute-metal/src/decode/moe_interleave/tests.rs:651",
    )
    .expect("command buffer completed");
    let took_zero_copy_path = m.try_inline_zero_copy_moe(
        &layer,
        &ctx,
        &bufs,
        &ictx,
        &h_post_attn_data,
        &mut cmd,
        &mut enc,
        &mut encoder_ended,
    );
    assert!(!took_zero_copy_path);
    // Early-return arms never touch `*encoder_ended` — it must come
    // back exactly as the caller left it (already ended, per the
    // real `handle_moe_interleave` calling convention above), not
    // flipped to `false` as the success path would.
    assert!(
        encoder_ended,
        "must leave caller state untouched on bail-out"
    );
}

// ── inline_moe_preconditions: the single authority both arms consume ─────
//
// The S2 GPU-route arm decides whether to SKIP attention's commit + wait by
// asking this function; the CPU fast path asks the same function whether to
// run. That shared answer is the point — two drifting copies would skip a
// wait some fallback arm still needs, which surfaces as intermittently
// wrong logits rather than as a crash.
//
// Every refusal is pinned to its own cause AND its own message, because the
// message is what `LARQL_MOE_INLINE_DIAG=1` prints: a diagnostic naming the
// wrong precondition sends the next reader to the wrong file.

/// Owns the expert bytes so a layer can borrow them for one assertion.
struct PreconditionFixture {
    gate_up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
}

const P_HIDDEN: usize = 256;
const P_INTER: usize = 128;
const P_TOP_K: usize = 2;

fn precondition_fixture() -> PreconditionFixture {
    let (gate_up, down) = make_q4k_experts(P_HIDDEN, P_INTER, 4);
    PreconditionFixture { gate_up, down }
}

impl PreconditionFixture {
    /// The admitting shape. Each test mutates exactly the one field whose
    /// refusal it is checking.
    fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            experts_gate_up: self.gate_up.iter().map(|v| v.as_slice()).collect(),
            experts_down: self.down.iter().map(|v| v.as_slice()).collect(),
            expert_scales: larql_compute::MoeExpertScales::Inline,
            fused_row_layout: larql_compute::MoeFusedRowLayout::ContiguousHalves,
            // `default()` is `gemma4_hybrid()`, which carries a post-expert
            // norm — stated explicitly here so the fixture IS the
            // identity-combine class this path serves.
            routing_policy: MoeRoutingPolicy {
                post_expert_norm: larql_compute::MoePostExpertNormPolicy::None,
                ..MoeRoutingPolicy::gemma4_hybrid()
            },
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q4_K,
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: self.gate_up.len(),
            top_k: P_TOP_K,
            intermediate_size: P_INTER,
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
        }
    }
}

fn precondition_ctx() -> MoeInterleaveCtx<'static> {
    MoeInterleaveCtx {
        layer_idx: 0,
        num_layers: 1,
        hidden: P_HIDDEN,
        inter: P_INTER,
        inter_padded: P_INTER,
        defer_ffn_for_split: false,
        stage_timing_split: false,
        layer_in_snapshot: None,
        dump_l0_dir: None,
    }
}

/// Assert the precondition check refuses, and that the reason it prints
/// names the actual cause.
fn assert_refuses(
    layer: &FullPipelineLayer<'_>,
    ctx: &MoeInterleaveCtx<'_>,
    scratch: &MoeScratch,
    needle: &str,
) {
    match MetalBackend::inline_moe_preconditions(layer, ctx, scratch) {
        Ok(_) => panic!("expected a refusal mentioning {needle:?}, but the layer was admitted"),
        Err(msg) => assert!(
            msg.contains(needle),
            "refused for the wrong reason: got {msg:?}, expected something mentioning {needle:?}"
        ),
    }
}

#[test]
fn admits_a_servable_inline_moe_layer() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    if let Err(e) = MetalBackend::inline_moe_preconditions(&layer, &precondition_ctx(), &scratch) {
        panic!(
            "the fixture must be the servable shape, else every refusal \
             assertion below is vacuous; refused with: {e}"
        );
    }
}

#[test]
fn refuses_a_layer_without_moe_weights() {
    let m = backend();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer::default();
    assert_refuses(&layer, &precondition_ctx(), &scratch, "no MoE weights");
}

#[test]
fn refuses_a_remote_ffn_layer() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ffn_is_remote: true,
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "ffn_is_remote");
}

/// The two context flags that mean "another arm owns this command buffer".
/// `stage_timing_split` in particular is why `LARQL_PROFILE_SPLIT=1` must
/// not be used to diagnose the merged-CB path — it disables it.
#[test]
fn refuses_when_another_arm_owns_the_command_buffer() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };

    let mut ctx = precondition_ctx();
    ctx.defer_ffn_for_split = true;
    assert_refuses(&layer, &ctx, &scratch, "defer_ffn_for_split");

    let mut ctx = precondition_ctx();
    ctx.stage_timing_split = true;
    assert_refuses(&layer, &ctx, &scratch, "stage_timing_split");
}

/// A capture hook needs the intermediate values the merged CB never
/// materialises on the host, so the two are mutually exclusive.
#[test]
fn refuses_while_a_capture_hook_is_active() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    let snapshot = vec![0.0f32; P_HIDDEN];

    let mut ctx = precondition_ctx();
    ctx.layer_in_snapshot = Some(&snapshot);
    assert_refuses(&layer, &ctx, &scratch, "layer_in_snapshot");

    let mut ctx = precondition_ctx();
    ctx.dump_l0_dir = Some("/tmp/does-not-need-to-exist");
    assert_refuses(&layer, &ctx, &scratch, "dump_l0_dir");
}

/// The identity-combine class: anything that post-processes the combined
/// output is a different shape than the merged CB encodes.
#[test]
fn refuses_layers_outside_the_identity_combine_class() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let ctx = precondition_ctx();

    let mut moe = f.moe();
    moe.routing_policy.post_expert_norm = larql_compute::MoePostExpertNormPolicy::RmsNorm;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "post_expert_norm");

    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        moe_combined_output_norm: true,
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "moe_combined_output_norm");

    // A layer scalar of 0 or 1 is absorbed; anything else must be applied
    // to the whole layer output, which this path does not do.
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        layer_scalar: 0.5,
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "layer_scalar");
}

/// A `Gated` layer carrying expert biases has no kernel — `ClampedGlu` is
/// the biased shape that does. Both directions are asserted so the check
/// cannot be satisfied by refusing every biased layer.
#[test]
fn refuses_a_gated_layer_with_expert_biases_but_admits_clamped_glu() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);
    let ctx = precondition_ctx();
    let gu_bias = vec![0.1f32; f.gate_up.len() * 2 * P_INTER];

    let mut gated = f.moe();
    gated.gate_rule = MoeGateRule::Gated(Activation::GeluTanh);
    gated.experts_gate_up_bias = &gu_bias;
    let layer = FullPipelineLayer {
        moe: Some(gated),
        ..Default::default()
    };
    assert_refuses(&layer, &ctx, &scratch, "no kernel");

    let mut clamped = f.moe();
    clamped.gate_rule = MoeGateRule::ClampedGlu {
        limit: 7.0,
        alpha: 1.702,
    };
    clamped.experts_gate_up_bias = &gu_bias;
    let layer = FullPipelineLayer {
        moe: Some(clamped),
        ..Default::default()
    };
    assert!(
        MetalBackend::inline_moe_preconditions(&layer, &ctx, &scratch).is_ok(),
        "ClampedGlu IS the biased shape with a kernel"
    );
}

/// Every dimension the scratch was allocated against. These are the checks
/// that stop a layer writing into slots sized for a different shape, and
/// each names the mismatched pair so the diagnostic is actionable.
#[test]
fn refuses_each_shape_that_disagrees_with_the_scratch() {
    let m = backend();
    let f = precondition_fixture();
    let scratch = MoeScratch::new_public(&m, P_TOP_K, P_HIDDEN, P_INTER);

    let mut moe = f.moe();
    moe.top_k = P_TOP_K + 1;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "top_k");

    let mut moe = f.moe();
    moe.intermediate_size = P_INTER * 2;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "intermediate_size");

    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    let mut ctx = precondition_ctx();
    ctx.hidden = P_HIDDEN * 2;
    assert_refuses(&layer, &ctx, &scratch, "hidden");

    let mut moe = f.moe();
    moe.expert_data_format = QuantFormat::Q6_K;
    let layer = FullPipelineLayer {
        moe: Some(moe),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &scratch, "expert_data_format");
}

/// The stored row width the scratch was sized for. A writer-padded bank
/// (gpt-oss stores 2880-wide rows at 3072) must match the scratch's
/// `weight_cols`, or every expert row is read at the wrong stride — the
/// one precondition here whose mismatch is silent rather than loud.
#[test]
fn refuses_a_stored_row_width_that_disagrees_with_the_scratch() {
    let m = backend();
    let f = precondition_fixture();
    // Scratch sized for a padded store; the fixture's bank is unpadded.
    let padded = MoeScratch::new_public_with_format(
        &m,
        P_TOP_K,
        P_HIDDEN,
        P_INTER,
        QuantFormat::Q4_K,
        P_HIDDEN + 256,
    );
    let layer = FullPipelineLayer {
        moe: Some(f.moe()),
        ..Default::default()
    };
    assert_refuses(&layer, &precondition_ctx(), &padded, "gate_up_cols");
}
