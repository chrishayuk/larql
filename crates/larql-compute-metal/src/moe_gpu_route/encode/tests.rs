//! Tests for [`super`] — the GPU-route admission check, the per-layer
//! descriptor-table cache, and the S1 production layer encode.
//!
//! These three sit between the ladder's kernel gates (which prove the
//! expert mathematics) and the real decode loop (which proves nothing in
//! isolation). What they own is *admission and assembly*: whether a layer
//! may take the GPU route at all, whether the table it uses is the right
//! one, and whether the assembled encode produces the same answer as the
//! CPU-routed path it replaces.
//!
//! `gpu_route_supported` gets the most attention because it is the only
//! thing standing between an unsupported policy and a silently wrong
//! route. Each refusal is pinned to its own cause, and the admitting case
//! is pinned beside them — a blanket `false` would satisfy every refusal
//! assertion on its own, so the positive case is what makes the set
//! meaningful.

use larql_compute::cpu::ops::q4_common::quantize_q6_k;
use larql_compute::{
    MoeExpertScalePolicy, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeInputSource,
    MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeRoutingPolicy,
    MoeTopKWeightPolicy, MoeWeightLayout, QuantFormat,
};

use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;

const PAGE: usize = 16384;
const NUM_EXPERTS: usize = 32;
const HIDDEN: usize = 256;
const INTER: usize = 256;
const TOP_K: usize = 4;
const ROW_BYTES: usize = HIDDEN / 256 * 210;
const GU_EXPERT_BYTES: usize = 2 * INTER * ROW_BYTES;
const DN_EXPERT_BYTES: usize = HIDDEN * (INTER / 256 * 210);

fn aligned_backing(size: usize) -> (Vec<u8>, usize) {
    let mem = vec![0u8; size + PAGE];
    let off = mem.as_ptr().align_offset(PAGE);
    (mem, off)
}

/// A registered Q6_K expert bank plus the router tensors a layer needs.
/// Deliberately the same shape the rung-E gate uses, so a divergence here
/// is about this module rather than about the fixture.
struct Fixture {
    _gu_mem: Vec<u8>,
    _dn_mem: Vec<u8>,
    gu_ptr: *const u8,
    dn_ptr: *const u8,
    router_w: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
    norm_w: Vec<f32>,
}

fn build_fixture(metal: &MetalBackend) -> Fixture {
    let gu_size = NUM_EXPERTS * GU_EXPERT_BYTES;
    let dn_size = NUM_EXPERTS * DN_EXPERT_BYTES;
    let (mut gu_mem, gu_off) = aligned_backing(gu_size);
    let (mut dn_mem, dn_off) = aligned_backing(dn_size);
    for e in 0..NUM_EXPERTS {
        let gu_vals: Vec<f32> = (0..2 * INTER * HIDDEN)
            .map(|i| ((e * 977 + i) as f32 * 0.011).sin() * 0.3)
            .collect();
        let dn_vals: Vec<f32> = (0..HIDDEN * INTER)
            .map(|i| ((e * 613 + i) as f32 * 0.017).cos() * 0.3)
            .collect();
        let gq = quantize_q6_k(&gu_vals);
        let dq = quantize_q6_k(&dn_vals);
        gu_mem[gu_off + e * GU_EXPERT_BYTES..gu_off + (e + 1) * GU_EXPERT_BYTES]
            .copy_from_slice(&gq);
        dn_mem[dn_off + e * DN_EXPERT_BYTES..dn_off + (e + 1) * DN_EXPERT_BYTES]
            .copy_from_slice(&dq);
    }
    let gu_region = &gu_mem[gu_off..gu_off + gu_size];
    let dn_region = &dn_mem[dn_off..dn_off + dn_size];
    assert!(metal.bufs.register_region(gu_region));
    assert!(metal.bufs.register_region(dn_region));

    Fixture {
        gu_ptr: gu_region.as_ptr(),
        dn_ptr: dn_region.as_ptr(),
        router_w: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.0007).sin() * 0.05)
            .collect(),
        router_bias: (0..NUM_EXPERTS)
            .map(|e| (e as f32 * 0.9).sin() * 0.3)
            .collect(),
        gate_up_bias: (0..NUM_EXPERTS * 2 * INTER)
            .map(|i| ((i as f32) * 0.023).sin() * 0.2)
            .collect(),
        down_bias: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.029).cos() * 0.2)
            .collect(),
        norm_w: (0..HIDDEN)
            .map(|i| 1.0 + (i as f32 * 0.01).sin() * 0.1)
            .collect(),
        _gu_mem: gu_mem,
        _dn_mem: dn_mem,
    }
}

impl Fixture {
    fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            expert_scales: MoeExpertScales::Inline,
            fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up: (0..NUM_EXPERTS)
                .map(|e| unsafe {
                    std::slice::from_raw_parts(
                        self.gu_ptr.add(e * GU_EXPERT_BYTES),
                        GU_EXPERT_BYTES,
                    )
                })
                .collect(),
            experts_down: (0..NUM_EXPERTS)
                .map(|e| unsafe {
                    std::slice::from_raw_parts(
                        self.dn_ptr.add(e * DN_EXPERT_BYTES),
                        DN_EXPERT_BYTES,
                    )
                })
                .collect(),
            routing_policy: MoeRoutingPolicy {
                expert_input: MoeInputSource::Residual,
                router_input: MoeInputSource::Residual,
                router_norm: MoeRouterNormPolicy::None,
                selected_weight: MoeTopKWeightPolicy::RawSoftmax,
                expert_scale: MoeExpertScalePolicy::None,
                post_expert_norm: MoePostExpertNormPolicy::None,
            },
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q6_K,
            router_proj: &self.router_w,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            intermediate_size: INTER,
            router_bias: &self.router_bias,
            experts_gate_up_bias: &self.gate_up_bias,
            experts_down_bias: &self.down_bias,
            gate_rule: MoeGateRule::ClampedGlu {
                limit: 7.0,
                alpha: 1.702,
            },
        }
    }
}

fn scratch_for(metal: &MetalBackend) -> MoeScratch {
    MoeScratch::new(
        &metal.bufs,
        TOP_K,
        HIDDEN,
        INTER,
        QuantFormat::Q6_K,
        HIDDEN, // weight_cols == hidden: no writer padding at this shape
    )
}

fn residual(seed: u32) -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 * 1e-9).sin())
        .collect()
}

// ── gpu_route_supported: the admission check ─────────────────────────────

#[test]
fn admits_a_supported_q6k_layer() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let s = scratch_for(&metal);
    assert!(
        metal.gpu_route_supported(&f.moe(), &s),
        "the fixture is the shape the GPU route was built for; if this \
         refuses, every refusal assertion below is vacuous"
    );
}

#[test]
fn refuses_q6k_with_interleaved_fused_rows() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let s = scratch_for(&metal);
    let mut moe = f.moe();
    moe.fused_row_layout = MoeFusedRowLayout::Interleaved;
    assert!(
        !metal.gpu_route_supported(&moe, &s),
        "the Q6_K grouped kernel reads contiguous halves; interleaved rows \
         would be read at the wrong stride"
    );
}

#[test]
fn refuses_a_format_the_descriptor_arm_does_not_serve() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    // Q4_K is a legitimate expert format elsewhere; the descriptor arm
    // just does not serve it, and must say so rather than mis-bind.
    let s = MoeScratch::new(&metal.bufs, TOP_K, HIDDEN, INTER, QuantFormat::Q4_K, HIDDEN);
    assert!(!metal.gpu_route_supported(&f.moe(), &s));
}

#[test]
fn refuses_mxfp4_without_paired_scales() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let s = MoeScratch::new(
        &metal.bufs,
        TOP_K,
        HIDDEN,
        INTER,
        QuantFormat::MXFP4,
        HIDDEN,
    );
    let mut moe = f.moe();
    moe.expert_data_format = QuantFormat::MXFP4;
    // `expert_scales` stays Inline — an MXFP4 bank without its e8m0
    // partner stream cannot be dequantised by the split kernel.
    assert!(
        !metal.gpu_route_supported(&moe, &s),
        "native MXFP4 needs paired scales; Inline is the transcoded shape"
    );
}

#[test]
fn refuses_a_post_expert_norm_it_does_not_encode() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let s = scratch_for(&metal);
    let mut moe = f.moe();
    moe.routing_policy.post_expert_norm = MoePostExpertNormPolicy::RmsNorm;
    assert!(!metal.gpu_route_supported(&moe, &s));
}

#[test]
fn refuses_expert_counts_and_top_k_outside_the_select_kernel() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let s = scratch_for(&metal);

    let mut too_many = f.moe();
    too_many.num_experts = crate::shaders::moe_router_select::MAX_EXPERTS + 1;
    assert!(!metal.gpu_route_supported(&too_many, &s));

    let mut zero_k = f.moe();
    zero_k.top_k = 0;
    assert!(!metal.gpu_route_supported(&zero_k, &s));

    let mut wide_k = f.moe();
    wide_k.top_k = crate::shaders::moe_router_select::MAX_TOP_K + 1;
    assert!(!metal.gpu_route_supported(&wide_k, &s));
}

#[test]
fn refuses_a_router_projection_of_the_wrong_shape() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let s = scratch_for(&metal);
    let short = vec![0.0f32; NUM_EXPERTS * HIDDEN - 1];
    let mut moe = f.moe();
    moe.router_proj = &short;
    assert!(
        !metal.gpu_route_supported(&moe, &s),
        "the projection kernel derives its grid from num_experts x hidden"
    );
}

/// The padded-row rule: `Identity` binds `h_post_attn` straight into the
/// matvec, so when the store's rows are wider than `hidden` there is no
/// zero tail to make the padding harmless. Only the norm transform, which
/// writes through `scratch.x_buf`, may serve that shape.
#[test]
fn refuses_padded_rows_under_identity_but_admits_them_under_the_norm() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let padded = MoeScratch::new(
        &metal.bufs,
        TOP_K,
        HIDDEN,
        INTER,
        QuantFormat::Q6_K,
        HIDDEN + 256, // writer-padded row width, as gpt-oss has
    );

    let identity = f.moe();
    assert!(
        !metal.gpu_route_supported(&identity, &padded),
        "Identity has no staging buffer to supply the zero tail"
    );

    let mut normed = f.moe();
    normed.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    normed.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
    normed.pre_experts_norm = &f.norm_w;
    assert!(
        metal.gpu_route_supported(&normed, &padded),
        "the norm writes into x_buf, whose tail is permanently zero"
    );
}

// ── descriptor_table_for_layer: the per-layer cache ──────────────────────

#[test]
fn descriptor_table_is_built_once_per_layer_and_bank() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let moe = f.moe();

    let a = metal
        .descriptor_table_for_layer(0, &moe, INTER, HIDDEN)
        .expect("table builds for a well-formed bank");
    let b = metal
        .descriptor_table_for_layer(0, &moe, INTER, HIDDEN)
        .expect("second call hits the cache");
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "the same (layer, bank) must return the cached Arc, not a rebuild"
    );

    // A different layer index is a different entry even for the same bank.
    let other = metal
        .descriptor_table_for_layer(1, &moe, INTER, HIDDEN)
        .expect("table builds for layer 1");
    assert!(!std::sync::Arc::ptr_eq(&a, &other));
}

#[test]
fn descriptor_table_refuses_a_bank_it_cannot_describe() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let mut moe = f.moe();
    // An empty bank has no first slice to key the cache on, and nothing
    // to describe — refusing keeps the caller on the CPU arm.
    moe.experts_gate_up = Vec::new();
    assert!(metal
        .descriptor_table_for_layer(9, &moe, INTER, HIDDEN)
        .is_none());
}

// ── encode_moe_layer_gpu_route: the assembled S1 encode ──────────────────

/// Run the production GPU-route encode for one layer and read back the
/// result, mirroring what `handle_moe_interleave` does per MoE layer.
fn run_gpu_route(metal: &MetalBackend, moe: &MoeLayerWeights<'_>, x: &[f32]) -> Vec<f32> {
    let s = scratch_for(metal);
    let table = metal
        .descriptor_table_for_layer(0, moe, INTER, HIDDEN)
        .expect("table");
    let h_post_attn = metal.bufs.transient_from_f32(x);
    let new_h = metal.bufs.output((HIDDEN * 4) as u64);

    let cmd = metal.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    metal.encode_moe_layer_gpu_route(enc, moe, &s, &table, &h_post_attn, &new_h, 1e-6);
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(
        cmd,
        "crates/larql-compute-metal/src/moe_gpu_route/encode/tests.rs:388",
    )
    .expect("command buffer completed");
    crate::buffers::read_buffer_f32(&new_h, HIDDEN)
}

/// The assembled encode must agree with the CPU-routed control the ladder
/// already pins — same layer, same input, routing decided on the GPU.
#[test]
fn gpu_route_layer_matches_the_cpu_routed_control() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let moe = f.moe();
    let x = residual(11);

    let got = run_gpu_route(&metal, &moe, &x);
    let want = metal
        .moe_layer_forward_control(&x, &moe, &x)
        .expect("the CPU-routed control runs on this device");

    assert_eq!(got.len(), want.len());
    let max_rel = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-3))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel < 1e-3,
        "GPU-routed layer diverges from the CPU-routed control: max rel {max_rel:.3e}\n\
         got[..4]={:?}\nwant[..4]={:?}",
        &got[..4],
        &want[..4]
    );
    assert!(got.iter().all(|v| v.is_finite()));
}

/// The pre-experts-norm transform is a different code path through the
/// same function: it norms into the staging buffer and routes on that,
/// rather than binding the residual directly.
#[test]
fn gpu_route_layer_runs_the_pre_experts_norm_transform() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let mut moe = f.moe();
    moe.routing_policy.router_input = MoeInputSource::PreExpertsNorm;
    moe.routing_policy.expert_input = MoeInputSource::PreExpertsNorm;
    moe.pre_experts_norm = &f.norm_w;
    let x = residual(23);

    let normed = run_gpu_route(&metal, &moe, &x);
    assert!(normed.iter().all(|v| v.is_finite()));

    // It must actually differ from the identity transform on the same
    // input — otherwise the norm dispatch could be a no-op and this test
    // would pass on a broken encode.
    let identity = run_gpu_route(&metal, &f.moe(), &x);
    let differs = normed
        .iter()
        .zip(&identity)
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        differs,
        "the norm transform produced the identity result — the rms_norm \
         dispatch is not reaching the router input"
    );
}

/// `Gated` is the other gate rule the activation dispatch has to serve,
/// and both of its pipelines (gelu-tanh and plain) are selected by an
/// architecture fact rather than a model name.
#[test]
fn gpu_route_layer_serves_both_gated_activations() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let x = residual(37);

    let mut silu = f.moe();
    silu.gate_rule = MoeGateRule::Gated(larql_compute::Activation::Silu);
    let silu_out = run_gpu_route(&metal, &silu, &x);
    assert!(silu_out.iter().all(|v| v.is_finite()));

    let mut gelu = f.moe();
    gelu.gate_rule = MoeGateRule::Gated(larql_compute::Activation::GeluTanh);
    let gelu_out = run_gpu_route(&metal, &gelu, &x);
    assert!(gelu_out.iter().all(|v| v.is_finite()));

    let differs = silu_out
        .iter()
        .zip(&gelu_out)
        .any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(
        differs,
        "silu and gelu-tanh produced identical output — the activation \
         selection is not reaching the dispatch"
    );
}

/// Renormalised top-k weights and a per-expert router scale are separate
/// policy inputs to the select kernel; both are read here, not baked.
#[test]
fn gpu_route_layer_honours_the_weight_and_scale_policies() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let f = build_fixture(&metal);
    let x = residual(51);
    let base = run_gpu_route(&metal, &f.moe(), &x);

    let mut renorm = f.moe();
    renorm.routing_policy.selected_weight = MoeTopKWeightPolicy::RenormalizedSoftmax;
    let renorm_out = run_gpu_route(&metal, &renorm, &x);
    assert!(renorm_out.iter().all(|v| v.is_finite()));
    assert!(
        renorm_out
            .iter()
            .zip(&base)
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "renormalising the top-k weights must change the combine"
    );

    let per_expert: Vec<f32> = (0..NUM_EXPERTS).map(|e| 1.0 + e as f32 * 0.01).collect();
    let mut scaled = f.moe();
    scaled.routing_policy.expert_scale = MoeExpertScalePolicy::PerExpert;
    scaled.router_per_expert_scale = &per_expert;
    let scaled_out = run_gpu_route(&metal, &scaled, &x);
    assert!(scaled_out.iter().all(|v| v.is_finite()));
}
