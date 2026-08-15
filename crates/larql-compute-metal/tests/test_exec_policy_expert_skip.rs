#![cfg(target_os = "macos")]

//! The production gate for the execution-policy seam: an expert group
//! can be deleted on the REAL Metal dispatch path, and the BW10 ledger
//! reports exactly the bytes that deletion avoided.
//!
//! Five proofs, all of them cheap — no model, no generation, one
//! synthetic layer (or a four-layer chain) per assertion:
//!
//! 1. DESCRIPTOR ARM — the `LARQL_GPU_ROUTE=1` serve path. Forced skip →
//!    output is BITWISE `h_post_attn` (residual identity), the ledger
//!    records exactly one avoided group and zero moved expert bytes, and
//!    uninstalling the policy restores canonical execution with the
//!    bytes back.
//! 2. ZERO-COPY ARM — the CPU-routed production path, same four claims.
//!    Both arms are gated because both are reachable in production and a
//!    seam honoured on one of them is a seam with a hole in it.
//! 3. NEGATIVE CONTROL — a mask armed on a layer the dispatch never
//!    reports must change nothing, bit for bit and byte for byte.
//!    Without this, proof 1 is also consistent with "installing ANY
//!    policy breaks the layer".
//! 4. LAYER ADDRESSING — in a four-layer chain with layer 2 masked,
//!    exactly one group is skipped and three execute. Proof 1 alone
//!    cannot distinguish "the seam addresses layers" from "the seam is
//!    all-or-nothing".
//! 5. TOKEN ADDRESSING — the same layer skips on one declared decode
//!    step and runs on the next. This is the "one known layer, one known
//!    token" statement in full.
//!
//! What is deliberately NOT claimed here: any latency result. These
//! assertions are about semantics and byte accounting. A wall-time claim
//! needs a steady-state A/B on a real model, and this file would be the
//! wrong instrument for it.

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use std::sync::{Arc, Mutex};

use larql_compute::cpu::ops::q4_common::quantize_q6_k;
use larql_compute::exec_policy::{
    install,
    policies::{LayerStepMask, StepSelector},
    step, uninstall,
};
use larql_compute::movement_ledger::{
    bytes, decisions, DecisionCounts, Phase, PhaseScope, Surface,
};
use larql_compute::{
    MoeExpertScalePolicy, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeInputSource,
    MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeRoutingPolicy,
    MoeTopKWeightPolicy, MoeWeightLayout, QuantFormat,
};
use larql_compute_metal::MetalBackend;

/// The ledger counters and the policy registry are process-wide, and
/// Cargo runs the tests in this binary on parallel threads. Every test
/// here reads a delta across a window it opens itself, so they must not
/// overlap.
static GATE_LOCK: Mutex<()> = Mutex::new(());

const PAGE: usize = 16384;
const NUM_EXPERTS: usize = 32;
const HIDDEN: usize = 256;
const INTER: usize = 256;
const TOP_K: usize = 4;
const ROW_BYTES: usize = HIDDEN / 256 * 210;
const GU_EXPERT_BYTES: usize = 2 * INTER * ROW_BYTES;
const DN_EXPERT_BYTES: usize = HIDDEN * (INTER / 256 * 210);
const GLU_LIMIT: f32 = 7.0;
const GLU_ALPHA: f32 = 1.702;

/// What one routed group of `TOP_K` experts costs, derived from the
/// FIXTURE's own byte layout rather than copied from the ledger's shape
/// arithmetic. That independence is the point: if the two ever disagree,
/// this assertion fails instead of confirming itself. HIDDEN and INTER
/// are exact multiples of the Q6_K block here, so there is no padding
/// term to account for and physical equals semantic.
const GROUP_PHYSICAL_BYTES: u64 = (TOP_K * (GU_EXPERT_BYTES + DN_EXPERT_BYTES)) as u64;

/// The single-layer test wrappers report themselves to the seam as layer
/// 0 (`moe_gpu_route::forward::SYNTHETIC_LAYER`).
const WRAPPER_LAYER: usize = 0;
/// A layer index the single-layer wrappers never report — the negative
/// control's target.
const UNREACHED_LAYER: usize = 1;

fn aligned_backing(size: usize) -> (Vec<u8>, usize) {
    let mem = vec![0u8; size + PAGE];
    let off = mem.as_ptr().align_offset(PAGE);
    (mem, off)
}

struct Fixture {
    _gu_mem: Vec<u8>,
    _dn_mem: Vec<u8>,
    gu_ptr: *const u8,
    dn_ptr: *const u8,
    router_w: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
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
    assert!(metal.bufs().register_region(gu_region));
    assert!(metal.bufs().register_region(dn_region));

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
                limit: GLU_LIMIT,
                alpha: GLU_ALPHA,
            },
        }
    }
}

fn router_input(seed: u32) -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 * 1e-9).sin())
        .collect()
}

fn h_post_attn() -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| ((i as f32) * 0.041).sin() * 0.5)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Everything one window of the ledger says about the expert surface.
#[derive(Debug)]
struct Window {
    physical_touched: u64,
    decisions: DecisionCounts,
    surface_fired: u64,
}

/// Run `f` inside a ledger window and report what it moved and decided.
/// Deltas, not resets: the counters are process-wide and production
/// readers take deltas for exactly this reason.
fn window<T>(f: impl FnOnce() -> T) -> (T, Window) {
    let b0 = bytes::snapshot();
    let d0 = decisions::snapshot();
    let f0 = Surface::MoeExperts.fired();
    let out = f();
    let b1 = bytes::snapshot();
    (
        out,
        Window {
            physical_touched: b1.physical_touched - b0.physical_touched,
            decisions: d0.delta(&decisions::snapshot()),
            surface_fired: Surface::MoeExperts.fired() - f0,
        },
    )
}

/// Proof 1 — the descriptor (GPU-route) arm.
#[test]
fn descriptor_arm_honours_skip_and_the_ledger_sees_the_avoided_bytes() {
    let _g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    uninstall();
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let h = h_post_attn();
    let x = router_input(7);

    // ── Canonical: the control every skip claim is measured against.
    let (canonical, canon_win) = window(|| {
        metal
            .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
            .expect("canonical forward")
    });
    assert_eq!(canon_win.decisions.requested, 1);
    assert_eq!(canon_win.decisions.executed, 1);
    assert_eq!(canon_win.decisions.skipped, 0);
    assert_eq!(canon_win.physical_touched, GROUP_PHYSICAL_BYTES);
    assert_eq!(canon_win.surface_fired, 1);
    assert_ne!(
        bits(&canonical),
        bits(&h),
        "fixture degenerated: the canonical layer must change the residual, \
         or the skip assertion below is trivially satisfied"
    );

    // ── Skipped.
    let (skipped, skip_win) = {
        let _policy = install(Arc::new(LayerStepMask::new([WRAPPER_LAYER])));
        window(|| {
            metal
                .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
                .expect("skipped forward")
        })
    };

    assert_eq!(
        bits(&skipped),
        bits(&h),
        "a skipped expert group must leave the residual BITWISE untouched — \
         the additive combine's identity, produced by the same combine \
         kernel at k = 0"
    );
    assert_eq!(skip_win.decisions.requested, 1);
    assert_eq!(skip_win.decisions.executed, 0);
    assert_eq!(
        skip_win.decisions.skipped, 1,
        "exactly one group, not zero, not two"
    );
    assert_eq!(
        skip_win.decisions.physical_avoided, GROUP_PHYSICAL_BYTES,
        "the avoided count must equal what the canonical arm actually moved"
    );
    assert_eq!(
        skip_win.decisions.physical_avoided, canon_win.physical_touched,
        "avoided and touched must be the same operation priced the same way"
    );
    assert_eq!(
        skip_win.physical_touched, 0,
        "no expert weight may be streamed by a skipped group"
    );
    assert_eq!(
        skip_win.surface_fired, 0,
        "coverage evidence is about bytes that MOVED — a skip is not coverage"
    );

    // ── Disarmed: canonical execution comes back, bit for bit.
    let (restored, restored_win) = window(|| {
        metal
            .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
            .expect("restored forward")
    });
    assert_eq!(
        bits(&restored),
        bits(&canonical),
        "uninstalling the policy must restore the canonical numerics exactly"
    );
    assert_eq!(restored_win.physical_touched, GROUP_PHYSICAL_BYTES);
    assert_eq!(restored_win.decisions.skipped, 0);
}

/// Proof 2 — the zero-copy (CPU-routed) arm. Same four claims: a seam
/// honoured on only one production path is a seam with a hole in it.
#[test]
fn zero_copy_arm_honours_skip_and_the_ledger_sees_the_avoided_bytes() {
    let _g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    uninstall();
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let h = h_post_attn();
    let x = router_input(11);

    let (canonical, canon_win) = window(|| {
        metal
            .moe_layer_forward_control(&x, &moe, &h)
            .expect("canonical control forward")
    });
    assert_eq!(canon_win.decisions.executed, 1);
    assert_eq!(canon_win.physical_touched, GROUP_PHYSICAL_BYTES);
    assert_ne!(bits(&canonical), bits(&h), "fixture degenerated");

    let (skipped, skip_win) = {
        let _policy = install(Arc::new(LayerStepMask::new([WRAPPER_LAYER])));
        window(|| {
            metal
                .moe_layer_forward_control(&x, &moe, &h)
                .expect("skipped control forward")
        })
    };
    assert_eq!(
        bits(&skipped),
        bits(&h),
        "the zero-copy arm must produce the same residual identity"
    );
    assert_eq!(skip_win.decisions.skipped, 1);
    assert_eq!(skip_win.decisions.physical_avoided, GROUP_PHYSICAL_BYTES);
    assert_eq!(skip_win.physical_touched, 0);
    assert_eq!(skip_win.surface_fired, 0);

    let (restored, _) = window(|| {
        metal
            .moe_layer_forward_control(&x, &moe, &h)
            .expect("restored control forward")
    });
    assert_eq!(bits(&restored), bits(&canonical));
}

/// Proof 3 — the negative control. A mask armed on a layer this dispatch
/// never reports must change NOTHING. Without it, proof 1 is equally
/// consistent with "installing any policy at all breaks the layer".
#[test]
fn a_mask_on_an_unreached_layer_changes_nothing() {
    let _g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    uninstall();
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let h = h_post_attn();
    let x = router_input(23);

    let (canonical, _) = window(|| {
        metal
            .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
            .expect("canonical forward")
    });

    let (masked, masked_win) = {
        let _policy = install(Arc::new(LayerStepMask::new([UNREACHED_LAYER])));
        window(|| {
            metal
                .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
                .expect("masked forward")
        })
    };

    assert_eq!(
        bits(&masked),
        bits(&canonical),
        "a policy armed elsewhere must not perturb this layer by a single bit"
    );
    assert_eq!(masked_win.decisions.skipped, 0);
    assert_eq!(masked_win.decisions.executed, 1);
    assert_eq!(masked_win.physical_touched, GROUP_PHYSICAL_BYTES);
}

/// Proof 4 — layer addressing. Four chained layers, one masked: exactly
/// one group's bytes are avoided and three are moved. A seam that were
/// all-or-nothing would pass proof 1 and fail here.
#[test]
fn one_layer_of_a_chain_skips_while_the_rest_execute() {
    let _g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    uninstall();
    const LAYERS: usize = 4;
    const MASKED: usize = 2;
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let x = router_input(31);

    let (canonical, canon_win) = window(|| {
        metal
            .moe_token_forward_descriptor(&x, &moe, &table, LAYERS, true)
            .expect("canonical chain")
    });
    assert_eq!(canon_win.decisions.requested, LAYERS as u64);
    assert_eq!(canon_win.decisions.executed, LAYERS as u64);
    assert_eq!(
        canon_win.physical_touched,
        GROUP_PHYSICAL_BYTES * LAYERS as u64
    );

    let (masked, masked_win) = {
        let _policy = install(Arc::new(LayerStepMask::new([MASKED])));
        window(|| {
            metal
                .moe_token_forward_descriptor(&x, &moe, &table, LAYERS, true)
                .expect("masked chain")
        })
    };

    assert_eq!(masked_win.decisions.requested, LAYERS as u64);
    assert_eq!(
        masked_win.decisions.skipped, 1,
        "exactly the one masked layer, out of {LAYERS}"
    );
    assert_eq!(masked_win.decisions.executed, (LAYERS - 1) as u64);
    assert_eq!(masked_win.decisions.physical_avoided, GROUP_PHYSICAL_BYTES);
    assert_eq!(
        masked_win.physical_touched,
        GROUP_PHYSICAL_BYTES * (LAYERS - 1) as u64
    );
    assert_eq!(
        masked_win.physical_touched + masked_win.decisions.physical_avoided,
        canon_win.physical_touched,
        "touched + avoided must reconstruct the canonical arm's traffic"
    );
    assert_ne!(
        bits(&masked.out),
        bits(&canonical.out),
        "deleting a mid-chain layer's experts must change the chain's output"
    );
}

/// Proof 5 — token addressing. The same layer skips on decode step 0 and
/// runs on decode step 1: "one known layer, one known token", stated
/// exactly rather than as a first-visit approximation.
#[test]
fn the_same_layer_skips_on_one_declared_token_and_runs_on_the_next() {
    let _g = GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    uninstall();
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let h = h_post_attn();
    let x = router_input(43);

    // The test IS the driver loop here: it declares the phase and crosses
    // the token boundaries, exactly as `MetalBackend::decode_token_*`
    // does around its own `TokenScope`.
    let _phase = PhaseScope::new(Phase::Decode);
    step::reset();

    let _policy = install(Arc::new(
        LayerStepMask::new([WRAPPER_LAYER])
            .with_phase(Phase::Decode)
            .with_steps(StepSelector::Exactly(0)),
    ));

    step::advance(); // decode step 0
    let (token0, win0) = window(|| {
        metal
            .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
            .expect("token 0")
    });
    assert_eq!(win0.decisions.skipped, 1, "step 0 is the armed token");
    assert_eq!(bits(&token0), bits(&h));

    step::advance(); // decode step 1
    let (token1, win1) = window(|| {
        metal
            .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
            .expect("token 1")
    });
    assert_eq!(
        win1.decisions.skipped, 0,
        "the SAME layer must run canonically on an unarmed token"
    );
    assert_eq!(win1.decisions.executed, 1);
    assert_eq!(win1.physical_touched, GROUP_PHYSICAL_BYTES);
    assert_ne!(bits(&token1), bits(&h));

    drop(_policy);
    step::reset();
}
