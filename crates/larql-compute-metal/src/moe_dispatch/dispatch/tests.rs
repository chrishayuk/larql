//! Tests for [`super`] — the staged GPU expert dispatch's refusal paths.
//!
//! The happy path through `gpu_moe_dispatch_with_scratch` is driven end to
//! end by the layout-parity and closure gates. What is *not* driven there
//! is what the staging loop does when an expert cannot be supplied, and
//! those branches all write into pre-allocated scratch with raw pointer
//! copies — the failure mode of getting one wrong is a buffer overrun or a
//! silently stale slot, not a clean error.
//!
//! So each guard is exercised for what it protects:
//!
//! - a missing expert and an undersized gate/up slice, which must skip the
//!   slot rather than stage a partial one;
//! - a short down slice, whose tail must be zero-filled rather than left
//!   holding the previous token's expert;
//! - no valid experts at all, which must yield zeros rather than whatever
//!   the scratch happened to contain.

use larql_compute::cpu::ops::q4_common::quantize_q6_k;
use larql_compute::{
    Activation, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy,
    MoeWeightLayout, QuantFormat,
};

use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;

const HIDDEN: usize = 256;
const INTER: usize = 256;
const NUM_EXPERTS: usize = 4;
const TOP_K: usize = 2;
const ROW_BYTES: usize = HIDDEN / 256 * 210;
const GU_BYTES: usize = 2 * INTER * ROW_BYTES;
const DN_BYTES: usize = HIDDEN * (INTER / 256 * 210);

/// Q6_K bytes for one expert's fused gate/up and down tensors.
struct Bank {
    gu: Vec<Vec<u8>>,
    dn: Vec<Vec<u8>>,
    router_w: Vec<f32>,
}

fn bank() -> Bank {
    let gu = (0..NUM_EXPERTS)
        .map(|e| {
            let v: Vec<f32> = (0..2 * INTER * HIDDEN)
                .map(|i| ((e * 977 + i) as f32 * 0.011).sin() * 0.3)
                .collect();
            quantize_q6_k(&v)
        })
        .collect();
    let dn = (0..NUM_EXPERTS)
        .map(|e| {
            let v: Vec<f32> = (0..HIDDEN * INTER)
                .map(|i| ((e * 613 + i) as f32 * 0.017).cos() * 0.3)
                .collect();
            quantize_q6_k(&v)
        })
        .collect();
    Bank {
        gu,
        dn,
        router_w: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.0007).sin() * 0.05)
            .collect(),
    }
}

impl Bank {
    fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            experts_gate_up: self.gu.iter().map(|v| v.as_slice()).collect(),
            experts_down: self.dn.iter().map(|v| v.as_slice()).collect(),
            expert_scales: MoeExpertScales::Inline,
            fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
            routing_policy: MoeRoutingPolicy::gemma4_hybrid(),
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
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            gate_rule: MoeGateRule::Gated(Activation::Silu),
        }
    }
}

fn scratch(metal: &MetalBackend, top_k: usize) -> MoeScratch {
    MoeScratch::new(&metal.bufs, top_k, HIDDEN, INTER, QuantFormat::Q6_K, HIDDEN)
}

fn h(seed: u32) -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 * 1e-9).sin())
        .collect()
}

/// Every test funnels through this ONE non-generic wrapper.
///
/// `moe_block_for_layer` is generic over the supplier closure, so each
/// distinct closure type monomorphises a fresh copy of the whole dispatch
/// body — and llvm-cov counts every copy. A test that returns `None`
/// immediately would then contribute an instantiation in which the entire
/// staging path reads as uncovered, so *adding* tests could lower the
/// file's coverage. Taking `&dyn Fn` here collapses all of them into a
/// single instantiation whose coverage is the union of what they exercise.
/// (The lifetime parameter is erased at codegen, so it does not
/// re-introduce the problem the way a type parameter would.)
fn block<'w>(
    metal: &MetalBackend,
    x: &[f32],
    moe: &MoeLayerWeights<'_>,
    scratch: &MoeScratch,
    supply: &dyn Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
) -> Vec<f32> {
    metal.moe_block_for_layer(x, moe, 1e-6, scratch, supply)
}

/// Baseline: every expert supplied, so the guards below are refusals of a
/// path that otherwise produces real output. Without this the "returns
/// zeros" assertions could pass on a dispatch that never works.
#[test]
fn a_fully_supplied_bank_produces_non_zero_output() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let out = block(&metal, &h(1), &b.moe(), &s, &|e| {
        Some((b.gu[e].as_slice(), b.dn[e].as_slice()))
    });
    assert_eq!(out.len(), HIDDEN);
    assert!(
        out.iter().any(|v| v.abs() > 1e-6),
        "the happy path must produce signal, or every refusal test below is vacuous"
    );
}

/// No expert can be supplied: the block contributes nothing, and must say
/// so with zeros rather than leaving the caller reading stale scratch.
#[test]
fn no_supplied_experts_yields_zeros() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let out = block(&metal, &h(2), &b.moe(), &s, &|_| None);
    assert_eq!(out.len(), HIDDEN);
    assert!(
        out.iter().all(|v| *v == 0.0),
        "an unsupplied bank must contribute exactly zero, not scratch residue"
    );
}

/// One expert missing: the slot is skipped and the remaining selection
/// still runs. The result must differ from both the full bank and zeros.
#[test]
fn a_missing_expert_skips_its_slot_without_aborting_the_block() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let x = h(3);
    let moe = b.moe();
    let full = block(&metal, &x, &moe, &s, &|e| {
        Some((b.gu[e].as_slice(), b.dn[e].as_slice()))
    });

    // Drop an expert the router ACTUALLY selected. Picking a fixed index
    // would make the assertion depend on whether the fixture happens to
    // route through it — the test would then pass by luck on a dispatch
    // that ignored the `None`.
    let (ids, _) = larql_compute::cpu::ops::moe::moe_route_from_router_input(&x, &moe);
    let dropped = ids[0];
    let partial = block(&metal, &x, &moe, &s, &|e| {
        (e != dropped).then(|| (b.gu[e].as_slice(), b.dn[e].as_slice()))
    });
    assert_eq!(partial.len(), HIDDEN);
    assert!(partial.iter().all(|v| v.is_finite()));
    assert!(
        partial.iter().zip(&full).any(|(a, c)| (a - c).abs() > 1e-6),
        "dropping selected expert {dropped} must change the block's contribution"
    );
}

/// A gate/up slice too short to hold both halves cannot be staged: the
/// copy would read past its end. The slot is skipped instead.
#[test]
fn an_undersized_gate_up_slice_is_skipped_not_truncated() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let out = block(&metal, &h(4), &b.moe(), &s, &|e| {
        // Half a fused row: enough to look plausible, not enough to hold
        // gate AND up.
        Some((&b.gu[e][..GU_BYTES / 2], b.dn[e].as_slice()))
    });
    assert!(
        out.iter().all(|v| *v == 0.0),
        "every expert was undersized, so nothing could be staged"
    );
}

/// A short down slice IS staged — what it must not do is leave the tail
/// of the slot holding the previous call's bytes. The zero-fill is what
/// makes two calls with the same short input agree.
#[test]
fn a_short_down_slice_has_its_tail_zero_filled() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let x = h(5);

    // Prime the scratch with a full bank so the down slots hold real
    // bytes, then run the short variant twice on the same scratch.
    let _ = block(&metal, &x, &b.moe(), &s, &|e| {
        Some((b.gu[e].as_slice(), b.dn[e].as_slice()))
    });
    let short = |e: usize| Some((b.gu[e].as_slice(), &b.dn[e][..DN_BYTES / 2]));
    let first = block(&metal, &x, &b.moe(), &s, &short);
    let second = block(&metal, &x, &b.moe(), &s, &short);

    assert_eq!(
        first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a short down slice must zero its tail; otherwise the result \
         depends on what the slot held from the previous call"
    );
    assert!(first.iter().all(|v| v.is_finite()));
}

// The `valid_count >= scratch.top_k` bound guards against `moe.top_k`
// drifting from the allocation in RELEASE builds; the same condition is a
// `debug_assert_eq!` a few lines above, which fires first under `cargo
// test`. There is therefore no way to drive that `break` here, and a test
// that appeared to would only be proving the debug assert fires. Left
// deliberately uncovered rather than papered over with a release-only
// harness for a single defensive line.

/// The gate rule selects the activation pipeline; both gated variants are
/// reachable from this dispatch, chosen from an architecture fact.
#[test]
fn both_gated_activations_reach_the_dispatch() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let x = h(7);

    let mut silu = b.moe();
    silu.gate_rule = MoeGateRule::Gated(Activation::Silu);
    let a = block(&metal, &x, &silu, &s, &|e| {
        Some((b.gu[e].as_slice(), b.dn[e].as_slice()))
    });

    let mut gelu = b.moe();
    gelu.gate_rule = MoeGateRule::Gated(Activation::GeluTanh);
    let c = block(&metal, &x, &gelu, &s, &|e| {
        Some((b.gu[e].as_slice(), b.dn[e].as_slice()))
    });

    assert!(a.iter().chain(&c).all(|v| v.is_finite()));
    assert!(
        a.iter().zip(&c).any(|(p, q)| (p - q).abs() > 1e-6),
        "silu and gelu-tanh must not produce identical output — the \
         activation selection is not reaching the dispatch"
    );
}

/// GLM-5.3-Flash's combine has no Metal expert-activation kernel, and
/// the nearest one computes a different function. The dispatch refuses
/// it — at the top of the body, before any encoder exists.
///
/// Reached through the same wrapper every other test here uses, so what
/// is being shown is that a *dispatch call* refuses, not that a
/// predicate returns false.
#[test]
#[should_panic(expected = "no Metal expert-activation kernel for MoeGateRule::ClampedGated")]
fn a_clamped_gated_layer_is_refused_before_the_dispatch_encodes() {
    let metal = MetalBackend::new().expect("Metal backend required");
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let x = h(11);

    let mut glm = b.moe();
    glm.gate_rule = MoeGateRule::ClampedGated {
        limit: 7.0,
        activation: Activation::Silu,
    };
    let _ = block(&metal, &x, &glm, &s, &|e| {
        Some((b.gu[e].as_slice(), b.dn[e].as_slice()))
    });
}

/// The positive arm the refusal above needs: the identical fixture,
/// identical supplier, one field changed, runs and produces output. A
/// gate that only ever refuses proves nothing about what it admits, and
/// without this the panic above could be coming from the bank rather
/// than from the combine rule.
#[test]
fn the_same_fixture_under_a_served_combine_dispatches() {
    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let b = bank();
    let s = scratch(&metal, TOP_K);
    let x = h(11);

    let mut served = b.moe();
    served.gate_rule = MoeGateRule::Gated(Activation::Silu);
    let out = block(&metal, &x, &served, &s, &|e| {
        Some((b.gu[e].as_slice(), b.dn[e].as_slice()))
    });
    assert!(
        out.iter().any(|v| v.abs() > 0.0),
        "the fixture must dispatch"
    );
}
