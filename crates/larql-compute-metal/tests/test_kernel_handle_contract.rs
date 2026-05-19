//! Per-shader contract tests for the `Kernel` markers + the live
//! `KernelHandle`s on `MetalBackend`. Every simdgroup-tiled shader
//! that ships a `Kernel` (impl `metal::kernel::TiledKernel`) shows up
//! here. The contract is:
//!
//! 1. The marker's compile-time constants match the shader file's
//!    documented `pub const ROWS_PER_TG` / `THREADS_PER_TG`. Compile-
//!    time check, but listing the markers explicitly here is what
//!    catches "added a new shader, forgot the marker."
//! 2. The runtime `KernelHandle` on `MetalBackend.<…>_pipeline`
//!    exposes those exact same values. If a future commit swaps the
//!    pipeline binding to a different `Kernel` marker, this test
//!    flips red — that's the bug class
//!    `q4_matvec_dispatch_geometry_matches_v4_kernel` already covers
//!    for `q4_matvec_v4`, generalised to every other tiled shader.
//! 3. The pipeline's `maxTotalThreadsPerThreadgroup` is
//!    `>= threads_per_tg` for every handle. Construction already
//!    asserts this (the `KernelHandle::from_kernel` constructor
//!    returns `None` if the cap is below the request and the backend
//!    creation fails); the test catches a future regression where
//!    someone adds a new tiled handle but forgets to go through
//!    `from_kernel`.
//!
//! These are kernel-level invariants — they don't depend on a real
//! vindex and run in milliseconds.

#![cfg(target_os = "macos")]

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use larql_compute_metal::kernels::{KernelHandle, TiledKernel};
use larql_compute_metal::shaders;

/// One row in the pipeline ↔ marker contract: the live `KernelHandle`
/// on `MetalBackend.<field>` must agree with the marker's compile-
/// time constants.
fn assert_handle_matches_marker<K: TiledKernel>(handle: &KernelHandle, label: &str) {
    assert_eq!(
        handle.kernel_name,
        K::KERNEL_NAME,
        "{label}: handle.kernel_name='{}' but marker expects '{}'",
        handle.kernel_name,
        K::KERNEL_NAME,
    );
    assert_eq!(
        handle.rows_per_tg,
        K::ROWS_PER_TG,
        "{label}: handle.rows_per_tg={} but marker expects {}",
        handle.rows_per_tg,
        K::ROWS_PER_TG,
    );
    assert_eq!(
        handle.threads_per_tg,
        K::THREADS_PER_TG,
        "{label}: handle.threads_per_tg={} but marker expects {}",
        handle.threads_per_tg,
        K::THREADS_PER_TG,
    );

    // Pipeline cap >= requested threads_per_tg. `KernelHandle::from_kernel`
    // already enforces this at construction; the assertion here pins
    // the invariant against a future "raw `device.new_compute_pipeline_…`
    // bypass `from_kernel`" regression.
    let cap = handle.state.max_total_threads_per_threadgroup();
    assert!(
        cap >= handle.threads_per_tg,
        "{label}: pipeline cap ({cap}) < threads_per_tg ({}). Metal would \
         silently dispatch fewer threads/TG → fewer simdgroups → rows dropped.",
        handle.threads_per_tg,
    );
}

fn assert_q4k_selected_handle_matches_active_marker(handle: &KernelHandle, label: &str) {
    match handle.kernel_name {
        <shaders::q4k_matvec::Kernel as TiledKernel>::KERNEL_NAME => {
            assert_handle_matches_marker::<shaders::q4k_matvec::Kernel>(handle, label);
        }
        <shaders::q4k_matvec_8sg::Kernel as TiledKernel>::KERNEL_NAME => {
            assert_handle_matches_marker::<shaders::q4k_matvec_8sg::Kernel>(handle, label);
        }
        other => panic!("{label}: q4k_matvec_pipeline is bound to unsupported kernel '{other}'"),
    }
}

/// The Q4 family — bundled in `Q4Pipelines`. Only `matvec` is a
/// `KernelHandle`; `vecmat` and `f32_matvec` are flat-dispatch and
/// stay as bare pipelines (intentional — see `metal/ops/q4_common.rs`).
#[test]
fn q4_pipelines_handle_contract() {
    let metal = get_metal();
    assert_handle_matches_marker::<shaders::q4_matvec_v4::Kernel>(&metal.q4.matvec, "q4.matvec");
}

/// The K-format matvec family — Q4_K, Q6_K, Q8.
#[test]
fn k_matvec_handle_contract() {
    let metal = get_metal();
    assert_q4k_selected_handle_matches_active_marker(
        &metal.quant.q4k_matvec_pipeline,
        "q4k_matvec_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4k_matvec::Kernel>(
        &metal.quant.q4k_matvec_4sg_pipeline,
        "q4k_matvec_4sg_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4k_matvec_8sg::Kernel>(
        &metal.quant.q4k_matvec_8sg_pipeline,
        "q4k_matvec_8sg_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4k_matvec_stride32::Kernel>(
        &metal.quant.q4k_matvec_stride32_pipeline,
        "q4k_matvec_stride32_pipeline",
    );
    assert_handle_matches_marker::<shaders::q6k_matvec::Kernel>(
        &metal.quant.q6k_matvec_pipeline,
        "q6k_matvec_pipeline",
    );
    assert_handle_matches_marker::<shaders::q8_matvec::Kernel>(
        &metal.quant.q8_matvec_pipeline,
        "q8_matvec_pipeline",
    );
}

/// The fused FFN gate+up family — Q4_K and Q4_KF.
#[test]
fn ffn_gate_up_handle_contract() {
    let metal = get_metal();
    assert_handle_matches_marker::<shaders::q4k_ffn_gate_up::Kernel>(
        &metal.ffn.q4k_ffn_gate_up_pipeline,
        "q4k_ffn_gate_up_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4kf_ffn_gate_up::Kernel>(
        &metal.ffn.q4kf_ffn_gate_up_pipeline,
        "q4kf_ffn_gate_up_pipeline",
    );
}

/// The QKV-projection family — fused (Q4_K, Q4_KF, mixed Q4_K/Q6_K)
/// and per-projection variants.
#[test]
fn qkv_proj_handle_contract() {
    let metal = get_metal();
    assert_handle_matches_marker::<shaders::q4k_qkv_proj::QkvKernel>(
        &metal.attention.q4k_qkv_proj_pipeline,
        "q4k_qkv_proj_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4k_qkv_proj::ProjKernel>(
        &metal.attention.q4k_proj_pipeline,
        "q4k_proj_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4kf_qkv_proj::QkvKernel>(
        &metal.attention.q4kf_qkv_proj_pipeline,
        "q4kf_qkv_proj_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4kf_qkv_proj::ProjKernel>(
        &metal.attention.q4kf_proj_pipeline,
        "q4kf_proj_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4k_q6k_qkv_proj::Kernel>(
        &metal.attention.q4k_q6k_qkv_proj_pipeline,
        "q4k_q6k_qkv_proj_pipeline",
    );
}

/// Fused Q8 QKV projection — tiled simdgroup, the only Q8-family
/// pipeline that needed migrating to KernelHandle. (Other Q8 paths use
/// flat dispatch_threads — `q8_matvec` is already a handle, the rest
/// don't need geometry.)
#[test]
fn q8_qkv_proj_handle_contract() {
    let metal = get_metal();
    assert_handle_matches_marker::<shaders::q8_attn_proj::QkvKernel>(
        &metal.attention.q8_qkv_proj_pipeline,
        "q8_qkv_proj_pipeline",
    );
}

/// The fused activation+down family — SiLU and GELU-tanh variants.
#[test]
fn geglu_down_handle_contract() {
    let metal = get_metal();
    assert_handle_matches_marker::<shaders::q4k_geglu_down::SiluKernel>(
        &metal.ffn.q4k_geglu_silu_down_pipeline,
        "q4k_geglu_silu_down_pipeline",
    );
    assert_handle_matches_marker::<shaders::q4k_geglu_down::GeluTanhKernel>(
        &metal.ffn.q4k_geglu_gelu_tanh_down_pipeline,
        "q4k_geglu_gelu_tanh_down_pipeline",
    );
}

/// The dense gemv family — f32 / f16 LM-head specialisations.
#[test]
fn gemv_handle_contract() {
    let metal = get_metal();
    assert_handle_matches_marker::<shaders::f32_gemv::Kernel>(
        &metal.f32_gemv_pipeline,
        "f32_gemv_pipeline",
    );
    assert_handle_matches_marker::<shaders::f16_gemv::Kernel>(
        &metal.f16_gemv_pipeline,
        "f16_gemv_pipeline",
    );
}

/// `Capability` truth table for `MetalBackend`. Mirrors the cpu
/// equivalent in `test_correctness.rs::cpu_backend_capability_truth_table`.
#[test]
fn metal_backend_capability_truth_table() {
    use larql_compute::prelude::*;
    use larql_compute::Capability;

    let metal = get_metal();
    // Metal accelerates everything in the menu — see
    // `metal/trait_impl/mod.rs::supports`.
    let all = [
        Capability::F32Gemv,
        Capability::F16Gemv,
        Capability::QuantMatVec,
        Capability::Q4VecMat,
        Capability::Q4PairBatch,
        Capability::FullPipelineQ4,
        Capability::MultiLayerQ4Ffn,
        Capability::DecodeToken,
        Capability::DecodeMoe,
        Capability::DecodeProfile,
        Capability::PrefillQ4,
    ];
    for cap in all {
        assert!(
            metal.supports(cap),
            "expected MetalBackend to support {cap:?}"
        );
    }
}
