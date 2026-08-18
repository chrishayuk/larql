//! Device-backend arms not reached by the parity gates.
//!
//! The parity gates in `device.rs`, `decode.rs` and `routed.rs` drive
//! `DevicePlanBackend` through whole plans, which reaches the happy
//! paths of the f32/f16/MXFP4 realisations and the *first* refusal a
//! kernelless device produces. What they cannot reach: the NVFP4
//! realisation (no fixture declared it), the single-gemv refusals that
//! sit behind a multi-gemv (a kernelless device dies at Q/K/V before the
//! output projection is ever asked), the geometry checks in front of
//! the device, the residency and diagnostic accessors, and the ungated
//! and unsupported-activation FFN arms. Each test here drives one of
//! those arms directly and asserts what the arm is *for*.

use std::sync::{Arc, Mutex};

use larql_compute::backend::MatMul;
use larql_compute::cpu::ops::geglu::silu;
use larql_models::config::{
    Activation, ExpertGatePolicy, ExpertRoutingPolicy, GateUpLayout, MoeRouterKind,
};
use larql_models::quant::nvfp4::{
    dequantize_into, round_trip, Nvfp4Matrix, NVFP4_GROUP_BYTES, NVFP4_GROUP_ELEMS,
};
use ndarray::{Array2, ArrayView2};

use super::device::LoopDevice;
use super::{dense_f32_model, lcg_values};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{
    FfnCall, MatrixClass, PlanBackend, ProjectCall, RoutedFfnCall, WeightFormat, WeightFormats,
    WeightSlice,
};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::weights::{quantize_mxfp4, quantize_nvfp4, LoadedWeight};
use crate::format::vindex3::opplan::exec::{execute_plan, ExecutionTrace};
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// A small matrix geometry aligned to both 4-bit group sizes (MXFP4's
/// 32 and NVFP4's 16), so every format can carry it.
const ROWS: usize = 4;
const COLS: usize = 32;

/// FFN geometry for the direct `ffn`/`routed_ffn` calls: `hidden` is
/// the up/gate K dimension and `intermediate` the down K dimension, so
/// both stay group-aligned.
const FFN_HIDDEN: usize = 32;
const FFN_INTERMEDIATE: usize = 32;

/// Routed-FFN geometry: two experts, top-1, so the selected expert is
/// determined by the router alone.
const EXPERTS: usize = 2;
const TOP_K: usize = 1;

/// Bytes per f16 element, for sizing f16 weight slices.
const F16_BYTES: usize = 2;

/// Above the reassociation noise of two in-order loops that sum the
/// same products (in practice they are bit-identical); far below any
/// semantic effect.
const LOOP_NOISE_CEILING: f32 = 1e-6;

/// The NVFP4 realisation of the dense fixture must stay in the f32
/// reference's neighbourhood: 4-bit weights are coarse, so this is a
/// direction gate (cosine), not a closeness gate.
const NVFP4_COSINE_FLOOR: f32 = 0.5;

/// Tokens for the dense Llama fixture; distinct positions so the
/// decode session's cache is exercised beyond one row.
const DENSE_TOKENS: [u32; 5] = [3, 17, 42, 99, 7];

/// A device with no gemv kernel of any format — every trait default.
struct KernellessDevice;

impl MatMul for KernellessDevice {
    fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }

    fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }
}

/// `LoopDevice` plus an NVFP4 kernel: decode through the models-crate
/// dequantiser (the independent reader of the format), then the same
/// in-order loop. The seam is what is under test; the arithmetic is
/// deliberately borrowed from the format's own reference decoder.
struct Nvfp4LoopDevice;

impl MatMul for Nvfp4LoopDevice {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        LoopDevice.matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        LoopDevice.matmul_transb(a, b)
    }

    fn f32_gemv_force(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        LoopDevice.f32_gemv_force(w, x)
    }

    fn nvfp4_gemv(
        &self,
        packed: &[u8],
        scales: &[u8],
        tensor_scale: f32,
        x: &[f32],
        n: usize,
        k: usize,
    ) -> Option<Vec<f32>> {
        if !k.is_multiple_of(NVFP4_GROUP_ELEMS) || x.len() != k {
            return None;
        }
        let groups = k / NVFP4_GROUP_ELEMS;
        if packed.len() < n * groups * NVFP4_GROUP_BYTES || scales.len() < n * groups {
            return None;
        }
        let matrix = Nvfp4Matrix {
            packed: packed[..n * groups * NVFP4_GROUP_BYTES].to_vec(),
            scales: scales[..n * groups].to_vec(),
            tensor_scale,
        };
        let mut w = vec![0.0f32; n * k];
        dequantize_into(&matrix, n, k, &mut w).ok()?;
        Some(matvec_rows(&w, n, k, x))
    }
}

/// A device that has the f16 *multi* kernel but no single f16 gemv:
/// Q/K/V (one multi submission) succeed and the output projection (a
/// single gemv) is refused, which reaches the refusal behind the multi
/// path that a fully kernelless device never gets to.
struct MultiOnlyF16Device;

impl MatMul for MultiOnlyF16Device {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        LoopDevice.matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        LoopDevice.matmul_transb(a, b)
    }

    fn f16_gemv_multi(
        &self,
        weights: &[(&[u8], usize, usize)],
        x: &[f32],
    ) -> Option<Vec<Vec<f32>>> {
        weights
            .iter()
            .map(|&(w, n, k)| LoopDevice.f16_gemv_force(w, x, n, k))
            .collect()
    }
}

/// A device that records what `wire_resident` was handed: the byte
/// length of every stream, in order.
struct WireRecorder {
    streams: Arc<Mutex<Vec<usize>>>,
}

impl MatMul for WireRecorder {
    fn matmul(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }

    fn matmul_transb(&self, _a: ArrayView2<f32>, _b: ArrayView2<f32>) -> Array2<f32> {
        unimplemented!("the plan backend only dispatches gemv")
    }

    fn wire_resident(&self, buffers: &[&[u8]]) {
        self.streams
            .lock()
            .unwrap()
            .extend(buffers.iter().map(|b| b.len()));
    }
}

/// `out[n] = W[n, k] · x`, plain in-order loop.
fn matvec_rows(w: &[f32], n: usize, k: usize, x: &[f32]) -> Vec<f32> {
    (0..n)
        .map(|row| (0..k).map(|col| w[row * k + col] * x[col]).sum())
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Encode the dense Llama fixture and open its plan + store.
fn dense_fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    dense_f32_model(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("dense-device".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Step every dense-fixture token through a fresh session; the last
/// position's logits.
fn decode_logits<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> Vec<f32> {
    let mut session = DecodeSession::new(plan, store, backend).unwrap();
    let mut last = None;
    for &token in DENSE_TOKENS.iter() {
        last = session.step(token).unwrap().logits;
    }
    last.expect("plan carries an output head")
}

fn project_on<M: MatMul + Send>(
    backend: &DevicePlanBackend<M>,
    weight: WeightSlice<'_>,
    x: &[f32],
) -> Result<Vec<f32>, crate::error::VindexError> {
    backend.project(ProjectCall {
        weight,
        out_dim: ROWS,
        in_dim: COLS,
        x,
    })
}

fn ffn_call<'a>(
    x: &'a [f32],
    gate: Option<WeightSlice<'a>>,
    up: WeightSlice<'a>,
    down: WeightSlice<'a>,
    activation: Activation,
) -> FfnCall<'a> {
    FfnCall {
        x,
        hidden: FFN_HIDDEN,
        intermediate: FFN_INTERMEDIATE,
        gate,
        up,
        down,
        activation,
        gate_policy: ExpertGatePolicy::Gated,
    }
}

// ── Diagnostics and residency ────────────────────────────────────────

/// `name` reports the injected name verbatim (it becomes the engine tag
/// on a dump) and `dispatch_stats` counts one submission per device
/// call: a lone `project` is one, a two-matrix f16 FFN is one multi
/// submission plus one for `down`.
#[test]
fn the_name_is_verbatim_and_dispatch_stats_count_one_submission_per_device_call() {
    const NAME: &str = "loop-device-stats";
    const PROJECT_CALLS: u64 = 3;
    /// Up+gate ride one multi submission; down is a second.
    const SUBMISSIONS_PER_GATED_F16_FFN: u64 = 2;

    let backend = DevicePlanBackend::new(LoopDevice, NAME, WeightFormat::F32);
    assert_eq!(backend.name(), NAME);
    let fresh = backend.dispatch_stats().unwrap();
    assert_eq!((fresh.submissions, fresh.device_nanos), (0, 0));

    let w = lcg_values(ROWS * COLS, 1);
    let x = lcg_values(COLS, 2);
    for _ in 0..PROJECT_CALLS {
        project_on(&backend, WeightSlice::F32(&w), &x).unwrap();
    }
    let after_projects = backend.dispatch_stats().unwrap();
    assert_eq!(after_projects.submissions, PROJECT_CALLS);

    let f16 = vec![0u8; FFN_INTERMEDIATE * FFN_HIDDEN * F16_BYTES];
    let x = lcg_values(FFN_HIDDEN, 3);
    backend
        .ffn(ffn_call(
            &x,
            Some(WeightSlice::F16(&f16)),
            WeightSlice::F16(&f16),
            WeightSlice::F16(&f16),
            Activation::Silu,
        ))
        .unwrap();
    let after_ffn = backend.dispatch_stats().unwrap();
    assert_eq!(
        after_ffn.submissions,
        PROJECT_CALLS + SUBMISSIONS_PER_GATED_F16_FFN
    );
}

/// `prepare` hands the device one stream per f16 weight and two per
/// packed 4-bit weight (codes and scales), skipping f32 — the layout
/// the residency hint must see for every byte the decode will touch.
#[test]
fn prepare_wires_one_stream_per_f16_weight_and_two_per_packed_weight() {
    let streams = Arc::new(Mutex::new(Vec::new()));
    let backend = DevicePlanBackend::new(
        WireRecorder {
            streams: Arc::clone(&streams),
        },
        "wire-recorder",
        WeightFormat::F16,
    );
    let f32_weight = lcg_values(ROWS * COLS, 4);
    let f16_bytes = vec![0u8; ROWS * COLS * F16_BYTES];
    let LoadedWeight::Mxfp4 {
        packed: mx_packed,
        scales: mx_scales,
    } = quantize_mxfp4(&f32_weight, ROWS, COLS, "mx").unwrap()
    else {
        unreachable!()
    };
    let LoadedWeight::Nvfp4 {
        packed: nv_packed,
        scales: nv_scales,
        tensor_scale,
    } = quantize_nvfp4(&f32_weight, ROWS, COLS, "nv").unwrap()
    else {
        unreachable!()
    };
    backend.prepare(&[
        WeightSlice::F32(&f32_weight),
        WeightSlice::F16(&f16_bytes),
        WeightSlice::Mxfp4 {
            packed: mx_packed.as_slice(),
            scales: mx_scales.as_slice(),
        },
        WeightSlice::Nvfp4 {
            packed: nv_packed.as_slice(),
            scales: nv_scales.as_slice(),
            tensor_scale,
        },
    ]);
    let seen = streams.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![
            f16_bytes.len(),
            mx_packed.as_slice().len(),
            mx_scales.as_slice().len(),
            nv_packed.as_slice().len(),
            nv_scales.as_slice().len(),
        ],
        "f32 contributes no stream; f16 one; each packed format its codes then its scales"
    );
}

// ── Geometry checks in front of the device ───────────────────────────

/// An f32 weight whose length is not `out × in` is refused by the seam,
/// naming the geometry, before any kernel sees it.
#[test]
fn a_misshapen_f32_weight_is_refused_naming_the_geometry() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-shape", WeightFormat::F32);
    let short = lcg_values(ROWS * COLS - 1, 5);
    let x = lcg_values(COLS, 6);
    let err = project_on(&backend, WeightSlice::F32(&short), &x).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!("is not [{ROWS}, {COLS}]")),
        "{message}"
    );
}

/// An f16 slice shorter than the matrix is refused by the seam with the
/// byte count — the device (which would also refuse) is never asked, so
/// the message names the seam's check, not a kernel.
#[test]
fn a_short_f16_weight_is_refused_before_reaching_the_device() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-f16-short", WeightFormat::F16);
    let short = vec![0u8; ROWS * COLS * F16_BYTES - F16_BYTES];
    let x = lcg_values(COLS, 7);
    let err = project_on(&backend, WeightSlice::F16(&short), &x).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!(
            "{} weight bytes cannot hold [{ROWS} x {COLS}]",
            short.len()
        )),
        "{message}"
    );
}

// ── Single-gemv refusals behind the multi path ───────────────────────

/// On a kernelless device the single-gemv seam refuses each format
/// naming its kernel and the shape — the f16/MXFP4/NVFP4 arms a whole-
/// plan run never reaches because it dies at the multi-gemv first.
#[test]
fn single_gemv_refusals_name_the_kernel_and_shape_for_every_packed_format() {
    let backend = DevicePlanBackend::new(KernellessDevice, "kernelless-single", WeightFormat::F32);
    let f32_weight = lcg_values(ROWS * COLS, 8);
    let x = lcg_values(COLS, 9);
    let f16_bytes = vec![0u8; ROWS * COLS * F16_BYTES];
    let LoadedWeight::Mxfp4 {
        packed: mx_packed,
        scales: mx_scales,
    } = quantize_mxfp4(&f32_weight, ROWS, COLS, "mx").unwrap()
    else {
        unreachable!()
    };
    let LoadedWeight::Nvfp4 {
        packed: nv_packed,
        scales: nv_scales,
        tensor_scale,
    } = quantize_nvfp4(&f32_weight, ROWS, COLS, "nv").unwrap()
    else {
        unreachable!()
    };
    let cases: [(&str, WeightSlice<'_>); 3] = [
        ("f16_gemv", WeightSlice::F16(&f16_bytes)),
        (
            "mxfp4_gemv",
            WeightSlice::Mxfp4 {
                packed: mx_packed.as_slice(),
                scales: mx_scales.as_slice(),
            },
        ),
        (
            "nvfp4_gemv",
            WeightSlice::Nvfp4 {
                packed: nv_packed.as_slice(),
                scales: nv_scales.as_slice(),
                tensor_scale,
            },
        ),
    ];
    for (kernel, weight) in cases {
        let message = project_on(&backend, weight, &x).unwrap_err().to_string();
        assert!(
            message.contains(&format!("device {kernel} [{ROWS} x {COLS}] refused")),
            "{kernel}: {message}"
        );
    }
}

/// A device that batches f16 matrices but has no single f16 kernel:
/// Q/K/V succeed through the multi path and the output projection —
/// a lone gemv — fails closed naming its own shape. Nothing widens or
/// substitutes.
#[test]
fn a_multi_only_f16_device_fails_closed_at_the_output_projection() {
    let (_c, plan, store) = dense_fixture();
    let backend = DevicePlanBackend::new(MultiOnlyF16Device, "multi-only-f16", WeightFormat::F16);
    let err = execute_plan(&plan, &store, &DENSE_TOKENS, &backend).unwrap_err();
    let message = err.to_string();
    let q_rows = super::Q_HEADS * super::HEAD_DIM;
    assert!(
        message.contains(&format!(
            "device f16_gemv [{} x {q_rows}] refused",
            super::HIDDEN
        )),
        "{message}"
    );
}

/// The NVFP4 multi path refuses on a kernelless device naming the
/// matrix count — the FFN's up+gate pair — and the FFN propagates it.
#[test]
fn an_nvfp4_multi_dispatch_refusal_names_the_matrix_count() {
    /// Up and gate.
    const FFN_PAIR: usize = 2;
    let backend = DevicePlanBackend::new(KernellessDevice, "kernelless-nvfp4", WeightFormat::Nvfp4);
    let values = lcg_values(FFN_INTERMEDIATE * FFN_HIDDEN, 10);
    let LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
    } = quantize_nvfp4(&values, FFN_INTERMEDIATE, FFN_HIDDEN, "nv").unwrap()
    else {
        unreachable!()
    };
    let weight = WeightSlice::Nvfp4 {
        packed: packed.as_slice(),
        scales: scales.as_slice(),
        tensor_scale,
    };
    let x = lcg_values(FFN_HIDDEN, 11);
    let err = backend
        .ffn(ffn_call(&x, Some(weight), weight, weight, Activation::Silu))
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!("nvfp4_gemv_multi ({FFN_PAIR} matrices) refused")),
        "{message}"
    );
}

/// The routed FFN's first device call is the f32 router; a kernelless
/// device refuses it and nothing routes.
#[test]
fn a_routed_ffn_whose_router_dispatch_is_refused_fails_closed() {
    let backend = DevicePlanBackend::new(KernellessDevice, "kernelless-routed", WeightFormat::F32);
    let router = lcg_values(EXPERTS * FFN_HIDDEN, 12);
    let expert_gate_up: Vec<Vec<f32>> = (0..EXPERTS)
        .map(|e| lcg_values(2 * FFN_INTERMEDIATE * FFN_HIDDEN, 20 + e as u64))
        .collect();
    let expert_down: Vec<Vec<f32>> = (0..EXPERTS)
        .map(|e| lcg_values(FFN_HIDDEN * FFN_INTERMEDIATE, 30 + e as u64))
        .collect();
    let gate_up: Vec<WeightSlice<'_>> =
        expert_gate_up.iter().map(|w| WeightSlice::F32(w)).collect();
    let down: Vec<WeightSlice<'_>> = expert_down.iter().map(|w| WeightSlice::F32(w)).collect();
    let x = lcg_values(FFN_HIDDEN, 13);
    let err = backend
        .routed_ffn(RoutedFfnCall {
            x: &x,
            hidden: FFN_HIDDEN,
            intermediate: FFN_INTERMEDIATE,
            experts: EXPERTS,
            top_k: TOP_K,
            router_kind: MoeRouterKind::TopKSoftmax,
            routing_policy: ExpertRoutingPolicy::SoftmaxThenSelect,
            activation: Activation::Silu,
            gate_policy: ExpertGatePolicy::Gated,
            gate_up_layout: GateUpLayout::ContiguousHalves,
            router: &router,
            router_bias: None,
            gate_up: &gate_up,
            gate_up_bias: None,
            down: &down,
            down_bias: None,
        })
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(&format!(
            "device f32_gemv [{EXPERTS} x {FFN_HIDDEN}] refused"
        )),
        "{message}"
    );
}

// ── FFN arms ─────────────────────────────────────────────────────────

/// An ungated FFN is `down · silu(up · x)`: no gate matrix, no
/// elementwise product — the shape the dense-Llama plan never carries.
#[test]
fn an_ungated_ffn_applies_silu_to_the_up_projection_alone() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-ungated", WeightFormat::F32);
    let up = lcg_values(FFN_INTERMEDIATE * FFN_HIDDEN, 14);
    let down = lcg_values(FFN_HIDDEN * FFN_INTERMEDIATE, 15);
    let x = lcg_values(FFN_HIDDEN, 16);
    let got = backend
        .ffn(ffn_call(
            &x,
            None,
            WeightSlice::F32(&up),
            WeightSlice::F32(&down),
            Activation::Silu,
        ))
        .unwrap();
    let inner: Vec<f32> = matvec_rows(&up, FFN_INTERMEDIATE, FFN_HIDDEN, &x)
        .into_iter()
        .map(silu)
        .collect();
    let expected = matvec_rows(&down, FFN_HIDDEN, FFN_INTERMEDIATE, &inner);
    assert_eq!(got.len(), FFN_HIDDEN);
    assert!(
        max_abs(&got, &expected) < LOOP_NOISE_CEILING,
        "ungated FFN diverges from down · silu(up · x)"
    );
}

/// The device FFN refuses an activation it has no kernel for, in both
/// the gated and the ungated shape, naming which shape refused.
#[test]
fn an_ffn_with_an_unsupported_activation_refuses_gated_and_ungated() {
    let backend = DevicePlanBackend::new(LoopDevice, "loop-device-activation", WeightFormat::F32);
    let up = lcg_values(FFN_INTERMEDIATE * FFN_HIDDEN, 17);
    let down = lcg_values(FFN_HIDDEN * FFN_INTERMEDIATE, 18);
    let x = lcg_values(FFN_HIDDEN, 19);
    for (shape, gate) in [("gated", Some(WeightSlice::F32(&up))), ("ungated", None)] {
        let err = backend
            .ffn(ffn_call(
                &x,
                gate,
                WeightSlice::F32(&up),
                WeightSlice::F32(&down),
                Activation::Gelu,
            ))
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains(&format!("{shape}-FFN")) && message.contains("Gelu"),
            "{shape}: {message}"
        );
    }
}

// ── The NVFP4 realisation ────────────────────────────────────────────

/// A single NVFP4 projection through the seam equals the round-tripped
/// matrix against the same input: the seam hands the device the codes,
/// the E4M3 scales *and* the tensor scale intact, with the geometry.
#[test]
fn an_nvfp4_projection_matches_the_round_tripped_matrix() {
    let backend =
        DevicePlanBackend::new(Nvfp4LoopDevice, "nvfp4-loop-project", WeightFormat::Nvfp4);
    let values = lcg_values(ROWS * COLS, 21);
    let x = lcg_values(COLS, 22);
    let LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
    } = quantize_nvfp4(&values, ROWS, COLS, "nv").unwrap()
    else {
        unreachable!()
    };
    let got = project_on(
        &backend,
        WeightSlice::Nvfp4 {
            packed: packed.as_slice(),
            scales: scales.as_slice(),
            tensor_scale,
        },
        &x,
    )
    .unwrap();
    let reconstructed = round_trip(&values, ROWS, COLS).unwrap();
    let expected = matvec_rows(&reconstructed, ROWS, COLS, &x);
    assert!(
        max_abs(&got, &expected) < LOOP_NOISE_CEILING,
        "seam-routed NVFP4 gemv diverges from the reference reconstruction"
    );
    // The realisation is lossy but must not be degenerate: the tensor
    // scale was applied (an all-zero output would mean it was dropped).
    assert!(got.iter().any(|v| *v != 0.0));
}

/// The NVFP4 realisation runs the dense plan end to end — attention
/// Q/K/V through the NVFP4 multi path, the FFN pair likewise, single
/// projections through the NVFP4 gemv — stays in the reference's
/// neighbourhood, and its decode session reproduces its own batch
/// traversal bit for bit (the step path with no gate, on the device).
#[test]
fn an_nvfp4_device_backend_executes_and_decodes_the_dense_plan() {
    let (_c, plan, store) = dense_fixture();
    let backend = DevicePlanBackend::with_formats(
        Nvfp4LoopDevice,
        "nvfp4-loop-dense",
        WeightFormats::uniform(WeightFormat::Nvfp4),
    );
    assert_eq!(
        backend.weight_format(MatrixClass::FfnProjection),
        WeightFormat::Nvfp4
    );
    let on_device: ExecutionTrace = execute_plan(&plan, &store, &DENSE_TOKENS, &backend).unwrap();
    let on_reference =
        execute_plan(&plan, &store, &DENSE_TOKENS, &ReferenceBackend::new()).unwrap();
    let logits = on_device.logits.as_ref().unwrap();
    assert!(logits.iter().all(|v| v.is_finite()));
    let cos = cosine(logits, on_reference.logits.as_ref().unwrap());
    assert!(
        cos > NVFP4_COSINE_FLOOR,
        "nvfp4 logits decorrelated from reference: cos {cos}"
    );

    let stepped = decode_logits(&plan, &store, &backend);
    assert_eq!(
        logits.as_slice(),
        stepped.as_slice(),
        "nvfp4 decode-session logits differ from the batch traversal"
    );
    // Every matrix went through the device: the batch pass and the
    // decode pass both submitted work.
    assert!(backend.dispatch_stats().unwrap().submissions > 0);
}
