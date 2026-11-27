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

use crate::format::vindex3::opplan::exec::backend::ExpertSlices;
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
    let mut session = DecodeSession::new(
        plan,
        store,
        backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
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
        up_bias: None,
        down_bias: None,
    }
}

mod diagnostics_and_residency;
mod single_gemv_refusals_behind_the_multi_pa;
mod the_nvfp4_realisation;
