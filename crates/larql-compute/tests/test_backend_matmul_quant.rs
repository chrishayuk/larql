//! Coverage for the backend trait default methods (matmul_batch, gemv stubs)
//! and quant_matvec dispatch for Q4_K / Q6_K / quant_matvec_q8_input.

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k, quantize_to_q8};
use larql_compute::prelude::*;
use larql_compute::{cpu_backend, MatMulOp, QuantFormat};
use ndarray::Array2;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ── MatMul::matmul_batch ─────────────────────────────────────────────────────

#[test]
fn matmul_batch_no_transpose_serial_dispatch() {
    let cpu = cpu_backend();
    let a1 = synth(3, 4, 1);
    let b1 = synth(4, 5, 2);
    let a2 = synth(2, 4, 3);
    let b2 = synth(4, 6, 4);
    let ops = vec![
        MatMulOp {
            a: a1.clone(),
            b: b1.clone(),
            transpose_b: false,
        },
        MatMulOp {
            a: a2.clone(),
            b: b2.clone(),
            transpose_b: false,
        },
    ];
    let results = cpu.matmul_batch(&ops);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].shape(), &[3, 5]);
    assert_eq!(results[1].shape(), &[2, 6]);
    // Verify against individual matmul calls
    let expected0 = cpu.matmul(a1.view(), b1.view());
    let expected1 = cpu.matmul(a2.view(), b2.view());
    let diff0: f32 = results[0]
        .iter()
        .zip(&expected0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let diff1: f32 = results[1]
        .iter()
        .zip(&expected1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff0 < 1e-5);
    assert!(diff1 < 1e-5);
}

#[test]
fn matmul_batch_with_transpose_serial_dispatch() {
    let cpu = cpu_backend();
    let a = synth(3, 8, 5);
    let b = synth(6, 8, 6); // B is [6, 8], transpose → [8, 6]
    let ops = vec![MatMulOp {
        a: a.clone(),
        b: b.clone(),
        transpose_b: true,
    }];
    let results = cpu.matmul_batch(&ops);
    assert_eq!(results[0].shape(), &[3, 6]);
    let expected = cpu.matmul_transb(a.view(), b.view());
    let diff: f32 = results[0]
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff < 1e-5);
}

// ── MatMul gemv stubs (CPU returns None) ─────────────────────────────────────

#[test]
fn f32_gemv_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let w = synth(512, 256, 7);
    let x = synth_vec(256, 8);
    assert!(cpu.f32_gemv(w.view(), &x).is_none());
    assert!(cpu.f32_gemv_topk1(w.view(), &x).is_none());
}

#[test]
fn f32_gemv_force_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let w = synth(512, 256, 9);
    let x = synth_vec(256, 10);
    // Default delegates to f32_gemv, so also None.
    assert!(cpu.f32_gemv_force(w.view(), &x).is_none());
}

#[test]
fn f16_gemv_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let n = 512usize;
    let k = 256usize;
    let w_f16 = vec![0u8; n * k * 2];
    let x = synth_vec(k, 11);
    assert!(cpu.f16_gemv(&w_f16, &x, n, k).is_none());
    assert!(cpu.f16_gemv_topk1(&w_f16, &x, n, k).is_none());
    assert!(cpu.f16_gemv_topk(&w_f16, &x, n, k, 8).is_none());
}

#[test]
fn f16_gemv_force_returns_none_on_cpu() {
    let cpu = cpu_backend();
    let n = 512usize;
    let k = 256usize;
    let w_f16 = vec![0u8; n * k * 2];
    let x = synth_vec(k, 12);
    // Default delegates to f16_gemv, so also None.
    assert!(cpu.f16_gemv_force(&w_f16, &x, n, k).is_none());
}

// ── QuantMatVec::quant_matvec for Q4_K and Q6_K ──────────────────────────────

#[test]
fn quant_matvec_q4k_dispatches_to_q4k_kernel() {
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 13);
    let x: Vec<f32> = synth_vec(hidden, 14);
    let q4k = quantize_q4_k(&weights);
    let result = cpu
        .quant_matvec(QuantFormat::Q4_K, &q4k, &x, rows, hidden)
        .expect("CPU should support Q4_K via q4k_matvec");
    assert_eq!(result.len(), rows);
    assert!(
        result.iter().any(|v| v.abs() > 1e-4),
        "expected nonzero output"
    );
}

#[test]
fn quant_matvec_q4kf_dispatches_same_as_q4k() {
    // Q4_KF is an alias → dispatches through q4k_matvec same as Q4_K.
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 15);
    let x: Vec<f32> = synth_vec(hidden, 16);
    let q4k = quantize_q4_k(&weights);
    let result = cpu
        .quant_matvec(QuantFormat::Q4_KF, &q4k, &x, rows, hidden)
        .expect("CPU should support Q4_KF via q4k_matvec");
    assert_eq!(result.len(), rows);
}

#[test]
fn quant_matvec_q6k_dispatches_to_q6k_kernel() {
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 17);
    let x: Vec<f32> = synth_vec(hidden, 18);
    let q6k = quantize_q6_k(&weights);
    let result = cpu
        .quant_matvec(QuantFormat::Q6_K, &q6k, &x, rows, hidden)
        .expect("CPU should support Q6_K via q6k_matvec");
    assert_eq!(result.len(), rows);
    assert!(
        result.iter().any(|v| v.abs() > 1e-4),
        "expected nonzero output"
    );
}

// ── QuantMatVec::quant_matvec_q8_input for Q4_K (triggers dequantise_q8) ────

#[test]
fn quant_matvec_q8_input_q4k_dequantises_then_dispatches() {
    // quant_matvec_q8_input with Q4_K hits the dequantise_q8 → f32 → q4k_matvec path.
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 19);
    let x: Vec<f32> = synth_vec(hidden, 20);
    let q4k = quantize_q4_k(&weights);
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let result = cpu
        .quant_matvec_q8_input(QuantFormat::Q4_K, &q4k, &q8_x, &q8_scales, rows, hidden)
        .expect("CPU should support Q4_K via quant_matvec_q8_input");
    assert_eq!(result.len(), rows);
    // Should approximately match quant_matvec (some Q8 round-trip error expected)
    let direct = cpu
        .quant_matvec(QuantFormat::Q4_K, &q4k, &x, rows, hidden)
        .unwrap();
    let max_diff: f32 = result
        .iter()
        .zip(&direct)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let mag: f32 = direct.iter().map(|v| v.abs()).fold(0.0, f32::max);
    // Allow up to 5% relative error from the Q8 round-trip
    assert!(
        max_diff < 0.05 * mag.max(1.0),
        "Q8-input path diverges from f32 path: {max_diff} vs mag {mag}"
    );
}

#[test]
fn quant_matvec_q8_input_q6k_dequantises_then_dispatches() {
    let cpu = cpu_backend();
    let hidden = 256usize;
    let rows = 4usize;
    let weights: Vec<f32> = synth_vec(rows * hidden, 21);
    let x: Vec<f32> = synth_vec(hidden, 22);
    let q6k = quantize_q6_k(&weights);
    let (q8_x, q8_scales) = quantize_to_q8(&x);

    let result = cpu
        .quant_matvec_q8_input(QuantFormat::Q6_K, &q6k, &q8_x, &q8_scales, rows, hidden)
        .expect("CPU should support Q6_K via quant_matvec_q8_input");
    assert_eq!(result.len(), rows);
}

// ── QuantMatVec::q4_vecmat via trait ─────────────────────────────────────────

#[test]
fn q4_vecmat_via_trait_nonzero() {
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    let cpu = cpu_backend();
    let inter = 128usize;
    let hidden = 256usize;
    let activation: Vec<f32> = synth_vec(inter, 23);
    let matrix: Vec<f32> = synth_vec(inter * hidden, 24);
    let q4 = quantize_q4_0(&matrix);
    let result = cpu
        .q4_vecmat(&activation, &q4, inter, hidden)
        .expect("CPU should support q4_vecmat");
    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|v| v.abs() > 1e-4));
}

// ── MinimalBackend — exercises default trait implementations ──────────────────

use larql_compute::backend::DecodeBackend;
use ndarray::ArrayView2;
use std::cell::{Cell, RefCell};

struct MinimalBackend;

impl MatMul for MinimalBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        a.dot(&b)
    }
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        a.dot(&b.t())
    }
}
impl QuantMatVec for MinimalBackend {} // all methods default to None/false
impl DecodeBackend for MinimalBackend {} // all methods default to None/no-op
impl larql_compute::ComputeBackend for MinimalBackend {
    fn name(&self) -> &str {
        "minimal"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    // device_info: default → self.name().to_string()
    // supports:    default → false
}

struct RecordingQuantBackend {
    q4_calls: Cell<usize>,
    q4k_inputs: RefCell<Vec<Vec<f32>>>,
}

impl RecordingQuantBackend {
    fn new() -> Self {
        Self {
            q4_calls: Cell::new(0),
            q4k_inputs: RefCell::new(Vec::new()),
        }
    }
}

impl QuantMatVec for RecordingQuantBackend {
    fn q4_matvec(
        &self,
        _q4_data: &[u8],
        _q8_x: &[i8],
        _q8_scales: &[f32],
        num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        self.q4_calls.set(self.q4_calls.get() + 1);
        Some(vec![self.q4_calls.get() as f32; num_rows])
    }

    fn q4k_matvec(
        &self,
        _q4k_data: &[u8],
        x: &[f32],
        num_rows: usize,
        _hidden: usize,
    ) -> Option<Vec<f32>> {
        self.q4k_inputs.borrow_mut().push(x.to_vec());
        Some(vec![x.iter().sum(); num_rows])
    }
}

struct ForwardingDecodeBackend {
    full_pipeline_calls: Cell<usize>,
    decode_calls: Cell<usize>,
    prefill_calls: Cell<usize>,
    moe_calls: Cell<usize>,
}

impl ForwardingDecodeBackend {
    fn new() -> Self {
        Self {
            full_pipeline_calls: Cell::new(0),
            decode_calls: Cell::new(0),
            prefill_calls: Cell::new(0),
            moe_calls: Cell::new(0),
        }
    }
}

impl DecodeBackend for ForwardingDecodeBackend {
    fn full_pipeline_q4(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        _hidden: usize,
        _inter: usize,
        seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        self.full_pipeline_calls
            .set(self.full_pipeline_calls.get() + 1);
        Some(vec![seq_len as f32])
    }

    fn decode_token(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        _hidden: usize,
        _inter: usize,
    ) -> Option<Vec<f32>> {
        self.decode_calls.set(self.decode_calls.get() + 1);
        Some(x.to_vec())
    }

    fn decode_token_with_moe(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        _hidden: usize,
        _inter: usize,
        moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        self.moe_calls.set(self.moe_calls.get() + 1);
        Some(moe_fn(7, x))
    }

    fn prefill_kquant(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        _hidden: usize,
        _inter: usize,
        seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        self.prefill_calls.set(self.prefill_calls.get() + 1);
        Some(x.iter().take(seq_len).copied().collect())
    }
}

#[test]
fn default_device_info_delegates_to_name() {
    let be = MinimalBackend;
    assert_eq!(be.device_info(), "minimal");
}

#[test]
fn default_supports_returns_false() {
    let be = MinimalBackend;
    assert!(!be.supports(larql_compute::Capability::F32Gemv));
    assert!(!be.supports(larql_compute::Capability::FullPipelineQ4));
}

#[test]
fn quant_matvec_defaults_forward_and_dequantise_q8_tail() {
    let be = RecordingQuantBackend::new();
    let weights = vec![0u8; 18];
    let x = vec![1.0f32; 32];

    assert_eq!(
        be.quant_matvec(QuantFormat::Q4_0, &weights, &x, 2, 32)
            .unwrap(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        be.quant_matvec_q8_input(QuantFormat::Q4_0, &weights, &[1, -1], &[0.5], 3, 2)
            .unwrap(),
        vec![2.0, 2.0, 2.0]
    );

    let mut q8 = vec![2i8; 35];
    q8[32] = -3;
    q8[33] = 4;
    q8[34] = -5;
    let out = be
        .quant_matvec_q8_input(QuantFormat::Q4_K, &weights, &q8, &[0.25], 1, 35)
        .unwrap();
    let captured = be.q4k_inputs.borrow();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0][0], 0.5);
    assert_eq!(&captured[0][32..], &[-3.0, 4.0, -5.0]);
    assert_eq!(out, vec![16.0 - 4.0]);
}

#[test]
fn decode_defaults_forward_to_specialized_entrypoints() {
    let be = ForwardingDecodeBackend::new();
    let layers: Vec<larql_compute::FullPipelineLayer<'_>> = Vec::new();
    let x = vec![1.0f32, 2.0, 3.0, 4.0];

    assert_eq!(
        be.full_pipeline_q4_with_head_replacement(&layers, &x, 4, 8, 3, false, 0.0, 2, 0, &x,)
            .unwrap(),
        vec![3.0]
    );
    assert_eq!(be.full_pipeline_calls.get(), 1);

    let mut ignored = 0usize;
    assert_eq!(
        be.decode_token_with_moe(&layers, &x, 4, 8, &mut |layer, h| {
            ignored += layer + h.len();
            vec![9.0; h.len()]
        },)
            .unwrap(),
        vec![9.0; 4]
    );
    assert_eq!(be.moe_calls.get(), 1);
    assert_eq!(ignored, 11);

    let mut fired = 0usize;
    let mut collected = 0usize;
    assert_eq!(
        be.decode_token_with_moe_split(
            &layers,
            &x,
            4,
            8,
            &mut |layer, h| fired += layer + h.len(),
            &mut |layer| {
                collected += layer;
                vec![layer as f32; 2]
            },
        )
        .unwrap(),
        vec![7.0, 7.0]
    );
    assert_eq!((fired, collected), (11, 7));

    assert_eq!(
        be.decode_token_split_profile(&layers, &x, 4, 8),
        (Some(x.clone()), 0.0, 0.0, 0.0)
    );
    assert_eq!(be.decode_calls.get(), 1);

    assert_eq!(
        be.prefill_kquant_with_head_replacement(&layers, &x, 4, 8, 2, false, 0.0, 0, 0, &x,)
            .unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(be.prefill_calls.get(), 1);
}

#[test]
fn default_quant_matvec_stubs_return_none() {
    let be = MinimalBackend;
    let dummy = vec![0u8; 18];
    let dummy_i8 = vec![0i8; 32];
    let dummy_f32 = vec![0.0f32; 256];
    let dummy_scales = vec![0.0f32; 1];
    assert!(be
        .quant_matvec(QuantFormat::BF16, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec(QuantFormat::F16, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec(QuantFormat::F32, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec(QuantFormat::Q4_0, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec(QuantFormat::Q8_0, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec(QuantFormat::Q4_K, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec(QuantFormat::Q4_KF, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec(QuantFormat::Q6_K, &dummy, &dummy_f32, 1, 256)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::BF16, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::F16, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::F32, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::Q4_0, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::Q8_0, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::Q4_K, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::Q4_KF, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .quant_matvec_q8_input(QuantFormat::Q6_K, &dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .q4_matvec(&dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .q4_matvec_topk1(&dummy, &dummy_i8, &dummy_scales, 1, 32)
        .is_none());
    assert!(be
        .q4_matvec_topk(&dummy, &dummy_i8, &dummy_scales, 1, 32, 4)
        .is_none());
    assert!(be.q4_vecmat(&dummy_f32[..32], &dummy, 32, 256).is_none());
    assert!(be.q4k_matvec(&dummy, &dummy_f32[..256], 1, 256).is_none());
    assert!(be
        .q4k_matvec_stride32(&dummy, &dummy_f32[..256], 1, 256)
        .is_none());
    assert!(be
        .q4k_matmul(&dummy, &dummy_f32[..256], 1, 256, 1)
        .is_none());
    assert!(be.q6k_matvec(&dummy, &dummy_f32[..256], 1, 256).is_none());
    assert!(be
        .q4_matvec_pair_batch(&dummy, &dummy, &dummy_f32[..256], 1, 1, 256)
        .is_none());
    assert!(!be.supports_quant(::larql_compute::QuantFormat::Q4_K));
}

#[test]
fn default_decode_stubs() {
    let be = MinimalBackend;
    let layers: Vec<larql_compute::FullPipelineLayer<'_>> = Vec::new();
    let x = vec![0.0f32; 4];
    assert!(!be.has_kv_cache());
    assert_eq!(be.kv_cache_len(), 0);
    be.populate_kv_layer(0, &x, &x, 1, 1, 4);
    be.preallocate_kv_cache_per_layer(&[(1, 4)], 16);
    be.truncate_kv_cache(0);
    be.reset_kv_cache(); // default no-op, must not panic
    assert!(be
        .full_pipeline_q4(&layers, &x, 4, 8, 1, false, 0.0)
        .is_none());
    assert!(be
        .full_pipeline_q4_with_head_replacement(&layers, &x, 4, 8, 1, false, 0.0, 0, 0, &x,)
        .is_none());
    assert!(be.multi_layer_q4_ffn(&[], &x, 8, 4).is_none());
    assert!(be.decode_token(&layers, &x, 4, 8).is_none());

    let mut fired = 0usize;
    let mut collected = 0usize;
    assert!(be
        .decode_token_with_moe(&layers, &x, 4, 8, &mut |layer, h| {
            fired += layer + h.len();
            vec![1.0; h.len()]
        },)
        .is_none());
    assert!(be
        .decode_token_with_moe_split(
            &layers,
            &x,
            4,
            8,
            &mut |layer, h| {
                fired += layer + h.len();
            },
            &mut |layer| {
                collected += layer + 1;
                vec![2.0; 4]
            },
        )
        .is_none());
    assert_eq!((fired, collected), (0, 0));
    assert_eq!(
        be.decode_token_split_profile(&layers, &x, 4, 8),
        (None, 0.0, 0.0, 0.0)
    );
    assert!(be
        .prefill_kquant(&layers, &x, 4, 8, 1, false, 0.0)
        .is_none());
    assert!(be
        .full_pipeline_kquant_capture_pre_wo(&layers, &x, 4, 8, 1, false, 0.0, 0, 0,)
        .is_none());
    assert!(be
        .prefill_kquant_with_head_replacement(&layers, &x, 4, 8, 1, false, 0.0, 0, 0, &x,)
        .is_none());
}
