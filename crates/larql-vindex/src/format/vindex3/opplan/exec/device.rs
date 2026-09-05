//! The device backend: the plan's matrix work on an injected [`MatMul`]
//! device.
//!
//! **Layering.** This crate never links a GPU API. The seam it binds is
//! `larql-compute`'s [`MatMul`] trait — the same abstraction the serving
//! path dispatches through — and the *caller* injects the concrete
//! device (the CLI hands in `larql-compute-metal`'s backend for
//! `--backend metal`). vindex stays device-agnostic on every target;
//! whatever implements `MatMul` tomorrow (a second GPU API, a remote
//! device) lowers the same plan with no change here.
//!
//! **What is on the device, and what is not.** Every matrix–vector
//! product — Q/K/V/O projections, the attention gate projection, the
//! three FFN projections, and the vocabulary head — dispatches through
//! the injected device's gemv kernels. The elementwise glue between
//! them (norms, RoPE, softmax, activations, residual adds) runs on the
//! CPU, **deliberately as the production backend's own code**: sharing
//! the glue with the CPU production backend means any device-vs-
//! production divergence is attributable to device matmul arithmetic
//! alone, and the reference backend remains the fully independent leg
//! of the triangle.
//!
//! **Weight residency rides on the format.** Constructed with
//! [`WeightFormat::F16`], the backend declares f16 matrix operands; the
//! interpreter (or a decode session) loads each one once into a stable,
//! page-aligned allocation, and a device whose buffer cache keys on
//! `(pointer, length)` — the Metal backend's does — keeps every weight
//! resident across calls instead of re-uploading it per forward. With
//! `F32` the backend behaves as the first device rung did: correct, and
//! paying a full upload per fresh allocation.
//!
//! **It fails closed.** Matmuls use the *force* gemv variants, which
//! never fall back below a FLOP threshold — a threshold fallback would
//! quietly turn "device parity" into "CPU parity" for small shapes. If
//! the device has no kernel for the declared weight format, the call
//! errors naming the shape; nothing silently substitutes other
//! arithmetic.
//!
//! **Dispatches are serialised** behind a mutex. A GPU executes one
//! command buffer at a time anyway, and this sidesteps the known Metal
//! test-parallelism race class while the interpreter drives positions
//! from worker threads.

use std::sync::Mutex;

use larql_compute::backend::MatMul;
use larql_models::config::{Activation, GateActivation, GateCombine, GatePlacement, GateSource};

use super::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, DispatchStats, FfnCall,
    GateCall, MatrixClass, NormCall, PlanBackend, ProjectCall, ProjectedQkv, RoutedFfnCall,
    WeightFormat, WeightFormats, WeightSlice,
};
use super::production::{
    add_expert_bias, add_output_bias, add_projection_biases, aggregate_heads,
    condition_qk_in_place, condition_v_in_place, expert_inner, router_input, select_experts,
    ProductionBackend, FUSED_BRANCHES,
};
use crate::error::VindexError;
use larql_compute::cpu::ops::geglu::{geglu_silu_alloc, silu};
use ndarray::ArrayView2;
use rayon::prelude::*;

use super::production::unsupported_activation;
use super::realization::{
    class_of, common_selection, resident_profile, RealizationBackend, RealizationForm,
    RealizationId, RefusalKind, RepresentationFacts, Selection, SelectionReason, SelectionRefusal,
};
use crate::format::vindex3::opplan::planned::PlannedOperand;
use larql_compute::ffn::gelu_tanh;

/// One MXFP4 matrix as the device trait consumes it:
/// `(packed, scales, n, k)`.
type Mxfp4Matrix<'a> = (&'a [u8], &'a [u8], usize, usize);

/// One NVFP4 matrix as the device trait consumes it:
/// `(packed, scales, tensor_scale, n, k)`.
type Nvfp4Matrix<'a> = (&'a [u8], &'a [u8], f32, usize, usize);

/// Device realisation: injected-device matmuls, production-CPU glue.
pub struct DevicePlanBackend<M: MatMul + Send> {
    /// The injected device — for `--backend metal`, the same serving
    /// backend `larql run` computes with, not a lookalike wrapper.
    device: Mutex<M>,
    /// CPU glue, shared with the production backend on purpose (see
    /// module docs).
    glue: ProductionBackend,
    /// Reported through [`PlanBackend::name`] and hence the engine tag,
    /// so a dump names the concrete device and realisation that
    /// produced it.
    name: String,
    /// The matrix-operand representations this backend asks the
    /// interpreter for, per matrix class (see module docs on residency).
    formats: WeightFormats,
    /// Cumulative time inside device dispatch calls, and how many
    /// submissions those were. Diagnostic only — never read by the
    /// arithmetic, so it cannot change a result.
    device_nanos: std::sync::atomic::AtomicU64,
    submissions: std::sync::atomic::AtomicU64,
}

impl<M: MatMul + Send> DevicePlanBackend<M> {
    /// `name` should carry device and realisation (e.g. `metal-r2-f16`)
    /// so a dump can never be mistaken for another lowering.
    pub fn new(device: M, name: impl Into<String>, format: WeightFormat) -> Self {
        Self::with_formats(device, name, WeightFormats::uniform(format))
    }

    /// Per-class formats, for realisations that keep numerically
    /// sensitive classes (the head, attention) in a wider format than
    /// the FFN bulk.
    pub fn with_formats(device: M, name: impl Into<String>, formats: WeightFormats) -> Self {
        Self {
            device: Mutex::new(device),
            glue: ProductionBackend::new(),
            name: name.into(),
            formats,
            device_nanos: std::sync::atomic::AtomicU64::new(0),
            submissions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Record one submission's wall time. `count` is the number of
    /// command buffers the call made, which is one for every path here.
    fn record(&self, started: std::time::Instant, count: u64) {
        use std::sync::atomic::Ordering;
        self.device_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.submissions.fetch_add(count, Ordering::Relaxed);
    }

    /// `out[out_dim] = W[out_dim, in_dim] · x` on the device, always,
    /// in whichever representation the weight arrived in.
    fn gemv(
        &self,
        weight: WeightSlice<'_>,
        out_dim: usize,
        in_dim: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        let started = std::time::Instant::now();
        let device = self.device.lock().expect("device dispatch lock");
        // No device kernel consumes stored bf16 yet; refusing names the
        // gap rather than silently widening 50 GB on the host.
        if matches!(
            weight,
            WeightSlice::Bf16(_) | WeightSlice::Q8 { .. } | WeightSlice::Q4 { .. }
        ) {
            return Err(VindexError::Parse(format!(
                "the device backend has no {} kernel; declare F16 or F32 for it",
                weight.representation()
            )));
        }
        let result = match weight {
            WeightSlice::Bf16(_)
            | WeightSlice::Q8 { .. }
            | WeightSlice::Q4 { .. }
            | WeightSlice::KQuant { .. } => {
                return Err(VindexError::Parse(format!(
                    "the device backend has no {} kernel; declare F16 or F32 for it",
                    weight.representation()
                )))
            }
            WeightSlice::F32(w) => {
                let view = ArrayView2::from_shape((out_dim, in_dim), w).map_err(|e| {
                    VindexError::Parse(format!(
                        "device gemv: weight slice is not [{out_dim}, {in_dim}]: {e}"
                    ))
                })?;
                device.f32_gemv_force(view, x).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "device f32_gemv [{out_dim} x {in_dim}] refused — no kernel or out of \
                         memory"
                    ))
                })
            }
            WeightSlice::F16(bytes) => {
                // The slice may be page-padded beyond the matrix (see
                // `weights::AlignedBytes`); geometry travels as n/k.
                if bytes.len() < out_dim * in_dim * 2 {
                    return Err(VindexError::Parse(format!(
                        "device f16 gemv: {} weight bytes cannot hold [{out_dim} x {in_dim}]",
                        bytes.len()
                    )));
                }
                device
                    .f16_gemv_force(bytes, x, out_dim, in_dim)
                    .ok_or_else(|| {
                        VindexError::Parse(format!(
                            "device f16_gemv [{out_dim} x {in_dim}] refused — no kernel or out \
                             of memory"
                        ))
                    })
            }
            WeightSlice::Mxfp4 { packed, scales } => device
                .mxfp4_gemv(packed, scales, x, out_dim, in_dim)
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "device mxfp4_gemv [{out_dim} x {in_dim}] refused — no kernel, bad \
                         geometry, or out of memory"
                    ))
                }),
            WeightSlice::Nvfp4 {
                packed,
                scales,
                tensor_scale,
            } => device
                .nvfp4_gemv(packed, scales, tensor_scale, x, out_dim, in_dim)
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "device nvfp4_gemv [{out_dim} x {in_dim}] refused — no kernel, bad \
                         geometry, or out of memory"
                    ))
                }),
        };
        drop(device);
        self.record(started, 1);
        result
    }

    /// Several matrices against one input vector. When every weight is
    /// f16 this goes through the device's multi-gemv — one submission,
    /// one wait, one input upload — which is where a serialised decode
    /// spends most of its time. Any f32 weight falls back to sequential
    /// dispatches; results are bit-identical either way (the multi path
    /// encodes the same kernel with the same per-dispatch arguments).
    fn gemv_multi(
        &self,
        weights: &[(WeightSlice<'_>, usize, usize)],
        x: &[f32],
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        let started = std::time::Instant::now();
        let all_f16: Option<Vec<(&[u8], usize, usize)>> = weights
            .iter()
            .map(|&(w, n, k)| match w {
                WeightSlice::F16(bytes) if bytes.len() >= n * k * 2 => Some((bytes, n, k)),
                _ => None,
            })
            .collect();
        if let Some(mats) = all_f16 {
            let out = self
                .device
                .lock()
                .expect("device dispatch lock")
                .f16_gemv_multi(&mats, x)
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "device f16_gemv_multi ({} matrices) refused — no kernel or out of memory",
                        mats.len()
                    ))
                });
            self.record(started, 1);
            return out;
        }
        let all_mxfp4: Option<Vec<Mxfp4Matrix>> = weights
            .iter()
            .map(|&(w, n, k)| match w {
                WeightSlice::Mxfp4 { packed, scales } => Some((packed, scales, n, k)),
                _ => None,
            })
            .collect();
        if let Some(mats) = all_mxfp4 {
            let out = self
                .device
                .lock()
                .expect("device dispatch lock")
                .mxfp4_gemv_multi(&mats, x)
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "device mxfp4_gemv_multi ({} matrices) refused — no kernel, bad \
                         geometry, or out of memory",
                        mats.len()
                    ))
                });
            self.record(started, 1);
            return out;
        }
        let all_nvfp4: Option<Vec<Nvfp4Matrix>> = weights
            .iter()
            .map(|&(w, n, k)| match w {
                WeightSlice::Nvfp4 {
                    packed,
                    scales,
                    tensor_scale,
                } => Some((packed, scales, tensor_scale, n, k)),
                _ => None,
            })
            .collect();
        if let Some(mats) = all_nvfp4 {
            let out = self
                .device
                .lock()
                .expect("device dispatch lock")
                .nvfp4_gemv_multi(&mats, x)
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "device nvfp4_gemv_multi ({} matrices) refused — no kernel, bad \
                         geometry, or out of memory",
                        mats.len()
                    ))
                });
            self.record(started, 1);
            return out;
        }
        // Mixed formats: falls back to one submission per matrix, each of
        // which records itself.
        weights
            .iter()
            .map(|&(w, n, k)| self.gemv(w, n, k, x))
            .collect()
    }

    /// One position's Q/K/V projections on the device — one submission —
    /// conditioned by the production glue; the arithmetic shared by the
    /// batch path and the decode step.
    fn project_position(
        &self,
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
    ) -> Result<ProjectedQkv, VindexError> {
        let q_rows = call.num_q_heads * call.head_dim;
        let kv_rows = call.num_kv_heads * call.head_dim;
        let mut qkv = self.gemv_multi(
            &[
                (call.w_q, q_rows, call.hidden),
                (call.w_k, kv_rows, call.hidden),
                (call.w_v, kv_rows, call.hidden),
            ],
            pre,
        )?;
        let mut v = qkv.pop().expect("three matrices in, three vectors out");
        let mut k = qkv.pop().expect("three matrices in, three vectors out");
        let mut q = qkv.pop().expect("three matrices in, three vectors out");
        add_projection_biases(call, &mut q, &mut k, &mut v);
        condition_v_in_place(call, &mut v);
        condition_qk_in_place(call, position, &mut q, &mut k)?;
        Ok((q, k, v))
    }
}

impl<M: MatMul + Send> PlanBackend for DevicePlanBackend<M> {
    fn name(&self) -> &str {
        &self.name
    }

    fn dispatch_stats(&self) -> Option<DispatchStats> {
        use std::sync::atomic::Ordering;
        Some(DispatchStats {
            device_nanos: self.device_nanos.load(Ordering::Relaxed),
            submissions: self.submissions.load(Ordering::Relaxed),
        })
    }

    /// Per class, and deliberately not per operand: a device backend's
    /// format is about what its kernels can execute and what stays
    /// resident in device memory, and neither of those turns on a host
    /// cache boundary. The operand's size is available and ignored on
    /// purpose.
    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        let bank = self.formats.for_class(MatrixClass::RoutedExpertBank);
        if let Some(common) = common_selection(operand, facts, bank) {
            return common;
        }
        let refuse = |kind| {
            Box::new(SelectionRefusal {
                operand: operand.operand.clone(),
                operation: operand.operation,
                representation: facts.label.clone(),
                requested: operand.access,
                kind,
                considered: vec![],
            })
        };
        let Some(class) = class_of(operand.operation) else {
            return Err(refuse(RefusalKind::MissingRealization));
        };
        if facts.registered.is_none() {
            return Err(refuse(RefusalKind::UnregisteredRepresentation));
        }
        // This backend's class table is its own declaration for its own
        // target: the one candidate it offers, named as such.
        let format = self.formats.for_class(class);
        let id = RealizationId {
            backend: RealizationBackend::Device,
            form: RealizationForm::DeviceResident(format),
        };
        Ok(Selection {
            realization: id,
            residency: resident_profile(format),
            reason: SelectionReason::DeviceClassTable,
            candidates: vec![id],
        })
    }

    fn prepare(&self, weights: &[WeightSlice<'_>]) {
        // Wire every f16 weight in one submission. Without this, a
        // driver's wired-page collector un-wires idle buffers between
        // submissions and a large-model decode pays a re-wire on every
        // touch — measured 10× on a 60 GB f16 working set. After one
        // pass, steps are fast enough to keep themselves wired.
        let mut streams: Vec<&[u8]> = Vec::with_capacity(weights.len() * 2);
        for w in weights {
            match w {
                // A residency hint computes nothing and must change no
                // number, so an unplaceable format is skipped here; the
                // refusal that matters fires where it would be USED.
                WeightSlice::Bf16(_)
                | WeightSlice::Q8 { .. }
                | WeightSlice::Q4 { .. }
                | WeightSlice::KQuant { .. } => continue,
                WeightSlice::F16(bytes) => streams.push(bytes),
                WeightSlice::Mxfp4 { packed, scales }
                | WeightSlice::Nvfp4 { packed, scales, .. } => {
                    streams.push(packed);
                    streams.push(scales);
                }
                WeightSlice::F32(_) => {}
            }
        }
        self.device
            .lock()
            .expect("device dispatch lock")
            .wire_resident(&streams);
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.glue.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.glue.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.gemv(call.weight, call.out_dim, call.in_dim, call.x)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        // Same structure as the production backend's attention; the only
        // substitution is which arithmetic performs each projection.
        // Projections stay serial over positions here — the GPU queue is
        // one lane, so parallel callers would only contend on the lock.
        let mut queries = Vec::with_capacity(call.inputs.len());
        let mut keys = Vec::with_capacity(call.inputs.len());
        let mut values = Vec::with_capacity(call.inputs.len());
        for (position, pre) in call.inputs.iter().enumerate() {
            let (q, k, v) = self.project_position(&call, position, pre)?;
            queries.push(q);
            keys.push(k);
            values.push(v);
        }

        // Score/softmax/weighted-V on the CPU (parallel over query
        // positions, arithmetic per position untouched); the gate and
        // output projections return to the device, serially — one GPU
        // lane again.
        let aggregated: Vec<Vec<f32>> = queries
            .par_iter()
            .enumerate()
            .map(|(position, query)| {
                aggregate_heads(
                    &call,
                    position,
                    query,
                    |p| keys[p].as_slice(),
                    |p| values[p].as_slice(),
                )
            })
            .collect();

        let mut out = Vec::with_capacity(aggregated.len());
        for (position, mut concat) in aggregated.into_iter().enumerate() {
            if let Some(GateCall { spec, weight }) = &call.gate {
                // Exhaustive on the judged semantics (see attend_position).
                // A fused query/gate projection needs the device
                // projection to emit `2 · head_dim` per head and gather
                // the halves; that is a kernel change, and this rung does
                // not touch Metal. Refused rather than silently gated
                // from the wrong rows.
                match spec.source {
                    GateSource::AttentionInput => {}
                    GateSource::FusedQueryProjection => {
                        return Err(VindexError::Parse(
                            "a fused query/gate projection has no device kernel; refusing"
                                .to_string(),
                        ))
                    }
                }
                let GateActivation::Sigmoid = spec.activation;
                let GateCombine::ElementwiseMultiply = spec.combine;
                let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
                let q_rows = call.num_q_heads * call.head_dim;
                let gate_values =
                    self.gemv(*weight, q_rows, call.hidden, &call.inputs[position])?;
                for (c, g) in concat.iter_mut().zip(&gate_values) {
                    *c *= 1.0 / (1.0 + (-g).exp());
                }
            }
            let mut projected = self.gemv(
                call.w_o,
                call.hidden,
                call.num_q_heads * call.head_dim,
                &concat,
            )?;
            add_output_bias(&call, &mut projected);
            out.push(projected);
        }
        Ok(AttentionOut {
            outputs: out,
            keys,
            values,
        })
    }

    fn attention_step(&self, step: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        let call = &step.op;
        let pre = &call.inputs[0];
        let q_rows = call.num_q_heads * call.head_dim;
        let kv_rows = call.num_kv_heads * call.head_dim;

        // Q/K/V and the judged gate all read the attention input, so
        // they ride one submission; the gate values are only *used*
        // after aggregation, exactly where the batch path applies them.
        let mut mats = vec![
            (call.w_q, q_rows, call.hidden),
            (call.w_k, kv_rows, call.hidden),
            (call.w_v, kv_rows, call.hidden),
        ];
        if let Some(GateCall { weight, .. }) = &call.gate {
            mats.push((*weight, q_rows, call.hidden));
        }
        let mut projected = self.gemv_multi(&mats, pre)?;
        let gate_values = call
            .gate
            .as_ref()
            .map(|_| projected.pop().expect("gate matrix in, gate vector out"));
        let mut v = projected.pop().expect("QKV in, three vectors out");
        let mut k = projected.pop().expect("QKV in, three vectors out");
        let mut q = projected.pop().expect("QKV in, three vectors out");
        add_projection_biases(call, &mut q, &mut k, &mut v);
        condition_qk_in_place(call, step.position, &mut q, &mut k)?;

        let mut concat = aggregate_heads(
            call,
            step.position,
            &q,
            |p| {
                if p == step.position {
                    k.as_slice()
                } else {
                    step.keys[p].as_slice()
                }
            },
            |p| {
                if p == step.position {
                    v.as_slice()
                } else {
                    step.values[p].as_slice()
                }
            },
        );
        if let Some(GateCall { spec, .. }) = &call.gate {
            // Exhaustive on the judged semantics, like both CPU
            // backends: a new variant must be implemented here before
            // it can execute on the device.
            match spec.source {
                GateSource::AttentionInput => {}
                GateSource::FusedQueryProjection => {
                    return Err(VindexError::Parse(
                        "a fused query/gate projection has no device kernel; refusing".to_string(),
                    ))
                }
            }
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let gate_values = gate_values.expect("projected alongside QKV above");
            for (c, g) in concat.iter_mut().zip(&gate_values) {
                *c *= 1.0 / (1.0 + (-g).exp());
            }
        }
        let mut output = self.gemv(call.w_o, call.hidden, q_rows, &concat)?;
        add_output_bias(call, &mut output);
        Ok(AttentionStepOut {
            key: k,
            value: v,
            output,
        })
    }

    /// The routed FFN on the device: router, then each selected expert's
    /// fused gate/up and down through the same `gemv` seam as every other
    /// matrix — in whatever representation the bank was loaded (native
    /// MXFP4 when this backend declared it). Selection and the gate rule
    /// are the production glue, so a divergence is device matmul
    /// arithmetic alone.
    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        let routed_input = router_input(&call)?;
        let mut logits = self.gemv(
            WeightSlice::F32(call.router),
            call.experts,
            call.hidden,
            &routed_input,
        )?;
        let selected = select_experts(&call, &mut logits)?;
        let two_inter = FUSED_BRANCHES * call.intermediate;
        let mut out = vec![0.0f32; call.hidden];
        for (expert, weight) in selected {
            let mut fused = self.gemv(call.gate_up[expert], two_inter, call.hidden, call.x)?;
            add_expert_bias(&mut fused, call.gate_up_bias, expert);
            let inner = expert_inner(&call, &fused);
            let mut expert_out =
                self.gemv(call.down[expert], call.hidden, call.intermediate, &inner)?;
            add_expert_bias(&mut expert_out, call.down_bias, expert);
            for (acc, v) in out.iter_mut().zip(&expert_out) {
                *acc += weight * v;
            }
        }
        Ok(out)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        super::production::require_plain_gate("device", call.gate_policy)?;
        let inner = match call.gate {
            Some(gate_weight) => {
                // Up and gate read the same input: one submission.
                let mut pair = self.gemv_multi(
                    &[
                        (call.up, call.intermediate, call.hidden),
                        (gate_weight, call.intermediate, call.hidden),
                    ],
                    call.x,
                )?;
                let gate = pair.pop().expect("two matrices in, two vectors out");
                let up = pair.pop().expect("two matrices in, two vectors out");
                match call.activation {
                    Activation::Silu => geglu_silu_alloc(&gate, &up),
                    Activation::GeluTanh => gate
                        .iter()
                        .zip(&up)
                        .map(|(g, u)| gelu_tanh(*g) * u)
                        .collect(),
                    other => return Err(unsupported_activation("gated", other)),
                }
            }
            None => {
                let up = self.gemv(call.up, call.intermediate, call.hidden, call.x)?;
                match call.activation {
                    Activation::Silu => up.iter().map(|u| silu(*u)).collect(),
                    Activation::GeluTanh => up.iter().map(|u| gelu_tanh(*u)).collect(),
                    other => return Err(unsupported_activation("ungated", other)),
                }
            }
        };
        self.gemv(call.down, call.hidden, call.intermediate, &inner)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut logits = self.gemv(projection, vocab, hidden, x)?;
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(logits)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.glue.residual_add(acc, delta);
    }
}
