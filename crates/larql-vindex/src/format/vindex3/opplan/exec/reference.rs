//! The reference backend: naive f32, the semantic anchor.
//!
//! Shares **nothing** with `larql-compute`'s kernels. That is the whole
//! point of it — a reference that called the production kernels would
//! agree with them by construction, and the agreement would prove
//! nothing. Plain loops, row-major `[out, in]` weights, no BLAS, no
//! SIMD, no fusion.
//!
//! When the production backend disagrees with this one, this one is
//! right about *meaning* and may well be wrong about speed. Divergence
//! is a bug in the production backend or a hole in the seam, never a
//! licence to change what the plan means.

use larql_models::config::{
    AttentionSinkSpec, GateActivation, GateCombine, GatePlacement, GateSource, QkNormScope,
};

use super::super::super::graph::policy::AttentionSpan;
use super::backend::{
    AttentionCall, AttentionStepCall, AttentionStepOut, FfnCall, GateCall, NormCall, PlanBackend,
    ProjectCall, ProjectedQkv, QkNormCall,
};
use super::kernels::{
    activate, matvec, norm, rope_rotate, sigmoid, softcap, softmax, softmax_with_sink,
};
use crate::error::VindexError;
use larql_models::config::NormType;
use larql_models::config::PositionPolicy;
use rayon::prelude::*;

/// Name reported by [`PlanBackend::name`].
const NAME: &str = "reference-f32";

/// Naive f32 realisation of every plan operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceBackend;

impl ReferenceBackend {
    pub fn new() -> Self {
        Self
    }

    /// Q/K normalisation: weighted per-head when the plan binds weights,
    /// parameter-free when the surface judged it. Both may apply.
    fn apply_qk_norm(
        call: &AttentionCall<'_>,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), VindexError> {
        let head_dim = call.head_dim;
        let eps = call.qk_norm_eps;
        if let Some(QkNormCall {
            scope,
            weight_offset,
            q_weight,
            k_weight,
        }) = &call.qk_norm
        {
            match scope {
                QkNormScope::PerHead => {
                    for head in q.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, q_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, k_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                }
                QkNormScope::FullProjection => {
                    return Err(VindexError::Parse(
                        "full-projection QK norm has no judged reference execution yet".to_string(),
                    ));
                }
            }
        }
        if call.parameter_free_qk_norm.q {
            for head in q.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        if call.parameter_free_qk_norm.k {
            for head in k.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        Ok(())
    }

    /// One position's Q/K/V projections with QK normalisation, query
    /// scale and position encoding applied in the judged order — the
    /// arithmetic both the batch path and the decode step share, so the
    /// two cannot disagree about a single position.
    fn project_position(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
    ) -> Result<ProjectedQkv, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        let mut q = matvec(call.w_q.as_f32()?, q_rows, call.hidden, pre);
        let mut k = matvec(call.w_k.as_f32()?, kv_rows, call.hidden, pre);
        let mut v = matvec(call.w_v.as_f32()?, kv_rows, call.hidden, pre);
        // Biases belong to the projections: added before anything reads
        // the projected values (QK-norm, rope, the cache).
        if let Some(bias) = &call.bias {
            add_in_place(&mut q, bias.q);
            add_in_place(&mut k, bias.k);
            add_in_place(&mut v, bias.v);
        }

        Self::apply_qk_norm(call, &mut q, &mut k)?;
        if let Some(query_scale) = call.query_scale {
            for value in &mut q {
                *value *= query_scale as f32;
            }
        }
        match call.position {
            PositionPolicy::Rope { theta } => {
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate(head, position, theta);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate(head, position, theta);
                }
            }
            // Represented, not yet executed here: YaRN is scaled frequencies
            // AND an attention amplitude, and rotating at the bare theta would
            // silently serve the wrong model. Refuse until A-9.3 lands it.
            PositionPolicy::Yarn { .. } => {
                return Err(VindexError::Parse(
                    "PositionPolicy::Yarn is carried by the container but this backend does not \
                     execute YaRN rotary scaling yet (A-9.3); refusing rather than rotating at the \
                     unscaled theta"
                        .to_string(),
                ));
            }
            PositionPolicy::None => {}
        }
        Ok((q, k, v))
    }

    /// One query position's scores, softmax, weighted-V aggregation,
    /// gate, and output projection. `key_of`/`value_of` abstract where
    /// K/V rows live (the batch path's local vectors, or the decode
    /// step's interpreter-owned cache plus the fresh row).
    fn attend_position<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let group = call.num_q_heads / call.num_kv_heads;
        // Span: which key positions this query may attend to. Exhaustive
        // over the vocabulary so a new span kind forces a decision here
        // instead of silently meaning "whole prefix".
        let start = match (call.span, call.window) {
            (AttentionSpan::Sliding, Some(window)) => (position + 1).saturating_sub(window),
            (AttentionSpan::Sliding, None) | (AttentionSpan::Full, _) => 0,
            // A spatial window bounds a region, not a position range; no
            // generic op lowers a perception component today.
            (AttentionSpan::Windowed, _) => 0,
        };
        let mut concat = vec![0.0f32; q_rows];
        for q_head in 0..call.num_q_heads {
            let kv_head = q_head / group;
            let q_slice = &query[q_head * head_dim..(q_head + 1) * head_dim];
            let mut scores: Vec<f32> = (start..=position)
                .map(|key_position| {
                    let k_slice =
                        &key_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for (a, b) in q_slice.iter().zip(k_slice) {
                        dot += a * b;
                    }
                    let mut score = dot * call.score_scale as f32;
                    if let Some(cap) = call.logit_softcapping {
                        score = softcap(score, cap);
                    }
                    score
                })
                .collect();
            match &call.sinks {
                // Exhaustive on the judged semantics: a new variant must
                // be implemented here before it can execute.
                Some(sinks) => {
                    let AttentionSinkSpec::SoftmaxDenominator = sinks.spec;
                    softmax_with_sink(&mut scores, sinks.logits[q_head]);
                }
                None => softmax(&mut scores),
            }
            let head_out = &mut concat[q_head * head_dim..(q_head + 1) * head_dim];
            for (offset, key_position) in (start..=position).enumerate() {
                let v_slice = &value_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
                let weight = scores[offset];
                for (acc, v) in head_out.iter_mut().zip(v_slice) {
                    *acc += weight * v;
                }
            }
        }

        if let Some(GateCall { spec, weight }) = &call.gate {
            // Exhaustive on the judged semantics: a new variant must
            // be implemented here before it can execute.
            let GateSource::AttentionInput = spec.source;
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let gate_values = matvec(weight.as_f32()?, q_rows, call.hidden, gate_input);
            for (c, g) in concat.iter_mut().zip(&gate_values) {
                *c *= sigmoid(*g);
            }
        }

        let mut out = matvec(call.w_o.as_f32()?, call.hidden, q_rows, &concat);
        if let Some(bias) = &call.bias {
            add_in_place(&mut out, bias.o);
        }
        Ok(out)
    }
}

/// `x[i] += b[i]`; a bias of the wrong length is a geometry bug closure
/// should have refused, so it panics rather than pads.
fn add_in_place(x: &mut [f32], b: &[f32]) {
    assert_eq!(
        x.len(),
        b.len(),
        "bias length must equal the projection's rows"
    );
    for (x, b) in x.iter_mut().zip(b) {
        *x += b;
    }
}

impl PlanBackend for ReferenceBackend {
    fn name(&self) -> &str {
        NAME
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        let row = &table[token as usize * hidden..(token as usize + 1) * hidden];
        match scale {
            Some(scale) => row.iter().map(|v| v * scale).collect(),
            None => row.to_vec(),
        }
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        norm(call.kind, call.x, call.weight, call.weight_offset, call.eps)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        Ok(matvec(
            call.weight.as_f32()?,
            call.out_dim,
            call.in_dim,
            call.x,
        ))
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        // Projections per position, with QK normalisation, query scale
        // and position encoding applied in the judged order. Positions
        // are independent, so they run in parallel with each position's
        // arithmetic untouched — bit-identical to the serial order.
        let projected: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = call
            .inputs
            .par_iter()
            .enumerate()
            .map(|(position, pre)| Self::project_position(&call, position, pre))
            .collect::<Result<_, VindexError>>()?;
        let mut queries = Vec::with_capacity(projected.len());
        let mut keys = Vec::with_capacity(projected.len());
        let mut values = Vec::with_capacity(projected.len());
        for (q, k, v) in projected {
            queries.push(q);
            keys.push(k);
            values.push(v);
        }

        // Each query position reads every position's K/V but writes only
        // its own output row — parallel over queries, arithmetic intact.
        queries
            .par_iter()
            .enumerate()
            .map(|(position, query)| {
                Self::attend_position(
                    &call,
                    position,
                    query,
                    |p| keys[p].as_slice(),
                    |p| values[p].as_slice(),
                    &call.inputs[position],
                )
            })
            .collect()
    }

    fn attention_step(&self, step: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        let call = &step.op;
        let pre = &call.inputs[0];
        let (q, k, v) = Self::project_position(call, step.position, pre)?;
        let output = Self::attend_position(
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
            pre,
        )?;
        Ok(AttentionStepOut {
            key: k,
            value: v,
            output,
        })
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        super::production::require_plain_gate("reference", call.gate_policy)?;
        let up = matvec(call.up.as_f32()?, call.intermediate, call.hidden, call.x);
        let inner: Vec<f32> = match call.gate {
            Some(gate_weight) => {
                let gate = matvec(
                    gate_weight.as_f32()?,
                    call.intermediate,
                    call.hidden,
                    call.x,
                );
                gate.iter()
                    .zip(&up)
                    .map(|(g, u)| activate(call.activation, *g) * u)
                    .collect()
            }
            None => up.iter().map(|u| activate(call.activation, *u)).collect(),
        };
        Ok(matvec(
            call.down.as_f32()?,
            call.hidden,
            call.intermediate,
            &inner,
        ))
    }

    fn output_head(
        &self,
        projection: super::backend::WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut logits = matvec(projection.as_f32()?, vocab, hidden, x);
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = softcap(*logit, cap);
            }
        }
        Ok(logits)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        for (a, b) in acc.iter_mut().zip(delta) {
            *a += b;
        }
    }
}
