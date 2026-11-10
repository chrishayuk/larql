//! Attention operands, KV writes and projection helpers.

use super::super::AttentionOp;
use crate::error::VindexError;
use backend::{
    AttentionCall, AttentionStepCall, BiasCall, GateCall, PlanBackend, ProjectCall, QkNormCall,
    SinkCall,
};
use kv::KvState;
use larql_models::config::GateSource;
use operands::OperandSource;
use weights::{load_weight, LoadedWeight};

#[allow(unused_imports)]
use super::*;

impl AttentionOperands {
    /// Load through the closure-verified path. QK-norm weights stay f32
    /// (elementwise glue, not matrix traffic).
    pub(in super::super) fn load(
        op: &AttentionOp,
        store: OperandSource<'_>,
        format: prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        // A K≡V layer names the K operand as `v`: the value projection is
        // the raw K projection (before the key's norm and rotation), which
        // is exactly what projecting `w_v` = W_k yields — the backends take
        // V before conditioning Q/K, and apply the parameter-free V norm
        // to it when the op carries one.
        Ok(Self {
            w_q: load_weight(store, &op.q, format(&op.q)?)?,
            w_k: load_weight(store, &op.k, format(&op.k)?)?,
            w_v: load_weight(store, &op.v, format(&op.v)?)?,
            w_o: load_weight(store, &op.o, format(&op.o)?)?,
            qk_weights: match &op.qk_norm {
                Some(qk) => Some((store.load(&qk.q)?, store.load(&qk.k)?)),
                None => None,
            },
            // **One physical projection, two consumers.**
            //
            // A `FusedQueryProjection` gate names the QUERY operand — the
            // op builder binds `OperandRole::AttnQ` for it — so loading it
            // here would hold Qwen3.8's `12288 x 5120` q_proj twice: 2.01
            // GB of the same bytes under two names. The call builder hands
            // both consumers the one slice instead.
            //
            // Keyed on the judged gate SOURCE rather than on the two
            // operands resolving to the same tensor. The source is what
            // the plan asserts; pointer identity would make this an
            // optimisation that silently stopped applying the day an
            // unrelated loader change broke the aliasing.
            gate: match &op.output_gate {
                Some(gate) if gate.spec.source != GateSource::FusedQueryProjection => Some(
                    load_weight(store, &gate.projection, format(&gate.projection)?)?,
                ),
                _ => None,
            },
            biases: match (&op.q_bias, &op.k_bias, &op.v_bias, &op.o_bias) {
                // All four (`attention_bias`) or Q/K/V alone (`qkv_bias`).
                (Some(q), Some(k), Some(v), o) => Some(AttentionBiases {
                    q: store.load(q)?,
                    k: store.load(k)?,
                    v: store.load(v)?,
                    o: o.as_ref().map(|o| store.load(o)).transpose()?,
                }),
                (None, None, None, None) => None,
                // Closure emits Q/K/V together, with or without O; any
                // other set is a plan the closure never produced.
                _ => {
                    return Err(VindexError::Parse(
                        "attention op carries a partial bias set; operand closure emits Q/K/V \
                         together, with the output bias or without it"
                            .to_string(),
                    ))
                }
            },
            sinks: match &op.sinks {
                Some(sinks) => Some(store.load(&sinks.logits)?),
                None => None,
            },
            rope_angles: super::rope_angles::RopeAngles::default(),
        })
    }

    /// Every matrix operand this attention holds, for residency
    /// preparation.
    /// Every matrix operand, for residency accounting.
    /// Each matrix paired with the operand it binds, field by field.
    pub(in super::super) fn bound<'a>(&'a self, op: &'a AttentionOp) -> Vec<accounting::Bound<'a>> {
        let mut out = vec![
            accounting::Bound::one(&op.q, &self.w_q),
            accounting::Bound::one(&op.k, &self.w_k),
            accounting::Bound::one(&op.v, &self.w_v),
            accounting::Bound::one(&op.o, &self.w_o),
        ];
        if let (Some(gate), Some(weight)) = (&op.output_gate, &self.gate) {
            out.push(accounting::Bound::one(&gate.projection, weight));
        }
        out
    }

    pub(in super::super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut all = vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o];
        if let Some(gate) = &self.gate {
            all.push(gate);
        }
        all
    }

    pub(in super::super) fn weight_slices(&self) -> Vec<backend::WeightSlice<'_>> {
        let mut slices = vec![
            self.w_q.slice(),
            self.w_k.slice(),
            self.w_v.slice(),
            self.w_o.slice(),
        ];
        if let Some(gate) = &self.gate {
            slices.push(gate.slice());
        }
        slices
    }

    /// A fully resolved call over `inputs`. Every judged fact travels as
    /// an argument; none is re-derived — and both the batch path and the
    /// decode step build their call here, so they cannot drift apart in
    /// what they carry.
    pub(in super::super) fn call<'a>(
        &'a self,
        op: &AttentionOp,
        inputs: &'a [Vec<f32>],
        qk_norm_eps: f64,
        hidden: usize,
    ) -> AttentionCall<'a> {
        let qk_norm = match (&op.qk_norm, &self.qk_weights) {
            (Some(qk), Some((q_weight, k_weight))) => Some(QkNormCall {
                scope: qk.scope,
                weight_offset: qk.weight_offset,
                q_weight,
                k_weight,
            }),
            _ => None,
        };
        let gate = match (&op.output_gate, &self.gate) {
            // Its own operand: an `AttentionInput` gate is a separate
            // matrix read over a separate activation.
            (Some(gate), Some(weight)) => Some(GateCall {
                spec: gate.spec,
                weight: weight.slice(),
            }),
            // The other half of the query projection — the same slice,
            // not a copy of it. A backend that computes both halves at
            // once reads these bytes once; one that projects again is
            // still CORRECT, because the operand really is `w_q`.
            (Some(gate), None) if gate.spec.source == GateSource::FusedQueryProjection => {
                Some(GateCall {
                    spec: gate.spec,
                    weight: self.w_q.slice(),
                })
            }
            _ => None,
        };
        let bias = self.biases.as_ref().map(|b| BiasCall {
            q: b.q.as_slice(),
            k: b.k.as_slice(),
            v: b.v.as_slice(),
            o: b.o.as_deref(),
        });
        let sinks = match (&op.sinks, &self.sinks) {
            (Some(op_sinks), Some(logits)) => Some(SinkCall {
                spec: op_sinks.spec,
                logits: logits.as_slice(),
            }),
            _ => None,
        };
        AttentionCall {
            inputs,
            hidden,
            num_q_heads: op.num_q_heads,
            num_kv_heads: op.num_kv_heads,
            head_dim: op.head_dim,
            w_q: self.w_q.slice(),
            w_k: self.w_k.slice(),
            w_v: self.w_v.slice(),
            w_o: self.w_o.slice(),
            qk_norm,
            parameter_free_qk_norm: op.parameter_free_qk_norm,
            qk_norm_eps,
            query_scale: op.query_scale,
            score_scale: op.score_scale,
            logit_softcapping: op.logit_softcapping,
            position: op.position,
            span: op.span,
            window: op.window,
            gate,
            bias,
            sinks,
            rope_angles: self.rope_angles,
        }
    }
}

/// One layer's attention driven position-by-position into the caller's
/// continuation state — the batch prefill's attention realisation.
///
/// The arithmetic per position is exactly the decode step's
/// (`attention_step` against the rows appended so far), so the rows
/// landing in the provider are the ones a later decode step reads,
/// and bit-identity with the batched [`attention`] is what the
/// decode-vs-batch parity gates already prove per backend — the
/// prefill gates re-pin it end to end. Positions are absolute:
/// appends continue from the provider's logical position, which the
/// caller advances once the whole traversal completes.
///
/// Sequential by necessity (each position reads the previous ones'
/// rows); the batch prefill is a semantic gate, not a fast path.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_into_kv<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
    op: &AttentionOp,
    operands: &AttentionOperands,
    inputs: &[Vec<f32>],
    qk_norm_eps: f64,
    hidden: usize,
    backend: &B,
    kv: &mut K,
    layer_index: usize,
) -> Result<Vec<Vec<f32>>, VindexError> {
    let base = kv.position();
    let mut outputs = Vec::with_capacity(inputs.len());
    for offset in 0..inputs.len() {
        let call = operands.call(op, &inputs[offset..=offset], qk_norm_eps, hidden);
        kv.prepare_layer(layer_index);
        let rows = kv.rows(layer_index);
        let out = backend.attention_step(AttentionStepCall::new(call, base + offset, rows)?)?;
        kv.append(layer_index, out.key, out.value);
        outputs.push(out.output);
    }
    Ok(outputs)
}

/// The one value of a `[1]` layer-scale operand — refused if the operand
/// is not exactly one value, since a silently-broadcast vector would be a
/// different op.
pub fn layer_scalar_of(values: &[f32]) -> Result<f32, VindexError> {
    match values {
        [scale] => Ok(*scale),
        other => Err(VindexError::Parse(format!(
            "layer scale operand holds {} values; the op is one scalar",
            other.len()
        ))),
    }
}

/// Project one vector through an `[out, in]` weight.
///
/// Kept as a named helper so the interpreter never open-codes a matvec:
/// every projection in a plan goes through the backend.
#[allow(dead_code)]
pub(super) fn project<B: PlanBackend + ?Sized>(
    backend: &B,
    weight: backend::WeightSlice<'_>,
    out_dim: usize,
    in_dim: usize,
    x: &[f32],
) -> Result<Vec<f32>, VindexError> {
    backend.project(ProjectCall {
        weight,
        out_dim,
        in_dim,
        x,
    })
}
