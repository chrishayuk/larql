//! The plan interpreter (V3-G5b-2 Stage A, V3-G5b-3b seam).
//!
//! Executes a [`ComponentOpPlan`] — and **nothing else**. Every argument
//! comes from the plan (which came from the container); every operand
//! loads through the closure-verified `object → representation → segment`
//! path; every judged enum is matched exhaustively so an unjudged variant
//! is a compile error, not a guess. There is no family name, no layer
//! arithmetic, no HF tensor name, and no default anywhere in this module.
//!
//! This file owns *meaning*: operation ordering, residual ordering, layer
//! traversal, whether an optional operation exists, and how position and
//! span policy dispatch. A [`PlanBackend`] owns only arithmetic. One
//! interpreter drives every backend, so a second implementation cannot
//! quietly become a second reading of the model — see [`backend`].
//!
//! The trace mirrors the production forward's hook points
//! (`post_attention` = after the attention residual add, `post_layer` =
//! after the FFN residual add) so parity can compare layer by layer
//! against a checkpoint-driven oracle.

pub mod backend;
pub mod decode;
pub mod device;
mod experts;
pub mod kernels;
pub mod operands;
pub mod production;
pub mod reference;
pub mod weights;

#[cfg(test)]
mod tests;

use super::{AttentionOp, ComponentOpPlan, LayerPlan, NormOp};
use crate::error::VindexError;
use backend::{
    AttentionCall, BiasCall, GateCall, MatrixClass, NormCall, PlanBackend, ProjectCall, QkNormCall,
    SinkCall, WeightFormat,
};
use operands::OperandStore;
use rayon::prelude::*;
use reference::ReferenceBackend;
use weights::{load_weight, LoadedWeight};

/// Per-layer hidden-state taps, mirroring the production hook points.
#[derive(Debug)]
pub struct LayerTrace {
    /// Hidden state after the attention residual add, per position.
    pub post_attention: Vec<Vec<f32>>,
    /// Hidden state after the FFN residual add, per position.
    pub post_layer: Vec<Vec<f32>>,
}

/// The full execution record of one component over one token sequence.
#[derive(Debug)]
pub struct ExecutionTrace {
    /// The residual *entering* layer 0, per position — everything the
    /// embedding op produced and nothing else.
    ///
    /// Captured because a layer-by-layer comparison needs somewhere to
    /// stand before layer 0: if the two sides already disagree here, no
    /// per-layer margin below means what it appears to mean. It is the
    /// same tap `scripts/dump_layers_hf.py` takes with a pre-hook on
    /// layer 0.
    pub embedded: Vec<Vec<f32>>,
    pub layers: Vec<LayerTrace>,
    /// Final-normed hidden state of the last position.
    pub final_hidden: Vec<f32>,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

/// A plane handed to the caller the moment it exists, so a long run can
/// persist progress incrementally instead of holding 52 layers of hidden
/// state until the end.
#[derive(Debug)]
pub enum PlaneEvent<'a> {
    /// The residual entering layer 0 — plane 000. Not emitted when a
    /// [`ResumePoint`] skips the embedding.
    Embedded(&'a [Vec<f32>]),
    /// One completed layer's taps, in layer order.
    Layer { index: usize, trace: LayerTrace },
}

/// Where an interrupted execution restarts.
///
/// The residual leaving layer `next_layer - 1` (plane `next_layer`) is
/// exactly the state entering `next_layer`, so a persisted plane resumes
/// the run bit-identically — no separate checkpoint format exists, and
/// none should: two formats could disagree.
#[derive(Debug)]
pub struct ResumePoint {
    /// Index of the first layer still to execute.
    pub next_layer: usize,
    /// The residual entering that layer, one row per position.
    pub hidden: Vec<Vec<f32>>,
}

/// What execution produces beyond the streamed planes.
#[derive(Debug)]
pub struct FinalOutput {
    /// Final-normed hidden state of the last position.
    pub final_hidden: Vec<f32>,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

/// Execute a text-component plan on the reference backend.
///
/// The semantic anchor: naive f32, sharing no arithmetic with
/// `larql-compute`.
pub fn execute_text(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    tokens: &[u32],
) -> Result<ExecutionTrace, VindexError> {
    execute_plan(plan, store, tokens, &ReferenceBackend::new())
}

/// Execute a text-component plan over `tokens` on `backend`, tracing
/// every layer.
///
/// The backend is a parameter, not a branch: nothing below reads its
/// identity, and swapping it must not change which operations run.
pub fn execute_plan<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    tokens: &[u32],
    backend: &B,
) -> Result<ExecutionTrace, VindexError> {
    let mut embedded = Vec::new();
    let mut layers = Vec::with_capacity(plan.layers.len());
    let out = execute_plan_streaming(plan, store, tokens, backend, None, &mut |event| {
        match event {
            PlaneEvent::Embedded(rows) => embedded = rows.to_vec(),
            PlaneEvent::Layer { trace, .. } => layers.push(trace),
        }
        Ok(())
    })?;
    Ok(ExecutionTrace {
        embedded,
        layers,
        final_hidden: out.final_hidden,
        logits: out.logits,
    })
}

/// Streaming form of [`execute_plan`]: one traversal, planes delivered
/// through `sink` as they complete, with an optional [`ResumePoint`] to
/// restart an interrupted run.
///
/// [`execute_plan`] is a wrapper over this function, so the two can
/// never disagree about what the program means — there is exactly one
/// traversal in this module.
pub fn execute_plan_streaming<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<FinalOutput, VindexError> {
    let embedding = plan.embedding.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "component `{}` has no embedding op — external hidden-state input is a later rung",
            plan.component
        ))
    })?;
    let hidden = embedding.table.shape[1];

    let (start_layer, mut h) = match resume {
        Some(point) => {
            if point.next_layer > plan.layers.len() {
                return Err(VindexError::Parse(format!(
                    "resume point at layer {} is past the plan's {} layers",
                    point.next_layer,
                    plan.layers.len()
                )));
            }
            if point.hidden.len() != tokens.len() {
                return Err(VindexError::Parse(format!(
                    "resume state carries {} positions but the fixture has {} tokens",
                    point.hidden.len(),
                    tokens.len()
                )));
            }
            if point.hidden.iter().any(|row| row.len() != hidden) {
                return Err(VindexError::Parse(format!(
                    "resume state rows do not match the plan's hidden size {hidden}"
                )));
            }
            (point.next_layer, point.hidden)
        }
        None => {
            let table = store.load(&embedding.table)?;
            let mut h: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| backend.embed(&table, hidden, t, embedding.scale))
                .collect();
            // The judged embedding normalisation, when the plan carries
            // one. It is weightless — no operand, hence the empty weight
            // slice — and it runs *after* any embedding scale, matching
            // the upstream order in which the scale belongs to the table
            // and the norm to the lookup.
            if let Some(norm) = embedding.norm {
                for row in h.iter_mut() {
                    *row = backend.norm(NormCall {
                        kind: norm.kind,
                        x: row,
                        weight: &[],
                        weight_offset: 0.0,
                        eps: norm.eps,
                    });
                }
            }
            sink(PlaneEvent::Embedded(&h))?;
            (0, h)
        }
    };

    for (index, layer) in plan.layers.iter().enumerate().skip(start_layer) {
        let trace = execute_layer(layer, store, &mut h, hidden, backend)?;
        sink(PlaneEvent::Layer { index, trace })?;
    }

    let last = h.last().ok_or_else(|| {
        VindexError::Parse("cannot execute over an empty token sequence".to_string())
    })?;
    let final_hidden = match &plan.final_norm {
        Some(op) => apply_norm_op(op, store, last, backend)?,
        None => last.clone(),
    };
    let logits = match &plan.output {
        Some(output) => {
            let weight = load_weight(
                store,
                &output.projection,
                backend.weight_format(MatrixClass::OutputHead),
            )?;
            let vocab = output.projection.shape[0];
            Some(backend.output_head(
                weight.slice(),
                vocab,
                hidden,
                &final_hidden,
                output.multiplier,
                output.softcapping,
            )?)
        }
        None => None,
    };
    Ok(FinalOutput {
        final_hidden,
        logits,
    })
}

/// One decoder layer: norms and residuals exactly where the plan puts
/// them — placement is data, not code structure.
fn execute_layer<B: PlanBackend + ?Sized>(
    layer: &LayerPlan,
    store: &OperandStore,
    h: &mut [Vec<f32>],
    hidden: usize,
    backend: &B,
) -> Result<LayerTrace, VindexError> {
    // The attention input is normalised here, once, and handed to the
    // backend — the judged gate reads the same vector, so producing it
    // in one place is what keeps the two from drifting apart.
    //
    // Position loops below run in parallel. Each position's arithmetic
    // is untouched and rows are disjoint, so the result is bit-identical
    // to the serial order — parallelism here is an execution strategy,
    // never a reassociation.
    let inputs: Vec<Vec<f32>> = h
        .par_iter()
        .map(|row| apply_norm_op(&layer.pre_attention_norm, store, row, backend))
        .collect::<Result<_, _>>()?;
    let attn_out = attention(
        &layer.attention,
        &inputs,
        layer.pre_attention_norm.eps,
        store,
        hidden,
        backend,
    )?;
    h.par_iter_mut()
        .zip(attn_out.par_iter())
        .try_for_each(|(row, out)| {
            let out = match &layer.post_attention_norm {
                Some(op) => apply_norm_op(op, store, out, backend)?,
                None => out.clone(),
            };
            backend.residual_add(row, &out);
            Ok::<(), VindexError>(())
        })?;
    let post_attention = h.to_vec();

    // FFN operands load once per layer, not once per position. They are
    // the bulk of a decoder layer's weight, and `OperandStore::load`
    // allocates a fresh copy per call — re-reading them for every
    // token would dominate the run on a real model without changing a
    // single number.
    let format = backend.weight_format(MatrixClass::FfnProjection);
    let ffn = experts::FfnOperands::load(&layer.ffn, store, format)?;
    h.par_iter_mut().try_for_each(|row| {
        let normed = apply_norm_op(&layer.pre_ffn_norm, store, row, backend)?;
        let ffn_out = ffn.apply(&layer.ffn, backend, &normed, hidden)?;
        let ffn_out = match &layer.post_ffn_norm {
            Some(op) => apply_norm_op(op, store, &ffn_out, backend)?,
            None => ffn_out,
        };
        backend.residual_add(row, &ffn_out);
        Ok::<(), VindexError>(())
    })?;
    Ok(LayerTrace {
        post_attention,
        post_layer: h.to_vec(),
    })
}

/// One layer's attention operands, loaded once in the backend's
/// declared format. Owned by whichever traversal is running — the batch
/// path loads them per forward, the decode session keeps them for its
/// lifetime — so both paths resolve operands through exactly one place.
pub(super) struct AttentionOperands {
    w_q: LoadedWeight,
    w_k: LoadedWeight,
    w_v: LoadedWeight,
    w_o: LoadedWeight,
    qk_weights: Option<(Vec<f32>, Vec<f32>)>,
    gate: Option<LoadedWeight>,
    /// Q/K/V/O biases, f32 (elementwise glue, not matrix traffic).
    biases: Option<[Vec<f32>; 4]>,
    /// Sink logits, f32.
    sinks: Option<Vec<f32>>,
}

impl AttentionOperands {
    /// Load through the closure-verified path. QK-norm weights stay f32
    /// (elementwise glue, not matrix traffic).
    pub(super) fn load(
        op: &AttentionOp,
        store: &OperandStore,
        format: WeightFormat,
    ) -> Result<Self, VindexError> {
        Ok(Self {
            w_q: load_weight(store, &op.q, format)?,
            w_k: load_weight(store, &op.k, format)?,
            w_v: load_weight(store, &op.v, format)?,
            w_o: load_weight(store, &op.o, format)?,
            qk_weights: match &op.qk_norm {
                Some(qk) => Some((store.load(&qk.q)?, store.load(&qk.k)?)),
                None => None,
            },
            gate: match &op.output_gate {
                Some(gate) => Some(load_weight(store, &gate.projection, format)?),
                None => None,
            },
            biases: match (&op.q_bias, &op.k_bias, &op.v_bias, &op.o_bias) {
                (Some(q), Some(k), Some(v), Some(o)) => Some([
                    store.load(q)?,
                    store.load(k)?,
                    store.load(v)?,
                    store.load(o)?,
                ]),
                (None, None, None, None) => None,
                // Closure emits all four or none; a partial set is a
                // plan the closure never produced.
                _ => {
                    return Err(VindexError::Parse(
                        "attention op carries a partial Q/K/V/O bias set; operand closure \
                         emits all four or none"
                            .to_string(),
                    ))
                }
            },
            sinks: match &op.sinks {
                Some(sinks) => Some(store.load(&sinks.logits)?),
                None => None,
            },
        })
    }

    /// Every matrix operand this attention holds, for residency
    /// preparation.
    pub(super) fn weight_slices(&self) -> Vec<backend::WeightSlice<'_>> {
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
    pub(super) fn call<'a>(
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
            (Some(gate), Some(weight)) => Some(GateCall {
                spec: gate.spec,
                weight: weight.slice(),
            }),
            _ => None,
        };
        let bias = self.biases.as_ref().map(|[q, k, v, o]| BiasCall {
            q: q.as_slice(),
            k: k.as_slice(),
            v: v.as_slice(),
            o: o.as_slice(),
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
        }
    }
}

/// Load the attention operands and hand the backend a fully resolved
/// call.
fn attention<B: PlanBackend + ?Sized>(
    op: &AttentionOp,
    inputs: &[Vec<f32>],
    qk_norm_eps: f64,
    store: &OperandStore,
    hidden: usize,
    backend: &B,
) -> Result<Vec<Vec<f32>>, VindexError> {
    let operands = AttentionOperands::load(
        op,
        store,
        backend.weight_format(MatrixClass::AttentionProjection),
    )?;
    backend.attention(operands.call(op, inputs, qk_norm_eps, hidden))
}

/// Apply one norm op to one vector.
fn apply_norm_op<B: PlanBackend + ?Sized>(
    op: &NormOp,
    store: &OperandStore,
    x: &[f32],
    backend: &B,
) -> Result<Vec<f32>, VindexError> {
    let weight = store.load(&op.weight)?;
    Ok(backend.norm(NormCall {
        kind: op.kind,
        x,
        weight: &weight,
        weight_offset: op.weight_offset,
        eps: op.eps,
    }))
}

/// Project one vector through an `[out, in]` weight.
///
/// Kept as a named helper so the interpreter never open-codes a matvec:
/// every projection in a plan goes through the backend.
#[allow(dead_code)]
fn project<B: PlanBackend + ?Sized>(
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
