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
pub mod continuation;
pub mod cpu;
pub mod decode;
pub mod device;
mod experts;
pub mod gated_delta;
pub mod kernels;
pub mod kv;
pub mod narrow;
pub mod observe;
pub mod operands;
pub mod prepared;
pub mod production;
pub mod quantise;
pub mod reference;
pub mod timing;
pub mod weights;

#[cfg(test)]
mod tests;

use larql_models::config::GateSource;

use super::{AttentionOp, ComponentOpPlan, LayerPlan};
use crate::error::VindexError;
use backend::{
    AttentionCall, AttentionStepCall, BiasCall, GateCall, NormCall, PlanBackend, ProjectCall,
    QkNormCall, SinkCall,
};
use kv::KvState;
use operands::OperandSource;
use prepared::{ExecutionSlice, PreparedAttention, PreparedLayer, PreparedOperands};
use rayon::prelude::*;
use reference::ReferenceBackend;
use weights::{load_weight, LoadedWeight};

/// Per-layer hidden-state taps, mirroring the production hook points.
#[derive(Debug)]
pub struct LayerTrace {
    /// Hidden state after the attention residual add, per position.
    pub post_attention: Vec<Vec<f32>>,
    /// The FFN's NORMED input (pre-FFN norm applied), per position —
    /// the vector the layer's gates multiply. This is the residual
    /// statistic V2's walk-FFN trace captures, and therefore the tap
    /// mutation capture must use: a gate built from anything else
    /// fires against a different vector than it was aimed at.
    pub ffn_input: Vec<Vec<f32>>,
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
    /// Plan indices of the layers that actually ran, in order.
    ///
    /// A reduced-depth run has to be able to prove it executed the
    /// prefix it asked for rather than silently falling back to the
    /// whole stack — and a full run has to be able to prove the reverse.
    /// `layers` alone cannot say that: a count is not an identity.
    pub executed_layers: Vec<usize>,
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
pub fn execute_text<'s>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
) -> Result<ExecutionTrace, VindexError> {
    execute_plan(plan, store.into(), tokens, &ReferenceBackend::new())
}

/// Execute a text-component plan over `tokens` on `backend`, tracing
/// every layer.
///
/// The backend is a parameter, not a branch: nothing below reads its
/// identity, and swapping it must not change which operations run.
pub fn execute_plan<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
) -> Result<ExecutionTrace, VindexError> {
    execute_slice(plan, store, tokens, backend, ExecutionSlice::Full)
}

/// [`execute_plan`] over a chosen [`ExecutionSlice`].
///
/// `execute_plan` is this with [`ExecutionSlice::Full`], so the two can
/// never disagree about what a whole model means.
pub fn execute_slice<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    slice: ExecutionSlice,
) -> Result<ExecutionTrace, VindexError> {
    let store = store.into();
    let mut embedded = Vec::new();
    let mut layers = Vec::with_capacity(plan.layers.len());
    let mut executed_layers = Vec::with_capacity(plan.layers.len());
    let ops = PreparedOperands::load(plan, store, backend, slice)?;
    let out = execute_prepared_streaming(plan, &ops, tokens, backend, None, &mut |event| {
        match event {
            PlaneEvent::Embedded(rows) => embedded = rows.to_vec(),
            PlaneEvent::Layer { index, trace } => {
                layers.push(trace);
                executed_layers.push(index);
            }
        }
        Ok(())
    })?;
    Ok(ExecutionTrace {
        embedded,
        layers,
        executed_layers,
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
pub fn execute_plan_streaming<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<FinalOutput, VindexError> {
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    execute_prepared_streaming(plan, &ops, tokens, backend, resume, sink)
}

/// [`execute_plan_streaming`] over operands the caller already
/// prepared. One-shot callers keep the source-taking form above, which
/// prepares and discards; a server prepares once and calls this.
pub fn execute_prepared_streaming<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<FinalOutput, VindexError> {
    // A one-shot forward owns whatever continuation state the plan needs.
    //
    // For a wholly-softmax stack that is nothing: `None` keeps the
    // existing behaviour exactly, including not materialising KV rows a
    // caller never asked for. A stack with a recurrence has no such
    // choice — its layers cannot run without durable buffers — so this
    // allocates them for the duration of the call. The provider starts at
    // position 0, which keeps every softmax layer on the batched
    // attention path it already took.
    if plan.layers.iter().all(|l| l.attention.softmax().is_some()) {
        return traverse(
            plan,
            ops,
            tokens,
            backend,
            resume,
            sink,
            None::<&mut dyn KvState>,
        );
    }
    let mut owned = kv::RowKvState::default();
    owned.prepare_continuation(
        &continuation::plan_continuation_geometry(plan).map_err(VindexError::Parse)?,
    )?;
    traverse(plan, ops, tokens, backend, resume, sink, Some(&mut owned))
}

/// Batch prefill (VI3-INF-3): the batch traversal over `tokens`,
/// populating the **caller's** continuation state — the same provider
/// a [`decode::DecodeSession`] then resumes via
/// [`with_kv_state`](decode::DecodeSession::with_kv_state). There is
/// no batch-state → decode-state translation and the executor never
/// manufactures a state implementation: continuation state belongs to
/// the caller for its entire lifetime, and execution modes merely
/// consume and update it.
///
/// The provider is `prepare`d with the plan's geometry, appended one
/// conditioned K/V row pair per layer per position (all positions for
/// layer 0, then layer 1 — the opposite interleaving to the decode
/// step's), and its logical position advanced past `tokens`. A
/// provider already holding state is *extended*: positions continue
/// from `kv.position()`, so a long prompt can prefill in chunks.
///
/// Returns the last position's final-normed hidden state and logits,
/// so generation can sample the first continuation token without an
/// extra step.
pub fn prefill_plan<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    kv: &mut dyn KvState,
) -> Result<FinalOutput, VindexError> {
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    prefill_prepared(plan, &ops, tokens, backend, kv)
}

/// [`prefill_plan`] over operands the caller already prepared.
///
/// This is the serve path's prefill: the point of preparing a model
/// once is that batch prefill and the decode session that follows it
/// read the *same* resident operands, instead of each materialising the
/// model for itself.
pub fn prefill_prepared<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    kv: &mut dyn KvState,
) -> Result<FinalOutput, VindexError> {
    // The FULL geometry, KV and recurrent alike. `plan_kv_geometry` is
    // the KV-only adapter and refuses a hybrid plan outright, which was
    // the right answer while nothing could execute one.
    kv.prepare_continuation(
        &continuation::plan_continuation_geometry(plan).map_err(VindexError::Parse)?,
    )?;
    let base = kv.position();
    let out = traverse(
        plan,
        ops,
        tokens,
        backend,
        None,
        &mut |_| Ok(()),
        Some(&mut *kv),
    )?;
    kv.set_position(base + tokens.len());
    Ok(out)
}

/// The one traversal in this module (see [`execute_plan_streaming`]'s
/// doc). `kv` switches the attention realisation: `None` runs the
/// backend's batched attention; `Some` runs the decode step's
/// per-position arithmetic into the provider — same numbers (the
/// decode-vs-batch parity gates are the guarantee), plus the rows.
/// `kv` and `resume` do not combine: a resumed run has already skipped
/// layers whose rows a provider would need.
fn traverse<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mut kv: Option<&mut K>,
) -> Result<FinalOutput, VindexError> {
    let embedding = plan.embedding.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "component `{}` has no embedding op — external hidden-state input is a later rung",
            plan.component
        ))
    })?;
    let hidden = embedding.table.shape[1];

    // **Refuse before any output.** A recurrence needs durable buffers,
    // and discovering at layer 63 that nobody can hold them would mean
    // every earlier layer had already been emitted — a caller left
    // holding 16 of 64 layers cannot tell that from a finished model.
    // QW-1 put this refusal up front for exactly that reason; the
    // question it asks has changed (from "can this run at all" to "can
    // this provider hold the state"), its position must not.
    //
    // Scoped to the layers this slice will actually execute. For `Full`
    // that is every layer and the check is unchanged; a reduced-depth
    // draft must not be refused — or charged state — for a recurrence in
    // a layer it never runs.
    let executed = ops.first_layer()..ops.first_layer() + ops.layers().len();
    for (offset, layer) in plan.layers.iter().enumerate() {
        if !executed.contains(&offset) {
            continue;
        }
        if layer.attention.softmax().is_some() {
            continue;
        }
        let Some(provider) = kv.as_mut() else {
            return Err(VindexError::Parse(format!(
                "layer {} carries `{}`, which keeps durable continuation state, and this \
                 traversal was given no provider to hold it",
                layer.layer,
                layer.attention.declared_name(),
            )));
        };
        provider.recurrent_state(offset).map_err(|e| {
            VindexError::Parse(format!(
                "layer {} carries `{}`: {e}",
                layer.layer,
                layer.attention.declared_name(),
            ))
        })?;
    }

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
            let table = ops.embed_table().ok_or_else(|| {
                VindexError::Parse(
                    "this prepared image carries no embedding table — a layer-range slice \
                     consumes hidden states, not token ids"
                        .to_string(),
                )
            })?;
            let mut h: Vec<Vec<f32>> = tokens
                .iter()
                .map(|&t| backend.embed(table, hidden, t, embedding.scale))
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

    let first = ops.first_layer();
    for (offset, prepared_layer) in ops.layers().iter().enumerate() {
        let index = first + offset;
        if index < start_layer {
            continue;
        }
        let capture = kv.as_mut().map(|state| (&mut **state, index));
        let trace = execute_layer(
            &plan.layers[index],
            prepared_layer,
            &mut h,
            hidden,
            backend,
            capture,
        )?;
        sink(PlaneEvent::Layer { index, trace })?;
    }

    let last = h.last().ok_or_else(|| {
        VindexError::Parse("cannot execute over an empty token sequence".to_string())
    })?;
    let final_hidden = match ops.final_norm() {
        Some(norm) => norm.apply(backend, last),
        None => last.clone(),
    };
    let logits = match ops.output() {
        Some((output, weight)) => {
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

/// Scale a sublayer's own output before its residual add, when the plan
/// declares a residual-scale operation (`LayerPlan::residual_scale`).
/// `None` leaves `delta` untouched — absence is not an identity multiply,
/// the same discipline every other optional op in the surface follows.
/// Backend-agnostic by design: every `PlanBackend` shares this one step
/// rather than each reimplementing the same multiply. `pub(super)`: the
/// stateful single-token driver in [`decode`] applies the same op at its
/// own residual-add sites, on the incremental decode path this batch path
/// does not cover.
pub(super) fn scale_residual_delta(scale: Option<f32>, delta: &mut [f32]) {
    if let Some(scale) = scale {
        for x in delta.iter_mut() {
            *x *= scale;
        }
    }
}

/// One decoder layer: norms and residuals exactly where the plan puts
/// them — placement is data, not code structure.
fn execute_layer<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
    layer: &LayerPlan,
    prepared: &PreparedLayer,
    h: &mut [Vec<f32>],
    hidden: usize,
    backend: &B,
    kv: Option<(&mut K, usize)>,
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
        .map(|row| prepared.pre_attention.apply(backend, row))
        .collect();
    // V3-SERVE-2: the attention realisation and the K/V behaviour are
    // separate decisions. Wanting a populated provider does not mean
    // wanting per-position arithmetic — the batched pass computes the
    // same conditioned rows and now returns them, so it can populate the
    // provider from the one traversal it already performs.
    //
    // The exception is not a preference but an expressibility limit: a
    // batched pass conditions position `p` as the `p`-th token of the
    // sequence it is given, so it cannot serve a prefill that resumes
    // part-way through one. Extending a populated provider therefore
    // still steps.
    // Dispatch on the OPERATOR the prepared layer holds. There is no
    // `softmax_or_refuse` here any more: a recurrence is something this
    // traversal runs, not something it declines. Both arms produce the
    // attention block's output for every position, and the residual and
    // FFN below are shared — the operator changes what attention IS, not
    // what a decoder layer does around it.
    let attn_out = match &prepared.attention {
        PreparedAttention::GatedDelta(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} runs a recurrence, which needs durable continuation state, \
                     and this traversal was given no provider to hold it",
                    layer.layer
                )));
            };
            // Resolved BEFORE any arithmetic. A provider that cannot hold
            // these buffers must not see a half-updated layer — the same
            // refuse-before-commit contract QW-1 established for the plan.
            let state = provider.recurrent_state(layer_index)?;
            gated_delta::layer_forward_with(
                &ops.op,
                &ops.weights()?,
                &inputs,
                state,
                gated_delta::Mutation::None,
                backend.dense_projector(),
            )
            .output
        }
        PreparedAttention::Softmax(ops) => {
            let attention_op = layer
                .attention
                .softmax()
                .expect("prepared softmax operands imply a softmax op");
            match kv {
                Some((kv, layer_index)) if kv.position() == 0 => {
                    let out = backend.attention(ops.call(
                        attention_op,
                        &inputs,
                        layer.pre_attention_norm.eps,
                        hidden,
                    ))?;
                    for (key, value) in out.keys.into_iter().zip(out.values) {
                        kv.append(layer_index, key, value);
                    }
                    out.outputs
                }
                Some((kv, layer_index)) => attention_into_kv(
                    attention_op,
                    ops,
                    &inputs,
                    layer.pre_attention_norm.eps,
                    hidden,
                    backend,
                    kv,
                    layer_index,
                )?,
                None => {
                    backend
                        .attention(ops.call(
                            attention_op,
                            &inputs,
                            layer.pre_attention_norm.eps,
                            hidden,
                        ))?
                        .outputs
                }
            }
        }
    };
    h.par_iter_mut()
        .zip(attn_out.par_iter())
        .try_for_each(|(row, out)| {
            let mut out = match &prepared.post_attention {
                Some(norm) => norm.apply(backend, out),
                None => out.clone(),
            };
            scale_residual_delta(layer.residual_scale, &mut out);
            backend.residual_add(row, &out);
            Ok::<(), VindexError>(())
        })?;
    let post_attention = h.to_vec();

    // FFN operands are resident on the prepared image — the bulk of a
    // decoder layer's weight, lowered once for the model rather than
    // once per layer per traversal.
    let ffn = &prepared.ffn;
    // The normed FFN inputs are computed once here (same values the
    // in-loop computation produced — one deterministic norm per row)
    // so the trace can carry the tap without a second norm pass.
    let ffn_inputs: Vec<Vec<f32>> = h
        .par_iter()
        .map(|row| prepared.pre_ffn.apply(backend, row))
        .collect();
    h.par_iter_mut()
        .zip(&ffn_inputs)
        .try_for_each(|(row, normed)| {
            let ffn_out = ffn.apply_from_residual(&layer.ffn, backend, row, normed, hidden)?;
            let mut ffn_out = match &prepared.post_ffn {
                Some(norm) => norm.apply(backend, &ffn_out),
                None => ffn_out,
            };
            scale_residual_delta(layer.residual_scale, &mut ffn_out);
            backend.residual_add(row, &ffn_out);
            if let Some(scale) = prepared.layer_scale {
                backend.scale_row(row, scale);
            }
            Ok::<(), VindexError>(())
        })?;
    Ok(LayerTrace {
        post_attention,
        ffn_input: ffn_inputs,
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
        store: OperandSource<'_>,
        format: prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        // A K≡V layer names the K operand as `v`: the value projection is
        // the raw K projection (before the key's norm and rotation), which
        // is exactly what projecting `w_v` = W_k yields — the backends take
        // V before conditioning Q/K, and apply the parameter-free V norm
        // to it when the op carries one.
        Ok(Self {
            w_q: load_weight(store, &op.q, format(&op.q))?,
            w_k: load_weight(store, &op.k, format(&op.k))?,
            w_v: load_weight(store, &op.v, format(&op.v))?,
            w_o: load_weight(store, &op.o, format(&op.o))?,
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
                    load_weight(store, &gate.projection, format(&gate.projection))?,
                ),
                _ => None,
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
    /// Every matrix operand, for residency accounting.
    pub(super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut all = vec![&self.w_q, &self.w_k, &self.w_v, &self.w_o];
        if let Some(gate) = &self.gate {
            all.push(gate);
        }
        all
    }

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
fn attention_into_kv<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
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
        let out = backend.attention_step(AttentionStepCall {
            op: call,
            position: base + offset,
            keys: kv.keys(layer_index),
            values: kv.values(layer_index),
        })?;
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
