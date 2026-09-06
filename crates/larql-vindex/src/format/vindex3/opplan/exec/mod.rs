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

pub mod accounting;
pub mod attention_residual;
pub mod backend;
pub mod continuation;
pub mod controls;
pub mod conv_qkv;
pub mod cpu;
pub mod decode;
pub mod device;
mod experts;
pub mod gated_delta;
pub mod hyper_connection;
pub mod kda;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod kda_metal;
pub mod kernels;
pub mod kimi_kda_layer;
pub mod kimi_mla_layer;
pub mod kimi_moe_block;
pub mod kimi_router;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod kimi_source;
pub mod kv;
pub mod mamba2;
pub mod mla;
pub mod narrow;
pub mod observe;
pub mod operands;
pub mod prepared;
pub mod production;
pub mod quantise;
pub mod realization;
pub mod reference;
pub mod requirements;
pub mod stack;
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub mod stack_metal;
pub mod timing;
pub mod token;
pub mod weights;

#[cfg(test)]
mod tests;

use std::borrow::Cow;

use larql_models::config::{GateSource, HyperConnection};

use super::{AttentionOp, ComponentOpPlan, LayerPlan};
use crate::error::VindexError;

use backend::{
    AttentionCall, AttentionStepCall, BiasCall, GateCall, NormCall, PlanBackend, ProjectCall,
    QkNormCall, SinkCall,
};
use hyper_connection::{Bundle, Mutation, SinkhornSplit, SiteReduction};
use kv::KvState;
use observe::HcSite;
use operands::OperandSource;
use prepared::{
    ExecutionSlice, PreparedAttention, PreparedHcSite, PreparedLayer, PreparedOperands,
};
use rayon::prelude::*;
use reference::ReferenceBackend;
use weights::{load_weight, LoadedWeight};

/// One plane of the traversal — the residual at a layer boundary, one
/// entry per position — typed by the component's residual topology
/// (wave 19b).
///
/// The batch carrier is `[positions, streams, hidden]` on a
/// hyper-connected component and `[positions, hidden]` everywhere else,
/// and the two are different types rather than one wider row: the
/// stream count and the hidden width are different semantic dimensions,
/// and a consumer that reads a plane must say which it expects. A
/// `[hidden]` row never comes out of a bundle here; it comes out of a
/// site's reduction or the head's, inside the traversal.
#[derive(Debug, Clone, PartialEq)]
pub enum Plane {
    /// One `[hidden]` row per position — every component before
    /// hyper-connections. Serialises and persists exactly as it always
    /// has.
    Rows(Vec<Vec<f32>>),
    /// One bundle per position — a hyper-connected component's residual.
    Bundles(Vec<Bundle>),
}

impl Plane {
    /// How many positions the plane holds.
    pub fn positions(&self) -> usize {
        match self {
            Self::Rows(rows) => rows.len(),
            Self::Bundles(bundles) => bundles.len(),
        }
    }

    /// The `[hidden]` rows, refused on a bundle plane. For consumers that
    /// can only mean one thing by a plane — persistence, comparison
    /// against a single-stream reference.
    pub fn try_rows(&self) -> Result<&[Vec<f32>], VindexError> {
        match self {
            Self::Rows(rows) => Ok(rows),
            Self::Bundles(bundles) => Err(VindexError::Parse(format!(
                "this plane holds {} bundles of {} streams, not [hidden] rows; the component is \
                 hyper-connected and the reader must say what a bundle means to it",
                bundles.len(),
                bundles.first().map_or(0, Bundle::streams)
            ))),
        }
    }

    /// [`Self::try_rows`] for callers on a single-stream component, where
    /// a bundle is a programming error rather than a model.
    pub fn rows(&self) -> &[Vec<f32>] {
        self.try_rows().unwrap_or_else(|e| panic!("{e}"))
    }

    /// The bundles, `None` on a row plane.
    pub fn bundles(&self) -> Option<&[Bundle]> {
        match self {
            Self::Rows(_) => None,
            Self::Bundles(bundles) => Some(bundles),
        }
    }
}

/// Per-layer hidden-state taps, mirroring the production hook points.
#[derive(Debug)]
pub struct LayerTrace {
    /// The residual after the attention sublayer's update, per position.
    pub post_attention: Plane,
    /// The FFN's NORMED input (pre-FFN norm applied), per position —
    /// the vector the layer's gates multiply. This is the residual
    /// statistic V2's walk-FFN trace captures, and therefore the tap
    /// mutation capture must use: a gate built from anything else
    /// fires against a different vector than it was aimed at. A
    /// `[hidden]` row on every topology: it is the ordinary operator's
    /// input, which a site has already reduced.
    pub ffn_input: Vec<Vec<f32>>,
    /// The residual after the FFN sublayer's update, per position.
    pub post_layer: Plane,
}

/// What the stack's exit produced: the `[hidden]` vector the final norm
/// and head read, or — on a layer-range image of a hyper-connected
/// component, which has no exit — the bundle after the last layer.
#[derive(Debug, Clone, PartialEq)]
pub enum FinalState {
    Hidden(Vec<f32>),
    Bundle(Bundle),
}

impl FinalState {
    /// The final `[hidden]` vector, refused when the run ended on a
    /// bundle.
    pub fn try_hidden(&self) -> Result<&[f32], VindexError> {
        match self {
            Self::Hidden(h) => Ok(h),
            Self::Bundle(b) => Err(VindexError::Parse(format!(
                "the run ended on a bundle of {} streams — a layer-range image of a \
                 hyper-connected component has no [hidden] exit",
                b.streams()
            ))),
        }
    }

    /// [`Self::try_hidden`] for callers on a single-stream component.
    pub fn hidden(&self) -> &[f32] {
        self.try_hidden().unwrap_or_else(|e| panic!("{e}"))
    }
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
    pub embedded: Plane,
    pub layers: Vec<LayerTrace>,
    /// Plan indices of the layers that actually ran, in order.
    ///
    /// A reduced-depth run has to be able to prove it executed the
    /// prefix it asked for rather than silently falling back to the
    /// whole stack — and a full run has to be able to prove the reverse.
    /// `layers` alone cannot say that: a count is not an identity.
    pub executed_layers: Vec<usize>,
    /// What the last position left the stack as.
    pub exit: FinalState,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

impl ExecutionTrace {
    /// Final-normed hidden state of the last position (single-stream
    /// callers; see [`FinalState::hidden`]).
    pub fn final_hidden(&self) -> &[f32] {
        self.exit.hidden()
    }
}

/// One hyper-connection site's intermediate state for every position of
/// the batch (wave 19b) — the batch form of
/// [`observe::HcSiteRecord`], indexed by position.
#[derive(Debug, Clone, Copy)]
pub struct HcSitePlane<'a> {
    pub layer: usize,
    pub site: HcSite,
    pub splits: &'a [SinkhornSplit],
    pub reduced: &'a [Vec<f32>],
    pub branch_outputs: &'a [Vec<f32>],
    pub bundles_out: &'a [Bundle],
}

/// A plane handed to the caller the moment it exists, so a long run can
/// persist progress incrementally instead of holding 52 layers of hidden
/// state until the end.
#[derive(Debug)]
pub enum PlaneEvent<'a> {
    /// The residual entering layer 0 — plane 000. Not emitted when a
    /// [`ResumePoint`] skips the embedding.
    Embedded(&'a Plane),
    /// One completed layer's taps, in layer order.
    Layer { index: usize, trace: LayerTrace },
    /// One hyper-connection site's state, every position, emitted after
    /// the site's update and before the layer's own plane. Only a
    /// hyper-connected component emits it.
    HyperConnectionSite(HcSitePlane<'a>),
}

/// Where an interrupted execution restarts.
///
/// The residual leaving layer `next_layer - 1` (plane `next_layer`) is
/// exactly the state entering `next_layer`, so a persisted plane resumes
/// the run bit-identically — no separate checkpoint format exists, and
/// none should: two formats could disagree. On a hyper-connected
/// component the plane is bundles, and a row plane is refused.
#[derive(Debug)]
pub struct ResumePoint {
    /// Index of the first layer still to execute.
    pub next_layer: usize,
    /// The residual entering that layer, one entry per position.
    pub hidden: Plane,
}

/// What execution produces beyond the streamed planes.
#[derive(Debug)]
pub struct FinalOutput {
    /// What the last position left the stack as.
    pub exit: FinalState,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

impl FinalOutput {
    /// Final-normed hidden state of the last position (single-stream
    /// callers; see [`FinalState::hidden`]).
    pub fn final_hidden(&self) -> &[f32] {
        self.exit.hidden()
    }
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
    let mut embedded = Plane::Rows(Vec::new());
    let mut layers = Vec::with_capacity(plan.layers.len());
    let mut executed_layers = Vec::with_capacity(plan.layers.len());
    let ops = PreparedOperands::load(plan, store, backend, slice)?;
    let out = execute_prepared_streaming(plan, &ops, tokens, backend, None, &mut |event| {
        match event {
            PlaneEvent::Embedded(plane) => embedded = plane.clone(),
            PlaneEvent::Layer { index, trace } => {
                layers.push(trace);
                executed_layers.push(index);
            }
            // The trace is the layer boundaries; a site's state is for
            // the streaming sink, which is where the witness reads it.
            PlaneEvent::HyperConnectionSite(_) => {}
        }
        Ok(())
    })?;
    Ok(ExecutionTrace {
        embedded,
        layers,
        executed_layers,
        exit: out.exit,
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
    execute_prepared_streaming_with(plan, ops, tokens, backend, resume, sink, Mutation::None)
}

/// [`execute_prepared_streaming`] under a deliberate defect — the
/// wave-19b negative controls on the batch path. Test-only: production
/// has exactly one way in, and it passes [`Mutation::None`].
#[cfg(test)]
pub(super) fn execute_prepared_streaming_mutated<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mutation: Mutation,
) -> Result<FinalOutput, VindexError> {
    execute_prepared_streaming_with(plan, ops, tokens, backend, resume, sink, mutation)
}

fn execute_prepared_streaming_with<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mutation: Mutation,
) -> Result<FinalOutput, VindexError> {
    // A pin whose provider has gone or changed invalidates the image;
    // nothing here falls back to another realization. The registry is
    // the image's own — the store's — never a built-in default.
    ops.ensure_providers_in(ops.registry())?;
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
            mutation,
        );
    }
    let mut owned = kv::RowKvState::default();
    owned.prepare_continuation(
        &continuation::plan_continuation_geometry(plan).map_err(VindexError::Parse)?,
    )?;
    traverse(
        plan,
        ops,
        tokens,
        backend,
        resume,
        sink,
        Some(&mut owned),
        mutation,
    )
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
        Mutation::None,
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
#[allow(clippy::too_many_arguments)]
fn traverse<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    tokens: &[u32],
    backend: &B,
    resume: Option<ResumePoint>,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mut kv: Option<&mut K>,
    mutation: Mutation,
) -> Result<FinalOutput, VindexError> {
    let embedding = plan.embedding.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "component `{}` has no embedding op — external hidden-state input is a later rung",
            plan.component
        ))
    })?;
    let hidden = embedding.table.shape[1];
    let topology = ops.hyper_connection();

    // **The batch path does not carry a residual history yet (2a).**
    // Refused by name, up front, so the decode seam that proves the
    // traversal cannot make this path look supported: the plane here is
    // one `[hidden]` row per position, and running an attention-residual
    // component over it would drop every snapshot and every boundary
    // event silently. 2b builds the per-position history and this
    // refusal goes with it.
    if ops.carries_attention_residual() {
        return Err(VindexError::Parse(format!(
            "component `{}` declares the attention-residual residual topology; the batch \
             traversal carries one [hidden] row per position and no snapshot history, so it \
             would run the component as an ordinary residual. The decode traversal carries \
             the history (K3-ATTNRES-1 2a); the batch one follows in 2b",
            plan.component
        )));
    }

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
    //
    // Driven by the CONTINUATION GEOMETRY, not by "is this layer
    // softmax": the question is which region a layer needs, and there
    // are now three answers. An earlier form asked every non-softmax
    // layer for recurrent buffers, which was right while a recurrence
    // was the only alternative to rows and became wrong the moment MLA
    // executed — it keeps a per-position latent cache and no recurrence
    // at all, so the pre-flight would have refused a state the provider
    // was holding perfectly well.
    let executed = ops.first_layer()..ops.first_layer() + ops.layers().len();
    let regions = continuation::plan_continuation_geometry(plan).map_err(|e| {
        VindexError::Parse(format!(
            "component `{}` declares continuation state this build cannot size: {e}",
            plan.component
        ))
    })?;
    for (offset, layer) in plan.layers.iter().enumerate() {
        if !executed.contains(&offset) {
            continue;
        }
        let region = &regions[offset];
        if matches!(
            region,
            continuation::LayerContinuationGeometry::Kv(_)
                | continuation::LayerContinuationGeometry::Stateless
        ) {
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
        let named = |e: kv::ContinuationError| {
            VindexError::Parse(format!(
                "layer {} carries `{}`: {e}",
                layer.layer,
                layer.attention.declared_name(),
            ))
        };
        match region {
            continuation::LayerContinuationGeometry::Recurrent(_)
            | continuation::LayerContinuationGeometry::KvAndRecurrent { .. } => {
                provider.recurrent_state(offset).map_err(named)?;
            }
            continuation::LayerContinuationGeometry::LatentKv(_) => {
                provider.latent_state(offset).map_err(named)?;
            }
            continuation::LayerContinuationGeometry::Kv(_)
            | continuation::LayerContinuationGeometry::Stateless => {}
        }
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
            if point.hidden.positions() != tokens.len() {
                return Err(VindexError::Parse(format!(
                    "resume state carries {} positions but the fixture has {} tokens",
                    point.hidden.positions(),
                    tokens.len()
                )));
            }
            // The plane's kind must be the component's residual: a row
            // plane on a hyper-connected component would run the bundle
            // as one stream, and a bundle plane on a single stream has
            // no reading at all.
            match (&point.hidden, topology) {
                (Plane::Rows(rows), None) => {
                    if rows.iter().any(|row| row.len() != hidden) {
                        return Err(VindexError::Parse(format!(
                            "resume state rows do not match the plan's hidden size {hidden}"
                        )));
                    }
                }
                (Plane::Bundles(bundles), Some(hc)) => {
                    if bundles
                        .iter()
                        .any(|b| b.streams() != hc.streams || b.hidden() != hidden)
                    {
                        return Err(VindexError::Parse(format!(
                            "resume bundles do not match the component's {} streams x {hidden}",
                            hc.streams
                        )));
                    }
                }
                (Plane::Rows(_), Some(hc)) => {
                    return Err(VindexError::Parse(format!(
                        "component `{}` declares {} residual streams; a resume point for it \
                         carries bundles, not rows",
                        plan.component, hc.streams
                    )));
                }
                (Plane::Bundles(_), None) => {
                    return Err(VindexError::Parse(format!(
                        "component `{}` declares a single residual stream; a resume point for \
                         it carries rows, not bundles",
                        plan.component
                    )));
                }
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
            // The embedding enters a hyper-connected stack replicated
            // into every stream, per position — after its scale and
            // norm, which belong to the lookup.
            let h = match topology {
                Some(hc) => Plane::Bundles(
                    h.iter()
                        .map(|row| Bundle::replicate(row, hc.streams))
                        .collect(),
                ),
                None => Plane::Rows(h),
            };
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
            topology,
            index,
            sink,
            mutation,
        )?;
        sink(PlaneEvent::Layer { index, trace })?;
    }

    // ── The exit ──
    //
    // A bundle leaves the stack through the head's OWN reduction (a
    // different operation from a site's — no Sinkhorn) when the image
    // carries one, and a whole-stack image of a hyper-connected
    // component always does (preparation refuses otherwise). A
    // layer-range image has no exit: the bundle after its last layer IS
    // its output, and it produces no logits.
    let empty = || VindexError::Parse("cannot execute over an empty token sequence".to_string());
    let (exit, logits) = match h {
        Plane::Rows(rows) => {
            let last = rows.last().ok_or_else(empty)?;
            let final_hidden = match ops.final_norm() {
                Some(norm) => norm.apply(backend, last),
                None => last.clone(),
            };
            let logits = output_logits(ops, backend, hidden, &final_hidden)?;
            (FinalState::Hidden(final_hidden), logits)
        }
        Plane::Bundles(mut bundles) => {
            let last = bundles.pop().ok_or_else(empty)?;
            match (ops.hyper_connection_head(), topology) {
                (Some(head), Some(hc)) => {
                    let reduced = hyper_connection::head_reduce(
                        last.as_flat(),
                        last.streams(),
                        hidden,
                        &head.weights(),
                        head.norm_eps(),
                        hc.sinkhorn_eps,
                    );
                    let final_hidden = match ops.final_norm() {
                        Some(norm) => norm.apply(backend, &reduced),
                        None => reduced,
                    };
                    let logits = output_logits(ops, backend, hidden, &final_hidden)?;
                    (FinalState::Hidden(final_hidden), logits)
                }
                _ => {
                    if ops.final_norm().is_some() || ops.output().is_some() {
                        return Err(VindexError::Parse(
                            "a hyper-connected bundle reached a whole-stack exit with no head \
                             reduction; preparation should have refused the image"
                                .to_string(),
                        ));
                    }
                    (FinalState::Bundle(last), None)
                }
            }
        }
    };
    Ok(FinalOutput { exit, logits })
}

/// The output head over the final-normed vector, when the image carries
/// one.
fn output_logits<B: PlanBackend + ?Sized>(
    ops: &PreparedOperands,
    backend: &B,
    hidden: usize,
    final_hidden: &[f32],
) -> Result<Option<Vec<f32>>, VindexError> {
    match ops.output() {
        Some((output, weight)) => {
            let vocab = output.projection.shape[0];
            Ok(Some(backend.output_head(
                weight.slice(),
                vocab,
                hidden,
                final_hidden,
                output.multiplier,
                output.softcapping,
            )?))
        }
        None => Ok(None),
    }
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
///
/// **Wave 19b.** The residual is a [`Plane`]. Each sublayer is a SITE:
/// on a bundle plane it enters through `hyper_connection::reduce`, per
/// position, and leaves through `hyper_connection::update`, per
/// position — while the ordinary operator in between runs exactly once
/// over `[positions, hidden]`, as it always has. On a row plane the site
/// is the identity in and the residual add out. `hidden` means the same
/// thing on both; the stream count is a different dimension and never
/// reaches the operator.
#[allow(clippy::too_many_arguments)]
fn execute_layer<B: PlanBackend + ?Sized, K: KvState + ?Sized>(
    layer: &LayerPlan,
    prepared: &PreparedLayer,
    h: &mut Plane,
    hidden: usize,
    backend: &B,
    kv: Option<(&mut K, usize)>,
    topology: Option<HyperConnection>,
    index: usize,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
    mutation: Mutation,
) -> Result<LayerTrace, VindexError> {
    // ── Attention site: enter ──
    //
    // The attention input is normalised here, once, and handed to the
    // backend — the judged gate reads the same vector, so producing it
    // in one place is what keeps the two from drifting apart. On a
    // bundle the norm reads the REDUCED vector, never the bundle.
    //
    // Position loops below run in parallel. Each position's arithmetic
    // is untouched and rows are disjoint, so the result is bit-identical
    // to the serial order — parallelism here is an execution strategy,
    // never a reassociation.
    // Under post-norm placement the sublayer reads the RAW residual; the
    // wrap norm applies to its output before the update. Same program as
    // the decode path, which must not be able to disagree with this one.
    let BatchSiteEntry {
        branch_inputs,
        reductions,
    } = enter_batch_site(
        h,
        prepared.hyper_connection.as_ref().map(|hc| &hc.attention),
        topology,
        layer.declared_norm_eps,
        mutation,
    )?;
    // The (c) controls feed the pre-attention norm a stream instead.
    let norm_source: Cow<'_, [Vec<f32>]> = match (mutation, &*h) {
        (Mutation::PreNormOnStreamZero, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(|x| x.stream(0).to_vec()).collect())
        }
        (Mutation::PreNormOnStreamMean, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(Bundle::stream_mean).collect())
        }
        _ => Cow::Borrowed(&branch_inputs),
    };
    let inputs: Vec<Vec<f32>> = match &prepared.pre_attention {
        Some(norm) => norm_source
            .par_iter()
            .map(|row| norm.apply(backend, row))
            .collect(),
        None => norm_source.into_owned(),
    };
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
        PreparedAttention::Mamba2(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} runs a recurrence, which needs durable continuation state, \
                     and this traversal was given no provider to hold it",
                    layer.layer
                )));
            };
            let state = provider.recurrent_state(layer_index)?;
            mamba2::layer_forward_with(
                &ops.op,
                &ops.weights()?,
                &inputs,
                state,
                backend.dense_projector(),
            )
            .output
        }
        PreparedAttention::Kda(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} runs a recurrence, which needs durable continuation state, \
                     and this traversal was given no provider to hold it",
                    layer.layer
                )));
            };
            let state = provider.recurrent_state(layer_index)?;
            // The batch is the sequence, flat: KDA's reference consumes
            // positions one after another because the recurrence IS
            // sequential, and this traversal differs from the decode
            // path only in how many positions it hands over.
            let flat: Vec<f32> = inputs.concat();
            let hidden_width = inputs.first().map_or(0, Vec::len);
            let planes = kda::layer_forward_with(
                &kda::BackendKdaProjections(backend.dense_projector()),
                &flat,
                hidden_width,
                ops.weights()?,
                ops.op.geometry(),
                state,
                kda::Mutation::None,
            );
            planes
                .output
                .chunks_exact(hidden_width.max(1))
                .map(<[f32]>::to_vec)
                .collect()
        }
        PreparedAttention::Mla(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} keeps a per-position latent cache, and this traversal was \
                     given no provider to hold it",
                    layer.layer
                )));
            };
            let weights = ops.weights()?;
            let geometry = ops.op.geometry();
            let projector = backend.dense_projector();
            let latent = provider.latent_state(layer_index)?;
            // Position by position, appending each one's latent before
            // reading the prefix back — the same call the decode path
            // makes, run `inputs.len()` times. A whole-sequence form
            // would need its own explicit causal mask; this one's
            // causality is the append-then-read order itself.
            inputs
                .iter()
                .map(|x| {
                    mla::mla_forward_with(
                        projector,
                        x,
                        x.len(),
                        weights,
                        geometry,
                        latent,
                        mla::Mutation::None,
                    )
                    .output
                })
                .collect()
        }
        PreparedAttention::ConvQkv(ops) => {
            let Some((provider, layer_index)) = kv else {
                return Err(VindexError::Parse(format!(
                    "layer {} keeps a KV cache and a conv history, and this traversal \
                     was given no provider to hold either",
                    layer.layer
                )));
            };
            // Two regions, one provider, borrowed in phases: the rows
            // already persisted are copied out first (the reference
            // executor is deliberately literal — speed is a later
            // rung's problem), the conv history is advanced by the
            // forward, and the batch's new rows are appended after.
            let past_keys: Vec<Vec<f32>> = provider.keys(layer_index).to_vec();
            let past_values: Vec<Vec<f32>> = provider.values(layer_index).to_vec();
            let base = provider.position();
            let state = provider.recurrent_state(layer_index)?;
            let planes = conv_qkv::layer_forward_with(
                &ops.op,
                &ops.weights()?,
                &inputs,
                state,
                &past_keys,
                &past_values,
                base,
                backend.dense_projector(),
            );
            for (key, value) in planes.keys.into_iter().zip(planes.values) {
                provider.append(layer_index, key, value);
            }
            planes.output
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
                        layer.declared_norm_eps,
                        hidden,
                    ))?;
                    for (key, value) in out.keys.into_iter().zip(out.values) {
                        kv.append(layer_index, key, value);
                    }
                    out.outputs
                }
                Some((kv, layer_index)) => {
                    let _site = cpu::ledger::in_site(cpu::ledger::Site::Attention);
                    attention_into_kv(
                        attention_op,
                        ops,
                        &inputs,
                        layer.declared_norm_eps,
                        hidden,
                        backend,
                        kv,
                        layer_index,
                    )?
                }
                None => {
                    backend
                        .attention(ops.call(
                            attention_op,
                            &inputs,
                            layer.declared_norm_eps,
                            hidden,
                        ))?
                        .outputs
                }
            }
        }
    };
    // The tail every position shares: post-norm and residual scaling,
    // then the site's update — the residual add on rows, stage five on
    // bundles.
    let deltas: Vec<Vec<f32>> = attn_out
        .into_par_iter()
        .map(|out| {
            let mut out = match &prepared.post_attention {
                Some(norm) => norm.apply(backend, &out),
                None => out,
            };
            scale_residual_delta(layer.residual_scale, &mut out);
            out
        })
        .collect();
    leave_batch_site(
        backend,
        h,
        deltas,
        reductions,
        BatchSiteContext {
            layer: index,
            site: HcSite::Attention,
            mutation,
        },
        sink,
    )?;
    let post_attention = h.clone();

    // A mixer-only (Mamba2) layer carries no FFN program: its one
    // residual update happened above, and the layer is complete.
    // Presence follows the program at execution time too.
    // The FFN OP is the discriminator, never its pre-norm — see the
    // decode path's note. A post-norm layer has no pre-FFN norm and must
    // still run its FFN, over the raw residual.
    let (Some(ffn), Some(ffn_op)) = (&prepared.ffn, &layer.ffn) else {
        return Ok(LayerTrace {
            post_attention,
            ffn_input: Vec::new(),
            post_layer: h.clone(),
        });
    };
    // ── FFN site: enter ──
    let BatchSiteEntry {
        branch_inputs: ffn_branch_inputs,
        reductions: ffn_reductions,
    } = enter_batch_site(
        h,
        prepared.hyper_connection.as_ref().map(|hc| &hc.ffn),
        topology,
        layer.declared_norm_eps,
        mutation,
    )?;
    // The normed FFN inputs are computed once here (same values the
    // in-loop computation produced — one deterministic norm per row)
    // so the trace can carry the tap without a second norm pass.
    let ffn_inputs: Vec<Vec<f32>> = match &prepared.pre_ffn {
        Some(pre_ffn) => ffn_branch_inputs
            .par_iter()
            .map(|row| pre_ffn.apply(backend, row))
            .collect(),
        None => ffn_branch_inputs.to_vec(),
    };
    // The FFN's raw residual — what a hybrid's router and expert
    // pre-norm read — is the vector the branch sees: the reduced one on
    // a bundle, never a stream of it (the (d) controls hand it one).
    let residual_source: Cow<'_, [Vec<f32>]> = match (mutation, &*h) {
        (Mutation::HybridResidualFromStreamZero, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(|x| x.stream(0).to_vec()).collect())
        }
        (Mutation::HybridResidualFromStreamMean, Plane::Bundles(bundles)) => {
            Cow::Owned(bundles.iter().map(Bundle::stream_mean).collect())
        }
        _ => Cow::Borrowed(&ffn_branch_inputs),
    };
    let ffn_outs: Vec<Vec<f32>> = if multi_position_ffn() {
        // **CPU-7C2.** One call for every position, rather than a parallel
        // loop over positions each re-entering the executor.
        //
        // The previous shape ran positions through `par_iter_mut`, so
        // every projection inside saw `caller_owns_the_machine` and
        // collapsed to a single worker. CPU-7C1 measured that as
        // `slabs/call` 5.03 -> 2.81 and a 42% loss against serial decode.
        // Here the executor partitions ROWS across its workers and the
        // positions live inside that traversal, which is the ownership
        // rule this module already states.
        let _site = cpu::ledger::in_site(cpu::ledger::Site::Ffn);
        let residuals: Vec<&[f32]> = residual_source.iter().map(Vec::as_slice).collect();
        let normed: Vec<&[f32]> = ffn_inputs.iter().map(Vec::as_slice).collect();
        ffn.apply_from_residual_many(ffn_op, backend, &residuals, &normed, hidden)?
    } else {
        // **Arm B.** The pre-CPU-7C2 shape, kept in the SAME binary so the
        // regression it exhibits is measured beside its fix rather than
        // carried in from another run — the anchor defect CPU-5's G1 was.
        residual_source
            .par_iter()
            .zip(&ffn_inputs)
            .map(|(residual, normed)| {
                // Inside the closure on purpose: this body runs on a
                // rayon worker with its own thread-local, so a guard
                // taken by the caller would attribute none of it.
                let _site = cpu::ledger::in_site(cpu::ledger::Site::Ffn);
                ffn.apply_from_residual(ffn_op, backend, residual, normed, hidden)
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    // The glue AFTER it stays position-parallel: norms, scaling and the
    // update are elementwise and issue no projections, so there is no
    // ownership to collapse.
    let deltas: Vec<Vec<f32>> = ffn_outs
        .into_par_iter()
        .map(|out| {
            let mut out = match &prepared.post_ffn {
                Some(norm) => norm.apply(backend, &out),
                None => out,
            };
            scale_residual_delta(layer.residual_scale, &mut out);
            out
        })
        .collect();
    leave_batch_site(
        backend,
        h,
        deltas,
        ffn_reductions,
        BatchSiteContext {
            layer: index,
            site: HcSite::Ffn,
            mutation,
        },
        sink,
    )?;
    if let Some(scale) = prepared.layer_scale {
        match h {
            Plane::Rows(rows) => rows
                .par_iter_mut()
                .for_each(|row| backend.scale_row(row, scale)),
            // Preparation refuses this combination; reaching it is an
            // executor bug, not a model.
            Plane::Bundles(_) => {
                return Err(VindexError::Parse(format!(
                    "layer {index} carries a layer scale on a hyper-connected component; \
                     preparation should have refused it"
                )))
            }
        }
    }
    Ok(LayerTrace {
        post_attention,
        ffn_input: ffn_inputs,
        post_layer: h.clone(),
    })
}

/// What entering one site produced for the whole batch: the `[hidden]`
/// vector each position's branch sees, and the reductions the update
/// needs afterwards — `None` on rows, where the update is the residual
/// add. The inputs BORROW a row plane (the branch reads the residual
/// itself) and are owned on a bundle plane (the branch reads what the
/// site reduced).
struct BatchSiteEntry<'a> {
    branch_inputs: Cow<'a, [Vec<f32>]>,
    reductions: Option<Vec<SiteReduction>>,
}

/// Which site a batch record belongs to, and under which control.
#[derive(Clone, Copy)]
struct BatchSiteContext {
    layer: usize,
    site: HcSite,
    mutation: Mutation,
}

/// Enter one site for every position. Stages one to three run per
/// position, in parallel; the carrier, the layer's site operands and the
/// topology must agree three ways, which preparation guarantees.
fn enter_batch_site<'a>(
    h: &'a Plane,
    site: Option<&PreparedHcSite>,
    topology: Option<HyperConnection>,
    norm_eps: f64,
    mutation: Mutation,
) -> Result<BatchSiteEntry<'a>, VindexError> {
    match (h, site, topology) {
        (Plane::Rows(rows), None, None) => Ok(BatchSiteEntry {
            branch_inputs: Cow::Borrowed(rows),
            reductions: None,
        }),
        (Plane::Bundles(bundles), Some(site), Some(hc)) => {
            if mutation == Mutation::BypassComposition {
                // The control: stream 0 stands in for every bundle and
                // no split exists to report.
                return Ok(BatchSiteEntry {
                    branch_inputs: Cow::Owned(
                        bundles.iter().map(|x| x.stream(0).to_vec()).collect(),
                    ),
                    reductions: None,
                });
            }
            let weights = site.weights();
            let mut reductions: Vec<SiteReduction> = bundles
                .par_iter()
                .map(|x| hyper_connection::reduce(x, &weights, hc, norm_eps, mutation))
                .collect();
            if mutation == Mutation::SplitFromPositionZero {
                // The control: one position's state applied to every row
                // — invisible at batch size one, which is why the witness
                // runs three distinguishable positions.
                if let Some(first) = reductions.first().cloned() {
                    for reduction in reductions.iter_mut() {
                        *reduction = first.clone();
                    }
                }
            }
            let branch_inputs = reductions.iter().map(|r| r.reduced.clone()).collect();
            Ok(BatchSiteEntry {
                branch_inputs: Cow::Owned(branch_inputs),
                reductions: Some(reductions),
            })
        }
        _ => Err(VindexError::Parse(
            "the residual plane, the layer's hyper-connection sites and the declared topology \
             disagree; preparation should have refused the image"
                .to_string(),
        )),
    }
}

/// Leave one site for every position: fold each position's `[hidden]`
/// delta back into the carrier. On rows that is the residual add; on
/// bundles it is stage five per position — one operation, not an add —
/// and the sink sees every position's state the moment it exists.
fn leave_batch_site<B: PlanBackend + ?Sized>(
    backend: &B,
    h: &mut Plane,
    deltas: Vec<Vec<f32>>,
    reductions: Option<Vec<SiteReduction>>,
    context: BatchSiteContext,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<(), VindexError> {
    match (h, reductions) {
        (Plane::Rows(rows), None) => {
            rows.par_iter_mut()
                .zip(deltas.par_iter())
                .for_each(|(row, delta)| backend.residual_add(row, delta));
            Ok(())
        }
        (Plane::Bundles(bundles), Some(reductions)) => {
            if context.mutation == Mutation::SwapPositionsBeforeUpdate && bundles.len() >= 2 {
                // The control: positions 0 and 1 exchange their bundles
                // between the reduction and the update, so the update
                // carries the wrong position's state forward.
                bundles.swap(0, 1);
            }
            let (splits, reduced): (Vec<SinkhornSplit>, Vec<Vec<f32>>) =
                reductions.into_iter().map(|r| (r.split, r.reduced)).unzip();
            let next: Vec<Bundle> = bundles
                .par_iter()
                .zip(deltas.par_iter())
                .zip(splits.par_iter())
                .map(|((x, delta), split)| {
                    hyper_connection::update(x, delta, split, context.mutation)
                })
                .collect();
            *bundles = next;
            sink(PlaneEvent::HyperConnectionSite(HcSitePlane {
                layer: context.layer,
                site: context.site,
                splits: &splits,
                reduced: &reduced,
                branch_outputs: &deltas,
                bundles_out: bundles,
            }))
        }
        (Plane::Bundles(bundles), None) => {
            // The bypass control: the delta lands in stream 0 alone and
            // no record is emitted, exactly what a traversal that never
            // ran the topology would do.
            debug_assert_eq!(context.mutation, Mutation::BypassComposition);
            bundles
                .par_iter_mut()
                .zip(deltas.par_iter())
                .for_each(|(x, delta)| backend.residual_add(x.stream_mut(0), delta));
            Ok(())
        }
        (Plane::Rows(_), Some(_)) => Err(VindexError::Parse(
            "a row plane received site reductions; preparation should have refused the image"
                .to_string(),
        )),
    }
}

/// Whether the FFN runs as ONE multi-position call (CPU-7C2) or as the
/// pre-C2 parallel loop over positions.
///
/// A CPU-7C2 arm switch, and it exists so arm B and arms C/E/D live in one
/// binary. The alternative — comparing against CPU-7C1's banked
/// `B/(2A) = 1.422` — would anchor a gate on a number measured in another
/// run, on another build, which is exactly the defect that cost CPU-5 its
/// Bank 2.
///
/// Default ON: the multi-position shape is the one that respects this
/// module's own ownership rule, and a default that did not would make
/// every other measurement in the repo a measurement of the defect.
/// Only `0` and `off` select the legacy shape.
pub const MULTI_POSITION_FFN_ENV: &str = "LARQL_FFN_MULTI_POSITION";

static FFN_SHAPE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn multi_position_ffn() -> bool {
    match FFN_SHAPE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = !matches!(
                std::env::var(MULTI_POSITION_FFN_ENV)
                    .ok()
                    .as_deref()
                    .map(str::trim),
                Some("0") | Some("off")
            );
            FFN_SHAPE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
            on
        }
    }
}

/// Select the FFN shape explicitly, for a harness running both arms in one
/// process. Not for production code, for the reason [`multi_position_ffn`]
/// gives.
pub fn set_multi_position_ffn(on: bool) {
    FFN_SHAPE.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
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
    /// Each matrix paired with the operand it binds, field by field.
    pub(super) fn bound<'a>(&'a self, op: &'a AttentionOp) -> Vec<accounting::Bound<'a>> {
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
