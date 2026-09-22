//! Observation points on the canonical decode step (LQL-2 TRACE).
//!
//! There is exactly one semantic execution path; TRACE and any other
//! observer **subscribe to it** — nothing re-enacts the plan to emit
//! events. [`DecodeSession::step_observed`] fires these events at the
//! step's existing operation boundaries and computes exactly what
//! [`step`] computes: the parity gate demands the observed and
//! unobserved paths stay bit-identical, so an observer can never
//! change arithmetic or execution order.
//!
//! The structural events are deliberately coarse: layer and sublayer
//! boundaries and the head's logits. Finer taps arrive the same way —
//! more events and borrowed records on the one executor, never a
//! second traversal: operand inputs (sensitivity), the topology site
//! records (waves 19 and K3-ATTNRES-1), and the carrier writes
//! themselves (V3-OBS-1, `docs/v3-obs-1-carrier-observation.md`).
//!
//! [`DecodeSession::step_observed`]: super::decode::DecodeSession::step_observed
//! [`step`]: super::decode::DecodeSession::step

use super::hyper_connection::{Bundle, SinkhornSplit};
use super::intervene::InterventionKind;

/// One decode step's observation events, in execution order.
///
/// Non-exhaustive on purpose: finer taps arrive as new variants
/// (routing, operand reads, representation resolution, refusals), and
/// a consumer in another crate that renders what it knows must keep
/// compiling when the executor learns to say more.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StepEvent {
    /// The token was embedded at this absolute position.
    Embedded { position: usize },
    /// A layer's attention sublayer completed (residual add included).
    AttentionDone { layer: usize },
    /// A layer's FFN sublayer completed (residual add and any layer
    /// scale included) — the layer boundary.
    FfnDone { layer: usize },
    /// The output head priced the vocabulary for this position.
    Logits { vocab: usize },
    /// A sublayer's output was written into the residual carrier
    /// (V3-OBS-1) — the write itself, as distinct from the sublayer's
    /// completion: `AttentionDone` and `FfnDone` are boundaries and fire
    /// on every layer, whereas this fires once per write the layer's
    /// program actually performs (a mixer-only layer writes once). On
    /// every carrier form; the values precede it, on
    /// [`StepObserver::carrier_write`] for a single stream or on the
    /// form's own site record for a bundle or a history.
    CarrierWrite {
        layer: usize,
        site: SublayerSite,
        carrier: CarrierForm,
    },
    /// V3-INTERVENE-1: an intervention fired on this site's write. Fires
    /// BEFORE the write's record and its structural event, so a reader of
    /// that record knows its `after` is not the branch's own.
    Intervened {
        layer: usize,
        site: SublayerSite,
        kind: InterventionKind,
    },
}

/// The form the residual carrier takes at a write (V3-OBS-1, property
/// C3). Named on the structural event so a structure-only consumer can
/// count writes per form, and so no consumer infers the topology from
/// which value record happened to arrive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CarrierForm {
    /// One `[hidden]` vector; the write is a residual add.
    Single,
    /// A hyper-connected bundle; the write is the site's expansion.
    Bundle,
    /// An attention-residual prefix plus its snapshot history; the write
    /// is `History::write`, which adds OR replaces.
    History,
}

/// One single-stream carrier write, borrowed at the write (V3-OBS-1).
///
/// `delta` is the branch output AS ADDED — after the sublayer's
/// post-norm and residual-delta scale where the plan has them — and
/// `after` is the carrier once the add has landed. `before` is not
/// carried: the add is in place, and copying the carrier first would
/// cost the unobserved path what observation must not. A consumer
/// chains instead — a write's `before` is the previous write's `after`
/// (times `layer_scale` where the previous write carried one), and the
/// first link is [`StepObserver::entering_carrier`].
#[derive(Debug, Clone, Copy)]
pub struct CarrierWriteRecord<'a> {
    pub layer: usize,
    pub site: SublayerSite,
    pub position: usize,
    pub delta: &'a [f32],
    pub after: &'a [f32],
    /// The per-layer scalar the executor multiplies the WHOLE carrier by
    /// after this write and before the layer boundary (Gemma 4
    /// `layer_scalar`), on the FFN site of a component that declares
    /// one; `None` where the program has no such scale. Carried so the
    /// layer output `layer_scale * after` is reconstructable and the
    /// batch path's `post_layer`, captured after the scale, is
    /// comparable.
    pub layer_scale: Option<f32>,
}

/// Where in a layer an activation was taken.
///
/// Two sites, because two suffice: everything else is derivable from them
/// offline. `q/k/v` read the attention input; `gate/up` read the FFN
/// input; and `down`'s input is `act(gate(x)) * up(x)`, which a screen can
/// reconstruct from the FFN input and those two operands rather than
/// needing its own tap.
///
/// `o_proj` is the exception and is *not* covered: its input is the
/// attention core's output, which never surfaces at this boundary. A
/// consumer must exclude `o_proj` rather than approximate it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputSite {
    /// Normalised residual entering attention — input to q, k and v.
    Attention,
    /// Normalised residual entering the FFN — input to gate and up.
    Ffn,
    /// The FFN sublayer's own output, before any post-norm or residual
    /// scaling.
    ///
    /// Not an input, and present for one reason: it is the control that
    /// proves `down_proj`'s reconstructed input is the executor's. A
    /// screen that reconstructs `act(gate(x)) ⊙ up(x)` can check itself by
    /// multiplying through `down_proj` and comparing here — so the
    /// reconstruction is verified rather than believed.
    FfnOutput,
}

/// Which of a transformer block's two sublayers a residual site wraps.
///
/// Topology-neutral on purpose: hyper-connections and attention
/// residuals both put a site at each of these two places, and a name
/// belonging to one of them would have had to be duplicated for the
/// other. `HcSite` remains as an alias so wave 19's call sites read as
/// they did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SublayerSite {
    Attention,
    Ffn,
}

pub use SublayerSite as HcSite;

/// One block-boundary event of the attention-residual topology
/// (K3-ATTNRES-1, transition 2a) — the THIRD contract point of a site.
///
/// Its own record, and not a field of the site record beside it, for two
/// reasons the freeze names. Layer 0 emits no attention-site record at
/// all, so an event carried on that record would be invisible exactly
/// where the schedule's first event happens; and the claim under test is
/// an ORDERING one — the attention site reads the set this event has not
/// yet extended, and the mlp site reads the set it has — which needs the
/// event to be a thing in the stream rather than an annotation on
/// something else.
#[derive(Debug, Clone, Copy)]
pub struct AttnResBoundaryRecord<'a> {
    pub layer: usize,
    pub position: usize,
    pub snapshots_before: usize,
    pub snapshots_after: usize,
    /// The vector appended. The reference snapshots the ENTERING prefix
    /// state, and two of the rung's controls perturb exactly this, so it
    /// is recorded rather than assumed.
    pub value: &'a [f32],
    /// The layer's entering prefix, so a witness can assert the equality
    /// rather than trusting the caller passed the right vector.
    pub entering_prefix: &'a [f32],
}

/// One attention-residual site's intermediate state at one position.
///
/// Distinct from [`HcSiteRecord`] rather than a generalisation of it:
/// that record carries a `SinkhornSplit`, which this topology has no
/// analogue of, and this one carries candidate counts and a snapshot
/// count, which that topology has no analogue of. A shared record would
/// have been a union with two empty halves.
///
/// **The counts are load-bearing, not diagnostics.** The rung's oracle
/// measured two of the topology's six properties at a divergence of
/// EXACTLY zero — softmax over one candidate is the identity, so layer
/// 0's skipped attention site computes what a regularised always-run
/// site computes; and the mlp site's guard never fires because no site
/// in the schedule ever sees an empty set. Neither can be caught by any
/// value comparison at any geometry. They are caught HERE, by which
/// records exist and what they count, or they are not caught at all.
#[derive(Debug, Clone, Copy)]
pub struct AttnResSiteRecord<'a> {
    pub layer: usize,
    pub site: SublayerSite,
    pub position: usize,
    /// Snapshots plus the prefix — never fewer than two anywhere in the
    /// reference's schedule.
    pub candidate_count: usize,
    /// The size of the set this reduction ACTUALLY read. At a boundary
    /// layer's attention site this is the count before the event, which
    /// is the whole ordering claim of the topology.
    pub snapshot_count_before: usize,
    /// The distribution over candidates. A single-stream traversal has
    /// no such object, so its existence is what says the topology ran.
    pub probs: &'a [f32],
    /// `probs @ candidates` over the RAW candidates — the vector the
    /// branch consumes, before the site's pre-norm.
    pub mixed_vector: &'a [f32],
    /// The prefix entering the site, and the prefix after the branch's
    /// delta was written. At a boundary layer's attention site the
    /// second is the branch output alone, because the event reset the
    /// first.
    pub prefix_before: &'a [f32],
    pub prefix_after: &'a [f32],
}

/// One hyper-connection site's intermediate state at one position
/// (wave 19a) — the values that exist ONLY if the bundle traversal ran.
///
/// A single-stream traversal of the same plan produces every other tap
/// this module offers; none of them can say whether the Sinkhorn split
/// happened. This record can: the split is stage two's output, the
/// reduced vector is what the ordinary operator actually saw, the
/// branch output is what the expansion consumed, and the bundle after
/// the update is what the next site reads. A witness holding the
/// bundle that entered the site can recompute every one of them.
///
/// Borrowed, like [`StepObserver::operand_input`]: the executor does not
/// clone its state to be observed.
#[derive(Debug, Clone, Copy)]
pub struct HcSiteRecord<'a> {
    pub layer: usize,
    pub site: HcSite,
    pub position: usize,
    /// Stage two's `pre`, `post` and `comb`.
    pub split: &'a SinkhornSplit,
    /// Stage three's `[hidden]` vector — the branch's input, before the
    /// site's pre-norm.
    pub reduced: &'a [f32],
    /// The `[hidden]` delta the update consumed: the branch's output
    /// after its post-norm and residual scaling, where the plan has them.
    pub branch_output: &'a [f32],
    /// The bundle after stage five.
    pub bundle_out: &'a Bundle,
}

/// Borrowed softmax head values from the actual production aggregation loop.
/// Pre output-gate, pre W_O, pre post-attention norm and residual scaling.
/// Source weights cover `source_start..=position`; sink mass is not renormalized.
/// Capturing these values is opt-in and timing-intrusive, separate from V3-OBS-1.
#[derive(Debug)]
pub struct AttentionHeadRecord<'a> {
    pub position: usize,
    pub head: usize,
    pub kv_head: usize,
    pub source_start: usize,
    pub weights: &'a [f32],
    pub values: &'a [f32],
    /// The conditioned query slice used for this head's scores.
    pub query: &'a [f32],
    /// Conditioned K slices, one per position in `source_start..=position`.
    /// Present only on the opt-in observed path.
    pub source_keys: &'a [Vec<f32>],
    /// Conditioned V slices aligned one-for-one with [`Self::source_keys`].
    pub source_values: &'a [Vec<f32>],
}

/// A subscriber to the canonical step's observation points.
pub trait StepObserver {
    fn event(&mut self, event: StepEvent);

    /// Request actual per-head softmax observations. Unsupported backends and
    /// non-softmax plans refuse, rather than silently emitting incomplete data.
    fn wants_attention_heads(&self) -> bool {
        false
    }

    /// Narrow an opt-in head capture to selected layer/position pairs. The
    /// default preserves the original all-head behaviour. This is a capture
    /// cost filter only and may not alter execution arithmetic.
    fn wants_attention_heads_at(&self, _layer: usize, _position: usize) -> bool {
        self.wants_attention_heads()
    }

    fn attention_head(&mut self, _layer: usize, _record: AttentionHeadRecord<'_>) {}

    /// Observe the attention branch after output projection and bias, but
    /// before a declared post-attention norm or residual-delta scale.
    ///
    /// This is the reconstruction authority for norm-aware head replay:
    /// per-head `W_O` contributions must sum to this value before the
    /// nonlinear post norm is applied. Borrowed and observation-only, like
    /// [`Self::attention_head`].
    fn attention_output(&mut self, _layer: usize, _position: usize, _values: &[f32]) {}

    /// Observe an operand input's values. Separate from [`event`] so the
    /// values are borrowed rather than cloned into an event: capturing
    /// second moments needs to read the vector, not own it.
    ///
    /// [`event`]: Self::event
    fn operand_input(&mut self, _layer: usize, _site: InputSite, _values: &[f32]) {}

    /// Observe the `[hidden]` vector entering layer 0 — the embedding
    /// after its scale and norm, before any topology replicates or
    /// wraps it (V3-OBS-1, property C6): the first link of the carrier
    /// chain. Fired once per step, after `Embedded`. Default: ignore.
    fn entering_carrier(&mut self, _position: usize, _values: &[f32]) {}

    /// Observe one single-stream carrier write (V3-OBS-1). Fired once per
    /// write the program performs on a single-stream component,
    /// immediately after the add and before the `CarrierWrite` event.
    /// Bundle and history writes deliver their values through their own
    /// site records instead. Default: ignore.
    fn carrier_write(&mut self, _record: CarrierWriteRecord<'_>) {}

    /// Observe one hyper-connection site's intermediate state. Fired
    /// only on a hyper-connected component, once per site per layer per
    /// step, immediately after the site's update and before the
    /// sublayer's completion event. Default: ignore.
    fn hyper_connection_site(&mut self, _record: HcSiteRecord<'_>) {}

    /// Observe one attention-residual site's intermediate state. Fired
    /// only on a component that declares the topology, and only where
    /// the reference REDUCES: layer 0's attention site emits nothing,
    /// because the reference does not reduce there. That absence is the
    /// observation. Default: ignore.
    fn attention_residual_site(&mut self, _record: AttnResSiteRecord<'_>) {}

    /// Observe one block-boundary event, fired between the attention
    /// site's reduction and the attention branch — the point in the
    /// schedule that wave 19's two-point site seam cannot express.
    /// Default: ignore.
    fn attention_residual_boundary(&mut self, _record: AttnResBoundaryRecord<'_>) {}
}

/// The default subscriber: observes nothing. [`DecodeSession::step`]
/// is `step_observed` with this observer, so the unobserved path is
/// the observed path by construction.
///
/// [`DecodeSession::step`]: super::decode::DecodeSession::step
pub struct NoopObserver;

impl StepObserver for NoopObserver {
    fn event(&mut self, _event: StepEvent) {}
}

/// Convenience subscriber: records every event, for tests and for
/// consumers that render after the step completes.
#[derive(Default)]
pub struct RecordingObserver {
    pub events: Vec<StepEvent>,
}

impl StepObserver for RecordingObserver {
    fn event(&mut self, event: StepEvent) {
        self.events.push(event);
    }
}
