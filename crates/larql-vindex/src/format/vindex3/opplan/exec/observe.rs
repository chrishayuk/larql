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
//! Deliberately coarse at this rung: layer and sublayer boundaries and
//! the head's logits — structure, not tensors. Finer taps (operand
//! reads, attention state, residual values) are later detail levels
//! and must arrive the same way: more events on the one executor,
//! never a second traversal.
//!
//! [`DecodeSession::step_observed`]: super::decode::DecodeSession::step_observed
//! [`step`]: super::decode::DecodeSession::step

use super::hyper_connection::{Bundle, SinkhornSplit};

/// One decode step's observation events, in execution order.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// A subscriber to the canonical step's observation points.
pub trait StepObserver {
    fn event(&mut self, event: StepEvent);

    /// Observe an operand input's values. Separate from [`event`] so the
    /// values are borrowed rather than cloned into an event: capturing
    /// second moments needs to read the vector, not own it.
    ///
    /// [`event`]: Self::event
    fn operand_input(&mut self, _layer: usize, _site: InputSite, _values: &[f32]) {}

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
