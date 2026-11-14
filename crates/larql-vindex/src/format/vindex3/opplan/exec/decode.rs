//! Incremental decode over a plan: operand residency plus a KV cache.
//!
//! A [`DecodeSession`] loads every operand **once** — in the format the
//! backend declares — and then advances one token per [`step`], feeding
//! the backend's [`attention_step`] against a per-layer K/V
//! continuation state ([`KvState`], default [`RowKvState`], or a
//! caller-owned provider via [`with_kv_state`](DecodeSession::with_kv_state)).
//! Each step therefore computes exactly one position through the
//! whole stack instead of re-running the forward over the grown
//! sequence, and every weight keeps a stable address for the session's
//! lifetime, which is what lets a pointer-keyed device buffer cache
//! hold the model resident.
//!
//! **This is the second traversal in the executor, and it is pinned to
//! the first.** [`execute_plan_streaming`](super::execute_plan_streaming)
//! remains the batch traversal (parallel positions, streamed planes,
//! resume); this session realises the same program one position at a
//! time. The two share the operand loaders and call construction
//! ([`AttentionOperands`]), and the decode-vs-batch parity tests assert
//! their outputs agree per backend — a change that moves one without
//! the other is a bug by definition.
//!
//! [`step`]: DecodeSession::step
//! [`attention_step`]: super::backend::PlanBackend::attention_step

use std::borrow::Cow;

use super::address::CarrierAddress;
use super::attention_residual::{self, BoundaryPhase};
use super::backend::PlanBackend;
use super::continuation_registry::BoxedContinuation;
use super::hyper_connection::{self, Bundle, Mutation, SiteReduction};
use super::intervene::{Firing, Intervention};
use super::intervene_heads::HeadFiring;
use super::kv::KvState;
use super::observe::{
    AttnResBoundaryRecord, AttnResSiteRecord, CarrierForm, CarrierTransition, CarrierWriteRecord,
    HcSite, HcSiteRecord, HistoryWriteMode, StepEvent, StepObserver,
};
use super::prepared::{
    PreparedAttentionResidual, PreparedAttnResSite, PreparedHcSite, PreparedOperands,
};
use crate::error::VindexError;
use larql_models::config::HyperConnection;

use super::super::ComponentOpPlan;

mod run;
mod steps;

/// Who holds the session's operands: an image the caller prepared —
/// typically at model lifetime, shared by every request on that model —
/// or one this session lowered for itself.
///
/// The borrowed arm is the point of operand residency; the owned arm
/// keeps the `new(plan, source, backend)` constructors working for
/// callers that legitimately want a one-shot session.
enum OperandsSlot<'a> {
    /// Boxed so the slot stays pointer-sized: the borrowed arm is the
    /// hot one (a request over a model-lifetime image), and the owned
    /// arm is a one-shot session that can afford the indirection.
    Owned(Box<PreparedOperands>),
    Borrowed(&'a PreparedOperands),
}

impl OperandsSlot<'_> {
    fn get(&self) -> &PreparedOperands {
        match self {
            OperandsSlot::Owned(ops) => ops,
            OperandsSlot::Borrowed(ops) => ops,
        }
    }
}

/// Who holds the session's continuation state (VI3-INF-2): a provider the
/// caller built and handed over (the session owns it and it ends with the
/// session), or a caller's provider borrowed for the session's lifetime so
/// the state outlives the session. There is no default: which provider
/// holds a conversation is the caller's selection (CONTINUATION-PLUGIN-1
/// C3), never the executor's.
enum KvSlot<'a> {
    Owned(BoxedContinuation),
    Borrowed(&'a mut dyn KvState),
}

impl KvSlot<'_> {
    fn state(&self) -> &dyn KvState {
        match self {
            KvSlot::Owned(state) => &**state,
            KvSlot::Borrowed(state) => &**state,
        }
    }

    fn state_mut(&mut self) -> &mut dyn KvState {
        match self {
            KvSlot::Owned(state) => &mut **state,
            KvSlot::Borrowed(state) => &mut **state,
        }
    }
}

/// What one decode step produces.
pub struct StepOutput {
    /// Logits for the position just consumed, when the plan carries an
    /// output head.
    pub logits: Option<Vec<f32>>,
}

/// How a step enters the stack: an ordinary token, or — for the wave-19a
/// witness — a bundle handed straight to the first executed layer, the
/// decode form of the layer-range contract.
enum Entry {
    Token(u32),
    /// An external embedding row standing in for the token lookup; it
    /// enters the declared topology exactly as an embedding would.
    Hidden(Vec<f32>),
    /// A single-stream entering carrier at the first executed layer.
    Single(Vec<f32>),
    #[cfg(test)]
    Bundle(Bundle),
}

/// The residual carrier through the layer loop (wave 19a).
///
/// One `[hidden]` vector on every component before hyper-connections,
/// and a typed [`Bundle`] on a hyper-connected one. The two arms share
/// every operator: a site REDUCES the bundle to the `[hidden]` vector the
/// operator consumes and UPDATES the bundle from the operator's output,
/// which on the single stream is the residual add. `hidden` means the
/// same thing on both arms; the stream count is a different dimension
/// and never widens it.
enum Carrier {
    Single(Vec<f32>),
    Bundle(Bundle),
    /// One prefix vector plus an ordered history of block-boundary
    /// snapshots (K3-ATTNRES-1, 2a). Not a `Bundle` with a growing
    /// stream count: a bundle's streams are interchangeable parallel
    /// residuals whose count is DECLARED once, and these are ordered
    /// historical states of the prefix produced by boundary EVENTS,
    /// whose count is a function of depth.
    History(attention_residual::History),
}

/// What entering one site produced: the `[hidden]` vector the branch
/// sees, and the reduction the update needs afterwards — `None` on the
/// single stream, where the update is the residual add.
struct SiteEntry {
    branch_input: Vec<f32>,
    reduction: Option<SiteReduction>,
    /// The attention-residual site's entry state, `None` on every other
    /// topology. Distinct from `reduction` above because the two
    /// topologies' reductions produce different objects — a Sinkhorn
    /// split against a distribution over candidates — and a shared field
    /// would have been a union with two empty halves.
    attn_res: Option<AttnResEntry>,
}

/// What entering one attention-residual site produced.
///
/// `reduction` is `None` exactly where the reference does not reduce:
/// layer 0's attention site, whose snapshot set is empty. That absence
/// travels to `leave_site`, which then emits NO record — and the missing
/// record is the observation, because the oracle measured this defect at
/// a divergence of exactly zero.
struct AttnResEntry {
    reduction: Option<attention_residual::Reduction>,
    prefix_before: Vec<f32>,
    snapshot_count_before: usize,
    candidate_count: usize,
}

/// One step's result before it is narrowed to [`StepOutput`]. The
/// witness fields exist only under test, so the production step keeps
/// nothing it does not return.
pub(super) struct StepRun {
    pub(super) logits: Option<Vec<f32>>,
    /// Every intervention that fired on this step, in execution order
    /// (V3-INTERVENE-1). Empty on the production step.
    pub(super) firings: Vec<Firing>,
    /// Every head intervention that fired on this step, in execution
    /// order (V3-INTERVENE-2). Empty on the production step.
    pub(super) head_firings: Vec<HeadFiring>,
    /// The `[hidden]` vector the exit reduced to, before the final norm.
    #[cfg(test)]
    pub(super) exit: Option<Vec<f32>>,
    /// The bundle after the last executed layer, on a hyper-connected
    /// component.
    #[cfg(test)]
    pub(super) bundle: Option<Bundle>,
}

/// Incremental executor over one component plan (see module docs).
pub struct DecodeSession<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    backend: &'a B,
    ops: OperandsSlot<'a>,
    kv: KvSlot<'a>,
    /// Set when a failed block advance could not be put back. The session
    /// then refuses every step, whether or not the provider records it.
    fault: Option<String>,
}

impl<'a, B: PlanBackend> DecodeSession<'a, B> {}

/// What the layer loop leaves for the exit.
struct Exit {
    hidden: Option<Vec<f32>>,
    #[cfg(test)]
    bundle: Option<Bundle>,
}

/// Which site a record belongs to, and under which control.
#[derive(Clone, Copy)]
struct SiteContext<'i> {
    layer: usize,
    site: HcSite,
    position: usize,
    mutation: Mutation,
    /// The intervention that fires on this site's write, if the run
    /// declared one at this address (V3-INTERVENE-1). `None` on the
    /// production step and at every other site.
    intervention: Option<&'i Intervention>,
    /// The whole-carrier scalar the layer applies AFTER this site's
    /// write (Gemma 4 `layer_scalar`) — `Some` only at the FFN site of
    /// a component that declares one, so the write record can carry
    /// what happens to its `after` before the layer boundary.
    layer_scale: Option<f32>,
}

/// Enter one site: produce the `[hidden]` vector the branch consumes.
///
/// The carrier, the layer's site operands and the topology must agree
/// three ways — preparation guarantees it, and a disagreement here is
/// an executor bug rather than a model to run anyway.
fn enter_site(
    carrier: &Carrier,
    site: Option<&PreparedHcSite>,
    topology: Option<HyperConnection>,
    norm_eps: f64,
    mutation: Mutation,
) -> Result<SiteEntry, VindexError> {
    match (carrier, site, topology) {
        (Carrier::Single(h), None, None) => Ok(SiteEntry {
            branch_input: h.clone(),
            reduction: None,
            attn_res: None,
        }),
        (Carrier::Bundle(x), Some(site), Some(hc)) => {
            if mutation == Mutation::BypassComposition {
                // The control: stream 0 stands in for the whole bundle
                // and no split exists to report.
                return Ok(SiteEntry {
                    branch_input: x.stream(0).to_vec(),
                    reduction: None,
                    attn_res: None,
                });
            }
            let reduction = hyper_connection::reduce(x, &site.weights(), hc, norm_eps, mutation);
            Ok(SiteEntry {
                branch_input: reduction.reduced.clone(),
                reduction: Some(reduction),
                attn_res: None,
            })
        }
        _ => Err(VindexError::Parse(
            "the residual carrier, the layer's hyper-connection sites and the declared topology \
             disagree; preparation should have refused the image"
                .to_string(),
        )),
    }
}

/// Enter one site, on whichever residual programme the carrier runs.
///
/// One call site per sublayer in the layer loop, so the loop reads the
/// same on every topology and the difference lives here.
#[allow(clippy::too_many_arguments)]
fn enter(
    carrier: &Carrier,
    hc_site: Option<&PreparedHcSite>,
    attn_res: Option<&PreparedAttentionResidual>,
    which: HcSite,
    topology: Option<HyperConnection>,
    norm_eps: f64,
    layer: usize,
    mutation: Mutation,
) -> Result<SiteEntry, VindexError> {
    match (carrier, attn_res) {
        (Carrier::History(history), Some(sites)) => {
            let site = match which {
                HcSite::Attention => &sites.attention,
                HcSite::Ffn => &sites.ffn,
            };
            enter_attention_residual_site(history, site, which, norm_eps, layer, mutation)
        }
        (Carrier::History(_), None) => Err(VindexError::Parse(format!(
            "layer {layer} carries a residual history and no attention-residual sites; \
             preparation should have refused the image"
        ))),
        _ => enter_site(carrier, hc_site, topology, norm_eps, mutation),
    }
}

/// Enter one attention-residual site.
///
/// The reduction is GUARDED at the attention site and UNCONDITIONAL at
/// the mlp site, which is the reference's own asymmetry
/// (`if block_residual is not None and block_residual.shape[1] > 0`
/// around the first, nothing around the second). Where it does not
/// reduce, the branch input is the prefix itself and no record will be
/// emitted — the reference's guard and a regularised always-run site
/// compute the same vector, so only the absence distinguishes them.
fn enter_attention_residual_site(
    history: &attention_residual::History,
    site: &PreparedAttnResSite,
    which: HcSite,
    norm_eps: f64,
    layer: usize,
    mutation: Mutation,
) -> Result<SiteEntry, VindexError> {
    let prefix = history.prefix().ok_or_else(|| {
        VindexError::Parse(format!(
            "layer {layer} entered a site with no prefix; a boundary reset it and no branch \
             has supplied one"
        ))
    })?;
    let guarded = match which {
        // The reference's guard, and the one control that drops it.
        HcSite::Attention => {
            history.snapshot_count() > 0 || mutation == Mutation::AttnResLayer0AttentionSiteRuns
        }
        // Unconditional in the reference; the control gives it the
        // attention site's guard, and a second skips it at layer 0.
        HcSite::Ffn => match mutation {
            Mutation::AttnResMlpSiteGuardedOnNonEmpty => history.snapshot_count() > 0,
            Mutation::AttnResMlpSiteSkippedAtLayer0 if layer == 0 => false,
            _ => true,
        },
    };
    let entry = AttnResEntry {
        prefix_before: prefix.to_vec(),
        snapshot_count_before: history.snapshot_count(),
        candidate_count: history.candidate_count(),
        reduction: None,
    };
    if !guarded {
        return Ok(SiteEntry {
            branch_input: prefix.to_vec(),
            reduction: None,
            attn_res: Some(entry),
        });
    }
    let reduction = attention_residual::reduce(history, site.pair(), norm_eps, mutation)?;
    Ok(SiteEntry {
        branch_input: reduction.mixed.clone(),
        reduction: None,
        attn_res: Some(AttnResEntry {
            reduction: Some(reduction),
            ..entry
        }),
    })
}

/// The block-boundary event, offered at each of the three points a
/// snapshot could be taken. Does something only at the point this run's
/// control selects; the reference's point is
/// [`BoundaryPhase::AfterAttentionReduce`].
///
/// The prefix RESET always happens at the reference's point, whatever
/// the snapshot's phase — the reference resets in the same statement it
/// appends, and a control that moved the append must not silently move
/// the reset with it.
fn boundary_event(
    carrier: &mut Carrier,
    phase: BoundaryPhase,
    entering_prefix: &[f32],
    mixed_vector: &[f32],
    context: SiteContext,
    observer: &mut dyn StepObserver,
) {
    let Carrier::History(history) = carrier else {
        return;
    };
    let snapshot_phase = match context.mutation {
        Mutation::AttnResSiteOverNewSnapshots => BoundaryPhase::BeforeAttentionReduce,
        Mutation::AttnResSnapshotAfterAttention => BoundaryPhase::AfterAttentionBranch,
        _ => BoundaryPhase::AfterAttentionReduce,
    };
    if phase == snapshot_phase {
        // WHICH vector is snapshotted is the other thing a control
        // perturbs, and the reference's answer is the ENTERING prefix
        // state — not the attention site's output, and not the
        // post-attention prefix.
        let value: Vec<f32> = match (context.mutation, phase) {
            (Mutation::AttnResSnapshotIsMixedVector, _) => mixed_vector.to_vec(),
            (_, BoundaryPhase::AfterAttentionBranch) => history
                .prefix()
                .map(<[f32]>::to_vec)
                .unwrap_or_else(|| entering_prefix.to_vec()),
            _ => entering_prefix.to_vec(),
        };
        let before = history.snapshot_count();
        history.push_snapshot(value.clone());
        name(
            observer,
            &context,
            CarrierForm::History,
            CarrierTransition::HistorySnapshot,
        );
        observer.attention_residual_boundary(AttnResBoundaryRecord {
            layer: context.layer,
            position: context.position,
            snapshots_before: before,
            snapshots_after: history.snapshot_count(),
            value: &value,
            entering_prefix,
        });
    }
    if phase == BoundaryPhase::AfterAttentionReduce {
        history.reset_prefix();
        name(
            observer,
            &context,
            CarrierForm::History,
            CarrierTransition::HistoryReset,
        );
    }
}

/// Name one transition at this site's address (RESIDUAL-BUS-1 T1, with
/// RESIDUAL-BUS-2's address): the decode position is already absolute.
fn name(
    observer: &mut dyn StepObserver,
    context: &SiteContext<'_>,
    form: CarrierForm,
    transition: CarrierTransition,
) {
    observer.transition(
        CarrierAddress::of(context.position, context.layer, form, &transition),
        transition,
    );
}

/// Leave one site: fold the branch's `[hidden]` delta back into the
/// carrier. On the single stream that is the residual add; on a bundle
/// it is stage five, which carries every stream forward through `comb`
/// and scatters the delta by `post` — one operation, not an add — and
/// the observer sees the site's state the moment it exists.
fn leave_site<B: PlanBackend + ?Sized>(
    backend: &B,
    carrier: &mut Carrier,
    delta: Vec<f32>,
    reduction: Option<SiteReduction>,
    attn_res: Option<AttnResEntry>,
    context: SiteContext<'_>,
    observer: &mut dyn StepObserver,
) -> Result<(), VindexError> {
    // The attention-residual write, and it is NOT always an add: a
    // boundary reset this prefix, and then the branch's output BECOMES
    // the prefix rather than being added to one. `History::write` is one
    // method for that reason — the reference is one expression, and
    // splitting it would let a caller forget the second arm.
    if let (Carrier::History(history), Some(entry)) = (&mut *carrier, attn_res) {
        let mode = if history.prefix().is_some() {
            HistoryWriteMode::Add
        } else {
            HistoryWriteMode::Replace
        };
        history.write(&delta);
        name(
            observer,
            &context,
            CarrierForm::History,
            CarrierTransition::HistoryWrite {
                site: context.site,
                mode,
            },
        );
        // Emitted only where the reference REDUCED. Layer 0's attention
        // site emits nothing, and that absence is the observation.
        if let Some(reduction) = &entry.reduction {
            let prefix_after = history.prefix().unwrap_or(&delta);
            observer.attention_residual_site(AttnResSiteRecord {
                layer: context.layer,
                site: context.site,
                position: context.position,
                candidate_count: entry.candidate_count,
                snapshot_count_before: entry.snapshot_count_before,
                probs: &reduction.probs,
                mixed_vector: &reduction.mixed,
                prefix_before: &entry.prefix_before,
                prefix_after,
            });
        }
        // The write happened whether or not the reference reduced here:
        // layer 0's attention site writes with no record, and the event
        // is what says so.
        observer.event(StepEvent::CarrierWrite {
            layer: context.layer,
            site: context.site,
            carrier: CarrierForm::History,
        });
        return Ok(());
    }
    match (carrier, reduction) {
        (Carrier::Single(h), None) => {
            // V3-INTERVENE-1: the write lands first, exactly as it would
            // unintervened; the intervention then acts on the written
            // carrier in the backend's residual path, and the record's
            // `delta` becomes `after − before` so it still includes what
            // landed. The event precedes the record so its reader knows.
            // An intervention that leaves the carrier bit-identical to
            // the unpatched write (`Add(0)`, `Replace` with its own value)
            // reports the branch's own `delta`, borrowed: `after − before`
            // is the ROUNDED write, not the branch output, and the no-op
            // law (I3/IP1) is a claim on the record, not only the logits.
            let before = context.intervention.map(|_| h.clone());
            backend.residual_add(h, &delta);
            name(
                observer,
                &context,
                CarrierForm::Single,
                CarrierTransition::Add { site: context.site },
            );
            let delta: Cow<'_, [f32]> = match (context.intervention, before) {
                (Some(intervention), Some(before)) => {
                    let unpatched = h.clone();
                    intervention.apply(backend, h);
                    observer.event(StepEvent::Intervened {
                        layer: context.layer,
                        site: context.site,
                        kind: intervention.kind(),
                    });
                    name(
                        observer,
                        &context,
                        CarrierForm::Single,
                        CarrierTransition::Intervene {
                            site: context.site,
                            kind: intervention.kind(),
                        },
                    );
                    if h.as_slice() == unpatched.as_slice() {
                        Cow::Borrowed(delta.as_slice())
                    } else {
                        Cow::Owned(h.iter().zip(&before).map(|(a, b)| a - b).collect())
                    }
                }
                _ => Cow::Borrowed(delta.as_slice()),
            };
            // V3-OBS-1: the write, borrowed where it landed. Values
            // first, then the structural event that closes it.
            observer.carrier_write(CarrierWriteRecord {
                layer: context.layer,
                site: context.site,
                position: context.position,
                delta: &delta,
                after: h.as_slice(),
                layer_scale: context.layer_scale,
            });
            observer.event(StepEvent::CarrierWrite {
                layer: context.layer,
                site: context.site,
                carrier: CarrierForm::Single,
            });
            Ok(())
        }
        (Carrier::Bundle(x), Some(reduction)) => {
            let next = hyper_connection::update(x, &delta, &reduction.split, context.mutation);
            observer.hyper_connection_site(HcSiteRecord {
                layer: context.layer,
                site: context.site,
                position: context.position,
                split: &reduction.split,
                reduced: &reduction.reduced,
                branch_output: &delta,
                bundle_out: &next,
            });
            *x = next;
            name(
                observer,
                &context,
                CarrierForm::Bundle,
                CarrierTransition::HcUpdate { site: context.site },
            );
            observer.event(StepEvent::CarrierWrite {
                layer: context.layer,
                site: context.site,
                carrier: CarrierForm::Bundle,
            });
            Ok(())
        }
        (Carrier::Bundle(x), None) => {
            // The bypass control: the delta lands in stream 0 alone and
            // no record is emitted, exactly what a traversal that never
            // ran the topology would do.
            debug_assert_eq!(context.mutation, Mutation::BypassComposition);
            backend.residual_add(x.stream_mut(0), &delta);
            name(
                observer,
                &context,
                CarrierForm::Bundle,
                CarrierTransition::Add { site: context.site },
            );
            // Still a write: the control drops the RECORD (no split
            // exists), never the fact that the carrier moved.
            observer.event(StepEvent::CarrierWrite {
                layer: context.layer,
                site: context.site,
                carrier: CarrierForm::Bundle,
            });
            Ok(())
        }
        // A history carrier reaches this match only if its entry was
        // lost on the way in; the arm above returns before it otherwise.
        (Carrier::History(_), _) => Err(VindexError::Parse(
            "an attention-residual carrier left a site with no site entry; the traversal built \
             one at every site it entered"
                .to_string(),
        )),
        (Carrier::Single(_), Some(_)) => Err(VindexError::Parse(
            "a single-stream carrier received a site reduction; preparation should have refused \
             the image"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests;
