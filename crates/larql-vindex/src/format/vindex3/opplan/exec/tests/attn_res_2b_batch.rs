//! **K3-ATTNRES-1 2b — the batch traversal carries ONE residual history
//! per position.**
//!
//! Frozen in `docs/arch-conformance/forecasts/k3-attnres-1-traverse.json`
//! against the oracle commit `ec7da08d`. 2a proved the decode traversal
//! against that oracle; this module proves the batch traversal carries a
//! DISTINCT history per position, agrees with the decode traversal to
//! the bit, and is caught by three positional controls. It lifts
//! nothing: the public loader still refuses, and a resume point carrying
//! a history plane is still refused by name.
//!
//! # The invariant, stated once
//!
//! **Batching may vectorise the branch computation. It may not merge,
//! share, reorder or reinterpret residual history.** Every position
//! snapshots at the same layers — the schedule is a property of depth,
//! not of the token — so all three histories always hold the same COUNT.
//! What separates them is their CONTENTS. A traversal that built one
//! shared history and handed it to every row would produce the right
//! counts, the right candidate counts, a plausible distribution at every
//! site, and a wrong model.
//!
//! # What is foreign here and what is not
//!
//! ```text
//! FOREIGN     the SCHEDULE and the probability BAND, both
//!             read from the oracle's export — which layers reduce, over
//!             how many candidates, where the boundary events fall, and
//!             the non-saturated interval the whole rung is only
//!             measurable inside.
//!
//! NOT FOREIGN A7 compares the batch traversal against the DECODE
//!             traversal. That is a self-consistency check, and it is
//!             worth having only because the decode arm is itself
//!             anchored against the oracle in 2a. Its provenance is
//!             borrowed, and this module says so rather than presenting
//!             batch/decode agreement as an external result.
//! ```
//!
//! The per-site ARITHMETIC is not re-compared against the oracle here:
//! a batch run from tokens enters on the substrate's own embeddings
//! rather than the oracle's, so its states are not the oracle's states.
//! 2a already made that comparison at the only place it is meaningful —
//! the reduction, replayed from the oracle's own recorded entering
//! state. What 2b adds instead is that every batch site reduction is
//! required to equal `attention_residual::reduce` called directly on the
//! state the witness recorded for that position, so the batch path
//! cannot have grown a second, batched implementation of the reduction.
//!
//! # The two traversals emit in different orders, and A7 accounts for it
//!
//! Decode is position-major: a whole stack for position 0, then for
//! position 1. Batch is layer-major: layer 0 for every position, then
//! layer 1. The global event sequences therefore CANNOT be equal, and a
//! test that compared them directly would be asserting an interleaving
//! rather than a topology. A7 compares the per-position SUBSEQUENCE,
//! which preserves every ordering claim the topology actually makes —
//! the order of sites and boundary events within one position's stack.
//!
//! # Why three positions, and the control that proves it was necessary
//!
//! `a_single_position_batch_hides_every_positional_defect` runs the same
//! witness at batch size one and shows all three controls are INVISIBLE
//! there — the swap and the offset need two rows to act on, and
//! broadcasting position 0's state onto position 0 is the identity. A
//! witness built on a one-position fixture would report this transition
//! green while proving nothing about it. The separation the three
//! positions actually achieve is measured first, and every later
//! assertion in the file is void if that measurement fails.

use super::super::attention_residual::{self, History};
use super::super::decode::DecodeSession;
use super::super::hyper_connection::Mutation;
use super::super::kv::RowKvState;
use super::super::observe::{
    AttnResBoundaryRecord, AttnResSiteRecord, HcSite, StepEvent, StepObserver,
};
use super::super::prepared::PreparedOperands;
use super::super::reference::ReferenceBackend;
use super::super::{
    execute_prepared_streaming_mutated, FinalOutput, FinalState, Plane, PlaneEvent, ResumePoint,
};
use super::attn_res_substrate::{
    prepare, substrate, Oracle, Substrate, HIDDEN, LAYERS, MAX_PROB, MIN_PROB, NORM_EPS, POSITIONS,
    TOLERANCE,
};

/// Three distinct tokens, deliberately neither ascending nor adjacent,
/// so no position can be confused with its index and no defect that
/// reverses the order is hidden by a monotone fixture.
pub(super) const TOKENS: [u32; POSITIONS] = [1, 4, 2];

/// The margin the separation precondition requires between any two
/// positions' state at any site.
///
/// A fixed multiple of the comparison tolerance rather than a number
/// read off this fixture: the question it answers is "could a positional
/// defect hide inside the noise floor", and the noise floor is
/// [`TOLERANCE`]. 100x is the margin that makes the answer unambiguous
/// without pinning a coincidence of these particular weights.
const SEPARATION: f32 = TOLERANCE * 100.0;

// ── The witness ─────────────────────────────────────────────────────

/// One site at one position, in the shape BOTH traversals emit.
///
/// Every field is compared by A7, and the derived ones are derived the
/// same way on both paths — `prefix_after` from the history's prefix
/// falling back to the branch output, exactly as the decode traversal
/// does when a boundary reset leaves the branch output standing alone.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SiteRow {
    pub(super) layer: usize,
    pub(super) site: HcSite,
    pub(super) position: usize,
    pub(super) candidate_count: usize,
    pub(super) snapshot_count_before: usize,
    pub(super) probs: Vec<f32>,
    pub(super) mixed: Vec<f32>,
    pub(super) prefix_before: Vec<f32>,
    pub(super) prefix_after: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct BoundaryRow {
    pub(super) layer: usize,
    pub(super) position: usize,
    pub(super) snapshots_before: usize,
    pub(super) snapshots_after: usize,
    pub(super) value: Vec<f32>,
    pub(super) entering_prefix: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Event {
    Site(SiteRow),
    Boundary(BoundaryRow),
}

impl Event {
    fn position(&self) -> usize {
        match self {
            Self::Site(s) => s.position,
            Self::Boundary(b) => b.position,
        }
    }

    fn layer(&self) -> usize {
        match self {
            Self::Site(s) => s.layer,
            Self::Boundary(b) => b.layer,
        }
    }
}

#[derive(Default, Clone)]
pub(super) struct Witness {
    pub(super) events: Vec<Event>,
}

impl Witness {
    /// One position's events, in emission order. The unit A7 compares:
    /// the two traversals interleave positions differently and agree
    /// within one.
    pub(super) fn at(&self, position: usize) -> Vec<Event> {
        self.events
            .iter()
            .filter(|e| e.position() == position)
            .cloned()
            .collect()
    }

    /// Every site record, excluding the exit — which the batch path
    /// reduces for the last position only, and which the exit assertion compares as an
    /// OUTPUT rather than as a record.
    pub(super) fn sites(&self) -> Vec<&SiteRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Site(s) if s.layer < LAYERS => Some(s),
                _ => None,
            })
            .collect()
    }

    pub(super) fn boundaries(&self) -> Vec<&BoundaryRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Boundary(b) => Some(b),
                Event::Site(_) => None,
            })
            .collect()
    }

    pub(super) fn site(&self, layer: usize, site: HcSite, position: usize) -> Option<&SiteRow> {
        self.sites()
            .into_iter()
            .find(|s| s.layer == layer && s.site == site && s.position == position)
    }

    /// The snapshot values this position had accumulated when the site
    /// at `layer` was entered, oldest first — reconstructed from the
    /// witness's OWN boundary rows.
    ///
    /// The boundary event of a layer falls BETWEEN its two sites, so the
    /// attention site of layer L reads the events of layers strictly
    /// before L and the FFN site reads L's own as well. The shared-
    /// reduction assertion rebuilds the
    /// reduction from this and is what proves the executor's recorded
    /// `snapshot_count_before` describes a real set.
    pub(super) fn snapshots_for(
        &self,
        layer: usize,
        site: HcSite,
        position: usize,
    ) -> Vec<Vec<f32>> {
        self.boundaries()
            .into_iter()
            .filter(|b| b.position == position)
            .filter(|b| match site {
                HcSite::Attention => b.layer < layer,
                HcSite::Ffn => b.layer <= layer,
            })
            .map(|b| b.value.clone())
            .collect()
    }
}

impl StepObserver for Witness {
    fn event(&mut self, _event: StepEvent) {}

    fn attention_residual_site(&mut self, r: AttnResSiteRecord<'_>) {
        self.events.push(Event::Site(SiteRow {
            layer: r.layer,
            site: r.site,
            position: r.position,
            candidate_count: r.candidate_count,
            snapshot_count_before: r.snapshot_count_before,
            probs: r.probs.to_vec(),
            mixed: r.mixed_vector.to_vec(),
            prefix_before: r.prefix_before.to_vec(),
            prefix_after: r.prefix_after.to_vec(),
        }));
    }

    fn attention_residual_boundary(&mut self, r: AttnResBoundaryRecord<'_>) {
        self.events.push(Event::Boundary(BoundaryRow {
            layer: r.layer,
            position: r.position,
            snapshots_before: r.snapshots_before,
            snapshots_after: r.snapshots_after,
            value: r.value.to_vec(),
            entering_prefix: r.entering_prefix.to_vec(),
        }));
    }
}

// ── The two runs ────────────────────────────────────────────────────

pub(super) struct Run {
    pub(super) witness: Witness,
    /// The last position's state as the run left it.
    ///
    /// The two paths tap this at DIFFERENT points, and the difference is
    /// not incidental: `StepRun::exit` is the vector the exit reduction
    /// produced, BEFORE the final norm, while a batch `FinalState` is
    /// what the component hands on, AFTER it. the exit assertion applies the norm to the
    /// decode tap rather than comparing the two raw — a test that
    /// compared them directly would be comparing a pre-norm vector to a
    /// post-norm one and calling the difference a defect.
    pub(super) exit: Vec<f32>,
    pub(super) logits: Option<Vec<f32>>,
}

/// Sequential decode over `tokens`, one step per position. The session
/// carries the KV across steps, so position p attends over 0..=p exactly
/// as the batch traversal does; the residual history is built fresh per
/// step, because it is a property of one forward pass.
pub(super) fn decode(sub: &Substrate, tokens: &[u32], mutation: Mutation) -> Run {
    let (_store, ops) = prepare(sub);
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let mut exit = Vec::new();
    let mut logits = None;
    for &token in tokens {
        let step = session
            .step_mutated(token, &mut witness, mutation)
            .expect("the decode step runs the topology");
        exit = step.exit.expect("a whole-stack image reduces at the exit");
        logits = step.logits;
    }
    Run {
        witness,
        exit,
        logits,
    }
}

/// One batch traversal over `tokens`, collected into the same record
/// shape the decode observer produces.
pub(super) fn batch(sub: &Substrate, tokens: &[u32], mutation: Mutation) -> Run {
    let (_store, ops) = prepare(sub);
    let backend = ReferenceBackend::new();
    let mut witness = Witness::default();
    let out = collect(sub, &ops, tokens, None, mutation, &backend, &mut witness)
        .expect("the batch traversal runs the topology");
    let exit = match out.exit {
        FinalState::Hidden(h) => h,
        other => panic!("a whole-stack batch image must exit on a hidden state, got {other:?}"),
    };
    Run {
        witness,
        exit,
        logits: out.logits,
    }
}

/// The plane-event sink, shared by every batch run in this file.
fn collect(
    sub: &Substrate,
    ops: &PreparedOperands,
    tokens: &[u32],
    resume: Option<ResumePoint>,
    mutation: Mutation,
    backend: &ReferenceBackend,
    witness: &mut Witness,
) -> Result<FinalOutput, crate::error::VindexError> {
    let out = execute_prepared_streaming_mutated(
        &sub.plan,
        ops,
        tokens,
        backend,
        resume,
        &mut |event| {
            match event {
                PlaneEvent::AttentionResidualSite(plane) => {
                    for position in 0..plane.reductions.len() {
                        let reduction = &plane.reductions[position];
                        // The same fallback the decode traversal uses: a
                        // boundary reset leaves the branch output as the
                        // whole prefix.
                        let prefix_after = plane.histories_out[position]
                            .prefix()
                            .unwrap_or(&plane.branch_outputs[position])
                            .to_vec();
                        witness.events.push(Event::Site(SiteRow {
                            layer: plane.layer,
                            site: plane.site,
                            position,
                            candidate_count: reduction.probs.len(),
                            snapshot_count_before: plane.snapshot_counts_before[position],
                            probs: reduction.probs.clone(),
                            mixed: reduction.mixed.clone(),
                            prefix_before: plane.prefixes_before[position].clone(),
                            prefix_after,
                        }));
                    }
                }
                PlaneEvent::AttentionResidualBoundary(plane) => {
                    for position in 0..plane.values.len() {
                        witness.events.push(Event::Boundary(BoundaryRow {
                            layer: plane.layer,
                            position,
                            snapshots_before: plane.snapshots_before,
                            snapshots_after: plane.snapshots_after,
                            value: plane.values[position].clone(),
                            entering_prefix: plane.entering_prefixes[position].clone(),
                        }));
                    }
                }
                PlaneEvent::Embedded(_) | PlaneEvent::Layer { .. } => {}
                PlaneEvent::HyperConnectionSite(_) => {
                    panic!("an attention-residual plan emitted a hyper-connection site")
                }
            }
            Ok(())
        },
        mutation,
    )?;
    Ok(out)
}

/// Every unordered pair of positions.
fn pairs() -> Vec<(usize, usize)> {
    (0..POSITIONS)
        .flat_map(|left| ((left + 1)..POSITIONS).map(move |right| (left, right)))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "comparing different shapes");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── The precondition every later assertion rests on ────────────────

/// **The three positions carry measurably different state, everywhere.**
///
/// Runs FIRST in intent and is cited by every control below. If the
/// positions were not separated, the swap would exchange near-identical
/// histories, the broadcast would replace a state with its own twin, and
/// all three controls would report green while rejecting nothing — the
/// exact failure the oracle demonstrated on itself with a saturated
/// softmax, in its positional form.
///
/// Measures the MINIMUM pairwise separation over every site and every
/// boundary event of a clean run and requires it to exceed a fixed
/// multiple of the comparison tolerance. The margin is reported on
/// failure so a fixture that drifts toward degeneracy says how far it
/// drifted.
#[test]
fn the_three_positions_are_separated_before_any_control_is_scored() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);

    let mut worst = f32::INFINITY;
    let mut where_worst = String::new();
    let mut compared = 0;
    // Every PAIR, not every position against position 0: two positions
    // that had collapsed into each other while both differing from the
    // first would be invisible to a star comparison, and that is exactly
    // the degeneracy this test exists to exclude.
    for (left, right) in pairs() {
        for site in run.witness.sites() {
            if site.position != left {
                continue;
            }
            let peer = run
                .witness
                .site(site.layer, site.site, right)
                .unwrap_or_else(|| panic!("no record at layer {} position {right}", site.layer));
            let gap = max_abs_diff(&site.mixed, &peer.mixed);
            if gap < worst {
                worst = gap;
                where_worst = format!(
                    "layer {} {:?} positions {left}v{right}",
                    site.layer, site.site
                );
            }
            compared += 1;
        }
        for boundary in run.witness.boundaries() {
            if boundary.position != left {
                continue;
            }
            let peer = run
                .witness
                .boundaries()
                .into_iter()
                .find(|b| b.layer == boundary.layer && b.position == right)
                .expect("every position takes the same boundary events");
            let gap = max_abs_diff(&boundary.value, &peer.value);
            if gap < worst {
                worst = gap;
                where_worst = format!("boundary {} positions {left}v{right}", boundary.layer);
            }
            compared += 1;
        }
    }

    assert!(compared > 0, "the separation precondition compared nothing");
    assert!(
        worst > SEPARATION,
        "separation: the closest two positions differ by only {worst:e} at {where_worst}, \
         under the required {SEPARATION:e}. Every positional control in this file is \
         void at this separation."
    );
}

// ── The foreign checks: the band and the schedule ───────────────────

/// **A6 — the band.** No record's distribution may leave the oracle's
/// measured non-saturated interval.
///
/// The freeze's hard rule: a run outside the band is an INSTRUMENT
/// FAILURE and proves nothing either way, so this is scored before any
/// value comparison rather than alongside them.
#[test]
fn every_batch_distribution_lies_inside_the_oracles_measured_band() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);
    let sites = run.witness.sites();
    assert!(!sites.is_empty(), "the run produced no distributions");
    for site in sites {
        for &p in &site.probs {
            assert!(
                (MIN_PROB..=MAX_PROB).contains(&p),
                "layer {} {:?} position {}: probability {p:e} outside the oracle's band \
                 [{MIN_PROB:e}, {MAX_PROB}] — a saturated softmax makes every candidate-set \
                 control invisible and this run is not evidence",
                site.layer,
                site.site,
                site.position
            );
        }
    }
}

/// **A4 — the schedule, per position, against the oracle.**
///
/// The oracle's export is the authority for which layers reduce, over
/// how many candidates, and where the boundary events fall. Every
/// position must walk that schedule independently: a traversal that
/// snapshotted at one position and not another, or that let two
/// positions drift apart in DEPTH, is caught here even though its
/// numbers would all be plausible.
///
/// # Emission order is not operation order, and the difference is real
///
/// At a boundary layer the reference reduces the attention site over the
/// OLD snapshot set, then takes the snapshot, then runs the branch. But
/// a site's record cannot be emitted until its branch has run — it
/// carries `prefix_after` — so the boundary record is emitted BEFORE the
/// attention record whose reduction preceded it. Both traversals do this
/// identically, and it is intrinsic rather than incidental: a record
/// completed at entry could not report what the site left behind.
///
/// So the expected sequence below is in EMISSION order, and the
/// operation order is pinned separately and explicitly by
/// `the_boundary_falls_between_the_two_sites_at_every_position`, which
/// reads it off the counts — the mechanism the freeze names. Ordering by
/// position in the stream alone would assert a convention; ordering by
/// the counts asserts the topology.
///
/// This is also where the rung's two NUMERICALLY INERT properties are
/// proven on the batch path — layer 0 emitting no attention record at
/// all, and every layer emitting an FFN record. The oracle measured both
/// at a divergence of exactly 0.0; if this file ever reports either
/// caught by a value comparison, that comparison is broken.
#[test]
fn every_position_walks_the_oracles_schedule() {
    let sub = substrate();
    let oracle = Oracle::load();
    let run = batch(&sub, &TOKENS, Mutation::None);

    let mut expected: Vec<String> = Vec::new();
    for layer in 0..LAYERS {
        // The boundary marker sits here, ahead of the attention entry,
        // for the emission reason above — never because the event
        // precedes the reduction, which it does not.
        if oracle.ran(&format!("/witness/{layer}/snapshot_event/taken")) {
            let before = oracle.count(&format!("/witness/{layer}/snapshots_before"));
            let after = oracle.count(&format!("/witness/{layer}/snapshots_after"));
            expected.push(format!("layer {layer} boundary {before}->{after}"));
        }
        if oracle.ran(&format!("/witness/{layer}/attention_site/ran")) {
            let n = oracle.count(&format!("/witness/{layer}/attention_site/candidate_count"));
            expected.push(format!("layer {layer} attn over {n}"));
        }
        assert!(
            oracle.ran(&format!("/witness/{layer}/mlp_site/ran")),
            "the oracle's mlp site is unconditional; layer {layer} says otherwise"
        );
        let n = oracle.count(&format!("/witness/{layer}/mlp_site/candidate_count"));
        expected.push(format!("layer {layer} mlp over {n}"));
    }
    // The schedule the oracle read out of the reference: layer 0 has no
    // attention entry at all, and no entry anywhere reduces over one.
    assert!(
        !expected.contains(&"layer 0 attn over 1".to_string())
            && !expected.iter().any(|e| e.ends_with(" over 1")),
        "no site in the reference's schedule reduces over one candidate: {expected:?}"
    );

    for position in 0..POSITIONS {
        let actual: Vec<String> = run
            .witness
            .at(position)
            .iter()
            .filter(|e| e.layer() < LAYERS)
            .map(|e| match e {
                Event::Site(s) => {
                    let name = match s.site {
                        HcSite::Attention => "attn",
                        HcSite::Ffn => "mlp",
                    };
                    format!("layer {} {name} over {}", s.layer, s.candidate_count)
                }
                Event::Boundary(b) => format!(
                    "layer {} boundary {}->{}",
                    b.layer, b.snapshots_before, b.snapshots_after
                ),
            })
            .collect();

        assert_eq!(
            actual, expected,
            "position {position} does not walk the oracle's schedule"
        );
    }
}

/// **The ordering claim of decision 2, read off the counts, per
/// position.**
///
/// At a boundary layer the attention site must have reduced over the set
/// the event had NOT yet extended, and the FFN site over the one it had.
/// The counts say so unambiguously whatever order the records were
/// emitted in, which is why this is the assertion that carries the claim
/// and the sequence check above is not.
///
/// Also pins the snapshot's identity: the value appended is the ENTERING
/// prefix state, per position, and not the mixed vector or the
/// post-attention prefix — the two controls the oracle scored at 9.11e-01
/// and 1.72e+00.
#[test]
fn the_boundary_falls_between_the_two_sites_at_every_position() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);

    for position in 0..POSITIONS {
        let boundaries: Vec<_> = run
            .witness
            .boundaries()
            .into_iter()
            .filter(|b| b.position == position)
            .collect();
        assert_eq!(
            boundaries.iter().map(|b| b.layer).collect::<Vec<_>>(),
            vec![0, 3, 6],
            "position {position}: a boundary at every layer where layer % block == 0"
        );
        for boundary in boundaries {
            assert_eq!(
                boundary.snapshots_after,
                boundary.snapshots_before + 1,
                "position {position} layer {}: one snapshot per event",
                boundary.layer
            );
            assert_eq!(
                boundary.value, boundary.entering_prefix,
                "position {position} layer {}: the event snapshots the ENTERING state",
                boundary.layer
            );
            // The attention site read the set BEFORE the event...
            if let Some(attention) = run
                .witness
                .site(boundary.layer, HcSite::Attention, position)
            {
                assert_eq!(
                    attention.snapshot_count_before, boundary.snapshots_before,
                    "position {position} layer {}: the attention site read the extended set",
                    boundary.layer
                );
            }
            // ...and the FFN site read the set after it.
            let ffn = run
                .witness
                .site(boundary.layer, HcSite::Ffn, position)
                .expect("the mlp site is unconditional");
            assert_eq!(
                ffn.snapshot_count_before, boundary.snapshots_after,
                "position {position} layer {}: the mlp site read the un-extended set",
                boundary.layer
            );
        }
    }
}

// ── A7: batch against decode ────────────────────────────────────────

/// **A7 — the batch and decode traversals agree to the bit, per
/// position.**
///
/// Exact equality, not a tolerance: the two paths run the same reduction
/// over the same state in the same order, so any difference is a
/// difference in what they did rather than in how they rounded.
///
/// Compared per position because the two paths interleave positions
/// differently — decode is position-major, batch layer-major — and the
/// ordering claim of the topology is a claim about one position's
/// sequence.
#[test]
fn the_batch_traversal_equals_the_decode_traversal_at_every_position() {
    let sub = substrate();
    let batched = batch(&sub, &TOKENS, Mutation::None);
    let stepped = decode(&sub, &TOKENS, Mutation::None);
    assert_a7(&batched.witness, &stepped.witness).expect("A7");

    // The exit. The batch path reduces at the exit for the LAST
    // position only — as every other topology's batch path does — so the
    // exit is compared as an OUTPUT rather than as a record.
    //
    // Compared at both taps, because they answer different questions.
    // The decode tap is the reduced vector before the final norm and the
    // batch one is after it, so the norm is applied to the decode tap
    // here: equality then says the two paths ran the same reduction AND
    // that the norm comes after it on both, which is A5's ordering claim.
    // The logits are compared as well, so nothing between the reduction
    // and the head is exempt.
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let normed = match ops.final_norm() {
        Some(norm) => norm.apply(&backend, &stepped.exit),
        None => stepped.exit.clone(),
    };
    assert_eq!(
        batched.exit, normed,
        "A7: the batch exit differs from the decode exit under the final norm"
    );
    assert_eq!(
        batched.exit.len(),
        HIDDEN,
        "the exit reduction produced the wrong width"
    );
    let batch_logits = batched.logits.expect("the batch path produces logits");
    let decode_logits = stepped.logits.expect("the decode path produces logits");
    assert_eq!(
        batch_logits, decode_logits,
        "A7: the batch and decode logits differ at the last position"
    );

    // And the decode arm really did run its exit reduction once per
    // step, so the equality above is an equality of two exits rather
    // than of two paths that both skipped one.
    let exit_records = stepped
        .witness
        .events
        .iter()
        .filter(|e| e.layer() == LAYERS)
        .count();
    assert_eq!(
        exit_records, POSITIONS,
        "each decode step must reduce once at the exit"
    );
}

/// A7's comparison, returning the first disagreement rather than
/// panicking, so the control tests can require it to FAIL and say where.
pub(super) fn assert_a7(batched: &Witness, stepped: &Witness) -> Result<(), String> {
    for position in 0..POSITIONS {
        let left = batched.at(position);
        let right: Vec<Event> = stepped
            .at(position)
            .into_iter()
            .filter(|e| e.layer() < LAYERS)
            .collect();
        if left.len() != right.len() {
            return Err(format!(
                "position {position}: {} batch events against {} decode events",
                left.len(),
                right.len()
            ));
        }
        for (index, (a, b)) in left.iter().zip(&right).enumerate() {
            if a != b {
                return Err(format!(
                    "position {position} event {index}: batch {a:?} != decode {b:?}"
                ));
            }
        }
    }
    Ok(())
}

// ── The reduction is the SHARED one ─────────────────────────────────

/// **The batch path reduces through `attention_residual::reduce`, over
/// the state the witness says it had.**
///
/// Rebuilds each site's history from the recorded prefix and the
/// recorded boundary values — the witness's own events, not the
/// executor's internals — calls the shared reduction directly, and
/// requires bit equality.
///
/// What it rules out: a batched re-implementation of the reduction that
/// agrees with the scalar one on this fixture but is a second source of
/// truth; and a `snapshot_count_before` that is a plausible number
/// rather than the size of the set actually read, since a wrong count
/// makes the rebuilt history the wrong length and the probabilities the
/// wrong width.
#[test]
fn every_batch_reduction_is_the_shared_reduction_over_the_recorded_state() {
    let sub = substrate();
    let run = batch(&sub, &TOKENS, Mutation::None);
    let oracle = Oracle::load();

    let mut checked = 0;
    for site in run.witness.sites() {
        let snapshots = run
            .witness
            .snapshots_for(site.layer, site.site, site.position);
        assert_eq!(
            snapshots.len(),
            site.snapshot_count_before,
            "layer {} {:?} position {}: the recorded snapshot count does not match the \
             events that produced it",
            site.layer,
            site.site,
            site.position
        );
        let mut history = History::new(site.prefix_before.clone());
        for snapshot in snapshots {
            history.push_snapshot(snapshot);
        }
        assert_eq!(history.candidate_count(), site.candidate_count);

        let (norm, proj) = oracle.site_pair(site.layer, site.site);
        let reduction = attention_residual::reduce(
            &history,
            attention_residual::SitePair {
                norm: &norm,
                proj: &proj,
            },
            NORM_EPS,
            Mutation::None,
        )
        .expect("the shared reduction runs");
        assert_eq!(
            reduction.probs, site.probs,
            "layer {} {:?} position {}: probabilities differ from the shared reduction",
            site.layer, site.site, site.position
        );
        assert_eq!(
            reduction.mixed, site.mixed,
            "layer {} {:?} position {}: mixed vector differs from the shared reduction",
            site.layer, site.site, site.position
        );
        checked += 1;
    }
    // 13 reducing sites per position, the schedule the oracle spells.
    assert_eq!(checked, 13 * POSITIONS, "sites rebuilt");
}

/// **The batch exit reduces the LAST position's own history, and its
/// distribution is the oracle's width.**
///
/// The batch path reduces once at the exit, for the last position, and
/// emits no record for it — so the exit's probabilities and mixed vector
/// are not directly observable the way a site's are. They are recovered
/// here instead of being left unchecked: the last position's final state
/// is rebuilt from the witness's own events — its last FFN site's
/// `prefix_after`, plus every boundary value it recorded — the shared
/// reduction is run over it with the SHIPPED exit pair, and the result
/// is required to equal what the traversal actually produced.
///
/// What this pins that the output comparison alone does not:
///
/// - the exit read the LAST position's history, not another position's.
///   Rebuilding from position 2's recorded events and matching is what
///   says so; a run that had reduced position 0's state would produce a
///   vector this rebuild does not predict.
/// - the exit reduced over four candidates — three snapshots plus the
///   prefix — which is the oracle's own count for this geometry.
/// - the exit's distribution lies inside the oracle's band, like every
///   other reduction in the run.
/// - the final norm is applied AFTER the reduction, which is A5's
///   ordering claim, since the rebuilt mixed vector matches only under
///   that order.
#[test]
fn the_batch_exit_reduces_the_last_positions_own_history() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let oracle = Oracle::load();
    let run = batch(&sub, &TOKENS, Mutation::None);
    let last = POSITIONS - 1;

    // The state the last position left the stack in, from the witness.
    let final_site = run
        .witness
        .site(LAYERS - 1, HcSite::Ffn, last)
        .expect("the last layer's mlp site");
    let mut history = History::new(final_site.prefix_after.clone());
    for boundary in run.witness.boundaries() {
        if boundary.position == last {
            history.push_snapshot(boundary.value.clone());
        }
    }

    let expected_candidates = oracle.count("/exit/candidate_count");
    assert_eq!(
        history.candidate_count(),
        expected_candidates,
        "the exit must reduce over the oracle's candidate count"
    );

    let exit = ops
        .attention_residual_exit()
        .expect("a whole-stack image ships the exit pair");
    let reduction =
        attention_residual::reduce(&history, exit.pair(), exit.norm_eps(), Mutation::None)
            .expect("the shared reduction runs at the exit");
    assert_eq!(
        reduction.probs.len(),
        expected_candidates,
        "the exit distribution's width"
    );
    for &p in &reduction.probs {
        assert!(
            (MIN_PROB..=MAX_PROB).contains(&p),
            "exit probability {p:e} outside the oracle's band"
        );
    }

    let normed = match ops.final_norm() {
        Some(norm) => norm.apply(&backend, &reduction.mixed),
        None => reduction.mixed.clone(),
    };
    assert_eq!(
        run.exit, normed,
        "the batch exit is not the shared reduction over the last position's own history,          under the final norm"
    );

    // And the rebuild is not vacuous: another position's history must
    // NOT predict the exit. Without this the assertion above would pass
    // on a traversal that reduced any position, as long as the rebuild
    // used the same one.
    let other_site = run
        .witness
        .site(LAYERS - 1, HcSite::Ffn, 0)
        .expect("position 0's last mlp site");
    let mut other = History::new(other_site.prefix_after.clone());
    for boundary in run.witness.boundaries() {
        if boundary.position == 0 {
            other.push_snapshot(boundary.value.clone());
        }
    }
    let other_reduction =
        attention_residual::reduce(&other, exit.pair(), exit.norm_eps(), Mutation::None).unwrap();
    assert_ne!(
        other_reduction.mixed, reduction.mixed,
        "position 0's history predicts the same exit as the last position's — the          rebuild cannot tell the positions apart and this test proves nothing"
    );
}

/// **Every position enters the stack as its OWN first prefix, with an
/// EMPTY history — and a reader that asks a history plane for rows is
/// refused by name.**
///
/// The entry condition is where every later divergence between positions
/// begins: nothing is replicated, nothing is shared, and no snapshot
/// exists until the first boundary event takes one. A traversal that
/// entered on one shared state would satisfy every count in this file
/// and none of its values.
///
/// The refusal is decision 1's other half. `try_rows` is what a caller
/// wanting `[positions, hidden]` reaches for — the CLI's layer dump is
/// exactly that caller — and on this topology it must say what the plane
/// holds instead of flattening a prefix-plus-snapshots state into a file
/// whose format nothing could read back.
#[test]
fn every_position_enters_as_its_own_prefix_with_an_empty_history() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let mut embedded: Option<Plane> = None;
    execute_prepared_streaming_mutated(
        &sub.plan,
        &ops,
        &TOKENS,
        &backend,
        None,
        &mut |event| {
            if let PlaneEvent::Embedded(plane) = event {
                embedded = Some(plane.clone());
            }
            Ok(())
        },
        Mutation::None,
    )
    .expect("the batch traversal runs");

    let plane = embedded.expect("the traversal emits its embedding");
    assert_eq!(plane.positions(), POSITIONS);
    assert!(
        plane.bundles().is_none(),
        "an attention-residual plane is not a bundle plane"
    );
    let histories = plane
        .histories()
        .expect("an attention-residual component enters on a history plane");
    assert_eq!(histories.len(), POSITIONS);
    for (position, history) in histories.iter().enumerate() {
        assert_eq!(
            history.snapshot_count(),
            0,
            "position {position} entered with a snapshot it could not have taken"
        );
        assert_eq!(
            history.candidate_count(),
            1,
            "position {position} enters as one candidate: its own prefix"
        );
        assert_eq!(history.hidden(), HIDDEN);
    }
    // ...and they are three DIFFERENT prefixes, which is the whole
    // premise of the separation precondition, measured at its source.
    for (left, right) in pairs() {
        let gap = max_abs_diff(
            histories[left].prefix().expect("an entering prefix"),
            histories[right].prefix().expect("an entering prefix"),
        );
        assert!(
            gap > SEPARATION,
            "positions {left} and {right} entered on the same state ({gap:e})"
        );
    }

    let refusal = match plane.try_rows() {
        Ok(_) => panic!("a history plane must not answer as [hidden] rows"),
        Err(err) => err.to_string(),
    };
    assert!(refusal.contains("residual histories"), "{refusal}");
    assert!(refusal.contains("attention-residual"), "{refusal}");
}

// ── What has NOT lifted ─────────────────────────────────────────────

/// **A resume point carrying a history plane is refused by name.**
///
/// Carrying a typed state through a traversal and reconstructing it from
/// an external representation are different capabilities. Nothing reads
/// a serialised prefix-plus-snapshots state back, and inventing a format
/// for one nothing consumes would be addressability without execution in
/// a new place. 2b builds the first and refuses the second.
#[test]
fn a_resume_point_carrying_a_history_plane_is_refused() {
    let sub = substrate();
    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let mut witness = Witness::default();
    let resume = ResumePoint {
        next_layer: 1,
        hidden: Plane::Histories(
            (0..POSITIONS)
                .map(|p| History::new(vec![p as f32 + 1.0; HIDDEN]))
                .collect(),
        ),
    };
    let refusal = match collect(
        &sub,
        &ops,
        &TOKENS,
        Some(resume),
        Mutation::None,
        &backend,
        &mut witness,
    ) {
        Ok(_) => panic!("a history resume point must be refused"),
        Err(err) => err.to_string(),
    };
    assert!(refusal.contains("resume point"), "{refusal}");
    assert!(refusal.contains("not supported"), "{refusal}");
    assert!(
        witness.events.is_empty(),
        "the refusal must come before any observation"
    );
}
