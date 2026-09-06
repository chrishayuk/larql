//! **K3-ATTNRES-1 2a — the decode traversal carries a residual history.**
//!
//! Frozen in `docs/arch-conformance/forecasts/k3-attnres-1-traverse.json`
//! against the oracle commit `ec7da08d`, and scored here. The claim is
//! narrow: the decode step carries an explicit prefix-plus-snapshots
//! state, reproduces the oracle's per-site probabilities and mixed
//! vectors, emits the schedule the reference spells, and is caught by
//! every named ordering mutation. Nothing lifts — the public loader
//! still refuses the topology. (2a also required the BATCH path to
//! refuse by name; 2b replaced that refusal with the per-position
//! history it was standing in for, and the obligation now lives in
//! `attn_res_2b_batch`.)
//!
//! # Two kinds of evidence, and they answer different questions
//!
//! ```text
//! FOREIGN     each site replayed from the ORACLE's own recorded
//!             entering state, its probabilities and mixed vector
//!             compared against the oracle's. This is the arithmetic,
//!             and its reference is a torch transcription of Kimi-K3's
//!             own file.
//!
//! STRUCTURAL  a real decode run over a synthetic stack, its record
//!             stream compared against the oracle's SCHEDULE — which
//!             layers reduce, over how many candidates, and where the
//!             boundary events fall relative to the two sites.
//! ```
//!
//! The full-stack vectors are deliberately NOT compared against the
//! oracle's. The oracle's sublayer is a stand-in (pre-norm, linear,
//! tanh) and this substrate's operators are real attention and a real
//! FFN, so a final-vector comparison would be measuring the branch
//! rather than the topology. What crosses that boundary is the
//! per-site arithmetic and the schedule; the controls are scored
//! against this substrate's own reference run.
//!
//! # Two properties are proven by SHAPE, and can be proven no other way
//!
//! The oracle measured `layer0_attention_site_runs` and
//! `mlp_site_guarded_on_nonempty` at a divergence of EXACTLY zero.
//! Softmax over one candidate is the identity, so layer 0's skipped
//! attention site computes what a regularised always-run site computes;
//! and the mlp site's guard never fires, because no site in this
//! schedule ever sees an empty snapshot set. No value comparison at any
//! geometry can catch either. They are caught by which records exist,
//! and if this file ever reports them caught by a value assertion, that
//! assertion is broken.

use super::super::attention_residual::{self, History};
use super::super::decode::DecodeSession;
use super::super::hyper_connection::Mutation;
use super::super::kv::RowKvState;
use super::super::observe::{
    AttnResBoundaryRecord, AttnResSiteRecord, HcSite, NoopObserver, StepEvent, StepObserver,
};
use super::super::operands::OperandStore;
use super::super::prepared::{ExecutionSlice, PreparedOperands};
use super::super::reference::ReferenceBackend;
use super::attn_res_substrate::{
    close, max_abs_diff, prepare, substrate, Oracle, Substrate, BLOCK, HIDDEN, LAYERS, MAX_PROB,
    MIN_PROB, NORM_EPS, POSITIONS,
};

// ── The witness ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct SiteRow {
    layer: usize,
    site: HcSite,
    candidate_count: usize,
    snapshot_count_before: usize,
    probs: Vec<f32>,
    mixed: Vec<f32>,
    prefix_before: Vec<f32>,
    prefix_after: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
struct BoundaryRow {
    layer: usize,
    snapshots_before: usize,
    snapshots_after: usize,
    value: Vec<f32>,
    entering_prefix: Vec<f32>,
}

/// One entry per observation, IN EMISSION ORDER — the ordering claim of
/// the topology is a claim about this sequence.
#[derive(Debug, Clone, PartialEq)]
enum Event {
    Site(SiteRow),
    Boundary(BoundaryRow),
}

#[derive(Default)]
struct Witness {
    events: Vec<Event>,
}

impl StepObserver for Witness {
    fn event(&mut self, _event: StepEvent) {}

    fn attention_residual_site(&mut self, r: AttnResSiteRecord<'_>) {
        self.events.push(Event::Site(SiteRow {
            layer: r.layer,
            site: r.site,
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
            snapshots_before: r.snapshots_before,
            snapshots_after: r.snapshots_after,
            value: r.value.to_vec(),
            entering_prefix: r.entering_prefix.to_vec(),
        }));
    }
}

impl Witness {
    fn sites(&self) -> Vec<&SiteRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Site(s) => Some(s),
                Event::Boundary(_) => None,
            })
            .collect()
    }

    fn boundaries(&self) -> Vec<&BoundaryRow> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::Boundary(b) => Some(b),
                Event::Site(_) => None,
            })
            .collect()
    }

    fn site(&self, layer: usize, site: HcSite) -> Option<&SiteRow> {
        self.sites()
            .into_iter()
            .find(|s| s.layer == layer && s.site == site)
    }
}

struct Run {
    witness: Witness,
    exit: Vec<f32>,
}

/// One decode step over the substrate under `mutation`, with everything
/// it observed.
fn run(sub: &Substrate, mutation: Mutation) -> Run {
    let (_store, ops) = prepare(sub);
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = Witness::default();
    let step = session
        .step_mutated(1, &mut witness, mutation)
        .expect("the decode step runs the topology");
    Run {
        witness,
        exit: step.exit.expect("a whole-stack image reduces at the exit"),
    }
}

// ── A1: the foreign comparison ──────────────────────────────────────

/// **The arithmetic, against a reference this build did not write.**
///
/// Every site the oracle recorded is replayed from the oracle's OWN
/// entering state — the layer's prefix and the snapshot set
/// reconstructed from its recorded boundary events — and the
/// probabilities and mixed vector are compared. Three positions, both
/// sites, seven layers, plus the exit.
///
/// This is what makes the module a transcription rather than an
/// invention: nothing here is compared against another Rust function.
#[test]
fn every_site_reproduces_the_oracles_probabilities_and_mixed_vector() {
    let oracle = Oracle::load();
    let mut checked = 0;
    for layer in 0..LAYERS {
        for (site, key) in [
            (HcSite::Attention, "attention_site"),
            (HcSite::Ffn, "mlp_site"),
        ] {
            if !oracle.ran(&format!("/witness/{layer}/{key}/ran")) {
                continue;
            }
            let candidates = oracle.count(&format!("/witness/{layer}/{key}/candidate_count"));
            let (norm, proj) = oracle.site_pair(layer, site);
            for position in 0..POSITIONS {
                // The entering state, from the oracle. The attention site
                // enters on the layer's own prefix; the mlp site enters on
                // the post-attention prefix the oracle recorded for it.
                let (prefix, snapshots) = match site {
                    HcSite::Attention => (
                        oracle.row(&format!("/witness/{layer}/prefix_in"), position, HIDDEN),
                        oracle.snapshots_before(layer, position),
                    ),
                    HcSite::Ffn => (
                        oracle.row(
                            &format!("/witness/{layer}/{key}/prefix_in"),
                            position,
                            HIDDEN,
                        ),
                        oracle.snapshots_through(layer, position),
                    ),
                };
                let mut history = History::new(prefix);
                for snapshot in snapshots {
                    history.push_snapshot(snapshot);
                }
                assert_eq!(
                    history.candidate_count(),
                    candidates,
                    "layer {layer} {key} position {position}: reconstructed candidate count"
                );
                let reduction = attention_residual::reduce(
                    &history,
                    attention_residual::SitePair {
                        norm: &norm,
                        proj: &proj,
                    },
                    NORM_EPS,
                    Mutation::None,
                )
                .expect("the reduction runs");
                close(
                    &reduction.probs,
                    &oracle.row(
                        &format!("/witness/{layer}/{key}/softmax_probs"),
                        position,
                        candidates,
                    ),
                    &format!("layer {layer} {key} position {position} probs"),
                );
                close(
                    &reduction.mixed,
                    &oracle.row(
                        &format!("/witness/{layer}/{key}/mixed_vector"),
                        position,
                        HIDDEN,
                    ),
                    &format!("layer {layer} {key} position {position} mixed"),
                );
                checked += 1;
            }
        }
    }
    // 13 reducing sites x 3 positions: layer 0 contributes its mlp site
    // alone, every other layer both.
    assert_eq!(checked, 13 * POSITIONS, "sites compared");

    // The exit, the same way.
    let (norm, proj) = oracle.exit_pair();
    let candidates = oracle.count("/exit/candidate_count");
    for position in 0..POSITIONS {
        let mut history = History::new(oracle.row("/exit/prefix_in", position, HIDDEN));
        for snapshot in oracle.snapshots_through(LAYERS - 1, position) {
            history.push_snapshot(snapshot);
        }
        assert_eq!(history.candidate_count(), candidates);
        let reduction = attention_residual::reduce(
            &history,
            attention_residual::SitePair {
                norm: &norm,
                proj: &proj,
            },
            NORM_EPS,
            Mutation::None,
        )
        .unwrap();
        close(
            &reduction.probs,
            &oracle.row("/exit/softmax_probs", position, candidates),
            "exit probs",
        );
        close(
            &reduction.mixed,
            &oracle.row("/exit/mixed_vector", position, HIDDEN),
            "exit mixed",
        );
    }
}

// ── A4: the schedule, structurally ──────────────────────────────────

/// **The schedule the reference spells, read off a real decode run.**
///
/// Every assertion here is about which records exist and what they
/// count, in emission order — the plane on which the two zero-delta
/// properties live.
#[test]
fn the_decode_traversal_emits_the_oracles_schedule() {
    let sub = substrate();
    let run = run(&sub, Mutation::None);
    let oracle = Oracle::load();

    // Layer 0 emits NO attention-site record. Not a record over one
    // candidate — none at all, because the reference's guard finds an
    // empty snapshot set and does not reduce.
    assert!(
        run.witness.site(0, HcSite::Attention).is_none(),
        "layer 0 must emit no attention-site record: {:?}",
        run.witness.sites()
    );
    // ...and every later layer does.
    for layer in 1..LAYERS {
        assert!(
            run.witness.site(layer, HcSite::Attention).is_some(),
            "layer {layer} attention site"
        );
    }
    // The mlp site is unconditional — every layer, layer 0 included.
    for layer in 0..LAYERS {
        assert!(
            run.witness.site(layer, HcSite::Ffn).is_some(),
            "layer {layer} mlp site"
        );
    }

    // Layer 0's mlp site sees TWO candidates: the snapshot the boundary
    // event has already taken, and the prefix. Never one — the oracle
    // falsified that reading of the reference, and this is the assertion
    // that keeps it falsified.
    assert_eq!(
        run.witness.site(0, HcSite::Ffn).unwrap().candidate_count,
        2,
        "layer 0's mlp site mixes the snapshot and the prefix"
    );

    // Every candidate count, against the oracle's own schedule.
    for layer in 0..LAYERS {
        for (site, key) in [
            (HcSite::Attention, "attention_site"),
            (HcSite::Ffn, "mlp_site"),
        ] {
            let expected = oracle.count(&format!("/witness/{layer}/{key}/candidate_count"));
            match run.witness.site(layer, site) {
                Some(row) => assert_eq!(
                    row.candidate_count, expected,
                    "layer {layer} {key} candidate count"
                ),
                None => assert_eq!(expected, 0, "layer {layer} {key} was expected to reduce"),
            }
        }
    }

    // The ordering claim, per boundary: the ATTENTION site read the set
    // the event had not yet extended, and the MLP site read the one it
    // had.
    let boundaries = run.witness.boundaries();
    assert_eq!(
        boundaries.iter().map(|b| b.layer).collect::<Vec<_>>(),
        vec![0, 3, 6],
        "a boundary at every layer where layer % block == 0"
    );
    for boundary in &boundaries {
        assert_eq!(boundary.snapshots_after, boundary.snapshots_before + 1);
        // The snapshot is the ENTERING prefix state.
        assert_eq!(
            boundary.value, boundary.entering_prefix,
            "layer {} snapshots the entering state",
            boundary.layer
        );
        if let Some(attention) = run.witness.site(boundary.layer, HcSite::Attention) {
            assert_eq!(
                attention.snapshot_count_before, boundary.snapshots_before,
                "layer {} attention reads the OLD set",
                boundary.layer
            );
        }
        let mlp = run.witness.site(boundary.layer, HcSite::Ffn).unwrap();
        assert_eq!(
            mlp.snapshot_count_before, boundary.snapshots_after,
            "layer {} mlp reads the EXTENDED set",
            boundary.layer
        );
    }

    // Layer 3 spelled out, because it is the case the whole ordering
    // claim rests on and a count table can hide.
    let l3_attn = run.witness.site(3, HcSite::Attention).unwrap();
    assert_eq!(
        (l3_attn.snapshot_count_before, l3_attn.candidate_count),
        (1, 2)
    );
    let l3_boundary = boundaries.iter().find(|b| b.layer == 3).unwrap();
    assert_eq!(
        (l3_boundary.snapshots_before, l3_boundary.snapshots_after),
        (1, 2)
    );
    let l3_mlp = run.witness.site(3, HcSite::Ffn).unwrap();
    assert_eq!(
        (l3_mlp.snapshot_count_before, l3_mlp.candidate_count),
        (2, 3)
    );

    // A2: the write is an ADD, except where a boundary reset the prefix
    // — there the attention branch's output BECOMES the prefix.
    for row in run.witness.sites() {
        if row.layer >= LAYERS {
            continue; // the exit's pseudo-record
        }
        let boundary_reset = row.site == HcSite::Attention
            && attention_residual::is_block_boundary(row.layer, BLOCK);
        if !boundary_reset {
            let delta: Vec<f32> = row
                .prefix_after
                .iter()
                .zip(&row.prefix_before)
                .map(|(a, b)| a - b)
                .collect();
            assert!(
                delta.iter().any(|d| d.abs() > 1e-6),
                "layer {} {:?}: the branch contributed nothing",
                row.layer,
                row.site
            );
        }
    }

    // A5: the exit ran, over every snapshot plus the prefix.
    let exit = run
        .witness
        .sites()
        .into_iter()
        .find(|s| s.layer == LAYERS)
        .expect("the exit emits a record");
    assert_eq!(exit.candidate_count, oracle.count("/exit/candidate_count"));
    assert_eq!(exit.snapshot_count_before, oracle.count("/exit/snapshots"));
    assert_eq!(
        run.exit, exit.mixed,
        "the step's exit IS the exit reduction"
    );

    // A6: the instrument can see. A substrate whose probabilities have
    // saturated proves nothing, and would report every control below as
    // passing while rejecting none of them.
    for row in run.witness.sites() {
        let max = row.probs.iter().copied().fold(f32::MIN, f32::max);
        let min = row.probs.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            max <= MAX_PROB && min >= MIN_PROB,
            "layer {} {:?}: probabilities {:?} outside the oracle's band — the substrate \
             cannot see the candidate-set controls",
            row.layer,
            row.site,
            row.probs
        );
    }
}

// ── The rejecting controls ──────────────────────────────────────────

/// Every mutation the oracle measured above zero must move this
/// substrate's exit vector too.
#[test]
fn every_value_visible_mutation_is_caught() {
    let sub = substrate();
    let reference = run(&sub, Mutation::None);
    for mutation in [
        Mutation::AttnResSiteOverNewSnapshots,
        Mutation::AttnResSnapshotIsMixedVector,
        Mutation::AttnResSnapshotAfterAttention,
        Mutation::AttnResMlpSiteSkippedAtLayer0,
        Mutation::AttnResMixOverNormalisedCandidates,
        Mutation::AttnResScoreWithoutRmsNorm,
        Mutation::AttnResExitSkipped,
        Mutation::AttnResExitUsesALayerPair,
    ] {
        let mutated = run(&sub, mutation);
        let diff = max_abs_diff(&mutated.exit, &reference.exit);
        assert!(
            diff > 1e-3,
            "{mutation:?} left the exit vector unchanged (max |diff| {diff:e}); the oracle \
             measured this defect above zero, so a substrate that cannot see it is not a witness"
        );
    }
}

/// **The two the oracle proved unreachable by value, caught by shape.**
///
/// Both must leave the exit vector bit-identical — that is the oracle's
/// measurement, reproduced here rather than taken on trust — and both
/// must change the record stream. A run that reported either as caught
/// by a value comparison would have a broken assertion, and this test
/// asserts the zero as firmly as it asserts the structural difference.
#[test]
fn the_two_numerically_inert_mutations_are_caught_by_the_witness_alone() {
    let sub = substrate();
    let reference = run(&sub, Mutation::None);

    // Layer 0's attention site, run instead of skipped. Softmax over one
    // candidate is the identity, so nothing moves.
    let regularised = run(&sub, Mutation::AttnResLayer0AttentionSiteRuns);
    assert_eq!(
        max_abs_diff(&regularised.exit, &reference.exit),
        0.0,
        "the oracle measured this at exactly 0.0; a non-zero here means the traversal \
         changed something else as well"
    );
    let extra = regularised.witness.site(0, HcSite::Attention).expect(
        "the regularised traversal emits a layer-0 attention record where the reference \
         emits none — the only observable difference",
    );
    assert_eq!(
        extra.candidate_count, 1,
        "it reduces over the prefix alone, which is why it is invisible"
    );
    assert!(reference.witness.site(0, HcSite::Attention).is_none());

    // The mlp site given the attention site's guard. The guard never
    // fires, because no site in this schedule sees an empty set.
    let guarded = run(&sub, Mutation::AttnResMlpSiteGuardedOnNonEmpty);
    assert_eq!(
        max_abs_diff(&guarded.exit, &reference.exit),
        0.0,
        "the oracle measured this at exactly 0.0"
    );
    assert_eq!(
        guarded.witness.sites().len(),
        reference.witness.sites().len(),
        "the guard fires nowhere: every mlp site still reduces"
    );
}

// ── What has NOT lifted ─────────────────────────────────────────────

/// **What 2a and 2b refused, the lift now prepares — through the public
/// loader, which is the one every witness in this rung reaches.**
///
/// This test asserted the opposite through both traversal transitions:
/// the public loader refused the topology by name, and the witnesses
/// prepared through a test-only seam so that proving the traversal could
/// not be mistaken for lifting it. At the lift the refusal went and the
/// seam went with it, so `prepare` in `attn_res_substrate` is now this
/// same call — every assertion in this file and in `attn_res_2b_batch`
/// scores the path a real caller takes.
///
/// Kept as its own test rather than left implicit in `prepare` because
/// the claim is about the PUBLIC surface, and a change that reintroduced
/// a refusal there would otherwise surface as thirteen confusing
/// failures instead of one clear one.
#[test]
fn the_public_loader_prepares_what_it_used_to_refuse() {
    let sub = substrate();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    PreparedOperands::load(
        &sub.plan,
        &store,
        &ReferenceBackend::new(),
        ExecutionSlice::Full,
    )
    .expect("the public loader prepares an attention-residual plan");
}

/// **Observation is optional, and the traversal must not depend on being
/// watched.**
///
/// `StepObserver`'s two attention-residual hooks have default no-op
/// bodies, so an observer that ignores them is a supported caller — and
/// the production path is exactly that caller. This runs the same step
/// under `NoopObserver`, which overrides neither, and requires the exit
/// vector to be bit-identical to the observed run's.
///
/// Not coverage padding: the site record borrows the reduction's own
/// `probs` and `mixed`, so a traversal that computed anything inside the
/// observer call — or skipped work when nobody was listening — would
/// diverge here and nowhere else. It is also the only test in this file
/// that executes those default bodies, which is how the gap was found.
#[test]
fn the_traversal_runs_identically_with_an_unobserving_observer() {
    let sub = substrate();
    let observed = run(&sub, Mutation::None);

    let (_store, ops) = prepare(&sub);
    let backend = ReferenceBackend::new();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let unobserved = session
        .step_mutated(1, &mut NoopObserver, Mutation::None)
        .expect("an unobserved step runs the topology")
        .exit
        .expect("a whole-stack image reduces at the exit");

    assert_eq!(
        unobserved, observed.exit,
        "the exit differs when nobody is observing; the traversal is doing work inside its \
         observation points"
    );
}
