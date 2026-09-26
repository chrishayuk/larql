//! V3-OBS-1 witnesses (`docs/v3-obs-1-carrier-observation.md`).
//!
//! P1 — a VALUE-consuming observer leaves the logits bit-identical;
//! P2 — every single-stream write reconstructs `before + delta == after`
//! bit-for-bit through the chain, and a one-ulp perturbation of either
//! vector is caught (the negative control that makes P2 evidence);
//! P3 — every layer emits exactly the writes its bound program declares,
//! on a plain stack, a mixer-only stack, a hyper-connected stack and an
//! attention-residual stack; C5/C6 — the layer scale rides on the FFN
//! write and the chain must apply it; and the batch/decode witness: the
//! decode tap exposes the states the batch traversal already records.
//!
//! The real-container form of the same witnesses, plus the P5 capture
//! cost measurement, lives in `carrier_write_real` and needs
//! `LARQL_V3_CONTAINER`.

use super::attn_res_substrate;
use super::decode::fixture as golden_fixture;
use super::gemma4;
use super::wave19_hc_decode::run_from_tokens;
use super::wave19_hc_substrate::{self, Variant};
use crate::format::vindex3::fixtures::{encode_fixture_container, G_LAYERS, G_TOKENS};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::hyper_connection::Mutation;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    AttnResSiteRecord, CarrierForm, CarrierWriteRecord, NoopObserver, StepEvent, StepObserver,
    SublayerSite,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::ExecutionTrace;
use crate::format::vindex3::opplan::tests::mamba2::miniature_mamba2;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

/// Tokens inside every miniature's vocabulary (the smallest is 7).
const SMALL_TOKENS: [u32; 3] = [3, 1, 5];

/// Which vector of one write the negative control perturbs.
#[derive(Clone, Copy, Debug)]
enum Perturb {
    Delta,
    After,
}

/// One write as the chain saw it, owned for later comparison.
pub(super) struct Write {
    pub(super) layer: usize,
    pub(super) site: SublayerSite,
    pub(super) position: usize,
    pub(super) after: Vec<f32>,
    pub(super) layer_scale: Option<f32>,
}

impl Write {
    /// What the next layer reads: `after`, scaled where the program
    /// scales it.
    fn layer_output(&self) -> Vec<f32> {
        match self.layer_scale {
            Some(scale) => self.after.iter().map(|v| v * scale).collect(),
            None => self.after.clone(),
        }
    }
}

/// P2's witness: chains every single-stream write from the entering
/// carrier and checks each add bit-for-bit; records every structural
/// event for P3.
#[derive(Default)]
pub(super) struct ChainWitness {
    carrier: Option<Vec<f32>>,
    pending_scale: Option<f32>,
    pub(super) events: Vec<StepEvent>,
    pub(super) entering: Vec<Vec<f32>>,
    pub(super) writes: Vec<Write>,
    pub(super) exact: usize,
    pub(super) inexact: Vec<(usize, SublayerSite, usize)>,
    /// The negative control: perturb this (layer, site)'s vector by one
    /// ulp before checking.
    perturb: Option<(usize, SublayerSite, Perturb)>,
    /// The C5 control: a chain that forgets the layer scale.
    ignore_scale: bool,
    /// Attention-residual site records seen, by (layer, site).
    attn_res_records: Vec<(usize, SublayerSite)>,
}

fn one_ulp(v: &mut f32) {
    *v = f32::from_bits(v.to_bits() ^ 1);
}

/// How many ulps of the sum the delta control moves by: enough that the
/// add cannot round it away, few enough to stay a perturbation.
const DELTA_CONTROL_ULPS: f32 = 4.0;

/// `n` ulps at the magnitude of `at` (at least `n` ulps of 1.0, so a
/// sum near zero still moves).
fn sum_ulps(at: f32, n: f32) -> f32 {
    at.abs().max(1.0) * f32::EPSILON * n
}

impl StepObserver for ChainWitness {
    fn event(&mut self, event: StepEvent) {
        self.events.push(event);
    }

    fn entering_carrier(&mut self, _position: usize, values: &[f32]) {
        self.entering.push(values.to_vec());
        self.carrier = Some(values.to_vec());
        self.pending_scale = None;
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        let mut before = self
            .carrier
            .take()
            .expect("a write follows the entering carrier or a previous write");
        if let (Some(scale), false) = (self.pending_scale.take(), self.ignore_scale) {
            for v in &mut before {
                *v *= scale;
            }
        }
        let mut delta = record.delta.to_vec();
        let mut after = record.after.to_vec();
        if let Some((layer, site, which)) = self.perturb {
            if layer == record.layer && site == record.site {
                match which {
                    // One ulp of the DELTA can vanish in the add's
                    // rounding when the delta is small against the
                    // carrier, so the delta control perturbs at the
                    // resolution of the SUM — the smallest change the
                    // witness could possibly be asked to see.
                    Perturb::Delta => delta[0] += sum_ulps(after[0], DELTA_CONTROL_ULPS),
                    Perturb::After => one_ulp(&mut after[0]),
                }
            }
        }
        let exact = before.len() == delta.len()
            && before.len() == after.len()
            && before
                .iter()
                .zip(&delta)
                .zip(&after)
                .all(|((b, d), a)| (b + d).to_bits() == a.to_bits());
        if exact {
            self.exact += 1;
        } else {
            self.inexact
                .push((record.layer, record.site, record.position));
        }
        self.writes.push(Write {
            layer: record.layer,
            site: record.site,
            position: record.position,
            after: record.after.to_vec(),
            layer_scale: record.layer_scale,
        });
        // The chain continues from what the executor actually holds,
        // so a perturbed check never contaminates the next link.
        self.carrier = Some(record.after.to_vec());
        self.pending_scale = record.layer_scale;
    }

    fn attention_residual_site(&mut self, record: AttnResSiteRecord<'_>) {
        self.attn_res_records.push((record.layer, record.site));
    }
}

impl ChainWitness {
    pub(super) fn writes_of(&self, form: CarrierForm) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, StepEvent::CarrierWrite { carrier, .. } if *carrier == form))
            .count()
    }

    pub(super) fn writes_at(&self, site: SublayerSite) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, StepEvent::CarrierWrite { site: s, .. } if *s == site))
            .count()
    }

    pub(super) fn boundaries(&self, ffn: bool) -> usize {
        self.events
            .iter()
            .filter(|e| match e {
                StepEvent::FfnDone { .. } => ffn,
                StepEvent::AttentionDone { .. } => !ffn,
                _ => false,
            })
            .count()
    }
}

/// P3's expectation, derived from the bound plan BEFORE the run: one
/// write per layer, plus one where the layer carries an FFN program.
pub(super) fn writes_declared_by(plan: &ComponentOpPlan) -> usize {
    plan.layers
        .iter()
        .map(|layer| 1 + usize::from(layer.ffn.is_some()))
        .sum()
}

pub(super) fn ffn_layers(plan: &ComponentOpPlan) -> usize {
    plan.layers.iter().filter(|l| l.ffn.is_some()).count()
}

/// Step every token through a fresh session with `witness`, returning
/// the per-position logits.
pub(super) fn observed<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
    witness: &mut dyn StepObserver,
) -> Vec<Vec<f32>> {
    let mut session = DecodeSession::new(
        plan,
        store,
        backend,
        Box::new(crate::format::vindex3::opplan::exec::kv::RowKvState::default()),
    )
    .unwrap();
    tokens
        .iter()
        .map(|&t| {
            session
                .step_observed(t, witness)
                .unwrap()
                .logits
                .expect("the plan carries an output head")
        })
        .collect()
}

pub(super) fn unobserved<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    observed(plan, store, backend, tokens, &mut NoopObserver)
}

// ── P1 ──────────────────────────────────────────────────────────────

#[test]
fn p1_a_value_consuming_observer_leaves_every_position_bit_identical() {
    let (_c, plan, store) = golden_fixture();
    let reference = ReferenceBackend::new();
    let production = ProductionBackend::new();
    for (name, plain, observed_logits) in [
        (
            "reference",
            unobserved(&plan, &store, &reference, &G_TOKENS),
            observed(
                &plan,
                &store,
                &reference,
                &G_TOKENS,
                &mut ChainWitness::default(),
            ),
        ),
        (
            "production",
            unobserved(&plan, &store, &production, &G_TOKENS),
            observed(
                &plan,
                &store,
                &production,
                &G_TOKENS,
                &mut ChainWitness::default(),
            ),
        ),
    ] {
        assert_eq!(
            plain, observed_logits,
            "{name}: observation changed the arithmetic"
        );
    }
}

// ── P2 ──────────────────────────────────────────────────────────────

#[test]
fn p2_every_single_stream_write_reconstructs_bit_for_bit_through_the_chain() {
    fn check<B: PlanBackend>(backend: &B) {
        let (_c, plan, store) = golden_fixture();
        let mut witness = ChainWitness::default();
        observed(&plan, &store, backend, &G_TOKENS, &mut witness);
        let expected = writes_declared_by(&plan) * G_TOKENS.len();
        assert_eq!(
            witness.entering.len(),
            G_TOKENS.len(),
            "one entering carrier per step"
        );
        assert_eq!(
            witness.writes.len(),
            expected,
            "every declared write was observed"
        );
        assert_eq!(
            witness.exact, expected,
            "inexact writes: {:?}",
            witness.inexact
        );
        assert!(witness.inexact.is_empty());
    }
    check(&ReferenceBackend::new());
    check(&ProductionBackend::new());
}

#[test]
fn p2_negative_control_one_ulp_in_the_delta_is_caught_at_exactly_that_write() {
    let (_c, plan, store) = golden_fixture();
    let mut witness = ChainWitness {
        perturb: Some((1, SublayerSite::Ffn, Perturb::Delta)),
        ..ChainWitness::default()
    };
    observed(
        &plan,
        &store,
        &ReferenceBackend::new(),
        &G_TOKENS,
        &mut witness,
    );
    let expected_inexact: Vec<(usize, SublayerSite, usize)> = (0..G_TOKENS.len())
        .map(|position| (1, SublayerSite::Ffn, position))
        .collect();
    assert_eq!(witness.inexact, expected_inexact);
    assert_eq!(
        witness.exact + witness.inexact.len(),
        writes_declared_by(&plan) * G_TOKENS.len()
    );
}

#[test]
fn p2_negative_control_one_ulp_in_the_after_is_caught_at_exactly_that_write() {
    let (_c, plan, store) = golden_fixture();
    let mut witness = ChainWitness {
        perturb: Some((0, SublayerSite::Attention, Perturb::After)),
        ..ChainWitness::default()
    };
    observed(
        &plan,
        &store,
        &ReferenceBackend::new(),
        &G_TOKENS,
        &mut witness,
    );
    let expected_inexact: Vec<(usize, SublayerSite, usize)> = (0..G_TOKENS.len())
        .map(|position| (0, SublayerSite::Attention, position))
        .collect();
    assert_eq!(witness.inexact, expected_inexact);
}

// ── P3 ──────────────────────────────────────────────────────────────

#[test]
fn p3_a_plain_stack_writes_twice_per_layer_and_only_single_stream() {
    let (_c, plan, store) = golden_fixture();
    let mut witness = ChainWitness::default();
    observed(
        &plan,
        &store,
        &ReferenceBackend::new(),
        &G_TOKENS,
        &mut witness,
    );
    let steps = G_TOKENS.len();
    assert_eq!(writes_declared_by(&plan), 2 * G_LAYERS);
    assert_eq!(witness.writes_of(CarrierForm::Single), 2 * G_LAYERS * steps);
    assert_eq!(witness.writes_of(CarrierForm::Bundle), 0);
    assert_eq!(witness.writes_of(CarrierForm::History), 0);
    assert_eq!(witness.writes_at(SublayerSite::Attention), G_LAYERS * steps);
    assert_eq!(witness.writes_at(SublayerSite::Ffn), G_LAYERS * steps);
    assert_eq!(witness.boundaries(true), G_LAYERS * steps);
}

fn mixer_only_fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        |dir| miniature_mamba2(dir, None),
        checkpoint.path(),
        container.path(),
        "mini-mamba2",
    );
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, outcome.plan.unwrap(), store)
}

/// A mixer-only stack writes ONCE per layer — there is no FFN write to
/// fabricate — while `FfnDone` still closes every layer as the layer
/// boundary. The two counts differ, and that difference is the
/// property.
#[test]
fn p3_a_mixer_only_stack_writes_once_per_layer_and_still_closes_every_layer() {
    let (_c, plan, store) = mixer_only_fixture();
    assert_eq!(ffn_layers(&plan), 0, "the fixture has no FFN program");
    let layers = plan.layers.len();
    let expected = writes_declared_by(&plan);
    assert_eq!(expected, layers);
    let mut witness = ChainWitness::default();
    let plain = unobserved(&plan, &store, &ReferenceBackend::new(), &SMALL_TOKENS);
    let seen = observed(
        &plan,
        &store,
        &ReferenceBackend::new(),
        &SMALL_TOKENS,
        &mut witness,
    );
    assert_eq!(plain, seen, "P1 holds on the mixer-only stack too");
    let steps = SMALL_TOKENS.len();
    assert_eq!(witness.writes_of(CarrierForm::Single), expected * steps);
    assert_eq!(
        witness.writes_at(SublayerSite::Ffn),
        0,
        "no FFN write was fabricated"
    );
    assert_eq!(witness.writes_at(SublayerSite::Attention), layers * steps);
    assert_eq!(
        witness.boundaries(true),
        layers * steps,
        "FfnDone is the boundary"
    );
    assert_eq!(witness.boundaries(false), layers * steps);
    // P2 holds across a mixer-only chain: consecutive attention-site
    // writes with nothing between them.
    assert_eq!(
        witness.exact,
        expected * steps,
        "inexact: {:?}",
        witness.inexact
    );
}

/// A hyper-connected stack writes BUNDLES: every write carries the
/// form, every write has its site record, and no single-stream record
/// is ever produced.
#[test]
fn p3_a_hyper_connected_stack_writes_bundles_with_a_record_per_write() {
    let sub = wave19_hc_substrate::build(Variant::HeadBearing);
    let expected = writes_declared_by(&sub.plan);
    assert_eq!(expected, 2 * wave19_hc_substrate::LAYERS);
    let run = run_from_tokens(&sub, &SMALL_TOKENS, Mutation::None);
    let bundle_writes = run
        .witness
        .events
        .iter()
        .filter(|e| {
            matches!(
                e,
                StepEvent::CarrierWrite {
                    carrier: CarrierForm::Bundle,
                    ..
                }
            )
        })
        .count();
    assert_eq!(bundle_writes, expected * SMALL_TOKENS.len());
    assert!(!run.witness.events.iter().any(|e| matches!(
        e,
        StepEvent::CarrierWrite {
            carrier: CarrierForm::Single | CarrierForm::History,
            ..
        }
    )));
    assert_eq!(
        run.witness.records.len(),
        bundle_writes,
        "every bundle write delivered its site record"
    );
}

/// An attention-residual stack writes HISTORY at every site, including
/// layer 0's attention site — where the reference does not reduce and
/// no site record exists. The write event is the observation there.
#[test]
fn p3_an_attention_residual_stack_writes_history_and_layer_zero_writes_without_a_record() {
    let sub = attn_res_substrate::substrate();
    let expected = writes_declared_by(&sub.plan);
    assert_eq!(expected, 2 * attn_res_substrate::LAYERS);
    let backend = ReferenceBackend::new();
    let store = OperandStore::open(sub.container.path(), &sub.inspection).unwrap();
    let ops = PreparedOperands::load(&sub.plan, &store, &backend, ExecutionSlice::Full).unwrap();
    let mut kv = RowKvState::default();
    let mut session = DecodeSession::over_prepared(&sub.plan, &ops, &backend, &mut kv).unwrap();
    let mut witness = ChainWitness::default();
    session.step_observed(1, &mut witness).unwrap();
    assert_eq!(witness.writes_of(CarrierForm::History), expected);
    assert_eq!(witness.writes_of(CarrierForm::Single), 0);
    assert_eq!(witness.writes_of(CarrierForm::Bundle), 0);
    assert!(
        witness.writes.is_empty(),
        "no single-stream value record on a history carrier"
    );
    assert!(witness.events.iter().any(|e| matches!(
        e,
        StepEvent::CarrierWrite {
            layer: 0,
            site: SublayerSite::Attention,
            carrier: CarrierForm::History
        }
    )));
    let layer_sites: Vec<_> = witness
        .attn_res_records
        .iter()
        .filter(|(layer, _)| *layer < attn_res_substrate::LAYERS)
        .collect();
    assert_eq!(
        layer_sites.len(),
        expected - 1,
        "one site per write except layer 0's attention"
    );
    assert!(!layer_sites.contains(&&(0, SublayerSite::Attention)));
}

// ── C5 / C6 ─────────────────────────────────────────────────────────

fn gemma4_fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    gemma4::miniature_gemma4(dir.path(), None);
    let container = gemma4::encoded(dir.path());
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = gemma4::closure(container.path());
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, outcome.plan.unwrap(), store)
}

/// On a component with a layer scalar, the FFN write carries the scale,
/// the attention write does not, and the chain reconstructs only when
/// it applies the scale between layers.
#[test]
fn c5_the_layer_scale_rides_on_the_ffn_write_and_the_chain_must_apply_it() {
    let (_c, plan, store) = gemma4_fixture();
    assert!(plan.layers.iter().all(|l| l.layer_scale.is_some()));
    let backend = ReferenceBackend::new();
    let mut witness = ChainWitness::default();
    observed(&plan, &store, &backend, &SMALL_TOKENS, &mut witness);
    let expected = writes_declared_by(&plan) * SMALL_TOKENS.len();
    assert_eq!(witness.writes.len(), expected);
    for write in &witness.writes {
        match write.site {
            SublayerSite::Ffn => assert!(write.layer_scale.is_some(), "layer {}", write.layer),
            SublayerSite::Attention => assert!(write.layer_scale.is_none()),
        }
    }
    assert_eq!(witness.exact, expected, "inexact: {:?}", witness.inexact);

    // The control: a chain that forgets the scale must fail at the NEXT
    // layer's attention write wherever the forgotten scale was not 1.0
    // — derived from the scales the records carried, not assumed to be
    // every layer (the miniature's last scalar happens to be exactly 1).
    let scaling_layers: Vec<usize> = witness
        .writes
        .iter()
        .filter(|w| {
            w.site == SublayerSite::Ffn
                && w.position == 0
                && w.layer + 1 < plan.layers.len()
                && w.layer_scale != Some(1.0)
        })
        .map(|w| w.layer)
        .collect();
    assert!(
        !scaling_layers.is_empty(),
        "the control needs a scale that does something"
    );
    let mut forgetful = ChainWitness {
        ignore_scale: true,
        ..ChainWitness::default()
    };
    observed(&plan, &store, &backend, &SMALL_TOKENS, &mut forgetful);
    let mut expected_inexact: Vec<(usize, SublayerSite, usize)> = (0..SMALL_TOKENS.len())
        .flat_map(|position| {
            scaling_layers
                .iter()
                .map(move |layer| (layer + 1, SublayerSite::Attention, position))
        })
        .collect();
    expected_inexact.sort_unstable();
    let mut actual = forgetful.inexact.clone();
    actual.sort_unstable();
    assert_eq!(actual, expected_inexact);
}

// ── Batch/decode ────────────────────────────────────────────────────

/// Compare every chained state to the batch traversal's planes.
pub(super) fn assert_batch_matches_chain(
    trace: &ExecutionTrace,
    witness: &ChainWitness,
    tokens: &[u32],
) {
    assert_eq!(trace.embedded.rows().len(), tokens.len());
    for (position, entering) in witness.entering.iter().enumerate() {
        assert_eq!(
            &trace.embedded.rows()[position],
            entering,
            "position {position}: entering carrier vs batch embedded plane"
        );
    }
    for write in &witness.writes {
        let plane = match write.site {
            SublayerSite::Attention => trace.layers[write.layer].post_attention.rows(),
            SublayerSite::Ffn => trace.layers[write.layer].post_layer.rows(),
        };
        let expected = match write.site {
            SublayerSite::Attention => write.after.clone(),
            SublayerSite::Ffn => write.layer_output(),
        };
        let batch = &plane[write.position];
        let bit_identical = batch.len() == expected.len()
            && batch
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            bit_identical,
            "layer {} {:?} position {}: decode tap vs batch plane differ",
            write.layer, write.site, write.position
        );
    }
}

#[test]
fn batch_and_decode_agree_on_every_carrier_state_of_the_plain_stack() {
    fn check<B: PlanBackend>(backend: &B) {
        let (_c, plan, store) = golden_fixture();
        let mut witness = ChainWitness::default();
        observed(&plan, &store, backend, &G_TOKENS, &mut witness);
        let trace = execute_plan(&plan, &store, &G_TOKENS, backend).unwrap();
        assert_batch_matches_chain(&trace, &witness, &G_TOKENS);
    }
    check(&ReferenceBackend::new());
    check(&ProductionBackend::new());
}

/// The batch path captures `post_layer` AFTER the layer scale; the
/// decode record carries the pre-scale `after` plus the scale. They
/// agree only through `layer_output`, which is the whole of C5.
#[test]
fn batch_and_decode_agree_through_the_layer_scale_on_the_gemma4_stack() {
    let (_c, plan, store) = gemma4_fixture();
    let backend = ReferenceBackend::new();
    let mut witness = ChainWitness::default();
    observed(&plan, &store, &backend, &SMALL_TOKENS, &mut witness);
    let trace = execute_plan(&plan, &store, &SMALL_TOKENS, &backend).unwrap();
    assert_batch_matches_chain(&trace, &witness, &SMALL_TOKENS);
    // And the pre-scale `after` is NOT what the batch plane holds.
    let first_ffn = witness
        .writes
        .iter()
        .find(|w| w.site == SublayerSite::Ffn)
        .unwrap();
    assert_ne!(
        trace.layers[first_ffn.layer].post_layer.rows()[first_ffn.position],
        first_ffn.after,
        "the batch plane is post-scale; the record's `after` is pre-scale"
    );
}

#[test]
fn recorded_carrier_readout_matches_canonical_exit_and_refuses_bad_inputs() {
    fn check<B: PlanBackend>(backend: &B) {
        let (_c, mut plan, store) = golden_fixture();
        plan.output.as_mut().unwrap().multiplier = Some(0.37);
        plan.output.as_mut().unwrap().softcapping = Some(1.5);
        let ops = PreparedOperands::load(&plan, &store, backend, ExecutionSlice::Full).unwrap();
        let mut kv = RowKvState::default();
        let mut session = DecodeSession::over_prepared(&plan, &ops, backend, &mut kv).unwrap();
        let mut witness = ChainWitness::default();
        for token in G_TOKENS {
            let expected = session
                .step_observed(token, &mut witness)
                .unwrap()
                .logits
                .unwrap();
            let exit = witness.writes.last().unwrap().layer_output();
            let actual = ops.readout_carrier(backend, &exit).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        let shard = PreparedOperands::load(
            &plan,
            &store,
            backend,
            ExecutionSlice::LayerRange { start: 0, end: 1 },
        )
        .unwrap();
        assert!(shard
            .readout_carrier(backend, &vec![0.; ops.hidden()])
            .is_err());
        assert!(ops.readout_carrier(backend, &[]).is_err());
        assert!(ops
            .readout_carrier(backend, &vec![f32::NAN; ops.hidden()])
            .is_err());
    }
    check(&ReferenceBackend::new());
    check(&ProductionBackend::new());
}
