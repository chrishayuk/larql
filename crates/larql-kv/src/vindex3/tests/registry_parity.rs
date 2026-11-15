//! **C3 of CONTINUATION-PLUGIN-1: the built-ins behind the registry are
//! the built-ins.**
//!
//! The forecast (`docs/represent/forecasts/continuation-plugin-1.json`)
//! extends the KV-1 gate to
//!
//! ```text
//! direct CanonicalKvState == registry canonical/v1 == registry row/v1
//! ```
//!
//! bit-identical through prefill, resume and decode, on a fixture crossing
//! a sliding-window boundary and on a hybrid fixture that keeps recurrent
//! AND latent state. Each arm is driven through the same harness and
//! leaves one [`Record`] of everything observable — logits, generated
//! ids, position, K/V rows, recurrent buffers, latent rows — and the
//! records must be equal. The direct arm is the original KV-1 reference,
//! so the registry arms are measured against the pre-registry behaviour.

use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_TOKENS,
};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LatentKvRows, LayerContinuationGeometry, RecurrentState,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, KvState, LayerKvGeometry, RowKvState,
};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

use super::super::{shipped_continuations, CanonicalKvState};

const STEPS: usize = 6;

/// One provider's complete observable state.
#[derive(Debug, PartialEq)]
pub(super) struct Snapshot {
    position: usize,
    keys: Vec<Vec<Vec<f32>>>,
    values: Vec<Vec<Vec<f32>>>,
    recurrent: Vec<Vec<Vec<f32>>>,
    latent: Vec<Vec<Vec<f32>>>,
}

impl Snapshot {
    /// Whether any recurrent cell left its zero start — without it, an
    /// equality over recurrent state is vacuous.
    pub(super) fn recurrent_moved(&self) -> bool {
        self.recurrent.iter().flatten().flatten().any(|v| *v != 0.0)
    }
}

/// Everything one arm produced.
#[derive(Debug, PartialEq)]
struct Record {
    prefill_logits: Vec<f32>,
    after_prefill: Snapshot,
    step_logits: Vec<Vec<f32>>,
    ids: Vec<u32>,
    last: Snapshot,
}

pub(super) fn open(
    model: fn(&std::path::Path),
    name: &str,
) -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(model, checkpoint.path(), container.path(), name);
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "{name} must close: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

pub(super) fn snapshot(
    state: &mut dyn KvState,
    geometry: &[LayerContinuationGeometry],
) -> Snapshot {
    let mut snap = Snapshot {
        position: state.position(),
        keys: Vec::new(),
        values: Vec::new(),
        recurrent: Vec::new(),
        latent: Vec::new(),
    };
    for (layer, g) in geometry.iter().enumerate() {
        if g.kv_side().is_some() {
            snap.keys.push(state.rows(layer).to_owned_rows().0.to_vec());
            snap.values
                .push(state.rows(layer).to_owned_rows().1.to_vec());
        }
        if let Some(r) = g.recurrent() {
            let held = state.recurrent_state(layer).unwrap();
            snap.recurrent.push(
                (0..r.buffers.len())
                    .map(|i| held.buffer(i).cells().to_vec())
                    .collect(),
            );
        }
        if g.latent_kv().is_some() {
            snap.latent
                .push(state.latent_state(layer).unwrap().rows().to_vec());
        }
    }
    snap
}

pub(super) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0 as u32
}

/// Prefill `prompt`, resume, and decode greedily for [`STEPS`], recording
/// everything the provider holds at both boundaries.
fn drive(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    prompt: &[u32],
    state: &mut dyn KvState,
) -> Record {
    let geometry = plan_continuation_geometry(plan).unwrap();
    let backend = ReferenceBackend::new();
    let prefill = prefill_plan(plan, store, prompt, &backend, state).unwrap();
    let prefill_logits = prefill.logits.unwrap();
    let after_prefill = snapshot(state, &geometry);
    let mut ids = vec![argmax(&prefill_logits)];
    let mut step_logits = Vec::new();
    {
        let mut session = DecodeSession::with_kv_state(plan, store, &backend, state).unwrap();
        for _ in 0..STEPS {
            let logits = session.step(*ids.last().unwrap()).unwrap().logits.unwrap();
            ids.push(argmax(&logits));
            step_logits.push(logits);
        }
    }
    let last = snapshot(state, &geometry);
    Record {
        prefill_logits,
        after_prefill,
        step_logits,
        ids,
        last,
    }
}

/// A fresh provider for `identity`, selected against `plan` from the
/// shipped registry — the path production takes.
fn registry_arm(
    plan: &ComponentOpPlan,
    identity: &ContinuationIdentity,
) -> Box<dyn KvState + Send> {
    let geometry = plan_continuation_geometry(plan).unwrap();
    shipped_continuations()
        .select(identity, &ContinuationConfig::empty(), &geometry)
        .unwrap()
        .build()
}

/// The three arms, each asserted equal to the direct canonical reference.
fn assert_arms_identical(plan: &ComponentOpPlan, store: &OperandStore, prompt: &[u32]) -> Record {
    let reference = drive(plan, store, prompt, &mut CanonicalKvState::new());
    let arms: [(&str, Box<dyn KvState + Send>); 3] = [
        (
            "registry canonical/v1",
            registry_arm(plan, &CanonicalKvState::identity()),
        ),
        (
            "registry row/v1",
            registry_arm(plan, &RowKvState::identity()),
        ),
        ("direct RowKvState", Box::new(RowKvState::default())),
    ];
    for (name, mut state) in arms {
        let record = drive(plan, store, prompt, &mut *state);
        assert!(
            record == reference,
            "{name} diverges from direct CanonicalKvState"
        );
    }
    reference
}

#[test]
fn registry_built_ins_match_direct_canonical_across_a_sliding_window() {
    let (_c, plan, store) = open(miniature_glimmer, "c3-sliding");
    let record = assert_arms_identical(&plan, &store, &G_TOKENS);
    // The fixture must actually cross its window, or the gate proves
    // nothing about sliding layers.
    let window = plan_continuation_geometry(&plan)
        .unwrap()
        .iter()
        .filter_map(|g| g.kv().and_then(|kv| kv.window))
        .min()
        .expect("a sliding layer");
    assert!(
        record.last.position > window,
        "{} ≤ window {window}",
        record.last.position
    );
}

#[test]
fn registry_built_ins_match_direct_canonical_on_a_recurrent_and_latent_hybrid() {
    let (_c, plan, store) = open(hybrid_kda_mla_f32_model, "c3-hybrid");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    // Both non-KV regions must be present, or this is not the hybrid the
    // forecast names.
    assert!(
        geometry.iter().any(|g| g.recurrent().is_some()),
        "no recurrent layer"
    );
    assert!(
        geometry.iter().any(|g| g.latent_kv().is_some()),
        "no latent layer"
    );
    let record = assert_arms_identical(&plan, &store, &[1, 0, 2]);
    assert!(!record.last.recurrent.is_empty() && !record.last.latent.is_empty());
    assert!(
        record
            .last
            .recurrent
            .iter()
            .flatten()
            .flatten()
            .any(|v| *v != 0.0),
        "the recurrence never moved, so equality would be vacuous"
    );
}

/// A row provider that moves ONE value by one ULP — the first K cell it is
/// handed, or the largest cell of the first recurrent buffer it serves once
/// the sequence is live — and is otherwise row/v1.
struct Perturbed {
    inner: RowKvState,
    moved: bool,
}

fn nudge(x: &mut f32) {
    *x = f32::from_bits(x.to_bits() + 1);
}

impl KvState for Perturbed {
    fn prepare(&mut self, layers: &[LayerKvGeometry]) {
        self.inner.prepare(layers)
    }
    fn append(&mut self, layer: usize, mut key: Vec<f32>, value: Vec<f32>) {
        if !self.moved {
            nudge(&mut key[0]);
            self.moved = true;
        }
        self.inner.append(layer, key, value)
    }
    fn rows(
        &self,
        layer: usize,
    ) -> larql_vindex::format::vindex3::opplan::exec::kv_view::KvView<'_> {
        self.inner.rows(layer)
    }
    fn prepare_layer(&mut self, layer: usize) {
        self.inner.prepare_layer(layer)
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
    fn set_position(&mut self, position: usize) {
        self.inner.set_position(position)
    }
    fn invalidate(&mut self, reason: &str) {
        self.inner.invalidate(reason)
    }
    fn invalidated(&self) -> Option<&str> {
        self.inner.invalidated()
    }
    fn prepare_continuation(
        &mut self,
        layers: &[LayerContinuationGeometry],
    ) -> Result<(), ContinuationError> {
        self.inner.prepare_continuation(layers)
    }
    fn recurrent_state(&mut self, layer: usize) -> Result<&mut RecurrentState, ContinuationError> {
        // Only once the state is live: a ULP of ZERO is a subnormal that
        // the first update absorbs exactly — measured, the first version
        // of this control nudged the zero start and went unseen.
        let live = self.inner.position() > 0 && !self.moved;
        let state = self.inner.recurrent_state(layer)?;
        if live {
            let cells = state.buffer_mut(0).cells_mut();
            let largest = (0..cells.len())
                .max_by(|&a, &b| cells[a].abs().total_cmp(&cells[b].abs()))
                .expect("a non-empty buffer");
            nudge(&mut cells[largest]);
            self.moved = true;
        }
        Ok(state)
    }
    fn latent_state(&mut self, layer: usize) -> Result<&mut LatentKvRows, ContinuationError> {
        self.inner.latent_state(layer)
    }
}

/// The control: one ULP anywhere in the held state is caught, on both
/// fixtures — so the equalities above are claims, not echoes.
#[test]
fn a_one_ulp_perturbation_is_caught_on_both_fixtures() {
    for (model, name, prompt) in [
        (
            miniature_glimmer as fn(&std::path::Path),
            "c3-sliding",
            &G_TOKENS[..],
        ),
        (hybrid_kda_mla_f32_model, "c3-hybrid", &[1u32, 0, 2][..]),
    ] {
        let (_c, plan, store) = open(model, name);
        let reference = drive(&plan, &store, prompt, &mut CanonicalKvState::new());
        let mut perturbed = Perturbed {
            inner: RowKvState::default(),
            moved: false,
        };
        let record = drive(&plan, &store, prompt, &mut perturbed);
        assert!(perturbed.moved, "{name}: the perturbation never fired");
        assert!(record != reference, "{name}: a one-ULP change went unseen");
    }
}

/// Codec width and exact-K window for the codec providers built here.
const CODEC_BITS: u8 = 4;
const CODEC_RECENT_WINDOW: usize = 2;

/// Every shipped provider answers an invalidation itself: it records the
/// reason, and from then on neither a batch prefill nor a decode step runs
/// over it. A failed rollback makes the STATE unusable, whichever provider
/// holds it.
#[test]
fn every_shipped_provider_records_an_invalidation_and_is_then_refused() {
    const REASON: &str = "a failed advance could not be put back";
    const UNUSABLE: &str = "unusable and must be discarded";
    let (_c, plan, store) = open(miniature_glimmer, "invalidated");
    let backend = ReferenceBackend::new();
    let providers: [(&str, Box<dyn KvState + Send>); 5] = [
        ("row/v1", Box::new(RowKvState::default())),
        ("canonical/v1", Box::new(CanonicalKvState::new())),
        ("window/v1", Box::new(super::super::WindowKvState::new())),
        (
            "codec/v1",
            Box::new(super::super::CodecKvState::new(CODEC_BITS)),
        ),
        (
            "codec-recent",
            Box::new(super::super::CodecRecentKvState::new(
                CODEC_BITS,
                CODEC_RECENT_WINDOW,
            )),
        ),
    ];
    for (name, mut state) in providers {
        assert!(state.invalidated().is_none(), "{name}: fresh");
        state.invalidate(REASON);
        assert_eq!(state.invalidated(), Some(REASON), "{name}: not recorded");

        let prefill = prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut *state);
        let refusal = prefill.map(|_| ()).unwrap_err().to_string();
        assert!(
            refusal.contains(UNUSABLE) && refusal.contains(REASON),
            "{name}: {refusal}"
        );
        assert_eq!(state.position(), 0, "{name}: a refused prefill moved it");

        let mut session =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut *state).unwrap();
        let refusal = session
            .step(G_TOKENS[0])
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(refusal.contains(UNUSABLE), "{name}: {refusal}");
    }
}
