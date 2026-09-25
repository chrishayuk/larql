//! **C4 of CONTINUATION-PLUGIN-1: the resume anti-cheat.**
//!
//! Frozen in `docs/represent/forecasts/continuation-plugin-1.json`: state
//! produced by `hostile-test-provider/v77` under configuration A; the
//! resuming registry holds `hostile-test-provider/v77` under configuration
//! B, `canonical/v1` and `row/v1`. Required: a refusal naming the
//! configuration mismatch. Forbidden: selecting another working provider,
//! reconstructing the state under `canonical/v1`, or silently restarting
//! from a fresh prefill.
//!
//! The notes put this at the resume-contract layer: every factory counts
//! its builds, so "nothing else was built" is measured, and the state is
//! a real prefill of the recurrent + latent hybrid, so the refusal is
//! about state that exists. The control is the same handoff resumed under
//! configuration A, which must succeed and decode bit-identically to an
//! uninterrupted `canonical/v1` run — without it, a resume that refused
//! everything would pass.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::{
    ConfigDigest, ContinuationConfig,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_handoff::{
    ContinuationHandoff, ResumeRefusal,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion, ContinuationRegistry,
    SelectedContinuation,
};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{KvState, RowFactory, RowKvState};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

use super::super::{CanonicalFactory, CanonicalKvState};
use super::registry_parity::{argmax, open, snapshot, Snapshot};

const PROMPT: [u32; 3] = [1, 0, 2];
const STEPS: usize = 4;
const CONFIG_A: &str = "bits=4";
const CONFIG_B: &str = "bits=3";

fn hostile(revision: u32) -> ContinuationIdentity {
    ContinuationIdentity::new("hostile-test-provider", revision)
}

/// Any factory, with every build counted.
struct Counted<F> {
    inner: F,
    builds: Arc<AtomicUsize>,
}

impl<F: ContinuationFactory> ContinuationFactory for Counted<F> {
    fn identity(&self) -> ContinuationIdentity {
        self.inner.identity()
    }
    fn regions(&self) -> &[ContinuationRegion] {
        self.inner.regions()
    }
    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        self.inner.validate_config(config)
    }
    fn build(&self, config: &ContinuationConfig) -> BoxedContinuation {
        self.builds.fetch_add(1, Ordering::SeqCst);
        self.inner.build(config)
    }
}

/// The hostile provider: row storage, every region, one `bits` option —
/// enough for configuration to be a second authority. (C5's is an
/// external crate; this one only has to be a provider core never names.)
struct Hostile {
    revision: u32,
}

impl ContinuationFactory for Hostile {
    fn identity(&self) -> ContinuationIdentity {
        hostile(self.revision)
    }
    fn regions(&self) -> &[ContinuationRegion] {
        &ContinuationRegion::ALL
    }
    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        match config.keys().find(|key| *key != "bits") {
            Some(key) => Err(format!("unknown option `{key}`")),
            None => Ok(()),
        }
    }
    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        Box::new(RowKvState::default())
    }
}

/// A registry whose every factory counts its builds.
struct Counting {
    registry: ContinuationRegistry,
    hostile: Arc<AtomicUsize>,
    canonical: Arc<AtomicUsize>,
    row: Arc<AtomicUsize>,
}

impl Counting {
    /// `hostile-test-provider/v{revision}`, `canonical/v1` and `row/v1`.
    fn with_hostile(revision: u32) -> Self {
        let mut counting = Self::built_ins_only();
        counting
            .registry
            .register(Box::new(Counted {
                inner: Hostile { revision },
                builds: Arc::clone(&counting.hostile),
            }))
            .unwrap();
        counting
    }

    fn built_ins_only() -> Self {
        let (hostile, canonical, row) = Default::default();
        let mut counting = Self {
            registry: ContinuationRegistry::new(),
            hostile,
            canonical,
            row,
        };
        counting
            .registry
            .register(Box::new(Counted {
                inner: CanonicalFactory,
                builds: Arc::clone(&counting.canonical),
            }))
            .unwrap();
        counting
            .registry
            .register(Box::new(Counted {
                inner: RowFactory,
                builds: Arc::clone(&counting.row),
            }))
            .unwrap();
        counting
    }

    fn select(
        &self,
        identity: &ContinuationIdentity,
        option: Option<&str>,
        geometry: &[LayerContinuationGeometry],
    ) -> SelectedContinuation {
        let config = ContinuationConfig::parse(&option.into_iter().collect::<Vec<_>>()).unwrap();
        self.registry.select(identity, &config, geometry).unwrap()
    }

    /// (hostile, canonical, row) builds so far.
    fn builds(&self) -> (usize, usize, usize) {
        (
            self.hostile.load(Ordering::SeqCst),
            self.canonical.load(Ordering::SeqCst),
            self.row.load(Ordering::SeqCst),
        )
    }
}

struct Fixture {
    _container: tempfile::TempDir,
    plan: ComponentOpPlan,
    store: OperandStore,
    geometry: Vec<LayerContinuationGeometry>,
}

fn hybrid() -> Fixture {
    let (container, plan, store) = open(hybrid_kda_mla_f32_model, "c4-anti-cheat");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    assert!(
        geometry.iter().any(|g| g.recurrent().is_some()),
        "no recurrent layer"
    );
    assert!(
        geometry.iter().any(|g| g.latent_kv().is_some()),
        "no latent layer"
    );
    Fixture {
        _container: container,
        plan,
        store,
        geometry,
    }
}

/// A real prefill of [`PROMPT`] into state begun from `selected`: the
/// handoff, its snapshot, and the prefill's logits.
fn prefilled(
    fixture: &Fixture,
    selected: &SelectedContinuation,
) -> (ContinuationHandoff, Snapshot, Vec<f32>) {
    let mut handoff = selected.begin();
    let logits = prefill_plan(
        &fixture.plan,
        &fixture.store,
        &PROMPT,
        &ReferenceBackend::new(),
        handoff.state_mut(),
    )
    .unwrap()
    .logits
    .unwrap();
    let snap = snapshot(handoff.state_mut(), &fixture.geometry);
    (handoff, snap, logits)
}

/// Greedy decode for [`STEPS`] from `state`, starting after `logits`.
fn decode(fixture: &Fixture, state: &mut dyn KvState, logits: &[f32]) -> (Vec<u32>, Snapshot) {
    let backend = ReferenceBackend::new();
    let mut ids = vec![argmax(logits)];
    {
        let mut session =
            DecodeSession::with_kv_state(&fixture.plan, &fixture.store, &backend, state).unwrap();
        for _ in 0..STEPS {
            let next = session.step(*ids.last().unwrap()).unwrap().logits.unwrap();
            ids.push(argmax(&next));
        }
    }
    (ids, snapshot(state, &fixture.geometry))
}

#[test]
fn state_from_configuration_a_refuses_under_configuration_b_and_nothing_else_answers() {
    let fixture = hybrid();
    let writer = Counting::with_hostile(77);
    let (handoff, written, _) = prefilled(
        &fixture,
        &writer.select(&hostile(77), Some(CONFIG_A), &fixture.geometry),
    );
    assert!(
        written.recurrent_moved(),
        "the recurrence never moved, so the state would be vacuous"
    );

    // The resuming side: same provider under B, plus both built-ins —
    // two working providers that could hold this plan.
    let resuming = Counting::with_hostile(77);
    let under_b = resuming.select(&hostile(77), Some(CONFIG_B), &fixture.geometry);
    let refusal = handoff.resume(under_b.authority()).unwrap_err();

    // Required: the configuration mismatch, naming both digests.
    let digest =
        |option| ConfigDigest::of(&hostile(77), &ContinuationConfig::parse(&[option]).unwrap());
    assert_eq!(
        refusal,
        ResumeRefusal::ConfigurationChanged {
            identity: hostile(77),
            recorded: digest(CONFIG_A),
            resolved: digest(CONFIG_B),
        }
    );
    let text = refusal.to_string();
    assert!(text.contains(digest(CONFIG_A).as_str()) && text.contains(digest(CONFIG_B).as_str()));

    // Forbidden: another provider selected, the state rebuilt under
    // canonical/v1, or a fresh start — none of the resuming registry's
    // factories built anything. (Selecting B builds nothing either:
    // selection is a promise, not a provider.)
    assert_eq!(resuming.builds(), (0, 0, 0), "(hostile, canonical, row)");
    assert_eq!(
        writer.builds(),
        (1, 0, 0),
        "only the original conversation's build"
    );
}

#[test]
fn control_the_same_handoff_resumes_under_configuration_a_and_continues_exactly() {
    // Without this, a resume that refused everything would pass the test
    // above. The handoff resumed under A — selected afresh from a SECOND
    // registry, so equality is by authority, not by object — must decode
    // bit-identically to an uninterrupted canonical/v1 run.
    let fixture = hybrid();
    let writer = Counting::with_hostile(77);
    let (handoff, written, logits) = prefilled(
        &fixture,
        &writer.select(&hostile(77), Some(CONFIG_A), &fixture.geometry),
    );
    let resuming = Counting::with_hostile(77);
    let under_a = resuming.select(&hostile(77), Some(CONFIG_A), &fixture.geometry);
    let mut resumed = handoff
        .resume(under_a.authority())
        .expect("the authority that wrote the state resumes it");
    assert_eq!(snapshot(resumed.state_mut(), &fixture.geometry), written);
    assert_eq!(resuming.builds(), (0, 0, 0), "resume built nothing");
    let (ids, last) = decode(&fixture, resumed.state_mut(), &logits);

    let mut reference = CanonicalKvState::new();
    let reference_logits = prefill_plan(
        &fixture.plan,
        &fixture.store,
        &PROMPT,
        &ReferenceBackend::new(),
        &mut reference,
    )
    .unwrap()
    .logits
    .unwrap();
    assert_eq!(
        reference_logits, logits,
        "prefill differs from canonical/v1"
    );
    let (reference_ids, reference_last) = decode(&fixture, &mut reference, &reference_logits);
    assert_eq!(ids, reference_ids);
    assert_eq!(last, reference_last);
}

#[test]
fn a_bumped_revision_refuses_as_revision_changed() {
    let fixture = hybrid();
    let writer = Counting::with_hostile(77);
    let (handoff, _, _) = prefilled(
        &fixture,
        &writer.select(&hostile(77), Some(CONFIG_A), &fixture.geometry),
    );
    let resuming = Counting::with_hostile(78);
    let under_78 = resuming.select(&hostile(78), Some(CONFIG_A), &fixture.geometry);
    assert_eq!(
        handoff.resume(under_78.authority()).unwrap_err(),
        ResumeRefusal::RevisionChanged {
            recorded: hostile(77),
            resolved: hostile(78),
        }
    );
    assert_eq!(resuming.builds(), (0, 0, 0));
}

#[test]
fn a_registry_without_the_provider_refuses_as_absent_and_never_falls_back_to_canonical() {
    let fixture = hybrid();
    let writer = Counting::with_hostile(77);
    let (handoff, _, _) = prefilled(
        &fixture,
        &writer.select(&hostile(77), Some(CONFIG_A), &fixture.geometry),
    );
    // The resolving side holds only the built-ins and resolves canonical/v1
    // — the fallback the forecast forbids.
    let resuming = Counting::built_ins_only();
    let canonical = resuming.select(&CanonicalKvState::identity(), None, &fixture.geometry);
    assert_eq!(
        handoff.resume(canonical.authority()).unwrap_err(),
        ResumeRefusal::ProviderAbsent {
            recorded: hostile(77),
            resolved: CanonicalKvState::identity(),
        }
    );
    assert_eq!(resuming.builds(), (0, 0, 0));
}
