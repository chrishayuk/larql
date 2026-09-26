//! **C4 of CONTINUATION-PLUGIN-1: state that outlives a call carries the
//! authority that built it, and resumes only under that authority.**
//!
//! The forecast predicts three distinct refusals — provider absent,
//! revision changed, configuration changed — each naming both sides, and
//! that identity and configuration digest are two authorities compared
//! separately. The notes fix the layer: resume compares and returns; it
//! builds, selects and prefills nothing. The anti-cheat against the
//! shipped built-ins lives in `larql-kv` (`vindex3/tests/resume_anti_cheat.rs`),
//! where both built-ins can be registered beside the hostile provider.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::super::continuation::LayerContinuationGeometry;
use super::super::continuation_authority::{
    ConfigDigest, ContinuationAuthority, ContinuationConfig,
};
use super::super::continuation_handoff::ResumeRefusal;
use super::super::continuation_identity::ContinuationIdentity;
use super::super::continuation_registry::{
    BoxedContinuation, ContinuationFactory, ContinuationRegion, ContinuationRegistry,
    SelectedContinuation,
};
use super::super::kv::{HistoryRange, LayerKvGeometry, RowKvState};

// ---- fixtures -------------------------------------------------------------

const KV_DIM: usize = 4;
const LAYERS: usize = 2;
const POSITIONS: usize = 3;

fn geometry() -> Vec<LayerContinuationGeometry> {
    vec![
        LayerContinuationGeometry::Kv(LayerKvGeometry {
            kv_dim: KV_DIM,
            window: None,
            history: HistoryRange::Full,
        });
        LAYERS
    ]
}

/// Row storage under any identity, counting every build so a test can
/// prove resume built nothing.
struct Counting {
    identity: ContinuationIdentity,
    builds: Arc<AtomicUsize>,
}

impl ContinuationFactory for Counting {
    fn identity(&self) -> ContinuationIdentity {
        self.identity.clone()
    }

    fn regions(&self) -> &[ContinuationRegion] {
        &[ContinuationRegion::Kv]
    }

    fn validate_config(&self, config: &ContinuationConfig) -> Result<(), String> {
        match config.keys().find(|key| *key != "bits") {
            Some(key) => Err(format!("unknown option `{key}`")),
            None => Ok(()),
        }
    }

    fn build(&self, _config: &ContinuationConfig) -> BoxedContinuation {
        self.builds.fetch_add(1, Ordering::SeqCst);
        Box::new(RowKvState::default())
    }
}

fn select(
    family: &str,
    revision: u32,
    options: &[&str],
) -> (SelectedContinuation, Arc<AtomicUsize>) {
    let builds = Arc::new(AtomicUsize::new(0));
    let mut registry = ContinuationRegistry::new();
    registry
        .register(Box::new(Counting {
            identity: ContinuationIdentity::new(family, revision),
            builds: Arc::clone(&builds),
        }))
        .unwrap();
    let selected = registry
        .select(
            &ContinuationIdentity::new(family, revision),
            &ContinuationConfig::parse(options).unwrap(),
            &geometry(),
        )
        .unwrap();
    (selected, builds)
}

fn row(layer: usize, position: usize, salt: f32) -> Vec<f32> {
    (0..KV_DIM)
        .map(|i| salt + (layer * 100 + position * 10 + i) as f32)
        .collect()
}

/// A handoff whose state has absorbed `POSITIONS` positions on every
/// layer — what a prefill leaves behind.
fn prefilled(
    selected: &SelectedContinuation,
) -> super::super::continuation_handoff::ContinuationHandoff {
    let mut handoff = selected.begin();
    let state = handoff.state_mut();
    state.prepare(
        &[LayerKvGeometry {
            kv_dim: KV_DIM,
            window: None,
            history: HistoryRange::Full,
        }; LAYERS],
    );
    for position in 0..POSITIONS {
        for layer in 0..LAYERS {
            state.append(layer, row(layer, position, 0.0), row(layer, position, 0.5));
        }
    }
    state.set_position(POSITIONS);
    handoff
}

fn authority(family: &str, revision: u32, options: &[&str]) -> ContinuationAuthority {
    ContinuationAuthority::new(
        ContinuationIdentity::new(family, revision),
        &ContinuationConfig::parse(options).unwrap(),
    )
}

// ---- sealing --------------------------------------------------------------

#[test]
fn begin_seals_fresh_state_with_the_selections_authority() {
    let (selected, builds) = select("hostile-test-provider", 77, &["bits=4"]);
    let handoff = selected.begin();
    assert_eq!(handoff.authority(), selected.authority());
    assert_eq!(handoff.position(), 0);
    assert_eq!(
        builds.load(Ordering::SeqCst),
        1,
        "begin builds exactly one provider"
    );
    assert!(format!("{handoff:?}").contains("hostile-test-provider/v77"));
}

// ---- resume under the same authority --------------------------------------

#[test]
fn the_same_authority_resumes_the_recorded_state_untouched() {
    let (selected, builds) = select("hostile-test-provider", 77, &["bits=4"]);
    let handoff = prefilled(&selected);
    let resumed = handoff
        .resume(selected.authority())
        .expect("the authority that wrote the state resumes it");
    assert_eq!(builds.load(Ordering::SeqCst), 1, "resume built nothing");
    assert_eq!(resumed.position(), POSITIONS);
    for layer in 0..LAYERS {
        for position in 0..POSITIONS {
            assert_eq!(
                resumed.state().rows(layer).to_owned_rows().0[position],
                row(layer, position, 0.0)
            );
            assert_eq!(
                resumed.state().rows(layer).to_owned_rows().1[position],
                row(layer, position, 0.5)
            );
        }
    }
}

#[test]
fn an_equal_authority_from_another_selection_resumes() {
    // Authority is by value, not by which selection object produced it:
    // a server that re-selects the same provider and configuration at a
    // later binding still resumes its own conversations.
    let (writer, _) = select("hostile-test-provider", 77, &["bits=4"]);
    let (reader, reader_builds) = select("hostile-test-provider", 77, &["bits=4"]);
    let resumed = prefilled(&writer).resume(reader.authority()).unwrap();
    assert_eq!(resumed.position(), POSITIONS);
    assert_eq!(reader_builds.load(Ordering::SeqCst), 0);
}

// ---- the three refusals ---------------------------------------------------

#[test]
fn another_family_refuses_as_provider_absent_naming_both() {
    let (writer, _) = select("hostile-test-provider", 77, &[]);
    let refusal = prefilled(&writer)
        .resume(&authority("canonical", 1, &[]))
        .unwrap_err();
    assert_eq!(
        refusal,
        ResumeRefusal::ProviderAbsent {
            recorded: ContinuationIdentity::new("hostile-test-provider", 77),
            resolved: ContinuationIdentity::new("canonical", 1),
        }
    );
    assert_eq!(refusal.kind(), "provider_absent");
    let text = refusal.to_string();
    assert!(text.contains("hostile-test-provider/v77"), "{text}");
    assert!(text.contains("canonical/v1"), "{text}");
    assert!(text.contains("absent"), "{text}");
}

#[test]
fn another_revision_refuses_as_revision_changed_naming_both() {
    let (writer, _) = select("hostile-test-provider", 77, &[]);
    let refusal = prefilled(&writer)
        .resume(&authority("hostile-test-provider", 78, &[]))
        .unwrap_err();
    assert_eq!(
        refusal,
        ResumeRefusal::RevisionChanged {
            recorded: ContinuationIdentity::new("hostile-test-provider", 77),
            resolved: ContinuationIdentity::new("hostile-test-provider", 78),
        }
    );
    assert_eq!(refusal.kind(), "revision_changed");
    let text = refusal.to_string();
    assert!(text.contains("hostile-test-provider/v77"), "{text}");
    assert!(text.contains("hostile-test-provider/v78"), "{text}");
    assert!(text.contains("revision changed"), "{text}");
}

#[test]
fn another_configuration_refuses_as_configuration_changed_naming_both_digests() {
    let (writer, _) = select("hostile-test-provider", 77, &["bits=4"]);
    let resolved = authority("hostile-test-provider", 77, &["bits=3"]);
    let refusal = prefilled(&writer).resume(&resolved).unwrap_err();
    let recorded = writer.authority().config_digest.clone();
    assert_eq!(
        refusal,
        ResumeRefusal::ConfigurationChanged {
            identity: ContinuationIdentity::new("hostile-test-provider", 77),
            recorded: recorded.clone(),
            resolved: resolved.config_digest.clone(),
        }
    );
    assert_ne!(recorded, resolved.config_digest);
    assert_eq!(refusal.kind(), "configuration_changed");
    let text = refusal.to_string();
    assert!(text.contains(recorded.as_str()), "{text}");
    assert!(text.contains(resolved.config_digest.as_str()), "{text}");
    assert!(text.contains("configuration changed"), "{text}");
}

#[test]
fn identity_is_compared_before_the_digest() {
    // The digest covers the identity (C2), so every identity change also
    // moves it. The refusal must still name the identity change, never a
    // configuration mismatch.
    let recorded = authority("hostile-test-provider", 77, &["bits=4"]);
    for (resolved, kind) in [
        (authority("row", 1, &["bits=4"]), "provider_absent"),
        (
            authority("hostile-test-provider", 76, &["bits=4"]),
            "revision_changed",
        ),
    ] {
        assert_ne!(recorded.config_digest, resolved.config_digest);
        assert_eq!(
            ResumeRefusal::between(&recorded, &resolved).unwrap().kind(),
            kind
        );
    }
}

#[test]
fn the_comparison_has_exactly_one_accepting_case() {
    // Control for every refusal above: the comparison is not a constant.
    // Only the identical authority passes; flipping any one component
    // refuses, each under its own kind.
    let base = authority("hostile-test-provider", 77, &["bits=4"]);
    assert_eq!(ResumeRefusal::between(&base, &base.clone()), None);
    let digest_only = ContinuationAuthority {
        identity: base.identity.clone(),
        config_digest: ConfigDigest::of(
            &base.identity,
            &ContinuationConfig::parse(&["bits=4", "mode=x"]).unwrap(),
        ),
    };
    let kinds: Vec<_> = [
        authority("other", 77, &["bits=4"]),
        authority("hostile-test-provider", 1, &["bits=4"]),
        digest_only,
    ]
    .iter()
    .map(|resolved| ResumeRefusal::between(&base, resolved).map(|r| r.kind()))
    .collect();
    assert_eq!(
        kinds,
        [
            Some("provider_absent"),
            Some("revision_changed"),
            Some("configuration_changed")
        ]
    );
}

#[test]
fn a_refusal_builds_nothing_on_either_side() {
    let (writer, writer_builds) = select("hostile-test-provider", 77, &["bits=4"]);
    let (reader, reader_builds) = select("hostile-test-provider", 77, &["bits=3"]);
    let handoff = prefilled(&writer);
    assert!(handoff.resume(reader.authority()).is_err());
    assert_eq!(
        writer_builds.load(Ordering::SeqCst),
        1,
        "only the original build"
    );
    assert_eq!(
        reader_builds.load(Ordering::SeqCst),
        0,
        "no replacement was built"
    );
}

#[test]
fn authority_round_trips_through_serde_as_a_handoff_records_it() {
    let recorded = authority("hostile-test-provider", 77, &["bits=4"]);
    let json = serde_json::to_string(&recorded).unwrap();
    let back: ContinuationAuthority = serde_json::from_str(&json).unwrap();
    assert_eq!(back, recorded);
    assert_eq!(ResumeRefusal::between(&back, &recorded), None);
}
