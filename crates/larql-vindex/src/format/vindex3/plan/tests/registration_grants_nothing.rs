//! Registering a family resolves its NAME. It grants nothing else.
//!
//! This is the control that guards the `architecture_identity` gate
//! against erosion. The gate exists because an unmatched `model_type`
//! used to fall through to `GenericArch` and be served Llama-shaped
//! defaults for norm placement, QK norm, embedding scaling and gating —
//! facts the checkpoint never declared. The failure mode when that gate
//! is "fixed" carelessly is the mirror image: a family entry that makes
//! a checkpoint admissible by supplying the same defaults from a
//! friendlier place.
//!
//! So the pair below is the whole test. The same registered identity is
//! admissible when every declaration has a home, and REFUSES when one
//! declaration does not — and the refusal names that declaration, not
//! the family.

use super::support::glimmer_shaped_target_with;
use crate::format::vindex3::plan::capability::Capability;
use crate::format::vindex3::plan::{plan_system, SystemPlan};

/// The OLMo-2 shape on the shared fixture: a registered identity and a
/// post-norm estate. Tensors are the fixture's; what is under test is
/// what the plan does with the DECLARATIONS.
fn olmo2_plan(edit: impl FnOnce(&mut serde_json::Value)) -> SystemPlan {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["model_type"] = serde_json::json!("olmo2");
        edit(config);
    });
    plan_system(&[("target-artifact".to_string(), inventory)])
}

fn blocking_subjects(plan: &SystemPlan) -> Vec<String> {
    let blockers: Vec<_> = plan
        .capabilities
        .iter()
        .find(|c| c.capability == Capability::TextGeneration)
        .map(|c| c.blocker_ids.clone())
        .unwrap_or_default();
    plan.artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| blockers.contains(&f.id))
        .map(|f| f.subject.clone())
        .collect()
}

/// A registered identity does NOT stop `architecture_family` being the
/// thing that was blocking — it removes it, and nothing else.
#[test]
fn a_registered_identity_removes_the_identity_blocker_and_only_that() {
    let registered = blocking_subjects(&olmo2_plan(|_| {}));
    assert!(
        !registered.iter().any(|s| s.contains("architecture_family")),
        "a registered identity must not block on its own name: {registered:?}"
    );

    // The control: the SAME fixture under a name nothing matches still
    // blocks on identity. Without this, the assertion above could pass
    // on a fixture that never had an identity blocker at all.
    let dir = tempfile::tempdir().unwrap();
    let unknown = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["model_type"] = serde_json::json!("olmo_no_such_generation");
    });
    let unknown = plan_system(&[("target-artifact".to_string(), unknown)]);
    assert!(
        blocking_subjects(&unknown)
            .iter()
            .any(|s| s.contains("architecture_family")),
        "an unmatched model_type must still refuse"
    );
}

/// **The negative witness.** One declaration the schema cannot carry,
/// under a fully registered identity, must still refuse — and the
/// refusal must name that declaration.
///
/// `sliding_window_pattern` is the real instance: EXAONE-4 declares
/// `"LLLG"`, a period string this schema has no field for, and it must
/// keep blocking after `exaone4` is registered. If a family entry ever
/// starts supplying defaults for what it does not know, this is the test
/// that says so.
#[test]
fn registration_does_not_admit_a_declaration_the_schema_cannot_carry() {
    let plan = olmo2_plan(|config| {
        config["text_config"]["sliding_window_pattern"] = serde_json::json!("LLLG");
    });
    let subjects = blocking_subjects(&plan);
    assert!(
        subjects
            .iter()
            .any(|s| s.ends_with("sliding_window_pattern")),
        "a registered family must not absorb an unrepresentable declaration: {subjects:?}"
    );
    assert!(
        !subjects.iter().any(|s| s.contains("architecture_family")),
        "and the refusal must be about the DECLARATION, not the name: {subjects:?}"
    );
}
