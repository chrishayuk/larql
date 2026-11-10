//! The shared admission gate: `encode` and `plan --capability` ask
//! [`admit`] and nothing else, so the two verbs cannot disagree.

use super::support::glimmer_shaped_target_with;
use crate::format::vindex3::plan::admission::admit;
use crate::format::vindex3::plan::capability::Capability;
use crate::format::vindex3::plan::{plan_system, SystemPlan};

/// Glimmer-shaped, with `edit` applied to its config.
fn plan_with(edit: impl FnOnce(&mut serde_json::Value)) -> SystemPlan {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), edit);
    plan_system(&[("target-artifact".to_string(), inventory)])
}

/// Whole-model inadmissible on a vision-tower fact the text closure
/// never reaches — the shape of a multimodal checkpoint run for text.
fn text_only_gap() -> SystemPlan {
    plan_with(|config| {
        config["vision_config"]["some_unjudged_tower_fact"] = serde_json::json!(7);
    })
}

#[test]
fn a_whole_model_gap_refuses_the_unscoped_gate_and_admits_text() {
    let plan = text_only_gap();
    assert!(
        !plan.admissible,
        "the fixture must be whole-model inadmissible"
    );

    let whole = admit(&plan, None).expect_err("whole-model must refuse");
    assert!(whole.to_string().contains("inadmissible plan"), "{whole}");

    let status = admit(&plan, Some(Capability::TextGeneration))
        .expect("text generation is admissible and supported")
        .expect("a scoped gate reports the capability it judged");
    assert_eq!(status.capability, Capability::TextGeneration);
    assert_eq!(status.blocking, 0);
}

#[test]
fn an_admissible_whole_model_plan_passes_the_unscoped_gate_with_no_status() {
    let plan = plan_with(|_| {});
    assert!(plan.admissible, "the clean fixture must be admissible");
    assert!(admit(&plan, None).unwrap().is_none());
}

#[test]
fn a_blocker_inside_the_closure_refuses_the_scoped_gate_by_count() {
    let plan = plan_with(|config| {
        config["text_config"]["some_unjudged_text_fact"] = serde_json::json!(7);
    });
    let err = admit(&plan, Some(Capability::TextGeneration))
        .expect_err("an unjudged text fact blocks text");
    let message = err.to_string();
    assert!(
        message.contains("inside that capability's closure"),
        "{message}"
    );
    assert!(message.contains("--capability"), "{message}");
}

#[test]
fn a_capability_with_no_executor_is_refused_even_when_understood() {
    let plan = plan_with(|_| {});
    let err = admit(&plan, Some(Capability::Drafting)).expect_err("no draft executor");
    assert!(err.to_string().contains("no executor"), "{err}");
}

#[test]
fn a_capability_the_plan_reports_nothing_about_is_refused_by_name() {
    let mut plan = plan_with(|_| {});
    plan.capabilities.clear();
    let err = admit(&plan, Some(Capability::TextGeneration)).expect_err("no verdict");
    assert!(
        err.to_string().contains("no verdict for TextGeneration"),
        "{err}"
    );
}
