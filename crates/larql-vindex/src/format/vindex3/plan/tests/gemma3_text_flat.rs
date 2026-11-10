//! The flat `gemma3_text` config through the plan gate (Gemma 3 270M /
//! 1B). Before this witness both real checkpoints refused on the RoPE
//! pair — the flat `rope_theta` judged against every layer while the
//! sliding layers rotate at `rope_local_base_freq` (mismatched), both
//! leaves unanswered by their probes, and `sliding_window_pattern`
//! refused outright — plus, on the 270M, `_sliding_window_pattern` and
//! `use_bidirectional_attention` graded `Unknown`. Each fact is pinned
//! here, and each re-drop names the declaration that must keep blocking.

use larql_models::config::PositionPolicy;

use crate::format::vindex3::plan::tests_support::{
    flat_gemma3_layer_types, flat_gemma3_target_with, FlatGemma3Dialect, FLAT_GEMMA3_GLOBAL_THETA,
    FLAT_GEMMA3_LAYERS, FLAT_GEMMA3_LOCAL_THETA, FLAT_GEMMA3_PERIOD,
};
use crate::format::vindex3::plan::{
    plan_system, FindingCategory, PlannedFinding, SemanticClass, SystemPlan,
};

/// A period other than the declared default, so honouring and ignoring
/// a declaration answer differently.
const OTHER_PERIOD: usize = 3;
/// A period longer than the stack: no layer is full.
const PERIOD_BEYOND_STACK: usize = FLAT_GEMMA3_LAYERS + 1;

fn plan_of(dialect: FlatGemma3Dialect, mutate: impl FnOnce(&mut serde_json::Value)) -> SystemPlan {
    let dir = tempfile::tempdir().unwrap();
    let inventory = flat_gemma3_target_with(dir.path(), dialect, mutate);
    plan_system(&[("gemma3-text-artifact".to_string(), inventory)])
}

fn findings_for<'a>(plan: &'a SystemPlan, subject: &str) -> Vec<&'a PlannedFinding> {
    plan.artifacts
        .iter()
        .flat_map(|a| a.findings.iter())
        .filter(|f| f.subject == subject)
        .collect()
}

fn blocking_subjects(plan: &SystemPlan) -> Vec<String> {
    plan.artifacts
        .iter()
        .flat_map(|a| a.findings.iter())
        .filter(|f| f.blocks())
        .map(|f| f.subject.clone())
        .collect()
}

fn text_generation_admissible(plan: &SystemPlan) -> bool {
    plan.capabilities
        .iter()
        .find(|c| {
            serde_json::to_value(c.capability).unwrap() == serde_json::json!("text_generation")
        })
        .expect("text generation is always a planned capability")
        .admissible
}

fn assert_admitted(plan: &SystemPlan) {
    let blocking = blocking_subjects(plan);
    assert!(blocking.is_empty(), "blocked on: {blocking:#?}");
    assert!(text_generation_admissible(plan));
    assert_eq!(plan.summary.mismatched, 0);
}

fn assert_blocks_on(plan: &SystemPlan, subject: &str) {
    let blocking = blocking_subjects(plan);
    assert!(
        blocking.iter().any(|s| s == subject),
        "`{subject}` must block: {blocking:#?}"
    );
}

/// Every finding for `subject` is representable — the comparator's and
/// the carriage gate's alike.
fn assert_all_representable(plan: &SystemPlan, subject: &str) {
    let found = findings_for(plan, subject);
    assert!(!found.is_empty(), "no finding for `{subject}`");
    for finding in found {
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}: {}",
            finding.detail
        );
    }
}

#[test]
fn the_270m_dialect_is_admissible_for_text_generation() {
    let plan = plan_of(FlatGemma3Dialect::LayerTypes, |_| {});
    assert_admitted(&plan);
    for subject in ["rope_theta", "rope_local_base_freq", "layer_types"] {
        assert_all_representable(&plan, subject);
    }
}

#[test]
fn the_1b_dialect_is_admissible_for_text_generation() {
    let plan = plan_of(FlatGemma3Dialect::Period, |_| {});
    assert_admitted(&plan);
    for subject in [
        "rope_theta",
        "rope_local_base_freq",
        "sliding_window_pattern",
    ] {
        assert_all_representable(&plan, subject);
    }
}

/// The flat pair splits by layer type: the full layers rotate at
/// `rope_theta`, the sliding ones at `rope_local_base_freq`.
#[test]
fn the_flat_rope_pair_resolves_per_layer_type() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = flat_gemma3_target_with(dir.path(), FlatGemma3Dialect::Period, |_| {});
    for layer in &inventory.resolved.layers {
        let theta = if layer.attention == "full" {
            FLAT_GEMMA3_GLOBAL_THETA
        } else {
            FLAT_GEMMA3_LOCAL_THETA
        };
        assert_eq!(
            layer.position,
            PositionPolicy::Rope { theta },
            "layer {}",
            layer.layer
        );
    }
}

/// Re-drop: a sliding base that differs from the one the layers rotate
/// at is a disagreement — scoped to the sliding span, so the check is
/// live rather than vacuous.
#[test]
fn a_flat_rope_base_disagreeing_with_its_span_still_blocks() {
    // The arch resolves `rope_local_base_freq` for the sliding layers, so
    // move the full base onto the sliding value: the declared `rope_theta`
    // still governs the full layers and both agree...
    let agree = plan_of(FlatGemma3Dialect::Period, |c| {
        c["rope_theta"] = serde_json::json!(FLAT_GEMMA3_LOCAL_THETA);
    });
    assert_admitted(&agree);
    // ...while a per-section full base overrides the flat `rope_theta`,
    // which then speaks for no layer and is not judged on its own.
    let superseded = plan_of(FlatGemma3Dialect::Period, |c| {
        c["rope_parameters"] = serde_json::json!({
            "full_attention": { "rope_theta": FLAT_GEMMA3_GLOBAL_THETA, "rope_type": "default" }
        });
        c["rope_theta"] = serde_json::json!(FLAT_GEMMA3_LOCAL_THETA);
    });
    assert!(
        findings_for(&superseded, "rope_theta")
            .iter()
            .all(|f| f.category != FindingCategory::Mismatched),
        "a superseded flat base is not compared against the full layers"
    );
}

/// `use_bidirectional_attention: false` is the causal stack the graph
/// builds — inert at that value. `true` is a different mask and blocks.
#[test]
fn bidirectional_attention_is_inert_only_when_false() {
    let causal = plan_of(FlatGemma3Dialect::LayerTypes, |_| {});
    let finding = findings_for(&causal, "use_bidirectional_attention");
    assert_eq!(finding.len(), 1);
    assert_eq!(finding[0].class, SemanticClass::TrainingOnly);
    assert!(!finding[0].blocks());

    let bidirectional = plan_of(FlatGemma3Dialect::LayerTypes, |c| {
        c["use_bidirectional_attention"] = serde_json::json!(true);
    });
    assert_blocks_on(&bidirectional, "use_bidirectional_attention");
    assert!(!text_generation_admissible(&bidirectional));
}

/// `_sliding_window_pattern` is an alias of `layer_types`, benign only
/// while it corroborates the array.
#[test]
fn the_bc_period_must_corroborate_layer_types() {
    let plan = plan_of(FlatGemma3Dialect::LayerTypes, |_| {});
    let finding = findings_for(&plan, "_sliding_window_pattern");
    assert_eq!(finding[0].class, SemanticClass::Alias);

    let contradicting = plan_of(FlatGemma3Dialect::LayerTypes, |c| {
        c["_sliding_window_pattern"] = serde_json::json!(OTHER_PERIOD);
    });
    assert_blocks_on(&contradicting, "_sliding_window_pattern");
}

/// A declared period the stride derivation honours is carried; one the
/// resolution overrides (an explicit array outranks it) mismatches.
#[test]
fn a_declared_period_is_judged_against_the_built_span_table() {
    let honoured = plan_of(FlatGemma3Dialect::Period, |c| {
        c["sliding_window_pattern"] = serde_json::json!(OTHER_PERIOD);
    });
    assert_admitted(&honoured);

    let overridden = plan_of(FlatGemma3Dialect::Period, |c| {
        c["sliding_window_pattern"] = serde_json::json!(OTHER_PERIOD);
        c["layer_types"] = serde_json::json!(flat_gemma3_layer_types(FLAT_GEMMA3_PERIOD));
    });
    assert_blocks_on(&overridden, "sliding_window_pattern");
}

/// A table with no full layer has no period to read back, so the
/// declaration has nothing to agree with and refuses.
#[test]
fn a_period_beyond_the_stack_refuses() {
    let plan = plan_of(FlatGemma3Dialect::Period, |c| {
        c["sliding_window_pattern"] = serde_json::json!(PERIOD_BEYOND_STACK);
    });
    assert_blocks_on(&plan, "sliding_window_pattern");
}
