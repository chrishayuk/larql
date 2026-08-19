//! Whole-system planning: verdicts, summaries, graph-backed interfaces.

use super::support::{drafter_shaped, glimmer_shaped_target, known_dense};
use crate::format::vindex3::plan::{plan_system, FindingCategory, SemanticClass};

/// THE G2 exit gate, on the fixture system: a Glimmer-shaped target +
/// drafter plans clean — every object placed, the interface resolved, the
/// NoPE policies honoured — with zero blocking findings and no
/// Muse-specific code anywhere in the path.
#[test]
fn glimmer_shaped_system_plans_clean() {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target_dir.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    let plan = plan_system(&named);

    assert_eq!(plan.summary.blocking, 0, "blocking findings: {:#?}", {
        let blockers: Vec<_> = plan
            .artifacts
            .iter()
            .flat_map(|a| &a.findings)
            .filter(|f| f.blocks())
            .collect();
        blockers
    });
    assert_eq!(plan.summary.mismatched, 0);
    assert!(plan.admissible);

    // The interface is a resolved edge with both sides named.
    assert_eq!(plan.interfaces.len(), 1);
    let interface = &plan.interfaces[0];
    assert_eq!(interface.producer_component, "target");
    assert_eq!(interface.producer_layers, vec![1, 3, 5]);
    assert_eq!(interface.consumer_object, "draft.feature_projector");
    assert_eq!(interface.block_size, Some(4));

    // The graph rides in the plan — the proof object for G3.
    assert!(plan.graph.validate().is_empty());
    assert_eq!(plan.graph.edges.len(), 1);

    // Summary counts stay derivable from the findings.
    let all: Vec<_> = plan.artifacts.iter().flat_map(|a| &a.findings).collect();
    assert_eq!(
        plan.summary.representable,
        all.iter()
            .filter(|f| f.category == FindingCategory::Representable)
            .count()
    );
}

/// The tripwire, flipped intentionally at G2: a known dense model now
/// plans admissible — the schema has homes for its objects. From here the
/// gate's job is to keep it admissible; a regression that starts blocking
/// clean dense models fails here first.
#[test]
fn known_dense_plans_admissible() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let plan = plan_system(&named);
    assert!(plan.admissible, "blocking: {:#?}", {
        let blockers: Vec<_> = plan
            .artifacts
            .iter()
            .flat_map(|a| &a.findings)
            .filter(|f| f.blocks())
            .collect();
        blockers
    });
    assert!(plan.interfaces.is_empty());
    assert!(plan.artifacts[0]
        .findings
        .iter()
        .any(|f| f.subject == "target.embedding" && f.category == FindingCategory::Representable));
}

/// A drafter alone cannot resolve its producer: the interface finding
/// blocks, and nothing is guessed.
#[test]
fn drafter_alone_is_inadmissible_on_the_unresolved_interface() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("drafter-artifact".to_string(), drafter_shaped(dir.path()))];
    let plan = plan_system(&named);
    assert!(!plan.admissible);
    assert!(plan.interfaces.is_empty());
    let interface_blockers: Vec<_> = plan.artifacts[0]
        .findings
        .iter()
        .filter(|f| f.category == FindingCategory::Interface && f.blocks())
        .collect();
    assert_eq!(interface_blockers.len(), 1);
    assert_eq!(
        interface_blockers[0].class,
        SemanticClass::InterfaceSemantic
    );
    assert!(interface_blockers[0].detail.contains("producer"));
}

/// Ambiguous producers block rather than guess.
#[test]
fn ambiguous_producer_blocks() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let dir_c = tempfile::tempdir().unwrap();
    let named = vec![
        ("deep-a".to_string(), glimmer_shaped_target(dir_a.path())),
        ("deep-b".to_string(), glimmer_shaped_target(dir_b.path())),
        ("drafter".to_string(), drafter_shaped(dir_c.path())),
    ];
    let plan = plan_system(&named);
    assert!(!plan.admissible);
    assert!(plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .any(|f| f.detail.contains("refusing to guess")));
}

/// An unjudged config key still blocks the whole plan — the fail-closed
/// rule survives the G2 flip.
#[test]
fn an_unjudged_key_still_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    inventory
        .config_keys
        .push(larql_models::inventory::ConfigKeyFact {
            path: "some_future_field_nobody_reviewed".to_string(),
            value: serde_json::json!(42),
            status: larql_models::inventory::KeyStatus::Unconsumed,
        });
    let named = vec![("llama-artifact".to_string(), inventory)];
    let plan = plan_system(&named);
    assert!(!plan.admissible);
    assert!(plan.artifacts[0]
        .findings
        .iter()
        .any(|f| f.class == SemanticClass::Unknown && f.blocks()));
}

/// The sibling case: a key some future parser reads (`Consumed`, not
/// `Unconsumed`) but the semantics registry has never classified. Parser
/// consumption is not representation authority, so this blocks exactly
/// like the unread case above — a key the parser happened to notice must
/// not silently outrank a key it never saw.
#[test]
fn a_consumed_but_unclassified_key_still_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    inventory
        .config_keys
        .push(larql_models::inventory::ConfigKeyFact {
            path: "some_future_field_a_parser_reads_but_nobody_classified".to_string(),
            value: serde_json::json!(42),
            status: larql_models::inventory::KeyStatus::Consumed,
        });
    let named = vec![("llama-artifact".to_string(), inventory)];
    let plan = plan_system(&named);
    assert!(!plan.admissible);
    let finding = plan.artifacts[0]
        .findings
        .iter()
        .find(|f| f.subject == "some_future_field_a_parser_reads_but_nobody_classified")
        .expect("finding for the unjudged consumed key");
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert_eq!(finding.class, SemanticClass::Unknown);
    assert!(finding.blocks());
    assert!(finding
        .detail
        .contains("parser consumption is not representation"));
}

/// Attention policy reports as representable per component, with the NoPE
/// count visible.
#[test]
fn attention_policy_is_representable_with_nope_count() {
    let target_dir = tempfile::tempdir().unwrap();
    let named = vec![(
        "target-artifact".to_string(),
        glimmer_shaped_target(target_dir.path()),
    )];
    let plan = plan_system(&named);
    let policy = plan.artifacts[0]
        .findings
        .iter()
        .find(|f| f.subject == "attention_policy")
        .expect("attention policy finding");
    assert_eq!(policy.category, FindingCategory::Representable);
    assert!(
        policy.detail.contains("2 NoPE layer(s)"),
        "{}",
        policy.detail
    );
}

#[test]
fn plan_round_trips_through_json() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![(
        "target-artifact".to_string(),
        glimmer_shaped_target(dir.path()),
    )];
    let plan = plan_system(&named);
    let json = serde_json::to_string_pretty(&plan).unwrap();
    let back: crate::format::vindex3::plan::SystemPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(back.admissible, plan.admissible);
    assert_eq!(back.summary, plan.summary);
    assert_eq!(back.graph.objects.len(), plan.graph.objects.len());
}
