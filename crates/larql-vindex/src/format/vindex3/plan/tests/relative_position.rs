//! A declared relative-position scheme is carried as itself, never as a
//! rotation.
//!
//! `rope_base` carries a default, so a checkpoint that declares no rope
//! key at all still resolved to `Rope { theta }` on every layer — a
//! rotation its author never asked for. Inkling-Small is the live case:
//! `d_rel: 16`, `rel_extent: 1024`, and no rope key anywhere.

use larql_models::config::PositionPolicy;

use super::support::glimmer_shaped_target_with;
use crate::format::vindex3::graph::build_from_inventories;
use crate::format::vindex3::plan::{plan_system, FindingCategory, PlannedFinding};

const D_REL: usize = 16;
const REL_EXTENT: usize = 1024;

fn declare_relative(config: &mut serde_json::Value) {
    config["text_config"]["d_rel"] = serde_json::json!(D_REL);
    config["text_config"]["rel_extent"] = serde_json::json!(REL_EXTENT);
}

#[test]
fn a_declared_relative_scheme_is_not_resolved_as_rotary() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), declare_relative);
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    let table = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.attention.as_ref())
        .expect("a per-layer table");

    for (i, layer) in table.iter().enumerate() {
        assert_eq!(
            layer.position,
            PositionPolicy::Relative {
                d_rel: D_REL,
                extent: REL_EXTENT,
            },
            "layer {i}"
        );
        // It does not rotate, so it has no base — and it is NOT NoPE
        // either: position enters, just not by rotation.
        assert_eq!(layer.position.rope_theta(), None, "layer {i}");
        assert_ne!(layer.position, PositionPolicy::None, "layer {i}");
    }
}

/// The paired control: the same fixture without the declaration still
/// resolves rotary, so the test above is about the declaration and not
/// about rotary having been removed.
#[test]
fn the_same_fixture_without_the_declaration_still_rotates() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |_| {});
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    let table = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.attention.as_ref())
        .expect("a per-layer table");
    assert!(
        table[0].position.rope_theta().is_some(),
        "the unmodified fixture is rotary: {:?}",
        table[0].position
    );
}

/// Both parameters carry, each answering with its own value — a composite
/// answer would never equal the scalar the checkpoint wrote and would read
/// as a mismatch on a policy that is carried correctly.
#[test]
fn both_relative_parameters_carry_to_the_position_policy() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), declare_relative);
    let findings: Vec<PlannedFinding> = plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect();

    for (leaf, expected) in [("d_rel", D_REL), ("rel_extent", REL_EXTENT)] {
        let finding = findings
            .iter()
            .find(|f| f.subject.ends_with(leaf))
            .unwrap_or_else(|| panic!("no finding for `{leaf}`"));
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{leaf}: {finding:?}"
        );
        assert!(!finding.blocks(), "{leaf}: {finding:?}");
        assert_eq!(
            finding.resolved,
            Some(serde_json::json!(expected)),
            "{leaf} must carry its own value, not a composite"
        );
    }
}
