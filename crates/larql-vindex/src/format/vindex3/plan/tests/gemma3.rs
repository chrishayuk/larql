//! Gemma 3 through the plan gate (issue #436). The sealed semantic-graph
//! substrate (`google/gemma-3-12b-it`) refused on exactly seven
//! text-generation blockers before this witness — three `*_index` token
//! roles and `mm_tokens_per_image` graded `Unknown`, `rope_type: linear`
//! mismatched and its `factor` unanswered, and a text execution surface
//! missing the `vocab_size` the config never declares. Each fact is
//! pinned by re-dropping it. The miniature mirrors the real 4B config
//! shape and tensor spelling (`plan::tests_support::gemma3_shaped_target`).
//!
//! The whole-model verdict stays blocked on the SigLIP tower's own
//! surface (its activation and norm epsilon are undeclared), which is the
//! real checkpoint's state too and is exactly what `--text_only` scopes
//! out. Nothing here vouches for the vision tower.

use larql_models::config::PositionPolicy;
use larql_models::inventory::report::VocabSizeProvenance;

use crate::format::vindex3::plan::tests_support::{
    gemma3_shaped_target, GEMMA3_EMBED_TENSOR, GEMMA3_FULL_LAYERS, GEMMA3_GLOBAL_THETA,
    GEMMA3_HEAD_DIM, GEMMA3_KV_HEADS, GEMMA3_LOCAL_THETA, GEMMA3_Q_HEADS, GEMMA3_ROPE_FACTOR,
    GEMMA3_VOCAB,
};
use crate::format::vindex3::plan::{
    plan_system, FindingCategory, PlannedFinding, SemanticClass, SystemPlan,
};

const VISION_SURFACE: &str = "vision.execution_surface";
const TEXT_SURFACE: &str = "target.execution_surface";
const ROPE_TYPE: &str = "text_config.rope_scaling.rope_type";
const ROPE_FACTOR: &str = "text_config.rope_scaling.factor";
const INTERFACE_SUBJECTS: [&str; 4] = [
    "boi_token_index",
    "eoi_token_index",
    "image_token_index",
    "mm_tokens_per_image",
];

fn plan_of(inventory: larql_models::inventory::ArchitectureInventory) -> SystemPlan {
    plan_system(&[("gemma3-artifact".to_string(), inventory)])
}

fn findings(plan: &SystemPlan) -> Vec<&PlannedFinding> {
    plan.artifacts
        .iter()
        .flat_map(|a| a.findings.iter())
        .collect()
}

fn finding_for<'a>(plan: &'a SystemPlan, subject: &str) -> &'a PlannedFinding {
    findings(plan)
        .into_iter()
        .find(|f| f.subject == subject)
        .unwrap_or_else(|| panic!("no finding for `{subject}`"))
}

fn blocking_subjects(plan: &SystemPlan) -> Vec<String> {
    findings(plan)
        .into_iter()
        .filter(|f| f.blocks())
        .map(|f| f.subject.clone())
        .collect()
}

/// Whether the text-generation capability admits the plan, read through
/// the capability's serialised name so the test states the public
/// spelling a caller of `--text_only` relies on.
fn text_generation_admissible(plan: &SystemPlan) -> bool {
    plan.capabilities
        .iter()
        .find(|c| {
            serde_json::to_value(c.capability).unwrap() == serde_json::json!("text_generation")
        })
        .expect("text generation is always a planned capability")
        .admissible
}

/// The gate itself: text generation admits, and the only whole-model
/// blocker left is the vision tower's own surface. The class-default
/// geometry the config leaves implicit bound the estate on the way.
#[test]
fn the_gemma3_shaped_checkpoint_is_admissible_for_text_generation() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma3_shaped_target(dir.path());
    assert_eq!(
        (
            inventory.resolved.num_q_heads,
            inventory.resolved.num_kv_heads,
            inventory.resolved.head_dim
        ),
        (GEMMA3_Q_HEADS, GEMMA3_KV_HEADS, GEMMA3_HEAD_DIM),
        "the parser supplies HF's class defaults for the geometry the config omits"
    );
    let plan = plan_of(inventory);
    let blocking = blocking_subjects(&plan);
    assert!(
        text_generation_admissible(&plan),
        "text generation blocked on: {blocking:#?}"
    );
    assert_eq!(blocking, vec![VISION_SURFACE.to_string()]);
    assert!(
        !plan.admissible,
        "the vision tower still refuses the whole model"
    );
    assert_eq!(plan.summary.mismatched, 0);
}

/// `rope_scaling = {linear, 8.0}` reaches the full-attention layers as
/// `Linear` at the global base, the sliding layers rotate plain at the
/// local base, and both leaves are judged against the carried block.
/// Re-drop: resolve the full layers as plain rotary and `rope_type`
/// mismatches while `factor` becomes unrepresented — the seven-blocker
/// state the real checkpoint planned to.
#[test]
fn linear_rope_reaches_the_full_layers_and_both_leaves_are_judged_against_it() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma3_shaped_target(dir.path());
    for layer in GEMMA3_FULL_LAYERS {
        assert_eq!(
            inventory.resolved.layers[layer].position,
            PositionPolicy::Linear {
                theta: GEMMA3_GLOBAL_THETA,
                factor: GEMMA3_ROPE_FACTOR,
            },
            "layer {layer}"
        );
    }
    for layer in [0, 4, 6] {
        assert_eq!(
            inventory.resolved.layers[layer].position,
            PositionPolicy::Rope {
                theta: GEMMA3_LOCAL_THETA
            },
            "layer {layer}"
        );
    }
    let plan = plan_of(inventory);
    let kind = finding_for(&plan, ROPE_TYPE);
    assert_eq!(
        kind.category,
        FindingCategory::Representable,
        "{}",
        kind.detail
    );
    assert_eq!(kind.resolved, Some(serde_json::json!("linear")));
    let factor = finding_for(&plan, ROPE_FACTOR);
    assert_eq!(
        factor.category,
        FindingCategory::Representable,
        "{}",
        factor.detail
    );
    assert_eq!(factor.resolved, Some(serde_json::json!(GEMMA3_ROPE_FACTOR)));

    let dir = tempfile::tempdir().unwrap();
    let mut dropped = gemma3_shaped_target(dir.path());
    for layer in GEMMA3_FULL_LAYERS {
        dropped.resolved.layers[layer].position = PositionPolicy::Rope {
            theta: GEMMA3_GLOBAL_THETA,
        };
    }
    let plan = plan_of(dropped);
    let kind = finding_for(&plan, ROPE_TYPE);
    assert_eq!(
        kind.category,
        FindingCategory::Mismatched,
        "{}",
        kind.detail
    );
    let factor = finding_for(&plan, ROPE_FACTOR);
    assert_eq!(
        factor.category,
        FindingCategory::Unrepresented,
        "{}",
        factor.detail
    );
    assert!(!text_generation_admissible(&plan));
}

/// The config declares no `vocab_size`; the embedding's rows answer, the
/// answer names its tensor, and the text surface completes. Re-drop:
/// take the answer away and the placed head has no width — the surface
/// refuses on it rather than assuming the class default. (Removing the
/// embedding tensor is not the re-drop: with no embedding there is no
/// head object at all, and a headless surface is complete by design.)
#[test]
fn an_undeclared_vocab_is_answered_by_the_embedding_and_the_surface_completes() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma3_shaped_target(dir.path());
    assert_eq!(inventory.resolved.vocab_size, Some(GEMMA3_VOCAB));
    assert_eq!(
        inventory.resolved.vocab_size_provenance,
        Some(VocabSizeProvenance::EmbeddingRows {
            tensor: GEMMA3_EMBED_TENSOR.to_string(),
        })
    );
    let plan = plan_of(inventory);
    let surface = finding_for(&plan, TEXT_SURFACE);
    assert_eq!(
        surface.category,
        FindingCategory::Representable,
        "{}",
        surface.detail
    );
    assert!(surface.detail.contains("head"), "{}", surface.detail);

    let dir = tempfile::tempdir().unwrap();
    let mut dropped = gemma3_shaped_target(dir.path());
    dropped.resolved.vocab_size = None;
    dropped.resolved.vocab_size_provenance = None;
    let plan = plan_of(dropped);
    let surface = finding_for(&plan, TEXT_SURFACE);
    assert_eq!(
        surface.category,
        FindingCategory::Unrepresented,
        "{}",
        surface.detail
    );
    assert!(surface.detail.contains("vocab_size"), "{}", surface.detail);
    assert!(!text_generation_admissible(&plan));
}

/// Gemma 3's spellings of the image join are the interface they are —
/// read, classified, and required by the image capability alone — where
/// they used to grade `Unknown` and block every capability.
#[test]
fn gemma3_interface_spellings_are_the_image_binding_not_unknowns() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma3_shaped_target(dir.path()));
    for subject in INTERFACE_SUBJECTS {
        let f = finding_for(&plan, subject);
        assert_eq!(
            f.class,
            SemanticClass::InterfaceSemantic,
            "{subject}: {}",
            f.detail
        );
        assert!(!f.blocks(), "{subject}: {}", f.detail);
        assert_ne!(f.category, FindingCategory::Unrepresented, "{subject}");
    }
}

/// The SigLIP pooling-head flag is a fact about which tensors the tower
/// holds, not an unknown.
#[test]
fn vision_use_head_has_a_fate() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma3_shaped_target(dir.path()));
    let f = finding_for(&plan, "vision_config.vision_use_head");
    assert_eq!(f.class, SemanticClass::TensorSemantic, "{}", f.detail);
    assert!(!f.blocks(), "{}", f.detail);
}
