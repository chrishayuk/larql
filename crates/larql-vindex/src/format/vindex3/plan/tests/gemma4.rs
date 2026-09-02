//! V3-F0 witness 3 — Gemma 4 through the plan gate (G4.0): every hostile
//! semantic the family declares is REPRESENTED, judged against what the
//! built graph holds, and each finding is pinned by re-dropping the fact
//! it guards. The miniature mirrors the real 26B-A4B checkpoint's config
//! shape and tensor spelling (`plan::tests_support::gemma4_shaped_target`).

use larql_models::config::{PositionPolicy, RotaryFrequencyBasis};

use crate::format::vindex3::plan::tests_support::{
    gemma4_shaped_target, gemma4_shaped_target_with, GEMMA4_FIXTURE_LAYERS, GEMMA4_FULL_LAYER,
    GEMMA4_FULL_THETA, GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_KV_HEADS, GEMMA4_HEAD_DIM,
    GEMMA4_KV_HEADS, GEMMA4_PARTIAL_ROTARY, GEMMA4_SLIDING_THETA, GEMMA4_TOP_K,
};
use crate::format::vindex3::plan::{
    plan_system, Finding, FindingCategory, PlannedFinding, SystemPlan,
};

fn plan_of(inventory: larql_models::inventory::ArchitectureInventory) -> SystemPlan {
    plan_system(&[("gemma4-artifact".to_string(), inventory)])
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

/// The theta a rope finding resolved to: the resolution comparator
/// reports the whole per-layer policy, the carriage probe the bare base;
/// both name the same number.
fn resolved_theta(finding: &Finding) -> Option<f64> {
    let resolved = finding.resolved.as_ref()?;
    resolved
        .get("theta")
        .and_then(serde_json::Value::as_f64)
        .or_else(|| resolved.as_f64())
}

/// The gate itself: the whole family plans admissibly — every declared
/// fact representable, none dropped, none disagreeing.
#[test]
fn the_gemma4_shaped_checkpoint_plans_admissibly() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    let blocking: Vec<String> = findings(&plan)
        .into_iter()
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(plan.admissible, "blocking findings: {blocking:#?}");
    assert_eq!(plan.summary.mismatched, 0);
    assert_eq!(plan.summary.unrepresented, 0);
}

/// Per-layer head geometry: the graph records each layer's own
/// `head_dim` / KV-head count, the full layer's differing from the
/// component's, and `global_head_dim` / `num_global_key_value_heads` are
/// judged against the full layers alone.
#[test]
fn per_layer_head_geometry_is_carried_and_the_global_facts_judged_against_full_layers() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    let target = plan
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .expect("text component");
    let table = target.attention.as_ref().expect("policy table");
    assert_eq!(table.len(), GEMMA4_FIXTURE_LAYERS);
    for (i, layer) in table.iter().enumerate() {
        let g = layer.geometry.expect("geometry recorded on every layer");
        let expected = if i == GEMMA4_FULL_LAYER {
            (GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_KV_HEADS)
        } else {
            (GEMMA4_HEAD_DIM, GEMMA4_KV_HEADS)
        };
        assert_eq!((g.head_dim, g.num_kv_heads), expected, "layer {i}");
    }
    let ghd = finding_for(&plan, "text_config.global_head_dim");
    assert_eq!(ghd.category, FindingCategory::Representable);
    assert_eq!(
        ghd.resolved,
        Some(serde_json::json!(GEMMA4_GLOBAL_HEAD_DIM))
    );
    let gkv = finding_for(&plan, "text_config.num_global_key_value_heads");
    assert_eq!(
        gkv.resolved,
        Some(serde_json::json!(GEMMA4_GLOBAL_KV_HEADS))
    );
}

/// Per-layer-type rope: each declared theta is judged against the layers
/// of its own type. Re-drop: change the resolved full layer's theta and
/// only the `full_attention` declaration mismatches; the sliding one
/// still agrees.
#[test]
fn per_layer_type_rope_thetas_are_judged_against_their_own_layers() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    let full = finding_for(
        &plan,
        "text_config.rope_parameters.full_attention.rope_theta",
    );
    let sliding = finding_for(
        &plan,
        "text_config.rope_parameters.sliding_attention.rope_theta",
    );
    assert_eq!(
        full.category,
        FindingCategory::Representable,
        "{}",
        full.detail
    );
    assert_eq!(
        sliding.category,
        FindingCategory::Representable,
        "{}",
        sliding.detail
    );
    assert_eq!(resolved_theta(full), Some(GEMMA4_FULL_THETA));
    assert_eq!(resolved_theta(sliding), Some(GEMMA4_SLIDING_THETA));

    let dir = tempfile::tempdir().unwrap();
    let mut dropped = gemma4_shaped_target(dir.path());
    let wrong_theta = 123.0;
    dropped.resolved.layers[GEMMA4_FULL_LAYER].position =
        PositionPolicy::Rope { theta: wrong_theta };
    let plan = plan_of(dropped);
    let full = finding_for(
        &plan,
        "text_config.rope_parameters.full_attention.rope_theta",
    );
    let sliding = finding_for(
        &plan,
        "text_config.rope_parameters.sliding_attention.rope_theta",
    );
    assert_eq!(
        full.category,
        FindingCategory::Mismatched,
        "{}",
        full.detail
    );
    assert_eq!(
        sliding.category,
        FindingCategory::Representable,
        "{}",
        sliding.detail
    );
    assert!(!plan.admissible);
}

/// The proportional partial rotary is a policy variant on the full
/// layers, and both its leaves (`rope_type: proportional`,
/// `partial_rotary_factor`) are judged against it. Re-drop: resolve the
/// full layer as plain rotary and `rope_type` mismatches (`default` vs
/// `proportional`) while `partial_rotary_factor` becomes unrepresented.
#[test]
fn proportional_partial_rotary_is_a_policy_the_full_layers_carry() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma4_shaped_target(dir.path());
    assert_eq!(
        inventory.resolved.layers[GEMMA4_FULL_LAYER].position,
        PositionPolicy::PartialRope {
            theta: GEMMA4_FULL_THETA,
            rotary_fraction: GEMMA4_PARTIAL_ROTARY,
            basis: RotaryFrequencyBasis::HeadWidth,
        }
    );
    assert_eq!(
        inventory.resolved.layers[0].position,
        PositionPolicy::Rope {
            theta: GEMMA4_SLIDING_THETA
        }
    );
    let plan = plan_of(inventory);
    let kind = finding_for(
        &plan,
        "text_config.rope_parameters.full_attention.rope_type",
    );
    assert_eq!(
        kind.category,
        FindingCategory::Representable,
        "{}",
        kind.detail
    );
    assert_eq!(kind.resolved, Some(serde_json::json!("proportional")));
    let fraction = finding_for(
        &plan,
        "text_config.rope_parameters.full_attention.partial_rotary_factor",
    );
    assert_eq!(
        fraction.category,
        FindingCategory::Representable,
        "{}",
        fraction.detail
    );
    assert_eq!(
        fraction.resolved,
        Some(serde_json::json!(GEMMA4_PARTIAL_ROTARY))
    );
    // The sliding declaration is the default class and says so.
    let sliding_kind = finding_for(
        &plan,
        "text_config.rope_parameters.sliding_attention.rope_type",
    );
    assert_eq!(sliding_kind.resolved, Some(serde_json::json!("default")));

    let dir = tempfile::tempdir().unwrap();
    let mut dropped = gemma4_shaped_target(dir.path());
    dropped.resolved.layers[GEMMA4_FULL_LAYER].position = PositionPolicy::Rope {
        theta: GEMMA4_FULL_THETA,
    };
    let plan = plan_of(dropped);
    let kind = finding_for(
        &plan,
        "text_config.rope_parameters.full_attention.rope_type",
    );
    assert_eq!(
        kind.category,
        FindingCategory::Mismatched,
        "{}",
        kind.detail
    );
    let fraction = finding_for(
        &plan,
        "text_config.rope_parameters.full_attention.partial_rotary_factor",
    );
    assert_eq!(
        fraction.category,
        FindingCategory::Unrepresented,
        "{}",
        fraction.detail
    );
    assert!(!plan.admissible);
}

/// K≡V is carried per layer and judged. Re-drop: resolve every layer as
/// projecting its own V and the declaration mismatches.
#[test]
fn k_eq_v_is_carried_on_the_full_layer_and_judged() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma4_shaped_target(dir.path());
    let shares: Vec<bool> = inventory
        .resolved
        .layers
        .iter()
        .map(|l| l.v_from_k)
        .collect();
    let mut expected = vec![false; GEMMA4_FIXTURE_LAYERS];
    expected[GEMMA4_FULL_LAYER] = true;
    assert_eq!(shares, expected);
    // The parameter-free V norm rides every layer.
    assert!(
        inventory
            .resolved
            .execution
            .as_ref()
            .unwrap()
            .parameter_free_qk_norm
            .v
    );
    let plan = plan_of(inventory);
    let k_eq_v = finding_for(&plan, "text_config.attention_k_eq_v");
    assert_eq!(
        k_eq_v.category,
        FindingCategory::Representable,
        "{}",
        k_eq_v.detail
    );
    assert_eq!(k_eq_v.resolved, Some(serde_json::json!(true)));

    let dir = tempfile::tempdir().unwrap();
    let mut dropped = gemma4_shaped_target(dir.path());
    for layer in &mut dropped.resolved.layers {
        layer.v_from_k = false;
    }
    let plan = plan_of(dropped);
    let k_eq_v = finding_for(&plan, "text_config.attention_k_eq_v");
    assert_eq!(
        k_eq_v.category,
        FindingCategory::Mismatched,
        "{}",
        k_eq_v.detail
    );
}

/// The knobs the graph represents only as absent agree at their absent
/// value and block at any other: PLE width, the double-wide MLP,
/// KV-shared layers. Each is a declared fact, so the mismatch names the
/// declared value rather than dropping it.
#[test]
fn absent_only_knobs_agree_when_off_and_block_when_on() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    for subject in [
        "text_config.hidden_size_per_layer_input",
        "text_config.use_double_wide_mlp",
        "text_config.num_kv_shared_layers",
        "text_config.vocab_size_per_layer_input",
    ] {
        let f = finding_for(&plan, subject);
        assert_eq!(
            f.category,
            FindingCategory::Representable,
            "{subject}: {}",
            f.detail
        );
    }

    let dir = tempfile::tempdir().unwrap();
    let ple_width = 256;
    let shared_layers = 2;
    let plan = plan_of(gemma4_shaped_target_with(
        dir.path(),
        |config| {
            config["text_config"]["hidden_size_per_layer_input"] = serde_json::json!(ple_width);
            config["text_config"]["use_double_wide_mlp"] = serde_json::json!(true);
            config["text_config"]["num_kv_shared_layers"] = serde_json::json!(shared_layers);
        },
        |_| {},
    ));
    for subject in [
        "text_config.hidden_size_per_layer_input",
        "text_config.use_double_wide_mlp",
        "text_config.num_kv_shared_layers",
    ] {
        let f = finding_for(&plan, subject);
        assert_eq!(
            f.category,
            FindingCategory::Mismatched,
            "{subject}: {}",
            f.detail
        );
        assert!(f.blocks());
    }
    assert!(!plan.admissible);
}

/// The hybrid-MoE declarations are judged against the routed surface:
/// `enable_moe_block` against the presence of a MoE judgment, `top_k_experts`
/// against its top-k. Re-drop: withdraw the block and both change.
#[test]
fn the_hybrid_moe_declarations_are_judged_against_the_routed_surface() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    let enabled = finding_for(&plan, "text_config.enable_moe_block");
    assert_eq!(enabled.resolved, Some(serde_json::json!(true)));
    let top_k = finding_for(&plan, "text_config.top_k_experts");
    assert_eq!(top_k.resolved, Some(serde_json::json!(GEMMA4_TOP_K)));

    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target_with(
        dir.path(),
        |config| {
            config["text_config"]["enable_moe_block"] = serde_json::json!(false);
        },
        |tensors| {
            tensors.retain(|(name, _)| !name.contains(".experts.") && !name.contains(".router."));
        },
    ));
    let enabled = finding_for(&plan, "text_config.enable_moe_block");
    assert_eq!(enabled.resolved, Some(serde_json::json!(false)));
    assert_eq!(enabled.category, FindingCategory::Representable);
}

/// The vision tower declares no `layer_types`: its table is full
/// attention on every declared layer, so its rope base and class are
/// judged (they were dropped when no table existed), and its
/// `attention_bias` reaches the tower surface.
#[test]
fn a_tower_without_layer_types_gets_a_full_table_and_its_facts_are_judged() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    let vision = plan
        .graph
        .components
        .iter()
        .find(|c| c.id == "vision")
        .expect("vision component");
    let table = vision.attention.as_ref().expect("tower table");
    assert_eq!(table.len(), 2);
    assert!(table
        .iter()
        .all(|l| l.span == Some(crate::format::vindex3::graph::policy::AttentionSpan::Full)));
    for subject in [
        "vision_config.rope_parameters.rope_theta",
        "vision_config.rope_parameters.rope_type",
        "vision_config.attention_bias",
        "vision_config.hidden_activation",
        "vision_config.global_head_dim",
    ] {
        let f = finding_for(&plan, subject);
        assert_eq!(
            f.category,
            FindingCategory::Representable,
            "{subject}: {}",
            f.detail
        );
    }
    assert_eq!(
        finding_for(&plan, "vision_config.rope_parameters.rope_theta").resolved,
        Some(serde_json::json!(100.0))
    );
    // The activation alias resolves in the checkpoint's own spelling.
    assert_eq!(
        finding_for(&plan, "vision_config.hidden_activation").resolved,
        Some(serde_json::json!("gelu_pytorch_tanh"))
    );
}

/// The multimodal interface and the tower's metadata are read and
/// classified: nothing at the root or under `id2label` is "read by
/// nothing".
#[test]
fn the_multimodal_interface_and_tower_metadata_are_read_and_classified() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    for subject in [
        "audio_config",
        "audio_token_id",
        "eoa_token_index",
        "vision_soft_tokens_per_image",
        "text_config.use_bidirectional_attention",
        "vision_config.id2label.0",
        "vision_config.label2id.LABEL_0",
        "vision_config.chunk_size_feed_forward",
        "vision_config.pooling_kernel_size",
        "vision_config.use_clipped_linears",
    ] {
        let f = finding_for(&plan, subject);
        assert_eq!(
            f.category,
            FindingCategory::Representable,
            "{subject}: {}",
            f.detail
        );
        assert!(
            !f.detail.contains("read by nothing"),
            "{subject} must be read: {}",
            f.detail
        );
    }
}

/// **Control for the duplicate-spelling gate (QW-3.5B).**
///
/// Gemma 4 declares `rope_theta` and `rope_type` at BOTH
/// `rope_parameters.full_attention.*` and
/// `rope_parameters.sliding_attention.*`, with deliberately different
/// values — two facts about two layer classes, not one fact spelled
/// twice. A first cut of the gate matched on "same leaf, same component"
/// and flagged all of it, turning eight Gemma 4 tests red.
///
/// The gate is a registered pair list because of this case. Without this
/// test the narrowing looks arbitrary, and the next person to reach for
/// the heuristic version has nothing telling them why it fails.
#[test]
fn a_per_layer_class_pair_is_not_a_duplicate_spelling() {
    let dir = tempfile::tempdir().unwrap();
    let plan = plan_of(gemma4_shaped_target(dir.path()));
    let flagged: Vec<&str> = findings(&plan)
        .into_iter()
        .filter(|f| f.subject.contains("two spellings"))
        .map(|f| f.subject.as_str())
        .collect();
    assert!(
        flagged.is_empty(),
        "per-layer-class rope facts are distinct facts, not duplicate spellings: {flagged:?}"
    );
    // And the pair really is present and really does disagree — without
    // this the assertion above would also pass on a fixture that simply
    // never declared them.
    let thetas: Vec<f64> = findings(&plan)
        .into_iter()
        .filter(|f| f.subject.ends_with("rope_theta"))
        .filter_map(|f| f.declared.as_ref().and_then(serde_json::Value::as_f64))
        .collect();
    assert!(
        thetas.contains(&GEMMA4_FULL_THETA) && thetas.contains(&GEMMA4_SLIDING_THETA),
        "the fixture must actually declare two disagreeing thetas: {thetas:?}"
    );
}
