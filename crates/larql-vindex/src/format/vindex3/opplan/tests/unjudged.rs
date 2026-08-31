//! Absence stays absence: withdrawing a declaration must not reappear as
//! a plausible identity.
//!
//! Each control removes one declared fact from an otherwise-complete
//! four-norm fixture and asserts the plan reports the operation as
//! *absent* rather than as a multiply by one. The failure these guard
//! against is not a crash — it is an ingestion regression producing a
//! fully executable program that is quietly wrong, which no downstream
//! authority can detect because every authority agrees on the default.

use larql_models::inventory::ArchitectureInventory;

use crate::format::vindex3::encode::encode_system_unenforced as encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect, OpPlanOutcome};
use crate::format::vindex3::plan::tests_support::{drafter_shaped, glimmer_shaped_target_with};

/// The text-config key each control withdraws.
const QK_SCALE_FACTOR: &str = "qk_scale_factor";
const OUTPUT_MULTIPLIER: &str = "output_multiplier";
const POST_NORM_EPS: &str = "post_norm_eps";
const TEXT_CONFIG: &str = "text_config";

/// Plan the target component of the Glimmer-shaped fixture, with `mutate`
/// applied to its config first.
fn plan_without(mutate: impl FnOnce(&mut serde_json::Value)) -> OpPlanOutcome {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named: Vec<(String, ArchitectureInventory)> = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target_with(target_dir.path(), mutate),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    let container = tempfile::tempdir().unwrap();
    encode_system(&named, container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    plan_component_ops(&inspection, container.path(), "target").unwrap()
}

/// Remove one key from the fixture's text config.
fn drop_text_key(config: &mut serde_json::Value, key: &str) {
    config[TEXT_CONFIG]
        .as_object_mut()
        .expect("fixture text_config is an object")
        .remove(key)
        .unwrap_or_else(|| panic!("fixture must declare `{key}` for this control to withdraw it"));
}

/// Baseline: with every declaration present, all three operations exist.
/// Without this the three withdrawal controls could pass on a fixture
/// that never carried the facts at all.
#[test]
fn the_intact_fixture_declares_all_three_operations() {
    let outcome = plan_without(|_| {});
    let plan = outcome.plan.expect("intact fixture plans");
    assert_eq!(
        plan.layers[0].attention.softmax().unwrap().query_scale,
        Some(3.87)
    );
    assert_eq!(plan.output.as_ref().unwrap().multiplier, Some(0.196));
    // The fixture's family declares no embedding-scale operation, so the
    // embedding control below asserts a state this baseline pins as
    // already-absent rather than newly-absent.
    assert_eq!(plan.embedding.as_ref().unwrap().scale, None);
}

/// Withdrawing `qk_scale_factor` yields no query-scale op — never 1.0.
///
/// This is the sharpest of the four: Muse-Glimmer's declared 3.87 is a
/// parity-critical fact, and 1.0 is indistinguishable from a correct
/// answer on the many models that have no query scale at all.
#[test]
fn a_withdrawn_query_scale_is_absent_not_identity() {
    let outcome = plan_without(|config| drop_text_key(config, QK_SCALE_FACTOR));
    let plan = outcome.plan.expect("withdrawing a scale must not block");
    for layer in &plan.layers {
        assert_eq!(
            layer.attention.softmax().unwrap().query_scale,
            None,
            "layer {}: withdrawn query scale came back as a value",
            layer.layer
        );
    }
}

/// Withdrawing `output_multiplier` yields no multiplier op — never 1.0.
#[test]
fn a_withdrawn_output_multiplier_is_absent_not_identity() {
    let outcome = plan_without(|config| drop_text_key(config, OUTPUT_MULTIPLIER));
    let plan = outcome
        .plan
        .expect("withdrawing a multiplier must not block");
    let output = plan.output.expect("target owns an output head");
    assert_eq!(output.multiplier, None);
}

/// A model declaring no embedding multiplier yields no embedding-scale
/// op — never 1.0. Paired with the Gemma families, whose computed
/// `sqrt(hidden_size)` must still arrive as `Some`.
#[test]
fn an_undeclared_embedding_scale_is_absent_not_identity() {
    let outcome = plan_without(|_| {});
    let plan = outcome.plan.expect("intact fixture plans");
    let embedding = plan.embedding.expect("target owns an embedding table");
    assert_eq!(embedding.scale, None);
}

/// A four-norm stack whose post-norm epsilon nothing established is
/// refused, with the fact and the structure requiring it both named.
///
/// The other three controls tolerate absence because absence is a
/// meaningful answer — the operation simply is not there. This one does
/// not: the post-norms exist and *must* apply some epsilon, so an
/// unjudged value is a hole in execution sufficiency, not an absent op.
#[test]
fn a_four_norm_stack_refuses_an_unjudged_post_norm_eps() {
    let outcome = plan_without(|config| drop_text_key(config, POST_NORM_EPS));
    assert!(
        outcome.plan.is_none(),
        "an unjudged post-norm epsilon must not produce a plan"
    );
    let (fact, required_by) = outcome
        .defects
        .iter()
        .find_map(|d| match d {
            ClosureDefect::UnjudgedSemantic {
                fact, required_by, ..
            } => Some((fact.clone(), required_by.clone())),
            _ => None,
        })
        .expect("the unjudged epsilon must be reported");
    assert!(fact.contains("post-norm epsilon"), "fact was {fact}");
    assert!(
        required_by.contains("four-norm"),
        "required_by was {required_by}"
    );
}

/// The refusal names its cause in the rendered message, not just in the
/// variant — a defect a human cannot read is a defect they will ignore.
#[test]
fn the_unjudged_refusal_renders_its_cause() {
    let outcome = plan_without(|config| drop_text_key(config, POST_NORM_EPS));
    let rendered = outcome
        .defects
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("post-norm epsilon"), "{rendered}");
    assert!(rendered.contains("not judged"), "{rendered}");
    assert!(rendered.contains("four-norm placement"), "{rendered}");
}
