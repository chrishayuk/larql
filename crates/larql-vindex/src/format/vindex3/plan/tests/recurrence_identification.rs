//! Which recurrence a `linear_attention` layer runs is a *separate fact*
//! from the fact that it is recurrent, and only the geometry answers it.
//!
//! `layer_types` spells every linear-attention family the same way. A
//! build that reads the spelling and answers "Gated DeltaNet" has made a
//! claim about an operator from evidence about a label — so these tests
//! pair the two fixtures that differ in exactly one thing: whether the
//! geometry that identifies the operator was declared.
//!
//! The defect this pins was live, with two real subjects: GLM-5.3-Flash
//! (34 KDA layers) and Kimi Linear (20). Both spell their recurrence
//! geometry under `linear_attn_config.*` — keys no parser reads — while
//! `layer_types` says `linear_attention`. Both graded as executable Gated
//! DeltaNet while every `linear_attn_config` key graded `unrepresented`
//! in the same plan. See `docs/glm5-flash-funnel.md` §4.2.

use super::support::{
    declare_gated_delta_geometry, declare_hybrid_cadence, glimmer_shaped_target_with,
};
use crate::format::vindex3::graph::{build_from_inventories, LayerOperator};
use crate::format::vindex3::plan::{plan_system, Finding, FindingCategory};

/// The plan findings for the hybrid cadence, with the identifying
/// geometry declared or withheld — the one difference between the arms.
fn findings_with_geometry(declare_geometry: bool) -> Vec<Finding> {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        declare_hybrid_cadence(config);
        if declare_geometry {
            declare_gated_delta_geometry(config);
        }
    });
    plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

fn attention_policy(findings: &[Finding]) -> &Finding {
    findings
        .iter()
        .find(|f| f.subject == "attention_policy")
        .expect("attention policy finding")
}

/// A declared recurrence with **no geometry to identify it** is
/// [`LayerOperator::Recurrent`], not [`LayerOperator::GatedDelta`].
///
/// The span half must not regress with it: an unidentified recurrence
/// still has no prefix to bound, so it must not acquire the `Full` span a
/// KV planner would read as liveness.
#[test]
fn an_unidentified_recurrence_is_never_graded_gated_delta() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), declare_hybrid_cadence);
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    let table = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.attention.as_ref())
        .expect("the text component carries a per-layer table");

    let recurrent = table
        .iter()
        .filter(|l| l.operator == LayerOperator::Recurrent)
        .count();
    let gated_delta = table
        .iter()
        .filter(|l| l.operator == LayerOperator::GatedDelta)
        .count();
    assert_eq!(gated_delta, 0, "nothing here identifies Gated DeltaNet");
    assert!(recurrent > 0, "the declared recurrence must be recorded");

    for layer in table
        .iter()
        .filter(|l| l.operator == LayerOperator::Recurrent)
    {
        assert_eq!(layer.span, None, "a recurrence carries no span");
        assert!(
            !layer.matches_declaration(),
            "it must not round-trip to an executable spelling"
        );
    }
}

/// The paired positive: declaring the geometry — and changing nothing
/// else — turns the same layers into [`LayerOperator::GatedDelta`].
///
/// Without this arm the test above would also pass on a build that had
/// simply deleted `GatedDelta`.
#[test]
fn declaring_the_geometry_identifies_the_same_layers_as_gated_delta() {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        declare_hybrid_cadence(config);
        declare_gated_delta_geometry(config);
    });
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    let table = built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.attention.as_ref())
        .expect("the text component carries a per-layer table");

    assert_eq!(
        table
            .iter()
            .filter(|l| l.operator == LayerOperator::Recurrent)
            .count(),
        0,
        "the geometry resolved, so nothing is unidentified"
    );
    assert!(table
        .iter()
        .any(|l| l.operator == LayerOperator::GatedDelta));
}

/// The plan-level consequence: an unidentified recurrence makes
/// `attention_policy` **block** and stops it claiming any gated-delta
/// layers.
///
/// Without this the graph could carry `Recurrent` honestly while the plan
/// still graded the component representable — the original defect one
/// level up, where the summary was informational and never blocked.
#[test]
fn an_unidentified_recurrence_blocks_the_attention_policy() {
    let unidentified = findings_with_geometry(false);
    let policy = attention_policy(&unidentified);
    assert_eq!(policy.category, FindingCategory::Unrepresented);
    assert!(policy.blocks(), "{}", policy.detail);
    assert!(
        policy
            .detail
            .contains("recurrent layer(s) whose operator this build cannot identify"),
        "the disclosure must name what kind of failure this is: {}",
        policy.detail
    );
    assert!(
        !policy.detail.contains("gated-delta recurrent"),
        "nothing here is identified Gated DeltaNet: {}",
        policy.detail
    );

    // Same fixture, geometry declared: it resolves and stops blocking.
    let identified = findings_with_geometry(true);
    let policy = attention_policy(&identified);
    assert_eq!(policy.category, FindingCategory::Representable);
    assert!(!policy.blocks());
    assert!(
        policy.detail.contains("gated-delta recurrent"),
        "{}",
        policy.detail
    );
}

/// `represented` and `executable` are independent facts, and the plan
/// must state both.
///
/// The regression this guards is the one P1 fixed one level down: a plan
/// that reports only "representable" invites a reader to conclude the
/// stack runs. KDA is the first operator to occupy the gap — fully
/// described, no executor — so it is the case that proves the two facts
/// are reported separately rather than derived from one another.
#[test]
fn a_represented_operator_without_an_executor_says_so() {
    use crate::format::vindex3::graph::LayerOperator;

    // The executor axis, stated directly on the operator.
    assert!(LayerOperator::Softmax.has_executor());
    assert!(LayerOperator::GatedDelta.has_executor());
    assert!(
        !LayerOperator::Kda.has_executor(),
        "KDA is represented, not executable"
    );
    assert!(!LayerOperator::Recurrent.has_executor());

    // And it is orthogonal to identification: KDA is identified (it is not
    // the unidentified-recurrence variant) yet still has no executor.
    assert!(!LayerOperator::Kda.is_unidentified_recurrence());
    assert!(LayerOperator::Kda.is_recurrent());
    assert!(LayerOperator::GatedDelta.is_recurrent());
    assert!(LayerOperator::Recurrent.is_recurrent());
    assert!(!LayerOperator::Softmax.is_recurrent());
}

/// The two-set interleave spelling carries, and so does the KDA conv width
/// beside it.
///
/// The set probes compare by **cardinality against the resolved table**
/// rather than re-rendering the indices: the declaration's index base is a
/// fact the resolver already proved, and re-deriving it here would be two
/// implementations of one rule. Both sets close against the same table, so
/// a resolution that dropped or doubled a layer fails one of them.
#[test]
fn the_two_set_spelling_and_the_kda_geometry_both_carry() {
    use crate::format::vindex3::plan::{plan_system, FindingCategory};

    const LAYERS: usize = 8;
    const CONV_KERNEL: usize = 4;

    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        // Zero-based, partitioning the fixture's eight layers 6:2.
        let full: Vec<usize> = (0..LAYERS).filter(|i| i % 4 == 3).collect();
        let kda: Vec<usize> = (0..LAYERS).filter(|i| i % 4 != 3).collect();
        config["text_config"]["linear_attn_config"] = serde_json::json!({
            "kda_layers": kda,
            "full_attn_layers": full,
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": CONV_KERNEL,
        });
        // `layer_types` would take precedence, so it must be absent for
        // the set spelling to be the one under test.
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("layer_types");
    });
    let findings: Vec<Finding> = plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect();

    for leaf in ["kda_layers", "full_attn_layers", "short_conv_kernel_size"] {
        let f = findings
            .iter()
            .find(|f| f.subject.ends_with(leaf))
            .unwrap_or_else(|| panic!("no finding for `{leaf}`"));
        assert_eq!(f.category, FindingCategory::Representable, "{leaf}: {f:?}");
        assert!(!f.blocks(), "{leaf}: {f:?}");
    }

    // The operator resolves to KDA, so the geometry above is what named it.
    let policy = attention_policy(&findings);
    assert!(policy.detail.contains("KDA recurrent"), "{}", policy.detail);
}
