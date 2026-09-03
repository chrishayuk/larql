//! One MoE surface, reached from either family's spelling.
//!
//! Kimi Linear writes `num_shared_experts`, `moe_renormalize`,
//! `moe_router_activation_func`, `num_expert_group`; the DeepSeek lineage
//! writes `n_shared_experts`, `norm_topk_prob`, `scoring_func`, `n_group`.
//! Reading only one set leaves the other checkpoint's facts declared and
//! unread — which, before this, meant Kimi Linear's 256 experts produced
//! an execution surface saying `ffn: dense`.

use super::support::gemma4_shaped_target_with;
use crate::format::vindex3::plan::{plan_system, FindingCategory, PlannedFinding};

const BRANCH_SCALE: f64 = 2.446;
const DENSE_PREFIX: usize = 1;

/// Kimi Linear's spellings, on a fixture that already carries an MoE.
fn kimi_spellings(config: &mut serde_json::Value) {
    let text = &mut config["text_config"];
    text["num_shared_experts"] = serde_json::json!(1);
    // The fixture's architecture selects experts then softmaxes over all
    // of them, so it does NOT renormalise over the selected set. Declaring
    // that truthfully is what makes this carry; the contradiction case is
    // its own test below.
    text["moe_renormalize"] = serde_json::json!(false);
    text["num_expert_group"] = serde_json::json!(1);
    text["topk_group"] = serde_json::json!(1);
    text["use_grouped_topk"] = serde_json::json!(true);
    text["moe_layer_freq"] = serde_json::json!(1);
    text["routed_scaling_factor"] = serde_json::json!(BRANCH_SCALE);
    text["first_k_dense_replace"] = serde_json::json!(DENSE_PREFIX);
}

fn findings_with(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    let inventory = gemma4_shaped_target_with(dir.path(), mutate, |_| {});
    plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

fn finding<'a>(findings: &'a [PlannedFinding], leaf: &str) -> &'a PlannedFinding {
    findings
        .iter()
        .find(|f| f.subject.ends_with(leaf))
        .unwrap_or_else(|| panic!("no finding for `{leaf}`"))
}

#[test]
fn every_kimi_moe_spelling_reaches_the_surface() {
    let findings = findings_with(kimi_spellings);
    for leaf in [
        "num_shared_experts",
        "moe_renormalize",
        "num_expert_group",
        "topk_group",
        "use_grouped_topk",
        "moe_layer_freq",
        "routed_scaling_factor",
        "first_k_dense_replace",
    ] {
        let f = finding(&findings, leaf);
        assert_eq!(f.category, FindingCategory::Representable, "{leaf}: {f:?}");
        assert!(!f.blocks(), "{leaf}: {f:?}");
    }
}

/// The two facts that became real surface fields answer with their own
/// declared values, not with a placeholder.
#[test]
fn the_branch_scale_and_dense_prefix_carry_their_values() {
    let findings = findings_with(kimi_spellings);
    assert_eq!(
        finding(&findings, "routed_scaling_factor").resolved,
        Some(serde_json::json!(BRANCH_SCALE))
    );
    assert_eq!(
        finding(&findings, "first_k_dense_replace").resolved,
        Some(serde_json::json!(DENSE_PREFIX))
    );
}

/// **The negative arm.** Grouping is representable only because one group
/// is ungrouped routing. A checkpoint declaring real groups states
/// something this schema cannot, and must block.
#[test]
fn more_than_one_expert_group_blocks() {
    let findings = findings_with(|config| {
        kimi_spellings(config);
        config["text_config"]["num_expert_group"] = serde_json::json!(8);
        config["text_config"]["topk_group"] = serde_json::json!(3);
    });
    for leaf in ["num_expert_group", "topk_group"] {
        let f = finding(&findings, leaf);
        assert!(
            f.blocks(),
            "{leaf} at a non-identity value must block: {f:?}"
        );
    }
}

/// The renormalisation flag reaches the surface, and a checkpoint that
/// declares the opposite gets the opposite policy.
///
/// Deliberately **not** asserted as a contradiction test: the routing
/// policy is derived from this key, so declared and resolved cannot
/// disagree and any "mismatch" assertion here would be a gate that cannot
/// fail. What is testable — and tested — is that the two settings produce
/// two different policies rather than one default.
#[test]
fn the_renormalisation_flag_selects_the_policy_it_declares() {
    for declared in [true, false] {
        let findings = findings_with(|config| {
            kimi_spellings(config);
            config["text_config"]["moe_renormalize"] = serde_json::json!(declared);
        });
        let f = finding(&findings, "moe_renormalize");
        assert_eq!(f.category, FindingCategory::Representable, "{f:?}");
        assert_eq!(
            f.resolved,
            Some(serde_json::json!(declared)),
            "the surface must follow the declaration, not a default"
        );
    }
}

/// A declared router activation that an architecture's own override
/// contradicts is caught as a mismatch.
///
/// The fixture's family hardcodes its router kind, so declaring `sigmoid`
/// beside it is a real disagreement between the checkpoint and the build —
/// exactly what the declared-vs-resolved comparison exists to surface, and
/// the reason the router kind is not simply believed from the string.
#[test]
fn a_router_activation_the_architecture_contradicts_is_caught() {
    let findings = findings_with(|config| {
        kimi_spellings(config);
        config["text_config"]["moe_router_activation_func"] = serde_json::json!("sigmoid");
    });
    let f = finding(&findings, "moe_router_activation_func");
    assert_eq!(f.category, FindingCategory::Mismatched, "{f:?}");
    assert!(f.blocks(), "{f:?}");
}

/// The shared branch's WIDTH is judged against the width the branch will
/// be built at — never waved through because a parser read it.
///
/// The two arms are the whole point. `shared_expert_intermediate_size` is
/// a consumed key either way, so "the parser reads it" cannot be the
/// evidence: what separates them is whether anything resolved a branch
/// for the width to describe.
#[test]
fn the_shared_expert_width_carries_only_where_a_branch_resolves() {
    const DECLARED: usize = 4096;

    // With a shared branch declared, the width reaches the surface and is
    // reported as the value the checkpoint stated.
    let with_branch = findings_with(|config| {
        config["text_config"]["n_shared_experts"] = serde_json::json!(1);
        config["text_config"]["shared_expert_intermediate_size"] = serde_json::json!(DECLARED);
    });
    let f = finding(&with_branch, "shared_expert_intermediate_size");
    assert_eq!(f.category, FindingCategory::Representable, "{f:?}");
    assert_eq!(f.resolved, Some(serde_json::json!(DECLARED)), "{f:?}");
    assert!(!f.blocks(), "{f:?}");

    // Without one, nothing resolves a width for it to be. The key is
    // still consumed — and it must still REFUSE, because a size for a
    // branch this build does not run is a fact that reached no operator.
    let without_branch = findings_with(|config| {
        config["text_config"]["shared_expert_intermediate_size"] = serde_json::json!(DECLARED);
    });
    let f = finding(&without_branch, "shared_expert_intermediate_size");
    assert_eq!(f.category, FindingCategory::Unrepresented, "{f:?}");
    assert!(
        f.blocks(),
        "a width with no branch to size must block, not pass as parsed: {f:?}"
    );
}
