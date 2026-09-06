//! K3-ACT-1: `hidden_act: "situ"` and its two softcaps, on the plan.
//!
//! Every test pairs against the state main was in before this rung, which
//! is the only reason the numbers here mean anything:
//!
//! ```text
//! hidden_act                  declared "situ"  ->  resolved "silu"   MISMATCHED
//! activation_situ_beta        declared 4.0     ->  read by nothing   UNKNOWN
//! activation_situ_linear_beta declared 25.0    ->  read by nothing   UNKNOWN
//! ```
//!
//! The mismatch is the interesting one. It was not a reporting bug — the
//! plan was telling the truth, because `ModelArchitecture::activation`
//! genuinely did resolve `situ` to SiLU. Retiring it therefore required
//! changing what the build COMPUTES, not what it says; the assertion
//! below that `resolved` reads `situ` is downstream of a real
//! `ExpertGatePolicy::SituGlu` on the surface, and there is no way to
//! satisfy it by adjusting the probe alone.

use super::support::glimmer_shaped_target_with;
use crate::format::vindex3::plan::carriage::{rule_for, Carriage};
use crate::format::vindex3::plan::{plan_system, FindingCategory, PlannedFinding, SemanticClass};

/// K3's declared parameters, from `moonshotai/Kimi-K3`'s `config.json`.
const BETA: f64 = 4.0;
const LINEAR_BETA: f64 = 25.0;

fn plan_with(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![(
        "target-artifact".to_string(),
        glimmer_shaped_target_with(dir.path(), mutate),
    )];
    plan_system(&named)
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

/// The SiTU declaration as K3 ships it: the name and both softcaps.
fn declare_situ(config: &mut serde_json::Value) {
    config["text_config"]["hidden_act"] = serde_json::json!("situ");
    config["text_config"]["activation_situ_beta"] = serde_json::json!(BETA);
    config["text_config"]["activation_situ_linear_beta"] = serde_json::json!(LINEAR_BETA);
}

/// The TEXT component's finding for one leaf, by exact path.
///
/// Exact rather than `ends_with`: this fixture's vision tower declares its
/// own `hidden_act`, and a suffix match would silently answer with the
/// vision component's finding on any test where the text one is absent —
/// passing for the wrong reason.
fn finding_for<'a>(findings: &'a [PlannedFinding], leaf: &str) -> &'a PlannedFinding {
    let path = format!("text_config.{leaf}");
    findings
        .iter()
        .find(|f| f.subject == path)
        .unwrap_or_else(|| panic!("no text finding for `{leaf}`"))
}

/// The declared name resolves to itself, not to SiLU.
///
/// Before this rung the probe answered `silu` — truthfully, because the
/// build computed SiLU. The assertion is on `resolved` rather than on the
/// finding's category alone, so an implementation that fixed the category
/// while still computing SwiGLU could not pass it.
#[test]
fn a_declared_situ_activation_resolves_to_situ() {
    let findings = plan_with(declare_situ);
    let finding = finding_for(&findings, "hidden_act");

    assert_eq!(finding.declared, Some(serde_json::json!("situ")));
    assert_eq!(
        finding.resolved,
        Some(serde_json::json!("situ")),
        "the FFN's combine is SiTU and the probe must answer from the combine"
    );
    assert_eq!(finding.category, FindingCategory::Representable);
    assert!(
        !finding.blocks(),
        "a carried activation must not block text closure"
    );
}

/// Both softcaps reach the op plan, and say so.
#[test]
fn both_situ_softcaps_are_carried_to_the_op_plan() {
    let findings = plan_with(declare_situ);

    for (leaf, declared) in [
        ("activation_situ_beta", serde_json::json!(BETA)),
        (
            "activation_situ_linear_beta",
            serde_json::json!(LINEAR_BETA),
        ),
    ] {
        let finding = finding_for(&findings, leaf);
        assert_eq!(finding.declared, Some(declared.clone()), "{leaf} declared");
        assert_eq!(finding.resolved, Some(declared), "{leaf} resolved");
        assert_eq!(finding.category, FindingCategory::Representable, "{leaf}");
        assert_eq!(
            finding.carriage,
            Some(Carriage::Lowered),
            "{leaf} reaches the op plan as a parameter of the combine"
        );
        assert!(!finding.blocks(), "{leaf} must not block");
    }
}

/// **The negative arm.** The softcaps beside an activation that does not
/// name SiTU find no home, and the plan says so rather than applying them.
///
/// This is what makes the positive arm above discriminative: the beta keys
/// are carried because the FFN's combine is SiTU, not because the keys
/// were declared. A build that read them unconditionally would pass the
/// positive arm and fail here.
#[test]
fn the_softcaps_alone_are_not_carried_when_the_activation_is_not_situ() {
    let findings = plan_with(|config| {
        config["text_config"]["hidden_act"] = serde_json::json!("silu");
        config["text_config"]["activation_situ_beta"] = serde_json::json!(BETA);
        config["text_config"]["activation_situ_linear_beta"] = serde_json::json!(LINEAR_BETA);
    });

    assert_eq!(
        finding_for(&findings, "hidden_act").resolved,
        Some(serde_json::json!("silu")),
        "an unrelated declaration of the softcaps must not change the activation"
    );
    for leaf in ["activation_situ_beta", "activation_situ_linear_beta"] {
        let finding = finding_for(&findings, leaf);
        assert_ne!(
            finding.category,
            FindingCategory::Representable,
            "{leaf} configures a combine this checkpoint says it does not use, and must not \
             read as carried"
        );
    }
}

/// A SiTU declaration with no `activation_situ_linear_beta` is a
/// DIFFERENT, and legitimate, function — the reference supports it — so
/// the name still resolves and the absent key produces no finding.
#[test]
fn situ_without_the_linear_softcap_is_still_situ() {
    let findings = plan_with(|config| {
        config["text_config"]["hidden_act"] = serde_json::json!("situ");
        config["text_config"]["activation_situ_beta"] = serde_json::json!(BETA);
    });

    assert_eq!(
        finding_for(&findings, "hidden_act").resolved,
        Some(serde_json::json!("situ"))
    );
    assert_eq!(
        finding_for(&findings, "activation_situ_beta").category,
        FindingCategory::Representable
    );
    assert!(
        !findings
            .iter()
            .any(|f| f.subject.ends_with("activation_situ_linear_beta")),
        "an undeclared key has nothing to report"
    );
}

/// The unchanged world: a plain `silu` checkpoint declaring no softcaps
/// plans exactly as it did before this rung.
///
/// 116 of the 117 rows in the conformance estate are this shape, and the
/// forecast for them is *byte-identical plans*. This is that claim in
/// miniature, where it can fail fast.
#[test]
fn a_plain_gated_checkpoint_is_untouched() {
    let findings = plan_with(|config| {
        config["text_config"]["hidden_act"] = serde_json::json!("silu");
    });
    let finding = finding_for(&findings, "hidden_act");
    assert_eq!(finding.declared, Some(serde_json::json!("silu")));
    assert_eq!(finding.resolved, Some(serde_json::json!("silu")));
    assert_eq!(finding.category, FindingCategory::Representable);
    assert!(!finding.blocks());
}

/// **The class this rung closes, on its second specimen.** BitNet's
/// `relu2` names no activation this build has judged and no gate policy.
///
/// It must STILL read `mismatched` and still block: the planner was
/// already honest about it, and K3-ACT-1's forecast says explicitly that
/// no row but K3 moves on the plan plane. What changed for `relu2` is on
/// the execution side — `gate_up_is_gelu_tanh` refuses it by name instead
/// of returning SiLU's answer — and this test exists so that a change
/// which quietly moved it here would be caught as a falsification.
#[test]
fn an_unjudged_activation_name_still_reports_as_a_dropped_fact() {
    let findings = plan_with(|config| {
        config["text_config"]["hidden_act"] = serde_json::json!("relu2");
    });
    let finding = finding_for(&findings, "hidden_act");

    assert_eq!(finding.declared, Some(serde_json::json!("relu2")));
    assert_eq!(finding.resolved, Some(serde_json::json!("silu")));
    assert_eq!(finding.category, FindingCategory::Mismatched);
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert!(finding.blocks());
}

/// Both new leaves have a carriage rule, and it claims what the rung
/// witnessed. A rule claiming more than the tests prove is the failure
/// `feedback_gate_claim_congruence` names.
#[test]
fn the_softcap_leaves_each_have_a_rule_claiming_lowered() {
    for leaf in ["activation_situ_beta", "activation_situ_linear_beta"] {
        let rule = rule_for(leaf).unwrap_or_else(|| panic!("no carriage rule for `{leaf}`"));
        assert_eq!(rule.reaches, Carriage::Lowered, "{leaf}");
        assert!(
            rule.probe.is_some(),
            "{leaf}'s rule must be checkable against the built graph, not asserted"
        );
        assert!(
            rule.site.contains("SituGlu"),
            "{leaf}'s site must name where it lands: {}",
            rule.site
        );
    }
}

/// The fixture this module mutates declares none of the three keys under
/// test, so every assertion above is a change from a known-empty state
/// rather than a coincidence of the fixture.
///
/// Asserted by planning the UNMUTATED fixture: reading the config literal
/// would prove only what the literal says, and the subject of every test
/// here is what the planner then reports.
#[test]
fn the_fixture_declares_none_of_the_keys_under_test() {
    let findings = plan_with(|_| {});
    for leaf in [
        "hidden_act",
        "activation_situ_beta",
        "activation_situ_linear_beta",
    ] {
        let path = format!("text_config.{leaf}");
        assert!(
            !findings.iter().any(|f| f.subject == path),
            "the baseline fixture already declares `{path}`, so this module's positive arms              are not changes from an empty state"
        );
    }
}
