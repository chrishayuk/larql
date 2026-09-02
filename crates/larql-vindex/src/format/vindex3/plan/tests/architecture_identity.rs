//! Who the checkpoint says it is, and whether this build may answer.
//!
//! The gate these cover exists because `GenericArch` is a *silent*
//! default: an unrecognised `model_type` produced no finding at all, and
//! the checkpoint was served with Llama-shaped norm placement, QK norm,
//! embedding scaling and gating that it never declared. Fifteen of the
//! forty-two `model_type` strings in the conformance corpus resolved that
//! way, over thirty checkpoints.

use super::support::known_dense_with_config;
use crate::format::vindex3::plan::{plan_system, FindingCategory, PlannedFinding, SemanticClass};

/// The findings this gate raises, by subject.
fn identity_findings(config: serde_json::Value) -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![(
        "target-artifact".to_string(),
        known_dense_with_config(dir.path(), config),
    )];
    plan_system(&named)
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .filter(|f| {
            matches!(
                f.subject.as_str(),
                "architecture_identity" | "architecture_family"
            )
        })
        .collect()
}

fn base(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["ForCausalLM"],
        "model_type": model_type,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    })
}

#[test]
fn a_registered_family_raises_no_identity_finding() {
    // The control. Without it every one of these tests would pass on a
    // gate that fired unconditionally, and the seventeen checkpoints that
    // are admissible today would all have regressed.
    assert!(identity_findings(base("llama")).is_empty());
}

#[test]
fn an_unregistered_family_is_refused_rather_than_served_generically() {
    let findings = identity_findings(base("jamba"));
    let finding = findings
        .iter()
        .find(|f| f.subject == "architecture_family")
        .expect("an unrecognised family must raise a finding");
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert!(finding.blocks(), "an unrecognised identity must block");
    assert_eq!(
        finding.declared,
        Some(serde_json::Value::String("jamba".into()))
    );
}

#[test]
fn an_unregistered_family_is_not_promoted_to_unsupported_component() {
    // The distinction AMBER carries: `UnsupportedComponent` claims the
    // semantics ARE understood and only the implementation is missing.
    // A `model_type` nothing has judged has not been understood, and
    // grading it that way would report every unrecognised checkpoint as
    // "we know what this is" — turning the corpus's six genuine AMBER
    // rows into thirty-odd meaningless ones.
    let findings = identity_findings(base("jamba"));
    let finding = findings
        .iter()
        .find(|f| f.subject == "architecture_family")
        .unwrap();
    assert_eq!(finding.class, SemanticClass::Unknown);
    assert_ne!(finding.class, SemanticClass::UnsupportedComponent);
}

#[test]
fn two_levels_that_resolve_differently_are_a_refused_conflict() {
    // Kimi K3's shape: the container declares `kimi_k3`, which nothing
    // registers, and the text component declares `kimi_linear`, which
    // resolves to the 48B implementation. Reading either level alone is a
    // decision the checkpoint did not authorise.
    let mut config = base("kimi_k3");
    config["text_config"] = serde_json::json!({
        "model_type": "kimi_linear",
        "num_hidden_layers": 93,
        "hidden_size": 7168
    });
    let findings = identity_findings(config);
    let conflict = findings
        .iter()
        .find(|f| f.subject == "architecture_identity")
        .expect("a divergent identity must raise a finding");
    assert_eq!(conflict.category, FindingCategory::Mismatched);
    assert!(conflict.blocks());
    assert_eq!(
        conflict.declared,
        Some(serde_json::Value::String("kimi_k3".into()))
    );
    assert_eq!(
        conflict.resolved,
        Some(serde_json::Value::String("kimi_linear".into()))
    );
}

#[test]
fn the_suffix_form_declares_one_identity_twice_and_is_not_a_conflict() {
    // The control that keeps the conflict gate from firing on ordinary
    // multimodal nesting. Twenty-seven of the twenty-eight corpus
    // checkpoints that declare at both levels use `<container>_text`, and
    // both spellings resolve to the same registry entry. Comparing the
    // strings instead of what they resolve to would refuse all of them —
    // including Gemma 4 26B-A4B and Qwen3.8-27B, which are admissible.
    let mut config = base("gemma3");
    config["text_config"] = serde_json::json!({
        "model_type": "gemma3_text",
        "num_hidden_layers": 2,
        "hidden_size": 64
    });
    assert!(
        identity_findings(config).is_empty(),
        "one identity spelled twice is not a disagreement"
    );
}
