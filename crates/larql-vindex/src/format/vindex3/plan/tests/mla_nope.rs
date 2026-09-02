//! `mla_use_nope` means no rotation, and only the combination the
//! reference implements resolves.
//!
//! The config reads as a contradiction — Kimi Linear declares
//! `mla_use_nope: true` *and* `qk_rope_head_dim: 64` — and only its own
//! `modeling_kimi.py` settles it:
//!
//! 1. the file contains **no rotary code at all**: `q_rot`/`k_rot` are
//!    split out and concatenated straight back, unrotated;
//! 2. `self.use_nope` is read exactly once, as `assert self.use_nope` —
//!    a precondition, not a switch.
//!
//! So `qk_rope_head_dim` is a structural width (it splits
//! `q_head_dim = 128 + 64 = 192`, and q_proj stores 32·192 = 6144 rows),
//! not a rotary subspace. `false` is a combination the reference refuses,
//! so this build has no ground truth for it and must not answer.

use larql_models::config::PositionPolicy;

use super::support::glimmer_shaped_target_with;
use crate::format::vindex3::graph::build_from_inventories;
use crate::format::vindex3::plan::{plan_system, FindingCategory, PlannedFinding};

const KIMI_ROPE_WIDTH: usize = 64;
const DECLARED_THETA: f64 = 10000.0;

fn positions(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<PositionPolicy> {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), mutate);
    let built = build_from_inventories(&[("target-artifact".to_string(), inventory)]);
    built
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .and_then(|c| c.attention.as_ref())
        .expect("a per-layer table")
        .iter()
        .map(|l| l.position)
        .collect()
}

/// Kimi Linear's exact combination: the flag true, a non-zero declared
/// rope width, and a declared base. Every layer resolves NoPE.
#[test]
fn the_flag_with_a_nonzero_rope_width_still_resolves_nope() {
    let policies = positions(|config| {
        config["text_config"]["mla_use_nope"] = serde_json::json!(true);
        config["text_config"]["qk_rope_head_dim"] = serde_json::json!(KIMI_ROPE_WIDTH);
        config["text_config"]["rope_theta"] = serde_json::json!(DECLARED_THETA);
    });
    assert!(
        policies.iter().all(|p| *p == PositionPolicy::None),
        "a declared rope WIDTH is geometry, not rotation: {policies:?}"
    );
}

/// GLM-5.3-Flash's combination: the flag true with a zero rope width.
#[test]
fn the_flag_with_a_zero_rope_width_resolves_nope() {
    let policies = positions(|config| {
        config["text_config"]["mla_use_nope"] = serde_json::json!(true);
        config["text_config"]["qk_rope_head_dim"] = serde_json::json!(0);
    });
    assert!(policies.iter().all(|p| *p == PositionPolicy::None));
}

/// **The control.** `false` is a combination the reference does not
/// implement — its own assert fires — so this build must not claim NoPE
/// for it. Without this arm, the two tests above would also pass on a
/// build that answered NoPE for every MLA checkpoint.
#[test]
fn the_flag_false_does_not_resolve_nope() {
    let policies = positions(|config| {
        config["text_config"]["mla_use_nope"] = serde_json::json!(false);
        config["text_config"]["qk_rope_head_dim"] = serde_json::json!(KIMI_ROPE_WIDTH);
    });
    // The fixture's own full-attention layers are already NoPE, so the
    // claim is that the FLAG did not make the whole stack NoPE.
    assert!(
        !policies.iter().all(|p| *p == PositionPolicy::None),
        "`false` is unimplemented upstream; answering NoPE would invent a \
         semantic: {policies:?}"
    );
}

/// An absent flag is not a false one, and is not a NoPE one either.
#[test]
fn an_absent_flag_does_not_resolve_nope() {
    let policies = positions(|_| {});
    assert!(!policies.iter().all(|p| *p == PositionPolicy::None));
}

fn findings(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<PlannedFinding> {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), mutate);
    plan_system(&[("target-artifact".to_string(), inventory)])
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

/// Every finding about the checkpoint-wide rope base. Two rules speak to
/// it — the declared-vs-resolved comparator and the carriage rule — and a
/// verdict is only honest if both agree.
fn rope_theta_findings(findings: &[PlannedFinding]) -> Vec<&PlannedFinding> {
    findings
        .iter()
        .filter(|f| f.subject.ends_with("rope_theta") && !f.subject.contains("layer_"))
        .collect()
}

/// A rope base declared on a stack that applies no rotation is **inert**,
/// not a mismatch: the field is a leftover the model's own forward never
/// reads.
#[test]
fn a_rope_base_on_a_nope_stack_is_inert() {
    let findings = findings(|config| {
        config["text_config"]["mla_use_nope"] = serde_json::json!(true);
        config["text_config"]["rope_theta"] = serde_json::json!(DECLARED_THETA);
        // A per-layer table would own the answer instead; Kimi Linear
        // declares only the checkpoint-wide base, which is the case here.
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("layer_rope_theta");
    });
    let about = rope_theta_findings(&findings);
    assert!(!about.is_empty());
    for f in &about {
        assert_eq!(f.category, FindingCategory::Representable, "{f:?}");
        assert!(!f.blocks(), "{f:?}");
    }
    assert!(
        about.iter().any(|f| f.detail.contains("not applied")),
        "the verdict must say WHY it is benign: {about:?}"
    );
}

/// **The arm that keeps the exemption honest.** On a stack that *does*
/// rotate, the real declared-vs-resolved comparison still runs.
///
/// The inertness rule short-circuits that comparison, so the risk is that
/// it swallows the rotating case too — and the rope-theta comparator
/// exists precisely because a theta once resolved 50× smaller than
/// declared. Asserted by the verdict's own wording: a rotating stack must
/// reach "declared and resolved agree", never "provably not applied".
#[test]
fn a_rope_base_on_a_rotating_stack_must_still_be_honoured() {
    let findings = findings(|config| {
        // No NoPE flag: the fixture rotates. Declare a base that
        // resolution will not produce.
        config["text_config"]["rope_parameters"]["rope_theta"] = serde_json::json!(12345.0);
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("layer_rope_theta");
    });
    let about = rope_theta_findings(&findings);
    assert!(
        about
            .iter()
            .any(|f| f.detail.contains("declared and resolved agree")),
        "the comparison must have actually run: {about:?}"
    );
    assert!(
        !about.iter().any(|f| f.detail.contains("not applied")),
        "the inertness exemption must not fire on a rotating stack: {about:?}"
    );
}
