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
    // **The fixture moved; the invariant did not.** This case was written
    // with Kimi K3's shape — container `kimi_k3`, which nothing then
    // registered, beside text `kimi_linear`. K3-ARCH-1 registered
    // `kimi_k3` and had it DECLARE `kimi_linear` as its text component,
    // so that exact config is now a declared containment and correctly
    // stops refusing (see `k3_arch_1_declared_containment_is_not_a_conflict`).
    //
    // What this case guards is unchanged and still needs a witness: two
    // levels resolving to different architectures with NO declaration
    // relating them. Reading either level alone would be a decision the
    // checkpoint did not authorise. `kimi_linear` declares no components,
    // so it stands in for the original K3 shape exactly.
    let mut config = base("kimi_linear");
    config["text_config"] = serde_json::json!({
        "model_type": "llama",
        "num_hidden_layers": 8,
        "hidden_size": 64
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
        Some(serde_json::Value::String("kimi_linear".into()))
    );
    assert_eq!(
        conflict.resolved,
        Some(serde_json::Value::String("llama".into()))
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

// ---- K3-ARCH-1 acceptance suite --------------------------------------
//
// **Container identity and component identity are not competing claims
// about the same level of abstraction.**
//
// K3 declares `kimi_k3` on the container and `kimi_linear` on the text
// component. Today that refuses, correctly, because the only relationship
// the gate understands is "these must resolve to the same architecture".
// But the truthful structure is containment:
//
// ```text
// container architecture   KimiK3
//     └── text component   KimiLinear
// ```
//
// The principle the gate protects — that which config level happened to be
// read must never decide which architecture the runtime serves — is
// preserved. What changes is recognising a DECLARED containment as a third
// answer beside "same" and "disagreeing".
//
// Registering `kimi_k3` alone does NOT fix this: the gate would then see
// `(Some(KimiK3), Some(KimiLinear))`, hit `!ptr::eq`, and still refuse.
// The relationship has to be declared, not inferred from both sides
// resolving to something.

/// K3-ARCH-1 case 1 — an unregistered container identity beside a
/// registered component one must refuse.
///
/// **The fixture changed and the invariant did not.** This case was
/// written with `kimi_k3` as its example of an unregistered container,
/// and pinning it BEFORE the implementation is what made the change
/// announce itself: registering `kimi_k3` moved that config onto the
/// declared-containment path, where it correctly no longer refuses. The
/// subject was invalidated by design; the rule it guards —
/// `(None, Some(_))` refuses — is untouched, so it keeps a fixture that
/// is genuinely unknown.
#[test]
fn k3_arch_1_unknown_container_beside_known_component_is_refused() {
    let mut config = base("no-such-container-architecture");
    config["text_config"] = serde_json::json!({
        "model_type": "kimi_linear",
        "num_hidden_layers": 93,
        "hidden_size": 7168
    });
    let findings = identity_findings(config);
    let conflict = findings
        .iter()
        .find(|f| f.subject == "architecture_identity")
        .expect("an unknown outer identity must refuse");
    assert_eq!(conflict.category, FindingCategory::Mismatched);
    assert!(
        conflict.blocks(),
        "an outer identity nothing registers stays refused, containment or not"
    );
}

/// K3-ARCH-1 case 4 — CURRENT BEHAVIOUR, pinned.
///
/// One architecture declared at both levels raises nothing. Any
/// containment support must leave this untouched.
#[test]
fn k3_arch_1_matching_levels_remain_silent() {
    let mut config = base("kimi_linear");
    config["text_config"] = serde_json::json!({
        "model_type": "kimi_linear",
        "num_hidden_layers": 93,
        "hidden_size": 7168
    });
    assert!(
        identity_findings(config)
            .iter()
            .all(|f| f.subject != "architecture_identity"),
        "one identity spelled twice is not a conflict"
    );
}

// Cases 2, 3, 5, 6 and 7 were specified here as comments BEFORE the
// containment relationship existed, and are now executable below. The
// prose is kept because it is the specification the implementation was
// written to satisfy, not a description of it:
//
//   2  registered KimiK3 declaring text = KimiLinear, config says
//      kimi_linear                                        -> ACCEPT
//   3  registered KimiK3, text config an incompatible family
//                                                         -> REFUSE
//   5  aliasing KimiK3 directly to KimiLinear must be IMPOSSIBLE to
//      express, not merely discouraged — it asserts K3 executes as its
//      ancestor, which is false
//   6  K3-specific semantics (AttnRes, SiTU-GLU, QB frozen bias,
//      LatentMoE wrapping) are NOT inherited merely because text ancestry
//      is declared. `text_component = KimiLinear` means "this is the
//      architectural lineage occupying the text slot", never "execute the
//      whole model with KimiLinear's implementation".
//
//   7  containment is DIRECTIONAL. `KimiK3 declares text = KimiLinear`
//      must not imply `KimiLinear declares KimiK3`, and must not imply
//      substitutability in either direction.
//
// Case 6 is load-bearing: the LatentMoE wrapper alone is 9.45 GB/token of
// BF16 machinery Kimi-Linear does not have.
//
// # A constraint on the API, not only on the semantics
//
// The relation must be spelled so that the wrong reading is hard to
// write:
//
// ```text
// declares_component(ComponentRole::Text, KimiLinear)   SAFE
// is_compatible_with(KimiLinear)                        UNSAFE
// ```
//
// A predicate named for compatibility invites symmetry, and a symmetric
// relation invites substitution — at which point a later caller expecting
// execution equivalence gets it, silently, from a relation that only ever
// meant lineage. Case 7 exists because that mistake is a naming accident
// away, not a design decision away.
//
// The relation must also NOT participate in capability inheritance unless
// some other explicit mechanism grants it. Declaring lineage answers "who
// occupies the text slot", never "what may this execute".

/// K3-ARCH-1 case 2 — a DECLARED container/component relationship is
/// accepted.
#[test]
fn k3_arch_1_declared_containment_is_not_a_conflict() {
    let mut config = base("kimi_k3");
    config["text_config"] = serde_json::json!({
        "model_type": "kimi_linear",
        "num_hidden_layers": 93,
        "hidden_size": 7168
    });
    assert!(
        identity_findings(config)
            .iter()
            .all(|f| f.subject != "architecture_identity"),
        "kimi_k3 declares text = kimi_linear, so the two levels agree by declaration"
    );
}

/// K3-ARCH-1 case 3 — a registered container beside an UNdeclared
/// component still refuses. The declaration is what accepts, not the mere
/// fact that both sides resolve.
#[test]
fn k3_arch_1_a_component_the_container_did_not_declare_is_refused() {
    let mut config = base("kimi_k3");
    config["text_config"] = serde_json::json!({
        "model_type": "llama",
        "num_hidden_layers": 8,
        "hidden_size": 64
    });
    let findings = identity_findings(config);
    let conflict = findings
        .iter()
        .find(|f| f.subject == "architecture_identity")
        .expect("kimi_k3 declares kimi_linear, not llama");
    assert_eq!(conflict.category, FindingCategory::Mismatched);
    assert!(conflict.blocks());
}

/// K3-ARCH-1 case 5 — containment is NOT aliasing.
#[test]
fn k3_arch_1_kimi_k3_is_not_kimi_linear() {
    use larql_models::detect::registry::find_architecture;
    let k3 = find_architecture("kimi_k3").expect("registered");
    let ancestor = find_architecture("kimi_linear").expect("registered");
    assert!(
        !std::ptr::eq(k3, ancestor),
        "aliasing would assert K3 executes as its ancestor, which is false"
    );
    assert_eq!(k3.model_type, "kimi_k3");
}

/// K3-ARCH-1 case 6 — declaring lineage grants no capability.
#[test]
fn k3_arch_1_lineage_is_not_capability_inheritance() {
    use larql_models::detect::detect_from_json;
    let arch = detect_from_json(&serde_json::json!({ "model_type": "kimi_k3" }));
    assert_eq!(arch.family(), "kimi_k3", "never the ancestor's family");
    assert_ne!(arch.family(), "kimi_linear");
    // And explicitly not the silent generic default, which is the failure
    // this whole gate exists to prevent.
    assert_ne!(arch.family(), "generic");
}

/// K3-ARCH-1 case 7 — containment is DIRECTIONAL.
#[test]
fn k3_arch_1_containment_is_directional_not_symmetric() {
    use larql_models::detect::registry::{find_architecture, ComponentRole};
    let k3 = find_architecture("kimi_k3").expect("registered");
    let ancestor = find_architecture("kimi_linear").expect("registered");

    assert_eq!(
        k3.declares_component(ComponentRole::Text),
        Some("kimi_linear")
    );
    assert_eq!(
        ancestor.declares_component(ComponentRole::Text),
        None,
        "the ancestor declares nothing about K3 — the relation is one-way"
    );

    // The reverse nesting must still refuse: kimi_linear does not declare
    // kimi_k3 as its text component, so this is an undeclared disagreement.
    let mut reversed = base("kimi_linear");
    reversed["text_config"] = serde_json::json!({
        "model_type": "kimi_k3",
        "num_hidden_layers": 93,
        "hidden_size": 7168
    });
    let findings = identity_findings(reversed);
    assert!(
        findings
            .iter()
            .any(|f| f.subject == "architecture_identity"
                && f.category == FindingCategory::Mismatched),
        "containment must not be readable backwards"
    );
}
