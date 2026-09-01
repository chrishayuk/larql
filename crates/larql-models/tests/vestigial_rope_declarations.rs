//! **Two declarations that no implementation reads — ours or upstream.**
//!
//! `use_mrope` appears nowhere in transformers 5.5.0, and `rope_interleaved`
//! (exact spelling) appears nowhere either — SmolLM2-135M declares it under
//! `model_type: llama`, which has no such field. Both are therefore unlike
//! `use_sliding_window`, which HF genuinely consults: there is no upstream
//! behaviour to match, only this build's.
//!
//! That makes reading them *more* important rather than less. An unread
//! declaration that happens to agree is one value away from a silent wrong
//! answer, and neither of these names a cosmetic fact:
//!
//! ```text
//! rope_interleaved   which dimensions rotate against which partner
//! use_mrope          whether position is one axis or three
//! ```
//!
//! What the parser must NOT do is act on them. The flags are checked
//! against the effective policy at the VINDEX3 boundary
//! (`plan::tests::carriage`); here the claim is narrower and is the
//! precondition for that check being meaningful — the declarations reach
//! `ModelConfig` verbatim, and neither one is allowed to manufacture a
//! policy on its own.

use larql_models::config::{PositionPolicy, ROPE_PAIRING_INTERLEAVED};
use larql_models::detect::detect_from_json;
use serde_json::json;

const THETA: f64 = 1_000_000.0;

fn config(mutate: impl FnOnce(&mut serde_json::Value)) -> serde_json::Value {
    let mut config = json!({
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": THETA,
    });
    mutate(&mut config);
    config
}

#[test]
fn both_declarations_reach_the_config_verbatim() {
    // The precondition for checking them at all. Absence stays absent —
    // a distinct state from a declared `false`, because "did not say" and
    // "said no" are different claims about the checkpoint.
    let declared = detect_from_json(&config(|c| {
        c["rope_interleaved"] = json!(false);
        c["use_mrope"] = json!(false);
    }));
    assert_eq!(declared.config().rope_interleaved, Some(false));
    assert_eq!(declared.config().use_mrope, Some(false));

    let silent = detect_from_json(&config(|_| {}));
    assert_eq!(silent.config().rope_interleaved, None);
    assert_eq!(silent.config().use_mrope, None);
}

#[test]
fn a_declared_pairing_does_not_change_the_resolved_policy() {
    // Reading the flag must not become acting on it. This build has one
    // pairing and the planner reports the disagreement; a parser that
    // quietly switched operators here would turn a reportable mismatch
    // into the silent wrong answer the report exists to prevent.
    // Declared as the OPPOSITE of whatever this build pairs, so the arm
    // stays a disagreement if the pairing ever changes.
    let disagreeing = !ROPE_PAIRING_INTERLEAVED;
    let interleaved = detect_from_json(&config(|c| c["rope_interleaved"] = json!(disagreeing)));
    assert_eq!(
        interleaved.config().rope_interleaved,
        Some(disagreeing),
        "the disagreeing declaration is carried"
    );
    assert_eq!(
        interleaved.position_policy_for_layer(0),
        PositionPolicy::Rope { theta: THETA },
        "and carried is not obeyed — the policy is unchanged by it"
    );
}

#[test]
fn a_declared_mrope_flag_cannot_manufacture_a_multi_axis_policy() {
    // M-RoPE needs the axis geometry — `mrope_section` and
    // `mrope_interleaved` jointly — and the flag supplies neither. If
    // `use_mrope: true` alone produced `PositionPolicy::MRope`, the
    // section would have to be invented, which is how a text model ends
    // up rotating against three axes it never declared.
    let claimed = detect_from_json(&config(|c| c["use_mrope"] = json!(true)));
    assert!(
        !matches!(
            claimed.position_policy_for_layer(0),
            PositionPolicy::MRope { .. }
        ),
        "a flag with no geometry behind it must not resolve a multi-axis policy"
    );
}
