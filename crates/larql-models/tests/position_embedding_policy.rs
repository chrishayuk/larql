//! **`granitemoehybrid` rotates only when it says so.**
//!
//! `position_embedding_type` looks like the most boring key in the file —
//! a string reading `"rope"` on a model that obviously uses RoPE. It is
//! not a restatement. `GraniteMoeHybridConfig` documents it as
//!
//! ```text
//! position_embedding_type (`str`, *optional*):
//!     Positional embedding type to be used; defaults to None.
//!     Allowed options: `[None, "rope"]`
//! ```
//!
//! and `modeling_granitemoehybrid.py` builds
//!
//! ```text
//! self.rotary_emb = GraniteMoeHybridRotaryEmbedding(config)
//!                   if config.position_embedding_type == "rope" else None
//! ```
//!
//! So the key is the **opt-in that turns rotation on**, and its absence
//! means the model encodes no position at all. `rope_theta` is declared
//! either way — granite-4.0-micro ships `10000000` — so a resolver that
//! reads the theta and rotates is right about that checkpoint by luck and
//! wrong about any `granitemoehybrid` that omits the opt-in, rotating
//! every position against the model's own instruction. Exactly the shape
//! of the `use_sliding_window` bug, one key over.
//!
//! The control carries as much weight as the fix. `granite`, `granitemoe`
//! and `granitemoeshared` construct their rotary unconditionally and
//! never mention the key, so a rule applied family-wide would silently
//! turn every dense Granite into a NoPE model — a much larger wrong
//! answer than the one being fixed.

use larql_models::config::PositionPolicy;
use larql_models::detect::detect_from_json;
use serde_json::json;

const THETA: f64 = 10_000_000.0;
const LAYERS: usize = 4;

/// A Granite-shaped config, `model_type` and the key under test varying.
fn config(model_type: &str, position_embedding_type: Option<&str>) -> serde_json::Value {
    let mut config = json!({
        "model_type": model_type,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        // Declared in every arm, because the whole hazard is that a theta
        // is present and inviting whether or not rotation was asked for.
        "rope_theta": THETA,
    });
    if let Some(declared) = position_embedding_type {
        config["position_embedding_type"] = json!(declared);
    }
    config
}

fn policies(model_type: &str, position_embedding_type: Option<&str>) -> Vec<PositionPolicy> {
    let arch = detect_from_json(&config(model_type, position_embedding_type));
    (0..LAYERS)
        .map(|l| arch.position_policy_for_layer(l))
        .collect()
}

#[test]
fn the_hybrid_without_the_opt_in_encodes_no_position() {
    // THE ADVERSARIAL CASE, and an ordinary config: everything a
    // granitemoehybrid needs to rotate is present except the one key that
    // authorises it. HF builds no rotary embedding here.
    let policies = policies("granitemoehybrid", None);
    assert!(
        policies.iter().all(|p| *p == PositionPolicy::None),
        "a granitemoehybrid that does not opt in must not rotate; got {policies:?}"
    );
}

#[test]
fn the_hybrid_with_the_opt_in_rotates_at_the_declared_theta() {
    // The other half: honouring absence must not degenerate into ignoring
    // the feature. granite-4.0-micro's real declaration.
    let policies = policies("granitemoehybrid", Some("rope"));
    assert!(
        policies
            .iter()
            .all(|p| *p == PositionPolicy::Rope { theta: THETA }),
        "the opt-in must rotate at the declared base; got {policies:?}"
    );
}

#[test]
fn control_a_dense_granite_still_rotates_without_the_key() {
    // The control that bounds the fix. `granite` has no
    // `position_embedding_type` at all and constructs its rotary
    // unconditionally, so scoping this rule to the wrong set of
    // model_types would turn the entire family into NoPE models —
    // a far bigger wrong answer than the one being fixed.
    for model_type in ["granite", "granitemoe", "granitemoeshared"] {
        let policies = policies(model_type, None);
        assert!(
            policies
                .iter()
                .all(|p| *p == PositionPolicy::Rope { theta: THETA }),
            "{model_type} rotates unconditionally upstream; got {policies:?}"
        );
    }
}

#[test]
fn an_out_of_contract_value_does_not_rotate() {
    // HF compares against `"rope"` and sends everything else down the
    // `else` branch that builds no rotary. Matching the reference
    // exactly matters more than guessing what a novel value meant: the
    // BERT lineage spells `absolute` here, and a granitemoehybrid
    // borrowing that spelling would get no rotation upstream.
    let policies = policies("granitemoehybrid", Some("absolute"));
    assert!(
        policies.iter().all(|p| *p == PositionPolicy::None),
        "only `rope` authorises rotation; got {policies:?}"
    );
}

#[test]
fn the_declaration_survives_into_the_config() {
    // The key must be READ, not merely acted on: an unread declaration is
    // an unconsumed config key, and the inventory's closure gate is what
    // turns that into a finding rather than a silent assumption.
    let arch = detect_from_json(&config("granitemoehybrid", Some("rope")));
    assert_eq!(
        arch.config().position_embedding_type.as_deref(),
        Some("rope"),
        "the declared scheme must reach ModelConfig verbatim"
    );

    let absent = detect_from_json(&config("granitemoehybrid", None));
    assert_eq!(
        absent.config().position_embedding_type,
        None,
        "absence is a distinct state from any declared value"
    );
}
