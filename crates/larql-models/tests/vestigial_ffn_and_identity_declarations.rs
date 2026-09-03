//! **Two more declarations that no implementation reads — ours or upstream.**
//!
//! Falcon3-1B-Base declares `activation: "swiglu"` beside `hidden_act: "silu"`
//! under `model_type: llama`; `LlamaConfig` has no `activation` field, so no
//! transformers-5.5.0 loader reads it for that family. SmolLM2-135M declares
//! `is_llama_config: true`; the key appears nowhere in transformers 5.5.0.
//! Both are the same kind as `use_mrope` / `rope_interleaved`
//! (`vestigial_rope_declarations`): no upstream behaviour to match, only this
//! build's, and neither names a cosmetic fact:
//!
//! ```text
//! activation        gated or ungated FFN, and what the gate applies
//! is_llama_config   which family serves the checkpoint
//! ```
//!
//! What the parser must NOT do is act on them. Both are checked at the
//! VINDEX3 boundary against what actually resolves — the FFN shape on the
//! execution surface, the registry's resolution of the identity
//! (`plan::tests::carriage`). Here the claim is the precondition for that
//! check: the declarations reach `ModelConfig` verbatim, and neither one is
//! allowed to manufacture an FFN or a family on its own.
use larql_models::config::{ffn_shape_from_hf_name, Activation, FfnType};
use larql_models::detect::detect_from_json;
use serde_json::json;

fn config(mutate: impl FnOnce(&mut serde_json::Value)) -> serde_json::Value {
    let mut config = json!({
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
    });
    mutate(&mut config);
    config
}

#[test]
fn both_declarations_reach_the_config_verbatim() {
    let declared = detect_from_json(&config(|c| {
        c["activation"] = json!("swiglu");
        c["is_llama_config"] = json!(true);
    }));
    assert_eq!(declared.config().ffn_shape_name.as_deref(), Some("swiglu"));
    assert_eq!(declared.config().is_llama_config, Some(true));

    let silent = detect_from_json(&config(|_| {}));
    assert_eq!(silent.config().ffn_shape_name, None);
    assert_eq!(silent.config().is_llama_config, None);
}

/// The FFN the family resolves is decided by the family, never by the
/// word. A Llama stack declaring `geglu` still parses as gated SiLU — the
/// disagreement is for the boundary to refuse, not for the parser to
/// honour by changing the arithmetic.
#[test]
fn the_declared_shape_word_does_not_change_the_resolved_ffn() {
    let claimed = detect_from_json(&config(|c| c["activation"] = json!("geglu")));
    assert_eq!(claimed.ffn_type(), FfnType::Gated);
    assert_eq!(claimed.activation(), Activation::Silu);
    assert_eq!(
        ffn_shape_from_hf_name("geglu"),
        Some((FfnType::Gated, Activation::Gelu)),
        "the word names a different FFN, which is exactly why it must be checked"
    );
}

/// Likewise the family: `is_llama_config: false` on a `model_type: llama`
/// checkpoint does not move detection off the Llama family.
#[test]
fn the_declared_family_flag_does_not_change_detection() {
    let denied = detect_from_json(&config(|c| c["is_llama_config"] = json!(false)));
    assert_eq!(denied.family(), "llama");
    assert_eq!(denied.config().is_llama_config, Some(false));
}
