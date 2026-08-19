//! Semantic-class registry behavior.

use crate::format::vindex3::plan::semantics::{classify_key, component_of, leaf_of};
use crate::format::vindex3::plan::SemanticClass;
use larql_models::inventory::config_keys::CONSUMED_LEAF_KEYS;

#[test]
fn dangerous_scalars_grade_execution_semantic() {
    for key in [
        "layer_rope_theta",
        "qk_scale_factor",
        "output_multiplier",
        "post_norm_eps",
        "hidden_activation",
        "attention_bias",
    ] {
        assert_eq!(classify_key(key), SemanticClass::ExecutionSemantic, "{key}");
    }
}

#[test]
fn shape_keys_grade_tensor_semantic() {
    for key in ["hidden_size", "patch_size", "out_hidden_size", "head_dim"] {
        assert_eq!(classify_key(key), SemanticClass::TensorSemantic, "{key}");
    }
}

#[test]
fn interface_keys_grade_interface_semantic() {
    for key in [
        "target_layer_ids",
        "block_size",
        "mask_token_id",
        "image_token_id",
    ] {
        assert_eq!(classify_key(key), SemanticClass::InterfaceSemantic, "{key}");
    }
}

#[test]
fn unseen_keys_grade_unknown_and_therefore_block() {
    let class = classify_key("some_future_field_nobody_reviewed");
    assert_eq!(class, SemanticClass::Unknown);
    assert!(class.is_critical());
}

#[test]
fn metadata_keys_do_not_block() {
    let class = classify_key("model_type");
    assert_eq!(class, SemanticClass::MetadataOnly);
    assert!(!class.is_critical());
}

#[test]
fn component_attribution_follows_config_nesting() {
    assert_eq!(component_of("text_config.qk_scale_factor"), "text");
    assert_eq!(component_of("vision_config.patch_size"), "vision");
    assert_eq!(component_of("image_token_id"), "root");
    assert_eq!(
        component_of("vision_config.rope_parameters.rope_theta"),
        "vision"
    );
}

/// The census: every leaf name a real parser reads
/// ([`CONSUMED_LEAF_KEYS`]) must be judged by this registry — placed in
/// some bucket, `Unknown` refused. `CONSUMED_LEAF_KEYS` is *the* list of
/// key names any `ModelConfig` parser reads, kept honest by its own sync
/// test against the parser source (`larql-models`' `tests/config_keys.rs`);
/// this test closes the other side, so a leaf cannot be `consumed` and
/// *unjudged* at once — a fact `Consumed`-but-`Unknown` used to grade
/// `Representable` in `carriage_finding` (`plan/mod.rs`), not
/// `Unrepresented`, so a key nobody had reviewed sailed through the plan
/// exactly like an unread one is supposed to be caught doing.
///
/// A-11 (2026-08-18) is the origin: Granite's four scaling multipliers
/// were consumed and silently `representable`/`unknown`. Auditing the
/// full 80-key list surfaced 41 more in the same state — GPT-OSS's other
/// YaRN leaves, MoE/MLA operand counts, four GPT-2 shape aliases, a
/// fourth norm-epsilon spelling. All 41 are now bucketed (`semantics.rs`);
/// this test is what keeps the count at zero from here.
#[test]
fn every_consumed_leaf_key_is_judged() {
    let unjudged: Vec<&str> = CONSUMED_LEAF_KEYS
        .iter()
        .copied()
        .filter(|key| classify_key(key) == SemanticClass::Unknown)
        .collect();
    assert!(
        unjudged.is_empty(),
        "consumed but never classified — add each to a bucket in \
         plan/semantics.rs before it can be silently `representable`: \
         {unjudged:?}"
    );
}

#[test]
fn leaf_extraction() {
    assert_eq!(
        leaf_of("text_config.rope_parameters.rope_theta"),
        "rope_theta"
    );
    assert_eq!(leaf_of("block_size"), "block_size");
}
