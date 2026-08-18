//! Classification of flattened config keys.

use serde_json::json;

use crate::inventory::config_keys::{classify_config, find_interfaces};
use crate::inventory::report::KeyStatus;

fn status_of(facts: &[crate::inventory::ConfigKeyFact], path: &str) -> KeyStatus {
    facts
        .iter()
        .find(|f| f.path == path)
        .unwrap_or_else(|| panic!("path {path} not found in facts"))
        .status
}

/// The Glimmer shape: consumed, metadata and unconsumed keys side by side,
/// with the four known-dangerous unconsumed scalars all flagged.
#[test]
fn glimmer_shaped_config_flags_the_dangerous_scalars() {
    let config = json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 6656,
            "num_hidden_layers": 52,
            "intermediate_size": 19968,
            "sliding_window": 2048,
            "final_logit_softcapping": 20.0,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "post_norm_eps": 1e-8,
            "layer_rope_theta": [500000.0, 0.0],
            "rms_norm_eps": 1e-5
        }
    });
    let facts = classify_config(&config, &Default::default());

    // Consumed: topology + softcapping the parser reads.
    assert_eq!(
        status_of(&facts, "text_config.hidden_size"),
        KeyStatus::Consumed
    );
    assert_eq!(
        status_of(&facts, "text_config.final_logit_softcapping"),
        KeyStatus::Consumed
    );
    assert_eq!(
        status_of(&facts, "text_config.rms_norm_eps"),
        KeyStatus::Consumed
    );

    // Metadata: identity facts.
    assert_eq!(status_of(&facts, "architectures"), KeyStatus::Metadata);
    assert_eq!(status_of(&facts, "dtype"), KeyStatus::Metadata);

    // The four scalars this instrument originally caught unconsumed are
    // consumed since G2a/G2b — parsed into ModelConfig with trait hooks.
    for path in [
        "text_config.qk_scale_factor",
        "text_config.output_multiplier",
        "text_config.post_norm_eps",
        "text_config.layer_rope_theta",
    ] {
        assert_eq!(status_of(&facts, path), KeyStatus::Consumed, "{path}");
    }
}

/// The instrument still fires: a key nobody has judged reads unconsumed.
#[test]
fn an_unjudged_key_still_reads_unconsumed() {
    let config = json!({
        "model_type": "x",
        "text_config": {
            "hidden_size": 64,
            "some_future_scalar_nobody_reviewed": 1.5
        }
    });
    let facts = classify_config(&config, &Default::default());
    assert_eq!(
        status_of(&facts, "text_config.some_future_scalar_nobody_reviewed"),
        KeyStatus::Unconsumed
    );
}

/// `vision_config` children never earn consumed credit, even when a child
/// shares its name with a consumed text-path key. The parser only checks the
/// container's presence.
#[test]
fn vision_config_children_are_never_consumed() {
    let config = json!({
        "model_type": "muse_glimmer",
        "vision_config": {
            "hidden_size": 1536,
            "layer_types": ["window_attention"],
            "rope_parameters": { "rope_theta": 10000.0 }
        }
    });
    let facts = classify_config(&config, &Default::default());
    assert_eq!(
        status_of(&facts, "vision_config.hidden_size"),
        KeyStatus::Unconsumed
    );
    assert_eq!(
        status_of(&facts, "vision_config.layer_types"),
        KeyStatus::Unconsumed
    );
    assert_eq!(
        status_of(&facts, "vision_config.rope_parameters.rope_theta"),
        KeyStatus::Unconsumed
    );
}

/// An unknown top-level container also cuts consumed credit: a leaf named
/// like a consumed key inside `foo_config` is not read by the parser.
#[test]
fn unknown_containers_cut_consumed_credit() {
    let config = json!({
        "model_type": "x",
        "foo_config": { "hidden_size": 128 }
    });
    let facts = classify_config(&config, &Default::default());
    assert_eq!(
        status_of(&facts, "foo_config.hidden_size"),
        KeyStatus::Unconsumed
    );
}

/// Consumed credit survives through the containers the parser recurses into.
#[test]
fn rope_scaling_leaves_keep_consumed_credit() {
    let config = json!({
        "model_type": "llama",
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0
        }
    });
    let facts = classify_config(&config, &Default::default());
    assert_eq!(
        status_of(&facts, "rope_scaling.factor"),
        KeyStatus::Consumed
    );
    assert_eq!(
        status_of(&facts, "rope_scaling.rope_type"),
        KeyStatus::Consumed
    );
    assert_eq!(
        status_of(&facts, "rope_scaling.low_freq_factor"),
        KeyStatus::Consumed
    );
}

/// Arrays are leaves: `layer_types` is one fact, not one per element.
#[test]
fn arrays_are_single_leaves() {
    let config = json!({
        "model_type": "x",
        "layer_types": ["sliding_attention", "full_attention"]
    });
    let facts = classify_config(&config, &Default::default());
    let matches: Vec<_> = facts
        .iter()
        .filter(|f| f.path.contains("layer_types"))
        .collect();
    assert_eq!(matches.len(), 1);
    assert!(matches[0].value.is_array());
}

/// Facts come back path-sorted so reports diff cleanly.
#[test]
fn facts_are_sorted_by_path() {
    let config = json!({
        "model_type": "x",
        "b_key": 1,
        "a_key": 2
    });
    let facts = classify_config(&config, &Default::default());
    let paths: Vec<&str> = facts.iter().map(|f| f.path.as_str()).collect();
    let mut sorted = paths.clone();
    sorted.sort();
    assert_eq!(paths, sorted);
}

/// `target_layer_ids` surfaces as a declared interface wherever it appears.
#[test]
fn target_layer_ids_is_reported_as_an_interface() {
    let config = json!({
        "model_type": "muse_glimmer_assistant",
        "target_layer_ids": [1, 13, 25, 37, 49],
        "block_size": 16
    });
    let facts = classify_config(&config, &Default::default());
    let interfaces = find_interfaces(&facts);
    assert_eq!(interfaces.len(), 1);
    assert_eq!(interfaces[0].path, "target_layer_ids");
    assert_eq!(interfaces[0].value, json!([1, 13, 25, 37, 49]));
    // Consumed since G2b (parsed as the drafter-interface declaration) —
    // and still surfaced as an interface, independently of key status.
    assert_eq!(status_of(&facts, "target_layer_ids"), KeyStatus::Consumed);
}

/// A config with no interface keys reports none.
#[test]
fn no_interfaces_when_none_declared() {
    let config = json!({ "model_type": "llama", "hidden_size": 4096 });
    let facts = classify_config(&config, &Default::default());
    assert!(find_interfaces(&facts).is_empty());
}

/// A non-object root has no key structure: one unconsumed fact, verbatim
/// — the defensive arm a real config.json never reaches.
#[test]
fn a_non_object_root_is_one_unconsumed_fact() {
    let config = json!("not an object");
    let facts = classify_config(&config, &Default::default());
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].status, KeyStatus::Unconsumed);
    assert_eq!(facts[0].value, json!("not an object"));
}

/// A label-map container (`id2label`, `label2id`) is metadata to its
/// leaves, however they are named — a numbered leaf cannot be named in
/// the leaf registry, and nothing under a label map moves a forward pass.
/// A nested map inside one is flattened the same way.
#[test]
fn label_map_containers_are_metadata_to_every_leaf() {
    let config = json!({
        "vision_config": {
            "id2label": { "0": "LABEL_0", "1": "LABEL_1", "nested": { "2": "LABEL_2" } },
            "label2id": { "LABEL_0": 0 },
            "problem_type": null,
            "chunk_size_feed_forward": 0
        }
    });
    let facts = classify_config(&config, &Default::default());
    for path in [
        "vision_config.id2label.0",
        "vision_config.id2label.1",
        "vision_config.id2label.nested.2",
        "vision_config.label2id.LABEL_0",
        "vision_config.problem_type",
        "vision_config.chunk_size_feed_forward",
    ] {
        let fact = facts
            .iter()
            .find(|f| f.path == path)
            .unwrap_or_else(|| panic!("{path}"));
        assert_eq!(fact.status, KeyStatus::Metadata, "{path}");
    }
}
