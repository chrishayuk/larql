//! Nested-component topology reading + recorded-read classification credit.

use serde_json::json;

use crate::inventory::components::read_components;
use crate::inventory::config_keys::classify_config;
use crate::inventory::report::KeyStatus;

fn glimmer_vision_config() -> serde_json::Value {
    json!({
        "model_type": "muse_glimmer",
        "text_config": { "model_type": "muse_glimmer_text", "hidden_size": 64 },
        "vision_config": {
            "model_type": "muse_glimmer_vision",
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 50,
            "num_attention_heads": 16,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-5,
            "layer_types": ["window_attention", "full_attention"],
            "max_position_embeddings": 1024,
            "merge_size": 2,
            "patch_size": 14,
            "patch_temporal": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" }
        }
    })
}

#[test]
fn vision_topology_is_read_into_typed_fields() {
    let readings = read_components(&glimmer_vision_config());
    assert_eq!(readings.len(), 1);
    let t = &readings[0].topology;
    assert_eq!(t.name, "vision");
    assert_eq!(t.model_type.as_deref(), Some("muse_glimmer_vision"));
    assert_eq!(t.hidden_size, Some(1536));
    assert_eq!(t.num_layers, Some(50));
    assert_eq!(t.num_attention_heads, Some(16));
    assert_eq!(t.norm_eps, Some(1e-5));
    assert_eq!(t.hidden_act.as_deref(), Some("gelu"));
    assert_eq!(t.rope_theta, Some(10000.0));
    assert_eq!(t.rope_type.as_deref(), Some("default"));
    assert_eq!(t.max_position_embeddings, Some(1024));
    assert_eq!(t.patch.patch_size, Some(14));
    assert_eq!(t.patch.patch_temporal, Some(2));
    assert_eq!(t.patch.merge_size, Some(2));
    assert_eq!(t.patch.pos_emb_height, Some(32));
    assert_eq!(t.patch.pos_emb_width, Some(32));
    assert_eq!(
        t.layer_types.as_deref(),
        Some(&["window_attention".to_string(), "full_attention".to_string()][..])
    );
}

/// Every key the reader stored classifies consumed; a key it did not store
/// stays unconsumed — the recorded read IS the credit.
#[test]
fn recorded_reads_are_the_consumed_credit() {
    let mut config = glimmer_vision_config();
    config["vision_config"]["some_unjudged_vision_field"] = json!(7);
    let readings = read_components(&config);
    let consumed: std::collections::BTreeSet<String> = readings
        .iter()
        .flat_map(|r| r.consumed_paths.iter().cloned())
        .collect();
    let facts = classify_config(&config, &consumed);

    let status = |path: &str| {
        facts
            .iter()
            .find(|f| f.path == path)
            .unwrap_or_else(|| panic!("missing {path}"))
            .status
    };
    for path in [
        "vision_config.hidden_size",
        "vision_config.layer_types",
        "vision_config.patch_size",
        "vision_config.rope_parameters.rope_theta",
        "vision_config.max_position_embeddings",
    ] {
        assert_eq!(status(path), KeyStatus::Consumed, "{path}");
    }
    assert_eq!(
        status("vision_config.some_unjudged_vision_field"),
        KeyStatus::Unconsumed
    );
}

/// `text_config`/`language_config` belong to the main parser, and only
/// object-valued `*_config` keys are components.
#[test]
fn main_parser_components_and_non_objects_are_skipped() {
    let config = json!({
        "text_config": { "hidden_size": 64 },
        "language_config": { "hidden_size": 64 },
        "vision_config": { "hidden_size": 32 },
        "audio_config": { "hidden_size": 16 },
        "not_a_config": 5,
        "weird_config": "flat-string"
    });
    let readings = read_components(&config);
    let names: Vec<&str> = readings.iter().map(|r| r.topology.name.as_str()).collect();
    assert_eq!(names, vec!["audio", "vision"]);
}

/// Absent keys are never credited: the consumed set contains only paths
/// that resolved to a present value.
#[test]
fn absent_keys_are_not_marked_consumed() {
    let config = json!({ "vision_config": { "hidden_size": 32 } });
    let readings = read_components(&config);
    let consumed = &readings[0].consumed_paths;
    assert!(consumed.contains("vision_config.hidden_size"));
    assert!(!consumed.iter().any(|p| p.contains("patch_size")));
    assert!(!consumed.iter().any(|p| p.contains("rope_parameters")));
}

#[test]
fn topology_round_trips_through_json() {
    let readings = read_components(&glimmer_vision_config());
    let json = serde_json::to_string(&readings[0].topology).unwrap();
    let back: crate::inventory::components::ComponentTopology =
        serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, "vision");
    assert_eq!(back.patch.patch_size, Some(14));
}

/// A `*_config` object that declares no topology is a policy block, not a
/// component. GPT-OSS's real `quantization_config` is the witness: read as
/// a component it produced a phantom with zero layers and zero heads,
/// whose execution surface then failed completeness and blocked the plan
/// for a reason about the reader rather than the model.
#[test]
fn a_config_block_with_no_topology_is_not_a_component() {
    let config = json!({
        "vision_config": { "hidden_size": 32 },
        "quantization_config": {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head"
            ],
            "quant_method": "mxfp4"
        }
    });
    let readings = read_components(&config);
    let names: Vec<&str> = readings.iter().map(|r| r.topology.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["vision"],
        "quantization_config is not a component"
    );
    // Its keys must stay uncredited, so they still face the semantics
    // registry rather than vanishing into a component that does not exist.
    assert!(
        !readings.iter().any(|r| r
            .consumed_paths
            .iter()
            .any(|p| p.starts_with("quantization_config"))),
        "a rejected reading must not credit its keys as consumed"
    );
}

/// The rule is evidence-based, not a name denylist: any single topology
/// fact is enough to make a nested block a component, and a block with
/// none is rejected whatever it is called.
#[test]
fn topology_evidence_alone_decides_componenthood() {
    for (fact, value) in [
        ("hidden_size", json!(32)),
        ("num_hidden_layers", json!(4)),
        ("layer_types", json!(["full_attention"])),
        ("patch_size", json!(14)),
    ] {
        let config = json!({ "mystery_config": { fact: value } });
        assert_eq!(
            read_components(&config).len(),
            1,
            "`{fact}` is topology evidence and must make a component"
        );
    }
    let inert = json!({ "mystery_config": { "some_policy": "value", "flag": true } });
    assert!(
        read_components(&inert).is_empty(),
        "a block with no topology fact is not a component"
    );
}

/// Gemma 3's SigLIP tower declares `vision_use_head`; the reader records
/// it on the tower's execution facts and credits the path, so the leaf
/// has a fate instead of grading "read by nothing".
#[test]
fn a_siglip_tower_reads_its_pooling_head_flag_and_credits_the_path() {
    let readings = read_components(&json!({
        "model_type": "gemma3",
        "text_config": { "model_type": "gemma3_text", "hidden_size": 64 },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "image_size": 896,
            "patch_size": 14,
            "vision_use_head": false
        }
    }));
    let vision = readings
        .iter()
        .find(|r| r.topology.name == "vision")
        .expect("the vision component is read");
    assert_eq!(vision.topology.tower.use_head, Some(false));
    assert!(vision
        .consumed_paths
        .contains("vision_config.vision_use_head"));
    assert!(!vision.topology.tower.is_empty());
}
