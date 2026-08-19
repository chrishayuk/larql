//! End-to-end inventory build over an on-disk fixture directory.

use std::io::Write;

use crate::inventory::{build_inventory, KeyStatus, INVENTORY_SCHEMA};

/// Minimal Glimmer-shaped checkpoint: config + one shard.
fn write_fixture(dir: &std::path::Path) {
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "sliding_window": 16,
            "vocab_size": 128,
            "qk_scale_factor": 3.87,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ]
        }
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();

    let header = serde_json::json!({
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "BF16", "shape": [256, 64], "data_offsets": [0, 32768]
        }
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
}

#[test]
fn builds_a_complete_inventory() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());

    let inv = build_inventory(dir.path()).unwrap();
    assert_eq!(inv.schema, INVENTORY_SCHEMA);
    assert_eq!(inv.identity.model_type, "muse_glimmer_text");
    // Glimmer is a registered family since its semantics were judged.
    assert!(!inv.detection.generic_fallback);
    assert_eq!(inv.detection.family, "muse_glimmer");
    assert_eq!(inv.resolved.num_layers, 4);
    assert_eq!(inv.resolved.attention.sliding_layers, 3);
    assert_eq!(inv.tensors.total_tensors, 1);
    assert_eq!(inv.tensors.total_bytes, 32768);

    // The once-dangerous scalar is consumed since G2b; the instrument's
    // negative case is covered by `an_unjudged_key_still_reads_unconsumed`.
    let qk = inv
        .config_keys
        .iter()
        .find(|f| f.path == "text_config.qk_scale_factor")
        .unwrap();
    assert_eq!(qk.status, KeyStatus::Consumed);
}

/// The report round-trips through JSON — it is the CLI contract.
#[test]
fn inventory_round_trips_through_json() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    let inv = build_inventory(dir.path()).unwrap();
    let json = serde_json::to_string_pretty(&inv).unwrap();
    let back: crate::inventory::ArchitectureInventory = serde_json::from_str(&json).unwrap();
    assert_eq!(back.schema, inv.schema);
    assert_eq!(back.config_keys.len(), inv.config_keys.len());
    assert_eq!(back.tensors.total_bytes, inv.tensors.total_bytes);
}

#[test]
fn missing_config_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = build_inventory(dir.path()).unwrap_err();
    assert!(err.to_string().contains("config.json"));
}

#[test]
fn non_directory_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("not_a_dir");
    std::fs::write(&file, "x").unwrap();
    assert!(build_inventory(&file).is_err());
}

/// Minimal gpt-oss-shaped checkpoint: a routed family with a
/// `quantization_config` block and one packed expert tensor.
fn write_routed_fixture(dir: &std::path::Path) {
    let config = serde_json::json!({
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_local_experts": 4,
        "experts_per_token": 2,
        "vocab_size": 128,
        "rope_theta": 150000.0,
        "swiglu_limit": 7.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 16,
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": ["model.layers.*.self_attn", "lm_head"]
        }
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();

    let header = serde_json::json!({
        "model.layers.0.mlp.experts.gate_up_proj_blocks": {
            "dtype": "U8", "shape": [4, 128, 2, 16], "data_offsets": [0, 16384]
        }
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
}

/// A routed checkpoint's inventory records the stored representation it
/// read (crediting the keys as consumed) and binds the one expert bank the
/// shard actually spells; the layer with no spelled bank stays unbound.
#[test]
fn routed_inventory_records_representation_and_binds_spelled_banks() {
    let dir = tempfile::tempdir().unwrap();
    write_routed_fixture(dir.path());

    let inv = build_inventory(dir.path()).unwrap();
    assert_eq!(inv.detection.family, "gpt_oss");
    let representation = inv
        .stored_representation
        .expect("quantization_config was read");
    assert_eq!(representation.method, "mxfp4");
    assert_eq!(representation.excluded_modules.len(), 2);
    for path in [
        "quantization_config.quant_method",
        "quantization_config.modules_to_not_convert",
    ] {
        let fact = inv.config_keys.iter().find(|f| f.path == path).unwrap();
        assert_eq!(fact.status, KeyStatus::Consumed, "{path}");
    }
    assert_eq!(
        inv.resolved.layers[0].expert_bank.as_deref(),
        Some("model.layers.0.mlp.experts")
    );
    assert_eq!(inv.resolved.layers[1].expert_bank, None);
    assert!(inv.resolved.execution.unwrap().moe.is_some());
}

/// A multimodal checkpoint's root interface facts are read by the
/// interface reader and credited as consumed, and the PLE-family knobs
/// are read by the main parser into `ModelConfig` — nothing at the root
/// or in `text_config` is "read by nothing".
#[test]
fn interface_reader_and_parser_credit_the_gemma4_root_and_ple_knobs() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    // Extend the Glimmer-shaped fixture's config with the Gemma-4-style
    // root interface and text knobs.
    let path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    config["audio_config"] = serde_json::Value::Null;
    config["boi_token_id"] = serde_json::json!(255999);
    config["vision_soft_tokens_per_image"] = serde_json::json!(280);
    config["text_config"]["use_bidirectional_attention"] = serde_json::json!("vision");
    config["text_config"]["use_double_wide_mlp"] = serde_json::json!(false);
    config["text_config"]["vocab_size_per_layer_input"] = serde_json::json!(128);
    std::fs::write(&path, serde_json::to_string_pretty(&config).unwrap()).unwrap();

    let inv = build_inventory(dir.path()).unwrap();
    let interface = inv.multimodal_interface.expect("interface read");
    assert_eq!(
        interface.absent_components,
        vec!["audio_config".to_string()]
    );
    assert_eq!(interface.soft_tokens_per_image, Some(280));
    assert_eq!(interface.bidirectional_attention.as_deref(), Some("vision"));
    assert!(interface
        .token_roles
        .contains(&("boi_token_id".to_string(), 255999)));
    for path in [
        "audio_config",
        "boi_token_id",
        "vision_soft_tokens_per_image",
        "text_config.use_bidirectional_attention",
        "text_config.use_double_wide_mlp",
        "text_config.vocab_size_per_layer_input",
    ] {
        let fact = inv.config_keys.iter().find(|f| f.path == path).unwrap();
        assert_eq!(fact.status, KeyStatus::Consumed, "{path}");
    }
}
