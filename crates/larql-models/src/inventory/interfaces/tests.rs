//! The multimodal interface reader.

use super::*;
use serde_json::Value;

fn gemma4_shaped() -> Value {
    serde_json::json!({
        "audio_config": null,
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "boi_token_id": 255999,
        "eoa_token_id": 258883,
        "eoa_token_index": 258883,
        "eoi_token_id": 258882,
        "image_token_id": 258880,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
        "text_config": { "use_bidirectional_attention": "vision" }
    })
}

/// Every declared interface fact is read, recorded, and credited by
/// full path — nothing is credited that was not read.
#[test]
fn reads_every_declared_interface_fact_and_credits_exactly_those_paths() {
    let reading = read_interface(&gemma4_shaped()).expect("declares an interface");
    let i = &reading.interface;
    assert_eq!(i.token_roles.len(), 8);
    assert!(i
        .token_roles
        .contains(&("image_token_id".to_string(), 258880)));
    assert_eq!(i.soft_tokens_per_image, Some(280));
    assert_eq!(i.absent_components, vec!["audio_config".to_string()]);
    assert_eq!(i.bidirectional_attention.as_deref(), Some("vision"));
    for path in [
        "audio_config",
        "eoa_token_index",
        "vision_soft_tokens_per_image",
        "text_config.use_bidirectional_attention",
    ] {
        assert!(reading.consumed_paths.contains(path), "{path}");
    }
    assert_eq!(reading.consumed_paths.len(), 11);
}

/// A present (non-null) `audio_config` is a component, not an absence:
/// it is left to the component reader and not credited here.
#[test]
fn a_present_optional_component_is_not_recorded_as_absent() {
    let mut config = gemma4_shaped();
    config["audio_config"] = serde_json::json!({ "hidden_size": 8 });
    let reading = read_interface(&config).unwrap();
    assert!(reading.interface.absent_components.is_empty());
    assert!(!reading.consumed_paths.contains("audio_config"));
}

/// A text-only config declares no interface: `None`, no credit.
#[test]
fn a_text_only_config_has_no_interface() {
    let config = serde_json::json!({ "model_type": "llama", "hidden_size": 64 });
    assert!(read_interface(&config).is_none());
}
