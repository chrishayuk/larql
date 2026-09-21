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

/// Gemma 3's spellings of the image join are read under their own names
/// and credited by path, and the soft-token count answers from
/// `mm_tokens_per_image` where Gemma 4 says `vision_soft_tokens_per_image`.
/// Values are `google/gemma-3-12b-it`'s own.
#[test]
fn gemma3_spellings_are_read_and_credited() {
    let reading = read_interface(&serde_json::json!({
        "boi_token_index": 255999,
        "eoi_token_index": 256000,
        "image_token_index": 262144,
        "mm_tokens_per_image": 256,
    }))
    .expect("declares an interface");
    let i = &reading.interface;
    assert_eq!(
        i.token_roles,
        vec![
            ("image_token_index".to_string(), 262144),
            ("boi_token_index".to_string(), 255999),
            ("eoi_token_index".to_string(), 256000),
        ]
    );
    assert_eq!(i.soft_tokens_per_image, Some(256));
    for path in [
        "boi_token_index",
        "eoi_token_index",
        "image_token_index",
        "mm_tokens_per_image",
    ] {
        assert!(reading.consumed_paths.contains(path), "{path} credited");
    }
    assert_eq!(
        reading.consumed_paths.len(),
        4,
        "nothing credited that was not read"
    );
}

/// One fact under two spellings: the first present answers, and only the
/// spelling actually read is credited — the other stays honestly unread.
#[test]
fn soft_token_count_credits_only_the_spelling_it_read() {
    let reading = read_interface(&serde_json::json!({
        "vision_soft_tokens_per_image": 280,
        "mm_tokens_per_image": 256,
    }))
    .expect("declares an interface");
    assert_eq!(reading.interface.soft_tokens_per_image, Some(280));
    assert!(reading
        .consumed_paths
        .contains("vision_soft_tokens_per_image"));
    assert!(!reading.consumed_paths.contains("mm_tokens_per_image"));
}
