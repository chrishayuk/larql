use super::*;
use serde_json::json;

/// GPT-OSS's real block, verbatim.
fn gpt_oss_config() -> Value {
    json!({
        "model_type": "gpt_oss",
        "quantization_config": {
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head"
            ],
            "quant_method": "mxfp4"
        }
    })
}

#[test]
fn reads_gpt_oss_block_and_records_exactly_what_it_read() {
    let r = read_stored_representation(&gpt_oss_config()).expect("declared");
    assert_eq!(r.representation.method, "mxfp4");
    assert_eq!(r.representation.excluded_modules.len(), 4);
    let paths: Vec<&str> = r.consumed_paths.iter().map(String::as_str).collect();
    assert_eq!(
        paths,
        [
            "quantization_config.modules_to_not_convert",
            "quantization_config.quant_method",
        ]
    );
}

#[test]
fn a_checkpoint_without_the_block_declares_nothing() {
    assert!(read_stored_representation(&json!({ "model_type": "llama" })).is_none());
    // A block without a method names no scheme: unread, so unconsumed.
    assert!(read_stored_representation(&json!({ "quantization_config": { "bits": 4 } })).is_none());
}

#[test]
fn a_block_without_the_exclusion_list_consumes_only_the_method() {
    let r = read_stored_representation(&json!({
        "quantization_config": { "quant_method": "mxfp4" }
    }))
    .expect("declared");
    assert!(r.representation.excluded_modules.is_empty());
    assert_eq!(r.consumed_paths.len(), 1);
    assert!(r
        .consumed_paths
        .contains("quantization_config.quant_method"));
}

#[test]
fn exclusion_globs_match_module_prefixes_on_dotted_boundaries() {
    let rep = read_stored_representation(&gpt_oss_config())
        .expect("declared")
        .representation;
    // Excluded: attention, router, embeddings, head.
    assert!(rep.excludes("model.layers.3.self_attn.q_proj.weight"));
    assert!(rep.excludes("model.layers.23.mlp.router.weight"));
    assert!(rep.excludes("model.embed_tokens.weight"));
    assert!(rep.excludes("lm_head.weight"));
    // Not excluded: the experts, whose blocks/scales the scheme applies to.
    assert!(!rep.excludes("model.layers.3.mlp.experts.gate_up_proj_blocks"));
    assert!(!rep.excludes("model.layers.3.mlp.experts.down_proj_scales"));
    // `*` is one segment, on dotted boundaries — no substring accidents.
    assert!(!rep.excludes("model.layers_extra.3.self_attn.q_proj.weight"));
    assert!(!rep.excludes("lm_header.weight"));
}

#[test]
fn round_trips_through_serde() {
    let rep = read_stored_representation(&gpt_oss_config())
        .expect("declared")
        .representation;
    let text = serde_json::to_string(&rep).expect("serialise");
    let back: StoredRepresentation = serde_json::from_str(&text).expect("deserialise");
    assert_eq!(back, rep);
}
