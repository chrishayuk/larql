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

// ── Fine-grained FP8: the three keys the scheme adds ──

fn fp8_config(extra: serde_json::Value) -> serde_json::Value {
    let mut block = serde_json::json!({
        "quant_method": "fp8",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    });
    for (k, v) in extra.as_object().expect("object") {
        block[k] = v.clone();
    }
    serde_json::json!({ "quantization_config": block })
}

#[test]
fn the_fine_grained_fp8_keys_are_read_and_their_paths_recorded() {
    let r = read_stored_representation(&fp8_config(serde_json::json!({})))
        .expect("a declared scheme is read");
    let rep = &r.representation;
    assert_eq!(rep.method, QUANT_METHOD_FP8);
    assert_eq!(rep.fmt.as_deref(), Some(FMT_E4M3));
    assert_eq!(rep.activation_scheme.as_deref(), Some("dynamic"));
    assert_eq!(rep.weight_block_size.as_deref(), Some(&[128, 128][..]));

    // Consumption is a RECORDED fact, not a name match — the planner
    // credits these paths and nothing else.
    for key in [FMT_KEY, WEIGHT_BLOCK_SIZE_KEY, ACTIVATION_SCHEME_KEY] {
        assert!(
            r.consumed_paths
                .contains(&format!("{QUANTIZATION_CONFIG_KEY}.{key}")),
            "`{key}` was read but its path not recorded"
        );
    }
}

/// Both halves are required. `quant_method: "fp8"` alone does not say
/// which element codec, and `e5m2` is a different format of the same
/// byte width — decoding one as the other yields plausible numbers from
/// every byte rather than an error.
#[test]
fn the_scheme_is_only_recognised_with_its_element_format() {
    let with = read_stored_representation(&fp8_config(serde_json::json!({})))
        .expect("read")
        .representation;
    assert!(with.is_finegrained_fp8_e4m3());

    // Case is not the distinction.
    let upper = read_stored_representation(&fp8_config(serde_json::json!({ "fmt": "E4M3" })))
        .expect("read")
        .representation;
    assert!(upper.is_finegrained_fp8_e4m3());

    let other = read_stored_representation(&fp8_config(serde_json::json!({ "fmt": "e5m2" })))
        .expect("read")
        .representation;
    assert!(
        !other.is_finegrained_fp8_e4m3(),
        "e5m2 must not be served as e4m3"
    );

    let mut none_fmt = fp8_config(serde_json::json!({}));
    none_fmt["quantization_config"]
        .as_object_mut()
        .expect("object")
        .remove(FMT_KEY);
    let bare = read_stored_representation(&none_fmt)
        .expect("read")
        .representation;
    assert!(
        !bare.is_finegrained_fp8_e4m3(),
        "an unstated element format is not an assumed one"
    );
    assert_eq!(bare.fmt, None);
}

/// The declared tile is carried as a PAIR or not at all — a malformed
/// declaration answers `None` rather than half a tile.
#[test]
fn the_declared_tile_is_a_pair_or_nothing() {
    let square = read_stored_representation(&fp8_config(serde_json::json!({})))
        .expect("read")
        .representation;
    assert_eq!(square.declared_tile(), Some((128, 128)));

    for bad in [
        serde_json::json!([128]),
        serde_json::json!([128, 128, 128]),
        serde_json::json!([]),
        serde_json::json!("128x128"),
    ] {
        let rep = read_stored_representation(&fp8_config(
            serde_json::json!({ "weight_block_size": bad }),
        ))
        .expect("read")
        .representation;
        assert_eq!(rep.declared_tile(), None, "malformed tile {bad} accepted");
    }
}

/// A scheme that declares none of the three still reads — the keys are
/// optional, and a checkpoint omitting them is not a malformed one.
#[test]
fn a_scheme_without_the_fp8_keys_still_reads() {
    let v = serde_json::json!({
        "quantization_config": { "quant_method": "mxfp4" }
    });
    let rep = read_stored_representation(&v).expect("read").representation;
    assert_eq!(rep.method, "mxfp4");
    assert_eq!(rep.fmt, None);
    assert_eq!(rep.weight_block_size, None);
    assert_eq!(rep.activation_scheme, None);
    assert!(!rep.is_finegrained_fp8_e4m3());
    assert_eq!(rep.declared_tile(), None);
}
