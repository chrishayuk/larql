//! The vindex facts, gated on a real encoded container.
//!
//! The invariant under test: every command answers from the artifact
//! alone — the fixture is encoded through the real pipeline, then the
//! facts must reconstruct identity, directory, derived precision, and
//! self-verification from its bytes, with no source checkpoint
//! present. A corrupted byte must fail verify by name.

use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, hybrid_lllf_f32_model, miniature_glimmer,
};
use vindex_cli::{
    describe_facts, diff_facts, inspect_facts, layers_facts, precision_facts,
    precision_matrix_facts, represent_facts, representations_facts, verify_facts,
};

fn container() -> tempfile::TempDir {
    let checkpoint = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        dir.path(),
        "vindex-cli-fixture",
    );
    dir
}

#[test]
fn inspect_reconstructs_identity_and_census_from_the_artifact_alone() {
    let dir = container();
    let v = inspect_facts(dir.path()).unwrap();
    assert_eq!(v["model"], "vindex-cli-fixture");
    assert_eq!(v["generation"], 3);
    assert!(v["coherent"].as_bool().unwrap(), "{v}");
    assert_eq!(v["components"][0]["id"], "target");
}

#[test]
fn representations_list_the_physical_directory_with_fidelity() {
    let dir = container();
    let v = representations_facts(dir.path()).unwrap();
    let entries = v["entries"].as_array().unwrap();
    assert!(!entries.is_empty());
    assert!(
        entries
            .iter()
            .all(|e| e["payload_bytes"].as_u64().unwrap() > 0),
        "{v}"
    );
}

#[test]
fn describe_finds_an_object_by_suffix_and_shows_its_tensor_table() {
    let dir = container();
    let v = describe_facts(dir.path(), "decoder_stack", 4, None).unwrap();
    assert_eq!(v["object"]["id"], "target.decoder_stack");
    let head = &v["directory"][0]["tensor_table_head"];
    assert!(!head.as_array().unwrap().is_empty(), "{v}");
}

#[test]
fn describe_peek_decodes_the_actual_values_of_one_tensor() {
    let dir = container();
    let v = describe_facts(
        dir.path(),
        "decoder_stack",
        4,
        Some("input_layernorm.weight"),
    )
    .unwrap();
    let vals = v["peek"]["values"].as_array().unwrap();
    assert_eq!(vals.len(), 4, "{v}");
    assert!(vals.iter().all(|x| x.as_f64().unwrap().is_finite()), "{v}");

    let err = describe_facts(dir.path(), "decoder_stack", 4, Some("no.such.tensor")).unwrap_err();
    assert!(err.contains("tensors:"), "{err}");
}

#[test]
fn describe_refuses_an_unknown_address_by_naming_the_holdings() {
    let dir = container();
    let err = describe_facts(dir.path(), "no.such.object", 4, None).unwrap_err();
    assert!(err.contains("the graph holds"), "{err}");
}

#[test]
fn precision_is_derived_from_bytes_over_elements_never_asserted() {
    let dir = container();
    let v = precision_facts(dir.path()).unwrap();
    let eff = v["stored_bits_per_weight_slot"].as_f64().unwrap();
    // The miniature encodes F32 source tensors: 32 bits per weight.
    assert!(
        (eff - 32.0).abs() < 0.5,
        "effective {eff} — expected ~32 for the F32 fixture"
    );
}

/// The dense (16-aligned) fixture, encoded and then compiled to NVFP4
/// through `represent_facts` — the two-representation container the
/// diff tests interrogate.
fn compiled_container() -> (tempfile::TempDir, serde_json::Value) {
    let checkpoint = tempfile::tempdir().unwrap();
    let src = tempfile::tempdir().unwrap();
    encode_fixture_container(
        dense_f32_model,
        checkpoint.path(),
        src.path(),
        "vindex-cli-dense",
    );
    let out = tempfile::tempdir().unwrap();
    let report = represent_facts(src.path(), out.path(), "nvfp4").unwrap();
    (out, report)
}

#[test]
fn represent_compiles_a_pack_and_the_result_still_verifies() {
    let (out, report) = compiled_container();
    let compiled = report["compiled"].as_array().unwrap();
    assert!(!compiled.is_empty(), "{report}");
    assert!(
        compiled[0]["compression"].as_f64().unwrap() > 1.0,
        "{report}"
    );
    let v = verify_facts(out.path()).unwrap();
    assert!(v["verified"].as_bool().unwrap(), "{v}");
}

#[test]
fn diff_of_a_representation_against_itself_is_identical() {
    let (out, report) = compiled_container();
    let object = report["compiled"][0]["object"].as_str().unwrap();
    let v = diff_facts(out.path(), "F32", "F32", object, 4, None).unwrap();
    assert!(v["identical"].as_bool().unwrap(), "{v}");
    assert_eq!(v["rms_error"], 0.0, "{v}");
}

#[test]
fn diff_derives_a_nonzero_error_between_source_and_compiled_bytes() {
    let (out, report) = compiled_container();
    let object = report["compiled"][0]["object"].as_str().unwrap();
    let v = diff_facts(out.path(), "F32", "NVFP4", object, 4, None).unwrap();
    assert!(!v["identical"].as_bool().unwrap(), "{v}");
    assert!(v["rms_error"].as_f64().unwrap() > 0.0, "{v}");
    assert!(
        v["max_error"].as_f64().unwrap() >= v["rms_error"].as_f64().unwrap(),
        "{v}"
    );
    let rows = v["values"]["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 4, "{v}");
}

#[test]
fn diff_refuses_an_encoding_the_container_does_not_hold() {
    let (out, report) = compiled_container();
    let object = report["compiled"][0]["object"].as_str().unwrap();
    let err = diff_facts(out.path(), "F32", "INT8", object, 4, None).unwrap_err();
    assert!(err.contains("the container holds"), "{err}");
}

#[test]
fn semantic_addresses_resolve_through_the_plan_not_filenames() {
    let (out, _) = compiled_container();
    let v = describe_facts(out.path(), "layer.0.ffn.down", 4, None).unwrap();
    assert_eq!(v["role"], "FFN DOWN PROJECTION", "{v}");
    assert_eq!(v["values"].as_array().unwrap().len(), 4, "{v}");
    let encs: Vec<&str> = v["representations"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["encoding"].as_str())
        .collect();
    assert!(encs.contains(&"NVFP4"), "{v}");

    let err = describe_facts(out.path(), "layer.99.ffn.down", 4, None).unwrap_err();
    assert!(err.contains("the plan holds layers"), "{err}");
}

#[test]
fn a_semantic_diff_scopes_to_its_one_tensor() {
    let (out, _) = compiled_container();
    let v = diff_facts(out.path(), "F32", "NVFP4", "layer.0.ffn.down", 4, None).unwrap();
    assert_eq!(v["tensors"].as_array().unwrap().len(), 1, "{v}");
    assert!(v["rms_error"].as_f64().unwrap() > 0.0, "{v}");
}

#[test]
fn the_precision_matrix_reads_the_compiled_representation_by_programme() {
    let (out, _) = compiled_container();
    let v = precision_matrix_facts(out.path()).unwrap();
    let programmes = v["programmes"].as_array().unwrap();
    assert!(!programmes.is_empty(), "{v}");
    let rows = programmes[0]["rows"].as_array().unwrap();
    let down = rows[0]["bits"]["down"].as_f64().unwrap();
    assert!(
        (down - 4.5).abs() < 0.1,
        "compiled down at {down} — expected ~4.5\n{v}"
    );
    assert!(
        !v["surfaces"].as_array().unwrap().is_empty(),
        "embedding/head surfaces expected: {v}"
    );
}

#[test]
fn a_deltanet_layer_refuses_qkvo_and_a_full_attention_layer_answers() {
    let checkpoint = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    encode_fixture_container(
        hybrid_lllf_f32_model,
        checkpoint.path(),
        dir.path(),
        "vindex-cli-hybrid",
    );
    // LLLF cadence: layers 0–2 are GatedDelta recurrences, layer 3 attends.
    let err = describe_facts(dir.path(), "layer.0.attention.q", 2, None).unwrap_err();
    assert!(err.contains("token mixer is GATED DELTANET"), "{err}");
    assert!(err.contains("layer.0.mixer"), "{err}");

    let v = describe_facts(dir.path(), "layer.3.attention.q", 2, None).unwrap();
    assert_eq!(v["role"], "ATTENTION QUERY PROJECTION", "{v}");
    assert_eq!(v["values"].as_array().unwrap().len(), 2, "{v}");
}

#[test]
fn the_mixer_is_a_first_class_semantic_component() {
    let checkpoint = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    encode_fixture_container(
        hybrid_lllf_f32_model,
        checkpoint.path(),
        dir.path(),
        "vindex-cli-hybrid",
    );
    let v = describe_facts(dir.path(), "layer.0.mixer", 2, None).unwrap();
    assert_eq!(v["mixer"], "GATED DELTANET", "{v}");
    let roles: Vec<&str> = v["operands"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|o| o["role"].as_str())
        .collect();
    assert!(roles.contains(&"fused recurrent q|k|v"), "{roles:?}");
    assert!(roles.contains(&"decay projection"), "{roles:?}");

    let v = describe_facts(dir.path(), "layer.3.mixer", 2, None).unwrap();
    assert!(
        v["mixer"] == "SOFTMAX ATTENTION" || v["mixer"] == "GATED ATTENTION",
        "{v}"
    );

    let l = layers_facts(dir.path()).unwrap();
    let mixers: Vec<&str> = l["layers"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|r| r["mixer"].as_str())
        .collect();
    assert_eq!(mixers.len(), 4, "{l}");
    assert_eq!(mixers[0], "GATED DELTANET", "{l}");
    assert_ne!(mixers[3], "GATED DELTANET", "{l}");
}

#[test]
fn verify_passes_on_the_intact_artifact_and_fails_on_a_flipped_byte() {
    let dir = container();
    let v = verify_facts(dir.path()).unwrap();
    assert!(v["verified"].as_bool().unwrap(), "{v}");

    // Flip one payload byte in one segment: verify must fail, by name.
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("index.json")).unwrap())
            .unwrap();
    let segment = index["representations"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()["segment"]
        .as_str()
        .unwrap()
        .to_string();
    let path = dir.path().join(&segment);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&path, bytes).unwrap();

    let v = verify_facts(dir.path()).unwrap();
    assert!(
        !v["verified"].as_bool().unwrap(),
        "corruption must not verify: {v}"
    );
    assert_eq!(v["failures"], 1, "{v}");
}
