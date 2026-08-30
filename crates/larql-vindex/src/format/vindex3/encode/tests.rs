//! Encode → inspect round trip: the G3 gate, on the fixture system.

use std::io::{Read, Seek, SeekFrom};

use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::encode::{encode_system, SEGMENTS_DIR, SYSTEM_GRAPH_JSON};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense, payload_pattern,
};

fn glimmer_system() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    Vec<(String, larql_models::inventory::ArchitectureInventory)>,
) {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target_dir.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    (target_dir, drafter_dir, named)
}

/// THE G3 gate: encode the pair, then reconstruct the system solely from
/// the container — components, topology (incl. NoPE), the edge — with a
/// coherent directory and no source access.
#[test]
fn encode_then_inspect_reconstructs_the_system_without_the_source() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    let outcome = encode_system(&named, out.path()).unwrap();
    assert!(outcome.representations >= 7);
    assert!(outcome.total_payload_bytes > 0);

    // Sources gone: move nothing, read nothing — inspect uses only `out`.
    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(
        inspection.is_coherent(),
        "defects: {:?}",
        inspection.defects
    );

    let target = inspection
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    assert_eq!(target.num_layers, 8);
    assert_eq!(target.hidden_size, 64);
    assert_eq!(target.sliding_layers, Some(6));
    assert_eq!(target.full_layers, Some(2));
    assert_eq!(target.nope_layers, Some(2));
    assert_eq!(target.window, Some(16));

    let draft = inspection
        .components
        .iter()
        .find(|c| c.id == "draft")
        .unwrap();
    assert_eq!(draft.num_layers, 2);

    assert_eq!(inspection.graph.edges.len(), 1);
    let edge = &inspection.graph.edges[0];
    assert_eq!(edge.producer_layers, vec![1, 3, 5]);
    assert_eq!(edge.block_size, Some(4));
    assert_eq!(edge.consumer_object, "draft.feature_projector");
}

/// Payload bytes survive the trip exactly: read a tensor back out of its
/// segment via the header table and compare with the deterministic source
/// pattern.
#[test]
fn payload_bytes_round_trip_exactly() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    // The drafter's projector: encoder.fc.weight is 192*64 BF16 = 24576
    // bytes at source offsets [8320, 32896).
    let segment_path = out
        .path()
        .join(SEGMENTS_DIR)
        .join("draft.feature_projector.bin");
    let (header, payload_start) = read_segment_header(&segment_path).unwrap();
    let entry = header
        .tensors
        .iter()
        .find(|t| t.name == "fc.weight")
        .expect("object-relative name for the fusion tensor");
    assert_eq!(entry.shape, vec![192, 64]);

    let mut file = std::fs::File::open(&segment_path).unwrap();
    file.seek(SeekFrom::Start(payload_start + entry.offset))
        .unwrap();
    let mut encoded = vec![0u8; entry.len as usize];
    file.read_exact(&mut encoded).unwrap();

    // Source pattern over the whole shard payload region, sliced at the
    // tensor's declared source offsets.
    let source_slice = &payload_pattern(32896)[8320..32896];
    assert_eq!(encoded, source_slice, "payload bytes differ from source");
}

/// Object-relative names carry no artifact-global prefixes.
#[test]
fn segment_tensor_names_are_object_relative() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let (header, _) = read_segment_header(
        &out.path()
            .join(SEGMENTS_DIR)
            .join("target.decoder_stack.bin"),
    )
    .unwrap();
    for tensor in &header.tensors {
        assert!(
            !tensor.name.starts_with("model."),
            "artifact-global name leaked: {}",
            tensor.name
        );
    }
    // Multi-binding object: names stay unique after prefix stripping.
    let (header, _) = read_segment_header(
        &out.path()
            .join(SEGMENTS_DIR)
            .join("draft.feature_projector.bin"),
    )
    .unwrap();
    let names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["fc.weight", "output_norm_enc.weight"]);
}

/// Encoding is deterministic: same inputs, same hashes.
#[test]
fn encode_is_deterministic() {
    let (_a, _b, named) = glimmer_system();
    let out1 = tempfile::tempdir().unwrap();
    let out2 = tempfile::tempdir().unwrap();
    encode_system(&named, out1.path()).unwrap();
    encode_system(&named, out2.path()).unwrap();
    let read = |p: &std::path::Path| {
        std::fs::read_to_string(p.join(crate::format::filenames::INDEX_JSON)).unwrap()
    };
    assert_eq!(read(out1.path()), read(out2.path()));
}

/// An inadmissible plan is refused before a single byte is written.
#[test]
fn inadmissible_plan_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    inventory
        .config_keys
        .push(larql_models::inventory::ConfigKeyFact {
            path: "some_future_field_nobody_reviewed".to_string(),
            value: serde_json::json!(42),
            status: larql_models::inventory::KeyStatus::Unconsumed,
        });
    let named = vec![("llama-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    let err = encode_system(&named, out.path()).unwrap_err();
    assert!(err.to_string().contains("inadmissible"), "{err}");
    assert!(
        !out.path()
            .join(crate::format::filenames::INDEX_JSON)
            .exists(),
        "a refused encode must not leave an index behind"
    );
}

/// A corrupted segment byte is caught by `inspect --verify` — with no
/// source access.
#[test]
fn verify_catches_a_flipped_payload_byte() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    let victim = out.path().join(SEGMENTS_DIR).join("target.embedding.bin");
    let mut bytes = std::fs::read(&victim).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&victim, bytes).unwrap();

    let clean = inspect_container(out.path(), false).unwrap();
    assert!(
        clean.is_coherent(),
        "structure still coherent without verify"
    );
    let verified = inspect_container(out.path(), true).unwrap();
    assert!(!verified.is_coherent());
    assert!(verified
        .defects
        .iter()
        .any(|d| format!("{d:?}").contains("target.embedding")));
}

/// The graph manifest is written verbatim and reloads as the same graph.
#[test]
fn graph_manifest_round_trips() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let graph: crate::format::vindex3::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.path().join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();
    assert!(graph.validate().is_empty());
    assert_eq!(graph.edges.len(), 1);
}

/// A known dense model encodes and reconstructs the same way — the path is
/// generic, not Glimmer-shaped.
#[test]
fn known_dense_encodes_and_inspects() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(inspection.is_coherent(), "{:?}", inspection.defects);
    assert!(inspection.graph.edges.is_empty());
    assert!(inspection
        .index
        .representations
        .keys()
        .any(|k| k.starts_with("target.embedding@")));
}

/// A header length past the sanity bound is refused, not allocated.
///
/// The guard exists because the length prefix is the first thing read from
/// an untrusted file: without it a corrupt or hostile prefix becomes a
/// multi-gigabyte `vec![0; header_len]` before anything has been validated.
/// The refusal must name the claimed size, so a reader can tell a corrupt
/// file from a merely unsupported one.
#[test]
fn a_header_length_past_the_bound_is_refused_before_allocating() {
    use std::io::Write;
    const ABSURD_HEADER_LEN: u64 = 512 * 1024 * 1024;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.segment");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&ABSURD_HEADER_LEN.to_le_bytes()).unwrap();
    // Deliberately no header body: the guard must fire on the prefix alone,
    // before any attempt to read that many bytes.
    f.flush().unwrap();
    drop(f);

    let err = read_segment_header(&path).unwrap_err().to_string();
    assert!(
        err.contains(&ABSURD_HEADER_LEN.to_string()),
        "refusal must name the claimed length: {err}"
    );
    assert!(err.contains("corrupt"), "{err}");
}

/// A header length inside the bound but with no body behind it is a short
/// read, not the corruption refusal — the two failures are different and a
/// reader must not be told the wrong one.
#[test]
fn a_truncated_header_is_a_short_read_not_a_corruption_claim() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.segment");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&64u64.to_le_bytes()).unwrap();
    f.write_all(b"not sixty-four bytes").unwrap();
    f.flush().unwrap();
    drop(f);

    let err = read_segment_header(&path).unwrap_err().to_string();
    assert!(
        !err.contains("corrupt"),
        "a short body is not the over-long-header failure: {err}"
    );
}

/// An object whose bindings select nothing is a disagreement between the
/// graph and the inventory, and it must be named as such.
///
/// Silently planning zero tensors would write a well-formed segment holding
/// no bytes — a container that passes every structural check and cannot
/// serve the object it claims to carry.
#[test]
fn an_object_matching_no_source_tensors_refuses() {
    use crate::format::vindex3::encode::plan_object_tensors;
    use crate::format::vindex3::graph::object::{
        LogicalObject, ObjectKind, Representation, SourceBinding,
    };
    use std::collections::BTreeMap;

    let dir = tempfile::tempdir().unwrap();
    let inventory = known_dense(dir.path());
    let mut inventories = BTreeMap::new();
    inventories.insert("only-artifact", &inventory);

    let object = LogicalObject {
        id: "target.embedding".to_string(),
        component: "target".to_string(),
        kind: ObjectKind::Embedding,
        source_bindings: vec![SourceBinding {
            artifact: "only-artifact".to_string(),
            // A prefix no tensor in the inventory carries.
            tensor_prefix: "no.such.prefix".to_string(),
            tensors: 0,
            bytes: 0,
        }],
        representations: Vec::<Representation>::new(),
    };

    let err = match plan_object_tensors(&object, &inventories, std::slice::from_ref(&object)) {
        Err(e) => e.to_string(),
        Ok(planned) => panic!(
            "planned {} tensors from a prefix nothing matches",
            planned.len()
        ),
    };
    assert!(err.contains("target.embedding"), "{err}");
    assert!(
        err.contains("bindings and inventory disagree"),
        "the refusal must say which two things disagree: {err}"
    );
}

/// `binding_owner` matches a binding's prefix only at a segment boundary.
///
/// A plain `starts_with` would make `model.layers_extra.0` resolve to the
/// binding for `model.layers`, quietly filing one object's tensors under
/// another. The boundary check is the thing being pinned, so the negative
/// case has to be a name that *shares a textual prefix* and is still not a
/// match — a name that merely differs would pass either way.
#[test]
fn binding_owner_matches_on_segment_boundaries_not_text_prefixes() {
    use crate::format::vindex3::encode::binding_owner;
    use crate::format::vindex3::graph::object::{
        LogicalObject, ObjectKind, Representation, SourceBinding,
    };

    let object = LogicalObject {
        id: "target.stack".to_string(),
        component: "target".to_string(),
        kind: ObjectKind::DecoderStack,
        source_bindings: vec![SourceBinding {
            artifact: "art".to_string(),
            tensor_prefix: "model.layers".to_string(),
            tensors: 1,
            bytes: 1,
        }],
        representations: Vec::<Representation>::new(),
    };

    assert_eq!(binding_owner(&object, "model.layers"), Some("art"));
    assert_eq!(binding_owner(&object, "model.layers.0.attn"), Some("art"));
    assert_eq!(binding_owner(&object, "model.layers_extra.0"), None);
    assert_eq!(binding_owner(&object, "other.tensor"), None);
}

// ── Directory/segment coherence: what inspection is FOR ──
//
// A container is two agreeing descriptions of the same bytes — the
// directory in `index.json` and each segment's own header. Inspection
// exists to notice when they stop agreeing, so every disagreement it can
// name needs a case where it actually names it. These tamper with the
// directory only: the segments stay exactly as written, so any defect
// reported is a real disagreement and not a corrupted fixture.

/// Encode the dense fixture, then rewrite `index.json` through `edit`.
fn tampered_directory(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("only-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    let index_path = out.path().join(crate::format::filenames::INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    edit(&mut index);
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    // Rendered as Debug: `InspectionDefect` carries its message as a
    // payload rather than implementing Display, and the message is what
    // these tests are about.
    inspect_container(out.path(), false)
        .unwrap()
        .defects
        .iter()
        .map(|d| format!("{d:?}"))
        .collect()
}

/// The first representation entry's id, for tests that need to name one.
fn first_representation(index: &serde_json::Value) -> String {
    index["representations"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone()
}

#[test]
fn a_directory_entry_naming_an_unknown_object_is_a_defect() {
    let defects = tampered_directory(|index| {
        let id = first_representation(index);
        index["representations"][&id]["object"] = serde_json::json!("target.no_such_object");
    });
    assert!(
        defects
            .iter()
            .any(|d| d.contains("references unknown object")
                && d.contains("target.no_such_object")),
        "{defects:?}"
    );
}

#[test]
fn a_directory_tensor_count_disagreeing_with_the_segment_is_a_defect() {
    let defects = tampered_directory(|index| {
        let id = first_representation(index);
        index["representations"][&id]["tensor_count"] = serde_json::json!(9_999);
    });
    assert!(
        defects
            .iter()
            .any(|d| d.contains("tensors, directory says 9999")),
        "{defects:?}"
    );
}

#[test]
fn a_directory_payload_size_disagreeing_with_the_file_is_a_defect() {
    let defects = tampered_directory(|index| {
        let id = first_representation(index);
        index["representations"][&id]["payload_bytes"] = serde_json::json!(1);
    });
    assert!(
        defects
            .iter()
            .any(|d| d.contains("bytes, expected") && d.contains("payload")),
        "{defects:?}"
    );
}

/// A clean container reports no defects and a complete execution surface —
/// the control the three tampering tests above are measured against. Without
/// it, "defect found" could just mean the fixture never inspects clean.
#[test]
fn an_untampered_container_inspects_clean_and_complete() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("only-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(inspection.is_coherent(), "{:?}", inspection.defects);
    assert!(
        inspection.execution_completeness().is_empty(),
        "{:?}",
        inspection.execution_completeness()
    );
}

// ── Reading an untrusted checkpoint ──
//
// `ArtifactSource` is the one place the encoder touches bytes it did not
// write. Every refusal below is about a shard that parses far enough to
// look usable and is not: the guards exist so a malformed checkpoint is
// named at open, not discovered as a wrong byte range halfway through a
// multi-gigabyte encode.

/// Write a `.safetensors` shard: 8-byte LE header length, header, payload.
fn write_shard(path: &std::path::Path, header: &str, payload: &[u8]) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(payload).unwrap();
    f.flush().unwrap();
}

fn open_err(dir: &std::path::Path) -> String {
    use crate::format::vindex3::encode::source::ArtifactSource;
    match ArtifactSource::open(dir) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("opened a shard that should have been refused"),
    }
}

#[test]
fn a_shard_header_past_the_bound_is_refused_before_allocating() {
    use std::io::Write;
    const ABSURD: u64 = 512 * 1024 * 1024;

    let dir = tempfile::tempdir().unwrap();
    let mut f = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
    // Length prefix only: the guard must fire on it, without ever trying
    // to read (or allocate) the half-gigabyte it claims.
    f.write_all(&ABSURD.to_le_bytes()).unwrap();
    f.flush().unwrap();
    drop(f);

    let err = open_err(dir.path());
    assert!(err.contains(&ABSURD.to_string()), "{err}");
    assert!(err.contains("corrupt"), "{err}");
}

#[test]
fn a_shard_header_that_is_not_an_object_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    // Valid JSON, wrong shape — the case a bare `serde_json` parse accepts.
    write_shard(&dir.path().join("model.safetensors"), "[1, 2, 3]", b"");
    assert!(open_err(dir.path()).contains("header is not an object"));
}

#[test]
fn a_tensor_without_usable_data_offsets_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    // `end < start` is the interesting case: the field is present and
    // well-typed, so only the ordering check rejects it. A missing field
    // would be caught by any parse.
    write_shard(
        &dir.path().join("model.safetensors"),
        r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}"#,
        b"",
    );
    let err = open_err(dir.path());
    assert!(err.contains("`weight`"), "{err}");
    assert!(err.contains("data_offsets"), "{err}");
}

#[test]
fn the_hf_shard_index_selects_and_dedupes_the_shards_it_names() {
    use crate::format::vindex3::encode::source::ArtifactSource;

    let dir = tempfile::tempdir().unwrap();
    let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    write_shard(&dir.path().join("one.safetensors"), header, &[0u8; 4]);
    let header_b = r#"{"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    write_shard(&dir.path().join("two.safetensors"), header_b, &[0u8; 4]);
    // A shard the index does NOT name: the index must be authoritative, so
    // this one's tensor must not be locatable afterwards.
    let header_c = r#"{"c":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    write_shard(&dir.path().join("three.safetensors"), header_c, &[0u8; 4]);

    // `a` appears twice on purpose — a weight_map names a file per TENSOR,
    // so a real multi-tensor shard is listed many times and must be deduped.
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"a":"one.safetensors","a2":"one.safetensors","b":"two.safetensors"}}"#,
    )
    .unwrap();

    let source = ArtifactSource::open(dir.path()).unwrap();
    assert!(source.locate("a").is_ok());
    assert!(source.locate("b").is_ok());
    assert!(
        source.locate("c").is_err(),
        "a shard the index does not name must not be indexed"
    );
}

#[test]
fn a_tensor_absent_from_every_shard_is_named_in_the_refusal() {
    use crate::format::vindex3::encode::source::ArtifactSource;

    let dir = tempfile::tempdir().unwrap();
    write_shard(
        &dir.path().join("model.safetensors"),
        r#"{"present":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0u8; 4],
    );
    let source = ArtifactSource::open(dir.path()).unwrap();
    let err = match source.locate("missing.tensor") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("located a tensor no shard carries"),
    };
    assert!(err.contains("missing.tensor"), "{err}");
    // The message must point at the likely cause — the directory moving
    // under an inventory taken earlier — not just report absence.
    assert!(err.contains("changed since inspection"), "{err}");
}

#[test]
fn a_container_recording_no_system_graph_is_refused_by_inspection() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("only-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    let index_path = out.path().join(crate::format::filenames::INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    index["system_graph"] = serde_json::Value::Null;
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    // A refusal, not an empty inspection: there is a real container shape
    // that records no graph (a routed-MoE bank), and reporting it as "no
    // defects" would say the container inspected clean when nothing was
    // inspected at all.
    let err = match inspect_container(out.path(), false) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("inspected a container with no graph to reconstruct"),
    };
    assert!(err.contains("records no system graph"), "{err}");
    assert!(
        err.contains("larql show"),
        "the refusal must point at the tool that does open such a container: {err}"
    );
}

#[test]
fn a_segment_claiming_a_different_representation_than_the_directory_is_a_defect() {
    let defects = tampered_directory(|index| {
        let id = first_representation(index);
        // Re-key the entry: the segment's own header still carries the old
        // id, so the directory and the file now name different things.
        let entry = index["representations"][&id].clone();
        index["representations"]
            .as_object_mut()
            .unwrap()
            .remove(&id);
        index["representations"][format!("{id}-renamed")] = entry;
    });
    assert!(
        defects.iter().any(|d| d.contains("says it materialises")),
        "{defects:?}"
    );
}

// ── checkpoint.rs: the shared one-checkpoint pipeline (rung M2) ──

/// The full pipeline on the miniature checkpoint: encode succeeds, the
/// container detects as V3, and the capability files present in the
/// checkpoint (tokenizer.json here) are placed beside the segments.
#[test]
fn encode_checkpoint_produces_a_v3_container_with_capabilities() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
    use crate::format::vindex3::fixtures::miniature_glimmer;

    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    std::fs::write(checkpoint.path().join("tokenizer.json"), "{}").unwrap();
    std::fs::write(checkpoint.path().join("generation_config.json"), "{}").unwrap();

    let out = tempfile::tempdir().unwrap();
    let encoded = encode_checkpoint(checkpoint.path(), out.path()).unwrap();

    assert!(encoded.outcome.representations > 0);
    assert_eq!(
        crate::format::generation::detect_generation(out.path()).unwrap(),
        crate::format::generation::ContainerGeneration::V3,
    );
    assert_eq!(
        encoded.capabilities,
        vec![
            "tokenizer.json".to_string(),
            "generation_config.json".to_string()
        ],
        "present capability files copy, in declaration order"
    );
    assert!(out.path().join("tokenizer.json").exists());
    assert!(out.path().join("generation_config.json").exists());
    assert!(!out.path().join("chat_template.jinja").exists());
}

/// A tokenizer-less checkpoint still encodes — absence narrows
/// capability, it is not an error.
#[test]
fn encode_checkpoint_tolerates_absent_capability_files() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
    use crate::format::vindex3::fixtures::miniature_glimmer;

    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let out = tempfile::tempdir().unwrap();
    let encoded = encode_checkpoint(checkpoint.path(), out.path()).unwrap();
    assert!(encoded.capabilities.is_empty());
    assert!(!out.path().join("tokenizer.json").exists());
}

/// A directory that is not an HF checkpoint refuses by name — the
/// message says what the encoder consumes.
#[test]
fn encode_checkpoint_refuses_a_non_checkpoint_dir() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;

    let not_a_checkpoint = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let err = encode_checkpoint(not_a_checkpoint.path(), out.path()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("config.json + safetensors"), "{msg}");
}

/// The artifact name is the checkpoint directory's stem — the same rule
/// `larql vindex3 encode` applies.
#[test]
fn encode_checkpoint_names_the_artifact_by_directory_stem() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
    use crate::format::vindex3::fixtures::miniature_glimmer;

    let base = tempfile::tempdir().unwrap();
    let checkpoint = base.path().join("mini-glimmer");
    std::fs::create_dir(&checkpoint).unwrap();
    miniature_glimmer(&checkpoint);
    let out = tempfile::tempdir().unwrap();
    let encoded = encode_checkpoint(&checkpoint, out.path()).unwrap();
    assert_eq!(encoded.artifact, "mini-glimmer");
}

/// An inadmissible plan refuses with the blocking findings ITEMISED —
/// `encode_system`'s own gate discards them and points at `vindex3
/// plan`; the shared pipeline must not make two surfaces do that dance.
#[test]
fn encode_checkpoint_renders_blocking_findings_into_the_refusal() {
    use crate::format::vindex3::encode::checkpoint::encode_checkpoint;

    // A drafter checkpoint alone is inadmissible: its producer
    // interface cannot resolve without the target artifact.
    let checkpoint = tempfile::tempdir().unwrap();
    drafter_shaped(checkpoint.path());

    let out = tempfile::tempdir().unwrap();
    let err = encode_checkpoint(checkpoint.path(), out.path()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("blocking finding"), "{msg}");
    assert!(
        msg.lines().count() > 1,
        "findings must be itemised, not counted: {msg}"
    );
}

/// **Every segment the encoder writes starts its payload aligned.**
///
/// A misaligned payload puts every tensor in the segment at a
/// misaligned address, which a zero-copy backend cannot bind — and on
/// Metal binds anyway as garbage, with a command buffer that reports
/// success. Checked across many tensor-table sizes, because the header
/// is JSON and its length is whatever the table happens to serialise to.
#[test]
fn every_segment_starts_its_payload_on_the_declared_alignment() {
    use crate::format::vindex3::encode::segment::{
        realign_segment, write_segment, SEGMENT_PAYLOAD_ALIGN,
    };

    use crate::format::vindex3::encode::segment::PlannedTensor;

    let dir = std::env::temp_dir().join(format!("larql-align-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    for tensors in 1..24usize {
        // Names of VARYING length, because the header is JSON and its
        // byte count — hence the payload's start — follows the table's
        // spelling, not the tensor count.
        let planned: Vec<PlannedTensor> = (0..tensors)
            .map(|i| {
                let name = format!("t{i}{}", "x".repeat(i % 7));
                PlannedTensor {
                    relative_name: name.clone(),
                    source_name: name,
                    dtype: "BF16".into(),
                    shape: vec![16],
                    len: 32,
                }
            })
            .collect();
        let path = dir.join(format!("seg_{tensors}.bin"));
        write_segment(&path, "test@BF16", planned, |_name, w, hash| {
            let bytes = [7u8; 32];
            w.write_all(&bytes).expect("payload");
            hash(&bytes);
            Ok(32)
        })
        .expect("write");
        let (_, payload_start) =
            crate::format::vindex3::encode::segment::read_segment_header(&path).expect("header");
        assert!(
            payload_start.is_multiple_of(SEGMENT_PAYLOAD_ALIGN as u64),
            "{tensors} tensors put the payload at {payload_start}, not a multiple of \
             {SEGMENT_PAYLOAD_ALIGN}"
        );
        // Realigning an already-aligned segment is a no-op in place.
        let (before, after) = realign_segment(&path, &path).expect("realign");
        assert_eq!((before, after), (payload_start, payload_start));
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// **Realigning a legacy misaligned segment preserves its payload
/// exactly** — the operation that makes an already-encoded container
/// bindable without re-encoding it.
///
/// Built by writing a conforming segment and then re-writing it with a
/// header one byte SHORTER, which is precisely the shape a container
/// encoded before the padding existed has: a valid tensor table over a
/// payload that begins at an odd byte.
#[test]
fn realigning_a_misaligned_segment_preserves_the_payload_byte_for_byte() {
    use crate::format::vindex3::encode::segment::{
        read_segment_header, realign_segment, write_segment, PlannedTensor, SEGMENT_PAYLOAD_ALIGN,
    };
    use std::io::{Read, Write};

    let dir = std::env::temp_dir().join(format!("larql-realign-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");
    let aligned = dir.join("aligned.bin");
    let payload: Vec<u8> = (0..96u16).map(|i| (i % 251) as u8).collect();
    write_segment(
        &aligned,
        "test@BF16",
        vec![PlannedTensor {
            relative_name: "w".into(),
            source_name: "w".into(),
            dtype: "BF16".into(),
            shape: vec![48],
            len: 96,
        }],
        |_n, w, hash| {
            w.write_all(&payload).expect("payload");
            hash(&payload);
            Ok(96)
        },
    )
    .expect("write");

    // Re-cut it with a one-byte-shorter header: same table, payload now
    // at an odd offset. This is a container encoded before the encoder
    // padded anything.
    let mut raw = Vec::new();
    std::fs::File::open(&aligned)
        .expect("open")
        .read_to_end(&mut raw)
        .expect("read");
    let header_len = u64::from_le_bytes(raw[..8].try_into().expect("len")) as usize;
    let header = &raw[8..8 + header_len];
    assert_eq!(
        header[header_len - 1],
        b' ',
        "the padding is trailing space"
    );
    let legacy = dir.join("legacy.bin");
    {
        let mut f = std::fs::File::create(&legacy).expect("create");
        f.write_all(&((header_len - 1) as u64).to_le_bytes())
            .expect("len");
        f.write_all(&header[..header_len - 1]).expect("header");
        f.write_all(&raw[8 + header_len..]).expect("payload");
    }
    let (_, legacy_start) = read_segment_header(&legacy).expect("legacy header parses");
    assert!(
        !legacy_start.is_multiple_of(SEGMENT_PAYLOAD_ALIGN as u64),
        "the fixture must actually be misaligned (starts at {legacy_start})"
    );

    let fixed = dir.join("fixed.bin");
    let (before, after) = realign_segment(&legacy, &fixed).expect("realign");
    assert_eq!(before, legacy_start);
    assert!(after.is_multiple_of(SEGMENT_PAYLOAD_ALIGN as u64));

    // The tensor table survives, and the payload is byte-identical.
    let (h_before, _) = read_segment_header(&legacy).expect("before");
    let (h_after, start_after) = read_segment_header(&fixed).expect("after");
    assert_eq!(h_before.tensors, h_after.tensors, "the table is unchanged");
    assert_eq!(h_before.representation, h_after.representation);
    assert_eq!(start_after, after);
    let mut got = Vec::new();
    let mut f = std::fs::File::open(&fixed).expect("open fixed");
    std::io::Read::read_to_end(&mut f, &mut got).expect("read fixed");
    assert_eq!(
        &got[start_after as usize..],
        &payload[..],
        "the payload bytes must be copied verbatim"
    );
    // Realigning an ALREADY-aligned segment to a new path still copies.
    let again = dir.join("again.bin");
    let (b2, a2) = realign_segment(&fixed, &again).expect("realign aligned");
    assert_eq!((b2, a2), (after, after));
    std::fs::remove_dir_all(&dir).ok();
}

/// A truncated or absurd header is refused by name rather than read.
#[test]
fn a_corrupt_segment_header_is_refused() {
    use crate::format::vindex3::encode::segment::read_segment_header;
    use std::io::Write;

    let dir = std::env::temp_dir().join(format!("larql-corrupt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp");

    let absurd = dir.join("absurd.bin");
    std::fs::File::create(&absurd)
        .expect("create")
        .write_all(&u64::MAX.to_le_bytes())
        .expect("len");
    let err = read_segment_header(&absurd).expect_err("an absurd header length must be refused");
    assert!(format!("{err}").contains("corrupt"), "{err}");

    let garbage = dir.join("garbage.bin");
    {
        let body = b"not json at all";
        let mut f = std::fs::File::create(&garbage).expect("create");
        f.write_all(&(body.len() as u64).to_le_bytes())
            .expect("len");
        f.write_all(body).expect("body");
    }
    assert!(
        read_segment_header(&garbage).is_err(),
        "a non-JSON header must be refused"
    );
    std::fs::remove_dir_all(&dir).ok();
}
