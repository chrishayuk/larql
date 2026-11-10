//! Reading an untrusted checkpoint

use super::*;

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
    encode_system_unenforced(&named, out.path()).unwrap();

    let index_path = out.path().join(crate::format::filenames::INDEX_JSON);
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    index["system_graph"] = serde_json::Value::Null;
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    // A refusal, not an empty inspection: reporting it as "no defects"
    // would say the container inspected clean when nothing was inspected
    // at all. With the graph gone and no routed-programme manifest, the
    // index names no container shape; a legacy bank (manifest, no graph)
    // is refused by name instead, in `shape_tests`.
    let err = match inspect_container(out.path(), false) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("inspected a container with no graph to reconstruct"),
    };
    assert!(err.contains("neither a system graph"), "{err}");
    assert!(err.contains("§5.5"), "{err}");
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

#[test]
fn a_config_key_naming_a_component_outside_the_container_writes_nothing_outside() {
    // `read_components` names a nested component after its `*_config`
    // key, and the component id becomes a segment filename. The key is
    // the checkpoint author's: `../../vision_config` must not make the
    // encoder write `out/segments/../../vision.*.bin`.
    use crate::format::vindex3::plan::tests_support::gemma3_shaped_target_with;

    let checkpoint = tempfile::tempdir().unwrap();
    let inventory = gemma3_shaped_target_with(
        checkpoint.path(),
        |config| {
            let object = config.as_object_mut().unwrap();
            let mut vision = object
                .remove("vision_config")
                .expect("the fixture has a tower");
            // A complete surface, so nothing but the name stands between the
            // encoder and the write.
            vision["hidden_act"] = "gelu_pytorch_tanh".into();
            vision["layer_norm_eps"] = 1e-6.into();
            object.insert("../../vision_config".to_string(), vision);
        },
        |_| {},
    );
    let sandbox = tempfile::tempdir().unwrap();
    let out = sandbox.path().join("a").join("container");

    let result = encode_system_unenforced(&[("hostile".to_string(), inventory)], &out);

    let escaped: Vec<_> = walk(sandbox.path())
        .into_iter()
        .filter(|p| !p.starts_with(&out))
        .collect();
    assert!(
        escaped.is_empty(),
        "written outside the container: {escaped:?} ({result:?})"
    );
    // Before the write gate this encoded "successfully" — and the
    // container it left could not be reopened.
    let err = result.expect_err("a container naming files outside itself must not be produced");
    assert!(
        err.to_string().contains("which is refused"),
        "refused for the wrong reason: {err}"
    );
}

fn walk(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}
