//! The server's plain completions feed the same raw-prompt ids as every
//! other VINDEX3 surface.
//!
//! [`V3Model::encode_raw_prompt`] is what `/v1/completions` calls on a V3
//! binding. For each BOS shape in the shared table
//! ([`larql_inference::test_utils::BosShape`]) — Granite/Qwen3 (declared,
//! not added), Gemma 4 (declared, required, not added by the tokenizer),
//! and a tokenizer that adds it — the bound model's ids must equal the
//! table's, which `larql run --raw`, `vindex3 observe` and LQL are held to
//! in their own crates. Before the shared encoder the server fed the bare
//! encoding, and the Gemma 4 shape failed here.

use std::path::{Path, PathBuf};

use larql_inference::test_utils::{BosShape, RAW_PROMPT};
use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_vindex::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, DENSE_VOCAB,
};

fn container_for(root: &Path, shape: BosShape) -> PathBuf {
    let name = format!("{shape:?}").to_lowercase();
    let checkpoint = root.join(format!("{name}-checkpoint"));
    let container = root.join(&name);
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, &name);
    shape.write_into(&container, DENSE_VOCAB);
    container
}

#[test]
fn plain_completions_encode_every_bos_shape_like_every_surface() {
    let root = tempfile::tempdir().unwrap();
    for shape in BosShape::ALL {
        let container = container_for(root.path(), shape);
        let model = match load_artifact(&container.to_string_lossy(), LoadVindexOptions::default())
            .unwrap()
        {
            LoadedArtifact::V3(m) => m,
            LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
        };
        let ids = model.encode_raw_prompt(RAW_PROMPT).unwrap();
        assert_eq!(ids, shape.expected_ids(), "{shape:?}");
    }
}

#[test]
fn an_empty_plain_prompt_is_a_bad_request() {
    let root = tempfile::tempdir().unwrap();
    let container = container_for(root.path(), BosShape::DeclaredRequired);
    let LoadedArtifact::V3(model) =
        load_artifact(&container.to_string_lossy(), LoadVindexOptions::default()).unwrap()
    else {
        panic!("a VINDEX3 container must bind as V3");
    };
    let err = model.encode_raw_prompt("").unwrap_err().to_string();
    assert!(err.contains("prompt tokenises to empty"), "{err}");
}
