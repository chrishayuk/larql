//! A container laid out like a Hugging Face cache snapshot, where every segment is a RELATIVE symlink into a sibling
//! blob store, compiles into a self-contained container: no segment of the output is a link that only resolved from
//! the source's directory.

#![cfg(unix)]

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::weight_patch::{apply_weight_patch, WeightEdit, WeightPatch};

use crate::fixture::*;

/// Move each segment into `<root>/../blobs/<name>` and leave a relative symlink in its place, as the HF hub does.
fn as_hf_snapshot(container: &std::path::Path) {
    let segments = container.join("segments");
    let blobs = container.parent().unwrap().join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    for entry in std::fs::read_dir(&segments).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_owned();
        std::fs::rename(&path, blobs.join(&name)).unwrap();
        std::os::unix::fs::symlink(std::path::Path::new("../../blobs").join(&name), &path).unwrap();
    }
}

#[test]
fn a_compile_from_relative_symlinked_segments_writes_real_files_that_read_back() {
    let src = source_with_pack();
    let before: Vec<(String, Vec<u8>)> = std::fs::read_dir(src.container.join("segments"))
        .unwrap()
        .map(|e| {
            let p = e.unwrap().path();
            (
                p.file_name().unwrap().to_string_lossy().to_string(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect();
    as_hf_snapshot(&src.container);

    let out = src.out("from-snapshot.vindex3");
    let patch = WeightPatch {
        tensor: TENSOR.to_string(),
        edits: vec![WeightEdit {
            row: 1,
            col: 2,
            delta: 0.5,
        }],
    };
    apply_weight_patch(&src.container, &patch, &out).unwrap();

    for entry in std::fs::read_dir(out.join("segments")).unwrap() {
        let path = entry.unwrap().path();
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert!(
            !meta.file_type().is_symlink(),
            "{} is a link in the compiled container",
            path.display()
        );
    }
    // The edited canonical segment is rewritten and the representations derived from it are re-derived; every
    // other segment must carry the source's bytes.
    let index = index_of(&out);
    let changed: Vec<String> = index
        .representations
        .iter()
        .filter(|(id, e)| id.as_str() == CANONICAL || e.compiled_from.as_deref() == Some(CANONICAL))
        .map(|(_, e)| e.segment.clone())
        .collect();
    for (name, bytes) in before {
        let rel = format!("segments/{name}");
        if changed.contains(&rel) || !out.join(&rel).exists() {
            continue;
        }
        assert!(
            std::fs::read(out.join(&rel)).unwrap() == bytes,
            "{rel} must carry the source's bytes"
        );
    }
    assert!(inspect_container(&out, true).unwrap().is_coherent());
}
