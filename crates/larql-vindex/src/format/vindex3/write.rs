//! Assemble a VINDEX3 container directory.
//!
//! This is the piece whose absence meant "VINDEX3 works" could only ever be
//! claimed about the *runtime*: every parity result to date bound VINDEX3
//! routes over operands sourced from a VINDEX2 file, because nothing could
//! produce a VINDEX3 file to source them from. `ContainerGeneration::V3`
//! existed only as a hand-written JSON string in a detection test.
//!
//! What a container is, physically:
//!
//! ```text
//! <root>/
//! ├── index.json           schema 4 (V3 reads the 3–4 span) — the root authority (§12)
//! ├── moe_manifest.json    which programme interprets the banks (§8)
//! └── <segment key>.lyrw   LYRW v2 bank files (§6)
//! ```
//!
//! The writer takes banks already planned and written by [`Lyrw2Writer`], and
//! its job is only to place them and declare them. It deliberately does not
//! *quantise* or *transcode* anything: a container assembler that also
//! converted formats would be the silent conversion §9.1 forbids, one layer
//! further down than anyone would think to look for it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::index::Vindex3Index;
use crate::format::filenames::INDEX_JSON;
use crate::format::moe_manifest::MoeManifest;
use crate::VindexError;

/// Filename of the MoE programme manifest within a container.
pub const MOE_MANIFEST_JSON: &str = "moe_manifest.json";
/// Extension for a LYRW v2 segment file.
pub const SEGMENT_EXT: &str = "lyrw";

/// One bank's bytes, already in LYRW v2 form, plus where it belongs.
pub struct SegmentSource {
    /// Path stem relative to the container root, e.g. `routed/layer_000`.
    /// Becomes `<root>/<key>.lyrw`.
    pub key: String,
    /// A complete LYRW v2 file as produced by `Lyrw2Writer`.
    pub bytes: Vec<u8>,
}

/// Everything a container needs before any byte is placed.
pub struct ContainerSpec {
    pub model: String,
    pub family: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub manifest: MoeManifest,
    pub segments: Vec<SegmentSource>,
}

/// Path a segment key resolves to. Composed, never globbed.
pub fn segment_path(root: &Path, key: &str) -> PathBuf {
    root.join(format!("{key}.{SEGMENT_EXT}"))
}

/// Write a complete VINDEX3 container to `root`.
///
/// Ordering is deliberate: segments, then the manifest, then `index.json`
/// last. `index.json` is the discriminator every reader dispatches on, so a
/// crash midway leaves a directory that is *not yet* a VINDEX3 container
/// rather than one that claims to be and is missing its banks.
pub fn write_container(root: &Path, spec: &ContainerSpec) -> Result<(), VindexError> {
    if spec.segments.is_empty() {
        return Err(VindexError::Parse(
            "a VINDEX3 container needs at least one segment; an index declaring \
             none cannot be bound and would fail at execution instead of here"
                .into(),
        ));
    }
    std::fs::create_dir_all(root).map_err(VindexError::Io)?;

    let mut declared: BTreeMap<String, u32> = BTreeMap::new();
    for segment in &spec.segments {
        let path = segment_path(root, &segment.key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(VindexError::Io)?;
        }
        std::fs::write(&path, &segment.bytes).map_err(VindexError::Io)?;
        *declared.entry(segment.key.clone()).or_insert(0) += 1;
    }

    let manifest_json = serde_json::to_string_pretty(&spec.manifest)
        .map_err(|e| VindexError::Parse(format!("serialise moe_manifest: {e}")))?;
    std::fs::write(root.join(MOE_MANIFEST_JSON), manifest_json).map_err(VindexError::Io)?;

    let index = Vindex3Index::new(
        spec.model.clone(),
        spec.family.clone(),
        spec.hidden_size,
        spec.num_layers,
        MOE_MANIFEST_JSON,
        declared,
    );
    let index_json = serde_json::to_string_pretty(&index)
        .map_err(|e| VindexError::Parse(format!("serialise index.json: {e}")))?;
    std::fs::write(root.join(INDEX_JSON), index_json).map_err(VindexError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::generation::{detect_generation, ContainerGeneration};
    use crate::format::vindex3::test_support::{fixture_a_spec, FIXTURE_A_SEGMENT_KEY};
    use tempfile::tempdir;

    #[test]
    fn a_written_container_is_detected_as_vindex3_from_disk() {
        // The claim the whole exercise exists to support: not a JSON literal
        // in a test asserting version 3, but a directory this code produced,
        // read back by the same detector production uses.
        let dir = tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write container");
        assert_eq!(
            detect_generation(dir.path()).unwrap(),
            ContainerGeneration::V3
        );
    }

    #[test]
    fn a_written_container_has_all_three_physical_parts() {
        let dir = tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write container");
        assert!(dir.path().join(INDEX_JSON).is_file());
        assert!(dir.path().join(MOE_MANIFEST_JSON).is_file());
        assert!(segment_path(dir.path(), FIXTURE_A_SEGMENT_KEY).is_file());
    }

    #[test]
    fn the_index_declares_every_segment_that_was_written() {
        // A segment on disk that the index does not declare is unreachable —
        // the loader composes paths from the index and never scans.
        let dir = tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write container");
        let raw = std::fs::read_to_string(dir.path().join(INDEX_JSON)).unwrap();
        let index: Vindex3Index = serde_json::from_str(&raw).unwrap();
        assert_eq!(index.segments.get(FIXTURE_A_SEGMENT_KEY), Some(&1));
    }

    #[test]
    fn a_container_with_no_segments_is_refused_at_the_api() {
        // Refused here, where the message can say why, rather than at bind
        // time where it would surface as a missing operand.
        let dir = tempdir().unwrap();
        let mut spec = fixture_a_spec();
        spec.segments.clear();
        let err = write_container(dir.path(), &spec).unwrap_err();
        assert!(
            format!("{err}").contains("at least one segment"),
            "unexpected: {err}"
        );
        assert!(
            !dir.path().join(INDEX_JSON).exists(),
            "a refused write must not leave a directory claiming to be a container"
        );
    }
}
