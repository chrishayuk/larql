//! Stream a many-layer VINDEX3 container to disk (container ladder c9).
//!
//! # Why c9 is not just c8 in a loop
//!
//! [`super::write::write_container`] takes a [`ContainerSpec`] whose segments
//! own their bytes. That is the right shape for one fixture-sized bank and the
//! wrong shape for a model: layer 0 of `gemma4-26b-a4b` is ~421 MB, so
//! describing thirty of them the c8 way would hold ~12 GB of already-mapped
//! weights a second time, in RAM, purely to hand them to a writer that puts
//! them straight back on disk.
//!
//! This builder inverts that. Each layer is written **directly at its final
//! path** as it arrives, and only the manifest entries — a few hundred bytes
//! per layer — accumulate:
//!
//! ```text
//! c8   describe all layers -> hold all bytes -> write everything
//! c9   describe one layer  -> write it       -> forget it -> next
//! ```
//!
//! Peak memory is therefore one expert's slice, whatever the model's size.
//!
//! # The write order is the crash contract
//!
//! Segments, then the manifest, then `index.json` **last** — the same ordering
//! [`super::write::write_container`] states, and for the same reason.
//! `index.json` is the discriminator every reader dispatches on, so a crash
//! part-way through a thirty-layer import must leave a directory that is *not
//! yet* a VINDEX3 container, rather than one that announces itself as VINDEX3
//! and is missing two thirds of its banks. That distinction matters more here
//! than at c8, because a multi-gigabyte import is long enough to actually be
//! interrupted.
//!
//! # Still verbatim
//!
//! Nothing here transcodes, requantises or repacks. The builder places bytes
//! and declares them, exactly as §9.1 requires of the one-layer path — a
//! streaming assembler that also converted formats would hide the silent
//! conversion one level below where c8 already forbids it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::import::{manifest_layer, routed_storage_key, write_segment_file, MoeLayerSource};
use super::index::Vindex3Index;
use super::write::{segment_file_name, MOE_MANIFEST_JSON};
use crate::format::filenames::INDEX_JSON;
use crate::format::moe_manifest::layer::MoeLayer;
use crate::format::moe_manifest::MoeManifest;
use crate::VindexError;

/// Every segment key a container declares appears exactly once. A repeat means
/// the caller would have silently overwritten a bank it had already written.
const SEGMENT_DECLARED_ONCE: u32 = 1;

/// Assembles a VINDEX3 container one layer at a time.
///
/// Construct with [`ContainerBuilder::create`], feed layers with
/// [`ContainerBuilder::add_moe_layer`], and close with
/// [`ContainerBuilder::finish`]. Dropping the builder without finishing leaves
/// the segments on disk but writes no `index.json`, so the directory is not a
/// container and no reader will treat it as one.
pub struct ContainerBuilder {
    root: PathBuf,
    layers: Vec<MoeLayer>,
    declared: BTreeMap<String, u32>,
    bytes_written: u64,
}

impl ContainerBuilder {
    /// Prepare `root` to receive segments. Creates the directory if absent.
    pub fn create(root: &Path) -> Result<Self, VindexError> {
        std::fs::create_dir_all(root).map_err(VindexError::Io)?;
        Ok(Self {
            root: root.to_path_buf(),
            layers: Vec::new(),
            declared: BTreeMap::new(),
            bytes_written: 0,
        })
    }

    /// Write one routed MoE layer's bank and record how to interpret it.
    ///
    /// Returns the segment's size on disk. Refuses a layer whose storage key
    /// this container already declares: importing it would overwrite the
    /// earlier bank while leaving both manifest entries in place, which
    /// produces a container that verifies clean and executes the wrong
    /// weights — the one failure mode worth spending a check on.
    pub fn add_moe_layer(&mut self, source: &MoeLayerSource<'_>) -> Result<u64, VindexError> {
        let storage = routed_storage_key(source.layer);
        if self.declared.contains_key(&storage) {
            return Err(VindexError::Parse(format!(
                "layer {} is already in this container as `{storage}` — importing \
                 it twice would overwrite the first bank and leave two manifest \
                 entries pointing at the survivor",
                source.layer
            )));
        }

        let path = super::container_path::prepare_write(&self.root, &segment_file_name(&storage))?;
        write_segment_file(source, &path)?;
        let size = std::fs::metadata(&path).map_err(VindexError::Io)?.len();

        self.layers.push(manifest_layer(source, &storage));
        self.declared.insert(storage, SEGMENT_DECLARED_ONCE);
        self.bytes_written += size;
        Ok(size)
    }

    /// How many layers have been written so far.
    pub fn layers_added(&self) -> usize {
        self.layers.len()
    }

    /// Total segment bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Write the manifest and then `index.json`, making `root` a container.
    ///
    /// `num_layers` is the **model's** layer count, not the number imported:
    /// a container holding only the MoE layers of a hybrid model still has to
    /// describe the model it came from, or a reader cannot tell a partial
    /// import from a shallower architecture.
    pub fn finish(
        self,
        model: impl Into<String>,
        family: impl Into<String>,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<(), VindexError> {
        if self.layers.is_empty() {
            return Err(VindexError::Parse(
                "a VINDEX3 container needs at least one segment; an index \
                 declaring none cannot be bound and would fail at execution \
                 instead of here"
                    .into(),
            ));
        }

        // Validate before publishing: a manifest this builder produced and
        // cannot itself parse would push the failure into the container it
        // just spent minutes writing, discovered only by whoever opens it.
        let manifest = MoeManifest::new(self.layers);
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| VindexError::Parse(format!("serialise moe_manifest: {e}")))?;
        MoeManifest::parse(&manifest_json)?;
        std::fs::write(self.root.join(MOE_MANIFEST_JSON), &manifest_json)
            .map_err(VindexError::Io)?;

        let index = Vindex3Index::new(
            model,
            family,
            hidden_size,
            num_layers,
            MOE_MANIFEST_JSON,
            self.declared,
        );
        let index_json = serde_json::to_string_pretty(&index)
            .map_err(|e| VindexError::Parse(format!("serialise index.json: {e}")))?;
        std::fs::write(self.root.join(INDEX_JSON), index_json).map_err(VindexError::Io)
    }
}

#[cfg(test)]
#[path = "build_tests.rs"]
mod tests;
