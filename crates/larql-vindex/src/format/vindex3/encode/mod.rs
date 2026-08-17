//! `larql vindex3 encode` — materialise the system graph (V3-G3).
//!
//! Deliberately boring, deterministic, source-independent output: one
//! logical object → one canonical representation → one contiguous segment.
//! No layout optimisation — optimised layouts arrive later as *additional*
//! representations over the same logical objects.
//!
//! The encoder consumes **the built graph** via the plan pipeline and
//! refuses an inadmissible plan outright. It never re-interprets the
//! checkpoint: interpretation happened once, in
//! [`graph::build_from_inventories`](super::graph::build_from_inventories),
//! and a second private reading here would reintroduce the two-authority
//! problem the graph exists to eliminate.
//!
//! Container layout (spec §6.1):
//!
//! ```text
//! <out>/
//! ├── index.json           root authority: version, system_graph ref,
//! │                        representation directory (id → segment + hashes)
//! ├── system_graph.json    the SystemGraph, verbatim
//! └── segments/<object>.bin  self-describing framed segments
//! ```

pub mod segment;
pub mod source;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_models::inventory::ArchitectureInventory;

use super::graph::{most_specific_owner, ComponentRole, LogicalObject, SystemGraph};
use super::index::{RepresentationEntry, Vindex3Index};
use super::plan::plan_system;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use segment::PlannedTensor;
use source::ArtifactSource;

/// Filename of the system-graph manifest within a container.
pub const SYSTEM_GRAPH_JSON: &str = "system_graph.json";
/// Directory holding representation segments.
pub const SEGMENTS_DIR: &str = "segments";
/// Extension for a framed representation segment.
pub const SEGMENT_BIN_EXT: &str = "bin";
/// Separator between object id and encoding in a representation id.
pub const REPRESENTATION_ID_SEP: char = '@';

/// Outcome summary of one encode.
#[derive(Debug)]
pub struct EncodeOutcome {
    pub container: PathBuf,
    pub representations: usize,
    pub total_payload_bytes: u64,
}

/// Encode a system of artifacts into a self-contained container.
///
/// Runs the plan first and refuses to encode anything inadmissible — the
/// gate is the same one `larql vindex3 plan` exits non-zero on.
pub fn encode_system(
    named: &[(String, ArchitectureInventory)],
    out: &Path,
) -> Result<EncodeOutcome, VindexError> {
    let plan = plan_system(named);
    if !plan.admissible {
        return Err(VindexError::Parse(format!(
            "refusing to encode an inadmissible plan: {} blocking finding(s); \
             run `larql vindex3 plan` for the itemised reasons",
            plan.summary.blocking
        )));
    }
    encode_graph(&plan.graph, named, out)
}

/// Encode an already-built system graph. `encode_system` is this after
/// the plan gate; a caller holding a graph it built (or edited) itself
/// comes in here and gets the same validation and the same bytes.
pub fn encode_graph(
    graph: &SystemGraph,
    named: &[(String, ArchitectureInventory)],
    out: &Path,
) -> Result<EncodeOutcome, VindexError> {
    let defects = graph.validate();
    if !defects.is_empty() {
        return Err(VindexError::Parse(format!(
            "system graph failed validation before encode: {defects:?}"
        )));
    }

    // Sources by artifact name, opened once.
    let mut sources: BTreeMap<&str, ArtifactSource> = BTreeMap::new();
    for (name, inventory) in named {
        sources.insert(
            name.as_str(),
            ArtifactSource::open(Path::new(&inventory.path))?,
        );
    }
    let inventories: BTreeMap<&str, &ArchitectureInventory> =
        named.iter().map(|(n, i)| (n.as_str(), i)).collect();

    std::fs::create_dir_all(out)?;
    let mut directory: BTreeMap<String, RepresentationEntry> = BTreeMap::new();
    let mut segment_keys: BTreeMap<String, u32> = BTreeMap::new();
    let mut total_payload_bytes = 0u64;

    for object in &graph.objects {
        let Some(representation) = object.representations.first() else {
            return Err(VindexError::Parse(format!(
                "object `{}` carries no representation — nothing to encode",
                object.id
            )));
        };
        let representation_id = format!(
            "{}{REPRESENTATION_ID_SEP}{}",
            object.id, representation.encoding
        );
        let segment_key = format!("{SEGMENTS_DIR}/{}", object.id);
        let segment_rel = format!("{segment_key}.{SEGMENT_BIN_EXT}");
        let planned = plan_object_tensors(object, &inventories, &graph.objects)?;

        let written = segment::write_segment(
            &out.join(&segment_rel),
            &representation_id,
            planned,
            |source_name, write, observe| {
                // Bindings within one object may span artifacts; resolve the
                // owning source per tensor.
                let owner = binding_owner(object, source_name).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "tensor `{source_name}` matches no binding of `{}`",
                        object.id
                    ))
                })?;
                sources[owner].stream_payload(source_name, write, observe)
            },
        )?;

        total_payload_bytes += written.payload_bytes;
        segment_keys.insert(segment_key, 1);
        directory.insert(
            representation_id,
            RepresentationEntry {
                object: object.id.clone(),
                encoding: representation.encoding.clone(),
                segment: segment_rel,
                tensor_count: written.tensor_count,
                payload_bytes: written.payload_bytes,
                payload_sha256: written.payload_sha256,
                segment_sha256: written.segment_sha256,
            },
        );
    }

    // Graph manifest, then the index last — a crash midway leaves a
    // directory that is not yet a container rather than one that lies.
    let graph_json = serde_json::to_string_pretty(graph)
        .map_err(|e| VindexError::Parse(format!("serialise system graph: {e}")))?;
    std::fs::write(out.join(SYSTEM_GRAPH_JSON), graph_json)?;

    let index = system_index(graph, named, directory, segment_keys)?;
    let index_json = serde_json::to_string_pretty(&index)
        .map_err(|e| VindexError::Parse(format!("serialise index.json: {e}")))?;
    std::fs::write(out.join(INDEX_JSON), index_json)?;

    Ok(EncodeOutcome {
        container: out.to_path_buf(),
        representations: index.representations.len(),
        total_payload_bytes,
    })
}

/// The artifact owning `source_name` under one of `object`'s bindings —
/// the single definition of binding membership, shared with the G4
/// re-hash so the two can never drift.
pub(crate) fn binding_owner<'a>(object: &'a LogicalObject, source_name: &str) -> Option<&'a str> {
    object
        .source_bindings
        .iter()
        .find(|b| b.covers(source_name))
        .map(|b| b.artifact.as_str())
}

/// Plan the tensor table for one object: every source tensor under its
/// bindings **that no other object owns more specifically** (see
/// [`most_specific_owner`]), named relative to the bindings' common
/// segment-prefix so no artifact-global name survives into the container.
pub(crate) fn plan_object_tensors(
    object: &LogicalObject,
    inventories: &BTreeMap<&str, &ArchitectureInventory>,
    all_objects: &[LogicalObject],
) -> Result<Vec<PlannedTensor>, VindexError> {
    let strip = common_segment_prefix(
        &object
            .source_bindings
            .iter()
            .map(|b| b.tensor_prefix.as_str())
            .collect::<Vec<_>>(),
    );
    let mut planned = Vec::new();
    for binding in &object.source_bindings {
        let inventory = inventories.get(binding.artifact.as_str()).ok_or_else(|| {
            VindexError::Parse(format!(
                "object `{}` binds artifact `{}` which is not in the encode set",
                object.id, binding.artifact
            ))
        })?;
        for tensor in inventory.tensors.tensors.iter().filter(|t| {
            binding.covers(&t.name)
                && most_specific_owner(all_objects, &t.name).is_none_or(|o| o.id == object.id)
        }) {
            let relative_name = tensor
                .name
                .strip_prefix(&strip)
                .unwrap_or(&tensor.name)
                .trim_start_matches('.')
                .to_string();
            planned.push(PlannedTensor {
                relative_name,
                source_name: tensor.name.clone(),
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                len: tensor.bytes,
            });
        }
    }
    if planned.is_empty() {
        return Err(VindexError::Parse(format!(
            "object `{}` matched no source tensors — bindings and inventory disagree",
            object.id
        )));
    }
    Ok(planned)
}

/// Longest common dot-segment prefix of the binding prefixes, as a string
/// (empty when nothing is shared).
fn common_segment_prefix(prefixes: &[&str]) -> String {
    let Some(first) = prefixes.first() else {
        return String::new();
    };
    let mut common: Vec<&str> = first.split('.').collect();
    for prefix in &prefixes[1..] {
        let segments: Vec<&str> = prefix.split('.').collect();
        let shared = common
            .iter()
            .zip(&segments)
            .take_while(|(a, b)| a == b)
            .count();
        common.truncate(shared);
    }
    common.join(".")
}

/// The container's root index: identity from the primary component,
/// the graph reference, and the representation directory.
fn system_index(
    graph: &SystemGraph,
    named: &[(String, ArchitectureInventory)],
    representations: BTreeMap<String, RepresentationEntry>,
    segments: BTreeMap<String, u32>,
) -> Result<Vindex3Index, VindexError> {
    let primary = graph
        .components
        .iter()
        .find(|c| c.role == ComponentRole::PrimaryText)
        .or_else(|| graph.components.first())
        .ok_or_else(|| VindexError::Parse("graph has no components".into()))?;
    let family = named
        .iter()
        .find(|(name, _)| *name == primary.source_artifact)
        .map(|(_, inventory)| inventory.identity.model_type.clone())
        .unwrap_or_default();
    let mut index = Vindex3Index::new(
        primary.source_artifact.clone(),
        family,
        primary.hidden_size,
        primary.num_layers,
        String::new(),
        segments,
    );
    // A system container has no routed programme; `new` is the MoE
    // constructor, so unset its manifest explicitly.
    index.moe_manifest = None;
    index.system_graph = Some(SYSTEM_GRAPH_JSON.to_string());
    index.representations = representations;
    Ok(index)
}
