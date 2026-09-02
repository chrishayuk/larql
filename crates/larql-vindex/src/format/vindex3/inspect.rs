//! `larql vindex3 inspect` — reconstruct a system from its container alone.
//!
//! This is the G3 gate: everything reported here comes from `index.json`,
//! `system_graph.json` and segment headers. No transformers config, no HF
//! filenames, no safetensors headers, no architecture registry — once
//! encoded, the source checkpoint is not an authority, and this module
//! proves it by never touching one.

use std::path::Path;

use larql_models::config::PositionPolicy;
use serde::Serialize;

use super::encode::segment::read_segment_header;
use super::graph::policy::AttentionSpan;
use super::graph::{SystemGraph, GRAPH_SCHEMA};
use super::index::Vindex3Index;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::generation::{detect_generation, ContainerGeneration};

/// A structural problem found while inspecting a container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum InspectionDefect {
    /// Graph and directory disagree about what exists.
    DirectoryIncoherent(String),
    /// A segment file disagrees with its directory entry.
    SegmentIncoherent(String),
    /// Payload hash did not match the directory (verify mode only).
    PayloadCorrupt(String),
}

/// One component, as reconstructed purely from the container.
#[derive(Debug, Clone, Serialize)]
pub struct ComponentSummary {
    pub id: String,
    pub role: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    /// Per-operator layer counts when the policy table exists. Softmax
    /// spans and recurrences are counted apart: a hybrid stack whose
    /// recurrences were folded into `full_layers` reports a tower it does
    /// not have.
    pub sliding_layers: Option<usize>,
    pub full_layers: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recurrent_layers: Option<usize>,
    /// Layers with no positional encoding.
    pub nope_layers: Option<usize>,
    pub window: Option<usize>,
}

/// The full reconstruction.
#[derive(Debug, Clone, Serialize)]
pub struct SystemInspection {
    pub components: Vec<ComponentSummary>,
    pub graph: SystemGraph,
    pub index: Vindex3Index,
    pub defects: Vec<InspectionDefect>,
}

impl SystemInspection {
    pub fn is_coherent(&self) -> bool {
        self.defects.is_empty()
    }

    /// Execution completeness of the persisted graph (the V3-G5a gate):
    /// every component with executable objects carries the surface those
    /// operations read — answered from the container alone.
    pub fn execution_completeness(&self) -> Vec<super::graph::CompletenessDefect> {
        super::graph::execution_completeness(&self.graph)
    }
}

/// Inspect a container. `verify_payloads` additionally re-hashes every
/// segment file and compares against the directory — slower, but detects
/// byte corruption with no source access.
/// Refuse a graph this build cannot read, naming the versions and the
/// remedy.
///
/// Deliberately separate from deserialising the graph: the schema must be
/// legible even when the rest of the document is not, which is the whole
/// situation a version field exists for.
fn check_graph_schema(graph_text: &str, graph_name: &str) -> Result<(), VindexError> {
    let probe: serde_json::Value = serde_json::from_str(graph_text)
        .map_err(|e| VindexError::Parse(format!("parse {graph_name}: {e}")))?;
    let found = probe
        .get("schema")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| VindexError::Parse(format!("{graph_name} records no schema version")))?;
    if found == u64::from(GRAPH_SCHEMA) {
        return Ok(());
    }
    Err(VindexError::Parse(format!(
        "{graph_name} is schema v{found}, this build reads v{GRAPH_SCHEMA} — re-encode the \
         container. Older graphs cannot be upgraded in place: each schema step added a judged \
         semantic fact whose absence is indistinguishable from a deliberate \"this model has \
         no such operation\", and inventing either answer is the guess the version exists to \
         prevent."
    )))
}

pub fn inspect_container(
    root: &Path,
    verify_payloads: bool,
) -> Result<SystemInspection, VindexError> {
    // The generation before any other byte. `index.json`'s `version` is
    // the sole schema discriminator, and this is the opening step for
    // every graph-path consumer (`vindex3 exec`, the runtime opener,
    // represent, compile, the download validator), so an old VINDEX2
    // directory, a future schema this build does not read, or an index
    // with no version at all must be refused here by name — not parsed
    // as if it were current and reported as a defect in something else.
    match detect_generation(root)? {
        ContainerGeneration::V3 => {}
        ContainerGeneration::V2 => {
            return Err(VindexError::WrongContainerGeneration {
                found: "VINDEX2",
                required: "VINDEX3",
            })
        }
    }
    let index: Vindex3Index = serde_json::from_str(
        &std::fs::read_to_string(root.join(INDEX_JSON))
            .map_err(|e| VindexError::Parse(format!("read {INDEX_JSON}: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    let graph_name = index.system_graph.as_deref().ok_or_else(|| {
        VindexError::Parse(
            "container records no system graph — nothing to reconstruct \
             (a routed-MoE container opens via `larql show`, not here)"
                .into(),
        )
    })?;
    let graph_text = std::fs::read_to_string(root.join(graph_name))
        .map_err(|e| VindexError::Parse(format!("read {graph_name}: {e}")))?;
    // Read the schema before the graph. A version field nothing checks is
    // not a version field: without this, an older container fails as a
    // parse error pointing at whichever line happens to have changed
    // shape, which says nothing about what to do next.
    check_graph_schema(&graph_text, graph_name)?;
    let graph: SystemGraph = serde_json::from_str(&graph_text)
        .map_err(|e| VindexError::Parse(format!("parse {graph_name}: {e}")))?;

    let mut defects: Vec<InspectionDefect> = graph
        .validate()
        .into_iter()
        .map(|d| InspectionDefect::DirectoryIncoherent(format!("graph defect: {d:?}")))
        .collect();

    // Directory ↔ graph coherence: every object materialised, every entry
    // grounded in an object.
    for object in &graph.objects {
        if !index
            .representations
            .values()
            .any(|e| e.object == object.id)
        {
            defects.push(InspectionDefect::DirectoryIncoherent(format!(
                "object `{}` has no representation in the directory",
                object.id
            )));
        }
    }
    for (id, entry) in &index.representations {
        if !graph.objects.iter().any(|o| o.id == entry.object) {
            defects.push(InspectionDefect::DirectoryIncoherent(format!(
                "directory entry `{id}` references unknown object `{}`",
                entry.object
            )));
        }
        // Segment header agreement.
        match read_segment_header(&root.join(&entry.segment)) {
            Ok((header, payload_start)) => {
                if header.representation != *id {
                    defects.push(InspectionDefect::SegmentIncoherent(format!(
                        "`{}` says it materialises `{}`, directory says `{id}`",
                        entry.segment, header.representation
                    )));
                }
                if header.tensors.len() != entry.tensor_count {
                    defects.push(InspectionDefect::SegmentIncoherent(format!(
                        "`{}` carries {} tensors, directory says {}",
                        entry.segment,
                        header.tensors.len(),
                        entry.tensor_count
                    )));
                }
                let file_len = std::fs::metadata(root.join(&entry.segment))
                    .map(|m| m.len())
                    .unwrap_or(0);
                if file_len != payload_start + entry.payload_bytes {
                    defects.push(InspectionDefect::SegmentIncoherent(format!(
                        "`{}` is {file_len} bytes, expected {} header + {} payload",
                        entry.segment, payload_start, entry.payload_bytes
                    )));
                }
                if verify_payloads {
                    verify_segment_hash(root, entry, &mut defects)?;
                }
            }
            Err(e) => defects.push(InspectionDefect::SegmentIncoherent(format!(
                "`{}`: {e}",
                entry.segment
            ))),
        }
    }

    let components = graph
        .components
        .iter()
        .map(|c| {
            let table = c.attention.as_ref();
            let sliding = table.map(|t| {
                t.iter()
                    .filter(|l| matches!(l.span, Some(AttentionSpan::Sliding)))
                    .count()
            });
            ComponentSummary {
                id: c.id.clone(),
                role: format!("{:?}", c.role).to_lowercase(),
                num_layers: c.num_layers,
                hidden_size: c.hidden_size,
                sliding_layers: sliding,
                // `n - sliding` counted every recurrence as a full-
                // attention layer, so the first real hybrid container
                // read back "0 sliding / 64 full" for a stack that is 48
                // recurrent. Counted by operator instead, and the
                // recurrent count is reported rather than folded away.
                full_layers: table.map(|t| {
                    t.iter()
                        .filter(|l| l.span == Some(AttentionSpan::Full))
                        .count()
                }),
                recurrent_layers: table
                    .map(|t| t.iter().filter(|l| l.operator.is_gated_delta()).count()),
                nope_layers: table.map(|t| {
                    t.iter()
                        .filter(|l| l.position == PositionPolicy::None)
                        .count()
                }),
                window: table.and_then(|t| t.iter().find_map(|l| l.window)),
            }
        })
        .collect();

    Ok(SystemInspection {
        components,
        graph,
        index,
        defects,
    })
}

/// Re-hash one segment file and compare against the directory entry.
fn verify_segment_hash(
    root: &Path,
    entry: &super::index::RepresentationEntry,
    defects: &mut Vec<InspectionDefect>,
) -> Result<(), VindexError> {
    let actual = crate::format::checksums::sha256_file(&root.join(&entry.segment))?;
    if actual != entry.segment_sha256 {
        defects.push(InspectionDefect::PayloadCorrupt(format!(
            "`{}` hashes {actual}, directory says {}",
            entry.segment, entry.segment_sha256
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
