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

pub mod checkpoint;
pub mod segment;
pub mod source;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larql_models::inventory::ArchitectureInventory;

use super::auxiliary_references::{AuxiliaryReference, AuxiliaryReferences, OperandAddress};
use super::graph::{most_specific_owner, LogicalObject, SystemGraph};
use super::index::{RepresentationEntry, Vindex3Index};
use super::plan::plan_system;
use crate::error::VindexError;
use crate::format::filenames::{AUXILIARY_REFERENCES_JSON, INDEX_JSON};
use segment::PlannedTensor;
use source::{ArtifactSource, TensorSource};

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
    encode_system_from_sources(named, None, out, None)
}

/// Encode a system gated on ONE capability's dependency closure instead of
/// the whole model's.
///
/// [`encode_system`] is unchanged and still means *every declared
/// execution-semantic fact of this checkpoint is understood*. This is the
/// narrower question the capability machinery exists to answer: **is
/// everything the requested capability actually executes understood?**
///
/// Qwen3.8-27B is why. Its text closure is complete — 0 blocking — while
/// 16 whole-model findings remain: an unexecutable vision surface, an
/// unsupported MTP draft head, and two metadata keys whose semantic
/// ownership is unresolved. None of those is reachable from a text
/// forward pass, and refusing to write the container because of them
/// would block a capability this build has proven it can run.
///
/// The graph encoded is the WHOLE graph, not a text-only projection: the
/// container stays a faithful record of the checkpoint, and it is the
/// admission GATE that is scoped, never the contents. A reader of the
/// container still sees the vision component and still finds it
/// inadmissible.
pub fn encode_system_for_capability(
    named: &[(String, ArchitectureInventory)],
    out: &Path,
    capability: crate::format::vindex3::plan::capability::Capability,
) -> Result<EncodeOutcome, VindexError> {
    encode_system_from_sources(named, None, out, Some(capability))
}

/// [`encode_system`] over payload sources the caller supplies.
///
/// The gate is unchanged and runs first, on the inventories — which is
/// the whole reason a remote encode is possible at all. Admission reads
/// headers, and headers are cheap; only once the plan is admissible does
/// anything ask a source for a payload byte.
///
/// `sources` maps artifact name to the place that artifact's tensors come
/// from. `None` opens the local directory each inventory records, which
/// is what every caller predating remote sources meant.
///
/// `capability` scopes the admission gate exactly as
/// [`encode_system_for_capability`] describes; `None` is the whole-model
/// bar.
pub fn encode_system_from_sources(
    named: &[(String, ArchitectureInventory)],
    sources: Option<&BTreeMap<&str, &dyn TensorSource>>,
    out: &Path,
    capability: Option<crate::format::vindex3::plan::capability::Capability>,
) -> Result<EncodeOutcome, VindexError> {
    let plan = plan_system(named);
    match capability {
        None => {
            if !plan.admissible {
                return Err(VindexError::Parse(format!(
                    "refusing to encode an inadmissible plan: {} blocking finding(s); \
                     run `larql vindex3 plan` for the itemised reasons",
                    plan.summary.blocking
                )));
            }
        }
        Some(capability) => {
            let status = plan
                .capabilities
                .iter()
                .find(|c| c.capability == capability)
                .ok_or_else(|| {
                    VindexError::Parse(format!("this build reports no verdict for {capability:?}"))
                })?;
            if !status.admissible {
                return Err(VindexError::Parse(format!(
                    "refusing to encode for {:?}: {} blocking finding(s) inside that \
                     capability's closure; run `larql vindex3 plan` for the itemised reasons",
                    capability, status.blocking
                )));
            }
            // Understanding the semantics is not the same as this build
            // being able to run them, and a container written for a
            // capability nothing can execute would be a promise the
            // runtime cannot keep.
            if !status.supported {
                return Err(VindexError::Parse(format!(
                    "refusing to encode for {capability:?}: this build has no executor for it"
                )));
            }
        }
    }
    let outcome = match sources {
        Some(sources) => encode_graph_with_sources(&plan.graph, named, sources, out)?,
        None => encode_graph(&plan.graph, named, out)?,
    };
    // Closure at encode holds on THIS path too (drill F4): a container
    // whose operands do not close is removed, never written — the
    // single-checkpoint path has enforced this since schema 6, and the
    // 2.7B hybrid witness caught this sibling path not enforcing it.
    checkpoint::enforce_closure_at_encode(out)?;
    Ok(outcome)
}

/// `encode_system` WITHOUT the closure-at-encode gate — the
/// doctored-write seam for in-crate tests that construct their subject
/// by encoding a deliberately defective source, or by editing the
/// persisted graph after the write, and then prove the defect is caught
/// downstream. The gate exists precisely so no PRODUCTION path can do
/// what this function permits; it is `cfg(test)` so none ever can.
#[cfg(test)]
pub(crate) fn encode_system_unenforced(
    named: &[(String, ArchitectureInventory)],
    out: &Path,
) -> Result<EncodeOutcome, VindexError> {
    let plan = plan_system(named);
    if !plan.admissible {
        // Itemised, because a test has no CLI to run for the reasons.
        let blocking: Vec<String> = plan
            .artifacts
            .iter()
            .flat_map(|a| &a.findings)
            .filter(|f| f.blocks())
            .map(|f| format!("{}: {}", f.subject, f.detail))
            .collect();
        return Err(VindexError::Parse(format!(
            "refusing to encode an inadmissible plan: {} blocking finding(s):\n  {}",
            plan.summary.blocking,
            blocking.join("\n  ")
        )));
    }
    encode_graph(&plan.graph, named, out)
}

/// Encode an already-built system graph. `encode_system` is this after
/// the plan gate; a caller holding a graph it built (or edited) itself
/// comes in here and gets the same validation and the same bytes.
///
/// **This seam does not enforce closure at encode**: a caller holding
/// its own (possibly deliberately doctored) graph owns that
/// responsibility, and the closure tests use exactly this property to
/// write containers whose defects they then prove are caught. Every
/// CLI writer path goes through the enforcing wrappers above.
pub fn encode_graph(
    graph: &SystemGraph,
    named: &[(String, ArchitectureInventory)],
    out: &Path,
) -> Result<EncodeOutcome, VindexError> {
    // Validation before I/O: a defective graph must report the defect,
    // not a failure to open the checkpoint it would never have read.
    // `encode_graph_with_sources` validates again — it is a public entry
    // and cannot assume it — and validating an in-memory graph twice
    // costs nothing.
    validate_graph(graph)?;

    // Sources by artifact name, opened once, from the directory each
    // inventory records.
    let opened: Vec<(&str, ArtifactSource)> = named
        .iter()
        .map(|(name, inventory)| {
            ArtifactSource::open(Path::new(&inventory.path)).map(|source| (name.as_str(), source))
        })
        .collect::<Result<_, _>>()?;
    let sources: BTreeMap<&str, &dyn TensorSource> = opened
        .iter()
        .map(|(name, source)| (*name, source as &dyn TensorSource))
        .collect();
    encode_graph_with_sources(graph, named, &sources, out)
}

/// [`encode_graph`] over payload sources the caller supplies, keyed by
/// artifact name.
///
/// This is where the container's bytes are actually written, and it is
/// the only place that reads a payload. Everything above it — plan,
/// admission, capability closure — runs on headers alone.
pub fn encode_graph_with_sources(
    graph: &SystemGraph,
    named: &[(String, ArchitectureInventory)],
    sources: &BTreeMap<&str, &dyn TensorSource>,
    out: &Path,
) -> Result<EncodeOutcome, VindexError> {
    validate_graph(graph)?;

    for (name, _) in named {
        if !sources.contains_key(name.as_str()) {
            return Err(VindexError::Parse(format!(
                "no payload source supplied for artifact `{name}`"
            )));
        }
    }
    let inventories: BTreeMap<&str, &ArchitectureInventory> =
        named.iter().map(|(n, i)| (n.as_str(), i)).collect();

    std::fs::create_dir_all(out)?;
    let mut directory: BTreeMap<String, RepresentationEntry> = BTreeMap::new();
    let mut segment_keys: BTreeMap<String, u32> = BTreeMap::new();
    let mut total_payload_bytes = 0u64;
    let mut references: Vec<AuxiliaryReference> = Vec::new();

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
        references.extend(declared_references(
            &object.id,
            planned
                .iter()
                .map(|t| (t.relative_name.as_str(), t.dtype.as_str())),
        ));

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
                // Encoded from the source checkpoint, not compiled from
                // another representation: there is no container-side
                // authority to name, and the bytes are the source's, so
                // this compiler's ABI does not describe them.
                compiled_from: None,
                codec: None,
                source_representation_digest: None,
                encoder: None,
            },
        );
    }

    // Graph manifest, then the index last — a crash midway leaves a
    // directory that is not yet a container rather than one that lies.
    let graph_json = serde_json::to_string_pretty(graph)
        .map_err(|e| VindexError::Parse(format!("serialise system graph: {e}")))?;
    std::fs::write(out.join(SYSTEM_GRAPH_JSON), graph_json)?;

    let mut index = system_index(graph, named, directory, segment_keys)?;
    // The dependencies the container DECLARES, written before the index
    // that names the table: a reader that finds the index finds the
    // table. Absent when nothing depends on anything — absence means "no
    // dependency is declared", never "look somewhere else".
    if !references.is_empty() {
        let table = AuxiliaryReferences::new(references);
        let table_json = serde_json::to_string_pretty(&table)
            .map_err(|e| VindexError::Parse(format!("serialise auxiliary references: {e}")))?;
        std::fs::write(out.join(AUXILIARY_REFERENCES_JSON), table_json)?;
        index.auxiliary_references = Some(AUXILIARY_REFERENCES_JSON.to_string());
    }
    let index_json = serde_json::to_string_pretty(&index)
        .map_err(|e| VindexError::Parse(format!("serialise index.json: {e}")))?;
    std::fs::write(out.join(INDEX_JSON), index_json)?;

    Ok(EncodeOutcome {
        container: out.to_path_buf(),
        representations: index.representations.len(),
        total_payload_bytes,
    })
}

/// The dependencies one object's tensors declare on each other, as the
/// container will address them.
///
/// This is where a checkpoint's naming convention becomes a declared
/// reference, and the only place that convention is consumed: fine-grained
/// FP8 ships a `*.weight_scale_inv` grid beside every E4M3 `*.weight`, and
/// the FP8 codec requires that grid under the name it declares. A sibling
/// beside a weight stored under any OTHER label declares nothing — the
/// pairing is the FP8 representation's, not a rule about names — and a
/// stray grid whose weight is absent is left for closure to report.
fn declared_references<'a>(
    object: &str,
    tensors: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Vec<AuxiliaryReference> {
    use super::represent::codec::codecs::fp8_block::{DTYPE_FP8_BLOCK, SCALES};
    use larql_models::quant::fp8_finegrained::weight_of_scale_sibling;
    let tensors: Vec<(&str, &str)> = tensors.into_iter().collect();
    let fp8_weights: std::collections::BTreeSet<&str> = tensors
        .iter()
        .filter(|(_, dtype)| *dtype == DTYPE_FP8_BLOCK)
        .map(|(name, _)| *name)
        .collect();
    tensors
        .iter()
        .filter_map(|(name, _)| {
            let weight = weight_of_scale_sibling(name)?;
            fp8_weights
                .contains(weight.as_str())
                .then(|| AuxiliaryReference {
                    owner: OperandAddress::new(object, &weight),
                    auxiliary: SCALES.to_string(),
                    target: OperandAddress::new(object, *name),
                })
        })
        .collect()
}

/// Declare the dependencies a container's tensors imply, for a container
/// encoded before the encoder declared them.
///
/// The rule is [`declared_references`] — the encoder's own, applied late
/// to the segment headers already on disk — so a container migrated here
/// carries exactly the table a fresh encode would have written, and the
/// convention it consumes stays consumed in one place. Writes the table
/// and names it in the index; returns how many references were declared,
/// and writes nothing when there are none (absence means "no dependency
/// is declared"). A container that already declares a table is refused
/// rather than merged with: which of two tables an encoder meant has no
/// answer, and the table that is there was written on purpose.
pub fn declare_references(root: &Path) -> Result<usize, VindexError> {
    let index_path = root.join(INDEX_JSON);
    let mut index: Vindex3Index = serde_json::from_str(&std::fs::read_to_string(&index_path)?)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    if let Some(name) = &index.auxiliary_references {
        return Err(VindexError::Parse(format!(
            "the container already declares its dependencies in `{name}`; refusing to \
             rewrite a table that was written on purpose"
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut references = Vec::new();
    for entry in index.representations.values() {
        let (header, _) = segment::read_segment_header(&root.join(&entry.segment))?;
        let tensors = header
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t.dtype.as_str()));
        // Two representations of one object — its canonical bytes and a
        // compiled pack — may both hold the pair; one dependency, declared
        // once, is what the table admits.
        for reference in declared_references(&entry.object, tensors) {
            if seen.insert((reference.owner.clone(), reference.auxiliary.clone())) {
                references.push(reference);
            }
        }
    }
    let declared = references.len();
    if declared == 0 {
        return Ok(0);
    }
    let table = AuxiliaryReferences::new(references);
    table.judge()?;
    let table_json = serde_json::to_string_pretty(&table)
        .map_err(|e| VindexError::Parse(format!("serialise auxiliary references: {e}")))?;
    std::fs::write(root.join(AUXILIARY_REFERENCES_JSON), table_json)?;
    index.auxiliary_references = Some(AUXILIARY_REFERENCES_JSON.to_string());
    let index_json = serde_json::to_string_pretty(&index)
        .map_err(|e| VindexError::Parse(format!("serialise {INDEX_JSON}: {e}")))?;
    std::fs::write(&index_path, index_json)?;
    Ok(declared)
}

/// Refuse a graph that does not validate, naming its defects.
fn validate_graph(graph: &SystemGraph) -> Result<(), VindexError> {
    let defects = graph.validate();
    if defects.is_empty() {
        return Ok(());
    }
    Err(VindexError::Parse(format!(
        "system graph failed validation before encode: {defects:?}"
    )))
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
    let primary =
        match graph.primary_text_component() {
            Ok(component) => component,
            // A graph with no primary_text (a lone drafter, say) falls back to
            // its only component; ambiguity never falls back (drill F10).
            Err(crate::format::vindex3::graph::PrimaryTextLookup::Absent) => graph
                .components
                .first()
                .ok_or_else(|| VindexError::Parse("graph has no components".into()))?,
            Err(ambiguous) => return Err(VindexError::Parse(ambiguous.to_string())),
        };
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
