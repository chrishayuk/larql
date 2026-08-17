//! Byte payload equivalence — spec §7's second, separate claim.
//!
//! Per representation, three hashes must agree:
//!
//! ```text
//! source    SHA-256 over the checkpoint payloads, recomputed NOW,
//!           in segment order (the encoder's own planner + ordering)
//! encoded   SHA-256 over the container's payload region, recomputed NOW
//! recorded  what the directory wrote at encode time
//! ```
//!
//! Recomputing *both* ends distinguishes the two failure stories a single
//! comparison would conflate: `source ≠ recorded` means the checkpoint
//! drifted since encoding; `encoded ≠ recorded` means the container did.
//!
//! Hashes cannot see a relabel: a dtype or shape edit over identical bytes
//! leaves every hash intact. The tensor *table* is therefore compared entry
//! by entry (name, dtype, shape, offset, length) as its own
//! Declared ≡ Encoded link.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use larql_models::inventory::ArchitectureInventory;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{Authority, EquivalenceCheck, PayloadCheck};
use crate::error::VindexError;
use crate::format::vindex3::encode::segment::{
    read_segment_header, sort_into_payload_order, PlannedTensor, SegmentTensor,
};
use crate::format::vindex3::encode::source::ArtifactSource;
use crate::format::vindex3::encode::{binding_owner, plan_object_tensors, REPRESENTATION_ID_SEP};
use crate::format::vindex3::graph::SystemGraph;
use crate::format::vindex3::index::Vindex3Index;

/// Streaming buffer for the encoded-side re-hash.
const STREAM_BUF: usize = 1 << 20;

/// Verify every representation of the rebuilt graph against the container.
///
/// Returns the per-representation payload verdicts plus the tensor-table
/// equivalence rows. Disagreements are findings, not errors — a missing
/// segment or a drifted source produces a failing row and the sweep
/// continues.
pub fn verify_payloads(
    named: &[(String, ArchitectureInventory)],
    graph: &SystemGraph,
    index: &Vindex3Index,
    root: &Path,
) -> Result<(Vec<PayloadCheck>, Vec<EquivalenceCheck>), VindexError> {
    let mut sources: BTreeMap<&str, ArtifactSource> = BTreeMap::new();
    for (name, inventory) in named {
        sources.insert(
            name.as_str(),
            ArtifactSource::open(Path::new(&inventory.path))?,
        );
    }
    let inventories: BTreeMap<&str, &ArchitectureInventory> =
        named.iter().map(|(n, i)| (n.as_str(), i)).collect();

    let mut payloads = Vec::new();
    let mut tables = Vec::new();
    for object in &graph.objects {
        let Some(representation) = object.representations.first() else {
            payloads.push(failed_check(
                object.id.clone(),
                "object carries no representation — nothing to compare".to_string(),
            ));
            continue;
        };
        let id = format!(
            "{}{REPRESENTATION_ID_SEP}{}",
            object.id, representation.encoding
        );
        let Some(entry) = index.representations.get(&id) else {
            payloads.push(failed_check(
                id,
                "no directory entry in the container".to_string(),
            ));
            continue;
        };

        let mut planned = match plan_object_tensors(object, &inventories, &graph.objects) {
            Ok(planned) => planned,
            Err(e) => {
                payloads.push(failed_check(id, format!("source no longer plans: {e}")));
                continue;
            }
        };
        sort_into_payload_order(&mut planned);

        let segment_path = root.join(&entry.segment);
        let (header, payload_start) = match read_segment_header(&segment_path) {
            Ok(read) => read,
            Err(e) => {
                payloads.push(failed_check(id, format!("segment unreadable: {e}")));
                continue;
            }
        };
        tables.push(table_check(&id, &planned, &header.tensors));

        let source_sha256 = match source_hash(object, &planned, &sources) {
            Ok(hash) => hash,
            Err(e) => {
                payloads.push(failed_check(id, format!("source unreadable: {e}")));
                continue;
            }
        };
        let encoded_sha256 = payload_region_hash(&segment_path, payload_start)?;

        let source_ok = source_sha256 == entry.payload_sha256;
        let encoded_ok = encoded_sha256 == entry.payload_sha256;
        let detail = match (source_ok, encoded_ok) {
            (true, true) => "source, container and directory agree byte-for-byte".to_string(),
            (false, true) => {
                "source ≠ recorded — the checkpoint changed since encoding".to_string()
            }
            (true, false) => {
                "encoded ≠ recorded — the container changed since encoding".to_string()
            }
            (false, false) => "source and container both disagree with the directory".to_string(),
        };
        payloads.push(PayloadCheck {
            representation: id,
            payload_bytes: entry.payload_bytes,
            source_sha256,
            encoded_sha256,
            recorded_sha256: entry.payload_sha256.clone(),
            pass: source_ok && encoded_ok,
            detail,
        });
    }
    Ok((payloads, tables))
}

/// A payload row that failed before any hash could be computed.
fn failed_check(representation: String, detail: String) -> PayloadCheck {
    PayloadCheck {
        representation,
        payload_bytes: 0,
        source_sha256: String::new(),
        encoded_sha256: String::new(),
        recorded_sha256: String::new(),
        pass: false,
        detail,
    }
}

/// SHA-256 over the source payloads in segment order — the same bytes the
/// encoder streamed, re-read from the checkpoint now.
fn source_hash(
    object: &crate::format::vindex3::graph::LogicalObject,
    planned: &[PlannedTensor],
    sources: &BTreeMap<&str, ArtifactSource>,
) -> Result<String, VindexError> {
    let mut hasher = Sha256::new();
    let mut sink = std::io::sink();
    for tensor in planned {
        let owner = binding_owner(object, &tensor.source_name).ok_or_else(|| {
            VindexError::Parse(format!(
                "tensor `{}` matches no binding of `{}`",
                tensor.source_name, object.id
            ))
        })?;
        let source = sources.get(owner).ok_or_else(|| {
            VindexError::Parse(format!("binding owner `{owner}` is not in the verify set"))
        })?;
        source.stream_payload(&tensor.source_name, &mut sink, &mut |chunk| {
            hasher.update(chunk);
        })?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// SHA-256 over a segment file's payload region (everything after the
/// header), recomputed from the container now.
fn payload_region_hash(path: &Path, payload_start: u64) -> Result<String, VindexError> {
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(payload_start))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; STREAM_BUF];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// The tensor table, planned-from-source vs stored-in-segment — the relabel
/// detector. Offsets on the planned side are recomputed cumulatively, the
/// same way the writer laid them out.
fn table_check(
    representation: &str,
    planned: &[PlannedTensor],
    stored: &[SegmentTensor],
) -> EquivalenceCheck {
    let mut offset = 0u64;
    let expected: Vec<Value> = planned
        .iter()
        .map(|t| {
            let row = json!({
                "name": t.relative_name,
                "dtype": t.dtype,
                "shape": t.shape,
                "offset": offset,
                "len": t.len,
            });
            offset += t.len;
            row
        })
        .collect();
    let actual: Vec<Value> = stored
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "dtype": t.dtype,
                "shape": t.shape,
                "offset": t.offset,
                "len": t.len,
            })
        })
        .collect();
    let first_mismatch = expected
        .iter()
        .zip(&actual)
        .position(|(a, b)| a != b)
        .or((expected.len() != actual.len()).then_some(expected.len().min(actual.len())));
    EquivalenceCheck {
        left: Authority::Declared,
        right: Authority::Encoded,
        subject: format!("{representation}.tensor_table"),
        pass: first_mismatch.is_none(),
        detail: match first_mismatch {
            None => format!(
                "{} entries agree (name, dtype, shape, offset, len)",
                actual.len()
            ),
            Some(i) => format!(
                "first disagreement at entry {i} ({} planned vs {} stored)",
                expected.len(),
                actual.len()
            ),
        },
        left_value: Value::Array(expected),
        right_value: Value::Array(actual),
    }
}
