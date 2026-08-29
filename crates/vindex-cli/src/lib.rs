//! vindex — the format-native facts, as data.
//!
//! Every function here reads only what the container declares —
//! `index.json`, the system graph, the segment headers — and returns
//! one `serde_json::Value`: the same object the binary prints with
//! `--json`, renders as text without it, and the web Explorer renders
//! as a designed panel. One result, three projections; the litmus
//! test for every fact is that an independent VINDEX3 implementation
//! could derive it from the artifact alone.

use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::index::{ContainerAuthority, Vindex3Index};
use larql_vindex::format::vindex3::inspect::inspect_container;

pub type Facts = Result<Value, String>;

fn read_index(root: &Path) -> Result<Vindex3Index, String> {
    let text = std::fs::read_to_string(root.join(INDEX_JSON))
        .map_err(|e| format!("read {INDEX_JSON}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {INDEX_JSON}: {e}"))
}

fn authority_str(a: &ContainerAuthority) -> &'static str {
    match a {
        ContainerAuthority::Canonical => "canonical",
        ContainerAuthority::Derived => "derived",
    }
}

/// `vindex inspect` — the container, reconstructed from itself.
pub fn inspect_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let index = &inspection.index;
    Ok(json!({
        "container": root.display().to_string(),
        "generation": 3,
        "model": index.model,
        "family": index.family,
        "hidden_size": index.hidden_size,
        "num_layers": index.num_layers,
        "authority": authority_str(&index.authority),
        "derived_from_model": index.derived_from_model,
        "components": inspection.components,
        "objects": inspection.graph.objects.len(),
        "edges": inspection.graph.edges.len(),
        "coherent": inspection.is_coherent(),
        "defects": inspection.defects.len(),
    }))
}

/// `vindex representations` — the physical directory, with the graph's
/// fidelity beside each entry.
pub fn representations_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let fidelity_of = |object: &str, encoding: &str| -> Option<String> {
        inspection
            .graph
            .objects
            .iter()
            .find(|o| o.id == object)
            .and_then(|o| o.representations.iter().find(|r| r.encoding == encoding))
            .map(|r| format!("{:?}", r.fidelity).to_lowercase())
    };
    let entries: Vec<Value> = inspection
        .index
        .representations
        .iter()
        .map(|(id, e)| {
            json!({
                "id": id,
                "object": e.object,
                "encoding": e.encoding,
                "fidelity": fidelity_of(&e.object, &e.encoding),
                "tensor_count": e.tensor_count,
                "payload_bytes": e.payload_bytes,
                "compiled_from": e.compiled_from,
            })
        })
        .collect();
    Ok(json!({ "container": root.display().to_string(), "entries": entries }))
}

/// `vindex describe <address>` — one logical object, in full: identity,
/// bindings, representations, and the head of its tensor table.
pub fn describe_facts(root: &Path, address: &str, values: usize) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let object = inspection
        .graph
        .objects
        .iter()
        .find(|o| o.id == address || o.id.ends_with(address))
        .ok_or_else(|| {
            let known: Vec<&str> = inspection
                .graph
                .objects
                .iter()
                .map(|o| o.id.as_str())
                .collect();
            format!(
                "no logical object matches `{address}` — the graph holds: {}",
                known.join(", ")
            )
        })?;
    let directory: Vec<Value> = inspection
        .index
        .representations
        .iter()
        .filter(|(_, e)| e.object == object.id)
        .map(|(id, e)| {
            let tensors = read_segment_header(&root.join(&e.segment))
                .ok()
                .map(|(header, _)| {
                    header
                        .tensors
                        .iter()
                        .take(values)
                        .map(|t| json!({ "name": t.name, "dtype": t.dtype, "shape": t.shape, "len": t.len }))
                        .collect::<Vec<Value>>()
                });
            json!({
                "id": id,
                "encoding": e.encoding,
                "segment": e.segment,
                "tensor_count": e.tensor_count,
                "payload_bytes": e.payload_bytes,
                "tensor_table_head": tensors,
            })
        })
        .collect();
    Ok(json!({
        "container": root.display().to_string(),
        "object": serde_json::to_value(object).map_err(|e| e.to_string())?,
        "directory": directory,
    }))
}

/// `vindex precision` — bits per weight, derived, never asserted: the
/// stored payload bytes over the tensor-table element counts, per
/// representation and effective across the container. The compiled
/// precision map is passed through verbatim when the index carries one.
pub fn precision_facts(root: &Path) -> Facts {
    let index = read_index(root)?;
    let mut total_bits = 0u128;
    let mut total_weights = 0u128;
    let mut per_entry: Vec<Value> = Vec::new();
    for (id, e) in &index.representations {
        let (header, _) = read_segment_header(&root.join(&e.segment))
            .map_err(|err| format!("segment {}: {err}", e.segment))?;
        let weights: u128 = header
            .tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>() as u128)
            .sum();
        let bits = e.payload_bytes as u128 * 8;
        if weights > 0 {
            total_bits += bits;
            total_weights += weights;
        }
        per_entry.push(json!({
            "id": id,
            "encoding": e.encoding,
            "weights": weights as u64,
            "payload_bytes": e.payload_bytes,
            "bits_per_weight": if weights > 0 { (bits as f64) / (weights as f64) } else { 0.0 },
        }));
    }
    Ok(json!({
        "container": root.display().to_string(),
        "entries": per_entry,
        "total_weights": total_weights as u64,
        "effective_bits_per_weight": if total_weights > 0 { (total_bits as f64) / (total_weights as f64) } else { 0.0 },
        "precision_map": index.precision_map,
    }))
}

/// `vindex verify` — the container against its own recorded hashes:
/// every segment file re-hashed whole, every payload region re-hashed,
/// both compared with what the directory recorded at encode time. This
/// is self-verification — corruption detection from the artifact
/// alone. Proving faithfulness to the *source* additionally needs the
/// source, and that lives with the reference implementation's G4 gate.
pub fn verify_facts(root: &Path) -> Facts {
    let index = read_index(root)?;
    let mut entries: Vec<Value> = Vec::new();
    let mut failures = 0usize;
    for (id, e) in &index.representations {
        let path = root.join(&e.segment);
        let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", e.segment))?;
        let segment_hash = format!("{:x}", Sha256::digest(&bytes));
        let payload_start = 8 + u64::from_le_bytes(
            bytes
                .get(0..8)
                .ok_or_else(|| format!("{}: shorter than its own framing", e.segment))?
                .try_into()
                .map_err(|_| "framing".to_string())?,
        ) as usize;
        let payload = bytes
            .get(payload_start..)
            .ok_or_else(|| format!("{}: payload offset beyond file", e.segment))?;
        let payload_hash = format!("{:x}", Sha256::digest(payload));
        let segment_ok = segment_hash == e.segment_sha256;
        let payload_ok = payload_hash == e.payload_sha256;
        if !segment_ok || !payload_ok {
            failures += 1;
        }
        entries.push(json!({
            "id": id,
            "segment_ok": segment_ok,
            "payload_ok": payload_ok,
        }));
    }
    Ok(json!({
        "container": root.display().to_string(),
        "entries": entries,
        "failures": failures,
        "verified": failures == 0,
        "scope": "self — recorded hashes re-derived from the artifact alone; source faithfulness is the reference implementation's G4",
    }))
}
