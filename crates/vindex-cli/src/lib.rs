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
use larql_vindex::format::vindex3::inspect::{inspect_container, SystemInspection};
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::OperandRef;
use larql_vindex::format::vindex3::represent::nvfp4_pack::{split, PackLayout, DTYPE_NVFP4};
use larql_vindex::format::vindex3::represent::{compile_representation, RepresentSpec};

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
    let object = find_object(&inspection, address)?;
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
        "total_weight_slots": total_weights as u64,
        "stored_bits_per_weight_slot": if total_weights > 0 { (total_bits as f64) / (total_weights as f64) } else { 0.0 },
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

fn find_object<'a>(
    inspection: &'a SystemInspection,
    address: &str,
) -> Result<&'a larql_vindex::format::vindex3::graph::object::LogicalObject, String> {
    inspection
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
        })
}

/// A tensor as one side's header declares it: name, dtype, shape.
type TensorEntry = (String, String, Vec<usize>);

/// One side of a diff: the store bound to `encoding`, plus that
/// encoding's tensor table for the object. Refuses an encoding the
/// container does not hold, naming what it does.
fn open_side(
    root: &Path,
    inspection: &SystemInspection,
    object: &str,
    encoding: &str,
) -> Result<(OperandStore, Vec<TensorEntry>), String> {
    let canonical = inspection
        .graph
        .objects
        .iter()
        .find(|o| o.id == object)
        .and_then(|o| o.representations.first())
        .map(|r| r.encoding.clone())
        .ok_or_else(|| format!("object `{object}` declares no representations"))?;
    let store = if encoding.eq_ignore_ascii_case(&canonical) {
        OperandStore::open(root, inspection).map_err(|e| format!("open {encoding}: {e}"))?
    } else {
        OperandStore::open_for(
            root,
            inspection,
            Some(encoding),
            RepresentationSource::Stored,
        )
        .map_err(|e| format!("open {encoding}: {e}"))?
    };
    let bound = store
        .selection()
        .get(object)
        .map(|s| s.encoding.clone())
        .unwrap_or_default();
    if !bound.eq_ignore_ascii_case(encoding) {
        let held: Vec<String> = inspection
            .index
            .representations
            .values()
            .filter(|e| e.object == object)
            .map(|e| e.encoding.clone())
            .collect();
        return Err(format!(
            "`{object}` has no {encoding} representation — the container holds: {}",
            held.join(", ")
        ));
    }
    let entry = inspection
        .index
        .representations
        .values()
        .find(|e| e.object == object && e.encoding.eq_ignore_ascii_case(encoding))
        .ok_or_else(|| format!("no directory entry for {object}@{encoding}"))?;
    let (header, _) = read_segment_header(&root.join(&entry.segment))
        .map_err(|e| format!("segment {}: {e}", entry.segment))?;
    let tensors = header
        .tensors
        .into_iter()
        .map(|t| (t.name, t.dtype, t.shape))
        .collect();
    Ok((store, tensors))
}

/// Decode one tensor to f32, whatever the container stored it as. The
/// arithmetic for a packed encoding is the spec's — `tensor_scale ·
/// e4m3(group scale) · e2m1(code)` — so what the diff compares is what
/// a matmul against those bytes would effectively use.
fn load_values(store: &OperandStore, operand: &OperandRef) -> Result<Vec<f32>, String> {
    if operand.dtype != DTYPE_NVFP4 {
        return store
            .load(operand)
            .map_err(|e| format!("load {}: {e}", operand.tensor));
    }
    let raw = store
        .load_raw(operand)
        .map_err(|e| format!("read {}: {e}", operand.tensor))?;
    let layout = PackLayout::derive(&operand.shape, &operand.tensor)
        .map_err(|e| format!("{}: {e}", operand.tensor))?;
    let (packed, scales, tensor_scale) = split(&raw.bytes, &layout, &operand.tensor)
        .map_err(|e| format!("{}: {e}", operand.tensor))?;
    let matrix = larql_models::quant::nvfp4::Nvfp4Matrix {
        packed: packed.to_vec(),
        scales: scales.to_vec(),
        tensor_scale,
    };
    let (rows, k) = (operand.shape[0], operand.shape[1]);
    let mut out = vec![0.0f32; rows * k];
    larql_models::quant::nvfp4::dequantize_into(&matrix, rows, k, &mut out)
        .map_err(|e| format!("decode {}: {e:?}", operand.tensor))?;
    Ok(out)
}

/// `vindex diff <a> <b> <address>` — one object decoded under two of
/// the container's own representations, compared value by value. The
/// error is derived, never asserted: both sides decode through the
/// same load path execution uses, and the numbers are whatever the
/// bytes disagree by.
pub fn diff_facts(
    root: &Path,
    encoding_a: &str,
    encoding_b: &str,
    address: &str,
    values: usize,
    tensor_filter: Option<&str>,
) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let object = find_object(&inspection, address)?.id.clone();
    let (encoding_a, encoding_b) = (encoding_a.to_uppercase(), encoding_b.to_uppercase());
    let (store_a, tensors_a) = open_side(root, &inspection, &object, &encoding_a)?;
    let (store_b, tensors_b) = open_side(root, &inspection, &object, &encoding_b)?;
    let dtypes_b: std::collections::BTreeMap<&str, (&str, &Vec<usize>)> = tensors_b
        .iter()
        .map(|(n, d, s)| (n.as_str(), (d.as_str(), s)))
        .collect();

    let mut rows: Vec<Value> = Vec::new();
    let mut head: Option<Value> = None;
    let mut worst: f64 = -1.0;
    let mut sum_sq = 0.0f64;
    let mut max_error = 0.0f64;
    let mut total_weights = 0u64;
    let mut changed_values = 0u64;
    for (name, dtype_a, shape) in &tensors_a {
        let Some((dtype_b, shape_b)) = dtypes_b.get(name.as_str()) else {
            rows.push(json!({ "tensor": name, "note": format!("only in {encoding_a}") }));
            continue;
        };
        if shape != *shape_b {
            rows.push(json!({
                "tensor": name,
                "note": format!("shape differs: {shape:?} vs {shape_b:?}"),
            }));
            continue;
        }
        let make_ref = |dtype: &str| OperandRef {
            object: object.clone(),
            tensor: name.clone(),
            dtype: dtype.to_string(),
            shape: shape.clone(),
        };
        let va = load_values(&store_a, &make_ref(dtype_a))
            .map_err(|e| format!("as {encoding_a}: {e}"))?;
        let vb = load_values(&store_b, &make_ref(dtype_b))
            .map_err(|e| format!("as {encoding_b}: {e}"))?;
        let n = va.len().min(vb.len());
        let mut t_sum_sq = 0.0f64;
        let mut t_max = 0.0f64;
        let mut t_changed = 0u64;
        for i in 0..n {
            let d = (va[i] as f64) - (vb[i] as f64);
            t_sum_sq += d * d;
            if d.abs() > t_max {
                t_max = d.abs();
            }
            if d != 0.0 {
                t_changed += 1;
            }
        }
        let rms = if n > 0 {
            (t_sum_sq / n as f64).sqrt()
        } else {
            0.0
        };
        sum_sq += t_sum_sq;
        total_weights += n as u64;
        changed_values += t_changed;
        if t_max > max_error {
            max_error = t_max;
        }
        rows.push(json!({
            "tensor": name,
            "dtype_a": dtype_a,
            "dtype_b": dtype_b,
            "weights": n as u64,
            "changed": t_changed,
            "rms_error": rms,
            "max_error": t_max,
        }));
        // A suffix only matches at a path boundary: `0.mlp.down` must not
        // match `30.mlp.down`. The first matching tensor wins.
        let wanted = match tensor_filter {
            Some(f) => head.is_none() && (name == f || name.ends_with(&format!(".{f}"))),
            None => rms > worst,
        };
        if wanted {
            worst = rms;
            head = Some(json!({
                "tensor": name,
                "rows": (0..n.min(values)).map(|i| json!({
                    "a": va[i],
                    "b": vb[i],
                    "error": vb[i] - va[i],
                })).collect::<Vec<Value>>(),
            }));
        }
    }
    if let Some(f) = tensor_filter {
        if head.is_none() {
            return Err(format!(
                "no tensor of `{object}` matches `{f}` — tensors: {}",
                tensors_a
                    .iter()
                    .map(|(n, _, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    Ok(json!({
        "container": root.display().to_string(),
        "object": object,
        "a": encoding_a,
        "b": encoding_b,
        "tensors": rows,
        "values": head,
        "total_weights": total_weights,
        "changed_values": changed_values,
        "rms_error": if total_weights > 0 { (sum_sq / total_weights as f64).sqrt() } else { 0.0 },
        "max_error": max_error,
        "identical": changed_values == 0,
    }))
}

/// `vindex represent <src> <out>` — compile a representation the spec
/// defines, through the reference compiler. The output container
/// carries every original segment plus the compiled packs; nothing is
/// destroyed, and `vindex verify` holds on the result.
pub fn represent_facts(src: &Path, out: &Path, encoding: &str) -> Facts {
    let mut spec = RepresentSpec::nvfp4();
    spec.encoding = encoding.to_uppercase();
    let report = compile_representation(src, out, &spec).map_err(|e| format!("represent: {e}"))?;
    let compiled: Vec<Value> = report
        .compiled_objects
        .iter()
        .map(|c| {
            json!({
                "object": c.object,
                "representation": c.representation_id,
                "compiled_tensors": c.compiled_tensors,
                "carried_tensors": c.carried_tensors,
                "source_bytes": c.source_bytes,
                "compiled_bytes": c.compiled_bytes,
                "compression": c.compression(),
                "preserved_roles": c.preserved.iter()
                    .map(|(role, n)| json!({ "role": format!("{role:?}"), "tensors": n }))
                    .collect::<Vec<Value>>(),
            })
        })
        .collect();
    let preserved: Vec<Value> = report
        .preserved_objects
        .iter()
        .map(|p| {
            json!({
                "object": p.object,
                "encoding": p.encoding,
                "bytes": p.bytes,
            })
        })
        .collect();
    Ok(json!({
        "source": src.display().to_string(),
        "out": out.display().to_string(),
        "encoding": spec.encoding,
        "map": spec.map_name(),
        "compiled": compiled,
        "preserved": preserved,
        "linked_segments": report.linked_segments,
    }))
}
