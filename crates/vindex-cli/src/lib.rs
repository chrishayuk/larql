//! vindex — the format-native facts, as data.
//!
//! Every function here reads only what the container declares —
//! `index.json`, the system graph, the segment headers — and returns
//! one `serde_json::Value`: the same object the binary prints with
//! `--json`, renders as text without it, and the web Explorer renders
//! as a designed panel. One result, three projections; the litmus
//! test for every fact is that an independent VINDEX3 implementation
//! could derive it from the artifact alone.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub mod mixer;
use mixer::{layer_range, MixerOperands};

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::artifact;
use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::graph::Component;
use larql_vindex::format::vindex3::index::{ContainerAuthority, Vindex3Index};
use larql_vindex::format::vindex3::inspect::{inspect_container, SystemInspection};
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use larql_vindex::format::vindex3::opplan::{
    plan_component_ops, ComponentOpPlan, LayerFfn, OperandRef,
};
use larql_vindex::format::vindex3::plan::capability::Capability;
use larql_vindex::format::vindex3::plan::{plan_system_with_sources, ArtifactSource};
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
/// bindings, representations, and the head of its tensor table. With
/// `peek`, the first `values` decoded weights of one named tensor —
/// the numbers themselves, read from the canonical bytes.
pub fn describe_facts(root: &Path, address: &str, values: usize, peek: Option<&str>) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    // `layer.N.mixer` — the token mixer as a first-class semantic
    // component: its programme, and every operand with the role the
    // plan assigns it. Architecture-neutral by construction.
    if let Some(n) = address
        .strip_prefix("layer.")
        .and_then(|r| r.strip_suffix(".mixer"))
        .and_then(|n| n.parse::<usize>().ok())
    {
        let (component, plan) = primary_plan(root, &inspection)?;
        let operator = mixer::declared_operator(component, n)?;
        let layer = plan
            .layers
            .get(n)
            .ok_or_else(|| layer_range(&format!("layer {n}"), plan.layers.len()))?;
        let (operands, undescribed) = match mixer::operands(operator, layer) {
            MixerOperands::Named(ops) => (
                ops.into_iter()
                    .map(|(role, op)| {
                        json!({ "role": role, "tensor": op.tensor, "shape": op.shape, "dtype": op.dtype })
                    })
                    .collect::<Vec<Value>>(),
                None,
            ),
            MixerOperands::Undescribed(why) => (Vec::new(), Some(why)),
        };
        return Ok(json!({
            "container": root.display().to_string(),
            "semantic": address,
            "mixer": mixer::label(operator, mixer::has_output_gate(layer)),
            "layer": n,
            "operands": operands,
            "undescribed": undescribed,
        }));
    }
    if let Some(resolved) = resolve_semantic(root, &inspection, address) {
        let (role, op) = resolved?;
        let mut representations: Vec<Value> = Vec::new();
        for entry in inspection
            .index
            .representations
            .values()
            .filter(|e| e.object == op.object)
        {
            let (header, _) = read_segment_header(&root.join(&entry.segment))
                .map_err(|e| format!("segment {}: {e}", entry.segment))?;
            if let Some(t) = header.tensors.iter().find(|t| t.name == op.tensor) {
                let weights: u128 = t.shape.iter().product::<usize>() as u128;
                representations.push(json!({
                    "encoding": entry.encoding,
                    "dtype": t.dtype,
                    "bits_per_weight": if weights > 0 { (t.len as u128 * 8) as f64 / weights as f64 } else { 0.0 },
                    "bytes": t.len,
                }));
            }
        }
        let store = OperandStore::open(root, &inspection).map_err(|e| format!("open: {e}"))?;
        let decoded = load_values(&store, &op)?;
        return Ok(json!({
            "container": root.display().to_string(),
            "semantic": address,
            "role": role,
            "object": op.object,
            "tensor": op.tensor,
            "shape": op.shape,
            "representations": representations,
            "values": decoded.iter().take(values).collect::<Vec<_>>(),
        }));
    }
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
    let peeked = match peek {
        None => Value::Null,
        Some(tensor) => {
            let store = OperandStore::open(root, &inspection).map_err(|e| format!("open: {e}"))?;
            let entry = inspection
                .index
                .representations
                .values()
                .find(|e| e.object == object.id)
                .ok_or_else(|| format!("`{}` has no directory entry", object.id))?;
            let (header, _) = read_segment_header(&root.join(&entry.segment))
                .map_err(|e| format!("segment {}: {e}", entry.segment))?;
            let t = header
                .tensors
                .iter()
                .find(|t| t.name == tensor || t.name.ends_with(&format!(".{tensor}")))
                .ok_or_else(|| {
                    format!(
                        "no tensor of `{}` matches `{tensor}` — tensors: {}",
                        object.id,
                        header
                            .tensors
                            .iter()
                            .map(|t| t.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            let decoded = load_values(
                &store,
                &OperandRef {
                    object: object.id.clone(),
                    tensor: t.name.clone(),
                    dtype: t.dtype.clone(),
                    shape: t.shape.clone(),
                },
            )?;
            json!({
                "tensor": t.name,
                "dtype": t.dtype,
                "shape": t.shape,
                "values": decoded.iter().take(values).collect::<Vec<_>>(),
            })
        }
    };
    Ok(json!({
        "container": root.display().to_string(),
        "object": serde_json::to_value(object).map_err(|e| e.to_string())?,
        "directory": directory,
        "peek": peeked,
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

/// `vindex layers` — every layer's token-mixer programme, from the
/// plan. Three seconds of terminal that says why a hybrid model is an
/// interesting quantization subject.
pub fn layers_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let (component, plan) = primary_plan(root, &inspection)?;
    let mut rows: Vec<Value> = Vec::with_capacity(plan.layers.len());
    for l in &plan.layers {
        let ffn = match &l.ffn {
            Some(LayerFfn::Dense(_)) => "dense",
            Some(LayerFfn::Routed(_)) => "routed",
            Some(LayerFfn::Hybrid(_)) => "hybrid",
            // A mixer-only (Mamba2) layer: the mixer is the whole
            // block and no FFN exists to report.
            None => "absent",
        };
        let operator = mixer::declared_operator(component, l.layer)?;
        rows.push(json!({
            "layer": l.layer,
            "mixer": mixer::label(operator, mixer::has_output_gate(l)),
            "ffn": ffn,
        }));
    }
    Ok(json!({
        "container": root.display().to_string(),
        "component": component.id,
        "layers": rows,
    }))
}

/// `vindex precision --matrix` — bits per weight, per layer and per
/// semantic role, read from the representation each object would
/// execute (the compiled pack when one exists, the canonical bytes
/// otherwise). Programme-aware: layers are grouped by their token
/// mixer, each group carrying the columns its programme actually has,
/// and the model's other surfaces (embedding, head, towers) are
/// listed with their own derived bits. No architecture is forced into
/// another's schema.
pub fn precision_matrix_facts(root: &Path) -> Facts {
    let inspection = inspect_container(root, false).map_err(|e| format!("inspect: {e}"))?;
    let (component, plan) = primary_plan(root, &inspection)?;

    // Per object: the representation execution would bind — a compiled
    // pack over the canonical bytes — and its tensor table of
    // (byte length, weight count) per tensor name.
    type TensorSizes = std::collections::BTreeMap<String, (u64, u128)>;
    let mut tables: std::collections::BTreeMap<String, (String, TensorSizes)> =
        std::collections::BTreeMap::new();
    for object in &inspection.graph.objects {
        let canonical = object.representations.first().map(|r| r.encoding.clone());
        let mut chosen: Option<(
            &String,
            &larql_vindex::format::vindex3::index::RepresentationEntry,
        )> = None;
        for (id, e) in &inspection.index.representations {
            if e.object != object.id {
                continue;
            }
            let is_pack = Some(&e.encoding) != canonical.as_ref();
            match &chosen {
                None => chosen = Some((id, e)),
                Some((_, cur)) => {
                    let cur_is_pack = Some(&cur.encoding) != canonical.as_ref();
                    if is_pack && !cur_is_pack {
                        chosen = Some((id, e));
                    }
                }
            }
        }
        if let Some((_, entry)) = chosen {
            let (header, _) = read_segment_header(&root.join(&entry.segment))
                .map_err(|e| format!("segment {}: {e}", entry.segment))?;
            let table = header
                .tensors
                .into_iter()
                .map(|t| {
                    let weights: u128 = t.shape.iter().product::<usize>() as u128;
                    (t.name, (t.len, weights))
                })
                .collect();
            tables.insert(object.id.clone(), (entry.encoding.clone(), table));
        }
    }
    let bits_of = |op: &OperandRef| -> Option<f64> {
        let (_, table) = tables.get(&op.object)?;
        let (len, weights) = table.get(&op.tensor)?;
        if *weights == 0 {
            return None;
        }
        Some((*len as u128 * 8) as f64 / *weights as f64)
    };

    // Group layers by programme; each group's columns are what its
    // programme actually computes with.
    //
    // The empty answer means "no column", and it is deliberate for
    // every per-head or per-channel vector — log decay, timestep
    // bias, norm weights, the Mamba2 skip. A bits-per-weight matrix
    // over a `[Hv]` operand says nothing about a representation and
    // would push a programme's row past readable width; the tensors
    // are still reachable through `describe layer.N.mixer`.
    let short = |role: &str| -> &'static str {
        match role {
            // softmax
            "query" => "q",
            "key" => "k",
            "value" => "v",
            "output" => "o",
            "output gate" => "zgate",
            // gated deltanet
            "fused recurrent q|k|v" => "qkv",
            "decay projection" => "decay",
            "write-strength projection" => "write",
            "output-gate projection" => "zgate",
            "causal conv over q|k|v" => "conv",
            // kda — split where gated deltanet fuses
            "query projection" => "q",
            "key projection" => "k",
            "value projection" => "v",
            "causal conv over q" => "qconv",
            "causal conv over k" => "kconv",
            "causal conv over v" => "vconv",
            "decay gate down" => "fa",
            "decay gate up" => "fb",
            "output gate down" => "ga",
            "output gate up" => "gb",
            // mla — the compressed-kv set
            "compressed kv projection" => "kv_a",
            "kv decompression" => "kv_b",
            // mamba2
            "fused in-projection z|x|B|C|dt" => "in",
            "causal conv over x|B|C" => "conv",
            // shared
            "output projection" => "out",
            _ => "",
        }
    };
    let mut programmes: Vec<(String, Vec<String>, Vec<Value>)> = Vec::new();
    for layer in &plan.layers {
        let operator = mixer::declared_operator(component, layer.layer)?;
        let label = mixer::label(operator, mixer::has_output_gate(layer)).to_string();
        let mut cells = serde_json::Map::new();
        let mut roles: Vec<String> = vec!["gate".into(), "up".into(), "down".into()];
        if let Some(ffn) = layer.ffn.as_ref().and_then(|f| f.dense()) {
            if let Some(g) = &ffn.gate {
                if let Some(b) = bits_of(g) {
                    cells.insert("gate".into(), json!(b));
                }
            }
            if let Some(b) = bits_of(&ffn.up) {
                cells.insert("up".into(), json!(b));
            }
            if let Some(b) = bits_of(&ffn.down) {
                cells.insert("down".into(), json!(b));
            }
        }
        if let MixerOperands::Named(ops) = mixer::operands(operator, layer) {
            for (role, op) in ops {
                let col = short(role);
                if col.is_empty() {
                    continue;
                }
                roles.push(col.to_string());
                if let Some(b) = bits_of(&op) {
                    cells.insert(col.to_string(), json!(b));
                }
            }
        }
        let row = json!({ "layer": layer.layer, "bits": Value::Object(cells) });
        match programmes.iter_mut().find(|(l, _, _)| *l == label) {
            Some((_, _, rows)) => rows.push(row),
            None => programmes.push((label, roles, vec![row])),
        }
    }
    // The model's other surfaces: every object outside the layer plan,
    // at the bits its bound representation derives to.
    let planned: std::collections::BTreeSet<String> = plan
        .layers
        .iter()
        .flat_map(|l| {
            let mut ops: Vec<String> = match mixer::declared_operator(component, l.layer)
                .map(|op| mixer::operands(op, l))
            {
                Ok(MixerOperands::Named(named)) => {
                    named.into_iter().map(|(_, o)| o.object).collect()
                }
                _ => Vec::new(),
            };
            if let Some(ffn) = l.ffn.as_ref().and_then(|f| f.dense()) {
                if let Some(g) = &ffn.gate {
                    ops.push(g.object.clone());
                }
                ops.push(ffn.up.object.clone());
                ops.push(ffn.down.object.clone());
            }
            ops
        })
        .collect();
    let surfaces: Vec<Value> = tables
        .iter()
        .filter(|(object, _)| !planned.contains(*object))
        .map(|(object, (encoding, table))| {
            let (bits, weights) = table.values().fold((0u128, 0u128), |(b, w), (len, n)| {
                (b + *len as u128 * 8, w + n)
            });
            json!({
                "object": object,
                "representation": encoding,
                "bits_per_weight": if weights > 0 { bits as f64 / weights as f64 } else { 0.0 },
            })
        })
        .collect();
    Ok(json!({
        "container": root.display().to_string(),
        "component": component.id,
        "programmes": programmes.into_iter().map(|(label, roles, rows)| json!({
            "label": label,
            "layers": rows.len(),
            "roles": roles,
            "rows": rows,
        })).collect::<Vec<Value>>(),
        "surfaces": surfaces,
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

/// The primary component: the graph node that declares it, and the
/// operation plan built over it.
///
/// Both, because the two answer different questions and neither
/// substitutes for the other — the node declares what each layer's
/// token mixer IS, the plan binds the operands that mixer computes
/// with. See [`mixer`].
fn primary_plan<'a>(
    root: &Path,
    inspection: &'a SystemInspection,
) -> Result<(&'a Component, ComponentOpPlan), String> {
    let component = inspection
        .graph
        .components
        .first()
        .ok_or("the graph holds no components")?;
    let outcome =
        plan_component_ops(inspection, root, &component.id).map_err(|e| format!("plan: {e}"))?;
    let plan = outcome
        .plan
        .ok_or_else(|| format!("component `{}` has no plan", component.id))?;
    Ok((component, plan))
}

/// A semantic address, resolved through the container's own operation
/// plan — the graph's judgement of what each tensor IS, never a
/// filename convention. `layer.N.ffn.{gate|up|down}` (mlp accepted)
/// and `layer.N.attention.{q|k|v|o}` (attn accepted). Returns the
/// role label with the operand.
fn resolve_semantic(
    root: &Path,
    inspection: &SystemInspection,
    address: &str,
) -> Option<Result<(String, OperandRef), String>> {
    let parts: Vec<&str> = address.split('.').collect();
    let [lit, layer, family, role] = parts.as_slice() else {
        return None;
    };
    if *lit != "layer" {
        return None;
    }
    let n: usize = layer.parse().ok()?;
    let family = match *family {
        "ffn" | "mlp" => "ffn",
        "attention" | "attn" => "attention",
        _ => return None,
    };
    let component = inspection.graph.components.first()?.id.clone();
    let go = || -> Result<(String, OperandRef), String> {
        let outcome =
            plan_component_ops(inspection, root, &component).map_err(|e| format!("plan: {e}"))?;
        let plan = outcome
            .plan
            .ok_or_else(|| format!("component `{component}` has no plan"))?;
        let layer_plan = plan
            .layers
            .get(n)
            .ok_or_else(|| layer_range(&format!("layer {n}"), plan.layers.len()))?;
        match family {
            "ffn" => {
                let ffn = layer_plan
                    .ffn
                    .as_ref()
                    .and_then(|f| f.dense())
                    .ok_or_else(|| {
                        let kind = match &layer_plan.ffn {
                            Some(LayerFfn::Routed(_)) => "a routed (mixture-of-experts) FFN",
                            Some(LayerFfn::Hybrid(_)) => "a hybrid FFN",
                            None => "no FFN at all (a mixer-only layer)",
                            Some(LayerFfn::Dense(_)) => unreachable!(),
                        };
                        format!(
                        "layer {n} carries {kind} — per-expert addressing is not yet a CLI surface"
                    )
                    })?;
                let (label, op) = match *role {
                    "gate" => (
                        "FFN GATE PROJECTION",
                        ffn.gate.as_ref().ok_or_else(|| {
                            format!("layer {n}'s FFN is ungated — no gate operand")
                        })?,
                    ),
                    "up" => ("FFN UP PROJECTION", &ffn.up),
                    "down" => ("FFN DOWN PROJECTION", &ffn.down),
                    other => return Err(format!("unknown ffn role `{other}` — gate, up, down")),
                };
                Ok((label.to_string(), op.clone()))
            }
            "attention" => {
                let attn = layer_plan.attention.softmax().ok_or_else(|| {
                    let mixer = inspection
                        .graph
                        .components
                        .first()
                        .and_then(|c| mixer::declared_operator(c, n).ok())
                        .map(|op| mixer::label(op, false))
                        .unwrap_or("a mixer this container does not name");
                    format!(
                        "layer {n} does not attend by softmax — its token mixer is {mixer}, \
                         so its projections are that operator's, not q/k/v/o; \
                         try `describe layer.{n}.mixer`"
                    )
                })?;
                let (label, op) = match *role {
                    "q" => ("ATTENTION QUERY PROJECTION", &attn.q),
                    "k" => ("ATTENTION KEY PROJECTION", &attn.k),
                    "v" => ("ATTENTION VALUE PROJECTION", &attn.v),
                    "o" => ("ATTENTION OUTPUT PROJECTION", &attn.o),
                    other => return Err(format!("unknown attention role `{other}` — q, k, v, o")),
                };
                Ok((label.to_string(), op.clone()))
            }
            _ => unreachable!(),
        }
    };
    Some(go())
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
    let mut semantic_tensor: Option<String> = None;
    let object = if let Some(resolved) = resolve_semantic(root, &inspection, address) {
        let (_, op) = resolved?;
        semantic_tensor = Some(op.tensor);
        op.object
    } else {
        find_object(&inspection, address)?.id.clone()
    };
    let tensor_filter = semantic_tensor.as_deref().or(tensor_filter);
    let (encoding_a, encoding_b) = (encoding_a.to_uppercase(), encoding_b.to_uppercase());
    let (store_a, tensors_a) = open_side(root, &inspection, &object, &encoding_a)?;
    let (store_b, tensors_b) = open_side(root, &inspection, &object, &encoding_b)?;
    let dtypes_b: std::collections::BTreeMap<&str, (&str, &Vec<usize>)> = tensors_b
        .iter()
        .map(|(n, d, s)| (n.as_str(), (d.as_str(), s)))
        .collect();

    // A semantic address names ONE tensor: the diff scopes to it, so
    // the result is that tensor's answer rather than the object's.
    let tensors_a: Vec<TensorEntry> = match &semantic_tensor {
        Some(t) => tensors_a.into_iter().filter(|(n, _, _)| n == t).collect(),
        None => tensors_a,
    };
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
/// `vindex export` — the container's selected representation, compiled
/// to a qwen35 GGUF and verified through the independent reader before
/// the function returns. Every count in the result was observed from
/// the finished file, none predicted.
pub fn export_facts(root: &Path, out: &Path) -> Facts {
    let report = larql_vindex::format::vindex3::gguf::export::export_qwen35(root, out)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "container": root.display().to_string(),
        "out": report.out.display().to_string(),
        "bytes": report.bytes,
        "selected": report.selected_encoding,
        "walk": {
            "source_tensors": report.ledger.source_total,
            "accounted": report.ledger.accounted,
            "geometry_reconciled": report.ledger.geometry_reconciled,
            "scale_siblings": report.ledger.generated_scale_tensors,
        },
        "emitted": {
            "tensors": report.emit.tensors,
            "scale_siblings": report.emit.scale_siblings,
            "metadata_keys": report.emit.metadata_keys,
        },
        "vocab": {
            "tokens": report.vocab_tokens,
            "padded": report.vocab_padded,
            "merges": report.vocab_merges,
        },
        "verified": {
            "tensors": report.verify.tensors,
            "nvfp4_tensors": report.verify.nvfp4_tensors,
            "scale_siblings": report.verify.scale_siblings,
            "metadata_keys": report.verify.metadata_keys,
        },
    }))
}

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

// ── Ingest: bringing a model in ──────────────────────────────────────────
//
// Every other verb reads a container that already exists. These two make
// one, and they are the reason `vindex` is a tool you can start with
// rather than a tool you reach for afterwards.
//
// Both resolve their arguments through
// `larql_vindex::format::vindex3::artifact`, which is also what
// `larql vindex3` uses — one authority on what an artifact argument means,
// so the two binaries cannot disagree about a model's identity or produce
// different containers from the same input.

/// The admission verdict for an artifact, without moving its weights.
///
/// A bring-up instrument, not the ordinary path: it answers "what does
/// VINDEX still need to understand about this model?" from configuration
/// and safetensors headers alone. On GLM-5.3-Flash that is ~39 MB of
/// staging against a 328 GB checkpoint.
pub fn plan_facts(artifacts: &[PathBuf]) -> Facts {
    let resolved = artifact::resolve_all(artifacts).map_err(|e| e.to_string())?;
    let staging: Vec<Value> = resolved.iter().filter_map(staging_value).collect();
    // The verdict names its subject: the argument as given and, for a
    // repo, the commit the facts were read at.
    let sources: Vec<ArtifactSource> = artifacts
        .iter()
        .zip(&resolved)
        .map(|(spec, a)| ArtifactSource {
            path: spec.display().to_string(),
            revision: a.commit().map(str::to_string),
            unpinned_revision: a.unpinned_revision().map(str::to_string),
        })
        .collect();
    let named: Vec<_> = resolved
        .into_iter()
        .map(|a| (a.name, a.inventory))
        .collect();
    let plan = plan_system_with_sources(&named, &sources).map_err(|e| e.to_string())?;
    let mut value = serde_json::to_value(&plan).map_err(|e| e.to_string())?;
    if !staging.is_empty() {
        value["staging"] = Value::Array(staging);
    }
    Ok(value)
}

/// Encode artifacts into a container.
///
/// An `hf://` argument is read over byte ranges: the canonical checkpoint
/// never needs to exist as a complete local file.
pub fn encode_facts(artifacts: &[PathBuf], output: &Path, text_only: bool) -> Facts {
    let resolved = artifact::resolve_all(artifacts).map_err(|e| e.to_string())?;
    let staging: Vec<Value> = resolved.iter().filter_map(staging_value).collect();
    let pinned: Vec<Value> = resolved
        .iter()
        .filter_map(|a| {
            Some(json!({
                "artifact": a.name,
                "commit": a.commit()?,
            }))
        })
        .collect();
    let unpinned: Vec<Value> = resolved
        .iter()
        .filter_map(|a| {
            Some(json!({
                "artifact": a.name,
                "revision": a.unpinned_revision()?,
            }))
        })
        .collect();

    let capability = text_only.then_some(Capability::TextGeneration);
    let outcome =
        artifact::encode_from_specs(resolved, output, capability).map_err(|e| e.to_string())?;

    Ok(json!({
        "container": outcome.container.display().to_string(),
        "representations": outcome.representations,
        "payload_bytes": outcome.total_payload_bytes,
        "payload": artifact::size(outcome.total_payload_bytes),
        "capabilities": outcome.capabilities,
        "staging": staging,
        "pinned": pinned,
        "unpinned": unpinned,
        "transfers": outcome.transfers.iter().map(|t| json!({
            "artifact": t.name,
            "tensors": t.tensors,
            "fetched_bytes": t.fetched,
            "fetched": artifact::size(t.fetched),
            "declared": artifact::size(t.declared),
            // The ratio IS the claim. Near 1.0 means the plan bound every
            // tensor; well under means the container carries less than the
            // checkpoint holds.
            "fraction": if t.declared == 0 { 0.0 } else { t.fetched as f64 / t.declared as f64 },
        })).collect::<Vec<_>>(),
    }))
}

/// One artifact's staging figures, when it was staged from a repo.
fn staging_value(a: &artifact::ResolvedArtifact) -> Option<Value> {
    let report = a.staging()?;
    Some(json!({
        "artifact": a.name,
        "commit": a.commit(),
        "shards": report.shards,
        "staged": artifact::size(report.staged_bytes()),
        "headers": artifact::size(report.header_bytes),
        "metadata": artifact::size(report.metadata_bytes),
        "stands_in_for": report.payload_bytes.as_ref().ok().map(|b| artifact::size(*b)),
        // Stated only when the index disagrees with its own headers, so
        // the difference reads as a fact about the checkpoint rather than
        // a units bug in the report.
        "index_declares": report
            .declared_total
            .filter(|d| report.payload_bytes.as_ref().is_ok_and(|p| d != p))
            .map(artifact::size),
    }))
}
