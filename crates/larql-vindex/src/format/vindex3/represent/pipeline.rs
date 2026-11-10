//! The compile pipeline behind `compile` / `compile_with`.

use super::super::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use super::super::encode::REPRESENTATION_ID_SEP;
use super::super::encode::{SEGMENTS_DIR, SEGMENT_BIN_EXT};
use super::super::graph::object::{Fidelity, Representation};
use super::super::index::{ContainerAuthority, RepresentationEntry, Vindex3Index};
use super::super::inspect::inspect_container;
use super::super::opplan::exec::operands::{OperandSource, OperandStore};
use super::super::opplan::OperandRef;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use codec::EncoderRegistry;
use map::PrecisionMap;
use nvfp4_pack::{CodecIdentity, EncoderRecipe, PackLayout, DTYPE_NVFP4};
use policy::Role;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

pub(super) fn compile_inner(
    src: &Path,
    out: &Path,
    spec: &RepresentSpec,
    encoders: &EncoderRegistry,
    weights: Option<&InputWeights>,
    recipes: Option<&recipe::Completed>,
) -> Result<RepresentReport, VindexError> {
    let target = if spec.encoding == DTYPE_NVFP4 {
        Target::Nvfp4
    } else if let Some(k) = kquant::lookup(&spec.encoding) {
        Target::KQuant(k)
    } else if let Some(encoder) = encoders.by_label(&spec.encoding) {
        Target::Encoder(encoder)
    } else {
        let registered = encoders.labels();
        return Err(VindexError::Parse(format!(
            "encoding `{}` has no representation compiler; known: {DTYPE_NVFP4}, {}{}",
            spec.encoding,
            kquant::compilable_names(),
            if registered.is_empty() {
                String::new()
            } else {
                format!("; registered encoders: {}", registered.join(", "))
            }
        )));
    };

    if weights.is_some() && !matches!(target, Target::Encoder(_)) {
        return Err(VindexError::Parse(format!(
            "input-feature weights were given, but `{}` is a shipped compiler, which \
             takes none; only a registered encoder encodes under weights",
            spec.encoding
        )));
    }

    let raw_index = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let mut index: Vindex3Index = serde_json::from_str(&raw_index)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    let inspection = inspect_container(src, false)?;
    // Which objects belong to the primary text model. A perception tower's
    // tensors are named exactly like a decoder's, so the component's
    // declared role is the only thing that separates them — see
    // `policy::classify_in`.
    let primary_text = primary_text_objects(&inspection);
    // The plan's own operand bindings, which outrank tensor spellings.
    // Best-effort: a component whose plan does not build contributes
    // nothing here and its tensors fall back to name classification.
    let declared_roles = plan_roles::plan_roles(src, &inspection);
    let map =
        PrecisionMap::from_policy(spec.map_name(), &spec.encoding, &spec.roles, &spec.protect);
    // Refuse a map with a dead rule before anything is encoded. The
    // surface is every tensor of every object this spec wants, including
    // objects a previous run already compiled: the map is recorded for
    // the whole candidate, so a resumed run must not refuse a protection
    // whose tensors happen to be done.
    let mut surface: BTreeSet<(Role, String)> = BTreeSet::new();
    for entry in index.representations.values() {
        if !spec.wants(&entry.object) {
            continue;
        }
        let (header, _) = read_segment_header(&src.join(&entry.segment))?;
        for t in &header.tensors {
            let role = tensor_role(&declared_roles, &primary_text, &entry.object, t);
            surface.insert((role, t.name.clone()));
        }
    }
    map.check_against(surface.iter().map(|(r, n)| (*r, n.as_str())))
        .map_err(|refusal| VindexError::Parse(refusal.to_string()))?;
    let mut candidate = candidate_authority::producer::CompilationAuthority::new(
        src,
        &index,
        map,
        &primary_text,
        &declared_roles,
    )?;
    let store = OperandStore::open(src, &inspection)?;
    let source = OperandSource::from(&store);

    std::fs::create_dir_all(out)?;

    let mut report = RepresentReport {
        compiled_objects: Vec::new(),
        linked_segments: 0,
        preserved_objects: Vec::new(),
    };

    let existing: Vec<(String, RepresentationEntry)> = index
        .representations
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let mut added: Vec<(String, RepresentationEntry)> = Vec::new();
    let mut added_segment_keys: Vec<String> = Vec::new();
    let mut compiled_object_ids: BTreeSet<String> = BTreeSet::new();

    for (rep_id, entry) in &existing {
        if !spec.wants(&entry.object) {
            continue;
        }
        // One compiled pack per object. An object already carrying the
        // target encoding is left alone rather than re-encoded — a second
        // pass must not quantise a quantised pack.
        let target_id = format!("{}{REPRESENTATION_ID_SEP}{}", entry.object, spec.encoding);
        if index.representations.contains_key(&target_id)
            || compiled_object_ids.contains(&entry.object)
        {
            continue;
        }

        let src_segment = src.join(&entry.segment);
        let (header, payload_start) = read_segment_header(&src_segment)?;

        // Plan first: which tensors the encoding applies to, and how long
        // each becomes. Nothing is written until every length is known,
        // because the segment writer needs the table before the payload.
        let mut planned: Vec<PlannedTensor> = Vec::new();
        let mut layouts: Vec<(String, Option<TensorEncoding>)> = Vec::new();
        let mut compiled_tensors = 0usize;
        let mut carried_tensors = 0usize;
        let mut source_bytes = 0u64;
        let mut preserved_roles: BTreeMap<Role, usize> = BTreeMap::new();
        // A registered encoder's bytes, produced while planning: its
        // length may depend on the values, and the table needs every
        // length before the payload. Held for one object at a time.
        let mut encoded_bytes: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut weighted_tensors = 0usize;

        for t in &header.tensors {
            // Role first, shape second. A tensor the policy preserves is
            // carried whatever its shape; a tensor the policy admits is
            // still refused by the layout if its `k` cannot be grouped.
            // A shape that cannot hold the encoding is still refused
            // below, whatever the role says.
            let role = tensor_role(&declared_roles, &primary_text, &entry.object, t);
            // Role says the encoding applies; protection says whether to
            // spend it here. A protected tensor is carried, and counted as
            // preserved under its own role so the report says what the map
            // actually held back.
            let eligible = spec.roles.compiles(role) && !spec.protect.protects(&t.name);
            // The role decided whether to spend the encoding here; the
            // shape decides only whether the encoding FITS. Asking the
            // target keeps that second question with the format that
            // owns it — NVFP4 needs a 2-D matrix, a K-quant needs a row
            // length that is a whole number of blocks, and neither rule
            // belongs to the other.
            let encoded = match (eligible, target) {
                (false, _) => None,
                (true, Target::Nvfp4) => PackLayout::derive(&t.shape, &t.name)
                    .ok()
                    .map(TensorEncoding::Nvfp4),
                (true, Target::KQuant(k)) => k
                    .plan(&t.shape, &t.name)
                    .ok()
                    .map(|len| TensorEncoding::KQuant(k, len)),
                (true, Target::Encoder(encoder)) => {
                    let w = weights.and_then(|w| {
                        w.by_tensor
                            .get(&(entry.object.clone(), t.name.clone()))
                            .map(Vec::as_slice)
                    });
                    encode_with(encoder, &source, &entry.object, t, w)?.map(|bytes| {
                        weighted_tensors += usize::from(w.is_some());
                        let len = bytes.len();
                        encoded_bytes.insert(t.name.clone(), bytes);
                        TensorEncoding::Encoder(len)
                    })
                }
            };
            candidate.decided(
                &entry.object,
                &t.name,
                match encoded {
                    Some(_) => state::ResolvedEncoding::Compiled(spec.encoding.clone()),
                    None if eligible => state::ResolvedEncoding::LayoutRefused {
                        encoding: spec.encoding.clone(),
                    },
                    None => state::ResolvedEncoding::Source,
                },
            );
            match encoded {
                Some(encoding) => {
                    source_bytes += t.len;
                    compiled_tensors += 1;
                    *preserved_roles.entry(role).or_insert(0) += 0;
                    planned.push(PlannedTensor {
                        relative_name: t.name.clone(),
                        source_name: t.name.clone(),
                        dtype: spec.encoding.clone(),
                        shape: t.shape.clone(),
                        len: encoding.len() as u64,
                    });
                    layouts.push((t.name.clone(), Some(encoding)));
                }
                None => {
                    // Either the policy preserves this role, or the shape
                    // cannot hold the encoding. Carried verbatim so the
                    // pack is a complete object, not a partial one its
                    // consumers would have to patch from elsewhere.
                    carried_tensors += 1;
                    *preserved_roles.entry(role).or_insert(0) += 1;
                    planned.push(PlannedTensor {
                        relative_name: t.name.clone(),
                        source_name: t.name.clone(),
                        dtype: t.dtype.clone(),
                        shape: t.shape.clone(),
                        len: t.len,
                    });
                    layouts.push((t.name.clone(), None));
                }
            }
        }

        if compiled_tensors == 0 {
            // Nothing eligible here; the object keeps its canonical
            // representation alone rather than gaining an identical copy
            // under a misleading name. Recorded, not silently skipped —
            // "the embedding is BF16 because the policy protects it" and
            // "the embedding is BF16 because nobody looked" are different
            // facts and a report that cannot tell them apart is useless.
            report.preserved_objects.push(PreservedObject {
                object: entry.object.clone(),
                encoding: entry.encoding.clone(),
                bytes: entry.payload_bytes,
                roles: preserved_roles
                    .into_iter()
                    .filter(|(_, n)| *n > 0)
                    .collect(),
            });
            continue;
        }

        // Same naming convention the encoder uses, so a pack is not a
        // second kind of file living somewhere else: `segments/<key>.bin`,
        // with the key registered in `index.segments` below.
        let segment_key = format!("{SEGMENTS_DIR}/{target_id}");
        let segment_rel = format!("{segment_key}.{SEGMENT_BIN_EXT}");
        // `target_id` is an object id from the source container.
        let out_segment = crate::format::vindex3::container_path::prepare_write(out, &segment_rel)?;

        let mut src_file = std::fs::File::open(&src_segment)?;
        let written = write_segment(&out_segment, &target_id, planned, |name, w, tap| {
            let tensor = header
                .tensors
                .iter()
                .find(|t| t.name == name)
                .expect("planned from the same header");
            let layout = layouts
                .iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, l)| *l);

            match layout {
                Some(TensorEncoding::Encoder(_)) => {
                    let bytes = encoded_bytes.remove(name).expect("encoded while planning");
                    w.write_all(&bytes)?;
                    tap(&bytes);
                    Ok(bytes.len() as u64)
                }
                Some(TensorEncoding::KQuant(k, planned_len)) => {
                    // Same shape as the NVFP4 arm below and for the same
                    // reason: load through `OperandSource::load`, then run
                    // exactly the encoder a transient arm would run, so
                    // persisted bytes are the transient ones.
                    let values = source.load(&OperandRef {
                        object: entry.object.clone(),
                        tensor: tensor.name.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                    })?;
                    // Row length, not a flattened count: ggml searches
                    // scales within a row, so the framing has to survive
                    // to the encoder or the bytes are not what a
                    // llama.cpp artifact of this tensor would hold.
                    let row_len = *tensor.shape.last().ok_or_else(|| {
                        VindexError::Parse(format!(
                            "tensor `{}`: a scalar has no row to block along",
                            tensor.name
                        ))
                    })?;
                    let bytes = encode_kquant(k, &values, row_len, &tensor.name)?;
                    if bytes.len() != planned_len {
                        return Err(VindexError::Parse(format!(
                            "tensor `{}`: encoded {} bytes, the plan reserved {planned_len} \
                             — the segment table would not describe its payload",
                            tensor.name,
                            bytes.len()
                        )));
                    }
                    w.write_all(&bytes)?;
                    tap(&bytes);
                    Ok(bytes.len() as u64)
                }
                Some(TensorEncoding::Nvfp4(layout)) => {
                    let bytes = if let Some(completed) = recipes {
                        let (region, record) = completed
                            .tensors
                            .get(&(entry.object.clone(), tensor.name.clone()))
                            .ok_or_else(|| {
                                VindexError::Parse(format!(
                                    "missing completed recipe for {}",
                                    tensor.name
                                ))
                            })?;
                        let bytes = region.bytes();
                        if bytes.len() != layout.total_len
                            || compile::hash_bytes(bytes) != record.payload_sha256
                        {
                            return Err(VindexError::Parse(
                                "completed recipe payload changed".into(),
                            ));
                        }
                        bytes.to_vec()
                    } else {
                        let values = source.load(&OperandRef {
                            object: entry.object.clone(),
                            tensor: tensor.name.clone(),
                            dtype: tensor.dtype.clone(),
                            shape: tensor.shape.clone(),
                        })?;
                        recipe::nearest(&values, layout, &tensor.name)?
                    };
                    w.write_all(&bytes)?;
                    tap(&bytes);
                    Ok(bytes.len() as u64)
                }
                None => {
                    src_file.seek(SeekFrom::Start(payload_start + tensor.offset))?;
                    let mut remaining = tensor.len;
                    let mut buf = vec![0u8; 1 << 20];
                    while remaining > 0 {
                        let take = remaining.min(buf.len() as u64) as usize;
                        src_file.read_exact(&mut buf[..take])?;
                        w.write_all(&buf[..take])?;
                        tap(&buf[..take]);
                        remaining -= take as u64;
                    }
                    Ok(tensor.len)
                }
            }
        })?;

        candidate.written(src, out, entry, &segment_rel)?;

        report.compiled_objects.push(CompiledObject {
            object: entry.object.clone(),
            representation_id: target_id.clone(),
            compiled_tensors,
            carried_tensors,
            source_bytes,
            compiled_bytes: written.payload_bytes,
            preserved: preserved_roles
                .into_iter()
                .filter(|(_, n)| *n > 0)
                .collect(),
            weighted_tensors,
        });
        compiled_object_ids.insert(entry.object.clone());

        added_segment_keys.push(segment_key);
        added.push((
            target_id,
            RepresentationEntry {
                object: entry.object.clone(),
                encoding: spec.encoding.clone(),
                segment: segment_rel,
                tensor_count: written.tensor_count,
                payload_bytes: written.payload_bytes,
                payload_sha256: written.payload_sha256,
                segment_sha256: written.segment_sha256,
                compiled_from: Some(rep_id.clone()),
                // The ABI these bytes were produced against, so a later
                // build refuses them rather than decoding under new rules.
                codec: Some(match target {
                    Target::Nvfp4 => CodecIdentity::nvfp4_v1(),
                    Target::KQuant(k) => k.codec_identity(),
                    Target::Encoder(encoder) => encoder.identity(),
                }),
                // Ties the pack to the exact source bytes even after it is
                // copied out of the container that holds them — which is
                // precisely what a deployment artifact does.
                source_representation_digest: Some(entry.payload_sha256.clone()),
                encoder: Some(match target {
                    Target::Nvfp4 => recipes
                        .map(|r| r.encoder(&entry.object))
                        .unwrap_or_else(EncoderRecipe::current),
                    Target::KQuant(k) => kquant_encoder_recipe(&k),
                    Target::Encoder(encoder) => match weights.filter(|_| weighted_tensors > 0) {
                        Some(w) => EncoderRecipe::codec_weighted(&encoder.identity(), &w.digest),
                        None => EncoderRecipe::codec(&encoder.identity()),
                    },
                }),
            },
        ));
    }

    // Source bytes travel unless this is a deployment image and their
    // object has a compiled replacement. A protected surface — the BF16
    // embedding, the norms — always travels: the image has to execute.
    for (rep_id, entry) in &existing {
        let superseded = spec.deployment && compiled_object_ids.contains(&entry.object);
        if superseded {
            continue;
        }
        let from = src.join(&entry.segment);
        let to = out.join(&entry.segment);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to)?;
        }
        report.linked_segments += 1;
        let _ = rep_id;
    }
    if spec.deployment {
        index
            .representations
            .retain(|_, e| !compiled_object_ids.contains(&e.object));
        index.segments.retain(|key, _| {
            let seg = format!("{key}.{SEGMENT_BIN_EXT}");
            !existing
                .iter()
                .any(|(_, e)| e.segment == seg && compiled_object_ids.contains(&e.object))
        });
        index.authority = ContainerAuthority::Derived;
        index.derived_from_model = Some(index.model.clone());
    }

    if added.is_empty() {
        return Err(VindexError::Parse(format!(
            "no tensor in this container is eligible for `{}` under the \
             active role policy; nothing was compiled and no container \
             was written",
            spec.encoding
        )));
    }

    for (id, entry) in added {
        index.representations.insert(id, entry);
    }
    // A segment a reader cannot resolve by key is a segment it refuses:
    // `Vindex3Container::segment` rejects anything `index.segments` does
    // not declare.
    for key in added_segment_keys {
        index.segments.insert(key, 1);
    }

    // The program that produced these packs, recorded as authority rather
    // than left to be inferred from the bytes it produced.
    index.precision_map = Some(PrecisionMap::from_policy(
        spec.map_name(),
        &spec.encoding,
        &spec.roles,
        &spec.protect,
    ));

    // The graph learns the object now has a second materialisation, marked
    // approximate: a profile may select it, and nothing may mistake it for
    // the bit-authoritative source.
    let graph_path = out.join(SYSTEM_GRAPH_JSON);
    let src_graph = src.join(SYSTEM_GRAPH_JSON);
    if src_graph.exists() {
        let graph_raw = std::fs::read_to_string(&src_graph)?;
        let mut graph: super::super::graph::SystemGraph = serde_json::from_str(&graph_raw)
            .map_err(|e| VindexError::Parse(format!("parse {SYSTEM_GRAPH_JSON}: {e}")))?;
        for object in &mut graph.objects {
            if !compiled_object_ids.contains(&object.id) {
                continue;
            }
            if !object
                .representations
                .iter()
                .any(|r| r.encoding == spec.encoding)
            {
                object.representations.push(Representation {
                    encoding: spec.encoding.clone(),
                    fidelity: Fidelity::Approximate,
                });
            }
            if spec.deployment {
                // The source bytes are not in this image, so declaring a
                // representation for them would point every reader at a
                // segment that is not there.
                object
                    .representations
                    .retain(|r| r.encoding == spec.encoding);
            }
        }
        let serialised = serde_json::to_string_pretty(&graph)
            .map_err(|e| VindexError::Parse(format!("serialise {SYSTEM_GRAPH_JSON}: {e}")))?;
        std::fs::write(&graph_path, serialised)?;
    }

    // The capability snapshot travels with the representation: a compiled
    // container that kept only tokenizer.json could tokenise and not
    // chat — no eos, no template — which reads as a broken model rather
    // than a missing file.
    for aux in [
        "moe_manifest.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        let from = src.join(aux);
        if from.exists() {
            std::fs::copy(&from, out.join(aux))?;
        }
    }

    // Index last: a crash mid-compile leaves a directory that is not yet a
    // container, matching the encode writer's ordering contract.
    let serialised = serde_json::to_string_pretty(&index)
        .map_err(|e| VindexError::Parse(format!("serialise {INDEX_JSON}: {e}")))?;
    std::fs::write(out.join(INDEX_JSON), serialised)?;
    if let Some(completed) = recipes {
        candidate.derivation(completed.derivation.clone());
    }
    let completed_candidate = candidate.finish(out)?;
    compiler::write_index_atomically(
        &completed_candidate,
        &out.join(candidate_authority::CANDIDATE_INDEX_FILE),
    )?;

    Ok(report)
}
