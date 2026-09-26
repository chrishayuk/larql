//! Representation compilation: add a compiled physical encoding of an
//! object to a container, without changing what the model *is*.
//!
//! ```text
//! source tensors → representation compiler → persisted representation pack
//!                                                      ↓
//!                                          a profile selects the pack
//! ```
//!
//! This is a third verb alongside the two that already exist, and the
//! distinction is the point:
//!
//! - **COMPILE** materialises overlay *meaning* into rewritten segments.
//! - **COMPACT** reorganises bytes while preserving meaning exactly
//!   (`SemanticDiff(input, output) == ∅`).
//! - **REPRESENT** adds a *lossy alternative encoding* beside the
//!   canonical bytes. It preserves neither byte-equality (the pack is new
//!   bytes) nor exact semantics (4-bit is an approximation), so it cannot
//!   hide behind either gate. Nearest packs must equal the transient nearest
//!   quantizer byte-for-byte. Calibrated recipes instead bind completed bytes
//!   to their derivation; consuming those bytes requires only the codec ABI.
//!
//! The default path still loads through [`OperandSource::load`] and calls the
//! transient nearest quantizer. [`compile_representation_recipe`] explicitly
//! selects nearest or calibrated GPTQ without changing that default.
//!
//! ## What it is for
//!
//! A 30B BF16 source is tens of gigabytes on disk and pays the
//! quantisation cost on every cold load. Neither is a property of the
//! model; both are properties of having only one stored representation.
//! Compiling one changes the artifact, not the semantics — and at K3
//! scale, where sparse expert fetches make bytes-per-expert an input to
//! the inference algorithm rather than a storage detail, it stops being a
//! convenience.
//!
//! ## What it does not do
//!
//! It does not replace the canonical representation. The source bytes stay
//! in the container and stay canonical; the compiled pack is added beside
//! them with [`Fidelity::Approximate`]. A profile then selects between
//! representations that *exist* — the rule
//! [`super::variants`] already states, and the reason a compiler is needed
//! at all: a profile cannot turn one encoding's bytes into another's.

pub mod activation;
pub mod actuate;
pub mod arena;
pub mod assessment;
pub mod bank;
pub mod byte_ledger;
pub mod calibration;
pub mod candidate_authority;
pub mod codec;
pub mod compile;
pub mod compiler;
pub mod constraint;
pub mod decision;
pub mod derivation;
pub mod diagnostic;
pub mod execution_cost;
pub mod experiment;
pub mod experiment_identity;
pub mod gptq;
pub mod ingest;
pub mod kda_candidate;
pub mod kquant;
pub mod map;
pub mod map_check;
pub mod measure;
#[cfg(test)]
mod measure_tests;
pub mod measurement;
pub mod nvfp4_pack;
pub mod observation_stream;
pub mod participation;
pub mod physical;
pub mod plan_roles;
#[cfg(test)]
mod plan_roles_tests;
pub mod policy;
pub mod promotion;
pub mod quality;
pub mod recipe;
pub use recipe::{compile_representation_recipe, Nvfp4Recipe};
#[cfg(feature = "reference-encoder")]
pub mod reference_encoder;
pub mod resampling;
pub mod search_evidence;
pub mod selection;
pub mod source_bank;
pub mod source_identity;
pub mod state;
pub mod statistic;
pub mod stream_replay;
pub mod token_bank;
pub mod view;

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use super::encode::REPRESENTATION_ID_SEP;
use super::encode::{SEGMENTS_DIR, SEGMENT_BIN_EXT};
use super::graph::object::{Fidelity, Representation};
use super::index::{ContainerAuthority, RepresentationEntry, Vindex3Index};
use super::inspect::inspect_container;
use super::opplan::exec::operands::{OperandSource, OperandStore};
#[cfg(test)]
use super::opplan::exec::weights::LoadedWeight;
use super::opplan::OperandRef;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use codec::{CodecError, EncoderRegistry, RepresentationEncoder};
use map::PrecisionMap;
use nvfp4_pack::{CodecIdentity, EncoderRecipe, PackLayout, DTYPE_NVFP4};
use policy::{classify_in, Protections, Role, RolePolicy};
/// Filename of the system graph, carried beside the index.
const SYSTEM_GRAPH_JSON: &str = "system_graph.json";

/// What one representation compilation produced.
#[derive(Debug, Clone)]
pub struct RepresentReport {
    /// Objects that gained a compiled representation.
    pub compiled_objects: Vec<CompiledObject>,
    /// Segments carried across untouched.
    pub linked_segments: usize,
    /// Objects the policy protected entirely — no tensor in them was
    /// eligible, so they have no compiled pack and execute at source
    /// precision. Naming them is the difference between a policy and a
    /// silent omission.
    pub preserved_objects: Vec<PreservedObject>,
}

/// An object left wholly at source precision, and why.
#[derive(Debug, Clone)]
pub struct PreservedObject {
    pub object: String,
    pub encoding: String,
    pub bytes: u64,
    /// The roles its tensors classified as.
    pub roles: BTreeMap<Role, usize>,
}

/// One object's compiled pack.
#[derive(Debug, Clone)]
pub struct CompiledObject {
    pub object: String,
    pub representation_id: String,
    /// Tensors re-encoded into the pack.
    pub compiled_tensors: usize,
    /// Tensors copied verbatim because they are not matrices the encoding
    /// applies to (norms, biases, 1-D vectors).
    pub carried_tensors: usize,
    pub source_bytes: u64,
    pub compiled_bytes: u64,
    /// Roles left at source precision, and how many tensors each covers —
    /// what a conservative policy actually protected.
    pub preserved: BTreeMap<Role, usize>,
    /// Of `compiled_tensors`, how many a registered encoder encoded under
    /// input-feature weights. The rest were encoded unweighted, because no
    /// weights were given for them.
    pub weighted_tensors: usize,
}

impl CompiledObject {
    /// Compression of the compiled pack against the bytes it was compiled
    /// from. `0.0` when the pack is empty.
    pub fn compression(&self) -> f64 {
        if self.compiled_bytes == 0 {
            return 0.0;
        }
        self.source_bytes as f64 / self.compiled_bytes as f64
    }
}

/// Which objects to compile, and into what.
#[derive(Debug, Clone)]
pub struct RepresentSpec {
    /// Target encoding: [`DTYPE_NVFP4`], any name [`kquant::lookup`]
    /// recognises, or the label of an encoder handed to
    /// [`compile_representation_with`]. The `Target` dispatch below is the
    /// single place a further encoding is added.
    pub encoding: String,
    /// Objects to compile. Empty means every object carrying a tensor the
    /// policy admits.
    pub objects: Vec<String>,
    /// Which tensor roles are eligible. Conservative by default — see
    /// [`policy`] for what that means and why.
    pub roles: RolePolicy,
    /// Write a deployment image rather than an archival container.
    ///
    /// A deployment image drops the source bytes of every object it
    /// compiled, keeping the compiled pack plus every surface the precision
    /// policy protected — the BF16 embedding and norms still have to
    /// travel, or the image would not execute. Nothing is destroyed: the
    /// canonical container is untouched and the image names the digests it
    /// derives from.
    ///
    /// This is a different artifact, not a smaller one. It is executable
    /// and not re-compilable, and [`ContainerAuthority::Derived`] says so.
    pub deployment: bool,
    /// Tensors held at source precision despite an eligible role — the
    /// material a precision map is made of. Empty is R0.
    pub protect: Protections,
}

impl RepresentSpec {
    pub fn nvfp4() -> Self {
        Self {
            encoding: DTYPE_NVFP4.to_string(),
            objects: Vec::new(),
            roles: RolePolicy::default(),
            deployment: false,
            protect: Protections::default(),
        }
    }

    /// A name for the program this spec describes, so two containers
    /// compiled the same way say so identically.
    pub fn map_name(&self) -> String {
        if self.protect.is_empty() {
            "r0-uniform".to_string()
        } else {
            format!(
                "r1-protect-{}",
                self.protect.describe().replace(", ", "+").replace(' ', "-")
            )
        }
    }

    fn wants(&self, object: &str) -> bool {
        self.objects.is_empty() || self.objects.iter().any(|o| o == object)
    }
}

/// Compile `spec`'s representations of the container at `src` into a new
/// container at `out`.
///
/// The output carries every original segment plus one compiled pack per
/// targeted object. Untouched segments are hard-linked where the
/// filesystem allows, so adding a representation costs the pack's bytes,
/// not the container's.
/// Which compiler writes an encoding's bytes.
///
/// ## The eligibility rule a search must not get wrong
///
/// K-quant eligibility is **not** `n_elements % block == 0`. It is
/// `inner_dim % block == 0`, with every outer-index row framed into
/// blocks independently, because that is how ggml lays a row out.
///
/// ```text
/// WRONG   N_elements mod B == 0
/// RIGHT   D_inner    mod B == 0     (each row blocked on its own)
/// ```
///
/// `[2, 128]` is the fixture that separates them: 256 elements, exactly
/// one Q6_K super-block, so the wrong rule admits it — and the resulting
/// block spans the end of row 0 and the start of row 1 under a single
/// shared scale. Nothing crashes. The bytes serialise, the segment table
/// is self-consistent, the container loads, and the arm produces
/// plausible behavioural numbers from semantically invalid bytes.
///
/// **This matters most once REPRESENT searches automatically.** The
/// optimiser enumerates candidate scopes; under the wrong rule it can
/// discover scopes that look valid, measure them, and fold the results
/// into a Pareto curve that is quietly part fiction. A representation
/// error that fails loudly costs an afternoon; this one would cost the
/// curve's credibility. `KQuant::plan` is where the rule lives, and
/// `a_shape_whose_total_divides_but_whose_row_does_not_is_refused`
/// asserts both halves so it cannot be "simplified" back to the total.
///
/// NVFP4 is a split pack — codes, then group scales, then a tensor scale
/// — whose planner derives a layout from a 2-D shape. A K-quant is
/// contiguous ggml blocks running along the row. The two share this
/// function's plan-then-write skeleton and nothing else, which is why
/// this is a target rather than a flag on one path.
///
/// A registered encoder is the third kind: a codec that writes its own
/// bytes ([`RepresentationEncoder`]), supplied by the caller rather than
/// compiled in. Its layout is its own business; what this function owns
/// is the plan-then-write skeleton, the policy, and the provenance.
#[derive(Debug, Clone, Copy)]
enum Target<'e> {
    Nvfp4,
    KQuant(kquant::KQuant),
    Encoder(&'e dyn RepresentationEncoder),
}

/// One tensor's decided encoding. `None` at the call sites means the
/// tensor is carried verbatim — because the policy preserves its role,
/// or because its shape cannot hold the encoding.
#[derive(Debug, Clone, Copy)]
enum TensorEncoding {
    Nvfp4(PackLayout),
    /// The encoding and the byte length its shape implies.
    KQuant(kquant::KQuant, usize),
    /// A registered encoder's bytes, already produced while planning, and
    /// their length.
    Encoder(usize),
}

impl TensorEncoding {
    /// Bytes this tensor occupies once encoded — what the segment table
    /// is planned with, before any payload is written.
    fn len(self) -> usize {
        match self {
            Self::Nvfp4(layout) => layout.total_len,
            Self::KQuant(_, len) | Self::Encoder(len) => len,
        }
    }
}

/// Which encoder chooses a K-quant pack's values.
///
/// **The control is architectural, not a runtime flag.** The feature
/// being compiled in IS the switch, so a comparative campaign cannot
/// accidentally call the native encoder, and a later regression in
/// `quantize_q6_k` cannot perturb an experiment that never called it.
///
/// Paired with [`kquant_encoder_recipe`] and derived in this one place,
/// so the bytes and the provenance that describes them cannot disagree.
#[cfg(feature = "reference-encoder")]
fn encode_kquant(
    k: kquant::KQuant,
    values: &[f32],
    row_len: usize,
    tensor: &str,
) -> Result<Vec<u8>, VindexError> {
    reference_encoder::encode(k, values, row_len, tensor)
}

#[cfg(not(feature = "reference-encoder"))]
fn encode_kquant(
    k: kquant::KQuant,
    values: &[f32],
    _row_len: usize,
    tensor: &str,
) -> Result<Vec<u8>, VindexError> {
    k.encode(values, tensor)
}

/// The provenance for whichever encoder [`encode_kquant`] is.
#[cfg(feature = "reference-encoder")]
fn kquant_encoder_recipe() -> EncoderRecipe {
    EncoderRecipe::kquant_ggml_reference(reference_encoder::PINNED_UPSTREAM)
}

#[cfg(not(feature = "reference-encoder"))]
fn kquant_encoder_recipe() -> EncoderRecipe {
    EncoderRecipe::kquant_native_v1()
}

/// Input-feature weights for a weighted compile: per tensor, one weight per
/// input feature (typically `E[x_i²]` over a calibration set), and the
/// digest of the artifact they came from, which the pack's encoder recipe
/// records.
#[derive(Debug, Clone, Default)]
pub struct InputWeights {
    /// `(object, tensor)` → one weight per element of a row.
    pub by_tensor: BTreeMap<(String, String), Vec<f64>>,
    /// sha256 of the artifact the weights were derived from.
    pub digest: String,
}

/// Encode one tensor with a registered encoder, at its terminal extent.
///
/// `Ok(None)` means the shape cannot hold the encoding — the encoder's own
/// `stored_bytes` refused it — and the tensor is carried, as a K-quant row
/// that does not divide is. A codec whose length depends on the values
/// (`InstanceSized`) is encoded to find out. When the codec can state the
/// length from the shape, the bytes must match it, or the segment table
/// would not describe its payload.
fn encode_with(
    encoder: &dyn RepresentationEncoder,
    source: &OperandSource<'_>,
    object: &str,
    tensor: &super::encode::segment::SegmentTensor,
    input_weights: Option<&[f64]>,
) -> Result<Option<Vec<u8>>, VindexError> {
    let extent = encoder.terminal_extent();
    let expected = match encoder.stored_bytes(&tensor.shape, extent, &tensor.name) {
        Ok(len) => Some(len as usize),
        Err(CodecError::InstanceSized { .. }) => None,
        Err(_) => return Ok(None),
    };
    let values = source.load(&OperandRef {
        object: object.to_string(),
        tensor: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
    })?;
    let bytes = match input_weights {
        Some(w) => encoder.encode_packed_weighted(&values, &tensor.shape, extent, &tensor.name, w),
        None => encoder.encode_packed(&values, &tensor.shape, extent, &tensor.name),
    }
    .map_err(|e| VindexError::Parse(format!("tensor `{}`: {e}", tensor.name)))?;
    if let Some(expected) = expected.filter(|&n| n != bytes.len()) {
        return Err(VindexError::Parse(format!(
            "tensor `{}`: `{}` encoded {} bytes, its codec states {expected} for this shape",
            tensor.name,
            encoder.encoding_label(),
            bytes.len()
        )));
    }
    Ok(Some(bytes))
}

/// The role a tensor compiles under. The plan first: it bound this tensor
/// to the role its operator computes with, and that is the container's
/// own judgement. The name heuristics answer only for what the plan does
/// not cover — object-level roles, components with no plan.
fn tensor_role(
    declared_roles: &plan_roles::PlanRoles,
    primary_text: &BTreeSet<String>,
    object: &str,
    t: &super::encode::segment::SegmentTensor,
) -> Role {
    let primary = primary_text.contains(object);
    declared_roles
        .get(&(object.to_string(), t.name.clone()))
        .copied()
        .filter(|_| primary)
        .unwrap_or_else(|| classify_in(primary, object, &t.name, &t.shape))
}

pub fn compile_representation(
    src: &Path,
    out: &Path,
    spec: &RepresentSpec,
) -> Result<RepresentReport, VindexError> {
    compile_representation_with(src, out, spec, &EncoderRegistry::new())
}

/// [`compile_representation`], with `encoders` available as targets
/// beside the compilers this build ships. A shipped compiler's name is
/// never answered by a registered encoder.
pub fn compile_representation_with(
    src: &Path,
    out: &Path,
    spec: &RepresentSpec,
    encoders: &EncoderRegistry,
) -> Result<RepresentReport, VindexError> {
    compile_inner(src, out, spec, encoders, None, None)
}

/// [`compile_representation_with`], with a registered encoder minimising
/// the input-weighted error for every tensor `weights` covers
/// ([`RepresentationEncoder::encode_packed_weighted`]); a tensor it does
/// not cover is encoded unweighted and counted apart. Only a registered
/// encoder takes weights: a shipped compiler refuses them rather than
/// ignore them.
pub fn compile_representation_weighted(
    src: &Path,
    out: &Path,
    spec: &RepresentSpec,
    encoders: &EncoderRegistry,
    weights: &InputWeights,
) -> Result<RepresentReport, VindexError> {
    compile_inner(src, out, spec, encoders, Some(weights), None)
}

fn compile_inner(
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
    let primary_text: BTreeSet<String> = {
        let text: BTreeSet<&str> = inspection
            .graph
            .components
            .iter()
            .filter(|c| c.role == super::graph::component::ComponentRole::PrimaryText)
            .map(|c| c.id.as_str())
            .collect();
        inspection
            .graph
            .objects
            .iter()
            .filter(|o| text.contains(o.component.as_str()))
            .map(|o| o.id.clone())
            .collect()
    };
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
        let out_segment = out.join(&segment_rel);
        if let Some(parent) = out_segment.parent() {
            std::fs::create_dir_all(parent)?;
        }

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
                    Target::KQuant(_) => kquant_encoder_recipe(),
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
        let mut graph: super::graph::SystemGraph = serde_json::from_str(&graph_raw)
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

#[cfg(test)]
#[path = "compile_real_tests.rs"]
mod compile_real_tests;

#[cfg(test)]
#[path = "compat_tests.rs"]
mod compat_tests;

#[cfg(test)]
#[path = "pareto_tests.rs"]
mod pareto_tests;

#[cfg(test)]
#[path = "frontier_scale_tests.rs"]
mod frontier_scale_tests;

#[cfg(test)]
#[path = "frontier_explore_tests.rs"]
mod frontier_explore_tests;

#[cfg(test)]
#[path = "frontier_spend_tests.rs"]
mod frontier_spend_tests;

#[cfg(test)]
#[path = "depth_invariance_tests.rs"]
mod depth_invariance_tests;

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod terminal_tests;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
