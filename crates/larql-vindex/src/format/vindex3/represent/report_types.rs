//! Represent specs, targets and the report types a compile returns.

use super::super::opplan::exec::operands::OperandSource;
use super::super::opplan::OperandRef;
use crate::error::VindexError;
use codec::{CodecError, EncoderRegistry, RepresentationEncoder};
use nvfp4_pack::{EncoderRecipe, PackLayout, DTYPE_NVFP4};
use policy::{classify_in, Protections, Role, RolePolicy};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

/// Filename of the system graph, carried beside the index.
pub(super) const SYSTEM_GRAPH_JSON: &str = "system_graph.json";

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

    pub(super) fn wants(&self, object: &str) -> bool {
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
pub(super) enum Target<'e> {
    Nvfp4,
    KQuant(kquant::KQuant),
    Encoder(&'e dyn RepresentationEncoder),
}

/// One tensor's decided encoding. `None` at the call sites means the
/// tensor is carried verbatim — because the policy preserves its role,
/// or because its shape cannot hold the encoding.
#[derive(Debug, Clone, Copy)]
pub(super) enum TensorEncoding {
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
    pub(super) fn len(self) -> usize {
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
pub(super) fn encode_kquant(
    k: kquant::KQuant,
    values: &[f32],
    row_len: usize,
    tensor: &str,
) -> Result<Vec<u8>, VindexError> {
    reference_encoder::encode(k, values, row_len, tensor)
}

#[cfg(not(feature = "reference-encoder"))]
pub(super) fn encode_kquant(
    k: kquant::KQuant,
    values: &[f32],
    _row_len: usize,
    tensor: &str,
) -> Result<Vec<u8>, VindexError> {
    k.encode(values, tensor)
}

/// The provenance for whichever encoder [`encode_kquant`] is.
#[cfg(feature = "reference-encoder")]
pub(super) fn kquant_encoder_recipe(_kquant: &kquant::KQuant) -> EncoderRecipe {
    EncoderRecipe::kquant_ggml_reference(reference_encoder::PINNED_UPSTREAM)
}

#[cfg(not(feature = "reference-encoder"))]
pub(super) fn kquant_encoder_recipe(kquant: &kquant::KQuant) -> EncoderRecipe {
    EncoderRecipe::kquant_native(kquant.name)
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
pub(super) fn encode_with(
    encoder: &dyn RepresentationEncoder,
    source: &OperandSource<'_>,
    object: &str,
    tensor: &super::super::encode::segment::SegmentTensor,
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
/// The objects that belong to the primary text model. A perception
/// tower's tensors are named exactly like a decoder's, so the component's
/// declared role is the only thing that separates them. Shared by the
/// compiler and the search-record producer, so both classify alike.
pub(crate) fn primary_text_objects(
    inspection: &super::super::inspect::SystemInspection,
) -> BTreeSet<String> {
    let text: BTreeSet<&str> = inspection
        .graph
        .components
        .iter()
        .filter(|c| c.role == super::super::graph::component::ComponentRole::PrimaryText)
        .map(|c| c.id.as_str())
        .collect();
    inspection
        .graph
        .objects
        .iter()
        .filter(|o| text.contains(o.component.as_str()))
        .map(|o| o.id.clone())
        .collect()
}

pub(crate) fn tensor_role(
    declared_roles: &plan_roles::PlanRoles,
    primary_text: &BTreeSet<String>,
    object: &str,
    t: &super::super::encode::segment::SegmentTensor,
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
