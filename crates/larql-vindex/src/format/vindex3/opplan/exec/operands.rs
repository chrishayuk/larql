//! Operand loading: an [`OperandRef`] to f32 values, from the container's
//! segments alone.
//!
//! Resolution is `object id → representation → segment → table entry →
//! payload bytes` — the same path closure verified, and no other. An
//! operand the store cannot resolve, or a dtype nobody has judged a
//! widening for, is an error naming the operand — never a zero-filled
//! buffer.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::super::super::encode::segment::SegmentTensor;
use super::super::OperandRef;
use crate::error::VindexError;
use crate::format::vindex3::represent::codec::streams::ResolvedAuxiliary;
use crate::format::vindex3::represent::codec::{
    CodecOperands, CodecRegistry, RepresentationExtent,
};
use crate::format::vindex3::represent::physical::{PhysicalStore, WeightRegion};

mod load;
mod open;

/// Safetensors dtype labels this reference executor can widen to f32.
const DTYPE_F32: &str = "F32";
const DTYPE_BF16: &str = "BF16";
const DTYPE_F16: &str = "F16";

/// One object's segment: file path, payload origin, and tensor table.
struct SegmentMap {
    path: PathBuf,
    /// The segment path as the container's index declares it, relative
    /// to the container root.
    declared: String,
    payload_start: u64,
    tensors: BTreeMap<String, SegmentTensor>,
}

/// Operand store over one container.
pub struct OperandStore {
    /// Container authority for associated artifacts such as the tokenizer.
    root: PathBuf,
    /// The codecs this store decodes through — the built-in registry
    /// unless a caller binds another, which is how a representation this
    /// build does not ship becomes executable through registration alone.
    registry: &'static CodecRegistry,
    segments: BTreeMap<String, SegmentMap>,
    /// Each object's segment mapped at most once, however many operands
    /// bind regions of it — the one physical binding a bank shares.
    mapped: std::sync::Mutex<BTreeMap<String, Arc<PhysicalStore>>>,
    /// Regions bound through `map_region`, counted beside `loads` so a
    /// test can prove a bank was bound and not read.
    regions: std::sync::atomic::AtomicUsize,
    /// Which representation each object was bound to.
    selected: BTreeMap<String, SelectedRepresentation>,
    /// Under `transient`, the encoding each tensor has in the compiled
    /// pack whose bytes are being ignored — the program the oracle must
    /// reproduce. Empty when there is no pack, which is R0.
    precision_map: BTreeMap<String, BTreeMap<String, String>>,
    /// The container's precision program, when it states one.
    program: Option<crate::format::vindex3::represent::map::PrecisionMap>,
    /// Every operand's role as the operation plan binds it, keyed
    /// `(object, tensor)`.
    ///
    /// The conformance check below must resolve a role exactly as the
    /// representation compiler did, or a legitimately compiled pack
    /// fails its own check. That is not hypothetical: when the compiler
    /// moved to plan-derived roles and this path was left on the name
    /// heuristics, Qwen3.8's `linear_attn.in_proj_qkv` compiled as
    /// `recurrence-projection` and then refused to load because
    /// `classify` still called it `unknown`. One resolution, two
    /// readers.
    plan_roles: crate::format::vindex3::represent::plan_roles::PlanRoles,
    /// Where representations were allowed to come from.
    source: RepresentationSource,
    /// Process-unique identity — see [`SourceStamp`].
    id: u64,
    /// How many operands have been read out of this store.
    ///
    /// Residency is an architectural claim ("a served model's operands
    /// are lowered once"), and a claim that can only be checked by
    /// stopwatch is a claim that regresses quietly. This counter lets a
    /// test assert the shape directly: prepare, then serve N requests,
    /// then assert the count did not move.
    loads: std::sync::atomic::AtomicU64,
    /// PHYSICAL bytes this store has actually read from disk.
    ///
    /// The observed half of preparation accounting. `loads` counts CALLS,
    /// which cannot be compared against a byte ledger — two reads of a
    /// small operand and one read of a large one are the same number.
    /// Incremented in [`Self::load_raw`] and [`Self::load_raw_range`], the paths
    /// that copy payload; `map_region` binds without reading and correctly
    /// moves neither counter.
    read_bytes: std::sync::atomic::AtomicU64,
    /// Tensors quantised at load in this session — see
    /// [`Self::runtime_quantised`].
    runtime_quantised: std::sync::atomic::AtomicU64,
    /// Tensors bound at their stored precision rather than the format the
    /// backend asked for — see [`Self::bound_at_stored_precision`].
    stored_precision: std::sync::atomic::AtomicU64,
    /// Objects the container describes whose segment is not on disk.
    ///
    /// Distinct from "not in the container": these are declared by the
    /// graph and the index, and only their bytes are elsewhere. Keeping
    /// them named is what lets the load path refuse by residency rather
    /// than by absence.
    absent: std::collections::BTreeSet<String>,
    /// Which represented object stands for each codec's declared
    /// dependency, as the container states it.
    ///
    /// Empty for every container that declares none, which is every
    /// container written before dependencies existed. The store holds it
    /// because the store is what resolves one: a codec asks for a
    /// dependency by name and never learns where it came from.
    references: crate::format::vindex3::auxiliary_references::ReferenceTable,
    /// What this container measures about its own representations.
    ///
    /// Empty for every container written before attestation existed,
    /// which is every container this build has ever read — so an empty
    /// table is the normal case and means "nothing measured", never "the
    /// file failed to load". A table the index NAMES and the container
    /// does not hold is a refusal at open, on the same terms as the
    /// reference table beside it.
    attestations: crate::format::vindex3::representation_attestations::AttestationTable,
    /// Whose measurements this build is willing to act on.
    ///
    /// [`RecognisedMethods::none`] by default, which is what a build that
    /// has qualified no measurement should say: an attestation is carried
    /// and checked either way, but an unrecognised one leaves the
    /// guarantee unavailable rather than optimistic. Recognition is a
    /// TRUST decision and belongs to the caller, not to the container
    /// making the claim about itself.
    recognised: crate::format::vindex3::representation_attestations::recognition::RecognisedMethods,
    /// Which objects this store has actually resolved an operand out of.
    ///
    /// The consumption half of the residency ledger. `load_count` says
    /// how much was read; this says *from where*, which is the question a
    /// hydration set has to answer: an execution that needs three objects
    /// must not be handed four, and the only way to know which three is
    /// to watch a real preparation ask.
    ///
    /// Recorded in [`Self::load_raw`] because that is the one resolution
    /// path — a second place to record would be a second answer.
    touched: std::sync::Mutex<std::collections::BTreeSet<String>>,
    /// Test witness at tensor granularity: one object may carry both local
    /// attention and remotely placed FFN operands.
    #[cfg(test)]
    touched_operands: std::sync::Mutex<std::collections::BTreeSet<(String, String)>>,
}

/// How much of each dependency to read, by the name its OWNER declared.
///
/// A name absent from this is read WHOLE — the container holds all of it,
/// and reading less is a decision someone has to have made. It is the
/// shape a pin will carry once selection chooses auxiliary extents; until
/// then it is how a caller states the choice explicitly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuxiliaryExtents {
    by_name: BTreeMap<String, RepresentationExtent>,
}

impl AuxiliaryExtents {
    /// Every dependency read whole.
    pub fn whole() -> Self {
        Self::default()
    }

    pub fn with(mut self, name: impl Into<String>, extent: RepresentationExtent) -> Self {
        self.by_name.insert(name.into(), extent);
        self
    }

    /// The extent chosen for `name`, or `None` for "whole", which the
    /// loader resolves against the dependency's own codec.
    pub fn get(&self, name: &str) -> Option<RepresentationExtent> {
        self.by_name.get(name).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// One dependency the loader resolved: the name its OWNER declared, the
/// shape the container records for it, and its decoded values. Owned,
/// because it lives exactly as long as the decode that reads it.
struct LoadedAuxiliary {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// Hand `resolved` dependencies to the operands a codec will see.
///
/// A free function with a NAMED lifetime: the values are owned by the
/// caller for exactly as long as the decode runs, and a closure cannot
/// say that.
fn attach_auxiliaries<'a>(
    mut operands: CodecOperands<'a>,
    resolved: &'a [LoadedAuxiliary],
) -> CodecOperands<'a> {
    for auxiliary in resolved {
        operands.auxiliaries = std::mem::take(&mut operands.auxiliaries).with(
            auxiliary.name.clone(),
            ResolvedAuxiliary {
                shape: &auxiliary.shape,
                values: &auxiliary.values,
            },
        );
    }
    operands
}

/// Where an execution representation is allowed to come from.
///
/// Deliberately separate from *which* representation execution wants. The
/// profile says "NVFP4 laid out this way"; this says whether the runtime
/// may manufacture that now or must find it already compiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RepresentationSource {
    /// Use a compiled pack when one exists, otherwise quantise at load.
    #[default]
    Auto,
    /// Forbid manufacturing a representation at load.
    ///
    /// Note what this does *not* say: that every object must have a pack.
    /// A conservative role policy deliberately leaves the embedding, the
    /// norms and the router at source precision, and binding those
    /// canonically manufactures nothing. The invariant is about work, not
    /// about coverage — if the runtime would have to quantise a tensor to
    /// proceed, the run fails naming it, rather than quietly doing the
    /// work persistence exists to avoid.
    Stored,
    /// Ignore any compiled pack and quantise at load.
    ///
    /// Retained permanently, not as a migration aid: it is the oracle the
    /// compiler is checked against, and an arm that fell through to a
    /// convenient pack would stop being one.
    Transient,
}

/// Which representation an object was bound to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedRepresentation {
    /// Encoding of the bytes actually opened.
    pub encoding: String,
    /// Whether those bytes came from a compiled pack.
    pub stored: bool,
    /// The decode ABI the pack declares, when it declares one — kept so
    /// the admission that ran at open can run again against another
    /// registry ([`OperandStore::with_registry`]).
    pub codec: Option<crate::format::vindex3::represent::nvfp4_pack::CodecIdentity>,
}

impl OperandStore {}

impl OperandStore {
    /// One operand's stored bytes as a region of its object's mapped
    /// segment — bound, not read: no payload byte is copied, and the
    /// object's segment is mapped once for every region taken from it.
    /// `expected_len` is the byte count the plan's declared geometry
    /// implies; a region of any other length is a container disagreeing
    /// with the declaration and is refused rather than sliced.
    pub fn map_region(
        &self,
        operand: &OperandRef,
        expected_len: u64,
    ) -> Result<WeightRegion, VindexError> {
        let segment = self.segments.get(&operand.object).ok_or_else(|| {
            VindexError::Parse(format!("no segment for object `{}`", operand.object))
        })?;
        self.touched.lock().unwrap().insert(operand.object.clone());
        #[cfg(test)]
        self.touched_operands
            .lock()
            .unwrap()
            .insert((operand.object.clone(), operand.tensor.clone()));
        let store = {
            let mut mapped = self.mapped.lock().unwrap();
            match mapped.get(&operand.object) {
                Some(store) => store.clone(),
                None => {
                    let store = Arc::new(
                        PhysicalStore::map_segment(operand.object.clone(), &segment.path)?
                            .declared_as(segment.declared.clone()),
                    );
                    mapped.insert(operand.object.clone(), store.clone());
                    store
                }
            }
        };
        let region = store.whole(&operand.tensor).ok_or_else(|| {
            VindexError::Parse(format!(
                "no tensor `{}` in `{}`'s segment",
                operand.tensor, operand.object
            ))
        })?;
        if region.len() != expected_len {
            return Err(VindexError::Parse(format!(
                "`{}`: {} stored bytes, the declared geometry implies {expected_len}; the \
                 container and the plan disagree",
                operand.tensor,
                region.len()
            )));
        }
        self.regions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(region)
    }
}

/// One operand exactly as stored: payload bytes plus the dtype label
/// that says how to read them.
pub struct RawOperand {
    pub dtype: String,
    pub bytes: Vec<u8>,
}

/// Widen stored bytes to f32 — judged dtypes only, fail-closed.
pub(crate) fn widen(dtype: &str, bytes: &[u8], name: &str) -> Result<Vec<f32>, VindexError> {
    match dtype {
        DTYPE_F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        DTYPE_BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        // IEEE half. Exact — every f16 value is representable in f32 —
        // through the one half-precision decoder the workspace already
        // judges, not a second bit-twiddling copy. First shipped estate:
        // mamba2-780m (the checkpoint is F16 throughout, where the prior
        // corpus was BF16).
        DTYPE_F16 => Ok(larql_models::quant::half::decode_f16(bytes)),
        other => Err(VindexError::Parse(format!(
            "tensor `{name}`: no judged f32 widening for dtype `{other}`"
        ))),
    }
}

/// One logical f32 edit to a stored operand (V3-LQL-3B compose): a row
/// or a column replaced by new values. Addressed semantically — the
/// operand's identity plus a slot index — never by byte offsets, so an
/// edit survives repacking or an alternative physical representation.
#[derive(Debug, Clone, PartialEq)]
pub enum OperandEdit {
    Row { index: usize, values: Vec<f32> },
    Column { index: usize, values: Vec<f32> },
}

/// Logical edits and immutable packed replacements, keyed by operand identity
/// (object + tensor). Applied inside [`OperandSource::load`] — after
/// widening to f32, before any backend requantization — so **every
/// weight format observes the same effective values** (`load_weight`
/// quantizes from the widened f32 buffer).
#[derive(Debug)]
pub struct OperandOverrides {
    edits: BTreeMap<(String, String), Vec<OperandEdit>>,
    /// Immutable candidate packs; decoded only when this operand is loaded.
    replacements: BTreeMap<(String, String), (Vec<usize>, WeightRegion)>,
    /// Process-unique identity, so two override sets are never
    /// mistaken for each other.
    id: u64,
    /// Bumped on every mutation. Together with `id` this is what lets a
    /// derived artefact — a [`PreparedOperands`](super::prepared::PreparedOperands)
    /// image — say whether it still describes these edits.
    generation: u64,
}

/// Hands out process-unique identities for override sets and stores.
fn next_identity() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Default for OperandOverrides {
    fn default() -> Self {
        Self {
            edits: BTreeMap::new(),
            replacements: BTreeMap::new(),
            id: next_identity(),
            generation: 0,
        }
    }
}

impl Clone for OperandOverrides {
    /// A clone takes a **fresh** identity. The two sets are equal now
    /// but diverge independently, and an artefact prepared from one
    /// must not silently pass as current for the other. Conservative by
    /// construction: the cost of a false "stale" is one re-preparation;
    /// the cost of a false "current" is executing the wrong model.
    fn clone(&self) -> Self {
        Self {
            edits: self.edits.clone(),
            replacements: self.replacements.clone(),
            id: next_identity(),
            generation: self.generation,
        }
    }
}

impl OperandOverrides {
    pub fn new() -> Self {
        Self::default()
    }

    /// This set's identity and mutation count — what a derived image
    /// stamps itself with.
    pub fn version(&self) -> (u64, u64) {
        (self.id, self.generation)
    }

    pub fn is_empty(&self) -> bool {
        self.edits.is_empty() && self.replacements.is_empty()
    }

    /// Record one edit for an operand; edits apply in insertion order.
    pub fn push(&mut self, operand: &OperandRef, edit: OperandEdit) {
        self.generation += 1;
        self.edits
            .entry((operand.object.clone(), operand.tensor.clone()))
            .or_default()
            .push(edit);
    }

    pub fn is_overridden(&self, operand: &OperandRef) -> bool {
        self.edits
            .contains_key(&(operand.object.clone(), operand.tensor.clone()))
    }

    fn packed_replacement(&self, operand: &OperandRef) -> Option<&WeightRegion> {
        self.replacements
            .get(&(operand.object.clone(), operand.tensor.clone()))
            .map(|(_, region)| region)
    }

    /// Install a completed immutable NVFP4 pack for sequential offline capture.
    /// Subsequent logical edits still apply after this replacement.
    pub(crate) fn replace_nvfp4(&mut self, operand: &OperandRef, region: WeightRegion) {
        self.generation += 1;
        self.replacements.insert(
            (operand.object.clone(), operand.tensor.clone()),
            (operand.shape.clone(), region),
        );
    }

    fn replacement(&self, operand: &OperandRef) -> Result<Option<Vec<f32>>, VindexError> {
        let Some((shape, region)) = self
            .replacements
            .get(&(operand.object.clone(), operand.tensor.clone()))
        else {
            return Ok(None);
        };
        if shape != &operand.shape {
            return Err(VindexError::Parse(
                "candidate replacement shape mismatch".into(),
            ));
        }
        use crate::format::vindex3::represent::codec::{codecs::nvfp4::NVFP4, RepresentationCodec};
        Ok(Some(NVFP4.decode_packed(
            region.bytes(),
            shape,
            NVFP4.terminal_extent(),
            &operand.tensor,
        )?))
    }

    /// Apply this operand's edits onto its widened f32 values.
    /// Row-major 2-D shape; an edit that does not fit the operand's
    /// declared shape is an error naming the operand — never a silent
    /// partial write.
    pub fn apply(&self, operand: &OperandRef, values: &mut [f32]) -> Result<(), VindexError> {
        let key = (operand.object.clone(), operand.tensor.clone());
        let Some(edits) = self.edits.get(&key) else {
            return Ok(());
        };
        let (rows, cols) = match operand.shape[..] {
            [rows, cols] => (rows, cols),
            _ => {
                return Err(VindexError::Parse(format!(
                    "operand `{}/{}` is not 2-D; overlay edits address rows/columns",
                    operand.object, operand.tensor
                )))
            }
        };
        for edit in edits {
            match edit {
                OperandEdit::Row { index, values: row } => {
                    if *index >= rows || row.len() != cols {
                        return Err(VindexError::Parse(format!(
                            "row edit {index} (len {}) does not fit `{}/{}` [{rows}, {cols}]",
                            row.len(),
                            operand.object,
                            operand.tensor
                        )));
                    }
                    values[index * cols..(index + 1) * cols].copy_from_slice(row);
                }
                OperandEdit::Column { index, values: col } => {
                    if *index >= cols || col.len() != rows {
                        return Err(VindexError::Parse(format!(
                            "column edit {index} (len {}) does not fit `{}/{}` [{rows}, {cols}]",
                            col.len(),
                            operand.object,
                            operand.tensor
                        )));
                    }
                    for (r, v) in col.iter().enumerate() {
                        values[r * cols + *index] = *v;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The executor's operand resolver: base representation + overlay
/// override → effective operand. Execution asks this seam, never the
/// store directly, so a mutation can alter what execution computes
/// without touching the container's bytes — and a source with no
/// overrides resolves bit-identically to the bare store.
/// The identity of one *effective* operand source: which store, and
/// which version of which overlay.
///
/// Preparation turns an effective source into a compiled artefact
/// ([`PreparedOperands`](super::prepared::PreparedOperands)), so that
/// artefact needs to be able to say which source it describes. Without
/// this, a prepared image outlives an overlay mutation and quietly
/// keeps executing the pre-edit model — the derived state becoming a
/// second authority for what the model means, which is exactly what the
/// operand seam exists to prevent.
///
/// Equality is deliberately conservative: reverting an edit produces a
/// new generation and therefore a different stamp, so a valid image can
/// be judged stale (costing one re-preparation) but a stale one can
/// never be judged valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceStamp {
    store: u64,
    /// `None` when the source is the bare store.
    overlay: Option<(u64, u64)>,
}

impl SourceStamp {
    /// Whether the source carried an overlay. An overlay's stamp is
    /// process-unique, so an overlaid image has no identity another
    /// process could name (RESIDUAL-BUS-2 D8).
    pub fn is_overlaid(&self) -> bool {
        self.overlay.is_some()
    }
}

#[derive(Clone, Copy)]
pub struct OperandSource<'a> {
    base: &'a OperandStore,
    overrides: Option<&'a OperandOverrides>,
}

impl<'a> OperandSource<'a> {
    /// A source with overlay edits. An empty overrides value behaves
    /// exactly like the bare store.
    pub fn overlaid(base: &'a OperandStore, overrides: &'a OperandOverrides) -> Self {
        Self {
            base,
            overrides: (!overrides.is_empty()).then_some(overrides),
        }
    }

    /// The store underneath, for the facts that belong to the session
    /// rather than to one operand — which representation was selected, and
    /// how many tensors were quantised at load.
    pub fn store(&self) -> &OperandStore {
        self.base
    }

    /// The codecs this source decodes through.
    pub fn registry(&self) -> &'static CodecRegistry {
        self.base.registry()
    }

    /// This source's identity, for stamping derived artefacts.
    pub fn stamp(&self) -> SourceStamp {
        SourceStamp {
            store: self.base.id(),
            overlay: self.overrides.map(OperandOverrides::version),
        }
    }

    /// Load one operand as f32, with any overlay edits applied.
    pub fn load(&self, operand: &OperandRef) -> Result<Vec<f32>, VindexError> {
        let mut values = match self
            .overrides
            .map(|o| o.replacement(operand))
            .transpose()?
            .flatten()
        {
            Some(values) => values,
            None => self.base.load(operand)?,
        };
        if let Some(overrides) = self.overrides {
            overrides.apply(operand, &mut values)?;
        }
        Ok(values)
    }

    /// Whether an overlay edit stands on this operand. An edit is an
    /// f32-space fact with no representation in stored bytes, so the only
    /// realization that can honour it decodes — the selector reads this
    /// beside the registry's facts, and [`Self::load_raw`] refuses it.
    pub fn is_overridden(&self, operand: &OperandRef) -> bool {
        self.overrides.is_some_and(|o| o.is_overridden(operand))
    }

    /// The effective stored length, including immutable candidate packs.
    /// Logical row/column edits alone do not change stored length.
    pub fn stored_len(&self, operand: &OperandRef) -> Option<u64> {
        self.overrides
            .and_then(|o| o.packed_replacement(operand))
            .map(|r| r.len())
            .or_else(|| self.base.stored_len(operand))
    }

    /// The effective stored representation, including an immutable candidate
    /// pack. Logical edits separately force a decoded realization.
    pub fn stored_dtype(&self, operand: &OperandRef) -> Option<&str> {
        if self
            .overrides
            .and_then(|o| o.packed_replacement(operand))
            .is_some()
        {
            Some("NVFP4")
        } else {
            self.base.stored_dtype(operand)
        }
    }

    /// Load one operand's stored bytes unwidened. Overlay edits are
    /// f32-space facts and cannot be represented in raw stored bytes,
    /// so an overridden operand refuses here rather than serving stale
    /// base bytes.
    pub fn load_raw(&self, operand: &OperandRef) -> Result<RawOperand, VindexError> {
        if let Some(overrides) = self.overrides {
            if overrides.is_overridden(operand) {
                return Err(VindexError::Parse(format!(
                    "operand `{}/{}` carries overlay edits — raw (unwidened) access would \
                     bypass them; load it widened instead",
                    operand.object, operand.tensor
                )));
            }
        }
        if let Some(overrides) = self.overrides {
            if let Some((shape, region)) = overrides
                .replacements
                .get(&(operand.object.clone(), operand.tensor.clone()))
            {
                if shape != &operand.shape {
                    return Err(VindexError::Parse(
                        "candidate replacement shape mismatch".into(),
                    ));
                }
                return Ok(RawOperand {
                    dtype: "NVFP4".into(),
                    bytes: region.bytes().to_vec(),
                });
            }
        }
        self.base.load_raw(operand)
    }
}

impl<'a> From<&'a OperandStore> for OperandSource<'a> {
    fn from(base: &'a OperandStore) -> Self {
        Self {
            base,
            overrides: None,
        }
    }
}
