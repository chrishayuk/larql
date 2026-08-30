//! Operand loading: an [`OperandRef`] to f32 values, from the container's
//! segments alone.
//!
//! Resolution is `object id → representation → segment → table entry →
//! payload bytes` — the same path closure verified, and no other. An
//! operand the store cannot resolve, or a dtype nobody has judged a
//! widening for, is an error naming the operand — never a zero-filled
//! buffer.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::super::super::encode::segment::{read_segment_header, SegmentTensor};
use super::super::super::encode::REPRESENTATION_ID_SEP;
use super::super::super::inspect::SystemInspection;
use super::super::OperandRef;
use crate::error::VindexError;

/// Safetensors dtype labels this reference executor can widen to f32.
const DTYPE_F32: &str = "F32";
const DTYPE_BF16: &str = "BF16";
const DTYPE_F16: &str = "F16";

/// One object's segment: file path, payload origin, and tensor table.
struct SegmentMap {
    path: PathBuf,
    payload_start: u64,
    tensors: BTreeMap<String, SegmentTensor>,
}

/// Operand store over one container.
pub struct OperandStore {
    segments: BTreeMap<String, SegmentMap>,
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
    /// Tensors quantised at load in this session — see
    /// [`Self::runtime_quantised`].
    runtime_quantised: std::sync::atomic::AtomicU64,
    /// Tensors bound at their stored precision rather than the format the
    /// backend asked for — see [`Self::bound_at_stored_precision`].
    stored_precision: std::sync::atomic::AtomicU64,
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
}

impl OperandStore {
    /// Open every canonical segment of every object in the inspection.
    pub fn open(root: &Path, inspection: &SystemInspection) -> Result<Self, VindexError> {
        Self::open_for(root, inspection, None, RepresentationSource::Auto)
    }

    /// Open each object at the representation `want` selects, subject to
    /// `source`.
    ///
    /// `want` is the execution encoding a profile asked for. `None` keeps
    /// every object on its canonical representation, which is what every
    /// caller predating compiled packs meant.
    pub fn open_for(
        root: &Path,
        inspection: &SystemInspection,
        want: Option<&str>,
        source: RepresentationSource,
    ) -> Result<Self, VindexError> {
        let mut segments = BTreeMap::new();
        let mut selected = BTreeMap::new();
        let mut precision_map: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        for object in &inspection.graph.objects {
            let Some(canonical) = object.representations.first() else {
                continue;
            };

            // A compiled pack is used only when one was asked for and one
            // exists. `Transient` never looks, so it stays an oracle.
            let packed_id = want.map(|enc| format!("{}{REPRESENTATION_ID_SEP}{}", object.id, enc));
            let stored_entry = match (source, &packed_id) {
                (RepresentationSource::Transient, _) | (_, None) => None,
                (_, Some(id)) => inspection.index.representations.get(id).map(|e| (id, e)),
            };

            // Under `transient` the pack's BYTES are deliberately ignored,
            // but its DECISIONS are not: which tensors a precision map
            // quantised is a property of the compiled representation, and
            // an oracle that re-decided would be measuring a different
            // program. Read the map from the pack's header even when the
            // canonical bytes will be bound.
            if source == RepresentationSource::Transient {
                if let Some(id) = &packed_id {
                    if let Some(pack) = inspection.index.representations.get(id) {
                        if let Ok((header, _)) = read_segment_header(&root.join(&pack.segment)) {
                            precision_map.insert(
                                object.id.clone(),
                                header
                                    .tensors
                                    .into_iter()
                                    .map(|t| (t.name, t.dtype))
                                    .collect(),
                            );
                        }
                    }
                }
            }

            let (id, entry, is_stored) = match stored_entry {
                Some((id, entry)) => {
                    // Bytes compiled by another build under a decode
                    // contract this one may not implement must be refused
                    // here, before anything reads them.
                    if let Some(codec) = &entry.codec {
                        codec.admit()?;
                    }
                    (id.clone(), entry, true)
                }
                None => {
                    // No pack for this object. That is not yet a problem —
                    // it becomes one only if execution asks for a format
                    // these bytes are not already in, which the load path
                    // catches by name.
                    let id = format!("{}{REPRESENTATION_ID_SEP}{}", object.id, canonical.encoding);
                    let Some(entry) = inspection.index.representations.get(&id) else {
                        continue;
                    };
                    (id, entry, false)
                }
            };
            let _ = &id;
            selected.insert(
                object.id.clone(),
                SelectedRepresentation {
                    encoding: entry.encoding.clone(),
                    stored: is_stored,
                },
            );
            let path = root.join(&entry.segment);
            let (header, payload_start) = read_segment_header(&path)?;
            segments.insert(
                object.id.clone(),
                SegmentMap {
                    path,
                    payload_start,
                    tensors: header
                        .tensors
                        .into_iter()
                        .map(|t| (t.name.clone(), t))
                        .collect(),
                },
            );
        }
        Ok(Self {
            segments,
            selected,
            precision_map,
            program: inspection.index.precision_map.clone(),
            // Resolved once at open, so every conformance check in the
            // load path reads the same answer the compiler wrote.
            plan_roles: crate::format::vindex3::represent::plan_roles::plan_roles(root, inspection),
            source,
            id: next_identity(),
            loads: std::sync::atomic::AtomicU64::new(0),
            runtime_quantised: std::sync::atomic::AtomicU64::new(0),
            stored_precision: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// What each object was bound to.
    pub fn selection(&self) -> &BTreeMap<String, SelectedRepresentation> {
        &self.selected
    }

    /// How many tensors this session quantised at load.
    ///
    /// Session-scoped rather than process-global so concurrent runs and
    /// tests cannot contaminate each other's count. Under
    /// [`RepresentationSource::Stored`] a non-zero value is an invariant
    /// violation, not a performance observation: it means the runtime
    /// manufactured a representation the caller required to be already
    /// compiled.
    pub fn runtime_quantised(&self) -> u64 {
        self.runtime_quantised
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Where this store was allowed to source representations from.
    pub fn representation_source(&self) -> RepresentationSource {
        self.source
    }

    /// What encoding a compiled precision map gives this tensor, when the
    /// store is reproducing one.
    ///
    /// `None` means no map is in force — either the store is not the
    /// transient oracle, or no pack exists — and the caller decides as it
    /// always did. `Some(enc)` is the compiled program's decision for this
    /// tensor, and the oracle honours it rather than re-deciding.
    pub fn mapped_encoding(&self, object: &str, tensor: &str) -> Option<&str> {
        self.precision_map
            .get(object)?
            .get(tensor)
            .map(String::as_str)
    }

    /// The container's precision program, when it declares one.
    pub fn program(&self) -> Option<&crate::format::vindex3::represent::map::PrecisionMap> {
        self.program.as_ref()
    }

    /// The role the plan binds this tensor to, falling back to the name
    /// heuristics for anything the plan does not cover — the same order
    /// the representation compiler resolves in.
    pub fn role_of(
        &self,
        object: &str,
        tensor: &str,
        shape: &[usize],
    ) -> crate::format::vindex3::represent::policy::Role {
        self.plan_roles
            .get(&(object.to_string(), tensor.to_string()))
            .copied()
            .unwrap_or_else(|| {
                crate::format::vindex3::represent::policy::classify(object, tensor, shape)
            })
    }

    /// Whether this object's bytes come from a compiled pack.
    ///
    /// Conformance is a claim about a *pack*. Under `transient` the bound
    /// bytes are the canonical ones by design, and they are expected not to
    /// match a map describing the pack — checking them against it would
    /// refuse the oracle for doing exactly its job.
    pub fn is_stored(&self, object: &str) -> bool {
        self.selected.get(object).is_some_and(|s| s.stored)
    }

    /// How many tensors ran at their stored precision instead of the
    /// format the backend asked for.
    ///
    /// A compiled pack is a precision map: it may store `gate_proj` as
    /// NVFP4 and `q_proj` as BF16 because a policy decided to spend bytes
    /// there. Backend arms declare a format per *class* — attention, FFN,
    /// head — which is a coarser instrument than the map, so under
    /// [`RepresentationSource::Stored`] the stored encoding wins and the
    /// arm's request acts as a ceiling rather than a demand.
    ///
    /// This is never a silent downgrade: honouring the map means running
    /// *higher* precision than asked, and the count says how often.
    pub fn bound_at_stored_precision(&self) -> u64 {
        self.stored_precision
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record one such binding.
    pub fn note_stored_precision(&self) {
        self.stored_precision
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record that a tensor is about to be quantised at load, and refuse
    /// under [`RepresentationSource::Stored`].
    ///
    /// The refusal is the gate: it makes "no runtime quantisation" an
    /// invariant the run enforces rather than a timing an operator infers.
    /// Called by the weight loader, the only place quantisation can happen.
    pub fn note_runtime_quantisation(&self, tensor: &str) -> Result<(), VindexError> {
        if self.source == RepresentationSource::Stored {
            return Err(VindexError::Parse(format!(
                "tensor `{tensor}` would be quantised at load, and \
                 `--representation-source stored` forbids manufacturing a \
                 representation. Compile one with `larql vindex3 represent`, \
                 or ask for `auto`."
            )));
        }
        self.runtime_quantised
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Load one operand as f32 values.
    pub fn load(&self, operand: &OperandRef) -> Result<Vec<f32>, VindexError> {
        let raw = self.load_raw(operand)?;
        widen(&raw.dtype, &raw.bytes, &operand.tensor)
    }

    /// This store's process-unique identity.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// How many operands have been read out of this store since it was
    /// opened. The residency gate reads this.
    pub fn load_count(&self) -> u64 {
        self.loads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The dtype the container stores this operand as — tensor-table
    /// metadata only, no payload read.
    ///
    /// Separate from [`Self::load_raw`] because the residency policy has
    /// to know what a 100 MB matrix is BEFORE deciding how to hold it,
    /// and a query that read the matrix to answer would load the model
    /// twice.
    pub fn stored_dtype(&self, operand: &OperandRef) -> Option<&str> {
        self.segments
            .get(&operand.object)?
            .tensors
            .get(&operand.tensor)
            .map(|t| t.dtype.as_str())
    }

    /// Load one operand's stored bytes and dtype, unwidened — for a
    /// caller that converts to a representation other than f32 (and for
    /// [`Self::load`] itself, so there is exactly one resolution path).
    pub fn load_raw(&self, operand: &OperandRef) -> Result<RawOperand, VindexError> {
        self.loads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let segment = self.segments.get(&operand.object).ok_or_else(|| {
            VindexError::Parse(format!("no segment for object `{}`", operand.object))
        })?;
        let tensor = segment.tensors.get(&operand.tensor).ok_or_else(|| {
            VindexError::Parse(format!(
                "no tensor `{}` in `{}`'s segment",
                operand.tensor, operand.object
            ))
        })?;
        let mut file = std::fs::File::open(&segment.path)?;
        file.seek(SeekFrom::Start(segment.payload_start + tensor.offset))?;
        let mut bytes = vec![0u8; tensor.len as usize];
        file.read_exact(&mut bytes)?;
        Ok(RawOperand {
            dtype: tensor.dtype.clone(),
            bytes,
        })
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

/// Logical edits over stored operands, keyed by operand identity
/// (object + tensor). Applied inside [`OperandSource::load`] — after
/// widening to f32, before any backend requantization — so **every
/// weight format observes the same effective values** (`load_weight`
/// quantizes from the widened f32 buffer).
#[derive(Debug)]
pub struct OperandOverrides {
    edits: BTreeMap<(String, String), Vec<OperandEdit>>,
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
        self.edits.is_empty()
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

    /// This source's identity, for stamping derived artefacts.
    pub fn stamp(&self) -> SourceStamp {
        SourceStamp {
            store: self.base.id(),
            overlay: self.overrides.map(OperandOverrides::version),
        }
    }

    /// Load one operand as f32, with any overlay edits applied.
    pub fn load(&self, operand: &OperandRef) -> Result<Vec<f32>, VindexError> {
        let mut values = self.base.load(operand)?;
        if let Some(overrides) = self.overrides {
            overrides.apply(operand, &mut values)?;
        }
        Ok(values)
    }

    /// Whether this operand can be held in the checkpoint's own compact
    /// bytes.
    ///
    /// Two conditions, and the second is easy to forget: the container
    /// must store bf16, AND no overlay edit may stand in the way. An edit
    /// is an f32-space fact with no representation in stored bytes, so an
    /// edited operand has to be widened to be honoured at all — see
    /// [`Self::load_raw`], which refuses it.
    ///
    /// False on anything it cannot establish. This decides how many bytes
    /// a weight occupies, never what it means, so an unknown answers
    /// "widen it" and the load path reports any real problem a moment
    /// later with the tensor's name.
    pub fn is_stored_bf16(&self, operand: &OperandRef) -> bool {
        if self.overrides.is_some_and(|o| o.is_overridden(operand)) {
            return false;
        }
        self.base.stored_dtype(operand) == Some(DTYPE_BF16)
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
