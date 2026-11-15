//! Logical edits and whole-operand replacements over a store's operands.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::super::super::OperandRef;
use super::{next_identity, OperandStore, RawOperand, DTYPE_BF16};
use crate::error::VindexError;
use crate::format::vindex3::represent::nvfp4_pack::DTYPE_NVFP4;
use crate::format::vindex3::represent::physical::WeightRegion;

/// Stored bytes per BF16 element.
const BF16_WIDTH: usize = 2;

/// One logical f32 edit to a stored operand (V3-LQL-3B compose): a row
/// or a column replaced by new values. Addressed semantically — the
/// operand's identity plus a slot index — never by byte offsets, so an
/// edit survives repacking or an alternative physical representation.
#[derive(Debug, Clone, PartialEq)]
pub enum OperandEdit {
    Row { index: usize, values: Vec<f32> },
    Column { index: usize, values: Vec<f32> },
}

/// Whole-operand stored bytes standing in for the container's, under the label that says how to read them.
///
/// A replacement is stored-byte state, not an f32 edit: the operand keeps a stored representation, so selection
/// pins it and the loader binds it exactly as it would the same bytes read from a segment.
#[derive(Debug, Clone)]
enum Replacement {
    /// A completed immutable NVFP4 pack, for sequential offline capture.
    Nvfp4Pack {
        shape: Vec<usize>,
        region: WeightRegion,
    },
    /// The tensor's whole BF16 bytes as a weight patch would store them.
    Bf16(PatchedBf16),
}

/// A BF16 replacement, and whether execution has read it.
struct PatchedBf16 {
    shape: Vec<usize>,
    bytes: Arc<[u8]>,
    /// Set when the bytes are handed to a loader. A replacement nothing read was not executed, whatever
    /// the run computed — see [`OperandOverrides::unread_patched`].
    read: AtomicBool,
}

impl std::fmt::Debug for PatchedBf16 {
    /// The tensor's size, never its bytes: a whole matrix does not belong in an assertion message.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PatchedBf16")
            .field("shape", &self.shape)
            .field("bytes", &self.bytes.len())
            .field("read", &self.read.load(Ordering::Relaxed))
            .finish()
    }
}

impl Clone for PatchedBf16 {
    /// The bytes are shared; the witness is not: a clone has been read by nothing.
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            bytes: Arc::clone(&self.bytes),
            read: AtomicBool::new(false),
        }
    }
}

impl Replacement {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Nvfp4Pack { shape, .. } => shape,
            Self::Bf16(patched) => &patched.shape,
        }
    }

    /// The label the replacing bytes are read under.
    fn label(&self) -> &'static str {
        match self {
            Self::Nvfp4Pack { .. } => DTYPE_NVFP4,
            Self::Bf16(_) => DTYPE_BF16,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Nvfp4Pack { region, .. } => region.bytes(),
            Self::Bf16(patched) => &patched.bytes,
        }
    }

    /// The bytes under their label, as the store hands back stored ones. Taking them is a read.
    fn take_raw(&self) -> RawOperand {
        match self {
            Self::Nvfp4Pack { .. } => {}
            Self::Bf16(patched) => patched.read.store(true, Ordering::Relaxed),
        }
        RawOperand {
            dtype: self.label().to_string(),
            bytes: self.bytes().to_vec(),
        }
    }
}

/// Logical edits and whole-operand replacements, keyed by operand identity
/// (object + tensor). Edits are applied inside [`OperandSource::load`](super::OperandSource::load) — after
/// widening to f32, before any backend requantization — so **every
/// weight format observes the same effective values** (`load_weight`
/// quantizes from the widened f32 buffer). A replacement stands in for the stored bytes themselves.
#[derive(Debug)]
pub struct OperandOverrides {
    edits: BTreeMap<(String, String), Vec<OperandEdit>>,
    /// Replacing bytes; decoded only when the operand is loaded.
    replacements: BTreeMap<(String, String), Replacement>,
    /// Process-unique identity, so two override sets are never
    /// mistaken for each other.
    id: u64,
    /// Bumped on every mutation. Together with `id` this is what lets a
    /// derived artefact — a [`PreparedOperands`](super::super::prepared::PreparedOperands)
    /// image — say whether it still describes these edits.
    generation: u64,
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

fn key(operand: &OperandRef) -> (String, String) {
    (operand.object.clone(), operand.tensor.clone())
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
        self.edits.entry(key(operand)).or_default().push(edit);
    }

    pub fn is_overridden(&self, operand: &OperandRef) -> bool {
        self.edits.contains_key(&key(operand))
    }

    /// Install a completed immutable NVFP4 pack for sequential offline capture.
    /// Subsequent logical edits still apply after this replacement.
    pub(crate) fn replace_nvfp4(&mut self, operand: &OperandRef, region: WeightRegion) {
        self.generation += 1;
        self.replacements.insert(
            key(operand),
            Replacement::Nvfp4Pack {
                shape: operand.shape.clone(),
                region,
            },
        );
    }

    /// Replace `operand`'s stored bytes with `bytes`: the whole tensor as BF16, little-endian and row-major,
    /// exactly as a container would store it. Refused unless `bytes` is the length the operand's shape implies.
    ///
    /// The operand is then selected and loaded as a stored BF16 operand holding these bytes, on every path that
    /// reads through an [`OperandSource`](super::OperandSource). A path that reads the store directly does not
    /// see it; [`Self::unread_patched`] says, after a run, whether anything was left unread.
    pub fn replace_bf16(
        &mut self,
        operand: &OperandRef,
        bytes: Arc<[u8]>,
    ) -> Result<(), VindexError> {
        let expected = operand
            .shape
            .iter()
            .try_fold(BF16_WIDTH, |n, &d| n.checked_mul(d));
        if expected != Some(bytes.len()) {
            return Err(VindexError::Parse(format!(
                "`{}/{}`: {} replacing bytes do not hold a BF16 tensor of shape {:?}",
                operand.object,
                operand.tensor,
                bytes.len(),
                operand.shape
            )));
        }
        self.generation += 1;
        self.replacements.insert(
            key(operand),
            Replacement::Bf16(PatchedBf16 {
                shape: operand.shape.clone(),
                bytes,
                read: AtomicBool::new(false),
            }),
        );
        Ok(())
    }

    /// Whether `operand`'s stored bytes are replaced by a weight patch's BF16 bytes.
    pub fn is_patched(&self, operand: &OperandRef) -> bool {
        matches!(
            self.replacements.get(&key(operand)),
            Some(Replacement::Bf16(_))
        )
    }

    /// The `(object, tensor)` of every BF16 replacement no loader has read since this set was made (or cloned).
    /// Empty after an execution that read every patched operand through the operand seam.
    pub fn unread_patched(&self) -> Vec<(String, String)> {
        self.replacements
            .iter()
            .filter_map(|(key, replacement)| match replacement {
                Replacement::Nvfp4Pack { .. } => None,
                Replacement::Bf16(patched) => {
                    (!patched.read.load(Ordering::Relaxed)).then(|| key.clone())
                }
            })
            .collect()
    }

    /// Refuse unless every BF16 replacement was read: the check an execution over patched weights ends with.
    /// An unread one means some path loaded that tensor from the container instead, so what ran was not the
    /// patched model, and its result must not be reported as if it were.
    pub fn ensure_patched_read(&self) -> Result<(), VindexError> {
        let unread = self.unread_patched();
        if unread.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = unread
            .iter()
            .map(|(object, tensor)| format!("{object}/{tensor}"))
            .collect();
        Err(VindexError::Parse(format!(
            "patched weights were not executed: {} loaded from the container, not from the patched bytes",
            names.join(", ")
        )))
    }

    /// The replacement standing on `operand`, checked against the shape execution asks for.
    fn replacement(&self, operand: &OperandRef) -> Result<Option<&Replacement>, VindexError> {
        let Some(replacement) = self.replacements.get(&key(operand)) else {
            return Ok(None);
        };
        if replacement.shape() != operand.shape.as_slice() {
            return Err(VindexError::Parse(
                "candidate replacement shape mismatch".into(),
            ));
        }
        Ok(Some(replacement))
    }

    /// The label and stored length of `operand`'s replacement — metadata only, nothing is read.
    pub(super) fn replacement_facts(&self, operand: &OperandRef) -> Option<(&'static str, u64)> {
        self.replacements
            .get(&key(operand))
            .map(|r| (r.label(), r.bytes().len() as u64))
    }

    /// `operand`'s replacing bytes under their label, as the store would hand back stored ones.
    pub(super) fn replacement_raw(
        &self,
        operand: &OperandRef,
    ) -> Result<Option<RawOperand>, VindexError> {
        Ok(self.replacement(operand)?.map(Replacement::take_raw))
    }

    /// `operand`'s replacement as f32 values. A pack decodes through its codec; BF16 bytes are widened by
    /// `base`'s own decode of a stored operand, so replaced and stored bytes widen in one place.
    pub(super) fn replacement_values(
        &self,
        base: &OperandStore,
        operand: &OperandRef,
    ) -> Result<Option<Vec<f32>>, VindexError> {
        let Some(replacement) = self.replacement(operand)? else {
            return Ok(None);
        };
        match replacement {
            Replacement::Nvfp4Pack { shape, region } => {
                use crate::format::vindex3::represent::codec::{
                    codecs::nvfp4::NVFP4, RepresentationCodec,
                };
                Ok(Some(NVFP4.decode_packed(
                    region.bytes(),
                    shape,
                    NVFP4.terminal_extent(),
                    &operand.tensor,
                )?))
            }
            Replacement::Bf16(_) => base
                .decode_stored(operand, replacement.take_raw())
                .map(Some),
        }
    }

    /// Apply this operand's edits onto its widened f32 values.
    /// Row-major 2-D shape; an edit that does not fit the operand's
    /// declared shape is an error naming the operand — never a silent
    /// partial write.
    pub fn apply(&self, operand: &OperandRef, values: &mut [f32]) -> Result<(), VindexError> {
        let Some(edits) = self.edits.get(&key(operand)) else {
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
