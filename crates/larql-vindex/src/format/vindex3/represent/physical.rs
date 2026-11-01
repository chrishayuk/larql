//! **Physical regions, and composing a model from several of them.**
//!
//! Execution holds a REFERENCE to a physical region; it never owns the
//! bytes. The region owns the backing and its lifetime.
//!
//! ```text
//! PhysicalStore (mmap'd segment | owned bytes)
//!        │
//!        └── WeightRegion { backing, offset, len }
//!                  │
//!                  └── execution binds this
//! ```
//!
//! Two things follow, and the second is why this is a seam rather than a
//! convenience.
//!
//! **A model can be composed from several representation layers without
//! touching the semantic graph.** A sparse candidate overlay supplies
//! the operands its precision map compiled; everything else falls back
//! to the source container. The graph does not know or care:
//!
//! ```text
//! layer 1 expert weights  -> candidate overlay (Q6_K)
//! everything else         -> source container  (BF16)
//! ```
//!
//! That is exactly the shape K3 needs — a cold source plus a hot compact
//! representation plus a resident cache — arrived at here because a
//! quality experiment needed it first.
//!
//! **A region carries the identity of the store it came from.** This
//! codebase has repeatedly been bitten by verifying VALUES and not
//! objects: an aliased buffer decodes to plausible numbers while being
//! the wrong physical thing. So a caller can assert that the operand it
//! believes is executing from the candidate really is, rather than
//! inferring it from the numbers coming out.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::map::{Precision, PrecisionMap};
use super::policy::Role;
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;

mod encoding;
pub use encoding::*;
mod residency;
pub use residency::{page_size, ResidencyReadError};

/// The table entry meaning "this projection has no address for that
/// expert". Mirrors the shader's own
/// `larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT`, kept as a
/// separate constant so the vindex side can read a container without
/// depending on the compute crate.
pub const NOT_ADDRESSABLE: u32 = u32::MAX;

/// **The execution requirement**: the alignment a region's offset must
/// have for a compute backend to bind it zero-copy.
///
/// This is the CONFORMANCE bar — a container meeting it is directly
/// bindable and nothing further is owed. It mirrors
/// `larql_compute_metal::buffers::WEIGHT_BINDING_ALIGN`, which carries
/// the measurement behind the number.
///
/// Distinct from `encode::segment::SEGMENT_PAYLOAD_ALIGN` (16), which
/// is what this project's ENCODER chooses to write. A container
/// aligned to 4 and not to 16 — the Kimi expert segment, at
/// 2,438,284 — is conforming and executes as-is; requiring the
/// encoder's number here would condemn it for no measurable reason.
pub const WEIGHT_BINDING_ALIGN: u64 = 4;

/// A physical store of operand bytes, named so regions can be attributed
/// to it.
pub struct PhysicalStore {
    id: String,
    backing: Backing,
    /// The file this store maps, when it maps one. The id names the
    /// operand OBJECT; this names the physical file, which is what a byte
    /// range read by another process is addressed in.
    source_path: Option<PathBuf>,
    /// The segment's path RELATIVE TO ITS CONTAINER, as the container's
    /// index declares it. Unlike `source_path` it survives the container
    /// being cloned or moved, so it is what a region's identity across
    /// processes is built from. `None` for a store no container declared.
    declared_segment: Option<String>,
    /// Tensor name → (offset from the payload start, length).
    tensors: BTreeMap<String, (u64, u64)>,
    payload_start: u64,
}

enum Backing {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl PhysicalStore {
    /// Map a container segment. The mapping lives as long as the store,
    /// so every region handed out stays valid without copying.
    pub fn map_segment(id: impl Into<String>, path: &Path) -> Result<Self, VindexError> {
        let (header, payload_start) =
            crate::format::vindex3::encode::segment::read_segment_header(path)?;
        let file = std::fs::File::open(path)?;
        // SAFETY: the file is opened read-only and the mapping is owned
        // by this store; regions borrow from it and cannot outlive it.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| VindexError::Parse(format!("{}: mmap failed: {e}", path.display())))?;
        Ok(Self {
            id: id.into(),
            backing: Backing::Mapped(mmap),
            source_path: Some(path.to_path_buf()),
            declared_segment: None,
            tensors: header
                .tensors
                .into_iter()
                .map(|t| (t.name, (t.offset, t.len)))
                .collect(),
            payload_start,
        })
    }

    /// A store over bytes already in memory, laid out by an explicit
    /// table — a compiled bank whose offsets came from its layout, or a
    /// fixture.
    pub fn owned(
        id: impl Into<String>,
        bytes: Vec<u8>,
        tensors: BTreeMap<String, (u64, u64)>,
    ) -> Self {
        Self {
            id: id.into(),
            backing: Backing::Owned(bytes),
            source_path: None,
            declared_segment: None,
            tensors,
            payload_start: 0,
        }
    }

    /// Map a compiled bank, taking its layout from a ledger rather than
    /// a segment header — the candidate overlay's own shape.
    pub fn map_compiled(
        id: impl Into<String>,
        path: &Path,
        ledger: &super::compile::CompilationLedger,
    ) -> Result<Self, VindexError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: as above.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| VindexError::Parse(format!("{}: mmap failed: {e}", path.display())))?;
        Ok(Self {
            id: id.into(),
            backing: Backing::Mapped(mmap),
            source_path: Some(path.to_path_buf()),
            declared_segment: None,
            tensors: ledger
                .sealed
                .values()
                .map(|s| (s.tensor.clone(), (s.target_offset, s.target_len)))
                .collect(),
            payload_start: 0,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// The file this store maps, or `None` for owned bytes.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// This store, recorded as the segment its container declares at
    /// `segment` (a path relative to the container root).
    pub fn declared_as(mut self, segment: impl Into<String>) -> Self {
        self.declared_segment = Some(segment.into());
        self
    }

    /// The container-relative segment path the container declared for
    /// this store, or `None` when no container declared one.
    pub fn declared_segment(&self) -> Option<&str> {
        self.declared_segment.as_deref()
    }

    fn all(&self) -> &[u8] {
        match &self.backing {
            Backing::Mapped(m) => &m[..],
            Backing::Owned(v) => &v[..],
        }
    }

    pub fn holds(&self, tensor: &str) -> bool {
        self.tensors.contains_key(tensor)
    }

    /// A region over one whole tensor of this store.
    ///
    /// The direct route for a caller that already knows which operand it
    /// wants and does not need a precision map to decide.
    pub fn whole(self: &Arc<Self>, tensor: &str) -> Option<WeightRegion> {
        self.region(tensor)
    }

    /// A region over an arbitrary span of this store's payload.
    ///
    /// Needed because a source segment's expert bank is addressed as
    /// three SHIFTED VIEWS of one mapping rather than as three named
    /// tensors — the projections of one expert are contiguous, so a
    /// single per-expert base table serves all three once each view
    /// starts at its own projection.
    pub fn span(self: &Arc<Self>, offset: u64, len: u64) -> Option<WeightRegion> {
        let start = self.payload_start + offset;
        (start + len <= self.all().len() as u64).then(|| WeightRegion {
            store: self.clone(),
            offset: start,
            len,
        })
    }

    /// Bytes of payload after the header.
    pub fn payload_len(&self) -> u64 {
        self.all().len() as u64 - self.payload_start
    }

    /// Where the payload starts inside the backing allocation.
    pub fn payload_start(&self) -> u64 {
        self.payload_start
    }

    /// The WHOLE backing allocation, header included.
    ///
    /// For registering the store with a compute backend's zero-copy
    /// region table: an mmap's base pointer is page-aligned by
    /// construction, while any payload span generally is not — so the
    /// registration slice must be cut from this allocation at a
    /// page boundary, not from a `WeightRegion`. Not for reading
    /// operands; regions stay the only sanctioned view of the payload.
    pub fn backing_bytes(&self) -> &[u8] {
        self.all()
    }

    pub(crate) fn region(self: &Arc<Self>, tensor: &str) -> Option<WeightRegion> {
        let (offset, len) = *self.tensors.get(tensor)?;
        let start = self.payload_start + offset;
        (start + len <= self.all().len() as u64).then(|| WeightRegion {
            store: self.clone(),
            offset: start,
            len,
        })
    }
}

/// A bound reference to physical bytes.
#[derive(Clone)]
pub struct WeightRegion {
    store: Arc<PhysicalStore>,
    offset: u64,
    len: u64,
}

impl std::fmt::Debug for WeightRegion {
    /// Names the STORE and the extent, never the bytes: a region can be
    /// gigabytes, and what a reader needs from a failure is which
    /// physical thing was bound.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WeightRegion({} @ {}+{})",
            self.store.id(),
            self.offset,
            self.len
        )
    }
}

impl WeightRegion {
    pub fn bytes(&self) -> &[u8] {
        &self.store.all()[self.offset as usize..(self.offset + self.len) as usize]
    }

    /// Which physical store these bytes are in.
    ///
    /// The assertion a quality run needs: not "the numbers differ" but
    /// "the operand I believe is executing from the candidate really is".
    pub fn store_id(&self) -> &str {
        self.store.id()
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes of this region whose pages are physically resident NOW, as
    /// the OS reports them — the committed half of a mapping, distinct
    /// from its address space — or why that could not be read. A failure
    /// is never a zero.
    pub fn residency(&self) -> Result<u64, ResidencyReadError> {
        residency::resident_bytes_of(self.bytes())
    }

    /// [`Self::residency`] with the reason dropped: `None` where the OS
    /// offers no report, never a guess.
    pub fn resident_bytes(&self) -> Option<u64> {
        self.residency().ok()
    }

    /// The file this region's store maps, or `None` for owned bytes.
    pub fn source_path(&self) -> Option<&Path> {
        self.store.source_path()
    }

    /// The container-relative segment path of this region's store, when
    /// a container declared one.
    pub fn declared_segment(&self) -> Option<&str> {
        self.store.declared_segment()
    }

    /// Byte offset of these bytes within their store's backing
    /// allocation. For a mapped store the backing is the whole file,
    /// header included, so this is a FILE offset.
    ///
    /// The number a zero-copy binding resolves to, and therefore the
    /// one whose ALIGNMENT decides whether this region can be bound at
    /// all — see [`ExpertBankBinding::validate`].
    pub fn store_offset(&self) -> u64 {
        self.offset
    }

    /// Whether this region can be bound zero-copy by a backend that
    /// requires `align`-byte offsets.
    pub fn is_bindable_at(&self, align: u64) -> bool {
        self.offset.is_multiple_of(align)
    }
}

/// How many operands came from where.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResolutionStats {
    /// Served by the candidate overlay, as its precision map intended.
    pub candidate_hits: u64,
    /// Fell back to the source container at source precision.
    pub source_fallback_hits: u64,
    /// The map said COMPILED and the overlay did not hold it. Never
    /// silently served from source: that would execute BF16 while every
    /// record claimed Q6_K.
    pub missing: u64,
}

/// A model composed from a candidate overlay over a source container.
pub struct LayeredOperands {
    map: PrecisionMap,
    candidate: Arc<PhysicalStore>,
    source: Arc<PhysicalStore>,
    /// What the SOURCE container's bytes are. A representation carries
    /// one encoding throughout — `target.expert_bank@BF16` says so in
    /// its own id — so this is a property of the container, not a guess.
    source_encoding: ExpertEncoding,
    stats: std::sync::Mutex<ResolutionStats>,
}

impl LayeredOperands {
    pub fn new(
        map: PrecisionMap,
        candidate: Arc<PhysicalStore>,
        source: Arc<PhysicalStore>,
    ) -> Self {
        Self::with_source_encoding(map, candidate, source, ExpertEncoding::Bf16)
    }

    pub fn with_source_encoding(
        map: PrecisionMap,
        candidate: Arc<PhysicalStore>,
        source: Arc<PhysicalStore>,
        source_encoding: ExpertEncoding,
    ) -> Self {
        Self {
            map,
            candidate,
            source,
            source_encoding,
            stats: std::sync::Mutex::new(ResolutionStats::default()),
        }
    }

    pub fn stats(&self) -> ResolutionStats {
        *self.stats.lock().unwrap()
    }

    /// The region this arm executes for `operand`.
    ///
    /// The precision map decides WHICH layer answers, not availability:
    /// an operand the map compiled but the overlay lacks is an error,
    /// because falling back would run source bytes under a compiled
    /// name and make the whole evidence chain a lie.
    pub fn resolve(&self, role: Role, operand: &OperandRef) -> Result<EncodedRegion, VindexError> {
        match self.map.resolve(role, &operand.tensor) {
            Precision::Compiled(enc) => match self.candidate.region(&operand.tensor) {
                Some(r) => {
                    self.stats.lock().unwrap().candidate_hits += 1;
                    // The MAP is the authority for what these bytes are.
                    // A caller never declares it, so the declaration
                    // cannot drift from the decision.
                    let encoding = ExpertEncoding::parse(enc).ok_or_else(|| {
                        VindexError::Parse(format!(
                            "map `{}` names encoding `{enc}`, which no grouped kernel reads",
                            self.map.name
                        ))
                    })?;
                    Ok(EncodedRegion {
                        region: r,
                        encoding,
                    })
                }
                None => {
                    self.stats.lock().unwrap().missing += 1;
                    Err(VindexError::Parse(format!(
                        "`{}` is compiled as {enc} by map `{}` but the candidate overlay does \
                         not hold it — refusing to fall back to source bytes under a \
                         compiled name",
                        operand.tensor, self.map.name
                    )))
                }
            },
            Precision::Source => match self.source.region(&operand.tensor) {
                Some(r) => {
                    self.stats.lock().unwrap().source_fallback_hits += 1;
                    Ok(EncodedRegion {
                        region: r,
                        encoding: self.source_encoding,
                    })
                }
                None => {
                    self.stats.lock().unwrap().missing += 1;
                    Err(VindexError::Parse(format!(
                        "`{}` is in neither the candidate overlay nor the source container",
                        operand.tensor
                    )))
                }
            },
        }
    }
}

/// How an expert's SEMANTIC id maps to its PHYSICAL slot in a bank.
///
/// The two are not the same fact, and conflating them is what makes a
/// packed fixture bank and a compiled full bank look like different
/// execution paths when they are one path with different layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertLayout {
    /// `expert_id == physical_slot`: a full execution-shaped bank
    /// holding every expert. What a compiled VINDEX writes, and what a
    /// K3 cold bank will be.
    Identity { experts: u32 },
    /// A packed subset: physical slot `i` holds expert `ids[i]`. The
    /// existing resident-union bank, and what a hot runtime cache over a
    /// large cold bank would be.
    ///
    /// A runtime VIEW, never the persistent format's ontology.
    Mapped { ids: Vec<u32> },
}

impl ExpertLayout {
    /// The physical slot holding `expert`, or `None` if this bank does
    /// not hold it.
    ///
    /// `None` is a real answer for `Mapped` — a packed bank genuinely
    /// holds a subset — and impossible for `Identity` within range,
    /// which is why a compiled bank makes route escape trivial.
    pub fn slot_of(&self, expert: u32) -> Option<u32> {
        match self {
            ExpertLayout::Identity { experts } => (expert < *experts).then_some(expert),
            ExpertLayout::Mapped { ids } => {
                ids.iter().position(|id| *id == expert).map(|i| i as u32)
            }
        }
    }

    pub fn slots(&self) -> usize {
        match self {
            ExpertLayout::Identity { experts } => *experts as usize,
            ExpertLayout::Mapped { ids } => ids.len(),
        }
    }
}

/// What a bank-pricing refusal names as its operand: a bank is priced
/// per projection, not per named tensor.
const EXPERT_BANK_OPERAND: &str = "expert-bank";

/// A physical representation a grouped kernel can execute.
///
/// The backend's job is to answer whether it can run one of these, never
/// to choose it — backend support is capability, not authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertEncoding {
    Bf16,
    /// Canonical ggml Q8_0: 34-byte blocks of 32 (f16 scale + 32 int8),
    /// 8.5 bpw — the precision ladder's rung between BF16 and Q6_K.
    Q80,
    Q6K,
    Q4K,
}

#[cfg(test)]
#[path = "physical_tests.rs"]
mod tests;
