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
use std::path::Path;
use std::sync::Arc;

use super::map::{Precision, PrecisionMap};
use super::policy::Role;
use crate::error::VindexError;
use crate::format::vindex3::opplan::OperandRef;

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

    fn region(self: &Arc<Self>, tensor: &str) -> Option<WeightRegion> {
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

    /// Byte offset of these bytes within their store's backing
    /// allocation.
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

/// A physical representation a grouped kernel can execute.
///
/// The backend's job is to answer whether it can run one of these, never
/// to choose it — backend support is capability, not authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertEncoding {
    Bf16,
    Q6K,
    Q4K,
}

impl ExpertEncoding {
    pub fn name(self) -> &'static str {
        match self {
            ExpertEncoding::Bf16 => "BF16",
            ExpertEncoding::Q6K => "Q6_K",
            ExpertEncoding::Q4K => "Q4_K",
        }
    }

    /// The encoding a precision map named, or `None` if no grouped
    /// kernel reads it — refused rather than approximated.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(ExpertEncoding::Bf16),
            "Q6_K" => Some(ExpertEncoding::Q6K),
            "Q4_K" => Some(ExpertEncoding::Q4K),
            _ => None,
        }
    }

    /// Bytes an `[n, k]` matrix occupies in this encoding.
    pub fn matrix_bytes(self, n: usize, k: usize) -> Result<u64, VindexError> {
        match self {
            ExpertEncoding::Bf16 => Ok((n * k) as u64 * 2),
            ExpertEncoding::Q6K | ExpertEncoding::Q4K => {
                if !k.is_multiple_of(256) {
                    return Err(VindexError::Parse(format!(
                        "k={k} is not a whole number of 256-element superblocks for {}",
                        self.name()
                    )));
                }
                let per = if self == ExpertEncoding::Q6K {
                    210
                } else {
                    144
                };
                Ok((n * k / 256) as u64 * per)
            }
        }
    }
}

/// A region together with what its bytes ARE.
///
/// Per projection, not per bank, because a precision map can already
/// name `gate/up at Q6_K, down at BF16` — the scope vocabulary supports
/// it, so the physical vocabulary must too or the next experiment
/// changes this type again.
///
/// The encoding is not a caller's declaration: it comes from the
/// resolver, which knows the precision map that decided it. The map says
/// Q6_K, the resolver returns Q6_K bytes, the binding says Q6_K, and the
/// kernel follows that one fact.
#[derive(Clone)]
pub struct EncodedRegion {
    pub region: WeightRegion,
    pub encoding: ExpertEncoding,
}

impl std::fmt::Debug for EncodedRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} as {}", self.region, self.encoding.name())
    }
}

impl EncodedRegion {
    /// Which physical store these bytes are in.
    pub fn store_id(&self) -> &str {
        self.region.store_id()
    }

    /// Refuse a region too small to hold what it claims to be.
    ///
    /// The execution analogue of the no-silent-fallback rule: bytes that
    /// are BF16 dispatched as Q6_K would decode to plausible garbage,
    /// and the failure would read as "quantisation is catastrophic"
    /// rather than "the wrong kernel ran". A size check catches every
    /// mismatch where the declared encoding is LARGER than the bytes;
    /// the reverse is caught by [`ExpertBankBinding::validate`], which
    /// knows the bank's extent.
    pub fn check_room(&self, top_offset: u64, n: usize, k: usize) -> Result<(), VindexError> {
        let need = top_offset + self.encoding.matrix_bytes(n, k)?;
        if need > self.region.len() {
            return Err(VindexError::Parse(format!(
                "a {} bank of [{n}, {k}] needs {need} bytes to reach its last expert, but the                  bound region ({:?}) is {} — the bytes are not what this encoding claims",
                self.encoding.name(),
                self.region,
                self.region.len()
            )));
        }
        Ok(())
    }
}

/// The shared expert's three projections, each its own region under its
/// own encoding.
///
/// A separate binding from the routed bank because `Shared` vs `Routed`
/// is SEMANTIC identity and must not imply physical co-location: a
/// source container keeps the shared expert in the decoder stack while
/// the routed experts live in an expert bank, and a candidate overlay
/// may compile the routed bank to Q6_K while the shared branch stays
/// source BF16. Placing the shared bytes next to the routed ones is a
/// layout an artifact MAY choose — the regions can be subranges of one
/// store — never something execution may assume.
#[derive(Clone)]
pub struct SharedExpertBinding {
    pub gate: EncodedRegion,
    pub up: EncodedRegion,
    pub down: EncodedRegion,
}

/// **Logical expert id → byte coordinate, for ONE projection.**
///
/// The owned form of the vocabulary the grouped kernel already speaks
/// (`larql_compute_metal::trait_impl::kimi_layer::ExpertAddressing`,
/// which borrows). One per projection rather than one per bank, and
/// that is not tidiness: a projection-scoped candidate needs `gate`
/// addressed by IDENTITY over a compiled Q6_K bank while `up` and
/// `down` stay TABLE-addressed over the arbitrarily-ordered source
/// segment, in the same layer, in the same forward pass. Metal could
/// express that from rung C onward; a bank-wide layout could not, and
/// a projection sweep is what proved the gap real rather than tidy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionAddressing {
    /// `offset = expert_id * stride`. Nothing is tabulated, and no
    /// selection can be unaddressable — a compiled full bank.
    Identity { experts: u32, stride: u32 },
    /// Byte offset per expert, indexed by logical expert id. What an
    /// arbitrarily-ordered source segment or a packed subset needs.
    Table(Vec<u32>),
}

impl ProjectionAddressing {
    /// How many experts this projection can address.
    pub fn experts(&self) -> u32 {
        match self {
            Self::Identity { experts, .. } => *experts,
            Self::Table(t) => t.len() as u32,
        }
    }

    /// The constant per-expert stride, when this projection has one.
    ///
    /// `None` for a table: a tabulated projection's entries need not be
    /// evenly spaced, and inventing a stride from two of them would be
    /// a claim about the layout nobody made.
    pub fn identity_stride(&self) -> Option<u32> {
        match self {
            Self::Identity { stride, .. } => Some(*stride),
            Self::Table(_) => None,
        }
    }

    /// The highest byte offset this projection can be asked to read
    /// from, or `None` if it addresses nothing.
    ///
    /// The quantity an extent check needs, and NOT the entry count: a
    /// table has one entry per SCORED expert while the bank behind it
    /// holds only the addressable subset — the fixture's is 256 entries
    /// over 65 blocks. Sizing the check by entries demanded a bank four
    /// times the size of the real one and refused a valid binding.
    pub fn max_offset(&self) -> Option<u64> {
        match self {
            Self::Identity { experts, stride } => {
                (*experts > 0).then(|| u64::from(experts - 1) * u64::from(*stride))
            }
            Self::Table(t) => t
                .iter()
                .filter(|o| **o != NOT_ADDRESSABLE)
                .map(|o| u64::from(*o))
                .max(),
        }
    }

    /// Byte offset of `expert`'s payload, or `None` when this projection
    /// cannot address it at all.
    pub fn offset_of(&self, expert: u32) -> Option<u64> {
        match self {
            Self::Identity { experts, stride } => {
                (expert < *experts).then(|| u64::from(expert) * u64::from(*stride))
            }
            Self::Table(t) => t
                .get(expert as usize)
                .copied()
                .filter(|o| *o != NOT_ADDRESSABLE)
                .map(u64::from),
        }
    }
}

/// One projection of a routed bank: its bytes, how they are addressed,
/// and whether the region IS the bank or a window onto a larger one.
///
/// The three travel together because they are one fact — where this
/// projection's bytes came from. A compiled candidate is `Exact` and
/// addressed by identity; a source view is a `ContainingView` addressed
/// by table. Splitting them across the bank, as this type replaced,
/// made a mixed binding inexpressible: an experiment compiling gate to
/// Q6_K while up and down stayed source-backed had to declare the whole
/// bank a `ContainingView`, which silently disabled the surplus-byte
/// check on the one projection under test — the only check that catches
/// BF16 bytes mislabelled as a smaller encoding.
#[derive(Clone)]
pub struct RoutedProjection {
    pub region: EncodedRegion,
    pub addressing: ProjectionAddressing,
    pub extent: ExtentPolicy,
}

impl RoutedProjection {
    /// Which physical store these bytes are in — the attribution a
    /// quality run asserts rather than infers from the numbers.
    pub fn store_id(&self) -> &str {
        self.region.store_id()
    }

    /// What these bytes ARE.
    pub fn encoding(&self) -> ExpertEncoding {
        self.region.encoding
    }
}

/// One layer's expert bank: three independently-bound routed
/// projections and — independently again — the shared expert's binding.
///
/// Deliberately one level above `DeviceLayer`, so execution never infers
/// addressing from the fact that it happens to hold regions. The same
/// binding covers an owned packed fixture, an mmap'd compiled bank, a
/// candidate overlay over some projections only, and eventually a
/// resident view over a K3 cold bank — one code path, several physical
/// stories.
///
/// Nothing physical is left at bank level. The only bank-wide fact is
/// the semantic one: these three projections constitute one routed MoE.
#[derive(Clone)]
pub struct ExpertBankBinding {
    pub gate: RoutedProjection,
    pub up: RoutedProjection,
    pub down: RoutedProjection,
    /// The shared expert, when the architecture declares one. `None` is
    /// a claim that the model HAS no shared branch, not that its bytes
    /// were not found — a loader that cannot find a declared shared
    /// expert must refuse, never construct a `None`.
    pub shared: Option<SharedExpertBinding>,
}

/// Why a region may be larger than the bank it addresses.
///
/// Two very different facts would otherwise be indistinguishable:
/// *these regions ARE the bank* (a compiled overlay), and *these regions
/// are windows onto a much larger segment* (a source container view).
/// Only the first can conclude that surplus bytes mean the declared
/// encoding is wrong — and that conclusion is the only thing that
/// catches BF16 bytes mislabelled Q6_K, since those are LARGER than the
/// claim and every room check passes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtentPolicy {
    /// The regions are exactly this bank. Surplus bytes are a defect.
    Exact,
    /// The regions are windows onto a larger backing. Surplus bytes are
    /// expected and say nothing about the encoding.
    ContainingView,
}

impl ExpertBankBinding {
    /// Byte offset of `expert`'s payload within the gate bank.
    ///
    /// Reads gate's OWN addressing. The stride is no longer a parameter
    /// because it is no longer a caller's guess: it belongs to the
    /// projection, and a caller that supplied a sibling's would have
    /// been believed.
    pub fn gate_offset(&self, expert: u32) -> Option<u64> {
        self.gate.addressing.offset_of(expert)
    }

    pub fn up_offset(&self, expert: u32) -> Option<u64> {
        self.up.addressing.offset_of(expert)
    }

    pub fn down_offset(&self, expert: u32) -> Option<u64> {
        self.down.addressing.offset_of(expert)
    }

    /// The three projections, labelled, for checks that must cover all
    /// of them. `(label, projection, n, k)` — down is the transpose.
    fn projections(
        &self,
        hidden: usize,
        inter: usize,
    ) -> [(&'static str, &RoutedProjection, usize, usize); 3] {
        [
            ("routed gate", &self.gate, inter, hidden),
            ("routed up", &self.up, inter, hidden),
            ("routed down", &self.down, hidden, inter),
        ]
    }

    /// Which physical store backs the gate projection.
    ///
    /// Gate's, not "the bank's": after per-projection binding the three
    /// may legitimately come from different stores, and a single answer
    /// is only meaningful where they agree. [`Self::stores_agree`] is
    /// the question to ask when that matters.
    pub fn store_id(&self) -> &str {
        self.gate.region.region.store_id()
    }

    /// Whether all three projections are backed by the same store.
    ///
    /// False is legitimate, not a defect: an experiment that compiles
    /// one projection to a candidate store and leaves the other two
    /// source-backed is exactly the asymmetry this binding exists to
    /// express. Callers that need a single provenance answer must ask
    /// this before believing [`Self::store_id`].
    pub fn stores_agree(&self) -> bool {
        let g = self.gate.region.region.store_id();
        g == self.up.region.region.store_id() && g == self.down.region.region.store_id()
    }

    /// Every projection has room for every addressable expert at the
    /// encoding it claims.
    ///
    /// [`ExtentPolicy::Exact`] additionally refuses a region LARGER than
    /// the encoding implies; a [`ExtentPolicy::ContainingView`] cannot
    /// make that claim and checks room only.
    pub fn validate(&self, hidden: usize, inter: usize) -> Result<(), VindexError> {
        // Extent is read PER PROJECTION. A mixed binding — a candidate
        // gate compiled Exact beside source-backed up/down windows — is
        // the whole point of the per-projection split, and the exact
        // check must still bite on the compiled one. Reading a single
        // bank-wide flag here meant one source-backed sibling disabled
        // the check on every projection, including the one under test.
        for (what, proj, n, k) in self.projections(hidden, inter) {
            let enc = &proj.region;
            // The highest OFFSET this projection can be asked to read,
            // from its own addressing. Not the expert count: a table
            // has an entry per scored expert while its bank holds only
            // the addressable subset.
            let top = proj.addressing.max_offset().unwrap_or(0);
            let per = enc.encoding.matrix_bytes(n, k)?;
            enc.check_room(top, n, k)?;
            if proj.extent == ExtentPolicy::Exact {
                let want = top + per;
                if enc.region.len() != want {
                    return Err(VindexError::Parse(format!(
                        "{what}: a {} bank reaching offset {top} at [{n}, {k}] is {want} bytes,                          but the bound region ({:?}) is {} — the bytes are not this encoding",
                        enc.encoding.name(),
                        enc.region,
                        enc.region.len()
                    )));
                }
            }
        }
        // The shared expert's regions, each under its own encoding.
        // Room-checked only: a shared region is typically a whole named
        // tensor or a window into a store whose extent semantics belong
        // to that store, so surplus bytes say nothing here.
        if let Some(shared) = &self.shared {
            for (enc, n, k) in [
                (&shared.gate, inter, hidden),
                (&shared.up, inter, hidden),
                (&shared.down, hidden, inter),
            ] {
                enc.check_room(0, n, k)?;
            }
        }
        self.check_bindable()
    }

    /// Every region sits at an offset a backend can bind zero-copy.
    ///
    /// A segment whose payload starts at an odd byte puts every tensor
    /// in it at an odd address. A backend that binds the mapping
    /// zero-copy then hands its kernel a misaligned pointer — on Metal,
    /// a `device const ushort*` at an odd address reads garbage and the
    /// command buffer still reports success. The backend declines such
    /// a binding and stages a copy instead, which is correct but means
    /// silently copying gigabytes per dispatch, so the condition is
    /// refused HERE, where the cause can be named.
    ///
    /// Measured: the Kimi container's `decoder_stack` payload began at
    /// 56,925, and every dense/shared-expert dispatch returned NaN.
    fn check_bindable(&self) -> Result<(), VindexError> {
        let mut regions: Vec<(&str, &EncodedRegion)> = vec![
            ("routed gate", &self.gate.region),
            ("routed up", &self.up.region),
            ("routed down", &self.down.region),
        ];
        if let Some(s) = &self.shared {
            regions.extend([
                ("shared gate", &s.gate),
                ("shared up", &s.up),
                ("shared down", &s.down),
            ]);
        }
        for (what, r) in regions {
            if !r.region.is_bindable_at(WEIGHT_BINDING_ALIGN) {
                return Err(VindexError::Parse(format!(
                    "{what} sits at byte {} of {:?}, which is not a multiple of                      {WEIGHT_BINDING_ALIGN} — a backend cannot bind it zero-copy, and the                      usual cause is a segment whose payload does not start on an aligned                      boundary. Re-write the container's segments with an aligned payload                      (`encode::segment::SEGMENT_PAYLOAD_ALIGN`); the payload bytes and their                      hash do not change.",
                    r.region.store_offset(),
                    r.region
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "physical_tests.rs"]
mod tests;
