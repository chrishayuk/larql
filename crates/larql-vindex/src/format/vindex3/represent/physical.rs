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

/// One layer's expert bank: three routed regions, the layout that
/// addresses them, and — independently — the shared expert's binding.
///
/// Deliberately one level above `DeviceLayer`, so execution never infers
/// a layout from the fact that it happens to hold regions. The same
/// binding covers an owned packed fixture, an mmap'd compiled bank, and
/// eventually a resident view over a K3 cold bank — one code path, three
/// physical stories.
#[derive(Clone)]
pub struct ExpertBankBinding {
    pub gate: EncodedRegion,
    pub up: EncodedRegion,
    pub down: EncodedRegion,
    pub layout: ExpertLayout,
    pub extent: ExtentPolicy,
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
    /// Byte offset of `expert`'s payload within the gate/up banks, given
    /// the per-expert stride.
    pub fn gate_up_offset(&self, expert: u32, stride: u64) -> Option<u64> {
        self.layout.slot_of(expert).map(|s| u64::from(s) * stride)
    }

    pub fn down_offset(&self, expert: u32, stride: u64) -> Option<u64> {
        self.layout.slot_of(expert).map(|s| u64::from(s) * stride)
    }

    /// Which physical store backs this bank — the assertion that the
    /// intended bytes are the ones executing.
    pub fn store_id(&self) -> &str {
        self.gate.region.store_id()
    }

    /// Every projection has room for every addressable expert at the
    /// encoding it claims.
    ///
    /// [`ExtentPolicy::Exact`] additionally refuses a region LARGER than
    /// the encoding implies; a [`ExtentPolicy::ContainingView`] cannot
    /// make that claim and checks room only.
    pub fn validate(&self, hidden: usize, inter: usize) -> Result<(), VindexError> {
        let exact_bank = self.extent == ExtentPolicy::Exact;
        // The highest PHYSICAL slot, which is one less than the slot
        // count for both layouts. Deriving it by mapping `0..slots()`
        // through `slot_of` was wrong for `Mapped`, whose entries are
        // arbitrary expert IDS rather than indices: it found only the
        // ids that happened to be small and under-counted the bank.
        // Routed slots only: the shared expert is its own binding with
        // its own regions, never a block appended past the routed ones.
        let blocks = self.layout.slots().max(1);
        let top = (blocks - 1) as u32;
        for (enc, n, k) in [
            (&self.gate, inter, hidden),
            (&self.up, inter, hidden),
            (&self.down, hidden, inter),
        ] {
            let per = enc.encoding.matrix_bytes(n, k)?;
            enc.check_room(u64::from(top) * per, n, k)?;
            if exact_bank {
                let want = (u64::from(top) + 1) * per;
                if enc.region.len() != want {
                    return Err(VindexError::Parse(format!(
                        "a {} bank of {} experts at [{n}, {k}] is {want} bytes, but the bound                          region ({:?}) is {} — the bytes are not this encoding",
                        enc.encoding.name(),
                        top + 1,
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
        Ok(())
    }
}

#[cfg(test)]
#[path = "physical_tests.rs"]
mod tests;
