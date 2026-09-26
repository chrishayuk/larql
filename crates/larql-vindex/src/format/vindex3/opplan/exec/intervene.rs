//! V3-INTERVENE-1: one carrier changed at one declared address, and
//! nothing else.
//!
//! An intervention is a declared object — an address `(layer, site,
//! positions)`, a kind (`Zero`, `Add`, `Replace`) and, for the two kinds
//! that carry a vector, that vector's provenance and SHA-256. It threads
//! through the one decode step beside the defect vocabulary
//! (`controls::Mutation`), never as a variant of it: a defect exists so a
//! witness can FAIL; an intervention exists so a claim can be TESTED.
//!
//! What it acts on is the `Single` carrier as it leaves the addressed
//! site — the vector `carrier_write` reports as `after`, before any
//! layer scale. `Bundle` and `History` carriers refuse at admission. The
//! arithmetic is the backend's own residual path: `Add(v)` is one
//! `residual_add` on the written carrier, so `Add(0)` is the unintervened
//! run bit for bit; `Replace(v)` copies; `Zero` fills. At an intervened
//! site the write record's `delta` is `after − before` so that it still
//! includes what landed; the `Intervened` event precedes it so a reader
//! knows the `after` is not the branch's own.
//!
//! Nothing here persists a vector. A vector arrives with where it came
//! from — a caller literal, or a carrier captured from a named run at a
//! named address — and its hash. [`CarrierCapture`] is the observer that
//! sources one from another run, in memory.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::super::ComponentOpPlan;
use super::backend::PlanBackend;
use super::observe::{CarrierWriteRecord, StepEvent, StepObserver, SublayerSite};
use super::prepared::PreparedOperands;
use crate::error::VindexError;

/// What an intervention does to the carrier at its address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InterventionKind {
    /// The carrier becomes zero.
    Zero,
    /// The carrier becomes `after + v`, through the backend's residual add.
    Add,
    /// The carrier becomes `v`.
    Replace,
}

impl InterventionKind {
    /// The name a record spells this kind with.
    pub fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Add => "add",
            Self::Replace => "replace",
        }
    }
}

/// Where a vector came from. The vector never travels with this; its
/// hash does, so a receipt can say which bytes were patched in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorProvenance {
    /// The caller supplied the vector.
    Literal { sha256: String },
    /// The vector is a carrier captured from another run at an address.
    Captured {
        run_id: String,
        layer: usize,
        site: SublayerSite,
        position: usize,
        sha256: String,
    },
    /// V3-INTERVENE-2: the vector is a query head's `ctx_h` captured
    /// from another run at a `(layer, head, position)` address — J7's
    /// head form of [`Self::Captured`].
    CapturedHead {
        run_id: String,
        layer: usize,
        head: usize,
        position: usize,
        sha256: String,
    },
}

impl VectorProvenance {
    /// The hash the provenance claims for its bytes.
    pub fn sha256(&self) -> &str {
        match self {
            Self::Literal { sha256 }
            | Self::Captured { sha256, .. }
            | Self::CapturedHead { sha256, .. } => sha256,
        }
    }
}

/// SHA-256 over a vector's exact f32 little-endian bytes, prefixed by its
/// width, as lowercase hex — the same discipline as a projection basis.
pub fn vector_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    let bytes: [u8; 32] = hasher.finalize().into();
    bytes.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// One site of one layer, at declared positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Address {
    pub layer: usize,
    pub site: SublayerSite,
    positions: BTreeSet<usize>,
}

impl Address {
    /// An address with at least one position; an empty one names nothing.
    pub fn new(
        layer: usize,
        site: SublayerSite,
        positions: impl IntoIterator<Item = usize>,
    ) -> Result<Self, VindexError> {
        let positions: BTreeSet<usize> = positions.into_iter().collect();
        if positions.is_empty() {
            return Err(VindexError::Parse(format!(
                "intervention at layer {layer} {site:?} names no position"
            )));
        }
        Ok(Self {
            layer,
            site,
            positions,
        })
    }

    /// The positions, ascending.
    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.positions.iter().copied()
    }

    /// Whether the address fires at this position.
    pub fn covers(&self, position: usize) -> bool {
        self.positions.contains(&position)
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.layer == other.layer
            && self.site == other.site
            && self.positions.iter().any(|p| other.positions.contains(p))
    }
}

/// One declared change to the carrier.
#[derive(Debug, Clone, PartialEq)]
pub struct Intervention {
    address: Address,
    kind: InterventionKind,
    vector: Option<Vec<f32>>,
    provenance: Option<VectorProvenance>,
}

impl Intervention {
    /// The carrier becomes zero at the address.
    pub fn zero(address: Address) -> Self {
        Self {
            address,
            kind: InterventionKind::Zero,
            vector: None,
            provenance: None,
        }
    }

    /// The carrier becomes `after + vector` at the address.
    pub fn add(
        address: Address,
        vector: Vec<f32>,
        provenance: VectorProvenance,
    ) -> Result<Self, VindexError> {
        Self::with_vector(address, InterventionKind::Add, vector, provenance)
    }

    /// The carrier becomes `vector` at the address.
    pub fn replace(
        address: Address,
        vector: Vec<f32>,
        provenance: VectorProvenance,
    ) -> Result<Self, VindexError> {
        Self::with_vector(address, InterventionKind::Replace, vector, provenance)
    }

    fn with_vector(
        address: Address,
        kind: InterventionKind,
        vector: Vec<f32>,
        provenance: VectorProvenance,
    ) -> Result<Self, VindexError> {
        if vector.is_empty() {
            return Err(VindexError::Parse(format!(
                "intervention `{}` at layer {} {:?} carries an empty vector",
                kind.name(),
                address.layer,
                address.site
            )));
        }
        if let Some(i) = vector.iter().position(|v| !v.is_finite()) {
            return Err(VindexError::Parse(format!(
                "intervention `{}` at layer {} {:?} carries a non-finite value at index {i}",
                kind.name(),
                address.layer,
                address.site
            )));
        }
        let actual = vector_sha256(&vector);
        if actual != provenance.sha256() {
            return Err(VindexError::Parse(format!(
                "intervention `{}` at layer {} {:?}: the vector's bytes hash to {actual} but its \
                 provenance claims {} — the bytes handed over are not the bytes declared",
                kind.name(),
                address.layer,
                address.site,
                provenance.sha256()
            )));
        }
        Ok(Self {
            address,
            kind,
            vector: Some(vector),
            provenance: Some(provenance),
        })
    }

    pub fn address(&self) -> &Address {
        &self.address
    }

    pub fn kind(&self) -> InterventionKind {
        self.kind
    }

    /// Where the vector came from; `None` for `Zero`, which carries none.
    pub fn provenance(&self) -> Option<&VectorProvenance> {
        self.provenance.as_ref()
    }

    /// The vector's hash; `None` for `Zero`.
    pub fn vector_sha256(&self) -> Option<&str> {
        self.provenance.as_ref().map(VectorProvenance::sha256)
    }

    fn width(&self) -> Option<usize> {
        self.vector.as_ref().map(Vec::len)
    }

    /// The arithmetic, in the backend's residual path, on the written
    /// carrier.
    pub(super) fn apply<B: PlanBackend + ?Sized>(&self, backend: &B, carrier: &mut [f32]) {
        match (self.kind, &self.vector) {
            (InterventionKind::Zero, _) => carrier.fill(0.0),
            (InterventionKind::Add, Some(v)) => backend.residual_add(carrier, v),
            (InterventionKind::Replace, Some(v)) => carrier.copy_from_slice(v),
            (InterventionKind::Add | InterventionKind::Replace, None) => {
                unreachable!("a vector kind is constructed with its vector")
            }
        }
    }
}

/// One firing, as the step reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Firing {
    pub layer: usize,
    pub site: SublayerSite,
    pub position: usize,
    pub kind: InterventionKind,
}

/// What an intervened step returns: the logits, and which declared
/// interventions applied on it.
#[derive(Debug, Clone, PartialEq)]
pub struct InterventionStepOutput {
    pub logits: Option<Vec<f32>>,
    pub firings: Vec<Firing>,
    /// V3-INTERVENE-2: which declared head interventions applied on this
    /// step. Empty when [`super::intervene_heads::HeadInterventionPlan::none`] was threaded.
    pub head_firings: Vec<super::intervene_heads::HeadFiring>,
}

/// An address that was declared and never reached by the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unreached {
    pub layer: usize,
    pub site: SublayerSite,
    pub position: usize,
}

/// Every intervention a run declares. Empty is what production threads.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InterventionPlan {
    interventions: Vec<Intervention>,
}

impl InterventionPlan {
    /// No intervention: the production step.
    pub const fn none() -> Self {
        Self {
            interventions: Vec::new(),
        }
    }

    pub fn is_none(&self) -> bool {
        self.interventions.is_empty()
    }

    /// Declare one more. Two interventions sharing a position at one site
    /// are refused: which applied first would depend on declaration
    /// order, and an arm whose meaning depends on order is not an arm.
    pub fn with(mut self, intervention: Intervention) -> Result<Self, VindexError> {
        if let Some(existing) = self
            .interventions
            .iter()
            .find(|i| i.address.overlaps(&intervention.address))
        {
            return Err(VindexError::Parse(format!(
                "two interventions share layer {} {:?} at a position — `{}` and `{}` — and which \
                 applied first would depend on declaration order",
                intervention.address.layer,
                intervention.address.site,
                existing.kind.name(),
                intervention.kind.name()
            )));
        }
        self.interventions.push(intervention);
        Ok(self)
    }

    pub fn declared(&self) -> usize {
        self.interventions.len()
    }

    pub fn interventions(&self) -> &[Intervention] {
        &self.interventions
    }

    /// Refuse before the first token executes: an address off the
    /// executed layers, an FFN site on a layer without an FFN program, a
    /// carrier that is not `Single`, or a vector of the wrong width.
    pub fn admit(&self, plan: &ComponentOpPlan, ops: &PreparedOperands) -> Result<(), VindexError> {
        if self.is_none() {
            return Ok(());
        }
        if ops.hyper_connection().is_some() {
            return Err(VindexError::Parse(
                "interventions act on a `Single` carrier; this component carries a `Bundle` \
                 (hyper-connected), which V3-INTERVENE-1 declares out of scope"
                    .to_string(),
            ));
        }
        if ops.attention_residual_block_size().is_some() {
            return Err(VindexError::Parse(
                "interventions act on a `Single` carrier; this component carries a `History` \
                 (attention-residual), which V3-INTERVENE-1 declares out of scope"
                    .to_string(),
            ));
        }
        let first = ops.first_layer();
        let executed = first..first + ops.layers().len();
        let hidden = ops.hidden();
        for intervention in &self.interventions {
            let address = &intervention.address;
            if !executed.contains(&address.layer) {
                return Err(VindexError::Parse(format!(
                    "intervention `{}` addresses layer {} but this image executes layers {}..{}",
                    intervention.kind.name(),
                    address.layer,
                    executed.start,
                    executed.end
                )));
            }
            if address.site == SublayerSite::Ffn && plan.layers[address.layer].ffn.is_none() {
                return Err(VindexError::Parse(format!(
                    "intervention `{}` addresses the FFN site of layer {}, which declares no FFN \
                     program and therefore writes no FFN carrier",
                    intervention.kind.name(),
                    address.layer
                )));
            }
            if let Some(width) = intervention.width() {
                if width != hidden {
                    return Err(VindexError::Parse(format!(
                        "intervention `{}` at layer {} {:?} carries a vector of width {width} on a \
                         carrier of width {hidden}",
                        intervention.kind.name(),
                        address.layer,
                        address.site
                    )));
                }
            }
        }
        Ok(())
    }

    /// The intervention that fires at this write, if any.
    pub(super) fn at(
        &self,
        layer: usize,
        site: SublayerSite,
        position: usize,
    ) -> Option<&Intervention> {
        self.interventions.iter().find(|i| {
            i.address.layer == layer && i.address.site == site && i.address.covers(position)
        })
    }

    /// The declaration's hash, for the run identity: over every
    /// intervention's address, kind and vector hash, in declaration
    /// order. `None` when nothing is declared, so an unintervened run's
    /// identity is unchanged.
    pub fn declaration_sha256(&self) -> Option<String> {
        if self.is_none() {
            return None;
        }
        let mut hasher = Sha256::new();
        for i in &self.interventions {
            hasher.update((i.address.layer as u64).to_le_bytes());
            hasher.update([match i.address.site {
                SublayerSite::Attention => 0u8,
                SublayerSite::Ffn => 1u8,
            }]);
            hasher.update((i.address.positions.len() as u64).to_le_bytes());
            for p in &i.address.positions {
                hasher.update((*p as u64).to_le_bytes());
            }
            hasher.update(i.kind.name().as_bytes());
            hasher.update(i.vector_sha256().unwrap_or("").as_bytes());
        }
        let bytes: [u8; 32] = hasher.finalize().into();
        Some(bytes.iter().fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        }))
    }

    /// Addresses declared at positions the run never executed
    /// (`positions_executed` is the count of positions stepped). A
    /// declared intervention that never fired is a refusal at the end of
    /// the run, on the receipt, never a silent no-op.
    pub fn unreached(&self, positions_executed: usize) -> Vec<Unreached> {
        self.interventions
            .iter()
            .flat_map(|i| {
                i.address
                    .positions()
                    .filter(move |&p| p >= positions_executed)
                    .map(move |position| Unreached {
                        layer: i.address.layer,
                        site: i.address.site,
                        position,
                    })
            })
            .collect()
    }
}

/// The observer that sources a carrier from another run: copies the
/// `after` of every declared address into memory, keyed by address, and
/// hands one back with its provenance. Nothing is persisted.
#[derive(Debug, Default)]
pub struct CarrierCapture {
    wanted: BTreeSet<(usize, SublayerSite, usize)>,
    captured: BTreeMap<(usize, SublayerSite, usize), Vec<f32>>,
}

impl CarrierCapture {
    /// Capture at these `(layer, site, position)` addresses.
    pub fn at(addresses: impl IntoIterator<Item = (usize, SublayerSite, usize)>) -> Self {
        Self {
            wanted: addresses.into_iter().collect(),
            captured: BTreeMap::new(),
        }
    }

    /// The captured carrier at an address, if the run reached it.
    pub fn get(&self, layer: usize, site: SublayerSite, position: usize) -> Option<&[f32]> {
        self.captured
            .get(&(layer, site, position))
            .map(Vec::as_slice)
    }

    /// How many declared addresses were captured.
    pub fn captured(&self) -> usize {
        self.captured.len()
    }

    /// An intervention whose vector is the carrier captured from `run_id`
    /// at `from`, applied with `kind` at `address`. Refuses when the run
    /// never reached `from`.
    pub fn intervention(
        &self,
        run_id: &str,
        from: (usize, SublayerSite, usize),
        kind: InterventionKind,
        address: Address,
    ) -> Result<Intervention, VindexError> {
        let (layer, site, position) = from;
        let vector = self.get(layer, site, position).ok_or_else(|| {
            VindexError::Parse(format!(
                "no carrier was captured from run `{run_id}` at layer {layer} {site:?} position \
                 {position} — the run did not reach it or the capture was not declared there"
            ))
        })?;
        let vector = vector.to_vec();
        let provenance = VectorProvenance::Captured {
            run_id: run_id.to_string(),
            layer,
            site,
            position,
            sha256: vector_sha256(&vector),
        };
        match kind {
            InterventionKind::Add => Intervention::add(address, vector, provenance),
            InterventionKind::Replace => Intervention::replace(address, vector, provenance),
            InterventionKind::Zero => Err(VindexError::Parse(
                "a captured carrier has no use in a `zero` intervention".to_string(),
            )),
        }
    }
}

impl StepObserver for CarrierCapture {
    fn event(&mut self, _event: StepEvent) {}

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        let key = (record.layer, record.site, record.position);
        if self.wanted.contains(&key) {
            self.captured.insert(key, record.after.to_vec());
        }
    }
}
