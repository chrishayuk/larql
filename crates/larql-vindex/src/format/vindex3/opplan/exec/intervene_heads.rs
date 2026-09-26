//! V3-INTERVENE-2: one query head's mixed value changed INSIDE the
//! attention kernel, before the model recombines it.
//!
//! A head intervention is a declared object — an address `(layer, head,
//! positions)`, a kind (`Zero`, `Scale(α)`, `Replace(v)`) and, for
//! `Replace`, a vector provenance with a checked hash (V3-INTERVENE-1's
//! vocabulary, [`super::intervene::VectorProvenance`]). It acts on
//! `ctx_h`, the head's mixed value, at the exact point V3-HEAD-OBS-1's
//! tap reads it: after aggregation, before the output gate, before
//! `o_proj`, before the post-attention norm and any residual scale.
//! Nothing downstream is patched again.
//!
//! This is deliberately NOT the additive shortcut of subtracting a
//! recorded head child from the carrier after the fact (ATTR-1C): under
//! a post-attention RMS norm, removing one head's contribution changes
//! the norm's scalar, which changes every OTHER head's effective
//! contribution too, so the shortcut is not the model's own
//! counterfactual. `docs/v3-intervene-2-head-intervention.md` measures
//! the gap between the two (J5) rather than assuming they agree.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::super::ComponentOpPlan;
use super::intervene::{vector_sha256, VectorProvenance};
use super::prepared::PreparedOperands;
use crate::error::VindexError;

/// What a head intervention does to `ctx_h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HeadInterventionKind {
    /// `ctx_h` becomes zero.
    Zero,
    /// `ctx_h` is multiplied by a declared finite scalar.
    Scale,
    /// `ctx_h` becomes a declared `head_dim`-wide vector.
    Replace,
}

impl HeadInterventionKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Scale => "scale",
            Self::Replace => "replace",
        }
    }
}

/// One layer's one query head, at declared positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadAddress {
    pub layer: usize,
    pub head: usize,
    positions: BTreeSet<usize>,
}

impl HeadAddress {
    pub fn new(
        layer: usize,
        head: usize,
        positions: impl IntoIterator<Item = usize>,
    ) -> Result<Self, VindexError> {
        let positions: BTreeSet<usize> = positions.into_iter().collect();
        if positions.is_empty() {
            return Err(VindexError::Parse(format!(
                "head intervention at layer {layer} head {head} names no position"
            )));
        }
        Ok(Self {
            layer,
            head,
            positions,
        })
    }

    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.positions.iter().copied()
    }

    pub fn covers(&self, position: usize) -> bool {
        self.positions.contains(&position)
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.layer == other.layer
            && self.head == other.head
            && self.positions.iter().any(|p| other.positions.contains(p))
    }
}

/// One declared change to `ctx_h`.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadIntervention {
    address: HeadAddress,
    kind: HeadInterventionKind,
    scale: Option<f32>,
    vector: Option<Vec<f32>>,
    provenance: Option<VectorProvenance>,
}

impl HeadIntervention {
    /// `ctx_h` becomes zero at the address.
    pub fn zero(address: HeadAddress) -> Self {
        Self {
            address,
            kind: HeadInterventionKind::Zero,
            scale: None,
            vector: None,
            provenance: None,
        }
    }

    /// `ctx_h` is multiplied by `factor` at the address. `factor` must be
    /// finite; `Scale(1.0)` is the no-op law (J4).
    pub fn scale(address: HeadAddress, factor: f32) -> Result<Self, VindexError> {
        if !factor.is_finite() {
            return Err(VindexError::Parse(format!(
                "head intervention `scale` at layer {} head {} carries a non-finite factor",
                address.layer, address.head
            )));
        }
        Ok(Self {
            address,
            kind: HeadInterventionKind::Scale,
            scale: Some(factor),
            vector: None,
            provenance: None,
        })
    }

    /// `ctx_h` becomes `vector` at the address. `vector` must be
    /// `head_dim` wide, finite, and its hash must match `provenance`'s.
    pub fn replace(
        address: HeadAddress,
        vector: Vec<f32>,
        provenance: VectorProvenance,
    ) -> Result<Self, VindexError> {
        if vector.is_empty() {
            return Err(VindexError::Parse(format!(
                "head intervention `replace` at layer {} head {} carries an empty vector",
                address.layer, address.head
            )));
        }
        if let Some(i) = vector.iter().position(|v| !v.is_finite()) {
            return Err(VindexError::Parse(format!(
                "head intervention `replace` at layer {} head {} carries a non-finite value at \
                 index {i}",
                address.layer, address.head
            )));
        }
        let actual = vector_sha256(&vector);
        if actual != provenance.sha256() {
            return Err(VindexError::Parse(format!(
                "head intervention `replace` at layer {} head {}: the vector's bytes hash to \
                 {actual} but its provenance claims {} — the bytes handed over are not the \
                 bytes declared",
                address.layer,
                address.head,
                provenance.sha256()
            )));
        }
        Ok(Self {
            address,
            kind: HeadInterventionKind::Replace,
            scale: None,
            vector: Some(vector),
            provenance: Some(provenance),
        })
    }

    pub fn address(&self) -> &HeadAddress {
        &self.address
    }

    pub fn kind(&self) -> HeadInterventionKind {
        self.kind
    }

    pub fn provenance(&self) -> Option<&VectorProvenance> {
        self.provenance.as_ref()
    }

    fn width(&self) -> Option<usize> {
        self.vector.as_ref().map(Vec::len)
    }

    /// The arithmetic, in place, on the head's borrowed `ctx_h` slice.
    pub(super) fn apply(&self, ctx_h: &mut [f32]) {
        match self.kind {
            HeadInterventionKind::Zero => ctx_h.fill(0.0),
            HeadInterventionKind::Scale => {
                let factor = self.scale.expect("a scale kind carries its factor");
                for v in ctx_h.iter_mut() {
                    *v *= factor;
                }
            }
            HeadInterventionKind::Replace => {
                ctx_h.copy_from_slice(
                    self.vector
                        .as_ref()
                        .expect("a replace kind carries its vector"),
                );
            }
        }
    }

    /// The declaration hash's own contribution, for [`HeadInterventionPlan::declaration_sha256`].
    fn hash_into(&self, hasher: &mut Sha256) {
        hasher.update((self.address.layer as u64).to_le_bytes());
        hasher.update((self.address.head as u64).to_le_bytes());
        hasher.update((self.address.positions.len() as u64).to_le_bytes());
        for p in &self.address.positions {
            hasher.update((*p as u64).to_le_bytes());
        }
        hasher.update(self.kind.name().as_bytes());
        if let Some(factor) = self.scale {
            hasher.update(factor.to_le_bytes());
        }
        hasher.update(
            self.provenance
                .as_ref()
                .map(VectorProvenance::sha256)
                .unwrap_or(""),
        );
    }
}

/// One head firing, as the step reports it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadFiring {
    pub layer: usize,
    pub head: usize,
    pub position: usize,
    pub kind: HeadInterventionKind,
}

/// An address that was declared and never reached by the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadUnreached {
    pub layer: usize,
    pub head: usize,
    pub position: usize,
}

/// Every head intervention a run declares. Empty is what production
/// threads — the existing `attention_step`/`attention_step_observed`
/// path runs unchanged, so declaring nothing costs nothing (J4/JF1).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HeadInterventionPlan {
    interventions: Vec<HeadIntervention>,
}

impl HeadInterventionPlan {
    pub const fn none() -> Self {
        Self {
            interventions: Vec::new(),
        }
    }

    pub fn is_none(&self) -> bool {
        self.interventions.is_empty()
    }

    /// Declare one more. Two interventions sharing a position on one head
    /// are refused (J6): which applied first would depend on declaration
    /// order.
    pub fn with(mut self, intervention: HeadIntervention) -> Result<Self, VindexError> {
        if let Some(existing) = self
            .interventions
            .iter()
            .find(|i| i.address.overlaps(&intervention.address))
        {
            return Err(VindexError::Parse(format!(
                "two head interventions share layer {} head {} at a position — `{}` and `{}` — \
                 and which applied first would depend on declaration order",
                intervention.address.layer,
                intervention.address.head,
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

    pub fn interventions(&self) -> &[HeadIntervention] {
        &self.interventions
    }

    /// Refuse before the first token executes (J6): an address off the
    /// executed layers, a layer without softmax heads, a head outside
    /// this layer's query heads, a `Single`-carrier violation (the same
    /// topology refusal V3-INTERVENE-1 makes), or a `Replace` vector of
    /// the wrong width.
    pub fn admit(&self, plan: &ComponentOpPlan, ops: &PreparedOperands) -> Result<(), VindexError> {
        if self.is_none() {
            return Ok(());
        }
        if ops.hyper_connection().is_some() {
            return Err(VindexError::Parse(
                "head interventions act on a `Single` carrier's attention write; this component \
                 carries a `Bundle` (hyper-connected), which V3-INTERVENE-2 declares out of scope"
                    .to_string(),
            ));
        }
        if ops.attention_residual_block_size().is_some() {
            return Err(VindexError::Parse(
                "head interventions act on a `Single` carrier's attention write; this component \
                 carries a `History` (attention-residual), which V3-INTERVENE-2 declares out of \
                 scope"
                    .to_string(),
            ));
        }
        let first = ops.first_layer();
        let executed = first..first + ops.layers().len();
        for intervention in &self.interventions {
            let address = &intervention.address;
            if !executed.contains(&address.layer) {
                return Err(VindexError::Parse(format!(
                    "head intervention `{}` addresses layer {} but this image executes layers \
                     {}..{}",
                    intervention.kind.name(),
                    address.layer,
                    executed.start,
                    executed.end
                )));
            }
            let Some(op) = plan.layers[address.layer].attention.softmax() else {
                return Err(VindexError::Parse(format!(
                    "head intervention `{}` addresses layer {}, which has no softmax attention \
                     heads",
                    intervention.kind.name(),
                    address.layer
                )));
            };
            if address.head >= op.num_q_heads {
                return Err(VindexError::Parse(format!(
                    "head intervention `{}` addresses head {} but layer {} has {} query heads",
                    intervention.kind.name(),
                    address.head,
                    address.layer,
                    op.num_q_heads
                )));
            }
            if let Some(width) = intervention.width() {
                if width != op.head_dim {
                    return Err(VindexError::Parse(format!(
                        "head intervention `{}` at layer {} head {} carries a vector of width \
                         {width} on a head of width {}",
                        intervention.kind.name(),
                        address.layer,
                        address.head,
                        op.head_dim
                    )));
                }
            }
        }
        Ok(())
    }

    /// The intervention that fires on this head's write, if any.
    pub(super) fn at(
        &self,
        layer: usize,
        head: usize,
        position: usize,
    ) -> Option<&HeadIntervention> {
        self.interventions.iter().find(|i| {
            i.address.layer == layer && i.address.head == head && i.address.covers(position)
        })
    }

    /// Whether ANY declared intervention addresses this layer — the
    /// cheap check the kernel makes before doing any per-head lookup.
    pub(super) fn touches_layer(&self, layer: usize) -> bool {
        self.interventions.iter().any(|i| i.address.layer == layer)
    }

    pub fn declaration_sha256(&self) -> Option<String> {
        if self.is_none() {
            return None;
        }
        let mut hasher = Sha256::new();
        for i in &self.interventions {
            i.hash_into(&mut hasher);
        }
        let bytes: [u8; 32] = hasher.finalize().into();
        Some(bytes.iter().fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        }))
    }

    pub fn unreached(&self, positions_executed: usize) -> Vec<HeadUnreached> {
        self.interventions
            .iter()
            .flat_map(|i| {
                i.address
                    .positions()
                    .filter(move |&p| p >= positions_executed)
                    .map(move |position| HeadUnreached {
                        layer: i.address.layer,
                        head: i.address.head,
                        position,
                    })
            })
            .collect()
    }
}

/// The observer that sources a head's `ctx_h` from another run (J7):
/// copies the UNINTERVENED value at declared `(layer, head, position)`
/// addresses into memory, keyed by address. Nothing is persisted.
///
/// Fed from [`super::observe::AttentionHeadRecord`] directly — the same
/// per-head tap V3-HEAD-OBS-1 reads, which always carries the
/// pre-intervention value (J3).
#[derive(Debug, Default)]
pub struct HeadCapture {
    wanted: BTreeSet<(usize, usize, usize)>,
    captured: BTreeMap<(usize, usize, usize), Vec<f32>>,
}

impl HeadCapture {
    pub fn at(addresses: impl IntoIterator<Item = (usize, usize, usize)>) -> Self {
        Self {
            wanted: addresses.into_iter().collect(),
            captured: BTreeMap::new(),
        }
    }

    pub fn get(&self, layer: usize, head: usize, position: usize) -> Option<&[f32]> {
        self.captured
            .get(&(layer, head, position))
            .map(Vec::as_slice)
    }

    pub fn captured(&self) -> usize {
        self.captured.len()
    }

    /// Observe one head's record; captures `ctx_h` (pre-gate) at a
    /// wanted address. Call from [`StepObserver::attention_head`].
    pub fn observe(&mut self, layer: usize, record: &super::observe::AttentionHeadRecord<'_>) {
        let key = (layer, record.head, record.position);
        if self.wanted.contains(&key) {
            self.captured.insert(key, record.values.to_vec());
        }
    }

    /// A head intervention whose vector is the `ctx_h` captured from
    /// `run_id` at `from`, applied with `kind` at `address`. Refuses when
    /// the run never reached `from`, or for `Zero`/`Scale`, which carry
    /// no vector.
    pub fn intervention(
        &self,
        run_id: &str,
        from: (usize, usize, usize),
        address: HeadAddress,
    ) -> Result<HeadIntervention, VindexError> {
        let (layer, head, position) = from;
        let vector = self.get(layer, head, position).ok_or_else(|| {
            VindexError::Parse(format!(
                "no head was captured from run `{run_id}` at layer {layer} head {head} position \
                 {position} — the run did not reach it or the capture was not declared there"
            ))
        })?;
        let vector = vector.to_vec();
        let provenance = VectorProvenance::CapturedHead {
            run_id: run_id.to_string(),
            layer,
            head,
            position,
            sha256: vector_sha256(&vector),
        };
        HeadIntervention::replace(address, vector, provenance)
    }
}
