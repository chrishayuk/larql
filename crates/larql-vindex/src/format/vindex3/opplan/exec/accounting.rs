//! **Accounting bound AGAINST the census, never into it.**
//!
//! Rung 3c of the representation/execution contract. Two instruments,
//! kept apart on purpose:
//!
//! * an [`Expectation`] is what a pinned realization DECLARES — derived from
//!   the pin, the codec's declared residency, the executor's own block
//!   geometry and the container's RECORDED operand length. It never reads
//!   a loaded object.
//! * an [`Observed`] is what the loader actually made resident — read off
//!   the bound `LoadedWeight`s, paired with the operand each one binds.
//!
//! [`reconcile`] compares the two. If both came from one declaration the
//! comparison would be circular; because they do not, a wrong block
//! constant, a wrong codec residency, or a loader that drifted from the
//! selector shows up as a mismatch. The residency census is a third
//! reading — the loaded objects summed by site — and the plan-level
//! witness checks it against the observed pairs.
//!
//! Three quantities, each with one definition:
//!
//! ```text
//! stored footprint   Σ recorded length over DISTINCT stored operands
//!                    (a tied head and its embedding are one object)
//! execution touch    Σ recorded length over OPERATIONS
//!                    (the same object read once per operation)
//! resident / staging what the realization holds, and what it
//!                    materialises transiently on the way there
//! ```

use std::collections::{BTreeMap, BTreeSet};

use super::backend::WeightFormat;
use super::cpu::integer::weight_index_enabled;
use super::cpu::ledger::{PlanTally, ProjectionLedger};
use super::cpu::physical::PhysicalProjectionPlan;
use super::quantise::{Q4_BLOCK, Q8_BLOCK};
use super::realization::{RealizationForm, RealizationId, RealizationRecord};
use super::weights::{LoadedWeight, DEVICE_PAGE_ALIGN};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{ResidencyClass, ResidencyProfile};

/// Width of the canonical decode target.
const F32_WIDTH: f64 = std::mem::size_of::<f32>() as f64;
/// Width of a half-precision resident element.
const HALF_WIDTH: f64 = std::mem::size_of::<u16>() as f64;
/// Bits in a byte, for the stored-bit pricings below.
const BITS_PER_BYTE: f64 = 8.0;
/// NVFP4's stored rate — 4-bit codes and one 8-bit scale per 16 elements.
const NVFP4_BITS_PER_WEIGHT: f64 = 4.5;
/// MXFP4's stored rate — 4-bit codes and one 8-bit scale per 32 elements.
const MXFP4_BITS_PER_WEIGHT: f64 = 4.25;
/// The widest of the three K-quant codecs; the bound operand carries which.
const KQUANT_WIDEST_BITS_PER_WEIGHT: f64 = 8.5;
/// One f32 scale per block of a re-quantised image.
const SCALE_WIDTH: f64 = F32_WIDTH;
/// One i16 code sum per block, when the weight-code index is on.
const SUM_WIDTH: f64 = std::mem::size_of::<i16>() as f64;

/// The executor's own resident geometry: what a re-quantised image costs
/// per weight depends on these, and they are the executor's declarations,
/// not the codec's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGeometry {
    pub q8_block: usize,
    pub q4_block: usize,
    /// Whether the Q8 image also carries one code sum per block.
    pub q8_indexed: bool,
}

impl BlockGeometry {
    /// The geometry this process's executor uses.
    pub fn executor() -> Self {
        Self {
            q8_block: Q8_BLOCK,
            q4_block: Q4_BLOCK,
            q8_indexed: weight_index_enabled(),
        }
    }
}

/// What the executor makes resident for `format`, priced from `geometry`:
/// a widened image is f32 per weight; a re-quantised image is its codes
/// plus one f32 scale (and one i16 sum, when indexed) per block; a
/// device's half-precision image is two bytes per weight and, being
/// rounded from the source, a re-quantisation; a compact pack bound as
/// stored is its stored bits.
pub fn resident_profile_with(format: WeightFormat, geometry: BlockGeometry) -> ResidencyProfile {
    match format {
        WeightFormat::F32 => ResidencyProfile::DECODED_F32,
        WeightFormat::Bf16 => ResidencyProfile::rebound(HALF_WIDTH * BITS_PER_BYTE),
        WeightFormat::F16 => ResidencyProfile {
            class: ResidencyClass::TransientRequantised,
            bytes_per_weight: HALF_WIDTH,
        },
        WeightFormat::Q8 => ResidencyProfile {
            class: ResidencyClass::TransientRequantised,
            bytes_per_weight: 1.0
                + (SCALE_WIDTH + if geometry.q8_indexed { SUM_WIDTH } else { 0.0 })
                    / geometry.q8_block as f64,
        },
        WeightFormat::Q4 => ResidencyProfile {
            class: ResidencyClass::TransientRequantised,
            bytes_per_weight: 0.5 + SCALE_WIDTH / geometry.q4_block as f64,
        },
        WeightFormat::Nvfp4 => ResidencyProfile::rebound(NVFP4_BITS_PER_WEIGHT),
        WeightFormat::Mxfp4 => ResidencyProfile::stored(MXFP4_BITS_PER_WEIGHT),
        WeightFormat::KQuant => ResidencyProfile::stored(KQUANT_WIDEST_BITS_PER_WEIGHT),
    }
}

/// What one pinned realization declares it will cost. Every field is
/// derived from the pin and the container's record; none from a loaded
/// object.
#[derive(Debug, Clone, PartialEq)]
pub struct Expectation {
    pub operand: OperandRef,
    pub operation: Operation,
    pub layer: Option<usize>,
    pub realization: RealizationId,
    /// The container's recorded length for the stored operand — an
    /// instance fact, which for an entropy-coded operand is not a
    /// function of its shape.
    pub stored_bytes: u64,
    pub logical_elements: usize,
    /// The declared resident image: the realization's residency profile
    /// over the logical elements.
    pub declared_resident: u64,
    /// Bytes materialised transiently on the way to residency: the f32
    /// image a decode or re-quantisation passes through, none for a
    /// realization that binds the stored bytes.
    pub staging: u64,
}

impl Expectation {
    /// The stored bytes read once for this operation.
    pub fn touch(&self) -> u64 {
        self.stored_bytes
    }

    /// The peak the realization holds while preparing: resident plus
    /// whatever it staged to get there.
    pub fn working_set(&self) -> u64 {
        self.declared_resident + self.staging
    }
}

/// Price every record. `stored_len` answers with the container's recorded
/// length for an operand, or `None` for one the container does not hold.
pub fn expectations(
    records: &[RealizationRecord],
    stored_len: impl Fn(&OperandRef) -> Option<u64>,
    geometry: BlockGeometry,
) -> Vec<Expectation> {
    records
        .iter()
        .map(|r| {
            let logical = r.planned.logical_elements;
            let realization = r.selection.realization;
            // Direct and decode carry the codec's own declaration; the
            // executor's forms are priced from its geometry, so a change
            // to that geometry re-prices them here and nowhere else.
            let profile = match realization.form {
                RealizationForm::Direct(_) | RealizationForm::Decode(_) => r.selection.residency,
                RealizationForm::Requantise(_)
                | RealizationForm::SliceStored { .. }
                | RealizationForm::DeviceResident(_) => {
                    resident_profile_with(realization.format(), geometry)
                }
                RealizationForm::DecodedGather => ResidencyProfile::DECODED_F32,
            };
            let staging = match realization.form {
                RealizationForm::Direct(_) => 0,
                RealizationForm::Decode(_)
                | RealizationForm::Requantise(_)
                | RealizationForm::DecodedGather => (logical as f64 * F32_WIDTH).round() as u64,
                RealizationForm::SliceStored { convert }
                | RealizationForm::DeviceResident(convert) => {
                    if convert == WeightFormat::F32 {
                        0
                    } else {
                        (logical as f64 * F32_WIDTH).round() as u64
                    }
                }
            };
            Expectation {
                operand: r.planned.operand.clone(),
                operation: r.planned.operation,
                layer: r.planned.layer,
                realization,
                stored_bytes: stored_len(&r.planned.operand).unwrap_or(0),
                logical_elements: logical,
                declared_resident: (profile.bytes_per_weight * logical as f64).round() as u64,
                staging,
            }
        })
        .collect()
}

/// One operand paired with the object(s) the loader bound for it: a
/// matrix is one object, a packed bank is one per expert. Every loader
/// names its own pairing, field by field, so the observation cannot be
/// derived from the plan's order.
pub struct Bound<'a> {
    pub operand: &'a OperandRef,
    pub weights: Vec<&'a LoadedWeight>,
}

impl<'a> Bound<'a> {
    pub fn one(operand: &'a OperandRef, weight: &'a LoadedWeight) -> Self {
        Self {
            operand,
            weights: vec![weight],
        }
    }

    pub fn observed(
        &self,
        operation: Operation,
        layer: Option<usize>,
    ) -> Result<Observed, VindexError> {
        let Some(first) = self.weights.first() else {
            return Err(VindexError::Parse(format!(
                "operand `{}`: bound to no object",
                self.operand.tensor
            )));
        };
        let format = first.format();
        if let Some(other) = self.weights.iter().find(|w| w.format() != format) {
            return Err(VindexError::Parse(format!(
                "operand `{}`: bound objects disagree on their representation ({format:?} vs {:?})",
                self.operand.tensor,
                other.format()
            )));
        }
        Ok(Observed {
            operand: self.operand.clone(),
            operation,
            layer,
            format,
            resident_bytes: self.weights.iter().map(|w| w.resident_bytes() as u64).sum(),
            allocations: self.weights.iter().map(|w| w.padded_allocations()).sum(),
        })
    }
}

/// What the loader actually made resident for one planned operand.
#[derive(Debug, Clone, PartialEq)]
pub struct Observed {
    pub operand: OperandRef,
    pub operation: Operation,
    pub layer: Option<usize>,
    pub format: WeightFormat,
    /// Bytes held, allocation padding included.
    pub resident_bytes: u64,
    /// Allocations the bytes live in: one for a matrix, one per expert
    /// for a bank.
    pub allocations: usize,
}

/// The result of a reconciliation that held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Reconciliation {
    pub matched: usize,
    pub declared_resident: u64,
    pub observed_resident: u64,
    /// Observed minus declared — allocation padding, bounded per
    /// allocation by the page.
    pub padding: u64,
}

/// Every expectation must meet one observation for its operand and
/// operation instance, in the pinned representation, holding the declared
/// bytes plus at most a page of padding per allocation; nothing observed
/// may be unexpected.
///
/// Instances are counted, not deduplicated: the same stored operand bound
/// twice — a tied head under two operations, Gemma-4's layer that binds
/// one tensor as both its key and value projection — is two expectations
/// meeting two objects, and the loader really does hold it twice.
pub fn reconcile(
    expected: &[Expectation],
    observed: &[Observed],
) -> Result<Reconciliation, VindexError> {
    let key = |op: &OperandRef, operation: Operation, layer: Option<usize>| {
        (
            op.object.clone(),
            op.tensor.clone(),
            operation.name(),
            layer,
        )
    };
    let mut pool: BTreeMap<(String, String, &'static str, Option<usize>), Vec<&Observed>> =
        BTreeMap::new();
    for o in observed {
        pool.entry(key(&o.operand, o.operation, o.layer))
            .or_default()
            .push(o);
    }
    let mut out = Reconciliation::default();
    for e in expected {
        let k = key(&e.operand, e.operation, e.layer);
        let Some(o) = pool.get_mut(&k).and_then(Vec::pop) else {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): pinned {} but nothing is resident for it",
                e.operand.tensor,
                e.operation.name(),
                e.realization.name()
            )));
        };
        let pinned = e.realization.format();
        if o.format != pinned {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): pinned {pinned:?} but {:?} is resident",
                e.operand.tensor,
                e.operation.name(),
                o.format
            )));
        }
        // Exact for an object held in plain vectors; up to a page of
        // padding per page-aligned allocation, and never less than declared.
        let ceiling = e.declared_resident + (o.allocations as u64) * DEVICE_PAGE_ALIGN as u64;
        let within = if o.allocations == 0 {
            o.resident_bytes == e.declared_resident
        } else {
            o.resident_bytes >= e.declared_resident && o.resident_bytes < ceiling
        };
        if !within {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): {} declares {} resident bytes over {} elements; {} are \
                 resident in {} allocation(s) — the declaration and the loader disagree",
                e.operand.tensor,
                e.operation.name(),
                e.realization.name(),
                e.declared_resident,
                e.logical_elements,
                o.resident_bytes,
                o.allocations
            )));
        }
        out.matched += 1;
        out.declared_resident += e.declared_resident;
        out.observed_resident += o.resident_bytes;
        out.padding += o.resident_bytes - e.declared_resident;
    }
    if let Some(stray) = pool.values().flatten().next() {
        return Err(VindexError::Parse(format!(
            "operand `{}` ({}) is resident but nothing was pinned for it",
            stray.operand.tensor,
            stray.operation.name()
        )));
    }
    Ok(out)
}

/// The stored footprint: each stored operand counted once, however many
/// operations read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StoredFootprint {
    pub bytes: u64,
    pub operands: usize,
}

pub fn stored_footprint(expected: &[Expectation]) -> StoredFootprint {
    let mut seen = BTreeSet::new();
    let mut out = StoredFootprint::default();
    for e in expected {
        if seen.insert((e.operand.object.clone(), e.operand.tensor.clone())) {
            out.bytes += e.stored_bytes;
            out.operands += 1;
        }
    }
    out
}

/// The execution touch: the stored bytes read, once per operation.
pub fn execution_touch(expected: &[Expectation]) -> u64 {
    expected.iter().map(Expectation::touch).sum()
}

/// The ledger's account of what ran, held against the pins: every CPU
/// plan the ledger tallied is a pinned realization, and every pinned CPU
/// realization ran. Returned per plan with the number of operands pinned
/// to it and its tally, so a caller can hold the tally's POSITIONS against
/// the operands — calls are not the unit, because the executor batches a
/// projection's positions differently per site (per position at the
/// attention, all at once in the FFN), while every position a pinned
/// operand processed is counted exactly once.
pub fn ledger_correspondence(
    records: &[RealizationRecord],
    ledger: &ProjectionLedger,
) -> Result<Vec<(PhysicalProjectionPlan, usize, PlanTally)>, VindexError> {
    let mut pinned: Vec<(PhysicalProjectionPlan, usize)> = Vec::new();
    for r in records {
        if let Some(plan) = r.selection.realization.cpu_plan() {
            match pinned.iter_mut().find(|(p, _)| *p == plan) {
                Some((_, n)) => *n += 1,
                None => pinned.push((plan, 1)),
            }
        }
    }
    let mut out = Vec::new();
    for (plan, tally) in ledger.all() {
        let pins = pinned.iter().find(|(p, _)| *p == plan).map(|(_, n)| *n);
        match (pins, tally.calls) {
            (None, 0) => {}
            (None, calls) => {
                return Err(VindexError::Parse(format!(
                    "{plan:?} ran {calls} call(s) but no operand was pinned to it"
                )))
            }
            (Some(n), 0) => {
                return Err(VindexError::Parse(format!(
                    "{n} operand(s) pinned to {plan:?} but it never ran"
                )))
            }
            (Some(n), _) => out.push((plan, n, tally)),
        }
    }
    Ok(out)
}

/// A selection summary for a report: presentation over the structured
/// records, one line per realization with the operands it serves.
pub fn render_selection_summary(records: &[RealizationRecord]) -> String {
    let mut groups: Vec<(String, String, String, usize, usize)> = Vec::new();
    for r in records {
        let key = (
            r.representation.clone(),
            r.selection.realization.name(),
            r.selection.reason.name().to_string(),
        );
        match groups
            .iter_mut()
            .find(|g| g.0 == key.0 && g.1 == key.1 && g.2 == key.2)
        {
            Some(g) => {
                g.3 += 1;
                g.4 += r.planned.logical_elements;
            }
            None => groups.push((key.0, key.1, key.2, 1, r.planned.logical_elements)),
        }
    }
    let mut out = String::from("realizations:\n");
    for (representation, realization, reason, operands, elements) in groups {
        out.push_str(&format!(
            "  {representation:<10} → {realization:<32} {operands:>4} operand(s) {:>12} weights  ({reason})\n",
            elements
        ));
    }
    out
}
