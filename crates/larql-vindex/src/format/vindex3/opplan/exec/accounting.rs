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
use super::quantise::{Q4_BLOCK, Q8_BLOCK, SUM_BLOCK};
use super::realization::{
    DependencyLifetime, DependencyPin, RealizationForm, RealizationId, RealizationRecord,
};
use super::weights::{LoadedWeight, DEVICE_PAGE_ALIGN};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::planned::PlannedOperand;
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
/// Fine-grained FP8's element width: E4M3 is one byte, exactly.
const FP8_BITS_PER_WEIGHT: f64 = 8.0;
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
        // Bound AS STORED, like a K-quant pack: the checkpoint's own
        // bytes, never widened at rest. That is the whole reason the
        // format is carried natively — a widened GLM-5.3-Flash would be
        // 612 GB of a 306 GB checkpoint, and the residency question this
        // ledger exists to answer would have no subject.
        //
        // The scales are counted with the codes: 8 bits per weight plus
        // one f32 per tile, which at the 128x128 grid GLM ships is
        // 32/16384 of a bit and rounds to nothing — but it is derived,
        // not waved away, because a [1, 32] grid (which the same scheme
        // permits) costs a full bit per weight.
        //
        // Priced at the CODES alone. The scale grid is a per-TENSOR fact
        // from the checkpoint — one scheme legally ships `[128, 128]` and
        // `[1, 32]` grids in one file — and `BlockGeometry` is by its own
        // definition the executor's geometry, not the codec's, so the
        // tile is not knowable here. The scales are accounted where the
        // tile IS known, on the bound operand
        // (`WeightRows::Fp8Block::bytes`, which counts them).
        //
        // The gap this leaves is stated rather than hidden: at GLM's
        // 128x128 grid it is one f32 per 16,384 weights — 0.02 bits per
        // weight, 0.2 % — but at a `[1, 32]` grid it would be a full bit,
        // and a forecast that silently omitted it would be 12 % light.
        WeightFormat::Fp8Block => ResidencyProfile::stored(FP8_BITS_PER_WEIGHT),
    }
}

/// The stored width a block runs along: the operand's inner dimension,
/// which for a packed bank of `[experts, rows, ...]` is the logical
/// elements per expert row rather than the packed shape's last axis.
fn inner_width(operation: Operation, shape: &[usize], logical: usize) -> usize {
    match operation {
        Operation::ExpertBankSlice if shape.len() >= 2 => logical / (shape[0] * shape[1]).max(1),
        _ => shape.last().copied().unwrap_or(logical),
    }
}

/// Bytes the executor's re-quantised image of `format` occupies over a
/// matrix of `rows × k` — EXACT, by the loader's own rule: codes per
/// element, one f32 scale per block, one i16 sum per sum-block when the
/// weight index is on, and blocks that never straddle a row, so a row
/// whose width is not a whole number of blocks carries a short last
/// block with its own scale. `None` for a format that is not a
/// re-quantised image, whose profile prices it per weight.
pub fn requantised_image_bytes(
    format: WeightFormat,
    rows: usize,
    k: usize,
    geometry: BlockGeometry,
) -> Option<u64> {
    let elements = (rows * k) as u64;
    match format {
        WeightFormat::Q8 => {
            let scales = (rows * k.div_ceil(geometry.q8_block)) as u64 * SCALE_WIDTH as u64;
            let sums = if geometry.q8_indexed {
                (rows * k.div_ceil(SUM_BLOCK)) as u64 * SUM_WIDTH as u64
            } else {
                0
            };
            Some(elements + scales + sums)
        }
        WeightFormat::Q4 => {
            let scales = (rows * k.div_ceil(geometry.q4_block)) as u64 * SCALE_WIDTH as u64;
            Some(elements / 2 + scales)
        }
        WeightFormat::F32
        | WeightFormat::Bf16
        | WeightFormat::F16
        | WeightFormat::Nvfp4
        | WeightFormat::Mxfp4
        | WeightFormat::KQuant
        // Stored as-is: there is no re-quantised image, so no bytes to price.
        | WeightFormat::Fp8Block => None,
    }
}

/// What `realization` declares it makes resident for `planned`: the
/// profile's bytes per weight over the logical elements, except that a
/// re-quantised image is priced by the loader's exact per-row rule. The
/// ONE pricing the ledger and the budget's re-selection share.
pub fn declared_resident_for(
    planned: &PlannedOperand,
    realization: RealizationId,
    profile: ResidencyProfile,
    geometry: BlockGeometry,
) -> u64 {
    let logical = planned.logical_elements;
    let exact = match realization.form {
        RealizationForm::Requantise(_)
        | RealizationForm::SliceStored { .. }
        | RealizationForm::DeviceResident(_) => {
            let k = inner_width(planned.operation, &planned.operand.shape, logical);
            let rows = logical.checked_div(k).unwrap_or(0);
            requantised_image_bytes(realization.format(), rows, k, geometry)
        }
        RealizationForm::Direct(_)
        | RealizationForm::Decode(_)
        | RealizationForm::DecodedGather
        | RealizationForm::MappedStored { .. } => None,
    };
    exact.unwrap_or_else(|| (profile.bytes_per_weight * logical as f64).round() as u64)
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
    /// over the logical elements — ADDRESSABLE bytes, whatever their
    /// physical form. [`Expectation::resources`] splits it into what is
    /// committed and what is merely mapped.
    pub declared_resident: u64,
    /// Bytes materialised transiently on the way to residency: the f32
    /// image a decode or re-quantisation passes through, none for a
    /// realization that binds the stored bytes.
    pub staging: u64,
    /// Bytes of the stored operand this pin OPENS to prepare it — the
    /// streams its selected extent reads, which for a terminal
    /// representation is everything the container holds and for a
    /// progressive one is the planes the extent reaches.
    ///
    /// Distinct from `stored_bytes`, which is the whole footprint on disk
    /// and does not move when an extent does, and from `touch_per_token`,
    /// which is the image the executor streams once the operand is
    /// resident. Under canonical decode this is the ONLY dimension a
    /// shallower extent moves.
    pub read_to_prepare: u64,
    /// The other represented objects this pin resolves, and what its
    /// realization does with each.
    ///
    /// Priced by the LEDGER rather than folded in here, because a
    /// dependency shared by many owners is one object: summing it per
    /// owner would count a codebook once per tensor that indexes it.
    pub dependencies: Vec<DependencyPin>,
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

    /// This expectation's demand on each resource, in the vocabulary a
    /// budget decision needs. A mapped realization's bytes are ADDRESS
    /// SPACE and page in per token as touched; a decoded, re-quantised,
    /// rebound or direct-copied realization's bytes are COMMITTED memory
    /// kept resident; a device realization's bytes live on its target.
    /// Touch is per operation instance per position, and it is the IMAGE
    /// the executor streams — the resident bytes of a decoded or
    /// re-quantised projection, the mapped bytes of a bank — not the
    /// stored bytes read once at load: a whole matrix for a projection,
    /// `top_k / experts` of an expert's matrix for a bank access, because
    /// a token selects that fraction of the bank.
    pub fn resources(&self) -> Resources {
        let per_token = match self.operation {
            Operation::ExpertProject { experts, top_k } if experts > 0 => {
                self.declared_resident as f64 * top_k as f64 / experts as f64
            }
            _ => self.declared_resident as f64,
        };
        let touch_per_token = per_token.round() as u64;
        match self.realization.form {
            RealizationForm::MappedStored { .. } => Resources {
                stored: self.stored_bytes,
                mapped: self.declared_resident,
                resident: 0,
                transient: self.staging,
                touch_per_token,
                page_in_per_token: touch_per_token,
                device: 0,
                read_to_prepare: self.read_to_prepare,
            },
            // On-device traffic is the device's; the host streams nothing.
            RealizationForm::DeviceResident(_) => Resources {
                stored: self.stored_bytes,
                mapped: 0,
                resident: 0,
                transient: self.staging,
                touch_per_token: 0,
                page_in_per_token: 0,
                device: self.declared_resident,
                read_to_prepare: self.read_to_prepare,
            },
            RealizationForm::Direct(_)
            | RealizationForm::Decode(_)
            | RealizationForm::Requantise(_)
            | RealizationForm::SliceStored { .. }
            | RealizationForm::DecodedGather => Resources {
                stored: self.stored_bytes,
                mapped: 0,
                resident: self.declared_resident,
                transient: self.staging,
                touch_per_token,
                page_in_per_token: 0,
                device: 0,
                read_to_prepare: self.read_to_prepare,
            },
        }
    }
}

/// What a preparation may hold and stream: the constraint selection
/// answers to. `None` on a dimension means unconstrained there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidencyBudget {
    /// Physical host memory the plan's working set — committed
    /// allocations, the staging peak, and a token's page-in — may reach.
    /// Never the address space a mapping occupies.
    pub physical_bytes: Option<u64>,
    /// Bytes the host may stream per token to reach a target rate.
    pub throughput: Option<ThroughputBudget>,
    /// Stored bytes the plan may OPEN to prepare itself — the cold cost of
    /// getting ready, as against the steady cost of running. The dimension
    /// a shallower extent moves: reading less of an artifact is what an
    /// extent buys under a realization that decodes.
    pub prepare_bytes: Option<u64>,
    /// How a mapped bank's selected experts are brought in per token —
    /// a policy on the ACCESS realization, stamped on every mapped pin
    /// the selection makes.
    pub expert_access: super::realization::MappedAccess,
    /// The reconstruction execution requires of the representations it
    /// selects. Representation-independent: a floor, never a depth.
    pub fidelity: RepresentationFloor,
}

/// What execution requires of a representation's reconstruction.
///
/// A quality REQUIREMENT, stated without naming a codec or a depth, so a
/// plan can carry it and any representation can answer it. It bounds which
/// extents a pin may take; it says nothing about how lossy the stored
/// representation is against the checkpoint it came from, which is the
/// graph's `Fidelity` and a different question.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RepresentationFloor {
    /// Only an extent that reconstructs the stored representation
    /// exactly: everything the artifact holds is read. The default, so a
    /// plan that asks for nothing gets no silent quality change — and a
    /// budget it cannot meet refuses rather than degrading.
    #[default]
    Exact,
    /// Any extent whose certificate declares a relative RMS at or under
    /// this bound. An extent that declares no radius is admissible only
    /// as the terminal one: an undeclared error is not a small one.
    RelativeRms(f64),
}

impl RepresentationFloor {
    /// Whether `option` satisfies this floor, given the representation's
    /// terminal extent.
    pub fn admits(
        self,
        option: &super::realization::ExtentOption,
        terminal: super::super::super::represent::codec::RepresentationExtent,
    ) -> bool {
        if option.certificate.extent == terminal {
            return true;
        }
        match self {
            Self::Exact => false,
            // v1 compares like with like: a bound stated in another
            // metric or over another domain does not satisfy this floor,
            // and is not converted into one that would.
            Self::RelativeRms(bound) => option.certificate.radius.as_ref().is_some_and(|r| {
                *r.metric() == super::super::super::represent::codec::MetricId::relative_rms()
                    && *r.domain()
                        == super::super::super::represent::codec::DomainId::finite_normals()
                    && r.radius() <= bound
            }),
        }
    }

    pub fn describe(self) -> String {
        match self {
            Self::Exact => "exact reconstruction".to_string(),
            Self::RelativeRms(bound) => format!("relative RMS at or under {bound:.3e}"),
        }
    }
}

/// A rate constraint: a plan can fit in memory and still be unusably
/// slow, so bytes touched per token are held against what the machine
/// moves per token at the rate the caller wants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThroughputBudget {
    pub bytes_per_second: u64,
    pub target_tokens_per_second: f64,
}

impl ThroughputBudget {
    /// Bytes a token may touch at the target rate.
    pub fn bytes_per_token(&self) -> u64 {
        (self.bytes_per_second as f64 / self.target_tokens_per_second).round() as u64
    }
}

impl ResidencyBudget {
    /// No constraint on either dimension — every selection is the
    /// backend's own preference, exactly as before a budget existed.
    pub const UNBOUNDED: Self = Self {
        physical_bytes: None,
        throughput: None,
        expert_access: super::realization::MappedAccess::Demand,
        prepare_bytes: None,
        fidelity: RepresentationFloor::Exact,
    };

    /// This machine's physical memory as the budget, read from the OS;
    /// unconstrained where the OS does not say.
    pub fn machine() -> Self {
        Self {
            physical_bytes: physical_memory_bytes(),
            throughput: None,
            expert_access: super::realization::MappedAccess::Demand,
            prepare_bytes: None,
            fidelity: RepresentationFloor::Exact,
        }
    }

    pub fn physical(bytes: u64) -> Self {
        Self {
            physical_bytes: Some(bytes),
            throughput: None,
            expert_access: super::realization::MappedAccess::Demand,
            prepare_bytes: None,
            fidelity: RepresentationFloor::Exact,
        }
    }

    pub fn with_throughput(mut self, throughput: ThroughputBudget) -> Self {
        self.throughput = Some(throughput);
        self
    }

    /// Stored bytes the plan may open to prepare itself.
    pub fn with_prepare_bytes(mut self, bytes: u64) -> Self {
        self.prepare_bytes = Some(bytes);
        self
    }

    /// The reconstruction the plan requires — the quality half of a
    /// budget, without which selection would take the cheapest extent
    /// every time and call it feasibility.
    pub fn with_fidelity(mut self, floor: RepresentationFloor) -> Self {
        self.fidelity = floor;
        self
    }

    pub fn with_expert_access(mut self, access: super::realization::MappedAccess) -> Self {
        self.expert_access = access;
        self
    }

    /// Whether `ledger` fits, and by how much it does not: the physical
    /// deficit and the per-token touch deficit, zero where it fits.
    pub fn deficit(&self, ledger: &ResourceLedger) -> BudgetDeficit {
        BudgetDeficit {
            physical: self
                .physical_bytes
                .map(|b| ledger.physical_working_set().saturating_sub(b))
                .unwrap_or(0),
            touch_per_token: self
                .throughput
                .map(|t| ledger.touch_per_token.saturating_sub(t.bytes_per_token()))
                .unwrap_or(0),
            prepare: self
                .prepare_bytes
                .map(|b| ledger.read_to_prepare.saturating_sub(b))
                .unwrap_or(0),
        }
    }
}

/// How far a ledger overshoots a budget, per constrained dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BudgetDeficit {
    pub physical: u64,
    pub touch_per_token: u64,
    /// Stored bytes the preparation would open over its budget.
    pub prepare: u64,
}

impl BudgetDeficit {
    pub fn is_zero(&self) -> bool {
        self.physical == 0 && self.touch_per_token == 0 && self.prepare == 0
    }
}

/// The machine's physical memory, from the OS.
pub fn physical_memory_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: sysconf reads two process-independent constants.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page > 0 {
            return Some(pages as u64 * page as u64);
        }
        None
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: MEMORYSTATUSEX is a plain C struct, so all-zero is a valid
        // value; `dwLength` is set as the API requires before the call, and
        // the struct outlives it.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok != 0 && status.ullTotalPhys > 0 {
            return Some(status.ullTotalPhys);
        }
        None
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

/// One expectation's demand on each resource a budget decision reads.
/// Seven numbers because they aggregate by SEVEN different rules — see
/// [`ResourceLedger::aggregate`] — and a single "resident" figure had
/// conflated a 98 GB mapping with 8 GB of committed memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Resources {
    /// Bytes the container stores for the operand.
    pub stored: u64,
    /// Address space a mapping of the stored bytes occupies; pages become
    /// resident only as touched.
    pub mapped: u64,
    /// Committed memory the realization keeps physically resident.
    pub resident: u64,
    /// Bytes materialised transiently on the way to residency.
    pub transient: u64,
    /// Bytes the host streams for this operation per position: the
    /// resident image, or the touched fraction of a mapping.
    pub touch_per_token: u64,
    /// Bytes a token is expected to page in cold from a mapping.
    pub page_in_per_token: u64,
    /// Bytes held on a device target.
    pub device: u64,
    /// Stored bytes opened once to prepare the operand at its pinned
    /// extent — every plane for a terminal representation, the extent's
    /// planes for a progressive one.
    pub read_to_prepare: u64,
}

/// A plan's demand on each resource, each aggregated by its own rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResourceLedger {
    /// Stored footprint: once per physical object (an operand bound
    /// under two operations is stored once).
    pub stored: u64,
    /// Mapped address space: once per mapping.
    pub mapped: u64,
    /// Persistent resident memory: every committed allocation, summed.
    pub resident: u64,
    /// Transient decode or staging: the MAXIMUM overlapping lifetime —
    /// the loader stages one operand at a time, so the peak is the
    /// largest, never the total.
    pub transient_peak: u64,
    /// Execution touch per token: summed over every operation instance.
    pub touch_per_token: u64,
    /// Expected cold page-in per token from mappings, summed.
    pub page_in_per_token: u64,
    /// Device memory, summed per target.
    pub device: u64,
    /// Stored bytes opened to prepare the plan: once per stored operand,
    /// like the footprint, because an operand bound once is read once
    /// however many operations it serves.
    pub read_to_prepare: u64,
}

impl ResourceLedger {
    /// Aggregate every expectation by the rule its resource carries.
    pub fn aggregate(expected: &[Expectation]) -> Self {
        let mut ledger = Self::default();
        let mut stored_seen = BTreeSet::new();
        let mut mapped_seen = BTreeSet::new();
        for e in expected {
            let r = e.resources();
            let object = (e.operand.object.clone(), e.operand.tensor.clone());
            if stored_seen.insert(object.clone()) {
                ledger.stored += r.stored;
                ledger.read_to_prepare += r.read_to_prepare;
            }
            // A dependency is ONE object however many owners resolve it:
            // its footprint and the reading that prepares it count once,
            // and only a realization that RETAINS it pays residency and
            // per-token touch for it.
            for dependency in &e.dependencies {
                let address = dependency.address();
                let first_time = stored_seen.insert(address);
                let bytes = dependency.stored_bytes.unwrap_or(0);
                if first_time {
                    ledger.stored += bytes;
                    ledger.read_to_prepare += bytes;
                }
                if dependency.lifetime == DependencyLifetime::Retained {
                    // Resident once, whoever keeps it; touched once per
                    // OPERATION that reads it, which is per owner.
                    let image = (dependency.elements as f64 * F32_WIDTH).round() as u64;
                    if first_time {
                        ledger.resident += image;
                    }
                    ledger.touch_per_token += image;
                }
            }
            if r.mapped > 0 && mapped_seen.insert(object) {
                ledger.mapped += r.mapped;
            }
            ledger.resident += r.resident;
            ledger.transient_peak = ledger.transient_peak.max(r.transient);
            ledger.touch_per_token += r.touch_per_token;
            ledger.page_in_per_token += r.page_in_per_token;
            ledger.device += r.device;
        }
        ledger
    }

    /// The physical working set a token needs on the host: committed
    /// memory, the staging peak, and the mapped bytes a token pages in.
    pub fn physical_working_set(&self) -> u64 {
        self.resident + self.transient_peak + self.page_in_per_token
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
                // Mapped as stored: resident exactly as the container
                // holds it, nothing staged on the way.
                RealizationForm::MappedStored { format, .. } => {
                    resident_profile_with(format, geometry)
                }
            };
            let staging = match realization.form {
                RealizationForm::Direct(_) | RealizationForm::MappedStored { .. } => 0,
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
            let stored_bytes = stored_len(&r.planned.operand).unwrap_or(0);
            Expectation {
                operand: r.planned.operand.clone(),
                operation: r.planned.operation,
                layer: r.planned.layer,
                realization,
                stored_bytes,
                logical_elements: logical,
                declared_resident: declared_resident_for(
                    &r.planned,
                    realization,
                    profile,
                    geometry,
                ),
                staging,
                // What the pin OPENS: the extent's own price where the
                // codec gives one, and otherwise the whole footprint —
                // an unpriced extent is read whole, never assumed cheap.
                read_to_prepare: r.extent.touch_bytes().unwrap_or(stored_bytes),
                dependencies: r.dependencies.clone(),
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
            mapped_bytes: self.weights.iter().map(|w| w.mapped_bytes() as u64).sum(),
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
    /// Bytes physically held, allocation padding included: committed
    /// allocations in full, a mapping's pages resident at the moment of
    /// observation.
    pub resident_bytes: u64,
    /// Address space held as a mapping of the container's segment; zero
    /// for an owned object.
    pub mapped_bytes: u64,
    /// Allocations the bytes live in: one for a matrix, one per expert
    /// for a bank.
    pub allocations: usize,
}

/// The result of a reconciliation that held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Reconciliation {
    pub matched: usize,
    /// Committed bytes declared and observed, over the owned objects.
    pub declared_resident: u64,
    pub observed_resident: u64,
    /// Observed minus declared — allocation padding, bounded per
    /// allocation by the page.
    pub padding: u64,
    /// Address space declared for mappings, held exactly against the
    /// mappings observed.
    pub mapped: u64,
    /// Pages of those mappings physically resident at observation — a
    /// fact about this moment, reported beside the declaration and never
    /// reconciled against it.
    pub mapped_resident: u64,
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
        // A mapping is held to its ADDRESS SPACE exactly; the pages of it
        // resident at this moment are a fact reported beside the
        // declaration, never reconciled against it.
        if e.resources().mapped > 0 {
            if o.mapped_bytes != e.declared_resident {
                return Err(VindexError::Parse(format!(
                    "operand `{}` ({}): {} declares {} mapped bytes over {} elements; {} are \
                     mapped — the declaration and the loader disagree",
                    e.operand.tensor,
                    e.operation.name(),
                    e.realization.name(),
                    e.declared_resident,
                    e.logical_elements,
                    o.mapped_bytes
                )));
            }
            out.matched += 1;
            out.mapped += e.declared_resident;
            out.mapped_resident += o.resident_bytes;
            continue;
        }
        if o.mapped_bytes != 0 {
            return Err(VindexError::Parse(format!(
                "operand `{}` ({}): pinned {} but a mapping of {} bytes is bound for it",
                e.operand.tensor,
                e.operation.name(),
                e.realization.name(),
                o.mapped_bytes
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
