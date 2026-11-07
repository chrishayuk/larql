//! Selecting realizations and the output head.

use super::super::super::{ComponentOpPlan, OperandRef, OutputOp};
use super::super::accounting::ResidencyBudget;
use super::super::backend::{PlanBackend, WeightFormat, WeightSlice};
use super::super::cpu::WeightRows;
use super::super::operands::{OperandSource, OperandStore, RepresentationSource, SourceStamp};
use super::super::quantise::SUM_BLOCK;
use super::super::realization::{
    DependencyLifetime, DependencyPin, ExtentOption, ExtentPin, RealizationRecord,
    RepresentationFacts, SelectionRefusals,
};
use super::super::weights::LoadedWeight;
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::represent::codec::{CodecRegistry, RepresentationExtent};
use larql_models::config::ResidualTopology;

#[allow(unused_imports)]
use super::*;

/// Every planned operand in `slice` with its representation facts and
/// the backend's selection, or the complete list of refusals.
pub(super) fn select_records<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: OperandSource<'_>,
    backend: &B,
    slice: &ExecutionSlice,
) -> Result<Vec<(RealizationRecord, RepresentationFacts)>, VindexError> {
    let registry = store.registry();
    // Who is lowering this plan, asked once: the pin records the
    // provider that qualified it, whether the caller resolved that
    // provider through a registry or handed it in directly.
    let lowering_provider = backend.identity();
    let mut records = Vec::new();
    let mut refusals = Vec::new();
    for mut planned in plan.planned_operands() {
        let in_scope = slice.contains(plan, &planned);
        if !in_scope {
            continue;
        }
        let Some(stored) = store.stored_dtype(&planned.operand) else {
            continue;
        };
        let mut facts = RepresentationFacts::resolve_declared(
            registry,
            stored,
            planned.declared_representation,
        );
        let label = facts.label.clone();
        if store.is_overridden(&planned.operand) {
            facts = facts.overlaid();
        }
        facts = facts.with_nvfp4_at_source(
            store
                .store()
                .nvfp4_request_binds_at_source(&planned.operand, stored),
        );
        facts = facts.with_nvfp4_compiled(OperandStore::f16_request_binds_compiled_nvfp4(stored));
        facts = facts.with_requantise_forbidden(
            store.store().representation_source() == RepresentationSource::Stored,
        );
        let codec_provider = facts.registered.as_ref().map(|r| r.identity.clone());
        // What the ARTIFACT offers, priced per extent from the codec's own
        // declaration. The pin starts on the whole of it; a budget may
        // move it shallower, and nothing else may.
        let extent = extent_pin(registry, &label, &planned);
        let mut dependencies = dependency_pins(registry, &label, &planned, extent.selected, store);
        match backend.select(&planned, &facts) {
            Ok(selection) => {
                // The lifetime is the REALIZATION's: a decode is finished
                // with its dependency once it has an f32 image, and a
                // direct kernel over the stored codes keeps it for every
                // token. Decided here, after the pin, because nothing
                // before the pin knows which.
                let lifetime = if selection.realization.retains_dependencies() {
                    DependencyLifetime::Retained
                } else {
                    DependencyLifetime::PreparationOnly
                };
                for dependency in &mut dependencies {
                    dependency.lifetime = lifetime;
                }
                if let ExecutionSlice::RoutedExperts {
                    expert_start,
                    expert_end,
                    ..
                } = slice
                {
                    let count = planned.operand.shape[0];
                    planned.logical_elements =
                        planned.logical_elements / count * (expert_end - expert_start);
                }
                records.push((
                    RealizationRecord {
                        representation: label.to_string(),
                        codec_provider,
                        lowering_provider: lowering_provider.clone(),
                        planned,
                        selection,
                        extent,
                        dependencies,
                        // Filled in by the carriage pass below, which is
                        // the only thing that reads a payload to verify a
                        // claim.
                        verified_bytes: 0,
                    },
                    facts,
                ))
            }
            Err(refusal) => refusals.push(*refusal),
        }
    }
    if !refusals.is_empty() {
        return Err(VindexError::Parse(SelectionRefusals(refusals).to_string()));
    }
    // The floor gates selection, so the certificate it judges has to be
    // the derived one BEFORE anything consults it. Refusals first: a
    // plan that is already refused should not pay for verification.
    super::super::fidelity_carriage::compose_extent_certificates(&mut records, store)?;
    Ok(records)
}

/// The largest saving in bytes-opened any record can make by taking a
/// shallower extent its fidelity floor admits: `(record, extent, saving)`.
///
/// One step at a time, like the realization re-selection beside it, so
/// every move is recorded and the plan gives up exactly as much fidelity
/// as the budget forced and no more.
pub(super) fn shallowest_saving(
    selected: &[(RealizationRecord, RepresentationFacts)],
    budget: &ResidencyBudget,
) -> Option<(usize, RepresentationExtent, u64)> {
    let mut best: Option<(usize, RepresentationExtent, u64)> = None;
    for (i, (record, _)) in selected.iter().enumerate() {
        let Some(now) = record.extent.touch_bytes() else {
            continue;
        };
        let terminal = record
            .extent
            .options
            .iter()
            .map(|o| o.certificate.extent)
            .max()
            .unwrap_or(RepresentationExtent::BASE);
        for option in &record.extent.options {
            if option.certificate.extent == record.extent.selected
                || !budget.fidelity.admits(option, terminal)
            {
                continue;
            }
            let Some(then) = option.stored_bytes else {
                continue;
            };
            if then >= now {
                continue;
            }
            let saving = now - then;
            if best.is_none_or(|(_, _, s)| saving > s) {
                best = Some((i, option.certificate.extent, saving));
            }
        }
    }
    best
}

/// What `planned`'s codec depends on at `extent`, as the container's
/// reference table addresses it, priced from the container's record.
///
/// The LIFETIME is the realization's and is not known here: every pin
/// starts `PreparationOnly` and [`select_records`] sets it from the
/// realization the backend pinned. A direct kernel over codes — the FP8
/// codes with their scale grid — retains its dependency, and the ledger
/// prices exactly that.
pub(super) fn dependency_pins(
    registry: &CodecRegistry,
    label: &str,
    planned: &PlannedOperand,
    extent: RepresentationExtent,
    store: OperandSource<'_>,
) -> Vec<DependencyPin> {
    let Some(codec) = registry.by_label(label) else {
        return Vec::new();
    };
    let required = codec.required_auxiliaries(extent);
    if required.is_empty() {
        return Vec::new();
    }
    let owner = crate::format::vindex3::auxiliary_references::OperandAddress::new(
        &planned.operand.object,
        &planned.operand.tensor,
    );
    let table = store.store().references();
    required
        .iter()
        .filter_map(|spec| {
            let target = table.target(&owner, spec.name)?;
            let reference = OperandRef {
                object: target.object.clone(),
                tensor: target.tensor.clone(),
                dtype: String::new(),
                shape: Vec::new(),
            };
            let label = store
                .store()
                .stored_dtype(&reference)
                .unwrap_or_default()
                .to_string();
            Some(DependencyPin {
                name: spec.name.to_string(),
                object: target.object.clone(),
                tensor: target.tensor.clone(),
                provider: registry.by_label(&label).map(|c| c.identity()),
                label,
                stored_bytes: store.stored_len(&reference),
                elements: store
                    .store()
                    .stored_shape(target)
                    .map(|shape| shape.iter().product())
                    .unwrap_or(0),
                lifetime: DependencyLifetime::PreparationOnly,
            })
        })
        .collect()
}

/// What extents `label` offers for `planned`, priced per extent, with the
/// pin on the whole of it.
///
/// The price is the CODEC's, from the shape: the container's recorded
/// length is the operand's whole stored footprint (every plane), and what
/// changes with the extent is how much of that footprint execution reads.
/// A codec that cannot price a shape — an entropy-coded one — offers its
/// extents unpriced, and the container's length stays the authority.
pub(super) fn extent_pin(
    registry: &CodecRegistry,
    label: &str,
    planned: &PlannedOperand,
) -> ExtentPin {
    let Some(codec) = registry.by_label(label) else {
        return ExtentPin::unknown();
    };
    ExtentPin::whole(
        codec
            .extents()
            .into_iter()
            .map(|certificate| {
                let stored_bytes = codec
                    .stored_bytes(
                        &planned.operand.shape,
                        certificate.extent,
                        &planned.operand.tensor,
                    )
                    .ok();
                ExtentOption {
                    certificate,
                    stored_bytes,
                }
            })
            .collect(),
    )
}

/// The representation pinned for `op` under `operation`.
///
/// An operand with no record is either absent from the container — the
/// loader refuses it by name a moment later, so any answer here is
/// unread — or one the plan's own view failed to list, which is a
/// disagreement between `planned_operands()` and the loader and is
/// refused as such rather than defaulted.
pub(in super::super) fn pinned_format(
    records: &[RealizationRecord],
    store: OperandSource<'_>,
    op: &OperandRef,
    operation: Operation,
) -> Result<WeightFormat, VindexError> {
    if let Some(record) = records.iter().find(|r| {
        r.planned.operand.object == op.object
            && r.planned.operand.tensor == op.tensor
            && r.planned.operation == operation
    }) {
        return Ok(record.selection.realization.format());
    }
    if store.stored_dtype(op).is_none() {
        return Ok(WeightFormat::F32);
    }
    Err(VindexError::Parse(format!(
        "tensor `{}` ({}): the loader resolves an operand the plan's own view does not list — \
         planned_operands() and the loader disagree",
        op.tensor,
        operation.name()
    )))
}

/// The resident form pinned for layer `index`'s bank: a packed bank's
/// one slice pin, or the one form every matrix of a per-expert bank was
/// pinned to — a bank whose experts were pinned differently is a plan
/// the loader cannot bind, and is refused by name.
/// What the bank's pins agree on: the resident form, and how a mapped
/// form is accessed per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankPin {
    pub format: WeightFormat,
    pub access: super::super::realization::MappedAccess,
}

impl From<WeightFormat> for BankPin {
    /// A format alone is a pin under demand access — the loader's
    /// behaviour before access was a declared policy, and the one every
    /// non-mapped form has.
    fn from(format: WeightFormat) -> Self {
        Self {
            format,
            access: super::super::realization::MappedAccess::Demand,
        }
    }
}

pub(super) fn bank_pin(
    records: &[RealizationRecord],
    index: usize,
) -> Result<BankPin, VindexError> {
    let mut pins = records
        .iter()
        .filter(|r| {
            r.planned.layer == Some(index)
                && matches!(
                    r.planned.operation,
                    Operation::ExpertBankSlice | Operation::ExpertProject { .. }
                )
        })
        .map(|r| BankPin {
            format: r.selection.realization.format(),
            access: r.selection.realization.access(),
        });
    let Some(first) = pins.next() else {
        return Err(VindexError::Parse(format!(
            "layer {index}: a routed FFN with no pinned bank realization — planned_operands() \
             and the loader disagree"
        )));
    };
    if let Some(other) = pins.find(|f| *f != first) {
        return Err(VindexError::Parse(format!(
            "layer {index}: the bank's experts were pinned to {first:?} and {other:?}; one bank \
             binds in one form"
        )));
    }
    Ok(first)
}

/// A component's operands, lowered once for a given slice and backend.
///
/// Immutable for its lifetime: this is the canonical base model. A
/// session that carries an overlay composes *over* these operands
/// rather than mutating them, so one prepared image can serve every
/// concurrent request on the model.
pub struct PreparedOperands {
    /// The registry this image was prepared through — the store's — and
    /// the one execution re-checks its providers against.
    pub(super) registry: &'static CodecRegistry,
    /// Which effective source this image was compiled from.
    pub(super) stamp: SourceStamp,
    pub(super) slice: ExecutionSlice,
    pub(super) hidden: usize,
    /// Present only for a slice that carries the stack's input end.
    pub(super) embed_table: Option<Vec<f32>>,
    /// Plan index of `layers[0]`, so a sliced image can still address
    /// the plan's per-layer ops and the KV state's layer rows.
    pub(super) first_layer: usize,
    pub(super) layers: Vec<PreparedLayer>,
    pub(super) dense_ffns: Option<super::super::dense_ffn::PreparedDenseFfns>,
    pub(super) routed_experts: Option<super::super::routed_experts::PreparedRoutedExperts>,
    pub(super) final_norm: Option<PreparedNorm>,
    pub(super) output: Option<(OutputOp, LoadedWeight)>,
    /// One pinned realization per planned operand this image executes,
    /// in the plan's order — the record the trace reads.
    pub(super) realizations: Vec<RealizationRecord>,
    /// The component's residual topology, carried so the traversal reads
    /// the stream count from the image it executes.
    pub(super) topology: ResidualTopology,
    /// The head's reduction — present only on a whole-stack image of a
    /// hyper-connected component.
    pub(super) hyper_connection_head: Option<PreparedHcHead>,
    /// The exit reduction — present only on a whole-stack image of an
    /// attention-residual component, and required there.
    pub(super) attention_residual_exit: Option<PreparedAttnResExit>,
}

/// A gathered subset of the prepared output head.
///
/// This is an offline evidence object: it preserves the exact resident
/// representation selected for the production image, but contains only the
/// declared vocabulary rows.  It exists so a frozen candidate lens does not
/// have to execute the entire vocabulary projection at every captured state.
pub struct SelectedOutputHead {
    pub(super) token_ids: Vec<u32>,
    pub(super) projection: SelectedProjection,
    pub(super) hidden: usize,
    pub(super) multiplier: Option<f64>,
    pub(super) softcapping: Option<f32>,
}

pub(super) enum SelectedProjection {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    Q8 {
        codes: Vec<i8>,
        scales: Vec<f32>,
        sums: Vec<i16>,
        block: usize,
    },
}

impl SelectedProjection {
    pub(super) fn slice(&self) -> WeightSlice<'_> {
        match self {
            Self::F32(values) => WeightSlice::F32(values),
            Self::Bf16(values) => WeightSlice::Bf16(values),
            Self::Q8 {
                codes,
                scales,
                sums,
                block,
            } => WeightSlice::Q8 {
                codes,
                scales,
                sums,
                block: *block,
            },
        }
    }

    pub(super) fn representation(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::Bf16(_) => "bf16",
            Self::Q8 { .. } => "q8",
        }
    }
}

impl SelectedOutputHead {
    /// Gather `token_ids`' rows of a resident `[vocab, hidden]` head, in its
    /// own representation. Every id must be a distinct row of the head.
    pub(in super::super) fn gather(
        head: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        token_ids: &[u32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Self, VindexError> {
        if token_ids.is_empty() {
            return Err(VindexError::Parse(
                "selected output head requires at least one token".into(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for &token in token_ids {
            let index = token as usize;
            if index >= vocab {
                return Err(VindexError::Parse(format!(
                    "selected output token {token} is outside vocabulary {vocab}"
                )));
            }
            if !seen.insert(token) {
                return Err(VindexError::Parse(format!(
                    "selected output token {token} is duplicated"
                )));
            }
        }

        let projection = match head.rows(vocab, hidden)? {
            WeightRows::F32(values) => {
                let mut selected = Vec::with_capacity(token_ids.len() * hidden);
                for &token in token_ids {
                    let start = token as usize * hidden;
                    selected.extend_from_slice(&values[start..start + hidden]);
                }
                SelectedProjection::F32(selected)
            }
            WeightRows::Bf16(values) => {
                let mut selected = Vec::with_capacity(token_ids.len() * hidden);
                for &token in token_ids {
                    let start = token as usize * hidden;
                    selected.extend_from_slice(&values[start..start + hidden]);
                }
                SelectedProjection::Bf16(selected)
            }
            WeightRows::Q8 {
                codes,
                scales,
                sums,
                block,
            } => {
                let scales_per_row = hidden.div_ceil(block);
                let sums_per_row = hidden.div_ceil(SUM_BLOCK);
                let mut selected_codes = Vec::with_capacity(token_ids.len() * hidden);
                let mut selected_scales = Vec::with_capacity(token_ids.len() * scales_per_row);
                let mut selected_sums = if sums.is_empty() {
                    Vec::new()
                } else {
                    Vec::with_capacity(token_ids.len() * sums_per_row)
                };
                for &token in token_ids {
                    let row = token as usize;
                    selected_codes.extend_from_slice(&codes[row * hidden..(row + 1) * hidden]);
                    selected_scales.extend_from_slice(
                        &scales[row * scales_per_row..(row + 1) * scales_per_row],
                    );
                    if !sums.is_empty() {
                        selected_sums
                            .extend_from_slice(&sums[row * sums_per_row..(row + 1) * sums_per_row]);
                    }
                }
                SelectedProjection::Q8 {
                    codes: selected_codes,
                    scales: selected_scales,
                    sums: selected_sums,
                    block,
                }
            }
            _ => {
                return Err(VindexError::Parse(
                    "selected output-head evidence refuses this prepared representation".into(),
                ))
            }
        };
        Ok(Self {
            token_ids: token_ids.to_vec(),
            projection,
            hidden,
            multiplier,
            softcapping,
        })
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn representation(&self) -> &'static str {
        self.projection.representation()
    }

    /// The raw output-head row for the `index`-th selected token, dequantised
    /// to f32 from the exact prepared production realization — never a
    /// second, independently reloaded or widened copy. Dequantisation
    /// (bf16 widening, Q8 `code * scale`) is arithmetic, not a different
    /// weight authority: it is the same conversion `backend.project`/
    /// `output_head` already apply on this exact resident data.
    pub fn row_f32(&self, index: usize) -> Result<Vec<f32>, VindexError> {
        if index >= self.token_ids.len() {
            return Err(VindexError::Parse(format!(
                "selected output head row {index} is outside {} selected tokens",
                self.token_ids.len()
            )));
        }
        match &self.projection {
            SelectedProjection::F32(values) => {
                let start = index * self.hidden;
                Ok(values[start..start + self.hidden].to_vec())
            }
            SelectedProjection::Bf16(values) => {
                let start = index * self.hidden;
                Ok(values[start..start + self.hidden]
                    .iter()
                    .map(|&bits| f32::from_bits(u32::from(bits) << 16))
                    .collect())
            }
            SelectedProjection::Q8 {
                codes,
                scales,
                block,
                ..
            } => {
                let scales_per_row = self.hidden.div_ceil(*block);
                let code_start = index * self.hidden;
                let scale_start = index * scales_per_row;
                Ok(codes[code_start..code_start + self.hidden]
                    .iter()
                    .enumerate()
                    .map(|(offset, &code)| {
                        let scale = scales[scale_start + offset / block];
                        f32::from(code) * scale
                    })
                    .collect())
            }
        }
    }
}

/// The effective dense FFN matrices selected for execution, widened only for
/// offline contribution accounting. A Q8 image contains the dequantised Q8
/// values, not the container's original BF16 values.
pub struct PreparedDenseFfnImage {
    pub gate: Option<Vec<f32>>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}
