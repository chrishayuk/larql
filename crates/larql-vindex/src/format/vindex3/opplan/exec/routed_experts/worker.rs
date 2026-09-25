use super::super::{
    accounting::{Bound, Observed},
    backend::{PlanBackend, WeightFormat},
    lowering::LoweringIdentity,
    operands::OperandSource,
    prepared::ExecutionSlice,
    realization::RealizationRecord,
    weights::{staged::StagedF32, LoadedWeight},
};
use super::*;
use crate::format::vindex3::{
    opplan::{planned::Operation, ComponentOpPlan, ExpertBank, LayerFfn, OperandRef, RoutedFfnOp},
    represent::codec::codecs::lyrw2::bind_region,
};

/// Declared before any payload read, including both quantization streams and
/// biases. Offsets are relative to the named tensor, never to a whole file.
#[derive(Clone, Debug, serde::Serialize, PartialEq)]
pub struct ExpertRegion {
    pub layer: usize,
    pub operand: OperandRef,
    pub total_bytes: u64,
    pub offset: u64,
    pub bytes: u64,
}
struct Layer {
    op: RoutedFfnOp,
    gate_up: Vec<LoadedWeight>,
    down: Vec<LoadedWeight>,
    gate_up_bias: Option<Vec<f32>>,
    down_bias: Option<Vec<f32>>,
}
pub struct PreparedRoutedExperts {
    first_layer: usize,
    first_expert: usize,
    hidden: usize,
    lowering: LoweringIdentity,
    layers: Vec<Layer>,
    regions: Vec<ExpertRegion>,
}
fn err(message: &str) -> VindexError {
    VindexError::Parse(message.into())
}
fn bytes(factors: &[usize]) -> Result<u64, VindexError> {
    factors.iter().try_fold(1u64, |n, x| {
        n.checked_mul(*x as u64)
            .ok_or_else(|| err("expert geometry overflow"))
    })
}
impl PreparedRoutedExperts {
    pub fn regions_for(
        plan: &ComponentOpPlan,
        store: OperandSource<'_>,
        slice: &ExecutionSlice,
    ) -> Result<Vec<ExpertRegion>, VindexError> {
        slice.validate(plan)?;
        let ExecutionSlice::RoutedExperts {
            expert_start,
            expert_end,
            ..
        } = slice
        else {
            return Err(err("expert worker slice required"));
        };
        let mut regions = Vec::new();
        for layer in slice.layers(plan) {
            let Some(LayerFfn::Routed(op)) = &plan.layers[layer].ffn else {
                unreachable!()
            };
            let hidden = op.router.shape[1];
            let inter = op.expert_intermediate_size;
            if !hidden.is_multiple_of(32) || !inter.is_multiple_of(32) {
                return Err(err("MXFP4 expert width must align to 32"));
            }
            let ExpertBank::Packed { gate_up, down } = &op.bank else {
                unreachable!()
            };
            for (projection, rows, k) in [(gate_up, 2 * inter, hidden), (down, hidden, inter)] {
                let scale = projection
                    .scales
                    .as_ref()
                    .ok_or_else(|| err("MXFP4 expert scale stream missing"))?;
                for (operand, per_expert) in [
                    (&projection.weights, bytes(&[rows, k / 32, 16])?),
                    (scale, bytes(&[rows, k / 32])?),
                ] {
                    regions.push(ExpertRegion {
                        layer,
                        operand: operand.clone(),
                        total_bytes: per_expert
                            .checked_mul(op.experts as u64)
                            .ok_or_else(|| err("expert bank size overflow"))?,
                        offset: per_expert * *expert_start as u64,
                        bytes: per_expert * (expert_end - expert_start) as u64,
                    });
                }
                if let Some(bias) = &projection.bias {
                    let width = match store.store().stored_dtype(bias) {
                        Some("F32") => 4,
                        Some("BF16" | "F16") => 2,
                        _ => return Err(err("expert biases require stored F32, BF16 or F16")),
                    };
                    let per_expert = bytes(&[rows, width])?;
                    regions.push(ExpertRegion {
                        layer,
                        operand: bias.clone(),
                        total_bytes: per_expert
                            .checked_mul(op.experts as u64)
                            .ok_or_else(|| err("expert bank size overflow"))?,
                        offset: per_expert * *expert_start as u64,
                        bytes: per_expert * (expert_end - expert_start) as u64,
                    });
                }
            }
        }
        Ok(regions)
    }
    pub(in super::super) fn load<B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: OperandSource<'_>,
        backend: &B,
        slice: &ExecutionSlice,
        pins: &[RealizationRecord],
        hidden: usize,
    ) -> Result<Self, VindexError> {
        if backend.identity() != LoweringIdentity::cpu_production()
            || pins.iter().any(|r| {
                r.selection.realization.format() != WeightFormat::F32 || r.representation != "MXFP4"
            })
        {
            return Err(err(
                "expert worker requires CPU production MXFP4-to-F32 lowering",
            ));
        }
        let regions = Self::regions_for(plan, store, slice)?;
        let ExecutionSlice::RoutedExperts {
            expert_start,
            expert_end,
            ..
        } = slice
        else {
            unreachable!()
        };
        let owned = expert_end - expert_start;
        let mut layers = Vec::new();
        for index in slice.layers(plan) {
            let Some(LayerFfn::Routed(op)) = &plan.layers[index].ffn else {
                unreachable!()
            };
            let read = |operand: &OperandRef| {
                let r = regions
                    .iter()
                    .find(|r| r.layer == index && r.operand == *operand)
                    .ok_or_else(|| err("expert operand outside preparation ledger"))?;
                store
                    .store()
                    .load_raw_range(operand, r.total_bytes, r.offset, r.bytes)
            };
            let projection = |p: &crate::format::vindex3::opplan::PackedProjection,
                              rows: usize,
                              k: usize|
             -> Result<Vec<LoadedWeight>, VindexError> {
                let raw = read(&p.weights)?;
                let scales = read(
                    p.scales
                        .as_ref()
                        .ok_or_else(|| err("missing expert scales"))?,
                )?;
                if raw.dtype != "U8" || scales.dtype != "U8" {
                    return Err(err("MXFP4 expert streams require U8 storage"));
                }
                let codec = store.registry().resolve("MXFP4", &p.weights.tensor)?;
                let shape = [owned * rows, k];
                let operands = bind_region(
                    codec,
                    &shape,
                    &raw.bytes,
                    Some(&scales.bytes),
                    &p.weights.tensor,
                )?;
                (0..owned)
                    .map(|e| {
                        let mut values = vec![0.0; rows * k];
                        codec.decode_rows(
                            &operands,
                            &shape,
                            e * rows..(e + 1) * rows,
                            codec.terminal_extent(),
                            &mut values,
                            &p.weights.tensor,
                        )?;
                        Ok(LoadedWeight::F32(StagedF32::stage(values)?))
                    })
                    .collect()
            };
            let bias = |operand: &Option<OperandRef>,
                        rows: usize|
             -> Result<Option<Vec<f32>>, VindexError> {
                operand
                    .as_ref()
                    .map(|operand| {
                        let raw = read(operand)?;
                        let codec = store.registry().resolve(&raw.dtype, &operand.tensor)?;
                        let shape = [owned, rows];
                        let operands =
                            bind_region(codec, &shape, &raw.bytes, None, &operand.tensor)?;
                        let mut values = vec![0.0; owned * rows];
                        codec.decode_rows(
                            &operands,
                            &shape,
                            0..owned,
                            codec.terminal_extent(),
                            &mut values,
                            &operand.tensor,
                        )?;
                        Ok(values)
                    })
                    .transpose()
            };
            let ExpertBank::Packed { gate_up, down } = &op.bank else {
                unreachable!()
            };
            layers.push(Layer {
                op: (**op).clone(),
                gate_up: projection(gate_up, 2 * op.expert_intermediate_size, hidden)?,
                down: projection(down, hidden, op.expert_intermediate_size)?,
                gate_up_bias: bias(&gate_up.bias, 2 * op.expert_intermediate_size)?,
                down_bias: bias(&down.bias, hidden)?,
            });
        }
        let prepared = Self {
            first_layer: slice.layers(plan).start,
            first_expert: *expert_start,
            hidden,
            lowering: backend.identity(),
            layers,
            regions,
        };
        backend.prepare(
            &prepared
                .matrices()
                .iter()
                .map(|w| w.slice())
                .collect::<Vec<_>>(),
        );
        Ok(prepared)
    }
    pub fn regions(&self) -> &[ExpertRegion] {
        &self.regions
    }
    pub fn apply<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        layer: usize,
        experts: &[usize],
        x: &[f32],
    ) -> Result<Vec<ExpertOutput>, VindexError> {
        if backend.identity() != self.lowering {
            return Err(err("expert numerical provider mismatch"));
        }
        super::super::dense_ffn::validate_row(x, self.hidden)?;
        let layer = layer
            .checked_sub(self.first_layer)
            .and_then(|l| self.layers.get(l))
            .ok_or_else(|| err("layer outside expert worker"))?;
        if experts.is_empty() || experts.len() > layer.op.top_k {
            return Err(err("expert request count outside top-k bound"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for expert in experts {
            if !seen.insert(*expert)
                || expert
                    .checked_sub(self.first_expert)
                    .is_none_or(|e| e >= layer.gate_up.len())
            {
                return Err(err("duplicate or unowned expert"));
            }
        }
        experts
            .iter()
            .map(|expert| {
                let e = expert - self.first_expert;
                let inter = layer.op.expert_intermediate_size;
                let row = backend.expert_transform(ExpertTransformCall {
                    x,
                    hidden: self.hidden,
                    intermediate: inter,
                    activation: layer.op.activation,
                    gate_policy: layer.op.gate_policy,
                    weights: ExpertWeights::Fused {
                        gate_up: layer.gate_up[e].slice(),
                        down: layer.down[e].slice(),
                        layout: layer.op.gate_up_layout.expect("validated layout"),
                        gate_up_bias: layer
                            .gate_up_bias
                            .as_ref()
                            .map(|b| &b[e * 2 * inter..(e + 1) * 2 * inter]),
                        down_bias: layer
                            .down_bias
                            .as_ref()
                            .map(|b| &b[e * self.hidden..(e + 1) * self.hidden]),
                    },
                })?;
                super::super::dense_ffn::validate_row(&row, self.hidden)?;
                Ok(ExpertOutput {
                    expert: *expert,
                    row,
                })
            })
            .collect()
    }
    pub(in super::super) fn bias_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                l.gate_up_bias
                    .as_ref()
                    .map_or(0, |b| std::mem::size_of_val(&b[..]))
                    + l.down_bias
                        .as_ref()
                        .map_or(0, |b| std::mem::size_of_val(&b[..]))
            })
            .sum()
    }
    pub(in super::super) fn matrices(&self) -> Vec<&LoadedWeight> {
        self.layers
            .iter()
            .flat_map(|l| l.gate_up.iter().chain(&l.down))
            .collect()
    }
    pub(in super::super) fn bound(&self) -> Result<Vec<Observed>, VindexError> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            let ExpertBank::Packed { gate_up, down } = &l.op.bank else {
                unreachable!()
            };
            for (operand, weights) in [(&gate_up.weights, &l.gate_up), (&down.weights, &l.down)] {
                out.push(
                    Bound {
                        operand,
                        weights: weights.iter().collect(),
                    }
                    .observed(Operation::ExpertBankSlice, Some(self.first_layer + i))?,
                );
            }
        }
        Ok(out)
    }
}
