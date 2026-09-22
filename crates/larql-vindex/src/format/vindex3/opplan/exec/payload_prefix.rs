//! Measurement adapter for a canonical prefix and one source V projection.
//! Uses the decode interpreter and prepared production operands. No new kernels.
use super::ComponentOpPlan;
use super::backend::{PlanBackend, ProjectCall, WeightSlice};
use super::decode::DecodeSession;
use super::kv::RowKvState;
use super::observe::{CarrierWriteRecord, StepEvent, StepObserver};
use super::operands::OperandStore;
use super::prepared::{ExecutionSlice, PreparedAttention, PreparedOperands};
use super::quantise::SUM_BLOCK;
use crate::error::VindexError;

/// Prefix-only image. It deliberately has no final norm or vocabulary head.
pub struct PayloadPrefix {
    plan: ComponentOpPlan,
    operands: PreparedOperands,
    depth: usize,
}

#[derive(Default)]
struct LastCarrier(Vec<f32>);
impl StepObserver for LastCarrier {
    fn event(&mut self, _: StepEvent) {}
    fn entering_carrier(&mut self, _: usize, values: &[f32]) {
        self.0 = values.to_vec();
    }
    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        self.0 = record.after.to_vec();
        if let Some(scale) = record.layer_scale {
            for v in &mut self.0 {
                *v *= scale;
            }
        }
    }
}

impl PayloadPrefix {
    /// Prepare exactly layers 0..depth, embedding included; source layer excluded.
    pub fn prepare<B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: &OperandStore,
        backend: &B,
        depth: usize,
    ) -> Result<Self, VindexError> {
        if depth > plan.layers.len() || !plan.residual_topology.is_single_stream() {
            return Err(VindexError::Parse(
                "payload prefix requires an in-range single-stream plan".into(),
            ));
        }
        let mut prefix = plan.clone();
        prefix.layers.truncate(depth);
        prefix.final_norm = None;
        prefix.output = None;
        let operands = PreparedOperands::load(&prefix, store, backend, ExecutionSlice::Full)?;
        Ok(Self {
            plan: prefix,
            operands,
            depth,
        })
    }

    /// Canonical lower-layer carrier at each supplied token, without an exit readout.
    pub fn carriers<B: PlanBackend>(
        &self,
        tokens: &[u32],
        backend: &B,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        if tokens.is_empty() {
            return Err(VindexError::Parse("empty payload prefix".into()));
        }
        let mut kv = RowKvState::default();
        let mut session =
            DecodeSession::over_prepared(&self.plan, &self.operands, backend, &mut kv)?;
        let mut result = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let mut observer = LastCarrier::default();
            session.step_observed(token, &mut observer)?;
            if observer.0.len() != self.operands.hidden() {
                return Err(VindexError::Parse("missing prefix carrier".into()));
            }
            result.push(observer.0);
        }
        Ok(result)
    }

    /// Natural query-head-1 V (KV head zero) at the requested source positions.
    /// Later prompt positions are not required to construct these causal states.
    pub fn values<B: PlanBackend>(
        &self,
        tokens: &[u32],
        positions: &[usize],
        full_plan: &ComponentOpPlan,
        full: &PreparedOperands,
        backend: &B,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        let last = positions
            .iter()
            .copied()
            .max()
            .ok_or_else(|| VindexError::Parse("empty subject positions".into()))?;
        if last >= tokens.len() {
            return Err(VindexError::Parse("subject position outside prompt".into()));
        }
        let carriers = self.carriers(&tokens[..=last], backend)?;
        let op = full_plan
            .layers
            .get(self.depth)
            .and_then(|l| l.attention.softmax())
            .ok_or_else(|| VindexError::Parse("source layer is not softmax".into()))?;
        if op.num_q_heads != 8
            || op.num_kv_heads != 4
            || op.head_dim != 256
            || op.v_bias.is_some()
            || op.v_from_k
            || op.parameter_free_qk_norm.v
        {
            return Err(VindexError::Parse(
                "payload measurement requires the frozen unbiased 8Q/4KV/256 V geometry".into(),
            ));
        }
        let layer = full
            .layers()
            .get(self.depth)
            .ok_or_else(|| VindexError::Parse("source layer absent".into()))?;
        let PreparedAttention::Softmax(attention) = &layer.attention else {
            return Err(VindexError::Parse("prepared source is not softmax".into()));
        };
        let hidden = full.hidden();
        let size = op.head_dim * hidden;
        // Head 1 shares KV head 0, so this is a row prefix, not a re-quantisation.
        let weight = match attention.w_v.slice() {
            WeightSlice::F32(v) => WeightSlice::F32(&v[..size]),
            WeightSlice::Bf16(v) => WeightSlice::Bf16(&v[..size]),
            WeightSlice::Q8 {
                codes,
                scales,
                sums,
                block,
            } => WeightSlice::Q8 {
                codes: &codes[..size],
                scales: &scales[..op.head_dim * hidden.div_ceil(block)],
                sums: if sums.is_empty() {
                    sums
                } else {
                    &sums[..op.head_dim * hidden.div_ceil(SUM_BLOCK)]
                },
                block,
            },
            _ => {
                return Err(VindexError::Parse(
                    "payload measurement refuses an untested V representation".into(),
                ));
            }
        };
        positions
            .iter()
            .map(|&p| {
                let normalized = match &layer.pre_attention {
                    Some(norm) => norm.apply(backend, &carriers[p]),
                    None => carriers[p].clone(),
                };
                backend.project(ProjectCall {
                    weight,
                    out_dim: op.head_dim,
                    in_dim: hidden,
                    x: &normalized,
                })
            })
            .collect()
    }

    pub fn resident_bytes(&self) -> usize {
        self.operands.residency_census().total()
    }
}
