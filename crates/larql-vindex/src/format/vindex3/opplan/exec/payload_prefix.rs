//! Measurement adapter for a canonical prefix and one source V projection.
//! Uses the decode interpreter and prepared production operands. No new kernels.
use super::backend::{PlanBackend, ProjectCall, WeightSlice};
use super::decode::DecodeSession;
use super::kv::RowKvState;
use super::observe::{CarrierWriteRecord, StepEvent, StepObserver};
use super::operands::OperandStore;
use super::prepared::{ExecutionSlice, PreparedAttention, PreparedOperands};
use super::quantise::SUM_BLOCK;
use super::ComponentOpPlan;
use crate::error::VindexError;
use std::ops::Range;

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

    /// Natural V for one declared query head at the requested source
    /// positions: the source layer's V rows for that head's KV group,
    /// applied to the canonical prefix carrier after the layer's
    /// pre-attention norm. Later prompt positions are not required to
    /// construct these causal states.
    ///
    /// The head is the caller's declaration; its KV group comes from the
    /// plan's own GQA geometry, so the rows taken are a row range of the
    /// prepared projection — never a re-quantisation. A V bias, a V taken
    /// from K, or a parameter-free V norm would make the projection alone
    /// not the head's V, and refuses.
    #[allow(clippy::too_many_arguments)]
    pub fn values<B: PlanBackend>(
        &self,
        tokens: &[u32],
        positions: &[usize],
        query_head: usize,
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
        let op = full_plan
            .layers
            .get(self.depth)
            .and_then(|l| l.attention.softmax())
            .ok_or_else(|| VindexError::Parse("source layer is not softmax".into()))?;
        if query_head >= op.num_q_heads {
            return Err(VindexError::Parse(format!(
                "query head {query_head} is outside the source layer's {} heads",
                op.num_q_heads
            )));
        }
        if op.v_bias.is_some() || op.v_from_k || op.parameter_free_qk_norm.v {
            return Err(VindexError::Parse(
                "payload measurement requires an unbiased, projected, unnormalised V".into(),
            ));
        }
        let layer = full
            .layers()
            .get(self.depth)
            .ok_or_else(|| VindexError::Parse("source layer absent".into()))?;
        let PreparedAttention::Softmax(attention) = &layer.attention else {
            return Err(VindexError::Parse("prepared source is not softmax".into()));
        };
        let carriers = self.carriers(&tokens[..=last], backend)?;
        let hidden = full.hidden();
        let kv_head = query_head / (op.num_q_heads / op.num_kv_heads);
        let rows = kv_head * op.head_dim..(kv_head + 1) * op.head_dim;
        let weight = row_range(attention.w_v.slice(), rows, hidden)?;
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

/// The contiguous `rows` of a resident `[_, in_dim]` matrix, in its own
/// representation: a view, never a re-quantisation. Q8 scales and sums are
/// per-row blocks, so they cut on the same row boundaries; an empty sums
/// index stays empty. A representation this has not been tested on
/// refuses rather than guessing its row geometry.
pub(super) fn row_range(
    weight: WeightSlice<'_>,
    rows: Range<usize>,
    in_dim: usize,
) -> Result<WeightSlice<'_>, VindexError> {
    let per_row = |width: usize| rows.start * width..rows.end * width;
    Ok(match weight {
        WeightSlice::F32(v) => WeightSlice::F32(&v[per_row(in_dim)]),
        WeightSlice::Bf16(v) => WeightSlice::Bf16(&v[per_row(in_dim)]),
        WeightSlice::Q8 {
            codes,
            scales,
            sums,
            block,
        } => WeightSlice::Q8 {
            codes: &codes[per_row(in_dim)],
            scales: &scales[per_row(in_dim.div_ceil(block))],
            sums: if sums.is_empty() {
                sums
            } else {
                &sums[per_row(in_dim.div_ceil(SUM_BLOCK))]
            },
            block,
        },
        _ => {
            return Err(VindexError::Parse(
                "payload measurement refuses an untested V representation".into(),
            ))
        }
    })
}
