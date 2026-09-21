//! V3-HEAD-OBS-1: the head reader — a consumer of the per-head tap that
//! computes, beside the executor's fused output and never in place of
//! it, each query head's contribution to the recorded attention write
//! and the stats-level summary the run record carries (property A9; the
//! evidence freeze's head-sum law HL1 and capture law HL5).
//!
//! The head-sum law, as this reader states it: with `ctx_h` the head's
//! mixed value, `g_h` its activated gate where the plan declares one,
//! `c_h = W_o[:, h·d..(h+1)·d] · (g_h ⊙ ctx_h)` through the prepared
//! image's own projection, `o = Σ_h c_h + bias`, and — where the layer
//! applies a post-attention RMS norm before the residual write —
//! `s = 1 / sqrt(mean(o²) + eps)` and `gain = offset + w_post`, the
//! children of the recorded write are `c′_h = s · gain ⊙ c_h ·
//! residual_scale` and the once-only term `bias′ = s · gain ⊙ bias ·
//! residual_scale`, so that `Σ_h c′_h + bias′ ≈ delta`, the delta
//! V3-OBS-1 records. The residual `‖Σ_h c′_h + bias′ − delta‖ / ‖delta‖`
//! is measured and recorded per write, never assumed; bit identity across
//! accumulation orders is not claimed.

use larql_models::config::NormType;

use super::super::ComponentOpPlan;
use super::backend::PlanBackend;
use super::observe::{
    AttentionHeadRecord, CarrierWriteRecord, StepEvent, StepObserver, SublayerSite,
};
use super::observe_stats::FixedBasis;
use super::prepared::PreparedOperands;
use crate::error::VindexError;

/// The method every head-sum residual in a [`HeadWrite`] was computed by.
pub const HEAD_SUM_METHOD: &str = "head-sum-through-post-norm/v1";

/// One query head's stats at one attention write.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadRow {
    pub head: usize,
    pub kv_head: usize,
    /// `‖c′_h‖`, Euclidean, accumulated in f64.
    pub norm: f64,
    /// `c′_h` projected on the run's basis, when one was given.
    pub projection: Vec<f32>,
    /// The top-`k` source positions by attention weight, `(position,
    /// weight)`, descending; ties keep the earlier position first.
    pub sources: Vec<(usize, f32)>,
    /// The sink's mass at this head; zero on a plan without sinks.
    pub sink: f32,
}

/// One attention write's decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadWrite {
    pub layer: usize,
    pub position: usize,
    /// The head-sum residual `‖Σ_h c′_h + bias′ − delta‖ / ‖delta‖`.
    pub residual: f64,
    pub rows: Vec<HeadRow>,
    /// `c′_h` per head, kept only when the reader was asked to retain them.
    pub children: Option<Vec<Vec<f32>>>,
    /// V3-INTERVENE-2, J5: `Σ_h c′_h + bias′` — reconstructed purely from
    /// this write's per-head records (never from `delta`), so on an
    /// UNINTERVENED run it equals `delta` to the head-sum law's own
    /// residual, and on an intervened run it is the same run's own
    /// "what the unintervened write would have been" (`delta_base`).
    pub sum: Vec<f32>,
}

/// What a recorder needs from a head reader: forward it every head
/// record, ask it for the write's decomposition at the write, and read
/// its counts and failure for the receipt.
pub trait HeadReader {
    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>);
    /// Called at every carrier write; returns the decomposition when the
    /// write is an attention write whose heads this reader saw.
    fn finish_write(&mut self, record: &CarrierWriteRecord<'_>) -> Option<HeadWrite>;
    fn records(&self) -> usize;
    fn failure(&self) -> Option<&VindexError>;
}

struct Pending {
    head: usize,
    kv_head: usize,
    /// `g_h ⊙ ctx_h`, or `ctx_h` where the plan declares no gate.
    input: Vec<f32>,
    weights: Vec<f32>,
    source_start: usize,
    sink: f32,
}

/// The stats-level head reader over one prepared image.
pub struct HeadStats<'a, B: PlanBackend + ?Sized> {
    ops: &'a PreparedOperands,
    plan: &'a ComponentOpPlan,
    backend: &'a B,
    basis: Option<FixedBasis>,
    top_k: usize,
    retain_children: bool,
    pending_layer: Option<usize>,
    pending: Vec<Pending>,
    pub writes: Vec<HeadWrite>,
    pub records: usize,
    pub failure: Option<VindexError>,
}

impl<'a, B: PlanBackend + ?Sized> HeadStats<'a, B> {
    pub fn new(
        ops: &'a PreparedOperands,
        plan: &'a ComponentOpPlan,
        backend: &'a B,
        basis: Option<FixedBasis>,
        top_k: usize,
    ) -> Self {
        Self {
            ops,
            plan,
            backend,
            basis,
            top_k,
            retain_children: false,
            pending_layer: None,
            pending: Vec::new(),
            writes: Vec::new(),
            records: 0,
            failure: None,
        }
    }

    /// Keep every `c′_h` on the decomposition, for a consumer that checks
    /// the law itself rather than trusting the residual.
    pub fn retaining_children(mut self) -> Self {
        self.retain_children = true;
        self
    }

    pub fn basis(&self) -> Option<&FixedBasis> {
        self.basis.as_ref()
    }

    /// The decomposition of one attention write from the heads pending
    /// for its layer. Refuses a norm kind it cannot linearise.
    fn decompose(
        &self,
        layer: usize,
        position: usize,
        delta: &[f32],
    ) -> Result<HeadWrite, VindexError> {
        let op = self.plan.layers[layer].attention.softmax().ok_or_else(|| {
            VindexError::Parse(format!(
                "layer {layer} has no softmax attention; its heads cannot have been observed"
            ))
        })?;
        let head_dim = op.head_dim;
        let num_q_heads = op.num_q_heads;
        let hidden = self.ops.hidden();
        if self.pending.len() != num_q_heads {
            return Err(VindexError::Parse(format!(
                "layer {layer} position {position}: {} head records for {num_q_heads} heads",
                self.pending.len()
            )));
        }
        // c_h through the image's own projection; o = Σ c_h + bias.
        let mut children: Vec<Vec<f32>> = Vec::with_capacity(num_q_heads);
        for pending in &self.pending {
            children.push(self.ops.head_projection(
                self.backend,
                layer,
                pending.head,
                head_dim,
                num_q_heads,
                &pending.input,
            )?);
        }
        let bias: Option<Vec<f32>> = self.ops.attention_output_bias(layer)?.map(<[f32]>::to_vec);
        let mut o = vec![0.0f64; hidden];
        for child in &children {
            for (acc, c) in o.iter_mut().zip(child) {
                *acc += f64::from(*c);
            }
        }
        if let Some(bias) = &bias {
            for (acc, b) in o.iter_mut().zip(bias) {
                *acc += f64::from(*b);
            }
        }
        // The post-attention norm's scalar and gain for THIS write, or
        // the identity where the family has none.
        let (scalar, gain): (f64, Option<Vec<f32>>) = match self.ops.post_attention_norm(layer)? {
            None => (1.0, None),
            Some(norm) => {
                if norm.kind() != NormType::RmsNorm {
                    return Err(VindexError::Parse(format!(
                        "layer {layer}'s post-attention norm is {:?}; this reader linearises \
                         RMS norms only",
                        norm.kind()
                    )));
                }
                let mean_sq = o.iter().map(|v| v * v).sum::<f64>() / hidden as f64;
                let scalar = 1.0 / (mean_sq + norm.eps()).sqrt();
                let offset = norm.weight_offset();
                let gain: Vec<f32> = norm.weight().iter().map(|w| offset + w).collect();
                (scalar, Some(gain))
            }
        };
        let residual_scale = f64::from(self.plan.layers[layer].residual_scale.unwrap_or(1.0));
        let child_of = |c: &[f32]| -> Vec<f32> {
            c.iter()
                .enumerate()
                .map(|(i, v)| {
                    let g = gain.as_ref().map_or(1.0, |g| f64::from(g[i]));
                    (f64::from(*v) * scalar * g * residual_scale) as f32
                })
                .collect()
        };
        let children: Vec<Vec<f32>> = children.iter().map(|c| child_of(c)).collect();
        let bias_child: Option<Vec<f32>> = bias.as_deref().map(child_of);
        // The law's residual against what V3-OBS-1 recorded.
        let mut sum = vec![0.0f64; hidden];
        for child in &children {
            for (acc, c) in sum.iter_mut().zip(child) {
                *acc += f64::from(*c);
            }
        }
        if let Some(b) = &bias_child {
            for (acc, c) in sum.iter_mut().zip(b) {
                *acc += f64::from(*c);
            }
        }
        let err: f64 = sum
            .iter()
            .zip(delta)
            .map(|(s, d)| (s - f64::from(*d)).powi(2))
            .sum::<f64>()
            .sqrt();
        let delta_norm: f64 = delta
            .iter()
            .map(|d| f64::from(*d).powi(2))
            .sum::<f64>()
            .sqrt();
        let residual = if delta_norm > 0.0 {
            err / delta_norm
        } else {
            err
        };
        let rows = self
            .pending
            .iter()
            .zip(&children)
            .map(|(pending, child)| {
                let norm = child
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let projection = self
                    .basis
                    .as_ref()
                    .map(|basis| {
                        basis
                            .rows()
                            .iter()
                            .map(|row| {
                                row.iter()
                                    .zip(child)
                                    .map(|(r, c)| f64::from(*r) * f64::from(*c))
                                    .sum::<f64>() as f32
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let mut sources: Vec<(usize, f32)> = pending
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| (pending.source_start + i, *w))
                    .collect();
                sources.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                sources.truncate(self.top_k);
                HeadRow {
                    head: pending.head,
                    kv_head: pending.kv_head,
                    norm,
                    projection,
                    sources,
                    sink: pending.sink,
                }
            })
            .collect();
        Ok(HeadWrite {
            layer,
            position,
            residual,
            rows,
            children: self.retain_children.then_some(children),
            sum: sum.iter().map(|v| *v as f32).collect(),
        })
    }
}

impl<B: PlanBackend + ?Sized> HeadReader for HeadStats<'_, B> {
    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        self.records += 1;
        if self.pending_layer != Some(layer) {
            self.pending.clear();
            self.pending_layer = Some(layer);
        }
        let input: Vec<f32> = match record.gate {
            Some(gate) => record.values.iter().zip(gate).map(|(v, g)| v * g).collect(),
            None => record.values.to_vec(),
        };
        self.pending.push(Pending {
            head: record.head,
            kv_head: record.kv_head,
            input,
            weights: record.weights.to_vec(),
            source_start: record.source_start,
            sink: record.sink,
        });
    }

    fn finish_write(&mut self, record: &CarrierWriteRecord<'_>) -> Option<HeadWrite> {
        if record.site != SublayerSite::Attention
            || self.pending_layer != Some(record.layer)
            || self.pending.is_empty()
            || self.failure.is_some()
        {
            return None;
        }
        let write = match self.decompose(record.layer, record.position, record.delta) {
            Ok(write) => Some(write),
            Err(e) => {
                self.failure = Some(e);
                None
            }
        };
        self.pending.clear();
        self.pending_layer = None;
        write
    }

    fn records(&self) -> usize {
        self.records
    }

    fn failure(&self) -> Option<&VindexError> {
        self.failure.as_ref()
    }
}

impl<B: PlanBackend + ?Sized> StepObserver for HeadStats<'_, B> {
    fn event(&mut self, _event: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        HeadReader::attention_head(self, layer, record);
    }

    fn carrier_write(&mut self, record: CarrierWriteRecord<'_>) {
        if let Some(write) = self.finish_write(&record) {
            self.writes.push(write);
        }
    }
}
