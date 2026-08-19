//! Incremental decode over a plan: operand residency plus a KV cache.
//!
//! A [`DecodeSession`] loads every operand **once** — in the format the
//! backend declares — and then advances one token per [`step`], feeding
//! the backend's [`attention_step`] against the session's per-layer K/V
//! cache. Each step therefore computes exactly one position through the
//! whole stack instead of re-running the forward over the grown
//! sequence, and every weight keeps a stable address for the session's
//! lifetime, which is what lets a pointer-keyed device buffer cache
//! hold the model resident.
//!
//! **This is the second traversal in the executor, and it is pinned to
//! the first.** [`execute_plan_streaming`](super::execute_plan_streaming)
//! remains the batch traversal (parallel positions, streamed planes,
//! resume); this session realises the same program one position at a
//! time. The two share the operand loaders and call construction
//! ([`AttentionOperands`]), and the decode-vs-batch parity tests assert
//! their outputs agree per backend — a change that moves one without
//! the other is a bug by definition.
//!
//! [`step`]: DecodeSession::step
//! [`attention_step`]: super::backend::PlanBackend::attention_step

use super::backend::{AttentionStepCall, MatrixClass, NormCall, PlanBackend};
use super::operands::OperandStore;
use super::weights::{load_weight, LoadedWeight};
use super::AttentionOperands;
use crate::error::VindexError;

use super::super::{ComponentOpPlan, NormOp, OutputOp};

/// One norm site's operation with its weight held resident.
struct LoadedNorm {
    op: NormOp,
    weight: Vec<f32>,
}

impl LoadedNorm {
    fn load(op: &NormOp, store: &OperandStore) -> Result<Self, VindexError> {
        Ok(Self {
            op: op.clone(),
            weight: store.load(&op.weight)?,
        })
    }

    fn apply<B: PlanBackend + ?Sized>(&self, backend: &B, x: &[f32]) -> Vec<f32> {
        backend.norm(NormCall {
            kind: self.op.kind,
            x,
            weight: &self.weight,
            weight_offset: self.op.weight_offset,
            eps: self.op.eps,
        })
    }
}

/// One layer's resident operands and its K/V cache.
struct LayerState {
    pre_attention: LoadedNorm,
    attention: AttentionOperands,
    post_attention: Option<LoadedNorm>,
    pre_ffn: LoadedNorm,
    ffn: super::experts::FfnOperands,
    post_ffn: Option<LoadedNorm>,
    /// The layer's output scalar, when the plan carries one.
    layer_scale: Option<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

/// What one decode step produces.
pub struct StepOutput {
    /// Logits for the position just consumed, when the plan carries an
    /// output head.
    pub logits: Option<Vec<f32>>,
}

/// Incremental executor over one component plan (see module docs).
pub struct DecodeSession<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    backend: &'a B,
    hidden: usize,
    embed_table: Vec<f32>,
    layers: Vec<LayerState>,
    final_norm: Option<LoadedNorm>,
    output: Option<(OutputOp, LoadedWeight)>,
    position: usize,
}

impl<'a, B: PlanBackend> DecodeSession<'a, B> {
    /// Load every operand the plan consumes, once, in the backend's
    /// declared weight format. The embedding table stays f32 — it is a
    /// row lookup, not matrix traffic.
    pub fn new(
        plan: &'a ComponentOpPlan,
        store: &OperandStore,
        backend: &'a B,
    ) -> Result<Self, VindexError> {
        let embedding = plan.embedding.as_ref().ok_or_else(|| {
            VindexError::Parse(format!(
                "component `{}` has no embedding op — external hidden-state input is a later rung",
                plan.component
            ))
        })?;
        let hidden = embedding.table.shape[1];

        let embed_table = store.load(&embedding.table)?;
        let mut layers = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            layers.push(LayerState {
                pre_attention: LoadedNorm::load(&layer.pre_attention_norm, store)?,
                attention: AttentionOperands::load(
                    &layer.attention,
                    store,
                    backend.weight_format(MatrixClass::AttentionProjection),
                )?,
                post_attention: layer
                    .post_attention_norm
                    .as_ref()
                    .map(|op| LoadedNorm::load(op, store))
                    .transpose()?,
                pre_ffn: LoadedNorm::load(&layer.pre_ffn_norm, store)?,
                ffn: super::experts::FfnOperands::load(
                    &layer.ffn,
                    store,
                    backend.weight_format(MatrixClass::FfnProjection),
                )?,
                post_ffn: layer
                    .post_ffn_norm
                    .as_ref()
                    .map(|op| LoadedNorm::load(op, store))
                    .transpose()?,
                layer_scale: layer
                    .layer_scale
                    .as_ref()
                    .map(|op| store.load(op).and_then(|v| super::layer_scalar_of(&v)))
                    .transpose()?,
                keys: Vec::new(),
                values: Vec::new(),
            });
        }
        let final_norm = plan
            .final_norm
            .as_ref()
            .map(|op| LoadedNorm::load(op, store))
            .transpose()?;
        let output = plan
            .output
            .as_ref()
            .map(|op| {
                Ok::<_, VindexError>((
                    op.clone(),
                    load_weight(
                        store,
                        &op.projection,
                        backend.weight_format(MatrixClass::OutputHead),
                    )?,
                ))
            })
            .transpose()?;

        let session = Self {
            plan,
            backend,
            hidden,
            embed_table,
            layers,
            final_norm,
            output,
            position: 0,
        };
        let mut weights: Vec<super::backend::WeightSlice<'_>> = Vec::new();
        for state in &session.layers {
            weights.extend(state.attention.weight_slices());
            weights.extend(state.ffn.weight_slices());
        }
        if let Some((_, projection)) = &session.output {
            weights.push(projection.slice());
        }
        backend.prepare(&weights);
        Ok(session)
    }

    /// Positions consumed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Advance one token: embed it, run it through every layer against
    /// the cached K/V, and return the head's logits for this position.
    ///
    /// Operation ordering mirrors the batch traversal exactly — the
    /// decode-vs-batch parity tests are the guarantee.
    pub fn step(&mut self, token: u32) -> Result<StepOutput, VindexError> {
        let embedding = self
            .plan
            .embedding
            .as_ref()
            .expect("session construction required an embedding op");
        if (token as usize + 1) * self.hidden > self.embed_table.len() {
            return Err(VindexError::Parse(format!(
                "token id {token} is outside the embedding table",
            )));
        }
        let mut h = self
            .backend
            .embed(&self.embed_table, self.hidden, token, embedding.scale);
        if let Some(norm) = embedding.norm {
            h = self.backend.norm(NormCall {
                kind: norm.kind,
                x: &h,
                weight: &[],
                weight_offset: 0.0,
                eps: norm.eps,
            });
        }

        for (state, layer) in self.layers.iter_mut().zip(&self.plan.layers) {
            // Attention input is normalised once and handed over; the
            // judged gate reads the same vector (same as the batch path).
            let inputs = [state.pre_attention.apply(self.backend, &h)];
            let call = state.attention.call(
                &layer.attention,
                &inputs,
                layer.pre_attention_norm.eps,
                self.hidden,
            );
            let out = self.backend.attention_step(AttentionStepCall {
                op: call,
                position: self.position,
                keys: &state.keys,
                values: &state.values,
            })?;
            state.keys.push(out.key);
            state.values.push(out.value);
            let attn_out = match &state.post_attention {
                Some(norm) => norm.apply(self.backend, &out.output),
                None => out.output,
            };
            self.backend.residual_add(&mut h, &attn_out);

            let normed = state.pre_ffn.apply(self.backend, &h);
            let ffn_out = state.ffn.apply_from_residual(
                &layer.ffn,
                self.backend,
                &h,
                &normed,
                self.hidden,
            )?;
            let ffn_out = match &state.post_ffn {
                Some(norm) => norm.apply(self.backend, &ffn_out),
                None => ffn_out,
            };
            self.backend.residual_add(&mut h, &ffn_out);
            if let Some(scale) = state.layer_scale {
                self.backend.scale_row(&mut h, scale);
            }
        }

        let final_hidden = match &self.final_norm {
            Some(norm) => norm.apply(self.backend, &h),
            None => h,
        };
        let logits = match &self.output {
            Some((op, weight)) => Some(self.backend.output_head(
                weight.slice(),
                op.projection.shape[0],
                self.hidden,
                &final_hidden,
                op.multiplier,
                op.softcapping,
            )?),
            None => None,
        };
        self.position += 1;
        Ok(StepOutput { logits })
    }
}
