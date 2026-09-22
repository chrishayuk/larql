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

use super::backend::{AttentionStepCall, FfnCall, MatrixClass, NormCall, PlanBackend};
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
    ffn_gate: Option<LoadedWeight>,
    ffn_up: LoadedWeight,
    ffn_down: LoadedWeight,
    post_ffn: Option<LoadedNorm>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

/// What one decode step produces.
pub struct StepOutput {
    /// Logits for the position just consumed, when the plan carries an
    /// output head.
    pub logits: Option<Vec<f32>>,
}

/// Read-only taps on one incremental position.
///
/// Observers are diagnostics, not execution policy: they see the resolved
/// values and calls but cannot replace them. [`DecodeSession::step`] uses the
/// same traversal with a no-op observer, so enabling a capture cannot change
/// which operation runs or what value reaches the next layer.
pub trait DecodeObserver {
    /// Residual entering `layer`, before its pre-attention norm.
    fn layer_input(&mut self, _layer: usize, _residual: &[f32]) -> Result<(), VindexError> {
        Ok(())
    }

    /// Normalised vector consumed by the attention projections.
    fn attention_input(&mut self, _layer: usize, _input: &[f32]) -> Result<(), VindexError> {
        Ok(())
    }

    /// Attention branch output after any judged branch norm, before the
    /// residual addition.
    fn attention_output(&mut self, _layer: usize, _output: &[f32]) -> Result<(), VindexError> {
        Ok(())
    }

    /// Residual after the attention branch has been added.
    fn post_attention(&mut self, _layer: usize, _residual: &[f32]) -> Result<(), VindexError> {
        Ok(())
    }

    /// Fully resolved FFN call before execution. This is the diagnostic seam
    /// for exact channel-block contribution accounting: the observer sees the
    /// same normalised input and resident operands the backend will consume.
    fn ffn_call(&mut self, _layer: usize, _call: &FfnCall<'_>) -> Result<(), VindexError> {
        Ok(())
    }

    /// FFN branch output after any judged branch norm, before the residual
    /// addition.
    fn ffn_output(&mut self, _layer: usize, _output: &[f32]) -> Result<(), VindexError> {
        Ok(())
    }

    /// Residual leaving `layer`, after its FFN branch has been added.
    fn post_layer(&mut self, _layer: usize, _residual: &[f32]) -> Result<(), VindexError> {
        Ok(())
    }
}

struct NoopObserver;

impl DecodeObserver for NoopObserver {}

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
                ffn_gate: layer
                    .ffn
                    .gate
                    .as_ref()
                    .map(|gate| {
                        load_weight(
                            store,
                            gate,
                            backend.weight_format(MatrixClass::FfnProjection),
                        )
                    })
                    .transpose()?,
                ffn_up: load_weight(
                    store,
                    &layer.ffn.up,
                    backend.weight_format(MatrixClass::FfnProjection),
                )?,
                ffn_down: load_weight(
                    store,
                    &layer.ffn.down,
                    backend.weight_format(MatrixClass::FfnProjection),
                )?,
                post_ffn: layer
                    .post_ffn_norm
                    .as_ref()
                    .map(|op| LoadedNorm::load(op, store))
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
        session.prepare_residency();
        Ok(session)
    }

    /// Re-issue the backend's numerical no-op residency hint for every loaded
    /// matrix operand. Long diagnostic corpora start independent sequences
    /// after clearing KV state; on unified-memory devices that boundary is
    /// long enough for a driver to begin unwiring a 60 GB working set and
    /// enter a self-reinforcing slow-step cycle. Callers decide when the hint
    /// is worthwhile; ordinary decode does not add it per token.
    pub fn prepare_residency(&self) {
        let mut weights: Vec<super::backend::WeightSlice<'_>> = Vec::new();
        for state in &self.layers {
            weights.extend(state.attention.weight_slices());
            if let Some(gate) = &state.ffn_gate {
                weights.push(gate.slice());
            }
            weights.push(state.ffn_up.slice());
            weights.push(state.ffn_down.slice());
        }
        if let Some((_, projection)) = &self.output {
            weights.push(projection.slice());
        }
        self.backend.prepare(&weights);
    }

    /// Positions consumed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Start an independent sequence while retaining every resident operand.
    ///
    /// Only sequence state is cleared. This is what makes a large oracle
    /// corpus practical: prompts do not share attention history, but they also
    /// do not reload the model between rows.
    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
    }

    /// Advance one token: embed it, run it through every layer against
    /// the cached K/V, and return the head's logits for this position.
    ///
    /// Operation ordering mirrors the batch traversal exactly — the
    /// decode-vs-batch parity tests are the guarantee.
    pub fn step(&mut self, token: u32) -> Result<StepOutput, VindexError> {
        self.step_observed(token, &mut NoopObserver)
    }

    /// Advance one token through the identical decode traversal while
    /// exposing read-only diagnostic taps.
    pub fn step_observed<O: DecodeObserver + ?Sized>(
        &mut self,
        token: u32,
        observer: &mut O,
    ) -> Result<StepOutput, VindexError> {
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

        for (layer_index, (state, layer)) in
            self.layers.iter_mut().zip(&self.plan.layers).enumerate()
        {
            observer.layer_input(layer_index, &h)?;
            // Attention input is normalised once and handed over; the
            // judged gate reads the same vector (same as the batch path).
            let inputs = [state.pre_attention.apply(self.backend, &h)];
            observer.attention_input(layer_index, &inputs[0])?;
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
            observer.attention_output(layer_index, &attn_out)?;
            self.backend.residual_add(&mut h, &attn_out);
            observer.post_attention(layer_index, &h)?;

            let normed = state.pre_ffn.apply(self.backend, &h);
            let call = FfnCall {
                x: &normed,
                hidden: self.hidden,
                intermediate: layer.ffn.intermediate_size,
                gate: state.ffn_gate.as_ref().map(LoadedWeight::slice),
                up: state.ffn_up.slice(),
                down: state.ffn_down.slice(),
                activation: layer.ffn.activation,
            };
            observer.ffn_call(layer_index, &call)?;
            let ffn_out = self.backend.ffn(call)?;
            let ffn_out = match &state.post_ffn {
                Some(norm) => norm.apply(self.backend, &ffn_out),
                None => ffn_out,
            };
            observer.ffn_output(layer_index, &ffn_out)?;
            self.backend.residual_add(&mut h, &ffn_out);
            observer.post_layer(layer_index, &h)?;
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
