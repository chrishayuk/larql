//! Accessors on prepared operands.

use super::super::super::{ComponentOpPlan, LayerFfn, OutputOp};
use super::super::accounting::ResidencyBudget;
use super::super::backend::{NormCall, PlanBackend, ProjectCall};
use super::super::experts::FfnOperands;
use super::super::lowering::{LoweringIdentity, LoweringRegistry};
use super::super::operands::OperandSource;
use super::super::weights::LoadedWeight;
use crate::error::VindexError;
use larql_models::config::{HyperConnection, ResidualTopology};

#[allow(unused_imports)]
use super::*;

impl PreparedOperands {
    /// Install an already artifact-bound provider before creating any sessions.
    pub fn bind_dense_ffn_provider(
        &mut self,
        provider: std::sync::Arc<dyn super::super::dense_ffn::DenseFfnProvider>,
    ) -> Result<(), VindexError> {
        if self.slice != ExecutionSlice::DenseFfnCoordinator {
            return Err(VindexError::Parse(
                "external FFN provider requires coordinator operands".into(),
            ));
        }
        for layer in &mut self.layers {
            if let Some(FfnOperands::External { provider: slot, .. }) = &mut layer.ffn {
                *slot = Some(provider.clone());
            }
        }
        Ok(())
    }

    pub(in super::super) fn ensure_stack_ready(&self) -> Result<(), VindexError> {
        if matches!(
            self.slice,
            ExecutionSlice::DenseFfns { .. } | ExecutionSlice::RoutedExperts { .. }
        ) {
            return Err(VindexError::Parse(
                "FFN/expert workers cannot execute a layer stack".into(),
            ));
        }
        if self.layers.iter().any(|l| {
            l.ffn
                .as_ref()
                .is_some_and(FfnOperands::routed_provider_missing)
        }) {
            return Err(VindexError::Parse(
                "routed expert provider is not bound".into(),
            ));
        }
        if self
            .layers
            .iter()
            .any(|l| matches!(l.ffn, Some(FfnOperands::External { provider: None, .. })))
        {
            return Err(VindexError::Parse("dense FFN provider is not bound".into()));
        }
        Ok(())
    }

    pub fn routed_experts(&self) -> Option<&super::super::routed_experts::PreparedRoutedExperts> {
        self.routed_experts.as_ref()
    }
    pub fn bind_routed_expert_provider(
        &mut self,
        provider: std::sync::Arc<dyn super::super::routed_experts::RoutedExpertProvider>,
    ) -> Result<(), VindexError> {
        if self.slice != ExecutionSlice::RoutedExpertCoordinator {
            return Err(VindexError::Parse(
                "routed provider requires coordinator operands".into(),
            ));
        }
        for layer in &mut self.layers {
            if let Some(ffn) = &mut layer.ffn {
                ffn.bind_routed_provider(provider.clone())?;
            }
        }
        Ok(())
    }

    pub fn dense_ffns(&self) -> Option<&super::super::dense_ffn::PreparedDenseFfns> {
        self.dense_ffns.as_ref()
    }

    pub fn dense_ffn_image(
        &self,
        plan: &ComponentOpPlan,
        layer: usize,
    ) -> Result<PreparedDenseFfnImage, VindexError> {
        let layer_plan = plan.layers.get(layer).ok_or_else(|| {
            VindexError::Parse(format!("layer {layer} is outside the component plan"))
        })?;
        let LayerFfn::Dense(op) = layer_plan
            .ffn
            .as_ref()
            .ok_or_else(|| VindexError::Parse(format!("layer {layer} carries no FFN")))?
        else {
            return Err(VindexError::Parse(format!(
                "layer {layer} is not a dense FFN"
            )));
        };
        let local = layer.checked_sub(self.first_layer).ok_or_else(|| {
            VindexError::Parse(format!("layer {layer} precedes this prepared slice"))
        })?;
        let prepared = self.layers.get(local).ok_or_else(|| {
            VindexError::Parse(format!("layer {layer} is outside this prepared slice"))
        })?;
        let ffn = prepared
            .ffn
            .as_ref()
            .ok_or_else(|| VindexError::Parse(format!("prepared layer {layer} carries no FFN")))?;
        let (gate, up, down) = ffn
            .dense_slices(layer_plan.ffn.as_ref().unwrap())
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "prepared layer {layer} is not the plan's dense FFN"
                ))
            })?;
        Ok(PreparedDenseFfnImage {
            gate: gate
                .map(|weight| weight.decode_f32(op.intermediate_size, self.hidden))
                .transpose()?,
            up: up.decode_f32(op.intermediate_size, self.hidden)?,
            down: down.decode_f32(self.hidden, op.intermediate_size)?,
        })
    }

    /// Lower `slice` of `plan`'s operands into `backend`'s execution
    /// form, and give the backend its chance to place them (device
    /// residency). Every operand this slice needs is loaded here, and
    /// none of it is loaded again.
    pub fn load<'s, B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &B,
        slice: ExecutionSlice,
    ) -> Result<Self, VindexError> {
        Self::load_within(plan, store, backend, slice, &ResidencyBudget::UNBOUNDED)
    }

    /// [`Self::load`] on the provider `provider` names in `lowerings` —
    /// the registry-carried path (LOWERING-PLUGIN-1, L2).
    ///
    /// A provider the registry does not hold is refused here, by identity
    /// and naming every provider it does hold, before selection and
    /// before any byte. Nothing below constructs a provider the caller
    /// did not register.
    pub fn load_via<'s>(
        plan: &ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        lowerings: &LoweringRegistry,
        provider: &LoweringIdentity,
        slice: ExecutionSlice,
    ) -> Result<Self, VindexError> {
        let backend = lowerings.provider(provider)?;
        Self::load(plan, store, backend, slice)
    }

    /// Hidden width, read from the plan's embedding op.
    /// The exit's arithmetic on one `[hidden]` carrier: the prepared
    /// final norm, then [`Self::head_over_normed`]. `None` when the image
    /// carries no output head (a layer-range slice).
    ///
    /// V3-LENS-1's one head path: the decode exit calls this, a logit
    /// lens calls this on an intermediate carrier, and there is no
    /// second spelling of "the head" for the two to disagree on.
    pub fn head_logits<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        carrier: &[f32],
    ) -> Result<Option<Vec<f32>>, VindexError> {
        self.head_logits_observed(backend, carrier, &mut |_| {})
    }

    /// [`Self::head_logits`], handing `exit` the vector the head reads —
    /// the final-normed carrier, or the carrier itself where the image
    /// declares no final norm — before the head runs (PREC-DIAG-1 E0a).
    /// Borrowed and observation-only; `head_logits` IS this with a no-op
    /// tap, so the two cannot drift. Not called when there is no head.
    pub fn head_logits_observed<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        carrier: &[f32],
        exit: &mut dyn FnMut(&[f32]),
    ) -> Result<Option<Vec<f32>>, VindexError> {
        if self.output().is_none() {
            return Ok(None);
        }
        let normed;
        let final_hidden: &[f32] = match self.final_norm() {
            Some(norm) => {
                normed = norm.apply(backend, carrier);
                &normed
            }
            None => carrier,
        };
        exit(final_hidden);
        self.head_over_normed(backend, final_hidden)
    }

    /// The prepared output head over an ALREADY final-normed vector, with
    /// the head's multiplier and softcap; `None` when the image carries
    /// no output head. The batch exit norms a plane row by row and then
    /// calls this per row; the decode exit and the lens reach it through
    /// [`Self::head_logits`].
    pub fn head_over_normed<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        final_hidden: &[f32],
    ) -> Result<Option<Vec<f32>>, VindexError> {
        match self.output() {
            Some((output, weight)) => Ok(Some(backend.output_head(
                weight.slice(),
                output.projection.shape[0],
                self.hidden(),
                final_hidden,
                output.multiplier,
                output.softcapping,
            )?)),
            None => Ok(None),
        }
    }

    /// V3-HEAD-OBS-1, property A4: one query head's share of the output
    /// projection, `W_o[:, h·d..(h+1)·d] · x`, computed by the SAME
    /// projection kernel and the same pinned realisation the executor's
    /// `o_proj` uses — the head's slice of the input is placed in a
    /// zero vector of the projection's full input width and the whole
    /// projection runs, so a consumer never carries a second `W_o`. The
    /// O bias is NOT added: it is a once-only term the consumer adds
    /// after summing heads (see [`Self::attention_output_bias`]).
    /// Refuses a layer outside the executed range, a layer whose
    /// attention has no softmax heads, or an `x` of the wrong width.
    pub fn head_projection<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        layer: usize,
        head: usize,
        head_dim: usize,
        num_q_heads: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        let prepared = self.prepared_layer(layer)?;
        let PreparedAttention::Softmax(ops) = &prepared.attention else {
            return Err(VindexError::Parse(format!(
                "layer {layer}'s attention has no softmax heads to project"
            )));
        };
        if x.len() != head_dim {
            return Err(VindexError::Parse(format!(
                "head projection expects a {head_dim}-wide head input, got {}",
                x.len()
            )));
        }
        if head >= num_q_heads {
            return Err(VindexError::Parse(format!(
                "head {head} is outside this layer's {num_q_heads} query heads"
            )));
        }
        let q_rows = num_q_heads * head_dim;
        let mut padded = vec![0.0f32; q_rows];
        padded[head * head_dim..(head + 1) * head_dim].copy_from_slice(x);
        backend.project(ProjectCall {
            weight: ops.w_o.slice(),
            out_dim: self.hidden,
            in_dim: q_rows,
            x: &padded,
        })
    }

    /// The attention output projection's bias for a layer, if the plan
    /// declares one — the once-only term of the head-sum law.
    pub fn attention_output_bias(&self, layer: usize) -> Result<Option<&[f32]>, VindexError> {
        let prepared = self.prepared_layer(layer)?;
        let PreparedAttention::Softmax(ops) = &prepared.attention else {
            return Ok(None);
        };
        Ok(ops.biases.as_ref().and_then(|b| b.o.as_deref()))
    }

    /// The post-attention norm a layer applies to its attention output
    /// before the residual write, if it has one.
    pub(in super::super) fn post_attention_norm(
        &self,
        layer: usize,
    ) -> Result<Option<&PreparedNorm>, VindexError> {
        Ok(self.prepared_layer(layer)?.post_attention.as_ref())
    }

    /// Whether a layer's attention is a softmax family this image taps.
    pub fn attention_has_heads(&self, layer: usize) -> Result<bool, VindexError> {
        Ok(matches!(
            self.prepared_layer(layer)?.attention,
            PreparedAttention::Softmax(_)
        ))
    }

    pub(super) fn prepared_layer(&self, layer: usize) -> Result<&PreparedLayer, VindexError> {
        let first = self.first_layer;
        layer
            .checked_sub(first)
            .and_then(|offset| self.layers.get(offset))
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "layer {layer} is outside this image's executed layers {first}..{}",
                    first + self.layers.len()
                ))
            })
    }

    /// The declared embedding operation, including its scale and optional norm.
    /// Shared by local token execution and distributed coordinators.
    pub fn embed_token<B: PlanBackend + ?Sized>(
        &self,
        plan: &ComponentOpPlan,
        backend: &B,
        token: u32,
    ) -> Result<Vec<f32>, VindexError> {
        let embedding = plan
            .embedding
            .as_ref()
            .ok_or_else(|| VindexError::Parse("component has no embedding op".into()))?;
        let table = self.embed_table().ok_or_else(|| {
            VindexError::Parse("this prepared image has no embedding table".into())
        })?;
        let hidden = self.hidden();
        if hidden == 0 || token as usize >= table.len() / hidden {
            return Err(VindexError::Parse(format!(
                "token id {token} is outside the embedding table"
            )));
        }
        let mut row = backend.embed(table, hidden, token, embedding.scale);
        if let Some(norm) = embedding.norm {
            row = backend.norm(NormCall {
                kind: norm.kind,
                x: &row,
                weight: &[],
                weight_offset: 0.0,
                eps: norm.eps,
            });
        }
        Ok(row)
    }

    /// Width of one residual row.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// How many layers this image can execute.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Whether this image carries an output head (only a whole-stack
    /// slice does).
    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    /// Vocabulary readout of an already reduced carrier. This performs only
    /// the prepared final norm and output head, with their pinned realization,
    /// multiplier and soft-capping. It does not execute layers or mutate KV.
    ///
    /// The caller owns boundary semantics: apply a recorded layer scale first
    /// when reading a post-add/pre-scale observation. Bundle/history reduction
    /// must not be guessed by an analysis caller.
    pub fn readout_carrier<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        carrier: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        let normalized = self.normalize_carrier_for_readout(backend, carrier)?;
        let (op, weight) = self
            .output
            .as_ref()
            .expect("normalization requires an output head");
        backend.output_head(
            weight.slice(),
            op.projection.shape[0],
            self.hidden,
            &normalized,
            op.multiplier,
            op.softcapping,
        )
    }

    /// Gather declared vocabulary rows from the output head in the exact
    /// representation held by this prepared image.
    ///
    /// The current evidence path admits the CPU production formats used by
    /// the GW programme (f32, stored bf16 and realised Q8).  Any other format
    /// refuses rather than widening or silently changing its arithmetic.
    pub fn select_output_head(&self, token_ids: &[u32]) -> Result<SelectedOutputHead, VindexError> {
        let (op, weight) = self
            .output
            .as_ref()
            .ok_or_else(|| VindexError::Parse("prepared slice has no output head".into()))?;
        SelectedOutputHead::gather(
            weight.slice(),
            op.projection.shape[0],
            self.hidden,
            token_ids,
            op.multiplier,
            op.softcapping,
        )
    }

    /// Read an already reduced carrier against a previously gathered output
    /// head.  Final normalisation, multiplier and softcap are identical to
    /// [`Self::readout_carrier`]; the projection is the same backend call over
    /// fewer rows, so it agrees up to that backend's summation order — equal
    /// on aarch64, within an ulp on x86 SIMD — not bit for bit in general.
    pub fn readout_carrier_selected<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        carrier: &[f32],
        head: &SelectedOutputHead,
    ) -> Result<Vec<f32>, VindexError> {
        if head.hidden != self.hidden {
            return Err(VindexError::Parse(
                "selected output head and prepared carrier widths differ".into(),
            ));
        }
        let normalized = self.normalize_carrier_for_readout(backend, carrier)?;
        backend.output_head(
            head.projection.slice(),
            head.token_ids.len(),
            self.hidden,
            &normalized,
            head.multiplier,
            head.softcapping,
        )
    }

    /// The exact prepared final-norm input to the output head, for a vocabulary
    /// geometry lens. Caller supplies an already reduced, layer-scaled carrier.
    /// No layers, output projection, or KV mutation are executed here.
    pub fn normalize_carrier_for_readout<B: PlanBackend + ?Sized>(
        &self,
        backend: &B,
        carrier: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        self.ensure_lowered_by(backend)?;
        if carrier.len() != self.hidden || carrier.iter().any(|v| !v.is_finite()) {
            return Err(VindexError::Parse(
                "readout requires a finite, hidden-width carrier".into(),
            ));
        }
        if self.output.is_none() {
            return Err(VindexError::Parse(
                "prepared slice has no output head for readout".into(),
            ));
        }
        Ok(match &self.final_norm {
            Some(norm) => norm.apply(backend, carrier),
            None => carrier.to_vec(),
        })
    }

    pub(in super::super) fn embed_table(&self) -> Option<&[f32]> {
        self.embed_table.as_deref()
    }

    pub(in super::super) fn first_layer(&self) -> usize {
        self.first_layer
    }

    /// The plan-layer indices this prepared set will execute, as a
    /// half-open range.
    ///
    /// Public because the identity of the work is part of a run's
    /// provenance: a caller banking logits has to be able to record —
    /// and a comparison to assert — which layers actually ran, and
    /// neither can be inferred from a layer count.
    pub fn executed_layers(&self) -> std::ops::Range<usize> {
        self.first_layer..self.first_layer + self.layers.len()
    }

    pub(in super::super) fn layers(&self) -> &[PreparedLayer] {
        &self.layers
    }

    pub(in super::super) fn final_norm(&self) -> Option<&PreparedNorm> {
        self.final_norm.as_ref()
    }

    pub(in super::super) fn output(&self) -> Option<&(OutputOp, LoadedWeight)> {
        self.output.as_ref()
    }

    /// The declared hyper-connection topology this image was prepared
    /// under, `None` for the single stream.
    pub(in super::super) fn hyper_connection(&self) -> Option<HyperConnection> {
        match self.topology {
            ResidualTopology::HyperConnection(hc) => Some(hc),
            ResidualTopology::SingleStream | ResidualTopology::AttentionResidual { .. } => None,
        }
    }

    /// Whether this image holds hyper-connection site operands — i.e.
    /// whether the residual it executes is a bundle.
    pub fn carries_hyper_connection(&self) -> bool {
        self.hyper_connection().is_some()
    }

    pub(in super::super) fn hyper_connection_head(&self) -> Option<&PreparedHcHead> {
        self.hyper_connection_head.as_ref()
    }

    /// The declared block period, `None` on every other topology. The
    /// traversal reads it to decide which layers carry the boundary
    /// event, and it is the ONE declared fact the schedule needs.
    pub(in super::super) fn attention_residual_block_size(&self) -> Option<usize> {
        match self.topology {
            ResidualTopology::AttentionResidual { block_size } => Some(block_size),
            ResidualTopology::SingleStream | ResidualTopology::HyperConnection(_) => None,
        }
    }

    pub(in super::super) fn attention_residual_exit(&self) -> Option<&PreparedAttnResExit> {
        self.attention_residual_exit.as_ref()
    }

    /// Whether this image holds attention-residual site operands — i.e.
    /// whether the residual it executes carries a snapshot history.
    pub fn carries_attention_residual(&self) -> bool {
        self.attention_residual_block_size().is_some()
    }
}
