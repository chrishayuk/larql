//! FFN operand selection and dense FFN operands.

use super::super::accounting::Bound;
use super::super::backend::{FfnCall, MatrixClass, WeightSlice};
use super::super::operands::OperandSource;
use super::super::weights::{load_weight, LoadedWeight};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::{FfnOp, LayerFfn};

#[allow(unused_imports)]
use super::*;

impl FfnOperands {
    pub(in super::super) fn load_routed_coordinator(
        ffn: &LayerFfn,
        store: OperandSource<'_>,
        layer: usize,
    ) -> Result<Self, VindexError> {
        let LayerFfn::Routed(op) = ffn else {
            return Err(VindexError::Parse(
                "routed coordinator requires routed layer".into(),
            ));
        };
        Ok(Self::Routed(Box::new(RoutedOperands {
            placement: Some((layer, None)),
            router: store.load(&op.router)?,
            router_bias: op.router_bias.as_ref().map(|r| store.load(r)).transpose()?,
            router_scale: op
                .router_scale
                .as_ref()
                .map(|r| store.load(r))
                .transpose()?,
            router_per_expert_scale: op
                .router_per_expert_scale
                .as_ref()
                .map(|r| store.load(r))
                .transpose()?,
            router_norm_eps: op.router_norm_eps,
            experts: ExpertMatrices::Fused {
                gate_up: Vec::new(),
                down: Vec::new(),
            },
            gate_up_bias: None,
            down_bias: None,
            shared: None,
            shared_gate: None,
            latent: None,
        })))
    }
    pub(in super::super) fn routed_provider_missing(&self) -> bool {
        matches!(self, Self::Routed(r) if matches!(r.placement, Some((_, None))))
    }
    /// Whether this layer's FFN sends a block of positions to remote
    /// experts as one call. A property of placement, not a tuning choice:
    /// the block traversal must take its grouped arm for such a layer
    /// whatever the multi-position switch says.
    pub(in super::super) fn groups_remote_rows(&self) -> bool {
        matches!(self, Self::Routed(r) if r.groups_rows())
    }
    pub(in super::super) fn bind_routed_provider(
        &mut self,
        provider: std::sync::Arc<dyn super::super::routed_experts::RoutedExpertProvider>,
    ) -> Result<(), VindexError> {
        let Self::Routed(r) = self else {
            return Err(VindexError::Parse("not a routed coordinator layer".into()));
        };
        let Some((_, slot)) = &mut r.placement else {
            return Err(VindexError::Parse(
                "routed layer is resident, not placed".into(),
            ));
        };
        *slot = Some(provider);
        Ok(())
    }
    pub(in super::super) fn dense_slices<'a>(
        &'a self,
        ffn: &'a LayerFfn,
    ) -> Option<(Option<WeightSlice<'a>>, WeightSlice<'a>, WeightSlice<'a>)> {
        match (self, ffn) {
            (Self::Dense(dense), LayerFfn::Dense(_)) => Some((
                dense.gate.as_ref().map(LoadedWeight::slice),
                dense.up.slice(),
                dense.down.slice(),
            )),
            _ => None,
        }
    }

    pub(in super::super) fn load(
        ffn: &LayerFfn,
        store: OperandSource<'_>,
        format: super::super::prepared::FormatFor<'_>,
        bank: super::super::prepared::BankPin,
        shared: super::super::prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        match ffn {
            LayerFfn::Dense(op) => Ok(Self::Dense(Box::new(DenseOperands::load(
                op, store, format,
            )?))),
            LayerFfn::Routed(op) => Ok(Self::Routed(Box::new(RoutedOperands::load(
                op, store, bank, shared, format,
            )?))),
            LayerFfn::Hybrid(op) => Ok(Self::Hybrid(Box::new(HybridOperands {
                dense: DenseOperands::load(&op.dense, store, format)?,
                routed: RoutedOperands::load(&op.routed, store, bank, shared, format)?,
                pre_experts_norm: LoadedNormWeight::load(&op.pre_experts_norm, store)?,
                post_dense_norm: LoadedNormWeight::load(&op.post_dense_norm, store)?,
                post_experts_norm: LoadedNormWeight::load(&op.post_experts_norm, store)?,
            }))),
        }
    }

    /// Every bound operand of this FFN, each under the OPERATION the
    /// loader bound it for — the dense projections one each, a packed
    /// bank as one operand over its per-expert objects, a per-expert bank
    /// one region per matrix, the shared branch as three projections.
    /// The loader names the operation because only it knows which object
    /// it bound for what; the accounting must not guess from counts.
    pub(in super::super) fn bound<'a>(
        &'a self,
        ffn: &'a LayerFfn,
    ) -> Result<Vec<(Operation, Bound<'a>)>, VindexError> {
        let dense = |bounds: Vec<Bound<'a>>| {
            bounds
                .into_iter()
                .map(|b| (Operation::Project(MatrixClass::FfnProjection), b))
        };
        match (self, ffn) {
            (Self::External { .. }, LayerFfn::Dense(_)) => Ok(Vec::new()),
            (Self::Dense(d), LayerFfn::Dense(op)) => Ok(dense(d.bound(op)).collect()),
            (Self::Routed(r), LayerFfn::Routed(op)) => Ok(r.bound(op)),
            (Self::Hybrid(h), LayerFfn::Hybrid(op)) => {
                let mut out: Vec<_> = dense(h.dense.bound(&op.dense)).collect();
                out.extend(h.routed.bound(&op.routed));
                Ok(out)
            }
            _ => Err(VindexError::Parse(
                "the prepared FFN and the plan's FFN are different programs".to_string(),
            )),
        }
    }

    /// The dense projections only — what a pinned projection realization
    /// is checked against. A bank's per-expert slices are a different
    /// realization and are not projections of the plan's operands.
    pub(in super::super) fn dense_matrices(&self) -> Vec<&LoadedWeight> {
        match self {
            Self::External { .. } => Vec::new(),
            Self::Dense(d) => d.loaded_matrices(),
            Self::Routed(_) => Vec::new(),
            Self::Hybrid(h) => h.dense.loaded_matrices(),
        }
    }

    /// Every matrix operand, for residency accounting.
    pub(in super::super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        match self {
            Self::External { .. } => Vec::new(),
            Self::Dense(dense) => dense.loaded_matrices(),
            Self::Routed(routed) => routed.loaded_matrices(),
            Self::Hybrid(hybrid) => {
                let mut all = hybrid.dense.loaded_matrices();
                all.extend(hybrid.routed.loaded_matrices());
                all
            }
        }
    }

    /// Every matrix operand, for residency preparation.
    pub(in super::super) fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        match self {
            Self::External { .. } => Vec::new(),
            Self::Dense(dense) => dense.weight_slices(),
            Self::Routed(routed) => routed.weight_slices(),
            Self::Hybrid(hybrid) => {
                let mut slices = hybrid.dense.weight_slices();
                slices.extend(hybrid.routed.weight_slices());
                slices
            }
        }
    }

    /// Run this layer's FFN over one normalised vector on `backend` — the
    /// dense-only and routed-only shapes, which read one input.
    pub(in super::super) fn apply<B: super::super::backend::PlanBackend + ?Sized>(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        x: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        match (self, ffn) {
            (Self::External { layer, provider }, LayerFfn::Dense(_)) => {
                super::super::dense_ffn::validate_row(x, hidden)?;
                let out = provider
                    .as_ref()
                    .ok_or_else(|| VindexError::Parse("dense FFN provider is not bound".into()))?
                    .apply(*layer, x)?;
                super::super::dense_ffn::validate_row(&out, hidden)?;
                Ok(out)
            }
            (Self::Dense(dense), LayerFfn::Dense(op)) => dense.apply(op, backend, x, hidden),
            (Self::Routed(routed), LayerFfn::Routed(op)) => routed.apply(op, backend, x, x, hidden),
            _ => Err(VindexError::Parse(
                "FFN operands were loaded for a different op kind than the plan carries"
                    .to_string(),
            )),
        }
    }

    /// Whether [`Self::apply_observed`] can serve this layer: a loaded
    /// dense FFN only. Hybrid, external and routed layers refuse.
    pub(in super::super) fn serves_down_input(&self, ffn: &LayerFfn) -> bool {
        matches!((self, ffn), (Self::Dense(_), LayerFfn::Dense(_)))
    }

    /// Capture the dense down input on the same backend call that consumes it.
    pub(in super::super) fn apply_observed<B: super::super::backend::PlanBackend + ?Sized>(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        x: &[f32],
        hidden: usize,
        tap: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<f32>, VindexError> {
        let (Self::Dense(dense), LayerFfn::Dense(op)) = (self, ffn) else {
            return Err(VindexError::Parse(
                "FFN down-input capture requires a dense FFN".into(),
            ));
        };
        backend.ffn_observed(dense.call(op, x, hidden), tap)
    }

    /// The whole FFN block from the post-attention residual up to — not
    /// including — the layer's post-FFN norm and residual add. Both
    /// drivers (batch and decode) call this, so the hybrid program lives
    /// in exactly one place:
    ///
    /// ```text
    /// dense/routed:  ffn(pre_ffn_normed)
    /// hybrid:        post_dense_norm(dense(pre_ffn_normed))
    ///              + post_experts_norm(routed(pre_experts_norm(residual), router ← residual))
    /// ```
    ///
    /// `pre_ffn_normed` is the layer's pre-FFN norm of `residual`, produced
    /// by the caller (it is also what the judged gate reads).
    pub(in super::super) fn apply_from_residual<B: super::super::backend::PlanBackend + ?Sized>(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        residual: &[f32],
        pre_ffn_normed: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        match (self, ffn) {
            (Self::Hybrid(hybrid), LayerFfn::Hybrid(op)) => {
                let dense_out = hybrid
                    .dense
                    .apply(&op.dense, backend, pre_ffn_normed, hidden)?;
                let dense_out = hybrid.post_dense_norm.apply(backend, &dense_out);
                let expert_input = hybrid.pre_experts_norm.apply(backend, residual);
                let experts_out =
                    hybrid
                        .routed
                        .apply(&op.routed, backend, &expert_input, residual, hidden)?;
                let experts_out = hybrid.post_experts_norm.apply(backend, &experts_out);
                Ok(dense_out
                    .iter()
                    .zip(&experts_out)
                    .map(|(d, e)| d + e)
                    .collect())
            }
            (Self::Hybrid(_), _) | (_, LayerFfn::Hybrid(_)) => Err(VindexError::Parse(
                "FFN operands were loaded for a different op kind than the plan carries"
                    .to_string(),
            )),
            _ => self.apply(ffn, backend, pre_ffn_normed, hidden),
        }
    }

    /// [`Self::apply_from_residual`] over several positions at once.
    ///
    /// The DENSE arm groups its projections. A ROUTED arm whose experts
    /// are placed behind a bound provider groups its provider call: every
    /// position's selected experts cross to the workers together, once
    /// per layer, and each position's arithmetic is unchanged. A routed
    /// arm with resident experts, and a hybrid, keep the per-position
    /// program, which is the same arithmetic they ran before.
    pub(in super::super) fn apply_from_residual_many<
        B: super::super::backend::PlanBackend + ?Sized,
    >(
        &self,
        ffn: &LayerFfn,
        backend: &B,
        residuals: &[&[f32]],
        pre_ffn_normed: &[&[f32]],
        hidden: usize,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        match (self, ffn) {
            (Self::Dense(dense), LayerFfn::Dense(op)) => {
                dense.apply_many(op, backend, pre_ffn_normed, hidden)
            }
            (Self::Routed(routed), LayerFfn::Routed(op)) if routed.groups_rows() => {
                routed.apply_many(op, backend, pre_ffn_normed, hidden)
            }
            _ => residuals
                .iter()
                .zip(pre_ffn_normed)
                .map(|(residual, normed)| {
                    self.apply_from_residual(ffn, backend, residual, normed, hidden)
                })
                .collect(),
        }
    }
}

impl DenseOperands {
    pub(super) fn load(
        op: &FfnOp,
        store: OperandSource<'_>,
        format: super::super::prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        Ok(Self {
            gate: match &op.gate {
                Some(gate) => Some(load_weight(store, gate, format(gate)?)?),
                None => None,
            },
            up: load_weight(store, &op.up, format(&op.up)?)?,
            down: load_weight(store, &op.down, format(&op.down)?)?,
            up_bias: op.up_bias.as_ref().map(|b| store.load(b)).transpose()?,
            down_bias: op.down_bias.as_ref().map(|b| store.load(b)).transpose()?,
        })
    }

    /// Each projection paired with the operand it binds.
    pub(in super::super) fn bound<'a>(&'a self, op: &'a FfnOp) -> Vec<Bound<'a>> {
        let mut out = Vec::new();
        if let (Some(gate), Some(weight)) = (&op.gate, &self.gate) {
            out.push(Bound::one(gate, weight));
        }
        out.push(Bound::one(&op.up, &self.up));
        out.push(Bound::one(&op.down, &self.down));
        out
    }

    pub(in super::super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut all = vec![&self.up, &self.down];
        if let Some(gate) = &self.gate {
            all.push(gate);
        }
        all
    }

    pub(super) fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        let mut slices = vec![self.up.slice(), self.down.slice()];
        if let Some(gate) = &self.gate {
            slices.push(gate.slice());
        }
        slices
    }

    pub(super) fn apply<B: super::super::backend::PlanBackend + ?Sized>(
        &self,
        op: &FfnOp,
        backend: &B,
        x: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        backend.ffn(self.call(op, x, hidden))
    }

    pub(super) fn call<'a>(&'a self, op: &FfnOp, x: &'a [f32], hidden: usize) -> FfnCall<'a> {
        FfnCall {
            x,
            hidden,
            intermediate: op.intermediate_size,
            gate: self.gate.as_ref().map(LoadedWeight::slice),
            up: self.up.slice(),
            down: self.down.slice(),
            activation: op.activation,
            gate_policy: op.gate_policy,
            up_bias: self.up_bias.as_deref(),
            down_bias: self.down_bias.as_deref(),
        }
    }

    pub(super) fn apply_many<B: super::super::backend::PlanBackend + ?Sized>(
        &self,
        op: &FfnOp,
        backend: &B,
        xs: &[&[f32]],
        hidden: usize,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        backend.ffn_many(super::super::backend::FfnManyCall {
            xs,
            hidden,
            intermediate: op.intermediate_size,
            gate: self.gate.as_ref().map(LoadedWeight::slice),
            up: self.up.slice(),
            down: self.down.slice(),
            activation: op.activation,
            gate_policy: op.gate_policy,
            up_bias: self.up_bias.as_deref(),
            down_bias: self.down_bias.as_deref(),
        })
    }
}
