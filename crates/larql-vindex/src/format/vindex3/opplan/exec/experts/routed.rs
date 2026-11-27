//! Routed-expert operands.

use super::super::accounting::Bound;
use super::super::backend::{ExpertSlices, MatrixClass, RoutedFfnCall, WeightFormat, WeightSlice};
use super::super::operands::OperandSource;
use super::super::weights::{load_weight, LoadedWeight, MappedForm};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use crate::format::vindex3::opplan::{ExpertBank, FfnOp, OperandRef, RoutedFfnOp};

#[allow(unused_imports)]
use super::*;

/// Refusals of [`RoutedOperands::apply_many`].
const GROUPED_NEEDS_PROVIDER: &str =
    "a grouped routed call needs placed experts and a bound provider";
const GROUPED_REFUSES_WRAPPERS: &str =
    "a grouped routed call does not support a latent or shared-expert branch";
const GROUPED_NEEDS_LAYOUT: &str =
    "routed FFN op carries a packed bank and no gate_up layout; closure requires one";

impl RoutedOperands {
    /// The two banks, each one operand over its per-expert objects. A
    /// per-expert bank is refused before it is loaded, so it binds nothing.
    pub(in super::super) fn bound<'a>(
        &'a self,
        op: &'a RoutedFfnOp,
    ) -> Vec<(Operation, Bound<'a>)> {
        if self.placement.is_some() {
            return Vec::new();
        }
        let mut out: Vec<(Operation, Bound<'a>)> = match (&op.bank, &self.experts) {
            (
                ExpertBank::Packed { gate_up, down },
                ExpertMatrices::Fused {
                    gate_up: g,
                    down: d,
                },
            ) => vec![
                (
                    Operation::ExpertBankSlice,
                    Bound {
                        operand: &gate_up.weights,
                        weights: g.iter().collect(),
                    },
                ),
                (
                    Operation::ExpertBankSlice,
                    Bound {
                        operand: &down.weights,
                        weights: d.iter().collect(),
                    },
                ),
            ],
            // One planned operand per expert matrix, one mapped region
            // each: multiplicity for touch, the mapping shared underneath.
            (
                ExpertBank::PerExpert { gate, up, down },
                ExpertMatrices::Separate {
                    gate: g,
                    up: u,
                    down: d,
                    ..
                },
            ) => {
                let bank = Operation::ExpertProject {
                    experts: op.experts,
                    top_k: op.top_k,
                };
                gate.iter()
                    .zip(g)
                    .chain(up.iter().zip(u))
                    .chain(down.iter().zip(d))
                    .map(|(operand, weight)| (bank, Bound::one(operand, weight)))
                    .collect()
            }
            // The loader binds the shape the plan declares; the other
            // pairing cannot be constructed.
            (ExpertBank::Packed { .. }, ExpertMatrices::Separate { .. })
            | (ExpertBank::PerExpert { .. }, ExpertMatrices::Fused { .. }) => Vec::new(),
        };
        if let Some((ffn, dense)) = &self.shared {
            out.extend(
                dense
                    .bound(ffn)
                    .into_iter()
                    .map(|b| (Operation::SharedExpertProject, b)),
            );
        }
        if let (Some((_, loaded)), Some(gate)) = (
            &self.shared_gate,
            op.shared.as_ref().and_then(|s| s.branch_gate.as_ref()),
        ) {
            out.push((
                Operation::SharedExpertBranchGate,
                Bound::one(&gate.weight, loaded),
            ));
        }
        // The latent wrapper's projections, under the same operation a
        // dense FFN projection binds: one whole matrix per token, read
        // sequentially. Reported here rather than left out because an
        // operand that executes and is not bound is invisible to every
        // residency and consumption instrument — the wrapper would move
        // real bytes that no ledger names.
        if let (Some(loaded), Some(l)) = (&self.latent, &op.latent) {
            let dense = Operation::Project(MatrixClass::FfnProjection);
            out.push((dense, Bound::one(&l.down, &loaded.down)));
            out.push((dense, Bound::one(&l.up, &loaded.up)));
        }
        out
    }

    /// Every expert matrix, for residency accounting. The router itself
    /// is f32 glue and is counted with the norms.
    pub(super) fn loaded_matrices(&self) -> Vec<&LoadedWeight> {
        let mut out = self.experts.all();
        if let Some((_, dense)) = &self.shared {
            out.extend(dense.loaded_matrices());
        }
        if let Some((_, gate)) = &self.shared_gate {
            out.push(gate);
        }
        // Two more real matrices per routed layer. `[3584, 7168]` and
        // `[7168, 3584]` at K3's widths is 51 MB of bf16 a layer — not
        // glue, and not something residency accounting may miss.
        if let Some(latent) = &self.latent {
            out.push(&latent.down);
            out.push(&latent.up);
        }
        out
    }

    pub(super) fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        self.loaded_matrices()
            .into_iter()
            .map(LoadedWeight::slice)
            .collect()
    }

    /// Whether a block of positions goes to this layer's experts as ONE
    /// provider call: the experts are placed and a provider is bound.
    pub(super) fn groups_rows(&self) -> bool {
        matches!(self.placement, Some((_, Some(_))))
    }

    /// The routed FFN for several positions of one layer through a single
    /// provider call ([`PlanBackend::routed_ffn_placed_rows`]): every
    /// position is routed locally, all their selected experts travel
    /// together, and each position is reduced in its own selection order.
    ///
    /// Only for placed experts with a bound provider. Each position's
    /// arithmetic is the single-position path's: the same call, built
    /// from the same operands, differing only in `x`. A layer with a
    /// latent bottleneck or a shared-expert branch is refused, because
    /// that program wraps the routed sum per position and remote
    /// placement does not admit such a plan; nothing has run when it is.
    pub(super) fn apply_many<B: super::super::backend::PlanBackend + ?Sized>(
        &self,
        op: &RoutedFfnOp,
        backend: &B,
        xs: &[&[f32]],
        hidden: usize,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        let refuse = |message: &str| VindexError::Parse(message.to_string());
        let placed = self
            .placement
            .as_ref()
            .and_then(|(l, p)| Some((l, p.as_ref()?)));
        let (layer, provider) = placed.ok_or_else(|| refuse(GROUPED_NEEDS_PROVIDER))?;
        let wrapped = self.latent.is_some() || op.latent.is_some() || self.shared.is_some();
        (!wrapped)
            .then_some(())
            .ok_or_else(|| refuse(GROUPED_REFUSES_WRAPPERS))?;
        let (gate_up, down, gate, up) = match &self.experts {
            ExpertMatrices::Fused {
                gate_up: g,
                down: d,
            } => (slices(g), slices(d), Vec::new(), Vec::new()),
            ExpertMatrices::Separate {
                gate: g,
                up: u,
                down: d,
                ..
            } => (Vec::new(), slices(d), slices(g), slices(u)),
        };
        let layout = op.gate_up_layout;
        let weights = || -> Result<ExpertSlices<'_>, VindexError> {
            Ok(match &self.experts {
                ExpertMatrices::Fused { .. } => ExpertSlices::Fused {
                    gate_up: &gate_up,
                    down: &down,
                    layout: layout.ok_or_else(|| refuse(GROUPED_NEEDS_LAYOUT))?,
                },
                ExpertMatrices::Separate { access, .. } => ExpertSlices::Separate {
                    gate: &gate,
                    up: &up,
                    down: &down,
                    access: *access,
                },
            })
        };
        let calls = xs
            .iter()
            .map(|x| {
                Ok(RoutedFfnCall {
                    x,
                    hidden,
                    intermediate: op.expert_intermediate_size,
                    experts: op.experts,
                    top_k: op.top_k,
                    router_kind: op.router_kind,
                    routing_policy: op.routing_policy,
                    branch_scale: op.executed_branch_scale(),
                    activation: op.activation,
                    gate_policy: op.gate_policy,
                    router: &self.router,
                    router_bias: self.router_bias.as_deref(),
                    weights: weights()?,
                    gate_up_bias: self.gate_up_bias.as_deref(),
                    down_bias: self.down_bias.as_deref(),
                    // The router reads what the experts consume: there is
                    // no latent projection here to make them differ.
                    router_input: None,
                    router_scale: self.router_scale.as_deref(),
                    router_per_expert_scale: self.router_per_expert_scale.as_deref(),
                    router_norm_eps: self.router_norm_eps,
                })
            })
            .collect::<Result<Vec<_>, VindexError>>()?;
        backend.routed_ffn_placed_rows(&calls, *layer, provider.as_ref())
    }

    /// The routed FFN over `x` (what the experts consume), routing on
    /// `router_input` (the same vector for every family but Gemma 4).
    pub(super) fn apply<B: super::super::backend::PlanBackend + ?Sized>(
        &self,
        op: &RoutedFfnOp,
        backend: &B,
        x: &[f32],
        router_input: &[f32],
        hidden: usize,
    ) -> Result<Vec<f32>, VindexError> {
        let (gate_up, down, gate, up);
        let weights = match &self.experts {
            ExpertMatrices::Fused {
                gate_up: g,
                down: d,
            } => {
                gate_up = slices(g);
                down = slices(d);
                ExpertSlices::Fused {
                    gate_up: &gate_up,
                    down: &down,
                    layout: op.gate_up_layout.ok_or_else(|| {
                        VindexError::Parse(
                            "routed FFN op carries a packed bank and no gate_up layout; closure \
                             requires one"
                                .to_string(),
                        )
                    })?,
                }
            }
            ExpertMatrices::Separate {
                gate: g,
                up: u,
                down: d,
                access,
            } => {
                gate = slices(g);
                up = slices(u);
                down = slices(d);
                ExpertSlices::Separate {
                    gate: &gate,
                    up: &up,
                    down: &down,
                    access: *access,
                }
            }
        };
        // The bottleneck the routed experts run behind, entered here so
        // that the two placement facts the oracle found SHAPE-PROTECTED
        // stay structural rather than maintained: the router below reads
        // `router_input`, which is the block input and never this
        // projection, and the shared branch is summed after this whole
        // block returns.
        //
        // `routed_sum -> routed_normed -> routed_out` are the oracle's
        // own boundaries, in its own order.
        let latent_in;
        let (expert_x, expert_width) = match (&self.latent, &op.latent) {
            (Some(loaded), Some(l)) => {
                latent_in = enter_latent(backend, loaded.down.slice(), x, l.width, hidden)?;
                (latent_in.as_slice(), l.width)
            }
            (None, None) => (x, hidden),
            // The loader builds `latent` from `op.latent` and nothing
            // else, so a disagreement is a defect in this file rather
            // than anything a checkpoint can cause — refused rather than
            // silently run at the wrong width.
            _ => {
                return Err(VindexError::Parse(
                    "routed FFN operands and op disagree about the latent branch".to_string(),
                ))
            }
        };
        let call = RoutedFfnCall {
            x: expert_x,
            hidden: expert_width,
            intermediate: op.expert_intermediate_size,
            experts: op.experts,
            top_k: op.top_k,
            router_kind: op.router_kind,
            routing_policy: op.routing_policy,
            branch_scale: op.executed_branch_scale(),
            activation: op.activation,
            gate_policy: op.gate_policy,
            router: &self.router,
            router_bias: self.router_bias.as_deref(),
            weights,
            gate_up_bias: self.gate_up_bias.as_deref(),
            down_bias: self.down_bias.as_deref(),
            // Compared against what the EXPERTS consume, not against the
            // block input: under a latent branch those differ, and the
            // router must still be handed the un-projected vector. This
            // is the placement fact the oracle could not mutate — the
            // reference reads `self.gate(hidden_states)` before any
            // projection exists.
            router_input: (!std::ptr::eq(router_input, expert_x)).then_some(router_input),
            router_scale: self.router_scale.as_deref(),
            router_per_expert_scale: self.router_per_expert_scale.as_deref(),
            router_norm_eps: self.router_norm_eps,
        };
        let mut routed = match &self.placement {
            Some((layer, Some(provider))) => {
                backend.routed_ffn_placed(call, *layer, provider.as_ref())?
            }
            Some((_, None)) => {
                return Err(VindexError::Parse(
                    "routed expert provider is not bound".into(),
                ))
            }
            None => backend.routed_ffn(call)?,
        };
        // Leave the bottleneck. `routed` is the WEIGHTED AGGREGATE at the
        // latent width — the oracle's `routed_sum` — so the norm applies
        // to one vector per token here, after top-k weighting and
        // summation and before the expansion. Normalising per expert, or
        // before the weighting, or after the up-projection are three
        // different models, and this is the only one of the operator's
        // placement facts that no shape can catch.
        if let (Some(loaded), Some(l)) = (&self.latent, &op.latent) {
            routed = exit_latent(
                backend,
                routed,
                loaded.norm.as_ref().map(|(w, eps)| (w.as_slice(), *eps)),
                loaded.up.slice(),
                hidden,
                l.width,
            )?;
        }
        // `y = moe(x) + shared_experts(x)` — the always-active branch is a
        // dense FFN over the same input, composed here, once, for every
        // backend: summed unscaled (DeepSeek / Kimi) or under its declared
        // scalar gate (Qwen MoE).
        //
        // `x`, not the latent: the shared experts read the un-projected
        // block input and are added AFTER the up-projection, so they never
        // enter the bottleneck.
        if let Some((ffn, dense)) = &self.shared {
            let _stage = super::super::stages::stage(super::super::stages::Stage::SharedExpert);
            let shared = dense.apply(ffn, backend, x, hidden)?;
            let scale = match &self.shared_gate {
                Some((spec, weight)) => shared_branch_scale(spec, weight.slice().as_f32()?, x)?,
                None => 1.0,
            };
            for (acc, v) in routed.iter_mut().zip(&shared) {
                *acc += scale * v;
            }
        }
        Ok(routed)
    }

    pub(super) fn load(
        op: &RoutedFfnOp,
        store: OperandSource<'_>,
        bank: super::super::prepared::BankPin,
        shared_format: super::super::prepared::FormatFor<'_>,
        // The latent wrapper's two projections. Resolved under
        // `Project(FfnProjection)` — the operation `bound` reports them
        // under — and NOT under the shared branch's or the bank's pin:
        // they are per-layer dense FFN matrices, and asking for a format
        // under one operation while accounting for another is how a
        // realization check passes against a weight nothing bound that
        // way.
        dense_format: super::super::prepared::FormatFor<'_>,
    ) -> Result<Self, VindexError> {
        let format = bank.format;
        let hidden = op.router.shape.get(1).copied().unwrap_or(0);
        let inter = op.expert_intermediate_size;
        // Where the EXPERTS live, which is not `hidden` once the plan
        // carries a bottleneck. Named apart from `hidden` because the two
        // are the same number on every model but this one, and a single
        // name for both is how a latent bank gets loaded at the residual
        // width and refuses on shape a step later, with a message about
        // bytes rather than about the fact that was misread.
        //
        // The execution-side twin of `MoeSurface::routed_expert_input_width`:
        // that decides what the bank's shape CONTRACT is, this decides
        // what is actually bound, and they read the same declaration.
        let expert_k = op.latent.as_ref().map_or(hidden, |l| l.width);
        // The bank first: its geometry is DECLARED — `k` follows from the
        // declared routed width — and a stray width refuses on the
        // declaration, before any operand's bytes are read.
        let (experts, gate_up_bias, down_bias) = match &op.bank {
            ExpertBank::Packed { gate_up, down } => (
                ExpertMatrices::Fused {
                    gate_up: load_packed(
                        store,
                        gate_up,
                        op,
                        FUSED_BRANCHES * inter,
                        expert_k,
                        format,
                    )?,
                    down: load_packed(store, down, op, expert_k, inter, format)?,
                },
                gate_up.bias.as_ref().map(|b| store.load(b)).transpose()?,
                down.bias.as_ref().map(|b| store.load(b)).transpose()?,
            ),
            // A per-expert bank is never read: each matrix is a region of
            // its object's one mapping, in the form the pin declares, and a
            // pin that is not a mapped form is the plan and the loader
            // disagreeing.
            ExpertBank::PerExpert { gate, up, down } => {
                let form = MappedForm::of(format).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "a per-expert bank pinned to {format:?}: only a mapped stored form \
                         (bf16, f32) binds a bank; planned_operands() and the loader disagree"
                    ))
                })?;
                let map = |operands: &[OperandRef], rows: usize, k: usize| {
                    operands
                        .iter()
                        .map(|operand| {
                            let region = store
                                .store()
                                .map_region(operand, (rows * k * form.width()) as u64)?;
                            Ok(LoadedWeight::Mapped { region, form })
                        })
                        .collect::<Result<Vec<_>, VindexError>>()
                };
                (
                    ExpertMatrices::Separate {
                        gate: map(gate, inter, expert_k)?,
                        up: map(up, inter, expert_k)?,
                        down: map(down, expert_k, inter)?,
                        access: bank.access,
                    },
                    None,
                    None,
                )
            }
        };
        let shared_gate = match op.shared.as_ref().and_then(|s| s.branch_gate.as_ref()) {
            Some(gate) => Some((
                gate.spec,
                load_weight(store, &gate.weight, WeightFormat::F32)?,
            )),
            None => None,
        };
        let shared = match &op.shared {
            Some(shared) => {
                let ffn = FfnOp {
                    intermediate_size: shared.intermediate_size,
                    activation: shared.activation,
                    gate_policy: shared.gate_policy,
                    gate: Some(shared.gate.clone()),
                    up: shared.up.clone(),
                    down: shared.down.clone(),
                    up_bias: None,
                    down_bias: None,
                };
                let dense = DenseOperands::load(&ffn, store, shared_format)?;
                Some((ffn, dense))
            }
            None => None,
        };
        Ok(Self {
            placement: None,
            router: store.load(&op.router)?,
            router_bias: op.router_bias.as_ref().map(|b| store.load(b)).transpose()?,
            router_scale: op
                .router_scale
                .as_ref()
                .map(|s| store.load(s))
                .transpose()?,
            router_per_expert_scale: op
                .router_per_expert_scale
                .as_ref()
                .map(|s| store.load(s))
                .transpose()?,
            router_norm_eps: op.router_norm_eps,
            experts,
            gate_up_bias,
            down_bias,
            shared,
            shared_gate,
            // Built from `op.latent` and nothing else, so operands and
            // op cannot disagree about whether a bottleneck exists.
            latent: op
                .latent
                .as_ref()
                .map(|l| -> Result<LatentOperands, VindexError> {
                    Ok(LatentOperands {
                        down: load_weight(store, &l.down, dense_format(&l.down)?)?,
                        up: load_weight(store, &l.up, dense_format(&l.up)?)?,
                        norm: l
                            .norm
                            .as_ref()
                            .map(|n| store.load(&n.weight).map(|w| (w, n.eps)))
                            .transpose()?,
                    })
                })
                .transpose()?,
        })
    }
}
