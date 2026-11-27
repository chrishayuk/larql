//! Loading and validating prepared operands.

use super::super::super::{ComponentOpPlan, LayerAttention, OperandRef};
use super::super::accounting::ResidencyBudget;
use super::super::backend::{MatrixClass, PlanBackend, WeightFormat};
use super::super::experts::FfnOperands;
use super::super::operands::OperandSource;
use super::super::weights::load_weight;
use super::super::AttentionOperands;
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::Operation;
use larql_models::config::ResidualTopology;

#[allow(unused_imports)]
use super::*;

impl PreparedOperands {
    /// [`Self::load`] under a residency budget: the pins are chosen so the
    /// plan's physical working set and per-token touch fit `budget`, or
    /// the preparation is refused before any payload byte with the
    /// deficit and the alternatives considered — see
    /// [`select_realizations_within`].
    pub fn load_within<'s, B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &B,
        slice: ExecutionSlice,
        budget: &ResidencyBudget,
    ) -> Result<Self, VindexError> {
        let store = store.into();
        slice.validate(plan)?;
        if matches!(
            slice,
            ExecutionSlice::DenseFfns { .. }
                | ExecutionSlice::DenseFfnCoordinator
                | ExecutionSlice::RoutedExperts { .. }
                | ExecutionSlice::RoutedExpertCoordinator
        ) && store.stamp() != OperandSource::from(store.store()).stamp()
        {
            return Err(VindexError::Parse("FFN placement requires base artifact operands; overlays are not bound by this protocol".into()));
        }
        // **Every declared residual topology is traversable here.**
        // Single-stream always was; hyper-connections joined it in wave
        // 19 when the bundle was witnessed on both the decode step and
        // the batch traversal; attention residuals join it now, their
        // decode (2a) and batch (2b) traversals each witnessed against a
        // Torch oracle transcribed from the reference. The authority
        // this used to consult — `ResidualTopology::unimplemented_reason`
        // — is deleted rather than left answering `None`, so there is no
        // dead refusal here for a reader to consult and conclude from.
        // A topology that cannot be traversed again must bring both the
        // authority and its readers back together.
        //
        // What a hyper-connected image still cannot be is said below by
        // name — a whole-stack image with no declared head reduction, a
        // layer scale under the topology — and the plan report reads the
        // same facts, so a plan it calls executable is one prepared here.
        Self::load_validated(plan, store, backend, slice, budget)
    }

    /// The loader proper, past slice validation. Every operand the slice
    /// needs is loaded here, and none of it is loaded again.
    pub(super) fn load_validated<B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: OperandSource<'_>,
        backend: &B,
        slice: ExecutionSlice,
        budget: &ResidencyBudget,
    ) -> Result<Self, VindexError> {
        let stamp = store.stamp();
        let whole = slice.is_whole_stack();
        // **Select before any operand is loaded.** Every planned operand
        // this slice executes is resolved against the registry, admitted,
        // and pinned to one realization — or the whole plan is refused
        // with every reason — before a byte of any of them is read.
        let realizations = select_realizations_within(plan, store, backend, &slice, budget)?;
        let embedding = plan.embedding.as_ref().ok_or_else(|| {
            VindexError::Parse(format!(
                "component `{}` has no embedding op — external hidden-state input is a later rung",
                plan.component
            ))
        })?;
        let hidden = embedding.table.shape[1];
        if matches!(slice, ExecutionSlice::DenseFfns { .. }) {
            let dense_ffns = super::super::dense_ffn::PreparedDenseFfns::load(
                plan,
                store,
                backend,
                &slice,
                &realizations,
                hidden,
            )?;
            let prepared = Self {
                stamp,
                first_layer: slice.layers(plan).start,
                slice,
                hidden,
                embed_table: None,
                position_table: None,
                layers: Vec::new(),
                final_norm: None,
                output: None,
                registry: store.registry(),
                realizations,
                topology: plan.residual_topology,
                hyper_connection_head: None,
                attention_residual_exit: None,
                dense_ffns: Some(dense_ffns),
                routed_experts: None,
            };
            prepared.verify_pins()?;
            prepared.reconcile(plan, store)?;
            return Ok(prepared);
        }
        if matches!(slice, ExecutionSlice::RoutedExperts { .. }) {
            let worker = super::super::routed_experts::PreparedRoutedExperts::load(
                plan,
                store,
                backend,
                &slice,
                &realizations,
                hidden,
            )?;
            let prepared = Self {
                stamp,
                first_layer: slice.layers(plan).start,
                slice,
                hidden,
                embed_table: None,
                position_table: None,
                layers: Vec::new(),
                final_norm: None,
                output: None,
                registry: store.registry(),
                realizations,
                topology: plan.residual_topology,
                hyper_connection_head: None,
                attention_residual_exit: None,
                dense_ffns: None,
                routed_experts: Some(worker),
            };
            prepared.verify_pins()?;
            prepared.reconcile(plan, store)?;
            return Ok(prepared);
        }
        let embed_table = if whole {
            Some(store.load(&embedding.table)?)
        } else {
            None
        };
        let position_table = match (&embedding.position, whole) {
            (Some(position), true) => Some(store.load(&position.table)?),
            _ => None,
        };
        let topology = plan.residual_topology;
        let hyper_connection = match topology {
            ResidualTopology::HyperConnection(hc) => Some(hc),
            // Neither of the others is a bundle. An attention-residual
            // plan never reaches this loader at all — `load` refuses it
            // above — and the arm answers what is true of the topology
            // rather than restating that refusal.
            ResidualTopology::SingleStream | ResidualTopology::AttentionResidual { .. } => None,
        };
        // The other topology's declaration, as a flag: its sites need no
        // parameter from it (the pair's geometry closes over the width
        // alone), only the fact that the component declares it.
        let attention_residual = matches!(topology, ResidualTopology::AttentionResidual { .. });

        // The loaders ask by operand and class; the answer is the pin.
        let pinned = |op: &OperandRef, operation: Operation| {
            pinned_format(&realizations, store, op, operation)
        };
        let attention_format =
            |op: &OperandRef| pinned(op, Operation::Project(MatrixClass::AttentionProjection));
        let ffn_format =
            |op: &OperandRef| pinned(op, Operation::Project(MatrixClass::FfnProjection));
        let shared_format = |op: &OperandRef| pinned(op, Operation::SharedExpertProject);
        let head_format = |op: &OperandRef| pinned(op, Operation::OutputHead);

        let range = slice.layers(plan);
        let first_layer = range.start;
        let mut layers = Vec::with_capacity(range.len());
        for index in range.clone() {
            let layer = &plan.layers[index];
            // A packed bank's realization is pinned per layer; a layer
            // with no bank never reads the value, and gets the widened
            // form so nothing compact is implied.
            let bank_format = match layer.ffn.as_ref().and_then(|f| f.routed()) {
                Some(_) if slice != ExecutionSlice::RoutedExpertCoordinator => {
                    bank_pin(&realizations, index)?
                }
                _ => BankPin {
                    format: WeightFormat::F32,
                    access: super::super::realization::MappedAccess::Demand,
                },
            };
            layers.push(PreparedLayer {
                // Absent under post-norm placement: the sublayer reads
                // the raw residual there. `None` is the program, and the
                // executor skips the site rather than applying identity.
                pre_attention: match &layer.pre_attention_norm {
                    Some(op) => Some(PreparedNorm::load(op, store)?),
                    None => None,
                },
                // The operator is decided here, from the plan, and the
                // operands follow it. No layer is prepared as softmax by
                // default.
                attention: match &layer.attention {
                    LayerAttention::Softmax(op) => PreparedAttention::Softmax(Box::new(
                        AttentionOperands::load(op, store, &attention_format)?,
                    )),
                    // No executor exists for this operator yet. The
                    // operands are bound and the geometry is stated, but
                    // binding is not running: preparing a KDA layer as
                    // anything else would execute the wrong recurrence on
                    // correctly-bound tensors, which is the failure the
                    // separate variant exists to make impossible.
                    // The layer's own pre-attention norm epsilon: KDA's
                    // gated output norm is built as
                    // `FusedRMSNormGated(head_dim, eps=config.rms_norm_eps)`
                    // in the checkpoint's own modeling code, the same
                    // value the layer norms use — unlike MLA's latent
                    // norm below, which is exactly why that one is
                    // carried per-op and this one is not.
                    LayerAttention::Kda(op) => PreparedAttention::Kda(Box::new(KdaOperands::load(
                        op,
                        store,
                        &attention_format,
                        layer.declared_norm_eps as f32,
                    )?)),
                    // Same posture as KDA above: represented, not
                    // executable. MLA's operands are bound and its
                    // geometry is stated, but no executor consumes them.
                    LayerAttention::Mla(op) => PreparedAttention::Mla(Box::new(MlaOperands::load(
                        op,
                        store,
                        &attention_format,
                    )?)),
                    LayerAttention::ConvQkv(op) => PreparedAttention::ConvQkv(Box::new(
                        ConvQkvOperands::load(op, store, &attention_format)?,
                    )),
                    LayerAttention::Mamba2(op) => PreparedAttention::Mamba2(Box::new(
                        Mamba2Operands::load(op, store, &attention_format)?,
                    )),
                    LayerAttention::GatedDelta(op) => {
                        PreparedAttention::GatedDelta(Box::new(GatedDeltaOperands::load(
                            op,
                            store,
                            &attention_format,
                            layer.declared_norm_eps as f32,
                        )?))
                    }
                },
                post_attention: layer
                    .post_attention_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                // Absent on a mixer-only layer: the plan carries no FFN
                // program there, and preparing one would fabricate work
                // the executor must then skip.
                pre_ffn: layer
                    .pre_ffn_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                ffn: layer
                    .ffn
                    .as_ref()
                    .map(|ffn| {
                        if slice == ExecutionSlice::DenseFfnCoordinator {
                            Ok(FfnOperands::External {
                                layer: index,
                                provider: None,
                            })
                        } else if slice == ExecutionSlice::RoutedExpertCoordinator {
                            FfnOperands::load_routed_coordinator(ffn, store, index)
                        } else {
                            FfnOperands::load(ffn, store, &ffn_format, bank_format, &shared_format)
                        }
                    })
                    .transpose()?,
                post_ffn: layer
                    .post_ffn_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                layer_scale: layer
                    .layer_scale
                    .as_ref()
                    .map(|op| {
                        store
                            .load(op)
                            .and_then(|v| super::super::layer_scalar_of(&v))
                    })
                    .transpose()?,
                hyper_connection: PreparedHyperConnection::for_layer(
                    layer,
                    hyper_connection,
                    hidden,
                    store,
                )?,
                attention_residual: PreparedAttentionResidual::for_layer(
                    layer,
                    attention_residual,
                    hidden,
                    store,
                )?,
            });
        }

        let final_norm = if whole {
            plan.final_norm
                .as_ref()
                .map(|op| PreparedNorm::load(op, store))
                .transpose()?
        } else {
            None
        };
        let output = if whole {
            plan.output
                .as_ref()
                .map(|op| {
                    Ok::<_, VindexError>((
                        op.clone(),
                        load_weight(store, &op.projection, head_format(&op.projection)?)?,
                    ))
                })
                .transpose()?
        } else {
            None
        };

        // The head's reduction belongs to the stack's END: a whole-stack
        // image of a hyper-connected component must carry one, and a
        // layer-range image must not consult one — the per-layer contract
        // needs no head (GLM-5.3-Flash has none to offer).
        let hyper_connection_head = match (whole, hyper_connection) {
            (true, Some(hc)) => Some(PreparedHcHead::load(plan, hc, hidden, store)?),
            _ => None,
        };
        // The exit reduction belongs to the stack's END for the same
        // reason the head's does — and unlike the head it is REQUIRED
        // under its declaration, so a whole-stack image without one
        // refuses here rather than running a stack whose history nothing
        // collapses. A layer-range image must not consult one: its
        // output IS the history it hands on.
        let attention_residual_exit = match (whole, attention_residual) {
            (true, true) => Some(PreparedAttnResExit::load(plan, hidden, store)?),
            _ => None,
        };
        let prepared = Self {
            stamp,
            slice,
            hidden,
            embed_table,
            position_table,
            first_layer,
            layers,
            final_norm,
            output,
            registry: store.registry(),
            realizations,
            topology,
            hyper_connection_head,
            attention_residual_exit,
            dense_ffns: None,
            routed_experts: None,
        };
        // **The executor runs what was pinned.** Every resident matrix
        // holds the representation its record named, checked here so a
        // loader that drifted from the selector cannot hand the executor
        // bytes the plan never pinned.
        prepared.verify_pins()?;
        prepared.place(backend);
        Ok(prepared)
    }
}
