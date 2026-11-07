//! CPU realisation selection for the production backend.

use super::super::backend::{
    AttentionCall, AttentionStepCall, AttentionStepOut, GateCall, Nvfp4Activation, WeightFormat,
};
use super::super::cpu::physical::{project_matrix, KQuantExecution};
use super::super::cpu::PhysicalProjectionPlan;
use super::super::kernels::{gather_fused_half, FusedHalf};
use super::super::observe::AttentionHeadRecord;
use super::super::realization::{
    class_of, common_selection, cpu_projection_candidates, realization_residency, RealizationForm,
    RealizationId, RefusalKind, RepresentationFacts, Selection, SelectionReason, SelectionRefusal,
};
use super::super::timing::{timed, OpClass};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::PlannedOperand;
use larql_models::config::{GateActivation, GateCombine, GatePlacement, GateSource};

#[allow(unused_imports)]
use super::*;

impl ProductionBackend {
    /// One position's Q/K/V projections through the production matvec,
    /// conditioned by the shared glue.
    pub(in super::super) fn project_position(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
    ) -> Result<ProjectedAttention, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        // A fused query/gate projection is `2 · head_dim` per head with
        // the halves INTERLEAVED; the first `q_rows` rows are not the
        // queries. See `gather_fused_half`.
        let fused_gate = matches!(
            call.gate.as_ref().map(|g| g.spec.source),
            Some(GateSource::FusedQueryProjection)
        );
        // **Both halves come out of one product.**
        //
        // `FusedQueryProjection` says the gate IS the other half of this
        // projection, over this same activation. Projecting again to
        // collect it read Qwen3.8's `12288 x 5120` q_proj a second time
        // per layer — 2.01 GB/token, 3.8% of every token's traffic, for a
        // vector already computed and discarded.
        //
        // The gather is per HEAD, not a contiguous range: head 0's query
        // rows, then head 0's gate rows, then head 1's. Taking the first
        // or second `q_rows` would have the right shape and the wrong
        // tensor.
        let (mut q, gate) = if fused_gate {
            let full = project_matrix(&call.w_q, pre, q_rows * 2, call.hidden)?;
            (
                gather_fused_half(&full, call.num_q_heads, head_dim, FusedHalf::Query),
                Some(gather_fused_half(
                    &full,
                    call.num_q_heads,
                    head_dim,
                    FusedHalf::Gate,
                )),
            )
        } else {
            (project_matrix(&call.w_q, pre, q_rows, call.hidden)?, None)
        };
        let mut k = project_matrix(&call.w_k, pre, kv_rows, call.hidden)?;
        let mut v = project_matrix(&call.w_v, pre, kv_rows, call.hidden)?;
        add_projection_biases(call, &mut q, &mut k, &mut v);
        condition_v_in_place(call, &mut v);
        condition_qk_in_place(call, position, &mut q, &mut k)?;
        Ok(ProjectedAttention {
            qkv: (q, k, v),
            gate,
        })
    }

    /// Aggregation plus this backend's own gate and output matmuls.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn attend_position<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
        projected_gate: Option<&[f32]>,
    ) -> Result<Vec<f32>, VindexError> {
        Self::attend_position_tapped(
            call,
            position,
            query,
            key_of,
            value_of,
            gate_input,
            projected_gate,
            None,
            None,
        )
    }

    /// [`Self::attend_position`] with the V3-HEAD-OBS-1 tap: when armed,
    /// each head's distribution is kept by the kernel and, once the gate
    /// values are known but BEFORE they multiply the heads, one record
    /// per query head is handed to `tap` — the head's mixed value
    /// pre-gate, its activated gate slice, its distribution and the sink
    /// mass. The arithmetic the executor performs is the same either way.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn attend_position_tapped<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
        projected_gate: Option<&[f32]>,
        tap: Option<&mut dyn FnMut(AttentionHeadRecord<'_>)>,
        head_intervene: Option<&mut super::super::backend::HeadIntervene<'_>>,
    ) -> Result<Vec<f32>, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let mut kept: Vec<Vec<f32>> = Vec::new();
        let mut concat = aggregate_heads_keeping(
            call,
            position,
            query,
            &key_of,
            &value_of,
            tap.as_ref().map(|_| &mut kept),
        );

        // The gate values, computed before anything multiplies the heads,
        // so a tap can carry each head's activated slice; the multiply
        // itself is unchanged below.
        let gate_values: Option<Vec<f32>> = if let Some(GateCall { spec, weight }) = &call.gate {
            // Exhaustive on the judged semantics, same as the
            // reference: a new variant must be implemented before it
            // can execute on this backend either.
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let values = match (spec.source, projected_gate) {
                // Already computed: the projection that produced the
                // queries produced these in the same pass.
                (GateSource::FusedQueryProjection, Some(values)) => values.to_vec(),
                // A fused gate with nothing handed over — the batched
                // path before it threads one through, or a caller that
                // reached here another way. Correct, and reads the
                // operand a second time; the ledger shows it as an extra
                // call rather than hiding it.
                (GateSource::FusedQueryProjection, None) => {
                    let full = project_matrix(weight, gate_input, q_rows * 2, call.hidden)?;
                    gather_fused_half(&full, call.num_q_heads, call.head_dim, FusedHalf::Gate)
                }
                // Its own matrix over its own activation: nothing to
                // share, and sharing would be wrong.
                (GateSource::AttentionInput, _) => {
                    project_matrix(weight, gate_input, q_rows, call.hidden)?
                }
            };
            Some(values)
        } else {
            None
        };

        // V3-HEAD-OBS-1: the records, between aggregation and the gate,
        // built in the one place every backend shares.
        if let Some(tap) = tap {
            let activated: Option<Vec<f32>> = gate_values
                .as_ref()
                .map(|g| g.iter().map(|g| 1.0 / (1.0 + (-g).exp())).collect());
            super::super::observe::fire_head_records(
                tap,
                position,
                call.num_q_heads,
                call.num_kv_heads,
                head_dim,
                source_start(call, position),
                call.sinks.is_some(),
                &concat,
                &kept,
                activated.as_deref(),
                query,
                &key_of,
                &value_of,
            );
        }

        // V3-INTERVENE-2: each head's `ctx_h`, mutated in place if the
        // caller declared an intervention there — AFTER the head record
        // above fired on the uninintervened value (J3), BEFORE the gate
        // multiply, `o_proj` and the post-attention norm below (J1).
        if let Some(head_intervene) = head_intervene {
            for (head, ctx_h) in concat.chunks_exact_mut(head_dim).enumerate() {
                head_intervene(head, ctx_h);
            }
        }

        if let Some(gate_values) = &gate_values {
            let _t = timed(OpClass::OutputGate);
            for (c, g) in concat.iter_mut().zip(gate_values) {
                *c *= 1.0 / (1.0 + (-g).exp());
            }
        }

        let mut out = project_matrix(&call.w_o, &concat, call.hidden, q_rows)?;
        add_output_bias(call, &mut out);
        Ok(out)
    }

    /// The decode step with an optional per-head tap and an optional
    /// per-head intervention: one projection, one attention over the
    /// cached rows plus the fresh one, both threaded into the
    /// aggregation. `attention_step` is this with `None, None`, so the
    /// observed step IS the step.
    pub(super) fn attention_step_tapped(
        step: AttentionStepCall<'_>,
        tap: Option<&mut dyn FnMut(AttentionHeadRecord<'_>)>,
        head_intervene: Option<&mut super::super::backend::HeadIntervene<'_>>,
    ) -> Result<AttentionStepOut, VindexError> {
        let call = &step.op;
        let pre = &call.inputs[0];
        let ProjectedAttention {
            qkv: (q, k, v),
            gate,
        } = Self::project_position(call, step.position, pre)?;
        let output = Self::attend_position_tapped(
            call,
            step.position,
            &q,
            |p| {
                if p == step.position {
                    k.as_slice()
                } else {
                    step.rows().key(p)
                }
            },
            |p| {
                if p == step.position {
                    v.as_slice()
                } else {
                    step.rows().value(p)
                }
            },
            pre,
            gate.as_deref(),
            tap,
            head_intervene,
        )?;
        Ok(AttentionStepOut {
            key: k,
            value: v,
            output,
        })
    }
}

/// The CPU executor's own re-quantised resident forms, offered as
/// candidates for a float source it knows how to narrow — which a codec
/// declares by naming the direct bf16 kernel.
pub(super) const REQUANTISE: [PhysicalProjectionPlan; 4] = [
    PhysicalProjectionPlan::FusedQ8,
    PhysicalProjectionPlan::Q8xQ8,
    PhysicalProjectionPlan::Q4xQ8,
    PhysicalProjectionPlan::FusedQ4,
];

/// [`ProductionBackend::select`]'s decision, with the K-quant execution
/// arm passed in rather than read from the environment — so both arms are
/// testable in one process without touching it.
///
/// **The policy, over a derived candidate set.** The candidates come from
/// the codec's declarations and the executor's own compact forms; the
/// ladder below only ORDERS them, and can answer nothing that is not a
/// candidate. A compiled NVFP4 pack outranks everything; a stored K-quant
/// runs in place or widens by the arm; a float source goes to the size
/// policy, which keeps a large bf16 image compact and widens a small one;
/// a codec with no direct realization decodes, and says so.
// Test-only since NVFP4-Q8-1: the provider selects through
// `select_cpu_with`, and the K-quant tests name only their own arm.
#[cfg(test)]
pub(crate) fn select_cpu(
    operand: &PlannedOperand,
    facts: &RepresentationFacts,
    kquant: KQuantExecution,
) -> Result<Selection, Box<SelectionRefusal>> {
    select_cpu_with(operand, facts, kquant, Nvfp4Activation::F32)
}

/// [`select_cpu`] with the NVFP4 activation arm named too — the
/// provider's, never the environment's.
pub(crate) fn select_cpu_with(
    operand: &PlannedOperand,
    facts: &RepresentationFacts,
    kquant: KQuantExecution,
    nvfp4: Nvfp4Activation,
) -> Result<Selection, Box<SelectionRefusal>> {
    use RealizationForm::{Decode, Direct, Requantise};
    if let Some(common) = common_selection(operand, facts, WeightFormat::F32) {
        return common;
    }
    let refuse = |kind, considered| {
        Box::new(SelectionRefusal {
            operand: operand.operand.clone(),
            operation: operand.operation,
            representation: facts.label.clone(),
            requested: operand.access,
            kind,
            considered,
        })
    };
    let Some(class) = class_of(operand.operation) else {
        return Err(refuse(RefusalKind::MissingRealization, vec![]));
    };
    if facts.registered.is_none() {
        return Err(refuse(RefusalKind::UnregisteredRepresentation, vec![]));
    }
    let candidates = cpu_projection_candidates(facts, PhysicalProjectionPlan::BlasF32, &REQUANTISE);
    let has = |form: RealizationForm| candidates.iter().any(|c| c.form == form);
    let decode = RealizationId::cpu(Decode(PhysicalProjectionPlan::BlasF32));
    let pick = |id: RealizationId, reason: SelectionReason| {
        Ok(Selection {
            realization: id,
            residency: realization_residency(facts, id),
            reason,
            candidates: candidates.clone(),
        })
    };
    if has(Direct(PhysicalProjectionPlan::FusedNvfp4)) {
        let plan = match nvfp4 {
            Nvfp4Activation::Q8 if has(Direct(PhysicalProjectionPlan::FusedNvfp4Q8)) => {
                PhysicalProjectionPlan::FusedNvfp4Q8
            }
            Nvfp4Activation::F32 | Nvfp4Activation::Q8 => PhysicalProjectionPlan::FusedNvfp4,
        };
        return pick(
            RealizationId::cpu(Direct(plan)),
            SelectionReason::DirectDeclared,
        );
    }
    if has(Direct(PhysicalProjectionPlan::FusedKQuant)) {
        return match kquant {
            KQuantExecution::DirectQ8k if has(Direct(PhysicalProjectionPlan::FusedKQuantQ8k)) => {
                pick(
                    RealizationId::cpu(Direct(PhysicalProjectionPlan::FusedKQuantQ8k)),
                    SelectionReason::DirectDeclared,
                )
            }
            // A member with no Q8_K kernel (Q8_0) keeps the in-place f32
            // realization: the arm changes the activation of the formats
            // it can, and nothing else.
            KQuantExecution::Direct | KQuantExecution::DirectQ8k => pick(
                RealizationId::cpu(Direct(PhysicalProjectionPlan::FusedKQuant)),
                SelectionReason::DirectDeclared,
            ),
            KQuantExecution::Widen => pick(decode, SelectionReason::ArmPrefersDecode),
        };
    }
    // The checkpoint's own fine-grained FP8 bytes, executed in place with
    // their scale grid retained. Ranked with the compiled packs above and
    // for the same reason: the stored bytes are the compact form, and the
    // only alternative is a widened image — 612 GB of a 306 GB checkpoint
    // on GLM-5.3-Flash — which stays a candidate for the oracle and is
    // never the policy's choice.
    if has(Direct(PhysicalProjectionPlan::FusedFp8Block)) {
        return pick(
            RealizationId::cpu(Direct(PhysicalProjectionPlan::FusedFp8Block)),
            SelectionReason::DirectDeclared,
        );
    }
    // The size policy is asked whether a bf16 image is worth keeping
    // compact — and the fact it is asked about is the codec DECLARING the
    // direct bf16 kernel, not a dtype the loader compared.
    let bf16_kernel_declared = has(Direct(PhysicalProjectionPlan::FusedBf16));
    let plan = PhysicalProjectionPlan::choose_for(
        Some(class),
        operand.logical_elements,
        bf16_kernel_declared,
    );
    let form = if has(Direct(plan)) {
        Direct(plan)
    } else if plan.format() == WeightFormat::F32 {
        Decode(plan)
    } else if facts.requantise_forbidden && has(Direct(PhysicalProjectionPlan::FusedBf16)) {
        // `stored` forbids the requantised image the size policy wanted;
        // the stored bf16 bytes run directly instead, at higher precision
        // than asked, and nothing is manufactured (#665).
        return pick(
            RealizationId::cpu(Direct(PhysicalProjectionPlan::FusedBf16)),
            SelectionReason::SourcePrecisionHeld,
        );
    } else {
        Requantise(plan)
    };
    if !has(form) {
        return Err(refuse(
            RefusalKind::MissingRealization,
            candidates
                .iter()
                .map(|c| {
                    (
                        *c,
                        "not the resident form the size policy chose".to_string(),
                    )
                })
                .collect(),
        ));
    }
    let reason = match form {
        _ if facts.overlaid => SelectionReason::OverlaidEdit,
        Direct(_) if !bf16_kernel_declared => SelectionReason::DirectDeclared,
        Decode(_) if !bf16_kernel_declared => SelectionReason::NoDirectRealization,
        _ => SelectionReason::SizePolicy,
    };
    pick(RealizationId::cpu(form), reason)
}
