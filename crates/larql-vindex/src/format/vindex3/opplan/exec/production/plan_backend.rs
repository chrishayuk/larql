//! The production `PlanBackend` impl.

use super::super::backend::{
    AttentionCall, AttentionOut, AttentionStepCall, AttentionStepOut, ExpertSlices, FfnCall,
    FfnManyCall, NormCall, PlanBackend, ProjectCall, RoutedFfnCall, WeightSlice,
};
use super::super::cpu::physical::{
    kquant_execution, project_matrix, project_matrix_many, ExecutorProjections,
};
use super::super::lowering::LoweringIdentity;
use super::super::observe::AttentionHeadRecord;
use super::super::prefetch;
use super::super::realization::{RepresentationFacts, Selection, SelectionRefusal};
use super::super::routing_trace;
use super::super::stages::{stage, Stage};
use super::super::timing::{timed, OpClass};
use crate::error::VindexError;
use crate::format::vindex3::opplan::planned::PlannedOperand;
use larql_compute::cpu::ops::moe::math::matmul_vec;
use larql_compute::residual::{layer_norm_eps, rms_norm_eps};
use larql_compute::MoeGateRule;
use larql_models::config::{GateUpBranch, NormType};

#[allow(unused_imports)]
use super::*;

impl PlanBackend for ProductionBackend {
    fn dense_projector(&self) -> &dyn super::super::gated_delta::DenseProjections {
        &ExecutorProjections
    }

    /// **The policy.** One decision per matrix, producing the resident
    /// form here and the kernel at [`project_rows`] — see
    /// [`PhysicalProjectionPlan`] and [`select_cpu`].
    fn select(
        &self,
        operand: &PlannedOperand,
        facts: &RepresentationFacts,
    ) -> Result<Selection, Box<SelectionRefusal>> {
        select_cpu_with(
            operand,
            facts,
            self.kquant.unwrap_or_else(kquant_execution),
            self.nvfp4,
        )
    }

    fn name(&self) -> &str {
        if self.is_q8k() {
            Q8K_NAME
        } else if self.is_nvfp4_q8() {
            NVFP4_Q8_NAME
        } else {
            NAME
        }
    }

    fn identity(&self) -> LoweringIdentity {
        if self.is_q8k() {
            LoweringIdentity::new(Q8K_IDENTITY_FAMILY, Q8K_IDENTITY_REVISION)
        } else if self.is_nvfp4_q8() {
            LoweringIdentity::new(NVFP4_Q8_IDENTITY_FAMILY, NVFP4_Q8_IDENTITY_REVISION)
        } else {
            LoweringIdentity::new(IDENTITY_FAMILY, IDENTITY_REVISION)
        }
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        let _t = timed(OpClass::Embed);
        let row = &table[token as usize * hidden..(token as usize + 1) * hidden];
        match scale {
            Some(scale) => row.iter().map(|v| v * scale).collect(),
            None => row.to_vec(),
        }
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        let _t = timed(OpClass::Norm);
        let weight = (!call.weight.is_empty()).then(|| call.weight.to_vec());
        let normed = match call.kind {
            NormType::RmsNorm => rms_norm_eps(
                &as_row(call.x),
                weight.as_ref(),
                call.weight_offset,
                call.eps,
            ),
            // The production layer-norm kernel takes a bias; the plan
            // carries none, so it is absent rather than zeroed.
            NormType::LayerNorm => layer_norm_eps(&as_row(call.x), weight.as_ref(), None, call.eps),
        };
        from_row(normed)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        project_matrix(&call.weight, call.x, call.out_dim, call.in_dim)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<AttentionOut, VindexError> {
        // Positions are independent, so projection runs in parallel with
        // each position's arithmetic untouched — bit-identical to the
        // serial order.
        let projected: Vec<ProjectedAttention> = call
            .inputs
            .par_iter()
            .enumerate()
            .map(|(position, pre)| Self::project_position(&call, position, pre))
            .collect::<Result<_, VindexError>>()?;
        let mut queries = Vec::with_capacity(projected.len());
        let mut keys = Vec::with_capacity(projected.len());
        let mut values = Vec::with_capacity(projected.len());
        // The gate halves travel with their positions: the batched path
        // shares the projection exactly as the step path does, so the two
        // do not differ in how many times they read `w_q`.
        let mut gates: Vec<Option<Vec<f32>>> = Vec::with_capacity(projected.len());
        for ProjectedAttention {
            qkv: (q, k, v),
            gate,
        } in projected
        {
            queries.push(q);
            keys.push(k);
            values.push(v);
            gates.push(gate);
        }

        // Each query position reads every position's K/V but writes only
        // its own output row — parallel over queries, arithmetic intact.
        let outputs: Vec<Vec<f32>> = queries
            .par_iter()
            .enumerate()
            .map(|(position, query)| {
                Self::attend_position(
                    &call,
                    position,
                    query,
                    |p| keys[p].as_slice(),
                    |p| values[p].as_slice(),
                    &call.inputs[position],
                    gates[position].as_deref(),
                )
            })
            .collect::<Result<_, VindexError>>()?;
        Ok(AttentionOut {
            outputs,
            keys,
            values,
        })
    }

    fn attention_step(&self, step: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        Self::attention_step_tapped(step, None, None)
    }

    fn serves_attention_heads(&self) -> bool {
        true
    }

    fn attention_step_observed(
        &self,
        step: AttentionStepCall<'_>,
        tap: &mut dyn FnMut(AttentionHeadRecord<'_>),
    ) -> Result<AttentionStepOut, VindexError> {
        Self::attention_step_tapped(step, Some(tap), None)
    }

    fn serves_head_intervention(&self) -> bool {
        true
    }

    fn attention_step_intervened(
        &self,
        step: AttentionStepCall<'_>,
        tap: Option<&mut dyn FnMut(AttentionHeadRecord<'_>)>,
        head_intervene: &mut super::super::backend::HeadIntervene<'_>,
    ) -> Result<AttentionStepOut, VindexError> {
        Self::attention_step_tapped(step, tap, Some(head_intervene))
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.ffn_observed(call, &mut |_| {})
    }

    fn serves_ffn_down_input(&self) -> bool {
        true
    }

    fn ffn_observed(
        &self,
        call: FfnCall<'_>,
        tap: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<f32>, VindexError> {
        require_executable_gate("production", call.gate_policy)?;
        let up = project_matrix(&call.up, call.x, call.intermediate, call.hidden)?;
        let gate = match call.gate {
            Some(w) => Some(project_matrix(&w, call.x, call.intermediate, call.hidden)?),
            None => None,
        };
        let inner = ffn_activation(gate.as_deref(), &up, call.activation, call.gate_policy)?;
        tap(&inner);
        project_matrix(&call.down, &inner, call.hidden, call.intermediate)
    }

    /// **CPU-7C2.** The dense FFN over several positions, with each
    /// projection taken as ONE weight traversal.
    ///
    /// The activation stays per position — it is elementwise, it is small
    /// against the projections, and grouping it would be a change to the
    /// arithmetic rather than to the schedule.
    ///
    /// Note what is NOT here: no `par_iter` over positions. Rows own the
    /// machine and positions live inside the row traversal. The previous
    /// shape ran positions in parallel and each of them re-entered the
    /// executor, where `caller_owns_the_machine` collapsed every
    /// projection to a single worker — CPU-7C1 measured that as
    /// `slabs/call` 5.03 -> 2.81 and a 42% loss against serial decode.
    fn ffn_many(&self, call: FfnManyCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        require_executable_gate("production", call.gate_policy)?;
        let ups = project_matrix_many(&call.up, call.xs, call.intermediate, call.hidden)?;
        let gates = match &call.gate {
            Some(w) => Some(project_matrix_many(
                w,
                call.xs,
                call.intermediate,
                call.hidden,
            )?),
            None => None,
        };
        let inners: Vec<Vec<f32>> = (0..call.xs.len())
            .map(|p| {
                ffn_activation(
                    gates.as_ref().map(|g| g[p].as_slice()),
                    &ups[p],
                    call.activation,
                    call.gate_policy,
                )
            })
            .collect::<Result<_, _>>()?;
        let refs: Vec<&[f32]> = inners.iter().map(Vec::as_slice).collect();
        project_matrix_many(&call.down, &refs, call.hidden, call.intermediate)
    }

    fn routed_ffn(&self, call: RoutedFfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.routed_ffn_impl(call, None)
    }
    fn routed_ffn_placed(
        &self,
        call: RoutedFfnCall<'_>,
        layer: usize,
        provider: &dyn super::super::routed_experts::RoutedExpertProvider,
    ) -> Result<Vec<f32>, VindexError> {
        self.routed_ffn_impl(call, Some((layer, provider)))
    }
    fn routed_ffn_placed_rows(
        &self,
        calls: &[RoutedFfnCall<'_>],
        layer: usize,
        provider: &dyn super::super::routed_experts::RoutedExpertProvider,
    ) -> Result<Vec<Vec<f32>>, VindexError> {
        use super::super::routed_experts::{reduce_rows, ExpertRow};
        if calls.is_empty() {
            return Err(VindexError::Parse(
                "multi-row routed call carries no row".into(),
            ));
        }
        let hidden = calls[0].hidden;
        if calls.iter().any(|call| call.hidden != hidden) {
            return Err(VindexError::Parse(
                "multi-row routed call mixes hidden widths".into(),
            ));
        }
        // Route every row exactly as the single-row path does, before any
        // expert runs: the provider only ever sees ids, never weights.
        let mut selected = Vec::with_capacity(calls.len());
        for (index, call) in calls.iter().enumerate() {
            let id = u32::try_from(index)
                .map_err(|_| VindexError::Parse("too many rows in one routed call".into()))?;
            let routed = self.route(call)?;
            routing_trace::record(&routed);
            selected.push((id, routed));
        }
        let ids: Vec<Vec<usize>> = selected
            .iter()
            .map(|(_, routed)| routed.iter().map(|(expert, _)| *expert).collect())
            .collect();
        let rows: Vec<ExpertRow<'_>> = calls
            .iter()
            .zip(&selected)
            .zip(&ids)
            .map(|((call, (id, _)), experts)| ExpertRow {
                id: *id,
                input: call.x,
                experts,
            })
            .collect();
        let outputs = provider.apply_rows(layer, &rows)?;
        reduce_rows(&selected, outputs, hidden)
    }
    fn expert_transform(
        &self,
        call: super::super::routed_experts::ExpertTransformCall<'_>,
    ) -> Result<Vec<f32>, VindexError> {
        use super::super::routed_experts::ExpertWeights;
        if call.hidden == 0
            || call.intermediate == 0
            || call.x.len() != call.hidden
            || call.x.iter().any(|v| !v.is_finite())
        {
            return Err(VindexError::Parse(
                "expert input dimensions or finiteness mismatch".into(),
            ));
        }
        match call.weights {
            ExpertWeights::Fused {
                gate_up,
                down,
                layout,
                gate_up_bias,
                down_bias,
            } => {
                let two_inter = call
                    .intermediate
                    .checked_mul(FUSED_BRANCHES)
                    .ok_or_else(|| VindexError::Parse("expert width overflow".into()))?;
                let g = gate_up.as_f32()?;
                let d = down.as_f32()?;
                if two_inter.checked_mul(call.hidden) != Some(g.len())
                    || call.hidden.checked_mul(call.intermediate) != Some(d.len())
                    || gate_up_bias.is_some_and(|b| b.len() != two_inter)
                    || down_bias.is_some_and(|b| b.len() != call.hidden)
                {
                    return Err(VindexError::Parse(
                        "expert projection or bias dimensions mismatch".into(),
                    ));
                }
                let mut fused = matmul_vec(call.x, g, two_inter, call.hidden);
                add_expert_bias(&mut fused, gate_up_bias, 0);
                let rule = MoeGateRule::from_arch(call.gate_policy, call.activation);
                let inner: Vec<f32> = (0..call.intermediate)
                    .map(|i| {
                        rule.combine(
                            fused[layout.row(GateUpBranch::Gate, i, call.intermediate)],
                            fused[layout.row(GateUpBranch::Up, i, call.intermediate)],
                        )
                    })
                    .collect();
                let mut out = matmul_vec(&inner, d, call.hidden, call.intermediate);
                add_expert_bias(&mut out, down_bias, 0);
                Ok(out)
            }
            ExpertWeights::Separate { gate, up, down } => {
                let g = project_matrix(&gate, call.x, call.intermediate, call.hidden)?;
                let u = project_matrix(&up, call.x, call.intermediate, call.hidden)?;
                let rule = MoeGateRule::from_arch(call.gate_policy, call.activation);
                let inner: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(g, u)| rule.combine(*g, *u))
                    .collect();
                project_matrix(&down, &inner, call.hidden, call.intermediate)
            }
        }
    }

    fn output_head(
        &self,
        projection: super::super::backend::WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut logits = project_matrix(&projection, x, vocab, hidden)?;
        // The vocabulary pass only — 248320 elements of multiplier and
        // softcap, which is real work and nothing to do with the matmul
        // that produced them.
        let _t = timed(OpClass::Logits);
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(logits)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        let _t = timed(OpClass::Residual);
        for (a, b) in acc.iter_mut().zip(delta) {
            *a += b;
        }
    }
}

impl ProductionBackend {
    /// Router projection and top-k selection for one row: `(expert, weight)`
    /// in production selection order. Shared by the single-row and multi-row
    /// paths so both select from identical arithmetic.
    fn route(&self, call: &RoutedFfnCall<'_>) -> Result<Vec<(usize, f32)>, VindexError> {
        let _stage = stage(Stage::Router);
        let routed_input = router_input(call)?;
        // The router's `k` is the width of what it READS, which is
        // not `call.hidden` once the experts run behind a bottleneck:
        // `call.hidden` is then the latent width, while the router
        // still projects from the block input. Taking it from the
        // vector itself keeps the two impossible to desync.
        let k = routed_input.len();
        let mut logits = matmul_vec(&routed_input, call.router, call.experts, k);
        select_experts(call, &mut logits)
    }

    pub(super) fn routed_ffn_impl(
        &self,
        call: RoutedFfnCall<'_>,
        placed: Option<(
            usize,
            &dyn super::super::routed_experts::RoutedExpertProvider,
        )>,
    ) -> Result<Vec<f32>, VindexError> {
        let started = super::super::profile::enabled().then(std::time::Instant::now);
        let selected = self.route(&call)?;
        let router_ns = started.map(|s| s.elapsed().as_nanos() as u64);
        routing_trace::record(&selected);
        if placed.is_none() {
            if let ExpertSlices::Separate {
                gate,
                up,
                down,
                access,
            } = &call.weights
            {
                // The selected experts' pages, ahead of the loop that reads
                // them — the access realization, timed apart from the loop
                // so a fault moved is a fault moved, not a fault removed.
                let _prefetch = stage(Stage::Prefetch);
                let ranges: Vec<prefetch::Range> = selected
                    .iter()
                    .flat_map(|(e, _)| [&gate[*e], &up[*e], &down[*e]])
                    .filter_map(|w| match w {
                        WeightSlice::Bf16(rows) => Some(prefetch::Range::of(rows)),
                        WeightSlice::F32(rows) => Some(prefetch::Range::of(rows)),
                        _ => None,
                    })
                    .collect();
                let parallelism = super::super::cpu::shared()
                    .map(|e| e.workers())
                    .unwrap_or(1);
                prefetch::prefetch(*access, &ranges, parallelism);
            }
        }
        let _stage = stage(Stage::RoutedExperts);
        let dispatch = started.map(|_| std::time::Instant::now());
        let mut local_expert_ns = 0u64;
        let rows = match placed {
            Some((layer, provider)) => {
                let ids: Vec<usize> = selected.iter().map(|(id, _)| *id).collect();
                provider.apply(layer, call.x, &ids)?
            }
            None => selected
                .iter()
                .map(|(expert, _)| {
                    let transform = started.map(|_| std::time::Instant::now());
                    let row = self.expert_transform(call.expert_transform(*expert)?)?;
                    if let Some(transform) = transform {
                        local_expert_ns += transform.elapsed().as_nanos() as u64;
                    }
                    Ok(super::super::routed_experts::ExpertOutput {
                        expert: *expert,
                        row,
                    })
                })
                .collect::<Result<Vec<_>, VindexError>>()?,
        };
        let dispatch_ns = dispatch.map(|s| s.elapsed().as_nanos() as u64);
        let reduction = started.map(|_| std::time::Instant::now());
        let result = super::super::routed_experts::reduce_selected(&selected, rows, call.hidden);
        if let (Some(started), Some(reduction)) = (started, reduction) {
            let reduction_ns = reduction.elapsed().as_nanos() as u64;
            super::super::profile::record_provider_call(serde_json::json!({
                "kind": "routed_ffn", "layer": super::super::profile::current_layer(),
                "remote": placed.is_some(), "selected_count": selected.len(),
                "router_ns": router_ns, "dispatch_ns": dispatch_ns,
                "local_expert_ns": placed.is_none().then_some(local_expert_ns),
                "reduction_ns": reduction_ns, "total_ns": started.elapsed().as_nanos() as u64,
                "complete": result.is_ok(),
            }));
        }
        result
    }
}
