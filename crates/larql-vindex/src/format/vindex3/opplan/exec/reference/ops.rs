//! The f32 reference backend operations.

use super::super::backend::{
    AttentionCall, AttentionStepCall, AttentionStepOut, GateCall, ProjectedQkv, QkNormCall,
};
use super::super::kernels::{
    gather_fused_half_mutated, linear_frequencies, llama3_frequencies, matvec, mrope_rotate, norm,
    partial_rotary_frequencies, partial_rotary_slice, rope_rotate, rope_rotate_scaled, sigmoid,
    softcap, softmax, softmax_with_sink, yarn_frequencies, FusedHalf, GateMutation,
};
use super::super::observe::{AttentionHeadRecord, UnpositionedQk};
use crate::error::VindexError;
use larql_models::config::NormType;
use larql_models::config::{
    AttentionSinkSpec, GateActivation, GateCombine, GatePlacement, GateSource, QkNormScope,
};
use larql_models::config::{PositionPolicy, RotaryFrequencyBasis};

#[allow(unused_imports)]
use super::*;

impl ReferenceBackend {
    pub fn new() -> Self {
        Self
    }

    /// Q/K normalisation: weighted per-head when the plan binds weights,
    /// parameter-free when the surface judged it. Both may apply.
    pub(super) fn apply_qk_norm(
        call: &AttentionCall<'_>,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), VindexError> {
        let head_dim = call.head_dim;
        let eps = call.qk_norm_eps;
        if let Some(QkNormCall {
            scope,
            weight_offset,
            q_weight,
            k_weight,
        }) = &call.qk_norm
        {
            match scope {
                QkNormScope::PerHead => {
                    for head in q.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, q_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, k_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                }
                QkNormScope::FullProjection => {
                    return Err(VindexError::Parse(
                        "full-projection QK norm has no judged reference execution yet".to_string(),
                    ));
                }
            }
        }
        if call.parameter_free_qk_norm.q {
            for head in q.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        if call.parameter_free_qk_norm.k {
            for head in k.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        Ok(())
    }

    /// One position's Q/K/V projections with QK normalisation, query
    /// scale and position encoding applied in the judged order — the
    /// arithmetic both the batch path and the decode step share, so the
    /// two cannot disagree about a single position.
    /// Whether this layer's gate is sourced from the query projection —
    /// the one case where `w_q` is wider than the attention width.
    pub(super) fn fused_query_gate(call: &AttentionCall<'_>) -> bool {
        matches!(
            call.gate.as_ref().map(|g| g.spec.source),
            Some(GateSource::FusedQueryProjection)
        )
    }

    /// [`Self::project_position`] with a deliberate defect in the fused
    /// query/gate split. The public path above always passes
    /// [`GateMutation::None`]; the mutation table drives THIS function,
    /// so the table exercises the shipped implementation.
    pub(in super::super) fn project_position_inner(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
        gate_mutation: GateMutation,
    ) -> Result<ProjectedQkv, VindexError> {
        Self::project_position_keeping(call, position, pre, gate_mutation, None)
    }

    /// [`Self::project_position_inner`], copying Q and K into `keep`
    /// immediately before position encoding when asked (PREC-DIAG-1
    /// E0a). The copy reads the buffers; the rotation then runs on them
    /// exactly as it does with `keep = None`.
    pub(in super::super) fn project_position_keeping(
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
        gate_mutation: GateMutation,
        keep: Option<&mut UnpositionedQk>,
    ) -> Result<ProjectedQkv, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        // A fused query projection carries `2 · head_dim` rows per head,
        // query and gate INTERLEAVED. Taking the first `q_rows` would
        // read the query of head 0, the gate of head 0, the query of
        // head 1 … and call the result "the queries" — right shape,
        // wrong tensor. The halves are gathered per head instead.
        let mut q = if Self::fused_query_gate(call) {
            let full = matvec(call.w_q.as_f32()?, q_rows * 2, call.hidden, pre);
            gather_fused_half_mutated(
                &full,
                call.num_q_heads,
                head_dim,
                FusedHalf::Query,
                gate_mutation,
            )
        } else {
            matvec(call.w_q.as_f32()?, q_rows, call.hidden, pre)
        };
        let mut k = matvec(call.w_k.as_f32()?, kv_rows, call.hidden, pre);
        let mut v = matvec(call.w_v.as_f32()?, kv_rows, call.hidden, pre);
        // Biases belong to the projections: added before anything reads
        // the projected values (QK-norm, rope, the cache).
        if let Some(bias) = &call.bias {
            add_in_place(&mut q, bias.q);
            add_in_place(&mut k, bias.k);
            add_in_place(&mut v, bias.v);
        }

        Self::apply_qk_norm(call, &mut q, &mut k)?;
        // The parameter-free V norm (Gemma 4 `v_norm`): per head, no
        // weight, the same epsilon — applied to the raw value projection
        // (on a K≡V layer that is the raw K projection, before its norm
        // and rotation, which is why V is taken before either).
        if call.parameter_free_qk_norm.v {
            for head in v.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, call.qk_norm_eps);
                head.copy_from_slice(&normed);
            }
        }
        if let Some(query_scale) = call.query_scale {
            for value in &mut q {
                *value *= query_scale as f32;
            }
        }
        if let Some(keep) = keep {
            keep.keep(&q, &k);
        }
        match call.position {
            PositionPolicy::Rope { theta } => {
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate(head, position, theta);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate(head, position, theta);
                }
            }
            // YaRN: the ramped frequency blend AND the amplitude on
            // cos/sin, from the reference transcription of the block the
            // container carries.
            PositionPolicy::Yarn { theta, scaling } => {
                let (inv_freq, amplitude) = yarn_frequencies(&scaling, head_dim, theta);
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, amplitude);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, amplitude);
                }
            }
            // Linear, transcribed: every frequency divided by the
            // factor, unit amplitude — Gemma 3's global layers. Written
            // as its own arm rather than a divisor on the plain arm, so
            // a plain layer can never inherit a divisor.
            PositionPolicy::Linear { theta, factor } => {
                const UNIT_AMPLITUDE: f32 = 1.0;
                let inv_freq = linear_frequencies(head_dim, theta, factor);
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, UNIT_AMPLITUDE);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, UNIT_AMPLITUDE);
                }
            }
            // Llama-3, transcribed: wavelength-band frequencies at unit
            // amplitude. `UNIT_AMPLITUDE` is passed explicitly rather
            // than defaulted, so an amplitude arriving here later has to
            // be written down rather than inherited.
            PositionPolicy::Llama3 { theta, scaling } => {
                const UNIT_AMPLITUDE: f32 = 1.0;
                let inv_freq = llama3_frequencies(&scaling, head_dim, theta);
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, UNIT_AMPLITUDE);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate_scaled(head, position, &inv_freq, UNIT_AMPLITUDE);
                }
            }
            // See `production.rs`: no backend rotates for a relative
            // scheme, and silently skipping position is a wrong answer
            // that still produces fluent output.
            PositionPolicy::Relative { d_rel, extent } => {
                return Err(VindexError::Parse(format!(
                    "relative position (d_rel {d_rel}, extent {extent}) is represented but not \
                     executable: no backend implements it"
                )))
            }
            PositionPolicy::None => {}
            // Partial rotary, transcribed: head-width basis is the full
            // rotate-half table with the top frequencies zero; rotary-width
            // basis rotates the prefix as its own block.
            PositionPolicy::PartialRope {
                theta,
                rotary_fraction,
                basis,
            } => match basis {
                RotaryFrequencyBasis::HeadWidth => {
                    let inv_freq = partial_rotary_frequencies(head_dim, rotary_fraction, theta);
                    for head in q.chunks_exact_mut(head_dim) {
                        rope_rotate_scaled(head, position, &inv_freq, 1.0);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        rope_rotate_scaled(head, position, &inv_freq, 1.0);
                    }
                }
                RotaryFrequencyBasis::RotaryWidth => {
                    let width = partial_rotary_slice(head_dim, rotary_fraction);
                    for head in q.chunks_exact_mut(head_dim) {
                        rope_rotate(&mut head[..width], position, theta);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        rope_rotate(&mut head[..width], position, theta);
                    }
                }
            },
            // Multi-axis rotary. The interpreter holds one scalar
            // position, so the grid is `(p, p, p)` — a text sequence,
            // where the axis assignment provably selects equal values.
            // The assignment still runs: see `mrope_rotate`.
            PositionPolicy::MRope {
                theta,
                rotary_fraction,
                basis,
                section,
                interleaved,
            } => {
                let width =
                    match basis {
                        RotaryFrequencyBasis::RotaryWidth => {
                            partial_rotary_slice(head_dim, rotary_fraction)
                        }
                        // No judged checkpoint pairs a head-width basis with
                        // M-RoPE; refusing beats guessing which block the
                        // sections index.
                        RotaryFrequencyBasis::HeadWidth => return Err(VindexError::Parse(
                            "M-RoPE with a head-width frequency basis is unjudged; no checkpoint \
                             declares it and the section-to-dimension mapping is undefined"
                                .to_string(),
                        )),
                    };
                let grid = [position, position, position];
                for head in q.chunks_exact_mut(head_dim) {
                    mrope_rotate(&mut head[..width], grid, theta, section, interleaved);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    mrope_rotate(&mut head[..width], grid, theta, section, interleaved);
                }
            }
        }
        Ok((q, k, v))
    }

    /// [`Self::attend_position`] with a deliberate defect in the gate
    /// stage. See [`Self::project_position_inner`].
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn attend_position_inner<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
        gate_mutation: GateMutation,
    ) -> Result<Vec<f32>, VindexError> {
        Self::attend_position_tapped(
            call,
            position,
            query,
            &key_of,
            &value_of,
            gate_input,
            gate_mutation,
            None,
            None,
        )
    }

    /// [`Self::attend_position_inner`] with the V3-HEAD-OBS-1 tap: the
    /// oracle's own loop keeps each head's distribution when asked and
    /// fires one record per query head after the gate values are known
    /// and before they multiply the heads. The arithmetic is the same
    /// either way; the tap is what the production kernel's tap is gated
    /// against.
    ///
    /// The row lookups are trait objects, not generics: the batch and the
    /// decode step then run ONE compiled body of this loop rather than a
    /// copy per caller's closure type, so the oracle the two paths are
    /// judged against is literally the same code.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn attend_position_tapped<'k>(
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: &dyn Fn(usize) -> &'k [f32],
        value_of: &dyn Fn(usize) -> &'k [f32],
        gate_input: &[f32],
        gate_mutation: GateMutation,
        mut tap: Option<&mut dyn FnMut(AttentionHeadRecord<'_>)>,
        mut head_intervene: Option<&mut super::super::backend::HeadIntervene<'_>>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut kept: Vec<Vec<f32>> = Vec::new();
        let mut fired = false;
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let group = call.num_q_heads / call.num_kv_heads;
        // Span: which key positions this query may attend to — the plan's
        // retention authority, never recomputed here.
        let start = call.history().required_start(position);
        let mut concat = vec![0.0f32; q_rows];
        for q_head in 0..call.num_q_heads {
            let kv_head = q_head / group;
            let q_slice = &query[q_head * head_dim..(q_head + 1) * head_dim];
            let mut scores: Vec<f32> = (start..=position)
                .map(|key_position| {
                    let k_slice =
                        &key_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for (a, b) in q_slice.iter().zip(k_slice) {
                        dot += a * b;
                    }
                    let mut score = dot * call.score_scale as f32;
                    if let Some(cap) = call.logit_softcapping {
                        score = softcap(score, cap);
                    }
                    score
                })
                .collect();
            match &call.sinks {
                // Exhaustive on the judged semantics: a new variant must
                // be implemented here before it can execute.
                Some(sinks) => {
                    let AttentionSinkSpec::SoftmaxDenominator = sinks.spec;
                    softmax_with_sink(&mut scores, sinks.logits[q_head]);
                }
                None => softmax(&mut scores),
            }
            let head_out = &mut concat[q_head * head_dim..(q_head + 1) * head_dim];
            for (offset, key_position) in (start..=position).enumerate() {
                let v_slice = &value_of(key_position)[kv_head * head_dim..(kv_head + 1) * head_dim];
                let weight = scores[offset];
                for (acc, v) in head_out.iter_mut().zip(v_slice) {
                    *acc += weight * v;
                }
            }
            if tap.is_some() {
                kept.push(scores);
            }
        }

        if let Some(GateCall { spec, weight }) = &call.gate {
            // Exhaustive on the judged semantics: a new variant must
            // be implemented here before it can execute.
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let gate_values = match spec.source {
                GateSource::AttentionInput => {
                    matvec(weight.as_f32()?, q_rows, call.hidden, gate_input)
                }
                // The gate rows live inside the query projection, so the
                // "gate weight" IS that projection and the gate is its
                // per-head second half. Recomputed from the same input
                // the query half was projected from — the reference
                // backend pays a second matvec rather than threading a
                // value through the call, which keeps this readable
                // beside the HF source it transcribes.
                GateSource::FusedQueryProjection => {
                    let full = matvec(weight.as_f32()?, q_rows * 2, call.hidden, gate_input);
                    let mut gate = gather_fused_half_mutated(
                        &full,
                        call.num_q_heads,
                        head_dim,
                        FusedHalf::Gate,
                        gate_mutation,
                    );
                    // The gate slice sees NEITHER the query norm nor the
                    // rotary — it is not a query. Both are mutations here
                    // rather than absent code, so a refactor that starts
                    // feeding the gate through either is caught by a
                    // number instead of by review.
                    if gate_mutation == GateMutation::GateGetsQNorm {
                        if let Some(qk) = &call.qk_norm {
                            for head in gate.chunks_exact_mut(head_dim) {
                                let normed = norm(
                                    NormType::RmsNorm,
                                    head,
                                    qk.q_weight,
                                    qk.weight_offset,
                                    call.qk_norm_eps,
                                );
                                head.copy_from_slice(&normed);
                            }
                        }
                    }
                    if gate_mutation == GateMutation::GateGetsRoPe {
                        for head in gate.chunks_exact_mut(head_dim) {
                            if let Some(theta) = call.position.rope_theta() {
                                rope_rotate(head, position, theta);
                            }
                        }
                    }
                    gate
                }
            };
            let activate_gate = |g: f32| match gate_mutation {
                // `silu(g)` is what `output_gate_type: "swish"` would mean
                // if it owned this gate. HF computes `sigmoid(g)`.
                GateMutation::SiluGate => g * sigmoid(g),
                _ => sigmoid(g),
            };
            // V3-HEAD-OBS-1: the records, with the activated gate the
            // multiply below will apply, before it applies.
            if let Some(tap) = tap.as_deref_mut() {
                let activated: Vec<f32> = gate_values.iter().map(|g| activate_gate(*g)).collect();
                super::super::observe::fire_head_records(
                    tap,
                    position,
                    call.num_q_heads,
                    call.num_kv_heads,
                    head_dim,
                    start,
                    call.sinks.is_some(),
                    &concat,
                    &kept,
                    Some(&activated),
                    query,
                    key_of,
                    value_of,
                );
                fired = true;
            }
            // V3-INTERVENE-2: each head's `ctx_h`, mutated in place if
            // declared — after the uninintervened head record above (J3),
            // before the gate multiply and `o_proj` below (J1).
            if let Some(hi) = head_intervene.as_deref_mut() {
                for (head, ctx_h) in concat.chunks_exact_mut(head_dim).enumerate() {
                    hi(head, ctx_h);
                }
            }
            if gate_mutation != GateMutation::NoGate
                && gate_mutation != GateMutation::GateAfterOProj
            {
                for (c, g) in concat.iter_mut().zip(&gate_values) {
                    *c *= activate_gate(*g);
                }
            }
            if gate_mutation == GateMutation::GateAfterOProj {
                let mut out = matvec(call.w_o.as_f32()?, call.hidden, q_rows, &concat);
                for (o, g) in out.iter_mut().zip(&gate_values) {
                    *o *= activate_gate(*g);
                }
                if let Some(o) = call.bias.as_ref().and_then(|bias| bias.o) {
                    add_in_place(&mut out, o);
                }
                return Ok(out);
            }
        }

        // V3-HEAD-OBS-1: a plan without a gate fires its records here,
        // with no gate slice; a gated plan fired them above.
        if let (Some(tap), false) = (tap, fired) {
            super::super::observe::fire_head_records(
                tap,
                position,
                call.num_q_heads,
                call.num_kv_heads,
                head_dim,
                start,
                call.sinks.is_some(),
                &concat,
                &kept,
                None,
                query,
                key_of,
                value_of,
            );
        }
        // V3-INTERVENE-2: a plan without a gate applies its head
        // intervention here; a gated plan applied it above regardless of
        // whether observation was armed — gated on the plan's OWN gate,
        // not on `fired`, which tracks the tap and stays false without one.
        if call.gate.is_none() {
            if let Some(hi) = head_intervene {
                for (head, ctx_h) in concat.chunks_exact_mut(head_dim).enumerate() {
                    hi(head, ctx_h);
                }
            }
        }

        let mut out = matvec(call.w_o.as_f32()?, call.hidden, q_rows, &concat);
        if let Some(o) = call.bias.as_ref().and_then(|bias| bias.o) {
            add_in_place(&mut out, o);
        }
        Ok(out)
    }

    /// The decode step with an optional per-head tap and an optional
    /// per-head intervention; `attention_step` is this with `None, None`,
    /// so the observed step IS the step.
    pub(super) fn attention_step_tapped(
        step: AttentionStepCall<'_>,
        tap: Option<&mut dyn FnMut(AttentionHeadRecord<'_>)>,
        head_intervene: Option<&mut super::super::backend::HeadIntervene<'_>>,
    ) -> Result<AttentionStepOut, VindexError> {
        let call = &step.op;
        let pre = &call.inputs[0];
        // PREC-DIAG-1 E0a: with a tap armed, keep the pre-rotation Q/K
        // and attach them to each head record. Unarmed, nothing is kept.
        let mut unpositioned = tap.is_some().then(UnpositionedQk::default);
        let (q, k, v) = Self::project_position_keeping(
            call,
            step.position,
            pre,
            GateMutation::None,
            unpositioned.as_mut(),
        )?;
        let output =
            UnpositionedQk::run_attached(unpositioned.as_ref(), call.head_dim, tap, |tap| {
                Self::attend_position_tapped(
                    call,
                    step.position,
                    &q,
                    &|p| {
                        if p == step.position {
                            k.as_slice()
                        } else {
                            step.rows().key(p)
                        }
                    },
                    &|p| {
                        if p == step.position {
                            v.as_slice()
                        } else {
                            step.rows().value(p)
                        }
                    },
                    pre,
                    GateMutation::None,
                    tap,
                    head_intervene,
                )
            })?;
        Ok(AttentionStepOut {
            key: k,
            value: v,
            output,
        })
    }
}
