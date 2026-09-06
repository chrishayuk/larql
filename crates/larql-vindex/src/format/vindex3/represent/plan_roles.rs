//! Semantic roles read from the operation plan, not from tensor names.
//!
//! [`super::policy::classify_in`] decides a tensor's role by substring —
//! `_proj.`, `self_attn.`, `attention.`, `mlp.`. That works for the
//! spellings softmax families happen to use and fails silently for every
//! other operator, because a name that matches nothing classifies
//! [`Role::Unknown`] and is carried verbatim.
//!
//! Measured on Qwen3.8-27B: its Gated DeltaNet operands are spelled
//! `linear_attn.in_proj_qkv`, `in_proj_a`, `in_proj_b`, `in_proj_z` —
//! `_proj_qkv.` is not `_proj.`, so none matched, and **4,050,124,800
//! weights, 16.6% of the decoder stack, stayed at BF16**. Only
//! `out_proj` matched, which is why it was the single recurrence column
//! reading 4.50 in `vindex precision --matrix`. The compiled decoder
//! measured 6.4138 bits/weight and looked like a precision decision.
//!
//! It was not one. `represent/mod.rs` already states the rule this
//! broke: *"the embedding is BF16 because the policy protects it" and
//! "the embedding is BF16 because nobody looked" are different facts and
//! a report that cannot tell them apart is useless.* Nobody had looked,
//! and the report said `unknown`.
//!
//! So the plan answers first. It has bound every operand to the role its
//! operator computes with — nine for Gated DeltaNet, fifteen for KDA,
//! five for MLA, eight for Mamba2 — and those bindings are the
//! container's own judgement, not a guess about a filename. Names remain
//! the fallback for anything the plan does not cover (object-level roles,
//! components with no plan, checkpoints whose plan does not build).
//!
//! **This module classifies. It does not decide precision.** Which
//! representation a role receives is [`super::policy::RolePolicy`]'s
//! answer, stated per role, and the two must not be collapsed — that
//! collapse is what made a classification accident read as a protection.

use std::collections::BTreeMap;
use std::path::Path;

use super::policy::Role;
use crate::format::vindex3::inspect::SystemInspection;
use crate::format::vindex3::opplan::{
    plan_component_ops, ComponentOpPlan, LayerAttention, LayerFfn, OperandRef,
};

/// Every tensor the plan binds, keyed `(object, tensor)`, with the role
/// its operator gives it.
pub type PlanRoles = BTreeMap<(String, String), Role>;

/// Read the plan for every component and collect its operand roles.
///
/// Best-effort by construction: a component whose plan does not build
/// contributes nothing and its tensors fall back to name classification,
/// which is exactly the behaviour before this module existed. A
/// representation compile must not fail because a plan does not.
pub fn plan_roles(root: &Path, inspection: &SystemInspection) -> PlanRoles {
    let mut roles = PlanRoles::new();
    for component in &inspection.graph.components {
        let Ok(outcome) = plan_component_ops(inspection, root, &component.id) else {
            continue;
        };
        let Some(plan) = outcome.plan else { continue };
        collect(&plan, &mut roles);
    }
    roles
}

fn put(roles: &mut PlanRoles, role: Role, op: &OperandRef) {
    roles.insert((op.object.clone(), op.tensor.clone()), role);
}

fn collect(plan: &ComponentOpPlan, roles: &mut PlanRoles) {
    for layer in &plan.layers {
        collect_attention(&layer.attention, roles);
        match &layer.ffn {
            Some(LayerFfn::Dense(f)) => {
                if let Some(g) = &f.gate {
                    put(roles, Role::DecoderLinear, g);
                }
                put(roles, Role::DecoderLinear, &f.up);
                put(roles, Role::DecoderLinear, &f.down);
            }
            // Routed and hybrid FFNs carve their experts into their own
            // object, which the expert-bank path already classifies; the
            // router is named there too. Nothing to add from here.
            Some(LayerFfn::Routed(_)) | Some(LayerFfn::Hybrid(_)) | None => {}
        }
    }
}

fn collect_attention(attention: &LayerAttention, roles: &mut PlanRoles) {
    match attention {
        LayerAttention::Softmax(a) => {
            for op in [&a.q, &a.k, &a.v, &a.o] {
                put(roles, Role::DecoderLinear, op);
            }
            if let Some(g) = &a.output_gate {
                put(roles, Role::DecoderLinear, &g.projection);
            }
        }
        // A recurrence's bulk matmuls are ordinary linear work — the same
        // shape of operation a softmax layer's q/k/v/o are, at the same
        // scale. Its *control* projections are structurally different:
        // `in_proj_a` and `in_proj_b` emit the per-head decay and
        // write-strength that drive the state update rather than
        // contributing to one position's output.
        //
        // Two roles because they are two structurally different
        // operands, which is a classification fact. Whether that
        // difference warrants different precision was a separate
        // question. Q-BANK-1 measured a small real benefit that did not
        // justify its cost, so both compile by default — see
        // `Role::RecurrenceControl` for the numbers.
        // The roles stay distinct because naming the operand is what
        // made the question askable at all.
        LayerAttention::GatedDelta(g) => {
            for op in [&g.in_proj_qkv, &g.in_proj_z, &g.out_proj] {
                put(roles, Role::RecurrenceProjection, op);
            }
            for op in [&g.in_proj_a, &g.in_proj_b] {
                put(roles, Role::RecurrenceControl, op);
            }
            put(roles, Role::SmallVector, &g.conv1d);
            put(roles, Role::SmallVector, &g.a_log);
            put(roles, Role::SmallVector, &g.dt_bias);
            put(roles, Role::Norm, &g.norm);
        }
        LayerAttention::Kda(k) => {
            for op in [&k.q_proj, &k.k_proj, &k.v_proj, &k.out_proj] {
                put(roles, Role::RecurrenceProjection, op);
            }
            // The gate factorisations and the write-strength projection
            // are this operator's control path — narrow when low-rank,
            // and still the control path when Kimi-K3 declares the output
            // gate full-rank: the operand's ROLE is what it drives, not its
            // width.
            for op in [&k.f_a_proj, &k.f_b_proj, &k.b_proj] {
                put(roles, Role::RecurrenceControl, op);
            }
            for (_, op) in k.output_gate.operands() {
                put(roles, Role::RecurrenceControl, op);
            }
            for op in [&k.q_conv1d, &k.k_conv1d, &k.v_conv1d, &k.a_log, &k.dt_bias] {
                put(roles, Role::SmallVector, op);
            }
            put(roles, Role::Norm, &k.o_norm);
        }
        // MLA retains a per-position cache and is not a recurrence: its
        // operands are ordinary decoder linear work at an unusual width.
        LayerAttention::Mla(m) => {
            for op in [&m.q_proj, &m.kv_a_proj, &m.kv_b_proj, &m.out_proj] {
                put(roles, Role::DecoderLinear, op);
            }
            put(roles, Role::Norm, &m.kv_a_norm);
        }
        // Conv-QKV attends by softmax and keeps a per-position cache —
        // the conv sits on the fused projection's output, it does not
        // make the block a recurrence. So its two matmuls are ordinary
        // decoder linear work, and the depthwise kernel is the same
        // small-vector shape every other operator's conv is.
        LayerAttention::ConvQkv(c) => {
            for op in [&c.in_proj, &c.out_proj] {
                put(roles, Role::DecoderLinear, op);
            }
            put(roles, Role::SmallVector, &c.conv1d);
            if let Some(b) = &c.conv1d_bias {
                put(roles, Role::SmallVector, b);
            }
        }
        LayerAttention::Mamba2(m) => {
            for op in [&m.in_proj, &m.out_proj] {
                put(roles, Role::RecurrenceProjection, op);
            }
            for op in [&m.conv1d, &m.a_log, &m.d, &m.dt_bias] {
                put(roles, Role::SmallVector, op);
            }
            if let Some(b) = &m.conv1d_bias {
                put(roles, Role::SmallVector, b);
            }
            if let Some(n) = &m.gated_norm {
                put(roles, Role::Norm, &n.weight);
            }
        }
    }
}
