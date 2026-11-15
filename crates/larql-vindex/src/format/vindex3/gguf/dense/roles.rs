//! Operand roles, read off the operation plan.
//!
//! The plan already decided what every tensor *is* — `3.self_attn.q_proj`
//! is layer 3's query projection because the plan bound it there, not
//! because of its spelling. This module only re-addresses those decisions
//! as [`DenseRole`]s and carries the one number a role's lowering needs:
//! the norm weight offset the plan's norm op declares.

use super::target::DenseRole;
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerAttention, LayerFfn, OperandRef};

/// One plan operand, addressed for the lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleAssignment {
    pub object: String,
    pub tensor: String,
    pub role: DenseRole,
    pub layer: Option<usize>,
    /// The weight offset the plan's norm op declares (Gemma's `1 + w`);
    /// `0.0` for every non-norm operand.
    pub weight_offset: f32,
}

/// Why the plan cannot be addressed as a dense softmax stack.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleRefusal(pub String);

impl std::fmt::Display for RoleRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn assign(
    out: &mut Vec<RoleAssignment>,
    op: &OperandRef,
    role: DenseRole,
    layer: Option<usize>,
    weight_offset: f32,
) {
    out.push(RoleAssignment {
        object: op.object.clone(),
        tensor: op.tensor.clone(),
        role,
        layer,
        weight_offset,
    });
}

/// Every operand of a dense softmax plan, with its role. A layer operator
/// or FFN this lowering has no roles for is refused by name.
pub fn assign_roles(plan: &ComponentOpPlan) -> Result<Vec<RoleAssignment>, RoleRefusal> {
    let mut out = Vec::new();
    let embedding = plan.embedding.as_ref().map(|e| &e.table);
    if let Some(table) = embedding {
        assign(&mut out, table, DenseRole::Embedding, None, 0.0);
    }
    for layer in &plan.layers {
        let l = Some(layer.layer);
        for (norm, role) in [
            (&layer.pre_attention_norm, DenseRole::PreAttentionNorm),
            (&layer.post_attention_norm, DenseRole::PostAttentionNorm),
            (&layer.pre_ffn_norm, DenseRole::PreFfnNorm),
            (&layer.post_ffn_norm, DenseRole::PostFfnNorm),
        ] {
            if let Some(n) = norm {
                assign(&mut out, &n.weight, role, l, n.weight_offset);
            }
        }
        let LayerAttention::Softmax(a) = &layer.attention else {
            return Err(RoleRefusal(format!(
                "layer {} runs {} — the dense GGUF lowering binds softmax attention only",
                layer.layer,
                layer.attention.declared_name()
            )));
        };
        assign(&mut out, &a.q, DenseRole::Query, l, 0.0);
        assign(&mut out, &a.k, DenseRole::Key, l, 0.0);
        assign(&mut out, &a.v, DenseRole::Value, l, 0.0);
        assign(&mut out, &a.o, DenseRole::AttentionOutput, l, 0.0);
        if let Some(qk) = &a.qk_norm {
            assign(&mut out, &qk.q, DenseRole::QueryNorm, l, qk.weight_offset);
            assign(&mut out, &qk.k, DenseRole::KeyNorm, l, qk.weight_offset);
        }
        match &layer.ffn {
            Some(LayerFfn::Dense(f)) => {
                if let Some(gate) = &f.gate {
                    assign(&mut out, gate, DenseRole::FfnGate, l, 0.0);
                }
                assign(&mut out, &f.up, DenseRole::FfnUp, l, 0.0);
                assign(&mut out, &f.down, DenseRole::FfnDown, l, 0.0);
            }
            Some(_) => {
                return Err(RoleRefusal(format!(
                    "layer {} has a routed FFN — the dense GGUF lowering binds dense FFNs only",
                    layer.layer
                )))
            }
            None => {
                return Err(RoleRefusal(format!(
                    "layer {} has no FFN — a dense decoder layer carries one",
                    layer.layer
                )))
            }
        }
    }
    if let Some(n) = &plan.final_norm {
        assign(
            &mut out,
            &n.weight,
            DenseRole::FinalNorm,
            None,
            n.weight_offset,
        );
    }
    // A tied head reads the embedding operand itself; it is the same
    // bytes, and the runtime ties when no `output.weight` is present.
    if let Some(o) = &plan.output {
        if embedding != Some(&o.projection) {
            assign(&mut out, &o.projection, DenseRole::OutputHead, None, 0.0);
        }
    }
    Ok(out)
}
