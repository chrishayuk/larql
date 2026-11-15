//! From segment tensors to lowered plans, in the reference's order.
//!
//! Every tensor of every included object must receive exactly one role
//! from the operation plan, and every role the target requires must be
//! received — a tensor left over or a role left empty is a different
//! model, so both refuse. Geometry is checked against the graph's own
//! declarations (the [`Programme`]), never against another tensor.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::programme::Programme;
use super::roles::RoleAssignment;
use super::target::{DenseRole, DenseTarget};
use crate::format::vindex3::gguf::emit::{TYPE_BF16, TYPE_F32};
use crate::format::vindex3::gguf::plan::{LoweredTensorPlan, RepresentationKind, ValueTransform};

pub const DTYPE_BF16: &str = "BF16";
pub const DTYPE_F32: &str = "F32";

/// One physical tensor of a selected representation.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceTensor {
    pub object: String,
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Where its payload lives: file, absolute offset, length.
    pub span: (PathBuf, u64, u64),
}

impl SourceTensor {
    /// The address `emit_gguf` opens a payload by.
    pub fn address(&self) -> String {
        format!("{}/{}", self.object, self.name)
    }
}

/// The graph facts the expected shapes derive from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeFacts<'a> {
    pub programme: &'a Programme,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

fn expected_shape(role: DenseRole, f: &ShapeFacts<'_>) -> Vec<u64> {
    let p = f.programme;
    let h = f.hidden_size as u64;
    let q = (p.num_q_heads * p.head_dim) as u64;
    let kv = (p.num_kv_heads * p.head_dim) as u64;
    let inter = p.intermediate_size as u64;
    match role {
        DenseRole::Embedding | DenseRole::OutputHead => vec![f.vocab_size as u64, h],
        DenseRole::FinalNorm
        | DenseRole::PreAttentionNorm
        | DenseRole::PostAttentionNorm
        | DenseRole::PreFfnNorm
        | DenseRole::PostFfnNorm => vec![h],
        DenseRole::QueryNorm | DenseRole::KeyNorm => vec![p.head_dim as u64],
        DenseRole::Query => vec![q, h],
        DenseRole::Key | DenseRole::Value => vec![kv, h],
        DenseRole::AttentionOutput => vec![h, q],
        DenseRole::FfnGate | DenseRole::FfnUp => vec![inter, h],
        DenseRole::FfnDown => vec![h, inter],
    }
}

/// Embedding first, then the layers in index order, then the final norm
/// and any untied head — the order the reference's sorted checkpoint
/// produces, so a reader diffing the two files sees the same sequence.
fn order_key(a: &RoleAssignment) -> (u8, usize) {
    match a.role {
        DenseRole::Embedding => (0, 0),
        DenseRole::FinalNorm => (2, 0),
        DenseRole::OutputHead => (3, 0),
        _ => (1, a.layer.unwrap_or(0)),
    }
}

/// The lowered plans, in emission order, or every refusal found.
pub fn plan_tensors(
    sources: &[SourceTensor],
    roles: &[RoleAssignment],
    target: &DenseTarget,
    facts: &ShapeFacts<'_>,
) -> Result<Vec<LoweredTensorPlan>, Vec<String>> {
    let mut refusals = Vec::new();
    let mut by_address: BTreeMap<(String, String), &RoleAssignment> = BTreeMap::new();
    for r in roles {
        if by_address
            .insert((r.object.clone(), r.tensor.clone()), r)
            .is_some()
        {
            refusals.push(format!("`{}/{}` carries two roles", r.object, r.tensor));
        }
    }
    let mut planned: Vec<((u8, usize), LoweredTensorPlan)> = Vec::new();
    let mut claimed = std::collections::BTreeSet::new();
    for s in sources {
        let Some(r) = by_address.get(&(s.object.clone(), s.name.clone())) else {
            refusals.push(format!(
                "`{}` has no role in the operation plan — exporting without it would drop \
                 part of the model",
                s.address()
            ));
            continue;
        };
        claimed.insert((s.object.clone(), s.name.clone()));
        let Some(name) = target.tensor_name(r.role, r.layer) else {
            refusals.push(format!(
                "`{}` is the {:?}, which {} has no tensor name for",
                s.address(),
                r.role,
                target.architecture
            ));
            continue;
        };
        let representation = match s.dtype.as_str() {
            DTYPE_BF16 => RepresentationKind::Bf16,
            DTYPE_F32 => RepresentationKind::F32,
            other => {
                refusals.push(format!(
                    "`{}` is stored {other}; the exporter writes BF16 and F32 sources only",
                    s.address()
                ));
                continue;
            }
        };
        let expected = expected_shape(r.role, facts);
        if s.shape != expected {
            refusals.push(format!(
                "`{}` ({:?}) has shape {:?}; the graph's geometry says {expected:?}",
                s.address(),
                r.role,
                s.shape
            ));
            continue;
        }
        // Norms and 1-D tensors leave as F32 (the reference's rule);
        // matrices keep the precision the container stores.
        let target_type = if s.shape.len() <= 1 || r.role.is_norm() {
            TYPE_F32
        } else if representation == RepresentationKind::Bf16 {
            TYPE_BF16
        } else {
            TYPE_F32
        };
        let value = if r.weight_offset != 0.0 {
            vec![ValueTransform::ApplyWeightOffset(r.weight_offset)]
        } else {
            vec![]
        };
        match LoweredTensorPlan::new(
            s.address(),
            name,
            representation,
            target_type,
            s.shape.clone(),
            vec![],
            value,
            None,
        ) {
            Ok(p) => planned.push((order_key(r), p)),
            Err(e) => refusals.push(e.to_string()),
        }
    }
    for r in roles {
        if !claimed.contains(&(r.object.clone(), r.tensor.clone())) {
            refusals.push(format!(
                "the plan binds `{}/{}` as the {:?}, and no selected segment holds it",
                r.object, r.tensor, r.role
            ));
        }
    }
    refusals.extend(missing_required(roles, target));
    if !refusals.is_empty() {
        return Err(refusals);
    }
    planned.sort_by_key(|(k, _)| *k);
    Ok(planned.into_iter().map(|(_, p)| p).collect())
}

/// Roles the runtime's loader needs and the plan did not bind.
fn missing_required(roles: &[RoleAssignment], target: &DenseTarget) -> Vec<String> {
    let has = |role: DenseRole, layer: Option<usize>| {
        roles.iter().any(|r| r.role == role && r.layer == layer)
    };
    let mut out = Vec::new();
    for role in [DenseRole::Embedding, DenseRole::FinalNorm] {
        if !has(role, None) {
            out.push(format!(
                "the plan binds no {role:?}, which every model needs"
            ));
        }
    }
    let layers: std::collections::BTreeSet<usize> = roles.iter().filter_map(|r| r.layer).collect();
    for layer in layers {
        for role in target.required_layer_roles {
            if !has(*role, Some(layer)) {
                out.push(format!(
                    "layer {layer} binds no {role:?}, which {} loads on every layer",
                    target.architecture
                ));
            }
        }
    }
    out
}
