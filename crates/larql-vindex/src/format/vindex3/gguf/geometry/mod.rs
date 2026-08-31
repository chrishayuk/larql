//! Do the metadata table and the tensor planner agree about the model?
//!
//! Coverage proves every tensor has a target. Geometry proves the target
//! is the *right shape* — and it does so from two authorities that must
//! never consult each other:
//!
//! ```text
//! planned    physical source tensor + its layout transforms
//! expected   graph-derived target metadata + the target ABI
//! ```
//!
//! Both modules can be individually correct and still disagree about the
//! model. That is the loophole this closes, and nothing else does: a
//! `q_proj` at ordinary width is perfectly plausible, passes coverage,
//! maps to a unique name, and is wrong.
//!
//! **Geometry is representation-independent.** `TargetGeometry` carries
//! a name and dimensions and nothing about encoding, so the BF16 and
//! NVFP4 selections of one model can be compared directly. The rule that
//! follows is sharper than shape agreement alone:
//!
//! > Representation choice may change target encoding and auxiliary
//! > tensors; it must not change the model's semantic geometry.
//!
//! `.scale` siblings are target-ABI auxiliaries, not model geometry, so
//! they never enter the comparison.
//!
//! **Where the two authorities meet.** [`qwen35_expected_shape`] derives
//! a role's target shape from [`ModelGeometry`] — graph facts and the
//! target ABI, never a tensor. The walk calls [`check_target`] on every
//! plan, so the comparison runs on the real container rather than only
//! in a unit test that hands `reconcile` two vectors it typed itself.
//! Until it was wired, the hero's matching digests proved
//! `planned == planned` across representations, which is real, and not
//! `planned == expected`, which is the loophole.

use std::fmt;

use crate::format::vindex3::graph::surface::ExecutionSurface;
use larql_models::config::GateSource;

/// The graph facts the target's shape rules consume.
///
/// Every field is the same authority the metadata table reads. None is
/// a tensor dimension: reading `q_heads x head_dim` off `q_proj`'s
/// height would make the expectation agree with the plan by
/// construction, and the comparison would be worth nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelGeometry {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// The query projection carries the output gate in its second half.
    /// From `attention.output_gate.source`, so a model without the gate
    /// expects an ordinary-width Q and is not refused for having one.
    pub query_carries_gate: bool,
    pub key_heads: usize,
    pub key_head_dim: usize,
    pub value_heads: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
}

impl ModelGeometry {
    /// Read the facts off a component's execution surface. `hidden_size`
    /// lives on the component rather than the surface, so it is passed.
    pub fn from_surface(
        surface: &ExecutionSurface,
        hidden_size: usize,
    ) -> Result<Self, GeometryError> {
        let attn = surface
            .attention
            .as_ref()
            .ok_or(GeometryError::MissingFact("execution.attention"))?;
        let ffn = surface
            .ffn
            .as_ref()
            .ok_or(GeometryError::MissingFact("execution.ffn"))?;
        let la = surface
            .linear_attention
            .as_ref()
            .ok_or(GeometryError::MissingFact("execution.linear_attention"))?;
        let head = surface
            .head
            .as_ref()
            .ok_or(GeometryError::MissingFact("execution.head.vocab_size"))?;
        Ok(Self {
            hidden_size,
            vocab_size: head.vocab_size,
            intermediate_size: ffn.intermediate_size,
            q_heads: attn.num_q_heads,
            kv_heads: attn.num_kv_heads,
            head_dim: attn.head_dim,
            query_carries_gate: attn
                .output_gate
                .as_ref()
                .is_some_and(|g| g.source == GateSource::FusedQueryProjection),
            key_heads: la.key_heads,
            key_head_dim: la.key_head_dim,
            value_heads: la.value_heads,
            value_head_dim: la.value_head_dim,
            conv_kernel: la.conv_kernel,
        })
    }

    fn qkv_channels(&self) -> u64 {
        (2 * self.key_heads * self.key_head_dim + self.value_heads * self.value_head_dim) as u64
    }

    fn value_width(&self) -> u64 {
        (self.value_heads * self.value_head_dim) as u64
    }
}

/// One target's expected shape and the facts that produced it.
///
/// Row-major, as the segment header stores it: `[out, in]`. The writer
/// reverses into GGUF's `ne` order at emission; the comparison happens
/// on this side of that flip so both derivations speak one convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expected {
    pub dims: Vec<u64>,
    pub from: &'static str,
}

/// What the target ABI expects a role's tensor to be.
///
/// The conv expectation is the *post-squeeze* shape: llama.cpp binds
/// `ssm_conv1d` as `[channels, kernel]`, so a plan that never squeezed
/// its singleton axis disagrees here, which is the point.
pub fn qwen35_expected_shape(role: &str, m: &ModelGeometry) -> Option<Expected> {
    let h = m.hidden_size as u64;
    let (dims, from) = match role {
        "embedding" | "output head" => (
            vec![m.vocab_size as u64, h],
            "head.vocab_size x hidden_size",
        ),
        "final norm" | "input layer norm" | "post-attention layer norm" => (vec![h], "hidden_size"),
        "query" => {
            let factor = if m.query_carries_gate { 2 } else { 1 };
            (
                vec![factor * (m.q_heads * m.head_dim) as u64, h],
                "(1 + fused output gate) x q_heads x head_dim, x hidden_size",
            )
        }
        "key" | "value" => (
            vec![(m.kv_heads * m.head_dim) as u64, h],
            "kv_heads x head_dim, x hidden_size",
        ),
        "output" => (
            vec![h, (m.q_heads * m.head_dim) as u64],
            "hidden_size x q_heads x head_dim",
        ),
        "attention q norm" | "attention k norm" => (vec![m.head_dim as u64], "head_dim"),
        "ffn gate" | "ffn up" => (
            vec![m.intermediate_size as u64, h],
            "ffn.intermediate_size x hidden_size",
        ),
        "ffn down" => (
            vec![h, m.intermediate_size as u64],
            "hidden_size x ffn.intermediate_size",
        ),
        "fused recurrent q|k|v" => (
            vec![m.qkv_channels(), h],
            "2 x key_heads x key_head_dim + value_heads x value_head_dim, x hidden_size",
        ),
        "causal conv over q|k|v" => (
            vec![m.qkv_channels(), m.conv_kernel as u64],
            "qkv channels x conv_kernel, singleton axis squeezed",
        ),
        "output-gate projection" => (
            vec![m.value_width(), h],
            "value_heads x value_head_dim, x hidden_size",
        ),
        "decay projection" | "write-strength projection" => {
            (vec![m.value_heads as u64, h], "value_heads x hidden_size")
        }
        "log decay" | "timestep bias" => (vec![m.value_heads as u64], "value_heads"),
        "gated norm" => (vec![m.value_head_dim as u64], "value_head_dim"),
        "output projection" => (
            vec![h, m.value_width()],
            "hidden_size x value_heads x value_head_dim",
        ),
        _ => return None,
    };
    Some(Expected { dims, from })
}

/// Reconcile one plan against the target's expectation for its role.
///
/// The fused-Q rule runs first, because when it is what went wrong the
/// dimensional refusal says why an ordinary-width Q is dangerous, where
/// a bare shape mismatch would not.
pub fn check_target(
    target: &str,
    role: &str,
    planned: &[u64],
    m: &ModelGeometry,
) -> Result<Expected, GeometryError> {
    let expected = qwen35_expected_shape(role, m).ok_or(GeometryError::NoExpectation {
        target: target.to_string(),
        role: role.to_string(),
    })?;
    if role == "query" && m.query_carries_gate {
        if let Some(&rows) = planned.first() {
            expect_fused_query_width(target, rows, m.q_heads, m.head_dim)?;
        }
    }
    reconcile(target, planned, &expected.dims, expected.from)?;
    Ok(expected)
}

/// A target tensor's semantic shape. No encoding, deliberately.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TargetGeometry {
    pub name: String,
    pub dims: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryError {
    /// The two authorities disagree. The message shows both derivations,
    /// because "shape mismatch" sends the reader to the wrong module.
    Disagreement {
        target: String,
        planned: Vec<u64>,
        expected: Vec<u64>,
        expected_from: String,
    },
    /// A conv axis the target may not remove.
    NonSingletonSqueeze {
        target: String,
        dims: Vec<u64>,
        axis: usize,
    },
    /// The fused Q convention, violated dimensionally rather than by name.
    UnfusedQueryWidth {
        target: String,
        found: u64,
        expected: u64,
        q_heads: usize,
        head_dim: usize,
    },
    /// A role the name table maps but the shape table does not. The two
    /// tables must cover the same roles, or a tensor can reach the file
    /// with a name and no checked geometry.
    NoExpectation { target: String, role: String },
    /// A graph fact the expectation needs and the surface lacks.
    MissingFact(&'static str),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disagreement {
                target,
                planned,
                expected,
                expected_from,
            } => write!(
                f,
                "geometry: `{target}` planned {planned:?} but the target expects {expected:?} \
                 ({expected_from}) — the metadata table and the tensor planner disagree about \
                 the model, and each is self-consistent"
            ),
            Self::NonSingletonSqueeze { target, dims, axis } => write!(
                f,
                "geometry: `{target}` has {dims:?} and axis {axis} is not a singleton — the \
                 target lowering may remove only a singleton convolution axis, never collapse \
                 real channels"
            ),
            Self::UnfusedQueryWidth {
                target,
                found,
                expected,
                q_heads,
                head_dim,
            } => write!(
                f,
                "geometry: `{target}` is {found} wide but qwen35 fuses the output gate into the \
                 query projection, so it must be 2 x {q_heads} x {head_dim} = {expected}. An \
                 ordinary-width Q is a plausible tensor and the wrong one"
            ),
            Self::NoExpectation { target, role } => write!(
                f,
                "geometry: `{target}` (role `{role}`) has a target name but no expected shape — \
                 the name table and the shape table cover different roles, so this tensor would \
                 reach the file unchecked"
            ),
            Self::MissingFact(fact) => write!(
                f,
                "geometry: cannot derive the target's expected shapes because the graph lacks \
                 `{fact}` — the expectation must come from the graph, or it is the plan \
                 checking itself"
            ),
        }
    }
}

/// `[c, 1, k]` → `[c, k]`. Refuses anything else.
pub fn squeeze_singleton(
    target: &str,
    dims: &[u64],
    axis: usize,
) -> Result<Vec<u64>, GeometryError> {
    if dims.get(axis) != Some(&1) {
        return Err(GeometryError::NonSingletonSqueeze {
            target: target.to_string(),
            dims: dims.to_vec(),
            axis,
        });
    }
    let mut out = dims.to_vec();
    out.remove(axis);
    Ok(out)
}

/// qwen35 stores Q and the output gate in one tensor at double width.
pub fn expect_fused_query_width(
    target: &str,
    found: u64,
    q_heads: usize,
    head_dim: usize,
) -> Result<(), GeometryError> {
    let expected = 2 * q_heads as u64 * head_dim as u64;
    if found != expected {
        return Err(GeometryError::UnfusedQueryWidth {
            target: target.to_string(),
            found,
            expected,
            q_heads,
            head_dim,
        });
    }
    Ok(())
}

/// Compare one tensor's two derivations.
pub fn reconcile(
    target: &str,
    planned: &[u64],
    expected: &[u64],
    expected_from: &str,
) -> Result<(), GeometryError> {
    if planned != expected {
        return Err(GeometryError::Disagreement {
            target: target.to_string(),
            planned: planned.to_vec(),
            expected: expected.to_vec(),
            expected_from: expected_from.to_string(),
        });
    }
    Ok(())
}

/// A stable summary of a walk's semantic geometry — target names and
/// dimensions only.
///
/// Two selections of one model must produce the same digest. If it moves
/// when someone edits NVFP4 lowering, they have changed the model rather
/// than its representation, and that is the signal worth having.
pub fn semantic_digest(mut geometry: Vec<TargetGeometry>) -> String {
    geometry.sort();
    let mut acc: u64 = 0xcbf29ce484222325;
    for g in &geometry {
        for byte in g.name.as_bytes() {
            acc ^= *byte as u64;
            acc = acc.wrapping_mul(0x100000001b3);
        }
        for d in &g.dims {
            acc ^= *d;
            acc = acc.wrapping_mul(0x100000001b3);
        }
    }
    format!("{acc:016x}")
}

#[cfg(test)]
mod tests;
