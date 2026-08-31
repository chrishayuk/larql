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

use std::fmt;

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
