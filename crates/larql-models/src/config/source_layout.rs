//! How a checkpoint *stores* its projections, when that differs from the
//! canonical `[out, in]` row-major layout every executor reads.
//!
//! Two storage conventions exist in the wild that the generic operations
//! do not read directly:
//!
//! - **input-major** projections (`transformers.Conv1D`, GPT-2's lineage):
//!   the weight is stored `[in, out]`, the transpose of `nn.Linear`.
//! - **fused** projections: one stored tensor holds several logical
//!   projections side by side along the output axis (GPT-2's `c_attn` is
//!   `q | k | v`), with its bias fused the same way.
//!
//! Both are *storage* facts, not semantics, and they are normalised once —
//! at the inventory boundary, recorded tensor by tensor as a
//! [`TensorDerivation`](crate::inventory::TensorDerivation) — so that every
//! consumer downstream (role classification, operand closure, encode, every
//! backend) sees canonical operands and no executor grows a transposed or
//! fused code path. The encoder materialises each derived tensor by the
//! recorded transform; it is a byte permutation, exact for every dtype.
//!
//! A family declares its layout through
//! [`TensorKeys::source_layout`](super::TensorKeys::source_layout); the
//! default is the canonical layout, which derives nothing.

/// One fused projection: a stored module whose output axis concatenates
/// `parts.len()` equal-width logical projections, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedProjection {
    /// Layer-relative module path of the stored tensor (`attn.c_attn`);
    /// its `.weight` and `.bias` are both split.
    pub module: &'static str,
    /// Layer-relative module paths the parts are written under, in
    /// output-axis order (`self_attn.q_proj`, `self_attn.k_proj`, …).
    pub parts: &'static [&'static str],
}

/// Tensors that ship in the checkpoint but are not parameters of the
/// forward pass: registered buffers the reference recomputes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonParameterBuffer {
    /// Layer-relative tensor name (`attn.bias`).
    pub suffix: &'static str,
    /// Why no operation reads it — printed in the inventory and the plan.
    pub reason: &'static str,
}

/// A family's storage layout. The default derives nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SourceLayout {
    /// Layer-relative module paths whose 2-D `.weight` is stored `[in,
    /// out]` (input-major). A fused module listed here is split first and
    /// each part transposed.
    pub input_major: &'static [&'static str],
    /// Fused projections, split into their parts along the output axis.
    pub fused: &'static [FusedProjection],
    /// Registered buffers excluded from the operand estate, each with the
    /// reason it is not an operand. Recorded, never silently dropped.
    pub buffers: &'static [NonParameterBuffer],
}

impl SourceLayout {
    /// Whether this layout derives or excludes anything at all.
    pub fn is_canonical(&self) -> bool {
        self.input_major.is_empty() && self.fused.is_empty() && self.buffers.is_empty()
    }
}
