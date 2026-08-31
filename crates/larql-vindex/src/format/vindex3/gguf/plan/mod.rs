//! What will be lowered, decided before anything is written.
//!
//! Three layers, each answering a different question:
//!
//! ```text
//! preflight   should this target exist for this artifact?
//! planner     is this particular lowering semantically legal?
//! writer      can these bytes be emitted?
//! ```
//!
//! The planner's own invariant — enforced in the constructor, not
//! documented and hoped for:
//!
//! > **A block-quantised representation may receive layout transforms
//! > only.** A value transform on NVFP4 would require decoding, applying
//! > arithmetic, and re-quantising — a new representation wearing the
//! > measured one's name.
//!
//! That is why the two transform kinds are separate types rather than
//! one enum with a comment. `-exp(log decay)` genuinely changes numbers;
//! the V-head permutation and the ABI repack do not. Collapsing them
//! under "lossless lowering" is precisely the refactor that would later
//! let someone put `ApplyWeightOffset` on a measured projection slab
//! without noticing, so the type system refuses it instead.

use std::fmt;

/// Moves bytes. Safe on a quantised representation provided every
/// permutation boundary lands on a quantisation-group boundary, which
/// [`super::preflight`] establishes before planning begins.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutTransform {
    /// V heads from grouped to tiled order along the output axis. On a
    /// fused operand only the V slice moves — Q and K stay put, and
    /// getting that wrong is a valid permutation applied to the wrong
    /// rows, which no shape check would catch.
    ReorderVRows {
        key_heads: usize,
        v_per_k: usize,
        head_dim: usize,
        /// Rows before the V region — `0` when the operand is V-only.
        v_offset_rows: usize,
    },
    /// The same reorder along the input axis, permuting whole
    /// quantisation groups rather than elements.
    ReorderVColumnsByGroups {
        key_heads: usize,
        v_per_k: usize,
        /// Groups per head — `head_dim / group`, and an integer by the
        /// preflight's invariant.
        groups_per_head: usize,
    },
    /// `[channels, 1, kernel]` → `[channels, kernel]`.
    SqueezeSingletonAxis { axis: usize },
}

/// Changes numbers. Legal only on a representation that stores numbers
/// directly.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueTransform {
    /// `A_log` → `-exp(A_log)`. Legitimate because the operand declares
    /// the `log decay` role, never because the tensor is called `A_log`.
    MaterializeLogDecay,
    /// Fold the norm's declared offset into the stored weight. The value
    /// comes from the graph; a literal here would be the family
    /// assumption this whole boundary exists to remove.
    ApplyWeightOffset(f32),
}

/// How the source stores this operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationKind {
    F32,
    Bf16,
    /// Block-quantised. Layout transforms only.
    Nvfp4,
}

impl RepresentationKind {
    /// Whether arithmetic can be applied without decoding first.
    pub fn holds_numbers_directly(self) -> bool {
        matches!(self, Self::F32 | Self::Bf16)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    /// The one the constructor exists for.
    ValueTransformOnQuantised {
        target: String,
        representation: RepresentationKind,
        transform: ValueTransform,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ValueTransformOnQuantised {
                target,
                representation,
                transform,
            } => write!(
                f,
                "cannot plan `{target}`: {transform:?} changes values, and the source is \
                 {representation:?} — applying it would mean decoding, computing and \
                 re-quantising, which produces a representation nobody measured while \
                 keeping the measured one's name"
            ),
        }
    }
}

/// One tensor's complete lowering.
#[derive(Debug, Clone, PartialEq)]
pub struct LoweredTensorPlan {
    /// The operand's semantic address in the container.
    pub source: String,
    /// The name llama.cpp will bind.
    pub target_name: String,
    pub layout: Vec<LayoutTransform>,
    pub value: Vec<ValueTransform>,
    pub source_representation: RepresentationKind,
    pub target_type: u32,
    /// NVFP4's per-tensor scale, which GGML has no room for in-block.
    pub scale_tensor: Option<String>,
}

impl LoweredTensorPlan {
    /// The only way to build one. Refuses a value transform on a
    /// block-quantised source rather than trusting callers to notice.
    pub fn new(
        source: impl Into<String>,
        target_name: impl Into<String>,
        source_representation: RepresentationKind,
        target_type: u32,
        layout: Vec<LayoutTransform>,
        value: Vec<ValueTransform>,
        scale_tensor: Option<String>,
    ) -> Result<Self, PlanError> {
        let target_name = target_name.into();
        if !source_representation.holds_numbers_directly() {
            if let Some(t) = value.first() {
                return Err(PlanError::ValueTransformOnQuantised {
                    target: target_name,
                    representation: source_representation,
                    transform: t.clone(),
                });
            }
        }
        Ok(Self {
            source: source.into(),
            target_name,
            layout,
            value,
            source_representation,
            target_type,
            scale_tensor,
        })
    }
}

/// Semantic role → the name llama.cpp binds.
///
/// A table rather than string-building, because three of these are
/// hazards that fail silently rather than loudly:
///
/// - `attn_q.weight` is the **fused** Q and output-gate projection at
///   double Q width. Splitting the gate into its own tensor produces a
///   file llama.cpp loads and then misreads.
/// - `ssm_a` carries **no `.weight` suffix**.
/// - `ssm_dt` is a **`.bias`**.
///
/// None of the three would error. They would fail to bind, or bind the
/// wrong thing.
pub fn qwen35_tensor_name(role: &str, layer: usize) -> Option<String> {
    let suffix = match role {
        // Full attention. `query` is Q + output gate, fused.
        "query" => "attn_q.weight",
        "key" => "attn_k.weight",
        "value" => "attn_v.weight",
        "output" => "attn_output.weight",
        // Gated DeltaNet.
        "fused recurrent q|k|v" => "attn_qkv.weight",
        "output-gate projection" => "attn_gate.weight",
        "causal conv over q|k|v" => "ssm_conv1d.weight",
        "log decay" => "ssm_a",
        "timestep bias" => "ssm_dt.bias",
        "decay projection" => "ssm_alpha.weight",
        "write-strength projection" => "ssm_beta.weight",
        "gated norm" => "ssm_norm.weight",
        "output projection" => "ssm_out.weight",
        // Dense FFN.
        "ffn gate" => "ffn_gate.weight",
        "ffn up" => "ffn_up.weight",
        "ffn down" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{suffix}"))
}

/// The three model-scope surfaces, which carry no layer index.
pub fn qwen35_global_name(role: &str) -> Option<&'static str> {
    match role {
        "embedding" => Some("token_embd.weight"),
        "final norm" => Some("output_norm.weight"),
        "output head" => Some("output.weight"),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
