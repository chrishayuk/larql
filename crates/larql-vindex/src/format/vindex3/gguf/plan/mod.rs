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

use super::geometry::{GeometryError, ModelGeometry};
use crate::format::vindex3::graph::surface::ExecutionSurface;

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

impl LayoutTransform {
    /// What this transform does to a shape.
    ///
    /// Both reorders permute within an axis and so preserve dims; only
    /// the squeeze changes rank, and only for a singleton. Keeping the
    /// effect here rather than in the writer means target geometry is
    /// derived from the source tensor plus its transforms, never from
    /// what the metadata expected — which is what makes the two
    /// authorities independent.
    pub fn apply_shape(&self, dims: &[u64]) -> Result<Vec<u64>, PlanError> {
        match self {
            Self::ReorderVRows { .. } | Self::ReorderVColumnsByGroups { .. } => Ok(dims.to_vec()),
            Self::SqueezeSingletonAxis { axis } => {
                if dims.get(*axis) != Some(&1) {
                    return Err(PlanError::NonSingletonSqueeze {
                        dims: dims.to_vec(),
                        axis: *axis,
                    });
                }
                let mut out = dims.to_vec();
                out.remove(*axis);
                Ok(out)
            }
        }
    }
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
    /// A squeeze of an axis that carries real channels.
    NonSingletonSqueeze { dims: Vec<u64>, axis: usize },
    /// V heads that do not group evenly under K heads have no tiled
    /// order, so the reorder is not definable.
    VHeadsDoNotGroup {
        value_heads: usize,
        key_heads: usize,
    },
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
            Self::VHeadsDoNotGroup {
                value_heads,
                key_heads,
            } => write!(
                f,
                "cannot plan: {value_heads} value heads do not group evenly under {key_heads} \
                 key heads, so the V-head reorder has no tiled order to produce"
            ),
            Self::NonSingletonSqueeze { dims, axis } => write!(
                f,
                "cannot plan: axis {axis} of {dims:?} is not a singleton — the target lowering \
                 may remove only a singleton convolution axis, never collapse real channels"
            ),
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
    /// The physical source shape, from the segment header.
    pub source_shape: Vec<u64>,
    /// Derived: source shape through the layout transforms. Value
    /// transforms never appear here, because changing numbers must not
    /// change geometry — asserted in the constructor.
    pub target_shape: Vec<u64>,
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
        source_shape: Vec<u64>,
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
        // Geometry is the source shape through the layout transforms,
        // in order. Value transforms are deliberately absent: they may
        // not move a dimension, and the type system keeps them out of
        // this fold rather than trusting a comment.
        let mut target_shape = source_shape.clone();
        for t in &layout {
            target_shape = t.apply_shape(&target_shape)?;
        }
        Ok(Self {
            source: source.into(),
            target_name,
            layout,
            value,
            source_representation,
            target_type,
            source_shape,
            target_shape,
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
        // Trunk norms. Both are per-layer across the WHOLE stack — the
        // real container carries 64 of each on a 64-layer model. The
        // second one's name says "attention" and its applicability does
        // not: keying it off layer kind would drop the 48 recurrent
        // layers' copies.
        "input layer norm" => "attn_norm.weight",
        "post-attention layer norm" => "post_attention_norm.weight",
        // Q/K norms, on attending layers only — 16 of each here. Their
        // offset comes from `attention.qk_norm_weight_offset`, a
        // separate authority from the trunk norms' `norm.pre`, so a
        // model may legitimately differ between them.
        "attention q norm" => "attn_q_norm.weight",
        "attention k norm" => "attn_k_norm.weight",
        // Dense FFN.
        "ffn gate" => "ffn_gate.weight",
        "ffn up" => "ffn_up.weight",
        "ffn down" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{suffix}"))
}

/// The weight offsets each norm family folds at lowering time.
///
/// Three separate authorities, read separately, because llama.cpp's
/// "+1 on every norm except the GDN's gated norm" is not an exception
/// written here — it falls out of the graph: `norm.pre` and
/// `norm.final_norm` declare `weight_offset: 1.0`, the q/k norms
/// declare their own via `attention.qk_norm_weight_offset`, and the
/// linear-attention surface declares no offset for its gated norm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormOffsets {
    pub trunk: f32,
    pub final_norm: f32,
    pub qk: f32,
}

/// Everything the transform table consumes: the model's geometry and
/// the declared norm offsets. All of it graph facts — none of it read
/// off a tensor, and none of it a family literal.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Lowering {
    pub model: ModelGeometry,
    pub offsets: NormOffsets,
}

impl Qwen35Lowering {
    pub fn from_surface(
        surface: &ExecutionSurface,
        hidden_size: usize,
    ) -> Result<Self, GeometryError> {
        let model = ModelGeometry::from_surface(surface, hidden_size)?;
        let attn = surface
            .attention
            .as_ref()
            .expect("ModelGeometry::from_surface already required attention");
        Ok(Self {
            model,
            offsets: NormOffsets {
                trunk: surface.norm.pre.weight_offset,
                final_norm: surface.norm.final_norm.weight_offset,
                qk: attn.qk_norm_weight_offset,
            },
        })
    }
}

/// The complete transform programme a role's tensor undergoes between
/// the container and the file. This is the target ABI's table, in one
/// place, so the emitter below it can execute plans without knowing a
/// single role name.
///
/// Two families of transform, both forced by llama.cpp's qwen35 graph:
///
/// - **V heads move from grouped to tiled order** so `ggml_repeat` can
///   broadcast K-head state across its value heads. The permutation
///   touches every tensor indexed by value head: the fused QKV's V rows
///   (at an offset past Q and K), the gate's rows, the per-head decay
///   and write-strength rows, the 1-D `log decay` and `timestep bias`
///   elements, the convolution's V channels, and the output
///   projection's input columns. Skipped entirely when `v_per_k == 1`,
///   where it is the identity.
/// - **Value arithmetic** the target stores pre-computed: `-exp` on the
///   log decay, and each norm family's declared weight offset folded
///   into the stored weight. An offset of exactly 0 attaches nothing —
///   the operation is the identity, and forcing a conversion to carry
///   it would change bytes for no semantic reason.
pub fn qwen35_transforms(
    role: &str,
    lowering: &Qwen35Lowering,
) -> Result<(Vec<LayoutTransform>, Vec<ValueTransform>), PlanError> {
    let m = &lowering.model;
    let o = &lowering.offsets;
    if m.key_heads == 0 || !m.value_heads.is_multiple_of(m.key_heads) {
        return Err(PlanError::VHeadsDoNotGroup {
            value_heads: m.value_heads,
            key_heads: m.key_heads,
        });
    }
    let v_per_k = m.value_heads / m.key_heads;
    let v_rows = |head_dim: usize, v_offset_rows: usize| {
        if v_per_k == 1 {
            vec![]
        } else {
            vec![LayoutTransform::ReorderVRows {
                key_heads: m.key_heads,
                v_per_k,
                head_dim,
                v_offset_rows,
            }]
        }
    };
    let offset = |value: f32| {
        if value == 0.0 {
            vec![]
        } else {
            vec![ValueTransform::ApplyWeightOffset(value)]
        }
    };
    let qk_rows = 2 * m.key_heads * m.key_head_dim;

    Ok(match role {
        "fused recurrent q|k|v" => (v_rows(m.value_head_dim, qk_rows), vec![]),
        "output-gate projection" => (v_rows(m.value_head_dim, 0), vec![]),
        "decay projection" | "write-strength projection" => (v_rows(1, 0), vec![]),
        "timestep bias" => (v_rows(1, 0), vec![]),
        "log decay" => (v_rows(1, 0), vec![ValueTransform::MaterializeLogDecay]),
        "causal conv over q|k|v" => {
            let mut layout = vec![LayoutTransform::SqueezeSingletonAxis { axis: 1 }];
            layout.extend(v_rows(m.value_head_dim, qk_rows));
            (layout, vec![])
        }
        "output projection" => {
            let layout = if v_per_k == 1 {
                vec![]
            } else {
                vec![LayoutTransform::ReorderVColumnsByGroups {
                    key_heads: m.key_heads,
                    v_per_k,
                    // Permutation moves whole heads, so for a quantised
                    // source every 16-element group moves intact —
                    // provided head_dim divides into groups, which the
                    // preflight establishes before planning begins.
                    groups_per_head: m.value_head_dim
                        / larql_models::quant::nvfp4::NVFP4_GROUP_ELEMS,
                }]
            };
            (layout, vec![])
        }
        "input layer norm" | "post-attention layer norm" => (vec![], offset(o.trunk)),
        "final norm" => (vec![], offset(o.final_norm)),
        "attention q norm" | "attention k norm" => (vec![], offset(o.qk)),
        // The gated norm's absence here is the graph speaking: its
        // surface declares no offset, so none is folded. llama.cpp's
        // converter writes the same distinction as a name-based
        // exception; this table never needs to.
        _ => (vec![], vec![]),
    })
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
