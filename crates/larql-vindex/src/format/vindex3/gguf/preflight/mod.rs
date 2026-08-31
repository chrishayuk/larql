//! Can this artifact be lowered to this target — and is the lowering
//! lossless?
//!
//! Two questions, and they fail for different reasons. The first is
//! **semantic completeness**: does the graph carry every fact the target
//! needs, without anyone reaching back to source-family configuration?
//! The second is **representation compatibility**: does the target's
//! required tensor layout survive contact with how these weights are
//! actually quantised?
//!
//! The second question is the one nothing internal ever had to ask.
//! qwen35 wants V heads in tiled rather than grouped order, which means
//! permuting the input axis of `out_proj` — and the input axis is
//! exactly the axis NVFP4 groups run along. Permute it carelessly and an
//! element lands in a new group while its E4M3 scale belongs to the old
//! one. Nothing errors. The weights stay finite and plausible and decode
//! to noise, and a fidelity number measured before the export describes
//! a representation the file does not contain.
//!
//! **The general invariant, of which Qwen's arithmetic is one instance:**
//!
//! ```text
//! every permutation boundary must lie on a quantisation-group boundary
//! ```
//!
//! For qwen35's V-head reorder the permuted axis reshapes to
//! `[k_heads, v_per_k, head_dim]` and the innermost `head_dim` stays
//! contiguous, so the condition reduces to `head_dim % group == 0`.
//! Qwen3.8 satisfies it at 128 % 16. A model at 120 would not, and the
//! preflight refuses it by name rather than shipping split groups.
//!
//! The report is a value, not a series of early returns, so `export`
//! can print it before writing anything.

use std::fmt;

use crate::format::vindex3::graph::surface::ExecutionSurface;
use larql_models::quant::nvfp4::NVFP4_GROUP_ELEMS;

/// One fact the target needs, and whether the graph carries it.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    pub name: &'static str,
    /// What the graph answered, or `None` when it has no answer.
    pub value: Option<String>,
}

impl Requirement {
    pub fn met(&self) -> bool {
        self.value.is_some()
    }
}

/// A layout condition the lowering depends on, checked against the
/// source's actual geometry rather than assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub name: &'static str,
    /// The invariant in the target's own terms.
    pub invariant: &'static str,
    pub satisfied: bool,
    /// The arithmetic, so a refusal shows its working.
    pub detail: String,
}

/// Why an export cannot proceed. Never a bare "unsupported".
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    /// The graph does not carry a fact the target needs. A defect in the
    /// artifact, not in the lowering — see the format spec's
    /// independent-backend test.
    MissingSemantic {
        requirement: &'static str,
        required_by: &'static str,
    },
    /// The target's required layout would damage this representation.
    IncompatibleGeometry {
        operation: &'static str,
        axis: &'static str,
        invariant: &'static str,
        detail: String,
    },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSemantic {
                requirement,
                required_by,
            } => write!(
                f,
                "cannot export: the graph does not carry `{requirement}`, required by {required_by} — \
                 this is the artifact missing a fact, not the target asking for too much"
            ),
            Self::IncompatibleGeometry {
                operation,
                axis,
                invariant,
                detail,
            } => write!(
                f,
                "cannot export: {operation} permutes {axis}, and {invariant} does not hold here — \
                 {detail}. Permuting anyway would split quantisation groups and require \
                 re-quantisation, producing a representation nobody measured"
            ),
        }
    }
}

/// The complete answer, printable before a byte is written.
#[derive(Debug, Clone, PartialEq)]
pub struct Preflight {
    pub target: &'static str,
    pub requirements: Vec<Requirement>,
    pub constraints: Vec<Constraint>,
    pub refusals: Vec<Refusal>,
}

impl Preflight {
    pub fn ready(&self) -> bool {
        self.refusals.is_empty()
    }
}

/// Geometry the qwen35 V-head reorder acts on, read from the graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VHeadGeometry {
    pub key_heads: usize,
    pub value_heads: usize,
    pub value_head_dim: usize,
}

impl VHeadGeometry {
    pub fn v_per_k(&self) -> Option<usize> {
        if self.key_heads == 0 || !self.value_heads.is_multiple_of(self.key_heads) {
            return None;
        }
        Some(self.value_heads / self.key_heads)
    }

    /// **The invariant that makes the reorder a permutation of whole
    /// quantisation groups rather than a re-quantisation.**
    ///
    /// The permuted axis reshapes to `[k_heads, v_per_k, head_dim]` and
    /// the swap moves the outer two, leaving `head_dim` contiguous. So
    /// every boundary the permutation creates lies on a group boundary
    /// exactly when `head_dim` is a whole number of groups.
    pub fn reorder_preserves_groups(&self, group: usize) -> bool {
        group != 0 && self.value_head_dim.is_multiple_of(group)
    }
}

/// Semantic facts the target's *value* transforms depend on, as opposed
/// to the ones its metadata needs.
///
/// qwen35 stores `ssm_a` as the materialised negative decay coefficient
/// rather than the log parameter, and stores the trunk norms with their
/// offset already folded in. Both are arithmetic on weights, and both
/// are only legitimate because the graph says which operand is a log
/// decay and which norms carry an offset.
///
/// Requiring them here is the difference between a lowering that reads
/// a fact and one that knows it is looking at Qwen. Without this, the
/// two transforms would be `+ 1.0 because qwen35` and `-exp because
/// A_log` — source-family assumptions smuggled back in one commit after
/// they were removed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformFacts {
    /// A GDN operand carries the `log decay` role, so the target can
    /// materialise `-exp(x)` from a semantic rather than a tensor name.
    pub log_decay_role_present: bool,
}

/// Preflight a component's execution surface against the qwen35 target.
pub fn qwen35_preflight(
    surface: &ExecutionSurface,
    nvfp4_in_use: bool,
    transforms: TransformFacts,
) -> Preflight {
    let mut requirements = Vec::new();
    let mut constraints = Vec::new();
    let mut refusals = Vec::new();

    let mut need = |name: &'static str, value: Option<String>, required_by: &'static str| {
        if value.is_none() {
            refusals.push(Refusal::MissingSemantic {
                requirement: name,
                required_by,
            });
        }
        requirements.push(Requirement { name, value });
    };

    need(
        "execution.context_length",
        surface.context_length.map(|v| v.to_string()),
        "qwen35.context_length",
    );
    let attn = surface.attention.as_ref();
    need(
        "execution.attention.num_q_heads",
        attn.map(|a| a.num_q_heads.to_string()),
        "qwen35.attention.head_count",
    );
    need(
        "execution.attention.num_kv_heads",
        attn.map(|a| a.num_kv_heads.to_string()),
        "qwen35.attention.head_count_kv",
    );
    need(
        "execution.attention.head_dim",
        attn.map(|a| a.head_dim.to_string()),
        "qwen35.attention.key_length",
    );
    need(
        "execution.ffn.intermediate_size",
        surface
            .ffn
            .as_ref()
            .map(|f| f.intermediate_size.to_string()),
        "qwen35.feed_forward_length",
    );
    need(
        "execution.norm.eps",
        Some(surface.norm.pre.eps.to_string()),
        "qwen35.attention.layer_norm_rms_epsilon",
    );

    // The linear-attention geometry, and the constraint that rides on it.
    match surface.linear_attention.as_ref() {
        None => need("execution.linear_attention", None, "qwen35 GDN lowering"),
        Some(la) => {
            let geom = VHeadGeometry {
                key_heads: la.key_heads,
                value_heads: la.value_heads,
                value_head_dim: la.value_head_dim,
            };
            requirements.push(Requirement {
                name: "execution.linear_attention.key_heads",
                value: Some(la.key_heads.to_string()),
            });
            requirements.push(Requirement {
                name: "execution.linear_attention.value_heads",
                value: Some(la.value_heads.to_string()),
            });
            requirements.push(Requirement {
                name: "execution.linear_attention.value_head_dim",
                value: Some(la.value_head_dim.to_string()),
            });

            let divides = geom.v_per_k().is_some();
            constraints.push(Constraint {
                name: "v-heads group under k-heads",
                invariant: "value_heads % key_heads == 0",
                satisfied: divides,
                detail: format!(
                    "{} value heads over {} key heads",
                    la.value_heads, la.key_heads
                ),
            });
            if !divides {
                refusals.push(Refusal::IncompatibleGeometry {
                    operation: "qwen35 V-head reorder",
                    axis: "rows and columns",
                    invariant: "value_heads % key_heads == 0",
                    detail: format!(
                        "{} value heads do not group evenly under {} key heads",
                        la.value_heads, la.key_heads
                    ),
                });
            }

            // Only binding when weights are actually block-quantised. A
            // BF16 export permutes freely — there are no groups to split.
            if nvfp4_in_use {
                let ok = geom.reorder_preserves_groups(NVFP4_GROUP_ELEMS);
                constraints.push(Constraint {
                    name: "V-head reorder lands on group boundaries",
                    invariant: "value_head_dim % nvfp4_group == 0",
                    satisfied: ok,
                    detail: format!(
                        "head dim {} over group width {} = {} whole groups per head",
                        la.value_head_dim,
                        NVFP4_GROUP_ELEMS,
                        la.value_head_dim / NVFP4_GROUP_ELEMS.max(1)
                    ),
                });
                if !ok {
                    refusals.push(Refusal::IncompatibleGeometry {
                        operation: "qwen35 V-head reorder",
                        axis: "columns (the input axis, which NVFP4 groups run along)",
                        invariant: "value_head_dim % nvfp4_group == 0",
                        detail: format!(
                            "head dimension {} is not a whole number of {}-element groups",
                            la.value_head_dim, NVFP4_GROUP_ELEMS
                        ),
                    });
                }
            }
        }
    }

    // The trunk norms qwen35 stores pre-offset must SAY they carry one.
    // The GDN's internal gated norm deliberately does not declare an
    // offset and deliberately does not receive one — llama.cpp's
    // converter makes the same exception, and here it falls out of the
    // graph rather than being written down twice.
    // Always declared, so this is not a presence check — it is a record
    // that the transform reads the graph's number. `0.0` and `1.0` are
    // both answers; the lowering must fold whichever is there, and never
    // a literal `1.0` justified by the family name.
    requirements.push(Requirement {
        name: "execution.norm.pre.weight_offset",
        value: Some(surface.norm.pre.weight_offset.to_string()),
    });
    requirements.push(Requirement {
        name: "execution.norm.final_norm.weight_offset",
        value: Some(surface.norm.final_norm.weight_offset.to_string()),
    });

    requirements.push(Requirement {
        name: "operand role `log decay`",
        value: transforms
            .log_decay_role_present
            .then(|| "present".to_string()),
    });
    if !transforms.log_decay_role_present {
        refusals.push(Refusal::MissingSemantic {
            requirement: "operand role `log decay`",
            required_by: "qwen35 ssm_a stores -exp(log decay), not the log parameter",
        });
    }

    Preflight {
        target: "gguf/qwen35",
        requirements,
        constraints,
        refusals,
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub mod tests_support;
