//! Declared hybrid topology: which layers of a scope run which operator,
//! whatever spelling the checkpoint used to say so.
//!
//! Four spellings of one fact are in the wild, and a fifth scope:
//!
//! | checkpoint | key | encoding | base |
//! |---|---|---|---|
//! | Qwen3.8 | `layer_types` | per-layer array | — |
//! | GLM-5.3-Flash | `linear_attn_config.{kda,full_attn}_layers` | two sets that partition | zero |
//! | Kimi Linear | same keys | two sets that partition | **one** |
//! | Inkling-Small | `local_layer_ids` | one set, complement implied | zero |
//! | Inkling-Small MTP | `mtp_config.local_layer_ids` | one set, **its own sub-stack** | zero |
//!
//! Reading only some of them is not a gap, it is a *wrong answer*: a
//! checkpoint whose declaration this build cannot read looks like a
//! checkpoint that declared nothing, and the caller's default then answers
//! for a topology the author actually stated. That has now happened three
//! times — Qwen3.8's 48 recurrent layers, Kimi Linear's 20, and
//! Inkling-Small's 35 sliding layers, each reported as full attention.
//!
//! Inkling's instance is the one that shows the cost is not cosmetic: 35
//! layers with a 512-token window, reported as retaining an unbounded
//! prefix, against a 1,048,576-token context. A KV planner reading that
//! sizes a cache two orders of magnitude too large.
//!
//! ## The invariant
//!
//! **A declared hybrid topology must resolve to exactly one kind for every
//! layer in its scope.** Overlap, a hole, an out-of-range index, an
//! ambiguous base, an unknown spelling or a length mismatch each make the
//! declaration *unresolved*, and an unresolved declaration blocks. None of
//! them may fall through to full attention.
//!
//! ## Scope
//!
//! Carried explicitly because Inkling-Small declares `local_layer_ids`
//! twice — once for its 42-layer decoder and once, in `mtp_config`, for
//! its 8-layer MTP sub-stack. The two index different layer spaces, so a
//! resolution is only meaningful against the scope it was read for.

mod resolve;
mod spellings;

#[cfg(test)]
mod tests;

pub use resolve::{resolve_declarations, resolve_per_layer_array};
pub use spellings::{read_declared_interleave, InterleaveScope};

use serde::{Deserialize, Serialize};

/// Which layer index a checkpoint's declared sets count from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerIndexBase {
    /// Sets index layers `0..layer_count` — GLM-5.3-Flash, Inkling-Small.
    Zero,
    /// Sets index layers `1..=layer_count` — Kimi Linear.
    One,
}

impl LayerIndexBase {
    /// Every base a declaration may be proven in, in the order tried.
    pub const ALL: [Self; 2] = [Self::Zero, Self::One];

    /// Offset to subtract to reach a zero-based layer index.
    pub fn offset(self) -> i64 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
        }
    }
}

/// Which recurrence a layer runs. Named families, because the operators
/// are not interchangeable — see `opplan::kda`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrenceFamily {
    /// Kimi Delta Attention (`linear_attn_config`).
    Kda,
    /// Gated DeltaNet (Qwen3.8's `linear_*` geometry).
    GatedDelta,
    /// Mamba2 / SSD (`state_size`-family geometry — see
    /// [`Mamba2Geometry`](crate::config::Mamba2Geometry)).
    Mamba2,
    /// Declared recurrent, family not identified by any geometry read.
    Unidentified,
}

/// What a layer runs — the *semantic* kind, never the key that declared it.
///
/// Deliberately not the `layer_types` vocabulary: that is one spelling of
/// this, and folding the two would make the canonical form depend on which
/// checkpoint happened to be read first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Attends to the whole prefix.
    Full,
    /// Attends to the last `window` positions. `window` is `None` when the
    /// checkpoint declares the interleave but states no size — an absence
    /// carried forward, never a default, because a wrong window is a wrong
    /// KV residency plan.
    Sliding { window: Option<usize> },
    /// A recurrence: no per-position prefix, so no span applies.
    Recurrent(RecurrenceFamily),
    /// The checkpoint declared this layer, and this build has no kind for
    /// what it said. The declaration is carried verbatim.
    ///
    /// A kind, rather than a resolution failure, because the invariant is
    /// **per layer**: one unreadable entry must not erase the layers
    /// beside it that read perfectly well. GLM-5.3-Flash is the case —
    /// its `layer_types` names 34 recurrent layers this build resolves and
    /// 11 `deepseek_sparse_attention` layers it cannot, and reporting 45
    /// unexpressed would hide the 34 that are understood.
    ///
    /// It still blocks: it round-trips to its own spelling and to nothing
    /// executable.
    Unexpressed { declared: String },
}

/// How a declaration states which layers it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Membership {
    /// The layers this kind names, in the checkpoint's own index base.
    ExplicitSet(Vec<i64>),
    /// Every layer no other declaration names. At most one declaration in
    /// a scope may be the complement — two would not determine an answer.
    Complement,
}

/// One kind and the layers it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    pub kind: LayerKind,
    pub membership: Membership,
}

/// The shape the checkpoint used, recorded so two checkpoints expressing
/// one concept differently can be shown to reach the same semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterleaveEncoding {
    /// One entry per layer (`layer_types`).
    PerLayerArray,
    /// Two or more explicit sets that together cover the scope.
    PartitionSets,
    /// One explicit set; every other layer takes the remaining kind.
    ExplicitSetWithComplement,
}

/// Where a resolution came from and how it was reached.
///
/// The "source spelling → canonical semantic" chain: without it, two
/// checkpoints that resolve to the same topology cannot be shown to have
/// *declared* it differently, and a wrong reading looks identical to a
/// right one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterleaveProvenance {
    /// Config paths read, in the order consulted.
    pub sources: Vec<String>,
    pub encoding: InterleaveEncoding,
    /// The base proven from the declaration. `None` for a per-layer array,
    /// which has no base to prove.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_base: Option<LayerIndexBase>,
    /// The layer space this resolution indexes.
    pub scope: String,
}

/// A declared topology, resolved: exactly one kind per layer of the scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedInterleave {
    pub layer_count: usize,
    pub provenance: InterleaveProvenance,
    /// Zero-based, `layer_count` long, one kind per layer. The invariant
    /// this whole module exists to uphold, stated in the type.
    pub layers: Vec<LayerKind>,
}

impl ResolvedInterleave {
    /// Layers of one kind, by predicate — the census a planner reads.
    pub fn count(&self, predicate: impl Fn(&LayerKind) -> bool) -> usize {
        self.layers.iter().filter(|k| predicate(k)).count()
    }
}

/// Why a declaration could not be resolved. Every variant blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterleaveError {
    /// Nothing declared in any known spelling — the ordinary case for a
    /// uniform-attention model, and the only variant that is not a defect.
    NotDeclared,
    /// No index base places every declared index inside the scope.
    NoConsistentBase {
        declared_indices: usize,
        layer_count: usize,
    },
    /// Both bases place the declaration validly, so neither is proven.
    /// Reachable only with an implied complement: a partition of the scope
    /// cannot satisfy both bases, but a bare set can.
    AmbiguousBase { layer_count: usize },
    /// Two declarations name the same layer.
    Overlap { layer: usize },
    /// No declaration names this layer, and none takes the complement.
    Uncovered { layer: usize },
    /// A per-layer array entry this build has no kind for.
    UnknownSpelling { entry: String },
    /// A per-layer array whose length is not the scope's layer count.
    LengthMismatch { declared: usize, layer_count: usize },
    /// More than one declaration claims the complement.
    MultipleComplements,
}

/// What a checkpoint declared, as an outcome.
///
/// Three states, not an `Option`: "declared nothing" and "declared
/// something unreadable" must stay distinguishable, or an unreadable
/// declaration silently becomes the caller's default — which is the whole
/// defect this module exists to close.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeclaredInterleave {
    #[default]
    Absent,
    Resolved(Box<ResolvedInterleave>),
    Unresolved(InterleaveError),
}

impl DeclaredInterleave {
    /// The resolution, when there is one.
    pub fn resolved(&self) -> Option<&ResolvedInterleave> {
        match self {
            Self::Resolved(r) => Some(r.as_ref()),
            Self::Absent | Self::Unresolved(_) => None,
        }
    }

    /// The refusal, when the checkpoint declared something unreadable.
    pub fn error(&self) -> Option<&InterleaveError> {
        match self {
            Self::Unresolved(e) => Some(e),
            Self::Absent | Self::Resolved(_) => None,
        }
    }
}
