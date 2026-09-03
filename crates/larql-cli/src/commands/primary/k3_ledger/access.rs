//! **Physical existence is not execution touch.**
//!
//! K3-DENSE-1 found two modelling errors in the dense ledger, and they
//! partly cancelled — which is exactly why the aggregate looked believable
//! for six weeks.
//!
//! ```text
//! layer 0 is structurally DENSE, not MoE      +1.07 GB  undercount
//! embed_tokens is a GATHER, not a full read   -2.35 GB  overcount
//!                                             ---------
//!                                             -1.28 GB  net
//! ```
//!
//! Both came from one assumption: that a tensor's presence in the
//! checkpoint means the whole tensor is read on every forward pass. It
//! does not.
//!
//! - `embed_tokens` is `[163840, 7168]`, 2,348.81 MB. A decode step reads
//!   **one row** — 14.3 KB. `lm_head` is the same shape and genuinely IS
//!   a full read, because it is a matmul over the vocabulary. They are
//!   untied (`tie_word_embeddings: false`), so this is two tensors of
//!   identical size with completely different access.
//! - `first_k_dense_replace: 1` makes layer 0 a DENSE layer: it has a
//!   `[33792, 7168]` MLP and no shared experts, no LatentMoE wrapper and
//!   no router. Modelling all 93 layers as MoE credits it with 0.38 GB it
//!   does not have and omits 1.45 GB it does.
//!
//! So access semantics are a TYPE. A family declares how it is touched,
//! and the ledger multiplies bytes by that rather than by 1.
//!
//! This is the same lesson as R4-F9 one layer up: there, physical file
//! length was not representation size. Here, resident footprint is not
//! activated traffic. Three quantities, three authorities:
//!
//! ```text
//! resident footprint    what must fit in memory
//! activated per token   what must move per token   <- the ledger's subject
//! logical bytes         what the representation costs
//! ```

use serde::{Deserialize, Serialize};

/// How a tensor is touched during one decode step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AccessMode {
    /// Every byte is read on every forward pass. Attention projections,
    /// `lm_head`, shared experts.
    FullRead,
    /// A gather: only `rows_per_token` of `total_rows` are touched.
    /// `embed_tokens`.
    Gather {
        rows_per_token: u64,
        total_rows: u64,
    },
    /// Only `active` of `total` units are touched, chosen per token.
    /// The MXFP4 expert bank.
    Routed { active: u64, total: u64 },
}

impl AccessMode {
    /// Bytes actually moved per token, given the tensor's resident size.
    pub fn activated_bytes(&self, resident_bytes: u64) -> u64 {
        match self {
            Self::FullRead => resident_bytes,
            Self::Gather {
                rows_per_token,
                total_rows,
            } => {
                if *total_rows == 0 {
                    0
                } else {
                    // Integer-safe: divide before multiplying would lose a
                    // row's worth on every gather.
                    (resident_bytes as u128 * *rows_per_token as u128 / *total_rows as u128) as u64
                }
            }
            Self::Routed { active, total } => {
                if *total == 0 {
                    0
                } else {
                    (resident_bytes as u128 * *active as u128 / *total as u128) as u64
                }
            }
        }
    }

    /// Whether the whole tensor moves. Only `FullRead` may be priced by
    /// its resident size.
    pub fn is_full_read(&self) -> bool {
        matches!(self, Self::FullRead)
    }
}

/// What a layer actually contains.
///
/// `first_k_dense_replace` layers are Dense; the rest are MoE. Treating
/// them uniformly is the layer-0 error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// A dense FFN, no router, no experts, no shared experts.
    Dense,
    /// Router + shared experts + LatentMoE wrapper + routed bank.
    Moe,
}

/// Which layers are dense and which are MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerTopology {
    /// Total decoder layers.
    pub n_layers: usize,
    /// Leading layers replaced by a dense FFN.
    pub n_dense: usize,
}

impl LayerTopology {
    /// Build a topology, clamping `n_dense` to the layer count.
    pub fn new(n_layers: usize, n_dense: usize) -> Self {
        Self {
            n_layers,
            n_dense: n_dense.min(n_layers),
        }
    }

    /// What layer `index` is.
    pub fn kind(&self, index: usize) -> LayerKind {
        if index < self.n_dense {
            LayerKind::Dense
        } else {
            LayerKind::Moe
        }
    }

    /// How many MoE layers there are — the multiplier for shared experts,
    /// the LatentMoE wrapper and the router.
    pub fn n_moe(&self) -> usize {
        self.n_layers - self.n_dense
    }

    /// How many dense layers there are.
    pub fn n_dense(&self) -> usize {
        self.n_dense
    }
}

/// One measured weight family and how it is touched.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Family {
    /// Name as it appears in a report.
    pub name: String,
    /// Resident bytes for ONE unit, measured from a safetensors header.
    pub bytes_per_unit: u64,
    /// How many units the model has.
    pub units: usize,
    /// How one unit is touched per token.
    pub access: AccessMode,
}

impl Family {
    /// A family read in full on every pass.
    pub fn full_read(name: impl Into<String>, bytes_per_unit: u64, units: usize) -> Self {
        Self {
            name: name.into(),
            bytes_per_unit,
            units,
            access: AccessMode::FullRead,
        }
    }

    /// Total bytes that must be RESIDENT.
    pub fn resident_bytes(&self) -> u64 {
        self.bytes_per_unit * self.units as u64
    }

    /// Total bytes ACTIVATED per token.
    pub fn activated_bytes(&self) -> u64 {
        self.access.activated_bytes(self.bytes_per_unit) * self.units as u64
    }
}

/// The measured dense-side census: every BF16 family and its access.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseCensus {
    /// Families, largest activated first.
    pub families: Vec<Family>,
}

impl DenseCensus {
    /// Build a census, ordering by activated bytes so a report cannot lead
    /// with a family that does not dominate.
    pub fn new(mut families: Vec<Family>) -> Self {
        families.sort_by_key(|f| std::cmp::Reverse(f.activated_bytes()));
        Self { families }
    }

    /// Bytes that must be resident, across every family.
    pub fn resident_bytes(&self) -> u64 {
        self.families.iter().map(Family::resident_bytes).sum()
    }

    /// Bytes moved per token, across every family.
    pub fn activated_bytes(&self) -> u64 {
        self.families.iter().map(Family::activated_bytes).sum()
    }

    /// A family by name.
    pub fn family(&self, name: &str) -> Option<&Family> {
        self.families.iter().find(|f| f.name == name)
    }

    /// The share of activated traffic one family carries.
    pub fn activated_share(&self, name: &str) -> f64 {
        let total = self.activated_bytes();
        if total == 0 {
            return 0.0;
        }
        self.family(name)
            .map_or(0.0, |f| f.activated_bytes() as f64 / total as f64)
    }
}

#[cfg(test)]
#[path = "access_tests.rs"]
mod access_tests;
