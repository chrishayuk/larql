//! **Can this surface be priced authoritatively at all?**
//!
//! That is 4b-d's entire question. It computes no cost, ranks nothing
//! and prunes nothing:
//!
//! ```text
//! PhysicalAccountingFacts
//! + TensorSurface
//!         ↓ bind()
//! BoundPhysicalAccounting
//! or
//! AccountingIncomplete { missing: [TensorIdentity, …] }
//! ```
//!
//! > **READY means every tensor on the REPRESENT surface has exactly
//! > one authoritative source price from the sealed container facts.**
//!
//! # Why this is a separate step
//!
//! The two populations are genuinely different. [`PhysicalAccountingFacts`]
//! describes what the CONTAINER stores; a [`TensorSurface`] is what
//! REPRESENT *enumerated* from it under one role classification. They
//! can disagree in both directions, and only one direction matters:
//!
//! ```text
//! surface tensor with no stored fact   → cannot be priced. INCOMPLETE.
//! stored fact no surface tensor names  → not this surface's business
//! ```
//!
//! An extra stored tensor neither satisfies nor damages completeness,
//! and pretending otherwise is how a missing price gets papered over by
//! an unrelated one that happened to be present.
//!
//! # Incompleteness is a failure to bind, not a fourth prune
//!
//! Stage 2's register holds exactly THREE usable pre-measurement
//! prunes, and "cannot be priced" is not among them: an unpriceable
//! candidate that arrived neither eligible nor pruned would break the
//! census conservation law. So the search does not start. Binding
//! happens once, before enumeration, and a
//! [`BoundPhysicalAccounting`] is what 4b-e implements `Footprint` over
//! — with no `Option`, no fallback and no missing-data branch, because
//! the absence was resolved here.
//!
//! # Blind to role and shape, on purpose
//!
//! Binding reads `(object, tensor)` and nothing else. A reclassified
//! role moves the surface identity and is a different search problem
//! (1a), and a shape is not what anything is priced by (4b-c) — so
//! neither may change whether a model can be priced at all.

use super::super::source_identity::SourceIdentity;
use super::accounting::{PhysicalAccountingFacts, SourceStorageFact, TensorIdentity};
use super::surface::{SurfaceTensor, TensorSurface};

/// The surface tensors the container stores no price for.
///
/// Complete and deterministic: every one of them, in `(object, tensor)`
/// order, so a caller reports the whole gap rather than the first
/// instance of it and gets the same list on every machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountingIncomplete {
    pub missing: Vec<TensorIdentity>,
}

impl std::fmt::Display for AccountingIncomplete {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the source container stores no price for {} surface tensor(s): {}",
            self.missing.len(),
            self.missing
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Why a surface could not be bound to a container's storage.
///
/// Two failures, kept apart because they call for different actions: a
/// foreign source is the wrong facts entirely and re-reading fixes it,
/// while incompleteness is a real gap between what REPRESENT enumerated
/// and what the container stores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccountingBindError {
    /// The facts were read from a different container.
    ForeignSource {
        facts: String,
        model: String,
    },
    /// A bound accounting was asked about a surface it did not bind.
    ForeignSurface {
        bound: String,
        asked: String,
    },
    Incomplete(AccountingIncomplete),
}

impl std::fmt::Display for AccountingBindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForeignSource { facts, model } => write!(
                f,
                "accounting facts were read from source {} and this model is {} — \
                 pricing one model's surface from another's storage",
                short(facts),
                short(model)
            ),
            Self::ForeignSurface { bound, asked } => write!(
                f,
                "this accounting was bound to surface {} and was asked about {} — \
                 completeness was proved for one population, not the other",
                short(bound),
                short(asked)
            ),
            Self::Incomplete(incomplete) => write!(f, "{incomplete}"),
        }
    }
}

impl std::error::Error for AccountingBindError {}

fn short(digest: &str) -> &str {
    &digest[..digest.len().min(12)]
}

/// **Every tensor of one surface, priced from one sealed container.**
///
/// Constructed only by [`PhysicalAccountingFacts::bind`], so holding one
/// IS the proof that the surface can be priced. That is the whole value
/// of the type: a consumer never re-checks and never handles an absence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundPhysicalAccounting {
    source: String,
    surface: String,
    priced: Vec<(TensorIdentity, SourceStorageFact)>,
}

impl BoundPhysicalAccounting {
    /// The `SourceSemanticIdentity` digest both sides agreed on.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The `TensorSurface::identity()` this was bound against.
    pub fn surface_identity(&self) -> &str {
        &self.surface
    }

    pub fn len(&self) -> usize {
        self.priced.len()
    }

    pub fn is_empty(&self) -> bool {
        self.priced.is_empty()
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&TensorIdentity, &SourceStorageFact)> {
        self.priced.iter().map(|(id, fact)| (id, fact))
    }

    /// **Every tensor of `surface`, paired with its source price.**
    ///
    /// Total by construction: binding proved each one is present, so
    /// there is no `Option` here and no branch for a caller to get
    /// wrong. One check — that this is the surface that was bound —
    /// stands in for the per-tensor ones, and a different surface is
    /// refused rather than answered over a different population.
    ///
    /// The pairing is positional, which is sound and not merely
    /// convenient: `priced` was built by walking `surface.entries()`,
    /// entries are sorted by `(object, tensor)`, and the identity is a
    /// digest of all of them — so equal identities are equal entry
    /// lists in equal order.
    pub fn prices_for<'a>(
        &'a self,
        surface: &'a TensorSurface,
    ) -> Result<Vec<(&'a SurfaceTensor, &'a SourceStorageFact)>, AccountingBindError> {
        if surface.identity() != self.surface {
            return Err(AccountingBindError::ForeignSurface {
                bound: self.surface.clone(),
                asked: surface.identity(),
            });
        }
        Ok(surface
            .entries()
            .iter()
            .zip(self.priced.iter().map(|(_, fact)| fact))
            .collect())
    }
}

impl PhysicalAccountingFacts {
    /// **Bind a surface to the storage that prices it, or say exactly
    /// what is missing.**
    ///
    /// `model` is checked against the source these facts were read
    /// from, on the SEMANTIC digest — so a re-exported container still
    /// binds (4b-b2) and a different model does not.
    pub fn bind(
        &self,
        model: &SourceIdentity,
        surface: &TensorSurface,
    ) -> Result<BoundPhysicalAccounting, AccountingBindError> {
        if !self.describe(model) {
            return Err(AccountingBindError::ForeignSource {
                facts: self.source_digest().to_string(),
                model: model.semantic_digest(),
            });
        }

        let mut priced = Vec::with_capacity(surface.len());
        let mut missing = Vec::new();
        for tensor in surface.entries() {
            // `(object, tensor)` and nothing else. An alias is two
            // surface entries, so it is two lookups and two required
            // prices — collapsing them here would let one stored fact
            // satisfy two enumerated tensors.
            let id = TensorIdentity::new(&tensor.object, &tensor.tensor);
            match self.get(&id) {
                Some(fact) => priced.push((id, fact.clone())),
                None => missing.push(id),
            }
        }
        if !missing.is_empty() {
            // `entries()` is already sorted by `(object, tensor)`, so
            // this list is deterministic; sorted again rather than
            // relying on that, because the guarantee this makes is its
            // own and should not move if surface ordering ever does.
            missing.sort();
            missing.dedup();
            return Err(AccountingBindError::Incomplete(AccountingIncomplete {
                missing,
            }));
        }

        Ok(BoundPhysicalAccounting {
            source: self.source_digest().to_string(),
            surface: surface.identity(),
            priced,
        })
    }
}
