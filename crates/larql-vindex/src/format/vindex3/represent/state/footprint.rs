//! **The production `Footprint`: what a bound search surface costs
//! under one representation state.**
//!
//! ```text
//! BoundPhysicalAccounting + resolved state
//!         ↓
//! Σ over every bound surface tensor:
//!     effective == source     → the sealed SourceStorageFact
//!     effective == encoding   → the pack layout's stored length
//! ```
//!
//! Total by construction. No missing branch, because 4b-d already
//! proved every surface tensor is priceable; no source-dtype
//! multiplication, because 4b-c made `dtype` uncomputable; no container
//! traversal, because the facts were read once.
//!
//! # What a footprint is a footprint OF
//!
//! > **The complete footprint of the bound REPRESENT surface under a
//! > representation state — not the byte size of the whole container.**
//!
//! ```text
//! container footprint            everything physically stored in the
//!                                VINDEX3 container
//!
//! representation footprint       every tensor in the bound
//!                                TensorSurface, priced under one
//!                                resolved state          ← this
//! ```
//!
//! A map resolves over a `TensorSurface`, so its domain is the surface
//! and not every artifact that happens to coexist in the container.
//! Anything outside the surface has no representation decision in this
//! search problem, and including it would make the optimiser account
//! for bytes it cannot transform. 4b-d's asymmetry is the proof: an
//! extra stored fact neither satisfies nor damages surface
//! completeness, so it must not reappear in the state's footprint
//! either.
//!
//! # Misses are made impossible, not handled
//!
//! [`super::candidate::Footprint`] returns a `LogicalBytes` with no
//! channel for "I could not price that", and inventing one would break
//! stage 2's census: an unpriced candidate is neither eligible nor
//! pruned. Stage 4 hit exactly this wall. The answer is not a fallible
//! trait but a constructor that enumerates the finite problem — every
//! bound tensor × every encoding the search may select — and refuses up
//! front:
//!
//! ```text
//! layout REFUSES (tensor, encoding)   → resolves to source; no price needed
//! layout ADMITS  (tensor, encoding)   → a compiled price is REQUIRED
//! ```
//!
//! The same [`LayoutAdmission`] the resolver uses, so the price table
//! and the decision vector cannot disagree about which tensors are
//! compiled. What remains is a state from another surface, which
//! [`SurfaceFootprint::try_logical_bytes`] reports and the trait method
//! cannot.

use std::collections::BTreeMap;

use super::super::nvfp4_pack::{PackLayout, DTYPE_NVFP4};
use super::accounting::TensorIdentity;
use super::bind::BoundPhysicalAccounting;
use super::candidate::Footprint;
use super::identity::RepresentationState;
use super::realization::LogicalBytes;
use super::resolved::{LayoutAdmission, SOURCE_PRECISION};
use super::surface::TensorSurface;

/// **What one tensor occupies once compiled to one encoding.**
///
/// Separate from [`LayoutAdmission`], which answers whether the
/// encoding can hold the tensor at all. An oracle that holds no rule
/// for an encoding answers `None`, and [`SurfaceFootprint::new`] turns
/// that into a refusal naming the tensor — never into a guess.
pub trait CompiledBytes {
    fn compiled_bytes(&self, encoding: &str, shape: &[usize]) -> Option<LogicalBytes>;
}

/// The pack layouts this build actually compiles.
///
/// Answers for `DTYPE_NVFP4` by asking [`PackLayout::derive`] — the same
/// call the compiler and [`super::resolved::PackLayoutAdmission`] make,
/// so a price, an admission and a written pack cannot drift — and
/// declares nothing about any other encoding. A search whose vocabulary
/// names one of those cannot be priced by this build, and
/// [`SurfaceFootprint::new`] says which encoding and which tensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct PackCompiledBytes;

impl CompiledBytes for PackCompiledBytes {
    fn compiled_bytes(&self, encoding: &str, shape: &[usize]) -> Option<LogicalBytes> {
        if encoding != DTYPE_NVFP4 {
            return None;
        }
        PackLayout::derive(shape, encoding)
            .ok()
            .map(|layout| LogicalBytes::new(layout.total_len as u64))
    }
}

/// Why a surface could not be given a total price table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FootprintError {
    /// The layout admits this tensor at this encoding and no oracle
    /// prices it, so a state selecting it would have no footprint.
    Unpriceable {
        tensor: TensorIdentity,
        encoding: String,
    },
    /// A state resolved against another surface.
    ForeignSurface { priced: String, asked: String },
}

impl std::fmt::Display for FootprintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unpriceable { tensor, encoding } => write!(
                f,
                "no oracle prices `{tensor}` at `{encoding}`, and the layout admits it — \
                 a state selecting that encoding would have no footprint at all"
            ),
            Self::ForeignSurface { priced, asked } => write!(
                f,
                "this footprint prices surface {} and was asked about a state resolved \
                 against {}",
                short(priced),
                short(asked)
            ),
        }
    }
}

impl std::error::Error for FootprintError {}

fn short(digest: &str) -> &str {
    &digest[..digest.len().min(12)]
}

/// Every price one tensor can present, resolved ahead of any state.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorPrices {
    source: LogicalBytes,
    /// Encoding → stored bytes, for every encoding the layout admits.
    /// An encoding the layout refuses is absent, because such a tensor
    /// presents source bytes and is never asked for a compiled price.
    compiled: BTreeMap<String, LogicalBytes>,
}

/// **A total price table for one bound surface.**
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceFootprint {
    surface: String,
    prices: BTreeMap<TensorIdentity, TensorPrices>,
}

impl SurfaceFootprint {
    /// Price every bound tensor at source and at every encoding the
    /// search may select, or refuse naming the first pair nothing can
    /// price.
    ///
    /// `encodings` is what the base map and the action vocabulary
    /// between them can put into a decision — an INPUT, exactly as
    /// `ActionVocabulary` is (R5-F6). Enumerating it here is what turns
    /// a per-call miss into a configuration error the search reports
    /// before it starts.
    pub fn new(
        bound: &BoundPhysicalAccounting,
        surface: &TensorSurface,
        layout: &dyn LayoutAdmission,
        compiled: &dyn CompiledBytes,
        encodings: &[String],
    ) -> Result<Self, FootprintError> {
        let mut prices = BTreeMap::new();
        for (tensor, price) in
            bound
                .prices_for(surface)
                .map_err(|_| FootprintError::ForeignSurface {
                    priced: bound.surface_identity().to_string(),
                    asked: surface.identity(),
                })?
        {
            let id = TensorIdentity::new(&tensor.object, &tensor.tensor);
            let mut per_encoding = BTreeMap::new();
            for encoding in encodings {
                if !layout.admits(encoding, tensor) {
                    continue;
                }
                let bytes = compiled
                    .compiled_bytes(encoding, &tensor.shape)
                    .ok_or_else(|| FootprintError::Unpriceable {
                        tensor: id.clone(),
                        encoding: encoding.clone(),
                    })?;
                per_encoding.insert(encoding.clone(), bytes);
            }
            prices.insert(
                id,
                TensorPrices {
                    source: price.logical_bytes,
                    compiled: per_encoding,
                },
            );
        }
        Ok(Self {
            surface: surface.identity(),
            prices,
        })
    }

    /// The surface this table prices.
    pub fn surface_identity(&self) -> &str {
        &self.surface
    }

    /// **Sum the state's presented bytes**, or say why it cannot be
    /// summed.
    ///
    /// The only reachable failure is a state resolved against another
    /// surface: within this one, every tensor has a source price and
    /// every admitted encoding has a compiled price, both established
    /// by [`Self::new`].
    pub fn try_logical_bytes(
        &self,
        state: &RepresentationState,
    ) -> Result<LogicalBytes, FootprintError> {
        if state.surface_identity() != self.surface {
            return Err(FootprintError::ForeignSurface {
                priced: self.surface.clone(),
                asked: state.surface_identity().to_string(),
            });
        }
        let mut total = 0u64;
        for decision in state.decisions().decisions() {
            let id = TensorIdentity::new(&decision.object, &decision.tensor);
            let prices = self
                .prices
                .get(&id)
                .expect("the state's surface is this surface, so every tensor is priced");
            // `effective()` and not the declared encoding: a protected
            // tensor and a layout-refused one present the same source
            // bytes, and pricing the refusal as compiled would claim a
            // saving the container never made.
            total += match decision.encoding.effective() {
                SOURCE_PRECISION => prices.source,
                encoding => *prices
                    .compiled
                    .get(encoding)
                    .expect("the layout admitted this encoding, so `new` priced it"),
            }
            .get();
        }
        Ok(LogicalBytes::new(total))
    }
}

impl Footprint for SurfaceFootprint {
    /// The trait has no channel for a miss — inventing one would break
    /// stage 2's census — so this is total over the surface it was
    /// built for and panics on a state from another one. That is a
    /// caller contract, checkable in advance by
    /// [`Self::try_logical_bytes`].
    fn logical_bytes(&self, state: &RepresentationState) -> LogicalBytes {
        self.try_logical_bytes(state)
            .unwrap_or_else(|e| panic!("{e}"))
    }
}
