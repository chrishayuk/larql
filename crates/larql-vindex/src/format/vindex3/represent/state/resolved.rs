//! **What the container will actually present, tensor by tensor.**
//!
//! [`PrecisionMap::resolve`] answers what the *rules* decide. That is
//! not the same as what gets stored, because a storage layout can refuse
//! a tensor the policy admits: NVFP4 holds 2-D matrices whose `k` is a
//! multiple of the 16-element group, and `represent`'s compiler carries
//! anything else verbatim whatever the map said. A search state built on
//! the declared decision would believe it had compiled tensors that are
//! still at source precision, and would price bytes it never saved.
//!
//! ```text
//! map says          layout says       presented        counts as
//! ────────────────────────────────────────────────────────────────
//! compile NVFP4     can hold it       NVFP4            Compiled
//! compile NVFP4     k not a multiple  source bytes     LayoutRefused
//! source precision  —                 source bytes     Source
//! ```
//!
//! `LayoutRefused` and `Source` present the same bytes, so they are the
//! same *state* — [`ResolvedEncoding::effective`] is what the state
//! digest reads. They are different *facts*, and the variants stay
//! distinct because the action generator needs them apart: removing a
//! protection is a legal move that changes the state, and "un-refusing"
//! a layout is not a move at all.
//!
//! # Refusal takes positive evidence
//!
//! [`LayoutAdmission`] answers *does this layout refuse this tensor*, and
//! an implementation that holds no rule for an encoding answers no. That
//! is deliberate and it is the safe direction here: a refusal this
//! module invents would silently mark a tensor as un-compilable and
//! delete it from the action space forever, whereas an admission that
//! the compiler later refuses shows up as a resolved-decision mismatch
//! the moment anything is actually compiled. A constraint has to be
//! stated by whoever owns the layout.

use serde::{Deserialize, Serialize};

use crate::error::VindexError;

use super::super::map::{Precision, PrecisionMap};
use super::super::nvfp4_pack::{PackLayout, DTYPE_NVFP4};
use super::surface::{SurfaceTensor, TensorSurface, FIELD, RECORD};

/// The canonical spelling of "whatever the checkpoint had".
///
/// A sentinel rather than an encoding name, because source precision is
/// not one encoding — it is BF16 here and F32 there — and the state is
/// the same state either way. What the source bytes actually are is
/// already pinned by the model identity's per-segment hashes.
pub const SOURCE_PRECISION: &str = "source";

/// What one tensor ends up represented as.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolvedEncoding {
    /// The map compiles it and the layout can hold it.
    Compiled(String),
    /// The map holds it at source precision — an unnamed role, or an
    /// exception that protects it.
    Source,
    /// The map compiles it and the layout cannot hold it, so the
    /// container carries the source bytes.
    LayoutRefused { encoding: String },
}

impl ResolvedEncoding {
    /// **What is presented.** Identity reads this and only this: a
    /// protected tensor and a layout-refused tensor are the same bytes.
    pub fn effective(&self) -> &str {
        match self {
            Self::Compiled(enc) => enc,
            Self::Source | Self::LayoutRefused { .. } => SOURCE_PRECISION,
        }
    }

    pub fn is_compiled(&self) -> bool {
        matches!(self, Self::Compiled(_))
    }

    /// **Which fact produced this decision**, as distinct from what it
    /// presents.
    ///
    /// `Source` and `LayoutRefused` present identical bytes and are one
    /// physical state; they are not one *realization*, because the
    /// actions available from them differ. Unprotecting a protected
    /// tensor is a legal move; un-refusing a layout is not a move at
    /// all. The search reads this; the physical digest does not.
    pub fn fact(&self) -> &'static str {
        match self {
            Self::Compiled(_) => "compiled",
            Self::Source => "source",
            Self::LayoutRefused { .. } => "layout-refused",
        }
    }
}

/// Whether an encoding's storage layout can hold a tensor.
///
/// A trait rather than a call into [`PackLayout`] so that resolution
/// does not become NVFP4-specific the moment a second encoding exists,
/// and so tests can state a surface's admissibility directly instead of
/// having to construct shapes that happen to trip a real layout.
pub trait LayoutAdmission {
    /// `false` only where the layout is known to refuse the tensor.
    fn admits(&self, encoding: &str, tensor: &SurfaceTensor) -> bool;
}

/// **The layout policies this build implements, by name.**
///
/// A snapshot stores the NAME and this resolves it. Injecting the
/// implementation and storing nothing would leave a second layout truth
/// next to the physical one 4b just closed: the policy that decided
/// which tensors were refused would be unrecorded, and a replay could
/// wire a different one without anything failing.
pub const PACK_LAYOUT_ADMISSION: &str = "pack-layout-admission/v1";
pub const NO_LAYOUT_CONSTRAINT: &str = "no-layout-constraint/v1";

/// Resolve a declared layout policy to the one implementation of it.
///
/// Deterministic and total over what this build knows; an unknown name
/// is refused rather than defaulted, because defaulting would silently
/// re-answer every refusal the record was built under.
pub fn layout_admission(id: &str) -> Result<&'static dyn LayoutAdmission, VindexError> {
    match id {
        PACK_LAYOUT_ADMISSION => Ok(&PackLayoutAdmission),
        NO_LAYOUT_CONSTRAINT => Ok(&NoLayoutConstraint),
        other => Err(VindexError::Parse(format!(
            "the record was built under layout policy `{other}`, which this build does not \
             implement — resolving it as any other policy would re-answer every layout \
             refusal the stored states were resolved under"
        ))),
    }
}

/// Declares no constraint for any encoding.
///
/// For encodings whose layout rules live elsewhere, and for tests that
/// are about map resolution rather than about layouts.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoLayoutConstraint;

impl LayoutAdmission for NoLayoutConstraint {
    fn admits(&self, _encoding: &str, _tensor: &SurfaceTensor) -> bool {
        true
    }
}

/// The constraint REPRESENT's own NVFP4 pack enforces.
///
/// Answers for `DTYPE_NVFP4` by asking [`PackLayout::derive`] — the same
/// call the compiler makes, so the two cannot drift — and declares
/// nothing about any other encoding.
#[derive(Debug, Clone, Copy, Default)]
pub struct PackLayoutAdmission;

impl LayoutAdmission for PackLayoutAdmission {
    fn admits(&self, encoding: &str, tensor: &SurfaceTensor) -> bool {
        if encoding != DTYPE_NVFP4 {
            return true;
        }
        PackLayout::derive(&tensor.shape, &tensor.tensor).is_ok()
    }
}

/// One tensor's resolved decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDecision {
    pub object: String,
    pub tensor: String,
    pub encoding: ResolvedEncoding,
}

impl ResolvedDecision {
    /// The canonical record identity reads: the pair, then what is
    /// presented. The *fact* — protected versus refused — is
    /// deliberately absent.
    fn canonical(&self) -> String {
        format!(
            "{}{FIELD}{}{FIELD}{}",
            self.object,
            self.tensor,
            self.encoding.effective()
        )
    }

    /// The canonical record with the fact retained — what the search,
    /// as opposed to the physical digest, is entitled to distinguish.
    fn canonical_full(&self) -> String {
        format!(
            "{}{FIELD}{}{FIELD}{}{FIELD}{}",
            self.object,
            self.tensor,
            self.encoding.effective(),
            self.encoding.fact()
        )
    }
}

/// **The resolved decision vector: one entry per surface tensor, in
/// surface order.**
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ResolvedDecisionVector {
    decisions: Vec<ResolvedDecision>,
}

impl ResolvedDecisionVector {
    pub fn decisions(&self) -> &[ResolvedDecision] {
        &self.decisions
    }

    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    pub fn get(&self, object: &str, tensor: &str) -> Option<&ResolvedEncoding> {
        self.decisions
            .iter()
            .find(|d| d.object == object && d.tensor == tensor)
            .map(|d| &d.encoding)
    }

    /// How many tensors this state actually compiles.
    pub fn compiled(&self) -> usize {
        self.decisions
            .iter()
            .filter(|d| d.encoding.is_compiled())
            .count()
    }

    /// Tensors the map wanted and the layout would not take. Reported
    /// rather than folded into "source", because "protected by policy"
    /// and "refused by the layout" are different facts and a report that
    /// cannot tell them apart is useless.
    pub fn layout_refused(&self) -> Vec<&ResolvedDecision> {
        self.decisions
            .iter()
            .filter(|d| matches!(d.encoding, ResolvedEncoding::LayoutRefused { .. }))
            .collect()
    }

    /// The canonical form the PHYSICAL digest reads: what is presented,
    /// with protection and layout refusal collapsed.
    pub(crate) fn canonical(&self) -> String {
        self.decisions
            .iter()
            .map(ResolvedDecision::canonical)
            .collect::<Vec<_>>()
            .join(&RECORD.to_string())
    }

    /// The canonical form the REALIZATION digest reads: every decision
    /// with the fact that produced it, nothing collapsed.
    ///
    /// Two vectors can share [`Self::canonical`] and differ here. That
    /// is not a defect of the physical digest — evidence may deduplicate
    /// more aggressively than search may, and this is the form that
    /// keeps the difference available to the action generator.
    pub(crate) fn canonical_full(&self) -> String {
        self.decisions
            .iter()
            .map(ResolvedDecision::canonical_full)
            .collect::<Vec<_>>()
            .join(&RECORD.to_string())
    }
}

/// **Resolve a map against a surface.**
///
/// Role first, then exceptions, then the layout — the same order the
/// compiler applies, because a state that resolved them in another order
/// would describe a container nobody can build.
pub fn resolve(
    map: &PrecisionMap,
    surface: &TensorSurface,
    layout: &dyn LayoutAdmission,
) -> ResolvedDecisionVector {
    let decisions = surface
        .entries()
        .iter()
        .map(|t| {
            let encoding = match map.resolve(t.role, &t.tensor) {
                Precision::Source => ResolvedEncoding::Source,
                Precision::Compiled(enc) if layout.admits(enc, t) => {
                    ResolvedEncoding::Compiled(enc.to_string())
                }
                Precision::Compiled(enc) => ResolvedEncoding::LayoutRefused {
                    encoding: enc.to_string(),
                },
            };
            ResolvedDecision {
                object: t.object.clone(),
                tensor: t.tensor.clone(),
                encoding,
            }
        })
        .collect();
    ResolvedDecisionVector { decisions }
}
