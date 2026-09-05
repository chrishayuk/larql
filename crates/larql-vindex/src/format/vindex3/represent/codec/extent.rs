//! How much of a representation, and what that much certifies.
//!
//! Today every codec is terminal: one extent, depth 0, the whole encoding.
//! A progressive codec — `R = R_0 + Δ_1 + … + Δ_n` — exposes one extent per
//! admissible prefix, each with its own cost and its own error bound, and
//! residency can then ask for the cheapest extent that satisfies a
//! requirement rather than for a named dtype. The dimension is in the
//! selection contract now, while every implementation answers `depth 0`,
//! because retrofitting it into stored variants later is the expensive
//! version of the same change.

/// Bits in a byte, for bits-per-weight arithmetic.
pub const BITS_PER_BYTE: f64 = 8.0;

/// A prefix of a representation. `depth` counts refinements past the base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepresentationExtent {
    pub depth: u32,
}

impl RepresentationExtent {
    /// The whole of a terminal representation, and the base of a
    /// progressive one.
    pub const TERMINAL: Self = Self { depth: 0 };

    pub const fn at_depth(depth: u32) -> Self {
        Self { depth }
    }
}

/// A reconstruction bound a codec is prepared to certify for an extent.
///
/// Absent means "measured, not declared": the K-quant and FP4 codecs this
/// build ships have their reconstruction error measured per encoder and
/// per tensor, and a number stated here would promote one measurement to
/// a property of the format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorRadius {
    /// Relative RMS of `decode(encode(x)) - x` over the codec's domain.
    pub relative_rms: f64,
}

/// What one extent costs and what it certifies.
///
/// Deliberately says nothing about *fidelity to a source*: that is a
/// property of an instance — set at extraction from provenance and carried
/// by the stored variant — not of the encoding. A native MXFP4 checkpoint
/// stored as MXFP4 is source-exact; the same bytes compiled from bf16 are
/// approximate; the codec is the same in both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtentCertificate {
    pub extent: RepresentationExtent,
    /// Asymptotic stored bits per weight at this extent, block overheads
    /// included and whole-tensor scales amortised away.
    pub bits_per_weight: f64,
    pub radius: Option<ErrorRadius>,
}

impl ExtentCertificate {
    /// The one certificate a terminal codec carries.
    pub const fn terminal(bits_per_weight: f64) -> Self {
        Self {
            extent: RepresentationExtent::TERMINAL,
            bits_per_weight,
            radius: None,
        }
    }
}
