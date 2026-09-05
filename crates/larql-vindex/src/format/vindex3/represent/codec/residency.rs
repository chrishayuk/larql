//! What executing a representation makes resident — declared per
//! realization, never per codec.
//!
//! A codec has one mandatory realization, decode to f32, and any number
//! of direct ones. They do not share a residency story: stored Q4_K
//! through a direct kernel touches 4.5 bits a weight, and the same bytes
//! decoded and staged touch 32. So the declaration belongs to the
//! realization, and a planner that selects one selects its cost with it.
//! That is what closes the silent-fallback hole: "planned direct, kernel
//! unavailable, quietly decode and stage" is no longer one path with a
//! surprise in it — it is a different realization with a different
//! declared profile, and the trace can name both.

use super::capability::RequiredAccess;
use super::extent::BITS_PER_BYTE;
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;

/// Width of the canonical decode target.
pub const F32_WIDTH_BYTES: usize = std::mem::size_of::<f32>();

/// Whether the bytes executed are the bytes stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyClass {
    /// The stored bytes themselves, bound in place.
    Stored,
    /// The stored bytes, copied once for alignment; no value changes.
    Rebound,
    /// A widened image of the stored values, computed at load.
    TransientDecoded,
    /// A re-quantised image: the values resident are not the values
    /// stored. The only lossy residency, and it says so.
    TransientRequantised,
}

impl ResidencyClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Rebound => "rebound",
            Self::TransientDecoded => "transient-decoded",
            Self::TransientRequantised => "transient-requantised",
        }
    }

    /// Whether the resident values equal the stored values.
    pub const fn preserves_values(self) -> bool {
        !matches!(self, Self::TransientRequantised)
    }
}

/// Bytes a weight touches at serve time under one realization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidencyProfile {
    pub class: ResidencyClass,
    /// Bytes of resident image per logical weight.
    pub bytes_per_weight: f64,
}

impl ResidencyProfile {
    /// The canonical decode realization's profile: an f32 image.
    pub const DECODED_F32: Self = Self {
        class: ResidencyClass::TransientDecoded,
        bytes_per_weight: F32_WIDTH_BYTES as f64,
    };

    /// A direct realization over stored bytes at `bits_per_weight`.
    pub const fn stored(bits_per_weight: f64) -> Self {
        Self {
            class: ResidencyClass::Stored,
            bytes_per_weight: bits_per_weight / BITS_PER_BYTE,
        }
    }

    /// A direct realization over an aligned copy of the stored bytes.
    pub const fn rebound(bits_per_weight: f64) -> Self {
        Self {
            class: ResidencyClass::Rebound,
            bytes_per_weight: bits_per_weight / BITS_PER_BYTE,
        }
    }
}

/// Where a direct realization runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationBackend {
    /// This crate's CPU executor. Device backends declare their own
    /// realizations from the peer crate that owns the kernels.
    Cpu,
}

/// One direct realization a codec offers, with the cost it declares.
///
/// `plan` is positive evidence: the kernel exists because the plan names
/// it, and a codec cannot claim an acceleration the executor does not
/// have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Acceleration {
    pub backend: AccelerationBackend,
    pub plan: PhysicalProjectionPlan,
    pub residency: ResidencyProfile,
    /// What the kernel addresses; a plan needing more is refused by name.
    pub requires: RequiredAccess,
}

impl Acceleration {
    pub const fn cpu(plan: PhysicalProjectionPlan, residency: ResidencyProfile) -> Self {
        Self {
            backend: AccelerationBackend::Cpu,
            plan,
            residency,
            requires: RequiredAccess::RowRandom,
        }
    }
}
