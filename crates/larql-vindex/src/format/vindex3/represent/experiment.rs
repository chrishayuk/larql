//! **The evidence a REPRESENT decision is made from.**
//!
//! REPRESENT adds a lossy alternative encoding beside the canonical
//! bytes. Which encoding to add is not a property of the format — it is
//! a property of *this component, of this model, on this hardware*, and
//! the only honest way to hold that is as a measurement with provenance.
//!
//! So a representation choice is recorded as an experiment rather than
//! asserted as a policy: what was measured, on what, against which
//! baseline, and — critically — **which fields were not measured**. A
//! record with a blank quality field is a record that cannot be used to
//! justify a quality claim, which is the whole point of making the
//! blank representable.
//!
//! ```text
//! RepresentationExperiment
//!   MODEL          Kimi-Linear-48B-A3B
//!   COMPONENT      RoutedExpertBank(layer 1)
//!   SOURCE         BF16          TARGET  Q6_K
//!   HARDWARE       Apple M3 Max
//!   BASELINE_TPS   37.33
//!   ...
//! ```

use serde::{Deserialize, Serialize};

use super::map::Exception;
use super::policy::Role;
use super::quality::QualityEvidence;

/// WHICH semantic objects a piece of evidence is about.
///
/// Deliberately the same selector [`Exception`] uses — role, optional
/// projection, optional depth range — because evidence and policy have
/// to speak one vocabulary. A record keyed by free text could describe
/// a region no precision map is able to govern, which is how a
/// quantisation benchmark ends up running beside REPRESENT instead of
/// feeding it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleScope {
    pub role: Role,
    /// Projection this evidence covers, e.g. `down_proj`. `None` = the
    /// whole role.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<String>,
    /// Inclusive depth range. `None` = every depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<(u32, u32)>,
}

impl RoleScope {
    pub fn role(role: Role) -> Self {
        Self {
            role,
            projection: None,
            layers: None,
        }
    }

    pub fn projection(mut self, p: impl Into<String>) -> Self {
        self.projection = Some(p.into());
        self
    }

    pub fn layers(mut self, lo: u32, hi: u32) -> Self {
        self.layers = Some((lo, hi));
        self
    }

    /// The precision-map exception this scope would install for
    /// `encoding` — the bridge from measurement to policy.
    pub fn as_exception(&self, encoding: impl Into<String>) -> Exception {
        Exception {
            projection: self.projection.clone(),
            layers: self.layers,
            encoding: Some(encoding.into()),
        }
    }
}

/// Independent facts about one (scope, representation) pair.
///
/// Not one boolean. A representation can be encodable but have no
/// kernel, dispatch correctly but never have been timed, be fast and
/// have no quality evidence at all. Collapsing those into "supported"
/// is what lets a backend's CAPABILITY read as authority to choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RepresentationStatus {
    /// An encoder exists that can produce these bytes.
    pub represented: bool,
    /// The bytes exist for this operand.
    pub available: bool,
    /// A backend has a kernel that reads them. **Capability, never
    /// authority** — this being true says nothing about whether the
    /// representation should be chosen.
    pub backend_supported: bool,
    /// It dispatched and produced finite, correct values.
    pub runnable: bool,
    /// Its throughput was measured through its own kernel.
    pub measured: bool,
    /// A precision map names it. Only [`super::selection`] sets this,
    /// and only from evidence.
    pub selected: bool,
}

impl RepresentationStatus {
    /// Everything the ladder can assert on its own.
    ///
    /// Quality is deliberately NOT here. It cannot be a boolean anyone
    /// sets: it has to name the gate it passed, so it lives on the
    /// evidence — see [`RepresentationExperiment::quality_proven_by`].
    ///
    /// Capability is absent too, on purpose: a kernel existing is a
    /// reason the choice is possible, never a reason it is right.
    pub fn ladder_complete(self) -> bool {
        self.represented && self.available && self.runnable && self.measured
    }
}

/// One measured (component, source, target) representation point.
///
/// Every quantitative field is `Option`, and absent means **not
/// measured** — never zero, never a default. A consumer that needs
/// `kl_p99` and finds `None` must decline to rank on quality rather
/// than treat it as perfect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepresentationExperiment {
    pub model: String,
    /// Which semantic objects this is evidence about, in the precision
    /// map's own selector vocabulary.
    pub scope: RoleScope,
    /// Free text naming the concrete thing measured, for a human reading
    /// the record — never the key anything resolves on.
    pub component: String,
    pub source: String,
    pub target: String,
    pub hardware: String,
    /// Bits per weight of `target`, from its block geometry.
    pub bits_per_weight: f64,

    /// What the component's bank occupies in each representation.
    pub source_bytes: u64,
    pub target_bytes: u64,

    // ── Throughput, measured on the GPU window ──
    pub baseline_tokens_per_second: Option<f64>,
    /// End-to-end tok/s with this representation in place. `None` until
    /// the representation is actually wired into the token loop — a
    /// projection is not a measurement and does not go in this field.
    pub result_tokens_per_second: Option<f64>,
    pub baseline_gpu_ms: Option<f64>,
    pub target_gpu_ms: Option<f64>,
    pub target_achieved_gb_per_s: Option<f64>,
    /// Speedup as a fraction of the byte ratio. 1.0 is bandwidth-bound;
    /// below it, dequantisation arithmetic is taking the difference.
    pub bandwidth_bound_fraction: Option<f64>,

    // ── Quality ──
    /// End-to-end relative RMS of the component's own output against the
    /// reference implementation's, with all of its projections
    /// re-represented. Evidence about the COMPONENT — never about the
    /// model's output distribution.
    pub component_rel_rms: Option<f64>,
    pub component_max_over_scale: Option<f64>,
    /// The logit-level bank and the gate judging it. `None` means no
    /// bank has been run, and no quality claim may rest on this record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<QualityEvidence>,

    /// The ladder: what is independently known about this pair.
    pub status: RepresentationStatus,

    /// How the numbers were produced: the gate that emitted them, the
    /// fixture, and anything a reader needs to reproduce or distrust it.
    pub provenance: Provenance,
}

/// Where a record came from, and what is known to be approximate about
/// it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// The test or tool that emitted the record.
    pub gate: String,
    pub fixture: String,
    /// Whether the target was dispatched by its OWN kernel, or through a
    /// decoded stand-in because no kernel exists on this convention.
    /// A simulated point may carry quality numbers but never throughput.
    pub native_kernel: bool,
    /// Free-text caveats a reader must see — the cost of a simulation
    /// carrier, a timer spread, a working set that fitted in cache.
    pub caveats: Vec<String>,
}

impl RepresentationExperiment {
    /// The byte ratio this representation buys.
    pub fn byte_ratio(&self) -> f64 {
        if self.target_bytes == 0 {
            return f64::NAN;
        }
        self.source_bytes as f64 / self.target_bytes as f64
    }

    /// The gate this record's quality claim rests on, if any.
    ///
    /// Component-level error is evidence about the component; it is not
    /// evidence about the model's output distribution. So a claim has to
    /// name a versioned gate it PASSED — "quality ok" with nothing
    /// behind it is unfalsifiable a month later, and indistinguishable
    /// from having passed a much weaker bar.
    pub fn quality_proven_by(&self) -> Option<&str> {
        self.quality.as_ref().and_then(|q| q.proven_by())
    }

    /// Whether this record may be used to justify a *quality* claim.
    pub fn supports_quality_claim(&self) -> bool {
        self.quality_proven_by().is_some()
    }

    /// Whether policy is ALLOWED to promote this to `selected`.
    pub fn promotable(&self) -> bool {
        self.status.ladder_complete()
            && self.supports_quality_claim()
            && self.supports_throughput_claim()
    }

    /// Whether this record may be used to justify a *throughput* claim.
    pub fn supports_throughput_claim(&self) -> bool {
        self.provenance.native_kernel && self.target_gpu_ms.is_some()
    }
}

#[cfg(test)]
mod tests;
