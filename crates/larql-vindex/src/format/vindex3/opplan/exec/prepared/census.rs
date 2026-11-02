//! Allocation and residency censuses.

use super::super::weights::LoadedWeight;

#[allow(unused_imports)]
use super::*;

/// Where a prepared image's allocations LAND, as distinct from how many
/// bytes they hold.
///
/// CPU-PERF-1 found the isolated kernel harness predicts real bf16
/// projection to +0.7% and misses real Q8 by 7.9%, and CPU-PERF-2 ruled
/// out machine state. What is left is the resident representation itself,
/// and the two formats differ in more than bytes: bf16 lands in
/// page-aligned `AlignedBytes`, one allocation per matrix, while Q8 uses
/// ordinary heap vectors and TWO allocations per matrix.
///
/// This measures that difference before anything is changed on the
/// strength of it — a large `Vec` may already receive a page-aligned VM
/// region, in which case "align it" would be an intervention with nothing
/// to intervene on.
#[derive(Default, Clone, Copy, Debug)]
pub struct AllocationCensus {
    pub allocations: usize,
    pub page_aligned: usize,
    /// The coarsest alignment every allocation shares, in bytes.
    pub common_alignment: usize,
    pub bytes: usize,
}

impl AllocationCensus {
    pub(super) fn add(&mut self, address: usize, bytes: usize) {
        self.allocations += 1;
        self.bytes += bytes;
        if address.is_multiple_of(super::super::weights::DEVICE_PAGE_ALIGN) {
            self.page_aligned += 1;
        }
        let align = 1usize << address.trailing_zeros().min(30);
        self.common_alignment = if self.allocations == 1 {
            align
        } else {
            self.common_alignment.min(align)
        };
    }
}

/// A prepared image's mappings at one moment: their address space and the
/// pages of it physically resident — the two figures a mapping must never
/// be reported as one of.
///
/// `resident_bytes` covers only the regions whose residency was READ;
/// `failed_regions` counts the rest, which are never summed as zero. Use
/// [`Self::complete_resident_bytes`] wherever the total must stand for
/// the whole image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MappedResidency {
    /// Every mapped region, read or not.
    pub regions: usize,
    /// Address space over every mapped region.
    pub mapped_bytes: u64,
    /// Resident bytes over the regions read successfully.
    pub resident_bytes: u64,
    /// Regions whose residency could not be read.
    pub failed_regions: usize,
    /// Address space over the regions read successfully.
    pub mapped_bytes_read: u64,
}

impl MappedResidency {
    /// Regions read successfully. Saturates so an inconsistent value
    /// cannot underflow; it does NOT make one valid — see
    /// [`Self::is_consistent`], which every reader of a total must ask.
    pub fn read_regions(&self) -> usize {
        self.regions.saturating_sub(self.failed_regions)
    }

    /// Whether the counts can describe one observation: no more failed
    /// regions than regions, and no more address space read than mapped.
    /// A summary derived by [`super::ResidencyObservation::summary`] is
    /// consistent by construction; the fields are public, so a value
    /// built any other way may not be.
    pub fn is_consistent(&self) -> bool {
        self.failed_regions <= self.regions && self.mapped_bytes_read <= self.mapped_bytes
    }

    /// The resident total as a figure for the WHOLE image: `Some` only
    /// when at least one region was observed, none failed, and the counts
    /// are consistent. An observation of nothing witnesses nothing, one
    /// with a failed region is a partial sum, and an inconsistent one
    /// witnesses nothing reliable.
    pub fn complete_resident_bytes(&self) -> Option<u64> {
        (self.is_consistent() && self.regions > 0 && self.failed_regions == 0)
            .then_some(self.resident_bytes)
    }
}

/// One site's bytes, split by whether the loader widened — and, apart
/// from both, what it MAPPED: address space over the container's own
/// segment, resident only as touched, never counted as committed.
#[derive(Default, Clone, Copy, Debug)]
pub struct SiteResidency {
    /// Bytes held as f32 — doubled, when the checkpoint stored bf16.
    pub widened_f32: usize,
    /// Bytes held exactly as the checkpoint holds them, committed.
    pub compact: usize,
    /// Bytes bound as a mapping of the container's segment.
    pub mapped: usize,
}

impl SiteResidency {
    pub(super) fn add(&mut self, w: &LoadedWeight) {
        let mapped = w.mapped_bytes();
        if mapped > 0 {
            self.mapped += mapped;
        } else if w.is_widened_f32() {
            self.widened_f32 += w.resident_bytes();
        } else {
            self.compact += w.resident_bytes();
        }
    }

    /// Committed bytes: what the process holds, mappings excluded.
    pub fn total(&self) -> usize {
        self.widened_f32 + self.compact
    }
}

/// Where a prepared image's bytes are, and in which representation.
#[derive(Default, Clone, Copy, Debug)]
pub struct ResidencyCensus {
    pub embedding: SiteResidency,
    pub attention: SiteResidency,
    pub delta: SiteResidency,
    pub ffn: SiteResidency,
    pub head: SiteResidency,
    /// Norms, biases, the depthwise convolution, gate biases — always
    /// f32, and small enough that widening them costs nothing worth
    /// recovering.
    pub glue: SiteResidency,
}

impl ResidencyCensus {
    /// Every site, in the order a decode reads them.
    pub fn sites(&self) -> [(&'static str, SiteResidency); 6] {
        [
            ("embedding", self.embedding),
            ("attention", self.attention),
            ("delta", self.delta),
            ("ffn", self.ffn),
            ("head", self.head),
            ("glue", self.glue),
        ]
    }

    pub fn total(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.total()).sum()
    }

    pub fn widened_f32(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.widened_f32).sum()
    }

    /// Address space held as mappings, across every site.
    pub fn mapped(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.mapped).sum()
    }

    pub fn compact(&self) -> usize {
        self.sites().iter().map(|(_, s)| s.compact).sum()
    }
}
