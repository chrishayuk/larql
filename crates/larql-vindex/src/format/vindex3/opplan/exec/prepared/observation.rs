//! One residency observation of a prepared image, and what may be read
//! from it.
//!
//! Every mapped region is read ONCE per observation, in a fixed order,
//! and both the aggregate and the per-region report are derived from that
//! single capture — two views computed from two separate scans could
//! disagree only because the page cache moved between them. A scan over
//! many regions is still not atomic: a region read late can reflect
//! later state than one read early, which is why the scan's bounds and
//! order are kept.
//!
//! A failed read stays a failure all the way out. It is never summed as
//! zero, an observation with one is never an admission-quality total,
//! and a delta never compares a successful reading with a failed one.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime};

use super::census::MappedResidency;
use crate::format::vindex3::represent::physical::{ResidencyReadError, WeightRegion};

/// A store an observation's regions point into, recorded once per
/// observation so a region carries an index instead of an owned name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedStore {
    /// The store's id: the operand OBJECT name, not a file.
    pub id: String,
    /// The file the store maps, when it maps one. Diagnostic only: it
    /// changes when the container is cloned or moved, so it is never a
    /// region's identity when a declared segment is available.
    pub source_path: Option<PathBuf>,
    /// The segment path relative to its container, as the container's
    /// index declared it.
    pub declared_segment: Option<String>,
}

impl ObservedStore {
    /// The segment half of a region key for regions in this store.
    pub fn segment_identity(&self) -> SegmentIdentity {
        match (&self.declared_segment, &self.source_path) {
            (Some(declared), _) => SegmentIdentity::Declared(declared.clone()),
            (None, Some(path)) => SegmentIdentity::Undeclared(path.clone()),
            (None, None) => SegmentIdentity::Owned,
        }
    }
}

/// One mapped region's reading within an observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionResidency {
    /// Position in the scan: regions are read in this order.
    pub scan_index: usize,
    /// Index into [`ResidencyObservation::stores`].
    pub store: usize,
    /// Offset in the store's backing; for a mapped store, a file offset.
    pub file_offset: u64,
    pub len: u64,
    pub resident: Result<u64, ResidencyReadError>,
}

/// One scan of every mapped region of an image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyObservation {
    pub started_wall: SystemTime,
    /// Monotonic nanoseconds since this process's first observation.
    pub started_mono_ns: u64,
    pub ended_wall: SystemTime,
    pub ended_mono_ns: u64,
    /// The page size the reads used, when the OS reported one.
    pub page_size: Option<u64>,
    pub stores: Vec<ObservedStore>,
    /// In scan order: `regions[i].scan_index == i`.
    pub regions: Vec<RegionResidency>,
}

/// Which segment a region lives in, as far as its identity is concerned.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SegmentIdentity {
    /// The container-relative path the container's index declares. It
    /// does not change when the container is cloned or moved, so the key
    /// SURVIVES cloning. That is not a licence to compare: two different
    /// containers with one layout give the same keys, and a segment
    /// rewritten in place keeps them. Regions from different processes may
    /// be compared only when both records carry equal, non-null container
    /// identity (`index_json_sha256`), checked by the procedure that pairs
    /// them; the key never authorises the comparison on its own.
    Declared(String),
    /// A mapped file no container declared, keyed by its absolute path:
    /// it pairs only with observations of that same path.
    Undeclared(PathBuf),
    /// Owned bytes: no file at all.
    Owned,
}

impl SegmentIdentity {
    /// Whether this identity is the same for every clone of a container —
    /// a fact about the key, never on its own a reason to pair two
    /// containers' records (see [`Self::Declared`]).
    pub fn is_portable(&self) -> bool {
        matches!(self, Self::Declared(_))
    }
}

/// A region's identity across observations: the operand object whose
/// store holds it, the segment that store is, the region's file offset
/// and its length. The container itself is identified once per record,
/// by its index digest, not per region.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionKey {
    pub object: String,
    pub segment: SegmentIdentity,
    pub file_offset: u64,
    pub len: u64,
}

/// The monotonic origin every observation in this process measures from.
fn mono_origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn mono_ns_now() -> u64 {
    mono_origin().elapsed().as_nanos() as u64
}

impl ResidencyObservation {
    /// Read every region in `regions`, once each, in order.
    pub fn capture<'a>(regions: impl IntoIterator<Item = &'a WeightRegion>) -> Self {
        // Fix the origin before the clock is read for this scan.
        let _ = mono_origin();
        let started_wall = SystemTime::now();
        let started_mono_ns = mono_ns_now();
        let page_size = crate::format::vindex3::represent::physical::page_size();
        let mut stores: Vec<ObservedStore> = Vec::new();
        let mut index_of: BTreeMap<(String, Option<PathBuf>, Option<String>), usize> =
            BTreeMap::new();
        let mut out = Vec::new();
        for (scan_index, region) in regions.into_iter().enumerate() {
            let key = (
                region.store_id().to_string(),
                region.source_path().map(|p| p.to_path_buf()),
                region.declared_segment().map(str::to_string),
            );
            let store = *index_of.entry(key.clone()).or_insert_with(|| {
                stores.push(ObservedStore {
                    id: key.0,
                    source_path: key.1,
                    declared_segment: key.2,
                });
                stores.len() - 1
            });
            out.push(RegionResidency {
                scan_index,
                store,
                file_offset: region.store_offset(),
                len: region.len(),
                resident: region.residency(),
            });
        }
        Self {
            started_wall,
            started_mono_ns,
            ended_wall: SystemTime::now(),
            ended_mono_ns: mono_ns_now(),
            page_size,
            stores,
            regions: out,
        }
    }

    /// The aggregate, derived from this capture alone.
    pub fn summary(&self) -> MappedResidency {
        let mut out = MappedResidency::default();
        for r in &self.regions {
            out.regions += 1;
            out.mapped_bytes += r.len;
            match r.resident {
                Ok(bytes) => {
                    out.resident_bytes += bytes;
                    out.mapped_bytes_read += r.len;
                }
                Err(_) => out.failed_regions += 1,
            }
        }
        out
    }

    /// The first failed region, for naming it.
    pub fn first_failure(&self) -> Option<(&RegionResidency, ResidencyReadError)> {
        self.regions
            .iter()
            .find_map(|r| r.resident.err().map(|e| (r, e)))
    }

    /// A region's identity across observations: built from the declared
    /// segment wherever the container declared one, never from where the
    /// container happens to sit on disk.
    pub fn key_of(&self, region: &RegionResidency) -> RegionKey {
        let store = &self.stores[region.store];
        RegionKey {
            object: store.id.clone(),
            segment: store.segment_identity(),
            file_offset: region.file_offset,
            len: region.len,
        }
    }
}

/// The change between two observations, over the regions read
/// successfully in BOTH.
///
/// Both observations are assumed to be of ONE container: this matches
/// regions by key and checks no container identity. The curve calls it
/// on two observations of one prepared image in one process. A caller
/// comparing observations from different processes or clones must first
/// establish that both name the same container (equal, non-null
/// `index_json_sha256`); a matching key is not that evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidencyDelta {
    /// Regions present and read in both observations.
    pub comparable: usize,
    /// Sum of the positive per-region changes, in bytes.
    pub increase: u64,
    /// Sum of the magnitudes of the negative per-region changes.
    pub decrease: u64,
    /// Regions left out: failed in either, or present in only one.
    pub excluded: usize,
}

impl ResidencyDelta {
    /// Per-region signed change, from `before` to `after`.
    pub fn between(before: &ResidencyObservation, after: &ResidencyObservation) -> Self {
        let earlier: BTreeMap<RegionKey, Result<u64, ResidencyReadError>> = before
            .regions
            .iter()
            .map(|r| (before.key_of(r), r.resident))
            .collect();
        let mut out = Self::default();
        let mut matched = 0usize;
        for r in &after.regions {
            match (earlier.get(&after.key_of(r)), r.resident) {
                (Some(Ok(was)), Ok(now)) => {
                    matched += 1;
                    out.comparable += 1;
                    let d = now as i64 - *was as i64;
                    if d >= 0 {
                        out.increase += d as u64;
                    } else {
                        out.decrease += d.unsigned_abs();
                    }
                }
                (Some(_), _) => {
                    matched += 1;
                    out.excluded += 1;
                }
                (None, _) => out.excluded += 1,
            }
        }
        // Regions only the earlier observation had.
        out.excluded += earlier.len().saturating_sub(matched);
        out
    }

    /// The signed total, or `None` when no region was comparable — never
    /// a zero that was not measured.
    pub fn signed_total(&self) -> Option<i64> {
        (self.comparable > 0).then(|| self.increase as i64 - self.decrease as i64)
    }
}
