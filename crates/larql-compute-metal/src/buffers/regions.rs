//! Registered weight regions — zero-copy Metal buffers over whole mmaps.
//!
//! The per-expert MoE decode previously memcpy'd every selected expert's
//! bytes into staging buffers each layer each token (GPT-OSS: top-4 ×
//! ~22 MB × 24 layers ≈ 2.1 GB of CPU memcpy per token — the dominant
//! decode cost once attention moved to the GPU). Registering each
//! layer-weights mmap ONCE as a `newBufferWithBytesNoCopy` buffer lets
//! the dispatch bind an expert as a byte offset into the shared buffer:
//! no staging, no duplication, no per-token CPU bandwidth.
//!
//! A region must be page-aligned at its base — true of every mmap by
//! construction — and lives for the process, the same contract
//! [`BufferCache::get_bytes`] already imposes on cached weight slices.

use metal::{Buffer, MTLResourceOptions};

use super::{BufferCache, PAGE_SIZE};

/// Byte alignment a weight binding's offset must satisfy.
///
/// **Measured, not taken from a feature table** — see
/// `the_binding_alignment_requirement_is_measured_not_assumed`, which
/// dispatches the real grouped kernel through a registered file mapping
/// at each residue: only an ODD offset misreads; 2, 4, 8 and 12 all
/// agree with the staged-copy arm bit for bit.
///
/// Four rather than that measured two, because two is bf16's own
/// element size and a weight buffer is also bound as `float` elsewhere
/// in this crate. Four covers every element type a weight binding uses
/// (bf16's `ushort`, a quantised format's `uchar`, `float`) without
/// demanding an alignment a real container may not have: at sixteen,
/// the Kimi container's 94 GB expert segment — whose payload starts at
/// 2,438,284, divisible by four and not by sixteen — would have had to
/// be rewritten to change nothing.
pub const WEIGHT_BINDING_ALIGN: usize = 4;

/// One registered region: `[start, start + len)` in host memory, plus the
/// no-copy Metal buffer that aliases it.
pub(super) struct Region {
    pub start: usize,
    pub len: usize,
    pub buf: Buffer,
}

impl BufferCache {
    /// Register `region` for zero-copy sub-slice resolution.
    ///
    /// Returns `false` (and registers nothing) when the base pointer is
    /// not page-aligned — the caller falls back to staged copies. Calling
    /// again with an already-registered base is a cheap no-op.
    ///
    /// The Metal buffer length is `region.len` rounded UP to the page
    /// size: `newBufferWithBytesNoCopy` requires a page-multiple length,
    /// and an mmap maps whole pages, so the bytes between `len` and the
    /// page boundary are readable (zero-filled by the kernel) even though
    /// they are past the slice. That tail is never bound at an offset —
    /// resolution is bounded by `len` — it only pads the allocation.
    ///
    /// # Contract
    ///
    /// `region` must point into an allocation that is page-granular
    /// (mmap-backed), never moves, and outlives this cache — the same
    /// stability contract as [`Self::get_bytes`].
    pub fn register_region(&self, region: &[u8]) -> bool {
        let start = region.as_ptr() as usize;
        if !start.is_multiple_of(PAGE_SIZE) || region.is_empty() {
            return false;
        }
        let mut regions = self.regions.lock().unwrap();
        if regions.iter().any(|r| r.start == start) {
            return true;
        }
        let rounded = region.len().div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let buf = self.device.new_buffer_with_bytes_no_copy(
            region.as_ptr() as *mut std::ffi::c_void,
            rounded as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        regions.push(Region {
            start,
            len: region.len(),
            buf,
        });
        true
    }

    /// Declare every registered region resident, per the selected arm.
    ///
    /// Idempotent and safe to call after each registration batch: the set is
    /// rebuilt from the current region list, committed, optionally
    /// `requestResidency`-ed, and attached to `queue`. Under
    /// [`ResidencyArm::Implicit`] it does nothing at all, so arm A is
    /// byte-identical to the pre-existing code path.
    pub fn seal_residency(&self, queue: &metal::CommandQueue) {
        use super::residency::{ResidencyArm, ResidencySet};
        let arm = ResidencyArm::from_env();
        if arm == ResidencyArm::Implicit {
            return;
        }
        let regions = self.regions.lock().unwrap();
        if regions.is_empty() {
            return;
        }
        let Some(mut set) = ResidencySet::new(&self.device) else {
            // Pre-macOS-15 runtime: implicit residency stays, which is
            // correct. Say so once rather than silently reporting arm B
            // numbers that are really arm A.
            eprintln!("[residency] MTLResidencySet unavailable — running implicit residency");
            return;
        };
        for r in regions.iter() {
            set.add_buffer(&r.buf);
        }
        set.commit();
        if arm == ResidencyArm::QueueSetRequested {
            set.request_residency();
        }
        let attached = set.add_to_queue(queue);
        eprintln!(
            "[residency] arm={:?} regions={} attached_to_queue={}",
            arm,
            regions.len(),
            attached
        );
        *self.residency.lock().unwrap() = Some(set);
    }

    /// Resolve `sub` to `(buffer, byte_offset)` if it lies wholly inside a
    /// registered region. `None` → the caller stages a copy instead.
    ///
    /// **A misaligned offset MISSES rather than resolving.** A kernel
    /// binding `device const ushort*` at an odd byte offset reads
    /// garbage on Metal with no error and a successful command buffer —
    /// the exact silent-wrong-answer shape this codebase keeps paying
    /// for. Missing costs a staged copy, which is slow and correct; the
    /// caller that cares about the cost is told why by whoever knows the
    /// alignment's cause, which is the container, not this cache.
    pub fn resolve_region(&self, sub: &[u8]) -> Option<(Buffer, u64)> {
        if sub.is_empty() {
            return None;
        }
        let p = sub.as_ptr() as usize;
        let regions = self.regions.lock().unwrap();
        for r in regions.iter() {
            if p >= r.start && p + sub.len() <= r.start + r.len {
                let offset = p - r.start;
                if !offset.is_multiple_of(WEIGHT_BINDING_ALIGN) {
                    return None;
                }
                return Some((r.buf.clone(), offset as u64));
            }
        }
        None
    }

    /// Number of registered regions (test/diagnostic surface).
    pub fn region_count(&self) -> usize {
        self.regions.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests;
