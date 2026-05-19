//! Metal GPU buffer management — caching, zero-copy mmap, transient allocation.
//!
//! Weight buffers (mmap'd, constant) are cached by pointer address.
//! Transient buffers (Q8 input, activation, output) are allocated fresh each call.
//! Page-aligned mmap data uses newBufferWithBytesNoCopy (zero-copy GPU access).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;

use metal::*;

/// Cache key: (pointer address, byte length) of the source data.
type CacheKey = (usize, usize);

/// Apple Silicon page size (16KB).
const PAGE_SIZE: usize = 16384;

/// Buffer cache for Metal GPU buffers.
/// Weight matrices from mmap'd files have stable addresses — their GPU buffers
/// are created once and reused for all subsequent calls.
/// Scratch output buffers are pooled by size — `output()` returns an existing
/// buffer of the requested size rather than calling `device.new_buffer` each
/// time. This eliminates ~21 GPU allocations per decode step which were the
/// dominant CPU overhead for large models (31B: 86KB × 21 = ~200ms/token).
pub struct BufferCache {
    device: Device,
    cache: Mutex<HashMap<CacheKey, Buffer>>,
    /// Pool of pre-allocated scratch buffers keyed by byte length.
    /// Each entry is a Vec of available (not currently in use) buffers.
    /// Grows on first use; reused on subsequent decode steps.
    scratch_pool: Mutex<HashMap<u64, Vec<Buffer>>>,
}

impl BufferCache {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            cache: Mutex::new(HashMap::new()),
            scratch_pool: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a cached GPU buffer for f32 data.
    /// Uses zero-copy for page-aligned mmap data, copies otherwise.
    /// Empty slices (absent optional weight arrays such as Q4_K scale vectors)
    /// return a minimal 4-byte stub — Metal rejects zero-length allocations.
    pub fn get_f32(&self, data: &[f32]) -> Buffer {
        if data.is_empty() {
            // All empty-slice calls share the same stub key so the stub is
            // allocated once and reused.
            let stub_key: CacheKey = (0, 0);
            let mut cache = self.cache.lock().unwrap();
            if let Some(buf) = cache.get(&stub_key) {
                return buf.clone();
            }
            let buf = self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
            cache.insert(stub_key, buf.clone());
            return buf;
        }

        let key: CacheKey = (data.as_ptr() as usize, data.len());
        let mut cache = self.cache.lock().unwrap();
        if let Some(buf) = cache.get(&key) {
            return buf.clone();
        }

        let bytes = data.len() * 4;
        let ptr = data.as_ptr() as *const c_void;

        let buf = if Self::is_page_aligned(ptr, bytes) {
            self.device.new_buffer_with_bytes_no_copy(
                ptr as *mut c_void,
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.device.new_buffer_with_data(
                ptr,
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        cache.insert(key, buf.clone());
        buf
    }

    /// Get or create a cached GPU buffer for raw byte data (Q4 packed weights).
    /// Uses zero-copy for page-aligned mmap data.
    /// Empty slices return a minimal 4-byte stub (see `get_f32` for rationale).
    pub fn get_bytes(&self, data: &[u8]) -> Buffer {
        if data.is_empty() {
            let stub_key: CacheKey = (1, 0);
            let mut cache = self.cache.lock().unwrap();
            if let Some(buf) = cache.get(&stub_key) {
                return buf.clone();
            }
            let buf = self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
            cache.insert(stub_key, buf.clone());
            return buf;
        }

        let key: CacheKey = (data.as_ptr() as usize, data.len());
        let mut cache = self.cache.lock().unwrap();
        if let Some(buf) = cache.get(&key) {
            return buf.clone();
        }

        let ptr = data.as_ptr() as *const c_void;
        let bytes = data.len();

        let buf = if Self::is_page_aligned(ptr, bytes) {
            self.device.new_buffer_with_bytes_no_copy(
                ptr as *mut c_void,
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            self.device.new_buffer_with_data(
                ptr,
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        cache.insert(key, buf.clone());
        buf
    }

    /// Create a transient buffer (NOT cached — contents change each call).
    /// Used for Q8 input vectors, activations, and output buffers.
    pub fn transient_from_i8(&self, data: &[i8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a transient buffer from f32 data.
    pub fn transient_from_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a transient buffer from raw bytes. Used for staging concatenated
    /// Q4K expert weight slices before a GPU matvec dispatch.
    pub fn transient_from_bytes(&self, data: &[u8]) -> Buffer {
        if data.is_empty() {
            return self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);
        }
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create an empty output buffer of given byte size.
    /// Return a scratch output buffer of at least `bytes` bytes.
    /// Reuses a pooled buffer when one of the exact size is available,
    /// otherwise allocates once and adds it to the pool for future calls.
    /// Callers treat the buffer as write-before-read scratch space.
    pub fn output(&self, bytes: u64) -> Buffer {
        let mut pool = self.scratch_pool.lock().unwrap();
        if let Some(buf) = pool.entry(bytes).or_default().pop() {
            return buf;
        }
        self.device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
    }

    /// Return a scratch buffer to the pool after it is no longer needed.
    /// Must be called after `cmd.wait_until_completed()` — the GPU must
    /// have finished writing before the buffer is recycled.
    pub fn recycle(&self, buf: Buffer) {
        let bytes = buf.length();
        self.scratch_pool
            .lock()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(buf);
    }

    /// Number of cached buffers (for diagnostics).
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_page_aligned(ptr: *const c_void, bytes: usize) -> bool {
        (ptr as usize).is_multiple_of(PAGE_SIZE) && bytes.is_multiple_of(PAGE_SIZE)
    }
}

/// RAII guard that returns scratch buffers to the pool when dropped.
/// Create one per decode step; it holds clones of all output buffers allocated
/// via `BufferCache::output`. Dropping the guard (at any function-exit path,
/// including early returns) recycles all held buffers automatically.
///
/// **Invariant**: only drop after `cmd.wait_until_completed()` so the GPU has
/// finished writing. The decode functions satisfy this: the guard is created
/// early, but by the time it drops the final command buffer has been waited on.
pub struct ScratchGuard<'a> {
    bufs: Vec<Buffer>,
    cache: &'a BufferCache,
}

impl<'a> ScratchGuard<'a> {
    pub fn new(cache: &'a BufferCache) -> Self {
        Self {
            bufs: Vec::new(),
            cache,
        }
    }

    /// Track a buffer for recycling. Call once per `BufferCache::output()` call.
    pub fn track(&mut self, buf: &Buffer) {
        self.bufs.push(buf.clone());
    }
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        for buf in self.bufs.drain(..) {
            self.cache.recycle(buf);
        }
    }
}

/// Read `len` f32 values from a completed Metal buffer.
///
/// # Safety (encapsulated)
/// The caller must ensure the buffer has been committed and completed
/// (i.e., `cmd.wait_until_completed()` has returned) before calling this.
/// The pointer is valid for the buffer's lifetime — we immediately copy
/// into a new Vec, so no dangling reference is possible.
///
/// # Panics
/// Panics if the buffer's contents pointer is null or the buffer is
/// smaller than `len * sizeof(f32)` bytes. Use [`try_read_buffer_f32`]
/// in paths where alloc may legitimately fail (low-GPU-memory benches).
pub fn read_buffer_f32(buf: &metal::Buffer, len: usize) -> Vec<f32> {
    try_read_buffer_f32(buf, len)
        .expect("Metal buffer contents pointer is null or buffer is undersized")
}

/// Fallible read: returns `None` if the buffer's contents pointer is
/// null or the buffer is smaller than `len * sizeof(f32)` bytes.
///
/// Metal's `newBufferWithBytes:length:options:` can return a buffer
/// whose `contents()` is null when the device is out of memory — the
/// Rust binding doesn't surface that as `None`, so callers that may
/// dispatch very large buffers (the 2.5 GB lm-head bench shape, for
/// example) use this to fall back rather than panic.
pub fn try_read_buffer_f32(buf: &metal::Buffer, len: usize) -> Option<Vec<f32>> {
    let ptr = buf.contents() as *const f32;
    if ptr.is_null() {
        return None;
    }
    if (buf.length() as usize) < len * std::mem::size_of::<f32>() {
        return None;
    }
    // SAFETY: ptr is non-null, buffer is large enough, and the command
    // buffer has completed (caller invariant). Data is immediately
    // copied into a new Vec, so no dangling reference is possible.
    Some(unsafe { std::slice::from_raw_parts(ptr, len).to_vec() })
}

#[cfg(test)]
mod tests {
    use super::*;

    // `dev()` previously returned `Option<Device>` so each test could
    // early-return on non-Metal hosts.  Since this module is already
    // `#[cfg(all(feature = "metal", target_os = "macos"))]`-gated by
    // `lib.rs`, no test runner that compiles these tests is missing a
    // Metal device — the early-return branch was permanently dead
    // (uncovered region count: 10).  Panic on a missing device instead;
    // we only reach this path on a misconfigured runner where failure
    // is the right outcome.
    fn dev() -> Device {
        Device::system_default().expect("Metal device available on test host")
    }

    /// `get_f32` caches by (pointer, len). The same slice handed in
    /// twice must return the same Buffer (one allocation, two clones).
    #[test]
    fn get_f32_caches_by_slice_identity() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(cache.len(), 0);
        let b1 = cache.get_f32(&data);
        let b2 = cache.get_f32(&data);
        assert_eq!(cache.len(), 1, "second call must hit cache, not allocate");
        // Same underlying GPU buffer.
        assert_eq!(b1.gpu_address(), b2.gpu_address());
    }

    /// Distinct slices → distinct cache entries even if contents
    /// happen to be byte-identical (cache key is pointer+len, not value).
    #[test]
    fn get_f32_distinct_slices_get_distinct_buffers() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let a = vec![1.0f32; 16];
        let b = vec![1.0f32; 16];
        let _ = cache.get_f32(&a);
        let _ = cache.get_f32(&b);
        assert_eq!(cache.len(), 2);
    }

    /// Empty f32 slice → reused 4-byte stub. Metal rejects 0-length
    /// allocations, so the cache returns a single shared stub buffer.
    #[test]
    fn get_f32_empty_slice_returns_shared_stub() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let empty: Vec<f32> = vec![];
        let b1 = cache.get_f32(&empty);
        let b2 = cache.get_f32(&empty);
        assert_eq!(cache.len(), 1, "empty slices share one stub");
        assert_eq!(b1.length(), 4);
        assert_eq!(b1.gpu_address(), b2.gpu_address());
    }

    /// `get_bytes` empty stub keyed separately from `get_f32` empty
    /// stub (cache keys are different — `(0,0)` vs `(1,0)`).
    #[test]
    fn empty_f32_and_empty_bytes_have_separate_stubs() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let _ = cache.get_f32(&[][..]);
        let _ = cache.get_bytes(&[][..]);
        assert_eq!(
            cache.len(),
            2,
            "f32 and bytes empty stubs are independent cache entries"
        );
    }

    /// `transient_from_*` does NOT cache. Ten calls = ten allocations.
    #[test]
    fn transient_buffers_are_not_cached() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let data = vec![0.0f32; 64];
        let _b1 = cache.transient_from_f32(&data);
        let _b2 = cache.transient_from_f32(&data);
        assert_eq!(cache.len(), 0, "transient calls must not touch the cache");
    }

    /// `output(bytes)` returns a buffer of at least the requested
    /// size (Metal may round up but never under).
    #[test]
    fn output_buffer_is_at_least_requested_size() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let buf = cache.output(1024);
        assert!(buf.length() >= 1024);
        let buf2 = cache.output(1024);
        assert_eq!(cache.len(), 0, "output() does not cache");
        // Distinct allocations (different gpu_address).
        assert_ne!(buf.gpu_address(), buf2.gpu_address());
    }

    /// `read_buffer_f32` round-trips bytes written via the contents
    /// pointer of a `transient_from_f32` buffer. Pin the
    /// "buffer-finished → CPU read" contract.
    #[test]
    fn read_buffer_f32_round_trip() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let src: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let buf = cache.transient_from_f32(&src);
        let got = read_buffer_f32(&buf, src.len());
        assert_eq!(got, src);
    }

    /// `read_buffer_f32` panics on an undersized buffer.
    #[test]
    #[should_panic(expected = "buffer is undersized")]
    fn read_buffer_f32_panics_when_buffer_undersized() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let buf = cache.output(4); // 1 f32
        let _ = read_buffer_f32(&buf, 100); // ask for 100 → must panic
    }

    /// `try_read_buffer_f32` returns `None` on undersized buffer
    /// instead of panicking — the bench-safe path.
    #[test]
    fn try_read_buffer_f32_none_when_buffer_undersized() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let buf = cache.output(4); // 1 f32
        assert!(try_read_buffer_f32(&buf, 100).is_none());
    }

    /// `try_read_buffer_f32` round-trips like `read_buffer_f32` on a
    /// healthy buffer — the fallible wrapper isn't supposed to lose
    /// data, only avoid the panic.
    #[test]
    fn try_read_buffer_f32_round_trips_on_healthy_buffer() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let src: Vec<f32> = (0..8).map(|i| i as f32 * 1.5).collect();
        let buf = cache.transient_from_f32(&src);
        assert_eq!(try_read_buffer_f32(&buf, src.len()), Some(src));
    }

    /// `get_bytes` empty slice returns the cached `(1,0)` stub on the
    /// second call rather than allocating a new 4-byte buffer.
    #[test]
    fn get_bytes_empty_slice_reuses_cached_stub() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let b1 = cache.get_bytes(&[][..]);
        let b2 = cache.get_bytes(&[][..]);
        assert_eq!(cache.len(), 1, "empty-bytes stub is reused, not realloc'd");
        assert_eq!(b1.gpu_address(), b2.gpu_address());
    }

    /// `transient_from_bytes` with empty input returns a fresh minimal
    /// (4-byte) buffer each call — uncached, no stub reuse.
    #[test]
    fn transient_from_bytes_empty_returns_uncached_minimal_buffer() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let b1 = cache.transient_from_bytes(&[][..]);
        let b2 = cache.transient_from_bytes(&[][..]);
        assert_eq!(b1.length(), 4);
        assert_eq!(b2.length(), 4);
        assert_eq!(
            cache.len(),
            0,
            "transient_from_bytes never touches the cache"
        );
        // Distinct allocations (uncached path).
        assert_ne!(b1.gpu_address(), b2.gpu_address());
    }

    /// `transient_from_bytes` non-empty path copies the data into a
    /// fresh shared-storage buffer and does not cache.
    #[test]
    fn transient_from_bytes_nonempty_copies_data() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let src: Vec<u8> = (0..64u8).collect();
        let buf = cache.transient_from_bytes(&src);
        assert!(buf.length() as usize >= src.len());
        assert_eq!(cache.len(), 0, "transient calls do not cache");
    }

    /// `transient_from_i8` copies an `i8` slice into a fresh buffer
    /// the same way `transient_from_bytes` does for `u8`.  Coverage
    /// pin for the i8 helper specifically — used by the Q8 quantised
    /// input path.
    #[test]
    fn transient_from_i8_copies_data() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let src: Vec<i8> = (-16..16).collect();
        let buf = cache.transient_from_i8(&src);
        assert!(buf.length() as usize >= src.len());
    }

    /// `output()` after `recycle()` returns the recycled buffer (pool
    /// hit path), not a fresh allocation.  Verified by gpu_address
    /// equality.
    #[test]
    fn output_pool_returns_recycled_buffer_on_size_match() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let first = cache.output(2048);
        let first_addr = first.gpu_address();
        cache.recycle(first);

        let second = cache.output(2048);
        assert_eq!(
            second.gpu_address(),
            first_addr,
            "second output() of the same size should reuse the recycled buffer"
        );
    }

    /// `is_empty()` is the trait-friendly form of `len() == 0`.  Add
    /// an explicit cover so the helper doesn't drift out of sync with
    /// `len()`.
    #[test]
    fn is_empty_tracks_len_zero() {
        let d = dev();
        let cache = BufferCache::new(&d);
        assert!(cache.is_empty());
        let _ = cache.get_f32(&[1.0f32, 2.0, 3.0]);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    /// `ScratchGuard::track` collects buffers; the drop impl recycles
    /// them into the underlying cache's scratch pool.  Verifies the
    /// RAII contract by checking the second `output(size)` returns
    /// the tracked buffer (pool-hit path).
    #[test]
    fn scratch_guard_recycles_tracked_buffers_on_drop() {
        let d = dev();
        let cache = BufferCache::new(&d);
        let first_addr = {
            let mut guard = ScratchGuard::new(&cache);
            let buf = cache.output(512);
            let addr = buf.gpu_address();
            guard.track(&buf);
            addr
            // guard drops here, recycling buf
        };
        let reused = cache.output(512);
        assert_eq!(
            reused.gpu_address(),
            first_addr,
            "ScratchGuard drop must recycle into the scratch pool"
        );
    }
}
