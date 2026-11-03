//! Input-size limits for [`super::ApolloStore`] files.
//!
//! These cap how many bytes a load will read. They are not an allocation-safety
//! proof: a file within them can be sparse, so a store holding little on disk can
//! make a load allocate and touch the whole peak below. Allocations sized from these inputs
//! are fallible, but on a host that overcommits a successful reservation can still
//! end in an OOM kill. Do not load untrusted stores on a host that cannot spare the
//! peak; lower the limits instead.
//!
//! # Peak memory
//!
//! ```text
//! peak <= max_store_bytes + 8/7 * max_archive_bytes + max_records * ~200 bytes
//! ```
//!
//! About 2.57 GiB plus the last term at the defaults. Every byte read may stay in memory, and the
//! total read is limited by `max_store_bytes`. The worst moment is decoding the
//! entries: the member's bytes (up to `max_archive_bytes`) are still held while its
//! decoded rows are built, at 16 bytes in memory for each 14 on disk. The last term
//! is the directory records, the window list and each array's own header, which is
//! an allowance rather than a measurement.
//!
//! Measured on aarch64 macOS, with a sparse store of 2.0 GiB logical bytes at the
//! defaults: 2,760,480,778 bytes live at peak (2.57 GiB) and 2.59 GiB maximum RSS,
//! for a store of 128 windows. It has not been run on any other platform or with
//! the record allowance at its maximum.
//!
//! # Basis of the defaults
//!
//! The writer is `chuk_lazarus.inference.context.knowledge.store`, which calls
//! `np.savez` for each archive and `np.save` for each boundary, with 512-token
//! windows by default. `apollo11_store`, the one real store found, has 176 windows,
//! 3,585 entries and a 2,560-wide boundary, in 2.8 MB. The writer's own spec
//! (`SPEC_V7.md`) sizes its largest case at 10 million tokens: 19,531 windows and
//! 625,000 entries. `VecInjectEntry` holds `window_id` and `position_in_window` as
//! `u16`, so the format's ceiling is 65,536 windows of 65,536 tokens. The largest
//! `hidden_size` in this repository is 7,168.
//!
//! The figures below are arithmetic from those facts; only `apollo11_store` was
//! actually measured.

/// `manifest.json` is nine scalar fields plus a four-field object: 315 bytes in
/// `apollo11_store`, and under 1 KiB with every number at 20 digits. One MiB
/// leaves room for whitespace and unknown keys, which the parser ignores.
pub const DEFAULT_MAX_MANIFEST_BYTES: u64 = 1 << 20;

/// One boundary or residual `.npy`: 8 MiB holds about 2 million `f32`s plus the
/// largest possible v1 header (64 KiB), about 290 times the largest `hidden_size`
/// here. It refuses a boundary wider than that.
pub const DEFAULT_MAX_ARRAY_FILE_BYTES: u64 = 8 << 20;

/// One `.npz`. The spec's largest store needs about 40 MB for token lists and 9 MB
/// for entries. 65,536 windows of 1,024 tokens, twice the default width at the
/// format's window ceiling, is about 271 MiB with per-member overhead. 512 MiB
/// covers that, and about 38 million entries.
///
/// It refuses stores the format allows: the format's own ceiling for
/// `window_token_lists.npz` is about 16 GiB. Those need
/// [`super::ApolloStore::load_with_limits`].
pub const DEFAULT_MAX_ARCHIVE_BYTES: u64 = 512 << 20;

/// Every file a load reads, summed, so aliases of one file count each time. Two full
/// archives are 1 GiB, which leaves 1 GiB for boundaries: 65,536 windows of about
/// 4,000 wide. It refuses the format's window ceiling at the largest `hidden_size`
/// here (65,536 × 7,168 wide is about 1.76 GiB), which again needs explicit limits.
/// Sized to keep the peak above reasonable, not to fit every format-legal store.
pub const DEFAULT_MAX_STORE_BYTES: u64 = 2 << 30;

/// Directory records in one archive, and windows a manifest may claim: twice the
/// format's 65,536 windows, leaving room for members the load ignores. This bounds
/// the directory's memory at about 8 MiB whatever the file claims.
pub const DEFAULT_MAX_RECORDS: u64 = 1 << 17;

/// What a load accepts. Lengths are logical, compared with `fstat` on the open
/// handle, and a file over its limit is refused before any of it is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreLimits {
    pub max_manifest_bytes: u64,
    pub max_array_file_bytes: u64,
    pub max_archive_bytes: u64,
    pub max_store_bytes: u64,
    pub max_records: u64,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: DEFAULT_MAX_MANIFEST_BYTES,
            max_array_file_bytes: DEFAULT_MAX_ARRAY_FILE_BYTES,
            max_archive_bytes: DEFAULT_MAX_ARCHIVE_BYTES,
            max_store_bytes: DEFAULT_MAX_STORE_BYTES,
            max_records: DEFAULT_MAX_RECORDS,
        }
    }
}
