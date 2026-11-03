//! Input-size limits for [`super::ApolloStore`] files.
//!
//! These cap how many bytes a load will read. They are not an allocation-safety
//! proof: a file within them can still be sparse, so a few KiB on disk can make
//! a load allocate and touch up to [`DEFAULT_MAX_STORE_BYTES`]. Allocations sized
//! from these inputs are fallible, but on a host that overcommits a successful
//! reservation can still end in an OOM kill. Do not load untrusted stores on a
//! host that cannot spare that much memory; lower the limits instead.
//!
//! # Basis of the defaults
//!
//! The writer is `chuk_lazarus.inference.context.knowledge.store`, which calls
//! `np.savez` for each archive and `np.save` for each boundary, with 512-token
//! windows by default. `apollo11_store`, the one real store found, has 176 windows,
//! 3,585 entries and a 2,560-wide boundary, in 2.8 MB. The writer's own spec
//! (`SPEC_V7.md`) sizes its largest case at 10 million tokens: 19,531 windows and
//! 625,000 entries, about 318 MB. `VecInjectEntry` holds `window_id` and
//! `position_in_window` as `u16`, so the format's ceiling is 65,536 windows of
//! 65,536 tokens. The largest `hidden_size` in this repository is 7,168.
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

/// One `.npz`. The spec's largest store needs about 40 MB for token lists and
/// 9 MB for entries. 65,536 windows of 2,048 tokens, four times the default width
/// at the format's window ceiling, needs about 552 MiB with per-member overhead.
/// One GiB covers that, and about 76 million entries.
///
/// It refuses stores the format allows: the format's own ceiling for
/// `window_token_lists.npz` is about 17 GiB. Those need
/// [`super::ApolloStore::load_with_limits`].
pub const DEFAULT_MAX_ARCHIVE_BYTES: u64 = 1 << 30;

/// Every file a load reads, summed, so aliases of one file count each time.
/// The largest store the other defaults should admit is two full archives (2 GiB)
/// and 65,536 boundaries of the largest `hidden_size` (about 1.9 GiB).
pub const DEFAULT_MAX_STORE_BYTES: u64 = 4 << 30;

/// Logical file lengths a load will accept, compared with `fstat` on the open
/// handle. A file over its limit is refused before any of it is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreLimits {
    pub max_manifest_bytes: u64,
    pub max_array_file_bytes: u64,
    pub max_archive_bytes: u64,
    pub max_store_bytes: u64,
}

impl Default for StoreLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: DEFAULT_MAX_MANIFEST_BYTES,
            max_array_file_bytes: DEFAULT_MAX_ARRAY_FILE_BYTES,
            max_archive_bytes: DEFAULT_MAX_ARCHIVE_BYTES,
            max_store_bytes: DEFAULT_MAX_STORE_BYTES,
        }
    }
}
