//! Where a widened f32 weight image LIVES, which is not what it holds.
//!
//! [`LoadedWeight::F32`](super::LoadedWeight::F32) is the residency of
//! last resort: it is what a matrix falls to when the stored bytes are
//! neither bf16 nor a format the policy may re-quantise. A K-quant
//! operand executed under `LARQL_CPU_MAX_FORMAT=bf16` is exactly that
//! case — the pack decodes to f32, `Q8` residency is forbidden, and
//! there are no stored bf16 code units to borrow. The image is then
//! twice the checkpoint's size, and on PARETO-1's rung A that measured
//! **96.8 GB RSS against a 45.36 GB bf16 decoder**, which the operating
//! system killed on a 128 GB machine before a single anchor-bank
//! evaluation completed.
//!
//! The values are not the problem. An anonymous `Vec<f32>` is: it is
//! dirty, private memory the kernel cannot reclaim under pressure, so
//! the process either fits or dies. The same bytes in a file-backed
//! mapping are clean — evictable when memory is short, re-read when
//! touched again — and the arithmetic that reads them cannot tell the
//! difference, because they are the same bytes.
//!
//! ```text
//! decode (unchanged) -> f32 values -> staging file -> mapped &[f32] -> backend
//! ```
//!
//! This deliberately changes RESIDENCY ONLY. It does not round, it does
//! not re-quantise, and it does not decode lazily: a per-use decoder
//! would re-expand tens of gigabytes of K-quant weights on every token,
//! trading a memory problem for a much larger time one.
//!
//! # One arena, not one file per weight
//!
//! Every staged image lives at a page-aligned offset in a single
//! unlinked file. One inode, one descriptor, and nothing to clean up:
//! the file is removed the moment it is created, so the bytes live
//! exactly as long as the process holds the handle, including if it is
//! killed.

use std::io::{Seek, SeekFrom, Write};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::error::VindexError;

/// Directory the staging arena is created in. Defaults to the platform
/// temporary directory, which is the wrong volume when the model is
/// larger than it — hence an override that names one explicitly.
pub const STAGE_DIR_ENV: &str = "LARQL_F32_STAGE_DIR";

/// Smallest image worth staging, in bytes.
///
/// Staging a small operand costs a page fault to save a few kilobytes,
/// and a model's norm and bias vectors are numerous enough that mapping
/// each one would add more mappings than it saves bytes. The default
/// sits well below the matrices that dominate a decoder — Qwen3.8's
/// `in_proj_qkv` is 105 MB widened — and well above the glue.
pub const STAGE_MIN_BYTES_ENV: &str = "LARQL_F32_STAGE_MIN_BYTES";

/// Default for [`STAGE_MIN_BYTES_ENV`].
pub const DEFAULT_STAGE_MIN_BYTES: usize = 16 * 1024 * 1024;

/// Set to `off` to keep every f32 image anonymous, as it was before
/// staging existed.
///
/// This is not a convenience. It is what makes the equivalence claim
/// testable inside ONE binary: the same build, the same weights, the
/// same arithmetic, differing only in where the bytes sit. A control
/// that requires two builds cannot separate "staging changed the answer"
/// from "the compiler did".
pub const STAGE_ENV: &str = "LARQL_F32_STAGE";

/// Value of [`STAGE_ENV`] that disables staging.
pub const STAGE_OFF: &str = "off";

/// Page granularity for arena offsets. `mmap` refuses an unaligned
/// offset, so each image starts on a boundary and the padding between
/// them is never read.
const ARENA_PAGE: u64 = 16 * 1024;

fn env_flag_disabled() -> bool {
    std::env::var(STAGE_ENV)
        .ok()
        .as_deref()
        .map(str::trim)
        .map(|v| v.eq_ignore_ascii_case(STAGE_OFF))
        .unwrap_or(false)
}

fn min_bytes() -> usize {
    std::env::var(STAGE_MIN_BYTES_ENV)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_STAGE_MIN_BYTES)
}

/// Bytes staged into the arena so far, for the residency report.
static STAGED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Images staged so far.
static STAGED_IMAGES: AtomicU64 = AtomicU64::new(0);

/// Total bytes this process has staged to the arena.
pub fn staged_bytes() -> u64 {
    STAGED_BYTES.load(Ordering::Relaxed)
}

/// Number of f32 images this process has staged.
pub fn staged_images() -> u64 {
    STAGED_IMAGES.load(Ordering::Relaxed)
}

struct Arena {
    file: std::fs::File,
    next: u64,
}

static ARENA: OnceLock<Mutex<Arena>> = OnceLock::new();

fn arena() -> Result<&'static Mutex<Arena>, VindexError> {
    // `OnceLock::get_or_init` cannot fail, so the fallible open happens
    // first and its error is reported rather than swallowed into a
    // silent fall back to anonymous memory — which is the exact failure
    // this module exists to make impossible.
    if let Some(a) = ARENA.get() {
        return Ok(a);
    }
    let dir = std::env::var_os(STAGE_DIR_ENV)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let path = dir.join(format!("larql-f32-stage-{}.bin", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| {
            VindexError::Parse(format!(
                "f32 staging arena `{}` could not be created: {e}. Set `{STAGE_DIR_ENV}` to a \
                 writable directory on a volume with room for the widened model, or \
                 `{STAGE_ENV}={STAGE_OFF}` to keep every image in anonymous memory",
                path.display()
            ))
        })?;
    // Unlink immediately: the descriptor keeps the inode alive, so the
    // bytes vanish when this process exits however it exits. A staging
    // file that outlives a killed run is a disk leak nobody will notice
    // until the volume is full.
    std::fs::remove_file(&path).map_err(|e| {
        VindexError::Parse(format!(
            "f32 staging arena `{}` could not be unlinked: {e}",
            path.display()
        ))
    })?;
    Ok(ARENA.get_or_init(|| Mutex::new(Arena { file, next: 0 })))
}

/// An f32 weight image, either owned outright or mapped from the arena.
///
/// Derefs to `[f32]`, so every reader sees one type and no kernel needs
/// to know which it got.
#[derive(Debug)]
pub struct StagedF32(Inner);

enum Inner {
    Owned(Vec<f32>),
    Mapped { map: memmap2::Mmap, elements: usize },
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Inner::Owned(v) => write!(f, "Owned({} f32)", v.len()),
            Inner::Mapped { elements, .. } => write!(f, "Mapped({elements} f32)"),
        }
    }
}

impl StagedF32 {
    /// Put `values` wherever policy says they belong.
    ///
    /// Small images stay owned; large ones move to the arena and the
    /// `Vec` is dropped, so the peak anonymous footprint is one operand
    /// rather than the whole model.
    pub fn stage(values: Vec<f32>) -> Result<Self, VindexError> {
        let bytes = std::mem::size_of_val(&values[..]);
        // An empty image is owned whatever the threshold says: a
        // zero-length mapping is an error on every platform, and a
        // threshold of zero would otherwise walk straight into it.
        if values.is_empty() || env_flag_disabled() || bytes < min_bytes() {
            return Ok(Self(Inner::Owned(values)));
        }
        // One lock for write AND map. Releasing it between the two would
        // let a second thread claim the same offset it just reserved,
        // and the corruption would land in weights rather than in an
        // error.
        let mut a = arena()?
            .lock()
            .map_err(|_| VindexError::Parse("f32 staging arena lock poisoned".into()))?;
        let offset = a.next;
        // SAFETY of the later cast rests on this: the mapping starts at a
        // page boundary, so the mapped address is 4-byte aligned for f32.
        let padded = (bytes as u64).div_ceil(ARENA_PAGE) * ARENA_PAGE;
        a.file
            .seek(SeekFrom::Start(offset))
            // The image's own bytes, verbatim. No conversion happens here
            // and none may: this function's whole contract is that what
            // comes back out is what went in.
            .and_then(|_| a.file.write_all(f32_bytes(&values)))
            .and_then(|_| a.file.set_len(offset + padded))
            .map_err(|e| VindexError::Parse(format!("f32 staging write failed: {e}")))?;
        a.next = offset + padded;
        let map = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset)
                .len(bytes)
                .map(&a.file)
        }
        .map_err(|e| VindexError::Parse(format!("f32 staging map failed: {e}")))?;
        STAGED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        STAGED_IMAGES.fetch_add(1, Ordering::Relaxed);
        Ok(Self(Inner::Mapped {
            map,
            elements: values.len(),
        }))
    }

    /// Whether these bytes are file-backed, for the residency report.
    pub fn is_mapped(&self) -> bool {
        matches!(self.0, Inner::Mapped { .. })
    }
}

/// `values` as the bytes that represent them, with no reinterpretation.
fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and no invalid bit patterns, so any
    // f32 slice is a valid byte slice of four times the length.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

impl Deref for StagedF32 {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        match &self.0 {
            Inner::Owned(v) => v,
            Inner::Mapped { map, elements } => {
                let bytes = &map[..];
                // SAFETY: the mapping begins at a page boundary so the
                // pointer is 4-byte aligned; `elements` counts exactly
                // the f32 values written at this offset; the file is
                // unlinked and privately mapped, so nothing else can
                // write to these bytes for the mapping's lifetime.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), *elements) }
            }
        }
    }
}

impl From<Vec<f32>> for StagedF32 {
    fn from(values: Vec<f32>) -> Self {
        Self(Inner::Owned(values))
    }
}
