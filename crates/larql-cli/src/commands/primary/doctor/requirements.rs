//! What the quick start needs — the one place these numbers live.
//!
//! They come from docs/getting-started.md ("About 3 GB of free disk space
//! and 4 GB of RAM for the starter model") and the README quick start
//! (Qwen3-0.6B, "about 1.5 GB"). Change them here and there together.

/// The quick start's first model, as `larql run hf://…` names it.
pub(crate) const QUICKSTART_REPO: &str = "Qwen/Qwen3-0.6B";

/// Bytes the quick start model downloads (1.5 GB, decimal as the docs
/// print it).
pub(crate) const QUICKSTART_DOWNLOAD_BYTES: u64 = 1_500_000_000;

/// Free disk the quick start needs on the volume holding the LARQL cache:
/// the download plus its encoded copy.
pub(crate) const QUICKSTART_FREE_DISK_BYTES: u64 = 3_000_000_000;

/// Physical memory the quick start model needs.
pub(crate) const QUICKSTART_RAM_BYTES: u64 = 4_000_000_000;

/// Where a newcomer reads the requirements above.
pub(crate) const GETTING_STARTED_DOC: &str = "docs/getting-started.md";
