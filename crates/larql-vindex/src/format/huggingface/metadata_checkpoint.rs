//! Stage a **header-only** checkpoint from a repo's metadata alone.
//!
//! [`larql_models::inventory::build_inventory`] and
//! [`crate::format::vindex3::plan`] read a checkpoint's `config.json` plus
//! each safetensors shard's JSON header — the 8-byte length prefix and the
//! header bytes it announces. No tensor data is ever touched. So the
//! admission instruments do not need the weights: they need the headers,
//! and a safetensors header is a few hundred kilobytes even when the
//! payload is hundreds of gigabytes.
//!
//! This fetches those headers over byte-range reads and writes a stub
//! directory containing `<u64 header_len><header json>` per shard and
//! nothing else. Inventory, plan and capability admission then run against
//! it **unmodified** and produce the same verdict they would produce
//! against the real weights — identity, per-layer attention policy, every
//! unconsumed config key, the full tensor inventory with exact shapes,
//! dtypes and byte counts.
//!
//! GLM-5.3-Flash: 18 MB of stub stands in for 328 GB of checkpoint.
//!
//! # The stub is also the payload manifest
//!
//! A stub carries every tensor's offset and length, because that is what
//! a safetensors header *is*. So [`super::super::vindex3::encode::source`]
//! indexes the stub with the same code that indexes a real shard, and the
//! remote source needs no second implementation of offset arithmetic —
//! which matters, because two implementations agreeing is not evidence
//! that either is right.
//!
//! # What a stub cannot do
//!
//! Anything that reads a tensor value. It answers "would this checkpoint
//! be admitted, and what would it cost" — the question worth answering
//! before spending the download.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::error::VindexError;
use crate::format::vindex3::encode::source::MAX_HEADER_BYTES;

use super::range::HfRangeClient;

/// Repo files copied verbatim when present.
///
/// The first group is what the inventory's identity, topology and
/// interface readers consult; the second is
/// [`crate::format::vindex3::encode::checkpoint::CHECKPOINT_CAPABILITY_FILES`],
/// staged here so a container compiled from `hf://` binds with full
/// capability rather than token-ids only.
pub const STAGED_METADATA_FILES: [&str; 8] = [
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
];

/// Filename of the HF shard index.
const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";

/// The single file an unsharded repo carries instead of an index.
const SINGLE_SHARD: &str = "model.safetensors";

/// File probed to resolve the revision to a commit — every model repo has
/// one, and it is the first thing staged anyway.
const COMMIT_PROBE: &str = "config.json";

/// Length of the safetensors length prefix.
const LENGTH_PREFIX_BYTES: u64 = 8;

/// Cache root for staged headers, under the same `LARQL_HOME` override
/// the vindex cache honours.
const HEADER_CACHE_SUBDIR: &str = "hf-headers";

/// Separator standing in for `/` in a repo id used as a directory name —
/// the same spelling the HF hub cache uses.
const REPO_DIR_SEP: &str = "--";

/// Concurrent shard-header reads. Enough to hide latency on a
/// hundred-shard repo, low enough not to trip the hub's rate limiter —
/// which answers by truncating bodies, the failure the range client
/// refuses rather than accepts.
const HEADER_FETCH_JOBS: usize = 8;

/// The commit `client`'s revision currently resolves to.
///
/// Probes the one file every model repo carries, so the answer costs a
/// request rather than a download. `None` when the hub sends no commit
/// header — a fact the caller reports, never one it invents a sha for.
pub fn resolve_commit(client: &HfRangeClient) -> Result<Option<String>, VindexError> {
    client.resolve_commit(COMMIT_PROBE)
}

/// Where headers for `repo` at `revision` are staged.
///
/// `~/.cache/larql/hf-headers/{owner}--{name}/{revision}/`, honouring
/// `LARQL_HOME` exactly as the vindex cache does. Keyed by revision, and
/// the caller keys by the resolved COMMIT once it has one, so two runs
/// against a moved `main` never share a stub whose offsets belong to a
/// different checkpoint.
pub fn header_cache_dir(repo: &str, revision: &str) -> Result<PathBuf, VindexError> {
    let root = match std::env::var("LARQL_HOME") {
        Ok(home) if !home.is_empty() => PathBuf::from(home),
        _ => {
            let home = std::env::var("HOME")
                .map_err(|_| VindexError::Parse("HOME is not set".to_string()))?;
            PathBuf::from(home).join(".cache").join("larql")
        }
    };
    Ok(root
        .join(HEADER_CACHE_SUBDIR)
        .join(repo.replace('/', REPO_DIR_SEP))
        .join(revision))
}

/// What one staging pass produced.
#[derive(Debug, Clone)]
pub struct StagedCheckpoint {
    /// The stub directory. Pass this to `build_inventory`.
    pub dir: PathBuf,
    /// The commit the revision resolved to, when the hub said. Provenance:
    /// `main` moves, and a container compiled from it must record which
    /// `main` it was.
    pub commit: Option<String>,
    /// Shard filenames, repo-relative, in index order.
    pub shards: Vec<String>,
    /// Metadata filenames actually found and staged.
    pub metadata: Vec<String>,
    /// Bytes the shard header stubs occupy.
    ///
    /// Reported separately from [`Self::metadata_bytes`] because they
    /// scale differently: header bytes track the tensor count, while a
    /// tokenizer is whatever size it is. GLM-5.3-Flash stages 10.7 MB of
    /// headers and 27 MB of metadata — quoting only the first would
    /// understate the transfer by a factor of four.
    pub stub_bytes: u64,
    /// Bytes the staged metadata files occupy.
    pub metadata_bytes: u64,
    /// `metadata.total_size` from the shard index, when it declares one —
    /// the checkpoint this stub stands in for.
    pub declared_total_size: Option<u64>,
}

/// Fetch metadata and shard headers for `client`'s repo into `dir`.
///
/// Idempotent: a shard whose stub is already present and non-trivial is
/// left alone, so an interrupted staging resumes.
pub fn stage_metadata_checkpoint(
    client: &HfRangeClient,
    dir: &Path,
) -> Result<StagedCheckpoint, VindexError> {
    std::fs::create_dir_all(dir)?;
    let commit = client.resolve_commit(COMMIT_PROBE)?;

    // Metadata is cached like the headers are. The cache directory is
    // keyed by the resolved COMMIT, so a file already staged there cannot
    // have changed — and re-fetching it is not free: GLM-5.3-Flash carries
    // a 19 MB tokenizer and an 8 MB shard index.
    let mut metadata = Vec::new();
    for name in STAGED_METADATA_FILES {
        let dest = dir.join(name);
        if std::fs::metadata(&dest).is_ok_and(|m| m.len() > 0) {
            metadata.push(name.to_string());
            continue;
        }
        if let Some(body) = client.fetch(name)? {
            std::fs::write(&dest, &body)?;
            metadata.push(name.to_string());
        }
    }
    if !metadata.iter().any(|n| n == COMMIT_PROBE) {
        return Err(VindexError::Parse(format!(
            "hf://{} has no {COMMIT_PROBE} — not a model checkpoint",
            client.repo()
        )));
    }

    let (shards, declared_total_size) = read_shard_list(dir)?;
    fetch_shard_headers(client, dir, &shards)?;

    let stub_bytes = staged_bytes(dir, &shards);
    let metadata_bytes = staged_bytes(dir, &metadata);

    Ok(StagedCheckpoint {
        dir: dir.to_path_buf(),
        commit,
        shards,
        metadata,
        stub_bytes,
        metadata_bytes,
        declared_total_size,
    })
}

/// Bytes `names` occupy under `dir`, skipping any that are absent.
fn staged_bytes(dir: &Path, names: &[String]) -> u64 {
    names
        .iter()
        .filter_map(|name| std::fs::metadata(dir.join(name)).ok())
        .map(|m| m.len())
        .sum()
}

/// Shard filenames and the index's declared total, from the staged index.
fn read_shard_list(dir: &Path) -> Result<(Vec<String>, Option<u64>), VindexError> {
    let index_path = dir.join(SAFETENSORS_INDEX_FILE);
    if !index_path.exists() {
        return Ok((vec![SINGLE_SHARD.to_string()], None));
    }
    let text = std::fs::read_to_string(&index_path)?;
    let index: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| VindexError::Parse(format!("parse {SAFETENSORS_INDEX_FILE}: {e}")))?;
    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .map(|map| {
            map.values()
                .filter_map(|v| v.as_str())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(VindexError::Parse(format!(
            "{SAFETENSORS_INDEX_FILE} names no shards"
        )));
    }
    let declared = index["metadata"]["total_size"].as_u64();
    Ok((shards, declared))
}

/// Read every shard's header into `dir`, `HEADER_FETCH_JOBS` at a time.
fn fetch_shard_headers(
    client: &HfRangeClient,
    dir: &Path,
    shards: &[String],
) -> Result<(), VindexError> {
    let next = AtomicUsize::new(0);
    let failures: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let jobs = HEADER_FETCH_JOBS.min(shards.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(shard) = shards.get(index) else {
                    return;
                };
                if let Err(err) = fetch_shard_header(client, dir, shard) {
                    failures.lock().unwrap().push(format!("{shard}: {err}"));
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    if failures.is_empty() {
        return Ok(());
    }
    // Every failure, not the first: a partial stub misreports the
    // checkpoint, and which shards are missing is the finding.
    Err(VindexError::Parse(format!(
        "{} of {} shard header(s) failed — the stub is INCOMPLETE:\n  {}",
        failures.len(),
        shards.len(),
        failures.join("\n  ")
    )))
}

/// One shard's stub: the length prefix and the header it announces.
fn fetch_shard_header(client: &HfRangeClient, dir: &Path, shard: &str) -> Result<(), VindexError> {
    let dest = dir.join(shard);
    if std::fs::metadata(&dest).is_ok_and(|m| m.len() > LENGTH_PREFIX_BYTES) {
        return Ok(());
    }
    let prefix = client.fetch_range(shard, 0, LENGTH_PREFIX_BYTES)?;
    let header_len = u64::from_le_bytes(
        prefix
            .as_slice()
            .try_into()
            .map_err(|_| VindexError::Parse(format!("{shard}: short length prefix")))?,
    );
    if header_len > MAX_HEADER_BYTES {
        return Err(VindexError::Parse(format!(
            "{shard}: safetensors header claims {header_len} bytes — \
             the range read did not land on a safetensors file"
        )));
    }
    let header = client.fetch_range(shard, LENGTH_PREFIX_BYTES, header_len)?;
    // Parse before writing: a stub that is not valid JSON would fail
    // later, inside the tool under test, and look like a tool defect.
    serde_json::from_slice::<serde_json::Value>(&header)
        .map_err(|e| VindexError::Parse(format!("{shard}: header is not JSON: {e}")))?;

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Write through a temporary: an interrupted write must not leave a
    // truncated stub that the resume check then accepts as complete.
    let staging = dest.with_extension("incomplete");
    let mut file = std::fs::File::create(&staging)?;
    file.write_all(&prefix)?;
    file.write_all(&header)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&staging, &dest)?;
    Ok(())
}
