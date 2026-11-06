//! `larql list` — show cached vindexes.
//!
//! Walks every cache (HF hub, LARQL local registry, encoded Hugging Face
//! checkpoints) and lists every cached vindex with its generation, size, layer count, hidden dim and
//! whether this build can actually open it. See
//! [`crate::commands::primary::cache`] for the scan logic shared with
//! `run` / `show` / `rm` / `link`.
//!
//! Identity (generation, layers, hidden) comes from the generation-neutral
//! [`summarize_container`] — the same fact source LQL's `SHOW VINDEXES`
//! reads — so a VINDEX3 container is never listed as a 0-layer VINDEX2 one.
//! Readiness is asked of each generation's own opener, without hashing
//! payloads: an entry this build cannot open is listed as `unusable` with
//! the opener's own reason, never as a row of zeros.

use clap::Args;
use larql_vindex::format::generation::{summarize_container, ContainerGeneration};

use crate::commands::primary::cache::{self, CachedVindex};

/// The size column's unit: decimal megabytes, as `SIZE (MB)` says.
const BYTES_PER_MB: f64 = 1e6;
/// Status of an entry this build can open.
const STATUS_READY: &str = "ready";
/// Status of an entry this build cannot open; the reason follows the table.
const STATUS_UNUSABLE: &str = "unusable";
/// Placeholder for a fact the entry could not supply.
const UNKNOWN_FIELD: &str = "-";
/// Where a reason's first sentence ends — the rest is remedy prose that
/// belongs to `larql show`, not to a one-line listing note.
const SENTENCE_END: &str = ". ";

#[derive(Args)]
pub struct ListArgs {}

/// Whether this build can open an entry, and if not, why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EntryStatus {
    Ready,
    Unusable(String),
}

/// What `larql list` says about one cache entry.
#[derive(Debug, Clone)]
pub(crate) struct ListRow {
    pub source: &'static str,
    pub repo: String,
    pub size_bytes: u64,
    pub generation: Option<ContainerGeneration>,
    pub layers: Option<usize>,
    pub hidden: Option<usize>,
    pub status: EntryStatus,
}

pub fn run(_args: ListArgs) -> Result<(), Box<dyn std::error::Error>> {
    let entries = cache::scan_cached_vindexes()?;

    if entries.is_empty() {
        println!(
            "No cached vindexes.\n\
             Try `larql run hf://owner/name` (a Hugging Face checkpoint is \
             encoded on first use), `larql pull hf://owner/name` (remote) or \
             `larql link <path>` (local)."
        );
        return Ok(());
    }

    let rows: Vec<ListRow> = entries.iter().map(describe).collect();
    print!("{}", render(&rows));
    Ok(())
}

/// Read one entry's identity and readiness.
pub(crate) fn describe(entry: &CachedVindex) -> ListRow {
    let mut row = ListRow {
        source: entry.source.label(),
        repo: entry.repo.clone(),
        size_bytes: entry.size_bytes,
        generation: None,
        layers: None,
        hidden: None,
        status: EntryStatus::Ready,
    };
    match summarize_container(&entry.snapshot) {
        Ok(summary) => {
            row.generation = Some(summary.generation);
            row.layers = Some(summary.num_layers);
            row.hidden = Some(summary.hidden_size);
            if let Err(why) = readiness(&entry.snapshot, summary.generation) {
                row.status = EntryStatus::Unusable(why);
            }
        }
        Err(e) => row.status = EntryStatus::Unusable(e.to_string()),
    }
    row
}

/// Ask the generation's own opener whether it accepts this directory.
fn readiness(path: &std::path::Path, generation: ContainerGeneration) -> Result<(), String> {
    match generation {
        ContainerGeneration::V2 => larql_vindex::load_vindex_config(path).map(|_| ()),
        ContainerGeneration::V3 => {
            larql_vindex::format::vindex3::check_container_structure(path, false)
        }
    }
    .map_err(|e| e.to_string())
}

/// The first sentence of an opener's refusal.
pub(crate) fn reason_summary(reason: &str) -> &str {
    reason
        .split_once(SENTENCE_END)
        .map_or(reason, |(first, _)| first)
}

fn field(value: Option<usize>) -> String {
    value.map_or_else(|| UNKNOWN_FIELD.to_string(), |v| v.to_string())
}

/// The table, then one note per unusable entry naming why.
pub(crate) fn render(rows: &[ListRow]) -> String {
    let mut out = format!(
        "{:<8}  {:<48}  {:<3}  {:>10}  {:>7}  {:>8}  {}\n",
        "SOURCE", "MODEL", "GEN", "SIZE (MB)", "LAYERS", "HIDDEN", "STATUS"
    );
    for row in rows {
        let status = match row.status {
            EntryStatus::Ready => STATUS_READY,
            EntryStatus::Unusable(_) => STATUS_UNUSABLE,
        };
        out.push_str(&format!(
            "{:<8}  {:<48}  {:<3}  {:>10.1}  {:>7}  {:>8}  {}\n",
            row.source,
            row.repo,
            row.generation.map_or(UNKNOWN_FIELD, |g| g.schema_label()),
            row.size_bytes as f64 / BYTES_PER_MB,
            field(row.layers),
            field(row.hidden),
            status,
        ));
    }
    let unusable: Vec<_> = rows
        .iter()
        .filter_map(|r| match &r.status {
            EntryStatus::Unusable(why) => Some((r, why)),
            EntryStatus::Ready => None,
        })
        .collect();
    if !unusable.is_empty() {
        out.push_str("\nUnusable with this build:\n");
        for (row, why) in unusable {
            out.push_str(&format!("  {}: {}\n", row.repo, reason_summary(why)));
        }
    }
    out
}

#[cfg(test)]
mod tests;
