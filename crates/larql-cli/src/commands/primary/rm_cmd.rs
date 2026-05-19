//! `larql rm <model>` — evict a cached vindex.
//!
//! Cache-only — never downloads. Accepts full `owner/name`, a cache
//! shorthand, or `<name>` for a local-cache entry.
//!
//! For local entries this unlinks the `<name>.vindex` symlink (or
//! removes the entry directory if it was a real dir). **The original
//! path `larql link` pointed at is never touched.**
//!
//! For HF entries this removes the whole
//! `{models,datasets}--<owner>--<name>` tree from the HF hub cache. The
//! prefix is whichever one hf-hub used when the vindex was pulled —
//! `cache::scan_hf_hub_at` recognises both.

use clap::Args;

use crate::commands::primary::cache::{self, CacheSource};

#[derive(Args)]
pub struct RmArgs {
    /// `owner/name` (HF), or cache shorthand.
    pub model: String,

    /// Skip the confirmation prompt.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

pub fn run(args: RmArgs) -> Result<(), Box<dyn std::error::Error>> {
    let entry = cache::resolve_cached(&args.model)?;

    let (target_desc, target_path, is_symlink) = match entry.source {
        CacheSource::Local => {
            // For local entries, `snapshot` IS the symlink / directory.
            let is_symlink = std::fs::symlink_metadata(&entry.snapshot)
                .map(|m| m.file_type().is_symlink())
                .unwrap_or(false);
            (
                format!(
                    "local link `{}` ({} MB)",
                    entry.repo,
                    entry.size_bytes as f64 / 1e6
                ),
                entry.snapshot.clone(),
                is_symlink,
            )
        }
        CacheSource::HuggingFace => {
            // Back up from `snapshots/<sha>/` → `{models,datasets}--<owner>--<name>/`.
            let hub_repo_dir = entry
                .snapshot
                .parent()
                .and_then(|p| p.parent())
                .ok_or("unexpected HF cache path structure")?
                .to_path_buf();
            (
                format!(
                    "HF cache `{}` ({} MB)",
                    entry.repo,
                    entry.size_bytes as f64 / 1e6
                ),
                hub_repo_dir,
                false,
            )
        }
    };

    if !target_path.exists() && !is_symlink {
        return Err(format!("not cached: {}", target_path.display()).into());
    }

    if !args.yes {
        use std::io::{self, Write};
        eprint!("Remove {target_desc}? [y/N] ");
        io::stderr().flush()?;
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        if !matches!(line.trim(), "y" | "Y" | "yes") {
            eprintln!("aborted.");
            return Ok(());
        }
    }

    if is_symlink {
        // Unlink only — never follow the symlink. Original stays put.
        std::fs::remove_file(&target_path)?;
    } else {
        std::fs::remove_dir_all(&target_path)?;
    }
    eprintln!("Removed {}.", entry.repo);
    Ok(())
}
