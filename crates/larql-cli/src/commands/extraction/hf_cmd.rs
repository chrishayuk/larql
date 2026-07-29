use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct HfArgs {
    #[command(subcommand)]
    command: HfCommand,
}

#[derive(Subcommand)]
enum HfCommand {
    /// Download a vindex from HuggingFace.
    Download {
        /// HuggingFace repo ID (e.g. chrishayuk/gemma-3-4b-it-vindex).
        repo: String,

        /// Output directory (default: downloads to HF cache).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Specific revision or tag.
        #[arg(long)]
        revision: Option<String>,
    },

    /// Publish a local vindex to HuggingFace.
    Publish {
        /// Local vindex directory.
        vindex: PathBuf,

        /// HuggingFace repo ID (e.g. chrishayuk/gemma-3-4b-it-vindex).
        #[arg(long)]
        repo: String,

        /// Create the repo private. Pair with `larql hf visibility
        /// --public` once verification passes.
        #[arg(long)]
        private: bool,

        /// HuggingFace repo type: `model` (default) or `dataset`.
        #[arg(long, default_value = "model")]
        repo_type: String,
    },

    /// Flip an already-published repo's visibility. Vindex Factory's
    /// RELEASE step (docs/vindex-factory.md §7): publish `--private`,
    /// verify the published bytes, then `visibility --public` once
    /// verification passes.
    Visibility {
        /// HuggingFace repo ID (e.g. chrishayuk/gemma-3-4b-it-vindex).
        repo: String,

        /// Make the repo public.
        #[arg(long, conflicts_with = "private")]
        public: bool,

        /// Make the repo private.
        #[arg(long, conflicts_with = "public")]
        private: bool,

        /// HuggingFace repo type: `model` (default) or `dataset`.
        #[arg(long, default_value = "model")]
        repo_type: String,
    },
}

pub fn run(args: HfArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        HfCommand::Download {
            repo,
            output,
            revision,
        } => run_download(&repo, output.as_deref(), revision.as_deref()),
        HfCommand::Publish {
            vindex,
            repo,
            private,
            repo_type,
        } => run_publish(&vindex, &repo, private, &repo_type),
        HfCommand::Visibility {
            repo,
            public,
            private,
            repo_type,
        } => run_visibility(&repo, public, private, &repo_type),
    }
}

fn run_download(
    repo: &str,
    output: Option<&std::path::Path>,
    revision: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let hf_path = if let Some(rev) = revision {
        format!("hf://{}@{}", repo, rev)
    } else {
        format!("hf://{}", repo)
    };

    eprintln!("Downloading vindex from HuggingFace: {}", hf_path);
    let cached_path = larql_vindex::resolve_hf_vindex(&hf_path)?;
    eprintln!("  Cached at: {}", cached_path.display());

    // If output specified, copy from cache to output
    if let Some(out) = output {
        eprintln!("  Copying to: {}", out.display());
        copy_dir(&cached_path, out)?;
        eprintln!("Done: {}", out.display());
    } else {
        eprintln!("Done. Use:");
        eprintln!("  larql repl");
        eprintln!("  larql> USE \"{}\";", cached_path.display());
    }

    // Load and show summary
    if let Ok(config) = larql_vindex::load_vindex_config(&cached_path) {
        let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
        eprintln!("\n  Model: {}", config.model);
        eprintln!(
            "  {} layers, {} features",
            config.num_layers, total_features
        );
        eprintln!("  Extract level: {}", config.extract_level);
    }

    Ok(())
}

fn run_visibility(
    repo: &str,
    public: bool,
    private: bool,
    repo_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let make_private = match (public, private) {
        (true, false) => false,
        (false, true) => true,
        (false, false) => return Err("specify exactly one of --public or --private".into()),
        (true, true) => unreachable!("clap's conflicts_with rules out both being set"),
    };

    larql_vindex::set_repo_visibility(repo, repo_type, make_private)?;
    eprintln!(
        "  {} is now {}",
        repo,
        if make_private { "private" } else { "public" }
    );
    Ok(())
}

fn run_publish(
    vindex: &std::path::Path,
    repo: &str,
    private: bool,
    repo_type: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Publishing vindex to HuggingFace: {}", repo);

    let mut callbacks = CliPublishCallbacks;
    let opts = larql_vindex::PublishOptions {
        repo_type: repo_type.to_string(),
        private,
        ..Default::default()
    };
    let url = larql_vindex::publish_vindex_with_opts(vindex, repo, &opts, &mut callbacks)?;

    eprintln!("\nPublished: {}", url);
    eprintln!("\nUsage:");
    eprintln!("  larql repl");
    eprintln!("  larql> USE \"hf://{}\";", repo);

    Ok(())
}

struct CliPublishCallbacks;

impl larql_vindex::PublishCallbacks for CliPublishCallbacks {
    fn on_start(&mut self, repo: &str) {
        eprintln!("  Creating repo: {}", repo);
    }

    fn on_file_start(&mut self, filename: &str, size: u64) {
        let size_str = if size > 1_073_741_824 {
            format!("{:.2} GB", size as f64 / 1_073_741_824.0)
        } else if size > 1_048_576 {
            format!("{:.1} MB", size as f64 / 1_048_576.0)
        } else {
            format!("{:.1} KB", size as f64 / 1024.0)
        };
        eprint!("  Uploading {} ({})...", filename, size_str);
    }

    fn on_file_done(&mut self, _filename: &str) {
        eprintln!(" done");
    }

    fn on_file_skipped(&mut self, filename: &str, size: u64, sha256: &str) {
        let short_sha = sha256.get(..12).unwrap_or(sha256);
        let size_str = if size > 1_073_741_824 {
            format!("{:.2} GB", size as f64 / 1_073_741_824.0)
        } else if size > 1_048_576 {
            format!("{:.1} MB", size as f64 / 1_048_576.0)
        } else {
            format!("{:.1} KB", size as f64 / 1024.0)
        };
        eprintln!(
            "  Skipping  {} ({}) — unchanged (sha256 {}…)",
            filename, size_str, short_sha
        );
    }

    fn on_complete(&mut self, url: &str) {
        eprintln!("  URL: {}", url);
    }
}

fn copy_dir(
    src: &std::path::Path,
    dst: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_file() {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}
