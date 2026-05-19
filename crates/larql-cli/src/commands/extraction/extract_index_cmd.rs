use larql_vindex::format::filenames::*;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_inference::InferenceModel;
use larql_vindex::IndexBuildCallbacks;

#[derive(Args)]
pub struct ExtractIndexArgs {
    /// Model path or HuggingFace model ID (extracts directly from weights).
    /// Not needed if --from-vectors is used.
    model: Option<String>,

    /// Output path for the .vindex directory.
    #[arg(short, long)]
    output: PathBuf,

    /// Build from already-extracted NDJSON vector files instead of model weights.
    /// Point to the directory containing ffn_gate.vectors.jsonl, etc.
    #[arg(long)]
    from_vectors: Option<PathBuf>,

    /// Top-K tokens to store per feature in down metadata (only for model extraction).
    #[arg(long, default_value = "10")]
    down_top_k: usize,

    /// Per-expert top-K right singular vectors of `gate_proj` to store
    /// instead of the full per-expert gate matrix. Default `0` = disabled
    /// (write full per-expert gate, original behaviour). Set e.g. `64` to
    /// produce a tractable summary vindex for many-experts MoE models
    /// (DeepSeek-V4-Pro at 384 experts/layer would otherwise need ~370 GB
    /// of gate_vectors; with `--summary-features-per-expert 64` it shrinks
    /// to ~11 GB).
    #[arg(long, default_value = "0")]
    summary_features_per_expert: usize,

    /// How much of the model to include in the vindex. Each tier is a
    /// strict superset of the previous:
    ///
    ///   browse     — gate + embed + down_meta only. WALK / DESCRIBE only.
    ///   attention  — + attention + norms. Client half of `run --ffn URL`.
    ///   inference  — + FFN up/down. Full local forward pass (default).
    ///   all        — + lm_head + anything for COMPILE.
    #[arg(long, default_value = "inference", value_parser = parse_extract_level)]
    level: larql_vindex::ExtractLevel,

    /// Include full model weights. Alias for --level all (deprecated, use --level instead).
    #[arg(long)]
    include_weights: bool,

    /// Opt out of the f16 default: store side-channel tensors
    /// (gate_vectors.bin, embeddings.bin, attn/norms/lm_head when
    /// `--quant none`) at f32 instead. Doubles file sizes for
    /// negligible accuracy gain. Rarely wanted.
    #[arg(long)]
    f32: bool,

    /// Quantise model forward-pass weights inline while extracting —
    /// skips any f32 intermediate. `q4k`: Q4_K for Q/K/O/gate/up, Q6_K
    /// for V/down (Ollama-compatible). Implies `--level all` (the Q4_K
    /// writer materialises all components in one pass) and forces f16
    /// on unquantised side-channels (gate_vectors, embeddings) even if
    /// `--f32` was passed.
    #[arg(long, default_value = "none", value_parser = parse_quant)]
    quant: larql_vindex::QuantFormat,

    /// Skip writing `up_weights.bin` + `down_weights.bin`. The up/down
    /// weights are reconstructable from `up_features.bin` /
    /// `down_features.bin` which are produced separately via
    /// `build_{up,down}_features`. This saves ~3.4 GB on a 4B f16 vindex
    /// / ~14 GB on a 31B vindex.
    ///
    /// **Caveat:** a compact vindex can only be read by `WalkFfn` (the
    /// default inference path). `WeightFfn` / `larql dev walk --compare`
    /// will panic on missing FFN tensors.
    #[arg(long)]
    compact: bool,

    /// Skip writing `gate_vectors.bin`. Only valid with `--quant q4k`
    /// — the loader rebuilds the f16 gate by dequantizing
    /// `interleaved_kquant.bin` at vindex-load time. Saves ~1.7 GB on a
    /// 4B q4k vindex / ~14 GB on a 31B q4k vindex; costs ~1.6 s / ~12 s
    /// of CPU at load. See
    /// `cargo run --release -p larql-vindex --example bench_gate_dequant`
    /// for the measured trade-off.
    #[arg(long)]
    drop_gate_vectors: bool,

    /// Quantise FFN down-proj as Q4_K instead of Q6_K. Only valid with
    /// `--quant q4k`. Default keeps the Ollama-compatible mix (Q4_K for
    /// gate/up, Q6_K for down). Enabling this saves ~30 MB/layer on 31B
    /// (~1.8 GB total) and drops down matmul cost ~1.5-1.7× at decode.
    /// Quantisation error on down is a scatter-sum over the intermediate
    /// dimension — noise averages — but quality must be validated
    /// against `walk_correctness` before adopting in production.
    #[arg(long)]
    down_q4k: bool,

    /// Emit `down_features_q4k.bin` (W2 feature-major down) so per-feature
    /// row decode can skip the `kquant_ffn_layer` cache. Adds ~14 MB / layer
    /// at Gemma 4B dims; eliminates the ~840 MB heap cache ceiling on
    /// CPU sparse walk and frees the same headroom across all grid shards.
    /// Requires `--quant q4k`.
    #[arg(long)]
    feature_major_down: bool,

    /// Skip stages that already have output files (resume interrupted builds).
    #[arg(long)]
    resume: bool,
}

fn parse_quant(s: &str) -> Result<larql_vindex::QuantFormat, String> {
    match s.to_lowercase().as_str() {
        "none" | "" => Ok(larql_vindex::QuantFormat::None),
        // `q4k` is the legacy tag preserved for back-compat; `kquant`
        // is the post-rename canonical tag. Both map to the same
        // `QuantFormat::Q4K` variant — they differ only in how the
        // value is spelled on disk in `index.json` / on the CLI.
        "q4k" | "q4_k" | "kquant" => Ok(larql_vindex::QuantFormat::Q4K),
        _ => Err(format!(
            "unknown quant format: {s} (expected: none, q4k, kquant)"
        )),
    }
}

fn parse_extract_level(s: &str) -> Result<larql_vindex::ExtractLevel, String> {
    match s.to_lowercase().as_str() {
        "browse" => Ok(larql_vindex::ExtractLevel::Browse),
        "attention" | "attn" => Ok(larql_vindex::ExtractLevel::Attention),
        "inference" | "infer" => Ok(larql_vindex::ExtractLevel::Inference),
        "all" => Ok(larql_vindex::ExtractLevel::All),
        _ => Err(format!(
            "unknown extract level: {s} \
             (expected: browse, attention, inference, all)"
        )),
    }
}

struct CliBuildCallbacks {
    stage_start: Option<Instant>,
    feature_bar: ProgressBar,
}

impl CliBuildCallbacks {
    fn new() -> Self {
        let feature_bar = ProgressBar::new(0);
        feature_bar.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        feature_bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());

        Self {
            stage_start: None,
            feature_bar,
        }
    }
}

impl IndexBuildCallbacks for CliBuildCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.feature_bar.finish_and_clear();
        eprintln!("\n── {stage} ──");
        self.stage_start = Some(Instant::now());
    }

    fn on_layer_start(&mut self, component: &str, layer: usize, total: usize) {
        self.feature_bar.reset();
        self.feature_bar
            .set_message(format!("{component} L{layer} ({}/{})", layer + 1, total));
    }

    fn on_feature_progress(&mut self, component: &str, _layer: usize, done: usize, total: usize) {
        if total > 0 {
            self.feature_bar.set_length(total as u64);
        }
        self.feature_bar.set_position(done as u64);
        if total == 0 {
            self.feature_bar
                .set_message(format!("{component} {done} records"));
        }
    }

    fn on_layer_done(&mut self, component: &str, layer: usize, elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        eprintln!("  {component} L{layer:2}: {:.1}s", elapsed_ms / 1000.0);
    }

    fn on_stage_done(&mut self, stage: &str, _elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        if let Some(start) = self.stage_start.take() {
            eprintln!("  {stage}: {:.1}s", start.elapsed().as_secs_f64());
        }
    }
}

pub fn run(args: ExtractIndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut callbacks = CliBuildCallbacks::new();
    let build_start = Instant::now();

    // Resolve extract level: --include-weights upgrades to All (backwards compat)
    let level = if args.include_weights {
        larql_vindex::ExtractLevel::All
    } else {
        args.level
    };

    // Dtype resolution:
    //   --f16                → F16
    //   --quant q4k          → F16 (Q4K quantizes attn + FFN; pairing that
    //                          with f32 gate_vectors/embeddings doubles
    //                          the side-channel footprint for zero accuracy
    //                          benefit. The f16 browse extract already
    //                          proves f16 side-channels are correct.)
    //   default              → F32
    // f16 is the default now; --f32 opts out. `--quant q4k` always
    // forces f16 on the side-channel tensors.
    let dtype = if args.f32 && args.quant != larql_vindex::QuantFormat::Q4K {
        larql_vindex::StorageDtype::F32
    } else {
        larql_vindex::StorageDtype::F16
    };

    if let Some(ref vectors_dir) = args.from_vectors {
        // Build from existing NDJSON files
        eprintln!("Building vindex from vectors: {}", vectors_dir.display());
        eprintln!("Output: {}", args.output.display());

        larql_vindex::build_vindex_from_vectors(vectors_dir, &args.output, &mut callbacks)?;

        if matches!(
            level,
            larql_vindex::ExtractLevel::Inference | larql_vindex::ExtractLevel::All
        ) {
            let model_name = args.model.as_deref().ok_or(
                "--model required with --level inference/all (need model to extract weights)",
            )?;
            eprintln!("\nLoading model for weights: {}", model_name);
            let model = InferenceModel::load(model_name)?;
            let weight_opts = larql_vindex::WriteWeightsOptions {
                level,
                ffn_compact: args.compact,
            };
            larql_vindex::write_model_weights_with_opts(
                model.weights(),
                &args.output,
                &mut callbacks,
                weight_opts,
            )?;
        }
    } else {
        // Build from model — streaming mode (mmap safetensors, no full model load)
        let model_name = args
            .model
            .as_deref()
            .ok_or("Either provide a model name or use --from-vectors")?;

        let model_path = larql_models::resolve_model_path(model_name)?;

        let level_str = match level {
            larql_vindex::ExtractLevel::Browse => "browse",
            larql_vindex::ExtractLevel::Attention => "attention",
            larql_vindex::ExtractLevel::Inference => "inference",
            larql_vindex::ExtractLevel::All => "all",
        };
        let dtype_str = match dtype {
            larql_vindex::StorageDtype::F32 => "f32",
            larql_vindex::StorageDtype::F16 => "f16",
        };
        eprintln!(
            "Extracting: {} → {} (level={}, dtype={}, quant={})",
            model_path.display(),
            args.output.display(),
            level_str,
            dtype_str,
            args.quant
        );

        let output = &args.output;

        // Find or create tokenizer
        let tok_path = model_path.join(TOKENIZER_JSON);
        let tokenizer = if tok_path.exists() {
            larql_vindex::tokenizers::Tokenizer::from_file(&tok_path)
                .map_err(|e| format!("failed to load tokenizer: {e}"))?
        } else {
            return Err(format!("tokenizer.json not found at {}", model_path.display()).into());
        };

        let weight_opts = larql_vindex::WriteWeightsOptions {
            level,
            ffn_compact: args.compact,
        };
        if args.drop_gate_vectors && args.quant != larql_vindex::QuantFormat::Q4K {
            return Err(
                "--drop-gate-vectors requires --quant q4k (gate is rebuilt from Q4K at load)"
                    .into(),
            );
        }
        if args.down_q4k && args.quant != larql_vindex::QuantFormat::Q4K {
            return Err(
                "--down-q4k requires --quant q4k (only the Q4K writer honours this flag)".into(),
            );
        }
        if args.feature_major_down && args.quant != larql_vindex::QuantFormat::Q4K {
            return Err(
                "--feature-major-down requires --quant q4k (only the Q4K writer honours this flag)"
                    .into(),
            );
        }
        let q4k_opts = larql_vindex::KquantWriteOptions {
            down_proj: if args.down_q4k {
                larql_vindex::DownProjFormat::Q4K
            } else {
                larql_vindex::DownProjFormat::Q6K
            },
            feature_major_down: args.feature_major_down,
        };

        // Per-expert SVD-summary tier — opt-in via flag. Threaded as env var
        // so the streaming gate path can read it without an API break.
        if args.summary_features_per_expert > 0 {
            std::env::set_var(
                "LARQL_SUMMARY_FEATURES_PER_EXPERT",
                args.summary_features_per_expert.to_string(),
            );
        }

        larql_vindex::build_vindex_streaming(
            &model_path,
            &tokenizer,
            model_name,
            output,
            args.down_top_k,
            level,
            dtype,
            args.quant,
            weight_opts,
            q4k_opts,
            args.drop_gate_vectors,
            &mut callbacks,
        )?;

        // Opportunistically copy HF metadata (tokenizer_config.json,
        // special_tokens_map.json, generation_config.json) from the source
        // directory into the vindex. Chat-template-aware runtimes read
        // `tokenizer_config.json::chat_template` from here; missing files
        // are silently skipped.
        if let Err(e) = larql_vindex::snapshot_hf_metadata(&model_path, output) {
            eprintln!("  warning: failed to snapshot HF metadata: {e}");
        }
    }

    callbacks.feature_bar.finish_and_clear();
    let build_elapsed = build_start.elapsed();

    // Print summary
    eprintln!("\n── Summary ──");
    eprintln!("  Output: {}", args.output.display());

    if build_elapsed.as_secs() >= 60 {
        eprintln!("  Build time: {:.1}min", build_elapsed.as_secs_f64() / 60.0);
    } else {
        eprintln!("  Build time: {:.1}s", build_elapsed.as_secs_f64());
    }

    for name in &[
        INDEX_JSON,
        GATE_VECTORS_BIN,
        EMBEDDINGS_BIN,
        "down_meta.jsonl",
        DOWN_META_BIN,
        TOKENIZER_JSON,
        ATTN_WEIGHTS_BIN,
        UP_WEIGHTS_BIN,
        DOWN_WEIGHTS_BIN,
        NORMS_BIN,
        LM_HEAD_BIN,
        WEIGHT_MANIFEST_JSON,
    ] {
        let path = args.output.join(name);
        if let Ok(meta) = std::fs::metadata(&path) {
            let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
            if size_mb > 1024.0 {
                eprintln!("  {name}: {:.2} GB", size_mb / 1024.0);
            } else if size_mb > 0.1 {
                eprintln!("  {name}: {:.1} MB", size_mb);
            } else {
                let size_kb = meta.len() as f64 / 1024.0;
                eprintln!("  {name}: {:.1} KB", size_kb);
            }
        } else {
            eprintln!("  {name}: (not found)");
        }
    }

    // Total: sum all files in the directory
    let total_size: u64 = std::fs::read_dir(&args.output)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    eprintln!(
        "  Total: {:.2} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    eprintln!("\nUsage:");
    eprintln!(
        "  larql walk --index {} -p \"The capital of France is\"",
        args.output.display()
    );

    Ok(())
}
