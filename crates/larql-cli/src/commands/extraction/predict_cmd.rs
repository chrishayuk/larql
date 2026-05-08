//! `larql predict` — graph-walk inference.
//!
//! One production backend: the walk kernel in `WalkFfn`. Density is controlled
//! by `--k` (`full` = dense walk, numeric = top-K sparse). `--ffn weights` stays
//! around as a debug/reference path — it is the classic matmul FFN and must be
//! bit-identical to the walk kernel on a sane model.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Args;

use larql_inference::{
    calibrate_scalar_gains, predict, predict_with_ffn, predict_with_strategy,
    vindex::{WalkFfn, WalkFfnConfig},
    FfnBackend, InferenceModel, InferenceWeights, LayerMode, WeightFfn,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

#[derive(Args)]
pub struct PredictArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Prompt text to predict the next token for.
    #[arg(short, long)]
    prompt: String,

    /// Number of top predictions to show.
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,

    /// FFN backend: `graph` (default, production) or `weights` (debug reference).
    #[arg(long, default_value = "graph")]
    ffn: String,

    /// Density for the graph backend. `full` = dense walk (all features), or
    /// a numeric K for top-K sparse walk.
    #[arg(long, default_value = "full")]
    k: String,

    /// Vindex directory (required for --ffn graph).
    #[arg(long)]
    vindex: Option<PathBuf>,

    /// Compare backends side by side: graph at K=full/5000/1000/500/200/100
    /// plus the weights debug reference.
    #[arg(long)]
    compare: bool,

    /// Layer strategy with scalar bypass: "walk:0-8,scalar:9-14,walk:15-33".
    /// Scalar gains are auto-calibrated from a forward pass on the same prompt.
    /// Supports: walk, sparse<K>, scalar.
    #[arg(long)]
    mode: Option<String>,
}

pub fn run(args: PredictArgs) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new(&args.model);
    if is_vindex_dir(model_path) {
        return run_vindex(&args, model_path);
    }

    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    eprintln!(
        "  {} layers, hidden_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        start.elapsed().as_secs_f64(),
    );

    eprintln!("Prompt: {:?}", args.prompt);
    let token_ids = larql_inference::encode_prompt(
        model.tokenizer(),
        &*model.weights().arch,
        args.prompt.as_str(),
    )
    .map_err(|e| format!("tokenize error: {e}"))?;
    eprintln!("  {} tokens: {:?}", token_ids.len(), token_ids);

    if args.compare {
        return run_comparison(&model, &token_ids, args.top_k, &args);
    }
    if let Some(ref spec) = args.mode {
        return run_with_mode(&model, &token_ids, args.top_k, spec, &args);
    }
    run_single(&model, &token_ids, args.top_k, &args)
}

fn is_vindex_dir(path: &Path) -> bool {
    path.join("index.json").is_file()
}

fn run_vindex(args: &PredictArgs, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if args.compare {
        return Err("--compare is only supported for raw model checkpoints".into());
    }
    if args.mode.is_some() {
        return Err("--mode is only supported for raw model checkpoints".into());
    }

    eprintln!("Loading vindex model: {}", path.display());
    let start = Instant::now();
    let config = larql_vindex::load_vindex_config(path)?;
    let mut cb = SilentLoadCallbacks;
    let mut weights = InferenceWeights::load(path, &config, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)?;
    eprintln!(
        "  {} layers, hidden_size={} ({}, {:.1}s)",
        weights.as_weights().num_layers,
        weights.as_weights().hidden_size,
        if weights.is_quantised() {
            "quantized"
        } else {
            "dense"
        },
        start.elapsed().as_secs_f64(),
    );

    eprintln!("Prompt: {:?}", args.prompt);
    let token_ids = larql_inference::encode_prompt(
        &tokenizer,
        &*weights.as_weights().arch,
        args.prompt.as_str(),
    )
    .map_err(|e| format!("tokenize error: {e}"))?;
    eprintln!("  {} tokens: {:?}", token_ids.len(), token_ids);

    let t = Instant::now();
    let result = weights.predict_dense(&tokenizer, &token_ids, args.top_k);
    eprintln!("  Forward pass: {:.1}s", t.elapsed().as_secs_f64());
    print_predictions("vindex", &result.predictions);

    Ok(())
}

// ── Single backend ─────────────────────────────────────────────────────

fn run_single(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    args: &PredictArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();

    match args.ffn.as_str() {
        "graph" => {
            let vindex_path = args.vindex.as_ref().ok_or(
                "--vindex required for --ffn graph. Build with: larql extract-index <model> -o out.vindex",
            )?;
            eprintln!("Loading vindex: {}", vindex_path.display());
            let t = Instant::now();
            let mut cb = SilentLoadCallbacks;
            let index = VectorIndex::load_vindex(vindex_path, &mut cb)?;
            eprintln!(
                "  {} layers, {} vectors ({:.1}s)",
                index.num_layers,
                index.total_gate_vectors(),
                t.elapsed().as_secs_f64(),
            );

            let config = parse_k(&args.k, weights.num_layers)?;
            eprintln!("FFN: graph (k={})", args.k);
            let walk = WalkFfn::from_config(weights, &index, config);
            run_ffn(&walk, weights, model.tokenizer(), token_ids, top_k, "graph");
        }
        "weights" => {
            eprintln!("FFN: weights (debug reference — classic matmul)");
            let ffn = WeightFfn { weights };
            run_ffn(
                &ffn,
                weights,
                model.tokenizer(),
                token_ids,
                top_k,
                "weights",
            );
        }
        other => return Err(format!("unknown --ffn: {other}. Use `graph` or `weights`.").into()),
    }

    Ok(())
}

fn run_ffn(
    ffn: &dyn FfnBackend,
    weights: &larql_inference::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    label: &str,
) {
    let t = Instant::now();
    let result = predict_with_ffn(weights, tokenizer, token_ids, top_k, ffn);
    eprintln!("  Forward pass: {:.1}s", t.elapsed().as_secs_f64());
    print_predictions(label, &result.predictions);
}

fn parse_k(k: &str, num_layers: usize) -> Result<WalkFfnConfig, Box<dyn std::error::Error>> {
    if k == "full" || k == "unlimited" {
        Ok(WalkFfnConfig::dense(num_layers))
    } else {
        let n: usize = k
            .parse()
            .map_err(|_| format!("--k must be `full` or a positive integer, got {k:?}"))?;
        Ok(WalkFfnConfig::sparse(num_layers, n))
    }
}

// ── --mode (scalar bypass research tool) ───────────────────────────────

fn run_with_mode(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    spec: &str,
    args: &PredictArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();
    let num_layers = weights.num_layers;

    #[derive(Debug, Clone)]
    enum Kind {
        Walk,
        Sparse(usize),
        Scalar,
    }

    let mut kinds = vec![Kind::Walk; num_layers];
    for part in spec.split(',') {
        let (name, range) = part
            .split_once(':')
            .ok_or_else(|| format!("invalid mode spec: {part}"))?;
        let (start, end) = if let Some((a, b)) = range.split_once('-') {
            (a.parse::<usize>()?, b.parse::<usize>()?)
        } else {
            let l: usize = range.parse()?;
            (l, l)
        };
        let kind = match name {
            "walk" | "dense" => Kind::Walk,
            "scalar" => Kind::Scalar,
            n if n.starts_with("sparse") => {
                let k_str = &n[6..];
                let k: usize = if k_str.is_empty() {
                    100
                } else {
                    k_str.parse()?
                };
                Kind::Sparse(k)
            }
            other => {
                return Err(format!("unknown mode: {other}. Use walk, sparse<K>, scalar.").into())
            }
        };
        for slot in kinds
            .iter_mut()
            .take(end.min(num_layers - 1) + 1)
            .skip(start)
        {
            *slot = kind.clone();
        }
    }

    let vindex_path = args.vindex.as_ref().ok_or(
        "--vindex required for --mode. Build with: larql extract-index <model> -o out.vindex",
    )?;
    eprintln!("Loading vindex: {}", vindex_path.display());
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(vindex_path, &mut cb)?;

    let has_scalar = kinds.iter().any(|k| matches!(k, Kind::Scalar));

    // Build per-layer K vector for a single WalkFfn driving walk + sparse layers.
    let mut k_per_layer: Vec<Option<usize>> = vec![None; num_layers];
    for (l, k) in kinds.iter().enumerate() {
        match k {
            Kind::Walk => k_per_layer[l] = None,
            Kind::Sparse(k) => k_per_layer[l] = Some(*k),
            Kind::Scalar => {} // scalar layers bypass compute entirely
        }
    }
    let walk = WalkFfn::from_config(
        weights,
        &index,
        WalkFfnConfig {
            k_per_layer,
            activation_floor: 0.0,
        },
    );

    if has_scalar {
        eprintln!("Calibrating scalar gains…");
        let t = Instant::now();
        let gains = calibrate_scalar_gains(weights, token_ids);
        eprintln!(
            "  {} layers in {:.1}s",
            gains.len(),
            t.elapsed().as_secs_f64()
        );

        let mut strategy: Vec<LayerMode> = Vec::with_capacity(num_layers);
        for (l, kind) in kinds.iter().enumerate() {
            match kind {
                Kind::Scalar => strategy.push(LayerMode::ScalarGain(gains[l])),
                _ => strategy.push(LayerMode::Compute(&walk)),
            }
        }

        eprintln!("Mode: {spec}");
        let t = Instant::now();
        let result = predict_with_strategy(weights, model.tokenizer(), token_ids, top_k, &strategy);
        eprintln!("  Forward pass: {:.1}s", t.elapsed().as_secs_f64());
        print_predictions(spec, &result.predictions);

        eprintln!("\nBaseline (walk all layers):");
        let t = Instant::now();
        let baseline = predict(weights, model.tokenizer(), token_ids, top_k);
        eprintln!("  Forward pass: {:.1}s", t.elapsed().as_secs_f64());
        print_predictions("walk (baseline)", &baseline.predictions);
    } else {
        // No scalar — one WalkFfn handles everything via its per-layer K vector.
        eprintln!("Mode: {spec}");
        let t = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &walk);
        eprintln!("  Forward pass: {:.1}s", t.elapsed().as_secs_f64());
        print_predictions(spec, &result.predictions);
    }

    Ok(())
}

// ── --compare ──────────────────────────────────────────────────────────

fn run_comparison(
    model: &InferenceModel,
    token_ids: &[u32],
    top_k: usize,
    args: &PredictArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let weights = model.weights();

    println!();
    println!(
        "{:<20} {:<15} {:>8} {:>10}  {:<20}",
        "Backend", "Top-1", "Prob", "Time", "Top-3"
    );
    println!("{}", "-".repeat(80));

    // Weights (debug reference)
    let t = Instant::now();
    let weight_ffn = WeightFfn { weights };
    let dense = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &weight_ffn);
    print_row("weights (reference)", &dense.predictions, t.elapsed());

    // Graph at various K values
    let vindex_path = args
        .vindex
        .as_ref()
        .ok_or("--vindex required for --compare. Build with: larql extract-index <model>.")?;
    eprintln!("  Loading vindex: {}", vindex_path.display());
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(vindex_path, &mut cb)?;

    let ks: Vec<(&str, WalkFfnConfig)> = vec![
        ("graph:full", WalkFfnConfig::dense(weights.num_layers)),
        (
            "graph:5000",
            WalkFfnConfig::sparse(weights.num_layers, 5000),
        ),
        (
            "graph:1000",
            WalkFfnConfig::sparse(weights.num_layers, 1000),
        ),
        ("graph:500", WalkFfnConfig::sparse(weights.num_layers, 500)),
        ("graph:200", WalkFfnConfig::sparse(weights.num_layers, 200)),
        ("graph:100", WalkFfnConfig::sparse(weights.num_layers, 100)),
    ];

    for (label, config) in ks {
        let walk = WalkFfn::from_config(weights, &index, config);
        let t = Instant::now();
        let result = predict_with_ffn(weights, model.tokenizer(), token_ids, top_k, &walk);
        print_row(label, &result.predictions, t.elapsed());
    }

    Ok(())
}

// ── Output helpers ─────────────────────────────────────────────────────

fn print_predictions(label: &str, predictions: &[(String, f64)]) {
    println!();
    println!("Top predictions ({label}):");
    for (i, (token, prob)) in predictions.iter().enumerate() {
        println!(
            "  {:2}. {:20} {:.4} ({:.2}%)",
            i + 1,
            token,
            prob,
            prob * 100.0,
        );
    }
}

fn print_row(label: &str, predictions: &[(String, f64)], elapsed: std::time::Duration) {
    let (top1, prob1) = predictions
        .first()
        .map(|(t, p)| (t.as_str(), *p))
        .unwrap_or(("?", 0.0));
    let top3: String = predictions
        .iter()
        .take(3)
        .map(|(t, _)| t.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "{:<20} {:<15} {:>7.2}% {:>8.0}ms  {:<20}",
        label,
        top1,
        prob1 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
        top3,
    );
}
