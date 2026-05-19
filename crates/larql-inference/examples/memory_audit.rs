//! memory_audit — RSS tracking for vindex + walk inference.
//!
//! Checkpoints resident set size (RSS) at every phase:
//!   (1) baseline
//!   (2) after InferenceModel::load_walk_only (drops FFN weights post-load)
//!   (3) after VectorIndex::load_vindex (+ interleaved Q4/f32 if present)
//!   (4) after residual warmup pass
//!   (5) per forward-pass over N iterations (leak check)
//!
//! Usage:
//!   cargo run --release -p larql-inference --example memory_audit -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex /path/to/vindex \
//!     [--prompt TEXT] [--iterations 20] [--walk-only]

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::{
    default_backend, predict_with_ffn,
    vindex::{WalkFfn, WalkFfnConfig},
    InferenceModel,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

// ── CLI ────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    vindex: PathBuf,
    prompt: String,
    iterations: usize,
    walk_only: bool,
    k: String,
    hnsw_ef: Option<usize>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut vindex = PathBuf::new();
    let mut prompt = "The capital of France is".to_string();
    let mut iterations: usize = 20;
    let mut walk_only = false;
    let mut k = "full".to_string();
    let mut hnsw_ef: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args[i].clone();
            }
            "--vindex" => {
                i += 1;
                vindex = PathBuf::from(&args[i]);
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--iterations" => {
                i += 1;
                iterations = args[i].parse().unwrap_or(20);
            }
            "--walk-only" => {
                walk_only = true;
            }
            "--k" => {
                i += 1;
                k = args[i].clone();
            }
            "--hnsw" => {
                i += 1;
                hnsw_ef = args[i].parse().ok();
            }
            _ => {}
        }
        i += 1;
    }

    if model.is_empty() || !vindex.is_dir() {
        eprintln!("Usage: memory_audit --model MODEL --vindex PATH [--walk-only] [--k full|N] [--hnsw EF] [--prompt TEXT] [--iterations N]");
        std::process::exit(1);
    }
    Args {
        model,
        vindex,
        prompt,
        iterations,
        walk_only,
        k,
        hnsw_ef,
    }
}

// ── RSS sampling ────────────────────────────────────────────────────────

/// Returns (resident_mb, virtual_mb) for the current process. macOS-tolerant
/// via `ps`. `ps` reports kilobytes; divide by 1024 for MB.
fn mem_mb() -> (u64, u64) {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=,vsz=", "-p", &pid])
        .output();
    match output {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = s.split_whitespace().collect();
            let rss_kb: u64 = parts.first().and_then(|p| p.parse().ok()).unwrap_or(0);
            let vsz_kb: u64 = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0);
            (rss_kb / 1024, vsz_kb / 1024)
        }
        Err(_) => (0, 0),
    }
}

fn checkpoint(label: &str, started: Instant, baseline: (u64, u64)) -> (u64, u64) {
    let (rss, vsz) = mem_mb();
    let dr = rss as i64 - baseline.0 as i64;
    let dv = vsz as i64 - baseline.1 as i64;
    println!(
        "  [{:>6.1}s] {label:<38}  RSS={rss:>7} MB  (Δ{dr:+>7} MB)  VSZ={vsz:>7} MB  (Δ{dv:+>7} MB)",
        started.elapsed().as_secs_f64()
    );
    (rss, vsz)
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    println!("=== Memory Audit ===\n");
    println!("Model:      {}", args.model);
    println!("Vindex:     {}", args.vindex.display());
    println!("Prompt:     {:?}", args.prompt);
    println!("Iterations: {}", args.iterations);
    println!("Walk-only:  {}\n", args.walk_only);

    let started = Instant::now();
    let baseline = mem_mb();
    println!(
        "  [{:>6.1}s] {:<38}  RSS={:>7} MB                   VSZ={:>7} MB",
        started.elapsed().as_secs_f64(),
        "baseline (before load)",
        baseline.0,
        baseline.1
    );

    // ── Load model ─────────────────────────────────────────────────────
    let model = if args.walk_only {
        InferenceModel::load_walk_only(&args.model)?
    } else {
        InferenceModel::load(&args.model)?
    };
    checkpoint(
        if args.walk_only {
            "after InferenceModel::load_walk_only"
        } else {
            "after InferenceModel::load (full)"
        },
        started,
        baseline,
    );

    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    println!(
        "\n  Model: {} layers, hidden={}, intermediate={}\n",
        num_layers, weights.hidden_size, weights.intermediate_size
    );

    // ── Load vindex ────────────────────────────────────────────────────
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.vindex, &mut cb)?;
    checkpoint("after VectorIndex::load_vindex", started, baseline);

    let q4 = index.load_interleaved_q4(&args.vindex).is_ok();
    let q4k = index.load_interleaved_kquant(&args.vindex).is_ok();
    let iv = index.load_interleaved(&args.vindex).is_ok();
    println!(
        "\n  Vindex: {} vectors, q4_interleaved={}, q4k_interleaved={}, f32_interleaved={}\n",
        index.total_gate_vectors(),
        q4,
        q4k,
        iv
    );
    checkpoint("after interleaved mmap loads", started, baseline);

    if let Some(ef) = args.hnsw_ef {
        index.enable_hnsw(ef);
        println!("  HNSW enabled with ef_search={ef} (indexes build lazily per layer)\n");
    }

    // ── Encode prompt ──────────────────────────────────────────────────
    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // ── Warmup forward pass ────────────────────────────────────────────
    let k_val: usize = if args.k == "full" || args.k == "unlimited" {
        usize::MAX
    } else {
        args.k.parse().unwrap_or(usize::MAX)
    };
    println!(
        "  K = {} ({})\n",
        args.k,
        if k_val == usize::MAX {
            "dense walk".into()
        } else {
            format!("sparse K={k_val}")
        }
    );
    // Detect best compute backend: Metal when available (Apple Silicon with
    // the `metal` feature), CPU-BLAS otherwise. Walk matmul paths route
    // through this backend automatically.
    let backend = default_backend();
    println!(
        "  Compute backend: {}\n",
        if backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
            "Metal (or CPU w/ Q4)"
        } else {
            "CPU (BLAS)"
        }
    );
    let walk = WalkFfn::from_config(weights, &index, WalkFfnConfig::sparse(num_layers, k_val))
        .with_backend(&*backend);

    let t = Instant::now();
    let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk);
    println!();
    println!("  Warmup forward: {:.1}s", t.elapsed().as_secs_f64());
    let prev = checkpoint("after warmup pass", started, baseline);

    // ── Leak check: run N more iterations, measure RSS between ─────────
    println!("\n--- Leak check: {} forward passes ---", args.iterations);
    let mut max_rss = prev.0;
    let mut prev_rss = prev.0;
    let mut rss_deltas: Vec<i64> = Vec::with_capacity(args.iterations);

    for i in 0..args.iterations {
        let t = Instant::now();
        let result = predict_with_ffn(weights, tokenizer, &token_ids, 1, &walk);
        let dur_ms = t.elapsed().as_secs_f64() * 1000.0;
        let top1 = result
            .predictions
            .first()
            .map(|(t, p)| format!("{t:?} {:.3}", p))
            .unwrap_or_else(|| "?".into());
        let (rss, _) = mem_mb();
        let drss = rss as i64 - prev_rss as i64;
        if rss > max_rss {
            max_rss = rss;
        }
        rss_deltas.push(drss);
        prev_rss = rss;
        println!(
            "  iter {:>3}  forward={:>6.1}ms  RSS={:>7} MB  (Δ{:+>6})  top1={top1}",
            i + 1,
            dur_ms,
            rss,
            drss,
        );
    }

    // ── Summary ────────────────────────────────────────────────────────
    let (final_rss, final_vsz) = mem_mb();
    let total_drift: i64 = rss_deltas.iter().sum();

    println!("\n=== Summary ===");
    println!(
        "  Baseline:       RSS={:>7} MB  VSZ={:>7} MB",
        baseline.0, baseline.1
    );
    println!("  Peak:           RSS={:>7} MB", max_rss);
    println!(
        "  Final:          RSS={:>7} MB  VSZ={:>7} MB",
        final_rss, final_vsz
    );
    println!(
        "  RSS drift over {} iters: {:+} MB",
        args.iterations, total_drift
    );
    let suspect = total_drift.abs() > (args.iterations as i64) * 5; // >5MB/iter drift is suspect
    println!(
        "  Leak verdict:   {}",
        if suspect {
            "SUSPECT (drift > 5 MB/iter)"
        } else {
            "OK"
        }
    );

    Ok(())
}
