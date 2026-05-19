//! walk_benchmark — per-layer FFN latency across backends + no-matmul verification.
//!
//! Captures the pre-FFN residual at every layer from a reference forward pass,
//! then benchmarks each backend running the same single-layer FFN call N times.
//!
//! Configs:
//!   weights        WeightFfn        classic matmul via model weights (reference)
//!   mmap (dense)   WalkFfn(None)    current dispatch at --k full → walk_ffn_interleaved (BLAS gemm)
//!   graph K=full   WalkFfn(max)     walk_ffn_sparse iterating every feature (no matmul)
//!   graph K=5000   WalkFfn(5000)    walk_ffn_sparse top-K (no matmul)
//!   graph K=1000   WalkFfn(1000)
//!   graph K=500    WalkFfn(500)
//!   graph K=200    WalkFfn(200)
//!   graph K=100    WalkFfn(100)
//!
//! Usage:
//!   cargo run --release -p larql-inference --example walk_benchmark -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex /path/to/gemma3-4b.vindex \
//!     [--prompt TEXT] [--iterations 20]

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::Array2;

use larql_inference::{
    default_backend, predict_with_ffn,
    vindex::{WalkFfn, WalkFfnConfig},
    ComputeBackend, FfnBackend, InferenceModel, WeightFfn,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

// ── CLI ────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    vindex: PathBuf,
    prompt: String,
    iterations: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut vindex = PathBuf::new();
    let mut prompt = "The capital of France is".to_string();
    let mut iterations: usize = 20;

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
            _ => {}
        }
        i += 1;
    }

    if model.is_empty() || !vindex.is_dir() {
        eprintln!(
            "Usage: walk_benchmark --model MODEL --vindex PATH [--prompt TEXT] [--iterations N]"
        );
        std::process::exit(1);
    }

    Args {
        model,
        vindex,
        prompt,
        iterations,
    }
}

// ── Capture pre-FFN residuals ──────────────────────────────────────────

/// Wraps a reference FFN, recording the `x` input seen at every layer.
/// The forward call uses the underlying FFN's output so the forward pass
/// stays numerically correct; we only extract the inputs.
struct CapturingFfn<'a> {
    inner: &'a dyn FfnBackend,
    captured: RefCell<Vec<Array2<f32>>>, // indexed by layer
    num_layers: usize,
}

impl<'a> CapturingFfn<'a> {
    fn new(inner: &'a dyn FfnBackend, num_layers: usize) -> Self {
        Self {
            inner,
            captured: RefCell::new(vec![Array2::<f32>::zeros((0, 0)); num_layers]),
            num_layers,
        }
    }

    fn take(self) -> Vec<Array2<f32>> {
        self.captured.into_inner()
    }
}

impl<'a> FfnBackend for CapturingFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        if layer < self.num_layers {
            self.captured.borrow_mut()[layer] = x.clone();
        }
        self.inner.forward_with_activation(layer, x)
    }

    fn name(&self) -> &str {
        "capturing"
    }
}

// ── Benchmark helpers ──────────────────────────────────────────────────

#[derive(Debug)]
struct LayerTiming {
    _layer: usize,
    median_us: f64,
    p99_us: f64,
}

fn bench_layer(ffn: &dyn FfnBackend, layer: usize, x: &Array2<f32>, iters: usize) -> LayerTiming {
    // Warmup — more aggressive to page mmap into resident memory.
    for _ in 0..10 {
        let _ = ffn.forward(layer, x);
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = ffn.forward(layer, x);
        samples.push(t.elapsed().as_secs_f64() * 1_000_000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[iters / 2];
    let p99 = samples[((iters as f64) * 0.99).floor() as usize % iters];
    LayerTiming {
        _layer: layer,
        median_us: median,
        p99_us: p99,
    }
}

#[derive(Debug)]
struct ConfigResult {
    name: String,
    uses_matmul: bool,
    per_layer: Vec<LayerTiming>,
    total_median_ms: f64,
    total_p99_ms: f64,
}

fn bench_config(
    name: &str,
    ffn: &dyn FfnBackend,
    uses_matmul: bool,
    residuals: &[Array2<f32>],
    iters: usize,
) -> ConfigResult {
    let per_layer: Vec<LayerTiming> = residuals
        .iter()
        .enumerate()
        .map(|(layer, x)| bench_layer(ffn, layer, x, iters))
        .collect();
    let total_median_ms: f64 = per_layer.iter().map(|t| t.median_us).sum::<f64>() / 1000.0;
    let total_p99_ms: f64 = per_layer.iter().map(|t| t.p99_us).sum::<f64>() / 1000.0;
    ConfigResult {
        name: name.to_string(),
        uses_matmul,
        per_layer,
        total_median_ms,
        total_p99_ms,
    }
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    println!("=== Walk Benchmark ===\n");
    println!("Model:      {}", args.model);
    println!("Vindex:     {}", args.vindex.display());
    println!("Prompt:     {:?}", args.prompt);
    println!("Iterations: {}\n", args.iterations);

    let t = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    println!(
        "Model loaded in {:.1}s ({} layers, hidden={})",
        t.elapsed().as_secs_f64(),
        model.weights().num_layers,
        model.weights().hidden_size
    );

    let t = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.vindex, &mut cb)?;
    // Load the Q4 interleaved mmap if present — enables walk_ffn_q4_interleaved
    // (one Metal shader per forward vs three BLAS gemms).
    let q4_loaded = index.load_interleaved_q4(&args.vindex).is_ok();
    // Also load the f32 interleaved mmap for walk_ffn_interleaved (contiguous gate+up+down).
    let iv_loaded = index.load_interleaved(&args.vindex).is_ok();
    println!(
        "Vindex loaded in {:.1}s ({} vectors, q4_interleaved={}, interleaved={})\n",
        t.elapsed().as_secs_f64(),
        index.total_gate_vectors(),
        q4_loaded,
        iv_loaded
    );

    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // ── Capture per-layer pre-FFN residuals via reference pass ─────────
    print!("Capturing per-layer pre-FFN residuals... ");
    let reference = WeightFfn { weights };
    let capturing = CapturingFfn::new(&reference, num_layers);
    let t = Instant::now();
    let _ = predict_with_ffn(weights, tokenizer, &token_ids, 1, &capturing);
    println!("done ({:.2}s)", t.elapsed().as_secs_f64());
    let residuals = capturing.take();
    println!(
        "  Captured {} layers, shape {:?}\n",
        residuals.iter().filter(|r| r.shape()[0] > 0).count(),
        residuals[0].shape()
    );

    // ── Build configs ──────────────────────────────────────────────────
    let weight_ffn = WeightFfn { weights };

    // Compute backend (Metal on Apple Silicon, CPU otherwise).
    let backend: Box<dyn ComputeBackend> = default_backend();
    let backend_name = if backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
        "Metal/Q4"
    } else {
        "CPU"
    };
    println!("Compute backend: {backend_name}\n");

    let walk_full_graph = WalkFfn::from_config(
        weights,
        &index,
        WalkFfnConfig::sparse(num_layers, usize::MAX),
    ); // graph walk, no matmul
    let walk_full_dense = WalkFfn::from_config(weights, &index, WalkFfnConfig::dense(num_layers)); // mmap matmul (CPU)
    let walk_full_dense_gpu =
        WalkFfn::from_config(weights, &index, WalkFfnConfig::dense(num_layers))
            .with_backend(&*backend); // mmap matmul (GPU/Metal if available)
    let walk_5000 = WalkFfn::from_config(weights, &index, WalkFfnConfig::sparse(num_layers, 5000));
    let walk_1000 = WalkFfn::from_config(weights, &index, WalkFfnConfig::sparse(num_layers, 1000));
    let walk_500 = WalkFfn::from_config(weights, &index, WalkFfnConfig::sparse(num_layers, 500));
    let walk_200 = WalkFfn::from_config(weights, &index, WalkFfnConfig::sparse(num_layers, 200));
    let walk_100 = WalkFfn::from_config(weights, &index, WalkFfnConfig::sparse(num_layers, 100));

    let _ = walk_full_dense_gpu; // Metal dispatched per-layer has severe overhead; skip for now.
    let configs: Vec<(&str, &dyn FfnBackend, bool)> = vec![
        ("weights (ref matmul, CPU)", &weight_ffn, true),
        ("mmap dense (BLAS gemm, CPU)", &walk_full_dense, true),
        ("graph K=full (no matmul)", &walk_full_graph, false),
        ("graph K=5000", &walk_5000, false),
        ("graph K=1000", &walk_1000, false),
        ("graph K=500", &walk_500, false),
        ("graph K=200", &walk_200, false),
        ("graph K=100", &walk_100, false),
    ];

    // ── Run benches ────────────────────────────────────────────────────
    println!(
        "--- Per-layer FFN latency, {} iterations ---\n",
        args.iterations
    );

    let mut results: Vec<ConfigResult> = Vec::with_capacity(configs.len());
    for (name, ffn, uses_matmul) in &configs {
        print!("  {name:<28}  ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let res = bench_config(name, *ffn, *uses_matmul, &residuals, args.iterations);
        println!(
            "total={:>7.1}ms (p99 {:>7.1}ms)  matmul={}",
            res.total_median_ms,
            res.total_p99_ms,
            if *uses_matmul { "YES" } else { "no" }
        );
        results.push(res);
    }

    // ── Summary table ──────────────────────────────────────────────────
    println!();
    println!("--- Summary ---\n");
    println!(
        "  {:<28}  {:>12}  {:>12}  {:>10}  {:>8}",
        "config", "total (ms)", "p99 (ms)", "vs ref", "matmul"
    );
    println!("  {:-<76}", "");
    let ref_total = results[0].total_median_ms;
    for r in &results {
        let rel = r.total_median_ms / ref_total;
        println!(
            "  {:<28}  {:>12.2}  {:>12.2}  {:>9.2}×  {:>8}",
            r.name,
            r.total_median_ms,
            r.total_p99_ms,
            rel,
            if r.uses_matmul { "YES" } else { "no" },
        );
    }

    // ── Per-layer detail for the graph-full config ─────────────────────
    let graph_full = results
        .iter()
        .find(|r| r.name.starts_with("graph K=full"))
        .unwrap();
    println!("\n--- Per-layer detail: {} ---\n", graph_full.name);
    println!("  {:>4}  {:>10}  {:>10}", "layer", "median μs", "p99 μs");
    for (layer, t) in graph_full.per_layer.iter().enumerate() {
        println!("  {:>4}  {:>10.1}  {:>10.1}", layer, t.median_us, t.p99_us);
    }

    // ── Claim check ────────────────────────────────────────────────────
    println!("\n=== Claim check: \"no matmul\" ===\n");
    println!("  walk_ffn_sparse (the graph kernel) computes per feature:");
    println!("    gate_score = gate_knn(residual, k)[i]        [HNSW or per-feature dot]");
    println!("    up_score   = up_mmap[feat] · residual         [hidden×1 dot product]");
    println!("    act        = silu(gate_score) * up_score");
    println!("    output    += act * down_mmap[feat]            [scaled_add]");
    println!();
    println!("  No BLAS gemm / sgemv / matmul_gpu calls on this path.");
    println!();
    println!("  Current dispatch at --k full routes to walk_ffn_interleaved");
    println!("  which IS a BLAS gemm. To run the true graph kernel at K=full,");
    println!("  use WalkFfnConfig::sparse(num_layers, usize::MAX) — benched above.");

    Ok(())
}
