//! walk_profile — decomposes walk_ffn_sparse cost into gate retrieval vs walk loop.
//!
//! The walk_benchmark example showed non-monotonic latency in K:
//!   K=full 357ms, K=5000 343ms, K=1000 690ms, K=500 450ms, K=200 240ms, K=100 185ms.
//! Mid-K is slower than either tail. This example isolates the two cost centres:
//!   (A) gate retrieval — `GateIndex::gate_knn` / `gate_walk` / `gate_knn_q4`
//!   (B) walk loop     — per-feature up.dot + silu(gate) * up + scaled_add(down)
//! to identify whether mid-K cost lives in KNN selection or in the walk loop.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example walk_profile -- \
//!     --model google/gemma-3-4b-it --vindex /path/to/vindex [--iterations 20]

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::Array2;

use larql_inference::{predict_with_ffn, vindex::WalkFfn, FfnBackend, InferenceModel, WeightFfn};
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
            "Usage: walk_profile --model MODEL --vindex PATH [--prompt TEXT] [--iterations N]"
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

// ── Residual capture ───────────────────────────────────────────────────

struct CapturingFfn<'a> {
    inner: &'a dyn FfnBackend,
    captured: RefCell<Vec<Array2<f32>>>,
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
        if layer < self.num_layers {
            self.captured.borrow_mut()[layer] = x.clone();
        }
        self.inner.forward(layer, x)
    }
    fn name(&self) -> &str {
        "capturing"
    }
}

// ── Timing helpers ─────────────────────────────────────────────────────

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[((samples.len() as f64) * p)
        .floor()
        .min(samples.len() as f64 - 1.0) as usize]
}

#[derive(Default, Debug)]
struct Stage {
    median_us: f64,
    #[allow(dead_code)]
    p99_us: f64,
}

fn measure<F: FnMut()>(iters: usize, mut f: F) -> Stage {
    for _ in 0..3 {
        f();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1_000_000.0);
    }
    Stage {
        median_us: percentile(&mut samples, 0.5),
        p99_us: percentile(&mut samples, 0.99),
    }
}

// ── Walk loop reimplementation (matches walk_ffn_sparse math) ──────────

fn walk_loop(
    index: &VectorIndex,
    weights: &larql_inference::ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    hits: &[(usize, f32)],
) -> Array2<f32> {
    let hidden = x.shape()[1];
    let seq_len = x.shape()[0];
    let arch = &*weights.arch;
    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = arch.gate_up_is_gelu_tanh();
    let up_view = index.up_layer_matrix(layer).expect("up mmap");
    let down_view = index.down_layer_matrix(layer).expect("down mmap");

    let mut out = Array2::<f32>::zeros((seq_len, hidden));
    for s in 0..seq_len {
        let x_row = x.row(s);
        let mut out_row = out.row_mut(s);
        for &(feat, gate_score) in hits {
            let act = if is_gated {
                let up_score = up_view.row(feat).dot(&x_row);
                let activated = if use_gelu {
                    larql_inference::ffn::gelu_tanh(gate_score)
                } else {
                    gate_score * larql_inference::ffn::sigmoid(gate_score)
                };
                activated * up_score
            } else if use_gelu {
                larql_inference::ffn::gelu_tanh(gate_score)
            } else {
                gate_score * larql_inference::ffn::sigmoid(gate_score)
            };
            if act.abs() > 1e-10 {
                out_row.scaled_add(act, &down_view.row(feat));
            }
        }
    }
    out
}

// ── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    println!("=== Walk Profile ===\n");
    println!("Model:      {}", args.model);
    println!("Vindex:     {}", args.vindex.display());
    println!("Prompt:     {:?}", args.prompt);
    println!("Iterations: {}\n", args.iterations);

    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    println!(
        "Loaded: {} layers, hidden={}",
        num_layers, weights.hidden_size
    );

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&args.vindex, &mut cb)?;
    println!("Vindex: {} vectors\n", index.total_gate_vectors());

    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Capture pre-FFN residuals
    print!("Capturing residuals... ");
    let reference = WeightFfn { weights };
    let capturing = CapturingFfn::new(&reference, num_layers);
    let _ = predict_with_ffn(weights, tokenizer, &token_ids, 1, &capturing);
    let residuals = capturing.take();
    let seq_len = residuals[0].shape()[0];
    println!("done, seq_len={}\n", seq_len);

    // Pick a representative layer for detailed analysis
    let target_layer = num_layers / 2; // layer 17 on Gemma 3 4B
    let num_features = index.num_features(target_layer);
    println!("Detailed profile on layer {target_layer} ({num_features} features)\n");

    let x = &residuals[target_layer];
    let last_row = x.row(seq_len - 1).to_owned();
    let ks: Vec<(String, usize)> = vec![
        ("K=full".to_string(), usize::MAX),
        ("K=5000".to_string(), 5000),
        ("K=2000".to_string(), 2000),
        ("K=1000".to_string(), 1000),
        ("K=500".to_string(), 500),
        ("K=200".to_string(), 200),
        ("K=100".to_string(), 100),
    ];

    // Stage A: gate retrieval at each K
    //   - gate_walk (per-feature + top-K)
    //   - gate_knn  (gemv + top-K)
    println!("--- Stage A: gate retrieval cost at layer {target_layer} ---\n");
    println!(
        "  {:>10}  {:>14}  {:>14}  {:>14}",
        "K", "gate_walk μs", "gate_knn μs", "returned"
    );
    println!("  {:-<60}", "");
    let mut walk_out: Vec<Option<Vec<(usize, f32)>>> = Vec::with_capacity(ks.len());
    let mut knn_out: Vec<Vec<(usize, f32)>> = Vec::with_capacity(ks.len());
    for (label, k) in &ks {
        let walk_stage = measure(args.iterations, || {
            let _ = index.gate_walk(target_layer, &last_row, *k);
        });
        let knn_stage = measure(args.iterations, || {
            let _ = index.gate_knn(target_layer, &last_row, *k);
        });
        // Also capture one sample for stage B
        let walk_sample = index.gate_walk(target_layer, &last_row, *k);
        let knn_sample = index.gate_knn(target_layer, &last_row, *k);
        let returned = walk_sample
            .as_ref()
            .map(|v| v.len())
            .unwrap_or_else(|| knn_sample.len());
        println!(
            "  {:>10}  {:>14.1}  {:>14.1}  {:>14}",
            label, walk_stage.median_us, knn_stage.median_us, returned
        );
        walk_out.push(walk_sample);
        knn_out.push(knn_sample);
    }
    println!();

    // Stage B: end-to-end single-layer walk_ffn_sparse.
    // Walk-loop cost is derived as (total - gate) × seq_len.
    println!(
        "--- Stage B: total forward vs gate vs derived walk-loop (layer {target_layer}) ---\n"
    );
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}  {:>12}  {:>8}  {:>10}",
        "K", "total μs", "total full x", "gate × seq", "walk = T-G", "hits", "μs/hit"
    );
    println!("  {:-<84}", "");
    use larql_inference::vindex::WalkFfnConfig;
    let x_full = residuals[target_layer].clone();
    let x_s1: Array2<f32> = {
        let row = x_full.row(seq_len - 1).to_owned();
        Array2::from_shape_vec((1, x_full.shape()[1]), row.to_vec()).unwrap()
    };
    for (i, (label, k)) in ks.iter().enumerate() {
        let config = if *k == usize::MAX {
            WalkFfnConfig::sparse(num_layers, usize::MAX)
        } else {
            WalkFfnConfig::sparse(num_layers, *k)
        };
        let ffn = WalkFfn::from_config(weights, &index, config);
        let s1_stage = measure(args.iterations, || {
            let _ = ffn.forward(target_layer, &x_s1);
        });
        let full_stage = measure(args.iterations, || {
            let _ = ffn.forward(target_layer, &x_full);
        });
        // gate-only measurement from Stage A (single residual, times seq_len)
        let gate_us = measure(args.iterations, || {
            let _ = index.gate_knn(target_layer, &last_row, *k);
        })
        .median_us
            * (seq_len as f64);
        let derived_walk = (full_stage.median_us - gate_us).max(0.0);
        let n_hits = knn_out[i].len();
        let us_per_hit = if n_hits > 0 {
            derived_walk / (n_hits as f64 * seq_len as f64)
        } else {
            0.0
        };
        println!(
            "  {:>10}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>8}  {:>10.3}",
            label,
            s1_stage.median_us,
            full_stage.median_us,
            gate_us,
            derived_walk,
            n_hits,
            us_per_hit,
        );
    }
    println!();

    // Also sanity-check: for K=100 and K=1000, print the spread of feature indices
    // (sequential vs scattered access predicts cache behaviour).
    println!("--- Stage C: hit distribution (feature-index pattern at layer {target_layer}) ---\n");
    for (i, (label, _k)) in ks.iter().enumerate() {
        let mut feats: Vec<usize> = knn_out[i].iter().map(|(f, _)| *f).collect();
        feats.sort_unstable();
        let n = feats.len();
        if n == 0 {
            continue;
        }
        // Gap statistics: average gap between consecutive feature indices
        let mut gaps = 0u64;
        for w in feats.windows(2) {
            gaps += (w[1] - w[0]) as u64;
        }
        let avg_gap = if n > 1 {
            gaps as f64 / (n - 1) as f64
        } else {
            0.0
        };
        let density = n as f64 / num_features as f64;
        println!(
            "  {:>10}  hits={:>5}  density={:>6.1}%  min={:>5}  max={:>5}  avg_gap={:>7.1}",
            label,
            n,
            density * 100.0,
            feats[0],
            feats[n - 1],
            avg_gap,
        );
    }
    let _ = walk_out;
    let _ = walk_loop; // silence unused helpers from earlier draft

    Ok(())
}
