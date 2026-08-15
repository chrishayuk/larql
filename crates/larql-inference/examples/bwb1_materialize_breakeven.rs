//! BW-B1 — closure: how many reuses does `materialize + N×compact` need to
//! beat `N×dense` and `N×gather`?
//!
//! BW-B (`bwb_compact_dense_oracle.rs`, `docs/diagnoses/
//! bwb-compact-dense-oracle.md`) measured the compact-dense kernel's
//! per-call cost GIVEN an already-materialized layer — it deliberately
//! did not measure `materialize` itself. That leaves the one question
//! that decides whether the result is operationally useful:
//!
//! ```text
//! N* = T_materialize / (T_dense - T_compact)     (break-even vs dense)
//! N* = T_materialize / (T_gather - T_compact)    (break-even vs gather)
//! ```
//!
//! A LOW N* (a handful of token reuses) means compact-dense is viable as
//! a live decode mechanism, refreshed whenever the route changes. A HIGH
//! N* (hundreds+) means it is a static/compiled-structure win only — a
//! real result, just not one that helps a per-token router.
//!
//! Deliberately bounded: a small K grid (not BW-B's full sweep), the
//! formula validated empirically at explicit reuse counts, and ONE
//! realistic-control arm asking the actual operative question — at the
//! cadence a real selector's mask actually changes, does the reuse count
//! clear N*?
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwb1_materialize_breakeven -- \
//!     --vindex /path/to/qwen3-0.6b-q4k-v2.vindex

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::vindex::{CompactDenseLayer, WalkFfn, WalkFfnConfig};
use larql_inference::FfnBackend;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;

/// K values to close the break-even question on — the three points from
/// BW-B's own headline numbers (8%, 33%, 67% of intermediate=3072).
const K_GRID: [usize; 3] = [256, 1024, 2048];
/// Reuse counts to validate the formula against directly, not just
/// algebraically.
const REUSE_STEPS: [usize; 6] = [1, 2, 4, 8, 16, 32];

const CALLS_PER_BLOCK: usize = 20;
const BLOCKS_PER_CELL: usize = 7;
const WARMUP_BLOCKS: usize = 4;
/// `materialize` is a single coarser-grained call (a multi-KB-to-MB
/// memcopy) — timed individually rather than block-batched.
const MATERIALIZE_REPEATS: usize = 7;
const MATERIALIZE_WARMUP: usize = 2;

/// Positions in the realistic-control trajectory — a smooth, small
/// incremental drift standing in for a short decode window. NOT a
/// captured real trajectory (see the module doc on
/// `bwb_compact_dense_oracle.rs` for why the CPU attention pipeline
/// isn't available for Q4K-only loads); disclosed synthetic proxy,
/// deliberately smooth rather than i.i.d. per-position noise — a real
/// residual stream drifts incrementally token to token, it doesn't
/// jump, so i.i.d. noise would bias the churn measurement toward an
/// unrealistically pessimistic worst case.
const TRAJECTORY_LEN: usize = 32;
/// K for the realistic-control arm — the 33% point, BW-B's cleanest
/// result (gather loses by 7%, compact wins by 52%, same bytes).
const TRAJECTORY_K: usize = 1024;

fn synthetic_x(hidden: usize, seed: usize) -> Vec<f32> {
    let phase = (seed as f32 + 1.0) * 0.37;
    (0..hidden)
        .map(|i| (i as f32 * 0.0137 + phase).sin() * 0.6 + (i as f32 * 0.071).cos() * 0.3)
        .collect()
}

/// A smoothly-drifting trajectory: `base` at position 0, incrementally
/// interpolated toward `base + drift_fraction * drift` by the last
/// position. `drift` is a DIFFERENT synthetic direction from `base`
/// (seeded differently), so the drift is not a pure rescaling.
fn trajectory(hidden: usize, layer: usize, n_positions: usize, drift_fraction: f32) -> Array2<f32> {
    let base = synthetic_x(hidden, layer);
    let drift = synthetic_x(hidden, layer + 10_000);
    let mut data = Vec::with_capacity(n_positions * hidden);
    for pos in 0..n_positions {
        let t = pos as f32 / (n_positions.max(2) - 1) as f32;
        for i in 0..hidden {
            data.push(base[i] + t * drift_fraction * drift[i]);
        }
    }
    Array2::from_shape_vec((n_positions, hidden), data).unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Block-median wall time in ms for one arm, `CALLS_PER_BLOCK` calls per
/// block. Mirrors `bwb_compact_dense_oracle.rs`'s methodology exactly —
/// same constants, same interleaving discipline — so the two harnesses'
/// numbers are comparable.
fn time_arm(warmup_blocks: usize, blocks: usize, mut call: impl FnMut()) -> f64 {
    for _ in 0..warmup_blocks {
        for _ in 0..CALLS_PER_BLOCK {
            call();
        }
    }
    let block_ms: Vec<f64> = (0..blocks)
        .map(|_| {
            let t0 = Instant::now();
            for _ in 0..CALLS_PER_BLOCK {
                call();
            }
            t0.elapsed().as_secs_f64() * 1000.0 / CALLS_PER_BLOCK as f64
        })
        .collect();
    median(block_ms)
}

/// Direct (non-blocked) median timing for a coarser one-shot call.
fn time_once(warmup: usize, repeats: usize, mut call: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        call();
    }
    let ms: Vec<f64> = (0..repeats)
        .map(|_| {
            let t0 = Instant::now();
            call();
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    median(ms)
}

struct KResult {
    k: usize,
    t_dense: f64,
    t_gather: f64,
    t_compact: f64,
    t_materialize: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = PathBuf::from(
        std::env::var("HOME").unwrap_or_default() + "/larql-vindex/qwen3-0.6b-q4k-v2.vindex",
    );
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" {
            i += 1;
            vindex_path = PathBuf::from(&args[i]);
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!("vindex directory not found: {}", vindex_path.display());
        eprintln!("Usage: bwb1_materialize_breakeven --vindex PATH");
        std::process::exit(1);
    }

    println!("=== BW-B1: materialize/break-even closure ===\n");
    let mut cb = SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    index.load_down_features_q4k(&vindex_path)?;
    if !index.has_down_features_kquant() {
        return Err("vindex has no down_features_q4k.bin sidecar".into());
    }
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let use_gelu = weights.arch.activation().uses_gelu_tanh_gate_up();
    let k_max = *K_GRID.iter().max().unwrap();
    println!("{num_layers} layers, hidden={hidden}, K grid={K_GRID:?}\n");

    let dense_cfg = WalkFfnConfig::dense(num_layers);
    let walk_dense = WalkFfn::from_config(&weights, &index, dense_cfg);

    // per-K accumulators, mean over layers
    let mut per_k: Vec<Vec<KResult>> = K_GRID.iter().map(|_| Vec::new()).collect();

    for layer in 0..num_layers {
        let x = synthetic_x(hidden, layer);
        let x_arr = Array2::from_shape_vec((1, hidden), x.clone())?;

        let capture_cfg = WalkFfnConfig::sparse(num_layers, k_max);
        let walk_capture = WalkFfn::from_config(&weights, &index, capture_cfg).with_trace();
        let _ = walk_capture.forward(layer, &x_arr);
        let trace = walk_capture.take_runtime_trace();
        let mut ranked: Vec<(usize, usize)> = trace
            .iter()
            .find(|r| r.layer == layer)
            .map(|r| r.features.iter().map(|f| (f.rank, f.feature)).collect())
            .unwrap_or_default();
        ranked.sort_by_key(|&(rank, _)| rank);
        let ranked: Vec<usize> = ranked.into_iter().map(|(_, feat)| feat).collect();
        if ranked.len() < k_max {
            continue;
        }

        // Layer-level warmup: settle this layer's Q4K bytes into cache
        // and let CPU clocks ramp before ANY timed cell — without this,
        // whichever K happens to run first in the loop below pays a
        // one-time cold-start tax that reads as a K-dependent cost even
        // though dense's byte count and kernel are IDENTICAL across K.
        // Confirmed present without this: T_dense measured 1720/1225/
        // 617us at K=256/1024/2048 in one early run, when it must be
        // constant (BW-B measured 582-586us flat across 5 K points with
        // more per-layer warmup runway). Caught by the same discipline
        // `feedback_bench_steady_state_protocol` and
        // `feedback_thermal_perf_artifacts` already require.
        for _ in 0..8 {
            let _ = walk_dense.forward(layer, &x_arr);
        }

        for (k_idx, &k) in K_GRID.iter().enumerate() {
            let mask = &ranked[..k];
            let gather_cfg = WalkFfnConfig::sparse(num_layers, k)
                .with_pool_per_layer(std::sync::Arc::new(vec![mask.to_vec(); num_layers]))
                .with_precomputed_routing(true);
            let walk_gather =
                WalkFfn::from_config(&weights, &index, gather_cfg).with_dispatch_trace();

            // Coverage guard, same as BW-B.
            let _ = walk_gather.forward(layer, &x_arr);
            let dispatched = walk_gather.take_dispatch_trace();
            if !dispatched.iter().any(|e| e.path == "sparse:gather_q4k") {
                continue;
            }

            let t_dense = time_arm(WARMUP_BLOCKS, BLOCKS_PER_CELL, || {
                let _ = walk_dense.forward(layer, &x_arr);
            });
            let t_gather = time_arm(WARMUP_BLOCKS, BLOCKS_PER_CELL, || {
                let _ = walk_gather.forward(layer, &x_arr);
            });

            let compact = CompactDenseLayer::materialize(&index, layer, mask, hidden)
                .expect("materialize succeeds — sidecar loaded, mask non-empty, in range");
            let t_compact = time_arm(WARMUP_BLOCKS, BLOCKS_PER_CELL, || {
                let _ = walk_gather.compact_dense_forward(&compact, &x, use_gelu, hidden);
            });
            let t_materialize = time_once(MATERIALIZE_WARMUP, MATERIALIZE_REPEATS, || {
                let _ = CompactDenseLayer::materialize(&index, layer, mask, hidden);
            });

            per_k[k_idx].push(KResult {
                k,
                t_dense,
                t_gather,
                t_compact,
                t_materialize,
            });
        }
    }

    // ── Mean over layers, per K. ──
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>12} {:>14} {:>14}",
        "K", "T_dense", "T_gather", "T_compact", "T_materialize", "N*_vs_dense", "N*_vs_gather"
    );
    println!("{}", "-".repeat(80));
    let mut summary: Vec<(usize, f64, f64, f64, f64)> = Vec::new(); // k, t_dense, t_gather, t_compact, t_materialize
    for results in &per_k {
        if results.is_empty() {
            continue;
        }
        let n = results.len() as f64;
        let k = results[0].k;
        let t_dense = results.iter().map(|r| r.t_dense).sum::<f64>() / n;
        let t_gather = results.iter().map(|r| r.t_gather).sum::<f64>() / n;
        let t_compact = results.iter().map(|r| r.t_compact).sum::<f64>() / n;
        let t_materialize = results.iter().map(|r| r.t_materialize).sum::<f64>() / n;

        let n_star_dense = if t_dense > t_compact {
            t_materialize / (t_dense - t_compact)
        } else {
            f64::INFINITY // compact never wins per-call, so no reuse count amortises it
        };
        let n_star_gather = if t_gather > t_compact {
            t_materialize / (t_gather - t_compact)
        } else {
            f64::INFINITY
        };

        println!(
            "{k:<6} {:>9.1}us {:>9.1}us {:>9.1}us {:>11.1}us {:>14.2} {:>14.2}",
            t_dense * 1000.0,
            t_gather * 1000.0,
            t_compact * 1000.0,
            t_materialize * 1000.0,
            n_star_dense,
            n_star_gather,
        );
        summary.push((k, t_dense, t_gather, t_compact, t_materialize));
    }

    // ── Empirical validation at explicit reuse steps. ──
    println!("\ncumulative cost at N reuses (materialize + N x compact, vs N x dense/gather):\n");
    println!(
        "{:<6} {:>4} {:>14} {:>14} {:>14} {:>10} {:>10}",
        "K", "N", "cum_compact", "cum_dense", "cum_gather", "<dense?", "<gather?"
    );
    println!("{}", "-".repeat(76));
    for &(k, t_dense, t_gather, t_compact, t_materialize) in &summary {
        for &nreuse in &REUSE_STEPS {
            let cum_compact = t_materialize + nreuse as f64 * t_compact;
            let cum_dense = nreuse as f64 * t_dense;
            let cum_gather = nreuse as f64 * t_gather;
            println!(
                "{k:<6} {nreuse:<4} {:>13.1}us {:>13.1}us {:>13.1}us {:>10} {:>10}",
                cum_compact * 1000.0,
                cum_dense * 1000.0,
                cum_gather * 1000.0,
                if cum_compact < cum_dense { "yes" } else { "no" },
                if cum_compact < cum_gather {
                    "yes"
                } else {
                    "no"
                },
            );
        }
        println!();
    }

    // ── Realistic control: mask churn at the cadence a real selector
    // actually produces, for TRAJECTORY_K on a representative layer. ──
    println!(
        "=== Realistic-control arm: mask churn over a {TRAJECTORY_LEN}-position \
              smooth-drift trajectory, K={TRAJECTORY_K} ===\n"
    );
    let mut all_overlaps = Vec::new();
    let mut all_run_lengths_strict = Vec::new();
    let mut all_run_lengths_tolerant = Vec::new();
    let sample_layers: Vec<usize> = (0..num_layers).step_by(4).collect(); // every 4th layer — bounded, still depth-spanning
    for &layer in &sample_layers {
        let traj = trajectory(hidden, layer, TRAJECTORY_LEN, 0.15);
        let cfg = WalkFfnConfig::sparse(num_layers, TRAJECTORY_K);
        let walk = WalkFfn::from_config(&weights, &index, cfg).with_trace();
        let _ = walk.forward(layer, &traj);
        let trace = walk.take_runtime_trace();
        let mut per_pos: Vec<(usize, std::collections::HashSet<usize>)> = trace
            .iter()
            .filter(|r| r.layer == layer)
            .map(|r| (r.position, r.features.iter().map(|f| f.feature).collect()))
            .collect();
        per_pos.sort_by_key(|&(pos, _)| pos);
        if per_pos.len() < 2 {
            continue;
        }

        let mut run_strict = 1usize;
        let mut run_tolerant = 1usize;
        for w in per_pos.windows(2) {
            let (_, a) = &w[0];
            let (_, b) = &w[1];
            let inter = a.intersection(b).count();
            let union = a.union(b).count().max(1);
            let jaccard = inter as f64 / union as f64;
            all_overlaps.push(jaccard);
            if jaccard >= 1.0 {
                run_strict += 1;
            } else {
                all_run_lengths_strict.push(run_strict);
                run_strict = 1;
            }
            if jaccard >= 0.95 {
                run_tolerant += 1;
            } else {
                all_run_lengths_tolerant.push(run_tolerant);
                run_tolerant = 1;
            }
        }
        all_run_lengths_strict.push(run_strict);
        all_run_lengths_tolerant.push(run_tolerant);
    }

    let mean_jaccard = all_overlaps.iter().sum::<f64>() / all_overlaps.len().max(1) as f64;
    let min_jaccard = all_overlaps.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_run_strict = all_run_lengths_strict.iter().sum::<usize>() as f64
        / all_run_lengths_strict.len().max(1) as f64;
    let mean_run_tolerant = all_run_lengths_tolerant.iter().sum::<usize>() as f64
        / all_run_lengths_tolerant.len().max(1) as f64;

    println!(
        "layers sampled: {} (every 4th of {num_layers})",
        sample_layers.len()
    );
    println!(
        "consecutive-position mask Jaccard overlap: mean={mean_jaccard:.3} min={min_jaccard:.3}"
    );
    println!("mean run length before ANY feature swap (strict, Jaccard<1.0 ends a run): {mean_run_strict:.2} positions");
    println!("mean run length before >5% mask drift (tolerant, Jaccard<0.95 ends a run): {mean_run_tolerant:.2} positions");

    if let Some(&(_, t_dense, t_gather, t_compact, t_materialize)) =
        summary.iter().find(|&&(k, ..)| k == TRAJECTORY_K)
    {
        let n_star_dense = t_materialize / (t_dense - t_compact).max(1e-9);
        let n_star_gather = t_materialize / (t_gather - t_compact).max(1e-9);
        println!(
            "\nAt K={TRAJECTORY_K}: N* vs dense = {n_star_dense:.2}, N* vs gather = {n_star_gather:.2}. \
             Observed mean run length (strict) = {mean_run_strict:.2}, (tolerant) = {mean_run_tolerant:.2}."
        );
        let verdict = if mean_run_strict >= n_star_dense.max(n_star_gather) {
            "DYNAMIC-VIABLE: the observed mask stability already clears break-even even under \
             the strict (zero-drift) reuse rule."
        } else if mean_run_tolerant >= n_star_dense.max(n_star_gather) {
            "MEDIUM-HORIZON: strict reuse doesn't clear break-even, but a small drift tolerance \
             (Jaccard>=0.95) does — a decode-time cache with an approximate-match refresh policy \
             could plausibly amortise this."
        } else {
            "STATIC/COMPILED-STRUCTURE ONLY: neither strict nor tolerant observed run length \
             clears break-even on this synthetic trajectory — real, but not yet evidence for a \
             live per-token router."
        };
        println!("\nVERDICT: {verdict}");
    }

    Ok(())
}
