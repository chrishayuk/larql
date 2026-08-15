//! BW-B — compact-dense vs. sparse-gather vs. dense, at a fixed oracle mask.
//!
//! The question BW-B answers, per `docs/diagnoses/bw10-live-gate.md`:
//!
//! > If an oracle gives us a useful subset of a matrix and we compile
//! > that subset into an actually contiguous dense representation, does
//! > the hardware reward us — or was R4's sparse-gather loss really
//! > about paying the gather cost on every call?
//!
//! Three arms, same route, same input, same underlying Q4K bytes:
//!
//!   1. **dense**    — `WalkFfnConfig::dense`, routes to
//!      `interleaved_kquant:native` (`kquant_matmul_transb` over the
//!      FULL layer — the production ceiling).
//!   2. **gather**    — `WalkFfnConfig::sparse(..).with_pool_per_layer
//!      (mask).with_precomputed_routing(true)`, routes to
//!      `sparse:gather_q4k` — gathers the mask's Q4K rows fresh on
//!      EVERY call (this is what R4 measured; BW-B does not re-derive
//!      R4's number on a different model, it reproduces R4's SHAPE as
//!      a control that the instrument and methodology are sound).
//!   3. **compact**   — `CompactDenseLayer::materialize` gathers the
//!      SAME mask's rows ONCE, outside the timed loop;
//!      `compact_dense_forward` then runs the identical fused kernel
//!      `gather_q4k_accumulate` uses, with zero gather cost per call.
//!
//! The oracle mask is the REAL top-K feature ranking
//! `WalkFfnConfig::sparse(.., K_MAX)` selects (default
//! `FeatureSelector::GateOnly`, real Q4K gate rows, real row-dot
//! kernel) — captured via `WalkFfn::with_trace` + `take_runtime_trace`
//! on a direct `WalkFfn::forward` call per layer.
//!
//! One disclosed simplification: the input `x` is a fixed, deterministic,
//! per-layer synthetic vector, not a residual captured from a live
//! `generate()` pass. `larql-inference`'s CPU reference attention
//! pipeline (`predict_with_ffn_trace`) expects full f32 attention
//! weights this Q4K-only-loaded vindex doesn't carry (`run_layer_with_ffn`
//! returns `None` for every layer — confirmed empty dispatch/runtime
//! traces before this file settled on calling `WalkFfn::forward`
//! directly). This does not affect BYTES or WALL TIME — Q4K row-dot /
//! scaled-add kernels are data-value-independent — only WHICH features
//! the real gate weights rank highest for this particular direction,
//! which is what the oracle mask needs to be non-trivial, not what
//! generation would have selected. Fixing the CPU attention pipeline
//! for Q4K-only loads is out of BW-B's scope.
//!
//! Bytes and the roofline join reuse `larql_compute::movement_ledger`
//! (BW10) directly — the same `ByteMovement` / `TimeAttribution` /
//! `MovementCost` this project's live gate (BW-A) used, declared
//! against the CPU-cluster attainable DRAM roofline (127 GB/s,
//! `docs/diagnoses/memory-bandwidth-roofline.md`) rather than the GPU
//! one BW-A used — this is CPU-side kernel work, not a Metal dispatch.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwb_compact_dense_oracle -- \
//!     --vindex /path/to/qwen3-0.6b-q4k-v2.vindex [--layers 0,5,10]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use larql_compute::movement_ledger::{
    ByteMovement, MovementCost, Rooflines, TierBandwidth, TimeAttribution,
};
use larql_inference::vindex::{CompactDenseLayer, WalkFfn, WalkFfnConfig};
use larql_inference::FfnBackend;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;

/// CPU-cluster attainable DRAM read bandwidth on the M3 Max development
/// machine — `docs/diagnoses/memory-bandwidth-roofline.md`
/// (`membw_probe`, 8-accumulator NEON read, 256 MiB/thread). A PROBE
/// RESULT, not a spec figure; re-measure on any other host before
/// quoting it. Distinct from `M3_MAX_ATTAINABLE_DRAM_GBPS` (367, the
/// GPU figure BW-A used) — this is CPU-side kernel work.
const M3_MAX_CPU_ATTAINABLE_DRAM_GBPS: f64 = 127.0;

/// Feature counts to sweep, as absolute K. The largest value doubles as
/// the trace-capture width (must stay below `hits_len_ge_intermediate`'s
/// 80% full-K-gemv threshold so the capture pass exercises the real
/// per-feature walk, not the dense-shaped fast path).
const K_SWEEP: [usize; 5] = [256, 512, 1024, 1536, 2048];

/// Timed calls per (layer, K, arm) block. Block-averaged rather than
/// timed individually — CPU matvecs at this size run in tens of
/// microseconds, well inside `Instant`'s per-call noise floor.
const CALLS_PER_BLOCK: usize = 20;
/// Blocks per cell — the MEDIAN of these is reported, not the mean, so
/// one scheduler-noise outlier block doesn't move the number.
const BLOCKS_PER_CELL: usize = 7;
/// Throwaway blocks before the timed ones, to settle the mmap'd Q4K
/// bytes into the page cache.
const WARMUP_BLOCKS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Dense,
    Gather,
    Compact,
}

impl Arm {
    const ALL: [Arm; 3] = [Arm::Dense, Arm::Gather, Arm::Compact];
    fn label(self) -> &'static str {
        match self {
            Arm::Dense => "dense",
            Arm::Gather => "gather",
            Arm::Compact => "compact",
        }
    }
    fn idx(self) -> usize {
        match self {
            Arm::Dense => 0,
            Arm::Gather => 1,
            Arm::Compact => 2,
        }
    }
}

/// One (layer, K, arm) cell's result: physical bytes touched (constant
/// across its blocks — the route doesn't change within a cell) and the
/// block-median wall time in ms.
struct Cell {
    physical_bytes: u64,
    wall_ms: f64,
}

/// A fixed, deterministic, per-layer input vector. Not a captured
/// residual — see the module doc for why. Non-degenerate (varies by
/// element and by layer) so the real gate weights produce a real,
/// non-uniform top-K ranking rather than a symmetric or all-equal one
/// (`feedback_fixture_symmetry_hides_representation_bugs`).
fn synthetic_x(hidden: usize, layer: usize) -> Vec<f32> {
    let phase = (layer as f32 + 1.0) * 0.37;
    (0..hidden)
        .map(|i| (i as f32 * 0.0137 + phase).sin() * 0.6 + (i as f32 * 0.071).cos() * 0.3)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = PathBuf::from(
        std::env::var("HOME").unwrap_or_default() + "/larql-vindex/qwen3-0.6b-q4k-v2.vindex",
    );
    let mut layers_arg: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = PathBuf::from(&args[i]);
            }
            "--layers" => {
                i += 1;
                layers_arg = Some(args[i].clone());
            }
            _ => {}
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!("vindex directory not found: {}", vindex_path.display());
        eprintln!("Usage: bwb_compact_dense_oracle --vindex PATH [--layers 0,5,10]");
        std::process::exit(1);
    }

    println!("=== BW-B: compiled compact-dense oracle vs. sparse-gather vs. dense ===\n");
    println!("vindex: {}", vindex_path.display());

    let mut cb = SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    index.load_down_features_q4k(&vindex_path)?;
    if !index.has_down_features_kquant() {
        return Err("vindex has no down_features_q4k.bin sidecar — run \
                     `larql convert add-feature-major-down --input <vindex>` first"
            .into());
    }

    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let use_gelu = weights.arch.activation().uses_gelu_tanh_gate_up();
    let k_max = *K_SWEEP.iter().max().unwrap();

    let layers: Vec<usize> = match layers_arg {
        Some(spec) => spec.split(',').filter_map(|s| s.parse().ok()).collect(),
        None => (0..num_layers).collect(),
    };
    println!(
        "{num_layers} layers, hidden={hidden}, K sweep={K_SWEEP:?}, testing {} layer(s)\n",
        layers.len()
    );

    // ── Capture the real oracle: one direct `forward` call per layer at
    // K_MAX, GateOnly selection, `with_trace` records exactly what the
    // production selector executed. ──
    let rooflines = Rooflines::dram_only(TierBandwidth::measured(
        M3_MAX_CPU_ATTAINABLE_DRAM_GBPS,
        "M3 Max CPU cluster attainable DRAM read probe (membw_probe, 2026-07-28)",
    ));
    let dense_cfg = WalkFfnConfig::dense(num_layers);
    let walk_dense = WalkFfn::from_config(&weights, &index, dense_cfg);

    let mut results: Vec<[Vec<Cell>; 3]> = (0..K_SWEEP.len())
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    let mut skipped_layers = Vec::new();

    for &layer in &layers {
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
            skipped_layers.push(layer);
            continue;
        }

        let physical_dense = {
            let slices = index
                .interleaved_kquant_layer_data(layer)
                .expect("dense arm requires interleaved Q4K bytes");
            (slices[0].0.len() + slices[1].0.len() + slices[2].0.len()) as u64
        };

        for (k_idx, &k) in K_SWEEP.iter().enumerate() {
            let mask = &ranked[..k];
            let gather_cfg = WalkFfnConfig::sparse(num_layers, k)
                .with_pool_per_layer(Arc::new(vec![mask.to_vec(); num_layers]))
                .with_precomputed_routing(true);
            let walk_gather =
                WalkFfn::from_config(&weights, &index, gather_cfg).with_dispatch_trace();
            let compact = CompactDenseLayer::materialize(&index, layer, mask, hidden)
                .expect("materialize succeeds — sidecar loaded, mask non-empty, in range");
            let physical_route = compact.physical_bytes();

            for arm in Arm::ALL {
                // Coverage guard: confirm `gather` actually dispatches
                // to `sparse:gather_q4k` and not a silent fallback — a
                // mis-routed comparison would read as a kernel result
                // but measure something else entirely.
                if arm == Arm::Gather {
                    let _ = walk_gather.forward(layer, &x_arr);
                    let dispatched = walk_gather.take_dispatch_trace();
                    let on_gather = dispatched.iter().any(|e| e.path == "sparse:gather_q4k");
                    if !on_gather {
                        println!(
                            "  layer {layer} K={k}: gather did NOT dispatch to sparse:gather_q4k \
                             (got {:?}) — skipped",
                            dispatched.iter().map(|e| e.path).collect::<Vec<_>>()
                        );
                        continue;
                    }
                }

                for _ in 0..WARMUP_BLOCKS {
                    run_block(
                        arm,
                        &walk_dense,
                        &walk_gather,
                        &compact,
                        layer,
                        &x_arr,
                        &x,
                        use_gelu,
                        hidden,
                    );
                }
                let mut block_ms: Vec<f64> = (0..BLOCKS_PER_CELL)
                    .map(|_| {
                        run_block(
                            arm,
                            &walk_dense,
                            &walk_gather,
                            &compact,
                            layer,
                            &x_arr,
                            &x,
                            use_gelu,
                            hidden,
                        )
                    })
                    .collect();
                block_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_ms = block_ms[block_ms.len() / 2];
                let physical_bytes = if arm == Arm::Dense {
                    physical_dense
                } else {
                    physical_route
                };
                results[k_idx][arm.idx()].push(Cell {
                    physical_bytes,
                    wall_ms: median_ms,
                });
            }
        }
    }

    if !skipped_layers.is_empty() {
        println!(
            "NOTE: {} layer(s) captured fewer than K_MAX={k_max} ranked features and were \
             skipped: {skipped_layers:?}\n",
            skipped_layers.len()
        );
    }

    // ── Report: logical fraction retained -> physical bytes -> kernel
    // eta -> wall time, per arm per K, mean across the tested layers. ──
    println!(
        "{:<8} {:>6} {:>8} {:>14} {:>12} {:>10} {:>8}",
        "arm", "K", "frac", "phys_bytes", "wall_us", "GB/s", "eta"
    );
    println!("{}", "-".repeat(72));
    for (k_idx, &k) in K_SWEEP.iter().enumerate() {
        for arm in Arm::ALL {
            let cells = &results[k_idx][arm.idx()];
            if cells.is_empty() {
                continue;
            }
            let n = cells.len() as f64;
            let mean_bytes =
                cells.iter().map(|c| c.physical_bytes).sum::<u64>() / cells.len() as u64;
            let mean_wall_ms = cells.iter().map(|c| c.wall_ms).sum::<f64>() / n;

            let bytes = ByteMovement {
                semantic_requested: mean_bytes,
                physical_touched: mean_bytes,
                useful_physical: mean_bytes,
                dram: mean_bytes,
                ..Default::default()
            };
            // CPU microbench, one fused kernel call: no separate
            // scheduling/host-wait split to report, so `gpu_busy_ms`
            // here means "kernel busy time" — the whole window, a
            // disclosed reuse of BW10's TimeAttribution shape (module
            // doc above).
            let time = TimeAttribution {
                wall_ms: mean_wall_ms,
                gpu_busy_ms: mean_wall_ms,
                ..Default::default()
            };
            let cost = MovementCost::derive(&bytes, &time, &rooflines);
            let frac = k as f64 / index.num_features(0).max(1) as f64;
            println!(
                "{:<8} {:>6} {:>7.1}% {:>14} {:>11.1}us {:>10.1} {:>8}",
                arm.label(),
                k,
                frac * 100.0,
                mean_bytes,
                mean_wall_ms * 1000.0,
                cost.implied_stream_gbps.unwrap_or(0.0),
                cost.roofline_utilisation
                    .map(|e| format!("{e:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
            );
        }
        println!();
    }

    Ok(())
}

/// Time one block of `CALLS_PER_BLOCK` calls for `arm`, return the
/// per-call mean in ms.
#[allow(clippy::too_many_arguments)]
fn run_block(
    arm: Arm,
    walk_dense: &WalkFfn,
    walk_gather: &WalkFfn,
    compact: &CompactDenseLayer,
    layer: usize,
    x_arr: &Array2<f32>,
    x_slice: &[f32],
    use_gelu: bool,
    hidden: usize,
) -> f64 {
    let t0 = Instant::now();
    for _ in 0..CALLS_PER_BLOCK {
        match arm {
            Arm::Dense => {
                let _ = walk_dense.forward(layer, x_arr);
            }
            Arm::Gather => {
                let _ = walk_gather.forward(layer, x_arr);
            }
            Arm::Compact => {
                let _ = walk_gather.compact_dense_forward(compact, x_slice, use_gelu, hidden);
            }
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / CALLS_PER_BLOCK as f64
}
