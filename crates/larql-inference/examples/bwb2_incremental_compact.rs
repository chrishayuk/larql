//! BW-B2 — incremental patch vs. full rebuild vs. always-gather, under
//! REAL empirically-observed mask-churn statistics.
//!
//! ## Chain context
//!
//! BW-B (`vindex/walk_ffn/compact_dense.rs`, ported into this worktree —
//! see the file-header note there) showed that a compiled compact-dense
//! layer wins once the gather cost is paid OUTSIDE the timed call. BW-B1
//! (`bwb1_materialize_breakeven.rs`, this worktree's port) then asked
//! how many reuses of an already-materialized layer are needed to amortise
//! `materialize`'s cost — and used a SYNTHETIC smooth-drift proxy for how
//! often a real selector's mask actually changes, because no captured
//! real trajectory was available at the time.
//!
//! BW-B1's break-even math assumed a FULL rebuild every time the mask
//! changes at all. This experiment (BW-B2) asks the next question: when
//! a mask changes only PARTIALLY, is a full rebuild wasteful compared to
//! patching only the rows that actually changed?
//!
//! ## Model-mismatch note — read before trusting any number below
//!
//! This session separately captured a REAL decode-time MoE-expert-
//! routing trace on gpt-oss-20b (`glm3_route_trace_final.jsonl`,
//! 1978 decode positions x 24 layers, top-4 experts per layer per
//! position). That trace is **whole-expert-block routing**, not
//! **dense-FFN top-K feature selection** — a completely different
//! object from what `CompactDenseLayer`/`GatheredRows` operate on
//! (individual Q4K rows within ONE qwen3 FFN, not whole MoE experts on
//! a different model entirely). Replaying gpt-oss-20b's literal expert
//! IDs through the qwen3 FFN-feature kernel would be a category error,
//! so this file does NOT do that.
//!
//! What DOES transfer legitimately, and is the ONLY thing borrowed from
//! that trace, is the empirically observed CHURN *DISTRIBUTION*:
//!   - `P(mask changes at all)` between consecutive decode positions —
//!     `p_change` below.
//!   - Given a change happens, `P(k of the top-4 experts differ)` for
//!     k in {1,2,3,4} — reinterpreted here as `P(a k/4 FRACTION of the
//!     selected K-feature set churns)`, `p_frac` below. Fraction, not
//!     absolute count, is what carries across a 4-slot MoE-expert set
//!     and a K in {256,1024,2048}-slot dense-FFN-feature set — see the
//!     `ChurnCell` doc for the explicit conversion and its consequence
//!     (the smallest observable churn granularity, "1 of 4", is 25% of
//!     ANY K — a much bigger swap than "1-2 rows" for K >= 256).
//!
//! `p_change`/`p_frac` per depth band were recomputed independently in
//! this worktree straight from the raw JSONL (not copied from the
//! other session's printed summary), matching that session's own log
//! (`glm3_final_run.log`) to within trace-length rounding (1977 vs 1978
//! transitions — an off-by-one in whether the first position counts).
//! Depth bands are gpt-oss-20b's own `syntax(early)=[0,8]
//! knowledge(mid)=[9,18] output(late)=[19,23]` bands (that session's
//! log) mapped IN SPIRIT — not by layer index, gpt-oss-20b and
//! qwen3-0.6b have different depths — onto qwen3-0.6b-q4k-v2.vindex's
//! own `index.json` `layer_bands` (`syntax=[0,10] knowledge=[11,21]
//! output=[22,27]`, 28 layers). One representative layer per band (the
//! band's midpoint) is used, matching BW-B1's one-layer-per-cell style
//! rather than BW-B's full-depth sweep, to keep this experiment's
//! runtime bounded.
//!
//! ## What this file measures, per (K, depth band)
//!
//! 1. **Full rebuild on any change** — `CompactDenseLayer::materialize`
//!    fresh whenever the simulated mask changes at all, reused
//!    otherwise (BW-B1's existing policy, now driven by real churn odds
//!    instead of a synthetic smooth-drift trajectory).
//! 2. **Incremental patch on change** — `CompactDenseLayer::patch`
//!    (new in this worktree) overwrites only the changed slots' rows,
//!    reused otherwise.
//! 3. **Always sparse-gather** — a fresh `CompactDenseLayer::materialize`
//!    EVERY step regardless of whether the mask changed. This is
//!    functionally identical to the pre-BW-B `gather_q4k_accumulate`
//!    baseline (same `gather_kquant_rows` + `score_and_accumulate`
//!    under the hood — BW-B's own parity test pins the two paths as
//!    bit-exact), reached here through the public `materialize` +
//!    `compact_dense_forward` pair instead of the crate-private
//!    `gather_q4k_accumulate` function, so this file needs no access
//!    to crate internals.
//!
//! Two complementary measurements are taken for each cell:
//!   - **Component costs**: `t_forward` (score_and_accumulate alone),
//!     `t_materialize_full` (a fresh full-K gather), and `t_patch(frac)`
//!     at each churn fraction {25%,50%,75%,100% of K} — BW-B1-style
//!     block/warmup timing, isolated from any RNG or schedule effects.
//!   - **End-to-end trajectory**: a few hundred simulated decode steps
//!     per (K, band), each step's mask evolution drawn from that band's
//!     real `p_change`/`p_frac`, replayed identically across all three
//!     arms (same seed, same schedule) and timed as whole-trajectory
//!     wall-clock (median over repeats, warmup trajectories discarded —
//!     the same cold-start discipline `bwb1_materialize_breakeven.rs`
//!     had to add after finding a K-dependent first-call tax).
//!
//! The end-to-end number is cross-checked against the component-cost
//! prediction (`t_forward + p_change-weighted overhead`) as an internal
//! consistency control, not asserted blindly.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwb2_incremental_compact -- \
//!     --vindex /path/to/qwen3-0.6b-q4k-v2.vindex

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use larql_inference::vindex::{CompactDenseLayer, WalkFfn, WalkFfnConfig};
use larql_inference::FfnBackend;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// K values to close the question on — identical to BW-B1's grid so the
/// two files' numbers sit side by side.
const K_GRID: [usize; 3] = [256, 1024, 2048];

/// Fixed RNG seed for the trajectory schedule generator — reproducible
/// across runs; the seed value itself carries no meaning.
const SCHEDULE_SEED: u64 = 0xB2_0000;

/// One depth band: its label, its representative layer on
/// qwen3-0.6b-q4k-v2.vindex (the band's midpoint from the vindex's own
/// `layer_bands`), and the real GLM-3-trace-derived churn parameters
/// for that band (see the module doc for the recomputation and the
/// fraction-not-count conversion).
struct ChurnCell {
    band: &'static str,
    layer: usize,
    /// P(mask changes at all between consecutive decode positions).
    p_change: f64,
    /// P(a {25%,50%,75%,100%} fraction of the K-set churns | a change
    /// happened) — GLM-3's real P(1/2/3/4 of top-4 experts differ |
    /// change), reindexed as a FRACTION of the selected set so it
    /// transfers to any K (see module doc).
    p_frac: [f64; 4],
}

/// qwen3-0.6b-q4k-v2.vindex has 28 layers, `layer_bands` syntax=[0,10]
/// knowledge=[11,21] output=[22,27]` (its own `index.json`) — midpoints
/// 5 / 16 / 24 used as the one representative layer per band.
///
/// `p_change` / `p_frac` recomputed independently in this worktree from
/// `glm3_route_trace_final.jsonl` (see module doc); matches that
/// session's own printed summary (`glm3_final_run.log`) to within
/// trace-length rounding.
const CHURN_CELLS: [ChurnCell; 3] = [
    ChurnCell {
        band: "early",
        layer: 5,
        p_change: 0.9747,
        p_frac: [0.1604, 0.2996, 0.3148, 0.2252],
    },
    ChurnCell {
        band: "mid",
        layer: 16,
        p_change: 0.9191,
        p_frac: [0.3270, 0.3494, 0.2366, 0.0871],
    },
    ChurnCell {
        band: "late",
        layer: 24,
        p_change: 0.9525,
        p_frac: [0.2273, 0.3257, 0.3016, 0.1454],
    },
];

/// The four churn-fraction buckets `p_frac` indexes into — "1 of 4",
/// "2 of 4", "3 of 4", "4 of 4" experts differing, reinterpreted as a
/// fraction of the K-feature set.
const CHURN_FRACTIONS: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

/// A few hundred simulated decode steps per (K, band) — matches GLM-3's
/// own trace scale in ORDER OF MAGNITUDE (that trace was ~2000
/// positions; this synthetic trajectory samples the same per-step churn
/// ODDS at a smaller, still-representative length to keep 3 arms x 3
/// bands x 3 K values' wall time bounded).
const TRAJECTORY_STEPS: usize = 300;
/// Whole-trajectory repeats: discarded (cold-start) then timed
/// (median taken) — same warmup-then-measure shape as
/// `bwb1_materialize_breakeven.rs`'s `time_arm`, just applied to a
/// whole trajectory instead of a single call.
const WARMUP_TRAJECTORIES: usize = 2;
const TIMED_TRAJECTORIES: usize = 5;

/// Component-cost block timing — identical constants to
/// `bwb1_materialize_breakeven.rs` so the two files' numbers are
/// directly comparable.
const CALLS_PER_BLOCK: usize = 20;
const BLOCKS_PER_CELL: usize = 7;
const WARMUP_BLOCKS: usize = 4;
const MATERIALIZE_REPEATS: usize = 7;
const MATERIALIZE_WARMUP: usize = 2;

fn synthetic_x(hidden: usize, seed: usize) -> Vec<f32> {
    let phase = (seed as f32 + 1.0) * 0.37;
    (0..hidden)
        .map(|i| (i as f32 * 0.0137 + phase).sin() * 0.6 + (i as f32 * 0.071).cos() * 0.3)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Ported from `bwb1_materialize_breakeven.rs` verbatim (see that
/// file's own doc comment for the methodology rationale).
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

/// Ported from `bwb1_materialize_breakeven.rs` verbatim.
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

/// Time a whole trajectory-shaped closure, warmup-then-measure, return
/// median ms for ONE full trajectory (not per-step).
fn time_trajectory(mut run_once: impl FnMut()) -> f64 {
    for _ in 0..WARMUP_TRAJECTORIES {
        run_once();
    }
    let ms: Vec<f64> = (0..TIMED_TRAJECTORIES)
        .map(|_| {
            let t0 = Instant::now();
            run_once();
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    median(ms)
}

/// Sample which churn-fraction bucket applies, given a change happens.
fn sample_frac(rng: &mut StdRng, p_frac: &[f64; 4]) -> f64 {
    let r: f64 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in p_frac.iter().enumerate() {
        cum += p;
        if r < cum {
            return CHURN_FRACTIONS[i];
        }
    }
    *CHURN_FRACTIONS.last().unwrap()
}

/// Pre-generate a fixed schedule of per-step slot changes, driven by
/// `cell`'s real churn odds. Each step's entry is empty (no change) or
/// `(slot, new_feature)` pairs — `slot` indexes the K-length mask
/// directly (never shifts), `new_feature` is a feature index not
/// already present in the CURRENT mask at generation time (so a "swap"
/// is a real swap, not a same-feature no-op). Generated ONCE, replayed
/// identically across all three arms and all repeats — the comparison
/// is only fair if every arm sees the exact same mask evolution.
fn gen_schedule(
    rng: &mut StdRng,
    k: usize,
    intermediate_size: usize,
    initial_mask: &[usize],
    cell: &ChurnCell,
    steps: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut mask = initial_mask.to_vec();
    let mut present: HashSet<usize> = initial_mask.iter().copied().collect();
    let mut schedule = Vec::with_capacity(steps);
    for _ in 0..steps {
        if rng.gen::<f64>() >= cell.p_change {
            schedule.push(Vec::new());
            continue;
        }
        let frac = sample_frac(rng, &cell.p_frac);
        let changed_count = ((frac * k as f64).round() as usize).clamp(1, k);
        let slots = rand::seq::index::sample(rng, k, changed_count).into_vec();
        let mut changes = Vec::with_capacity(changed_count);
        for slot in slots {
            // Draw a replacement feature not already in the mask —
            // bounded retries; `intermediate_size` is always >> k in
            // this experiment's grid so this terminates immediately in
            // practice.
            loop {
                let cand = rng.gen_range(0..intermediate_size);
                if !present.contains(&cand) {
                    present.remove(&mask[slot]);
                    present.insert(cand);
                    mask[slot] = cand;
                    changes.push((slot, cand));
                    break;
                }
            }
        }
        schedule.push(changes);
    }
    schedule
}

/// Arm 1: full rebuild on any change, reuse otherwise. Returns the
/// forward output count touched (kept alive so the compiler can't hoist
/// the calls out from under the timer) — the value itself is discarded.
#[allow(clippy::too_many_arguments)]
fn run_full_rebuild(
    index: &VectorIndex,
    layer: usize,
    hidden: usize,
    initial_mask: &[usize],
    schedule: &[Vec<(usize, usize)>],
    ffn: &WalkFfn,
    x: &[f32],
    use_gelu: bool,
) -> usize {
    let mut mask = initial_mask.to_vec();
    let mut compact = CompactDenseLayer::materialize(index, layer, &mask, hidden)
        .expect("initial materialize succeeds");
    let mut touched = 0usize;
    for changes in schedule {
        if !changes.is_empty() {
            for &(slot, feat) in changes {
                mask[slot] = feat;
            }
            compact = CompactDenseLayer::materialize(index, layer, &mask, hidden)
                .expect("materialize succeeds on every in-range mask this schedule produces");
        }
        if let Some(out) = ffn.compact_dense_forward(&compact, x, use_gelu, hidden) {
            touched = touched.wrapping_add(out.len());
        }
    }
    touched
}

/// Arm 2: incremental patch on change, reuse otherwise.
#[allow(clippy::too_many_arguments)]
fn run_incremental_patch(
    index: &VectorIndex,
    layer: usize,
    hidden: usize,
    initial_mask: &[usize],
    schedule: &[Vec<(usize, usize)>],
    ffn: &WalkFfn,
    x: &[f32],
    use_gelu: bool,
) -> usize {
    let mut compact = CompactDenseLayer::materialize(index, layer, initial_mask, hidden)
        .expect("initial materialize succeeds");
    let mut touched = 0usize;
    for changes in schedule {
        if !changes.is_empty() {
            compact
                .patch(index, layer, hidden, changes)
                .expect("patch succeeds on every in-range change this schedule produces");
        }
        if let Some(out) = ffn.compact_dense_forward(&compact, x, use_gelu, hidden) {
            touched = touched.wrapping_add(out.len());
        }
    }
    touched
}

/// Arm 3: always sparse-gather — a fresh full-K materialize EVERY step,
/// functionally identical to `gather_q4k_accumulate` (see module doc).
#[allow(clippy::too_many_arguments)]
fn run_always_gather(
    index: &VectorIndex,
    layer: usize,
    hidden: usize,
    initial_mask: &[usize],
    schedule: &[Vec<(usize, usize)>],
    ffn: &WalkFfn,
    x: &[f32],
    use_gelu: bool,
) -> usize {
    let mut mask = initial_mask.to_vec();
    let mut touched = 0usize;
    for changes in schedule {
        for &(slot, feat) in changes {
            mask[slot] = feat;
        }
        let compact = CompactDenseLayer::materialize(index, layer, &mask, hidden)
            .expect("materialize succeeds on every in-range mask this schedule produces");
        if let Some(out) = ffn.compact_dense_forward(&compact, x, use_gelu, hidden) {
            touched = touched.wrapping_add(out.len());
        }
    }
    touched
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
        eprintln!("Usage: bwb2_incremental_compact --vindex PATH");
        std::process::exit(1);
    }

    println!("=== BW-B2: incremental patch vs full rebuild vs always-gather, real churn ===\n");
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
    let intermediate_size = weights.intermediate_size;
    let use_gelu = weights.arch.activation().uses_gelu_tanh_gate_up();
    let k_max = *K_GRID.iter().max().unwrap();
    println!(
        "{num_layers} layers, hidden={hidden}, intermediate={intermediate_size}, K grid={K_GRID:?}\n"
    );

    println!(
        "{:<7} {:<6} {:>10} {:>14} {:>14} {:>14} {:>14}",
        "band", "K", "t_forward", "t_matz_full", "t_patch(avg)", "t_matz(traj)", "t_patch(traj)"
    );
    println!("{}", "-".repeat(96));

    for cell in &CHURN_CELLS {
        let layer = cell.layer;
        let x = synthetic_x(hidden, layer);
        let x_arr = Array2::from_shape_vec((1, hidden), x.clone())?;

        // Real oracle mask: capture the top-K_MAX gate ranking at this
        // layer exactly as BW-B/BW-B1 do (GateOnly selection, real Q4K
        // gate rows, real row-dot kernel).
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
            println!(
                "  layer {layer} ({}) captured only {} < K_MAX={k_max} ranked features — skipped",
                cell.band,
                ranked.len()
            );
            continue;
        }

        // Layer-level warmup — settle this layer's Q4K bytes into cache
        // before ANY timed cell, exactly the discipline
        // `bwb1_materialize_breakeven.rs` had to add after finding a
        // K-dependent first-call cold-start tax.
        for _ in 0..8 {
            let warm = CompactDenseLayer::materialize(&index, layer, &ranked[..k_max], hidden);
            let _ = warm;
        }

        for &k in &K_GRID {
            let mask = &ranked[..k];
            let ffn_cfg = WalkFfnConfig::sparse(num_layers, k);
            let ffn = WalkFfn::from_config(&weights, &index, ffn_cfg);

            // ── Component costs (isolated, BW-B1-style block timing). ──
            let compact = CompactDenseLayer::materialize(&index, layer, mask, hidden)
                .expect("materialize succeeds — sidecar loaded, mask non-empty, in range");
            let t_forward = time_arm(WARMUP_BLOCKS, BLOCKS_PER_CELL, || {
                let _ = ffn.compact_dense_forward(&compact, &x, use_gelu, hidden);
            });
            let t_materialize_full = time_once(MATERIALIZE_WARMUP, MATERIALIZE_REPEATS, || {
                let _ = CompactDenseLayer::materialize(&index, layer, mask, hidden);
            });

            // t_patch at each churn-fraction bucket, weighted by this
            // band's real P(fraction | change) to get one representative
            // average patch cost for the summary line.
            let mut t_patch_weighted = 0.0;
            for (frac_idx, &frac) in CHURN_FRACTIONS.iter().enumerate() {
                let changed_count = ((frac * k as f64).round() as usize).clamp(1, k);
                // Deterministic, valid replacement features: shift each
                // slot's current feature by half the pool width so the
                // replacement is always in range and (for k < pool/2)
                // always different from the original — a synthetic but
                // uniform stand-in for "a fresh feature", chosen for
                // reproducibility rather than drawn from the trajectory
                // RNG (this is the isolated-component measurement, kept
                // independent of the trajectory schedule below).
                let changes: Vec<(usize, usize)> = (0..changed_count)
                    .map(|slot| {
                        (
                            slot,
                            (mask[slot] + intermediate_size / 2) % intermediate_size,
                        )
                    })
                    .collect();
                let mut patch_target = CompactDenseLayer::materialize(&index, layer, mask, hidden)
                    .expect("materialize succeeds for the patch-timing base layer");
                let t_patch_frac = time_once(MATERIALIZE_WARMUP, MATERIALIZE_REPEATS, || {
                    let _ = patch_target.patch(&index, layer, hidden, &changes);
                });
                t_patch_weighted += cell.p_frac[frac_idx] * t_patch_frac;
            }

            // ── End-to-end trajectory, real churn schedule, all three
            // arms replaying the IDENTICAL schedule. ──
            let mut rng = StdRng::seed_from_u64(SCHEDULE_SEED ^ (k as u64) << 8 ^ layer as u64);
            let schedule =
                gen_schedule(&mut rng, k, intermediate_size, mask, cell, TRAJECTORY_STEPS);

            let t_full_traj = time_trajectory(|| {
                let _ =
                    run_full_rebuild(&index, layer, hidden, mask, &schedule, &ffn, &x, use_gelu);
            }) / TRAJECTORY_STEPS as f64;
            let t_patch_traj = time_trajectory(|| {
                let _ = run_incremental_patch(
                    &index, layer, hidden, mask, &schedule, &ffn, &x, use_gelu,
                );
            }) / TRAJECTORY_STEPS as f64;
            let t_gather_traj = time_trajectory(|| {
                let _ =
                    run_always_gather(&index, layer, hidden, mask, &schedule, &ffn, &x, use_gelu);
            }) / TRAJECTORY_STEPS as f64;

            println!(
                "{:<7} {:<6} {:>9.1}us {:>11.1}us {:>11.1}us {:>11.1}us {:>11.1}us",
                cell.band,
                k,
                t_forward * 1000.0,
                t_materialize_full * 1000.0,
                t_patch_weighted * 1000.0,
                t_full_traj * 1000.0,
                t_patch_traj * 1000.0,
            );

            // Analytical prediction vs measured trajectory — internal
            // consistency control, not a blind assertion.
            let predicted_full = t_forward + cell.p_change * t_materialize_full;
            let predicted_patch = t_forward + cell.p_change * t_patch_weighted;
            let predicted_gather = t_forward + t_materialize_full;
            println!(
                "{:<7} {:<6} {:>10} predicted: full={:>7.1}us (meas {:>7.1}us, {:+.1}%)  \
                 patch={:>7.1}us (meas {:>7.1}us, {:+.1}%)  gather={:>7.1}us (meas {:>7.1}us, {:+.1}%)",
                "",
                "",
                "",
                predicted_full * 1000.0,
                t_full_traj * 1000.0,
                (t_full_traj - predicted_full) / predicted_full * 100.0,
                predicted_patch * 1000.0,
                t_patch_traj * 1000.0,
                (t_patch_traj - predicted_patch) / predicted_patch * 100.0,
                predicted_gather * 1000.0,
                t_gather_traj * 1000.0,
                (t_gather_traj - predicted_gather) / predicted_gather * 100.0,
            );

            // Patch vs full-rebuild win, under THIS band's real churn.
            let patch_vs_full_speedup = t_full_traj / t_patch_traj;
            let patch_vs_gather_speedup = t_gather_traj / t_patch_traj;
            println!(
                "{:<7} {:<6} {:>10} patch/full={:.3}x  patch/gather={:.3}x  full/gather={:.3}x\n",
                "",
                "",
                "",
                patch_vs_full_speedup,
                patch_vs_gather_speedup,
                t_full_traj / t_gather_traj,
            );
        }
    }

    Ok(())
}
