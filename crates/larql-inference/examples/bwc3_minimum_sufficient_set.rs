//! BW-C3 — minimum sufficient expert set: individually-safe (74% per
//! BW-C1/C2) does NOT imply jointly-removable. Experts A, B, C could
//! each be individually dispensable while mutually redundant with each
//! other — not all three simultaneously droppable. This exhaustively
//! tests all 15 non-empty subsets of a real top-4 routing at each
//! checkpoint and reports the minimum KEPT-expert cardinality that
//! still preserves the trajectory — the number that actually bounds
//! how much routed-expert compute could be cut, if some policy could
//! find it.
//!
//! Method: same KV-fork checkpoint machinery as BW-C1/C2
//! (`bwc1_kvfork_sanity.rs` validated the R1/R4 gates on this exact
//! restore/decode path; not re-validated here — no new checkpoint
//! surface, same `BoundaryCheckpoint` primitives). For each
//! (checkpoint, target layer): capture the real top-4 `(expert,
//! weight)` routing, then for the 15 non-empty subsets of those 4
//! experts, `arm_set` (BW-C3's generalisation of BW-C's one-expert
//! `arm_once` to simultaneous multi-expert ablation — see
//! `expert_override`'s module doc), decode `N_CONTINUATION` tokens,
//! compare against ONE clean baseline for that checkpoint.
//! `fired_mask()` is checked against the intended subset on every
//! test — a partial fire means one or more targets never actually ran
//! at this layer (a mis-specified target, not a null result), and that
//! test is excluded rather than silently counted.
//!
//! `minimum_sufficient_size(checkpoint, layer) = 4 - max(|R| : removing
//! R is safe)` — the smallest number of experts you'd need to KEEP for
//! there to exist SOME safe choice of which ones. Not assumed
//! monotonic in `|R|` (removing more is not guaranteed to be "at least
//! as safe" as removing less), so all 15 subsets are tested per point,
//! not a search along one dimension.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc3_minimum_sufficient_set -- \
//!     --vindex /path/to/gpt-oss-20b-q4k.vindex

use std::path::PathBuf;

use larql_compute::cpu::ops::moe::expert_override;
use larql_inference::ffn::LocalMoeFfn;
use larql_inference::kv_engine::PerLayerKvAccess;
use larql_inference::ModelWeights;
use larql_kv::engines::semantic_promotion::checkpoint::BoundaryCheckpoint;
use larql_kv::engines::semantic_promotion::ids::CheckpointId;
use larql_kv::AnyEngine;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

/// Prompts, kept diverse on purpose — same rationale as BW-C1/C2's
/// `bwc1_skippability_correlates.rs`: a single prompt can land on a
/// greedy repetition attractor that reads as "safe" for the wrong
/// reason (attractors resist ANY perturbation, not just genuinely
/// redundant ones).
const PROMPTS: [&str; 6] = [
    "The history of the Roman Empire began when",
    "In the field of quantum computing, researchers recently discovered",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:",
    "The recipe calls for two cups of flour, one teaspoon of",
    "Dear Sir or Madam, I am writing to formally request",
    "The three primary colors are red, blue, and",
];

/// Fewer positions per prompt than BW-C1/C2's 8 — each checkpoint now
/// costs 15 subset tests instead of 1, so this stays bounded at
/// 6 × 4 × 3 = 72 (checkpoint, layer) points, 1,080 subset tests
/// total, roughly the same wall-clock budget as the BW-C1/C2 census.
const CHECKPOINT_STEPS: [usize; 4] = [0, 4, 8, 12];
/// Tokens decoded post-intervention for the trajectory comparison —
/// unchanged from BW-C1/C2, for direct comparability.
const N_CONTINUATION: usize = 6;
/// Same three depths as BW-C1/C2 (early/mid/late), so the minimum-
/// sufficient-size histogram below can be read against that census's
/// depth-dependent safe% directly.
const LAYER_FRACTIONS: [f64; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];

fn per_layer_kv(engine: &mut AnyEngine) -> Option<&mut dyn PerLayerKvAccess> {
    match engine {
        AnyEngine::Kv(e) => e.per_layer_kv_mut(),
        AnyEngine::Retrieval(_) => None,
    }
}

fn restore(engine: &mut AnyEngine, ckpt: &BoundaryCheckpoint) -> Result<(), String> {
    let kv = per_layer_kv(engine).ok_or("per_layer_kv_mut returned None on restore")?;
    ckpt.restore(kv)
        .map_err(|e| format!("restore failed: {e:?}"))
}

fn step(
    engine: &mut AnyEngine,
    weights: &ModelWeights,
    ffn: &LocalMoeFfn,
    index: &VectorIndex,
    current: u32,
) -> u32 {
    let h = engine
        .decode_step_resident(weights, ffn, index, current)
        .expect("decode_step_resident failed");
    let logits = larql_inference::research::hidden_to_raw_logits(weights, &h);
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn decode_n(
    engine: &mut AnyEngine,
    weights: &ModelWeights,
    ffn: &LocalMoeFfn,
    index: &VectorIndex,
    mut current: u32,
    n: usize,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        current = step(engine, weights, ffn, index, current);
        out.push(current);
    }
    out
}

/// Every non-empty subset of {0, 1, 2, 3} as a 4-bit mask (1..=15) —
/// bit `i` set means "index `i` of the top-4 is in this subset".
fn nonempty_subset_masks() -> impl Iterator<Item = u8> {
    1u8..=15
}

struct SubsetPoint {
    prompt_idx: usize,
    position: usize,
    layer: usize,
    /// Size of the REMOVED set this subset test ablated (1..=4).
    removed_size: usize,
    safe: bool,
    /// Distinct tokens among this checkpoint's `N_CONTINUATION` clean
    /// baseline tokens (same value for every subset test at this
    /// checkpoint — carried per-point rather than in a separate lookup
    /// for simplicity). A LOW count (bwc1_kvfork_sanity.rs found a
    /// live example of all 6 identical) means the un-perturbed
    /// trajectory was already a repetition attractor — those resist
    /// ANY intervention by construction, so a "safe" label there is
    /// evidence about the attractor, not about whether the ablated
    /// computation was genuinely redundant. Every headline number
    /// below is reported both including and excluding these.
    baseline_distinct_tokens: usize,
}

/// Below this many distinct tokens (out of `N_CONTINUATION`), a
/// baseline continuation is treated as a likely repetition attractor
/// rather than a genuine trajectory. 6 tokens with only 1-2 distinct
/// values is the same signature `bwc1_kvfork_sanity.rs` found on a
/// real decode (`[1261, 1261, 1261, 1261, 1261, 1261]`).
const DEGENERATE_BASELINE_MAX_DISTINCT: usize = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = PathBuf::from(
        std::env::var("HOME").unwrap_or_default() + "/chris-models/gpt-oss-20b-q4k.vindex",
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
        std::process::exit(1);
    }

    println!("=== BW-C3: minimum sufficient expert set ===\n");
    let mut cb = SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
    for layer in 0..weights.num_layers {
        larql_inference::vindex::insert_q4k_layer_tensors_resident(&mut weights, &index, layer)?;
    }
    let weights_ref = &weights;
    let moe_ffn = LocalMoeFfn {
        weights: weights_ref,
        index: Some(&index),
    };
    let num_layers = weights_ref.num_layers;
    let target_layers: Vec<usize> = LAYER_FRACTIONS
        .iter()
        .map(|f| ((*f * num_layers as f64) as usize).min(num_layers - 1))
        .collect();
    println!("target layer depths sampled per checkpoint: {target_layers:?}\n");

    let mut points: Vec<SubsetPoint> = Vec::new();
    let mut skipped_mismatched_fire = 0usize;
    let mut skipped_not_four_experts = 0usize;

    for (prompt_idx, prompt) in PROMPTS.iter().enumerate() {
        let encoding = tokenizer
            .encode(*prompt, true)
            .map_err(|e| format!("{e}"))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        println!(
            "--- prompt {prompt_idx}: {:?} ({} tokens) ---",
            prompt,
            prompt_ids.len()
        );

        let mut engine = larql_kv::EngineKind::from_name("standard")
            .expect("standard engine exists")
            .build(larql_inference::cpu_engine_backend());
        let h0 = engine
            .prefill_resident(weights_ref, &moe_ffn, &index, &prompt_ids)
            .map_err(|e| format!("prefill failed: {e}"))?;
        let mut current = {
            let logits = larql_inference::research::hidden_to_raw_logits(weights_ref, &h0);
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };

        let max_checkpoint_step = *CHECKPOINT_STEPS.iter().max().unwrap();
        let mut step_idx = 0usize;
        loop {
            if CHECKPOINT_STEPS.contains(&step_idx) {
                let kv = per_layer_kv(&mut engine).ok_or("per_layer_kv_mut returned None")?;
                let ckpt = BoundaryCheckpoint::capture(
                    CheckpointId::from_counter((prompt_idx * 1000 + step_idx) as u128),
                    kv,
                )
                .map_err(|e| format!("capture failed: {e:?}"))?;

                expert_override::start_observing();
                let _ = step(&mut engine, weights_ref, &moe_ffn, &index, current);
                let observed = expert_override::stop_observing();
                restore(&mut engine, &ckpt)?;

                let baseline_tokens = decode_n(
                    &mut engine,
                    weights_ref,
                    &moe_ffn,
                    &index,
                    current,
                    N_CONTINUATION,
                );
                restore(&mut engine, &ckpt)?;
                let baseline_distinct_tokens: usize = baseline_tokens
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                if baseline_distinct_tokens <= DEGENERATE_BASELINE_MAX_DISTINCT {
                    println!(
                        "  checkpoint step={step_idx}: DEGENERATE baseline ({baseline_distinct_tokens} \
                         distinct/{N_CONTINUATION}) — {baseline_tokens:?}"
                    );
                }

                for &layer in &target_layers {
                    let at_layer: Vec<(usize, f32)> = observed
                        .iter()
                        .filter(|obs| obs.layer == layer)
                        .map(|obs| (obs.expert, obs.router_weight))
                        .collect();
                    if at_layer.len() != 4 {
                        // Top-4 routing didn't produce exactly 4 distinct
                        // observed calls at this layer (e.g. two ranks
                        // landed on the same expert) — refuse to guess a
                        // subset structure that doesn't match reality.
                        skipped_not_four_experts += 1;
                        continue;
                    }

                    for mask in nonempty_subset_masks() {
                        let removed: Vec<usize> = (0..4)
                            .filter(|&i| (mask >> i) & 1 == 1)
                            .map(|i| at_layer[i].0)
                            .collect();
                        let expected_fired: u64 =
                            removed.iter().fold(0u64, |m, &e| m | (1u64 << e));

                        expert_override::arm_set(layer, &removed);
                        let ablated_tokens = decode_n(
                            &mut engine,
                            weights_ref,
                            &moe_ffn,
                            &index,
                            current,
                            N_CONTINUATION,
                        );
                        let fired_mask = expert_override::fired_mask();
                        expert_override::disarm();
                        restore(&mut engine, &ckpt)?;

                        if fired_mask != expected_fired {
                            // One or more targeted experts never actually
                            // ran at this layer this step — a mis-specified
                            // target (shouldn't happen off REAL observed
                            // routing, but refuse rather than record a
                            // fabricated result if it ever does).
                            skipped_mismatched_fire += 1;
                            continue;
                        }

                        let safe = ablated_tokens == baseline_tokens;
                        points.push(SubsetPoint {
                            prompt_idx,
                            position: step_idx,
                            layer,
                            removed_size: removed.len(),
                            safe,
                            baseline_distinct_tokens,
                        });
                    }
                }
                println!(
                    "  checkpoint step={step_idx}: {} subset tests so far",
                    points.len()
                );
            }
            if step_idx >= max_checkpoint_step {
                break;
            }
            current = step(&mut engine, weights_ref, &moe_ffn, &index, current);
            step_idx += 1;
        }
    }

    println!("\ntotal subset tests: {}", points.len());
    println!(
        "skipped (fired_mask mismatch): {skipped_mismatched_fire}, \
         skipped (top-4 didn't yield 4 distinct experts): {skipped_not_four_experts}"
    );

    // ── Per (prompt, position, layer): minimum_sufficient_size = 4 -
    // max(|removed| : removing it was safe). Groups are keyed by the
    // checkpoint+layer identity, not assumed contiguous in `points`. ──
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<(usize, usize, usize), Vec<&SubsetPoint>> = BTreeMap::new();
    for p in &points {
        groups
            .entry((p.prompt_idx, p.position, p.layer))
            .or_default()
            .push(p);
    }

    let mut histogram = [0usize; 5]; // index = minimum_sufficient_size (0..=4)
    let mut histogram_genuine = [0usize; 5]; // excludes degenerate-baseline points
    let mut histogram_degenerate = [0usize; 5]; // degenerate-baseline points ONLY
    let mut histogram_by_depth: [[usize; 5]; 3] = [[0; 5]; 3]; // [depth_idx][size]
    println!(
        "\n{:<3} {:>4} {:>6} {:>12} {:>12} {:>10}",
        "p", "pos", "layer", "max_safe_rm", "min_suff", "distinct"
    );
    let mut n_degenerate_groups = 0usize;
    for ((prompt_idx, position, layer), group) in &groups {
        let max_safe_removed = group
            .iter()
            .filter(|p| p.safe)
            .map(|p| p.removed_size)
            .max()
            .unwrap_or(0);
        let min_sufficient = 4 - max_safe_removed;
        let baseline_distinct = group[0].baseline_distinct_tokens;
        let degenerate = baseline_distinct <= DEGENERATE_BASELINE_MAX_DISTINCT;
        histogram[min_sufficient] += 1;
        if degenerate {
            n_degenerate_groups += 1;
            histogram_degenerate[min_sufficient] += 1;
        } else {
            histogram_genuine[min_sufficient] += 1;
        }
        if let Some(depth_idx) = target_layers.iter().position(|&l| l == *layer) {
            histogram_by_depth[depth_idx][min_sufficient] += 1;
        }
        println!(
            "{prompt_idx:<3} {position:>4} {layer:>6} {max_safe_removed:>12} {min_sufficient:>12} \
             {baseline_distinct:>7}/{N_CONTINUATION}{}",
            if degenerate { "  <-- attractor?" } else { "" }
        );
    }

    println!(
        "\n{:=<60}\nminimum sufficient expert set size — histogram (n={} checkpoint×layer points, \
         {n_degenerate_groups} flagged as a likely repetition attractor: baseline had \
         <= {DEGENERATE_BASELINE_MAX_DISTINCT} distinct tokens out of {N_CONTINUATION})",
        "",
        groups.len()
    );
    println!("  ALL points (includes attractors — the number from the smoke test, do not headline this alone):");
    for (size, &n) in histogram.iter().enumerate() {
        let pct = 100.0 * n as f64 / groups.len().max(1) as f64;
        println!("    {size} expert(s) sufficient: {n:>3} ({pct:>5.1}%)");
    }
    let n_genuine = groups.len() - n_degenerate_groups;
    println!("  GENUINE points only (baseline was NOT a repetition attractor, n={n_genuine}):");
    for (size, &n) in histogram_genuine.iter().enumerate() {
        let pct = 100.0 * n as f64 / n_genuine.max(1) as f64;
        println!("    {size} expert(s) sufficient: {n:>3} ({pct:>5.1}%)");
    }
    println!(
        "  DEGENERATE (attractor) points only, for comparison (n={n_degenerate_groups}) — if \
         this distribution looks similar to the genuine one, attractors are NOT driving the \
         headline; if it's skewed heavily toward 0, they are:"
    );
    for (size, &n) in histogram_degenerate.iter().enumerate() {
        let pct = 100.0 * n as f64 / n_degenerate_groups.max(1) as f64;
        println!("    {size} expert(s) sufficient: {n:>3} ({pct:>5.1}%)");
    }

    println!("\nsplit by depth (early / mid / late = layers {target_layers:?}):");
    let depth_names = ["early", "mid", "late"];
    for (depth_idx, name) in depth_names.iter().enumerate() {
        let total: usize = histogram_by_depth[depth_idx].iter().sum();
        print!("  {name:<6} (n={total:>3}): ");
        for (size, &n) in histogram_by_depth[depth_idx].iter().enumerate() {
            print!("{size}={n:>3}  ");
        }
        println!();
    }

    println!(
        "\nminimum_sufficient_size=0 means removing ALL 4 top-k experts at that checkpoint left \
         the {N_CONTINUATION}-token continuation byte-identical — the whole routed group was \
         jointly unnecessary there. =4 means no tested removal (of any size) preserved the \
         trajectory — every expert was individually necessary given the other three still ran, \
         OR necessary in every combination tested. This does NOT by itself mean each of the 4 \
         is individually unskippable (see BW-C1/C2 for that direct measurement) — it means no \
         REMOVAL preserved the trajectory in THIS joint test."
    );
    println!(
        "\nspace + guarantee (do not quote a stronger claim than this): min_suff is \
         minimum-CARDINALITY, not greedy or inclusion-minimal — earned by exhaustively \
         enumerating all 15 non-empty subsets of THIS checkpoint's real top-4, never by a \
         stopping rule. It is minimum ONLY within that space: (a) the 4 experts this specific \
         step's router actually selected — never tested against swapping in a DIFFERENT \
         expert; (b) safety = exact {N_CONTINUATION}-token greedy match, not any looser \
         quality bar; (c) one single decode step's snapshot — the same layer at a DIFFERENT \
         position can and does have a different real top-4 and a different min_suff. It says \
         nothing about longer horizons (BW-C4) or about a REPEATED policy applied throughout \
         generation (BW-C5)."
    );

    Ok(())
}
