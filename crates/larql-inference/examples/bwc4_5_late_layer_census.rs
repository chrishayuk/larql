//! BW-C4.5 — late-layer horizon census at scale: BW-C4's late-depth,
//! open-ended-prompt cell (the 70%-flat-at-16/32/64 plateau) rested on
//! only n=10. This deliberately narrows scope to widen sample: LATE
//! LAYER ONLY, 20 diverse open-ended prompts (no recipe/letter/list
//! patterns — see `feedback_templated_completion_robustness_confound`
//! for why those were excluded), 4 positions each = 80 checkpoints.
//! No 15-subset search (BW-C3) and no multi-depth sweep (BW-C1-C4) —
//! every cycle goes into the one cell that matters right now.
//!
//! Adds a CONTINUOUS entropy covariate — mean baseline top-1 logit
//! margin across the clean continuation — instead of the eyeballed
//! open-ended/templated label BW-C4 used post-hoc. Low margin =
//! contested/high-entropy decisions; high margin = obvious/low-entropy
//! (the templated-prompt signature). This lets survival-vs-entropy be
//! read as a correlation, not a binary split chosen by inspection.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc4_5_late_layer_census -- \
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

/// 20 deliberately open-ended prompts — narrative, technical, causal-
/// analytic — chosen to have no near-forced continuation. Explicitly
/// avoids recipe/list, form-letter, and fill-in-the-blank patterns:
/// exactly the shape BW-C4 found reads as 100%-robust regardless of
/// which computation ran.
const PROMPTS: [&str; 20] = [
    "The history of the Roman Empire began when",
    "In the field of quantum computing, researchers recently discovered",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:",
    "The detective walked into the room and immediately noticed",
    "Scientists have long debated whether",
    "The economic policy sparked controversy because",
    "According to the latest research on climate change,",
    "The novel's protagonist decided to",
    "When analyzing the algorithm's time complexity, one must consider",
    "The philosopher argued that free will",
    "In a surprising turn of events, the company announced",
    "The ancient civilization's collapse was likely caused by",
    "Modern architecture has evolved to prioritize",
    "The experiment's results suggested that",
    "Critics of the new policy argue that",
    "The spacecraft's trajectory required scientists to",
    "Throughout history, revolutions have often begun with",
    "The startup's founder explained that their approach",
    "Recent advances in machine learning have enabled",
    "The negotiation between the two countries stalled when",
];
const CHECKPOINT_STEPS: [usize; 4] = [0, 4, 8, 12];
/// Late depth only — the fraction that produced BW-C4's flat plateau.
const LATE_LAYER_FRACTION: f64 = 5.0 / 6.0;
const N_HORIZON: usize = 64;
const HORIZON_MARKERS: [usize; 4] = [6, 16, 32, 64];

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
) -> (u32, Vec<f32>) {
    let h = engine
        .decode_step_resident(weights, ffn, index, current)
        .expect("decode_step_resident failed");
    let logits = larql_inference::research::hidden_to_raw_logits(weights, &h);
    let tok = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    (tok, logits)
}

fn top1_margin(logits: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sorted.first().copied().unwrap_or(0.0) - sorted.get(1).copied().unwrap_or(0.0)
}

/// Decode `n` steps: every token, logits at `save_logits_at` markers
/// (bounded memory — see BW-C4's rationale), and the mean top-1 margin
/// across ALL `n` steps (cheap — margin is derived from logits already
/// computed for token selection, never needs the full vector kept).
fn decode_n_with_markers(
    engine: &mut AnyEngine,
    weights: &ModelWeights,
    ffn: &LocalMoeFfn,
    index: &VectorIndex,
    mut current: u32,
    n: usize,
    save_logits_at: &[usize],
) -> (Vec<u32>, std::collections::HashMap<usize, Vec<f32>>, f32) {
    let mut tokens = Vec::with_capacity(n);
    let mut marker_logits = std::collections::HashMap::new();
    let mut margin_sum = 0.0f32;
    for i in 1..=n {
        let (tok, logits) = step(engine, weights, ffn, index, current);
        margin_sum += top1_margin(&logits);
        if save_logits_at.contains(&i) {
            marker_logits.insert(i, logits);
        }
        tokens.push(tok);
        current = tok;
    }
    (tokens, marker_logits, margin_sum / n as f32)
}

fn softmax_bits_kl(a: &[f32], b: &[f32]) -> f32 {
    let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_b = b.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_a: Vec<f32> = a.iter().map(|v| (v - max_a).exp()).collect();
    let exp_b: Vec<f32> = b.iter().map(|v| (v - max_b).exp()).collect();
    let sum_a: f32 = exp_a.iter().sum();
    let sum_b: f32 = exp_b.iter().sum();
    let mut kl_nats = 0.0f32;
    for (ea, eb) in exp_a.iter().zip(&exp_b) {
        let pa = ea / sum_a;
        let pb = (eb / sum_b).max(1e-30);
        if pa > 1e-30 {
            kl_nats += pa * (pa / pb).ln();
        }
    }
    kl_nats / std::f32::consts::LN_2
}

struct CensusPoint {
    prompt_idx: usize,
    position: usize,
    /// Mean baseline top-1 margin across the clean continuation — the
    /// continuous entropy proxy. High = obvious/templated-like,
    /// low = contested/open-ended.
    baseline_mean_margin: f32,
    first_divergence: Option<usize>,
    kl_at_marker: [Option<f32>; 4],
}

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

    println!("=== BW-C4.5: late-layer horizon census at scale ===\n");
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
    let late_layer = ((LATE_LAYER_FRACTION * num_layers as f64) as usize).min(num_layers - 1);
    println!("late layer: {late_layer} (of {num_layers})\n");

    let mut total_points = 0usize;
    let mut census: Vec<CensusPoint> = Vec::new();
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
                let (_, _) = step(&mut engine, weights_ref, &moe_ffn, &index, current);
                let observed = expert_override::stop_observing();
                restore(&mut engine, &ckpt)?;

                let (baseline_tokens, baseline_markers, baseline_mean_margin) =
                    decode_n_with_markers(
                        &mut engine,
                        weights_ref,
                        &moe_ffn,
                        &index,
                        current,
                        N_HORIZON,
                        &HORIZON_MARKERS,
                    );
                restore(&mut engine, &ckpt)?;

                total_points += 1;
                let at_layer: Vec<usize> = observed
                    .iter()
                    .filter(|obs| obs.layer == late_layer)
                    .map(|obs| obs.expert)
                    .collect();
                if at_layer.len() != 4 {
                    skipped_not_four_experts += 1;
                } else {
                    let expected_fired: u64 = at_layer.iter().fold(0u64, |m, &e| m | (1u64 << e));

                    expert_override::arm_set(late_layer, &at_layer);
                    let (ablated_tokens, ablated_markers, _) = decode_n_with_markers(
                        &mut engine,
                        weights_ref,
                        &moe_ffn,
                        &index,
                        current,
                        N_HORIZON,
                        &HORIZON_MARKERS,
                    );
                    let fired_mask = expert_override::fired_mask();
                    expert_override::disarm();
                    restore(&mut engine, &ckpt)?;

                    if fired_mask != expected_fired {
                        skipped_mismatched_fire += 1;
                    } else {
                        let safe_at_6 = baseline_tokens[..6] == ablated_tokens[..6];
                        if safe_at_6 {
                            let first_divergence = baseline_tokens
                                .iter()
                                .zip(&ablated_tokens)
                                .position(|(a, b)| a != b);

                            let mut kl_at_marker = [None; 4];
                            for (mi, &marker) in HORIZON_MARKERS.iter().enumerate() {
                                let still_matching = first_divergence.is_none_or(|d| d >= marker);
                                if still_matching {
                                    if let (Some(bl), Some(al)) = (
                                        baseline_markers.get(&marker),
                                        ablated_markers.get(&marker),
                                    ) {
                                        kl_at_marker[mi] = Some(softmax_bits_kl(bl, al));
                                    }
                                }
                            }

                            census.push(CensusPoint {
                                prompt_idx,
                                position: step_idx,
                                baseline_mean_margin,
                                first_divergence,
                                kl_at_marker,
                            });
                        }
                    }
                }
                println!(
                    "  checkpoint step={step_idx}: {}/{} safe-at-6 so far",
                    census.len(),
                    total_points
                );
            }
            if step_idx >= max_checkpoint_step {
                break;
            }
            current = step(&mut engine, weights_ref, &moe_ffn, &index, current).0;
            step_idx += 1;
        }
    }

    println!(
        "\ntotal checkpoints: {total_points}, safe-at-6: {} ({:.1}%)",
        census.len(),
        100.0 * census.len() as f64 / total_points.max(1) as f64
    );
    println!(
        "skipped (fired_mask mismatch): {skipped_mismatched_fire}, skipped (top-4 didn't yield \
         4 distinct experts): {skipped_not_four_experts}"
    );

    let n = census.len();
    println!(
        "\n{:=<70}\nsurvival curve (late layer, open-ended prompts, n={n})",
        ""
    );
    for &marker in &HORIZON_MARKERS {
        let survived = census
            .iter()
            .filter(|p| p.first_divergence.is_none_or(|d| d >= marker))
            .count();
        println!(
            "  h{marker:>3}: {survived:>3}/{n} = {:.1}%",
            100.0 * survived as f64 / n.max(1) as f64
        );
    }

    // ── Entropy covariate: does baseline_mean_margin predict survival
    // to horizon 64? Replaces BW-C4's eyeballed open-ended/templated
    // label with a continuous, pre-registered proxy. ──
    let margins: Vec<f64> = census
        .iter()
        .map(|p| p.baseline_mean_margin as f64)
        .collect();
    let survives_64: Vec<f64> = census
        .iter()
        .map(|p| {
            if p.first_divergence.is_none_or(|d| d >= 64) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let mm = mean(&margins);
    let ms = mean(&survives_64);
    let cov: f64 = margins
        .iter()
        .zip(&survives_64)
        .map(|(m, s)| (m - mm) * (s - ms))
        .sum::<f64>()
        / margins.len().max(1) as f64;
    let sm = (margins.iter().map(|m| (m - mm).powi(2)).sum::<f64>() / margins.len().max(1) as f64)
        .sqrt();
    let ss = (survives_64.iter().map(|s| (s - ms).powi(2)).sum::<f64>()
        / survives_64.len().max(1) as f64)
        .sqrt();
    let r = if sm > 1e-12 && ss > 1e-12 {
        cov / (sm * ss)
    } else {
        0.0
    };
    println!(
        "\ncorrelation(baseline_mean_top1_margin, survives_to_h64) = {r:.4} (n={n})\n  \
         POSITIVE and large would mean survival-to-64 is largely explained by the prompt's own \
         predictability (the templated-prompt confound, now measured continuously instead of \
         eyeballed) rather than a property of the ablation. Mean margin: {mm:.4} \
         (min={:.4}, max={:.4}).",
        margins.iter().cloned().fold(f64::INFINITY, f64::min),
        margins.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // ── KL drift among survivors ──
    println!("\nKL-bits at each horizon marker, among points still matching at that marker:");
    println!(
        "{:>8} {:>6} {:>10} {:>10}",
        "horizon", "n", "mean_kl", "median_kl"
    );
    for (mi, &marker) in HORIZON_MARKERS.iter().enumerate() {
        let vals: Vec<f32> = census.iter().filter_map(|p| p.kl_at_marker[mi]).collect();
        if vals.is_empty() {
            println!("{marker:>8} {:>6} {:>10} {:>10}", 0, "n/a", "n/a");
            continue;
        }
        let mv = vals.iter().sum::<f32>() / vals.len() as f32;
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        println!("{marker:>8} {:>6} {mv:>10.4} {median:>10.4}", vals.len());
    }

    // ── Raw per-point table ──
    println!(
        "\n{:<3} {:>4} {:>9} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "p", "pos", "margin", "first_div", "kl@6", "kl@16", "kl@32", "kl@64"
    );
    let fmt_kl = |k: Option<f32>| {
        k.map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "-".to_string())
    };
    for p in &census {
        let first_div = p
            .first_divergence
            .map(|d| d.to_string())
            .unwrap_or_else(|| "none".to_string());
        println!(
            "{:<3} {:>4} {:>9.4} {:>10} {:>9} {:>9} {:>9} {:>9}",
            p.prompt_idx,
            p.position,
            p.baseline_mean_margin,
            first_div,
            fmt_kl(p.kl_at_marker[0]),
            fmt_kl(p.kl_at_marker[1]),
            fmt_kl(p.kl_at_marker[2]),
            fmt_kl(p.kl_at_marker[3]),
        );
    }

    println!(
        "\nspace + guarantee: same single-ablation-in-time test as BW-C4 (remove all 4 real \
         top-routed experts at ONE late layer, one checkpoint), scaled to 20 prompts x 4 \
         positions with prompts deliberately chosen to avoid the templated-completion confound \
         BW-C4 found. Still oracle, not predictor; still one layer at a time; still greedy CPU \
         reference decode."
    );

    Ok(())
}
