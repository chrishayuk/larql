//! BW-C4 — horizon survival: does "safe at 6 tokens" (BW-C3) mean the
//! whole-group deletion's perturbation was truly ABSORBED, or has it
//! merely not surfaced yet? Extends BW-C3's most provocative claim —
//! deleting the ENTIRE real top-4 routing at one layer/position — from
//! a 6-token exact-match window out to 64, and retains KL/logit
//! divergence at every horizon marker even where the greedy token
//! still matches, to separate absorbed perturbation from hidden
//! probability drift that hasn't crossed an argmax boundary yet.
//!
//! Method: re-derives BW-C3's 72 (checkpoint, layer) points with the
//! IDENTICAL prompts/positions/depths, but tests only ONE ablation per
//! point — `arm_set(layer, ALL 4 real top-routed experts)` — instead
//! of the full 15-subset search. `min_suff=0` (BW-C3) is exactly
//! equivalent to "removing all 4 is safe at 6 tokens" (there is only
//! one size-4 removed-set), so re-deriving this classification here
//! and cross-checking the count against BW-C3's own 48/72 is a free
//! consistency check that both runs are measuring the same thing.
//! Points that reproduce as safe-at-6 get decoded to
//! `N_HORIZON` (64) tokens for both the clean baseline and the
//! all-4-ablated run, from the SAME checkpoint.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc4_horizon_survival -- \
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

/// Identical to BW-C3's set — this is a re-derivation of the SAME
/// points, not a new sample, so the safe-at-6 subset must reproduce
/// BW-C3's 48/72 exactly.
const PROMPTS: [&str; 6] = [
    "The history of the Roman Empire began when",
    "In the field of quantum computing, researchers recently discovered",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:",
    "The recipe calls for two cups of flour, one teaspoon of",
    "Dear Sir or Madam, I am writing to formally request",
    "The three primary colors are red, blue, and",
];
const CHECKPOINT_STEPS: [usize; 4] = [0, 4, 8, 12];
const LAYER_FRACTIONS: [f64; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];

/// How far to decode past the intervention — BW-C3's window (6) is
/// the first marker; the rest are the horizon extension.
const N_HORIZON: usize = 64;
/// Survival/KL-drift checkpoints along the horizon. 6 must match
/// BW-C3's window exactly (it's the re-derivation gate); the rest are
/// the new question.
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

/// Decode `n` steps, returning every token and the logits at each
/// position in `save_logits_at` (1-indexed step number, matching
/// `HORIZON_MARKERS`) — NOT every step's logits, to keep memory
/// bounded (vocab-sized vectors × 64 steps × 2 sequences × 72 points
/// would be wasteful when only the marker positions are ever read).
fn decode_n_with_markers(
    engine: &mut AnyEngine,
    weights: &ModelWeights,
    ffn: &LocalMoeFfn,
    index: &VectorIndex,
    mut current: u32,
    n: usize,
    save_logits_at: &[usize],
) -> (Vec<u32>, std::collections::HashMap<usize, Vec<f32>>) {
    let mut tokens = Vec::with_capacity(n);
    let mut marker_logits = std::collections::HashMap::new();
    for i in 1..=n {
        let (tok, logits) = step(engine, weights, ffn, index, current);
        if save_logits_at.contains(&i) {
            marker_logits.insert(i, logits);
        }
        tokens.push(tok);
        current = tok;
    }
    (tokens, marker_logits)
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

struct HorizonPoint {
    prompt_idx: usize,
    position: usize,
    layer: usize,
    /// 0-indexed position of the first token mismatch, or `None` if
    /// `baseline_tokens[..N_HORIZON] == ablated_tokens[..N_HORIZON]`
    /// held all the way out.
    first_divergence: Option<usize>,
    /// KL(baseline || ablated) at each horizon marker where tokens
    /// were STILL matching at that marker — `None` once the tokens
    /// have already diverged before that marker (KL there is just
    /// restating a divergence already visible in the token stream,
    /// not hidden drift).
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

    println!("=== BW-C4: horizon survival of whole-group deletion ===\n");
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

    let mut total_points = 0usize;
    let mut safe_at_6_points: Vec<HorizonPoint> = Vec::new();
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

                let (baseline_tokens, baseline_markers) = decode_n_with_markers(
                    &mut engine,
                    weights_ref,
                    &moe_ffn,
                    &index,
                    current,
                    N_HORIZON,
                    &HORIZON_MARKERS,
                );
                restore(&mut engine, &ckpt)?;

                for &layer in &target_layers {
                    total_points += 1;
                    let at_layer: Vec<usize> = observed
                        .iter()
                        .filter(|obs| obs.layer == layer)
                        .map(|obs| obs.expert)
                        .collect();
                    if at_layer.len() != 4 {
                        skipped_not_four_experts += 1;
                        continue;
                    }
                    let expected_fired: u64 = at_layer.iter().fold(0u64, |m, &e| m | (1u64 << e));

                    expert_override::arm_set(layer, &at_layer);
                    let (ablated_tokens, ablated_markers) = decode_n_with_markers(
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
                        continue;
                    }

                    // BW-C3 re-derivation gate: min_suff=0 there is
                    // EXACTLY "removing all 4 was safe at 6 tokens" —
                    // there is only one size-4 removed-set.
                    let safe_at_6 = baseline_tokens[..6] == ablated_tokens[..6];
                    if !safe_at_6 {
                        continue;
                    }

                    let first_divergence = baseline_tokens
                        .iter()
                        .zip(&ablated_tokens)
                        .position(|(a, b)| a != b);

                    let mut kl_at_marker = [None; 4];
                    for (mi, &marker) in HORIZON_MARKERS.iter().enumerate() {
                        let still_matching = first_divergence.is_none_or(|d| d >= marker);
                        if still_matching {
                            if let (Some(bl), Some(al)) =
                                (baseline_markers.get(&marker), ablated_markers.get(&marker))
                            {
                                kl_at_marker[mi] = Some(softmax_bits_kl(bl, al));
                            }
                        }
                    }

                    safe_at_6_points.push(HorizonPoint {
                        prompt_idx,
                        position: step_idx,
                        layer,
                        first_divergence,
                        kl_at_marker,
                    });
                }
                println!(
                    "  checkpoint step={step_idx}: {}/{} points safe-at-6 so far",
                    safe_at_6_points.len(),
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
        "\ntotal (checkpoint, layer) points: {total_points}, safe-at-6 (BW-C3's min_suff=0 \
         set, re-derived): {}",
        safe_at_6_points.len()
    );
    println!(
        "skipped (fired_mask mismatch): {skipped_mismatched_fire}, skipped (top-4 didn't yield \
         4 distinct experts): {skipped_not_four_experts}"
    );
    println!(
        "consistency check against BW-C3: expect 48/72 (adjust for any skips above) — got \
         {}/{total_points}",
        safe_at_6_points.len()
    );

    // ── Raw per-point arms, before any aggregate — read these before
    // the verdict lines below. ──
    println!(
        "\n{:<3} {:>4} {:>6} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "p", "pos", "layer", "first_div", "kl@6", "kl@16", "kl@32", "kl@64"
    );
    let fmt_kl = |k: Option<f32>| {
        k.map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "-".to_string())
    };
    for p in &safe_at_6_points {
        let first_div = p
            .first_divergence
            .map(|d| d.to_string())
            .unwrap_or_else(|| "none".to_string());
        println!(
            "{:<3} {:>4} {:>6} {:>10} {:>9} {:>9} {:>9} {:>9}",
            p.prompt_idx,
            p.position,
            p.layer,
            first_div,
            fmt_kl(p.kl_at_marker[0]),
            fmt_kl(p.kl_at_marker[1]),
            fmt_kl(p.kl_at_marker[2]),
            fmt_kl(p.kl_at_marker[3]),
        );
    }

    // ── Survival curve ──
    let n = safe_at_6_points.len().max(1);
    println!(
        "\n{:=<70}\nsurvival curve (of the safe-at-6 set, n={n})",
        ""
    );
    println!("{:>8} {:>10} {:>8}", "horizon", "survived", "%");
    for &marker in &HORIZON_MARKERS {
        let survived = safe_at_6_points
            .iter()
            .filter(|p| p.first_divergence.is_none_or(|d| d >= marker))
            .count();
        println!(
            "{marker:>8} {survived:>10} {:>7.1}%",
            100.0 * survived as f64 / n as f64
        );
    }

    // ── Survival curve, split by depth ──
    println!("\nsurvival curve split by depth (early / mid / late = layers {target_layers:?}):");
    let depth_names = ["early", "mid", "late"];
    for (depth_idx, &layer) in target_layers.iter().enumerate() {
        let at_depth: Vec<&HorizonPoint> = safe_at_6_points
            .iter()
            .filter(|p| p.layer == layer)
            .collect();
        let dn = at_depth.len().max(1);
        print!(
            "  {:<6} (n={:>3}): ",
            depth_names[depth_idx],
            at_depth.len()
        );
        for &marker in &HORIZON_MARKERS {
            let survived = at_depth
                .iter()
                .filter(|p| p.first_divergence.is_none_or(|d| d >= marker))
                .count();
            print!("h{marker}={:>5.1}%  ", 100.0 * survived as f64 / dn as f64);
        }
        println!();
    }

    // ── KL drift beneath a still-matching surface ──
    println!(
        "\nKL-bits at each horizon marker, AMONG POINTS STILL MATCHING at that marker (rising \
         KL despite identical tokens = hidden probability drift that hasn't crossed an argmax \
         boundary yet):"
    );
    println!(
        "{:>8} {:>6} {:>10} {:>10} {:>10}",
        "horizon", "n", "mean_kl", "median_kl", "max_kl"
    );
    for (mi, &marker) in HORIZON_MARKERS.iter().enumerate() {
        let vals: Vec<f32> = safe_at_6_points
            .iter()
            .filter_map(|p| p.kl_at_marker[mi])
            .collect();
        if vals.is_empty() {
            println!(
                "{marker:>8} {:>6} {:>10} {:>10} {:>10}",
                0, "n/a", "n/a", "n/a"
            );
            continue;
        }
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let max = sorted.last().copied().unwrap_or(0.0);
        println!(
            "{marker:>8} {:>6} {mean:>10.4} {median:>10.4} {max:>10.4}",
            vals.len()
        );
    }

    println!(
        "\ninterpretation guide: if the survival % collapses sharply from horizon 6 to 32/64, \
         BW-C3's 'safe' cases were mostly delayed rather than absorbed. If a large fraction \
         survives all the way to 64, whole-group deletion is genuinely durable, not just \
         locally invisible. Rising KL among survivors at later horizons — even where the \
         survival % stays high — would mean the perturbation is real and growing but hasn't \
         yet flipped an argmax; falling/flat KL among survivors is the stronger case for \
         real absorption."
    );

    println!(
        "\nspace + guarantee: this tests ONE ablation (all 4 real top-routed experts, one \
         layer, one checkpoint) extended in TIME, not a new search — every scope caveat from \
         BW-C3 (this checkpoint's real top-4 only, one layer at a time, oracle not predictor) \
         still applies. A survival curve here is NOT a production decode-length guarantee: it \
         is measured on ONE prompt continuation per point, greedy, CPU reference path."
    );

    Ok(())
}
