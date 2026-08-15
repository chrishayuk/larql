//! BW-C5.1 — multi-layer composability ladder: BW-C5 showed ONE late
//! layer's repeated skips mostly compose (88.7% skip rate, 75% full
//! 32-token fidelity). This tests the actual lever for a meaningful
//! aggregate reduction — MULTIPLE late layers skipping SIMULTANEOUSLY
//! within the same forward pass — as a controlled ladder (1 -> 2 -> 4
//! -> 8 candidate late layers), holding everything else (prompts,
//! window, generation length) fixed so the only changed variable is
//! how many layers participate. Also tests whether BW-C5's two
//! divergent cases were caused by oracle MYOPIA (a too-short
//! lookahead window) by re-running the single-layer rung at both a
//! 6-token and a 16-token lookahead on the SAME prompts.
//!
//! Method — per token, for the rung's K candidate layers: observe real
//! routing at all K fresh from the CURRENT (possibly already-modified)
//! state, compute ONE clean lookahead (shared across all K layers'
//! tests — the "no ablation at all" baseline doesn't depend on which
//! layer is being tested), then for EACH layer independently test
//! whether ablating JUST that layer's group (holding the other K-1
//! layers unablated) leaves the lookahead byte-identical to the clean
//! baseline. Whichever subset of the K layers pass their OWN
//! individual test get COMMITTED TOGETHER via `arm_multi` for this
//! token's real advance — this is what lets multiple layers' skips
//! compose within one forward pass, using BW-C5.1's `arm_multi`
//! generalisation of `expert_override` (up to `MAX_TARGETS` = 8
//! simultaneous layers).
//!
//! Deliberately bounded for a first pass: 2 prompts (reused from
//! BW-C5 — one with full fidelity, one that diverged, for direct
//! comparability), rungs [1, 2, 4, 8] at the production-realistic
//! 6-token lookahead, PLUS rung 1 also at a 16-token lookahead (the
//! myopia check — cheapest rung, and the one with BW-C5's own concrete
//! divergent examples to re-test). Testing every rung at both
//! lookaheads multiplies cost well past what's tractable in one pass —
//! a natural follow-up if this shows a real trend.
//!
//! The fidelity horizon is the one bound that is SELECTABLE rather than
//! fixed: `--generation-length` takes 32 (BW-C5's exact window, the
//! default, directly comparable) or 64 (BW-C4.5's horizon, where
//! individually-safe cases were still eroding — 82.5% pooled, 66.7% in
//! the most-contested tertile). 32 is roughly half the wall clock and
//! reports an UPPER BOUND on the 64-token fidelity count; the harness
//! prints that caveat itself whenever it ran at less than 64, so a
//! result cannot be quoted without it.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc5_1_multilayer_ladder -- \
//!     --vindex /path/to/gpt-oss-20b-q4k.vindex [--generation-length 32|64]

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use larql_compute::cpu::ops::moe::expert_override;
use larql_inference::ffn::LocalMoeFfn;
use larql_inference::kv_engine::PerLayerKvAccess;
use larql_inference::ModelWeights;
use larql_kv::engines::semantic_promotion::checkpoint::BoundaryCheckpoint;
use larql_kv::engines::semantic_promotion::ids::CheckpointId;
use larql_kv::AnyEngine;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

/// Reused from BW-C5: prompt 0 kept full 32-token fidelity at 96.9%
/// skip; prompt 1 diverged at position 7 despite 90.6% skip — the
/// single-layer myopia check re-tests THIS EXACT prompt at a longer
/// lookahead.
const PROMPTS: [&str; 2] = [
    "The history of the Roman Empire began when",
    "In the field of quantum computing, researchers recently discovered",
];
/// Matches BW-C5's window exactly, for direct comparability — and the
/// DEFAULT, not the only choice. BW-C4.5 found that even individually
/// "safe at 6" cases keep eroding out to 64 tokens (82.5% pooled, 66.7%
/// in the most-contested tertile), so a 32-token full-fidelity result
/// here is an upper bound on what a 64-token one would report. Pass
/// `--generation-length 64` to measure that directly; it costs roughly
/// twice the wall clock per config, which is why it is not the default.
const DEFAULT_GENERATION_LENGTH: usize = 32;
/// Generation lengths this harness reports against. Anything else is
/// refused rather than silently accepted: these two are the windows the
/// BW-C results are stated over, and a third would produce a number with
/// nothing to compare it to.
const SUPPORTED_GENERATION_LENGTHS: [usize; 2] = [32, 64];

/// Candidate late layers as fractions of num_layers, ORDERED so index
/// 0 is BW-C3/C4/C4.5/C5's already-characterized reference layer
/// (5/6) and each later entry extends outward across the rest of the
/// top third of the network. `RUNG_SIZES[i]` always uses the FIRST
/// `RUNG_SIZES[i]` fractions here — larger rungs are supersets of
/// smaller ones (adding layers, never resampling), so the ladder
/// isolates "how many layers participate" as the only variable.
const CANDIDATE_LATE_LAYER_FRACTIONS: [f64; 8] = [
    5.0 / 6.0,
    2.0 / 3.0,
    17.0 / 24.0,
    3.0 / 4.0,
    19.0 / 24.0,
    7.0 / 8.0,
    11.0 / 12.0,
    23.0 / 24.0,
];
const RUNG_SIZES: [usize; 4] = [1, 2, 4, 8];
/// (rung_size, lookahead_window) configurations actually tested. Every
/// rung at the production-realistic window (6), plus the cheapest rung
/// (1) ALSO at a longer window (16) — the myopia check.
const CONFIGS: [(usize, usize); 5] = [(1, 6), (1, 16), (2, 6), (4, 6), (8, 6)];

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

struct ConfigResult {
    prompt_idx: usize,
    rung_size: usize,
    lookahead: usize,
    /// Total (layer, token) opportunities tested — `rung_size *
    /// generation length` minus any refused (malformed routing).
    opportunities: usize,
    skipped: usize,
    first_divergence: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = PathBuf::from(
        std::env::var("HOME").unwrap_or_default() + "/chris-models/gpt-oss-20b-q4k.vindex",
    );
    let args: Vec<String> = std::env::args().collect();
    let mut gen_len = DEFAULT_GENERATION_LENGTH;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" {
            i += 1;
            vindex_path = PathBuf::from(&args[i]);
        } else if args[i] == "--generation-length" {
            i += 1;
            gen_len = args[i].parse().unwrap_or(DEFAULT_GENERATION_LENGTH);
        }
        i += 1;
    }
    if !SUPPORTED_GENERATION_LENGTHS.contains(&gen_len) {
        eprintln!(
            "--generation-length must be one of {SUPPORTED_GENERATION_LENGTHS:?}; got {gen_len}"
        );
        std::process::exit(1);
    }
    if !vindex_path.is_dir() {
        eprintln!("vindex directory not found: {}", vindex_path.display());
        std::process::exit(1);
    }

    println!("=== BW-C5.1: multi-layer composability ladder ===\n");
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
    let candidate_layers: Vec<usize> = CANDIDATE_LATE_LAYER_FRACTIONS
        .iter()
        .map(|f| ((*f * num_layers as f64) as usize).min(num_layers - 1))
        .collect();
    println!("candidate late layers (nested, index 0 = BW-C5's reference): {candidate_layers:?}");
    println!(
        "rungs tested: {RUNG_SIZES:?}, configs (rung_size, lookahead): {CONFIGS:?}, \
         generation length: {gen_len}\n"
    );

    if expert_override::MAX_TARGETS < *RUNG_SIZES.iter().max().unwrap() {
        return Err(format!(
            "expert_override::MAX_TARGETS ({}) is smaller than the largest rung ({}) — \
             raise MAX_TARGETS before running the full ladder",
            expert_override::MAX_TARGETS,
            RUNG_SIZES.iter().max().unwrap()
        )
        .into());
    }

    let mut results: Vec<ConfigResult> = Vec::new();

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
        let start_token = {
            let logits = larql_inference::research::hidden_to_raw_logits(weights_ref, &h0);
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };

        let kv = per_layer_kv(&mut engine).ok_or("per_layer_kv_mut returned None")?;
        let start_ckpt = BoundaryCheckpoint::capture(
            CheckpointId::from_counter((prompt_idx * 100_000) as u128),
            kv,
        )
        .map_err(|e| format!("capture failed: {e:?}"))?;

        // Canonical baseline: zero ablations, computed ONCE per
        // prompt, reused across every (rung, lookahead) config below.
        let canonical_start = std::time::Instant::now();
        let canonical_tokens = decode_n(
            &mut engine,
            weights_ref,
            &moe_ffn,
            &index,
            start_token,
            gen_len,
        );
        restore(&mut engine, &start_ckpt)?;
        println!(
            "  canonical baseline decoded in {:.1}s",
            canonical_start.elapsed().as_secs_f64()
        );
        std::io::stdout().flush().ok();

        for &(rung_size, lookahead) in &CONFIGS {
            let rung_layers: Vec<usize> = candidate_layers[..rung_size].to_vec();
            let config_start = std::time::Instant::now();
            println!("  rung={rung_size} lookahead={lookahead}: starting {gen_len} tokens...");
            std::io::stdout().flush().ok();

            let mut current = start_token;
            let mut repeated_policy_tokens: Vec<u32> = Vec::with_capacity(gen_len);
            let mut opportunities = 0usize;
            let mut skipped = 0usize;

            for token_idx in 0..gen_len {
                let kv = per_layer_kv(&mut engine).ok_or("per_layer_kv_mut returned None")?;
                let step_ckpt = BoundaryCheckpoint::capture(
                    CheckpointId::from_counter(
                        (prompt_idx * 100_000
                            + rung_size * 1000
                            + lookahead * 100
                            + 1
                            + repeated_policy_tokens.len()) as u128,
                    ),
                    kv,
                )
                .map_err(|e| format!("capture failed: {e:?}"))?;

                expert_override::start_observing();
                let _ = step(&mut engine, weights_ref, &moe_ffn, &index, current);
                let observed = expert_override::stop_observing();
                restore(&mut engine, &step_ckpt)?;

                let mut experts_at: HashMap<usize, Vec<usize>> = HashMap::new();
                for &layer in &rung_layers {
                    let at_layer: Vec<usize> = observed
                        .iter()
                        .filter(|obs| obs.layer == layer)
                        .map(|obs| obs.expert)
                        .collect();
                    if at_layer.len() == 4 {
                        experts_at.insert(layer, at_layer);
                    }
                }
                opportunities += experts_at.len();

                // Shared clean lookahead — independent of which layer
                // is being tested, computed once per token.
                let clean_lookahead = decode_n(
                    &mut engine,
                    weights_ref,
                    &moe_ffn,
                    &index,
                    current,
                    lookahead,
                );
                restore(&mut engine, &step_ckpt)?;

                // Test each candidate layer's group INDEPENDENTLY
                // (holding the others un-ablated), collect the subset
                // that individually passed.
                let mut safe_targets: Vec<(usize, Vec<usize>)> = Vec::new();
                for (&layer, layer_experts) in &experts_at {
                    expert_override::arm_set(layer, layer_experts);
                    let ablated_lookahead = decode_n(
                        &mut engine,
                        weights_ref,
                        &moe_ffn,
                        &index,
                        current,
                        lookahead,
                    );
                    let expected: u64 = layer_experts.iter().fold(0u64, |m, &e| m | (1u64 << e));
                    let fired_ok = expert_override::fired_mask() == expected;
                    expert_override::disarm();
                    restore(&mut engine, &step_ckpt)?;

                    if fired_ok && ablated_lookahead == clean_lookahead {
                        safe_targets.push((layer, layer_experts.clone()));
                    }
                }

                if safe_targets.is_empty() {
                    current = step(&mut engine, weights_ref, &moe_ffn, &index, current);
                } else {
                    let arm_targets: Vec<(usize, &[usize])> = safe_targets
                        .iter()
                        .map(|(l, e)| (*l, e.as_slice()))
                        .collect();
                    expert_override::arm_multi(&arm_targets);
                    current = step(&mut engine, weights_ref, &moe_ffn, &index, current);
                    expert_override::disarm();
                    skipped += safe_targets.len();
                }
                repeated_policy_tokens.push(current);

                println!(
                    "    token {}/{gen_len}: skipped_so_far={skipped} \
                     opportunities_so_far={opportunities} elapsed={:.1}s",
                    token_idx + 1,
                    config_start.elapsed().as_secs_f64()
                );
                std::io::stdout().flush().ok();
            }

            let first_divergence = canonical_tokens
                .iter()
                .zip(&repeated_policy_tokens)
                .position(|(a, b)| a != b);

            println!(
                "  rung={rung_size} lookahead={lookahead} DONE in {:.1}s: opportunities={opportunities} \
                 skipped={skipped} ({:.1}%) first_divergence={}",
                config_start.elapsed().as_secs_f64(),
                100.0 * skipped as f64 / opportunities.max(1) as f64,
                first_divergence
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );

            results.push(ConfigResult {
                prompt_idx,
                rung_size,
                lookahead,
                opportunities,
                skipped,
                first_divergence,
            });

            restore(&mut engine, &start_ckpt)?;
        }
    }

    println!("\n{:=<80}", "");
    println!(
        "{:<3} {:>5} {:>10} {:>14} {:>10} {:>8} {:>12}",
        "p", "rung", "lookahead", "opportunities", "skipped", "skip%", "first_div"
    );
    for r in &results {
        println!(
            "{:<3} {:>5} {:>10} {:>14} {:>10} {:>7.1}% {:>12}",
            r.prompt_idx,
            r.rung_size,
            r.lookahead,
            r.opportunities,
            r.skipped,
            100.0 * r.skipped as f64 / r.opportunities.max(1) as f64,
            r.first_divergence
                .map(|d| d.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
    }

    println!("\n=== ladder summary (lookahead=6, pooled across prompts) ===");
    for &rung in &RUNG_SIZES {
        let at_rung: Vec<&ConfigResult> = results
            .iter()
            .filter(|r| r.rung_size == rung && r.lookahead == 6)
            .collect();
        let total_opp: usize = at_rung.iter().map(|r| r.opportunities).sum();
        let total_skip: usize = at_rung.iter().map(|r| r.skipped).sum();
        let full_fidelity = at_rung
            .iter()
            .filter(|r| r.first_divergence.is_none())
            .count();
        println!(
            "  rung={rung}: skip_rate={total_skip}/{total_opp} ({:.1}%)  \
             full_fidelity={full_fidelity}/{}",
            100.0 * total_skip as f64 / total_opp.max(1) as f64,
            at_rung.len()
        );
    }

    println!("\n=== myopia check: rung=1, lookahead 6 vs 16 (same prompts) ===");
    for prompt_idx in 0..PROMPTS.len() {
        let w6 = results
            .iter()
            .find(|r| r.prompt_idx == prompt_idx && r.rung_size == 1 && r.lookahead == 6);
        let w16 = results
            .iter()
            .find(|r| r.prompt_idx == prompt_idx && r.rung_size == 1 && r.lookahead == 16);
        if let (Some(a), Some(b)) = (w6, w16) {
            println!(
                "  prompt {prompt_idx}: lookahead=6 first_div={} vs lookahead=16 first_div={}",
                a.first_divergence
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "none".into()),
                b.first_divergence
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "none".into()),
            );
        }
    }

    println!(
        "\ninterpretation guide: if skip_rate stays high and full_fidelity stays high as rung \
         grows, layers compose almost independently — a strong result. If full_fidelity drops \
         sharply as rung grows (even with a high skip_rate), individually-safe layers are NOT \
         jointly safe — composition across LAYERS fails even where composition across TIME \
         (BW-C5, one layer) held. For the myopia check: if lookahead=16 fixes a divergence that \
         lookahead=6 didn't, the failure is oracle myopia, not cumulative perturbation — a \
         better pre-execution predictor could plausibly recover it; if lookahead=16 does NOT \
         fix it, the perturbation itself is the problem, and no amount of one-step lookahead \
         will save that policy."
    );
    println!(
        "\nspace + guarantee: per-layer decisions are tested INDEPENDENTLY (each candidate \
         layer's lookahead holds every OTHER candidate layer un-ablated), then the individually-\
         passing subset is committed TOGETHER via arm_multi — this is NOT a joint search over \
         all 2^k subsets (that's BW-C3's exhaustive design, intractable to repeat at every \
         token of a real generation); a genuinely joint-optimal subset could differ from what \
         this greedy per-layer-then-commit policy finds. Still oracle, not predictor; still \
         greedy/local in TIME (BW-C5's caveat) as well as now in LAYER SELECTION; still strict \
         exact-match only. 2 prompts — exploratory scope, not a powered census."
    );
    if gen_len < 64 {
        println!(
            "\nFIDELITY HORIZON: this run measured {gen_len}-token fidelity. BW-C4.5 found \
             individually-safe cases keep eroding out to 64 (82.5% pooled, 66.7% in the \
             most-contested tertile), so every full_fidelity count above is an UPPER BOUND on \
             the 64-token one. Re-run with --generation-length 64 before quoting a durable \
             composition result."
        );
    }

    Ok(())
}
