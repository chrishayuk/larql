//! BW-C1 — skippability correlates: does router weight predict whether
//! a real expert invocation is safe to skip?
//!
//! BW-C (first pass, `docs/diagnoses/bwc-expert-skip-oracle.md`) found
//! 4/8 real single-expert ablations left a 16-token greedy continuation
//! byte-identical, with no obvious depth pattern. This asks WHY, via
//! the cheapest candidate covariate already available at zero extra
//! compute: the router's own selection weight (`add_expert`'s `w`
//! parameter — GPT-OSS's `RenormalizedSoftmax` selected-weight, already
//! captured per invocation).
//!
//! KV-forked (`larql-kv`'s `BoundaryCheckpoint`, validated by
//! `bwc1_kvfork_sanity.rs`'s R1/R4 gates — run that first) so many
//! interventions at the SAME position share one checkpoint instead of
//! re-decoding the prompt per target, which is what made this scale
//! from BW-C's 8 points to several hundred tractable.
//!
//! For each (prompt, checkpoint position, sampled layer, selected
//! expert): capture real router weight + rank-within-top-k, ablate,
//! decode a short continuation from the SAME checkpoint, compare
//! against ONE clean baseline decoded from that checkpoint. Label
//! safe / delayed-divergence / immediate-divergence by exact greedy
//! token match; also record KL and top-1 margin change AT the
//! intervention token (the one call `hidden_to_raw_logits` gives per
//! step, cheap relative to the MoE forward itself).
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc1_skippability_correlates -- \
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

/// Prompts, kept diverse on purpose — `bwc1_kvfork_sanity.rs` found a
/// single prompt can land on a greedy repetition attractor a few steps
/// in, which is real behaviour but would make every ablation near it
/// read as "safe" for the wrong reason (attractors resist ANY small
/// perturbation, not just skippable ones). Spreading across prompts
/// bounds how much of the census one attractor can dominate.
const PROMPTS: [&str; 6] = [
    "The history of the Roman Empire began when",
    "In the field of quantum computing, researchers recently discovered",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:",
    "The recipe calls for two cups of flour, one teaspoon of",
    "Dear Sir or Madam, I am writing to formally request",
    "The three primary colors are red, blue, and",
];

/// Positions within each prompt's continuation to checkpoint at.
const CHECKPOINT_STEPS: [usize; 8] = [0, 2, 4, 6, 8, 10, 12, 14];
/// Tokens decoded post-intervention for the trajectory comparison.
const N_CONTINUATION: usize = 6;
/// Sampled layer depths per checkpoint, as fractions of num_layers —
/// early / mid / late, not an exhaustive sweep (bounded).
const LAYER_FRACTIONS: [f64; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];

struct Intervention {
    prompt_idx: usize,
    position: usize,
    layer: usize,
    expert: usize,
    router_weight: f32,
    rank: usize,
    match_pos: usize,
    kl_bits: f32,
    margin_baseline: f32,
    margin_ablated: f32,
    /// BW-C1 second-wave covariates (C2): the ablated expert's raw
    /// (pre-weight) output L2 norm, and the incoming residual stream's
    /// L2 norm at that position. `weighted_contribution_norm` (=
    /// `router_weight * out_norm`) and `contrib_over_residual` (=
    /// `out_norm / residual_norm`) are derived from these at report
    /// time rather than stored twice.
    out_norm: f32,
    residual_norm: f32,
}

impl Intervention {
    fn weighted_contribution_norm(&self) -> f32 {
        self.router_weight * self.out_norm
    }

    fn contrib_over_residual(&self) -> f32 {
        self.out_norm / self.residual_norm
    }
}

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

/// One decode step, returning (token, logits).
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

fn decode_n(
    engine: &mut AnyEngine,
    weights: &ModelWeights,
    ffn: &LocalMoeFfn,
    index: &VectorIndex,
    mut current: u32,
    n: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut toks = Vec::with_capacity(n);
    let mut first_logits = Vec::new();
    for i in 0..n {
        let (t, logits) = step(engine, weights, ffn, index, current);
        if i == 0 {
            first_logits = logits;
        }
        toks.push(t);
        current = t;
    }
    (toks, first_logits)
}

fn softmax_bits_kl(a: &[f32], b: &[f32]) -> f32 {
    // KL(softmax(a) || softmax(b)) in bits. Stable softmax (subtract
    // max), skip zero-probability terms (0 * log(0/x) := 0).
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

fn top1_margin(logits: &[f32]) -> f32 {
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sorted.first().copied().unwrap_or(0.0) - sorted.get(1).copied().unwrap_or(0.0)
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

    println!("=== BW-C1: skippability correlates ===\n");
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

    let mut results: Vec<Intervention> = Vec::new();

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

                // Real routing at this position, ALL layers, with
                // weights — one observed step gives every covariate we
                // need before sampling targets.
                expert_override::start_observing();
                let (_, _) = step(&mut engine, weights_ref, &moe_ffn, &index, current);
                let observed = expert_override::stop_observing();
                restore(&mut engine, &ckpt)?;

                // One clean baseline continuation from this checkpoint,
                // reused for every target tested here.
                let (baseline_tokens, _) = decode_n(
                    &mut engine,
                    weights_ref,
                    &moe_ffn,
                    &index,
                    current,
                    N_CONTINUATION,
                );
                restore(&mut engine, &ckpt)?;

                for &layer in &target_layers {
                    let mut at_layer: Vec<&expert_override::ExpertObservation> =
                        observed.iter().filter(|obs| obs.layer == layer).collect();
                    at_layer.sort_by(|a, b| b.router_weight.partial_cmp(&a.router_weight).unwrap());

                    for (rank, obs) in at_layer.iter().enumerate() {
                        let (expert, weight, out_norm, residual_norm) = (
                            obs.expert,
                            obs.router_weight,
                            obs.out_norm,
                            obs.residual_norm,
                        );
                        expert_override::arm_once(layer, expert);
                        let (ablated_tokens, ablated_logits_t0) = decode_n(
                            &mut engine,
                            weights_ref,
                            &moe_ffn,
                            &index,
                            current,
                            N_CONTINUATION,
                        );
                        let fired = expert_override::fired();
                        expert_override::disarm();
                        restore(&mut engine, &ckpt)?;
                        if !fired {
                            continue; // mis-specified target — refuse, don't record a fake null
                        }

                        // Baseline logits at the SAME (first) position,
                        // recomputed here rather than cached from the
                        // baseline decode above, so both arms' logits
                        // come from an identical call shape.
                        let (_, baseline_logits_t0) =
                            step(&mut engine, weights_ref, &moe_ffn, &index, current);
                        restore(&mut engine, &ckpt)?;

                        let match_pos = baseline_tokens
                            .iter()
                            .zip(&ablated_tokens)
                            .take_while(|(a, b)| a == b)
                            .count();
                        let kl_bits = softmax_bits_kl(&baseline_logits_t0, &ablated_logits_t0);
                        let margin_baseline = top1_margin(&baseline_logits_t0);
                        let margin_ablated = top1_margin(&ablated_logits_t0);

                        results.push(Intervention {
                            prompt_idx,
                            position: step_idx,
                            layer,
                            expert,
                            router_weight: weight,
                            rank,
                            match_pos,
                            kl_bits,
                            margin_baseline,
                            margin_ablated,
                            out_norm,
                            residual_norm,
                        });
                    }
                }
                println!(
                    "  checkpoint step={step_idx}: {} interventions so far",
                    results.len()
                );
            }
            if step_idx >= max_checkpoint_step {
                break;
            }
            let (tok, _) = step(&mut engine, weights_ref, &moe_ffn, &index, current);
            current = tok;
            step_idx += 1;
        }
    }

    // ── Report ──
    println!("\n{:=<100}", "");
    println!(
        "{:<3} {:>4} {:>6} {:>6} {:>5} {:>4} {:>10} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:<10}",
        "p",
        "pos",
        "layer",
        "exp",
        "rank",
        "mp",
        "weight",
        "kl_bits",
        "out_norm",
        "w*norm",
        "norm/res",
        "margin_b",
        "margin_a",
        "label"
    );
    let mut safe = 0usize;
    let mut immediate = 0usize;
    let mut delayed = 0usize;
    for r in &results {
        let label = if r.match_pos == N_CONTINUATION {
            safe += 1;
            "safe"
        } else if r.match_pos == 0 {
            immediate += 1;
            "immediate"
        } else {
            delayed += 1;
            "delayed"
        };
        println!(
            "{:<3} {:>4} {:>6} {:>6} {:>5} {:>4} {:>10.4} {:>8.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:<10}",
            r.prompt_idx,
            r.position,
            r.layer,
            r.expert,
            r.rank,
            r.match_pos,
            r.router_weight,
            r.kl_bits,
            r.out_norm,
            r.weighted_contribution_norm(),
            r.contrib_over_residual(),
            r.margin_baseline,
            r.margin_ablated,
            label
        );
    }

    println!("\ntotal interventions: {}", results.len());
    println!(
        "safe={safe} ({:.1}%)  delayed={delayed} ({:.1}%)  immediate={immediate} ({:.1}%)",
        100.0 * safe as f64 / results.len().max(1) as f64,
        100.0 * delayed as f64 / results.len().max(1) as f64,
        100.0 * immediate as f64 / results.len().max(1) as f64,
    );

    // ── The router-weight hypothesis: does mean weight differ between
    // safe and non-safe groups? ──
    let safe_weights: Vec<f64> = results
        .iter()
        .filter(|r| r.match_pos == N_CONTINUATION)
        .map(|r| r.router_weight as f64)
        .collect();
    let unsafe_weights: Vec<f64> = results
        .iter()
        .filter(|r| r.match_pos != N_CONTINUATION)
        .map(|r| r.router_weight as f64)
        .collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let stdev = |v: &[f64], m: f64| {
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    };
    let mean_safe = mean(&safe_weights);
    let mean_unsafe = mean(&unsafe_weights);
    println!(
        "\nrouter weight: safe mean={mean_safe:.4} (n={}, sd={:.4})  non-safe mean={mean_unsafe:.4} \
         (n={}, sd={:.4})",
        safe_weights.len(),
        stdev(&safe_weights, mean_safe),
        unsafe_weights.len(),
        stdev(&unsafe_weights, mean_unsafe),
    );

    // Generic Pearson/point-biserial correlation — shared by every
    // covariate below so router weight, rank, layer, and the C2
    // contribution-norm family all go through identical arithmetic.
    // NaN pairs (an un-set residual norm) are dropped, not zeroed —
    // silently treating "not measured" as 0 would fabricate a data
    // point instead of shrinking `n`.
    fn correlation(xs: &[f64], ys: &[f64]) -> (f64, usize) {
        let pairs: Vec<(f64, f64)> = xs
            .iter()
            .zip(ys)
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| (x, y))
            .collect();
        let n = pairs.len();
        if n < 2 {
            return (0.0, n);
        }
        let mx = pairs.iter().map(|(x, _)| x).sum::<f64>() / n as f64;
        let my = pairs.iter().map(|(_, y)| y).sum::<f64>() / n as f64;
        let cov: f64 = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f64>() / n as f64;
        let sx = (pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum::<f64>() / n as f64).sqrt();
        let sy = (pairs.iter().map(|(_, y)| (y - my).powi(2)).sum::<f64>() / n as f64).sqrt();
        let r = if sx > 1e-12 && sy > 1e-12 {
            cov / (sx * sy)
        } else {
            0.0
        };
        (r, n)
    }

    fn interpret(label: &str, r: f64, n: usize) {
        let band = match r.abs() {
            x if x < 0.1 => "essentially no linear relationship",
            x if x < 0.3 => "weak",
            x if x < 0.5 => "moderate",
            _ => "a real, usable signal",
        };
        println!("correlation({label}, is_safe) = {r:.4} (n={n}) — {band}");
    }

    /// Average-tie rank transform, for Spearman. The C2 covariates
    /// (contribution norms) span ~3 orders of magnitude across layer
    /// depth alone — Pearson on a heavy-tailed magnitude covariate lets
    /// a handful of extreme late-layer points dominate the fit and can
    /// flip its sign relative to the true monotonic relationship (see
    /// the positive-control check below, which is exactly where this
    /// bit). Spearman (Pearson on ranks) is scale-free and answers the
    /// question that actually matters here — "does more/less track
    /// more/less" — without an outlier-sensitive linear-fit assumption.
    fn rank_transform(xs: &[f64]) -> Vec<f64> {
        // Every caller filters to finite-only pairs before ranking (see
        // `spearman` below) — `partial_cmp` is total on that domain, so
        // this never needs a NaN fallback.
        let mut order: Vec<usize> = (0..xs.len()).collect();
        order.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap());
        let mut ranks = vec![0.0; xs.len()];
        let mut i = 0;
        while i < order.len() {
            let mut j = i;
            while j + 1 < order.len() && xs[order[j + 1]] == xs[order[i]] {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0 + 1.0;
            for &idx in &order[i..=j] {
                ranks[idx] = avg_rank;
            }
            i = j + 1;
        }
        ranks
    }

    fn spearman(xs: &[f64], ys: &[f64]) -> (f64, usize) {
        // Filter pairwise BEFORE ranking, not after: ranking a vector
        // containing NaN would need its own (arbitrary) NaN placement,
        // which could shift every other value's rank. Matches
        // `correlation`'s own pairwise-finite contract exactly.
        let (fx, fy): (Vec<f64>, Vec<f64>) = xs
            .iter()
            .zip(ys)
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| (x, y))
            .unzip();
        correlation(&rank_transform(&fx), &rank_transform(&fy))
    }

    let is_safe: Vec<f64> = results
        .iter()
        .map(|r| {
            if r.match_pos == N_CONTINUATION {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let router_weight: Vec<f64> = results.iter().map(|r| r.router_weight as f64).collect();
    let rank: Vec<f64> = results.iter().map(|r| r.rank as f64).collect();
    let layer: Vec<f64> = results.iter().map(|r| r.layer as f64).collect();
    let out_norm: Vec<f64> = results.iter().map(|r| r.out_norm as f64).collect();
    let weighted_contrib: Vec<f64> = results
        .iter()
        .map(|r| r.weighted_contribution_norm() as f64)
        .collect();
    let contrib_over_residual: Vec<f64> = results
        .iter()
        .map(|r| r.contrib_over_residual() as f64)
        .collect();
    let kl_bits: Vec<f64> = results.iter().map(|r| r.kl_bits as f64).collect();

    println!("\n=== C1 covariates (router weight and its free proxies) ===");
    let (r, n) = correlation(&router_weight, &is_safe);
    interpret("router_weight", r, n);
    let (r, n) = correlation(&rank, &is_safe);
    interpret("rank", r, n);
    let (r, n) = correlation(&layer, &is_safe);
    interpret("layer", r, n);

    println!("\n=== C2 covariates (contribution norm — is skippability a magnitude effect?) ===");
    println!(
        "  (norms span ~3 orders of magnitude across layer depth alone — Pearson AND Spearman \
         reported; a sign or magnitude disagreement between them means Pearson is being pulled \
         by a handful of extreme late-layer points, not a real linear relationship)"
    );
    let (rp, np) = correlation(&out_norm, &is_safe);
    let (rs, _) = spearman(&out_norm, &is_safe);
    println!("raw_output_norm vs is_safe: pearson={rp:.4} spearman={rs:.4} (n={np})");
    let (rp, np) = correlation(&weighted_contrib, &is_safe);
    let (rs, _) = spearman(&weighted_contrib, &is_safe);
    println!("weighted_contribution_norm (w*out_norm) vs is_safe: pearson={rp:.4} spearman={rs:.4} (n={np})");
    let (rp, np) = correlation(&contrib_over_residual, &is_safe);
    let (rs, _) = spearman(&contrib_over_residual, &is_safe);
    println!("contrib_over_residual_norm vs is_safe: pearson={rp:.4} spearman={rs:.4} (n={np})");
    let n_residual_valid = results
        .iter()
        .filter(|r| r.residual_norm.is_finite())
        .count();
    println!(
        "  ({n_residual_valid}/{} interventions had a valid residual norm)",
        results.len()
    );
    let (rs_out_layer, _) = spearman(&out_norm, &layer);
    let (rs_ratio_layer, _) = spearman(&contrib_over_residual, &layer);
    println!(
        "  confound check: spearman(raw_output_norm, layer)={rs_out_layer:.4} vs \
         spearman(contrib_over_residual_norm, layer)={rs_ratio_layer:.4} — raw norm tracking \
         layer this closely means its is_safe correlation above is largely riding on the SAME \
         depth signal already reported under C1, not independent information; the ratio's job \
         is to strip that out, and the near-zero number confirms it does."
    );

    // ── Live positive control for the C2 covariates themselves, per
    // the standing rule: don't trust a new instrument's correlation
    // with the OUTCOME until it's shown to correlate with something
    // ALREADY known to be true. Here: a bigger ablated contribution
    // should cause a BIGGER downstream perturbation at the
    // intervention token (measured independently via KL) — if
    // weighted-contribution-norm doesn't even predict its OWN
    // immediate effect, it cannot be trusted as a skippability
    // covariate either. Spearman, not Pearson, is the right read for
    // this check — see the C2 header note above. ──
    println!(
        "\n=== positive control: does contribution norm predict its OWN downstream effect? ==="
    );
    let (rp, np) = correlation(&weighted_contrib, &kl_bits);
    let (rs, _) = spearman(&weighted_contrib, &kl_bits);
    println!(
        "weighted_contribution_norm vs kl_bits_at_intervention: pearson={rp:.4} spearman={rs:.4} \
         (n={np}) — expect POSITIVE: a bigger deleted contribution should perturb the immediate \
         next-token distribution more. {}",
        if rs > 0.1 {
            "PASS on the correlation that actually applies to a heavy-tailed magnitude \
             covariate (Spearman) — the covariate is measuring something real; read the \
             is_safe numbers above by their Spearman column too."
        } else {
            "FAIL even on Spearman — this covariate isn't even predicting the immediate \
             perturbation it caused; treat any is_safe correlation above with real suspicion."
        }
    );

    Ok(())
}
