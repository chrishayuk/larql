//! BW-C — whole-expert compute-skip oracle: does the generated trajectory
//! survive deleting ONE real MoE expert invocation?
//!
//! Registered in `docs/diagnoses/bw10-live-gate.md` as the sibling of
//! BW-A/BW-B/BW-B1 (all CLOSED 2026-08-14): "whole-expert compute-skip
//! oracle (skip the operation, not just bytes within it)". BW-B's whole
//! family (compact-dense, materialize break-even) answered "reduce the
//! representation of an operation". BW-C answers the other lever:
//! "delete the operation entirely" — no gather, no compact
//! materialization, no partial kernel; the entire weight movement for
//! one expert call disappears.
//!
//! Method, per the brief: oracle ONE expert invocation at a time on a
//! REAL serving trajectory (gpt-oss-20b, greedy decode), no predictor —
//! the target is picked from a REAL observed routing decision, not
//! guessed. Substitute: zero (== residual/identity pass-through for
//! GPT-OSS's un-normalised combine — see `larql_compute::cpu::ops::
//! moe::expert_override`'s module doc for the precise caveat about
//! renormalised top-k weights). Scored by TRAJECTORY preservation, not
//! local cosine: does the CONTINUATION's greedy token sequence, for M
//! tokens after the ablation point, match the unperturbed baseline —
//! not just whether the one perturbed layer's output vector looks
//! similar.
//!
//! Deliberately bounded: greedy (deterministic) decode only, a small
//! number of oracle targets sampled from one real routing trace, one
//! substitute type (zero/identity). Mean-response and previous-
//! invocation substitutes, and non-greedy/sampled trajectories, are
//! explicitly out of scope for this first pass.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc_expert_skip_oracle -- \
//!     --vindex /path/to/gpt-oss-20b-q4k.vindex [--continuation 16] [--targets 5]

use std::path::PathBuf;

use larql_compute::cpu::ops::moe::expert_override;
use larql_inference::ffn::LocalMoeFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = PathBuf::from(
        std::env::var("HOME").unwrap_or_default() + "/chris-models/gpt-oss-20b-q4k.vindex",
    );
    let mut continuation_len: usize = 16;
    let mut n_targets: usize = 5;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = PathBuf::from(&args[i]);
            }
            "--continuation" => {
                i += 1;
                continuation_len = args[i].parse().unwrap_or(16);
            }
            "--targets" => {
                i += 1;
                n_targets = args[i].parse().unwrap_or(5);
            }
            _ => {}
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!("vindex directory not found: {}", vindex_path.display());
        eprintln!("Usage: bwc_expert_skip_oracle --vindex PATH [--continuation N] [--targets N]");
        std::process::exit(1);
    }

    println!("=== BW-C: whole-expert compute-skip oracle ===\n");
    println!("vindex: {}", vindex_path.display());

    let mut cb = SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;

    if weights.arch.per_layer_input_gate_key(0).is_some() {
        return Err(
            "BW-C needs the resident in-process MoE path, which doesn't support \
                     Per-Layer-Embedding architectures"
                .into(),
        );
    }

    // Dequantize attention + dense FFN to f32, resident for the whole
    // run — matches `larql bench --cpu`'s in-process MoE path exactly
    // (`bench/local_moe_runtime.rs`), so this experiment measures the
    // same object that path serves.
    for layer in 0..weights.num_layers {
        larql_inference::vindex::insert_q4k_layer_tensors_resident(&mut weights, &index, layer)
            .map_err(|e| format!("failed to dequantize layer {layer} to f32: {e}"))?;
    }
    let weights_ref = &weights;
    let moe_ffn = LocalMoeFfn {
        weights: weights_ref,
        index: Some(&index),
    };

    let prompt = "The history of the Roman Empire began when";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!(
        "prompt: {:?} ({} tokens), continuation={continuation_len} tokens, targets={n_targets}\n",
        prompt,
        prompt_ids.len()
    );

    let build_engine = || {
        let kind = larql_kv::EngineKind::from_name("standard").expect("standard engine exists");
        kind.build(larql_inference::cpu_engine_backend())
    };

    // ── Baseline: real greedy decode, WITH observation on, so the
    // oracle targets below are real (layer, expert) pairs this exact
    // trajectory actually routed to — not guessed indices. ──
    expert_override::start_observing();
    let mut engine = build_engine();
    let mut baseline_tokens: Vec<u32> = Vec::with_capacity(continuation_len);
    let _ = larql_kv::generation::generate_with_engine_resident(
        &mut engine,
        weights_ref,
        &tokenizer,
        &moe_ffn,
        &index,
        &prompt_ids,
        continuation_len,
        |id, _tok| baseline_tokens.push(id),
    );
    let observed = expert_override::stop_observing();
    println!(
        "baseline: {} tokens generated, {} expert calls observed\n",
        baseline_tokens.len(),
        observed.len()
    );
    if baseline_tokens.len() < continuation_len {
        println!(
            "NOTE: baseline hit EOS or a decode failure early ({}/{} tokens) — the \
             continuation window is shorter than requested.\n",
            baseline_tokens.len(),
            continuation_len
        );
    }

    // ── Oracle targets: real (layer, expert) pairs from the observed
    // trace, sampled to span depth (no clever predictor — just spread
    // across the observed call sequence). ──
    let mut seen = std::collections::HashSet::new();
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for obs in &observed {
        if seen.insert((obs.layer, obs.expert)) {
            candidates.push((obs.layer, obs.expert));
        }
    }
    if candidates.is_empty() {
        return Err(
            "no expert calls observed — the resident MoE path may not be reaching \
                     cpu_moe_forward for this vindex"
                .into(),
        );
    }
    let stride = (candidates.len() / n_targets.max(1)).max(1);
    let targets: Vec<(usize, usize)> = candidates
        .iter()
        .step_by(stride)
        .copied()
        .take(n_targets)
        .collect();
    println!("oracle targets (layer, expert), spread across the observed trace: {targets:?}\n");

    // Approximate bytes ONE expert call touches — gate+up+down at
    // hidden/intermediate width, Q4_K's ~4.5 bits/weight. A theoretical
    // figure (uniform across experts/layers for this architecture), not
    // a measured one — BW-A/BW-B's `movement_ledger`/bump-site machinery
    // is the instrument for a load-bearing byte claim; this is a
    // disclosed approximation for context only.
    let approx_expert_bytes = {
        let bits_per_weight = 4.5_f64;
        let weights_per_expert =
            3.0 * weights_ref.hidden_size as f64 * weights_ref.intermediate_size as f64;
        (weights_per_expert * bits_per_weight / 8.0) as u64
    };
    println!(
        "approx bytes/expert call (Q4_K, hidden={} x inter={}, ~4.5 bits/weight): {approx_expert_bytes}\n",
        weights_ref.hidden_size, weights_ref.intermediate_size
    );

    // ── Per target: arm the ablation, re-run the SAME decode from the
    // SAME prompt (fresh — no KV-fork available on this path, but the
    // resident CPU decode is cheap enough that full re-decode is fine
    // for a bounded target count), compare the continuation. ──
    println!(
        "{:<8} {:<8} {:>10} {:>10} {:>14} {:>10}",
        "layer", "expert", "fired", "match_pos", "first_diverge", "approx_bytes"
    );
    println!("{}", "-".repeat(64));
    for &(layer, expert) in &targets {
        expert_override::disarm();
        expert_override::arm_once(layer, expert);
        let mut engine = build_engine();
        let mut perturbed_tokens: Vec<u32> = Vec::with_capacity(continuation_len);
        let _ = larql_kv::generation::generate_with_engine_resident(
            &mut engine,
            weights_ref,
            &tokenizer,
            &moe_ffn,
            &index,
            &prompt_ids,
            continuation_len,
            |id, _tok| perturbed_tokens.push(id),
        );
        let fired = expert_override::fired();
        expert_override::disarm();

        let n = baseline_tokens.len().min(perturbed_tokens.len());
        let matching = baseline_tokens[..n]
            .iter()
            .zip(&perturbed_tokens[..n])
            .take_while(|(a, b)| a == b)
            .count();
        let first_diverge = if matching < n {
            format!("{matching}")
        } else {
            "none".to_string()
        };

        println!(
            "{layer:<8} {expert:<8} {:>10} {matching:>10} {:>14} {approx_expert_bytes:>10}",
            fired, first_diverge,
        );
    }

    println!(
        "\nmatch_pos = N means the first N continuation tokens are IDENTICAL between baseline \
         and the ablated run (out of {} baseline tokens); first_diverge = the 0-indexed \
         position where they first differ, or \"none\" if the whole continuation matched.",
        baseline_tokens.len()
    );

    Ok(())
}
