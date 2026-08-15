//! BW-C5 — oracle repeated-policy ceiling: BW-C1-C4.5 all tested ONE
//! deletion at a time against the untouched canonical trajectory.
//! That answers "is this opportunity real" but not "do opportunities
//! COMPOSE" — if expert group A is safe to skip when B/C/D still run,
//! that says nothing about whether A and B are BOTH safe to skip
//! simultaneously, because skipping A changes the state B's own
//! safety was evaluated against. This is the actual inference
//! question: `canonical state -> one deletion -> observe` (BW-C3/C4)
//! becomes `modified state -> next decision -> maybe delete again ->
//! modified state -> ...` (BW-C5).
//!
//! Method — STRICT oracle, single late layer (the one BW-C4.5
//! characterized), greedy/unrestricted: for each token of a REAL
//! generation, capture the CURRENT actual state (which may already
//! reflect earlier skips in this same generation), observe the real
//! top-4 routing fresh at this point, and test via a short
//! (`LOOKAHEAD_WINDOW`) lookahead whether ablating the whole group
//! HERE leaves that window byte-identical to not ablating. If yes,
//! COMMIT the skip and advance the real trajectory from the ablated
//! state (composability); if no, advance normally. At the end,
//! compare the accumulated real trajectory against a SEPARATE,
//! never-ablated canonical decode of the same length from the same
//! prompt — this is the number that actually answers "does locally-
//! invisible compose into globally-invisible".
//!
//! Deliberately narrow for a first pass, per the standing ladder:
//! ONE late layer (not a multi-layer/percentage-capped policy — that
//! is the natural next increment if this shows real signal), STRICT
//! exact-match safety (no KL/quality threshold — an unambiguous upper
//! bound, not muddied by a quality metric), greedy/unrestricted
//! (skip whenever locally safe, no cap).
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bwc5_oracle_repeated_policy -- \
//!     --vindex /path/to/gpt-oss-20b-q4k.vindex [--emit-trace <dir>]
//!
//! `--emit-trace <dir>` writes one replayable trace per prompt (see
//! `larql_compute::exec_policy::trace`), which
//! `LARQL_EXEC_POLICY=trace:<file>` then replays on the Metal serve
//! path — the bridge from this offline oracle to a bytes-and-latency
//! A/B. Match the prompt: a trace replayed against a different prompt
//! addresses a trajectory that does not exist.

use std::path::PathBuf;

use larql_compute::cpu::ops::moe::expert_override;
use larql_inference::ffn::LocalMoeFfn;
use larql_inference::kv_engine::PerLayerKvAccess;
use larql_inference::ModelWeights;
use larql_kv::engines::semantic_promotion::checkpoint::BoundaryCheckpoint;
use larql_kv::engines::semantic_promotion::ids::CheckpointId;
use larql_kv::AnyEngine;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

/// 8 of BW-C4.5's 20 open-ended prompts — a representative subset,
/// not re-selected for margin (this experiment's question is
/// composability, not re-litigating the entropy confound).
const PROMPTS: [&str; 8] = [
    "The history of the Roman Empire began when",
    "In the field of quantum computing, researchers recently discovered",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:",
    "The detective walked into the room and immediately noticed",
    "Scientists have long debated whether",
    "The economic policy sparked controversy because",
    "According to the latest research on climate change,",
    "The novel's protagonist decided to",
];
/// Real generation length under the repeated policy. Trimmed from an
/// initially-planned 48 after the smoke test showed this harness's
/// per-token cost is higher than BW-C1-C4.5's (a fresh checkpoint
/// capture/restore triplet every token here, vs one capture reused
/// across many tests there) — 32 keeps the full run inside the
/// session's established ~30-40 min per-census budget.
const REAL_GENERATION_LENGTH: usize = 32;
/// Local safety test window at each decision point — same convention
/// as BW-C1-C4.5's minimal window, kept identical for comparability.
const LOOKAHEAD_WINDOW: usize = 6;
const LATE_LAYER_FRACTION: f64 = 5.0 / 6.0;

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

struct PromptResult {
    prompt_idx: usize,
    opportunities: usize,
    skipped: usize,
    /// Position where the repeated-policy trajectory first differs
    /// from the never-ablated canonical trajectory, or `None` if all
    /// `REAL_GENERATION_LENGTH` tokens matched.
    first_divergence: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = PathBuf::from(
        std::env::var("HOME").unwrap_or_default() + "/chris-models/gpt-oss-20b-q4k.vindex",
    );
    let args: Vec<String> = std::env::args().collect();
    let mut emit_trace_dir: Option<PathBuf> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" {
            i += 1;
            vindex_path = PathBuf::from(&args[i]);
        } else if args[i] == "--emit-trace" {
            i += 1;
            emit_trace_dir = Some(PathBuf::from(&args[i]));
        }
        i += 1;
    }
    if let Some(dir) = &emit_trace_dir {
        std::fs::create_dir_all(dir)?;
    }
    if !vindex_path.is_dir() {
        eprintln!("vindex directory not found: {}", vindex_path.display());
        std::process::exit(1);
    }

    println!("=== BW-C5: oracle repeated-policy ceiling (strict, single late layer) ===\n");
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
    println!("late layer: {late_layer} (of {num_layers})");
    println!(
        "real generation length: {REAL_GENERATION_LENGTH}, lookahead window: {LOOKAHEAD_WINDOW}\n"
    );

    // Approx bytes for the WHOLE 4-expert group at this layer's shape
    // — same disclosed-approximation formula as BW-C's original
    // harness (Q4_K, ~4.5 bits/weight), scaled by 4 since a successful
    // skip here removes the entire top-4 group, not one expert.
    let approx_bytes_per_group = {
        let bits_per_weight = 4.5_f64;
        let weights_per_expert =
            3.0 * weights_ref.hidden_size as f64 * weights_ref.intermediate_size as f64;
        (4.0 * weights_per_expert * bits_per_weight / 8.0) as u64
    };
    println!("approx bytes/successful skip (4-expert group, Q4_K, disclosed approximation): {approx_bytes_per_group}\n");

    let mut results: Vec<PromptResult> = Vec::new();
    let mut skip_reasons_mismatched_fire = 0usize;
    let mut skip_reasons_not_four_experts = 0usize;

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

        // Checkpoint right after prefill — the shared starting point
        // for BOTH the canonical baseline and the repeated-policy run,
        // so they are the same prompt/position, not just the same text.
        let kv = per_layer_kv(&mut engine).ok_or("per_layer_kv_mut returned None")?;
        let start_ckpt = BoundaryCheckpoint::capture(
            CheckpointId::from_counter((prompt_idx * 10_000) as u128),
            kv,
        )
        .map_err(|e| format!("capture failed: {e:?}"))?;

        // ── Canonical baseline: zero ablations, ever. ──
        let canonical_tokens = decode_n(
            &mut engine,
            weights_ref,
            &moe_ffn,
            &index,
            start_token,
            REAL_GENERATION_LENGTH,
        );
        restore(&mut engine, &start_ckpt)?;

        // ── Repeated-policy run: at every step, test local safety
        // FROM THE CURRENT ACTUAL STATE (which may already include
        // earlier skips), commit if safe, advance either way. ──
        let mut current = start_token;
        let mut repeated_policy_tokens: Vec<u32> = Vec::with_capacity(REAL_GENERATION_LENGTH);
        let mut opportunities = 0usize;
        let mut skipped = 0usize;

        // One trace per prompt: `TraceReplay` addresses (layer, decode
        // step) within ONE generation, so a trace is only meaningful
        // replayed against the prompt that produced it.
        let mut trace = larql_compute::exec_policy::trace::Trace::new(format!(
            "bwc5_oracle_repeated_policy layer={late_layer} lookahead={LOOKAHEAD_WINDOW} \
             generation_length={REAL_GENERATION_LENGTH} prompt_idx={prompt_idx} \
             prompt={:?} | REPLAY PRECONDITIONS: same prompt, greedy decode, \
             deterministic engine; decode step 0 is the FIRST GENERATED token (prefill \
             positions carry their own phase index and are never skipped). Safety was \
             established on the CPU resident decode path — the serve path's routing may \
             differ in fp provenance, so the SKIP DECISIONS transfer exactly but their \
             safety verdict does not.",
            PROMPTS[prompt_idx]
        ));

        for token_idx in 0..REAL_GENERATION_LENGTH {
            let kv = per_layer_kv(&mut engine).ok_or("per_layer_kv_mut returned None")?;
            let step_ckpt = BoundaryCheckpoint::capture(
                CheckpointId::from_counter(
                    (prompt_idx * 10_000 + 1 + repeated_policy_tokens.len()) as u128,
                ),
                kv,
            )
            .map_err(|e| format!("capture failed: {e:?}"))?;

            expert_override::start_observing();
            let _ = step(&mut engine, weights_ref, &moe_ffn, &index, current);
            let observed = expert_override::stop_observing();
            restore(&mut engine, &step_ckpt)?;

            let at_layer: Vec<usize> = observed
                .iter()
                .filter(|obs| obs.layer == late_layer)
                .map(|obs| obs.expert)
                .collect();

            let mut committed = false;
            if at_layer.len() == 4 {
                opportunities += 1;
                let expected_fired: u64 = at_layer.iter().fold(0u64, |m, &e| m | (1u64 << e));

                let baseline_lookahead = decode_n(
                    &mut engine,
                    weights_ref,
                    &moe_ffn,
                    &index,
                    current,
                    LOOKAHEAD_WINDOW,
                );
                restore(&mut engine, &step_ckpt)?;

                expert_override::arm_set(late_layer, &at_layer);
                let ablated_lookahead = decode_n(
                    &mut engine,
                    weights_ref,
                    &moe_ffn,
                    &index,
                    current,
                    LOOKAHEAD_WINDOW,
                );
                let fired_mask = expert_override::fired_mask();
                expert_override::disarm();
                restore(&mut engine, &step_ckpt)?;

                if fired_mask != expected_fired {
                    skip_reasons_mismatched_fire += 1;
                } else {
                    let locally_safe = baseline_lookahead == ablated_lookahead;
                    if locally_safe {
                        // Commit the skip: re-arm (the test above
                        // already consumed the one-shot) and take
                        // exactly ONE real step from the checkpoint —
                        // this is what advances the REAL trajectory
                        // with the ablation baked in, composability
                        // intact for the next iteration.
                        expert_override::arm_set(late_layer, &at_layer);
                        current = step(&mut engine, weights_ref, &moe_ffn, &index, current);
                        expert_override::disarm();
                        skipped += 1;
                        committed = true;
                        trace.record(late_layer, token_idx as u64);
                    }
                }
            } else {
                skip_reasons_not_four_experts += 1;
            }

            if !committed {
                // Not skipped (unsafe, mismatched, or not exactly 4
                // experts) — advance normally from the SAME checkpoint.
                current = step(&mut engine, weights_ref, &moe_ffn, &index, current);
            }
            repeated_policy_tokens.push(current);
        }

        let first_divergence = canonical_tokens
            .iter()
            .zip(&repeated_policy_tokens)
            .position(|(a, b)| a != b);

        println!(
            "  opportunities={opportunities} skipped={skipped} ({:.1}%) first_divergence={}",
            100.0 * skipped as f64 / opportunities.max(1) as f64,
            first_divergence
                .map(|d| d.to_string())
                .unwrap_or_else(|| "none".to_string())
        );

        if let Some(dir) = &emit_trace_dir {
            // The fidelity verdict goes in the header, not just the log:
            // a trace whose source run DIVERGED is still replayable, and
            // an operator picking a file out of a directory must be able
            // to see which ones held parity without cross-referencing.
            let fidelity = match first_divergence {
                None => format!("FULL {REAL_GENERATION_LENGTH}-token parity"),
                Some(d) => format!("DIVERGED at token {d}"),
            };
            let mut trace = trace;
            trace.source = trace
                .source
                .map(|s| format!("{s} | skipped={skipped}/{opportunities} fidelity={fidelity}"));
            let path = dir.join(format!("bwc5-prompt{prompt_idx}.trace"));
            trace.write(&path)?;
            println!(
                "  trace written: {} ({} skips)",
                path.display(),
                trace.len()
            );
        }

        results.push(PromptResult {
            prompt_idx,
            opportunities,
            skipped,
            first_divergence,
        });
    }

    // ── Aggregate ──
    let total_opportunities: usize = results.iter().map(|r| r.opportunities).sum();
    let total_skipped: usize = results.iter().map(|r| r.skipped).sum();
    let fully_preserved = results
        .iter()
        .filter(|r| r.first_divergence.is_none())
        .count();

    println!("\n{:=<70}", "");
    println!(
        "aggregate skip rate: {total_skipped}/{total_opportunities} = {:.1}%",
        100.0 * total_skipped as f64 / total_opportunities.max(1) as f64
    );
    println!(
        "prompts with FULL {REAL_GENERATION_LENGTH}-token global fidelity vs canonical: \
         {fully_preserved}/{} ({:.1}%)",
        results.len(),
        100.0 * fully_preserved as f64 / results.len().max(1) as f64
    );
    println!(
        "approx bytes avoided (total across all prompts): {}",
        total_skipped as u64 * approx_bytes_per_group
    );
    println!(
        "skip attempts refused (fired_mask mismatch): {skip_reasons_mismatched_fire}, refused \
         (top-4 didn't yield 4 distinct experts): {skip_reasons_not_four_experts}"
    );

    println!(
        "\n{:<3} {:>14} {:>10} {:>8} {:>12}",
        "p", "opportunities", "skipped", "skip%", "first_div"
    );
    for r in &results {
        println!(
            "{:<3} {:>14} {:>10} {:>7.1}% {:>12}",
            r.prompt_idx,
            r.opportunities,
            r.skipped,
            100.0 * r.skipped as f64 / r.opportunities.max(1) as f64,
            r.first_divergence
                .map(|d| d.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
    }

    println!(
        "\ninterpretation guide: the skip rate is what the strict local-safety test allows; \
         'full global fidelity' is the number that actually matters — a HIGH skip rate with \
         LOW global fidelity means locally-invisible skips do NOT reliably compose (each one \
         individually undetectable, but their combination drifts the trajectory). A high skip \
         rate WITH high full-fidelity would be the genuinely strong result: opportunities \
         compose almost for free."
    );
    println!(
        "\nspace + guarantee: this is a GREEDY, MYOPIC, LOCAL-window ({LOOKAHEAD_WINDOW}-token) \
         oracle — 'locally safe' is judged against the CURRENT (possibly already-modified) \
         state, not against the original canonical trajectory. It is an upper bound under this \
         specific greedy policy, not a global optimum (a different sequence of skip/no-skip \
         decisions might do better or worse) and not yet a KL/quality-thresholded policy \
         (strict exact-match only, by design — see the module doc). Single late layer only; \
         multi-layer or percentage-capped policies are the natural next increment."
    );

    Ok(())
}
