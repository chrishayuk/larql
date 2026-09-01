//! **The decode-rate benchmark for a composed REPRESENT candidate.**
//!
//! The same two arms `q2a_teacher_forced` judges for QUALITY — the
//! source container's BF16 stack beside the candidate overlay's
//! compiled banks — timed instead of judged. Quality authority belongs
//! to the frozen contract at 8192 positions; this file answers the
//! other half of the economics: what the admitted map buys in decode
//! rate, on this machine, in this session.
//!
//! Protocol, per the repo's measured bench discipline:
//!
//! * **Both arms in one process, interleaved in blocks**, the order
//!   alternating block to block — the BF16 figure is re-earned beside
//!   the candidate, never compared against a number from another
//!   session or day (machine state shifts move e2e timings by ±6%).
//! * **Warm-up precedes measurement** (pipeline caches, page cache),
//!   and the per-arm result is the MINIMUM block time — the floor the
//!   machine demonstrated — with every block printed, not averaged
//!   away.
//! * Timing wraps ONLY [`HybridStack::forward`]: the same
//!   one-command-buffer-per-token step serving takes, logits readback
//!   included, instrumentation off (`forward`, not `forward_traced`).
//!
//! The candidate map is the compiled expert overlay plus whichever
//! TRANSIENT scopes the quality probe's own environment variables
//! name, so the arm that is timed is the arm that was judged:
//!
//! ```text
//! LARQL_KIMI_VINDEX3=~/chris-models/Kimi-Linear-48B-A3B-Instruct.lift2.vindex3 \
//! LARQL_KIMI_Q6_CANDIDATE=/tmp/kimi-map-l20-26q80-lift2.vindex3 \
//! LARQL_KIMI_QUALITY_BANK=~/chris-models/qbanks/kimi-quality-bank-256x32 \
//! LARQL_KDA_Q8_LAYER=20,21,22,24,25 LARQL_MLA_Q8_LAYER=23,26 LARQL_LMHEAD_Q8=1 \
//!   cargo test -p larql-vindex --features gpu --release --lib \
//!   q2a_decode_bench -- --nocapture
//! ```

use std::time::Instant;

use larql_compute::backend::ComputeBackend;
use larql_compute_metal::MetalBackend;
use serde_json::Value;

use super::kda_q8_real::{
    assemble, assemble_with_head, build_layers, build_layers_scoped, head_q8, layer_list,
    LAYER_ENV, MLA_ENV, SHARED_ENV,
};
use super::q2a_teacher_forced::{
    build_stack, env_dir, sequence_embeddings, BANK_ENV, CANDIDATE_ENV, SOURCE_ENV,
};
use crate::format::vindex3::opplan::exec::kimi_source::{CandidateOverlay, KimiSourceModel};
use crate::format::vindex3::opplan::exec::stack_metal::HybridStack;

/// Tokens per measured block. At the BF16 stack's known ~27 ms/token a
/// 128-token block is ~3.5 s — long enough to average submission
/// jitter, short enough that all blocks of both arms interleave inside
/// one machine state.
const TOKENS_ENV: &str = "LARQL_BENCH_TOKENS";
const TOKENS_DEFAULT: usize = 128;
/// Measured blocks per arm. Every block is printed; the summary takes
/// the minimum.
const BLOCKS_ENV: &str = "LARQL_BENCH_BLOCKS";
const BLOCKS_DEFAULT: usize = 4;
/// Unmeasured warm-up tokens per arm before any block.
const WARMUP_TOKENS: usize = 64;

fn env_count(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{var}={v} is not a count"))
        })
        .unwrap_or(default)
}

/// Feed `tokens` bank rows through one arm, resetting recurrent state
/// at each sequence boundary exactly as the quality runner does, and
/// return (wall seconds, device-reported GPU milliseconds).
fn run_tokens(
    metal: &MetalBackend,
    stack: &mut HybridStack<'_>,
    rows: &[Vec<Vec<f32>>],
    hidden: usize,
    tokens: usize,
) -> (f64, f64) {
    let positions = rows[0].len();
    let mut gpu_ms = 0.0;
    let t0 = Instant::now();
    for t in 0..tokens {
        let (seq, pos) = (t / positions % rows.len(), t % positions);
        if pos == 0 {
            stack.reset_states().expect("stack resets");
        }
        let (logits, _traces, timing) = stack
            .forward(metal, &rows[seq][pos], hidden)
            .expect("a decode step must not refuse");
        assert!(!logits.is_empty() && logits[0].is_finite());
        gpu_ms += timing.device_gpu_ms;
    }
    (t0.elapsed().as_secs_f64(), gpu_ms)
}

#[test]
fn decode_rate_of_the_candidate_map_beside_its_own_bf16_baseline() {
    let (Some(source_dir), Some(candidate_dir), Some(bank_dir)) = (
        env_dir(SOURCE_ENV),
        env_dir(CANDIDATE_ENV),
        env_dir(BANK_ENV),
    ) else {
        eprintln!("skipped: set {SOURCE_ENV}, {CANDIDATE_ENV} and {BANK_ENV}");
        return;
    };
    // Same refusal as the quality runner: the residency SET would wire
    // the ~94 GB expert bank past the wired-collector wall.
    if std::env::var("LARQL_RESIDENCY_SET").is_ok() {
        panic!("unset LARQL_RESIDENCY_SET: the bench must use implicit residency");
    }
    let Some(metal) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("MetalBackend::new() returned None on macOS — the shader library failed");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let tokens = env_count(TOKENS_ENV, TOKENS_DEFAULT);
    let blocks = env_count(BLOCKS_ENV, BLOCKS_DEFAULT);

    let manifest: Value =
        serde_json::from_slice(&std::fs::read(bank_dir.join("manifest.json")).expect("manifest"))
            .expect("bank manifest parses");
    let positions = manifest["positions"].as_u64().unwrap() as usize;
    let hidden = manifest["hidden"].as_u64().unwrap() as usize;

    let t0 = Instant::now();
    let model = KimiSourceModel::open(&source_dir).expect("source container opens");
    let g = model.geometry.clone();
    assert_eq!(g.hidden, hidden, "the bank was exported for this model");
    let overlay =
        CandidateOverlay::open(&candidate_dir, &source_dir, &g).expect("candidate overlay opens");
    let moe_layers: Vec<u32> = (g.dense_prefix_layers..g.num_layers)
        .map(|l| l as u32)
        .collect();
    model
        .register_stores(&metal, &moe_layers)
        .expect("stores register");
    overlay.register_store(&metal);
    // Optional TRANSIENT scopes beside the compiled expert overlay —
    // the same probe vocabulary the quality runner judges: KDA and MLA
    // projections, shared-expert branches, the output head. The device
    // executes real Q8_0 buffers through the real quantised kernels
    // either way, so the decode timing is real; only the STORAGE is
    // transient — which makes this a **Q8 EXECUTION PREVIEW**: valid
    // for decode compute economics and steady-state tok/s, NOT a
    // measurement of native artifact size, cold-load, disk traffic or
    // materialisation overhead.
    let kda_layers = layer_list(LAYER_ENV);
    let mla_layers = layer_list(MLA_ENV);
    let shared_layers = layer_list(SHARED_ENV);
    let q8_head = head_q8();
    // The head counts: a map whose only transient member is the head
    // would otherwise time two arms that differ in nothing.
    let transient =
        !kda_layers.is_empty() || !mla_layers.is_empty() || !shared_layers.is_empty() || q8_head;
    let (mut baseline, mut candidate) = if transient {
        let (base, _) = build_layers(&metal, &model, &[], None);
        let (cand, swapped) = build_layers_scoped(
            &metal,
            &model,
            &kda_layers,
            &mla_layers,
            &shared_layers,
            Some(&overlay),
        );
        assert_eq!(
            swapped.len(),
            kda_layers.len() + mla_layers.len() + shared_layers.len(),
            "every transient target re-encoded"
        );
        (
            assemble(&metal, &model, base),
            assemble_with_head(&metal, &model, cand, q8_head),
        )
    } else {
        (
            build_stack(&metal, &model, None),
            build_stack(&metal, &model, Some(&overlay)),
        )
    };
    metal.seal_weight_regions();
    let mut transient_scope = String::new();
    for (name, layers) in [
        ("KDA", &kda_layers),
        ("MLA", &mla_layers),
        ("shared", &shared_layers),
    ] {
        if !layers.is_empty() {
            transient_scope.push_str(&format!(" + TRANSIENT {name} Q8_0 at {layers:?}"));
        }
    }
    if q8_head {
        transient_scope.push_str(" + TRANSIENT head Q8_0");
    }
    if !transient_scope.is_empty() {
        transient_scope.push_str(" (Q8 execution preview)");
    }
    eprintln!(
        "[bench] both arms loaded in {:.1}s; candidate scope {}{transient_scope}",
        t0.elapsed().as_secs_f64(),
        overlay.scope(),
    );

    // Enough distinct rows that a block never re-times one hot row, cycled
    // deterministically.
    let seqs_needed = (tokens.max(WARMUP_TOKENS)).div_ceil(positions).max(1);
    let rows: Vec<Vec<Vec<f32>>> = (0..seqs_needed)
        .map(|s| sequence_embeddings(&bank_dir, s, positions, hidden))
        .collect();

    for (name, stack) in [("baseline", &mut baseline), ("candidate", &mut candidate)] {
        let (wall, _gpu) = run_tokens(&metal, stack, &rows, hidden, WARMUP_TOKENS);
        eprintln!("[bench] warm-up {name}: {WARMUP_TOKENS} tokens in {wall:.2}s (unmeasured)");
    }

    // Interleaved blocks, order alternating so a monotone machine-state
    // drift cannot systematically favour one arm.
    let mut wall_ms: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut gpu_ms: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for block in 0..blocks {
        let order: [usize; 2] = if block % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            let (name, stack) = match arm {
                0 => ("baseline", &mut baseline),
                _ => ("candidate", &mut candidate),
            };
            let (wall, gpu) = run_tokens(&metal, stack, &rows, hidden, tokens);
            let per_tok = wall * 1000.0 / tokens as f64;
            wall_ms[arm].push(per_tok);
            gpu_ms[arm].push(gpu / tokens as f64);
            eprintln!(
                "[bench] block {block} {name}: {per_tok:.2} ms/token = {:.2} tok/s \
                 (gpu {:.2} ms/token)",
                1000.0 / per_tok,
                gpu / tokens as f64,
            );
        }
    }

    let floor = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let (b_ms, c_ms) = (floor(&wall_ms[0]), floor(&wall_ms[1]));
    let (b_gpu, c_gpu) = (floor(&gpu_ms[0]), floor(&gpu_ms[1]));
    eprintln!(
        "[bench] FLOOR baseline {:.2} tok/s ({b_ms:.2} ms wall, {b_gpu:.2} ms gpu) | \
         candidate {:.2} tok/s ({c_ms:.2} ms wall, {c_gpu:.2} ms gpu) | speedup {:.3}x \
         over {blocks} blocks x {tokens} tokens/arm",
        1000.0 / b_ms,
        1000.0 / c_ms,
        b_ms / c_ms,
    );
}
