//! The first real decode timing — deliberately taken AFTER correctness
//! closed (`generate_real.rs`), not before, and deliberately taken
//! BEFORE any performance work, per the user's own instruction: "record
//! the baseline immediately... that ugly baseline is hugely valuable
//! because every later optimization can be measured against the exact
//! fully-verified implementation."
//!
//! Reuses the SAME fixture `generate_real.rs` already verified correct
//! — no new export, no new weights, just a stopwatch around the exact
//! code path that just proved itself. `exec::timing`'s `OpClass::{Kda,
//! Mla, MoeRouter, MoeRoutedExpert, MoeSharedExpert, LmHead}` (new this
//! rung, additive — `Norm`/`Residual`/`Embed` already existed and are
//! reused unchanged) are wired into the composition files
//! (`kimi_kda_layer.rs`/`kimi_mla_layer.rs`/`kimi_moe_block.rs`/
//! `token.rs`) at their outer call sites — coarse, "KDA total" rather
//! than KDA's own internal sub-breakdown, matching what was asked for
//! and avoiding re-opening the already-proven recurrence.
//!
//! **Prefill here is NOT batched** — a real, load-bearing observation,
//! not a limitation of this benchmark: `token_forward`/`stack_forward`
//! only ever process one position per call (the interface every
//! correctness rung above this one relies on), so even the 3-token
//! prompt runs as three sequential single-token passes, exactly like
//! decode. That gap — batching the prefill — is itself one of the
//! findings this baseline exists to surface, not something this file
//! works around.
//!
//! Not measured: allocation counts (`if easy` in the user's own
//! request — a global allocator wrapper is a crate-wide, always-on
//! change, not a scoped one, so it is out of scope for a single
//! baseline run and left as a stated gap rather than silently skipped).
//!
//! Prints a report; does not assert performance thresholds — a
//! regression gate on a number this file is establishing FOR THE FIRST
//! TIME would be circular. `#[ignore]`d by default: run explicitly.
//!
//! ```text
//! LARQL_KIMI_GENERATE_FIXTURE=/tmp/kimi_generate_fixture \
//!     cargo test -p larql-vindex --lib generate_baseline --release -- --ignored --nocapture
//! ```

use std::time::{Duration, Instant};

use larql_models::config::{KdaGeometry, MlaGeometry};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::stack::{LayerSpec, LayerState, LoadedExpert};
use crate::format::vindex3::opplan::exec::timing::{ledger, OpClass};
use crate::format::vindex3::opplan::exec::token::{token_forward, EmbeddingRow};

use super::stack_real::{expert_list_for, load_real_layer, read_f32, spec};

const FIXTURE_ENV: &str = "LARQL_KIMI_GENERATE_FIXTURE";
/// One expert's three projections, BF16 (P4a) — `3 * hidden * inter * 2`
/// bytes, computed from the fixture's own declared widths rather than a
/// remembered constant, so this number tracks the checkpoint it is
/// actually reading.
fn expert_bytes(hidden: usize, inter: usize) -> usize {
    3 * hidden * inter * 2
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[test]
#[ignore = "prints a timing report against the largest real-weight fixture; run explicitly"]
fn first_real_decode_baseline() {
    let Some(dir) = std::env::var_os(FIXTURE_ENV).map(std::path::PathBuf::from) else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");

    let hidden = manifest["hidden"].as_u64().unwrap() as usize;
    let eps = manifest["rms_eps"].as_f64().unwrap();
    let kda_geometry = KdaGeometry {
        num_heads: manifest["kda_num_heads"].as_u64().unwrap() as usize,
        head_dim: manifest["kda_head_dim"].as_u64().unwrap() as usize,
        conv_kernel: manifest["kda_conv_kernel"].as_u64().unwrap() as usize,
    };
    let mla_geometry = MlaGeometry {
        num_heads: manifest["mla_num_heads"].as_u64().unwrap() as usize,
        kv_lora_rank: manifest["mla_kv_lora_rank"].as_u64().unwrap() as usize,
        qk_nope_head_dim: manifest["mla_qk_nope_head_dim"].as_u64().unwrap() as usize,
        qk_rope_head_dim: manifest["mla_qk_rope_head_dim"].as_u64().unwrap() as usize,
        v_head_dim: manifest["mla_v_head_dim"].as_u64().unwrap() as usize,
    };
    let mla_kv_a_norm_eps = manifest["mla_kv_a_norm_eps"].as_f64().unwrap();
    let experts = manifest["experts"].as_u64().unwrap() as usize;
    let top_k = manifest["top_k"].as_u64().unwrap() as usize;
    let moe_intermediate = manifest["moe_intermediate_size"].as_u64().unwrap() as usize;
    let dense_intermediate = manifest["dense_intermediate_size"].as_u64().unwrap() as usize;
    let renormalize = manifest["moe_renormalize"].as_bool().unwrap();
    let branch_scale = manifest["routed_scaling_factor"].as_f64().unwrap();
    let num_layers = manifest["num_layers"].as_u64().unwrap() as usize;
    let vocab_size = manifest["vocab_size"].as_u64().unwrap() as usize;
    let dense_layer_count = manifest["dense_layers"].as_array().unwrap().len();
    let moe_layer_count = num_layers - dense_layer_count;
    let prompt_tokens: Vec<usize> = manifest["prompt_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let token_ids: Vec<usize> = manifest["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    eprintln!(
        "  loading {num_layers} real layers from {}...",
        dir.display()
    );
    let load_start = Instant::now();
    let layers: Vec<_> = (0..num_layers)
        .map(|i| load_real_layer(&dir, i, &manifest))
        .collect();
    let expert_lists: Vec<Vec<LoadedExpert<'_>>> = layers.iter().map(expert_list_for).collect();
    let specs: Vec<LayerSpec<'_>> = layers
        .iter()
        .zip(&expert_lists)
        .map(|(l, loaded)| {
            spec(
                l,
                kda_geometry,
                mla_geometry,
                mla_kv_a_norm_eps,
                experts,
                top_k,
                moe_intermediate,
                dense_intermediate,
                renormalize,
                branch_scale,
                eps,
                loaded,
            )
        })
        .collect();
    let mut states: Vec<LayerState> = layers
        .iter()
        .map(|l| {
            if l.kind == "kda" {
                LayerState::Kda(crate::format::vindex3::opplan::exec::kda::KdaState::zeros(
                    kda_geometry,
                ))
            } else {
                LayerState::Mla(crate::format::vindex3::opplan::exec::mla::MlaState::empty())
            }
        })
        .collect();
    let embedding_rows_owned: Vec<(usize, Vec<f32>)> = token_ids
        .iter()
        .enumerate()
        .map(|(p, &id)| (id, read_f32(&dir, &format!("embedding_{p}"))))
        .collect();
    let embedding_rows: Vec<EmbeddingRow<'_>> = embedding_rows_owned
        .iter()
        .map(|(id, vector)| EmbeddingRow { id: *id, vector })
        .collect();
    let final_norm_weight = read_f32(&dir, "final_norm_weight");
    let lm_head_weight = read_f32(&dir, "lm_head_weight");
    eprintln!(
        "  loaded in {:.1}s (not part of the decode measurement)",
        load_start.elapsed().as_secs_f64()
    );

    // ── Prefill: the 3-token prompt, one position at a time (see this
    // file's own doc comment — this is not batched, and that is a real
    // finding, not a shortcut) ──
    ledger().reset();
    let prefill_start = Instant::now();
    for &token_id in &prompt_tokens {
        std::hint::black_box(token_forward(
            token_id,
            hidden,
            &embedding_rows,
            &specs,
            &mut states,
            &final_norm_weight,
            eps,
            &lm_head_weight,
            vocab_size,
        ));
    }
    let prefill_wall = prefill_start.elapsed();

    // ── Decode: 16 positions, one at a time, timed individually ──
    ledger().reset();
    let mut step_wall: Vec<Duration> = Vec::with_capacity(16);
    let decode_start = Instant::now();
    for &token_id in &token_ids[prompt_tokens.len()..] {
        let t0 = Instant::now();
        std::hint::black_box(token_forward(
            token_id,
            hidden,
            &embedding_rows,
            &specs,
            &mut states,
            &final_norm_weight,
            eps,
            &lm_head_weight,
            vocab_size,
        ));
        step_wall.push(t0.elapsed());
    }
    let decode_wall = decode_start.elapsed();

    let mut sorted_steps = step_wall.clone();
    sorted_steps.sort();
    let first_token = step_wall[0];
    let steady = &step_wall[1..]; // tokens 2..16
    let steady_mean = steady.iter().sum::<Duration>().as_secs_f64() / steady.len().max(1) as f64;

    // ── Physical: from the fixture's OWN per-position routing record,
    // no re-derivation ──
    let mut naive_loads_per_token = 0usize;
    let mut union_across_decode = 0usize;
    for l in manifest["layers"].as_array().unwrap() {
        let Some(per_pos) = l.get("selected_ids_per_position").and_then(Value::as_array) else {
            continue; // dense layer 0, no routing
        };
        naive_loads_per_token += top_k;
        let mut seen = std::collections::BTreeSet::new();
        for row in &per_pos[prompt_tokens.len()..] {
            for id in row.as_array().unwrap() {
                seen.insert(id.as_u64().unwrap());
            }
        }
        union_across_decode += seen.len();
    }
    let bytes_per_expert = expert_bytes(hidden, moe_intermediate);

    println!("\n=== P3d-m baseline: first real decode timing, fully-verified implementation ===\n");
    println!("prompt/prefill:");
    println!(
        "  total time      {:>10.2} ms",
        prefill_wall.as_secs_f64() * 1e3
    );
    println!("  positions       {:>10}", prompt_tokens.len());
    println!(
        "  tok/s           {:>10.2}   (NOT batched — {} sequential single-token passes)",
        prompt_tokens.len() as f64 / prefill_wall.as_secs_f64(),
        prompt_tokens.len()
    );
    println!("\ndecode (16 tokens):");
    println!(
        "  first-token latency {:>8.2} ms",
        first_token.as_secs_f64() * 1e3
    );
    println!("  mean tok/s (2..16)  {:>8.2}", 1.0 / steady_mean);
    println!(
        "  p50 token latency   {:>8.2} ms",
        percentile(&sorted_steps, 0.50).as_secs_f64() * 1e3
    );
    println!(
        "  p95 token latency   {:>8.2} ms",
        percentile(&sorted_steps, 0.95).as_secs_f64() * 1e3
    );
    println!(
        "  total decode wall   {:>8.2} ms",
        decode_wall.as_secs_f64() * 1e3
    );

    println!(
        "\nper-token (decode-only, ledger reset before decode; {} steps averaged):",
        step_wall.len()
    );
    // KDA's own decomposition, eleven leaves summed for "KDA total" —
    // reverted to this shape at P4c-4: `KdaBranchFanout` (P4c-2a) is no
    // longer wired to any call site (branch concurrency was found to
    // cost more than it saved, see that rung's own writeup), and
    // q/k/v/o_proj now run sequentially through `matvec_bf16` ->
    // `executor::project_as`, which times under the CALLER's class
    // directly rather than nesting the executor's own generic
    // `OpClass::Projection` inside it — so these eleven are genuine,
    // disjoint leaves again, safe to sum without the P4c-2a-era
    // diagnostic/summed split.
    let kda_rows = [
        (OpClass::KdaQProj, "KDA q_proj"),
        (OpClass::KdaKProj, "KDA k_proj"),
        (OpClass::KdaVProj, "KDA v_proj"),
        (OpClass::KdaConv, "KDA conv x3"),
        (OpClass::KdaQkNorm, "KDA q/k norm"),
        (OpClass::KdaDecayGate, "KDA decay gate"),
        (OpClass::KdaOutputGate, "KDA output gate"),
        (OpClass::KdaBProj, "KDA b_proj"),
        (OpClass::KdaRecurrence, "KDA recurrence"),
        (OpClass::KdaGatedNorm, "KDA gated norm"),
        (OpClass::KdaOProj, "KDA o_proj"),
    ];
    let rows = [
        (OpClass::Mla, "MLA total"),
        (OpClass::MoeRouter, "Router"),
        (OpClass::MoeFanout, "MoE fan-out"),
        (OpClass::Norm, "Norms"),
        (OpClass::Residual, "Residual/glue"),
        (OpClass::Embed, "Embed"),
        (OpClass::LmHead, "lm_head"),
    ];
    // Diagnostic ONLY, excluded from `attributed_ns`: since P4d, routed
    // experts and the shared branch run inside the SAME `MoeFanout`
    // dispatch (see that class's own doc comment) — these two show each
    // side's OWN summed per-job cost, not wall-clock, and would
    // double-count against `MoeFanout` if added to the total.
    let moe_branch_diagnostic = [
        (OpClass::MoeRoutedExpert, "  routed (diag)"),
        (OpClass::MoeSharedExpert, "  shared (diag)"),
    ];
    let mut attributed_ns = 0u64;
    let mut kda_total_ns = 0u64;
    for (class, label) in kda_rows {
        let t = ledger().get(class);
        attributed_ns += t.nanos;
        kda_total_ns += t.nanos;
        println!(
            "  {:<16} {:>9.2} ms/tok  ({:>6} calls, {:>8.1} us/call)",
            label,
            t.nanos as f64 / step_wall.len() as f64 / 1e6,
            t.calls,
            t.nanos_per_call() / 1e3,
        );
    }
    println!(
        "  {:<16} {:>9.2} ms/tok  (sum of the eleven leaves above)",
        "KDA total",
        kda_total_ns as f64 / step_wall.len() as f64 / 1e6,
    );
    for (class, label) in rows {
        let t = ledger().get(class);
        attributed_ns += t.nanos;
        println!(
            "  {:<16} {:>9.2} ms/tok  ({:>6} calls, {:>8.1} us/call)",
            label,
            t.nanos as f64 / step_wall.len() as f64 / 1e6,
            t.calls,
            t.nanos_per_call() / 1e3,
        );
        if class == OpClass::MoeFanout {
            for (diag_class, diag_label) in moe_branch_diagnostic {
                let dt = ledger().get(diag_class);
                println!(
                    "  {:<16} {:>9.2} ms/tok  ({:>6} calls, {:>8.1} us/call)",
                    diag_label,
                    dt.nanos as f64 / step_wall.len() as f64 / 1e6,
                    dt.calls,
                    dt.nanos_per_call() / 1e3,
                );
            }
        }
    }
    let unattributed_ns = decode_wall.as_nanos() as u64 - attributed_ns;
    println!(
        "  {:<16} {:>9.2} ms/tok  — command/allocation/glue overhead not covered by a named leaf",
        "Unattributed",
        unattributed_ns as f64 / step_wall.len() as f64 / 1e6
    );
    if ledger().nested() > 0 {
        println!(
            "  WARNING: {} nested timer(s) — the breakdown above double-counts",
            ledger().nested()
        );
    }

    println!("\nphysical (MoE-routed layers only, {moe_layer_count} of {num_layers}):");
    println!("  expert loads/token, no cache  {naive_loads_per_token:>6}   (= top_k[{top_k}] x moe_layers[{moe_layer_count}], worst case)");
    println!(
        "  unique experts, whole 16-tok decode  {union_across_decode:>6}   ({:.1} per layer, vs {} naive-per-token x 16 = {})",
        union_across_decode as f64 / moe_layer_count as f64,
        naive_loads_per_token, naive_loads_per_token * 16
    );
    println!(
        "  bytes/expert    {:>10.1} MiB",
        bytes_per_expert as f64 / 2f64.powi(20)
    );
    println!(
        "  bytes/token, no cache   {:>10.1} MiB  ({} experts x {:.1} MiB)",
        (naive_loads_per_token * bytes_per_expert) as f64 / 2f64.powi(20),
        naive_loads_per_token,
        bytes_per_expert as f64 / 2f64.powi(20)
    );
    println!(
        "  lm_head.weight  {:>10.1} MiB  (read from disk ONCE, held in memory — but the matvec touches",
        (vocab_size * hidden * 4) as f64 / 2f64.powi(20)
    );
    println!(
        "                              the FULL {vocab_size} x {hidden} matrix on EVERY decode step: argmax needs the whole vocabulary)"
    );

    println!("\n(this run does not cache expert or lm_head loads across the 16 steps at all —");
    println!(
        " every number above is the CURRENT, fully-verified, zero-optimisation implementation.)"
    );
}
