//! Kimi router + MoE block parity against real Kimi Linear layer-1
//! weights — the gate: same hidden input → same selected ids + weights →
//! same eight expert outputs → same routed sum → same shared branch →
//! same final MoE output, against `modeling_kimi.py`.
//!
//! The synthetic fixtures (`kimi_router.rs`, `kimi_moe_block.rs`) prove
//! the arithmetic and the combine logic at hand-checkable sizes; this
//! proves indexing and layout at Kimi's real geometry (256 experts,
//! hidden 2304, `moe_intermediate_size` 1024) — a transposed weight
//! matrix or a wrong row stride is invisible at the tiny sizes and fatal
//! here.
//!
//! Env-gated because the fixture is real per-expert weight data (~245
//! MiB for 8 experts + shared, at full width) — too large to commit,
//! regenerable in seconds:
//!
//! ```text
//! python scripts/kimi_moe_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --layer 1 --out /tmp/kimi_moe_fixture
//! LARQL_KIMI_MOE_FIXTURE=/tmp/kimi_moe_fixture cargo test -p larql-vindex --lib kimi_moe_real
//! ```

use std::path::PathBuf;

use serde_json::Value;

use crate::format::vindex3::opplan::exec::kimi_moe_block::{moe_block_forward, ExpertWeights};

const FIXTURE_ENV: &str = "LARQL_KIMI_MOE_FIXTURE";
/// Full-width real weights sum hundreds of terms per element, so two
/// orderings of the same arithmetic separate further than a tiny fixture
/// — the same reasoning `kda_parity_real.rs`'s looser tolerance states.
const TOLERANCE: f32 = 3e-4;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(PathBuf::from)
}

fn read_f32(dir: &std::path::Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("{name}.f32: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Expert weights only (P4a — `ExpertWeights` is BF16 code units, never
/// F32). See `stack_real.rs`'s own `read_bf16` for why truncation here
/// recovers the checkpoint's own bits exactly.
fn read_bf16(dir: &std::path::Path, name: &str) -> Vec<u16> {
    let bytes = std::fs::read(dir.join(format!("{name}.bf16")))
        .unwrap_or_else(|e| panic!("{name}.bf16: {e}"));
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    let d = max_abs_diff(got, want);
    assert!(d < TOLERANCE, "{name}: max|Δ| {d:e}");
}

#[test]
fn router_and_block_match_the_oracle_at_kimis_real_geometry() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");

    let hidden = manifest["hidden"].as_u64().unwrap() as usize;
    let inter = manifest["moe_intermediate_size"].as_u64().unwrap() as usize;
    let shared_count = manifest["num_shared_experts"].as_u64().unwrap() as usize;
    let experts = manifest["experts"].as_u64().unwrap() as usize;
    let top_k = manifest["top_k"].as_u64().unwrap() as usize;
    let renormalize = manifest["moe_renormalize"].as_bool().unwrap();
    let branch_scale = manifest["routed_scaling_factor"].as_f64().unwrap();
    assert_eq!(
        (experts, top_k),
        (256, 8),
        "this gate exists to run REAL geometry"
    );

    let ids_order: Vec<usize> = manifest["selected_ids_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let x = read_f32(&dir, "input");
    let router_weight = read_f32(&dir, "router_weight");
    let router_bias = read_f32(&dir, "router_bias");

    // Load every selected expert's real weights up front — sparse by
    // construction: only `top_k` of `experts` are ever read, matching the
    // export script's own claim.
    let loaded: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)> = ids_order
        .iter()
        .map(|&id| {
            (
                read_bf16(&dir, &format!("expert{id}_w1")),
                read_bf16(&dir, &format!("expert{id}_w3")),
                read_bf16(&dir, &format!("expert{id}_w2")),
            )
        })
        .collect();
    let by_id = |id: usize| -> ExpertWeights<'_> {
        let slot = ids_order
            .iter()
            .position(|&i| i == id)
            .unwrap_or_else(|| panic!("moe_block_forward asked for un-selected expert {id}"));
        let (gate, up, down) = &loaded[slot];
        ExpertWeights { gate, up, down }
    };

    let (shared_gate, shared_up, shared_down) = (
        read_bf16(&dir, "shared_w1"),
        read_bf16(&dir, "shared_w3"),
        read_bf16(&dir, "shared_w2"),
    );
    let shared_weights = ExpertWeights {
        gate: &shared_gate,
        up: &shared_up,
        down: &shared_down,
    };

    let trace = moe_block_forward(
        &x,
        hidden,
        inter,
        &router_weight,
        &router_bias,
        experts,
        top_k,
        renormalize,
        branch_scale,
        by_id,
        Some((shared_weights, inter * shared_count)),
    );

    // ── Router boundaries ──
    assert_close(
        "logits",
        &trace.router.logits,
        &read_f32(&dir, "router_logits"),
    );
    assert_close(
        "scores",
        &trace.router.scores,
        &read_f32(&dir, "router_scores"),
    );
    assert_close(
        "selection_scores",
        &trace.router.selection_scores,
        &read_f32(&dir, "router_selection_scores"),
    );

    let mut got_ids = trace.router.selected_ids.clone();
    got_ids.sort_unstable();
    let mut want_ids = ids_order.clone();
    want_ids.sort_unstable();
    assert_eq!(got_ids, want_ids, "selected ids must match, as a set");

    // Weight-bearing arrays are order-dependent on the ORACLE's own
    // `ids_order` (the order `route()` — Python or Rust — happened to
    // return); reorder this crate's output into that same order before
    // comparing elementwise.
    let reorder = |values: &[f32], ids: &[usize]| -> Vec<f32> {
        ids_order
            .iter()
            .map(|want_id| {
                let pos = ids.iter().position(|&i| i == *want_id).unwrap();
                values[pos]
            })
            .collect()
    };
    assert_close(
        "gathered_weights",
        &reorder(&trace.router.gathered_weights, &trace.router.selected_ids),
        &read_f32(&dir, "router_gathered_weights"),
    );
    assert_close(
        "normalized_weights",
        &reorder(&trace.router.normalized_weights, &trace.router.selected_ids),
        &read_f32(&dir, "router_normalized_weights"),
    );
    assert_close(
        "weights",
        &reorder(&trace.router.weights, &trace.router.selected_ids),
        &read_f32(&dir, "router_weights"),
    );

    // ── Per-expert outputs, routed sum, shared branch, final output ──
    for (slot, &id) in trace.router.selected_ids.iter().enumerate() {
        let oracle_slot = ids_order.iter().position(|&i| i == id).unwrap();
        let want = read_f32(&dir, &format!("expert_output_{oracle_slot}"));
        assert_close(
            &format!("expert {id} output"),
            &trace.expert_outputs[slot],
            &want,
        );
    }
    assert_close(
        "routed_sum",
        &trace.routed_sum,
        &read_f32(&dir, "routed_sum"),
    );
    assert_close(
        "shared_output",
        &trace.shared_output,
        &read_f32(&dir, "shared_output"),
    );
    assert_close("output", &trace.output, &read_f32(&dir, "output"));

    eprintln!(
        "  layer {}: {} experts matched, routed sum + shared branch + output all within {:e}",
        manifest["layer"], top_k, TOLERANCE
    );
}
