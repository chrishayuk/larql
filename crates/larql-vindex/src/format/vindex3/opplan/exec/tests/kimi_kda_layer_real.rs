//! One complete Kimi Linear KDA decoder layer, at real weights — the gate
//! the user specified: same real hidden state in → same complete layer
//! output out, against the Python reference, with every internal
//! boundary (input norm, KDA attention's own fifteen boundaries, the
//! post-attention residual, the post-attention norm, the router, each
//! selected expert, the routed sum, the shared branch) exposed so a
//! disagreement is immediately attributable rather than debugged
//! backwards from one final vector.
//!
//! Env-gated, same reason `kda_parity_real.rs`/`kimi_moe_real.rs` are:
//! real per-expert weight data, ~400 MiB at full width, regenerable in
//! seconds:
//!
//! ```text
//! python scripts/kimi_kda_layer_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --layer 1 --out /tmp/kimi_kda_layer_fixture
//! LARQL_KIMI_KDA_LAYER_FIXTURE=/tmp/kimi_kda_layer_fixture \
//!     cargo test -p larql-vindex --lib kimi_kda_layer_real
//! ```

use std::path::PathBuf;

use larql_models::config::KdaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kda::{zero_state, KdaOutputGateWeights, KdaWeights};
use crate::format::vindex3::opplan::exec::kimi_kda_layer::kda_decoder_layer_forward;
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;

const FIXTURE_ENV: &str = "LARQL_KIMI_KDA_LAYER_FIXTURE";
/// Same as `kda_parity_real.rs`/`kimi_moe_real.rs`: full-width real
/// weights sum hundreds of terms per element, so two orderings of the
/// same arithmetic separate further than a tiny fixture.
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

/// Expert weights (P4a) and KDA's q/k/v/o_proj (P4c-4) — both BF16 code
/// units, never F32. See `stack_real.rs`'s own `read_bf16` for why
/// truncation here recovers the checkpoint's own bits exactly.
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
fn one_complete_kda_layer_matches_the_oracle_at_kimis_real_geometry() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
        return;
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");

    let hidden = manifest["hidden"].as_u64().unwrap() as usize;
    let eps = manifest["rms_eps"].as_f64().unwrap();
    let inter = manifest["moe_intermediate_size"].as_u64().unwrap() as usize;
    let shared_count = manifest["num_shared_experts"].as_u64().unwrap() as usize;
    let experts = manifest["experts"].as_u64().unwrap() as usize;
    let top_k = manifest["top_k"].as_u64().unwrap() as usize;
    let renormalize = manifest["moe_renormalize"].as_bool().unwrap();
    let branch_scale = manifest["routed_scaling_factor"].as_f64().unwrap();
    let geometry = KdaGeometry {
        num_heads: manifest["num_heads"].as_u64().unwrap() as usize,
        head_dim: manifest["head_dim"].as_u64().unwrap() as usize,
        conv_kernel: 4,
    };
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
    let input_norm_weight = read_f32(&dir, "input_norm_weight");
    let post_attention_norm_weight = read_f32(&dir, "post_attention_norm_weight");

    let kda = |n: &str| read_f32(&dir, &format!("kda_{n}"));
    let kda_bf16 = |n: &str| read_bf16(&dir, &format!("kda_{n}"));
    let (qp, kp, vp) = (kda_bf16("q_proj"), kda_bf16("k_proj"), kda_bf16("v_proj"));
    let (qc, kc, vc) = (kda("q_conv1d"), kda("k_conv1d"), kda("v_conv1d"));
    let (fa, fb) = (kda("f_a_proj"), kda("f_b_proj"));
    let (ga, gb) = (kda("g_a_proj"), kda("g_b_proj"));
    let (bp, al, dt) = (kda("b_proj"), kda("a_log"), kda("dt_bias"));
    let (on, op) = (kda("o_norm"), kda_bf16("o_proj"));
    let kda_weights = KdaWeights {
        q_proj: WeightRows::Bf16(&qp),
        k_proj: WeightRows::Bf16(&kp),
        v_proj: WeightRows::Bf16(&vp),
        q_conv1d: &qc,
        k_conv1d: &kc,
        v_conv1d: &vc,
        f_a_proj: &fa,
        f_b_proj: &fb,
        output_gate: KdaOutputGateWeights::LowRank {
            g_a_proj: &ga,
            g_b_proj: &gb,
        },
        b_proj: &bp,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        o_proj: WeightRows::Bf16(&op),
        norm_eps: eps as f32,
        // The rank the two gate factorisations meet at, read from this
        // fixture's own `f_a_proj` rather than assumed equal to the head
        // dim: on this checkpoint the two coincide, and the executor no
        // longer takes that coincidence as its definition.
        gate_rank: fa.len() / hidden,
    };

    let router_weight = read_f32(&dir, "router_weight");
    let router_bias = read_f32(&dir, "router_bias");

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
            .unwrap_or_else(|| panic!("layer forward asked for un-selected expert {id}"));
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

    let mut state = zero_state(geometry);
    let trace = kda_decoder_layer_forward(
        &x,
        hidden,
        &input_norm_weight,
        &post_attention_norm_weight,
        eps,
        kda_weights,
        geometry,
        &mut state,
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

    // ── Boundary by boundary, so a disagreement names its own stage ──
    assert_close(
        "input_normed",
        &trace.input_normed,
        &read_f32(&dir, "out_input_normed"),
    );
    assert_close(
        "attention.output",
        &trace.attention.output,
        &read_f32(&dir, "out_attention_output"),
    );
    assert_close(
        "after_attention",
        &trace.after_attention,
        &read_f32(&dir, "out_after_attention"),
    );
    assert_close(
        "post_attention_normed",
        &trace.post_attention_normed,
        &read_f32(&dir, "out_post_attention_normed"),
    );

    let mut got_ids = trace.moe.router.selected_ids.clone();
    got_ids.sort_unstable();
    let mut want_ids = ids_order.clone();
    want_ids.sort_unstable();
    assert_eq!(
        got_ids, want_ids,
        "the router must select the SAME experts the oracle did, from the SAME \
         post-attention-normed input this layer just computed"
    );

    for (slot, &id) in trace.moe.router.selected_ids.iter().enumerate() {
        let oracle_slot = ids_order.iter().position(|&i| i == id).unwrap();
        assert_close(
            &format!("expert {id} output"),
            &trace.moe.expert_outputs[slot],
            &read_f32(&dir, &format!("out_expert_output_{oracle_slot}")),
        );
    }
    assert_close(
        "routed_sum",
        &trace.moe.routed_sum,
        &read_f32(&dir, "out_routed_sum"),
    );
    assert_close(
        "shared_output",
        &trace.moe.shared_output,
        &read_f32(&dir, "out_shared_output"),
    );
    assert_close("output", &trace.output, &read_f32(&dir, "out_layer_output"));

    eprintln!(
        "  layer {}: input norm + KDA(32x128) + residual + post-attention norm + \
         router + {} experts + shared, all within {:e}",
        manifest["layer"], top_k, TOLERANCE
    );
}
