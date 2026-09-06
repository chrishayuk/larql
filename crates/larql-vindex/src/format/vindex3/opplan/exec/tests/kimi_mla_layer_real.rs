//! One complete Kimi Linear MLA decoder layer, at real weights — the
//! SAME gate `kimi_kda_layer_real.rs` proved for the KDA family, around
//! the OTHER attention operator: same real hidden state in → same
//! complete layer output out, against the Python reference, with every
//! internal boundary exposed so a disagreement names its own stage.
//!
//! **Three real positions, not one.** A single cached position cannot
//! exercise MLA's attention math at all — softmax over one score is
//! `1.0` regardless of its value (see `exec::mla`'s own doc comment) —
//! so this fixture threads `MlaState` across three real calls, exactly
//! the KV-cache-carrying shape the roadmap's next rung (mixed stack,
//! token 2 with carried state) needs anyway.
//!
//! Env-gated, same reason every other real-weight gate in this file is:
//!
//! ```text
//! python scripts/kimi_mla_layer_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --layer 3 --out /tmp/kimi_mla_layer_fixture
//! LARQL_KIMI_MLA_LAYER_FIXTURE=/tmp/kimi_mla_layer_fixture \
//!     cargo test -p larql-vindex --lib kimi_mla_layer_real
//! ```

use std::path::PathBuf;

use larql_models::config::MlaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kimi_mla_layer::mla_decoder_layer_forward;
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;
use crate::format::vindex3::opplan::exec::mla::{MlaQueryWeights, MlaState, MlaWeights};

const FIXTURE_ENV: &str = "LARQL_KIMI_MLA_LAYER_FIXTURE";
/// Same as the other full-width real-weight gates: hundreds of terms
/// summed per element separates two valid orderings of the same
/// arithmetic further than a tiny fixture would.
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
fn one_complete_mla_layer_matches_the_oracle_at_kimis_real_geometry() {
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
    let positions = manifest["positions"].as_u64().unwrap() as usize;
    let geometry = MlaGeometry {
        num_heads: manifest["num_heads"].as_u64().unwrap() as usize,
        kv_lora_rank: manifest["kv_lora_rank"].as_u64().unwrap() as usize,
        qk_nope_head_dim: manifest["qk_nope_head_dim"].as_u64().unwrap() as usize,
        qk_rope_head_dim: manifest["qk_rope_head_dim"].as_u64().unwrap() as usize,
        v_head_dim: manifest["v_head_dim"].as_u64().unwrap() as usize,
    };
    let kv_a_norm_eps = manifest["kv_a_norm_eps"].as_f64().unwrap();
    assert_eq!(
        (experts, top_k),
        (256, 8),
        "this gate exists to run REAL geometry"
    );
    assert_eq!(
        positions, 3,
        "MLA's attention math needs ≥2 real positions to be non-degenerate"
    );

    let union_ids: Vec<usize> = manifest["selected_ids_union_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let ids_per_position: Vec<Vec<usize>> = manifest["selected_ids_per_position"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();

    let inputs: Vec<Vec<f32>> = (0..positions)
        .map(|p| read_f32(&dir, &format!("input_{p}")))
        .collect();
    let input_norm_weight = read_f32(&dir, "input_norm_weight");
    let post_attention_norm_weight = read_f32(&dir, "post_attention_norm_weight");

    let (qp, kap, kan, kbp, op) = (
        read_f32(&dir, "mla_q_proj"),
        read_f32(&dir, "mla_kv_a_proj"),
        read_f32(&dir, "mla_kv_a_norm"),
        read_f32(&dir, "mla_kv_b_proj"),
        read_f32(&dir, "mla_o_proj"),
    );
    let mla_weights = MlaWeights {
        output_gate: None,
        query: MlaQueryWeights::Direct {
            q_proj: WeightRows::F32(&qp),
        },
        kv_a_proj: WeightRows::F32(&kap),
        kv_a_norm: &kan,
        kv_b_proj: WeightRows::F32(&kbp),
        o_proj: WeightRows::F32(&op),
        kv_a_norm_eps,
    };

    let router_weight = read_f32(&dir, "router_weight");
    let router_bias = read_f32(&dir, "router_bias");

    let loaded: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)> = union_ids
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
        let slot = union_ids
            .iter()
            .position(|&i| i == id)
            .unwrap_or_else(|| panic!("layer forward asked for un-loaded expert {id}"));
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

    let mut mla_state = MlaState::default();
    for p in 0..positions {
        let trace = mla_decoder_layer_forward(
            &inputs[p],
            hidden,
            &input_norm_weight,
            &post_attention_norm_weight,
            eps,
            mla_weights,
            geometry,
            &mut mla_state,
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
            &format!("position {p} input_normed"),
            &trace.input_normed,
            &read_f32(&dir, &format!("out_input_normed_{p}")),
        );
        assert_close(
            &format!("position {p} attention.output"),
            &trace.attention.output,
            &read_f32(&dir, &format!("out_attention_output_{p}")),
        );
        assert_close(
            &format!("position {p} after_attention"),
            &trace.after_attention,
            &read_f32(&dir, &format!("out_after_attention_{p}")),
        );
        assert_close(
            &format!("position {p} post_attention_normed"),
            &trace.post_attention_normed,
            &read_f32(&dir, &format!("out_post_attention_normed_{p}")),
        );

        let mut got_ids = trace.moe.router.selected_ids.clone();
        got_ids.sort_unstable();
        let mut want_ids = ids_per_position[p].clone();
        want_ids.sort_unstable();
        assert_eq!(
            got_ids, want_ids,
            "position {p}: the router must select the SAME experts the oracle did, from \
             the SAME post-attention-normed input this layer just computed"
        );

        for (slot, &id) in trace.moe.router.selected_ids.iter().enumerate() {
            let oracle_slot = ids_per_position[p].iter().position(|&i| i == id).unwrap();
            assert_close(
                &format!("position {p} expert {id} output"),
                &trace.moe.expert_outputs[slot],
                &read_f32(&dir, &format!("out_expert_output_{p}_{oracle_slot}")),
            );
        }
        assert_close(
            &format!("position {p} routed_sum"),
            &trace.moe.routed_sum,
            &read_f32(&dir, &format!("out_routed_sum_{p}")),
        );
        assert_close(
            &format!("position {p} shared_output"),
            &trace.moe.shared_output,
            &read_f32(&dir, &format!("out_shared_output_{p}")),
        );
        assert_close(
            &format!("position {p} output"),
            &trace.output,
            &read_f32(&dir, &format!("out_layer_output_{p}")),
        );
    }

    assert_eq!(
        mla_state.len(),
        positions,
        "the KV cache must hold exactly the positions run"
    );

    eprintln!(
        "  layer {}: {positions} positions, input norm + MLA({} heads) + residual + \
         post-attention norm + router + {top_k} experts + shared, all within {TOLERANCE:e}",
        manifest["layer"], geometry.num_heads
    );
}
