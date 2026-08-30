//! Token IDs to logits, at REAL Kimi Linear weights — the exit
//! condition the user specified for P3d-k:
//!
//! ```text
//! token IDs -> embedding -> proven 27-layer stack -> final RMSNorm ->
//! lm_head -> logits
//! ```
//!
//! No sampling, no chat template, no performance work. Reuses every
//! real-weight loader `stack_real.rs` already proved (`load_real_layer`,
//! `spec`, `expert_list_for`) UNCHANGED — this file's only new claims
//! are the embedding gather and the `lm_head` projection.
//!
//! Three real tokenizer ids ("The capital of", ids `[1008, 10484,
//! 318]`), not two: the extra position costs nothing once ≥2 is already
//! required for MLA, and a 3-token real prompt is closer to what serving
//! will actually see than the minimum that merely avoids the softmax
//! degeneracy.
//!
//! Env-gated, same reason every other real-weight gate in this file is
//! — the largest fixture yet at ~20 GiB (`lm_head.weight` alone
//! contributes ~1.4 GiB at f32, over the FULL 163,840-token vocabulary —
//! required, since top-k ranking and argmax are claims about the WHOLE
//! distribution, not a subset):
//!
//! ```text
//! python scripts/kimi_token_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --tokens 1008 10484 318 --out /tmp/kimi_token_fixture
//! LARQL_KIMI_TOKEN_FIXTURE=/tmp/kimi_token_fixture \
//!     cargo test -p larql-vindex --lib token_real --release
//! ```

use larql_models::config::{KdaGeometry, MlaGeometry};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::stack::{LayerSpec, LayerState, LoadedExpert};
use crate::format::vindex3::opplan::exec::token::{token_forward, EmbeddingRow};

use super::stack_real::{expert_list_for, load_real_layer, read_f32, spec};

const FIXTURE_ENV: &str = "LARQL_KIMI_TOKEN_FIXTURE";
/// Wider than the stack's own `2e-3`: `lm_head`'s matvec sums 2304 terms
/// per one of 163,840 vocabulary rows, on top of the 27-layer stack's
/// own already-compounded noise.
const TOLERANCE: f32 = 4e-3;
/// How wide a "top-k ranking" claim this gate makes — the model's own
/// export dumps the top 10 ids per position, so this is what is
/// actually checkable, not an arbitrary smaller number.
const TOP_K_RANK: usize = 10;

fn fixture_dir() -> Option<std::path::PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(std::path::PathBuf::from)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// `‖got − want‖₂ / ‖want‖₂` — scale-normalised, so a 163,840-wide logit
/// vector's disagreement is judged relative to its own magnitude rather
/// than an absolute threshold picked for a 2,304-wide hidden state.
fn relative_rms(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = want.iter().map(|&b| (b as f64).powi(2)).sum();
    ((num / den.max(1e-30)).sqrt()) as f32
}

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    let d = max_abs_diff(got, want);
    assert!(d < TOLERANCE, "{name}: max|Δ| {d:e}");
}

fn top_k_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .expect("logits are never NaN")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

#[test]
fn token_ids_to_logits_match_the_oracle_at_every_position() {
    let Some(dir) = fixture_dir() else {
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
    let positions = manifest["positions"].as_u64().unwrap() as usize;
    let vocab_size = manifest["vocab_size"].as_u64().unwrap() as usize;
    let token_ids: Vec<usize> = manifest["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let want_argmax: Vec<usize> = manifest["argmax_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let want_top10: Vec<Vec<usize>> = manifest["top10_ids"]
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

    assert_eq!(
        num_layers, 27,
        "this gate exists to run the REAL 27-layer topology"
    );
    assert_eq!(positions, token_ids.len());
    assert!(
        positions >= 2,
        "MLA's attention math needs ≥2 real positions to be non-degenerate"
    );
    assert_eq!(
        (experts, top_k),
        (256, 8),
        "this gate exists to run REAL geometry"
    );
    assert_eq!(
        vocab_size, 163_840,
        "the checkpoint's REAL vocab, not the tokenizer's self-reported count"
    );

    eprintln!(
        "  loading {num_layers} real layers from {}...",
        dir.display()
    );
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

    eprintln!("  loading embedding rows, final norm, lm_head ({vocab_size}-wide)...");
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
    assert_eq!(
        lm_head_weight.len(),
        vocab_size * hidden,
        "lm_head.weight shape"
    );

    let mut got_argmax = Vec::with_capacity(positions);
    for (p, &token_id) in token_ids.iter().enumerate() {
        eprintln!("  position {p} (token {token_id}): running embedding + all {num_layers} layers + lm_head...");
        let trace = token_forward(
            token_id,
            hidden,
            &embedding_rows,
            &specs,
            &mut states,
            &final_norm_weight,
            eps,
            &lm_head_weight,
            vocab_size,
        );

        assert_close(
            &format!("position {p} embedding"),
            &trace.embedding,
            &read_f32(&dir, &format!("embedding_{p}")),
        );
        assert_close(
            &format!("position {p} stack_output"),
            trace.stack_output(),
            &read_f32(&dir, &format!("stack_output_{p}")),
        );
        assert_close(
            &format!("position {p} final_normed"),
            &trace.final_normed,
            &read_f32(&dir, &format!("final_normed_{p}")),
        );

        let want_logits = read_f32(&dir, &format!("logits_{p}"));
        let d = max_abs_diff(&trace.logits, &want_logits);
        assert!(d < TOLERANCE, "position {p} logits: max|Δ| {d:e}");
        let rel = relative_rms(&trace.logits, &want_logits);
        assert!(
            rel < 0.05,
            "position {p} logits: relative RMS {rel:e} (5% bound)"
        );

        let got_top = top_k_ids(&trace.logits, TOP_K_RANK);
        assert_eq!(
            got_top, want_top10[p],
            "position {p}: top-{TOP_K_RANK} ranking must match the oracle exactly"
        );

        got_argmax.push(trace.argmax);
    }

    assert_eq!(
        got_argmax, want_argmax,
        "argmax next-token id must match the oracle at every position"
    );

    eprintln!(
        "  token ids {token_ids:?}, argmax next-token ids {got_argmax:?}, every boundary within {TOLERANCE:e}"
    );
}
