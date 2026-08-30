//! P3d-l — autoregressive continuation, at REAL weights:
//!
//! ```text
//! prompt tokens -> stack -> token 1 -> append token 1 ->
//!     reuse EVERY layer's state -> stack one new position -> token 2
//! ```
//!
//! `token_real.rs` already proved token 1 from the 3-token prompt
//! (`[1008, 10484, 318]`, argmax `276` = " the") in isolation. This file
//! proves the CONTINUATION claim: append that predicted token and run a
//! fourth position through the same carried state, exactly the shape
//! real decode needs.
//!
//! **No new incremental machinery in the Python oracle.** `kimi_token_
//! export.py` computes the 4-token sequence `[1008, 10484, 318, 276]`
//! WHOLE, the same way the 3-token fixture was computed — mathematically
//! identical to "decode token 1, then decode token 2 with KV/recurrent
//! state carried from token 1" for any internally-causal stack (the
//! SAME equivalence `stack_real.rs`'s own doc comment already leans on).
//! The RUST side, in contrast, calls `token_forward` once per position
//! with ONE `states` array threaded across all four calls — genuinely
//! incremental, genuinely carrying state. Agreement between a whole-
//! sequence oracle and an incremental Rust run, across FOUR positions
//! this time, is the same "two different computational orders agree"
//! proof `stack_real.rs`/`stack_parity.rs` already established, now at
//! the token/continuation level.
//!
//! Env-gated:
//!
//! ```text
//! python scripts/kimi_token_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --tokens 1008 10484 318 276 --out /tmp/kimi_token2_fixture
//! LARQL_KIMI_TOKEN2_FIXTURE=/tmp/kimi_token2_fixture \
//!     cargo test -p larql-vindex --lib token2_real --release
//! ```

use larql_models::config::{KdaGeometry, MlaGeometry};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::stack::{
    AttentionKind, LayerSpec, LayerState, LoadedExpert,
};
use crate::format::vindex3::opplan::exec::token::{token_forward, EmbeddingRow};

use super::stack_real::{expert_list_for, load_real_layer, read_f32, spec};

const FIXTURE_ENV: &str = "LARQL_KIMI_TOKEN2_FIXTURE";
const TOLERANCE: f32 = 4e-3;
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
fn token_two_agrees_with_state_reused_across_the_whole_stack() {
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
    assert_eq!(
        positions, 4,
        "3-token prompt + 1 appended prediction = token 1's continuation"
    );
    assert_eq!(token_ids.len(), 4);
    assert_eq!(
        &token_ids[..3],
        &[1008, 10484, 318],
        "must be the SAME prompt token_real.rs already proved token 1 from"
    );
    assert_eq!(
        (experts, top_k),
        (256, 8),
        "this gate exists to run REAL geometry"
    );
    assert_eq!(vocab_size, 163_840);

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

    // KDA layers' recurrent state size and MLA layers' cached-position
    // count, read back at every position — the two state MODELS this
    // whole programme cares about, now proven at real weights across a
    // GENUINE autoregressive continuation, not just a fresh 3-position
    // run: `kda_state_size_is_constant_and_mla_state_size_grows_by_one_
    // per_position` in `stack_parity.rs` already proved this shape on
    // the synthetic oracle; this is the same claim, real weights, one
    // token appended after a real argmax.
    let mut kda_sizes: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    let mut mla_sizes: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();

    let mut got_argmax = Vec::with_capacity(positions);
    let mut got_logits_at_last = Vec::new();
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

        for lt in &trace.layers {
            let sizes = match lt.kind {
                AttentionKind::Kda => kda_sizes.entry(lt.layer).or_default(),
                AttentionKind::Mla => mla_sizes.entry(lt.layer).or_default(),
            };
            sizes.push(lt.state_size);
        }

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
        assert_eq!(
            top_k_ids(&trace.logits, TOP_K_RANK),
            want_top10[p],
            "position {p}: top-{TOP_K_RANK} ranking"
        );

        got_argmax.push(trace.argmax);
        if p == positions - 1 {
            got_logits_at_last = trace.logits.clone();
        }
    }
    let _ = got_logits_at_last; // kept for future inspection, not asserted beyond the checks above

    // ── The two consecutive tokens ──
    let token_1 = got_argmax[2]; // predicted from the original 3-token prompt
    let token_2 = got_argmax[3]; // predicted after appending token 1 and reusing state
    assert_eq!(want_argmax[2], token_ids[3], "the fixture's own 4th token must BE token 1's prediction — otherwise this isn't testing continuation");
    assert_eq!(token_1, token_ids[3], "token 1 (this run's own prediction at position 2) must equal what got appended as position 3's input");
    assert_eq!(
        got_argmax, want_argmax,
        "argmax next-token id must match the oracle at every position, including token 2"
    );
    eprintln!("  token 1 = {token_1}, token 2 = {token_2} — two consecutive matching autoregressive tokens");

    // ── KDA state is REUSED, not reset: constant size across all 4 calls ──
    for (layer, sizes) in &kda_sizes {
        assert!(
            sizes.iter().all(|&s| s == sizes[0]),
            "KDA layer {layer}: recurrent state size must stay O(1) across positions, got {sizes:?}"
        );
        assert_eq!(
            sizes[0],
            kda_geometry.num_heads * kda_geometry.head_dim * kda_geometry.head_dim,
            "KDA layer {layer}: state size must match the declared geometry — a zeroed-and-rebuilt \
             state of the WRONG size would still be caught here even before the numeric parity check above"
        );
    }
    // ── MLA cache grows by exactly one position per call ──
    for (layer, sizes) in &mla_sizes {
        let expected: Vec<usize> = (1..=sizes.len()).collect();
        assert_eq!(*sizes, expected, "MLA layer {layer}: cached-position count must grow by exactly one per call, got {sizes:?}");
    }
    assert!(
        !kda_sizes.is_empty() && !mla_sizes.is_empty(),
        "both operator families must actually run in this topology"
    );

    eprintln!(
        "  all {num_layers} layers own independent state (kda_sizes and mla_sizes keyed by layer, \
         disjoint layer sets, {} kda + {} mla = {num_layers}); every boundary within {TOLERANCE:e}",
        kda_sizes.len(),
        mla_sizes.len()
    );
}
