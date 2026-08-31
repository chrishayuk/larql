//! P3d-m — 16 deterministic greedy tokens, at REAL weights. The
//! milestone the user's plan closes correctness on:
//!
//! ```text
//! same prompt -> greedy argmax x16, one persistent per-layer state
//! array reused for the whole decode -> every generated token id
//! matches the oracle -> Kimi Linear correctness CLOSED
//! ```
//!
//! **The hard semantic gate is 16/16 token ids identical, top-10
//! ranking stable, and no unexplained numerical discontinuity** —
//! deliberately NOT "every logit stays within the single-layer `3e-4`
//! bound indefinitely". Across 16 autoregressive positions, summation-
//! order noise compounds even when the model's actual TRAJECTORY (the
//! sequence of tokens it greedily picks) stays identical; the id/rank
//! agreement is the claim that matters, and the numeric checks stay as
//! generously-bounded diagnostics rather than the primary gate. If a
//! token were ever to first disagree, this test prints the oracle's own
//! top1/top2 margin at that exact step (`kimi_generate_export.py`
//! records it during its own greedy loop) — a margin near zero means a
//! numerical tie, a wide margin means a real defect.
//!
//! `kimi_generate_export.py` builds this fixture with an in-process
//! tensor cache shared across all 16 greedy steps: attention/norm/
//! router weights are checkpoint-static and read once; only NEWLY
//! selected experts are read as later positions select them — so the
//! disk cost is close to one large 19-position export, not sixteen.
//!
//! Env-gated:
//!
//! ```text
//! python scripts/kimi_generate_export.py ~/chris-models/Kimi-Linear-48B-A3B-Instruct \
//!     --tokens 1008 10484 318 --new 16 --out /tmp/kimi_generate_fixture
//! LARQL_KIMI_GENERATE_FIXTURE=/tmp/kimi_generate_fixture \
//!     cargo test -p larql-vindex --lib generate_real --release
//! ```

use larql_models::config::{KdaGeometry, MlaGeometry};
use serde_json::Value;

use crate::format::vindex3::opplan::exec::stack::{
    AttentionKind, LayerSpec, LayerState, LoadedExpert,
};
use crate::format::vindex3::opplan::exec::token::{token_forward, EmbeddingRow};

use super::stack_real::{expert_list_for, load_real_layer, read_f32, spec};

const FIXTURE_ENV: &str = "LARQL_KIMI_GENERATE_FIXTURE";
/// Generous and deliberately NOT the single-layer `3e-4`/`4e-3` bound —
/// see this module's own doc comment. A diagnostic, not the gate.
const SOFT_TOLERANCE: f32 = 2e-2;
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

fn top2_margin(logits: &[f32]) -> f32 {
    let mut top2 = [f32::NEG_INFINITY; 2];
    for &v in logits {
        if v > top2[0] {
            top2[1] = top2[0];
            top2[0] = v;
        } else if v > top2[1] {
            top2[1] = v;
        }
    }
    top2[0] - top2[1]
}

#[test]
fn sixteen_greedy_tokens_match_the_oracle_with_persistent_state() {
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
    let prompt_tokens: Vec<usize> = manifest["prompt_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let generated_tokens: Vec<usize> = manifest["generated_tokens"]
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
    let per_step_margins: Vec<f32> = manifest["per_step"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["top1_top2_margin"].as_f64().unwrap() as f32)
        .collect();

    assert_eq!(
        num_layers, 27,
        "this gate exists to run the REAL 27-layer topology"
    );
    assert_eq!(
        generated_tokens.len(),
        16,
        "P3d-m generates exactly 16 tokens"
    );
    assert_eq!(positions, prompt_tokens.len() + 16);
    assert_eq!(token_ids.len(), positions);
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

    // ONE persistent per-layer state array, reused for the WHOLE 19-position
    // decode — exactly the user's own framing: "one persistent per-layer
    // state array for the whole decode".
    let mut states: Vec<LayerState> = layers
        .iter()
        .map(|l| {
            if l.kind == "kda" {
                LayerState::Kda(crate::format::vindex3::opplan::exec::kda::zero_state(
                    kda_geometry,
                ))
            } else {
                LayerState::Mla(crate::format::vindex3::opplan::exec::mla::MlaState::default())
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

    let mut kda_sizes: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    let mut mla_sizes: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    let mut got_argmax = Vec::with_capacity(positions);
    let mut got_logits: Vec<Vec<f32>> = Vec::with_capacity(positions);

    // The CPU baseline the Metal trajectory is measured against. Timed
    // over the same span: stack + final norm + lm_head + argmax, with
    // the top-10 ranking done afterwards because sorting 163840 logits
    // a position is gate instrumentation, not decode work.
    let decode = std::time::Instant::now();
    for &token_id in &token_ids {
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
        got_argmax.push(trace.argmax);
        got_logits.push(trace.logits);
    }
    let decode_s = decode.elapsed().as_secs_f64();
    eprintln!(
        "  [baseline] {positions} positions in {decode_s:.3} s = {:.2} tok/s \
         ({:.1} ms/token), all stages on CPU",
        positions as f64 / decode_s,
        1000.0 * decode_s / positions as f64,
    );

    // ── The hard gate: every one of the 16 generated tokens ──
    let prompt_len = prompt_tokens.len();
    let mut first_divergence: Option<usize> = None;
    for (step, &oracle_margin) in per_step_margins.iter().enumerate().take(16) {
        let pos = prompt_len - 1 + step; // predicts token_ids[pos + 1]
        let got = got_argmax[pos];
        let want = token_ids[pos + 1];
        if got != want && first_divergence.is_none() {
            first_divergence = Some(step);
            let got_margin = top2_margin(&got_logits[pos]);
            eprintln!(
                "  FIRST DIVERGENCE at generation step {step} (position {pos}): got token {got}, \
                 want {want}. Oracle's own top1/top2 margin at this step was {oracle_margin:.6}; \
                 this run's margin is {got_margin:.6}. Near zero ⇒ numerical tie; wide ⇒ real defect."
            );
        }
        assert_eq!(
            got, want,
            "generation step {step} (position {pos}): token id must match — see the diagnostic above for the margin"
        );
        assert_eq!(
            top_k_ids(&got_logits[pos], TOP_K_RANK),
            want_top10[pos],
            "generation step {step} (position {pos}): top-{TOP_K_RANK} ranking must stay stable"
        );
    }
    assert!(
        first_divergence.is_none(),
        "16/16 token ids must match — see the diagnostic printed above"
    );

    // ── Every position, not just the 16 generated ones: the same
    // argmax/logit agreement `token_real.rs`/`token2_real.rs` already
    // proved, extended to the full 19-length sequence ──
    assert_eq!(
        got_argmax, want_argmax,
        "argmax must match the oracle at every position, not just the generated 16"
    );
    for (p, got_logits_at_p) in got_logits.iter().enumerate().take(positions) {
        let want_logits = read_f32(&dir, &format!("logits_{p}"));
        let d = max_abs_diff(got_logits_at_p, &want_logits);
        let rel = relative_rms(got_logits_at_p, &want_logits);
        assert!(
            d < SOFT_TOLERANCE,
            "position {p} logits: max|Δ| {d:e} exceeds the generous soft bound"
        );
        assert!(
            rel < 0.1,
            "position {p} logits: relative RMS {rel:e} exceeds the generous soft bound"
        );
    }

    // ── State model: KDA constant, MLA grows by one, across all 19 calls ──
    for (layer, sizes) in &kda_sizes {
        assert!(
            sizes.iter().all(|&s| s == sizes[0]),
            "KDA layer {layer}: recurrent state size must stay O(1) across all {positions} positions, got {sizes:?}"
        );
        assert_eq!(
            sizes[0],
            kda_geometry.num_heads * kda_geometry.head_dim * kda_geometry.head_dim,
            "KDA layer {layer}: state size must match the declared geometry — never zeroed, never wrong-sized"
        );
    }
    for (layer, sizes) in &mla_sizes {
        let expected: Vec<usize> = (1..=sizes.len()).collect();
        assert_eq!(*sizes, expected, "MLA layer {layer}: cached-position count must grow by exactly one per call, got {sizes:?}");
    }
    assert_eq!(
        kda_sizes.len() + mla_sizes.len(),
        num_layers,
        "every layer must own independent, correctly-typed state"
    );

    eprintln!(
        "  prompt {prompt_tokens:?} -> generated {generated_tokens:?} — 16/16 tokens match, \
         top-{TOP_K_RANK} stable at every position, state model holds across {positions} positions. \
         Kimi Linear correctness: CLOSED."
    );
}
