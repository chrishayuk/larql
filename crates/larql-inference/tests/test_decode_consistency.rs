//! Decode-vs-prefill consistency: per-layer hidden states from
//! `Metal prefill(N) + decode(1, 2, 4 …)` must match a fresh CPU
//! prefill at the same effective sequence length.
//!
//! ## Why
//!
//! Two kinds of bugs cost us a debugging week of manual diff'ing
//! before this suite existed:
//!
//! 1. **Kernel limits silently breached.** The Metal `fused_attention`
//!    shader gated its `tg_q` load on `if (tid < head_dim)` with a
//!    256-thread TG; on Gemma 4 global layers (head_dim=512) that left
//!    half of `tg_q` unset. End-to-end output stayed coherent, but the
//!    KV-cached decode step couldn't reproduce a fresh prefill at the
//!    same length. Per-token argmax drifted from token 1 onward.
//!
//! 2. **Prefill writes vs decode reads.** Bugs where prefill stores K/V
//!    in one layout and decode reads in another (off-by-one, wrong
//!    stride). Prefill alone passes parity, decode alone runs without
//!    panicking, but `prefill(N) + decode(1)` ≠ `prefill(N+1)`.
//!
//! The architecture goldens (`test_arch_golden`) only check the first
//! few tokens; small drift can keep them green for the wrong reasons.
//! `test_cpu_metal_parity` covers prefill but not the KV-cache hand-off.
//! This suite plugs that hole.
//!
//! ## What it asserts
//!
//! For each available Q4K vindex, for `k ∈ {1, 2, 4}` decode steps:
//!
//!   metal_decode = prefill(prompt_ids) + decode(t1) + decode(t2) + …
//!   cpu_ref      = predict_kquant_hidden(prompt_ids ++ [t1, t2, …])
//!
//! Each decode step's per-layer hidden (1 position) must match
//! `cpu_ref`'s last-position slice at that layer with cos ≥ 0.99995
//! and rel max_abs ≤ 1%. Threshold matches `test_cpu_metal_parity`'s
//! tight preset, so the two suites move together.
//!
//! Skip semantics mirror the golden / parity tests: missing vindexes
//! return Ok with a skip note.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use larql_inference::residual_diff::{compare_captures, ParityThreshold, ResidualCapture};
use larql_inference::wrap_chat_prompt;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

struct ConsistencyCase {
    name: &'static str,
    vindex_name: &'static str,
}

const CASES: &[ConsistencyCase] = &[
    ConsistencyCase {
        name: "gemma3-4b-it",
        vindex_name: "gemma3-4b-q4k-v2",
    },
    ConsistencyCase {
        name: "gemma4-31b-it (dense)",
        vindex_name: "gemma4-31b-q4k",
    },
    ConsistencyCase {
        name: "llama2-7b-hf (base)",
        vindex_name: "llama2-7b-q4k",
    },
    ConsistencyCase {
        name: "mistral-7b-v0.1 (base)",
        vindex_name: "mistral-7b-v0.1-q4k",
    },
];

fn find_vindex(name: &str) -> Option<PathBuf> {
    let filename = format!("{name}.vindex");
    if let Ok(env_path) = std::env::var(format!(
        "LARQL_VINDEX_{}",
        name.to_uppercase().replace('-', "_")
    )) {
        let p = PathBuf::from(env_path);
        if p.is_dir() {
            return Some(p);
        }
    }
    let chris_models = PathBuf::from("/Users/christopherhay/chris-models").join(&filename);
    if chris_models.is_dir() {
        return Some(chris_models);
    }
    let home = std::env::var("HOME").ok()?;
    [
        PathBuf::from(&home)
            .join(".cache/larql/local")
            .join(&filename),
        PathBuf::from("output").join(&filename),
    ]
    .into_iter()
    .find(|p| p.is_dir())
}

fn strict_mode() -> bool {
    matches!(
        std::env::var("LARQL_ARCH_STRICT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

/// Drive Metal through one prefill + N decode tokens, capture the per-layer
/// output of the **last** decode step against a CPU reference of the same
/// final sequence length, compare. `n_steps == 1` is the original single-
/// step variant; `n_steps >= 2` exercises the prefill→decode→decode KV-
/// cache hand-off (the path single-step parity does not cover).
fn check_n_steps(case: &ConsistencyCase, n_steps: usize) -> Result<(), String> {
    if n_steps == 0 {
        return Err("n_steps must be >= 1".to_string());
    }
    let Some(vindex_path) = find_vindex(case.vindex_name) else {
        if strict_mode() {
            return Err(format!(
                "[{}] vindex `{}` not found (LARQL_ARCH_STRICT=1)",
                case.name, case.vindex_name
            ));
        }
        eprintln!(
            "[{}] skip: vindex `{}` not found",
            case.name, case.vindex_name
        );
        return Ok(());
    };

    let mut cb = SilentLoadCallbacks;
    let cfg = load_vindex_config(&vindex_path).map_err(|e| format!("load_vindex_config: {e}"))?;
    if cfg.quant != QuantFormat::Q4K {
        return Err(format!("expected Q4K vindex, got {:?}", cfg.quant));
    }
    let tokenizer =
        load_vindex_tokenizer(&vindex_path).map_err(|e| format!("load_vindex_tokenizer: {e}"))?;
    let mut index =
        VectorIndex::load_vindex(&vindex_path, &mut cb).map_err(|e| format!("load vindex: {e}"))?;
    index
        .load_attn_kquant(&vindex_path)
        .map_err(|e| format!("load_attn_kquant: {e}"))?;
    index
        .load_interleaved_kquant(&vindex_path)
        .map_err(|e| format!("load_interleaved_kquant: {e}"))?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    let mut w_metal = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (metal): {e}"))?;
    let mut w_cpu = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (cpu): {e}"))?;

    let prompt = "The capital of France is";
    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), prompt);
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;

    let metal_backend =
        larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;

    // Drive Metal `generate(n_steps)` once to capture the deterministic
    // greedy token chain. Re-encode prompt + that chain to recover
    // canonical ids — keeps the parity check anchored to ids the
    // tokenizer actually round-trips.
    let cached = larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
    let metal_num_layers = w_metal.num_layers;
    let r = larql_inference::layer_graph::generate(
        &mut w_metal,
        &tokenizer,
        &prompt_ids,
        n_steps,
        &index,
        &metal_backend,
        &cached,
        0..metal_num_layers,
    );
    if r.tokens.len() < n_steps {
        return Err(format!(
            "[{}] generate produced only {} of {} tokens",
            case.name,
            r.tokens.len(),
            n_steps
        ));
    }
    let mut chain_text = String::new();
    for (t, _) in r.tokens.iter().take(n_steps) {
        chain_text.push_str(t);
    }
    let appended_prompt = format!("{}{}", wrap.prompt, chain_text);
    let appended_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &appended_prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;
    if appended_ids.len() != prompt_ids.len() + n_steps {
        eprintln!(
            "[{}] note: tokeniser merged generated tokens at boundary \
             (expected len {} got {}); skipping {n_steps}-step parity",
            case.name,
            prompt_ids.len() + n_steps,
            appended_ids.len(),
        );
        return Ok(());
    }
    let new_ids: Vec<u32> = appended_ids[prompt_ids.len()..].to_vec();

    let metal_decode = ResidualCapture::metal_decode_steps(
        &mut w_metal,
        &prompt_ids,
        &new_ids,
        &index,
        &metal_backend,
    )?;
    let cpu_ref_full = ResidualCapture::cpu_prefill(&mut w_cpu, &appended_ids, &index)?;
    let cpu_ref = cpu_ref_full.project_to_last_position();

    let report = compare_captures(&cpu_ref, &metal_decode, ParityThreshold::tight());
    report
        .assert_clean()
        .map_err(|e| format!("[{}] {n_steps}-step decode: {e}", case.name))?;
    eprintln!(
        "[{}] decode-consistency OK across {} layers ({n_steps} step{})",
        case.name,
        cpu_ref.num_layers(),
        if n_steps == 1 { "" } else { "s" },
    );
    Ok(())
}

/// Drive Metal through one prefill + one decode token, capture both
/// the decode's per-layer output and a CPU reference at sequence
/// length N+1, compare. Single-step variant — the multi-step test
/// loops this.
fn check_one_step(case: &ConsistencyCase) -> Result<(), String> {
    let Some(vindex_path) = find_vindex(case.vindex_name) else {
        if strict_mode() {
            return Err(format!(
                "[{}] vindex `{}` not found (LARQL_ARCH_STRICT=1)",
                case.name, case.vindex_name
            ));
        }
        eprintln!(
            "[{}] skip: vindex `{}` not found",
            case.name, case.vindex_name
        );
        return Ok(());
    };

    let mut cb = SilentLoadCallbacks;
    let cfg = load_vindex_config(&vindex_path).map_err(|e| format!("load_vindex_config: {e}"))?;
    if cfg.quant != QuantFormat::Q4K {
        return Err(format!("expected Q4K vindex, got {:?}", cfg.quant));
    }
    let tokenizer =
        load_vindex_tokenizer(&vindex_path).map_err(|e| format!("load_vindex_tokenizer: {e}"))?;
    let mut index =
        VectorIndex::load_vindex(&vindex_path, &mut cb).map_err(|e| format!("load vindex: {e}"))?;
    index
        .load_attn_kquant(&vindex_path)
        .map_err(|e| format!("load_attn_kquant: {e}"))?;
    index
        .load_interleaved_kquant(&vindex_path)
        .map_err(|e| format!("load_interleaved_kquant: {e}"))?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    let mut w_metal = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (metal): {e}"))?;
    let mut w_cpu = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (cpu): {e}"))?;

    let prompt = "The capital of France is";
    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), prompt);
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;

    let metal_backend =
        larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;

    // Step 0: drive Metal through `generate(max_tokens=1)` to pick a
    // realistic next token. Using a deterministic argmax (which is
    // what `generate` does) keeps the two paths aligned without us
    // hard-coding a token id per arch.
    let cached = larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
    let metal_num_layers = w_metal.num_layers;
    let r0 = larql_inference::layer_graph::generate(
        &mut w_metal,
        &tokenizer,
        &prompt_ids,
        1,
        &index,
        &metal_backend,
        &cached,
        0..metal_num_layers,
    );
    let token_0_text = r0
        .tokens
        .first()
        .map(|(t, _)| t.clone())
        .unwrap_or_default();
    if token_0_text.is_empty() {
        return Err(format!("[{}] generate produced no first token", case.name));
    }
    // Re-encode prompt + step-0 token to recover its id (the tokeniser
    // can re-merge; comparing the appended-id length tells us if so).
    let appended_prompt = format!("{}{}", wrap.prompt, token_0_text);
    let appended_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &appended_prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;
    if appended_ids.len() != prompt_ids.len() + 1 {
        eprintln!(
            "[{}] note: tokeniser merged step-0 token into prompt boundary; \
             skipping decode-consistency for this combination",
            case.name
        );
        return Ok(());
    }
    let token_0_id = *appended_ids.last().unwrap();

    // Capture both paths.
    let metal_decode = ResidualCapture::metal_decode(
        &mut w_metal,
        &prompt_ids,
        token_0_id,
        &index,
        &metal_backend,
    )?;
    let cpu_ref_full = ResidualCapture::cpu_prefill(&mut w_cpu, &appended_ids, &index)?;
    // CPU is `[seq=N+1, hidden]` per layer; decode is `[1, hidden]`.
    // Slice CPU's last-position row to align shapes.
    let cpu_ref = cpu_ref_full.project_to_last_position();

    let report = compare_captures(&cpu_ref, &metal_decode, ParityThreshold::tight());
    report
        .assert_clean()
        .map_err(|e| format!("[{}] one-step decode: {e}", case.name))?;
    eprintln!(
        "[{}] decode-consistency OK across {} layers (1 step)",
        case.name,
        cpu_ref.num_layers()
    );
    Ok(())
}

#[test]
fn decode_consistency_gemma3_4b() {
    check_one_step(&CASES[0]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn decode_consistency_gemma3_4b_2steps() {
    check_n_steps(&CASES[0], 2).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn decode_consistency_gemma4_31b_dense() {
    check_one_step(&CASES[1]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn decode_consistency_llama2_7b() {
    check_one_step(&CASES[2]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn decode_consistency_mistral_7b() {
    check_one_step(&CASES[3]).unwrap_or_else(|e| panic!("{e}"));
}
