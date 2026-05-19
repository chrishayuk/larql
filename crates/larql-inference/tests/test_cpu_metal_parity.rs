//! Per-layer CPU↔Metal prefill parity regression guard.
//!
//! Companion to the architecture golden tests (`test_arch_golden`) —
//! the goldens check token-level output, this suite checks the
//! per-layer hidden state. Both are needed: a kernel can drift
//! quietly enough to keep the argmax token unchanged for a few steps
//! while compounding into a real bug at longer generations. The
//! per-layer check rejects "good output by luck".
//!
//! Driven entirely through [`larql_inference::residual_diff`] —
//! captures both backends in memory, compares with [`compare_captures`]
//! at the [`ParityThreshold::tight`] preset, asserts via
//! [`ParityReport::assert_clean`]. No tempdirs, no env vars in the
//! test body. The capture module owns that plumbing.
//!
//! ### Caught regressions
//!
//! - **Metal `fused_attention` head_dim>256 bug** — `tg_q[256..512]`
//!   left uninitialised, dropped attention magnitude ~6% per global
//!   layer. Compounded to cos≈0.91 by L59 on Gemma 4 31B; this suite
//!   would surface it at L5 (the first global layer) within the cos
//!   threshold of `tight()`.
//!
//! ### Skip semantics
//!
//! Vindexes can be tens of GB; missing ones print a skip note and
//! return `Ok` so CI stays green. `LARQL_ARCH_STRICT=1` flips skips
//! to hard failures (useful locally to confirm the test actually ran).

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use larql_inference::residual_diff::{compare_captures, ParityThreshold, ResidualCapture};
use larql_inference::wrap_chat_prompt;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

struct ParityCase {
    name: &'static str,
    vindex_name: &'static str,
}

/// One row per arch we want covered. `gemma-4-26B-A4B-it` is omitted
/// because its Metal MoE prefill goes through `decode_token` per-position
/// (`metal/trait_impl.rs:215-229`), bypassing the per-layer dump that
/// `prefill_kquant` populates. Re-add when MoE prefill batches.
const CASES: &[ParityCase] = &[
    ParityCase {
        name: "gemma3-4b-it",
        vindex_name: "gemma3-4b-q4k-v2",
    },
    ParityCase {
        name: "gemma4-31b-it (dense)",
        vindex_name: "gemma4-31b-q4k",
    },
    ParityCase {
        name: "llama2-7b-hf (base)",
        vindex_name: "llama2-7b-q4k",
    },
    ParityCase {
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

fn run_case(case: &ParityCase) -> Result<(), String> {
    let Some(vindex_path) = find_vindex(case.vindex_name) else {
        if strict_mode() {
            return Err(format!(
                "[{}] vindex `{}` not found (LARQL_ARCH_STRICT=1)",
                case.name, case.vindex_name
            ));
        }
        eprintln!(
            "[{}] skip: vindex `{}` not found in cache",
            case.name, case.vindex_name
        );
        return Ok(());
    };

    let mut cb = SilentLoadCallbacks;
    let cfg = load_vindex_config(&vindex_path).map_err(|e| format!("load_vindex_config: {e}"))?;
    if cfg.quant != QuantFormat::Q4K {
        return Err(format!("expected Q4K vindex (got {:?})", cfg.quant));
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

    // Disjoint weight handles — CPU's per-layer dequant inserts into
    // `weights.tensors`, which would race if both backends shared a
    // single ModelWeights.
    let mut w_metal = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (metal): {e}"))?;
    let mut w_cpu = load_model_weights_kquant(&vindex_path, &mut cb)
        .map_err(|e| format!("load weights (cpu): {e}"))?;

    let prompt = "The capital of France is";
    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), prompt);
    let token_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)
        .map_err(|e| format!("encode_prompt: {e}"))?;

    let metal_backend = larql_compute_metal::MetalBackend::new()
        .ok_or("Metal backend unavailable — rebuild with --features metal")?;

    let metal = ResidualCapture::metal_prefill(&mut w_metal, &token_ids, &index, &metal_backend)?;
    let cpu = ResidualCapture::cpu_prefill(&mut w_cpu, &token_ids, &index)?;

    if cpu.num_layers() != metal.num_layers() {
        return Err(format!(
            "[{}] backend produced different layer counts: cpu={}, metal={}",
            case.name,
            cpu.num_layers(),
            metal.num_layers()
        ));
    }

    let report = compare_captures(&cpu, &metal, ParityThreshold::tight());
    report
        .assert_clean()
        .map_err(|e| format!("[{}] {e}", case.name))?;
    eprintln!(
        "[{}] parity OK across {} layers (rel max_abs ≤ {:.1}%)",
        case.name,
        cpu.num_layers(),
        100.0 * ParityThreshold::tight().rel_max_abs
    );
    Ok(())
}

#[test]
fn parity_gemma3_4b_prefill() {
    run_case(&CASES[0]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn parity_gemma4_31b_dense_prefill() {
    run_case(&CASES[1]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn parity_llama2_7b_prefill() {
    run_case(&CASES[2]).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn parity_mistral_7b_prefill() {
    run_case(&CASES[3]).unwrap_or_else(|e| panic!("{e}"));
}
