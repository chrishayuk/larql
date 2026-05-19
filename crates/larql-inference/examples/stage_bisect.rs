//! Per-stage decode-vs-prefill bisect — locates the *first sub-stage*
//! of a layer where Metal KV-cached decode disagrees with a fresh CPU
//! prefill at the same effective sequence length.
//!
//! Companion to `examples/residual_diff.rs`. That tool diffs CPU vs
//! Metal *prefill* at end-of-layer granularity. This one diffs CPU
//! prefill vs Metal *decode* (the production hot path) and goes one
//! level deeper — splitting each layer into its sub-stages
//! (`norm_out`, `q_out`, `k_out`, `v_out`, `attn_out`, `o_out`,
//! `h_post_attn`, `ffn_norm_out`, `ffn_out_raw`/`down_out`) so a
//! drift signal points at a specific stage of the encoder.
//!
//! Built directly on the public
//! `larql_inference::residual_diff::stages::StageCapture` +
//! `compare_stages` API. The `test_decode_stage_bisect` test suite
//! pins the same calls in CI; this binary is the interactive form
//! you reach for when you're hunting an ad-hoc divergence.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --features metal -p larql-inference \
//!     --example stage_bisect -- <vindex-dir> [prompt] [layer]
//! ```
//!
//! `layer` defaults to 0. Override `LARQL_STAGE_DUMP_LAYER` if you
//! prefer the env-var route (the kernel test suite uses both).
//!
//! ## What you'll see
//!
//! For Gemma 3 4B / Llama 2 / Mistral on a known-good build, every
//! stage reports `cos≈1.0 max_abs≈1e-4`. For Gemma 4 31B on a build
//! before the 2026-04-25 q4k_matvec / q4k_ffn_gate_up shared-memory
//! cap fix, every stage up through `ffn_norm_out` matches at
//! `cos=1.0` and the divergence first appears at `ffn_out_raw`
//! (`cos≈0.97 / max_abs≈5.7`) — the bisect signature that pointed
//! at the FFN gate+up shader.

#[cfg(all(feature = "metal", target_os = "macos"))]
extern crate blas_src;

#[cfg(all(feature = "metal", target_os = "macos"))]
use std::path::PathBuf;

#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_compute::DecodeBackend;
#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_inference::residual_diff::{compare_stages, ParityThreshold, StageCapture};
#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_inference::wrap_chat_prompt;
#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

/// Pair list mapping the CPU dump's per-stage names to the
/// Metal-decode dump's per-stage names. Order = walk order; the first
/// failing pair under the chosen threshold is the localised divergence.
///
/// CPU prefill captures Q at three points (`q_out_raw`,
/// `q_out_after_qk_norm`, `q_out_after_rope`) because each is a separate
/// `Array2<f32>` allocation; Metal decode does the same operations
/// in-place on a single buffer and only sees the post-everything
/// `q_out`. The right comparison for the cached/decoded form is
/// CPU's `q_out_after_rope` ↔ Metal's `q_out`.
#[cfg(all(feature = "metal", target_os = "macos"))]
const STAGE_PAIRS: &[(&str, &str)] = &[
    // Pre-attention
    ("norm_out", "norm_out"),
    ("q_out_after_rope", "q_out"),
    ("k_out_after_rope", "k_out"),
    ("v_out", "v_out"),
    // Attention block
    ("attn_out", "attn_out"),
    ("o_out", "o_out"),
    ("h_post_attn", "h_post_attn"),
    // FFN block
    ("ffn_norm_out", "ffn_norm_out"),
    ("ffn_out_raw", "down_out"),
];

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let vindex_path = PathBuf::from(
        args.next()
            .ok_or("usage: stage_bisect <vindex-dir> [prompt] [layer]")?,
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let layer: usize = args
        .next()
        .or_else(|| std::env::var("LARQL_STAGE_DUMP_LAYER").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if !vindex_path.is_dir() {
        return Err(format!("not a vindex dir: {}", vindex_path.display()).into());
    }

    let mut cb = SilentLoadCallbacks;
    let cfg = load_vindex_config(&vindex_path)?;
    if cfg.quant != QuantFormat::Q4K {
        return Err(format!("expected Q4K vindex, got {:?}", cfg.quant).into());
    }
    let tokenizer = load_vindex_tokenizer(&vindex_path)?;

    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    let mut w_metal = load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut w_cpu = load_model_weights_kquant(&vindex_path, &mut cb)?;

    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), &prompt);
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)?;

    let metal_backend =
        larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;

    println!("━━━ Per-stage decode-vs-prefill bisect ────────────────────────────");
    println!("  vindex: {}", vindex_path.display());
    println!("  model:  {}", cfg.model);
    println!("  prompt: {prompt:?}");
    println!("  layer:  L{layer}");
    println!(
        "  prompt_ids ({}): {:?}…",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(8)]
    );
    println!();

    // Step 0: deterministic next token via greedy Metal decode. Mirrors
    // what `test_decode_stage_bisect` does so the interactive bisect
    // and the regression test agree on (prompt, t1).
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
        return Err("generate produced no first token".into());
    }
    println!("  step-0 token: {token_0_text:?}");

    let appended_prompt = format!("{}{}", wrap.prompt, token_0_text);
    let appended_ids =
        larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &appended_prompt)?;
    if appended_ids.len() != prompt_ids.len() + 1 {
        eprintln!(
            "note: tokeniser merged step-0 token at the prompt boundary; \
             stage bisect skipped for this combination."
        );
        return Ok(());
    }
    let token_0_id = *appended_ids.last().unwrap();
    println!();

    // Step 1: capture stages from both backends.
    metal_backend.reset_kv_cache();
    println!(
        "Running Metal prefill({prefill_n}) + decode(1) with stage dump …",
        prefill_n = prompt_ids.len()
    );
    let metal_stages = StageCapture::metal_decode(
        &mut w_metal,
        &prompt_ids,
        token_0_id,
        &index,
        &metal_backend,
        layer,
    )?;

    println!(
        "Running CPU prefill({}) with stage dump …",
        appended_ids.len()
    );
    let cpu_stages = StageCapture::cpu_prefill(&mut w_cpu, &appended_ids, &index, layer)?
        .project_to_last_position();

    if cpu_stages.is_empty() {
        return Err("CPU stage capture empty — env var or path bug".into());
    }
    if metal_stages.is_empty() {
        return Err("Metal stage capture empty — env var or path bug".into());
    }

    // Step 2: compare stage-by-stage. Loose threshold: this is a
    // diagnostic, not a strict parity test. A real divergence shows
    // up as cos<<0.999 (kernel-noise drift sits in the 1e-4 .. 1e-6
    // range across architectures).
    let report = compare_stages(
        &cpu_stages,
        &metal_stages,
        STAGE_PAIRS,
        ParityThreshold::loose(),
    );
    println!();
    print!("{}", report.summary());
    println!();
    if report.is_clean() {
        println!(
            "✓ no stage diverges past the loose threshold — decode and prefill agree at L{layer}."
        );
    } else {
        let i = report.first_bad.unwrap();
        let p = &report.pairs[i];
        if p.missing {
            println!(
                "✗ first divergence at stage `{}` (capture missing on one side)",
                p.name_a
            );
        } else {
            println!(
                "✗ first divergence at stage `{}` (cos={:.6} rel={:.3}%)",
                p.name_a,
                p.stat.cos,
                100.0 * p.stat.rel_max_abs(),
            );
        }
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("stage_bisect requires `--features metal`.");
}
