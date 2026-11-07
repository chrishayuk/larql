//! `larql shannon decode-diff` — CPU-vs-Metal parity across the
//! prefill→decode boundary.
//!
//! ## Why this is a separate instrument from `layer-diff`
//!
//! `layer-dump` / `layer-diff` compare *this engine* against an *external
//! reference* (`scripts/dump_layers_hf.py`) over a prefill. That axis
//! cannot see a decode defect at all, because both sides prefill.
//!
//! This subcommand compares *two of our own backends* across the
//! boundary that prefill parity does not cover:
//!
//!   reference = CPU prefill(N tokens), last row
//!   subject   = Metal prefill(N-1) + decode_token(token N)
//!
//! That distinction is not academic. ROADMAP M1 was exactly this shape:
//! Metal decode roped in-shader from `rope_base` alone and honoured no
//! scaling family, while prefill roped on the host and honoured all of
//! them. Every prefill comparison stayed clean while KV-cached decode
//! rotated global-layer positions at the wrong frequency.
//!
//! It replaces `examples/residual_diff.rs`, which carried this pass
//! until the examples tree was reorganised. The comparison logic itself
//! lives in `larql_inference::residual_diff`, so this is a front end,
//! not a reimplementation.

#[cfg(all(feature = "gpu", target_os = "macos"))]
use larql_inference::residual_diff::{compare_captures, ParityThreshold, ResidualCapture};

use super::DecodeDiffArgs;

// The `gpu` feature alone is not enough to select the real implementation: it
// compiles on every target, but `larql_compute_metal::MetalBackend` is
// `#[cfg(target_os = "macos")]`, so a Linux build with the feature on reaches
// for a type that is not there. Cargo cannot express "this feature, on this
// OS", so the call site carries it — as two whole definitions rather than
// `cfg` blocks inside one body, so the unsupported build pulls in neither the
// imports nor the locals of the supported one.
#[cfg(not(all(feature = "gpu", target_os = "macos")))]
pub fn run_decode_diff(_args: DecodeDiffArgs) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "decode-diff compares the CPU and Metal backends, so it needs a macOS host with the \
         `gpu` feature; this build has one or neither"
            .into(),
    )
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
pub fn run_decode_diff(args: DecodeDiffArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex = &crate::commands::primary::cache::resolve_installed_model(&args.vindex)?;
    if !vindex.is_dir() {
        return Err(format!("not a vindex dir: {}", vindex.display()).into());
    }

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let cfg = larql_vindex::load_vindex_config(vindex)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex, &mut cb)?;
    index.load_attn_kquant(vindex)?;
    index.load_interleaved_kquant(vindex)?;
    let _ = index.load_lm_head_kquant(vindex);
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex)?;

    let mut w_cpu = larql_vindex::load_model_weights_kquant(vindex, &mut cb)?;
    let mut w_metal = larql_vindex::load_model_weights_kquant(vindex, &mut cb)?;

    let wrap = larql_inference::wrap_chat_prompt(vindex, Some(cfg.model.as_str()), &args.prompt);
    let token_ids = larql_inference::encode_prompt(&tokenizer, &*w_cpu.arch, &wrap.prompt)?;
    if token_ids.len() < 2 {
        return Err(format!(
            "need at least 2 tokens to split into prefix + decoded token; got {}",
            token_ids.len()
        )
        .into());
    }

    let backend =
        larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable on this host")?;

    let split = token_ids.len() - args.steps.max(1).min(token_ids.len() - 1);
    let (prefix, tail) = token_ids.split_at(split);

    println!("━━━ decode-diff ────────────────────────────────────────────────");
    println!("  vindex:   {}", vindex.display());
    println!("  model:    {}", cfg.model);
    println!("  prompt:   {:?}", args.prompt);
    println!(
        "  split:    Metal prefill({}) + decode({})  vs  CPU prefill({})",
        prefix.len(),
        tail.len(),
        token_ids.len()
    );
    println!();

    // Reference: a fresh CPU prefill over the whole sequence, projected
    // to its last row so it is shape-compatible with a decode capture.
    let reference = ResidualCapture::cpu_prefill(&mut w_cpu, &token_ids, &index)
        .map_err(|e| format!("CPU prefill capture failed: {e}"))?
        .project_to_last_position();

    let subject = if tail.len() == 1 {
        ResidualCapture::metal_decode(&mut w_metal, prefix, tail[0], &index, &backend)
    } else {
        ResidualCapture::metal_decode_steps(&mut w_metal, prefix, tail, &index, &backend)
    }
    .map_err(|e| format!("Metal decode capture failed: {e}"))?;

    let report = compare_captures(&reference, &subject, ParityThreshold::tight());

    println!(
        "  {:<5} {:>10}  {:>12}  {:>12}  {:>12}",
        "L", "cos_sim", "max_abs_Δ", "||cpu||", "||mtl||"
    );
    let mut first_bad: Option<usize> = None;
    for (l, s) in report.layers.iter().enumerate() {
        let bad = s.cos < args.threshold_cos as f32;
        if bad && first_bad.is_none() {
            first_bad = Some(l);
        }
        println!(
            "  L{l:02}   {:>10.6}  {:>12.3e}  {:>12.3}  {:>12.3}{}",
            s.cos,
            s.max_abs,
            s.a_norm,
            s.b_norm,
            if bad { " ←" } else { "" }
        );
    }

    println!();
    match first_bad {
        Some(l) => {
            println!("  L{l} is the first layer where KV-cached decode leaves the");
            println!("  prefill reference. Prefill parity does not cover this — check");
            println!("  the decode path's RoPE plan (position divisor, base, scaling");
            println!("  family) and its KV read/write layout before the kernels.");
        }
        None => println!(
            "  decode reproduces prefill at every layer (cos >= {})",
            args.threshold_cos
        ),
    }

    if args.json {
        println!(
            "RESULT {{\"layers\":{},\"first_drift\":{},\"threshold_cos\":{}}}",
            report.layers.len(),
            match first_bad {
                Some(l) => l.to_string(),
                None => "null".to_string(),
            },
            args.threshold_cos
        );
    }

    if first_bad.is_some() {
        return Err("decode diverged from prefill".into());
    }
    Ok(())
}
