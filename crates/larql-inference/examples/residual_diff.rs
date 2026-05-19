//! Per-layer residual diff between CPU (`predict_kquant_hidden`) and Metal
//! (`dispatch_full_pipeline`) forward passes.
//!
//! Invariant under test: for the same input prompt, both backends should
//! produce the same `[seq_len, hidden]` residual at the end of every
//! layer. Any drift compounds into the final logits, so the first layer
//! where cosine similarity drops below 1.0 is usually the one to fix.
//!
//! How it works:
//!   1. Triggers both backends on the same prompt with max_tokens=1
//!      (single prefill pass — no KV cache involvement) with the
//!      respective per-layer dump env vars set to disjoint temp dirs.
//!   2. Reads the `.f32` dumps each backend emits per layer.
//!      CPU:   `cpu_layer_{LL}.f32`           — LARQL_CPU_DUMP_LAYERS
//!      Metal: `metal_layer_{LL}_h_out.f32`   — LARQL_METAL_DUMP_LAYERS
//!      Both are raw little-endian `f32[seq_len * hidden]` of the
//!      end-of-layer residual.
//!   3. Computes cosine similarity + max abs diff per layer, flagging
//!      the first layer where cos_sim drops below 0.9999.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference --example residual_diff -- \
//!       <vindex-dir> [prompt]
//!
//! Metal prefill dumps only fire on the dense (non-MoE) path — MoE models
//! use `decode_token` which doesn't hook the dump. For MoE, the CPU dump
//! still works; pair it with the existing `LARQL_DUMP_RESIDUALS` for
//! Metal's MoE path (packed format, parsed differently).

extern crate blas_src;

use std::path::{Path, PathBuf};

use larql_inference::layer_graph::generate::generate;
use larql_inference::layer_graph::CachedLayerGraph;
use larql_inference::wrap_chat_prompt;

const DRIFT_THRESHOLD: f32 = 0.9999;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let vindex_path = PathBuf::from(
        args.next()
            .ok_or("usage: residual_diff <vindex-dir> [prompt]")?,
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());

    if !vindex_path.is_dir() {
        return Err(format!("not a vindex dir: {}", vindex_path.display()).into());
    }

    // Disjoint scratch dirs for the two backends' dumps. `tempfile`
    // auto-cleans on drop; we stash the paths before the guards leave
    // scope so the post-run readers see the files. When the env vars are
    // set by the caller (for interactive inspection of intermediate
    // files), we use those paths directly and skip the TempDir guard so
    // the files survive the run.
    let external_cpu = std::env::var_os("LARQL_CPU_DUMP_LAYERS").map(std::path::PathBuf::from);
    let external_metal = std::env::var_os("LARQL_METAL_DUMP_LAYERS").map(std::path::PathBuf::from);
    let _cpu_guard: Option<tempfile::TempDir>;
    let _metal_guard: Option<tempfile::TempDir>;
    let cpu_path: std::path::PathBuf = if let Some(p) = external_cpu {
        _cpu_guard = None;
        std::fs::create_dir_all(&p).ok();
        p
    } else {
        let d = tempfile::tempdir()?;
        let p = d.path().to_path_buf();
        _cpu_guard = Some(d);
        p
    };
    let metal_path: std::path::PathBuf = if let Some(p) = external_metal {
        _metal_guard = None;
        std::fs::create_dir_all(&p).ok();
        p
    } else {
        let d = tempfile::tempdir()?;
        let p = d.path().to_path_buf();
        _metal_guard = Some(d);
        p
    };
    std::env::set_var("LARQL_CPU_DUMP_LAYERS", &cpu_path);
    std::env::set_var("LARQL_METAL_DUMP_LAYERS", &metal_path);
    // Stage dumps: Metal writes to LARQL_METAL_DUMP_LAYERS (same dir) with
    // `metal_layer_{LL}_<stage>.f32` names; CPU writes its stages into a
    // shared stage dir via LARQL_CPU_STAGE_DUMP using `cpu_L0_<stage>.f32`.
    // Place CPU stage files alongside CPU layer files for simpler reading.
    std::env::set_var("LARQL_CPU_STAGE_DUMP", &cpu_path);
    // Which layer's per-stage snapshots to compare. Override with the env
    // var if you want to bisect somewhere other than L0.
    let stage_layer: usize = std::env::var("LARQL_STAGE_DUMP_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // ── Load vindex ────────────────────────────────────────────────────
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let _ = index.load_lm_head_kquant(&vindex_path);
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;

    let mut w_metal = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut w_cpu = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;

    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), &prompt);
    let token_ids = larql_inference::encode_prompt(&tokenizer, &*w_metal.arch, &wrap.prompt)?;
    let num_layers = w_metal.num_layers;
    let hidden = w_metal.hidden_size;
    let seq_len = token_ids.len();

    println!("━━━ Per-layer residual diff ─────────────────────────────────────────");
    println!("  vindex:       {}", vindex_path.display());
    println!("  model:        {}", cfg.model);
    println!("  family:       {}", cfg.family);
    println!("  prompt:       {prompt:?}");
    println!(
        "  seq_len:      {seq_len}  ({} tokens post-template)",
        token_ids.len()
    );
    println!("  num_layers:   {num_layers}");
    println!("  hidden:       {hidden}");
    println!();

    // ── Drive both backends (max_tokens=1 → just prefill once each) ─────
    let metal_backend =
        larql_compute_metal::MetalBackend::new().ok_or("Metal backend unavailable")?;
    let metal_cached = CachedLayerGraph::from_residuals(Vec::new());
    println!(
        "Running Metal prefill (dumps → {})",
        metal_path.as_path().display()
    );
    let _ = generate(
        &mut w_metal,
        &tokenizer,
        &token_ids,
        1,
        &index,
        &metal_backend,
        &metal_cached,
        0..num_layers,
    );

    let cpu_backend = larql_compute::CpuBackend;
    let cpu_cached = CachedLayerGraph::from_residuals(Vec::new());
    println!(
        "Running CPU prefill (dumps → {})",
        cpu_path.as_path().display()
    );
    let _ = generate(
        &mut w_cpu,
        &tokenizer,
        &token_ids,
        1,
        &index,
        &cpu_backend,
        &cpu_cached,
        0..num_layers,
    );

    println!();
    println!("━━━ Layer-by-layer comparison ──────────────────────────────────────");
    println!("  L    h_post_attn cos / maxΔ    h_out cos / maxΔ         attn vs ffn");
    println!("  ─── ─────────────────────────  ─────────────────────────  ─────────");

    let mut first_bad: Option<usize> = None;
    for l in 0..num_layers {
        let load = |cpu_name: &str, metal_name: &str| -> Option<(Vec<f32>, Vec<f32>)> {
            let c = read_f32(&cpu_path.as_path().join(cpu_name))?;
            let m = read_f32(&metal_path.as_path().join(metal_name))?;
            if c.len() != m.len() {
                return None;
            }
            Some((c, m))
        };

        let hpa = load(
            &format!("cpu_layer_{l:02}_h_post_attn.f32"),
            &format!("metal_layer_{l:02}_h_post_attn.f32"),
        );
        let hout = load(
            &format!("cpu_layer_{l:02}.f32"),
            &format!("metal_layer_{l:02}_h_out.f32"),
        );

        let Some((cpu_out, mtl_out)) = hout else {
            println!("  L{l:02}  <h_out dump missing>");
            continue;
        };
        let stat_out = layer_stats(&cpu_out, &mtl_out);
        let stat_hpa = hpa.as_ref().map(|(c, m)| layer_stats(c, m));

        if stat_out.cos < DRIFT_THRESHOLD && first_bad.is_none() {
            first_bad = Some(l);
        }
        let flag = if stat_out.cos < DRIFT_THRESHOLD {
            " ←"
        } else {
            ""
        };

        // Diagnostic: which piece (attention vs FFN) introduces the drift.
        // If h_post_attn already differs, attention is the culprit;
        // otherwise drift is in FFN+PLE+scalar.
        let diagnosis = match stat_hpa {
            Some(ref s) if s.cos < DRIFT_THRESHOLD && stat_out.cos < DRIFT_THRESHOLD => "attn+ffn",
            Some(ref s) if s.cos < DRIFT_THRESHOLD => "attn",
            Some(_) if stat_out.cos < DRIFT_THRESHOLD => "ffn",
            Some(_) => "clean",
            None => "?",
        };

        let hpa_cell = match stat_hpa {
            Some(s) => format!("{:>8.6} / {:>8.2e}", s.cos, s.max_abs_diff),
            None => "         -    /        -".to_string(),
        };
        println!(
            "  L{l:02}  {}  {:>8.6} / {:>8.2e}  {:>9}{flag}",
            hpa_cell, stat_out.cos, stat_out.max_abs_diff, diagnosis,
        );
    }

    println!();
    match first_bad {
        Some(l) => {
            println!(
                "━━━ First layer with cos_sim < {} ─────────────────────────",
                DRIFT_THRESHOLD
            );
            println!("  L{l} is where CPU and Metal first diverge meaningfully.");
            if l == 0 {
                println!("  Layer 0 drift → culprit is in the embedding or layer-0 pre-norm / attention / FFN.");
            } else {
                println!(
                    "  Earlier layers match; focus on L{l} attention, FFN, or per-layer scalar."
                );
            }
            // Also point at stages (dumped for L0 only by the Metal
            // prefill hook) so the user can cross-reference.
            let stage_dumps = [
                "norm_out",
                "q_out",
                "k_out",
                "v_out",
                "attn_out",
                "o_out",
                "h_post_attn",
            ];
            if l == 0 {
                println!();
                println!(
                    "  L0 stage files available in {}:",
                    metal_path.as_path().display()
                );
                for s in &stage_dumps {
                    let p = metal_path.as_path().join(format!("metal_layer_00_{s}.f32"));
                    if p.is_file() {
                        println!("    {}", p.display());
                    }
                }
            }
        }
        None => {
            println!("━━━ No layer divergence above threshold ─────────────────────");
            println!("  All layers match within cos_sim >= {DRIFT_THRESHOLD}. Drift");
            println!("  (if any) is below threshold or comes from the lm_head / sampling step.");
        }
    }

    // ── Stage-by-stage comparison at `stage_layer` ──────────────────────
    // Naming convention: Metal writes `metal_layer_{LL}_{stage}.f32` for
    // arbitrary layers (when set via LARQL_STAGE_DUMP_LAYER). Layer 0 also
    // writes `metal_L0_q_out_after_qk_norm.f32` via a separate hook. CPU
    // writes `cpu_L0_<stage>.f32` from `attention::block::run_attention_block_core`.
    // We match both sides' layout below for a unified comparison table.
    println!();
    println!("━━━ Stage-by-stage comparison @ L{stage_layer} ──────────────────────────");
    println!(
        "  {:<28} {:>10}  {:>12}  {:>10}  {:>10}",
        "stage", "cos_sim", "max_abs_Δ", "||cpu||", "||mtl||"
    );
    let ll = format!("{stage_layer:02}");
    // Pairs of (pretty name, cpu file suffix, metal file suffix). CPU's
    // stage dump is always L0-prefixed by current block.rs convention, so
    // we read from that name — any layer picked up by the dump infra
    // still writes under `cpu_L0_*` for historical reasons.
    let pairs: &[(&str, String, String)] = &[
        (
            "norm_out (pre-Q/K/V)",
            "cpu_L0_norm_out.f32".to_string(),
            format!("metal_layer_{ll}_norm_out.f32"),
        ),
        (
            "q_out (raw, pre QK-norm)",
            "cpu_L0_q_out_raw.f32".to_string(),
            format!("metal_layer_{ll}_q_out.f32"),
        ),
        (
            "q_out_after_qk_norm",
            "cpu_L0_q_out_after_qk_norm.f32".to_string(),
            "metal_L0_q_out_after_qk_norm.f32".to_string(),
        ),
        (
            "q_out_after_rope",
            "cpu_L0_q_out_after_rope.f32".to_string(),
            String::new(),
        ),
        (
            "attn_out (softmax·V)",
            "cpu_L0_attn_out.f32".to_string(),
            format!("metal_layer_{ll}_attn_out.f32"),
        ),
        (
            "o_out (post Wo-proj)",
            "cpu_L0_o_out.f32".to_string(),
            format!("metal_layer_{ll}_o_out.f32"),
        ),
    ];
    for (name, cpu_name, metal_name) in pairs {
        if metal_name.is_empty() {
            continue;
        }
        let cpu_path = cpu_path.as_path().join(cpu_name);
        let metal_path = metal_path.as_path().join(metal_name);
        let cpu = read_f32(&cpu_path);
        let metal = read_f32(&metal_path);
        match (cpu, metal) {
            (Some(c), Some(m)) if c.len() == m.len() => {
                let s = layer_stats(&c, &m);
                let flag = if s.cos < DRIFT_THRESHOLD { " ←" } else { "" };
                println!(
                    "  {:<28} {:>10.6}  {:>12.3e}  {:>10.3}  {:>10.3}{flag}",
                    name, s.cos, s.max_abs_diff, s.cpu_norm, s.metal_norm
                );
            }
            (Some(c), Some(m)) => {
                println!(
                    "  {:<28} <len mismatch: cpu={} mtl={}>",
                    name,
                    c.len(),
                    m.len()
                );
            }
            (None, _) => println!("  {:<28} <cpu missing: {}>", name, cpu_path.display()),
            (_, None) => println!("  {:<28} <mtl missing: {}>", name, metal_path.display()),
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct LayerStat {
    cos: f32,
    max_abs_diff: f32,
    cpu_norm: f32,
    metal_norm: f32,
}

/// Cosine similarity + max absolute element-wise difference, plus each
/// side's L2 norm for scale debugging.
fn layer_stats(cpu: &[f32], metal: &[f32]) -> LayerStat {
    let n = cpu.len().min(metal.len());
    let mut dot = 0.0f64;
    let mut cn = 0.0f64;
    let mut mn = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..n {
        let a = cpu[i] as f64;
        let b = metal[i] as f64;
        dot += a * b;
        cn += a * a;
        mn += b * b;
        let d = (cpu[i] - metal[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cos = if cn > 0.0 && mn > 0.0 {
        (dot / (cn.sqrt() * mn.sqrt())) as f32
    } else {
        0.0
    };
    LayerStat {
        cos,
        max_abs_diff: max_abs,
        cpu_norm: cn.sqrt() as f32,
        metal_norm: mn.sqrt() as f32,
    }
}

/// Read a raw `f32[]` little-endian file. Returns `None` on any I/O
/// error or non-multiple-of-4 file size.
fn read_f32(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}
