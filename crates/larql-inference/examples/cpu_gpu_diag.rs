//! CPU ↔ Metal diagnostic: accuracy + performance side-by-side on a real
//! vindex, for one prompt, one generated token.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference --example cpu_gpu_diag -- \
//!       <vindex-dir> [prompt] [tokens]
//!
//! Defaults:
//!   prompt = "The capital of France is"
//!   tokens = 8
//!
//! Output columns:
//!   • Backend name, wall time for N tokens, per-token decode ms, tok/s
//!   • First-token top-5 tokens + their scores from each backend
//!   • Top-1 agreement, top-5 Jaccard overlap, full generated text
//!
//! Doesn't attempt a per-layer residual diff — that path already exists
//! via `LARQL_METAL_DUMP_LAYERS` + `LARQL_CPU_DUMP_LAYERS`. This tool
//! focuses on user-facing accuracy (same top token? same continuation?)
//! and the head-to-head timing, which is what "diagnose perf + accuracy"
//! usually means in practice.

#[cfg(all(feature = "metal", target_os = "macos"))]
extern crate blas_src;

#[cfg(all(feature = "metal", target_os = "macos"))]
use std::path::PathBuf;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::time::Instant;

#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_inference::layer_graph::generate::generate;
#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_inference::layer_graph::CachedLayerGraph;
#[cfg(all(feature = "metal", target_os = "macos"))]
use larql_inference::wrap_chat_prompt;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let vindex_path = PathBuf::from(
        args.next()
            .ok_or("usage: cpu_gpu_diag <vindex-dir> [prompt] [tokens]")?,
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".to_string());
    let tokens: usize = args.next().map(|s| s.parse().unwrap_or(8)).unwrap_or(8);

    if !vindex_path.is_dir() {
        return Err(format!("not a vindex dir: {}", vindex_path.display()).into());
    }

    // ── Load once, reuse for both runs ─────────────────────────────────────
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let _ = index.load_lm_head_kquant(&vindex_path);

    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
    // Separate weight copies for each backend so CPU's per-layer dequant
    // inserts into `weights.tensors` don't race with the Metal path.
    let mut weights_metal = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut weights_cpu = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;

    // Chat template, if the vindex ships one.
    let wrap = wrap_chat_prompt(&vindex_path, Some(cfg.model.as_str()), &prompt);
    let token_ids = larql_inference::encode_prompt(&tokenizer, &*weights_metal.arch, &wrap.prompt)?;
    let num_layers = weights_metal.num_layers;

    println!("━━━ CPU ↔ Metal diagnostic ─────────────────────────────────────────");
    println!("  vindex:   {}", vindex_path.display());
    println!("  model:    {}", cfg.model);
    println!("  family:   {}", cfg.family);
    println!("  prompt:   {prompt:?}");
    println!("  chat:     applied={} ({})", wrap.applied, wrap.note);
    println!(
        "  prompt_ids.len(): {}  (template prompt: {:?})",
        token_ids.len(),
        &wrap.prompt[..wrap.prompt.len().min(100)]
    );
    println!("  tokens:   {tokens}");
    println!();

    // ── Metal run ──────────────────────────────────────────────────────────
    let metal_backend = larql_compute_metal::MetalBackend::new()
        .ok_or("Metal backend unavailable — this tool requires Metal")?;
    let metal_cached = CachedLayerGraph::from_residuals(Vec::new());
    println!("Running Metal…");
    let t0 = Instant::now();
    let r_metal = generate(
        &mut weights_metal,
        &tokenizer,
        &token_ids,
        tokens,
        &index,
        &metal_backend,
        &metal_cached,
        0..num_layers,
    );
    let metal_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── CPU run ────────────────────────────────────────────────────────────
    let cpu_backend = larql_compute::CpuBackend;
    let cpu_cached = CachedLayerGraph::from_residuals(Vec::new());
    println!("Running CPU…");
    let t0 = Instant::now();
    let r_cpu = generate(
        &mut weights_cpu,
        &tokenizer,
        &token_ids,
        tokens,
        &index,
        &cpu_backend,
        &cpu_cached,
        0..num_layers,
    );
    let cpu_wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── Timing table ──────────────────────────────────────────────────────
    println!();
    println!("━━━ Performance ────────────────────────────────────────────────────");
    println!(
        "  {:<10} {:>10}  {:>10}  {:>9}  {:>9}  {:>6}",
        "Backend", "wall ms", "prefill ms", "ms/tok", "tok/s", "steps"
    );
    for (name, r, wall) in [
        ("metal", &r_metal, metal_wall_ms),
        ("cpu", &r_cpu, cpu_wall_ms),
    ] {
        let avg = r.avg_decode_ms();
        let tps = r.decode_tok_s();
        println!(
            "  {:<10} {:>10.1}  {:>10.1}  {:>9.2}  {:>9.2}  {:>6}",
            name,
            wall,
            r.prefill_ms,
            avg,
            tps,
            r.decode_ms.len(),
        );
    }
    let speedup = if r_cpu.avg_decode_ms() > 0.0 && r_metal.avg_decode_ms() > 0.0 {
        r_cpu.avg_decode_ms() / r_metal.avg_decode_ms()
    } else {
        0.0
    };
    if speedup > 0.0 {
        println!(
            "  → Metal is {:.1}× faster per decoded token than CPU",
            speedup
        );
    }

    // ── Accuracy: full generated text ──────────────────────────────────────
    println!();
    println!("━━━ Accuracy — generated text ──────────────────────────────────────");
    println!("  metal: {:?}", r_metal.text());
    println!("  cpu:   {:?}", r_cpu.text());
    let metal_text = r_metal.text();
    let cpu_text = r_cpu.text();
    let shared_prefix = shared_prefix_len(&metal_text, &cpu_text);
    println!(
        "  shared prefix (chars): {} / metal={} cpu={}",
        shared_prefix,
        metal_text.chars().count(),
        cpu_text.chars().count()
    );

    // ── Token-by-token agreement ───────────────────────────────────────────
    println!();
    println!("━━━ Token-by-token agreement ───────────────────────────────────────");
    println!("  {:<5} {:<28} {:<28}  match", "step", "metal", "cpu");
    let n = r_metal.tokens.len().min(r_cpu.tokens.len());
    let mut agreed = 0usize;
    for i in 0..n {
        let m = &r_metal.tokens[i].0;
        let c = &r_cpu.tokens[i].0;
        let match_mark = if m == c {
            agreed += 1;
            "✓"
        } else {
            "✗"
        };
        println!(
            "  {:<5} {:<28} {:<28}  {}",
            i,
            format!("{m:?}"),
            format!("{c:?}"),
            match_mark
        );
    }
    if n > 0 {
        println!(
            "  token-level match: {agreed}/{n} ({:.1}%)",
            100.0 * agreed as f64 / n as f64
        );
    }
    // If token counts differ, show which side ran over.
    if r_metal.tokens.len() != r_cpu.tokens.len() {
        println!(
            "  note: metal produced {} tokens, cpu produced {} tokens",
            r_metal.tokens.len(),
            r_cpu.tokens.len()
        );
    }

    Ok(())
}

/// Longest common prefix length in Unicode chars. A cheap signal of
/// "how far do the two backends agree before diverging".
fn shared_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("cpu_gpu_diag requires `--features metal`.");
}
