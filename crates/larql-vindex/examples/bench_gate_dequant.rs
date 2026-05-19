//! Benchmark for dedup #2 — dequantize gate vectors from Q4K on load
//! instead of storing `gate_vectors.bin` separately.
//!
//! In a Q4_K vindex the gate projection lives in two places:
//!
//! 1. `gate_vectors.bin` — the feature-major, f16-or-f32 copy used by
//!    the gate KNN on every `DESCRIBE`/`WALK`/`INFER` call.
//! 2. `interleaved_kquant.bin` — the Q4_K-packed copy used by the FFN
//!    forward pass.
//!
//! These are the same numbers at two different precisions. If startup
//! cost allows it, (1) can be reconstructed from (2) at load time,
//! dropping `gate_vectors.bin` entirely:
//!
//! - 4B q4k: saves ~1.7 GB
//! - 31B q4k: saves ~13.9 GB
//!
//! This benchmark measures the per-layer wall-clock cost of the dequant
//! path so you can decide whether the saving is worth the startup time.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p larql-vindex --example bench_gate_dequant -- \
//!   --vindex path/to/q4k.vindex [--iters 3]
//! ```
//!
//! Requires a vindex extracted with `--quant q4k` (so
//! `interleaved_kquant.bin` + its manifest exist) *and* still carrying
//! `gate_vectors.bin` (so approach A can be measured against it).
//! Every q4k extract today satisfies both.

use std::path::PathBuf;
use std::time::Instant;

use larql_models::quant::{ggml, half};
use larql_vindex::{load_vindex_config, SilentLoadCallbacks, VectorIndex};

fn rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok();
        return out
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|kb| kb as f64 / 1024.0)
            .unwrap_or(0.0);
    }
    #[allow(unreachable_code)]
    0.0
}

fn file_size_gb(p: &std::path::Path) -> f64 {
    std::fs::metadata(p)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0)
}

/// f32 → f16 bytes (how we'd store the dequantised gate in-memory for
/// KNN at half the size of f32). Uses the same encoder the writer uses
/// so precision matches what `gate_vectors.bin` would have stored.
fn pack_as_f16(floats: &[f32]) -> Vec<u8> {
    half::encode_f16(floats)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = PathBuf::new();
    let mut iters: usize = 3;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = PathBuf::from(&args[i]);
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse()?;
            }
            _ => eprintln!("unknown arg: {}", args[i]),
        }
        i += 1;
    }
    if !vindex_path.is_dir() {
        eprintln!(
            "usage: bench_gate_dequant --vindex PATH [--iters N]\n\
             Requires a Q4K vindex containing both gate_vectors.bin and interleaved_kquant.bin.",
        );
        std::process::exit(1);
    }

    let config = load_vindex_config(&vindex_path)?;
    if config.quant != larql_vindex::QuantFormat::Q4K {
        return Err(format!(
            "vindex quant is {}, expected Q4K — this benchmark is Q4K-specific",
            config.quant
        )
        .into());
    }
    let num_layers = config.num_layers;
    let hidden = config.hidden_size;

    println!("== bench_gate_dequant ==");
    println!("  vindex:    {}", vindex_path.display());
    println!("  layers:    {num_layers}");
    println!("  hidden:    {hidden}");
    println!("  iters:     {iters}");

    let gate_path = vindex_path.join("gate_vectors.bin");
    let interleaved_path = vindex_path.join("interleaved_q4k.bin");
    let gate_gb = file_size_gb(&gate_path);
    let interleaved_gb = file_size_gb(&interleaved_path);
    println!("\n  gate_vectors.bin:   {gate_gb:.2} GB   (savings if dropped)");
    println!("  interleaved_kquant.bin: {interleaved_gb:.2} GB  (kept, contains gate slice)");

    // ── Load the index (both gate_vectors and interleaved_kquant must be mmap'd) ──
    let mut cb = SilentLoadCallbacks;
    let rss_before_load = rss_mb();
    let t0 = Instant::now();
    let mut idx = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    idx.load_interleaved_kquant(&vindex_path)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let rss_after_load = rss_mb();
    println!(
        "\nIndex loaded in {load_ms:.1}ms, RSS +{:.1} MB (mmap is cold)",
        rss_after_load - rss_before_load,
    );

    // ── Approach A: read gate_vectors.bin per layer ──
    //
    // Produce a heap f32 buffer per layer via `gate_vectors_flat`
    // (reads the mmap slice, copies to Vec<f32> — equivalent to what
    // the KNN path will naturally pull into hot cache on first use).
    println!("\n── Approach A: load gate from gate_vectors.bin (mmap → f32 buffer) ──");
    let mut a_times = Vec::with_capacity(iters);
    for iter in 0..iters {
        let t = Instant::now();
        let mut sum: f64 = 0.0;
        for layer in 0..num_layers {
            if let Some((data, rows, cols)) = idx.gate_vectors_flat(layer) {
                // Prevent DCE. Touching the first and last elements is
                // enough to guarantee pages are faulted in.
                sum += data[0] as f64 + data.last().copied().unwrap_or(0.0) as f64;
                debug_assert_eq!(rows * cols, data.len());
            }
        }
        let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
        a_times.push(elapsed_ms);
        println!("  iter {iter}: {elapsed_ms:7.1}ms  (checksum {sum:+.4e})");
    }

    // ── Approach B: dequantize gate slice from interleaved_kquant.bin, pack as f16 ──
    println!("\n── Approach B: dequantize Q4K gate per layer → f16 buffer ──");
    let mut b_times = Vec::with_capacity(iters);
    let mut peak_layer_f16_bytes: usize = 0;
    let mut peak_rss_delta: f64 = 0.0;
    for iter in 0..iters {
        let t = Instant::now();
        let mut bytes_produced: usize = 0;
        let rss_start = rss_mb();
        let mut peak_rss_iter: f64 = rss_start;
        for layer in 0..num_layers {
            let layer_data = idx
                .interleaved_kquant_layer_data(layer)
                .ok_or("missing interleaved manifest entry")?;
            let (gate_bytes, gate_format) = layer_data[0];
            if gate_format != "Q4_K" {
                return Err(format!(
                    "expected Q4_K gate format at layer {layer}, got {gate_format}"
                )
                .into());
            }
            let nf = idx.num_features(layer);
            let n = nf * hidden;
            let padded = n.div_ceil(256) * 256;
            let gate_f32 = ggml::dequantize_q4_k(gate_bytes, padded)
                .map_err(|e| format!("layer {layer} dequant: {e}"))?;
            // Pack to f16 — that's how the reconstructed gate_vectors
            // would live in RAM (twice as cheap as f32).
            let gate_f16 = pack_as_f16(&gate_f32[..n]);
            bytes_produced += gate_f16.len();
            peak_layer_f16_bytes = peak_layer_f16_bytes.max(gate_f16.len());
            drop(gate_f16);
            let rss_now = rss_mb();
            if rss_now > peak_rss_iter {
                peak_rss_iter = rss_now;
            }
            // Drop the buffer here — simulating "write to contiguous
            // layer slot in a preallocated in-memory gate_vectors
            // buffer and move on". Real implementation would write into
            // an mmap-anon region directly.
            drop(gate_f32);
        }
        let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
        b_times.push(elapsed_ms);
        peak_rss_delta = peak_rss_delta.max(peak_rss_iter - rss_start);
        println!(
            "  iter {iter}: {elapsed_ms:7.1}ms  (f16 bytes produced: {:.2} GB, peak layer: {:.1} MB)",
            bytes_produced as f64 / (1024.0 * 1024.0 * 1024.0),
            peak_layer_f16_bytes as f64 / (1024.0 * 1024.0),
        );
    }

    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let a_med = median(&mut a_times.clone());
    let b_med = median(&mut b_times.clone());

    println!("\n── Summary ──");
    println!("  A (gate_vectors.bin mmap touch):  median {a_med:7.1}ms");
    println!("  B (Q4K dequant → f16 buffer):     median {b_med:7.1}ms   (peak RSS +{peak_rss_delta:.1} MB)");
    println!(
        "  B − A:  {:+.1}ms startup cost, saves {gate_gb:.2} GB on disk",
        b_med - a_med
    );
    println!(
        "\n  Per-layer avg (approach B): {:.1}ms",
        b_med / num_layers as f64
    );

    Ok(())
}
