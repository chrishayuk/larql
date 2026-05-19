//! Generate benchmark: prefill + decode timing on a real vindex.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference --example bench_generate -- \
//!     --vindex output/gemma3-4b-q4k-v2.vindex
//!
//! Optional flags:
//!   --prompt "<text>"   (default: "The capital of France is")
//!   --max-tokens N      (default: 20)
//!   --warmup N          (default: 0; discard the first N generated tokens)
//!   --model HF_ID       (override; default reads it from vindex index.json)
//!
//! Like `streaming_demo`, this loads weights + tokenizer + arch from the
//! vindex (`load_model_weights_kquant`) rather than re-downloading the
//! safetensors via `InferenceModel::load`. The vindex's transformed
//! `norms.bin` doesn't match HF's raw norms — using the wrong source
//! produced first-token gibberish on Gemma 4 26B-A4B even though every
//! per-layer residual matched cos=1.0 in the parity diagnostic.

use larql_inference::{
    default_backend, encode_prompt, generate, open_inference_vindex, wrap_chat_prompt,
    CachedLayerGraph,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-q4k-v2.vindex");
    let mut max_tokens = 20usize;
    let mut warmup = 0usize;
    let mut prompt = "The capital of France is".to_string();
    let mut model_override: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex_path = std::path::PathBuf::from(&args[i]);
            }
            "--model" => {
                i += 1;
                model_override = Some(args[i].clone());
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse()?;
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    // Load weights + tokenizer + arch directly from the vindex. See the
    // module-level comment for why `InferenceModel::load(<hf_id>)` is
    // not used here.
    let config = larql_vindex::load_vindex_config(&vindex_path)?;
    let model_name: String = model_override.unwrap_or(config.model.clone());

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
    let num_layers = weights.num_layers;

    let index = open_inference_vindex(&vindex_path)?;
    let gpu_be = default_backend();

    // Apply the chat template for instruction-tuned models — bare-prompt
    // encoding produces multilingual gibberish on `-it` / `-instruct`
    // variants since they're trained only on chat-wrapped sequences.
    let wrapped = wrap_chat_prompt(&vindex_path, Some(&model_name), &prompt);
    let token_ids: Vec<u32> = encode_prompt(&tokenizer, &*weights.arch, &wrapped.prompt)?;

    // Empty cache + full layer range. The earlier
    // `CachedLayerGraph::build(0..=12)` + `generate(13..num_layers)`
    // shortcut is invalid for any model whose layers 0-12 contribute
    // anything beyond a dense FFN: hybrid-MoE in particular skips every
    // expert block in those layers (the cache is built from `WeightFfn`)
    // and emits multilingual gibberish. Match `streaming_demo` /
    // `walk_cmd` instead.
    let cache = CachedLayerGraph::from_residuals(Vec::new());

    println!("╔═══════════════════════════════════════════════╗");
    println!("║       LARQL Generate Benchmark                ║");
    println!("╚═══════════════════════════════════════════════╝");
    println!();
    println!("  Model:   {model_name} ({num_layers} layers)");
    println!("  Vindex:  {}", vindex_path.display());
    println!("  Prompt:  \"{prompt}\" ({} tokens)", token_ids.len());
    println!("  Backend: {}", gpu_be.name());
    println!();

    if warmup > 0 {
        // Discard a short warmup run so JIT compilation, command-buffer
        // pool growth, and KV-cache first-allocation costs don't drag
        // the measured average. Compute-layer benchmarks (78.7 tok/s
        // headline) use 8 warmup + 100 measured.
        let _ = generate(
            &mut weights,
            &tokenizer,
            &token_ids,
            warmup,
            &index,
            &*gpu_be,
            &cache,
            0..num_layers,
        );
    }
    let result = generate(
        &mut weights,
        &tokenizer,
        &token_ids,
        max_tokens,
        &index,
        &*gpu_be,
        &cache,
        0..num_layers,
    );

    println!("  Prefill:       {:.0}ms", result.prefill_ms);
    println!("  Generated:     \"{}\"", result.text());
    println!("  Tokens:        {}", result.tokens.len());
    println!();

    if !result.decode_ms.is_empty() {
        println!("  Decode timing:");
        for (i, ms) in result.decode_ms.iter().enumerate() {
            let tok = &result.tokens[i + 1].0;
            println!(
                "    Token {}: {:>8} {:>7.1}ms  ({:.0} tok/s)",
                i + 1,
                tok,
                ms,
                1000.0 / ms
            );
        }
        println!();
        println!(
            "  Average decode: {:.1}ms/tok = {:.0} tok/s",
            result.avg_decode_ms(),
            result.decode_tok_s()
        );

        // Cold vs warm split — when run with `--warmup 0`, token 1 is
        // the first decode call against a fresh backend and pays any
        // one-time costs (Metal buffer allocation for `MoeScratch`,
        // KV-cache first-touch, mmap fault-in). Tokens 2+ are
        // steady-state. The ratio answers "is the per-decode allocation
        // claim in ROADMAPs (~120ms on Gemma 4 26B A4B) actually
        // amortized after the first token?" — see
        // `crates/larql-vindex/docs/per-layer-ffn-phase2-research.md`.
        if warmup == 0 && result.decode_ms.len() >= 2 {
            let cold = result.decode_ms[0];
            let warm: f64 =
                result.decode_ms[1..].iter().sum::<f64>() / (result.decode_ms.len() - 1) as f64;
            let overhead = cold - warm;
            let ratio = if warm > 0.0 { cold / warm } else { 0.0 };
            println!();
            println!("  Cold vs warm (warmup=0, MoE Phase 2 acceptance check):");
            println!(
                "    Cold (token 1):              {:>7.1}ms  ({:.0} tok/s)",
                cold,
                1000.0 / cold
            );
            println!(
                "    Warm (mean of tokens 2-{}): {:>7.1}ms  ({:.0} tok/s)",
                result.decode_ms.len(),
                warm,
                1000.0 / warm
            );
            println!(
                "    First-token overhead:        {:>7.1}ms  (cold/warm = {:.2}×)",
                overhead, ratio
            );
        }
    }

    println!();
    println!("  ┌───────────────────────────────────────────┐");
    println!(
        "  │ Prefill: {:>6.0}ms (one-time)              │",
        result.prefill_ms
    );
    if result.decode_ms.is_empty() {
        println!("  │ Decode:  (no GPU decode tokens)           │");
    } else {
        println!(
            "  │ Decode:  {:>6.1}ms/tok = {:>3.0} tok/s          │",
            result.avg_decode_ms(),
            result.decode_tok_s()
        );
    }
    // Reference: median of 5×100-tok runs on the same M3 Max against
    // `gemma3:4b` at ollama 0.20 (2026-04-27, gemma3-4b-q4k-v2.vindex).
    // Update via `larql bench <vindex> --ollama gemma3:4b` if the gap
    // closes — the older "117 tok/s" footer was stale by ~25%.
    println!("  │ Ollama:   10.5ms/tok =  95 tok/s (median)  │");
    println!("  └───────────────────────────────────────────┘");

    Ok(())
}
