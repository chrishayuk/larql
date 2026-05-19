//! Streaming demo — print each token as the model emits it.
//!
//! Demonstrates [`generate_streaming`]'s `on_token` callback. Each token
//! is printed live with stdout flushed after every write so the user sees
//! the response unfold rather than appearing all at once at the end.
//!
//! Compare against `bench_generate.rs` which collects the full result
//! before printing — the buffered version completes faster wall-clock
//! but the streaming version delivers visible tokens with the same
//! latency profile as Ollama / llama.cpp.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference \
//!     --example streaming_demo -- --vindex output/gemma3-4b-q4k-v2.vindex
//!
//! Optional flags:
//!   --prompt "<text>"   (default: "The capital of France is")
//!   --max-tokens N      (default: 32)
//!   --temperature F     (default: 0.0 = greedy)
//!   --top-p F           (default: not applied)
//!   --top-k N           (default: not applied)
//!   --seed N            (default: 42 if any sampling flag is set)
//!   --model HF_ID       (override; default reads it from vindex index.json)
//!
//! The model architecture (layer count, head dims, etc.) comes from the HF
//! model name. If you point `--vindex` at a non-4B vindex without overriding
//! `--model`, the example used to panic on `attn Q4K slices missing for
//! layer N` because the loaded arch had a different layer count than the
//! vindex shipped. The `--model` flag (or `index.json`'s `model` field)
//! keeps the two in sync.

use std::io::Write;
use std::time::Instant;

use larql_inference::{
    default_backend, encode_prompt, generate_streaming, open_inference_vindex, wrap_chat_prompt,
    CachedLayerGraph, EosConfig, SamplingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-q4k-v2.vindex");
    let mut prompt = "The capital of France is".to_string();
    let mut max_tokens = 32usize;
    let mut temperature: f32 = 0.0;
    let mut top_p: Option<f32> = None;
    let mut top_k: Option<usize> = None;
    let mut seed: u64 = 42;
    let mut model_override: Option<String> = None;

    let args: Vec<String> = std::env::args().collect();
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
            "--temperature" => {
                i += 1;
                temperature = args[i].parse()?;
            }
            "--top-p" => {
                i += 1;
                top_p = Some(args[i].parse()?);
            }
            "--top-k" => {
                i += 1;
                top_k = Some(args[i].parse()?);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    let mut sampling = SamplingConfig::temperature(temperature);
    if let Some(p) = top_p {
        sampling = sampling.with_top_p(p);
    }
    if let Some(k) = top_k {
        sampling = sampling.with_top_k(k);
    }
    if !sampling.is_greedy() {
        sampling = sampling.with_seed(seed);
    }

    // Load weights, tokenizer, and arch directly from the vindex — same
    // path the `larql parity` tool uses. Earlier this loaded HF weights
    // via `InferenceModel::load(<hardcoded model name>)`, which had two
    // failure modes on non-4B vindexes: (a) `weights.num_layers` came
    // from the HF arch (e.g. 34 for 4B) and panicked when the vindex
    // only shipped 30 layers; (b) the HF f32 norms didn't match the
    // vindex's transformed `norms.bin`, producing first-token gibberish
    // on the same input that parity decoded as "Paris". The vindex's
    // `index.json` carries the canonical model name; pass `--model` to
    // override.
    let config = larql_vindex::load_vindex_config(&vindex_path)?;
    let model_name: String = model_override.unwrap_or(config.model.clone());

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut weights = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
    let num_layers = weights.num_layers;

    let index = open_inference_vindex(&vindex_path)?;

    let gpu_be = default_backend();

    // Apply the chat template when the model is instruction-tuned. The
    // bare-prompt path works for Gemma 3 4B, but Gemma 4 26B-A4B-it (and
    // any other `-it` / `-instruct` variant) trained only on chat-wrapped
    // sequences emits multilingual gibberish on raw prompts. `wrap_chat_prompt`
    // reads `vindex/chat_template.jinja` first, falls back to model-name
    // hints, and finally passes through unchanged for base models.
    let wrapped = wrap_chat_prompt(&vindex_path, Some(&model_name), &prompt);
    let token_ids: Vec<u32> = encode_prompt(&tokenizer, &*weights.arch, &wrapped.prompt)?;
    // No precomputed cache — stream the full transformer end-to-end. The
    // earlier `CachedLayerGraph::build` over `(0..=12)` + generate range
    // `13..num_layers` is invalid for any model whose layers 0-12 contribute
    // anything beyond a dense FFN (hybrid-MoE in particular: the cache built
    // from `WeightFfn` would skip every MoE expert block in those layers and
    // produce multilingual gibberish). Match the convention used by
    // `walk_cmd` and `bench_generate`: empty cache, full layer range.
    let cache = CachedLayerGraph::from_residuals(Vec::new());
    let eos = EosConfig::from_vindex_dir(&vindex_path);

    println!("=== larql-inference: Streaming Demo ===\n");
    println!("Model:       {model_name} ({num_layers} layers)");
    println!("Vindex:      {}", vindex_path.display());
    println!("Prompt:      \"{prompt}\"");
    println!("Sampling:    {sampling:?}");
    println!("Max tokens:  {max_tokens}");
    println!("Backend:     {}\n", gpu_be.name());
    print!("Output:      ");
    std::io::stdout().flush().ok();

    let start = Instant::now();
    let result = generate_streaming(
        &mut weights,
        &tokenizer,
        &token_ids,
        max_tokens,
        &index,
        &*gpu_be,
        &cache,
        0..num_layers,
        sampling,
        &eos,
        |_id, text, _prob| {
            print!("{text}");
            std::io::stdout().flush().ok();
        },
    );
    let wall = start.elapsed().as_secs_f64();
    println!("\n");
    println!("(buffered text: \"{}\")", result.text());
    println!("Tokens emitted: {}", result.tokens.len());
    println!(
        "Decode rate:    {:.1} tok/s ({:.1} ms/tok)",
        result.decode_tok_s(),
        result.avg_decode_ms()
    );
    println!(
        "Wall time:      {wall:.2}s (prefill {:.0}ms)",
        result.prefill_ms
    );

    Ok(())
}
