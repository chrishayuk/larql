//! Real-model proof for the A5 async-engine opt-in on Gemma 3 4B Q4K.
//!
//! Now that `StandardEngine::prefill_quant` routes through the dispatch
//! trait (via the `kv-dispatch-quantization.md` Phase 1 refactor),
//! this example loads `gemma3-4b-q4k-v2.vindex` and runs N tokens
//! through both `StandardEngine::new` (sync) and
//! `StandardEngine::with_async_backend(CpuBackend)` (async), asserting:
//!
//! 1. Token streams bit-identical between sync and async paths.
//! 2. Async tok/s within ±5% of sync (zero-overhead `Ready*` wrapper).
//!
//! Run:
//! ```text
//! cargo run -p larql-kv --release --example async_parity_real_model -- \
//!     --vindex output/gemma3-4b-q4k-v2.vindex --tokens 16
//! ```

use std::path::PathBuf;
use std::time::Instant;

use larql_compute::CpuBackend;
use larql_inference::forward::hidden_to_raw_logits;
use larql_inference::AsyncComputeBackend;
use larql_kv::engines::standard::StandardEngine;
use larql_kv::KvEngine;
use larql_vindex::SilentLoadCallbacks;

fn parse_args() -> (PathBuf, usize, String) {
    let mut vindex = std::env::var("LARQL_ASYNC_PARITY_VINDEX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("output/gemma3-4b-q4k-v2.vindex"));
    let mut tokens = 16usize;
    let mut prompt = String::from("The capital of France is");

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--vindex" => {
                vindex = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--tokens" => {
                tokens = argv[i + 1].parse().expect("--tokens must be a number");
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    (vindex, tokens, prompt)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn run_engine_q4k(
    engine: &mut dyn KvEngine,
    weights: &mut larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    prompt_tokens: &[u32],
    n_tokens: usize,
) -> (Vec<u32>, f64, f64) {
    let ffn = larql_inference::ffn::NullFfn;

    let t_pre = Instant::now();
    let mut hidden = engine
        .prefill_quant(weights, &ffn, index, prompt_tokens, backend)
        .expect("prefill_quant");
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    let mut emitted = Vec::with_capacity(n_tokens);
    let mut decode_times = Vec::with_capacity(n_tokens);
    let mut last = argmax(&hidden_to_raw_logits(weights, &hidden));

    for _ in 0..n_tokens {
        let t = Instant::now();
        hidden = engine
            .decode_step_quant(weights, &ffn, index, last, backend)
            .expect("decode_step_quant");
        decode_times.push(t.elapsed().as_secs_f64() * 1000.0);
        last = argmax(&hidden_to_raw_logits(weights, &hidden));
        emitted.push(last);
    }

    let mean_decode_ms = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
    (emitted, prefill_ms, mean_decode_ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (vindex_path, n_tokens, prompt) = parse_args();

    if !vindex_path.exists() {
        eprintln!("vindex not found at {}; skipping", vindex_path.display());
        return Ok(());
    }

    eprintln!("Loading Q4K vindex: {}", vindex_path.display());
    let mut cb = SilentLoadCallbacks;
    let mut weights_sync = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut weights_async = larql_vindex::load_model_weights_kquant(&vindex_path, &mut cb)?;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_attn_kquant(&vindex_path)?;
    index.load_interleaved_kquant(&vindex_path)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
    let token_ids =
        larql_inference::encode_prompt(&tokenizer, &*weights_sync.arch, prompt.as_str())
            .map_err(|e| format!("tokenize: {e}"))?;

    let backend_for_compute: Box<dyn larql_compute::ComputeBackend> = Box::new(CpuBackend);

    eprintln!(
        "Prompt tokens: {} ({:?}); decode steps: {n_tokens}",
        token_ids.len(),
        &token_ids[..token_ids.len().min(8)]
    );

    eprintln!("\n== Sync (StandardEngine::new) ==");
    let mut sync_engine = StandardEngine::new(None);
    let (sync_tokens, sync_prefill, sync_decode) = run_engine_q4k(
        &mut sync_engine,
        &mut weights_sync,
        &index,
        backend_for_compute.as_ref(),
        &token_ids,
        n_tokens,
    );
    let sync_tok_per_s = 1000.0 / sync_decode;
    eprintln!(
        "  prefill={sync_prefill:.2} ms  mean_decode={sync_decode:.3} ms  tok/s={sync_tok_per_s:.2}"
    );

    eprintln!("\n== Async (StandardEngine::with_async_backend(CpuBackend)) ==");
    let async_backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
    let mut async_engine = StandardEngine::with_async_backend(None, async_backend);
    let (async_tokens, async_prefill, async_decode) = run_engine_q4k(
        &mut async_engine,
        &mut weights_async,
        &index,
        backend_for_compute.as_ref(),
        &token_ids,
        n_tokens,
    );
    let async_tok_per_s = 1000.0 / async_decode;
    eprintln!(
        "  prefill={async_prefill:.2} ms  mean_decode={async_decode:.3} ms  tok/s={async_tok_per_s:.2}"
    );

    eprintln!("\n== Verifying parity ==");
    assert_eq!(
        sync_tokens.len(),
        async_tokens.len(),
        "token count mismatch"
    );
    for (i, (s, a)) in sync_tokens.iter().zip(async_tokens.iter()).enumerate() {
        assert_eq!(
            s, a,
            "token mismatch at step {i}: sync={s} async={a} — accuracy contract violated"
        );
    }
    eprintln!("  ✓ all {n_tokens} tokens bit-identical between sync and async paths");
    eprintln!(
        "  Emitted tokens: {:?}",
        &sync_tokens[..sync_tokens.len().min(8)]
    );

    let overhead = (async_decode - sync_decode) / sync_decode * 100.0;
    eprintln!(
        "  Δ mean_decode = {overhead:+.2}%  ({sync_decode:.3} ms sync → {async_decode:.3} ms async)"
    );
    if overhead.abs() > 5.0 {
        eprintln!("  ⚠ async overhead exceeds ±5% noise threshold");
        std::process::exit(2);
    } else {
        eprintln!("  ✓ within ±5% noise threshold");
    }

    Ok(())
}
