//! `larql bench <model>` — end-to-end decode benchmark on a real vindex.
//!
//! Measures prefill + autoregressive decode on a vindex, reports per-stage
//! breakdown (GPU forward / lm_head / norm / embed / detok), and optionally
//! queries a running Ollama server on the same machine for a side-by-side
//! tok/s comparison.
//!
//! This is the real-vindex counterpart of `crates/larql-compute/examples/
//! compare_ollama.rs`, which benchmarks synthetic weights. The synthetic
//! version measures the kernel ceiling; this one measures what an actual
//! decode loop delivers on the vindex bytes shipped by `larql extract`.
//!
//! Flag surface:
//!   <model>          vindex dir, `hf://owner/name`, or cache shorthand.
//!   --prompt STR     prompt to time (default: "The capital of France is").
//!   -n, --tokens N   decode steps to time (default: 50).
//!   --warmup N       decode steps to run first and discard (default: 3).
//!   --backends LIST  comma-separated: `metal`, `cuda`, `cpu`. Default: `metal`.
//!   --ollama MODEL   also query Ollama (e.g. `gemma3:4b`) via localhost.
//!   --ffn URL        bench remote FFN path (attention local, FFN remote).
//!   -v, --verbose

use std::time::Instant;

use clap::Args;
use larql_inference::engines::EngineKind;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct BenchArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Prompt to time. Kept short by default to keep prefill consistent
    /// across runs.
    #[arg(long, default_value = "The capital of France is")]
    pub prompt: String,

    /// Number of decode steps to measure.
    #[arg(short = 'n', long = "tokens", default_value = "50")]
    pub tokens: usize,

    /// Discarded warmup steps before measurement (smooths first-call
    /// allocation / JIT effects in the Metal library).
    #[arg(long, default_value = "3")]
    pub warmup: usize,

    /// Comma-separated backend list. Supported: `metal`, `cuda`, `cpu`.
    #[arg(long, default_value = "metal")]
    pub backends: String,

    /// Also query a local Ollama server on the default port with this
    /// model name (e.g. `gemma3:4b`). Requires `ollama serve` running.
    #[arg(long, value_name = "MODEL")]
    pub ollama: Option<String>,

    /// Comma-separated KV engines to bench alongside the GPU path.
    /// Supported: `markov-rs`, `unlimited-context`.
    /// Example: `--engine markov-rs,unlimited-context`.
    #[arg(long, value_name = "ENGINE,...")]
    pub engine: Option<String>,

    /// Route FFN to a remote larql-server for the bench run.
    /// Attention runs locally on Metal; each layer's FFN is a round trip to
    /// the URL. Use this to bench the grid path for large models like 31B.
    /// Example: `--ffn http://127.0.0.1:8080`
    #[arg(long, value_name = "URL")]
    pub ffn: Option<String>,

    /// HTTP timeout in seconds for --ffn.
    #[arg(long, default_value = "60")]
    pub ffn_timeout_secs: u64,

    /// Dispatch strategy for --ffn.
    ///   streaming  (default) — one HTTP round-trip per layer per token.
    ///   batch      — all layers in parallel (Q8K NEON) per token.
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub ffn_dispatch: String,

    /// Bench the remote MoE expert path (Gemma 4 26B A4B etc.).
    /// Shard map: `"START-END=URL,START-END=URL,..."`.
    /// Example: `--moe-shards "0-63=http://a:8081,64-127=http://b:8082"`
    #[arg(long, value_name = "SHARDS")]
    pub moe_shards: Option<String>,

    /// Dispatch strategy for --moe-shards.
    ///   streaming  (default) — one round-trip per layer per token.
    ///   batch      — all layers in one round-trip per token (approximate).
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub moe_dispatch: String,

    /// Refinement iterations for `--moe-dispatch batch`.
    /// 1 = one dispatch + two Metal passes (fast, approximate).
    /// 2 = two dispatches + three passes (correct answer, ~half the speed).
    #[arg(long, default_value = "2")]
    pub moe_predispatch_iters: usize,

    /// Print per-stage timing breakdown for each engine (markov-rs only for now).
    #[arg(long)]
    pub profile: bool,

    /// Verbose load / warmup logging.
    #[arg(short, long)]
    pub verbose: bool,

    /// Off-the-shelf draft model vindex for speculative decoding.
    /// When set together with `LARQL_SPECULATIVE_DECODE=1`, the bench
    /// loads this vindex as a `SmallModelDrafter` for the speculative
    /// path. When unset (or env flag off), bench runs the existing
    /// non-speculative path bit-exactly.
    ///
    /// Example: `--draft-model /path/to/gemma-3-270m-vindex`
    ///
    /// First slice: drafter is loaded and held by the bench process
    /// (proves the loader works on the target hardware) but not yet
    /// dispatched into the decode loop. Wiring the call site is the
    /// next slice. Until then this flag is purely a "did the drafter
    /// load successfully" check.
    #[arg(long, value_name = "VINDEX_PATH")]
    pub draft_model: Option<String>,
}

struct BenchRow {
    backend: String,
    prefill_ms: f64,
    avg_decode_ms: f64,
    tok_per_s: f64,
    stages: Option<larql_inference::layer_graph::generate::StageTimings>,
    /// Remote FFN path breakdown: average FFN round-trip ms per token.
    ffn_rtt_ms: Option<f64>,
    /// Estimated local attention+norm+lmhead ms per token (= decode - ffn_rtt).
    attn_ms: Option<f64>,
    n_steps: usize,
    note: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LarqlBenchBackend {
    Metal,
    Cuda,
    Cpu,
}

impl LarqlBenchBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Metal => "larql-metal",
            Self::Cuda => "larql-cuda",
            Self::Cpu => "larql-cpu",
        }
    }

    fn request_name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }

    fn is_gpu(self) -> bool {
        matches!(self, Self::Metal | Self::Cuda)
    }
}

pub fn run(args: BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_path = cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display(),
        )
        .into());
    }

    let requested_backends: Vec<&str> = args
        .backends
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let want_metal = requested_backends.contains(&"metal");
    let want_cuda = requested_backends.contains(&"cuda");
    let want_cpu = requested_backends.contains(&"cpu");
    let want_engine = args.engine.is_some();
    let want_ffn = args.ffn.is_some();
    let want_moe = args.moe_shards.is_some();
    if !want_metal
        && !want_cuda
        && !want_cpu
        && args.ollama.is_none()
        && !want_engine
        && !want_ffn
        && !want_moe
    {
        return Err(
            "no backends selected: pass --backends metal,cuda,cpu, --ollama, --engine, --ffn, or --moe-shards".into(),
        );
    }

    println!("larql bench: {}", vindex_path.display());
    println!("Prompt: {:?}", args.prompt);
    println!(
        "Decode: {} tokens after {} warmup; backends={}{}",
        args.tokens,
        args.warmup,
        args.backends,
        args.ollama
            .as_deref()
            .map(|m| format!(", ollama={m}"))
            .unwrap_or_default(),
    );

    // Speculative-decode drafter install — phase 4c task C.2.d (v3
    // dispatch). Loads the draft model if `--draft-model <path>` is
    // given and installs the thread-local drafter + spec config + RNG
    // that `generate_streaming` reads when `LARQL_SPECULATIVE_DECODE=1`.
    //
    // v3 uses the canonical backend's KV cache via decode_token, so it
    // does NOT need the v2-era `SpeculativeTargetExecutor` (a second
    // ModelWeights load). The v2 install was retired in C.2.d.
    if let Some(draft_path) = args.draft_model.as_deref() {
        let resolved = cache::resolve_model(draft_path)?;
        match larql_inference::speculative::SmallModelDrafter::from_vindex(&resolved) {
            Ok(drafter) => {
                let env_on = larql_inference::speculative::enabled();
                larql_inference::speculative::set_thread_drafter(Some(drafter));
                larql_inference::speculative::set_thread_spec_config(
                    larql_inference::speculative::SpecConfig {
                        depth: 2,
                        branches: 1,
                        swa_window: None,
                    },
                );
                larql_inference::speculative::set_thread_rng(0xCAFE_BABE_DEAD_F00D);
                println!(
                    "Speculative drafter: loaded from {} ({}) — env LARQL_SPECULATIVE_DECODE={}",
                    resolved.display(),
                    if env_on {
                        "active (v3 dispatch installed)"
                    } else {
                        "loaded but env disabled"
                    },
                    if env_on { "1" } else { "unset/0" },
                );
            }
            Err(e) => {
                eprintln!(
                    "warning: --draft-model {} failed to load: {e}; continuing without speculative path",
                    resolved.display(),
                );
            }
        }
    }
    println!();

    let mut rows: Vec<BenchRow> = Vec::new();

    // GPU/CPU bench requires Q4K vindex. Skip silently when running engine-only
    // (engines need f32 weights from a non-Q4K vindex).
    let cfg = larql_vindex::load_vindex_config(&vindex_path)?;
    let is_q4k = cfg.quant == larql_vindex::QuantFormat::Q4K;

    if want_metal {
        if is_q4k {
            rows.push(run_larql(&vindex_path, &args, LarqlBenchBackend::Metal)?);
        } else if !want_engine {
            return Err(format!(
                "GPU bench requires a Q4K vindex (got quant={:?}). \
                 Use a q4k vindex for GPU bench, or omit --backends and use --engine only.",
                cfg.quant,
            )
            .into());
        }
    }
    if want_cuda {
        if is_q4k {
            rows.push(run_larql(&vindex_path, &args, LarqlBenchBackend::Cuda)?);
        } else if !want_engine {
            return Err(format!(
                "CUDA bench requires a Q4K vindex (got quant={:?}). \
                 Use a q4k vindex for CUDA bench, or omit --backends and use --engine only.",
                cfg.quant,
            )
            .into());
        }
    }
    if want_cpu {
        if is_q4k {
            rows.push(run_larql(&vindex_path, &args, LarqlBenchBackend::Cpu)?);
        } else if !want_engine {
            return Err(format!(
                "CPU bench requires a Q4K vindex (got quant={:?}).",
                cfg.quant,
            )
            .into());
        }
    }
    if let Some(ref ollama_model) = args.ollama {
        rows.push(run_ollama(ollama_model, &args.prompt, args.tokens));
    }

    // KV engine rows.
    //
    // Q4K vindex → prefill_q4k / decode_step_q4k (Metal pipeline, fast path).
    // f16/f32 vindex → prefill / decode_step (f32 CPU path, slow but correct).
    if let Some(ref engine_list) = args.engine {
        let mut cb = larql_vindex::SilentLoadCallbacks;

        if is_q4k {
            // Fast path: load Q4K weights + Q4K VectorIndex (for attention bytes + WalkFfn FFN).
            let mut weights = larql_vindex::load_model_weights_q4k(&vindex_path, &mut cb)?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
            let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
            index.load_attn_q4k(&vindex_path)?;
            index.load_interleaved_q4k(&vindex_path)?;
            let token_ids =
                larql_inference::encode_prompt(&tokenizer, &*weights.arch, args.prompt.as_str())
                    .map_err(|e| format!("tokenize: {e}"))?;
            let kv_ref_bytes = larql_inference::engines::markov_residual::kv_memory_bytes_for_seq(
                &weights,
                token_ids.len(),
            );

            for engine_name in engine_list
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                match EngineKind::from_name(engine_name) {
                    Some(kind) => {
                        let backend = if want_metal {
                            larql_inference::default_backend()
                        } else {
                            larql_inference::cpu_backend()
                        };
                        rows.push(run_engine_q4k(
                            &mut weights,
                            &index,
                            &token_ids,
                            kv_ref_bytes,
                            kind,
                            backend,
                            &args,
                        )?);
                    }
                    None => eprintln!(
                        "unknown engine {:?} — supported: markov-rs, unlimited-context",
                        engine_name
                    ),
                }
            }
        } else {
            // Slow path: f32 weights (f16 vindex or similar).
            let weights = larql_vindex::load_model_weights(&vindex_path, &mut cb)?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(&vindex_path)?;
            let token_ids =
                larql_inference::encode_prompt(&tokenizer, &*weights.arch, args.prompt.as_str())
                    .map_err(|e| format!("tokenize: {e}"))?;
            let kv_ref_bytes = larql_inference::engines::markov_residual::kv_memory_bytes_for_seq(
                &weights,
                token_ids.len(),
            );

            for engine_name in engine_list
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                match EngineKind::from_name(engine_name) {
                    Some(kind) => {
                        let backend = if want_metal {
                            larql_inference::default_backend()
                        } else {
                            larql_inference::cpu_backend()
                        };
                        rows.push(run_engine(
                            &weights,
                            &token_ids,
                            kv_ref_bytes,
                            kind,
                            backend,
                            &args,
                        )?);
                    }
                    None => eprintln!(
                        "unknown engine {:?} — supported: markov-rs, unlimited-context",
                        engine_name
                    ),
                }
            }
        }
    }

    if let Some(ref ffn_url) = args.ffn {
        rows.push(run_remote_ffn_bench(&vindex_path, &args, ffn_url)?);
    }

    if let Some(ref shards_str) = args.moe_shards {
        rows.push(run_remote_moe_bench(&vindex_path, &args, shards_str)?);
    }

    print_table(&rows);
    Ok(())
}

/// Run the larql generate loop once with the selected backend.
///
/// Warmup runs are discarded; the measured window is `args.tokens` steps
/// AFTER warmup. Because the shared `generate()` doesn't expose a "run
/// N extra steps silently" hook, we run a single call with
/// `max_tokens = warmup + tokens` and subtract. Good enough — the
/// variance between the first-call warmup and later steady-state is
/// absorbed into the discarded prefix.
fn run_larql(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    kind: LarqlBenchBackend,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::layer_graph::generate::generate;
    use larql_inference::layer_graph::CachedLayerGraph;

    if args.verbose {
        eprintln!("[bench] loading vindex for {}…", kind.request_name());
    }

    // Load the vindex once per backend. This mirrors `walk_cmd`'s Q4K
    // path — attention + interleaved Q4K mmaps, weights via the
    // Q4K-specific loader (the plain `load_model_weights` rejects
    // quantised vindexes).
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut q4_index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)?;
    q4_index.load_attn_q4k(vindex_path)?;
    q4_index.load_interleaved_q4k(vindex_path)?;
    if let Err(e) = q4_index.load_lm_head_q4(vindex_path) {
        if args.verbose {
            eprintln!(
                "[bench] lm_head_q4 unavailable for {}: {e}",
                kind.request_name()
            );
        }
    }

    let cfg = larql_vindex::load_vindex_config(vindex_path)?;
    if cfg.quant != larql_vindex::QuantFormat::Q4K {
        return Err(format!(
            "larql bench currently requires a Q4K vindex (got {:?})",
            cfg.quant,
        )
        .into());
    }
    let mut weights = larql_vindex::load_model_weights_q4k(vindex_path, &mut cb)?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)?;
    // Apply chat template so IT models (Gemma 4 31B, etc.) get the same
    // prompt shape as `larql run`. Falls back to raw prompt if wrapping fails
    // (base models, non-IT vindexes without a chat template).
    let wrapped_prompt = larql_inference::chat::render_user_prompt(
        vindex_path,
        weights.arch.family(),
        args.prompt.as_str(),
    )
    .unwrap_or_else(|_| args.prompt.to_string());
    let token_ids: Vec<u32> =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
            .map_err(|e| format!("tokenize: {e}"))?;

    let backend: Box<dyn larql_compute::ComputeBackend> = match kind {
        LarqlBenchBackend::Metal => {
            #[cfg(feature = "metal")]
            {
                let b = larql_compute::metal::MetalBackend::new().ok_or(
                    "Metal backend unavailable — rebuild with `--features metal` on an M-series Mac",
                )?;
                Box::new(b)
            }
            #[cfg(not(feature = "metal"))]
            {
                return Err(
                    "--backends metal requested but binary built without `--features metal`".into(),
                );
            }
        }
        LarqlBenchBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let b = larql_compute::cuda::CudaBackend::new()
                    .map_err(|e| format!("CUDA backend unavailable: {e}"))?;
                Box::new(b)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(
                    "--backends cuda requested but binary built without `--features cuda`".into(),
                );
            }
        }
        LarqlBenchBackend::Cpu => Box::new(larql_compute::CpuBackend),
    };

    let cached_layers = CachedLayerGraph::from_residuals(Vec::new());

    // Pre-warm: one generate call to allocate the KV cache (~1 GB on Gemma 3 4B)
    // and populate the Metal buffer caches. The prefill timer would otherwise
    // include this one-time allocation cost even though it is amortized to zero
    // in real multi-turn usage.
    if kind.is_gpu() {
        let num_layers = weights.num_layers;
        let _ = generate(
            &mut weights,
            &tokenizer,
            &token_ids,
            1,
            &q4_index,
            &*backend,
            &cached_layers,
            0..num_layers,
        );
    }

    if args.profile {
        std::env::set_var("LARQL_PROFILE_SPLIT", "1");
    }
    let max_tokens = args.warmup + args.tokens;
    let num_layers = weights.num_layers;
    let t0 = Instant::now();
    let result = generate(
        &mut weights,
        &tokenizer,
        &token_ids,
        max_tokens,
        &q4_index,
        &*backend,
        &cached_layers,
        0..num_layers,
    );
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Q4_K dequant cache footprint after the run. The full-K Metal fast
    // path streams Q4_K bytes through `q4k_matmul_transb` and should NOT
    // populate this cache; the per-position fallback in walk_ffn/sparse
    // does. Print it on `-v` so the perf audit can verify which path
    // was taken without running vmmap.
    if args.verbose {
        let (slots, bytes) = q4_index.q4k_ffn_cache_stats();
        eprintln!(
            "[bench] q4k_ffn_cache after {}: {} populated slots, {:.1} MB",
            kind.label(),
            slots,
            bytes as f64 / 1_048_576.0,
        );
    }

    let n_warm = args.warmup.min(result.decode_ms.len());
    let measured = &result.decode_ms[n_warm..];
    let measured_n = measured.len();
    let (prefill_ms, avg_decode_ms, tok_per_s) = if measured_n == 0 {
        (result.prefill_ms, 0.0, 0.0)
    } else {
        let avg = measured.iter().sum::<f64>() / measured_n as f64;
        (result.prefill_ms, avg, 1000.0 / avg)
    };

    let backend_name = kind.label();
    let note = if measured_n < args.tokens {
        format!(
            "early stop @{}/{} (EOS or GPU fallback)",
            measured_n, args.tokens
        )
    } else if measured_n == 0 {
        format!("no decode steps completed (wall {:.0}ms)", wall_ms)
    } else {
        String::new()
    };

    // StageTimings across ALL decode steps (including warmup); we'd need
    // to re-architect `generate` to bucket post-warmup only. Report the
    // raw totals and let the caller compute the post-warmup ratio
    // heuristically (~same within noise on 50-token runs).
    let stages = Some(result.stage_timings.avg_per_step(result.decode_ms.len()));

    Ok(BenchRow {
        backend: backend_name.to_string(),
        prefill_ms,
        avg_decode_ms,
        tok_per_s,
        stages,
        ffn_rtt_ms: None,
        attn_ms: None,
        n_steps: measured_n,
        note,
    })
}

/// Run the CPU KV-engine bench path for a single engine kind.
///
/// Runs prefill on `token_ids` then decodes `args.tokens` steps with greedy
/// argmax. Reports prefill time, avg decode time, and engine memory.
fn run_engine(
    weights: &larql_inference::ModelWeights,
    token_ids: &[u32],
    kv_ref_bytes: usize,
    kind: EngineKind,
    backend: Box<dyn larql_inference::ComputeBackend>,
    args: &BenchArgs,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::forward::hidden_to_raw_logits;

    let mut engine = kind.build_with_profiling(backend, args.profile);
    let info = engine.info();
    let label = if info.config.is_empty() {
        format!("{} [{}]", info.name, info.backend)
    } else {
        format!("{} [{}] ({})", info.name, info.backend, info.config)
    };

    if args.verbose {
        eprintln!("[bench] {}", info.summary());
    }

    // Prefill.
    let t_pre = Instant::now();
    let mut hidden = engine
        .prefill(weights, token_ids)
        .ok_or("engine prefill failed")?;
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    // Decode loop: greedy argmax over vocab.
    let max_steps = args.warmup + args.tokens;
    let mut decode_ms_all: Vec<f64> = Vec::with_capacity(max_steps);
    let mut last_token = {
        let logits = hidden_to_raw_logits(weights, &hidden);
        argmax_token(&logits)
    };

    for _ in 0..max_steps {
        let t = Instant::now();
        hidden = engine
            .decode_step(weights, last_token)
            .ok_or("engine decode_step failed")?;
        decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
        last_token = argmax_token(&hidden_to_raw_logits(weights, &hidden));
    }

    let n_warm = args.warmup.min(decode_ms_all.len());
    let measured = &decode_ms_all[n_warm..];
    let measured_n = measured.len();
    let (avg_decode_ms, tok_per_s) = if measured_n == 0 {
        (0.0, 0.0)
    } else {
        let avg = measured.iter().sum::<f64>() / measured_n as f64;
        (avg, 1000.0 / avg)
    };

    // Memory breakdown and compression ratio vs Standard KV (FP16).
    let total_mem = engine.memory_bytes();
    let cold_mem = engine.cold_bytes();
    let hot_mem = total_mem.saturating_sub(cold_mem);
    let ratio = if total_mem > 0 {
        kv_ref_bytes as f64 / total_mem as f64
    } else {
        0.0
    };
    let note = format!(
        "hot={:.1}MB cold={:.1}MB  {:.0}× vs std-kv",
        hot_mem as f64 / 1_048_576.0,
        cold_mem as f64 / 1_048_576.0,
        ratio,
    );

    if args.verbose {
        eprintln!(
            "[bench] {} post-decode: {}",
            info.name,
            engine.info().description
        );
    }
    if args.profile {
        if let Some(summary) = engine.stage_summary() {
            summary.print();
        }
    }

    Ok(BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        n_steps: measured_n,
        note,
    })
}

fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Q4K engine bench: uses `prefill_q4k`/`decode_step_q4k` which route through
/// the Metal pipeline (`decode_token`) for UnlimitedContext and WalkFfn Q4K FFN
/// for MarkovRS — both significantly faster than the f32 path.
fn run_engine_q4k(
    weights: &mut larql_inference::ModelWeights,
    index: &larql_vindex::VectorIndex,
    token_ids: &[u32],
    kv_ref_bytes: usize,
    kind: EngineKind,
    backend: Box<dyn larql_inference::ComputeBackend>,
    args: &BenchArgs,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    // We need two backend instances: one owned by the engine, one for Q4K calls.
    let want_metal_q4k = args.backends.contains("metal");
    let backend_for_q4k: Box<dyn larql_inference::ComputeBackend> = if want_metal_q4k {
        larql_inference::default_backend()
    } else {
        larql_inference::cpu_backend()
    };
    let mut engine = kind.build_with_profiling(backend, args.profile);
    let info = engine.info();
    let label = if info.config.is_empty() {
        format!("{} [{}] Q4K", info.name, info.backend)
    } else {
        format!("{} [{}] ({}) Q4K", info.name, info.backend, info.config)
    };

    if args.verbose {
        eprintln!("[bench] Q4K engine: {}", info.summary());
    }

    use larql_inference::layer_graph::generate::lm_head_topk;
    let be = backend_for_q4k.as_ref();

    // Pick next token via Metal lm_head (matches production path).
    // Defined as a macro-style helper to avoid closure borrow conflicts with &mut weights.
    macro_rules! pick_next {
        ($h:expr) => {{
            let h_1d = ndarray::Array1::from_iter($h.iter().copied());
            lm_head_topk(index, weights, &h_1d, 1, be)
                .first()
                .map(|(t, _)| *t)
                .unwrap_or_else(|| {
                    argmax_token(&larql_inference::forward::hidden_to_raw_logits(weights, $h))
                })
        }};
    }

    // Prefill via Q4K path.
    let t_pre = Instant::now();
    let mut hidden = engine
        .prefill_q4k(weights, index, token_ids, be)
        .ok_or("Q4K engine prefill failed")?;
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

    // Decode loop using Metal lm_head for token selection.
    let max_steps = args.warmup + args.tokens;
    let mut decode_ms_all: Vec<f64> = Vec::with_capacity(max_steps);
    let mut last_token = pick_next!(&hidden);

    for _ in 0..max_steps {
        let t = Instant::now();
        hidden = engine
            .decode_step_q4k(weights, index, last_token, be)
            .ok_or("Q4K engine decode_step failed")?;
        decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
        last_token = pick_next!(&hidden);
    }

    let n_warm = args.warmup.min(decode_ms_all.len());
    let measured = &decode_ms_all[n_warm..];
    let measured_n = measured.len();
    let (avg_decode_ms, tok_per_s) = if measured_n == 0 {
        (0.0, 0.0)
    } else {
        let avg = measured.iter().sum::<f64>() / measured_n as f64;
        (avg, 1000.0 / avg)
    };

    let total_mem = engine.memory_bytes();
    let cold_mem = engine.cold_bytes();
    let hot_mem = total_mem.saturating_sub(cold_mem);
    let ratio = if total_mem > 0 {
        kv_ref_bytes as f64 / total_mem as f64
    } else {
        0.0
    };
    let note = format!(
        "hot={:.1}MB cold={:.1}MB  {:.0}× vs std-kv",
        hot_mem as f64 / 1_048_576.0,
        cold_mem as f64 / 1_048_576.0,
        ratio,
    );

    if args.profile {
        if let Some(summary) = engine.stage_summary() {
            summary.print();
        }
    }

    Ok(BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        n_steps: measured_n,
        note,
    })
}

/// Bench the remote-FFN path: attention runs locally on Metal, FFN is a
/// round-trip to `ffn_url` via `LayerShardedBackend`.
///
/// Reports overall tok/s plus a breakdown:
///   ffn-rtt  — time spent in the remote FFN closure (all layers summed)
///   attn+    — remainder = local attn + norm + lm_head + embed
fn run_remote_ffn_bench(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    ffn_url: &str,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, LayerShardedBackend,
    };
    use std::time::Duration;

    if args.verbose {
        eprintln!("[bench] loading vindex for remote-ffn…");
    }

    let timeout = Duration::from_secs(args.ffn_timeout_secs);
    let backend = larql_compute::default_backend();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_q4k(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index.load_attn_q4k(vindex_path)?;
    index.load_interleaved_q4k(vindex_path)?;
    let _ = index.load_lm_head_q4(vindex_path);

    eprintln!("Connecting to remote FFN at {ffn_url}…");
    let remote = LayerShardedBackend::connect(ffn_url, timeout)
        .map_err(|e| format!("failed to connect to remote FFN: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  FFN:        remote  ({})", ffn_url);

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), &args.prompt)
            .unwrap_or_else(|_| args.prompt.clone());
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("tokenise: {e}"))?;

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let max_tokens = args.warmup + args.tokens;

    let is_batch = args.ffn_dispatch.trim() == "batch";

    // Warmup run — discarded. Amortises TCP connection, Metal init.
    if args.verbose {
        eprintln!("[bench] remote-ffn warmup ({} tokens)…", args.warmup.max(1));
    }
    if is_batch {
        let _ = generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            args.warmup.max(1),
            &index,
            &*backend,
            &remote,
            &eos,
            1,
        );
    } else {
        let _ = generate_with_remote_ffn(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            args.warmup.max(1),
            &index,
            &*backend,
            &remote,
            &eos,
        );
    }

    // Measured run.
    let t_wall = std::time::Instant::now();
    let result = if is_batch {
        generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            1,
        )
        .map_err(|e| format!("remote-ffn generate failed (batch): {e}"))?
    } else {
        generate_with_remote_ffn(
            &weights,
            &tokenizer,
            prompt_ids.clone(),
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
        )
        .map_err(|e| format!("remote-ffn generate failed: {e}"))?
    };
    let _wall_ms = t_wall.elapsed().as_secs_f64() * 1000.0;

    let n_warm = args.warmup.min(result.decode_ms.len());
    let measured_decode = &result.decode_ms[n_warm..];
    let measured_ffn = &result.ffn_rtt_ms[n_warm.min(result.ffn_rtt_ms.len())..];
    let n = measured_decode.len();

    let (prefill_ms, avg_decode_ms, tok_per_s, ffn_rtt_ms, attn_ms) = if n == 0 {
        (0.0, 0.0, 0.0, None, None)
    } else {
        let avg_decode = measured_decode.iter().sum::<f64>() / n as f64;
        let avg_ffn = if measured_ffn.len() == n {
            Some(measured_ffn.iter().sum::<f64>() / n as f64)
        } else {
            None
        };
        let avg_attn = avg_ffn.map(|f| (avg_decode - f).max(0.0));
        (0.0, avg_decode, 1000.0 / avg_decode, avg_ffn, avg_attn)
    };

    let note = if n < args.tokens {
        format!("early stop @{}/{}", n, args.tokens)
    } else {
        String::new()
    };

    let _ = weights; // keep alive

    Ok(BenchRow {
        backend: format!(
            "remote-ffn-{} ({})",
            if is_batch { "batch" } else { "stream" },
            ffn_url
        ),
        prefill_ms,
        avg_decode_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms,
        attn_ms,
        n_steps: n,
        note,
    })
}

/// Bench the remote MoE expert path. Attention + router run locally; expert
/// blocks are dispatched to remote shards via `RemoteMoeBackend`.
///
/// Reports overall tok/s plus a breakdown:
///   expert-rtt  — time spent in remote expert dispatch per token
///   attn+       — remainder = local attn + router + dense FFN
fn run_remote_moe_bench(
    vindex_path: &std::path::Path,
    args: &BenchArgs,
    shards_str: &str,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::ffn::moe_remote::{RemoteMoeBackend, ShardConfig};
    use larql_inference::{generate_with_remote_moe, generate_with_remote_moe_batch};

    // Parse "START-END=URL,..." shard map.
    let mut configs: Vec<ShardConfig> = Vec::new();
    for segment in shards_str.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let mut parts = segment.splitn(2, '=');
        let range_str = parts
            .next()
            .ok_or_else(|| format!("malformed shard segment: {segment:?}"))?;
        let url = parts
            .next()
            .ok_or_else(|| format!("missing URL in shard segment: {segment:?}"))?;
        let (start, end_incl) = ShardConfig::parse_range(range_str)
            .ok_or_else(|| format!("bad expert range {range_str:?} in --moe-shards"))?;
        configs.push(ShardConfig::new(start, end_incl, url));
    }
    if configs.is_empty() {
        return Err("--moe-shards: no valid shard segments".into());
    }

    let num_shards = configs.len();
    let backend = larql_compute::default_backend();
    eprintln!("Connecting to {} MoE shard(s)…", num_shards);
    let remote = RemoteMoeBackend::connect(configs)
        .map_err(|e| format!("failed to connect to MoE shards: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  Router:     local");
    eprintln!(
        "  Experts:    remote  (sharded across {} endpoint{})",
        num_shards,
        if num_shards == 1 { "" } else { "s" }
    );

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_q4k(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index.load_attn_q4k(vindex_path)?;
    index.load_interleaved_q4k(vindex_path)?;
    let _ = index.load_lm_head_q4(vindex_path);

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), &args.prompt)
            .unwrap_or_else(|_| args.prompt.clone());
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("tokenise: {e}"))?;

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let max_tokens = args.warmup + args.tokens;
    let is_batch = args.moe_dispatch.trim() == "batch";
    let iters = args.moe_predispatch_iters.max(1);

    // Warmup.
    let run_once =
        |n: usize| -> Result<larql_inference::layer_graph::grid::GridGenerateResult, String> {
            if is_batch {
                generate_with_remote_moe_batch(
                    &weights,
                    &tokenizer,
                    prompt_ids.clone(),
                    n,
                    &index,
                    &remote,
                    &*backend,
                    &eos,
                    iters,
                )
                .map_err(|e| e.to_string())
            } else {
                generate_with_remote_moe(
                    &weights,
                    &tokenizer,
                    prompt_ids.clone(),
                    n,
                    &index,
                    &remote,
                    &*backend,
                    &eos,
                )
                .map_err(|e| e.to_string())
            }
        };

    let _ = run_once(args.warmup.max(1));

    let result = run_once(max_tokens).map_err(|e| format!("moe bench generate failed: {e}"))?;

    let n_warm = args.warmup.min(result.decode_ms.len());
    let measured = &result.decode_ms[n_warm..];
    let measured_ffn = &result.ffn_rtt_ms[n_warm.min(result.ffn_rtt_ms.len())..];
    let n = measured.len();

    let (avg_decode_ms, tok_per_s, ffn_rtt_ms, attn_ms) = if n == 0 {
        (0.0, 0.0, None, None)
    } else {
        let avg = measured.iter().sum::<f64>() / n as f64;
        let avg_ffn = if measured_ffn.len() == n {
            Some(measured_ffn.iter().sum::<f64>() / n as f64)
        } else {
            None
        };
        let avg_attn = avg_ffn.map(|f| (avg - f).max(0.0));
        (avg, 1000.0 / avg, avg_ffn, avg_attn)
    };

    let note = if n < args.tokens {
        format!("early stop @{}/{}", n, args.tokens)
    } else {
        String::new()
    };

    Ok(BenchRow {
        backend: format!(
            "remote-moe-{} ({} shards)",
            if is_batch { "batch" } else { "stream" },
            num_shards
        ),
        prefill_ms: 0.0,
        avg_decode_ms,
        tok_per_s,
        stages: None,
        ffn_rtt_ms,
        attn_ms,
        n_steps: n,
        note,
    })
}

/// Query a local Ollama server for a one-shot generate at `n` tokens.
/// Reports tok/s based on Ollama's own `eval_duration` / `eval_count`
/// (GPU wall time on its end, excludes HTTP overhead).
fn run_ollama(model: &str, prompt: &str, num_predict: usize) -> BenchRow {
    // Warm up with a small generate to avoid measuring model-load latency.
    let _ = std::process::Command::new("curl")
        .args(["-s", "http://localhost:11434/api/generate",
               "-d", &format!(r#"{{"model":"{model}","prompt":"Hi","stream":false,"options":{{"num_predict":5}}}}"#)])
        .output();

    let body = format!(
        r#"{{"model":"{model}","prompt":"{}","stream":false,"options":{{"num_predict":{num_predict}}}}}"#,
        prompt.replace('"', "\\\""),
    );
    let out = std::process::Command::new("curl")
        .args(["-s", "http://localhost:11434/api/generate", "-d", &body])
        .output()
        .ok();

    let mut row = BenchRow {
        backend: format!("ollama {model}"),
        prefill_ms: 0.0,
        avg_decode_ms: 0.0,
        tok_per_s: 0.0,
        stages: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        n_steps: 0,
        note: "not reachable (ollama serve on :11434?)".into(),
    };

    let o = match out {
        Some(o) => o,
        None => return row,
    };
    let text = String::from_utf8_lossy(&o.stdout);
    let val: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return row,
    };

    // Ollama reports durations in nanoseconds.
    let eval_count = val["eval_count"].as_f64().unwrap_or(0.0);
    let eval_dur_ns = val["eval_duration"].as_f64().unwrap_or(0.0);
    let prompt_dur_ns = val["prompt_eval_duration"].as_f64().unwrap_or(0.0);
    if eval_count > 0.0 && eval_dur_ns > 0.0 {
        let avg_ms = eval_dur_ns / 1e6 / eval_count;
        row.avg_decode_ms = avg_ms;
        row.tok_per_s = 1000.0 / avg_ms;
        row.prefill_ms = prompt_dur_ns / 1e6;
        row.n_steps = eval_count as usize;
        row.note = String::new();
    }
    row
}

fn print_table(rows: &[BenchRow]) {
    println!(
        "  {:<20} {:>10} {:>12} {:>10} {:>6}  notes",
        "Backend", "prefill", "ms/tok", "tok/s", "steps",
    );
    println!("  {}", "─".repeat(78));
    for r in rows {
        println!(
            "  {:<20} {:>9.1}ms {:>10.2}ms {:>9.1}  {:>6}  {}",
            r.backend, r.prefill_ms, r.avg_decode_ms, r.tok_per_s, r.n_steps, r.note,
        );
    }

    // Per-stage breakdown for whichever row has one.
    let stage_row = rows.iter().find(|r| r.stages.is_some());
    if let Some(r) = stage_row {
        let s = r.stages.unwrap();
        let total = s.embed_ms_total
            + s.gpu_ms_total
            + s.norm_ms_total
            + s.lm_head_ms_total
            + s.detok_ms_total;
        if total > 0.0 {
            let pct = |v: f64| (v / total) * 100.0;
            println!();
            println!("  Per-stage average ({}):", r.backend);
            println!(
                "    embed     {:>6.3}ms  ({:>4.1}%)",
                s.embed_ms_total,
                pct(s.embed_ms_total)
            );
            println!(
                "    GPU fwd   {:>6.3}ms  ({:>4.1}%)",
                s.gpu_ms_total,
                pct(s.gpu_ms_total)
            );
            if s.gate_up_ms_total > 0.0 {
                println!(
                    "      gate+up {:>6.3}ms  ({:>4.1}%)",
                    s.gate_up_ms_total,
                    pct(s.gate_up_ms_total)
                );
                println!(
                    "      act+down{:>6.3}ms  ({:>4.1}%)",
                    s.down_ms_total,
                    pct(s.down_ms_total)
                );
            }
            println!(
                "    final_norm{:>6.3}ms  ({:>4.1}%)",
                s.norm_ms_total,
                pct(s.norm_ms_total)
            );
            println!(
                "    lm_head   {:>6.3}ms  ({:>4.1}%)",
                s.lm_head_ms_total,
                pct(s.lm_head_ms_total)
            );
            println!(
                "    detok     {:>6.3}ms  ({:>4.1}%)",
                s.detok_ms_total,
                pct(s.detok_ms_total)
            );
        }
    }

    // Remote FFN breakdown for whichever row has it.
    let ffn_row = rows.iter().find(|r| r.ffn_rtt_ms.is_some());
    if let Some(r) = ffn_row {
        let ffn = r.ffn_rtt_ms.unwrap();
        let attn = r.attn_ms.unwrap_or(r.avg_decode_ms);
        let total = r.avg_decode_ms;
        let pct = |v: f64| {
            if total > 0.0 {
                (v / total) * 100.0
            } else {
                0.0
            }
        };
        println!();
        println!(
            "  Per-stage average (remote-ffn, {} layers × RTT):",
            r.backend.split('(').next().unwrap_or("").trim()
        );
        println!(
            "    attn+norm+lmhead {:>7.2}ms  ({:>4.1}%)",
            attn,
            pct(attn)
        );
        println!(
            "    ffn round-trips  {:>7.2}ms  ({:>4.1}%)  ← remote",
            ffn,
            pct(ffn)
        );
        println!(
            "    total/tok        {:>7.2}ms  →  {:.1} tok/s",
            total, r.tok_per_s
        );
    }

    // Top-line comparison: larql vs ollama, if both present.
    let metal = rows
        .iter()
        .find(|r| r.backend == "larql-metal" && r.tok_per_s > 0.0);
    let ollama = rows
        .iter()
        .find(|r| r.backend.starts_with("ollama") && r.tok_per_s > 0.0);
    if let (Some(m), Some(o)) = (metal, ollama) {
        println!();
        let ratio = m.tok_per_s / o.tok_per_s;
        let (verb, sign) = if ratio >= 1.0 {
            ("faster", '>')
        } else {
            ("slower", '<')
        };
        println!(
            "  → larql-metal is {:.2}× {} {} ollama ({:.1} {} {:.1} tok/s)",
            if ratio >= 1.0 { ratio } else { 1.0 / ratio },
            verb,
            sign,
            m.tok_per_s,
            sign,
            o.tok_per_s,
        );
    }
}
