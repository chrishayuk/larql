//! `larql run <model> [prompt]` — ollama-style one-shot inference / chat.
//!
//! Wraps the richer `larql dev walk --predict` pipeline behind a slim flag
//! set. If a prompt is given, runs one forward pass and prints the top-N
//! predictions. If no prompt is given, drops into a stdin chat loop — one
//! line in, one forward pass out, repeat until EOF.
//!
//! Flag surface:
//!   <model>         required; vindex directory, `hf://owner/name`, or a
//!                   cache shorthand (e.g. `gemma-3-4b-it-vindex`).
//!   [prompt]        optional; enters chat mode if omitted.
//!   -n, --top N     number of predictions to show (default 10).
//!   --ffn URL       route FFN to a remote larql-server.
//!   --experts       enable WASM-expert dispatch (gcd, base64, …).
//!   --experts-dir   directory of `.wasm` experts (overrides default lookup).
//!   -v, --verbose
//!
//! All other walk tuning (top-K, layers, compare, metal opt-in) lives
//! under `larql dev walk` for power users.

use larql_vindex::format::filenames::*;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use clap::Args;

use crate::commands::extraction::walk_cmd;
use crate::commands::primary::cache;

/// Legacy `--kv-cache` flag enum. Retained for backward compatibility;
/// each variant resolves to an `EngineKind` in
/// `walk_cmd::generate_stream`:
///
/// | `--kv-cache` value | `EngineKind` |
/// |---|---|
/// | `standard` (default) | `Standard { window_size: None }` |
/// | `markov-bounded` | `Standard { window_size: Some(--context-window) }` |
/// | `none` | `NoCache` |
///
/// New callers should prefer `--engine SPEC` / `LARQL_KV_ENGINE` instead
/// — they accept the full engine catalog (MarkovResidual, UnlimitedContext,
/// TurboQuant, Apollo) not just the three legacy cache strategies.
/// See `crates/larql-inference/docs/specs/kv-engine-unification.md` §6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheKind {
    /// → `EngineKind::Standard { window_size: None }`. Full FP32 K/V per
    /// layer, unbounded growth. Correct over any context length.
    Standard,
    /// → `EngineKind::Standard { window_size: Some(--context-window) }`.
    /// Sliding window — keep only the last `context_window` positions.
    /// Memory stays O(window). Older tokens drop off the back of the
    /// cache (StreamingLLM-style).
    MarkovBounded,
    /// → `EngineKind::NoCache`. Re-runs full forward over the growing
    /// sequence every step. O(N²) wall time. Correctness fallback.
    None,
}

pub fn parse_kv_cache(s: &str) -> Result<KvCacheKind, String> {
    match s.to_lowercase().as_str() {
        "standard" | "full" | "fp32" => Ok(KvCacheKind::Standard),
        "markov-bounded" | "markov" | "bounded" | "sliding" => Ok(KvCacheKind::MarkovBounded),
        "none" | "off" => Ok(KvCacheKind::None),
        _ => Err(format!(
            "unknown kv-cache strategy: {s} \
             (expected: standard, markov-bounded, none)"
        )),
    }
}

#[derive(Args)]
pub struct RunArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Prompt text. Omit to enter chat mode (line-by-line stdin).
    pub prompt: Option<String>,

    /// Maximum number of tokens to generate autoregressively. Set to
    /// 1 for single-token "what comes next" behavior.
    ///
    /// Uses a CPU KV cache (prefill captures K/V per layer, decode
    /// step attends new Q against cached K/V + new K/V). On
    /// Gemma 3 4B f32 that's ~0.5-0.6 s/token — ollama-shaped.
    /// Q4K CPU path still uses the no-cache loop (slow); prefer
    /// `--metal` for Q4K speed.
    #[arg(short = 'n', long = "max-tokens", default_value = "64")]
    pub max_tokens: usize,

    /// KV cache strategy for autoregressive decode (legacy flag).
    ///
    ///   standard         — Full FP32 K/V, unbounded. Correct over any
    ///                      context length. Memory grows O(context).
    ///   markov-bounded   — Sliding window. Keep the last N positions'
    ///                      K/V, evict older. Memory O(window). Attention
    ///                      only sees the last N tokens — older drops off.
    ///   none             — No cache. Re-runs full forward per decode
    ///                      step (O(N²) total). Useful for correctness
    ///                      checks; unusable for long outputs.
    ///
    /// Each value maps to an `EngineKind` internally (see `KvCacheKind`
    /// docs). For the full engine catalog (MarkovResidual,
    /// UnlimitedContext, TurboQuant, Apollo), use `--engine` instead.
    #[arg(long, default_value = "standard", value_parser = parse_kv_cache)]
    pub kv_cache: KvCacheKind,

    /// Sliding-window size when `--kv-cache markov-bounded`. Ignored
    /// otherwise. `0` = unbounded (same as `standard`).
    #[arg(long, default_value = "0")]
    pub context_window: usize,

    /// KV engine spec, overrides `--kv-cache` when set. Accepts the same
    /// syntax `larql bench --engine` parses:
    ///
    ///   standard                    — production K/V cache (default)
    ///   standard:window=1024        — sliding-window K/V
    ///   no-cache                    — full re-forward per step (O(N²))
    ///   markov-rs[:window=N]        — residual-stream replacement
    ///   unlimited-context:window=N  — per-window K/V checkpoints
    ///   turbo-quant[:bits=3|4]      — WHT + Lloyd-Max codec
    ///   apollo:layer=N,coef=F,top_k=K — boundary-residual injection (bench-only)
    ///
    /// Falls back to the `LARQL_KV_ENGINE` env var when unset, and to
    /// the `--kv-cache` mapping when both are absent. CLI flag wins over
    /// env var; env var wins over `--kv-cache`. See
    /// `crates/larql-inference/docs/specs/kv-engine-unification.md`.
    #[arg(long, value_name = "SPEC")]
    pub engine: Option<String>,

    /// Show the top-K prediction table for each step instead of just
    /// the argmax. Implied by `--verbose`.
    #[arg(long, default_value = "1")]
    pub top: usize,

    /// Route FFN to a remote larql-server (e.g. `http://127.0.0.1:8080`).
    /// Attention runs locally; each layer's FFN is a round trip to the URL.
    #[arg(long, value_name = "URL")]
    pub ffn: Option<String>,

    /// HTTP timeout in seconds for --ffn.
    #[arg(long, default_value = "60")]
    pub ffn_timeout_secs: u64,

    /// Dense FFN dispatch strategy when `--ffn` is set.
    ///
    ///   streaming  (default) — 60 sequential round-trips per decode token,
    ///              one per layer.  Exact: each layer's FFN input uses the
    ///              correct h_post_attn from the previous layer.
    ///
    ///   batch      — parallel predispatch: all 60 layers fired in parallel
    ///              threads, then injected in a second Metal pass.
    ///              Approximate but much faster: wall time ≈ one HTTP round
    ///              trip instead of 60.  Combine with
    ///              `--ffn-predispatch-iters 2` for better accuracy.
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub ffn_dispatch: String,

    /// Number of predispatch iterations per token when `--ffn-dispatch batch`
    /// is set.  1 (default) = one parallel dispatch + two Metal passes;
    /// 2 = two dispatches + three passes, more accurate.
    #[arg(long, default_value = "1", value_name = "N")]
    pub ffn_predispatch_iters: usize,

    /// Use Metal GPU backend for Q4K inference (macOS only).
    #[arg(long)]
    pub metal: bool,

    /// Verbose load / timing output.
    #[arg(short, long)]
    pub verbose: bool,

    /// Enable WASM-expert dispatch. The model is prompted to emit a structured
    /// op-call (`{"op":"...","args":{...}}`); the parser extracts it and the
    /// matching expert (gcd, base64, sql, …) computes the answer.
    ///
    /// Requires Metal (`--metal`) on macOS. Use `--experts-dir` to point at a
    /// custom WASM build directory; otherwise the default lookup is used.
    #[arg(long)]
    pub experts: bool,

    /// Override the WASM experts directory. Defaults to the workspace build
    /// dir at `crates/larql-experts/target/wasm32-wasip1/release/`, or
    /// `$LARQL_EXPERTS_DIR` if set.
    #[arg(long, value_name = "DIR")]
    pub experts_dir: Option<PathBuf>,

    /// Restrict `--experts` to a comma-separated subset of op names. The
    /// system prompt enumerates only these ops, which dramatically improves
    /// weak / mid-sized models' ability to pick the right op. Example:
    /// `--ops gcd,is_prime,factorial,to_roman`.
    #[arg(long, value_name = "OP1,OP2,...", value_delimiter = ',')]
    pub ops: Vec<String>,

    /// Constrain the op-name field of generated `{"op":"...","args":{...}}`
    /// to a prefix of one of the advertised op names. Forces weak models to
    /// pick a real op instead of hallucinating (`gcdd`, `to_number`, etc.).
    /// Slightly slower per token; large reliability win on small Q4K models.
    #[arg(long)]
    pub constrained: bool,

    /// MoE expert shard map: `"START-END=URL,START-END=URL,..."`
    ///
    /// Enables remote expert dispatch for hybrid-MoE models (e.g. Gemma 4 26B-A4B).
    /// Each segment maps an inclusive expert-ID range to a shard server URL.
    ///
    ///   larql serve output/gemma4-26b-a4b-q4k.vindex --experts 0-63 --port 8081
    ///   larql serve output/gemma4-26b-a4b-q4k.vindex --experts 64-127 --port 8082
    ///   larql run   output/gemma4-26b-a4b-q4k.vindex \
    ///               --moe-shards "0-63=http://localhost:8081,64-127=http://localhost:8082" \
    ///               "The capital of France is"
    ///
    /// Client loads attention + dense-FFN + router weights locally (~2 GB).
    /// Expert weights (4 MB × experts_owned × layers) stay on the shard servers.
    /// Router runs locally per layer; top-K expert residuals are dispatched in
    /// parallel to the owning shard(s) via `POST /v1/expert/batch`.
    #[arg(long, value_name = "SHARDS")]
    pub moe_shards: Option<String>,

    /// Path to a JSON manifest for fine-grained per-(layer, expert) shard
    /// ownership.  Format:
    ///
    /// ```json
    /// { "shards": [
    ///     { "url": "grpc://hostA:9081",
    ///       "layer_experts": {"0": [[0,31]], "1": [[0,15]]} },
    ///     { "url": "grpc://hostB:9082",
    ///       "layer_experts": {"0": [[32,63]], "1": [[16,31]]} }
    ///   ] }
    /// ```
    ///
    /// Each shard owns an explicit `(layer, expert_id)` set instead of a
    /// layer-uniform expert range — pairs naturally with the server's
    /// `--units PATH` flag.  Mutually exclusive with `--moe-shards`.
    #[arg(long, value_name = "PATH")]
    pub moe_units_manifest: Option<std::path::PathBuf>,

    /// MoE dispatch strategy when `--moe-shards` is set.
    ///
    ///   streaming  (default) — one gRPC stream per shard, 30 sequential
    ///              round-trips per decode token.  Exact: each layer's expert
    ///              input uses the correct h_post_attn.
    ///
    ///   batch      — parallel batch dispatch: all layers in one round trip,
    ///              approximate.  Combine with `--moe-predispatch-iters 2` for
    ///              better accuracy.
    #[arg(long, default_value = "streaming", value_name = "streaming|batch")]
    pub moe_dispatch: String,

    /// Number of predispatch iterations per token when `--moe-dispatch batch`
    /// is set.  1 (default) = one dispatch + two passes; 2 = two dispatches +
    /// three passes.  Each additional iteration improves routing accuracy by
    /// incorporating prior expert contributions into h_post_attn before
    /// re-routing, at the cost of one extra remote round-trip per token.
    #[arg(long, default_value = "1", value_name = "N")]
    pub moe_predispatch_iters: usize,
}

pub fn run(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_path = cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display()
        )
        .into());
    }

    if args.experts {
        return experts::run(&vindex_path, &args);
    }

    if let Some(ref ffn_url) = args.ffn {
        let prompt = args.prompt.as_deref().ok_or(
            "--ffn requires a prompt argument (chat mode not yet supported with --ffn-dispatch batch)",
        )?;
        return run_with_remote_ffn(
            &vindex_path,
            prompt,
            ffn_url,
            args.ffn_timeout_secs,
            args.max_tokens,
            &args.ffn_dispatch,
            args.ffn_predispatch_iters,
        );
    }

    if args.moe_shards.is_some() && args.moe_units_manifest.is_some() {
        return Err(
            "--moe-shards and --moe-units-manifest are mutually exclusive — \
             use --moe-shards for layer-uniform expert ranges, \
             --moe-units-manifest for per-(layer, expert) ownership"
                .into(),
        );
    }
    if args.moe_shards.is_some() || args.moe_units_manifest.is_some() {
        let prompt = args.prompt.as_deref().ok_or(
            "--moe-shards / --moe-units-manifest requires a prompt argument \
             (chat mode not yet supported)",
        )?;
        return run_with_moe_shards(
            &vindex_path,
            prompt,
            args.moe_shards.as_deref(),
            args.moe_units_manifest.as_deref(),
            args.max_tokens,
            &args.moe_dispatch,
            args.moe_predispatch_iters,
        );
    }

    if let Some(prompt) = args.prompt.as_deref() {
        run_once(&vindex_path, prompt, &args)
    } else {
        run_chat(&vindex_path, &args)
    }
}

/// One forward pass on `prompt`, print predictions, return.
fn run_once(
    vindex_path: &std::path::Path,
    prompt: &str,
    args: &RunArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let walk_args = build_walk_args(vindex_path, prompt, args);
    walk_cmd::run(walk_args)
}

/// REPL loop: read a line from stdin, run a forward pass, print, repeat.
/// EOF (Ctrl-D) exits cleanly. Empty lines are skipped.
fn run_chat(
    vindex_path: &std::path::Path,
    args: &RunArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "larql chat — {} (Ctrl-D to exit)",
        vindex_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model")
    );
    let stdin = io::stdin();
    let mut out = io::stderr();
    loop {
        write!(out, "> ")?;
        out.flush()?;

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                eprintln!();
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(Box::new(e)),
        }
        let prompt = line.trim();
        if prompt.is_empty() {
            continue;
        }

        let walk_args = build_walk_args(vindex_path, prompt, args);
        if let Err(e) = walk_cmd::run(walk_args) {
            eprintln!("Error: {e}");
        }
    }
}

/// Build a `WalkArgs` with sensible defaults from the slim `RunArgs`. The
/// fields we don't surface to end users get stable defaults here.
fn build_walk_args(
    vindex_path: &std::path::Path,
    prompt: &str,
    args: &RunArgs,
) -> walk_cmd::WalkArgs {
    walk_cmd::WalkArgs {
        prompt: prompt.to_string(),
        index: Some(vindex_path.to_path_buf()),
        model: None,
        gate_vectors: None,
        down_vectors: None,
        top_k: usize::MAX,
        max_tokens: args.max_tokens,
        kv_cache: args.kv_cache,
        context_window: args.context_window,
        engine: args.engine.clone(),
        layers: None,
        predict_top_k: args.top,
        predict: true,
        compare: false,
        down_top_k: 5,
        verbose: args.verbose,
        metal: args.metal,
        ffn_remote: args.ffn.clone(),
        ffn_remote_timeout_secs: args.ffn_timeout_secs,
        ffn_dispatch: args.ffn_dispatch.clone(),
        ffn_predispatch_iters: args.ffn_predispatch_iters,
    }
}

/// `--moe-shards` dispatch path.
///
/// Metal runs attention + dense FFN on GPU (same as normal `larql run --metal`).
/// MoE expert blocks are dispatched to remote mini-processes via binary
/// `POST /v1/expert/batch` instead of running locally.
fn run_with_moe_shards(
    vindex_path: &std::path::Path,
    prompt: &str,
    shards_str: Option<&str>,
    units_manifest: Option<&std::path::Path>,
    max_tokens: usize,
    dispatch: &str,
    predispatch_iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::ffn::moe_remote::{parse_unit_manifest, RemoteMoeBackend, ShardConfig};
    use larql_inference::{generate_with_remote_moe, generate_with_remote_moe_batch};

    // Pick ownership mode: legacy `--moe-shards` (layer-uniform ranges) or
    // `--moe-units-manifest` (fine-grained per-(layer, expert) sets).  The
    // mutually-exclusive guard at the caller means at most one is set here.
    let configs: Vec<ShardConfig> = if let Some(path) = units_manifest {
        let cfgs = parse_unit_manifest(path).map_err(|e| format!("--moe-units-manifest: {e}"))?;
        if cfgs.is_empty() {
            return Err("--moe-units-manifest: manifest contains no shards".into());
        }
        eprintln!(
            "Loaded {} shard(s) from unit manifest at {}",
            cfgs.len(),
            path.display()
        );
        cfgs
    } else if let Some(s) = shards_str {
        // Parse "START-END=URL,START-END=URL,..." into Vec<ShardConfig>.
        let mut cfgs: Vec<ShardConfig> = Vec::new();
        for segment in s.split(',') {
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
            cfgs.push(ShardConfig::new(start, end_incl, url));
        }
        if cfgs.is_empty() {
            return Err("--moe-shards: no valid shard segments found".into());
        }
        cfgs
    } else {
        return Err("internal error: run_with_moe_shards called with neither flag".into());
    };

    let num_shards = configs.len();
    // Initialise compute backend early so we can report it in the topology banner.
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

    // Client loads attn + dense FFN + norms + router weights — no expert bytes.
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index
        .load_attn_kquant(vindex_path)
        .map_err(|e| format!("failed to load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(vindex_path)
        .map_err(|e| format!("failed to load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(vindex_path);

    // Prompt-shape options (centralised in `larql_inference::chat::render_user_prompt`):
    //   default              → chat_template.jinja with auto-injected default system prompt for Gemma 4
    //   LARQL_RAW_PROMPT=1   → raw user string with <bos> prepended (no template)
    //   LARQL_THINKING=1     → enable_thinking=true (skips empty thought block)
    //   LARQL_SYSTEM=<text>  → explicit system message
    //   LARQL_NO_DEFAULT_SYSTEM=1 → suppress the auto-injected Gemma 4 default
    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), prompt)?;
    if std::env::var("LARQL_DUMP_PROMPT").is_ok() {
        let mode = if std::env::var("LARQL_RAW_PROMPT").is_ok() {
            "raw"
        } else if std::env::var("LARQL_THINKING").is_ok() {
            "thinking"
        } else {
            "default"
        };
        eprintln!(
            "[chat] mode={mode} ---PROMPT START---\n{wrapped_prompt}\n[chat] ---PROMPT END---"
        );
    }
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("failed to tokenise prompt: {e}"))?;
    eprintln!("[chat] tokenised to {} ids", prompt_ids.len());

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let result = if dispatch == "batch" {
        generate_with_remote_moe_batch(
            &weights,
            &tokenizer,
            prompt_ids,
            max_tokens,
            &index,
            &remote,
            &*backend,
            &eos,
            predispatch_iters,
        )
    } else {
        generate_with_remote_moe(
            &weights, &tokenizer, prompt_ids, max_tokens, &index, &remote, &*backend, &eos,
        )
    }
    .map_err(|e| format!("grid generate failed ({dispatch}): {e}"))?;

    for tok in &result.tokens {
        print!("{tok}");
    }
    if !result.tokens.is_empty() {
        println!();
    }
    let n = result.decode_ms.len();
    if n > 0 {
        let avg = result.decode_ms.iter().sum::<f64>() / n as f64;
        let tok_s = 1000.0 / avg;
        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let top_k = weights.arch.num_experts_per_token();
        let experts_invoked = num_layers * top_k * n;
        // One f32 residual vector per layer per shard in each direction.
        let bytes_per_token = num_layers * num_shards * hidden * std::mem::size_of::<f32>();
        let kb = |b: usize| b as f64 / 1024.0;
        eprintln!();
        eprintln!("  decode:          {tok_s:.1} tok/s");
        eprintln!(
            "  experts invoked: {experts_invoked}  ({num_layers} layers × top-{top_k} × {n} token{})",
            if n == 1 { "" } else { "s" }
        );
        eprintln!(
            "  bytes sent:      ~{:.0} KB  ({num_layers} layers × {num_shards} shard{} × hidden × f32)",
            kb(bytes_per_token * n),
            if num_shards == 1 { "" } else { "s" }
        );
        eprintln!(
            "  bytes recv:      ~{:.0} KB  ({num_layers} layers × {num_shards} shard{} × hidden × f32)",
            kb(bytes_per_token * n),
            if num_shards == 1 { "" } else { "s" }
        );
    }
    Ok(())
}

/// `--ffn URL` dispatch path for dense models.
///
/// Metal runs attention on the local GPU. Every layer's FFN is a round trip
/// to the remote server at `ffn_url` via `LayerShardedBackend`. The local
/// vindex supplies attention weights; the remote server supplies FFN outputs.
///
/// This is analogous to `run_with_moe_shards` for hybrid-MoE models, but
/// simpler: there is no local FFN and no router — every layer unconditionally
/// calls the remote server.
fn run_with_remote_ffn(
    vindex_path: &std::path::Path,
    prompt: &str,
    ffn_url: &str,
    ffn_timeout_secs: u64,
    max_tokens: usize,
    dispatch: &str,
    predispatch_iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use larql_inference::{
        generate_with_remote_ffn, generate_with_remote_ffn_batch, LayerShardedBackend,
    };
    use std::time::Duration;

    let timeout = Duration::from_secs(ffn_timeout_secs);
    let backend = larql_compute::default_backend();
    eprintln!("Connecting to remote FFN at {ffn_url}…");
    let remote = LayerShardedBackend::connect(ffn_url, timeout)
        .map_err(|e| format!("failed to connect to remote FFN server: {e}"))?;
    eprintln!("  Attention:  {} (local)", backend.name());
    eprintln!("  FFN:        remote  ({})  dispatch={dispatch}", ffn_url);

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load client weights: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;
    let mut index = larql_vindex::VectorIndex::load_vindex(vindex_path, &mut cb)
        .map_err(|e| format!("failed to load vindex: {e}"))?;
    index
        .load_attn_kquant(vindex_path)
        .map_err(|e| format!("failed to load attn Q4K: {e}"))?;
    index
        .load_interleaved_kquant(vindex_path)
        .map_err(|e| format!("failed to load interleaved Q4K: {e}"))?;
    let _ = index.load_lm_head_kquant(vindex_path);

    let wrapped_prompt =
        larql_inference::chat::render_user_prompt(vindex_path, weights.arch.family(), prompt)?;
    let prompt_ids = larql_inference::encode_prompt(&tokenizer, &*weights.arch, &wrapped_prompt)
        .map_err(|e| format!("failed to tokenise prompt: {e}"))?;
    eprintln!("[chat] tokenised to {} ids", prompt_ids.len());

    let eos = larql_inference::layer_graph::generate::eos::EosConfig::from_vindex_dir(vindex_path);
    let result = if dispatch == "batch" {
        generate_with_remote_ffn_batch(
            &weights,
            &tokenizer,
            prompt_ids,
            max_tokens,
            &index,
            &*backend,
            &remote,
            &eos,
            predispatch_iters,
        )
    } else {
        generate_with_remote_ffn(
            &weights, &tokenizer, prompt_ids, max_tokens, &index, &*backend, &remote, &eos,
        )
    }
    .map_err(|e| format!("remote-ffn generate failed ({dispatch}): {e}"))?;

    for tok in &result.tokens {
        print!("{tok}");
    }
    if !result.tokens.is_empty() {
        println!();
    }
    let n = result.decode_ms.len();
    if n > 0 {
        let avg = result.decode_ms.iter().sum::<f64>() / n as f64;
        let tok_s = 1000.0 / avg;
        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        // One f32 residual in each direction per layer.
        let bytes_per_token = num_layers * hidden * std::mem::size_of::<f32>();
        let kb = |b: usize| b as f64 / 1024.0;
        eprintln!();
        eprintln!("  decode:     {tok_s:.1} tok/s");
        eprintln!(
            "  bytes sent: ~{:.0} KB  ({num_layers} layers × hidden × f32)",
            kb(bytes_per_token * n)
        );
        eprintln!(
            "  bytes recv: ~{:.0} KB  ({num_layers} layers × hidden × f32)",
            kb(bytes_per_token * n)
        );
    }
    Ok(())
}

/// `--experts` wiring: load registry, wrap prompt, generate, dispatch.
///
/// Self-contained — does not call into `walk_cmd` because we need the raw
/// generated text for op-call extraction (walk_cmd streams to stdout).
///
/// Backend matrix:
///
/// | vindex quant | `--metal` | strategy                                    |
/// |--------------|-----------|---------------------------------------------|
/// | Q4_K         | yes       | `layer_graph::generate` (KV-cached, fast)   |
/// | Q4_K         | no        | `vindex::generate_kquant_cpu` (per-step, slow) |
/// | f32          | any       | `forward::generate_cached` (CPU, F32)       |
///
/// Chat mode (no prompt): drops into a stdin REPL over the same loaded model.
mod experts {
    use super::*;
    use larql_inference::experts::{
        DispatchOutcome, DispatchSkip, Dispatcher, ExpertRegistry, ExpertSession,
        FilteredDispatcher, OpNameMask,
    };
    use larql_inference::prompt::ChatTemplate;
    use larql_inference::WeightFfn;
    use larql_vindex::{load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex};

    type BoxErr = Box<dyn std::error::Error>;

    /// Which decode strategy to use for this `--experts` invocation.
    enum Strategy {
        /// Q4_K vindex + Metal backend. KV-cached decode via `layer_graph::generate`.
        MetalQ4K,
        /// Q4_K vindex, no Metal. Loops `predict_kquant` per token (O(N²)).
        CpuQ4K,
        /// Non-quantised vindex. CPU `generate_cached` with full f32 weights.
        CpuF32,
    }

    impl Strategy {
        fn name(&self) -> &'static str {
            match self {
                Self::MetalQ4K => "metal-q4k",
                Self::CpuQ4K => "cpu-q4k",
                Self::CpuF32 => "cpu-f32",
            }
        }
    }

    /// Resolved runtime — model + index + chosen strategy. Lives across
    /// REPL turns so loads only happen once.
    struct Runtime {
        weights: larql_inference::ModelWeights,
        tokenizer: tokenizers::Tokenizer,
        index: Option<VectorIndex>,
        strategy: Strategy,
    }

    /// Teacher-forced prefix that drops the model into the op-name field
    /// of an op-call JSON immediately. Used by `--constrained`.
    const OP_CALL_PREFIX: &str = r#"{"op":""#;

    impl Runtime {
        /// Generate text from `wrapped`. When `mask_op_names` is `Some`,
        /// constrained decoding (a) injects [`OP_CALL_PREFIX`] into the
        /// prompt as teacher-forcing and (b) restricts the op-name field
        /// of the generated text to a prefix of one of those op names.
        /// `None` is unconstrained generation.
        ///
        /// Returns the generated text. When constrained, the returned
        /// string includes the teacher-forced prefix so downstream
        /// `parse_op_call` sees a complete `{"op":"..."}` block.
        fn generate(
            &mut self,
            wrapped: &str,
            max_tokens: usize,
            mask_op_names: Option<&[String]>,
        ) -> Result<String, BoxErr> {
            // Teacher-force the JSON prefix when constrained — the model
            // never has to "decide" to emit the op-call.
            let effective_prompt: String = if mask_op_names.is_some() {
                format!("{wrapped}{OP_CALL_PREFIX}")
            } else {
                wrapped.to_string()
            };
            let token_ids = larql_inference::encode_prompt(
                &self.tokenizer,
                &*self.weights.arch,
                &effective_prompt,
            )
            .map_err(|e| format!("tokenize: {e}"))?;

            let text = match self.strategy {
                Strategy::MetalQ4K => {
                    let index = self.index.as_ref().expect("metal-q4k needs index");
                    let backend = larql_compute::default_backend();
                    let cached_layers =
                        larql_inference::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
                    let num_layers = self.weights.num_layers;
                    let result = if let Some(ops) = mask_op_names {
                        let mut mask = OpNameMask::new(ops.to_vec(), &self.tokenizer);
                        mask.set_seed_text(OP_CALL_PREFIX);
                        larql_inference::layer_graph::generate_constrained(
                            &mut self.weights,
                            &self.tokenizer,
                            &token_ids,
                            max_tokens,
                            index,
                            &*backend,
                            &cached_layers,
                            0..num_layers,
                            |ids, logits| mask.apply(ids, logits),
                        )
                    } else {
                        larql_inference::layer_graph::generate(
                            &mut self.weights,
                            &self.tokenizer,
                            &token_ids,
                            max_tokens,
                            index,
                            &*backend,
                            &cached_layers,
                            0..num_layers,
                        )
                    };
                    result.tokens.iter().map(|(t, _)| t.as_str()).collect()
                }
                Strategy::CpuQ4K => {
                    let index = self.index.as_ref().expect("cpu-q4k needs index");
                    let toks = if let Some(ops) = mask_op_names {
                        let mut mask = OpNameMask::new(ops.to_vec(), &self.tokenizer);
                        mask.set_seed_text(OP_CALL_PREFIX);
                        larql_inference::vindex::generate_kquant_cpu_constrained(
                            &mut self.weights,
                            &self.tokenizer,
                            &token_ids,
                            max_tokens,
                            index,
                            |ids, logits| mask.apply(ids, logits),
                        )
                    } else {
                        larql_inference::vindex::generate_kquant_cpu(
                            &mut self.weights,
                            &self.tokenizer,
                            &token_ids,
                            max_tokens,
                            index,
                        )
                    };
                    toks.into_iter().map(|(t, _)| t).collect()
                }
                Strategy::CpuF32 => {
                    let ffn = WeightFfn {
                        weights: &self.weights,
                    };
                    let mut text = String::new();
                    if let Some(ops) = mask_op_names {
                        let mut mask = OpNameMask::new(ops.to_vec(), &self.tokenizer);
                        mask.set_seed_text(OP_CALL_PREFIX);
                        larql_kv::generation::generate_cached_constrained(
                            &self.weights,
                            &self.tokenizer,
                            &ffn,
                            &token_ids,
                            max_tokens,
                            |ids, logits| mask.apply(ids, logits),
                            |_id, tok| text.push_str(tok),
                        );
                    } else {
                        larql_kv::generation::generate_cached(
                            &self.weights,
                            &self.tokenizer,
                            &ffn,
                            &token_ids,
                            max_tokens,
                            |_id, tok| text.push_str(tok),
                        );
                    }
                    text
                }
            };
            // When constrained, prepend the teacher-forced prefix so the
            // dispatcher sees a complete op-call JSON block.
            let result = if mask_op_names.is_some() {
                format!("{OP_CALL_PREFIX}{text}")
            } else {
                text
            };
            Ok(result)
        }
    }

    /// Locate the WASM experts directory.
    ///
    /// Search order:
    ///   1. `--experts-dir <PATH>` flag (if provided).
    ///   2. `LARQL_EXPERTS_DIR` env var.
    ///   3. Workspace build dir relative to the running CLI binary location.
    fn resolve_experts_dir(args: &RunArgs) -> Result<PathBuf, BoxErr> {
        resolve_experts_dir_inner(
            args.experts_dir.clone(),
            std::env::var("LARQL_EXPERTS_DIR").ok().map(PathBuf::from),
            std::env::current_exe().ok(),
        )
    }

    /// Pure version of [`resolve_experts_dir`] — env var + current exe are
    /// passed in. Lets unit tests exercise the precedence chain without
    /// mutating shared process state.
    fn resolve_experts_dir_inner(
        arg_dir: Option<PathBuf>,
        env_dir: Option<PathBuf>,
        exe_path: Option<PathBuf>,
    ) -> Result<PathBuf, BoxErr> {
        if let Some(p) = arg_dir {
            if !p.is_dir() {
                return Err(format!("--experts-dir does not exist: {}", p.display()).into());
            }
            return Ok(p);
        }
        if let Some(path) = env_dir {
            if path.is_dir() {
                return Ok(path);
            }
        }
        if let Some(exe) = exe_path {
            for ancestor in exe.ancestors() {
                let candidate = ancestor.join("crates/larql-experts/target/wasm32-wasip1/release");
                if candidate.is_dir() {
                    return Ok(candidate);
                }
            }
        }
        Err(
            "could not locate WASM experts directory; pass --experts-dir or set LARQL_EXPERTS_DIR"
                .into(),
        )
    }

    /// Detect the chat template from a vindex.
    ///
    /// Vindexes ship their family in `index.json` (no `config.json` — that
    /// only exists in raw safetensors directories), so we read it directly.
    /// Falls back to [`larql_models::detect_architecture`] for non-vindex
    /// model dirs, then to `Plain` if neither resolves.
    fn detect_template(vindex_path: &Path) -> ChatTemplate {
        // Try vindex index.json first.
        let index_path = vindex_path.join(INDEX_JSON);
        if let Ok(text) = std::fs::read_to_string(&index_path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(family) = value.get("family").and_then(|v| v.as_str()) {
                    return ChatTemplate::for_family(family);
                }
                // Fall back to model id → for_model_id heuristic if family is absent.
                if let Some(id) = value.get("model").and_then(|v| v.as_str()) {
                    return ChatTemplate::for_model_id(id);
                }
            }
        }
        // Fall back to safetensors-style config.json detection.
        match larql_models::detect_architecture(vindex_path) {
            Ok(arch) => ChatTemplate::for_family(arch.family()),
            Err(_) => ChatTemplate::Plain,
        }
    }

    /// Whether the active compute backend can serve Q4 work-sets via Metal.
    /// Wraps the impure `default_backend()` call so [`pick_strategy`] stays pure.
    fn metal_ready_for_q4(want_metal: bool) -> bool {
        want_metal
            && larql_compute::default_backend().supports_quant(::larql_compute::QuantFormat::Q4_K)
    }

    /// Pure strategy selector: given the vindex quant format and whether
    /// Metal is available + requested, pick a decode strategy.
    fn pick_strategy(quant: larql_vindex::QuantFormat, metal_ready: bool) -> Strategy {
        match (quant, metal_ready) {
            (larql_vindex::QuantFormat::Q4K, true) => Strategy::MetalQ4K,
            (larql_vindex::QuantFormat::Q4K, false) => Strategy::CpuQ4K,
            _ => Strategy::CpuF32,
        }
    }

    /// Load the runtime: weights + tokenizer + q4 index (when needed).
    fn load_runtime(vindex_path: &Path, args: &RunArgs) -> Result<Runtime, BoxErr> {
        let mut cb = SilentLoadCallbacks;
        let cfg = larql_vindex::load_vindex_config(vindex_path)?;
        let strategy = pick_strategy(cfg.quant, metal_ready_for_q4(args.metal));

        if args.verbose {
            eprintln!(
                "strategy: {} (quant={:?}, metal_requested={})",
                strategy.name(),
                cfg.quant,
                args.metal
            );
        }

        let (weights, index) = match strategy {
            Strategy::MetalQ4K | Strategy::CpuQ4K => {
                let weights = larql_vindex::load_model_weights_kquant(vindex_path, &mut cb)?;
                let mut idx = VectorIndex::load_vindex(vindex_path, &mut cb)?;
                idx.load_attn_kquant(vindex_path)?;
                idx.load_interleaved_kquant(vindex_path)?;
                let _ = idx.load_lm_head_kquant(vindex_path);
                (weights, Some(idx))
            }
            Strategy::CpuF32 => {
                let weights = larql_vindex::load_model_weights_with_opts(
                    vindex_path,
                    &mut cb,
                    larql_vindex::LoadWeightsOptions::default(),
                )?;
                (weights, None)
            }
        };
        let tokenizer = load_vindex_tokenizer(vindex_path)?;
        Ok(Runtime {
            weights,
            tokenizer,
            index,
            strategy,
        })
    }

    /// Print a single dispatch outcome (or skip reason) to stdout/stderr.
    fn print_dispatch(
        model_output: &str,
        outcome: Result<DispatchOutcome, DispatchSkip>,
    ) -> Result<(), BoxErr> {
        match outcome {
            Ok(DispatchOutcome { call, result }) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "op": call.op,
                        "args": call.args,
                        "value": result.value,
                        "expert_id": result.expert_id,
                    })
                );
                Ok(())
            }
            Err(DispatchSkip::NoOpCall) => {
                eprintln!("no op-call extracted; raw output:");
                println!("{model_output}");
                Ok(())
            }
            Err(DispatchSkip::UnknownOp(op)) => {
                Err(format!("model emitted unknown op `{op}`; raw output: {model_output}").into())
            }
            Err(DispatchSkip::ExpertDeclined { op, args }) => Err(format!(
                "expert `{op}` declined args {args}; raw output: {model_output}"
            )
            .into()),
        }
    }

    pub fn run(vindex_path: &Path, args: &RunArgs) -> Result<(), BoxErr> {
        // ── Load experts ──
        let experts_dir = resolve_experts_dir(args)?;
        if args.verbose {
            eprintln!("experts: loading from {}", experts_dir.display());
        }
        let registry = ExpertRegistry::load_dir(&experts_dir)?;
        if args.verbose {
            eprintln!(
                "experts: loaded {} modules ({} ops)",
                registry.len(),
                registry.ops().len()
            );
        }

        // Optionally narrow the registry to a focused subset — small models
        // pick the right op far more reliably with 5–15 options than 126.
        let dispatcher: Box<dyn larql_inference::experts::Dispatcher> = if args.ops.is_empty() {
            Box::new(registry)
        } else {
            if args.verbose {
                eprintln!("experts: filtering to {} ops", args.ops.len());
            }
            Box::new(FilteredDispatcher::new(registry, args.ops.clone()))
        };
        let mut session = ExpertSession::new(dispatcher);

        // ── Detect template + load model ──
        let template = detect_template(vindex_path);
        if args.verbose {
            eprintln!("template: {}", template.name());
        }
        let mut runtime = load_runtime(vindex_path, args)?;

        if let Some(prompt) = args.prompt.as_deref() {
            run_one(&mut session, &mut runtime, prompt, template, args)
        } else {
            run_chat(&mut session, &mut runtime, template, args)
        }
    }

    /// Single dispatch: wrap → generate → dispatch → print.
    fn run_one(
        session: &mut ExpertSession<Box<dyn larql_inference::experts::Dispatcher>>,
        runtime: &mut Runtime,
        prompt: &str,
        template: ChatTemplate,
        args: &RunArgs,
    ) -> Result<(), BoxErr> {
        let wrapped = session.build_prompt(prompt, template);
        let mask_op_names: Option<Vec<String>> = if args.constrained {
            Some(
                session
                    .registry()
                    .op_specs()
                    .into_iter()
                    .map(|s| s.name)
                    .collect(),
            )
        } else {
            None
        };
        let model_output = runtime.generate(&wrapped, args.max_tokens, mask_op_names.as_deref())?;
        if args.verbose {
            eprintln!("model output: {model_output:?}");
        }
        print_dispatch(&model_output, session.dispatch(&model_output))
    }

    /// REPL: read line → run_one → repeat. Loads model exactly once.
    fn run_chat(
        session: &mut ExpertSession<Box<dyn larql_inference::experts::Dispatcher>>,
        runtime: &mut Runtime,
        template: ChatTemplate,
        args: &RunArgs,
    ) -> Result<(), BoxErr> {
        eprintln!("larql experts chat — Ctrl-D to exit");
        let stdin = io::stdin();
        let mut stderr = io::stderr();
        loop {
            write!(stderr, "> ")?;
            stderr.flush()?;
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => {
                    eprintln!();
                    return Ok(());
                }
                Ok(_) => {}
                Err(e) => return Err(Box::new(e)),
            }
            let prompt = line.trim();
            if prompt.is_empty() {
                continue;
            }
            // Per-turn errors don't kill the REPL — print and continue.
            if let Err(e) = run_one(session, runtime, prompt, template, args) {
                eprintln!("error: {e}");
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use larql_vindex::QuantFormat;

        // ── pick_strategy ──────────────────────────────────────────────────

        #[test]
        fn pick_strategy_q4k_with_metal_picks_metal() {
            assert!(matches!(
                pick_strategy(QuantFormat::Q4K, true),
                Strategy::MetalQ4K
            ));
        }

        #[test]
        fn pick_strategy_q4k_without_metal_picks_cpu_q4k() {
            assert!(matches!(
                pick_strategy(QuantFormat::Q4K, false),
                Strategy::CpuQ4K
            ));
        }

        #[test]
        fn pick_strategy_non_q4k_with_metal_falls_back_to_f32() {
            // Metal can't help with non-Q4K weights — backend has no f32 path.
            assert!(matches!(
                pick_strategy(QuantFormat::None, true),
                Strategy::CpuF32
            ));
        }

        #[test]
        fn pick_strategy_non_q4k_without_metal_picks_cpu_f32() {
            assert!(matches!(
                pick_strategy(QuantFormat::None, false),
                Strategy::CpuF32
            ));
        }

        // ── resolve_experts_dir_inner ──────────────────────────────────────

        #[test]
        fn resolve_arg_dir_when_valid() {
            let dir = tempfile::tempdir().expect("tempdir");
            let p = dir.path().to_path_buf();
            let resolved = resolve_experts_dir_inner(Some(p.clone()), None, None).expect("ok");
            assert_eq!(resolved, p);
        }

        #[test]
        fn resolve_arg_dir_invalid_errors() {
            let bogus = PathBuf::from("/this/path/does/not/exist/xyz");
            let err = resolve_experts_dir_inner(Some(bogus.clone()), None, None).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("--experts-dir does not exist"), "got: {msg}");
            assert!(
                msg.contains(bogus.to_str().unwrap()),
                "msg should name the path; got: {msg}"
            );
        }

        #[test]
        fn resolve_falls_through_to_env_dir() {
            let env = tempfile::tempdir().expect("tempdir");
            let resolved =
                resolve_experts_dir_inner(None, Some(env.path().to_path_buf()), None).expect("ok");
            assert_eq!(resolved, env.path());
        }

        #[test]
        fn resolve_skips_invalid_env_dir_falls_to_workspace_walk() {
            // env dir doesn't exist; workspace walk must then succeed.
            // Build a fake "exe" inside a workspace-shaped tempdir tree.
            let root = tempfile::tempdir().expect("tempdir");
            let wasm_dir = root
                .path()
                .join("crates/larql-experts/target/wasm32-wasip1/release");
            std::fs::create_dir_all(&wasm_dir).unwrap();
            // exe is conceptually somewhere inside root, e.g. target/debug/larql.
            let exe = root.path().join("target/debug/larql");
            std::fs::create_dir_all(exe.parent().unwrap()).unwrap();

            let resolved = resolve_experts_dir_inner(
                None,
                Some(PathBuf::from("/nonexistent/env/dir")),
                Some(exe),
            )
            .expect("ok");
            assert_eq!(
                resolved.canonicalize().unwrap(),
                wasm_dir.canonicalize().unwrap()
            );
        }

        #[test]
        fn resolve_returns_error_when_nothing_resolves() {
            let err = resolve_experts_dir_inner(
                None,
                Some(PathBuf::from("/nope/env")),
                Some(PathBuf::from("/nope/exe")),
            )
            .unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("could not locate"), "got: {msg}");
            assert!(
                msg.contains("--experts-dir"),
                "should hint at the flag; got: {msg}"
            );
        }

        // ── print_dispatch ─────────────────────────────────────────────────

        #[test]
        fn print_dispatch_unknown_op_errors() {
            let outcome = Err(DispatchSkip::UnknownOp("foo".into()));
            let err = print_dispatch("raw model output", outcome).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("unknown op `foo`"), "got: {msg}");
            assert!(
                msg.contains("raw model output"),
                "should include raw output; got: {msg}"
            );
        }

        #[test]
        fn print_dispatch_expert_declined_errors() {
            let outcome = Err(DispatchSkip::ExpertDeclined {
                op: "gcd".into(),
                args: serde_json::json!({"bad": true}),
            });
            let err = print_dispatch("output", outcome).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("expert `gcd` declined"), "got: {msg}");
        }

        #[test]
        fn print_dispatch_no_op_call_succeeds() {
            // No op-call is a soft case — print raw, return Ok.
            let outcome = Err(DispatchSkip::NoOpCall);
            assert!(print_dispatch("free text", outcome).is_ok());
        }
    }
}
