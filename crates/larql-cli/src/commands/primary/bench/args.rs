//! Clap-derived CLI args for `larql bench`. Kept in its own file so
//! flag-surface changes don't churn the dispatch logic in `run.rs`.

use clap::Args;

#[derive(Args, Clone)]
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

    /// Comma-separated backend list. Supported: `metal`, `cpu`.
    #[arg(long, default_value = "metal")]
    pub backends: String,

    /// Shorthand for `--backends cpu`.
    #[arg(long)]
    pub cpu: bool,

    /// Also query a local Ollama server on the default port with this
    /// model name (e.g. `gemma3:4b`). Requires `ollama serve` running.
    #[arg(long, value_name = "MODEL")]
    pub ollama: Option<String>,

    /// KV engines to bench alongside the GPU path.
    ///
    /// Supported engine specs (same syntax as `EngineKind::from_name`):
    ///   standard                              — production K/V cache (default)
    ///   standard:window=N                     — sliding-window K/V
    ///   no-cache                              — full re-forward per step (O(N²)); debug
    ///   markov-rs[:window=N]                  — residual-stream replacement
    ///   markov-rs-codec[:window=N]            — markov-rs with bf16 cold tier (2× cold saving)
    ///   unlimited-context:window=N            — per-window K/V checkpoints
    ///   turbo-quant[:bits=3|4]                — WHT + Lloyd-Max codec; experimental
    ///   apollo:layer=N,coef=F,top_k=K         — boundary-residual injection; experimental
    ///   boundary-kv:chunk_tokens=N,sequence_id=S  — Standard + larql-boundary frame emission
    ///
    /// List separator: `;` (preferred) or `,` (legacy). Use `;` when any engine
    /// carries multiple params, since `,` collides with the param separator.
    ///
    /// Example (single param each — `,` is safe):
    ///   `--engine standard,markov-rs:window=512`
    ///
    /// Example (multi-param engine — use `;`):
    ///   `--engine "standard;boundary-kv:chunk_tokens=64,sequence_id=demo"`
    #[arg(long, value_name = "ENGINE[;ENGINE]...")]
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

    /// Route Q4K engine benches through the new `LayerExecutor` surface
    /// (`prefill_quant_via_executor` / `decode_step_quant_via_executor`)
    /// instead of `prefill_quant` / `decode_step_quant`. For migrated
    /// engines this honors the caller-supplied FFN backend — required
    /// for `--ffn http://shard:8080` to actually route through the
    /// remote shard. Unmigrated engines transparently fall through to
    /// the non-executor path via the trait's default impl, so the flag
    /// is safe to set globally.
    ///
    /// Note: as of the 2026-05-17 bypass-removal cut, every per-layer
    /// engine (`markov-rs`, `markov-rs-codec`, `unlimited-context`,
    /// `turbo-quant`, `apollo`, `boundary-per-layer`) always runs its
    /// own state-policy code regardless of this flag. The fused fast
    /// path is exclusive to `standard` / `boundary-kv`. This flag now
    /// only controls FFN-backend honoring; it no longer toggles
    /// "executor vs fused".
    #[arg(long)]
    pub via_executor: bool,

    /// Comma-separated wire formats to compare end-to-end. Requires --ffn.
    /// Supported: f32, f16, i8.
    /// Example: --wire f32,f16,i8
    #[arg(long, value_name = "f32,f16,i8")]
    pub wire: Option<String>,

    /// Run a shard-count scaling sweep.
    /// With --moe-shards: reruns with 1..N shards from the provided map.
    /// With --ffn: runs the same URL 1..3 times (simulated replicas).
    #[arg(long)]
    pub bench_grid: bool,

    /// LAN preregistration matrix (Exp 41): take a JSON config that
    /// lists named `larql bench …` invocations, run each one with the
    /// configured repeats, and emit a JSONL manifest + summary table.
    /// See `experiments/41_residual_transport_grid/config.example.json`
    /// for the schema. When set, all other bench backends are skipped.
    #[arg(long, value_name = "PATH")]
    pub bench_grid_lan: Option<std::path::PathBuf>,

    /// Where to write the grid-lan JSONL manifest + per-run stdout/stderr
    /// captures. Defaults to `<config-dir>/results/`.
    #[arg(long, value_name = "DIR")]
    pub grid_lan_out: Option<std::path::PathBuf>,

    /// Restrict the grid-lan matrix to the named run IDs (repeatable).
    /// Mirrors `run.py --only`.
    #[arg(long = "grid-lan-only", value_name = "ID")]
    pub grid_lan_only: Vec<String>,

    /// Include runs marked `enabled: false` in the JSON config.
    #[arg(long)]
    pub grid_lan_include_disabled: bool,

    /// Dry-run the grid-lan matrix: print the substituted command for
    /// each run but don't spawn anything.
    #[arg(long)]
    pub grid_lan_dry_run: bool,

    /// Exp 41 retry rule: when the per-row CoV across repeats exceeds
    /// this fraction, run up to `--grid-lan-extra-repeats` more times.
    #[arg(long, default_value = "0.15", value_name = "FRAC")]
    pub grid_lan_cov_threshold: f64,

    /// Maximum extra repeats issued when the CoV gate trips.
    #[arg(long, default_value = "2", value_name = "N")]
    pub grid_lan_extra_repeats: u32,

    /// Simulate N concurrent clients. Each runs the full bench independently;
    /// reports aggregate tok/s and per-client p99.
    #[arg(long, default_value = "1", value_name = "N")]
    pub concurrent: usize,

    /// Emit machine-readable JSON alongside the table output.
    /// Supported: json.
    #[arg(long, value_name = "json")]
    pub output: Option<String>,

    /// Write JSON output to this file instead of stdout.
    #[arg(long, value_name = "PATH")]
    pub output_file: Option<String>,

    /// Verbose load / warmup logging.
    #[arg(short, long)]
    pub verbose: bool,

    /// CPU thread count for the rayon pool. `0` (default) auto-selects:
    /// 8 on M3-class Apple silicon (best on memory-channel-saturated
    /// Q4_K matvec), otherwise rayon's default. Honors
    /// `RAYON_NUM_THREADS` if set in the environment.
    #[arg(long, default_value = "0", value_name = "N")]
    pub threads: usize,
}
