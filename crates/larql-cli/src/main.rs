#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::type_complexity)]
// Architectural lints suppressed crate-wide:
//
//  - `too_many_arguments`: most command runners have wide signatures
//    (`run(args, output, callbacks, ...)`) reflecting CLI surface, not
//    internal coupling. Reducing to ≤7 args would mean introducing
//    parameter structs across every command, which is a bigger refactor
//    than a lint cleanup.
//  - `large_enum_variant`: a few command-arg enums (notably `OvRdArgs`)
//    have one or two large variants. Boxing them would change call
//    sites everywhere.
//
// Mechanical lints (iter-on-map-keys, ptr_arg, needless_deref,
// map_entry, etc.) are fixed in place rather than suppressed.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::large_enum_variant)]

use clap::{Parser, Subcommand};

mod commands;
mod formatting;
mod utils;

use commands::dev::*;
use commands::diagnostics::*;
use commands::extraction::*;
use commands::primary::*;
use commands::query::*;

#[derive(Parser)]
#[command(
    name = "larql",
    version,
    about = "LARQL — decompile transformer weights into a queryable vindex"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// ══════════════════════════════════════════════════════════════════════
// Top-level commands
//
// Grouped in --help output via `next_help_heading`:
//   * (unspecified)   — Primary user verbs
//   * "Build"         — Extract / compile / publish
//   * "Query"         — Graph file introspection (legacy pre-LQL)
//   * "LQL"           — Query-language surface
//   * "Server"        — Serve a vindex
//   * "Research"      — `larql dev <subcmd>`
// ══════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum Commands {
    // ── Primary user-facing ─────────────────────────────────────────
    /// Run inference (one-shot if prompt is given, chat if not).
    Run(run_cmd::RunArgs),

    /// Interactive chat — alias for `run <model>` with no prompt.
    Chat(ChatArgs),

    /// Download a vindex from HuggingFace and cache it locally.
    Pull(pull_cmd::PullArgs),

    /// Manage HuggingFace *model* repos (safetensors + tokenizer + config).
    /// Companion to `pull` (which is vindex-only). Use `model pull` to
    /// stage a raw HF model for `convert safetensors-to-vindex`.
    Model(model_cmd::ModelArgs),

    /// Register a local vindex directory with the cache so `run` / `list`
    /// / `show` can find it by shorthand.
    Link(link_cmd::LinkArgs),

    /// List cached vindexes.
    List(list_cmd::ListArgs),

    /// Show metadata for a vindex.
    Show(show_cmd::ShowArgs),

    /// Carve a subset of a vindex (client / server / browse / router slice).
    Slice(slice_cmd::SliceArgs),

    /// Publish a vindex to HuggingFace — full vindex plus slice siblings.
    Publish(publish_cmd::PublishArgs),

    /// Remove a cached vindex.
    Rm(rm_cmd::RmArgs),

    /// Benchmark decode throughput on a real vindex (Metal / CPU / Ollama).
    Bench(bench::BenchArgs),

    /// Split-axis accuracy suite — parametric vs in-context vs conflict,
    /// scored with top-1 match and Shannon bits-per-token.
    Accuracy(accuracy_cmd::AccuracyArgs),

    /// Shannon-style next-token bit measurements and demo compression.
    #[command(subcommand)]
    Shannon(shannon_cmd::ShannonCommand),

    // ── Server ──────────────────────────────────────────────────────
    #[command(next_help_heading = "Server")]
    /// Serve a vindex over HTTP + gRPC.
    Serve(ServeArgs),

    // ── LQL ─────────────────────────────────────────────────────────
    #[command(next_help_heading = "LQL")]
    /// Launch the LQL interactive REPL.
    Repl,

    #[command(next_help_heading = "LQL")]
    /// Execute a one-shot LQL statement.
    Lql(LqlArgs),

    // ── Build / extract ─────────────────────────────────────────────
    #[command(next_help_heading = "Build")]
    /// Build a .vindex by decompiling a HuggingFace model.
    Extract(extract_index_cmd::ExtractIndexArgs),

    #[command(next_help_heading = "Build")]
    /// Backwards-compat alias for `extract` (identical behavior).
    ExtractIndex(extract_index_cmd::ExtractIndexArgs),

    #[command(next_help_heading = "Build")]
    /// Build a custom vindex from a Vindexfile (declarative: FROM + PATCH + INSERT).
    Build(build_cmd::BuildArgs),

    #[command(next_help_heading = "Build")]
    /// Compile vindex patches into model weights (AOT compilation).
    Compile(compile_cmd::CompileArgs),

    #[command(next_help_heading = "Build")]
    /// Convert between model formats (GGUF ↔ vindex, safetensors → vindex).
    Convert(convert_cmd::ConvertArgs),

    #[command(next_help_heading = "Build")]
    /// HuggingFace Hub: upload a vindex.
    Hf(hf_cmd::HfArgs),

    #[command(next_help_heading = "Build")]
    /// Verify vindex file integrity (SHA256 checksums).
    Verify(verify_cmd::VerifyArgs),

    #[command(next_help_heading = "Build")]
    /// Engine diagnostic — print which kernel paths fire for a vindex.
    Diag(diag_cmd::DiagArgs),

    #[command(next_help_heading = "Build")]
    /// Cross-backend numerical parity diff (CPU vs Metal vs reference).
    Parity(parity::ParityArgs),

    // ── Query (legacy, pre-LQL graph-file surface) ──────────────────
    #[command(next_help_heading = "Query")]
    /// Query a graph file for facts.
    Query(query_cmd::QueryArgs),

    #[command(next_help_heading = "Query")]
    /// Describe an entity (all edges).
    Describe(describe_cmd::DescribeArgs),

    #[command(next_help_heading = "Query")]
    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    #[command(next_help_heading = "Query")]
    /// Validate a graph file.
    Validate(validate_cmd::ValidateArgs),

    #[command(next_help_heading = "Query")]
    /// Merge multiple graph files.
    Merge(merge_cmd::MergeArgs),

    #[command(next_help_heading = "Query")]
    /// Filter graph edges by confidence, layer, selectivity, relation, source.
    Filter(filter_cmd::FilterArgs),

    // ── Research / power-user tooling ───────────────────────────────
    #[command(next_help_heading = "Research", subcommand)]
    /// Research / interpretability tools (weight-extract, qk-rank, …).
    Dev(DevCommand),
}

// ══════════════════════════════════════════════════════════════════════
// Research subcommand group — `larql dev <subcmd>`.
//
// Everything in here is unchanged from the pre-redesign top-level surface
// except its invocation path. A small argv trampoline in `main()` rewrites
// `larql <legacy-name>` → `larql dev <legacy-name>` so existing scripts
// continue to work without a breaking change.
// ══════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum DevCommand {
    /// Extract edges from FFN weights. Zero forward passes.
    WeightExtract(weight_walk_cmd::WeightWalkArgs),

    /// Extract routing edges from attention OV circuits. Zero forward passes.
    AttentionExtract(attention_walk_cmd::AttentionWalkArgs),

    /// Extract full vectors from model weights to NDJSON files.
    VectorExtract(vector_extract_cmd::VectorExtractArgs),

    /// Capture residual stream vectors for entities via forward passes.
    Residuals(residuals_cmd::ResidualsArgs),

    /// Run full forward pass and predict next token.
    Predict(predict_cmd::PredictArgs),

    /// Build gate index for graph-based FFN (offline, run once per model).
    IndexGates(index_gates_cmd::IndexGatesArgs),

    /// Walk the model as a local vector index — gate KNN + down token lookup.
    Walk(walk_cmd::WalkArgs),

    /// Capture and compare attention patterns across prompts.
    AttentionCapture(attention_capture_cmd::AttentionCaptureArgs),

    /// Extract attention template circuits from QK weight decomposition.
    QkTemplates(qk_templates_cmd::QkTemplatesArgs),

    /// SVD rank analysis of attention QK products.
    QkRank(qk_rank_cmd::QkRankArgs),

    /// Extract interpretable modes from low-rank QK heads via SVD → gate projection.
    QkModes(qk_modes_cmd::QkModesArgs),

    /// Map attention OV circuits to FFN gate features.
    OvGate(ov_gate_cmd::OvGateArgs),

    /// OV rate-distortion and residual-table attention compilation experiments.
    OvRd(ov_rd::cmd::OvRdArgs),

    /// Discover attention → FFN circuits from weight decomposition.
    CircuitDiscover(circuit_discover_cmd::CircuitDiscoverArgs),

    /// Bottleneck analysis of attention components.
    AttnBottleneck(attn_bottleneck_cmd::AttnBottleneckArgs),

    /// Bottleneck analysis of FFN components.
    FfnBottleneck(ffn_bottleneck_cmd::FfnBottleneckArgs),

    /// Measure overlap between entity-routed and ground-truth gate features.
    FfnOverlap(ffn_overlap_cmd::FfnOverlapArgs),

    /// Knowledge graph retrieval benchmark.
    KgBench(kg_bench_cmd::KgBenchArgs),

    /// Trace residual stream trajectories on the sphere across layers.
    TrajectoryTrace(trajectory_trace_cmd::TrajectoryTraceArgs),

    /// Test rank-k projection through the residual stream.
    ProjectionTest(projection_test_cmd::ProjectionTestArgs),

    /// Extract OV fingerprint basis from attention weights.
    FingerprintExtract(fingerprint_extract_cmd::FingerprintExtractArgs),

    /// Test rule-based bottleneck — if-else rules replace early layers.
    BottleneckTest(bottleneck_test_cmd::BottleneckTestArgs),

    /// Embedding jump — raw token embeddings → projected L13 → decoder.
    EmbeddingJump(embedding_jump_cmd::EmbeddingJumpArgs),

    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    /// Measure round-trip latency breakdown against a remote FFN server.
    FfnLatency(ffn_latency_cmd::FfnLatencyArgs),
}

// ══════════════════════════════════════════════════════════════════════
// Minor glue types
// ══════════════════════════════════════════════════════════════════════

#[derive(clap::Args)]
struct ChatArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    model: String,

    /// Max tokens to generate per chat response.
    #[arg(short = 'n', long = "max-tokens", default_value = "64")]
    max_tokens: usize,

    /// Route FFN to a remote larql-server.
    #[arg(long, value_name = "URL")]
    ffn: Option<String>,

    /// HTTP timeout in seconds for --ffn.
    #[arg(long, default_value = "60")]
    ffn_timeout_secs: u64,

    /// Verbose load / timing output.
    #[arg(short, long)]
    verbose: bool,
}

impl From<ChatArgs> for run_cmd::RunArgs {
    fn from(c: ChatArgs) -> Self {
        run_cmd::RunArgs {
            model: c.model,
            prompt: None,
            max_tokens: c.max_tokens,
            top: 1,
            kv_cache: run_cmd::KvCacheKind::Standard,
            context_window: 0,
            engine: None,
            ffn: c.ffn,
            ffn_timeout_secs: c.ffn_timeout_secs,
            metal: false,
            verbose: c.verbose,
            experts: false,
            experts_dir: None,
            ops: Vec::new(),
            constrained: false,
            moe_shards: None,
            moe_units_manifest: None,
            moe_dispatch: "streaming".to_string(),
            moe_predispatch_iters: 1,
            ffn_dispatch: "streaming".to_string(),
            ffn_predispatch_iters: 1,
        }
    }
}

#[derive(clap::Args)]
struct LqlArgs {
    /// LQL statement (e.g. `WALK "The capital of France is" TOP 5;`).
    statement: String,
}

#[derive(clap::Args)]
struct ServeArgs {
    /// Path to a .vindex directory (or `hf://` path).
    #[arg(value_name = "VINDEX_PATH")]
    vindex_path: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    dir: Option<std::path::PathBuf>,

    /// Listen port.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    no_infer: bool,

    /// Run as an FFN-service endpoint for remote clients using
    /// `larql run --ffn URL`. Disables `/v1/infer` and advertises
    /// `mode: ffn-service` in `/v1/stats`. Act 2 of the demo.
    #[arg(long)]
    ffn_only: bool,

    /// Cap decoded f16 gate layers via LRU (bounds server RSS). 0 = unlimited.
    /// On 31B each layer decodes to ~433 MB, so 60 layers = ~26 GB.
    /// Set to N to cap at N layers; evicted layers are re-decoded on access.
    #[arg(long, default_value = "0")]
    max_gate_cache_layers: usize,

    /// madvise(MADV_DONTNEED) on all mmaps after each walk-ffn request.
    /// Enforces a hard RSS bound alongside --max-gate-cache-layers at the
    /// cost of re-fault per request. Prefer --layers sharding for real
    /// deployments (sharding never touches out-of-range pages).
    #[arg(long)]
    release_mmap_after_request: bool,

    /// Enable CORS for browser access.
    #[arg(long)]
    cors: bool,

    /// API key for authentication.
    #[arg(long)]
    api_key: Option<String>,

    /// Rate limit per IP (e.g. "100/min", "10/sec").
    #[arg(long)]
    rate_limit: Option<String>,

    /// Max concurrent requests.
    #[arg(long, default_value = "100")]
    max_concurrent: usize,

    /// Cache TTL for DESCRIBE results in seconds (0 = disabled).
    #[arg(long, default_value = "0")]
    cache_ttl: u64,

    /// gRPC port.
    #[arg(long)]
    grpc_port: Option<u16>,

    /// TLS certificate path.
    #[arg(long)]
    tls_cert: Option<std::path::PathBuf>,

    /// TLS private key path.
    #[arg(long)]
    tls_key: Option<std::path::PathBuf>,

    /// Logging level.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Only load and serve layers in this range (inclusive, e.g. "0-19").
    /// Pages outside the range are never touched; RSS scales with shard size.
    #[arg(long)]
    layers: Option<String>,

    /// Only load and serve experts in this range (inclusive, e.g. "0-63").
    /// Used to shard the expert bank across servers for MoE models.
    /// Mutually exclusive with --units.
    #[arg(long)]
    experts: Option<String>,

    /// Path to a JSON manifest for fine-grained per-(layer, expert) ownership.
    /// Mutually exclusive with --experts.
    #[arg(long, value_name = "PATH")]
    units: Option<std::path::PathBuf>,

    /// Run as an embed-service endpoint (loads only embeddings + lm_head).
    #[arg(long)]
    embed_only: bool,

    /// Eager-build HNSW index for every owned layer at startup. Requires --hnsw.
    #[arg(long)]
    warmup_hnsw: bool,

    /// Pre-load inference weights and prefetch all owned layer mmap pages at boot.
    #[arg(long)]
    warmup_walk_ffn: bool,

    /// Bind a Unix domain socket alongside TCP for same-host MoE shard clients.
    #[arg(long, value_name = "PATH")]
    uds_path: Option<std::path::PathBuf>,

    /// Join one or more router grids (comma-separated gRPC addresses).
    /// Example: "grpc://router-a:50052,grpc://router-b:50052"
    /// Requires --public-url so routers know where to direct clients.
    #[arg(long)]
    join: Option<String>,

    /// Public HTTP URL clients use to reach this server (used with --join).
    #[arg(long)]
    public_url: Option<String>,

    /// Shared secret matching the router's --grid-key (or set LARQL_GRID_KEY env var).
    #[arg(long)]
    grid_key: Option<String>,

    /// Trust X-Forwarded-For when rate limiting (enable only behind a trusted proxy).
    #[arg(long)]
    trust_forwarded_for: bool,

    /// Server-side MoE expert shard map: `"START-END=URL,START-END=URL,..."`
    /// The walk-ffn handler will dispatch MoE expert calls to these remote servers.
    /// Combine with --layers for full 2D (layer × expert) sharding.
    #[arg(long)]
    moe_shards: Option<String>,

    /// Path to a JSON manifest for fine-grained per-(layer, expert) shard ownership.
    /// Mutually exclusive with --moe-shards.
    #[arg(long, value_name = "PATH")]
    moe_units_manifest: Option<std::path::PathBuf>,
}

// ══════════════════════════════════════════════════════════════════════
// Main entry + argv trampoline
// ══════════════════════════════════════════════════════════════════════

/// Research subcommands previously lived at the top level. Rewrite
/// `larql <legacy-name> …` → `larql dev <legacy-name> …` before clap
/// parses so existing scripts keep working.
const LEGACY_DEV_NAMES: &[&str] = &[
    "weight-extract",
    "attention-extract",
    "vector-extract",
    "residuals",
    "predict",
    "index-gates",
    "extract-routes",
    "walk",
    "attention-capture",
    "qk-templates",
    "qk-rank",
    "qk-modes",
    "ov-gate",
    "circuit-discover",
    "attn-bottleneck",
    "ffn-bench",
    "ffn-bottleneck",
    "ffn-overlap",
    "kg-bench",
    "ffn-throughput",
    "trajectory-trace",
    "projection-test",
    "fingerprint-extract",
    "bottleneck-test",
    "embedding-jump",
    "bfs",
    "ffn-latency",
];

fn rewrite_legacy_argv(args: Vec<String>) -> Vec<String> {
    if args.len() >= 2 && LEGACY_DEV_NAMES.contains(&args[1].as_str()) {
        let mut rewritten = Vec::with_capacity(args.len() + 1);
        rewritten.push(args[0].clone());
        rewritten.push("dev".to_string());
        rewritten.extend(args.into_iter().skip(1));
        return rewritten;
    }
    args
}

fn main() {
    // Windows defaults the main thread to a 1 MiB stack, which our large
    // clap-derived `Commands` enum overflows during parse_from in debug
    // builds. Spawn the real entrypoint on a worker thread with a roomy
    // stack so the binary behaves the same as on Linux/macOS (where the
    // default main stack is ~8 MiB).
    let code = std::thread::Builder::new()
        .name("larql-main".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(real_main)
        .expect("spawn larql-main thread")
        .join()
        .expect("larql-main thread panicked");
    std::process::exit(code);
}

fn real_main() -> i32 {
    let raw_args: Vec<String> = std::env::args().collect();
    let args = rewrite_legacy_argv(raw_args);
    let cli = Cli::parse_from(args);

    let result = match cli.command {
        // ── Primary ──
        Commands::Run(args) => run_cmd::run(args),
        Commands::Chat(args) => run_cmd::run(args.into()),
        Commands::Bench(args) => bench::run(args),
        Commands::Accuracy(args) => accuracy_cmd::run(args),
        Commands::Shannon(cmd) => shannon_cmd::run(cmd),
        Commands::Pull(args) => pull_cmd::run(args),
        Commands::Model(args) => model_cmd::run(args),
        Commands::Link(args) => link_cmd::run(args),
        Commands::List(args) => list_cmd::run(args),
        Commands::Show(args) => show_cmd::run(args),
        Commands::Slice(args) => slice_cmd::run(args),
        Commands::Publish(args) => publish_cmd::run(args),
        Commands::Rm(args) => rm_cmd::run(args),

        // ── Build / extract ──
        Commands::Extract(args) => extract_index_cmd::run(args),
        Commands::ExtractIndex(args) => extract_index_cmd::run(args),
        Commands::Build(args) => build_cmd::run(args),
        Commands::Compile(args) => compile_cmd::run(args),
        Commands::Convert(args) => convert_cmd::run(args),
        Commands::Hf(args) => hf_cmd::run(args),
        Commands::Verify(args) => verify_cmd::run(args),
        Commands::Diag(args) => diag_cmd::run(args),
        Commands::Parity(args) => parity::run(args),

        // ── Query (legacy graph-file surface) ──
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Merge(args) => merge_cmd::run(args),
        Commands::Filter(args) => filter_cmd::run(args),

        // ── LQL ──
        Commands::Repl => {
            larql_lql::run_repl();
            Ok(())
        }
        Commands::Lql(args) => match larql_lql::run_batch(&args.statement) {
            Ok(lines) => {
                for line in &lines {
                    println!("{line}");
                }
                Ok(())
            }
            Err(e) => Err(e),
        },

        // ── Serve (exec into larql-server) ──
        Commands::Serve(args) => run_serve(args),

        // ── Research / dev tools ──
        Commands::Dev(cmd) => run_dev(cmd),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        return 1;
    }
    0
}

fn run_dev(cmd: DevCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        DevCommand::WeightExtract(a) => weight_walk_cmd::run(a),
        DevCommand::AttentionExtract(a) => attention_walk_cmd::run(a),
        DevCommand::VectorExtract(a) => vector_extract_cmd::run(a),
        DevCommand::Residuals(a) => residuals_cmd::run(a),
        DevCommand::Predict(a) => predict_cmd::run(a),
        DevCommand::IndexGates(a) => index_gates_cmd::run(a),
        DevCommand::Walk(a) => walk_cmd::run(a),
        DevCommand::AttentionCapture(a) => attention_capture_cmd::run(a),
        DevCommand::QkTemplates(a) => qk_templates_cmd::run(a),
        DevCommand::QkRank(a) => qk_rank_cmd::run(a),
        DevCommand::QkModes(a) => qk_modes_cmd::run(a),
        DevCommand::OvGate(a) => ov_gate_cmd::run(a),
        DevCommand::OvRd(a) => ov_rd::cmd::run(a),
        DevCommand::CircuitDiscover(a) => circuit_discover_cmd::run(a),
        DevCommand::AttnBottleneck(a) => attn_bottleneck_cmd::run(a),
        DevCommand::FfnBottleneck(a) => ffn_bottleneck_cmd::run(a),
        DevCommand::FfnOverlap(a) => ffn_overlap_cmd::run(a),
        DevCommand::KgBench(a) => kg_bench_cmd::run(a),
        DevCommand::TrajectoryTrace(a) => trajectory_trace_cmd::run(a),
        DevCommand::ProjectionTest(a) => projection_test_cmd::run(a),
        DevCommand::FingerprintExtract(a) => fingerprint_extract_cmd::run(a),
        DevCommand::BottleneckTest(a) => bottleneck_test_cmd::run(a),
        DevCommand::EmbeddingJump(a) => embedding_jump_cmd::run(a),
        DevCommand::Bfs(a) => bfs_cmd::run(a),
        DevCommand::FfnLatency(a) => ffn_latency_cmd::run(a),
    }
}

fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd_args = Vec::new();
    if let Some(ref path) = args.vindex_path {
        // Resolve cache shorthands / owner-name / hf:// → actual path
        // so `larql serve gemma3-4b-v2` works the same as `larql run`.
        // Explicit directories and already-resolved paths pass through.
        let resolved = commands::primary::cache::resolve_model(path)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.clone());
        cmd_args.push(resolved);
    }
    if let Some(ref dir) = args.dir {
        cmd_args.push("--dir".into());
        cmd_args.push(dir.display().to_string());
    }
    cmd_args.push("--port".into());
    cmd_args.push(args.port.to_string());
    cmd_args.push("--host".into());
    cmd_args.push(args.host.clone());
    cmd_args.push("--log-level".into());
    cmd_args.push(args.log_level.clone());
    cmd_args.push("--max-concurrent".into());
    cmd_args.push(args.max_concurrent.to_string());
    if args.no_infer {
        cmd_args.push("--no-infer".into());
    }
    if args.ffn_only {
        cmd_args.push("--ffn-only".into());
    }
    if args.max_gate_cache_layers > 0 {
        cmd_args.push("--max-gate-cache-layers".into());
        cmd_args.push(args.max_gate_cache_layers.to_string());
    }
    if args.release_mmap_after_request {
        cmd_args.push("--release-mmap-after-request".into());
    }
    if args.cors {
        cmd_args.push("--cors".into());
    }
    if let Some(ref key) = args.api_key {
        cmd_args.push("--api-key".into());
        cmd_args.push(key.clone());
    }
    if let Some(ref rl) = args.rate_limit {
        cmd_args.push("--rate-limit".into());
        cmd_args.push(rl.clone());
    }
    if args.cache_ttl > 0 {
        cmd_args.push("--cache-ttl".into());
        cmd_args.push(args.cache_ttl.to_string());
    }
    if let Some(port) = args.grpc_port {
        cmd_args.push("--grpc-port".into());
        cmd_args.push(port.to_string());
    }
    if let Some(ref cert) = args.tls_cert {
        cmd_args.push("--tls-cert".into());
        cmd_args.push(cert.display().to_string());
    }
    if let Some(ref key) = args.tls_key {
        cmd_args.push("--tls-key".into());
        cmd_args.push(key.display().to_string());
    }
    if let Some(ref range) = args.layers {
        cmd_args.push("--layers".into());
        cmd_args.push(range.clone());
    }
    if let Some(ref range) = args.experts {
        cmd_args.push("--experts".into());
        cmd_args.push(range.clone());
    }
    if let Some(ref path) = args.units {
        cmd_args.push("--units".into());
        cmd_args.push(path.display().to_string());
    }
    if args.embed_only {
        cmd_args.push("--embed-only".into());
    }
    if args.warmup_hnsw {
        cmd_args.push("--warmup-hnsw".into());
    }
    if args.warmup_walk_ffn {
        cmd_args.push("--warmup-walk-ffn".into());
    }
    if let Some(ref path) = args.uds_path {
        cmd_args.push("--uds-path".into());
        cmd_args.push(path.display().to_string());
    }
    if let Some(ref addrs) = args.join {
        cmd_args.push("--join".into());
        cmd_args.push(addrs.clone());
    }
    if let Some(ref url) = args.public_url {
        cmd_args.push("--public-url".into());
        cmd_args.push(url.clone());
    }
    if let Some(ref key) = args.grid_key {
        cmd_args.push("--grid-key".into());
        cmd_args.push(key.clone());
    }
    if args.trust_forwarded_for {
        cmd_args.push("--trust-forwarded-for".into());
    }
    if let Some(ref s) = args.moe_shards {
        cmd_args.push("--moe-shards".into());
        cmd_args.push(s.clone());
    }
    if let Some(ref path) = args.moe_units_manifest {
        cmd_args.push("--moe-units-manifest".into());
        cmd_args.push(path.display().to_string());
    }

    let exe = std::env::current_exe().ok();
    let server_bin = exe
        .as_ref()
        .and_then(|e| e.parent())
        .map(|d| d.join("larql-server"))
        .filter(|p| p.exists());

    let bin = server_bin
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "larql-server".into());

    let status = std::process::Command::new(&bin).args(&cmd_args).status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("larql-server exited with: {s}").into()),
        Err(e) => {
            eprintln!("Failed to exec larql-server: {e}");
            eprintln!(
                "Make sure larql-server is installed (cargo install --path crates/larql-server)"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod trampoline_tests {
    use super::*;

    fn args(tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn primary_verb_is_untouched() {
        let input = args(&["larql", "run", "gemma3-4b.vindex", "hello"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn top_level_extract_is_untouched() {
        let input = args(&["larql", "extract", "google/gemma-3-4b-it", "-o", "out"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn extract_index_alias_is_untouched() {
        // `extract-index` is a distinct top-level variant, not a legacy
        // research command — must not be rewritten to `dev extract-index`.
        let input = args(&["larql", "extract-index", "google/gemma-3-4b-it"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn legacy_research_verb_is_rewritten() {
        let input = args(&[
            "larql",
            "walk",
            "--index",
            "x.vindex",
            "--prompt",
            "hi",
            "--predict",
        ]);
        let out = rewrite_legacy_argv(input);
        assert_eq!(
            out,
            args(&[
                "larql",
                "dev",
                "walk",
                "--index",
                "x.vindex",
                "--prompt",
                "hi",
                "--predict"
            ])
        );
    }

    #[test]
    fn legacy_research_flag_names_all_rewrite() {
        // Spot-check each legacy name survives the rewrite.
        for name in LEGACY_DEV_NAMES {
            let input = args(&["larql", name, "--help"]);
            let out = rewrite_legacy_argv(input);
            assert_eq!(out[0], "larql");
            assert_eq!(out[1], "dev");
            assert_eq!(out[2], *name);
            assert_eq!(out[3], "--help");
        }
    }

    #[test]
    fn no_args_returns_unchanged() {
        let input = args(&["larql"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn unknown_verb_is_not_rewritten() {
        // If `larql typo-command` comes in, don't wrap in `dev` — let
        // clap produce its own "unrecognized subcommand" error.
        let input = args(&["larql", "typo-command"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out, input);
    }

    #[test]
    fn rewrite_preserves_argument_count_plus_one() {
        let input = args(&["larql", "walk", "--flag", "value"]);
        let out = rewrite_legacy_argv(input.clone());
        assert_eq!(out.len(), input.len() + 1);
    }
}
