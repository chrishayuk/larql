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

mod anyres_tiler;
mod backend_select;
mod commands;
mod formatting;
mod image_input;
mod trampoline;
mod utils;

#[cfg(feature = "research")]
use commands::dev::{run_dev, DevCommand};
#[cfg(feature = "research")]
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
//
// Variants marked `#[cfg(feature = "research")]` exist only in research
// builds (the default); tagged release binaries are built without them,
// and `trampoline::prepare_argv` refuses those names with a clear error.
// ══════════════════════════════════════════════════════════════════════

#[derive(Subcommand)]
enum Commands {
    // ── Primary user-facing ─────────────────────────────────────────
    /// Run inference (one-shot if prompt is given, chat if not).
    Run(run_cmd::RunArgs),

    /// Interactive chat — alias for `run <model>` with no prompt.
    Chat(ChatArgs),

    /// Keep a speech model resident and synthesise utterance after
    /// utterance against it — `run --speak` without re-paying the load
    /// and quantisation bill every time.
    SpeakServe(speak_serve::SpeakServeArgs),

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

    /// Validate the VINDEX3 production registry (registry/index.json +
    /// registry/models/*.json).
    #[command(subcommand)]
    Registry(registry_cmd::RegistryCommand),

    /// Remove a cached vindex.
    Rm(rm_cmd::RmArgs),

    /// Benchmark decode throughput on a real vindex (Metal / CPU / Ollama).
    Bench(bench::BenchArgs),

    /// DEC residual-replay loadgen — capture real per-layer residuals, then
    /// replay batch × wire × dispatch sweeps against an expert server
    /// (docs/dec-funnel.md).
    DecBench(dec_bench::DecBenchArgs),

    #[cfg(feature = "research")]
    /// K3 serving ledger — miss budget, weight touch, dense-precision
    /// frontier and speculative block economics, derived from the
    /// checkpoint's own tensor table (docs/dec-funnel.md).
    K3Ledger(k3_ledger::K3LedgerArgs),

    /// Split-axis accuracy suite — parametric vs in-context vs conflict,
    /// scored with top-1 match and Shannon bits-per-token.
    Accuracy(accuracy_cmd::AccuracyArgs),

    /// Shannon-style next-token bit measurements and demo compression.
    #[command(subcommand)]
    Shannon(shannon_cmd::ShannonCommand),

    // ── Server ──────────────────────────────────────────────────────
    #[command(next_help_heading = "Server")]
    /// Serve a vindex over HTTP + gRPC.
    Serve(serve_cmd::ServeArgs),

    #[command(next_help_heading = "Server")]
    /// Ask a running LARQL server what it will and will not do
    /// (`GET /v1/capabilities`). Distinct from `capabilities`, which
    /// reports what this release recognises rather than what one
    /// server offers.
    ServerCapabilities(server_capabilities_cmd::ServerCapabilitiesArgs),

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

    #[cfg(feature = "research")]
    #[command(next_help_heading = "Build")]
    /// Cross-backend numerical parity diff (CPU vs Metal vs reference).
    Parity(parity::ParityArgs),

    #[cfg(feature = "research")]
    #[command(next_help_heading = "Build")]
    /// Expert-selection locality over a routing trace: does speculative
    /// decoding amortise the expert bank, and can a hot cache work?
    /// Collect the trace with `LARQL_MOE_ROUTE_TRACE=<path> larql shannon score`.
    MoeLocality(moe_locality::MoeLocalityArgs),

    // ── Factory (docs/vindex-factory.md) ─────────────────────────────
    #[command(next_help_heading = "Factory", subcommand)]
    /// Vindex Factory recipe tooling — validate a recipe, compute its
    /// build_id (docs/vindex-factory.md).
    Recipe(recipe_cmd::RecipeCommand),

    #[command(next_help_heading = "Factory")]
    /// Print this release's capability manifest as JSON — which
    /// architectures it recognises and what each one supports.
    Capabilities,

    #[command(next_help_heading = "Factory", name = "inspect-hf")]
    /// Machine-readable architecture inventory of an HF checkpoint dir —
    /// identity, per-layer attention policy, tensors, and every config key
    /// this build does not consume.
    InspectHf(inspect_hf_cmd::InspectHfArgs),

    #[command(next_help_heading = "Factory", subcommand)]
    /// VINDEX3 container programme verbs (`plan`: semantic
    /// representability check before conversion).
    Vindex3(vindex3_cmd::Vindex3Command),

    #[command(next_help_heading = "Factory", subcommand)]
    /// Render a Hub model card for a build (docs/vindex-factory.md §9).
    Card(card_cmd::CardCommand),

    #[cfg(feature = "research")]
    #[command(next_help_heading = "Factory")]
    /// Serve a stored physical-plan search record over MCP, read-only
    /// (docs/represent-optimizer-mcp.md §4h).
    OptimizerMcp(optimizer_mcp::OptimizerMcpArgs),

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
    #[cfg(feature = "research")]
    #[command(next_help_heading = "Research", subcommand)]
    /// Research / interpretability tools (weight-extract, qk-rank, …).
    Dev(DevCommand),
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
    routed_from: Option<String>,

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
            continuation: None,
            continuation_options: Vec::new(),
            ffn: c.ffn,
            routed_from: c.routed_from,
            emit_ids: false,
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
            // Chat doesn't accept images today; default to empty for the
            // shim. When chat-with-images becomes a thing (Phase 2+), the
            // ChatArgs struct will grow its own --image flag.
            image: Vec::new(),
            mm_weights: None,
            v3_shards: Vec::new(),
            v3_ffn_shards: Vec::new(),
            v3_ffn_wire: None,
            v3_profile: None,
            v3_shard_token_env: None,
            // Chat is text-only today; speech arrives via `run --speak`
            // (and later a chat session feeding the speech stream).
            speak: false,
            plugin: Default::default(),
            voice: None,
            codec_cmd: None,
            speech_out: None,
            play: false,
            max_frames: 0,
            greedy: false,
            seed: 0,
            q4: false,
        }
    }
}

#[derive(clap::Args)]
struct LqlArgs {
    /// LQL statement (e.g. `WALK "The capital of France is" TOP 5;`).
    statement: String,
}

// ══════════════════════════════════════════════════════════════════════
// Main entry + argv trampoline
// ══════════════════════════════════════════════════════════════════════

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
    // Flush the latent-mask channel survival counts, if that probe was
    // collecting them. No-op unless `LARQL_MOE_LATENT_STATS` is set.
    larql_compute::cpu::ops::moe::latent_mask::dump_stats();
    std::process::exit(code);
}

fn real_main() -> i32 {
    let raw_args: Vec<String> = std::env::args().collect();
    let args = match trampoline::prepare_argv(raw_args) {
        Ok(args) => args,
        Err(refusal) => {
            eprintln!("Error: {refusal}");
            return trampoline::RESEARCH_UNAVAILABLE_EXIT_CODE;
        }
    };
    let cli = Cli::parse_from(args);

    let result = match cli.command {
        // ── Primary ──
        Commands::Run(args) => run_cmd::run(args),
        Commands::Chat(args) => run_cmd::run(args.into()),
        Commands::SpeakServe(args) => speak_serve::run(args),
        Commands::Bench(args) => bench::run(args),
        Commands::DecBench(args) => dec_bench::run(args),
        #[cfg(feature = "research")]
        Commands::K3Ledger(args) => k3_ledger::run(args),
        Commands::Accuracy(args) => accuracy_cmd::run(args),
        Commands::Shannon(cmd) => shannon_cmd::run(cmd),
        Commands::Pull(args) => pull_cmd::run(args),
        Commands::Model(args) => model_cmd::run(args),
        Commands::Link(args) => link_cmd::run(args),
        Commands::List(args) => list_cmd::run(args),
        Commands::Show(args) => show_cmd::run(args),
        Commands::Slice(args) => slice_cmd::run(args),
        Commands::Publish(args) => publish_cmd::run(args),
        Commands::Registry(cmd) => registry_cmd::run(cmd),
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
        #[cfg(feature = "research")]
        Commands::Parity(args) => parity::run(args),
        #[cfg(feature = "research")]
        Commands::MoeLocality(args) => moe_locality::run(args),

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

        // ── Factory ──
        Commands::Recipe(cmd) => recipe_cmd::run(cmd),
        Commands::Capabilities => capabilities_cmd::run(),
        Commands::ServerCapabilities(args) => server_capabilities_cmd::run(args),
        Commands::InspectHf(args) => inspect_hf_cmd::run(args),
        Commands::Vindex3(cmd) => vindex3_cmd::run(cmd),
        #[cfg(feature = "research")]
        Commands::OptimizerMcp(args) => optimizer_mcp::run(args),
        Commands::Card(cmd) => card_cmd::run(cmd),

        // ── Serve (exec into larql-server) ──
        Commands::Serve(args) => serve_cmd::run_serve(args),

        // ── Research / dev tools ──
        #[cfg(feature = "research")]
        Commands::Dev(cmd) => run_dev(cmd),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        return 1;
    }
    0
}

#[cfg(test)]
mod documentation_tests {
    use clap::CommandFactory;

    #[test]
    fn current_documentation_facts_match_clap() {
        let facts: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/generated/current-facts.json"))
                .unwrap();
        let command = super::Cli::command();
        let vindex3 = command.find_subcommand("vindex3").unwrap();
        let mut names: Vec<_> = vindex3.get_subcommands().map(|c| c.get_name()).collect();
        names.sort_unstable();
        assert_eq!(facts["commands"]["larql_vindex3"], serde_json::json!(names));
    }
}
