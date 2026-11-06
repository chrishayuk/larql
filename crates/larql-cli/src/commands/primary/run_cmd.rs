//! `larql run <model> [prompt]` — ollama-style one-shot inference / chat.
//!
//! Wraps the richer `larql dev walk --predict` pipeline behind a slim flag
//! set. If a prompt is given, runs one forward pass and prints the top-N
//! predictions. If no prompt is given, drops into a stdin chat loop — one
//! line in, one forward pass out, repeat until EOF.
//!
//! A VINDEX3 container is routed first, to `run_cmd_vindex3`: it executes
//! its own program through the VINDEX3 interpreter, with the tokenizer the
//! container carries, and honours only the flags that apply to it.
//!
//! Flag surface:
//!   <model>         required; vindex directory, VINDEX3 container,
//!                   `hf://owner/name`, or a cache shorthand
//!                   (e.g. `gemma-3-4b-it-vindex`).
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

mod bitnet;
mod moe_shards;
mod remote_ffn;
mod routed_container;
use bitnet::*;
use moe_shards::*;
use remote_ffn::*;
use routed_container::*;

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
/// — they accept the full engine catalog (MarkovResidual, WindowedCheckpoint,
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

    /// VINDEX3 only: send the prompt verbatim as a plain completion,
    /// without the chat template the container carries. By default a
    /// VINDEX3 run (one-shot and chat) renders each turn through that
    /// template. VINDEX2 runs read `LARQL_RAW_PROMPT=1` for the same
    /// choice and refuse this flag.
    #[arg(long)]
    pub raw: bool,

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
    /// WindowedCheckpoint, TurboQuant, Apollo), use `--engine` instead.
    #[arg(long, default_value = "standard", value_parser = parse_kv_cache)]
    pub kv_cache: KvCacheKind,

    /// Sliding-window size when `--kv-cache markov-bounded`. Ignored
    /// otherwise. `0` = unbounded (same as `standard`).
    #[arg(long, default_value = "0")]
    pub context_window: usize,

    /// VINDEX3 accepts row, standard and no-cache; other engine specs refuse.
    /// KV engine spec, overrides `--kv-cache` when set. Accepts the same
    /// syntax `larql bench --engine` parses:
    ///
    ///   standard                    — production K/V cache (default)
    ///   standard:window=1024        — sliding-window K/V
    ///   no-cache                    — full re-forward per step (O(N²))
    ///   markov-rs[:window=N]        — residual-stream replacement
    ///   windowed-checkpoint:window=N  — per-window K/V checkpoints
    ///   turbo-quant[:bits=3|4]      — WHT + Lloyd-Max codec
    ///   apollo:layer=N,coef=F,top_k=K,bos=B — boundary-residual injection (bench-only)
    ///
    /// Falls back to the `LARQL_KV_ENGINE` env var when unset, and to
    /// the `--kv-cache` mapping when both are absent. CLI flag wins over
    /// env var; env var wins over `--kv-cache`. See
    /// `crates/larql-inference/docs/specs/kv-engine-unification.md`.
    #[arg(long, value_name = "SPEC")]
    pub engine: Option<String>,

    /// VINDEX3 only: hold continuation state with the provider of this
    /// identity (`family/vN`), e.g. one a `--plugin` registered, instead
    /// of the one `--engine` names. Refused together with `--engine`.
    #[arg(long, value_name = "FAMILY/vN")]
    pub continuation: Option<String>,

    /// A `key=value` option for the `--continuation` provider. Repeatable;
    /// the provider accepts or refuses each before anything runs.
    #[arg(long = "continuation-option", value_name = "KEY=VALUE")]
    pub continuation_options: Vec<String>,

    /// Show the top-K prediction table for each step instead of just
    /// the argmax. Implied by `--verbose`.
    #[arg(long, default_value = "1")]
    pub top: usize,

    /// Route FFN to a remote larql-server (e.g. `http://127.0.0.1:8080`).
    /// Attention runs locally; each layer's FFN is a round trip to the URL.
    #[arg(long, value_name = "URL")]
    pub ffn: Option<String>,

    /// Serve the routed expert banks from a legacy LYRW bank-shape
    /// container (`extract-index --expert-banks-out`; not a VINDEX3 3.0
    /// model, ADR-0027), keeping the rest of the model (tokenizer, config, embeddings, attention, norms,
    /// routers, dense/shared FFN, LM head) from the VINDEX2 `MODEL` argument.
    ///
    /// Exactly one operand source is replaced — spec §4 classes 4 and 5 —
    /// so a comparison against the same prompt without this flag is a
    /// statement about the routed bytes and nothing else.
    ///
    /// Never falls back: if the container cannot serve every routed layer the
    /// model needs, the run is refused before the prompt is encoded.
    #[arg(long, value_name = "DIR")]
    pub routed_from: Option<String>,

    /// With `--routed-from --metal`: print the exact prompt token ids (after
    /// chat wrapping) and the generated token ids to stderr, so the run can
    /// serve as an id-level oracle for `larql vindex3 exec --tokens ...`
    /// on the same model. Text output is unchanged.
    #[arg(long, default_value_t = false)]
    pub emit_ids: bool,

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

    /// Path to one or more image files to splice into the prompt prefix
    /// (multi-modal Phase 1d, prefix-only). Repeat the flag to pass
    /// multiple images:
    ///
    ///   larql run gemma-3-4b-it --image cat.jpg --image stop_sign.jpg "describe both"
    ///
    /// Currently supported on Gemma 3 multimodal checkpoints only.
    /// VINDEX3 supports row, standard and no-cache on CPU. V2
    /// requires `--engine standard` (other engines lack
    /// `prefill_from_hidden`; the CLI will fail fast with a clear
    /// message if `--image` is combined with an MM-incapable engine —
    /// see ADR-0023). Also requires `--mm-weights` to point at the
    /// directory containing the SigLIP vision_tower and
    /// multi_modal_projector safetensors shards (typically the HF
    /// snapshot dir, e.g.
    /// `~/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/<hash>`).
    #[arg(long, value_name = "PATH", num_args = 0..)]
    pub image: Vec<PathBuf>,

    /// Directory containing the SigLIP and multi_modal_projector
    /// safetensors shards. Required when `--image` is set. The vindex
    /// itself only carries LM weights (FFN + attention + embeddings);
    /// the vision tower and projector live in the original HF snapshot
    /// alongside `config.json`. Both Phase 1b and Phase 1c loaders
    /// scan this directory for `*.safetensors` files filtered by tensor
    /// key prefix.
    #[arg(long, value_name = "DIR")]
    pub mm_weights: Option<PathBuf>,

    /// Ordered VINDEX3 CPU layer workers. Replays the full prefix each step.
    #[arg(long, value_delimiter = ',', value_name = "URL,...")]
    pub v3_shards: Vec<String>,

    /// VINDEX3 CPU dense FFN or routed-expert workers; attention, routing and KV remain local.
    #[arg(
        long,
        value_delimiter = ',',
        value_name = "URL,...",
        conflicts_with = "v3_shards"
    )]
    pub v3_ffn_shards: Vec<String>,

    /// FFN wire: binary f32 (default), JSON control, or experimental stream. Routed experts require binary.
    #[arg(long, value_parser = ["binary", "json", "stream"], requires = "v3_ffn_shards")]
    pub v3_ffn_wire: Option<String>,

    /// Write per-position CPU V3 timings and exact FFN HTTP body bytes to a new JSONL file.
    #[arg(long, value_name = "PATH")]
    pub v3_profile: Option<PathBuf>,

    /// Environment variable holding the bearer token for V3 layer, FFN or expert workers.
    #[arg(long, value_name = "ENV")]
    pub v3_shard_token_env: Option<String>,

    /// Speak the prompt: run the model as a speech generator
    /// (MOSS-TTS-Realtime) and synthesise audio tokens instead of text.
    /// Pure TTS — the prompt is what gets said, no chat LLM in front.
    /// `model` is the checkpoint's safetensors directory until the model
    /// moves into the vindex (TTS funnel step 6).
    #[arg(long)]
    pub speak: bool,

    /// Voice reference for `--speak`: a token-rows file produced by the
    /// external codec's encode mode (one frame per line). Omit for the
    /// model's unconditioned voice.
    #[arg(long, value_name = "TOKENS")]
    pub voice: Option<PathBuf>,

    /// External codec command for `--speak`, with `{tokens}` and `{wav}`
    /// placeholders (falls back to $LARQL_MOSS_CODEC_CMD). Without it,
    /// audio tokens are written and WAV synthesis is skipped.
    #[arg(long, value_name = "CMD")]
    pub codec_cmd: Option<String>,

    /// Output WAV path for `--speak` (default: speech.wav).
    #[arg(long, value_name = "WAV")]
    pub speech_out: Option<PathBuf>,

    /// Play the synthesised WAV after codec decode (`--speak`, macOS).
    #[arg(long)]
    pub play: bool,

    /// Frame cap for `--speak` (12.5 frames per second of audio).
    #[arg(long, default_value = "1500")]
    pub max_frames: usize,

    /// Greedy decoding for `--speak` (parity/debug). Default is the
    /// reference's sampled mode — greedy does not terminate reliably on
    /// novel text.
    #[arg(long)]
    pub greedy: bool,

    /// RNG seed for `--speak` sampled mode (reproducible synthesis).
    #[arg(long, default_value = "0")]
    pub seed: u64,

    /// Quantise FFN weights to Q4_K for `--speak` (both transformers;
    /// attention stays fp32). The realtime experiment of
    /// docs/tts-funnel.md — check voice quality before trusting speed.
    #[arg(long)]
    pub q4: bool,

    /// `--plugin` / `--lowering`: codecs and lowering providers loaded
    /// from shared libraries, for a VINDEX3 container.
    #[command(flatten)]
    pub plugin: super::vindex3_cmd::plugins::PluginArgs,
}

pub fn run(mut args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Speech mode routes before vindex resolution: the speech model lives
    // in its safetensors directory until TTS funnel step 6.
    if args.speak {
        return super::run_cmd_speak::run_speak(&args);
    }

    let vindex_path = cache::resolve_model(&args.model)?;
    if !vindex_path.is_dir() {
        return Err(format!(
            "resolved model path is not a directory: {}",
            vindex_path.display()
        )
        .into());
    }

    // A VINDEX3 container executes its own program through the VINDEX3
    // interpreter — detect early and route, before any VINDEX2-only
    // reader is asked to open it. A directory that is not a VINDEX3
    // container falls through so the dense path surfaces its own error.
    if super::run_cmd_vindex3::is_vindex3_container(&vindex_path) {
        if args.engine.is_none() {
            args.engine = std::env::var("LARQL_KV_ENGINE")
                .ok()
                .filter(|s| !s.is_empty());
        }
        return super::run_cmd_vindex3::run(&vindex_path, &args);
    }
    if !args.v3_shards.is_empty() || !args.v3_ffn_shards.is_empty() || args.v3_profile.is_some() {
        return Err(
            "--v3-shards, --v3-ffn-shards and --v3-profile require a VINDEX3 container".into(),
        );
    }
    if args.raw {
        return Err(
            "--raw applies to a VINDEX3 container; a VINDEX2 run reads LARQL_RAW_PROMPT=1 \
             to send the prompt without its chat template"
                .into(),
        );
    }

    if args.experts {
        return experts::run(&vindex_path, &args);
    }

    // BitNet b1.58 native-ternary vindex: served by the ternary forward
    // (`larql_inference::ternary`), not the dense engine dispatch — detect
    // early and route, bypassing the walk_cmd path the dense run delegates to.
    // A failed config load falls through so the dense path surfaces the error.
    if larql_vindex::load_vindex_config(&vindex_path)
        .map(|c| c.bitnet_layout.is_some())
        .unwrap_or(false)
    {
        return run_bitnet(&vindex_path, &args);
    }

    if let Some(ref routed_dir) = args.routed_from {
        let prompt = args
            .prompt
            .as_deref()
            .ok_or("--routed-from requires a prompt argument (chat mode not yet supported)")?;
        return run_with_routed_container(
            &vindex_path,
            routed_dir,
            prompt,
            args.max_tokens,
            args.metal,
            args.emit_ids,
        );
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
            args.metal,
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
            args.metal,
            args.engine.as_deref(),
        );
    }

    if !args.image.is_empty() {
        // Multi-modal Phase 1d, prefix-only. Mutually exclusive with
        // the experts / ffn / moe-shards paths above (they're
        // tokenizer-deep and don't compose with hidden-state prefill).
        if args.ffn.is_some() {
            return Err("--image is incompatible with --ffn (Phase 1d scope)".into());
        }
        if args.moe_shards.is_some() || args.moe_units_manifest.is_some() {
            return Err(
                "--image is incompatible with --moe-shards / --moe-units-manifest \
                 (Phase 1d scope)"
                    .into(),
            );
        }
        if args.experts {
            return Err("--image is incompatible with --experts (Phase 1d scope)".into());
        }
        let prompt = args
            .prompt
            .as_deref()
            .ok_or("--image requires a prompt argument (chat mode with images is Phase 2+ work)")?;
        return super::run_cmd_image::run_with_images(&vindex_path, prompt, &args);
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

mod experts;
