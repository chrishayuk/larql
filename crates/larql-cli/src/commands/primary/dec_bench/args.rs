//! Clap args for `larql dec-bench` — the DEC residual-replay loadgen
//! (docs/dec-funnel.md §3, DEC-0). Two sub-modes:
//!
//!   * `capture` — run single-stream decode over N prompts against a live
//!     expert server, recording each step's pre-normed per-layer residuals
//!     into a portable pool.
//!   * `replay`  — sweep batch × wire × dispatch against the server using
//!     B-row frames built from the pool; emit the `dec/*` schema.

use clap::{Args, Subcommand};

#[derive(Args, Clone)]
pub struct DecBenchArgs {
    #[command(subcommand)]
    pub cmd: DecBenchCmd,
}

#[derive(Subcommand, Clone)]
pub enum DecBenchCmd {
    /// Capture a residual pool from live decode over N prompts.
    Capture(CaptureArgs),
    /// Replay the pool as B-row requests, sweeping batch × wire × dispatch.
    Replay(ReplayArgs),
    /// C6 wire-fidelity gate: teacher-forced bits/char per wire arm vs the
    /// in-run f32/f32 baseline, pass/fail at a pre-set drift gate.
    Drift(DriftArgs),
    /// BW11-1: same-sequence consecutive-position expert-union ceiling,
    /// from `LARQL_MOE_ROUTE_TRACE` JSONL files (no HTTP, no capture pool
    /// — reads a decode run's routing trace directly).
    WindowUnion(WindowUnionArgs),
}

#[derive(Args, Clone)]
pub struct CaptureArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand.
    pub model: String,

    /// Expert-server URL (loopback for DEC-0 arm M).
    #[arg(long)]
    pub ffn: String,

    /// Prompts file, one prompt per line (fixed set, checked in).
    #[arg(long)]
    pub prompts_file: std::path::PathBuf,

    /// Number of prompts to capture (= max replay batch size).
    #[arg(long, default_value = "64")]
    pub num_prompts: usize,

    /// Decode steps to capture per prompt.
    #[arg(long, default_value = "16")]
    pub steps: usize,

    /// Use the Metal GPU attention backend (required on macOS for the
    /// remote-FFN decode path).
    #[arg(long)]
    pub metal: bool,

    /// Output pool directory (`manifest.json` + `residuals.bin`, plus
    /// `raw.bin`/`normed.bin`/`routing.bin` with `--routing`).
    #[arg(long)]
    pub out: std::path::PathBuf,

    /// Also capture the routed-experts sidecars (raw + pre-experts-normed
    /// residual planes and per-layer top-k expert routing) for the DEC
    /// routed replay arm. `top_k` comes from the model arch; errors if the
    /// model has no MoE. Without this flag the capture is byte-identical to
    /// the walk-ffn-only pool format.
    #[arg(long)]
    pub routing: bool,

    /// Per-request timeout for the expert server.
    #[arg(long, default_value = "60")]
    pub ffn_timeout_secs: u64,

    #[arg(short, long)]
    pub verbose: bool,
}

/// `larql dec-bench drift` — the C6 wire-fidelity drift instrument
/// (docs/dec-funnel.md §3 DEC-1A). Scores a pinned corpus teacher-forced
/// (NLL of fixed text — NOT generation) through the production remote-FFN
/// decode path once per wire arm, plus once at f32/f32 as the in-run
/// baseline, and gates each arm's bits/char drift.
///
/// The metric is exact math given weights + wire: no sampling, no timing.
/// Thermal state / battery / load cannot move it, so this instrument is
/// battery-safe — unlike `dec-bench replay` and `larql bench`.
#[derive(Args, Clone)]
pub struct DriftArgs {
    /// Vindex directory, `hf://owner/name`, or cache shorthand (client-side
    /// attention weights — must match the expert server's model).
    pub model: String,

    /// Expert-server URL (or `A-B=URL,...` shard map).
    #[arg(long)]
    pub ffn: String,

    /// UTF-8 corpus to score. Defaults to the corpus the shannon-verify CI
    /// gate pins.
    #[arg(long, default_value = "tests/fixtures/shannon_frankenstein_2k.txt")]
    pub corpus: std::path::PathBuf,

    /// Limit the corpus to the first N bytes (UTF-8 boundary), like
    /// `larql shannon score --bytes`.
    #[arg(long)]
    pub bytes: Option<usize>,

    /// Comma-separated wire arms: plain (`f32`, `f16`, `i8`, `q8k` — f32
    /// request frames, Accept per arm; q8k = its own endpoint) and/or
    /// asymmetric in/return pairs (`f16/i8`, `i8/f16`, ...). The f32/f32
    /// baseline always runs first regardless of this list.
    #[arg(long, default_value = "f32,f16,i8,q8k")]
    pub wire: String,

    /// Max |bits/char drift| vs the in-run f32 baseline, in percent
    /// (pre-registered C6 gate: 0.5).
    #[arg(long, default_value_t = super::drift::DEFAULT_DRIFT_GATE_PCT)]
    pub gate_pct: f64,

    /// Use the Metal GPU attention backend (required on macOS — the
    /// remote-FFN decode path needs the fused GPU decode).
    #[arg(long)]
    pub metal: bool,

    /// Per-request timeout for the expert server.
    #[arg(long, default_value = "60")]
    pub ffn_timeout_secs: u64,

    /// Write the full JSON run record here.
    #[arg(long)]
    pub output_file: Option<std::path::PathBuf>,

    /// Write `dec/*` pulse JSONL here (rig `$CHUK_METRICS` format): one line
    /// per arm with dec/bits_per_char, dec/drift_pct, dec/drift_gate,
    /// dec/wire_in, dec/wire_out.
    #[arg(long)]
    pub pulse_file: Option<std::path::PathBuf>,

    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Args, Clone)]
pub struct ReplayArgs {
    /// Expert-server URL.
    #[arg(long)]
    pub ffn: String,

    /// Capture pool directory (from `dec-bench capture`).
    #[arg(long)]
    pub capture: std::path::PathBuf,

    /// Server endpoint family: `walk-ffn` (dense/shared-expert FFN path) or
    /// `experts` (routed multi-layer expert path — needs a pool captured
    /// with `--routing`; wire axis limited to f32/q8k, mapping to
    /// /v1/experts/multi-layer-batch[-q8k]).
    #[arg(long, default_value = "walk-ffn")]
    pub endpoint: String,

    /// Comma-separated batch sizes (rows per request).
    #[arg(long, default_value = "1,8,16,32,64")]
    pub batch: String,

    /// Comma-separated wire arms: f32, f16, i8, q8k (symmetric — unchanged
    /// behaviour: f32 request frames, Accept per arm), or asymmetric
    /// in/return pairs like `f16/i8`, `i8/f16` (request Content-Type and
    /// Accept set independently; walk-ffn endpoint only — q8k is its own
    /// endpoint and cannot pair). i8 returns need LARQL_I8_WIRE=1 on the
    /// server or the return direction falls back (recorded in
    /// served_wire_out).
    #[arg(long, default_value = "f32,f16,i8,q8k")]
    pub wire: String,

    /// Comma-separated dispatch modes: streaming (sequential per-layer),
    /// batch (all layers fired in parallel, mirrors predispatch).
    #[arg(long, default_value = "streaming,batch")]
    pub dispatch: String,

    /// Inclusive layer range `A-B` to replay (default: all layers).
    #[arg(long)]
    pub layers: Option<String>,

    /// Steps to replay per repeat (default: all captured steps).
    #[arg(long)]
    pub steps: Option<usize>,

    /// Repeats per sweep point.
    #[arg(long, default_value = "3")]
    pub repeats: usize,

    /// Warmup passes over all replayed layers per sweep point (discarded).
    /// Full passes, not single requests — mmap-backed layer weights pay
    /// first-touch page faults per layer.
    #[arg(long, default_value = "1")]
    pub warmup_passes: usize,

    /// Per-request timeout.
    #[arg(long, default_value = "60")]
    pub timeout_secs: u64,

    /// Write the full JSON run record here.
    #[arg(long)]
    pub output_file: Option<std::path::PathBuf>,

    /// Write `dec/*` pulse JSONL here (rig `$CHUK_METRICS` format).
    #[arg(long)]
    pub pulse_file: Option<std::path::PathBuf>,

    /// Include per-layer p50/p99 keys in the pulse lines (always present in
    /// the JSON run record).
    #[arg(long)]
    pub pulse_per_layer: bool,

    /// Measured link RTT to record alongside the run (spec §6 gate).
    #[arg(long)]
    pub net_rtt_ms: Option<f64>,

    /// Measured link bandwidth to record alongside the run (spec §6 gate).
    #[arg(long)]
    pub net_gbps: Option<f64>,

    #[arg(short, long)]
    pub verbose: bool,
}

/// `larql dec-bench window-union` — BW11-1: does K consecutive positions of
/// ONE sequence (the shape a speculative-verification batch actually has)
/// share routed experts more than DEC-0's cross-prompt batch union? Reads
/// one or more `LARQL_MOE_ROUTE_TRACE` JSONL files (produced by any driver
/// loop that installs `moe_route_observe::LayerScope`, e.g. `larql run
/// --routed-from` without `LARQL_GPU_ROUTE=1`) and pools the sliding-window
/// union/naive ratio across layers and files. A ceiling measurement (R6):
/// no kernel yet groups multiple tokens through one expert's weight read,
/// so this licenses building one, not a bandwidth claim on its own.
#[derive(Args, Clone)]
pub struct WindowUnionArgs {
    /// One or more `LARQL_MOE_ROUTE_TRACE` JSONL files, one per captured
    /// prompt. Each is windowed independently (it is one continuous
    /// sequence: prefill positions then decode steps, in call order) and
    /// pooled across files for the reported spread.
    #[arg(long, value_delimiter = ',', required = true)]
    pub trace: Vec<std::path::PathBuf>,

    /// Comma-separated window widths K to report.
    #[arg(long, default_value = "1,2,4,8")]
    pub k: String,

    /// Total routed experts for this model — the activation-fraction
    /// denominator that must be quoted with every ratio (R2). Never
    /// defaulted: this is the workspace's no-hardcoded-types rule applied
    /// to a measurement, not just to model code.
    #[arg(long)]
    pub num_experts: usize,

    /// Bytes read per selected expert (gate_up + down, this container's
    /// format — from the loaded slice length, never a literal).
    #[arg(long)]
    pub per_expert_bytes: f64,

    /// Marginal-preserving shuffled-null trials per layer (R9): pool a
    /// layer's rows across all trace files, shuffle, re-measure on disjoint
    /// K-chunks, average. Rules out "the ratio just reflects popular
    /// experts colliding" as the whole explanation before this instrument's
    /// number is read as same-sequence structure. 0 disables the control.
    #[arg(long, default_value = "200")]
    pub shuffle_trials: usize,

    /// Seed for the shuffled-null control (deterministic — not wall-clock —
    /// so the control is re-derivable from the same trace files).
    #[arg(long, default_value = "1")]
    pub shuffle_seed: u64,

    /// Write `dec/*` pulse JSONL here (one line per K).
    #[arg(long)]
    pub pulse_file: Option<std::path::PathBuf>,

    #[arg(short, long)]
    pub verbose: bool,
}
