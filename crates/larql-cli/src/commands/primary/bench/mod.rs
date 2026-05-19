//! `larql bench <model>` — end-to-end decode benchmark on a real vindex.
//!
//! Measures prefill + autoregressive decode on a vindex, reports per-stage
//! breakdown (GPU forward / lm_head / norm / embed / detok), and optionally
//! queries a running Ollama server on the same machine for a side-by-side
//! tok/s comparison.
//!
//! Flag surface (see `args` module for the full clap-derive struct):
//!   <model>               vindex dir, `hf://owner/name`, or cache shorthand.
//!   --prompt STR          prompt to time.
//!   -n, --tokens N        decode steps to time (default: 50).
//!   --warmup N            warmup steps before measurement (default: 3).
//!   --backends LIST       comma-separated: `metal`, `cpu`. Default: `metal`.
//!   --cpu                 shorthand for `--backends cpu`.
//!   --ollama MODEL        also query Ollama via localhost.
//!   --ffn URL             bench remote FFN path.
//!   --wire f32,f16,i8     compare wire formats end-to-end (requires --ffn).
//!   --bench-grid          shard-count scaling sweep (requires --moe-shards or --ffn).
//!   --bench-grid-lan PATH LAN preregistration matrix (Exp 41) from a JSON config.
//!   --concurrent N        simulate N concurrent clients (default: 1).
//!   --output json         also emit machine-readable JSON.
//!   --output-file PATH    write JSON to file instead of stdout.
//!   -v, --verbose
//!
//! Module map:
//!   args       — clap `BenchArgs` struct.
//!   row        — internal `BenchRow` + JSON serialization types + percentile helpers.
//!   helpers    — pure helpers (wire-list parsing, concurrent aggregation, efficiency).
//!   run        — orchestration entry point.
//!   local      — local Metal/CPU bench (`run_larql`).
//!   engine     — KV-engine bench (markov-rs / unlimited-context).
//!   remote_ffn — remote FFN HTTP path + `--concurrent` aggregation.
//!   remote_moe — remote MoE expert path + `--concurrent` aggregation.
//!   ollama     — Ollama side-by-side comparison.
//!   output     — table printer.

pub mod args;
pub mod helpers;
pub mod row;
pub mod run;

pub(super) mod engine;
pub(super) mod engine_runtime;
pub(super) mod grid_lan;
pub(super) mod grid_lan_runtime;
pub(super) mod local;
pub(super) mod local_runtime;
pub(super) mod ollama;
pub(super) mod output;
pub(super) mod remote_ffn;
pub(super) mod remote_ffn_runtime;
pub(super) mod remote_moe;
pub(super) mod remote_moe_runtime;

// Public surface kept identical to the pre-split bench_cmd: callers only
// see `BenchArgs` and `run`.
pub use args::BenchArgs;
pub use run::run;
