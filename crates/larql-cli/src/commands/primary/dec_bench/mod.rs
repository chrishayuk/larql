//! `larql dec-bench` — DEC residual-replay loadgen (docs/dec-funnel.md).
//!
//! Measures the expert tier's batch behaviour (claim C1/C2) by replaying
//! real captured residuals as B-row requests, without client-side batched
//! decode. Module map:
//!
//!   args           — clap surface (`capture` / `replay` / `drift` sub-modes).
//!   capture_format — pool on-disk format (pure, coverage-gated).
//!   capture_runtime— model load + live decode capture (I/O, excluded).
//!   replay         — sweep plan, frame builders, summaries (pure, gated).
//!   replay_runtime — HTTP driver (I/O, excluded).
//!   drift          — C6 wire-fidelity gate: bits math, arm plan, drift/gate
//!                    arithmetic, records (pure, gated).
//!   drift_runtime  — teacher-forced scoring driver (I/O, excluded).
//!   pulse          — `dec/*` JSONL emission (pure, gated).
//!   output         — full JSON run record (pure, gated).
//!   window_union   — BW11-1 same-sequence consecutive-position expert-union
//!                    ceiling: parsing, windowing, percentiles (pure, gated).
//!   window_union_runtime — trace-file driver (plain file I/O, tempfile-testable).

pub mod args;
pub mod capture_format;
mod capture_runtime;
pub mod drift;
mod drift_runtime;
pub mod output;
pub mod pulse;
pub mod replay;
mod replay_runtime;
pub mod window_union;
mod window_union_runtime;

pub use args::DecBenchArgs;

pub fn run(args: DecBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.cmd {
        args::DecBenchCmd::Capture(a) => capture_runtime::run_capture(&a),
        args::DecBenchCmd::Replay(a) => replay_runtime::run_replay(&a),
        args::DecBenchCmd::Drift(a) => drift_runtime::run_drift(&a),
        args::DecBenchCmd::WindowUnion(a) => window_union_runtime::run_window_union(&a),
    }
}
