//! Process-wide runtime toggles for the remote-MoE backend, plus the
//! canonical `LARQL_MOE_*` / `SKIP_MOE` env-var names that the grid
//! config and metrics modules also reference.
//!
//! Replaces six scattered `std::env::var(..)` reads (some via per-thread
//! `thread_local!` caches, some on every shard call) with a single typed
//! config read once via `OnceLock`. Centralises the diagnostic surface so
//! adding a new toggle is a single edit instead of a hunt across
//! `shard.rs` / `backend.rs` / `metrics.rs` / `grid/config.rs`.
//!
//! Recognised vars (semantics unchanged from prior inline reads):
//!
//! - [`ENV_HTTP_TIMING`] = `LARQL_HTTP_TIMING=1` — print per-stage
//!   HTTP/UDS client-side timings on every shard call.
//! - [`ENV_MOE_WIRE_F16`] = `LARQL_MOE_WIRE_F16=1` — encode residual as
//!   f16 on the wire (LAN deployments where wire bytes matter more than
//!   f32↔f16 conversion CPU). Default off — loopback / same-host grids
//!   prefer f32.
//! - [`ENV_DISABLE_Q8K_WIRE`] = `LARQL_DISABLE_Q8K_WIRE=1` — disable Q8K
//!   residual upload (4× smaller). Default enabled; flag exists for
//!   debugging.
//! - [`ENV_VERBOSE`] = `LARQL_VERBOSE=1` — emit dispatch-step timings for
//!   the multi-layer gRPC path (route / dispatch / accum breakdown per
//!   call).
//! - [`ENV_MOE_BYTES`] = `LARQL_MOE_BYTES=1` — accumulate per-shard
//!   transport bytes; consumed by `metrics.rs`.
//! - [`ENV_MOE_TIMING`] = `LARQL_MOE_TIMING=1` — implies `MOE_BYTES` AND
//!   enables grid-side timing prints. Read by both `metrics.rs` (for
//!   the byte-tracking implication) and `grid/config.rs` (for the timing
//!   enable).
//! - [`ENV_MOE_SHARD_TIMING`] = `LARQL_MOE_SHARD_TIMING=1` — finer-grained
//!   per-shard timing in `metrics.rs`.
//! - [`ENV_MOE_TOP_K`] = `LARQL_MOE_TOP_K=<n>` — override architectural
//!   `top_k` for the grid. Consumed by `grid/config.rs`.
//! - [`ENV_MOE_NO_SPLIT`] = `LARQL_MOE_NO_SPLIT=1` — disable grid-side
//!   split execution. Consumed by `grid/config.rs`.
//! - [`ENV_SKIP_MOE`] = `SKIP_MOE=1` — bypass MoE entirely on the grid
//!   (CPU dense fallback). Consumed by `grid/config.rs`.

use std::sync::OnceLock;

// ── Env var names (single source of truth — grid/config.rs and
// metrics.rs reference these instead of raw literals). ────────────────────

/// `LARQL_HTTP_TIMING=1`
pub const ENV_HTTP_TIMING: &str = "LARQL_HTTP_TIMING";
/// `LARQL_MOE_WIRE_F16=1`
pub const ENV_MOE_WIRE_F16: &str = "LARQL_MOE_WIRE_F16";
/// `LARQL_DISABLE_Q8K_WIRE=1`
pub const ENV_DISABLE_Q8K_WIRE: &str = "LARQL_DISABLE_Q8K_WIRE";
/// `LARQL_VERBOSE=1`
pub const ENV_VERBOSE: &str = "LARQL_VERBOSE";
/// `LARQL_MOE_BYTES=1`
pub const ENV_MOE_BYTES: &str = "LARQL_MOE_BYTES";
/// `LARQL_MOE_TIMING=1` — implies `LARQL_MOE_BYTES`.
pub const ENV_MOE_TIMING: &str = "LARQL_MOE_TIMING";
/// `LARQL_MOE_SHARD_TIMING=1`
pub const ENV_MOE_SHARD_TIMING: &str = "LARQL_MOE_SHARD_TIMING";
/// `LARQL_MOE_TOP_K=<n>`
pub const ENV_MOE_TOP_K: &str = "LARQL_MOE_TOP_K";
/// `LARQL_MOE_NO_SPLIT=1`
pub const ENV_MOE_NO_SPLIT: &str = "LARQL_MOE_NO_SPLIT";
/// `SKIP_MOE=1` — bypasses MoE entirely on the grid path.
pub const ENV_SKIP_MOE: &str = "SKIP_MOE";

/// Snapshot of the MoE runtime toggles, captured once per process. The
/// grid-only vars (`MOE_TOP_K`, `MOE_NO_SPLIT`, `SKIP_MOE`) live on
/// `GridRuntimeConfig`, not here, but their *names* are exported above
/// so the grid module can reference them.
#[derive(Clone, Debug, Default)]
pub struct RemoteMoeRuntime {
    /// Print per-stage HTTP/UDS client timings for each shard call.
    pub http_timing: bool,
    /// Use f16 residual on the wire (off by default — only wins on LAN).
    pub wire_f16: bool,
    /// Disable Q8K residual upload (debug flag — default enabled).
    pub q8k_disabled: bool,
    /// Emit gRPC predispatch / multi-layer timing breakdown.
    pub verbose: bool,
    /// Accumulate per-shard transport bytes (also implied by
    /// `LARQL_MOE_TIMING`).
    pub moe_bytes_enabled: bool,
    /// Finer-grained per-shard timing in `metrics.rs`.
    pub moe_shard_timing: bool,
}

impl RemoteMoeRuntime {
    /// Read the env vars and assemble the runtime config.
    pub fn from_env() -> Self {
        let moe_bytes_raw = std::env::var(ENV_MOE_BYTES).is_ok();
        let moe_timing_raw = std::env::var(ENV_MOE_TIMING).is_ok();
        Self {
            http_timing: std::env::var(ENV_HTTP_TIMING).is_ok(),
            wire_f16: std::env::var(ENV_MOE_WIRE_F16).is_ok(),
            q8k_disabled: std::env::var(ENV_DISABLE_Q8K_WIRE).is_ok(),
            verbose: std::env::var(ENV_VERBOSE).is_ok(),
            // MOE_TIMING implies MOE_BYTES (the byte tracker is the
            // substrate that timing summaries accumulate against).
            moe_bytes_enabled: moe_bytes_raw || moe_timing_raw,
            moe_shard_timing: std::env::var(ENV_MOE_SHARD_TIMING).is_ok(),
        }
    }

    /// Process-wide singleton. First caller pays the env-read cost; every
    /// subsequent caller returns a cheap `&'static`.
    pub fn get() -> &'static Self {
        static CFG: OnceLock<RemoteMoeRuntime> = OnceLock::new();
        CFG.get_or_init(Self::from_env)
    }

    /// Inverse of `q8k_disabled` — what most callers actually want to
    /// branch on. Q8K is on by default; the env var is an opt-out.
    pub fn q8k_enabled(&self) -> bool {
        !self.q8k_disabled
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_documented_off_state() {
        let cfg = RemoteMoeRuntime::default();
        assert!(!cfg.http_timing);
        assert!(!cfg.wire_f16);
        assert!(!cfg.q8k_disabled);
        assert!(cfg.q8k_enabled(), "q8k is enabled by default");
        assert!(!cfg.verbose);
        assert!(!cfg.moe_bytes_enabled);
        assert!(!cfg.moe_shard_timing);
    }

    #[test]
    fn q8k_enabled_inverts_q8k_disabled() {
        let cfg = RemoteMoeRuntime {
            q8k_disabled: true,
            ..Default::default()
        };
        assert!(!cfg.q8k_enabled());
    }

    #[test]
    fn singleton_is_stable() {
        let a = RemoteMoeRuntime::get() as *const _;
        let b = RemoteMoeRuntime::get() as *const _;
        assert_eq!(a, b);
    }

    #[test]
    fn env_var_consts_match_legacy_literal_names() {
        // Pin the on-the-wire env var names — refactors that rename a
        // const must not silently break LARQL users' shell scripts.
        assert_eq!(ENV_HTTP_TIMING, "LARQL_HTTP_TIMING");
        assert_eq!(ENV_MOE_WIRE_F16, "LARQL_MOE_WIRE_F16");
        assert_eq!(ENV_DISABLE_Q8K_WIRE, "LARQL_DISABLE_Q8K_WIRE");
        assert_eq!(ENV_VERBOSE, "LARQL_VERBOSE");
        assert_eq!(ENV_MOE_BYTES, "LARQL_MOE_BYTES");
        assert_eq!(ENV_MOE_TIMING, "LARQL_MOE_TIMING");
        assert_eq!(ENV_MOE_SHARD_TIMING, "LARQL_MOE_SHARD_TIMING");
        assert_eq!(ENV_MOE_TOP_K, "LARQL_MOE_TOP_K");
        assert_eq!(ENV_MOE_NO_SPLIT, "LARQL_MOE_NO_SPLIT");
        assert_eq!(ENV_SKIP_MOE, "SKIP_MOE");
    }
}
