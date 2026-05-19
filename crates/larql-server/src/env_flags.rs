//! Centralised environment-variable knobs.
//!
//! Every `LARQL_*` env var read by the server lives here. The names are
//! exported as `pub const` so call sites and the README env-var table
//! reference the same string. Cached accessors avoid the per-call syscall on
//! hot paths (`forward_moe`, `handle_layer_batch`).
//!
//! See README.md → "Environment variables" for what each flag does.

use std::sync::OnceLock;

// ── Names ──────────────────────────────────────────────────────────────────────
//
// Strings only — no semantics. README cross-references these by name.

/// Per-token MoE timing summary.
pub const MOE_TIMING: &str = "LARQL_MOE_TIMING";
/// Per-call HTTP/UDS timing breakdown.
pub const HTTP_TIMING: &str = "LARQL_HTTP_TIMING";
/// Skip Metal expert / HNSW cache pre-population at boot.
pub const NO_WARMUP: &str = "LARQL_NO_WARMUP";
/// Force the legacy CPU-rayon expert path (skip the layer-batch fast path).
pub const USE_LEGACY_CPU: &str = "LARQL_USE_LEGACY_CPU";
/// Opt-in: route experts through Metal (correctness-blocked, see ROADMAP).
pub const USE_METAL_EXPERTS: &str = "LARQL_USE_METAL_EXPERTS";
/// Hard-disable the Metal expert path even on `metal-experts` builds.
pub const DISABLE_METAL_EXPERTS: &str = "LARQL_DISABLE_METAL_EXPERTS";
/// Disable the SDOT direct-Q4K matvec; fall back to BLAS-on-cached-f32.
pub const DISABLE_Q4K_DIRECT: &str = "LARQL_DISABLE_Q4K_DIRECT";
/// Server-side per-call A/B compare Metal vs CPU expert outputs.
pub const METAL_VS_CPU_DEBUG: &str = "LARQL_METAL_VS_CPU_DEBUG";
/// Override the auto-selected MoE batch dispatch mode.
pub const MOE_BATCH_MODE: &str = "LARQL_MOE_BATCH_MODE";
/// Opt-out of f16 wire format for grid traffic. Set to any value to force f32.
/// Default (unset): f16 wire is used when the client advertises Accept: f16.
pub const F16_WIRE_DISABLE: &str = "LARQL_F16_WIRE_DISABLE";
/// Opt-in to i8 symmetric quantised residuals on the wire.
pub const I8_WIRE: &str = "LARQL_I8_WIRE";

// ── Cached presence ────────────────────────────────────────────────────────────
//
// `is_ok()` semantics: any value (including empty) enables the flag. Cached
// in process-wide `OnceLock`s — env vars don't change at runtime, and the
// per-call syscall used to show up in HTTP-path traces.

fn cached_is_set(slot: &OnceLock<bool>, name: &'static str) -> bool {
    *slot.get_or_init(|| std::env::var(name).is_ok())
}

/// `LARQL_MOE_TIMING=1` — per-token MoE breakdown on stderr.
pub fn moe_timing_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, MOE_TIMING)
}

/// `LARQL_HTTP_TIMING=1` — per-call HTTP/UDS breakdown on stderr.
pub fn http_timing_enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, HTTP_TIMING)
}

/// `LARQL_NO_WARMUP=1` — skip warmup helpers at boot.
pub fn no_warmup() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, NO_WARMUP)
}

/// `LARQL_USE_LEGACY_CPU=1` — route experts through the legacy CPU path.
pub fn use_legacy_cpu() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, USE_LEGACY_CPU)
}

/// `LARQL_USE_METAL_EXPERTS=1` — opt in to the Metal expert kernel.
pub fn use_metal_experts() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, USE_METAL_EXPERTS)
}

/// `LARQL_DISABLE_METAL_EXPERTS=1` — hard-disable Metal experts.
pub fn disable_metal_experts() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, DISABLE_METAL_EXPERTS)
}

/// `LARQL_DISABLE_Q4K_DIRECT=1` — fall back to BLAS for the gate/up matvec.
pub fn disable_q4k_direct() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, DISABLE_Q4K_DIRECT)
}

/// `LARQL_METAL_VS_CPU_DEBUG=1` — run both Metal and CPU per call, log diff.
pub fn metal_vs_cpu_debug() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    cached_is_set(&CACHE, METAL_VS_CPU_DEBUG)
}

/// `LARQL_MOE_BATCH_MODE=<mode>` — override the auto-selected batch mode.
/// Returns `None` when unset; the caller decides what's valid.
pub fn moe_batch_mode() -> Option<String> {
    std::env::var(MOE_BATCH_MODE).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(test, target_arch = "wasm32"))]
    use wasm_bindgen_test::wasm_bindgen_test;
    #[cfg(all(test, target_arch = "wasm32"))]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn names_are_larql_prefixed_and_unique() {
        let names = [
            MOE_TIMING,
            HTTP_TIMING,
            NO_WARMUP,
            USE_LEGACY_CPU,
            USE_METAL_EXPERTS,
            DISABLE_METAL_EXPERTS,
            DISABLE_Q4K_DIRECT,
            METAL_VS_CPU_DEBUG,
            MOE_BATCH_MODE,
            F16_WIRE_DISABLE,
            I8_WIRE,
        ];
        for n in names {
            assert!(n.starts_with("LARQL_"), "{n} must be LARQL_-prefixed");
        }
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "env-var names must be unique");
    }
}
