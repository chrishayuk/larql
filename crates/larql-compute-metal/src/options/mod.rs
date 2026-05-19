//! Decode hot-path env flags + backend-startup options, captured once
//! at `MetalBackend::with_options(...)`.
//!
//! Two structs live here:
//!
//! - [`DecodeFlags`] — per-token / per-layer fusion choices read by
//!   `decode_token_with_moe_split_fn` and its `encode_*` callees.
//!   Resolving these via `getenv` per layer would cost ~12 syscalls ×
//!   34 layers = ~400/token on Gemma 3 4B; settling them to a single
//!   load at construction keeps the hot path free of syscalls.
//!
//! - [`BackendOptions`] — programmatic surface for non-debug startup
//!   choices (kernel-variant selection, decode flags). Existing env
//!   vars stay as a compatibility / shell-debug bridge: the `from_env`
//!   constructors of both structs preserve the historical behaviour;
//!   calling code that wants explicit control passes a configured
//!   `BackendOptions` to `MetalBackend::with_options`.
//!
//! ## Contract for callers
//!
//! Once a backend is constructed, its flags are fixed. To reflect an
//! env-var flip, build a fresh backend (`MetalBackend::new()` or
//! `MetalBackend::with_options(...)`). Tests that toggle these vars
//! must do this — see the env-mutating tests in
//! `tests/test_metal_decode_synthetic.rs` and the `ENV_TEST_LOCK` that
//! serialises them.
//!
//! ## What lives here vs what doesn't
//!
//! Cached: flags consulted from the decode hot path. Off the hot path
//! (debug dumps, timing diagnostics, prefill-only paths) intentionally
//! read env per-call so they can be flipped without restarting — see
//! `LARQL_DEBUG_NAN_LAYERS`, `LARQL_GPU_TIMING`, `LARQL_DECODE_DUMP_LAYERS`.

use larql_compute::options::{
    env_flag, env_not_zero_or_default, env_opt_in, env_opt_out, ENV_F16_ACC, ENV_FUSED_ATTN,
    ENV_FUSED_DOWN, ENV_FUSED_KV_APPEND_ATTEND, ENV_FUSED_POST_ATTN_NORM, ENV_FUSED_POST_FFN_NORM,
    ENV_FUSED_PRELAYER_NORM, ENV_FUSED_Q6K_DOWN, ENV_FUSED_QK_NORM_ROPE, ENV_GATE_UP_8SG,
    ENV_GATE_UP_COOP, ENV_Q4K_MATVEC_8SG, ENV_Q6K_8SG, ENV_QKV_FUSED,
};

/// Decode-path flag snapshot captured at backend startup.
#[derive(Copy, Clone, Debug)]
pub struct DecodeFlags {
    /// `LARQL_GATE_UP_COOP=1` — cooperative-scale-load Q4_K gate+up
    /// variant. Opt-in (default false).
    pub gate_up_coop: bool,
    /// `LARQL_GATE_UP_8SG=0` opts OUT of the default 8sg Q4_K gate+up
    /// back to the older 4sg kernel. Stored as `use_4sg` so the
    /// reading site doesn't have to re-invert.
    pub gate_up_use_4sg: bool,
    /// `LARQL_F16_ACC=1` — f16 accumulator on the legacy 4sg gate+up.
    /// Opt-in (default false).
    pub f16_acc: bool,
    /// `LARQL_FUSED_Q6K_DOWN=1` — opt into the cached-activation Q6_K
    /// fused down. Currently no-op pending kernel parity (see
    /// `encode_ffn.rs` block doc).
    pub fused_q6k_down: bool,
    /// `LARQL_FUSED_DOWN` — fused Q4_K GEGLU+down on the decode hot
    /// path. Default true; opt out with `=0`. (The prefill path in
    /// `stages/ffn.rs` reads the same env var with default OFF — that
    /// asymmetry is intentional and not cached here; prefill is
    /// allocation-bound, the per-call env read is in the noise.)
    pub fused_down: bool,
    /// `LARQL_FUSED_PRELAYER_NORM=1` — D-RMS-FUSE Phase 1: fold the
    /// post-FFN residual_add into the next-layer input rms_norm. Opt-in.
    pub fused_prelayer_norm: bool,
    /// `LARQL_QKV_FUSED=1` — opt back IN to the (now-defaulted-off)
    /// `q4k_q6k_qkv_proj_normed` shader. Opt-in (ADR-016 flipped the
    /// default to defused on 2026-05-09).
    pub qkv_fused: bool,
    /// `LARQL_FUSED_ATTN=1` — opt into the triple-fused QK-norm + RoPE
    /// + KV-append + attend kernel. Opt-in.
    pub fused_attn: bool,
    /// `LARQL_FUSED_QK_NORM_ROPE` — default true; opt out with `=0`.
    pub fused_qk_norm_rope: bool,
    /// `LARQL_FUSED_KV_APPEND_ATTEND` — default true; opt out with `=0`.
    pub fused_kv_append_attend: bool,
    /// `LARQL_FUSED_POST_ATTN_NORM` — default true; opt out with `=0`.
    pub fused_post_attn_norm: bool,
    /// `LARQL_FUSED_POST_FFN_NORM` — default true; opt out with `=0`.
    pub fused_post_ffn_norm: bool,
}

impl DecodeFlags {
    /// Snapshot the env. Call once per `MetalBackend::new()`.
    pub fn from_env() -> Self {
        Self {
            gate_up_coop: env_opt_in(ENV_GATE_UP_COOP),
            gate_up_use_4sg: env_opt_out(ENV_GATE_UP_8SG),
            f16_acc: env_flag(ENV_F16_ACC),
            fused_q6k_down: env_flag(ENV_FUSED_Q6K_DOWN),
            fused_down: env_not_zero_or_default(ENV_FUSED_DOWN, true),
            fused_prelayer_norm: env_opt_in(ENV_FUSED_PRELAYER_NORM),
            qkv_fused: env_opt_in(ENV_QKV_FUSED),
            fused_attn: env_opt_in(ENV_FUSED_ATTN),
            fused_qk_norm_rope: !env_opt_out(ENV_FUSED_QK_NORM_ROPE),
            fused_kv_append_attend: !env_opt_out(ENV_FUSED_KV_APPEND_ATTEND),
            fused_post_attn_norm: !env_opt_out(ENV_FUSED_POST_ATTN_NORM),
            fused_post_ffn_norm: !env_opt_out(ENV_FUSED_POST_FFN_NORM),
        }
    }
}

/// Programmatic backend-startup options.
///
/// Bundles every kernel-variant choice + decode-path flag set the
/// Metal backend reads at construction. Callers that want explicit
/// control build a `BackendOptions`, mutate fields, and pass it to
/// [`crate::MetalBackend::with_options`]. The bare
/// [`crate::MetalBackend::new`] keeps the env-driven default
/// behaviour as a compatibility / shell-debug bridge.
///
/// New non-debug startup options should land here, not as fresh env
/// reads inside `MetalBackend::new()`.
#[derive(Copy, Clone, Debug)]
pub struct BackendOptions {
    /// Decode hot-path fusion flags. See [`DecodeFlags`].
    pub decode_flags: DecodeFlags,
    /// Use the legacy 4-simdgroup Q4_K matvec kernel instead of the
    /// 8sg default. The 8sg variant became the default 2026-04-28 after
    /// production-batched profiling; the 4sg kernel stays available as
    /// the explicit fallback (cross-hardware A/B + bisect aid).
    /// Env: `LARQL_Q4K_MATVEC_8SG=0` opts out of 8sg → sets this true.
    pub q4k_matvec_use_4sg: bool,
    /// Use the 8-simdgroup Q6_K matvec kernel. The 4sg variant is the
    /// production default (q6k was already at 84% LPDDR5X peak; 8sg
    /// kernel-isolated 1.96× did not translate end-to-end on M3 Max).
    /// Env: `LARQL_Q6K_8SG=1` opts in → sets this true.
    pub q6k_use_8sg: bool,
}

impl BackendOptions {
    /// Snapshot env-derived defaults. Used by
    /// [`crate::MetalBackend::new`] so the historical env
    /// behaviour keeps working unchanged.
    pub fn from_env() -> Self {
        Self {
            decode_flags: DecodeFlags::from_env(),
            q4k_matvec_use_4sg: env_opt_out(ENV_Q4K_MATVEC_8SG),
            q6k_use_8sg: env_opt_in(ENV_Q6K_8SG),
        }
    }
}

impl Default for BackendOptions {
    /// Programmatic defaults — independent of any env var. Use this
    /// from production code that doesn't want shell env to leak in.
    /// Today this lines up with `from_env()` on a clean process; if
    /// they ever drift, this default tracks the codebase's current
    /// "production" choices and `from_env()` tracks env semantics.
    fn default() -> Self {
        Self {
            decode_flags: DecodeFlags::default(),
            q4k_matvec_use_4sg: false, // 8sg default since 2026-04-28
            q6k_use_8sg: false,        // 4sg default
        }
    }
}

impl Default for DecodeFlags {
    /// Programmatic defaults — see [`BackendOptions::default`]. Mirrors
    /// the no-env-set state pinned by `from_env_defaults_when_no_vars_set`.
    fn default() -> Self {
        Self {
            gate_up_coop: false,
            gate_up_use_4sg: false,
            f16_acc: false,
            fused_q6k_down: false,
            fused_down: true,
            fused_prelayer_norm: false,
            qkv_fused: false,
            fused_attn: false,
            fused_qk_norm_rope: true,
            fused_kv_append_attend: true,
            fused_post_attn_norm: true,
            fused_post_ffn_norm: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the defaults: every flag must resolve to a known value with
    /// no env vars set. Captures the contract that
    /// `MetalBackend::new()` on a clean process is deterministic.
    #[test]
    fn from_env_defaults_when_no_vars_set() {
        // ENV vars may leak in from the running process. To make the
        // test robust we explicitly clear ours, snapshot, and restore.
        // Held under a static lock so this doesn't race other tests in
        // the file.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().expect("flag-test lock poisoned");

        let names = [
            ENV_GATE_UP_COOP,
            ENV_GATE_UP_8SG,
            ENV_F16_ACC,
            ENV_FUSED_Q6K_DOWN,
            ENV_FUSED_DOWN,
            ENV_FUSED_PRELAYER_NORM,
            ENV_QKV_FUSED,
            ENV_FUSED_ATTN,
            ENV_FUSED_QK_NORM_ROPE,
            ENV_FUSED_KV_APPEND_ATTEND,
            ENV_FUSED_POST_ATTN_NORM,
            ENV_FUSED_POST_FFN_NORM,
        ];
        let prior: Vec<_> = names.iter().map(|n| (*n, std::env::var_os(n))).collect();
        for n in names {
            std::env::remove_var(n);
        }

        let flags = DecodeFlags::from_env();

        // Opt-in flags (default false).
        assert!(!flags.gate_up_coop);
        assert!(!flags.gate_up_use_4sg); // opt-OUT of 8sg, so default false → use 8sg
        assert!(!flags.f16_acc);
        assert!(!flags.fused_q6k_down);
        assert!(!flags.fused_prelayer_norm);
        assert!(!flags.qkv_fused);
        assert!(!flags.fused_attn);

        // Default-on flags.
        assert!(flags.fused_down);
        assert!(flags.fused_qk_norm_rope);
        assert!(flags.fused_kv_append_attend);
        assert!(flags.fused_post_attn_norm);
        assert!(flags.fused_post_ffn_norm);

        for (n, v) in prior {
            match v {
                Some(val) => std::env::set_var(n, val),
                None => std::env::remove_var(n),
            }
        }
    }

    /// `BackendOptions::default()` and `BackendOptions::from_env()` on
    /// a clean process must produce the same struct. If they ever
    /// diverge it should be a deliberate codebase decision (see the
    /// doc on `Default`).
    #[test]
    fn backend_options_default_matches_clean_env() {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().expect("flag-test lock poisoned");

        let names = [
            ENV_GATE_UP_COOP,
            ENV_GATE_UP_8SG,
            ENV_F16_ACC,
            ENV_FUSED_Q6K_DOWN,
            ENV_FUSED_DOWN,
            ENV_FUSED_PRELAYER_NORM,
            ENV_QKV_FUSED,
            ENV_FUSED_ATTN,
            ENV_FUSED_QK_NORM_ROPE,
            ENV_FUSED_KV_APPEND_ATTEND,
            ENV_FUSED_POST_ATTN_NORM,
            ENV_FUSED_POST_FFN_NORM,
            ENV_Q4K_MATVEC_8SG,
            ENV_Q6K_8SG,
        ];
        let prior: Vec<_> = names.iter().map(|n| (*n, std::env::var_os(n))).collect();
        for n in names {
            std::env::remove_var(n);
        }

        let from_env = BackendOptions::from_env();
        let default = BackendOptions::default();

        // Field-by-field — `BackendOptions: !PartialEq` because
        // `DecodeFlags` doesn't derive it (deliberate; debug-only struct).
        assert_eq!(from_env.q4k_matvec_use_4sg, default.q4k_matvec_use_4sg);
        assert_eq!(from_env.q6k_use_8sg, default.q6k_use_8sg);
        assert_eq!(
            from_env.decode_flags.fused_down,
            default.decode_flags.fused_down
        );
        assert_eq!(
            from_env.decode_flags.fused_qk_norm_rope,
            default.decode_flags.fused_qk_norm_rope
        );
        assert_eq!(
            from_env.decode_flags.gate_up_coop,
            default.decode_flags.gate_up_coop
        );
        assert_eq!(
            from_env.decode_flags.qkv_fused,
            default.decode_flags.qkv_fused
        );

        for (n, v) in prior {
            match v {
                Some(val) => std::env::set_var(n, val),
                None => std::env::remove_var(n),
            }
        }
    }

    /// `BackendOptions::default()` is independent of env state — toggling
    /// vars must not change what `default()` returns.
    #[test]
    fn backend_options_default_ignores_env() {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().expect("flag-test lock poisoned");

        let prior_qkv = std::env::var_os(ENV_QKV_FUSED);
        let prior_4sg = std::env::var_os(ENV_Q4K_MATVEC_8SG);

        std::env::set_var(ENV_QKV_FUSED, "1");
        std::env::set_var(ENV_Q4K_MATVEC_8SG, "0");

        let default = BackendOptions::default();
        // Default ignores env entirely.
        assert!(!default.decode_flags.qkv_fused);
        assert!(!default.q4k_matvec_use_4sg);

        // Sanity: from_env DOES pick them up under the same conditions.
        let from_env = BackendOptions::from_env();
        assert!(from_env.decode_flags.qkv_fused);
        assert!(from_env.q4k_matvec_use_4sg);

        match prior_qkv {
            Some(v) => std::env::set_var(ENV_QKV_FUSED, v),
            None => std::env::remove_var(ENV_QKV_FUSED),
        }
        match prior_4sg {
            Some(v) => std::env::set_var(ENV_Q4K_MATVEC_8SG, v),
            None => std::env::remove_var(ENV_Q4K_MATVEC_8SG),
        }
    }
}
