//! Runtime options and environment-variable names for compute backends.
//!
//! Keep process-global debug and experiment toggles here instead of spelling
//! string literals through hot paths. Most callers should eventually pass an
//! explicit options struct; this module is the compatibility bridge while those
//! APIs are split out.
//!
//! ## Environment-variable surface (the categories)
//!
//! - **Decode fast path — default ON, opt out with `=0`.** The shipped CPU
//!   decode default; you do *not* set anything to go fast. Resolvers:
//!   [`q4k_direct_attn_enabled`], [`q4k_attn_int8_enabled`],
//!   [`q4k_lm_head_enabled`], [`q4k_direct_ffn_enabled`], [`q4k_asm_enabled`],
//!   [`spin_pool_enabled`] (env `LARQL_Q4K_*`, `LARQL_SPIN_POOL`). Set any to
//!   `0`/`false`/`off`/`no` to force the f32/rayon path (A/B, kernel debug).
//! - **Diagnostics / dumps — presence = on** (`env_flag`): the `LARQL_*_DUMP_*`,
//!   `*_TIMING`, `LARQL_PROFILE_SPLIT`, `LARQL_DECODE_STAGES`,
//!   `LARQL_VINDEX_DESCRIBE`, `LARQL_MOE_DEBUG` toggles. Off unless set.
//! - **Retained comparison knobs** (ADR-017 shader/kernel retention): the
//!   fused-shader flags (`LARQL_QKV_FUSED`, `LARQL_FUSED_*`) and the `asm_v2`
//!   bench arm — deliberately kept for A/B, *not* dead code.
//! - **Config / paths / experiment / test** live with their feature, not here
//!   (`HF_*`, `LARQL_HOME`, `LARQL_MODEL`, `LARQL_MEMIT_*`, `LARQL_TEST_*`, …).
//!
//! ## Helper taxonomy — pick the matching one for the flag's intended default
//!
//! Mixing these on the same env var is the bug class flagged in the
//! larql-compute review (`stages/ffn.rs` reads `LARQL_FUSED_DOWN` with
//! [`env_flag`] = default OFF, while `decode/encode_ffn.rs` reads it with
//! [`env_not_zero_or_default(_, true)`] = default ON). When in doubt
//! about a flag, prefer [`env_opt_in`] / [`env_opt_out`] — they ignore
//! `set-but-empty`, which is a common shell-export footgun.
//!
//! | Helper                                  | Default | True when env is …                          | Best for                                  |
//! |-----------------------------------------|---------|---------------------------------------------|-------------------------------------------|
//! | [`env_flag`]                            | false   | set (any value, including empty)            | debug toggles, dump destinations          |
//! | [`env_opt_in`]                          | false   | exactly `1` / `true` / `on` / `yes`         | opt-in experiments (cooperative kernels)  |
//! | [`env_opt_out`]                         | false   | exactly `0` / `false` / `off` / `no`        | opt-OUT of a default-on path              |
//! | `!env_opt_out(name)`                    | true    | env unset OR not in opt-out vocabulary      | default-on fusion paths                   |
//! | [`env_not_zero_or_default`]`(name, d)`  | `d`     | env set AND not exactly `0`                 | "default true unless explicitly disabled" |
//!
//! ### Picking helpers for new flags
//!
//! - **Default-OFF, opt-in**: use [`env_opt_in`]. Setting `LARQL_X=` (empty)
//!   stays OFF.  This is the right shape for new experiments where bare
//!   shell-exports (`export LARQL_X` with no value) shouldn't accidentally
//!   activate the path.
//! - **Default-ON, opt-out**: use `!env_opt_out(name)`. Setting `LARQL_X=0`
//!   disables; `LARQL_X=` (empty) keeps the default.
//! - **Diagnostic toggle, presence-as-truth**: use [`env_flag`]. Convenient
//!   for "set this var to anything, I just need to know it was requested".
//!
//! Cache hot-path env reads at backend construction (see
//! `metal::flags::DecodeFlags`) — repeated `getenv` per layer per token
//! costs measurable syscalls.

/// Enable timing around the full CPU MoE forward pass.
pub const ENV_MOE_FWD_TIMING: &str = "LARQL_MOE_FWD_TIMING";
/// Enable timing around one CPU MoE expert.
pub const ENV_MOE_EXPERT_TIMING: &str = "LARQL_MOE_EXPERT_TIMING";
/// Enable timing inside the direct Q4_K expert kernel.
pub const ENV_KERNEL_TIMING: &str = "LARQL_KERNEL_TIMING";
/// Disable the direct Q4_K/Q8_K CPU MoE path.
pub const ENV_DISABLE_Q4K_DIRECT: &str = "LARQL_DISABLE_Q4K_DIRECT";
/// Opt in to the older scalar Q4_K direct path in `run_single_expert_into`.
pub const ENV_Q4K_DIRECT: &str = "LARQL_Q4K_DIRECT";
/// Max entries in the dequantised MoE expert cache.
pub const ENV_MOE_CACHE_ENTRIES: &str = "LARQL_MOE_CACHE_ENTRIES";
/// MoE bypass toggle (diagnostic). The DEC-0 "SKIP_MOE ceiling" anchor
/// arm is measured with this flag — one canonical name protects it.
pub const ENV_SKIP_MOE: &str = "LARQL_SKIP_MOE";
/// Legacy unprefixed alias for [`ENV_SKIP_MOE`] (the grid path's historical
/// name) — honoured with a one-time warning via [`skip_moe_enabled`].
pub const ENV_SKIP_MOE_LEGACY: &str = "SKIP_MOE";
/// MoE route/debug output toggle.
pub const ENV_MOE_DEBUG: &str = "LARQL_MOE_DEBUG";
/// Path for the per-token expert-routing trace written by the reference MoE
/// backend. Value-carrying and opt-in: unset or empty means no trace.
/// See `larql-compute/src/ffn/expert_weight/trace.rs`.
pub const ENV_MOE_ROUTE_TRACE: &str = "LARQL_MOE_ROUTE_TRACE";
/// Enable Metal MoE dispatch timing.
pub const ENV_METAL_MOE_TIMING: &str = "LARQL_MOE_TIMING";
/// Select the 8-simdgroup Q4_K matvec kernel; set to a false value to opt out.
pub const ENV_Q4K_MATVEC_8SG: &str = "LARQL_Q4K_MATVEC_8SG";
/// Opt in to the 8-simdgroup Q6_K matvec kernel.
pub const ENV_Q6K_8SG: &str = "LARQL_Q6K_8SG";
/// Select the MXFP4 grouped-expert arm (`a`..`d`, or the arm name).
/// Unset or unrecognised selects the exact split-stream arm.
pub const ENV_MXFP4_ARM: &str = "LARQL_MXFP4_ARM";
/// Opt in to fused attention.
pub const ENV_FUSED_ATTN: &str = "LARQL_FUSED_ATTN";
/// Disable fused QK-norm + RoPE when set to a false value.
pub const ENV_FUSED_QK_NORM_ROPE: &str = "LARQL_FUSED_QK_NORM_ROPE";
/// Disable fused KV append + attend when set to a false value.
pub const ENV_FUSED_KV_APPEND_ATTEND: &str = "LARQL_FUSED_KV_APPEND_ATTEND";
/// Disable fused post-attention norm when set to a false value.
pub const ENV_FUSED_POST_ATTN_NORM: &str = "LARQL_FUSED_POST_ATTN_NORM";
/// Disable fused post-FFN norm when set to a false value.
pub const ENV_FUSED_POST_FFN_NORM: &str = "LARQL_FUSED_POST_FFN_NORM";
/// Opt in to fusing the post-FFN residual_add (non-Gemma archs) with the
/// NEXT layer's input rms_norm in one `residual_norm_store` dispatch.
/// Saves 1 rms_norm dispatch per layer × num_layers on Llama / Mistral /
/// Qwen / etc. (Gemma 3/4 already use the post_norms triple-fusion path,
/// so this is a no-op there.) D-RMS-FUSE Phase 1.
pub const ENV_FUSED_PRELAYER_NORM: &str = "LARQL_FUSED_PRELAYER_NORM";
/// Opt in to the cooperative gate+up kernel variant.
pub const ENV_GATE_UP_COOP: &str = "LARQL_GATE_UP_COOP";
/// Opt back in to the fused `q4k_q6k_qkv_proj_normed` shader (RMS norm
/// rolled into the matmul). The defused path (separate `rms_norm` +
/// non-fused `q4k_q6k_qkv_proj`) is the default since 2026-05-09 because
/// end-to-end A/B on Gemma 3 4B showed +1.6 tok/s (−0.30 ms/tok GPU fwd):
/// the fused kernel rereads H + norm_w 3× per TG (dropping it from 287
/// to 199 GB/s) and that bandwidth waste exceeds the 0.24 ms/tok dispatch
/// saving the fusion gave. Set this to compare against the old default.
pub const ENV_QKV_FUSED: &str = "LARQL_QKV_FUSED";
/// Select the 8-simdgroup gate+up kernel; set to a false value to opt out.
pub const ENV_GATE_UP_8SG: &str = "LARQL_GATE_UP_8SG";
/// Opt in to f16 accumulation for the legacy gate+up kernel.
pub const ENV_F16_ACC: &str = "LARQL_F16_ACC";
/// Opt in to experimental fused Q6_K down routing.
pub const ENV_FUSED_Q6K_DOWN: &str = "LARQL_FUSED_Q6K_DOWN";
/// Fused Q4_K down routing toggle. Existing decode code only treats `0` as off.
pub const ENV_FUSED_DOWN: &str = "LARQL_FUSED_DOWN";
/// Print the Q4_K quant-matvec dispatch route.
pub const ENV_DBG_QM: &str = "LARQL_DBG_QM";
/// One-line summary for the first few Metal decode calls.
pub const ENV_DECODE_DEBUG: &str = "LARQL_DECODE_DEBUG";
/// Legacy unprefixed alias for [`ENV_DECODE_DEBUG`] — honoured with a
/// one-time warning via [`decode_debug_enabled`]. Unprefixed names can
/// collide with a rented host's ambient env (dec-readiness review §3e).
pub const ENV_DECODE_DEBUG_LEGACY: &str = "DECODE_DEBUG";
/// Dump per-layer residuals to a binary file.
pub const ENV_DUMP_RESIDUALS: &str = "LARQL_DUMP_RESIDUALS";
/// Stop Metal decode at this layer and dump intermediate buffers.
pub const ENV_DECODE_DIAG_LAYER: &str = "LARQL_DECODE_DIAG_LAYER";
/// Dump Gemma-4-MoE layer-0 intermediates.
pub const ENV_DUMP_L0: &str = "LARQL_DUMP_L0";
/// Force per-layer NaN diagnostics in Metal decode.
pub const ENV_DEBUG_NAN_LAYERS: &str = "LARQL_DEBUG_NAN_LAYERS";
/// Dump Metal decode layer outputs.
pub const ENV_DECODE_DUMP_LAYERS: &str = "LARQL_DECODE_DUMP_LAYERS";
/// Dump Metal full-pipeline layer outputs.
pub const ENV_METAL_DUMP_LAYERS: &str = "LARQL_METAL_DUMP_LAYERS";
/// Layer index for stage-level dump helpers.
pub const ENV_STAGE_DUMP_LAYER: &str = "LARQL_STAGE_DUMP_LAYER";
/// Print GPU-side command-buffer timing.
pub const ENV_GPU_TIMING: &str = "LARQL_GPU_TIMING";
/// Print the BW10 per-token movement/causality ledger. `1` emits the
/// steady-state summary at end of run; `token` additionally emits the
/// compact per-token line.
pub const ENV_MOVEMENT_LEDGER: &str = "LARQL_MOVEMENT_LEDGER";
/// Declare which regime the ledger's measurement targets — one of
/// `resident`, `capacity-constrained`, `cold-estate`. Without it the
/// ledger prints raw counters but REFUSES the verdict line, because the
/// same byte delta licenses opposite conclusions across regimes.
pub const ENV_MOVEMENT_REGIME: &str = "LARQL_MOVEMENT_REGIME";
/// Install an execution policy for the decode path — the one way to arm
/// `larql_compute::exec_policy`'s seam from outside the process, so a
/// bytes-and-latency A/B is a change of environment rather than a change
/// of code. `skip-layers:<L>[,<L>...][:every-<n>|:token-<n>]` or
/// `trace:<path>`; see that module's `spec` submodule. Unset means
/// canonical execution; an UNPARSEABLE value is a hard error, never a
/// silent fallback (a silent one turns an A/B into canonical-vs-canonical
/// and reports "no change").
pub const ENV_EXEC_POLICY: &str = "LARQL_EXEC_POLICY";
/// Request paired commit/wait decode stage profiling.
pub const ENV_PROFILE_SPLIT: &str = "LARQL_PROFILE_SPLIT";
/// Debug-only outer norm bypass in Metal MoE combine.
pub const ENV_SKIP_OUTER_NORM: &str = "LARQL_SKIP_OUTER_NORM";
/// Legacy unprefixed alias for [`ENV_SKIP_OUTER_NORM`] — honoured with a
/// one-time warning via [`skip_outer_norm_enabled`].
pub const ENV_SKIP_OUTER_NORM_LEGACY: &str = "SKIP_OUTER_NORM";

// ── CPU decode fast path — default ON, opt out with `=0` ─────────────────────
//
// These graduated from opt-in experiments (2026-06) to the shipped default:
// together they take CPU MoE decode from ~7 tok/s (f32 fallback) to ~35 on the
// 26B-A4B, parity-safe, with per-layer/format fallbacks (a layer/model that
// can't take the fast route silently uses the f32 one). Disable any single
// stage with `LARQL_<NAME>=0` (also accepts `false`/`off`/`no`) — e.g. for an
// A/B against the f32 path or to debug a kernel.
//
/// Q4_K-direct attention projections (read Q4_K weights straight from the index
/// instead of dequantising to f32 first).
pub const ENV_Q4K_DIRECT_ATTN: &str = "LARQL_Q4K_DIRECT_ATTN";
/// Int8 (Q8_K) activation route for the Q4_K-direct attention projections.
pub const ENV_Q4K_ATTN_INT8: &str = "LARQL_Q4K_ATTN_INT8";
/// Q4_K lm_head (vocab projection straight from the Q4_K view; ~4× the
/// bandwidth of the f32 head). Falls back to f32 when no Q4_K head view exists.
pub const ENV_Q4K_LM_HEAD: &str = "LARQL_Q4K_LM_HEAD";
/// Q4_K-direct dense-FFN slab on the decode path (prefill stays f32 gemm).
pub const ENV_Q4K_DIRECT_FFN: &str = "LARQL_Q4K_DIRECT_FFN";
/// Hand-asm aarch64 Q4_K/Q6_K × Q8_K kernels (bit-exact with the intrinsic path).
pub const ENV_Q4K_ASM: &str = "LARQL_Q4K_ASM";
/// Spin-barrier thread pool for the decode hot path (vs rayon's sleeping pool).
pub const ENV_SPIN_POOL: &str = "LARQL_SPIN_POOL";

thread_local! {
    /// Per-thread override for env-var reads ([`env_override`]). Tests inject
    /// values here to toggle a flag WITHOUT `std::env::set_var`, which is
    /// thread-unsafe against the concurrent `getenv` every other parallel test
    /// does on the decode path — that race SIGSEGVs libc. Each entry is the raw
    /// value the env helper should see: `Some("v")` = "set to v", `None` = "act
    /// as if unset". Production never touches this; the map is empty so every
    /// helper falls through to the process env unchanged.
    static ENV_OVERRIDES: std::cell::RefCell<
        std::collections::HashMap<&'static str, Option<String>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The current thread's test override for `name`, if any. The outer `Option`
/// tells overridden-vs-not; the inner is the (possibly-unset) raw value.
fn env_override(name: &str) -> Option<Option<String>> {
    ENV_OVERRIDES.with(|o| o.borrow().get(name).cloned())
}

/// Effective raw value for `name`: the thread-local override if present, else
/// the process env. The single choke point every env helper reads through.
fn env_effective(name: &str) -> Option<String> {
    match env_override(name) {
        Some(v) => v,
        None => std::env::var(name).ok(),
    }
}

// ── Pure value parsers (no env) — directly unit-tested; the env helpers below
//    just feed them the effective raw value. Keeps the "0"/"true"/… vocabulary
//    in one place and testable without touching process env.
fn is_opt_out_value(v: Option<&str>) -> bool {
    matches!(v, Some("0") | Some("false") | Some("off") | Some("no"))
}
fn is_opt_in_value(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true") | Some("on") | Some("yes"))
}

/// The current thread's override for a fast-path stage flag as a bool, if set.
/// `None` in production → the accessor uses the cached [`decode_options`] value.
fn fast_path_override(name: &'static str) -> Option<bool> {
    env_override(name).map(|v| !is_opt_out_value(v.as_deref()))
}

/// Override an env flag on the current thread to a raw string value (`Some`) or
/// unset (`None`) — test-only escape hatch ([`ENV_OVERRIDES`]). Lets tests
/// toggle any flag without process-global env mutation (which segfaults under
/// parallel `getenv`). Clear with [`clear_fast_path_overrides`] on teardown.
#[doc(hidden)]
pub fn set_env_override(name: &'static str, value: Option<&str>) {
    ENV_OVERRIDES.with(|o| {
        o.borrow_mut().insert(name, value.map(str::to_string));
    });
}

/// Override a decode fast-path stage flag on the current thread (test-only).
/// Bool convenience over [`set_env_override`] (`true` → `"1"`, `false` → `"0"`).
#[doc(hidden)]
pub fn set_fast_path_override(name: &'static str, on: bool) {
    set_env_override(name, Some(if on { "1" } else { "0" }));
}

/// Clear all thread-local env overrides (test-only).
#[doc(hidden)]
pub fn clear_fast_path_overrides() {
    ENV_OVERRIDES.with(|o| o.borrow_mut().clear());
}

/// RAII guard that sets fast-path stage overrides on the current thread and
/// clears them on drop (test-only). Replaces the `std::env::set_var` pattern,
/// which races concurrent `getenv` on the decode path and SIGSEGVs libc.
#[cfg(test)]
pub(crate) struct FastPathGuard;

#[cfg(test)]
impl FastPathGuard {
    pub(crate) fn set(flags: &[(&'static str, bool)]) -> Self {
        for &(name, on) in flags {
            set_fast_path_override(name, on);
        }
        FastPathGuard
    }
}

#[cfg(test)]
impl Drop for FastPathGuard {
    fn drop(&mut self) {
        clear_fast_path_overrides();
    }
}

/// The decode fast-path stage flags — the single source of truth for "which
/// decode stages are on". Read ONCE from the process env at first use and
/// cached (see [`decode_options`]); each stage is default-ON, opt out with
/// `LARQL_<X>=0`. This folds what were four per-token `getenv`s and two ad-hoc
/// per-stage `OnceLock`s (`asm`, `spin_pool`) into one typed registry. Tests
/// toggle stages per-thread via [`set_fast_path_override`] (no `set_var`, which
/// races the per-token `getenv` and SIGSEGVs libc), and the override wins over
/// this cache.
#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    /// Q4_K-direct attention projections (read Q4_K bytes from the index).
    pub q4k_direct_attn: bool,
    /// Int8 (Q8_K) activation route for the Q4_K-direct attention projections.
    pub q4k_attn_int8: bool,
    /// Q4_K lm_head (vocab projection straight from the Q4_K view).
    pub q4k_lm_head: bool,
    /// Q4_K-direct dense-FFN decode slab (prefill stays f32 gemm).
    pub q4k_direct_ffn: bool,
    /// Hand-asm aarch64 Q4_K/Q6_K kernels (bit-exact with the intrinsic path).
    pub q4k_asm: bool,
    /// Spin-barrier thread pool for the decode hot path (vs rayon's pool).
    pub spin_pool: bool,
}

impl DecodeOptions {
    fn from_env() -> Self {
        // RAW process env (bypass the per-thread override): this is the
        // process-wide cached production value, and a test's thread-local
        // override must not be baked into it (the accessors apply the override
        // per-call instead, via `fast_path_override`).
        let on = |name: &str| !is_opt_out_value(std::env::var(name).ok().as_deref());
        Self {
            q4k_direct_attn: on(ENV_Q4K_DIRECT_ATTN),
            q4k_attn_int8: on(ENV_Q4K_ATTN_INT8),
            q4k_lm_head: on(ENV_Q4K_LM_HEAD),
            q4k_direct_ffn: on(ENV_Q4K_DIRECT_FFN),
            q4k_asm: on(ENV_Q4K_ASM),
            spin_pool: on(ENV_SPIN_POOL),
        }
    }
}

/// Process-wide decode fast-path flags, built from env on first use and cached.
/// The single registry the per-stage `*_enabled()` accessors read.
pub fn decode_options() -> &'static DecodeOptions {
    static OPTS: std::sync::OnceLock<DecodeOptions> = std::sync::OnceLock::new();
    OPTS.get_or_init(DecodeOptions::from_env)
}

/// Q4_K-direct attention projections enabled (default on).
pub fn q4k_direct_attn_enabled() -> bool {
    fast_path_override(ENV_Q4K_DIRECT_ATTN).unwrap_or(decode_options().q4k_direct_attn)
}
/// Int8 attention projection route enabled (default on).
pub fn q4k_attn_int8_enabled() -> bool {
    fast_path_override(ENV_Q4K_ATTN_INT8).unwrap_or(decode_options().q4k_attn_int8)
}
/// Q4_K lm_head enabled (default on; falls back to f32 without a head view).
pub fn q4k_lm_head_enabled() -> bool {
    fast_path_override(ENV_Q4K_LM_HEAD).unwrap_or(decode_options().q4k_lm_head)
}
/// Q4_K-direct dense-FFN decode slab enabled (default on).
pub fn q4k_direct_ffn_enabled() -> bool {
    fast_path_override(ENV_Q4K_DIRECT_FFN).unwrap_or(decode_options().q4k_direct_ffn)
}
/// Hand-asm Q4_K/Q6_K kernels enabled (default on; aarch64 only).
pub fn q4k_asm_enabled() -> bool {
    fast_path_override(ENV_Q4K_ASM).unwrap_or(decode_options().q4k_asm)
}
/// Spin-barrier decode pool enabled (default on).
pub fn spin_pool_enabled() -> bool {
    fast_path_override(ENV_SPIN_POOL).unwrap_or(decode_options().spin_pool)
}

/// One-line summary of every env toggle that changes numerical output or
/// decode-path selection — the flag-state twin of
/// `q4k_q8k_dot::kernel_class_summary()`. Logged once at server startup
/// (and quotable in DEC run records) so no measured number is ever
/// recorded against unlogged flag state (dec-readiness review §3e / run-
/// record hygiene). Bypass flags that CORRUPT a production number
/// (`skip_moe`, `skip_outer_norm`) are only mentioned when set, in
/// upper case, so they stand out in a log grep.
pub fn decode_options_summary() -> String {
    let opts = decode_options();
    let on_off = |b: bool| if b { "on" } else { "off" };
    let mut s = format!(
        "q4k_direct_attn={} q4k_attn_int8={} q4k_lm_head={} q4k_direct_ffn={} q4k_asm={} spin_pool={} moe_q4k_direct={}",
        on_off(opts.q4k_direct_attn),
        on_off(opts.q4k_attn_int8),
        on_off(opts.q4k_lm_head),
        on_off(opts.q4k_direct_ffn),
        on_off(opts.q4k_asm),
        on_off(opts.spin_pool),
        // MoE expert dispatch kernel path: direct Q4K×Q8K matvec (default)
        // vs BLAS on cached f32 dequant — a silently different
        // byte-movement regime if left unlogged (dec-readiness review).
        on_off(!env_flag(ENV_DISABLE_Q4K_DIRECT)),
    );
    if skip_moe_enabled() {
        s.push_str(" SKIP_MOE=ON");
    }
    if skip_outer_norm_enabled() {
        s.push_str(" SKIP_OUTER_NORM=ON");
    }
    s
}

// Helpers below are `pub` (not `pub(crate)`) because sibling backend
// crates (`larql-compute-metal`, future `larql-compute-vulkan`, …)
// share the same env-toggle vocabulary defined above.  Keeping the
// parsers private would force every backend to duplicate the
// `env::var_os`/`parse::<usize>` boilerplate and risk drift in how
// "set" / "true" / "1" are interpreted across backends.

// All of these read through `env_effective` so the thread-local test override
// applies uniformly (no `std::env::set_var` in tests → no `setenv`/`getenv`
// SIGSEGV race). In production the override map is empty, so each is exactly
// the prior `std::env::var*` read.
pub fn env_flag(name: &str) -> bool {
    match env_override(name) {
        Some(v) => v.is_some(),
        None => std::env::var_os(name).is_some(),
    }
}

pub fn env_flag_any(names: &[&str]) -> bool {
    names.iter().any(|name| env_flag(name))
}

pub fn env_usize(name: &str) -> Option<usize> {
    env_effective(name)?.parse().ok()
}

pub fn env_value(name: &str) -> Option<String> {
    env_effective(name)
}

pub fn env_nonempty_value(name: &str) -> Option<String> {
    env_value(name).filter(|value| !value.is_empty())
}

pub fn env_opt_in(name: &str) -> bool {
    is_opt_in_value(env_effective(name).as_deref())
}

pub fn env_opt_out(name: &str) -> bool {
    is_opt_out_value(env_effective(name).as_deref())
}

pub fn env_not_zero_or_default(name: &str, default: bool) -> bool {
    env_effective(name)
        .map(|value| value != "0")
        .unwrap_or(default)
}

/// Presence check on a canonical env name with a deprecated unprefixed
/// alias: the canonical name wins; the alias still works but logs a
/// one-time warning (unprefixed names can collide with a rented host's
/// ambient env — dec-readiness review §3e).
fn env_flag_with_loud_legacy(
    canonical: &'static str,
    legacy: &'static str,
    warned: &'static std::sync::Once,
) -> bool {
    if env_flag(canonical) {
        return true;
    }
    if env_flag(legacy) {
        warned.call_once(|| {
            eprintln!(
                "[larql] WARNING: env var `{legacy}` is a deprecated alias — \
                 use `{canonical}` (unprefixed names can collide with ambient host env)"
            );
        });
        return true;
    }
    false
}

pub(crate) fn moe_debug_enabled() -> bool {
    env_flag(ENV_MOE_DEBUG)
}

/// MoE bypass (the DEC-0 ceiling arm's flag): `LARQL_SKIP_MOE`, with the
/// grid path's historical `SKIP_MOE` as a loud deprecated alias.
pub fn skip_moe_enabled() -> bool {
    static WARNED: std::sync::Once = std::sync::Once::new();
    env_flag_with_loud_legacy(ENV_SKIP_MOE, ENV_SKIP_MOE_LEGACY, &WARNED)
}

/// Metal decode-call debug summary: `LARQL_DECODE_DEBUG`, with unprefixed
/// `DECODE_DEBUG` as a loud deprecated alias.
pub fn decode_debug_enabled() -> bool {
    static WARNED: std::sync::Once = std::sync::Once::new();
    env_flag_with_loud_legacy(ENV_DECODE_DEBUG, ENV_DECODE_DEBUG_LEGACY, &WARNED)
}

/// Debug-only outer-norm bypass: `LARQL_SKIP_OUTER_NORM`, with unprefixed
/// `SKIP_OUTER_NORM` as a loud deprecated alias.
pub fn skip_outer_norm_enabled() -> bool {
    static WARNED: std::sync::Once = std::sync::Once::new();
    env_flag_with_loud_legacy(ENV_SKIP_OUTER_NORM, ENV_SKIP_OUTER_NORM_LEGACY, &WARNED)
}

pub fn split_profile_requested() -> bool {
    env_flag(ENV_PROFILE_SPLIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `f` with the given env flags overridden on the current thread via the
    /// thread-local override (NOT `std::env::set_var`, which races concurrent
    /// `getenv` → SIGSEGV). Cleared on drop, so no cross-test leakage and no
    /// serialization needed.
    fn with_env_vars<T>(vars: &[(&'static str, Option<&str>)], f: impl FnOnce() -> T) -> T {
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                clear_fast_path_overrides();
            }
        }
        let _clear = Clear;
        for (name, value) in vars {
            set_env_override(name, *value);
        }
        f()
    }

    fn with_env<T>(name: &'static str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        with_env_vars(&[(name, value)], f)
    }

    #[test]
    fn opt_value_parsers_recognise_the_vocabulary() {
        for v in ["0", "false", "off", "no"] {
            assert!(is_opt_out_value(Some(v)));
            assert!(!is_opt_in_value(Some(v)));
        }
        for v in ["1", "true", "on", "yes"] {
            assert!(is_opt_in_value(Some(v)));
            assert!(!is_opt_out_value(Some(v)));
        }
        assert!(!is_opt_out_value(None));
        assert!(!is_opt_in_value(None));
        assert!(!is_opt_out_value(Some("maybe")));
    }

    #[test]
    fn env_flag_and_value_helpers_read_presence_and_content() {
        with_env(ENV_GPU_TIMING, Some("1"), || {
            assert!(env_flag(ENV_GPU_TIMING));
            assert_eq!(env_value(ENV_GPU_TIMING).as_deref(), Some("1"));
            assert_eq!(env_nonempty_value(ENV_GPU_TIMING).as_deref(), Some("1"));
        });

        with_env(ENV_GPU_TIMING, Some(""), || {
            assert!(env_flag(ENV_GPU_TIMING));
            assert_eq!(env_value(ENV_GPU_TIMING).as_deref(), Some(""));
            assert!(env_nonempty_value(ENV_GPU_TIMING).is_none());
        });

        with_env(ENV_GPU_TIMING, None, || {
            assert!(!env_flag(ENV_GPU_TIMING));
            assert!(env_value(ENV_GPU_TIMING).is_none());
        });
    }

    #[test]
    fn env_numeric_and_boolean_helpers_parse_expected_forms() {
        with_env(ENV_STAGE_DUMP_LAYER, Some("7"), || {
            assert_eq!(env_usize(ENV_STAGE_DUMP_LAYER), Some(7));
        });
        with_env(ENV_STAGE_DUMP_LAYER, Some("not-a-number"), || {
            assert_eq!(env_usize(ENV_STAGE_DUMP_LAYER), None);
        });

        for value in ["1", "true", "on", "yes"] {
            with_env(ENV_FUSED_ATTN, Some(value), || {
                assert!(env_opt_in(ENV_FUSED_ATTN));
                assert!(!env_opt_out(ENV_FUSED_ATTN));
            });
        }

        for value in ["0", "false", "off", "no"] {
            with_env(ENV_FUSED_ATTN, Some(value), || {
                assert!(!env_opt_in(ENV_FUSED_ATTN));
                assert!(env_opt_out(ENV_FUSED_ATTN));
            });
        }

        with_env(ENV_FUSED_DOWN, None, || {
            assert!(env_not_zero_or_default(ENV_FUSED_DOWN, true));
            assert!(!env_not_zero_or_default(ENV_FUSED_DOWN, false));
        });
        with_env(ENV_FUSED_DOWN, Some("0"), || {
            assert!(!env_not_zero_or_default(ENV_FUSED_DOWN, true));
        });
        with_env(ENV_FUSED_DOWN, Some("1"), || {
            assert!(env_not_zero_or_default(ENV_FUSED_DOWN, false));
        });
    }

    #[test]
    fn namespaced_toggle_helpers_read_their_flag() {
        with_env(ENV_SKIP_MOE, Some("1"), || assert!(skip_moe_enabled()));
    }

    /// The historical unprefixed names still work as loud deprecated
    /// aliases, and the canonical prefixed name is authoritative
    /// (dec-readiness review §3e — the DEC-0 ceiling-arm flag must mean
    /// the same thing on the local and grid paths).
    #[test]
    fn unprefixed_legacy_aliases_still_enable_their_flags() {
        with_env(ENV_SKIP_MOE_LEGACY, Some("1"), || {
            assert!(skip_moe_enabled(), "legacy SKIP_MOE alias must work");
        });
        with_env(ENV_DECODE_DEBUG_LEGACY, Some("1"), || {
            assert!(
                decode_debug_enabled(),
                "legacy DECODE_DEBUG alias must work"
            );
        });
        with_env(ENV_SKIP_OUTER_NORM_LEGACY, Some("1"), || {
            assert!(
                skip_outer_norm_enabled(),
                "legacy SKIP_OUTER_NORM alias must work"
            );
        });
        // Unset everywhere → off.
        assert!(!skip_moe_enabled());
        assert!(!decode_debug_enabled());
        assert!(!skip_outer_norm_enabled());
        with_env(ENV_MOE_DEBUG, Some("1"), || assert!(moe_debug_enabled()));
        with_env(ENV_PROFILE_SPLIT, Some("1"), || {
            assert!(split_profile_requested())
        });
    }

    #[test]
    fn env_flag_any_and_debug_helpers_cover_absent_and_present_cases() {
        with_env_vars(
            &[(ENV_SKIP_OUTER_NORM, None), (ENV_MOE_DEBUG, None)],
            || {
                assert!(!env_flag(ENV_SKIP_OUTER_NORM));
                assert!(!env_flag_any(&[ENV_SKIP_OUTER_NORM, ENV_MOE_DEBUG]));
                assert!(!moe_debug_enabled());
            },
        );

        with_env_vars(
            &[(ENV_SKIP_OUTER_NORM, Some("1")), (ENV_MOE_DEBUG, Some("1"))],
            || {
                assert!(env_flag(ENV_SKIP_OUTER_NORM));
                assert!(env_flag_any(&[ENV_SKIP_OUTER_NORM, ENV_MOE_DEBUG]));
                assert!(moe_debug_enabled());
            },
        );
    }
    /// The startup flag-state line names every fast-path stage, and the
    /// number-corrupting bypasses only appear (upper-case) when set.
    #[test]
    fn decode_options_summary_names_every_stage_and_flags_bypasses() {
        let s = decode_options_summary();
        for key in [
            "q4k_direct_attn=",
            "q4k_attn_int8=",
            "q4k_lm_head=",
            "q4k_direct_ffn=",
            "q4k_asm=",
            "spin_pool=",
            "moe_q4k_direct=",
        ] {
            assert!(s.contains(key), "{s}");
        }
        assert!(!s.contains("SKIP_MOE"), "bypass shown while unset: {s}");
        with_env(ENV_SKIP_MOE, Some("1"), || {
            assert!(decode_options_summary().contains("SKIP_MOE=ON"));
        });
        // MoE kernel-path toggle: default (unset) is the direct Q4K path;
        // LARQL_DISABLE_Q4K_DIRECT flips the summary to off.
        with_env(ENV_DISABLE_Q4K_DIRECT, None, || {
            assert!(
                decode_options_summary().contains("moe_q4k_direct=on"),
                "{}",
                decode_options_summary()
            );
        });
        with_env(ENV_DISABLE_Q4K_DIRECT, Some("1"), || {
            assert!(
                decode_options_summary().contains("moe_q4k_direct=off"),
                "{}",
                decode_options_summary()
            );
        });
    }
}
