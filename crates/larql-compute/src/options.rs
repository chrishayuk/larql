//! Runtime options and environment-variable names for compute backends.
//!
//! Keep process-global debug and experiment toggles here instead of spelling
//! string literals through hot paths. Most callers should eventually pass an
//! explicit options struct; this module is the compatibility bridge while those
//! APIs are split out.
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
/// Namespaced MoE bypass toggle.
pub const ENV_SKIP_MOE: &str = "LARQL_SKIP_MOE";
/// Legacy MoE bypass toggle. Prefer [`ENV_SKIP_MOE`] in new scripts.
pub const ENV_SKIP_MOE_LEGACY: &str = "SKIP_MOE";
/// Namespaced MoE route/debug output toggle.
pub const ENV_MOE_DEBUG: &str = "LARQL_MOE_DEBUG";
/// Legacy MoE route/debug output toggle. Prefer [`ENV_MOE_DEBUG`].
pub const ENV_MOE_DEBUG_LEGACY: &str = "MOE_DEBUG";
/// Enable Metal MoE dispatch timing.
pub const ENV_METAL_MOE_TIMING: &str = "LARQL_MOE_TIMING";
/// Select the 8-simdgroup Q4_K matvec kernel; set to a false value to opt out.
pub const ENV_Q4K_MATVEC_8SG: &str = "LARQL_Q4K_MATVEC_8SG";
/// Opt in to the 8-simdgroup Q6_K matvec kernel.
pub const ENV_Q6K_8SG: &str = "LARQL_Q6K_8SG";
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
pub const ENV_DECODE_DEBUG: &str = "DECODE_DEBUG";
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
/// Request paired commit/wait decode stage profiling.
pub const ENV_PROFILE_SPLIT: &str = "LARQL_PROFILE_SPLIT";
/// Legacy alias for [`ENV_PROFILE_SPLIT`].
pub const ENV_DECODE_STAGE_TIMING: &str = "LARQL_DECODE_STAGE_TIMING";
/// Debug-only outer norm bypass in Metal MoE combine.
pub const ENV_SKIP_OUTER_NORM: &str = "SKIP_OUTER_NORM";

// Helpers below are `pub` (not `pub(crate)`) because sibling backend
// crates (`larql-compute-metal`, future `larql-compute-vulkan`, …)
// share the same env-toggle vocabulary defined above.  Keeping the
// parsers private would force every backend to duplicate the
// `env::var_os`/`parse::<usize>` boilerplate and risk drift in how
// "set" / "true" / "1" are interpreted across backends.

pub fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

pub fn env_flag_any(names: &[&str]) -> bool {
    names.iter().any(|name| env_flag(name))
}

pub fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

pub fn env_value(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

pub fn env_nonempty_value(name: &str) -> Option<String> {
    env_value(name).filter(|value| !value.is_empty())
}

pub fn env_opt_in(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("on") | Ok("yes")
    )
}

pub fn env_opt_out(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("0") | Ok("false") | Ok("off") | Ok("no")
    )
}

pub fn env_not_zero_or_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| value != "0")
        .unwrap_or(default)
}

pub(crate) fn moe_debug_enabled() -> bool {
    env_flag_any(&[ENV_MOE_DEBUG, ENV_MOE_DEBUG_LEGACY])
}

pub(crate) fn skip_moe_enabled() -> bool {
    env_flag_any(&[ENV_SKIP_MOE, ENV_SKIP_MOE_LEGACY])
}

pub fn split_profile_requested() -> bool {
    env_flag_any(&[ENV_PROFILE_SPLIT, ENV_DECODE_STAGE_TIMING])
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env_vars<T>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().expect("env test mutex poisoned");
        let previous: Vec<_> = vars
            .iter()
            .map(|(name, _)| (*name, std::env::var_os(name)))
            .collect();
        for (name, value) in vars {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        let result = f();
        for (name, value) in previous {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
        result
    }

    fn with_env<T>(name: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        with_env_vars(&[(name, value)], f)
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
    fn legacy_alias_helpers_still_work() {
        with_env_vars(
            &[(ENV_SKIP_MOE, None), (ENV_SKIP_MOE_LEGACY, Some("1"))],
            || {
                assert!(skip_moe_enabled());
            },
        );
        with_env_vars(
            &[(ENV_MOE_DEBUG, Some("1")), (ENV_MOE_DEBUG_LEGACY, None)],
            || {
                assert!(moe_debug_enabled());
            },
        );
        with_env_vars(
            &[
                (ENV_PROFILE_SPLIT, None),
                (ENV_DECODE_STAGE_TIMING, Some("1")),
            ],
            || {
                assert!(split_profile_requested());
            },
        );
    }

    #[test]
    fn env_flag_any_and_debug_helpers_cover_absent_and_present_cases() {
        with_env_vars(
            &[
                (ENV_SKIP_OUTER_NORM, None),
                (ENV_MOE_DEBUG, None),
                (ENV_MOE_DEBUG_LEGACY, None),
            ],
            || {
                assert!(!env_flag(ENV_SKIP_OUTER_NORM));
                assert!(!env_flag_any(&[ENV_SKIP_OUTER_NORM, ENV_MOE_DEBUG]));
                assert!(!moe_debug_enabled());
            },
        );

        with_env_vars(
            &[
                (ENV_SKIP_OUTER_NORM, Some("1")),
                (ENV_MOE_DEBUG, Some("1")),
                (ENV_MOE_DEBUG_LEGACY, None),
            ],
            || {
                assert!(env_flag(ENV_SKIP_OUTER_NORM));
                assert!(env_flag_any(&[ENV_SKIP_OUTER_NORM, ENV_MOE_DEBUG_LEGACY]));
                assert!(moe_debug_enabled());
            },
        );
    }
}
