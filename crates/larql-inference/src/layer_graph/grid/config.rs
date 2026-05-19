use crate::ffn::moe_remote::runtime::{
    ENV_MOE_NO_SPLIT, ENV_MOE_TIMING, ENV_MOE_TOP_K, ENV_SKIP_MOE,
};
use crate::layer_graph::generate::policy::TokenSelectionPolicy;

#[derive(Clone, Debug)]
pub(super) struct GridRuntimeConfig {
    pub moe_top_k_override: Option<usize>,
    pub skip_moe: bool,
    pub timing_enabled: bool,
    pub split_disabled: bool,
    pub token_policy: TokenSelectionPolicy,
}

impl GridRuntimeConfig {
    pub fn from_env() -> Self {
        Self {
            moe_top_k_override: std::env::var(ENV_MOE_TOP_K)
                .ok()
                .and_then(|s| s.parse::<usize>().ok()),
            skip_moe: std::env::var(ENV_SKIP_MOE).is_ok(),
            timing_enabled: std::env::var(ENV_MOE_TIMING).is_ok(),
            split_disabled: std::env::var(ENV_MOE_NO_SPLIT).is_ok(),
            token_policy: TokenSelectionPolicy::from_env(),
        }
    }

    pub fn moe_top_k(&self, arch_top_k: usize) -> usize {
        self.moe_top_k_override
            .map(|k| k.clamp(1, arch_top_k))
            .unwrap_or(arch_top_k)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn from_env_returns_default_when_no_vars_set() {
        // Ensure none of the env vars are set so we exercise the default
        // arms of every `.is_ok()` / `.parse().ok()` chain.
        let prev_topk = std::env::var(ENV_MOE_TOP_K).ok();
        let prev_skip = std::env::var(ENV_SKIP_MOE).ok();
        let prev_timing = std::env::var(ENV_MOE_TIMING).ok();
        let prev_split = std::env::var(ENV_MOE_NO_SPLIT).ok();
        std::env::remove_var(ENV_MOE_TOP_K);
        std::env::remove_var(ENV_SKIP_MOE);
        std::env::remove_var(ENV_MOE_TIMING);
        std::env::remove_var(ENV_MOE_NO_SPLIT);

        let cfg = GridRuntimeConfig::from_env();
        assert!(cfg.moe_top_k_override.is_none());
        assert!(!cfg.skip_moe);

        // Restore.
        if let Some(v) = prev_topk {
            std::env::set_var(ENV_MOE_TOP_K, v);
        }
        if let Some(v) = prev_skip {
            std::env::set_var(ENV_SKIP_MOE, v);
        }
        if let Some(v) = prev_timing {
            std::env::set_var(ENV_MOE_TIMING, v);
        }
        if let Some(v) = prev_split {
            std::env::set_var(ENV_MOE_NO_SPLIT, v);
        }
    }

    #[test]
    fn moe_top_k_falls_back_to_arch_when_no_override() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: None,
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        assert_eq!(cfg.moe_top_k(8), 8);
    }

    #[test]
    fn moe_top_k_clamps_override_to_arch_max() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: Some(99),
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        // Override 99 > arch 8 → clamped to 8.
        assert_eq!(cfg.moe_top_k(8), 8);
    }

    #[test]
    fn moe_top_k_clamps_override_to_min_one() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: Some(0),
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        // Override 0 < 1 → clamped to 1.
        assert_eq!(cfg.moe_top_k(8), 1);
    }

    #[test]
    fn moe_top_k_uses_override_when_in_range() {
        let cfg = GridRuntimeConfig {
            moe_top_k_override: Some(4),
            skip_moe: false,
            timing_enabled: false,
            split_disabled: false,
            token_policy: TokenSelectionPolicy::from_env(),
        };
        assert_eq!(cfg.moe_top_k(8), 4);
    }
}
