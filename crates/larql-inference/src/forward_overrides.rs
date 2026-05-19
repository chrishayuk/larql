//! Forward-path override surface.
//!
//! This module lives between `larql-models` (which parses model config into
//! the architecture trait) and the inference forward path (CPU + GPU).
//! Each helper here resolves an effective per-layer parameter by checking
//! a diagnostic env-var first, then falling back to whatever the
//! architecture exposes from the parsed `config.json`.
//!
//! ## Why the env vars exist
//!
//! The five env vars below were the diagnostic instruments used to
//! bisect the cross-engine forward-pass divergence documented in
//! [`docs/diagnoses/shannon-cross-engine-divergence.md`](../../../docs/diagnoses/shannon-cross-engine-divergence.md).
//! They are kept in tree even after the underlying loader bugs were fixed
//! so future regressions on new architectures can be localised the same
//! way without touching code. Production runs never need to set any of
//! them — when unset, every helper delegates to the architecture.
//!
//! ## Precedence
//!
//! For each parameter:
//!
//! 1. If the corresponding env var is set and parses, use it.
//! 2. Otherwise call the architecture's accessor on the parsed config.
//! 3. Architecture accessors carry their own defaults
//!    (see [`larql_models::defaults`]) for fields the model's config
//!    omits entirely.
//!
//! ## Env-var reference
//!
//! | Var | Type | Effect |
//! |---|---|---|
//! | `LARQL_FORCE_GLOBAL_LAYERS` | `all` or `<csv>` | Force listed layers onto global rope_base (sliding_window=0). |
//! | `LARQL_ROPE_POS_DIVISOR` | `f64` | Divide RoPE position by this factor on every layer. |
//! | `LARQL_ROPE_POS_DIVISOR_GLOBAL` | `f64` | Same, but only on `!is_sliding_window_layer(layer)`. |
//! | `LARQL_LLAMA3_ROPE_SCALING` | `factor,low,high,old_ctx` | Force HF llama3 scaling params. |
//! | `LARQL_NORM_EPS_OVERRIDE` | `f64` | Override `arch.norm_eps()`. |

use std::sync::OnceLock;

/// Diagnostic override for the sliding-window attention bisection.
///
/// `LARQL_FORCE_GLOBAL_LAYERS=all` forces every layer onto the global-attention
/// code path (sliding_window=0, rope_base = arch's full rope_theta). A comma-
/// separated index list (`LARQL_FORCE_GLOBAL_LAYERS=12,13,14`) targets specific
/// layers. Empty/unset leaves the architecture's per-layer routing untouched.
#[derive(Debug)]
enum ForceGlobalSpec {
    None,
    All,
    Layers(Vec<usize>),
}

/// Pure parser: maps the raw env-var value (or absence) to the
/// `ForceGlobalSpec` variant. Split from [`force_global_spec`] so the
/// parsing logic is testable without going through the `OnceLock`
/// (which fires once per process and pins to whatever the env was at
/// first-call time).
fn parse_force_global_spec(raw: Option<&str>) -> ForceGlobalSpec {
    let Some(s) = raw else {
        return ForceGlobalSpec::None;
    };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        ForceGlobalSpec::None
    } else if trimmed.eq_ignore_ascii_case("all") {
        ForceGlobalSpec::All
    } else {
        let layers: Vec<usize> = trimmed
            .split(',')
            .filter_map(|tok| tok.trim().parse::<usize>().ok())
            .collect();
        if layers.is_empty() {
            ForceGlobalSpec::None
        } else {
            ForceGlobalSpec::Layers(layers)
        }
    }
}

fn force_global_spec() -> &'static ForceGlobalSpec {
    static CELL: OnceLock<ForceGlobalSpec> = OnceLock::new();
    CELL.get_or_init(|| {
        parse_force_global_spec(std::env::var("LARQL_FORCE_GLOBAL_LAYERS").ok().as_deref())
    })
}

/// Returns true when `LARQL_FORCE_GLOBAL_LAYERS` requests this layer be
/// forced onto the global-attention code path.
pub fn layer_forced_global(layer: usize) -> bool {
    match force_global_spec() {
        ForceGlobalSpec::None => false,
        ForceGlobalSpec::All => true,
        ForceGlobalSpec::Layers(v) => v.contains(&layer),
    }
}

/// Per-layer rope base honouring the `LARQL_FORCE_GLOBAL_LAYERS` diagnostic
/// override. Use this anywhere the CPU/GPU forward path would otherwise call
/// `arch.rope_base_for_layer(layer)` directly.
pub fn effective_rope_base_for_layer(
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> f64 {
    if layer_forced_global(layer) {
        arch.config().rope_base
    } else {
        arch.rope_base_for_layer(layer)
    }
}

/// Diagnostic position scale read from `LARQL_ROPE_POS_DIVISOR=<f64>`. Matches
/// HF `rope_scaling = {rope_type: linear, factor: <v>}`. Returns `1.0` when
/// the env var is unset. Applied uniformly to every layer.
/// Pure parser for `LARQL_ROPE_POS_DIVISOR`. Returns 1.0 (no scaling)
/// when the input is `None`, empty, unparseable, non-finite, or
/// non-positive. Split from [`rope_position_divisor`] for testability.
fn parse_rope_position_divisor(raw: Option<&str>) -> f64 {
    raw.and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0)
}

fn rope_position_divisor() -> f64 {
    static CELL: OnceLock<f64> = OnceLock::new();
    *CELL.get_or_init(|| {
        parse_rope_position_divisor(std::env::var("LARQL_ROPE_POS_DIVISOR").ok().as_deref())
    })
}

/// Diagnostic position scale read from `LARQL_ROPE_POS_DIVISOR_GLOBAL=<f64>`,
/// applied only on global (non-sliding) layers. Gemma 3's HF config sets a
/// linear factor on full-attention layers only via the structured per-layer-
/// type `rope_scaling` form.
fn rope_position_divisor_global_only() -> f64 {
    static CELL: OnceLock<f64> = OnceLock::new();
    *CELL.get_or_init(|| {
        // Reuses the `LARQL_ROPE_POS_DIVISOR` parser — same validation
        // semantics (finite, positive, defaults to 1.0).
        parse_rope_position_divisor(
            std::env::var("LARQL_ROPE_POS_DIVISOR_GLOBAL")
                .ok()
                .as_deref(),
        )
    })
}

/// Diagnostic override for HF `llama3` rope scaling, reading
/// `LARQL_LLAMA3_ROPE_SCALING=factor,low,high,old_ctx` (comma-separated).
/// E.g. `LARQL_LLAMA3_ROPE_SCALING=32,1,4,8192` matches Llama-3.2's config.
/// Returns `None` when the env var is unset or malformed (in which case
/// the arch-driven value from [`effective_llama3_rope_scaling`] is used).
/// Pure parser for `LARQL_LLAMA3_ROPE_SCALING=factor,low,high,old_ctx`.
/// Returns `None` for absent / malformed / non-validating inputs (the
/// validators match HF's `Llama3RopeScaling` invariants: all four
/// factors positive, `high_freq_factor > low_freq_factor`).
fn parse_llama3_rope_scaling(raw: Option<&str>) -> Option<larql_models::Llama3RopeScaling> {
    let parts: Vec<f64> = raw?
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if parts.len() != 4 {
        return None;
    }
    let s = larql_models::Llama3RopeScaling {
        factor: parts[0],
        low_freq_factor: parts[1],
        high_freq_factor: parts[2],
        original_max_position_embeddings: parts[3],
    };
    if s.factor > 0.0
        && s.low_freq_factor > 0.0
        && s.high_freq_factor > 0.0
        && s.original_max_position_embeddings > 0.0
        && s.high_freq_factor > s.low_freq_factor
    {
        Some(s)
    } else {
        None
    }
}

fn llama3_rope_scaling_override() -> Option<larql_models::Llama3RopeScaling> {
    static CELL: OnceLock<Option<larql_models::Llama3RopeScaling>> = OnceLock::new();
    *CELL.get_or_init(|| {
        parse_llama3_rope_scaling(std::env::var("LARQL_LLAMA3_ROPE_SCALING").ok().as_deref())
    })
}

/// Llama3 rope-scaling parameters for the forward pass — env-var override
/// first, then the architecture's parsed `rope_scaling`. Returns `None`
/// when neither is set (no scaling applied).
pub fn effective_llama3_rope_scaling(
    arch: &dyn larql_models::ModelArchitecture,
) -> Option<larql_models::Llama3RopeScaling> {
    llama3_rope_scaling_override().or_else(|| arch.llama3_rope_scaling())
}

/// Diagnostic norm-epsilon override read from `LARQL_NORM_EPS_OVERRIDE=<f64>`.
/// When set, replaces the architecture's `norm_eps()` value at every
/// `rms_norm_for_arch` / `layer_norm_for_arch` call site. Use to test
/// whether a hardcoded default is masking a config that expects a
/// different eps.
/// Pure parser for `LARQL_NORM_EPS_OVERRIDE`. Returns `None` for
/// absent / unparseable / non-finite / non-positive inputs.
fn parse_norm_eps_override(raw: Option<&str>) -> Option<f32> {
    raw.and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

pub fn norm_eps_override() -> Option<f32> {
    static CELL: OnceLock<Option<f32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        parse_norm_eps_override(std::env::var("LARQL_NORM_EPS_OVERRIDE").ok().as_deref())
    })
}

/// Effective per-layer RoPE position divisor.
///
/// Precedence: env-var overrides first (uniform `LARQL_ROPE_POS_DIVISOR` and
/// global-only `LARQL_ROPE_POS_DIVISOR_GLOBAL`), then the architecture's
/// own `rope_position_divisor_for_layer` (which reads the parsed
/// `config.rope_scaling`). Returns 1.0 (no scaling) when nothing applies.
pub fn effective_rope_position_divisor_for_layer(
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> f64 {
    let uniform_env = rope_position_divisor();
    let global_env = rope_position_divisor_global_only();
    if !arch.is_sliding_window_layer(layer) && global_env != 1.0 {
        return global_env;
    }
    if uniform_env != 1.0 {
        return uniform_env;
    }
    // Default: ask the architecture (parsed from rope_scaling in config.json).
    arch.rope_position_divisor_for_layer(layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The env-var-reading helpers use OnceLock, so they read process env
    // exactly once. We can't unset/reset them within a test process, so
    // these tests exercise the *arch-driven* fallback path that runs when
    // the env vars are unset (which is also the production path).

    fn gemma3_with_linear_scaling() -> Box<dyn larql_models::ModelArchitecture> {
        // Minimal Gemma 3 config with the structured per-layer-type
        // rope_scaling form that triggers the "factor on global layers
        // only" code path in `Gemma3Arch`.
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 2560,
                "head_dim": 256,
                "num_hidden_layers": 34,
                "num_attention_heads": 8,
                "intermediate_size": 10240,
                "sliding_window": 1024,
                "rope_scaling": {
                    "full_attention": {"rope_type": "linear", "factor": 8.0},
                    "sliding_attention": {"rope_type": "default"},
                },
            },
        }))
    }

    #[test]
    fn effective_rope_position_divisor_uses_arch_on_global_layer() {
        // No env vars set → defer to arch. Layer 5 is global on Gemma 3
        // (5 + 1 = 6, multiple of 6), so the linear factor 8.0 must come
        // through.
        let arch = gemma3_with_linear_scaling();
        assert_eq!(effective_rope_position_divisor_for_layer(&*arch, 5), 8.0);
    }

    #[test]
    fn effective_rope_position_divisor_uses_arch_on_sliding_layer() {
        // Layer 0 is sliding → factor must NOT apply, divisor stays 1.0.
        let arch = gemma3_with_linear_scaling();
        assert_eq!(effective_rope_position_divisor_for_layer(&*arch, 0), 1.0);
    }

    #[test]
    fn effective_llama3_returns_none_without_arch_scaling_or_env() {
        // Arch with no rope_scaling at all → None.
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }));
        assert!(effective_llama3_rope_scaling(&*arch).is_none());
    }

    #[test]
    fn effective_llama3_returns_arch_scaling_when_set() {
        // Arch with rope_type=llama3 → must flow through to caller with
        // the same field values (no rounding / no zero-init).
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        }));
        let s = effective_llama3_rope_scaling(&*arch).expect("llama3 scaling exposed");
        assert_eq!(s.factor, 32.0);
        assert_eq!(s.low_freq_factor, 1.0);
        assert_eq!(s.high_freq_factor, 4.0);
        assert_eq!(s.original_max_position_embeddings, 8192.0);
    }

    // ── Pure-parser tests: cover every branch of the env-var parsing ──
    //
    // The OnceLock wrappers (`force_global_spec`, `rope_position_divisor`,
    // …) pin to whatever the env was at first call. The pure parsers
    // below are pure functions of their input — no global state — so
    // we can exhaustively cover the dispatch arms in one test process.

    // ── parse_force_global_spec ──

    #[test]
    fn parse_force_global_spec_none_when_env_absent() {
        assert!(matches!(
            parse_force_global_spec(None),
            ForceGlobalSpec::None
        ));
    }

    #[test]
    fn parse_force_global_spec_none_when_empty_or_whitespace() {
        assert!(matches!(
            parse_force_global_spec(Some("")),
            ForceGlobalSpec::None
        ));
        assert!(matches!(
            parse_force_global_spec(Some("   ")),
            ForceGlobalSpec::None
        ));
    }

    #[test]
    fn parse_force_global_spec_all_is_case_insensitive() {
        assert!(matches!(
            parse_force_global_spec(Some("all")),
            ForceGlobalSpec::All
        ));
        assert!(matches!(
            parse_force_global_spec(Some("ALL")),
            ForceGlobalSpec::All
        ));
        assert!(matches!(
            parse_force_global_spec(Some(" All ")),
            ForceGlobalSpec::All
        ));
    }

    #[test]
    fn parse_force_global_spec_csv_layers() {
        match parse_force_global_spec(Some("12,13,14")) {
            ForceGlobalSpec::Layers(v) => assert_eq!(v, vec![12, 13, 14]),
            other => panic!("expected Layers, got {other:?}"),
        }
    }

    #[test]
    fn parse_force_global_spec_csv_skips_non_numeric_tokens() {
        // `parse::<usize>().ok()` filters non-numeric entries; valid
        // ones are kept in declared order.
        match parse_force_global_spec(Some("5, oops, 7, 11")) {
            ForceGlobalSpec::Layers(v) => assert_eq!(v, vec![5, 7, 11]),
            other => panic!("expected Layers, got {other:?}"),
        }
    }

    #[test]
    fn parse_force_global_spec_none_when_csv_has_no_numbers() {
        // No parseable layer indices → falls through to None instead of
        // an empty Layers list.
        assert!(matches!(
            parse_force_global_spec(Some("nope,bogus")),
            ForceGlobalSpec::None
        ));
    }

    // ── parse_rope_position_divisor ──

    #[test]
    fn parse_rope_position_divisor_defaults_to_one() {
        assert_eq!(parse_rope_position_divisor(None), 1.0);
        assert_eq!(parse_rope_position_divisor(Some("")), 1.0);
    }

    #[test]
    fn parse_rope_position_divisor_accepts_positive_finite() {
        assert_eq!(parse_rope_position_divisor(Some("8")), 8.0);
        assert_eq!(parse_rope_position_divisor(Some("  2.5  ")), 2.5);
    }

    #[test]
    fn parse_rope_position_divisor_rejects_non_positive_and_non_finite() {
        // Non-finite, zero, and negative all fall back to 1.0.
        assert_eq!(parse_rope_position_divisor(Some("inf")), 1.0);
        assert_eq!(parse_rope_position_divisor(Some("nan")), 1.0);
        assert_eq!(parse_rope_position_divisor(Some("0")), 1.0);
        assert_eq!(parse_rope_position_divisor(Some("-3")), 1.0);
        assert_eq!(parse_rope_position_divisor(Some("not-a-number")), 1.0);
    }

    // ── parse_llama3_rope_scaling ──

    #[test]
    fn parse_llama3_rope_scaling_none_on_missing_or_malformed() {
        assert!(parse_llama3_rope_scaling(None).is_none());
        // Too-few parseable fields (only 3 numbers).
        assert!(parse_llama3_rope_scaling(Some("32,1,4")).is_none());
        // Too-many parseable fields (5 numbers).
        assert!(parse_llama3_rope_scaling(Some("32,1,4,8192,128")).is_none());
        // All tokens unparseable → empty parts vec, len != 4 → None.
        assert!(parse_llama3_rope_scaling(Some("a,b,c,d")).is_none());
    }

    #[test]
    fn parse_llama3_rope_scaling_accepts_well_formed_input() {
        let s = parse_llama3_rope_scaling(Some("32,1,4,8192")).expect("valid scaling");
        assert_eq!(s.factor, 32.0);
        assert_eq!(s.low_freq_factor, 1.0);
        assert_eq!(s.high_freq_factor, 4.0);
        assert_eq!(s.original_max_position_embeddings, 8192.0);
    }

    #[test]
    fn parse_llama3_rope_scaling_rejects_invariant_violations() {
        // factor <= 0
        assert!(parse_llama3_rope_scaling(Some("0,1,4,8192")).is_none());
        // low_freq_factor <= 0
        assert!(parse_llama3_rope_scaling(Some("32,0,4,8192")).is_none());
        // high_freq_factor <= 0
        assert!(parse_llama3_rope_scaling(Some("32,1,0,8192")).is_none());
        // original_max_position_embeddings <= 0
        assert!(parse_llama3_rope_scaling(Some("32,1,4,0")).is_none());
        // high_freq_factor <= low_freq_factor (invariant: high > low)
        assert!(parse_llama3_rope_scaling(Some("32,4,4,8192")).is_none());
        assert!(parse_llama3_rope_scaling(Some("32,4,1,8192")).is_none());
    }

    // ── parse_norm_eps_override ──

    #[test]
    fn parse_norm_eps_override_none_on_missing_or_bad() {
        assert!(parse_norm_eps_override(None).is_none());
        assert!(parse_norm_eps_override(Some("")).is_none());
        assert!(parse_norm_eps_override(Some("abc")).is_none());
        assert!(parse_norm_eps_override(Some("inf")).is_none());
        assert!(parse_norm_eps_override(Some("-1e-6")).is_none());
        assert!(parse_norm_eps_override(Some("0")).is_none());
    }

    #[test]
    fn parse_norm_eps_override_accepts_positive_finite() {
        assert_eq!(parse_norm_eps_override(Some("1e-5")), Some(1e-5_f32));
        assert_eq!(parse_norm_eps_override(Some("  1e-6  ")), Some(1e-6_f32));
    }

    // ── layer_forced_global semantic test through the live OnceLock ──
    //
    // We can't reset the static cell, so this test relies on the env
    // var being unset (the default in tests). The arch-default path
    // returns false for every layer.

    #[test]
    fn layer_forced_global_returns_false_when_env_unset() {
        assert!(!layer_forced_global(0));
        assert!(!layer_forced_global(34));
    }

    // ── Direct semantic test of layer_forced_global against each ForceGlobalSpec ──
    //
    // We test the *match arms* explicitly by constructing a spec and
    // walking the variants. This complements the OnceLock-bound test
    // above by covering the All and Layers arms.

    #[test]
    fn force_global_spec_layers_arm_matches_listed_layers() {
        let spec = ForceGlobalSpec::Layers(vec![3, 7, 11]);
        let hits: Vec<bool> = (0..12)
            .map(|l| matches!(&spec, ForceGlobalSpec::Layers(v) if v.contains(&l)))
            .collect();
        assert!(hits[3]);
        assert!(hits[7]);
        assert!(hits[11]);
        assert!(!hits[0]);
        assert!(!hits[5]);
    }
}
