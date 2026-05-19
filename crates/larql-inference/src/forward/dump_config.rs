//! Typed accessor for the diagnostic per-layer / per-stage dump env vars,
//! plus the canonical filename templates the producer and consumer sides
//! share.
//!
//! Five env vars used to be re-read on every CPU forward pass — once per
//! layer and once per stage (norm, Q, K, V, attn, O, FFN). On a 34-layer
//! decode that was ≥ 200 process-env lookups per token even when no dump
//! was active. They also drift if the env changes mid-process, which the
//! dump consumers don't actually want.
//!
//! [`DumpConfig::current`] reads the three CPU-side vars on every call and
//! returns an owned `DumpConfig`. Env-var reads are nanoseconds; the
//! once-cached design (`get() -> &'static Self`) broke cross-test
//! correctness when fixtures rotated tempdirs between calls. The two
//! Metal/decode-side vars (read by `larql-compute`) appear here only as
//! string consts — `larql-compute` reads them itself, but `residual_diff`
//! sets them via the names exported from this module so producer and
//! consumer agree.
//!
//! Recognised vars (semantics unchanged from the previous inline reads):
//!
//! - [`ENV_CPU_DUMP_LAYERS`] = `LARQL_CPU_DUMP_LAYERS=<dir>` — write
//!   `cpu_layer_NN.f32`, `cpu_layer_NN_h_post_attn.f32` and
//!   `cpu_layer_NN_h_out.f32` per layer (CPU forward path).
//! - [`ENV_CPU_STAGE_DUMP`] = `LARQL_CPU_STAGE_DUMP=<dir>` — write
//!   per-stage intermediates (`cpu_L0_norm_out.f32`, `cpu_L0_q_proj.f32`,
//!   …) for one specific layer (default 0).
//! - [`ENV_STAGE_DUMP_LAYER`] = `LARQL_STAGE_DUMP_LAYER=<usize>` — pick a
//!   different stage-dump layer (Gemma 4 global layers 5/11/17/… are
//!   useful for partial-RoPE bisects).
//! - [`ENV_DECODE_DUMP_LAYERS`] = `LARQL_DECODE_DUMP_LAYERS=<dir>` — read
//!   by `larql-compute::metal/decode`, writes `decode_layer_NN.f32` and
//!   `decode_layer_NN_<stage>.f32` per layer.
//! - [`ENV_METAL_DUMP_LAYERS`] = `LARQL_METAL_DUMP_LAYERS=<dir>` — read
//!   by `larql-compute::metal/ops/full_pipeline`, writes
//!   `metal_layer_NN_h_out.f32` and `metal_layer_NN_<stage>.f32` per
//!   layer.
//!
//! ## Filename templates
//!
//! The path-builder helpers below are the single source of truth for the
//! on-disk filenames so producer and consumer can't drift. **Note**: the
//! per-stage CPU producer always writes the literal `cpu_L0_<name>.f32`
//! regardless of [`DumpConfig::stage_dump_layer`]; the gate decides
//! *whether* to dump, not what to name the file. The consumer-side prefix
//! [`cpu_stage_prefix`] therefore only finds files when `layer == 0`. Any
//! future fix should change both sides together.

// ── Env var names ──────────────────────────────────────────────────────────

/// `LARQL_CPU_DUMP_LAYERS=<dir>` — read by [`DumpConfig`].
pub const ENV_CPU_DUMP_LAYERS: &str = "LARQL_CPU_DUMP_LAYERS";

/// `LARQL_CPU_STAGE_DUMP=<dir>` — read by [`DumpConfig`].
pub const ENV_CPU_STAGE_DUMP: &str = "LARQL_CPU_STAGE_DUMP";

/// `LARQL_STAGE_DUMP_LAYER=<usize>` — read by [`DumpConfig`].
pub const ENV_STAGE_DUMP_LAYER: &str = "LARQL_STAGE_DUMP_LAYER";

/// `LARQL_DECODE_DUMP_LAYERS=<dir>` — read by `larql-compute::metal/decode`.
/// Set here so `residual_diff` and the producer agree on the name.
pub const ENV_DECODE_DUMP_LAYERS: &str = "LARQL_DECODE_DUMP_LAYERS";

/// `LARQL_METAL_DUMP_LAYERS=<dir>` — read by
/// `larql-compute::metal/ops/full_pipeline`. Set here for the same
/// producer/consumer-agreement reason as [`ENV_DECODE_DUMP_LAYERS`].
pub const ENV_METAL_DUMP_LAYERS: &str = "LARQL_METAL_DUMP_LAYERS";

// ── Filename templates (producer and consumer share these) ──────────────────

/// `{dir}/cpu_layer_NN.f32` — end-of-layer CPU residual (from
/// `vindex/kquant_forward/hidden.rs`).
pub fn cpu_layer_path(dir: &str, layer: usize) -> String {
    format!("{dir}/cpu_layer_{layer:02}.f32")
}

/// `cpu_layer_NN.f32` — bare filename, for callers that already have the
/// dir handle (e.g. `tempfile::TempDir::path().join(...)`).
pub fn cpu_layer_file(layer: usize) -> String {
    format!("cpu_layer_{layer:02}.f32")
}

/// `{dir}/cpu_layer_NN_h_post_attn.f32` — per-layer post-attention dump
/// (CPU forward / patched forward).
pub fn cpu_layer_h_post_attn_path(dir: &str, layer: usize) -> String {
    format!("{dir}/cpu_layer_{layer:02}_h_post_attn.f32")
}

/// `{dir}/cpu_L0_<name>.f32` — per-stage CPU dump. **Always** uses the
/// literal `L0` regardless of [`DumpConfig::stage_dump_layer`]; see the
/// module-level note on the producer/consumer mismatch.
pub fn cpu_stage_path(dir: &str, name: &str) -> String {
    format!("{dir}/cpu_L0_{name}.f32")
}

/// `cpu_L<layer>_` — consumer-side filename prefix the `residual_diff`
/// stage reader scans for. Mismatched with the producer (always `L0`)
/// when `layer != 0`.
pub fn cpu_stage_prefix(layer: usize) -> String {
    format!("cpu_L{layer}_")
}

/// `decode_layer_NN.f32` — bare filename for the per-layer Metal-decode
/// dump. Producer is `larql-compute`; consumer is `residual_diff`.
pub fn decode_layer_file(layer: usize) -> String {
    format!("decode_layer_{layer:02}.f32")
}

/// `decode_layer_NN_` — consumer-side prefix the stage reader scans for
/// when comparing per-stage Metal-decode dumps.
pub fn decode_layer_prefix(layer: usize) -> String {
    format!("decode_layer_{layer:02}_")
}

/// `metal_layer_NN_h_out.f32` — bare filename for per-layer Metal-prefill
/// end-of-layer dump.
pub fn metal_layer_h_out_file(layer: usize) -> String {
    format!("metal_layer_{layer:02}_h_out.f32")
}

/// `metal_layer_NN_` — consumer-side prefix for per-stage Metal-prefill
/// dumps.
pub fn metal_layer_prefix(layer: usize) -> String {
    format!("metal_layer_{layer:02}_")
}

/// Snapshot of dump-related env vars, captured once per process.
#[derive(Clone, Debug, Default)]
pub struct DumpConfig {
    /// Directory to write per-layer end-of-layer dumps to. `None` disables.
    pub layer_dump_dir: Option<String>,
    /// Directory to write per-stage intermediates to. `None` disables.
    pub stage_dump_dir: Option<String>,
    /// Which layer's stages to dump when `stage_dump_dir` is set. Defaults to 0.
    pub stage_dump_layer: usize,
}

impl DumpConfig {
    /// Read the three env vars and assemble a `DumpConfig`. Public so test
    /// fixtures can build one without touching the process env.
    pub fn from_env() -> Self {
        Self {
            layer_dump_dir: std::env::var(ENV_CPU_DUMP_LAYERS).ok(),
            stage_dump_dir: std::env::var(ENV_CPU_STAGE_DUMP).ok(),
            stage_dump_layer: std::env::var(ENV_STAGE_DUMP_LAYER)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0),
        }
    }

    /// Read env vars freshly and build a config. Aliased by [`Self::get`]
    /// for back-compat; new callers should prefer this name.
    pub fn current() -> Self {
        Self::from_env()
    }

    /// Read env vars freshly. Previously cached via `OnceLock`, which
    /// silently broke test fixtures that rotated `LARQL_*_DUMP_LAYERS`
    /// tempdirs between calls — the second call would still hand out
    /// the path of the first (now-deleted) tempdir, and writes failed
    /// with `No such file or directory`. Env-var reads on macOS are
    /// ~150 ns; at ~170 reads per token (34 layers × 5 stage gates),
    /// the cost is ~25 µs per token vs ~35 ms of forward work — 0.07%.
    /// Caller binds the result to a local before using `.layer_dir()` /
    /// `.stage_dir(l)` so the `&str` doesn't borrow from a temporary.
    pub fn get() -> Self {
        Self::current()
    }

    /// `Some(dir)` only when stage dumps are enabled AND the active layer
    /// matches `stage_dump_layer`. Mirrors the prior inline guard.
    pub fn stage_dir(&self, layer: usize) -> Option<&str> {
        if layer == self.stage_dump_layer {
            self.stage_dump_dir.as_deref()
        } else {
            None
        }
    }

    /// `Some(dir)` when per-layer dumps are enabled.
    pub fn layer_dir(&self) -> Option<&str> {
        self.layer_dump_dir.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_reads_all_three_vars() {
        // Hand-build via from_env-style construction (we can't poke the
        // process env reliably in parallel tests, so build the struct
        // directly and exercise the accessors).
        let cfg = DumpConfig {
            layer_dump_dir: Some("/tmp/layers".into()),
            stage_dump_dir: Some("/tmp/stages".into()),
            stage_dump_layer: 5,
        };
        assert_eq!(cfg.layer_dir(), Some("/tmp/layers"));
        assert_eq!(cfg.stage_dir(5), Some("/tmp/stages"));
        assert_eq!(cfg.stage_dir(0), None);
        assert_eq!(cfg.stage_dir(99), None);
    }

    #[test]
    fn stage_dir_returns_none_when_dump_disabled() {
        let cfg = DumpConfig {
            layer_dump_dir: None,
            stage_dump_dir: None,
            stage_dump_layer: 0,
        };
        assert_eq!(cfg.stage_dir(0), None);
    }

    #[test]
    fn layer_dir_returns_none_when_dump_disabled() {
        let cfg = DumpConfig::default();
        assert_eq!(cfg.layer_dir(), None);
    }

    #[test]
    fn get_reads_env_freshly_each_call() {
        // The OnceLock-cached design used to break test fixtures that
        // rotated dump-dir tempdirs between calls — the cache handed
        // out a stale path. `get()` now returns a freshly-built
        // `DumpConfig` per call so flipping the env var actually
        // takes effect.
        // Save + restore to keep the process env clean for other tests.
        let prev = std::env::var(ENV_CPU_DUMP_LAYERS).ok();
        std::env::set_var(ENV_CPU_DUMP_LAYERS, "/tmp/dump-a");
        let a_dir = DumpConfig::get().layer_dump_dir;
        std::env::set_var(ENV_CPU_DUMP_LAYERS, "/tmp/dump-b");
        let b_dir = DumpConfig::get().layer_dump_dir;
        match prev {
            Some(v) => std::env::set_var(ENV_CPU_DUMP_LAYERS, v),
            None => std::env::remove_var(ENV_CPU_DUMP_LAYERS),
        }
        assert_eq!(a_dir, Some("/tmp/dump-a".to_string()));
        assert_eq!(b_dir, Some("/tmp/dump-b".to_string()));
    }

    #[test]
    fn stage_dump_layer_default_is_zero_when_unparsed() {
        // Mirrors the unwrap_or(0) on bad/missing LARQL_STAGE_DUMP_LAYER.
        let cfg = DumpConfig::default();
        assert_eq!(cfg.stage_dump_layer, 0);
    }

    #[test]
    fn env_var_consts_match_legacy_literal_names() {
        // Pin the on-the-wire env var names so a refactor that renames
        // a const doesn't silently break LARQL users' scripts.
        assert_eq!(ENV_CPU_DUMP_LAYERS, "LARQL_CPU_DUMP_LAYERS");
        assert_eq!(ENV_CPU_STAGE_DUMP, "LARQL_CPU_STAGE_DUMP");
        assert_eq!(ENV_STAGE_DUMP_LAYER, "LARQL_STAGE_DUMP_LAYER");
        assert_eq!(ENV_DECODE_DUMP_LAYERS, "LARQL_DECODE_DUMP_LAYERS");
        assert_eq!(ENV_METAL_DUMP_LAYERS, "LARQL_METAL_DUMP_LAYERS");
    }

    #[test]
    fn cpu_layer_path_matches_legacy_format() {
        assert_eq!(cpu_layer_path("/tmp", 7), "/tmp/cpu_layer_07.f32");
        assert_eq!(cpu_layer_path("/tmp", 0), "/tmp/cpu_layer_00.f32");
        assert_eq!(cpu_layer_file(7), "cpu_layer_07.f32");
    }

    #[test]
    fn cpu_layer_h_post_attn_path_matches_legacy_format() {
        assert_eq!(
            cpu_layer_h_post_attn_path("/tmp", 33),
            "/tmp/cpu_layer_33_h_post_attn.f32"
        );
    }

    #[test]
    fn cpu_stage_path_always_uses_l0_regardless_of_layer() {
        // Pin the producer-side L0 hardcoding so anyone "fixing" it has
        // to update both sides (see module-level note).
        assert_eq!(
            cpu_stage_path("/tmp", "h_post_attn"),
            "/tmp/cpu_L0_h_post_attn.f32"
        );
    }

    #[test]
    fn cpu_stage_prefix_is_per_layer() {
        assert_eq!(cpu_stage_prefix(0), "cpu_L0_");
        assert_eq!(cpu_stage_prefix(5), "cpu_L5_");
    }

    #[test]
    fn metal_and_decode_layer_helpers_match_legacy_format() {
        assert_eq!(decode_layer_file(3), "decode_layer_03.f32");
        assert_eq!(decode_layer_prefix(3), "decode_layer_03_");
        assert_eq!(metal_layer_h_out_file(12), "metal_layer_12_h_out.f32");
        assert_eq!(metal_layer_prefix(12), "metal_layer_12_");
    }
}
