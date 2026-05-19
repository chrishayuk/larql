//! Per-stage residual capture for backend bisecting.
//!
//! [`ResidualCapture`] captures a *single* `Vec<f32>` per layer (the
//! end-of-layer hidden). That's enough to spot which **layer** first
//! diverges between two backends, but not which **stage within a
//! layer**: norm? QKV proj? QK-norm? RoPE? V-norm? attention? O proj?
//! FFN gate+up? down? When end-to-end parity drifts but every
//! kernel-level test passes, the divergence has to live in stage
//! ordering, parameter binding, or a stage we haven't pinned — and
//! the only way to find it is to dump every intermediate buffer at
//! one layer and diff stage-by-stage.
//!
//! The decode and prefill backends already write per-stage `.f32`
//! files when the right env vars are set:
//! - CPU prefill — `LARQL_CPU_STAGE_DUMP=<dir>` +
//!   `LARQL_STAGE_DUMP_LAYER=<L>` writes `cpu_L0_<stage>.f32`.
//! - Metal prefill — `LARQL_METAL_DUMP_LAYERS=<dir>` +
//!   `LARQL_STAGE_DUMP_LAYER=<L>` writes `metal_layer_NN_<stage>.f32`.
//! - Metal decode — `LARQL_DECODE_DUMP_LAYERS=<dir>` +
//!   `LARQL_STAGE_DUMP_LAYER=<L>` writes `decode_layer_NN_<stage>.f32`.
//!
//! This module owns the temp-dir + env-var plumbing, reads every
//! stage file back into memory as a typed [`StageCapture`], and
//! exposes [`compare_stages`] which walks a caller-supplied list of
//! `(stage_a, stage_b)` name pairs and reports the first divergence.
//!
//! ## Why explicit name pairs
//!
//! CPU prefill captures Q at three points (`q_out_raw`,
//! `q_out_after_qk_norm`, `q_out_after_rope`) because each stage is
//! an `Array2<f32>` allocation; Metal decode does the same work
//! in-place on a single buffer and only sees the final
//! post-everything `q_out`. That asymmetry means a one-to-one stage
//! map doesn't exist: the CPU buffer to compare against Metal's
//! `q_out` is `q_out_after_rope`. Defaulting to magic-string
//! conversion would silently compare against the wrong file the
//! moment a backend grows or trims a stage; the explicit pair list
//! makes the intent visible at the test site.

use std::collections::HashMap;
use std::path::Path;

use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use super::compare::{LayerStat, ParityThreshold};
use crate::forward::dump_config::{
    cpu_stage_prefix, decode_layer_prefix, metal_layer_prefix, ENV_CPU_STAGE_DUMP,
    ENV_DECODE_DUMP_LAYERS, ENV_METAL_DUMP_LAYERS, ENV_STAGE_DUMP_LAYER,
};
use crate::layer_graph::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ;

/// In-memory representation of one backend's per-stage dump for one
/// layer. Stage names are exactly the suffixes the producer wrote
/// (`cpu_L<L>_<stage>` / `metal_layer_NN_<stage>` / `decode_layer_NN_<stage>`).
/// We strip the prefix on read so callers can pair stages by their
/// short name regardless of which backend produced them.
#[derive(Debug, Clone)]
pub struct StageCapture {
    /// Stage suffix → flat float buffer.
    pub stages: HashMap<String, Vec<f32>>,
    /// Layer the dump was captured at.
    pub layer: usize,
    /// Sequence length the dump covers — `> 1` for prefill captures,
    /// `1` for decode captures. Used by [`Self::project_to_last_position`]
    /// to slice prefill stages down to their last row so a multi-position
    /// CPU dump can compare 1:1 against a single-position Metal-decode
    /// dump.
    pub seq_len: usize,
    /// Backend label — for diagnostics in [`StageReport`].
    pub backend: &'static str,
}

impl StageCapture {
    /// Number of stages captured. Useful when callers want to assert
    /// the dump fired (zero stages means the backend didn't honour the
    /// env var, e.g. an env-var typo or the layer didn't reach the
    /// dump point).
    pub fn len(&self) -> usize {
        self.stages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Look up one stage by its short name (no `cpu_L0_` /
    /// `decode_layer_NN_` prefix).
    pub fn get(&self, stage: &str) -> Option<&[f32]> {
        self.stages.get(stage).map(|v| v.as_slice())
    }

    /// Slice every stage down to its last position. CPU prefill
    /// captures the full `[seq_len, stride]` per stage, Metal decode
    /// captures only the single new position; this method bridges
    /// the shape gap so [`compare_stages`] sees `[stride]` on both
    /// sides.
    ///
    /// Per-stage stride is inferred as `len / seq_len`. Stages whose
    /// length isn't an exact multiple of `seq_len` (which would
    /// indicate a different shape contract — e.g. router scores
    /// `[seq_len, num_experts]` accidentally lumped in) are kept
    /// as-is rather than truncated, so an unexpected shape surfaces
    /// as a length mismatch in the comparison rather than getting
    /// silently sliced.
    pub fn project_to_last_position(&self) -> Self {
        let mut out: HashMap<String, Vec<f32>> = HashMap::with_capacity(self.stages.len());
        for (name, v) in &self.stages {
            if self.seq_len <= 1 || !v.len().is_multiple_of(self.seq_len) {
                out.insert(name.clone(), v.clone());
                continue;
            }
            let stride = v.len() / self.seq_len;
            let start = (self.seq_len - 1) * stride;
            out.insert(name.clone(), v[start..start + stride].to_vec());
        }
        Self {
            stages: out,
            layer: self.layer,
            seq_len: 1,
            backend: self.backend,
        }
    }

    /// Drive a CPU prefill with `LARQL_CPU_STAGE_DUMP` + `LARQL_STAGE_DUMP_LAYER`
    /// active for `layer`, then collect every `cpu_L<layer>_<stage>.f32` it
    /// wrote. Stages produced by the CPU path:
    ///   `norm_out`, `q_out_raw`, `q_out_after_qk_norm`,
    ///   `q_out_after_rope`, `k_out_after_rope`, `v_out`, `attn_out`,
    ///   `o_out`, `h_post_attn`, `ffn_norm_out`, `ffn_out_raw`.
    /// The exact set may grow as more dumps are wired into
    /// `attention/block.rs` / `forward/layer.rs`.
    pub fn cpu_prefill(
        weights: &mut ModelWeights,
        ids: &[u32],
        index: &VectorIndex,
        layer: usize,
    ) -> Result<Self, String> {
        let dir = run_with_two_env_vars(
            ENV_CPU_STAGE_DUMP,
            ENV_STAGE_DUMP_LAYER,
            &layer.to_string(),
            || {
                let _ = crate::vindex::predict_q4k_hidden(weights, ids, index, None);
            },
        )?;
        let prefix = cpu_stage_prefix(layer);
        Ok(Self {
            stages: read_stage_dir(dir.path(), &prefix)?,
            layer,
            seq_len: ids.len(),
            backend: "cpu_prefill",
        })
    }

    /// Drive Metal prefill with `LARQL_METAL_DUMP_LAYERS` +
    /// `LARQL_STAGE_DUMP_LAYER`. Stages produced by the Metal-prefill
    /// path: `norm_out`, `q_out`, `k_out`, `v_out`, `attn_out`,
    /// `o_out`, `ffn_norm_out`, `gate_out`, `up_out`, `act_buf`,
    /// `down_out`. Note the absence of `h_post_attn` in the per-stage
    /// dump — Metal-prefill writes that one to `metal_layer_NN_h_post_attn.f32`
    /// for *every* layer, not just the named stage layer; this
    /// reader picks it up regardless.
    pub fn metal_prefill(
        weights: &mut ModelWeights,
        ids: &[u32],
        index: &VectorIndex,
        backend: &dyn ComputeBackend,
        layer: usize,
    ) -> Result<Self, String> {
        let dir = run_with_two_env_vars(
            ENV_METAL_DUMP_LAYERS,
            ENV_STAGE_DUMP_LAYER,
            &layer.to_string(),
            || {
                let cached = crate::layer_graph::CachedLayerGraph::from_residuals(Vec::new());
                let dummy_tok = build_dummy_tokenizer();
                let n = weights.num_layers;
                let _ = crate::layer_graph::generate::generate(
                    weights,
                    &dummy_tok,
                    ids,
                    1,
                    index,
                    backend,
                    &cached,
                    0..n,
                );
            },
        )?;
        let prefix = metal_layer_prefix(layer);
        Ok(Self {
            stages: read_stage_dir(dir.path(), &prefix)?,
            layer,
            seq_len: ids.len(),
            backend: "metal_prefill",
        })
    }

    /// Drive Metal prefill on `prefix_ids` then a single
    /// `decode_token(new_id)` with `LARQL_DECODE_DUMP_LAYERS` +
    /// `LARQL_STAGE_DUMP_LAYER` active for `layer`. Stages produced:
    /// `norm_out`, `q_out`, `k_out`, `v_out`, `attn_out`, `o_out`,
    /// `h_post_attn`, `ffn_norm_out`, `gate_out`, `up_out`,
    /// `act_buf`, `down_out`. Names match the Metal-prefill set so
    /// callers can pair them 1:1 via [`compare_stages`].
    pub fn metal_decode(
        weights: &mut ModelWeights,
        prefix_ids: &[u32],
        new_id: u32,
        index: &VectorIndex,
        backend: &dyn ComputeBackend,
        layer: usize,
    ) -> Result<Self, String> {
        // Driver mirrors `ResidualCapture::metal_decode` — we go
        // through the same backend prefill+decode entry point so the
        // shaders dispatched are identical to production.
        let hidden = weights.hidden_size;
        let num_layers = weights.num_layers;
        let arch = &*weights.arch;

        backend.reset_kv_cache();
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, DEFAULT_GPU_KV_CACHE_MAX_SEQ);

        use larql_vindex::GateIndex;
        let gate_index: &dyn GateIndex = index;
        let (q4_ffn, ffn_is_q4k) = if let Some(m) = gate_index.interleaved_q4k_mmap_ref() {
            (Some(m), true)
        } else {
            (gate_index.interleaved_q4_mmap_ref(), false)
        };
        let q4_ffn_mmap = q4_ffn.ok_or("no Q4 FFN mmap available for decode capture")?;
        let intermediate = gate_index.num_features(0);
        let ffn_format = if ffn_is_q4k {
            larql_compute::QuantFormat::Q4_K
        } else {
            larql_compute::QuantFormat::Q4_0
        };
        let q4_ffn_per_matrix = ffn_format
            .packed_matrix_bytes(intermediate, hidden)
            .ok_or("unsupported Q4 FFN format for decode capture")?;
        let pipeline_layers = crate::layer_graph::pipeline_layer::build_pipeline_layers(
            weights,
            index,
            0..num_layers,
            q4_ffn_mmap,
            q4_ffn_per_matrix,
            ffn_format,
        );

        let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
        let qk_norm_val = arch.attn_q_norm_key(0).is_some();

        let h_embed = crate::forward::embed_tokens_pub(weights, prefix_ids);
        let prefill_x: Vec<f32> = h_embed.as_slice().unwrap().to_vec();
        backend
            .prefill_q4(
                &pipeline_layers,
                &prefill_x,
                hidden,
                intermediate,
                prefix_ids.len(),
                qk_norm_val,
                softcap,
            )
            .ok_or("Metal prefill_q4 returned None")?;

        let dec_embed = crate::forward::embed_tokens_pub(weights, &[new_id]);
        let dec_x: Vec<f32> = dec_embed.row(0).to_vec();
        let dir = run_with_two_env_vars(
            ENV_DECODE_DUMP_LAYERS,
            ENV_STAGE_DUMP_LAYER,
            &layer.to_string(),
            || {
                let _ = backend.decode_token(&pipeline_layers, &dec_x, hidden, intermediate);
            },
        )?;
        let prefix = decode_layer_prefix(layer);
        Ok(Self {
            stages: read_stage_dir(dir.path(), &prefix)?,
            layer,
            seq_len: 1,
            backend: "metal_decode",
        })
    }
}

// ── Comparison ──────────────────────────────────────────────────────────────

/// One stage's diff. `stat` carries the same cos / max_abs metrics
/// [`LayerStat`] uses; `name_a`/`name_b` are the file-suffix names so
/// the report can name which file pair was diffed.
#[derive(Debug, Clone)]
pub struct StagePair {
    pub name_a: String,
    pub name_b: String,
    pub stat: LayerStat,
    /// True when the stage was missing on either side. Inspect this
    /// before reading `stat` — a missing stage surfaces as cos=0,
    /// max_abs=inf so `assert_clean` flags it, but the cause is
    /// "wasn't dumped" not "diverged".
    pub missing: bool,
}

#[derive(Debug, Clone)]
pub struct StageReport {
    pub a_backend: &'static str,
    pub b_backend: &'static str,
    pub layer: usize,
    pub pairs: Vec<StagePair>,
    pub first_bad: Option<usize>,
    pub threshold: ParityThreshold,
}

impl StageReport {
    pub fn is_clean(&self) -> bool {
        self.first_bad.is_none()
    }

    /// Emit a one-line summary per stage, marking the first-bad row
    /// with a "←" so the diverging stage stands out at a glance. Used
    /// directly in test failure messages.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "stage diff @L{} ({} vs {}, threshold cos≥{} rel≤{}):\n",
            self.layer,
            self.a_backend,
            self.b_backend,
            self.threshold.cos,
            self.threshold.rel_max_abs,
        );
        for (i, p) in self.pairs.iter().enumerate() {
            let mark = if Some(i) == self.first_bad {
                " ←"
            } else {
                ""
            };
            if p.missing {
                s.push_str(&format!(
                    "  {:<24} MISSING ({}/{}){}\n",
                    p.name_a, p.name_a, p.name_b, mark,
                ));
            } else {
                s.push_str(&format!(
                    "  {:<24} cos={:.6} max_abs={:.3e} rel={:.3}%{}\n",
                    p.name_a,
                    p.stat.cos,
                    p.stat.max_abs,
                    100.0 * p.stat.rel_max_abs(),
                    mark,
                ));
            }
        }
        s
    }

    pub fn assert_clean(&self) -> Result<(), String> {
        if self.first_bad.is_none() {
            return Ok(());
        }
        Err(self.summary())
    }
}

/// Compare a list of `(stage_in_a, stage_in_b)` name pairs between
/// two captures. Pairs are evaluated **in order** so the first
/// divergence (per the threshold) is identifiable as the localised
/// stage where two backends start to disagree.
pub fn compare_stages(
    a: &StageCapture,
    b: &StageCapture,
    pairs: &[(&str, &str)],
    threshold: ParityThreshold,
) -> StageReport {
    let mut out = Vec::with_capacity(pairs.len());
    let mut first_bad: Option<usize> = None;
    for (i, &(name_a, name_b)) in pairs.iter().enumerate() {
        let (av, bv) = match (a.get(name_a), b.get(name_b)) {
            (Some(av), Some(bv)) => (av, bv),
            _ => {
                out.push(StagePair {
                    name_a: name_a.into(),
                    name_b: name_b.into(),
                    stat: LayerStat {
                        layer: a.layer,
                        cos: 0.0,
                        max_abs: f32::INFINITY,
                        a_norm: 0.0,
                        b_norm: 0.0,
                    },
                    missing: true,
                });
                if first_bad.is_none() {
                    first_bad = Some(i);
                }
                continue;
            }
        };
        let stat = stage_stat(a.layer, av, bv);
        let bad = av.len() != bv.len()
            || stat.cos < threshold.cos
            || stat.rel_max_abs() > threshold.rel_max_abs;
        if bad && first_bad.is_none() {
            first_bad = Some(i);
        }
        out.push(StagePair {
            name_a: name_a.into(),
            name_b: name_b.into(),
            stat,
            missing: false,
        });
    }
    StageReport {
        a_backend: a.backend,
        b_backend: b.backend,
        layer: a.layer,
        pairs: out,
        first_bad,
        threshold,
    }
}

// ── Internals ──────────────────────────────────────────────────────────────

fn stage_stat(layer: usize, a: &[f32], b: &[f32]) -> LayerStat {
    if a.len() != b.len() {
        return LayerStat {
            layer,
            cos: 0.0,
            max_abs: f32::INFINITY,
            a_norm: 0.0,
            b_norm: 0.0,
        };
    }
    let mut dot = 0.0f64;
    let mut a_sq = 0.0f64;
    let mut b_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        a_sq += x * x;
        b_sq += y * y;
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cos = if a_sq > 0.0 && b_sq > 0.0 {
        (dot / (a_sq.sqrt() * b_sq.sqrt())) as f32
    } else {
        0.0
    };
    LayerStat {
        layer,
        cos,
        max_abs,
        a_norm: a_sq.sqrt() as f32,
        b_norm: b_sq.sqrt() as f32,
    }
}

/// Set two env vars together (a dir-typed one and a layer-index one),
/// run `f`, restore them. Used because every stage dump is gated by
/// the *pair* (output dir + which layer to dump).
fn run_with_two_env_vars(
    dir_var: &str,
    layer_var: &str,
    layer_value: &str,
    f: impl FnOnce(),
) -> Result<tempfile::TempDir, String> {
    let dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let prev_dir = std::env::var(dir_var).ok();
    let prev_layer = std::env::var(layer_var).ok();
    std::env::set_var(dir_var, dir.path());
    std::env::set_var(layer_var, layer_value);
    f();
    match prev_dir {
        Some(v) => std::env::set_var(dir_var, v),
        None => std::env::remove_var(dir_var),
    }
    match prev_layer {
        Some(v) => std::env::set_var(layer_var, v),
        None => std::env::remove_var(layer_var),
    }
    Ok(dir)
}

/// Walk `dir`, pick up every `*.f32` whose name starts with `prefix`,
/// strip the prefix and the trailing `.f32`, return the rest as the
/// stage name. Errors only on filesystem read failures — a totally
/// empty directory returns an empty map (the caller's `is_empty()`
/// catches that).
fn read_stage_dir(dir: &Path, prefix: &str) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut out = HashMap::new();
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("read_dir({}): {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry: {e}"))?;
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(rest) = fname.strip_prefix(prefix) else {
            continue;
        };
        let Some(stage) = rest.strip_suffix(".f32") else {
            continue;
        };
        let Some(v) = read_f32_vec(&path) else {
            return Err(format!("could not read f32 file {}", path.display()));
        };
        out.insert(stage.to_string(), v);
    }
    Ok(out)
}

fn read_f32_vec(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn build_dummy_tokenizer() -> tokenizers::Tokenizer {
    use tokenizers::models::wordpiece::WordPiece;
    let model = WordPiece::default();
    tokenizers::Tokenizer::new(model)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn cap(stages: &[(&str, Vec<f32>)], layer: usize, backend: &'static str) -> StageCapture {
        StageCapture {
            stages: stages
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            layer,
            seq_len: 1,
            backend,
        }
    }

    fn cap_with_seq(
        stages: &[(&str, Vec<f32>)],
        layer: usize,
        seq_len: usize,
        backend: &'static str,
    ) -> StageCapture {
        StageCapture {
            stages: stages
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            layer,
            seq_len,
            backend,
        }
    }

    #[test]
    fn project_to_last_position_slices_per_stride() {
        // [seq=3, hidden=2] for s0; [seq=3, qdim=4] for s1.
        let s0 = vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0];
        let s1 = vec![0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 9.1, 9.2, 9.3, 9.4];
        let cap = cap_with_seq(&[("s0", s0), ("s1", s1)], 0, 3, "cpu");
        let proj = cap.project_to_last_position();
        assert_eq!(proj.seq_len, 1);
        assert_eq!(proj.get("s0").unwrap(), &[100.0, 200.0]);
        assert_eq!(proj.get("s1").unwrap(), &[9.1, 9.2, 9.3, 9.4]);
    }

    #[test]
    fn project_to_last_position_keeps_unaligned_stages_unchanged() {
        // seq_len=3 but stage has 7 floats (not a multiple of 3) —
        // unexpected shape. Don't truncate; let the comparison
        // surface it as a length mismatch.
        let cap = cap_with_seq(&[("weird", vec![1.0; 7])], 0, 3, "cpu");
        let proj = cap.project_to_last_position();
        assert_eq!(proj.get("weird").unwrap().len(), 7);
    }

    #[test]
    fn compare_stages_clean_when_all_match() {
        let a = cap(
            &[("norm_out", vec![1.0, 2.0]), ("q_out", vec![3.0, 4.0])],
            0,
            "a",
        );
        let b = cap(
            &[("norm_out", vec![1.0, 2.0]), ("q_out", vec![3.0, 4.0])],
            0,
            "b",
        );
        let r = compare_stages(
            &a,
            &b,
            &[("norm_out", "norm_out"), ("q_out", "q_out")],
            ParityThreshold::tight(),
        );
        assert!(r.is_clean(), "{}", r.summary());
    }

    #[test]
    fn compare_stages_first_bad_is_first_diverging() {
        // Stage 0 matches, stage 1 diverges — first_bad must be 1.
        let a = cap(&[("s0", vec![1.0; 4]), ("s1", vec![1.0; 4])], 0, "a");
        let mut b1 = vec![1.0; 4];
        b1[0] = 100.0;
        let b = cap(&[("s0", vec![1.0; 4]), ("s1", b1)], 0, "b");
        let r = compare_stages(
            &a,
            &b,
            &[("s0", "s0"), ("s1", "s1")],
            ParityThreshold::tight(),
        );
        assert_eq!(r.first_bad, Some(1));
        assert!(!r.is_clean());
        assert!(r.summary().contains("s1"));
    }

    #[test]
    fn compare_stages_missing_stage_flags_first_bad() {
        let a = cap(&[("s0", vec![1.0])], 0, "a");
        let b = cap(&[("s0", vec![1.0])], 0, "b");
        // Asking for "s1" which neither side has.
        let r = compare_stages(
            &a,
            &b,
            &[("s0", "s0"), ("s1", "s1")],
            ParityThreshold::tight(),
        );
        assert_eq!(r.first_bad, Some(1));
        assert!(r.pairs[1].missing);
    }

    #[test]
    fn compare_stages_supports_asymmetric_names() {
        // CPU's "q_out_after_rope" pairs with Metal's "q_out".
        let a = cap(&[("q_out_after_rope", vec![1.0, 2.0])], 0, "cpu");
        let b = cap(&[("q_out", vec![1.0, 2.0])], 0, "metal");
        let r = compare_stages(
            &a,
            &b,
            &[("q_out_after_rope", "q_out")],
            ParityThreshold::tight(),
        );
        assert!(r.is_clean());
    }
}
