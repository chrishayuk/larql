//! Layer normalization and residual stream operations.

use ndarray::Array2;

/// Default norm epsilon. Most models use 1e-5 or 1e-6.
/// Callers should prefer passing `arch.norm_eps()` explicitly.
pub const DEFAULT_EPS: f64 = 1e-6;

/// RMS norm with the legacy default epsilon ([`DEFAULT_EPS`] = 1e-6).
///
/// Prefer [`rms_norm_for_arch`] when an architecture handle is available —
/// it reads `arch.norm_eps()` from the parsed config (Mistral / Llama 3 /
/// Gemma 3 ship `rms_norm_eps = 1e-5`, not 1e-6), with the
/// `LARQL_NORM_EPS_OVERRIDE` env var taking precedence for diagnostic
/// bisection. This bare version is for tests and for code paths that
/// genuinely have no model context.
pub fn rms_norm(x: &Array2<f32>, weight: Option<&Vec<f32>>, offset: f32) -> Array2<f32> {
    rms_norm_eps(x, weight, offset, DEFAULT_EPS)
}

/// RMS norm with eps sourced from `arch.norm_eps()` (parsed from config.json)
/// or overridden by `LARQL_NORM_EPS_OVERRIDE`. The arch-driven path is the
/// permanent fix for bug 2 in
/// `docs/diagnoses/shannon-cross-engine-divergence.md`; the env var stays
/// as a diagnostic instrument.
pub fn rms_norm_for_arch(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    offset: f32,
    arch: &dyn larql_models::ModelArchitecture,
) -> Array2<f32> {
    let eps = crate::forward_overrides::norm_eps_override()
        .map(|v| v as f64)
        .unwrap_or_else(|| arch.norm_eps() as f64);
    rms_norm_eps(x, weight, offset, eps)
}

/// RMS norm with explicit epsilon.
pub fn rms_norm_eps(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    offset: f32,
    eps: f64,
) -> Array2<f32> {
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((rows, cols));

    for i in 0..rows {
        let row = x.row(i);
        let sq_sum: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = (sq_sum / cols as f64 + eps).sqrt() as f32;
        for j in 0..cols {
            let w = match weight {
                Some(wt) => offset + wt[j],
                None => 1.0,
            };
            out[[i, j]] = row[j] / rms * w;
        }
    }
    out
}

/// LayerNorm: (x - mean) / std * weight + bias.
/// Uses f64 accumulation for mean/variance.
///
/// Prefer [`layer_norm_for_arch`] when an architecture handle is available
/// — it reads `arch.norm_eps()` from the parsed config (StarCoder2 ships
/// `norm_epsilon = 1e-5`, GPT-2 ships `layer_norm_epsilon = 1e-5`).
pub fn layer_norm(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    bias: Option<&Vec<f32>>,
) -> Array2<f32> {
    layer_norm_eps(x, weight, bias, DEFAULT_EPS)
}

/// LayerNorm with eps sourced from `arch.norm_eps()` (parsed from
/// `norm_epsilon` / `layer_norm_eps` / `layer_norm_epsilon`), overridden
/// by `LARQL_NORM_EPS_OVERRIDE`. Companion to [`rms_norm_for_arch`].
pub fn layer_norm_for_arch(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    bias: Option<&Vec<f32>>,
    arch: &dyn larql_models::ModelArchitecture,
) -> Array2<f32> {
    let eps = crate::forward_overrides::norm_eps_override()
        .map(|v| v as f64)
        .unwrap_or_else(|| arch.norm_eps() as f64);
    layer_norm_eps(x, weight, bias, eps)
}

/// LayerNorm with explicit epsilon.
pub fn layer_norm_eps(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    bias: Option<&Vec<f32>>,
    eps: f64,
) -> Array2<f32> {
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((rows, cols));

    for i in 0..rows {
        let row = x.row(i);
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
        let var: f64 = row
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / cols as f64;
        let std = (var + eps).sqrt() as f32;
        let mean_f = mean as f32;
        for j in 0..cols {
            let normed = (row[j] - mean_f) / std;
            let w = weight.map_or(1.0, |wt| wt[j]);
            let b = bias.map_or(0.0, |bt| bt[j]);
            out[[i, j]] = normed * w + b;
        }
    }
    out
}

/// Per-head RMS norm without learned weights (parameter-free normalization).
/// Used for V-norm in Gemma 4: just normalizes, no scaling.
pub fn rms_norm_heads_no_weight(x: &Array2<f32>, num_heads: usize, head_dim: usize) -> Array2<f32> {
    rms_norm_heads_no_weight_eps(x, num_heads, head_dim, DEFAULT_EPS)
}

/// Per-head parameter-free RMS norm with explicit epsilon.
pub fn rms_norm_heads_no_weight_eps(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = h * head_dim;
            let mut sq_sum = 0.0f64;
            for d in 0..head_dim {
                let v = x[[s, off + d]] as f64;
                sq_sum += v * v;
            }
            let rms = (sq_sum / head_dim as f64 + eps).sqrt() as f32;
            for d in 0..head_dim {
                out[[s, off + d]] = x[[s, off + d]] / rms;
            }
        }
    }
    out
}

/// Per-head RMS norm for Q/K projections with configurable weight offset.
/// Uses f64 accumulation for the sum-of-squares.
pub fn rms_norm_heads(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
) -> Array2<f32> {
    rms_norm_heads_eps(x, weight, num_heads, head_dim, offset, DEFAULT_EPS)
}

/// Per-head RMS norm with explicit epsilon.
pub fn rms_norm_heads_eps(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
    eps: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = h * head_dim;
            let mut sq_sum = 0.0f64;
            for d in 0..head_dim {
                let v = x[[s, off + d]] as f64;
                sq_sum += v * v;
            }
            let rms = (sq_sum / head_dim as f64 + eps).sqrt() as f32;
            for d in 0..head_dim {
                out[[s, off + d]] = x[[s, off + d]] / rms * (offset + weight[d]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── rms_norm ──────────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_shape_preserved() {
        let x = Array2::from_shape_vec((3, 4), vec![1.0f32; 12]).unwrap();
        let out = rms_norm(&x, None, 0.0);
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn rms_norm_output_is_finite() {
        let x = Array2::from_shape_vec((2, 8), (0..16).map(|i| i as f32 * 0.1).collect()).unwrap();
        let out = rms_norm(&x, None, 0.0);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "rms_norm produced non-finite values"
        );
    }

    #[test]
    fn rms_norm_with_ones_weight_and_offset_one() {
        // weight=ones, offset=1.0 → Gemma-style: weight = 1.0 + learned (learned=0 here)
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = vec![0.0f32; 4]; // learned weight = zeros
        let out = rms_norm(&x, Some(&w), 1.0); // effective weight = 1.0 + 0.0 = 1.0
        let out_no_w = rms_norm(&x, None, 0.0);
        // Both paths should give the same result since effective weight=1 for both
        for (a, b) in out.iter().zip(out_no_w.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "offset=1 with zero weight should match no-weight norm"
            );
        }
    }

    #[test]
    fn rms_norm_zero_row_is_finite() {
        // Zero input → norm = 0 → eps prevents div-by-zero
        let x = Array2::zeros((1, 4));
        let out = rms_norm(&x, None, 0.0);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── layer_norm ────────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_shape_and_finite() {
        let x = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32).collect()).unwrap();
        let w = vec![1.0f32; 4];
        let b = vec![0.0f32; 4];
        let out = layer_norm(&x, Some(&w), Some(&b));
        assert_eq!(out.shape(), x.shape());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let x = Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32).collect()).unwrap();
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];
        let out = layer_norm(&x, Some(&w), Some(&b));
        let mean: f32 = out.row(0).iter().sum::<f32>() / 8.0;
        let var: f32 = out.row(0).iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.1, "var should be ~1, got {var}");
    }

    // ── rms_norm_heads ────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_heads_no_weight_shape() {
        // [seq, num_heads * head_dim]
        let x = Array2::from_shape_vec((3, 8), (0..24).map(|i| i as f32 * 0.1).collect()).unwrap();
        let out = rms_norm_heads_no_weight(&x, 2, 4);
        assert_eq!(out.shape(), &[3, 8]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rms_norm_heads_normalises_each_head_independently() {
        // Two heads with very different magnitudes → both normalised
        let mut data = vec![0.0f32; 8];
        for (i, slot) in data.iter_mut().enumerate().take(4) {
            *slot = (i + 1) as f32;
        } // head 0: [1,2,3,4]
        for (i, slot) in data.iter_mut().enumerate().skip(4).take(4) {
            *slot = 100.0 * (i - 4 + 1) as f32;
        } // head 1: [100,200,300,400]
        let x = Array2::from_shape_vec((1, 8), data).unwrap();
        let out = rms_norm_heads_no_weight(&x, 2, 4);
        // Both heads should have similar L2 norm after per-head normalisation
        let h0_norm: f32 = out.row(0).iter().take(4).map(|v| v * v).sum::<f32>().sqrt();
        let h1_norm: f32 = out.row(0).iter().skip(4).map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (h0_norm - h1_norm).abs() < 0.1,
            "both heads should have similar L2 norm"
        );
    }

    #[test]
    fn rms_norm_heads_with_weight_scales() {
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = vec![2.0f32, 2.0, 2.0, 2.0]; // scale by 2
        let out_scaled = rms_norm_heads(&x, &w, 1, 4, 0.0);
        let out_unscaled = rms_norm_heads_no_weight(&x, 1, 4);
        // Scaled output should be ~2× the unscaled
        for (s, u) in out_scaled.iter().zip(out_unscaled.iter()) {
            assert!(
                (s - 2.0 * u).abs() < 1e-5,
                "weight=2 should double the output"
            );
        }
    }

    // ── rms_norm_for_arch / layer_norm_for_arch eps wiring ───────────────────
    //
    // The default `rms_norm` (legacy) uses DEFAULT_EPS = 1e-6. The
    // arch-aware variants must read `arch.norm_eps()`. These are the
    // unit-level gates for bug 2 in
    // docs/diagnoses/shannon-cross-engine-divergence.md.

    fn build_arch_with_eps(eps: f64) -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": eps,
        }))
    }

    #[test]
    fn rms_norm_for_arch_reads_arch_eps_not_default() {
        // Construct a residual where the squared mean is small enough that
        // 1e-6 vs 1e-5 changes the rsqrt term measurably (>1e-4 ULP), so
        // the two eps values produce visibly different outputs.
        let x = Array2::from_shape_vec((1, 4), vec![0.001_f32, 0.001, 0.001, 0.001]).unwrap();
        let arch_strict = build_arch_with_eps(1e-6);
        let arch_loose = build_arch_with_eps(1e-5);
        let out_strict = rms_norm_for_arch(&x, None, 0.0, &*arch_strict);
        let out_loose = rms_norm_for_arch(&x, None, 0.0, &*arch_loose);
        let max_diff = out_strict
            .iter()
            .zip(out_loose.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 0.01,
            "arch.norm_eps() not honoured: outputs match with eps 1e-6 vs 1e-5 (max diff {max_diff})"
        );
    }

    #[test]
    fn rms_norm_for_arch_falls_back_to_default_when_arch_omits_eps() {
        // Build an arch without rms_norm_eps so arch.norm_eps() returns
        // the trait default (DEFAULT_NORM_EPS = 1e-6). Result should match
        // bare rms_norm exactly.
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        }));
        let x = Array2::from_shape_vec((1, 4), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let out_arch = rms_norm_for_arch(&x, None, 0.0, &*arch);
        let out_default = rms_norm(&x, None, 0.0);
        for (a, d) in out_arch.iter().zip(out_default.iter()) {
            assert!(
                (a - d).abs() < 1e-7,
                "arch.norm_eps() fallback must equal DEFAULT_EPS path"
            );
        }
    }

    #[test]
    fn layer_norm_for_arch_reads_arch_eps_not_default() {
        let x = Array2::from_shape_vec((1, 4), vec![0.001_f32, 0.001, 0.001, 0.001]).unwrap();
        let arch_strict = build_arch_with_eps(1e-6);
        let arch_loose = build_arch_with_eps(1e-5);
        let out_strict = layer_norm_for_arch(&x, None, None, &*arch_strict);
        let out_loose = layer_norm_for_arch(&x, None, None, &*arch_loose);
        let max_diff = out_strict
            .iter()
            .zip(out_loose.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // LayerNorm subtracts mean first; for a uniform vector the
        // numerator is 0 and the output is determined entirely by eps
        // through the variance term. Either eps produces 0/sqrt(eps) = 0
        // for both, so this case is degenerate. Use a non-uniform vector
        // to exercise the eps difference.
        if max_diff <= 1e-6 {
            let x2 = Array2::from_shape_vec((1, 4), vec![0.001_f32, 0.002, 0.003, 0.004]).unwrap();
            let s = layer_norm_for_arch(&x2, None, None, &*arch_strict);
            let l = layer_norm_for_arch(&x2, None, None, &*arch_loose);
            let d = s
                .iter()
                .zip(l.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                d > 1e-5,
                "layer_norm_for_arch did not honour arch.norm_eps()"
            );
        }
    }
}
