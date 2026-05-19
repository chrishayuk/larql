//! Small math utilities shared by `forward/` and `attention/`.

use crate::model::ModelWeights;
use crate::residual::{layer_norm_for_arch, rms_norm_for_arch};
use larql_models::NormType;
use ndarray::Array2;

/// Apply the appropriate norm (RMSNorm or LayerNorm) based on architecture.
pub fn apply_norm(
    weights: &ModelWeights,
    x: &Array2<f32>,
    weight_key: &str,
    norm_offset: f32,
) -> Array2<f32> {
    match weights.arch.norm_type() {
        NormType::LayerNorm => {
            let bias_key = weight_key.replace(".weight", ".bias");
            layer_norm_for_arch(
                x,
                weights.vectors.get(weight_key),
                weights.vectors.get(&bias_key),
                &*weights.arch,
            )
        }
        _ => rms_norm_for_arch(
            x,
            weights.vectors.get(weight_key),
            norm_offset,
            &*weights.arch,
        ),
    }
}

/// Compute x @ w.T via BLAS.
pub fn dot_proj(
    x: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    w: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
) -> Array2<f32> {
    x.dot(&w.t())
}

/// Numerically-stable softmax. Returns an empty vec for empty input.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

/// Add a 1D bias vector to each row of a 2D matrix.
pub fn add_bias(x: &mut Array2<f32>, bias: &[f32]) {
    let cols = x.shape()[1];
    let n = cols.min(bias.len());
    for mut row in x.rows_mut() {
        for j in 0..n {
            row[j] += bias[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    // ── dot_proj ──────────────────────────────────────────────────────────────

    #[test]
    fn dot_proj_shape() {
        let x = Array2::<f32>::from_elem((3, 4), 1.0);
        let w = Array2::<f32>::from_elem((5, 4), 1.0);
        let out = dot_proj(&x, &w);
        assert_eq!(out.shape(), &[3, 5]);
    }

    #[test]
    fn dot_proj_identity_weight() {
        // x @ I^T = x when w is identity
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let w = Array2::eye(3);
        let out = dot_proj(&x, &w);
        for i in 0..2 {
            for j in 0..3 {
                assert!((out[[i, j]] - x[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn dot_proj_values_correct() {
        // [1,2] @ [[3],[4]]^T = [1*3+2*4] = [11]
        let x = Array2::from_shape_vec((1, 2), vec![1.0f32, 2.0]).unwrap();
        let w = Array2::from_shape_vec((1, 2), vec![3.0f32, 4.0]).unwrap();
        let out = dot_proj(&x, &w);
        assert_eq!(out.shape(), &[1, 1]);
        assert!((out[[0, 0]] - 11.0).abs() < 1e-5);
    }

    // ── add_bias ──────────────────────────────────────────────────────────────

    #[test]
    fn add_bias_all_rows_updated() {
        let mut x = Array2::from_elem((3, 4), 1.0f32);
        let bias = vec![0.1f32, 0.2, 0.3, 0.4];
        add_bias(&mut x, &bias);
        for row in x.rows() {
            for (j, v) in row.iter().enumerate() {
                assert!(
                    (v - (1.0 + bias[j])).abs() < 1e-6,
                    "row val wrong at col {j}"
                );
            }
        }
    }

    #[test]
    fn add_bias_shorter_bias_does_not_overflow() {
        let mut x = Array2::from_elem((2, 4), 0.0f32);
        let bias = vec![1.0f32, 2.0]; // shorter than cols
        add_bias(&mut x, &bias);
        for row in x.rows() {
            assert!((row[0] - 1.0).abs() < 1e-6);
            assert!((row[1] - 2.0).abs() < 1e-6);
            assert!(row[2].abs() < 1e-6, "col 2 should be unmodified");
            assert!(row[3].abs() < 1e-6, "col 3 should be unmodified");
        }
    }

    #[test]
    fn add_bias_zero_bias_is_noop() {
        let orig = Array2::from_elem((2, 3), 5.0f32);
        let mut x = orig.clone();
        add_bias(&mut x, &[0.0, 0.0, 0.0]);
        assert_eq!(x, orig);
    }

    // ── apply_norm ────────────────────────────────────────────────────────────

    #[test]
    fn apply_norm_output_shape_matches_input() {
        let weights = make_test_weights();
        let x = Array2::from_elem((2, weights.hidden_size), 0.5f32);
        let norm_key = weights.arch.input_layernorm_key(0);
        let out = apply_norm(&weights, &x, &norm_key, 0.0);
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn apply_norm_output_is_finite() {
        let weights = make_test_weights();
        let x = Array2::from_elem((1, weights.hidden_size), 1.0f32);
        let norm_key = weights.arch.input_layernorm_key(0);
        let out = apply_norm(&weights, &x, &norm_key, 0.0);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "apply_norm produced non-finite values"
        );
    }

    #[test]
    fn apply_norm_with_offset_differs_from_without() {
        let weights = make_test_weights();
        let x = Array2::from_elem((1, weights.hidden_size), 1.0f32);
        let norm_key = weights.arch.input_layernorm_key(0);
        let out0 = apply_norm(&weights, &x, &norm_key, 0.0);
        let out1 = apply_norm(&weights, &x, &norm_key, 1.0);
        // offset=1.0 means weight = 1 + learned; result should differ
        assert_ne!(
            out0, out1,
            "different offsets should produce different norms"
        );
    }
}
