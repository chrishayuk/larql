//! Shared sparse FFN computation — architecture-correct.
//!
//! Given a set of pre-selected feature indices, computes the FFN output using
//! the model architecture trait for activation function, gating, and bias.
//! Backends only need to provide the feature selection; this module handles
//! the gather, activation, and down projection.
//!
//! Supports gated (Gemma/Llama/Mistral) and non-gated (StarCoder2) models,
//! SiLU and GELU activations, optional up/down bias, and down vector overrides.

use ndarray::Array2;

use super::weight::dense_ffn_forward;
use super::{gelu_tanh, sigmoid};
use crate::forward::add_bias;
use crate::model::ModelWeights;

/// Compute FFN output for a pre-selected set of features.
///
/// Architecture-correct: reads ffn_type, activation, and bias from the model.
/// Falls back to dense (via `weight::dense_ffn_forward`) when K >= 80%.
///
/// `overrides`: optional down vector replacements for patched features.
/// When a feature has an override, its custom down vector is used instead of
/// the model's W_down column. This enables training-free INSERT.
pub fn sparse_ffn_forward(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
) -> (Array2<f32>, Array2<f32>) {
    sparse_ffn_forward_impl(weights, layer, x, features, &[])
}

/// Sparse FFN with down vector overrides (for patched features).
pub fn sparse_ffn_forward_with_overrides(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[(usize, &[f32])],
) -> (Array2<f32>, Array2<f32>) {
    sparse_ffn_forward_impl(weights, layer, x, features, overrides)
}

/// Per-slot override carrying any combination of gate / up / down vectors.
/// Used by `sparse_ffn_forward_with_full_overrides` so that an INSERT's
/// (gate * g_ref * 30, up * u_ref, down * d_ref * alpha_mul) all flow into
/// the activation computation, rather than only the down vector being
/// swapped post-activation. Each field is `None` when the original
/// model weight at that slot should be used.
#[derive(Debug, Clone, Copy)]
pub struct FeatureSlotOverride<'a> {
    pub feature: usize,
    pub gate: Option<&'a [f32]>,
    pub up: Option<&'a [f32]>,
    pub down: Option<&'a [f32]>,
}

/// Sparse FFN with full slot overrides — gate, up, and down can each
/// be replaced per feature. This is the path the LQL `INSERT` constellation
/// takes when the vindex hasn't loaded `up_features.bin` (the LQL `USE`
/// fast path doesn't load it). Without this entry point, the activation
/// at an installed slot would compute `silu(weak_gate · x) * (weak_up · x)`
/// using the original free-slot vectors, and the strong overlay gate
/// would never reach inference.
pub fn sparse_ffn_forward_with_full_overrides(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[FeatureSlotOverride<'_>],
) -> (Array2<f32>, Array2<f32>) {
    sparse_ffn_forward_full_impl(weights, layer, x, features, overrides)
}

fn sparse_ffn_forward_impl(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[(usize, &[f32])],
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
    let hidden = x.shape()[1];
    let intermediate = w_up.shape()[0];
    let seq_len = x.shape()[0];
    let k = features.len();

    if k == 0 {
        return (
            Array2::<f32>::zeros((seq_len, hidden)),
            Array2::<f32>::zeros((seq_len, intermediate)),
        );
    }

    // Fall back to dense when most features are selected
    if k * 5 >= intermediate * 4 && overrides.is_empty() {
        return dense_ffn_forward(weights, layer, x);
    }

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // Gather weight rows for selected features
    let up_buf = gather_rows(w_up, features, hidden);
    let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();

    let _gate_buf;
    let gate_sub = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        _gate_buf = gather_rows(w_gate, features, hidden);
        Some(ndarray::ArrayView2::from_shape((k, hidden), &_gate_buf).unwrap())
    } else {
        _gate_buf = Vec::new();
        None
    };

    // Gather down-projection columns: w_down[:, features] → [hidden, K]
    let down_sub = gather_columns(w_down, features, hidden);
    let down_view = ndarray::ArrayView2::from_shape((hidden, k), &down_sub).unwrap();

    // Override lookup (only built when overrides are present)
    let override_map: std::collections::HashMap<usize, &[f32]> = if overrides.is_empty() {
        std::collections::HashMap::new()
    } else {
        overrides.iter().copied().collect()
    };

    let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
    let mut sparse_act = vec![0.0f32; k];
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        // Compute sparse activations
        if let Some(ref gate_sub) = gate_sub {
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);
            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                let activated = if use_gelu {
                    gelu_tanh(g)
                } else {
                    g * sigmoid(g)
                };
                let val = activated * up_proj[i];
                sparse_act[i] = val;
                full_activation[[s, feat]] = val;
            }
        } else {
            let up_proj = up_sub.dot(&x_row);
            let mut vals = up_proj.to_vec();
            if let Some(bias) = arch
                .ffn_up_bias_key(layer)
                .and_then(|bk| weights.vectors.get(&bk))
            {
                for (i, &feat) in features.iter().enumerate() {
                    if feat < bias.len() {
                        vals[i] += bias[feat];
                    }
                }
            }
            for (i, &feat) in features.iter().enumerate() {
                let v = vals[i];
                let val = if use_gelu {
                    gelu_tanh(v)
                } else {
                    v * sigmoid(v)
                };
                sparse_act[i] = val;
                full_activation[[s, feat]] = val;
            }
        }

        // Sparse down projection: w_down[:, features] @ sparse_act
        let act_view = ndarray::ArrayView1::from(&sparse_act[..k]);
        let out_vec = down_view.dot(&act_view);
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row)
            .and(&out_vec)
            .for_each(|o, &v| *o = v);

        // Apply overrides: swap standard down contribution with custom vector
        if !override_map.is_empty() {
            for (local_i, &feat) in features.iter().enumerate() {
                if let Some(override_down) = override_map.get(&feat) {
                    let activation = sparse_act[local_i];
                    if activation.abs() > 1e-8 && override_down.len() == hidden {
                        for j in 0..hidden {
                            out_row[j] -= down_view[[j, local_i]] * activation;
                            out_row[j] += override_down[j] * activation;
                        }
                    }
                }
            }
        }
    }

    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    (out, full_activation)
}

fn sparse_ffn_forward_full_impl(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[FeatureSlotOverride<'_>],
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
    let hidden = x.shape()[1];
    let intermediate = w_up.shape()[0];
    let seq_len = x.shape()[0];
    let k = features.len();

    if k == 0 {
        return (
            Array2::<f32>::zeros((seq_len, hidden)),
            Array2::<f32>::zeros((seq_len, intermediate)),
        );
    }

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // Gather original weight rows. Per-feature overrides will be
    // applied below by re-computing the dot products for the
    // overridden slots only — the unchanged slots use the gathered
    // values from the dense weights.
    let up_buf = gather_rows(w_up, features, hidden);
    let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();

    let _gate_buf;
    let gate_sub = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        _gate_buf = gather_rows(w_gate, features, hidden);
        Some(ndarray::ArrayView2::from_shape((k, hidden), &_gate_buf).unwrap())
    } else {
        _gate_buf = Vec::new();
        None
    };

    let down_sub = gather_columns(w_down, features, hidden);
    let down_view = ndarray::ArrayView2::from_shape((hidden, k), &down_sub).unwrap();

    // Per-feature override lookup. Built once.
    let override_map: std::collections::HashMap<usize, &FeatureSlotOverride<'_>> =
        overrides.iter().map(|o| (o.feature, o)).collect();

    let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
    let mut sparse_act = vec![0.0f32; k];
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        // Phase 1: compute baseline activations for every feature
        // using gathered dense weights.
        if let Some(ref gate_sub) = gate_sub {
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);
            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                let activated = if use_gelu {
                    gelu_tanh(g)
                } else {
                    g * sigmoid(g)
                };
                let val = activated * up_proj[i];
                sparse_act[i] = val;
                full_activation[[s, feat]] = val;
            }
        } else {
            let up_proj = up_sub.dot(&x_row);
            let mut vals = up_proj.to_vec();
            if let Some(bias) = arch
                .ffn_up_bias_key(layer)
                .and_then(|bk| weights.vectors.get(&bk))
            {
                for (i, &feat) in features.iter().enumerate() {
                    if feat < bias.len() {
                        vals[i] += bias[feat];
                    }
                }
            }
            for (i, &feat) in features.iter().enumerate() {
                let v = vals[i];
                let val = if use_gelu {
                    gelu_tanh(v)
                } else {
                    v * sigmoid(v)
                };
                sparse_act[i] = val;
                full_activation[[s, feat]] = val;
            }
        }

        // Phase 2: re-compute activation for overridden slots using
        // the overlay's gate / up vectors. The slot's contribution to
        // the residual is `silu(gate_override · x) * (up_override · x)`
        // — exactly the install_compiled_slot Python semantics.
        for (i, &feat) in features.iter().enumerate() {
            let Some(ov) = override_map.get(&feat) else {
                continue;
            };
            // Only recompute if at least one of gate / up is overridden.
            if ov.gate.is_none() && ov.up.is_none() {
                continue;
            }
            // Gate dot product (override or original gathered row).
            let g = if let Some(g_ov) = ov.gate {
                if g_ov.len() == hidden {
                    g_ov.iter()
                        .zip(x_row.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                } else {
                    // Length mismatch — fall through to original.
                    if let Some(ref gate_sub) = gate_sub {
                        gate_sub.row(i).dot(&x_row)
                    } else {
                        0.0
                    }
                }
            } else if let Some(ref gate_sub) = gate_sub {
                gate_sub.row(i).dot(&x_row)
            } else {
                0.0
            };
            let activated = if use_gelu {
                gelu_tanh(g)
            } else {
                g * sigmoid(g)
            };

            // Up dot product (override or original).
            let up_score = if let Some(u_ov) = ov.up {
                if u_ov.len() == hidden {
                    u_ov.iter()
                        .zip(x_row.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                } else {
                    up_sub.row(i).dot(&x_row)
                }
            } else {
                up_sub.row(i).dot(&x_row)
            };

            let new_act = if is_gated {
                activated * up_score
            } else {
                activated
            };
            sparse_act[i] = new_act;
            full_activation[[s, feat]] = new_act;
        }

        // Phase 3: down projection using gathered dense down vectors,
        // then swap in down overrides for any overridden slots.
        let act_view = ndarray::ArrayView1::from(&sparse_act[..k]);
        let out_vec = down_view.dot(&act_view);
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row)
            .and(&out_vec)
            .for_each(|o, &v| *o = v);

        for (i, &feat) in features.iter().enumerate() {
            let Some(ov) = override_map.get(&feat) else {
                continue;
            };
            let Some(d_ov) = ov.down else {
                continue;
            };
            if d_ov.len() != hidden {
                continue;
            }
            let activation = sparse_act[i];
            if activation.abs() <= 1e-8 {
                continue;
            }
            // Subtract the dense column contribution and add the override.
            for j in 0..hidden {
                out_row[j] -= down_view[[j, i]] * activation;
                out_row[j] += d_ov[j] * activation;
            }
        }
    }

    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    (out, full_activation)
}

/// Gather rows from a weight matrix for selected features.
/// Input: w is [num_features, hidden], output: [K, hidden] contiguous.
fn gather_rows(
    w: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    features: &[usize],
    hidden: usize,
) -> Vec<f32> {
    let k = features.len();
    let raw = w.as_slice().unwrap();
    let mut buf = vec![0.0f32; k * hidden];
    for (i, &feat) in features.iter().enumerate() {
        let src = feat * hidden;
        buf[i * hidden..(i + 1) * hidden].copy_from_slice(&raw[src..src + hidden]);
    }
    buf
}

/// Gather columns from a weight matrix for selected features.
/// Input: w is [hidden, intermediate] (row-major), output: [hidden, K] contiguous.
fn gather_columns(
    w: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    features: &[usize],
    hidden: usize,
) -> Vec<f32> {
    let k = features.len();
    let cols = w.shape()[1];
    let raw = w.as_slice().unwrap();
    let mut buf = vec![0.0f32; hidden * k];
    for row in 0..hidden {
        let row_start = row * cols;
        for (j, &feat) in features.iter().enumerate() {
            buf[row * k + j] = raw[row_start + feat];
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::make_test_weights;
    use ndarray::Array2;

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Array2::from_shape_vec((seq, hidden), data).unwrap()
    }

    // ── sparse_ffn_forward ────────────────────────────────────────────────────

    #[test]
    fn sparse_forward_empty_features_returns_zeros() {
        let weights = make_test_weights();
        let x = input(2, weights.hidden_size);
        let (out, act) = sparse_ffn_forward(&weights, 0, &x, &[]);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(
            out.iter().all(|v| v.abs() < 1e-9),
            "empty features → zero output"
        );
        assert_eq!(act.shape()[0], 2);
    }

    #[test]
    fn sparse_forward_single_feature_output_shape() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let (out, act) = sparse_ffn_forward(&weights, 0, &x, &[0]);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert_eq!(act.shape()[0], 1);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_forward_multi_token_shape() {
        let weights = make_test_weights();
        let x = input(3, weights.hidden_size);
        let (out, act) = sparse_ffn_forward(&weights, 0, &x, &[0, 1, 2]);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert_eq!(act.shape()[0], 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_forward_top_k_selection_is_sorted() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let x_row = x.row(0);
        let feats = select_top_k_features(&weights, 0, &x_row, 4);
        // select_top_k_features sorts by feature index (ascending)
        for w in feats.windows(2) {
            assert!(w[0] <= w[1], "features not sorted: {:?}", feats);
        }
    }

    #[test]
    fn sparse_forward_top_k_respects_k() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let x_row = x.row(0);
        for k in [1, 4, 8] {
            let feats = select_top_k_features(&weights, 0, &x_row, k);
            assert!(
                feats.len() <= k,
                "got {} features but requested {k}",
                feats.len()
            );
        }
    }

    #[test]
    fn sparse_forward_all_features_matches_dense_fallback() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        // When K >= 80% of intermediate, sparse_ffn_forward falls back to dense.
        // Request all features to trigger that path.
        let all: Vec<usize> = (0..weights.intermediate_size).collect();
        let (sparse_out, _) = sparse_ffn_forward(&weights, 0, &x, &all);
        let (dense_out, _) = crate::ffn::weight::dense_ffn_forward(&weights, 0, &x);
        for (s, d) in sparse_out.iter().zip(dense_out.iter()) {
            assert!((s - d).abs() < 1e-4, "sparse/dense mismatch: {s} vs {d}");
        }
    }

    // ── sparse_ffn_forward_with_overrides ─────────────────────────────────────

    #[test]
    fn overrides_replace_down_contribution() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let feats = &[0usize];
        let custom_down = vec![99.0f32; weights.hidden_size];
        let (out_override, _) =
            sparse_ffn_forward_with_overrides(&weights, 0, &x, feats, &[(0, &custom_down)]);
        let (out_baseline, _) = sparse_ffn_forward(&weights, 0, &x, feats);
        // The two outputs should differ because the down vector was replaced.
        let diff: f32 = out_override
            .iter()
            .zip(out_baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "override had no effect on output");
    }

    // ── gather_rows / gather_columns (indirectly) ─────────────────────────────

    #[test]
    fn gather_rows_all_features_produces_correct_shape() {
        // Test via sparse_ffn_forward by requesting two specific features
        let weights = make_test_weights();
        let x = input(2, weights.hidden_size);
        let (out, _) = sparse_ffn_forward(&weights, 0, &x, &[0, weights.intermediate_size - 1]);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
    }
}

/// Select top-K features by gate activation magnitude (architecture-correct).
pub fn select_top_k_features(
    weights: &ModelWeights,
    layer: usize,
    x_row: &ndarray::ArrayView1<f32>,
    top_k: usize,
) -> Vec<usize> {
    let arch = &*weights.arch;
    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    let proj = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        w_gate.dot(x_row)
    } else {
        let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let mut p = w_up.dot(x_row);
        if let Some(bias) = arch
            .ffn_up_bias_key(layer)
            .and_then(|bk| weights.vectors.get(&bk))
        {
            for i in 0..p.len().min(bias.len()) {
                p[i] += bias[i];
            }
        }
        p
    };

    let intermediate = proj.len();
    let k = top_k.min(intermediate);

    let mut indexed: Vec<(usize, f32)> = proj
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| {
            let act = if use_gelu {
                gelu_tanh(v)
            } else {
                v * sigmoid(v)
            };
            (i, act)
        })
        .collect();

    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);
    }
    indexed.sort_unstable_by_key(|x| x.0);
    indexed.into_iter().map(|(id, _)| id).collect()
}
