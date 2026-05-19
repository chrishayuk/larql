//! Dense FFN backend — full matrix multiply, architecture-correct.
//! This is the ground truth: identical to model inference.

use larql_compute::{dot_proj_gpu, ComputeBackend};
use ndarray::Array2;

use super::{gelu_tanh, gelu_tanh_gate_up, sigmoid, silu_gate_up, FfnBackend};
use crate::forward::add_bias;
use crate::model::ModelWeights;

/// Dense FFN: follows the model architecture exactly (CPU BLAS).
/// Gated: activation(x @ gate.T) * (x @ up.T) @ down.T + bias
/// Non-gated: activation(x @ up.T + bias) @ down.T + bias
pub struct WeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

impl<'a> FfnBackend for WeightFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward(self.weights, layer, x).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward(self.weights, layer, x)
    }

    fn name(&self) -> &str {
        "weights"
    }
}

/// Backend-dispatched dense FFN. Matmuls route through `ComputeBackend` when
/// `backend` is `Some` — useful for prefill on Metal where gate/up/down
/// projections are the dominant cost.
pub struct BackendFfn<'a, 'b> {
    pub weights: &'a ModelWeights,
    pub backend: &'b dyn ComputeBackend,
}

impl<'a, 'b> FfnBackend for BackendFfn<'a, 'b> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward_backend(self.weights, layer, x, Some(self.backend)).0
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        dense_ffn_forward_backend(self.weights, layer, x, Some(self.backend))
    }

    fn name(&self) -> &str {
        "weights+backend"
    }
}

/// Stub FFN that returns the input unchanged. Used by callers that must
/// satisfy the `KvEngine::{prefill,decode_step}` signature but know the
/// engine they're calling doesn't consult an FFN router (e.g. engines
/// that recompute FFN internally from `weights`). Cheap to construct;
/// holds no references.
pub struct NullFfn;

impl FfnBackend for NullFfn {
    fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
        x.clone()
    }

    fn forward_with_activation(
        &self,
        _layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        (x.clone(), x.clone())
    }

    fn name(&self) -> &str {
        "null"
    }
}

/// Architecture-correct dense FFN — CPU BLAS path.
pub fn dense_ffn_forward(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    dense_ffn_forward_backend(weights, layer, x, None)
}

/// Architecture-correct dense FFN with optional backend dispatch.
/// `backend = None` → plain ndarray BLAS (same as `dense_ffn_forward`).
/// `backend = Some(be)` → gate/up/down matmuls through `be.matmul_transb`.
pub fn dense_ffn_forward_backend(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    backend: Option<&dyn ComputeBackend>,
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let compact_hint = "FFN weight tensor missing — this is a `--compact` \
        vindex. Use `WalkFfn` instead of `WeightFfn` for inference \
        (or re-extract without `--compact` if you need dense matmul).";

    let w_up = weights
        .tensors
        .get(&arch.ffn_up_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_up_key(layer)));
    let w_down = weights
        .tensors
        .get(&arch.ffn_down_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_down_key(layer)));

    let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
        let w_gate = weights
            .tensors
            .get(&arch.ffn_gate_key(layer))
            .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_gate_key(layer)));
        let gate = dot_proj_gpu(x, w_gate, backend);
        let up = dot_proj_gpu(x, w_up, backend);
        match arch.activation() {
            larql_models::Activation::GeluTanh => gelu_tanh_gate_up(&gate, &up),
            _ => silu_gate_up(&gate, &up),
        }
    } else {
        let mut projected = dot_proj_gpu(x, w_up, backend);
        if let Some(bias) = arch
            .ffn_up_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
        {
            add_bias(&mut projected, bias);
        }
        match arch.activation() {
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => {
                projected.mapv(gelu_tanh)
            }
            _ => projected.mapv(|v| v * sigmoid(v)),
        }
    };

    let mut out = dot_proj_gpu(&activation, w_down, backend);
    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    (out, activation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    fn x(rows: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (rows, hidden),
            (0..rows * hidden)
                .map(|i| (i as f32 + 1.0) * 0.05)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn dense_ffn_forward_shape() {
        let weights = make_test_weights();
        let input = x(3, weights.hidden_size);
        let (out, act) = dense_ffn_forward(&weights, 0, &input);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert_eq!(act.shape(), &[3, weights.intermediate_size]);
    }

    #[test]
    fn dense_ffn_forward_output_finite() {
        let weights = make_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, act) = dense_ffn_forward(&weights, 0, &input);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "FFN output has non-finite values"
        );
        assert!(
            act.iter().all(|v| v.is_finite()),
            "FFN activation has non-finite values"
        );
    }

    #[test]
    fn dense_ffn_forward_backend_matches_no_backend() {
        // backend=None should produce the same result as dense_ffn_forward
        let weights = make_test_weights();
        let input = x(2, weights.hidden_size);
        let (out1, act1) = dense_ffn_forward(&weights, 0, &input);
        let (out2, act2) = dense_ffn_forward_backend(&weights, 0, &input, None);
        assert_eq!(
            out1, out2,
            "output should match between dense_ffn_forward and backend(None)"
        );
        assert_eq!(act1, act2, "activation should match");
    }

    #[test]
    fn dense_ffn_forward_all_layers() {
        let weights = make_test_weights();
        let input = x(1, weights.hidden_size);
        for layer in 0..weights.num_layers {
            let (out, _) = dense_ffn_forward(&weights, layer, &input);
            assert_eq!(
                out.shape(),
                &[1, weights.hidden_size],
                "layer {layer} wrong shape"
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "layer {layer} non-finite"
            );
        }
    }

    #[test]
    fn weight_ffn_implements_ffn_backend() {
        use crate::ffn::FfnBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        assert_eq!(ffn.name(), "weights");
        let input = x(2, weights.hidden_size);
        let out = ffn.forward(0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn backend_ffn_matches_weight_ffn() {
        use crate::ffn::FfnBackend;
        let weights = make_test_weights();
        let wffn = WeightFfn { weights: &weights };
        let bffn = BackendFfn {
            weights: &weights,
            backend: &larql_compute::CpuBackend,
        };
        let input = x(2, weights.hidden_size);
        let out_w = wffn.forward(0, &input);
        let out_b = bffn.forward(0, &input);
        for (w, b) in out_w.iter().zip(out_b.iter()) {
            assert!(
                (w - b).abs() < 1e-4,
                "WeightFfn and BackendFfn differ: {w} vs {b}"
            );
        }
    }

    #[test]
    fn weight_ffn_forward_with_activation_returns_both_arrays() {
        use crate::ffn::FfnBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = x(3, weights.hidden_size);
        let (out, act) = ffn.forward_with_activation(0, &input);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert_eq!(act.shape(), &[3, weights.intermediate_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(act.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn backend_ffn_forward_with_activation_returns_both_arrays() {
        use crate::ffn::FfnBackend;
        let weights = make_test_weights();
        let ffn = BackendFfn {
            weights: &weights,
            backend: &larql_compute::CpuBackend,
        };
        let input = x(2, weights.hidden_size);
        let (out, act) = ffn.forward_with_activation(0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert_eq!(act.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn backend_ffn_name_is_weights_plus_backend() {
        let weights = make_test_weights();
        let ffn = BackendFfn {
            weights: &weights,
            backend: &larql_compute::CpuBackend,
        };
        assert_eq!(ffn.name(), "weights+backend");
    }

    #[test]
    fn dense_ffn_forward_single_token_shape() {
        // Edge case: one row at the smallest meaningful seq_len.
        let weights = make_test_weights();
        let input = x(1, weights.hidden_size);
        let (out, act) = dense_ffn_forward(&weights, 0, &input);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert_eq!(act.shape(), &[1, weights.intermediate_size]);
    }

    #[test]
    fn dense_ffn_zero_input_produces_finite_output() {
        // Activation at x=0 is well-defined (silu(0) = 0); output must be
        // finite — pins against any future NaN-introducing activation
        // change to the gated path.
        let weights = make_test_weights();
        let input = Array2::<f32>::zeros((2, weights.hidden_size));
        let (out, act) = dense_ffn_forward(&weights, 0, &input);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(act.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dense_ffn_forward_backend_with_some_matches_no_backend() {
        // backend=Some(CpuBackend) and backend=None route through
        // different `dot_proj_gpu` branches but must produce identical
        // output (within float noise).
        let weights = make_test_weights();
        let input = x(2, weights.hidden_size);
        let (out_none, act_none) = dense_ffn_forward_backend(&weights, 0, &input, None);
        let (out_some, act_some) =
            dense_ffn_forward_backend(&weights, 0, &input, Some(&larql_compute::CpuBackend));
        for (a, b) in out_none.iter().zip(out_some.iter()) {
            assert!((a - b).abs() < 1e-4, "out diverged: {a} vs {b}");
        }
        for (a, b) in act_none.iter().zip(act_some.iter()) {
            assert!((a - b).abs() < 1e-4, "act diverged: {a} vs {b}");
        }
    }

    // ── Starcoder2-arch: non-gated FFN + biases ────────────────────────

    #[test]
    fn dense_ffn_forward_starcoder2_runs_non_gated_branch() {
        // Starcoder2 has ffn_type = NonGated, so dense_ffn_forward takes
        // the `else` branch (no gate matrix; just up + activation + down).
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, act) = dense_ffn_forward(&weights, 0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        // Non-gated activation has shape (seq, intermediate).
        assert_eq!(act.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn dense_ffn_forward_starcoder2_bias_paths_fire() {
        // Starcoder2 returns Some from ffn_up_bias_key + ffn_down_bias_key,
        // so the `add_bias(&mut projected, bias)` and `add_bias(&mut out,
        // bias)` calls fire.
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let input = x(1, weights.hidden_size);
        let (out, _) = dense_ffn_forward(&weights, 0, &input);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Gemma3-arch: GeluTanh activation in gated FFN ──────────────────

    #[test]
    fn dense_ffn_forward_gemma3_runs_gelu_tanh_gate_up_branch() {
        // Gemma3 has activation = GeluTanh, exercising the
        // `gelu_tanh_gate_up` branch instead of the default silu.
        let weights = crate::test_utils::make_gemma3_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, _) = dense_ffn_forward(&weights, 0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
