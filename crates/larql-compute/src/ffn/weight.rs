//! Dense FFN backend — full matrix multiply, architecture-correct.
//! This is the ground truth: identical to model inference.

use crate::{dot_proj_gpu, ComputeBackend};
use ndarray::Array2;

use super::{gelu_tanh, gelu_tanh_gate_up, sigmoid, silu_gate_up, FfnActivations, FfnBackend};
use crate::forward::add_bias;
use larql_models::{ModelWeights, WeightsView};

/// Amortised quant matmul for formats with a direct CPU kernel (Q4_K / Q6_K):
/// `x[seq, cols] -> [seq, rows]`, reading the quantised bytes once and never
/// materialising the full f32 weight. Returns `None` for a format without a
/// kernel so the caller can dequantise + matmul. Shared by the q4k-direct FFN
/// and the attention Q/K/V/O projections.
pub(crate) fn quant_matmul(
    bytes: &[u8],
    fmt: &str,
    x: &[f32],
    rows: usize,
    cols: usize,
    seq: usize,
) -> Option<Array2<f32>> {
    let kernel = crate::QuantFormat::from_registry_tag(fmt).and_then(|f| f.route().quant_matmul)?;
    let mut out = vec![0.0f32; seq * rows];
    kernel(&mut out, x, bytes, rows, cols, seq);
    Some(Array2::from_shape_vec((seq, rows), out).expect("quant_matmul output shape [seq, rows]"))
}

/// Project `x[seq, in_dim] -> [seq, out_rows]` from quantised weight bytes;
/// `in_dim` must be a 256-multiple. Q4_K/Q6_K read the bytes once via the
/// amortised kernel; any other format dequantises then matmuls. Used for the
/// FFN gate/up and the attention Q/K/V/O projections.
pub(crate) fn quant_proj(
    bytes: &[u8],
    fmt: &str,
    x: &Array2<f32>,
    out_rows: usize,
    in_dim: usize,
    seq: usize,
) -> Array2<f32> {
    if let Some(r) = quant_matmul(
        bytes,
        fmt,
        x.as_slice().expect("contiguous x"),
        out_rows,
        in_dim,
        seq,
    ) {
        r
    } else {
        let w = crate::kquant_forward::dequant::dequantize_matrix(bytes, fmt, out_rows, in_dim);
        dot_proj_gpu(x, &w, None)
    }
}

/// Dense FFN: follows the model architecture exactly (CPU BLAS).
/// Gated: activation(x @ gate.T) * (x @ up.T) @ down.T + bias
/// Non-gated: activation(x @ up.T + bias) @ down.T + bias
pub struct WeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

impl<'a> FfnBackend for WeightFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward(WeightsView::dense(self.weights), layer, x).0
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        let (out, act) = dense_ffn_forward(WeightsView::dense(self.weights), layer, x);
        (out, FfnActivations::Dense(act))
    }

    fn name(&self) -> &str {
        "weights"
    }
}

/// FFN backend over a [`WeightsView`] — the quant forward's scratch-aware FFN.
/// Identical math to [`WeightFfn`], but resolves gate/up/down through the view
/// (engine scratch first, then canonical), so the per-layer dequantised FFN
/// tensors are visible without mutating `ModelWeights`. The quant forward loops
/// construct this with a `with_scratch` view; everything else keeps `WeightFfn`.
pub struct ViewFfn<'a> {
    pub view: WeightsView<'a>,
}

impl FfnBackend for ViewFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        dense_ffn_forward(self.view, layer, x).0
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        let (out, act) = dense_ffn_forward(self.view, layer, x);
        (out, FfnActivations::Dense(act))
    }

    fn name(&self) -> &str {
        "view"
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
        dense_ffn_forward_backend(
            WeightsView::dense(self.weights),
            layer,
            x,
            Some(self.backend),
        )
        .0
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        let (out, act) = dense_ffn_forward_backend(
            WeightsView::dense(self.weights),
            layer,
            x,
            Some(self.backend),
        );
        (out, FfnActivations::Dense(act))
    }

    fn name(&self) -> &str {
        "weights+backend"
    }
}

/// FFN backend that runs gate/up/down on the vindex's quantised weight bytes
/// **without a per-layer full dequant**. Each component is dispatched on its
/// stored format tag (from [`KvIndex::interleaved_kquant_layer_data`]):
/// `Q4_K` and `Q6_K` (the default `down_proj`) each go through their amortised
/// matmul kernel ([`q4k_matmul_into`](crate::cpu::ops::q4_common::q4k_matmul_into)
/// / [`q6k_matmul_into`](crate::cpu::ops::q4_common::q6k_matmul_into)) — reads
/// the quantised bytes once, no f32 materialisation. Only a format without a
/// direct kernel falls back to dequantise + matmul. The FFN weights are ~4× the
/// attention weights, so avoiding their f32 staging is the bulk of the prefill
/// saving.
///
/// The surrounding gating/activation/bias math mirrors [`dense_ffn_forward`];
/// only the projection source differs. Callers confirm the interleaved FFN
/// bytes are present and `hidden` is a 256-multiple before constructing this.
pub struct Q4kMatmulFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn crate::KvIndex,
}

impl Q4kMatmulFfn<'_> {
    /// Bytes per 256-element super-block for a quant format with a
    /// direct matmul kernel (the formats [`quant_matmul`] serves).
    #[inline]
    fn block_bytes(fmt: &str) -> usize {
        crate::QuantFormat::from_registry_tag(fmt)
            .filter(|f| f.route().quant_matmul.is_some())
            .and_then(|f| f.packed_block_layout())
            .map(|(_, block_bytes)| block_bytes)
            .unwrap_or_else(|| panic!("Q4kMatmulFfn: unsupported FFN quant format {fmt}"))
    }

    /// gate/up projection: `x[seq, in_dim] -> [seq, out_rows]`, where `in_dim`
    /// (= `hidden`) is a 256-multiple. Q4_K/Q6_K read the bytes once via the
    /// amortised kernel; any other format dequantises then matmuls.
    fn project(
        bytes: &[u8],
        fmt: &str,
        x: &Array2<f32>,
        out_rows: usize,
        in_dim: usize,
        seq: usize,
    ) -> Array2<f32> {
        quant_proj(bytes, fmt, x, out_rows, in_dim, seq)
    }

    /// down projection: `act[seq, intermediate] -> [seq, hidden]`. The stored
    /// down weight pads `intermediate` up to a 256-multiple. Q4_K/Q6_K zero-pad
    /// the activation to the stored width and use the amortised kernel; any
    /// other format dequantises the padded weight and slices to `intermediate`
    /// before the f32 matmul.
    fn project_down(
        bytes: &[u8],
        fmt: &str,
        act: &Array2<f32>,
        hidden: usize,
        intermediate: usize,
        seq: usize,
    ) -> Array2<f32> {
        let inter_padded = bytes.len() / hidden / Self::block_bytes(fmt) * 256;
        let act_slice = act.as_slice().expect("contiguous activation");

        if inter_padded == intermediate {
            if let Some(r) = quant_matmul(bytes, fmt, act_slice, hidden, inter_padded, seq) {
                return r;
            }
            let w =
                crate::kquant_forward::dequant::dequantize_matrix(bytes, fmt, hidden, intermediate);
            return dot_proj_gpu(act, &w, None);
        }

        // `intermediate` isn't a 256-multiple: the stored weight is padded, so
        // zero-pad the activation columns to match before projecting.
        let mut padded = vec![0.0f32; seq * inter_padded];
        for s in 0..seq {
            padded[s * inter_padded..s * inter_padded + intermediate]
                .copy_from_slice(&act_slice[s * intermediate..(s + 1) * intermediate]);
        }
        if let Some(r) = quant_matmul(bytes, fmt, &padded, hidden, inter_padded, seq) {
            return r;
        }
        let w_full =
            crate::kquant_forward::dequant::dequantize_matrix(bytes, fmt, hidden, inter_padded);
        let w_down = w_full.slice(ndarray::s![.., ..intermediate]).to_owned();
        dot_proj_gpu(act, &w_down, None)
    }
}

impl Q4kMatmulFfn<'_> {
    /// Shared forward body: the pre-down activation is an intrinsic
    /// intermediate here (the down projection consumes it), so both
    /// trait entry points route through this at identical cost.
    fn forward_full(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let seq = x.nrows();
        let hidden = x.ncols();
        let intermediate = self.index.num_features(layer);
        let ffn = self
            .index
            .interleaved_kquant_layer_data(layer)
            .expect("Q4kMatmulFfn requires interleaved FFN bytes for this layer");
        let (gate_bytes, gate_fmt) = ffn[0];
        let (up_bytes, up_fmt) = ffn[1];
        let (down_bytes, down_fmt) = ffn[2];

        let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
            let gate = Self::project(gate_bytes, gate_fmt, x, intermediate, hidden, seq);
            let up = Self::project(up_bytes, up_fmt, x, intermediate, hidden, seq);
            if arch.gate_up_is_gelu_tanh() {
                gelu_tanh_gate_up(&gate, &up)
            } else {
                silu_gate_up(&gate, &up)
            }
        } else {
            let mut projected = Self::project(up_bytes, up_fmt, x, intermediate, hidden, seq);
            if let Some(bias) = arch
                .ffn_up_bias_key(layer)
                .and_then(|k| self.weights.vectors.get(&k))
            {
                add_bias(&mut projected, bias);
            }
            if arch.gate_up_is_gelu_tanh() {
                projected.mapv(gelu_tanh)
            } else {
                projected.mapv(|v| v * sigmoid(v))
            }
        };

        let mut out =
            Self::project_down(down_bytes, down_fmt, &activation, hidden, intermediate, seq);
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            add_bias(&mut out, bias);
        }

        (out, activation)
    }
}

impl FfnBackend for Q4kMatmulFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_full(layer, x).0
    }

    fn forward_observed(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, FfnActivations) {
        let (out, act) = self.forward_full(layer, x);
        (out, FfnActivations::Dense(act))
    }

    fn name(&self) -> &str {
        "q4k-matmul"
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

    // `forward_observed` keeps the trait default (Absent): a pass-through
    // stub computes no activations, and it must not fabricate any.

    fn name(&self) -> &str {
        "null"
    }
}

/// Architecture-correct dense FFN — CPU BLAS path.
pub fn dense_ffn_forward(
    weights: WeightsView,
    layer: usize,
    x: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    dense_ffn_forward_backend(weights, layer, x, None)
}

/// Architecture-correct dense FFN with optional backend dispatch.
/// `backend = None` → plain ndarray BLAS (same as `dense_ffn_forward`).
/// `backend = Some(be)` → gate/up/down matmuls through `be.matmul_transb`.
///
/// Resolves FFN weights through [`WeightsView::tensor`] (engine scratch first,
/// then canonical) so the quant forward's dequantised FFN tensors are visible
/// without mutating `ModelWeights`. Dense callers pass a `dense()` view.
pub fn dense_ffn_forward_backend(
    weights: WeightsView,
    layer: usize,
    x: &Array2<f32>,
    backend: Option<&dyn ComputeBackend>,
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let compact_hint = "FFN weight tensor missing — this is a `--compact` \
        vindex. Use `WalkFfn` instead of `WeightFfn` for inference \
        (or re-extract without `--compact` if you need dense matmul).";

    let w_up = weights
        .tensor(&arch.ffn_up_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_up_key(layer)));
    let w_down = weights
        .tensor(&arch.ffn_down_key(layer))
        .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_down_key(layer)));

    let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
        let w_gate = weights
            .tensor(&arch.ffn_gate_key(layer))
            .unwrap_or_else(|| panic!("{compact_hint} (key: {})", arch.ffn_gate_key(layer)));
        let gate = dot_proj_gpu(x, w_gate, backend);
        let up = dot_proj_gpu(x, w_up, backend);
        if arch.gate_up_is_gelu_tanh() {
            gelu_tanh_gate_up(&gate, &up)
        } else {
            silu_gate_up(&gate, &up)
        }
    } else {
        let mut projected = dot_proj_gpu(x, w_up, backend);
        if let Some(bias) = arch
            .ffn_up_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
        {
            add_bias(&mut projected, bias);
        }
        if arch.gate_up_is_gelu_tanh() {
            projected.mapv(gelu_tanh)
        } else {
            projected.mapv(|v| v * sigmoid(v))
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
    use larql_models::test_fixtures::make_test_weights;
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
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert_eq!(act.shape(), &[3, weights.intermediate_size]);
    }

    #[test]
    fn dense_ffn_forward_output_finite() {
        let weights = make_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
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
        let (out1, act1) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        let (out2, act2) = dense_ffn_forward_backend(WeightsView::dense(&weights), 0, &input, None);
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
            let (out, _) = dense_ffn_forward(WeightsView::dense(&weights), layer, &input);
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
        use super::FfnBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        assert_eq!(ffn.name(), "weights");
        let input = x(2, weights.hidden_size);
        let out = ffn.forward(0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn backend_ffn_matches_weight_ffn() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let wffn = WeightFfn { weights: &weights };
        let bffn = BackendFfn {
            weights: &weights,
            backend: &crate::CpuBackend,
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
    fn weight_ffn_forward_observed_reports_dense_activation() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = x(3, weights.hidden_size);
        let (out, obs) = ffn.forward_observed(0, &input);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        let act = obs.into_dense().expect("dense path observes densely");
        assert_eq!(act.shape(), &[3, weights.intermediate_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(act.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn backend_ffn_forward_observed_reports_dense_activation() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let ffn = BackendFfn {
            weights: &weights,
            backend: &crate::CpuBackend,
        };
        let input = x(2, weights.hidden_size);
        let (out, obs) = ffn.forward_observed(0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        let act = obs.into_dense().expect("dense path observes densely");
        assert_eq!(act.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn view_ffn_forward_and_observed_match_weight_ffn() {
        use super::FfnBackend;
        let weights = make_test_weights();
        let view = ViewFfn {
            view: WeightsView::dense(&weights),
        };
        let dense = WeightFfn { weights: &weights };
        let input = x(2, weights.hidden_size);
        assert_eq!(view.name(), "view");
        assert_eq!(view.forward(0, &input), dense.forward(0, &input));
        let (out_v, obs_v) = view.forward_observed(0, &input);
        let (out_d, obs_d) = dense.forward_observed(0, &input);
        assert_eq!(out_v, out_d, "a dense view resolves the same tensors");
        assert_eq!(obs_v, obs_d, "and observes the same dense activation");
    }

    #[test]
    fn null_ffn_forward_observed_is_absent_not_fabricated() {
        use super::FfnBackend;
        let input = x(2, 4);
        let (out, obs) = NullFfn.forward_observed(0, &input);
        assert_eq!(out, input, "NullFfn passes the residual through");
        assert!(
            obs.is_absent(),
            "a pass-through stub must not fabricate activations"
        );
        assert_eq!(NullFfn.name(), "null");
    }

    #[test]
    fn backend_ffn_name_is_weights_plus_backend() {
        let weights = make_test_weights();
        let ffn = BackendFfn {
            weights: &weights,
            backend: &crate::CpuBackend,
        };
        assert_eq!(ffn.name(), "weights+backend");
    }

    #[test]
    fn dense_ffn_forward_single_token_shape() {
        // Edge case: one row at the smallest meaningful seq_len.
        let weights = make_test_weights();
        let input = x(1, weights.hidden_size);
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
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
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
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
        let (out_none, act_none) =
            dense_ffn_forward_backend(WeightsView::dense(&weights), 0, &input, None);
        let (out_some, act_some) = dense_ffn_forward_backend(
            WeightsView::dense(&weights),
            0,
            &input,
            Some(&crate::CpuBackend),
        );
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
        let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, act) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
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
        let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
        let input = x(1, weights.hidden_size);
        let (out, _) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Gemma3-arch: GeluTanh activation in gated FFN ──────────────────

    #[test]
    fn dense_ffn_forward_gemma3_runs_gelu_tanh_gate_up_branch() {
        // Gemma3 has activation = GeluTanh, exercising the
        // `gelu_tanh_gate_up` branch instead of the default silu.
        let weights = larql_models::test_fixtures::make_gemma3_test_weights();
        let input = x(2, weights.hidden_size);
        let (out, _) = dense_ffn_forward(WeightsView::dense(&weights), 0, &input);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── q4k-direct FFN parity ──────────────────────────────────────────

    /// `Q4kMatmulFfn` must match dequantising the SAME Q4_K bytes and
    /// running the dense FFN — both decode identical weights, so they agree
    /// within fp summation noise. This is the prefill correctness contract:
    /// swapping the FFN to q4k-direct must not change the output.
    #[test]
    fn q4k_matmul_ffn_matches_dequant_dense() {
        use super::Q4kMatmulFfn;
        use crate::test_fixtures::make_q4k_fixture_index;
        use larql_models::test_fixtures::make_test_q4k_weights;

        let weights = make_test_q4k_weights();
        let index = make_q4k_fixture_index(&weights);
        let input = x(3, weights.hidden_size);

        // Reference: dequant the layer's Q4_K FFN bytes into scratch, then
        // run the dense FFN against those f32 tensors (the current path).
        let mut scratch = larql_models::DequantScratch::new();
        crate::kquant_forward::insert_q4k_layer_tensors(&mut scratch, &weights, &index, 0)
            .expect("dequant layer 0");
        let (ref_out, ref_act) =
            dense_ffn_forward(WeightsView::with_scratch(&weights, &scratch), 0, &input);

        // q4k-direct: same bytes, no dequant.
        let ffn = Q4kMatmulFfn {
            weights: &weights,
            index: &index,
        };
        assert_eq!(ffn.name(), "q4k-matmul");
        let (got_out, got_obs) = ffn.forward_observed(0, &input);
        let got_act = got_obs
            .into_dense()
            .expect("q4k-direct is a dense path — observation must be Dense");
        // `forward` shares `forward_full` — identical output, no observation.
        assert_eq!(ffn.forward(0, &input), got_out);

        let max_out: f32 = ref_out
            .iter()
            .zip(&got_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let max_act: f32 = ref_act
            .iter()
            .zip(&got_act)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_out < 5e-3,
            "q4k FFN output diverged: max_diff={max_out}"
        );
        assert!(
            max_act < 5e-3,
            "q4k FFN activation diverged: max_diff={max_act}"
        );
        assert_eq!(got_out.shape(), &[3, weights.hidden_size]);
    }

    /// Regression for the Q6_K `down_proj` mis-decode: the *default* q4k vindex
    /// stores FFN down at Q6_K (210 B/block), not Q4_K (144 B/block). The down
    /// projection must dispatch on the format tag — decoding Q6_K bytes through
    /// `q4k_matmul` produces garbage and a wrong derived column count. Compare
    /// the Q6_K `project_down` against dequantise+matmul on the same bytes.
    #[test]
    fn project_down_dispatches_on_q6k_format() {
        use super::Q4kMatmulFfn;
        use crate::cpu::ops::q4_common::quantize_q6_k;

        let hidden = 256;
        let intermediate = 512; // 2 super-blocks; 256-multiple (matches real gemma)
        let seq = 3;

        let down: Vec<f32> = (0..hidden * intermediate)
            .map(|i| (i as f32 * 0.0007).sin() * 0.6)
            .collect();
        let down_q6 = quantize_q6_k(&down);
        // Bounded activation: the amortised kernel sums per-super-block while the
        // BLAS reference sums the whole row, so an unbounded/growing activation
        // amplifies fp-reorder noise via cancellation. This is a correctness
        // check (same decoded weights), so keep the inputs well-conditioned.
        let act = Array2::from_shape_fn((seq, intermediate), |(s, j)| {
            (((s * 13 + j) % 97) as f32 / 97.0 - 0.5) * 0.8
        });

        // Reference: dequantise the Q6_K weight to f32, then act @ down.T.
        let w = crate::kquant_forward::dequant::dequantize_matrix(
            &down_q6,
            "Q6_K",
            hidden,
            intermediate,
        );
        let reference = crate::dot_proj_gpu(&act, &w, None);

        let got = Q4kMatmulFfn::project_down(&down_q6, "Q6_K", &act, hidden, intermediate, seq);
        assert_eq!(got.shape(), &[seq, hidden]);

        let max: f32 = reference
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max < 5e-3,
            "Q6_K project_down diverged from dequant+matmul: {max}"
        );
    }
}
