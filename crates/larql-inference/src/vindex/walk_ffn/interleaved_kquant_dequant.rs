//! Q4K dequant walk — dequantises gate/up/down from `interleaved_kquant.bin`
//! for the given layer, then runs the standard dense GEGLU forward.
//!
//! Used by the INFER pipeline on Q4K vindexes without a GPU backend.
//! Peak memory is one layer's worth of dequantised f32 matrices;
//! cheap on 4B (120 MB), tight on 31B (1.8 GB).

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_kquant_dequant(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let ffn = self.index.interleaved_kquant_layer_data(layer)?;
        // Stream layer N+1 in while we dequant N — same trick the Q4_0
        // path uses. No-op when `layer + 1` is out of range.
        self.index.prefetch_interleaved_kquant_layer(layer + 1);
        let arch = &*self.weights.arch;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let hidden = x.shape()[1];

        let dequant = |bytes: &[u8], fmt: &str, rows: usize, cols: usize| -> Array2<f32> {
            let padded = rows * cols;
            let info = larql_vindex::quant::registry::lookup(fmt)
                .unwrap_or_else(|| panic!("unknown quant format: {fmt}"));
            let flat =
                (info.dequantize)(bytes, padded).unwrap_or_else(|e| panic!("{fmt} dequant: {e}"));
            Array2::from_shape_vec((rows, cols), flat[..rows * cols].to_vec())
                .expect("dequant shape mismatch")
        };

        let w_gate = dequant(ffn[0].0, ffn[0].1, intermediate, hidden);
        let w_up = dequant(ffn[1].0, ffn[1].1, intermediate, hidden);
        let w_down = dequant(ffn[2].0, ffn[2].1, hidden, intermediate);

        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );
        let gate = crate::forward::dot_proj(x, &w_gate);
        let up = crate::forward::dot_proj(x, &w_up);
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate, &up)
        } else {
            crate::ffn::silu_gate_up(&gate, &up)
        };
        let out = crate::forward::dot_proj(&activation, &w_down);
        self.trace_path(layer, "interleaved_kquant:dequant");
        Some((out, activation))
    }
}

#[cfg(test)]
mod tests {
    //! `walk_ffn_kquant_dequant` is `pub(super)`, so tests live in the
    //! sibling `walk_ffn/` modules. Coverage here drives:
    //!
    //! - the Gelu activation branch (via the Gemma3 Q4K fixture);
    //! - the Silu activation branch (via the TinyModel Q4K fixture).
    //!
    //! The `intermediate == 0` early-return (line 25 in the original
    //! file) is unreachable with the existing `make_test_q4k_vindex`
    //! shape — the manifest always produces a positive feature count.
    //! Leaving it uncovered is the conscious trade.
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_q4k_weights_silu,
    };
    use crate::vindex::WalkFfn;
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    #[test]
    fn dequant_path_runs_for_gelu_tanh_arch() {
        // Gemma3 → GeluTanh activation → `gelu_tanh_gate_up` branch.
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = WalkFfn::new_unlimited(&weights, &index);
        // Call directly through the pub(super) method so the routing
        // ladder's priority-4 fallthrough (preferred by `forward()`)
        // doesn't matter — we want to hit the dequant body explicitly.
        let (out, activation) = ffn
            .walk_ffn_kquant_dequant(0, &x(1, weights.hidden_size))
            .expect("dequant path should produce output on Q4K fixture");
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert_eq!(activation.shape(), &[1, weights.intermediate_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dequant_path_runs_for_silu_arch() {
        // TinyModel → Silu activation → `silu_gate_up` branch (line 52
        // in the original module).
        let weights = make_test_q4k_weights_silu();
        let index = make_test_q4k_vindex(&weights);
        let ffn = WalkFfn::new_unlimited(&weights, &index);
        let (out, activation) = ffn
            .walk_ffn_kquant_dequant(0, &x(1, weights.hidden_size))
            .expect("dequant path should produce output on Silu Q4K fixture");
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert_eq!(activation.shape(), &[1, weights.intermediate_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
