//! Full mmap walk — gate + up + down from three separate mmap files.
//! Zero safetensor reads. Three BLAS gemms over mmap'd matrices.
//!
//! Used by vindexes that have `up_features.bin` and `down_features.bin`
//! but not the interleaved layout. Same FLOP count as dense; the only
//! win is that all weight reads come from the vindex so the safetensors
//! can be unloaded after extraction.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_full_mmap(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let gate_scores = self.index.gate_scores_batch(layer, x)?;
        let up_view = self.index.up_layer_matrix(layer)?;
        let down_view = self.index.down_layer_matrix(layer)?;

        let arch = &*self.weights.arch;
        let use_gelu = arch.gate_up_is_gelu_tanh();

        let up_scores = larql_compute::dot_proj_gpu(x, &up_view, self.backend);

        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
        } else {
            crate::ffn::silu_gate_up(&gate_scores, &up_scores)
        };

        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, "full_mmap");
        Some((out, activation))
    }
}

#[cfg(test)]
mod tests {
    //! First tests for this path (2026-07-30 review, roadmap item 20).
    //! With the vindex gates set to the MODEL's gate tensors
    //! (`make_model_gate_vindex`) and feature-major up/down from the
    //! same weights, the three mmap gemms compute exactly the dense FFN
    //! — parity vs `WeightFfn` is the correctness bar.

    use crate::ffn::{FfnBackend, WeightFfn};
    use crate::test_utils::{
        attach_feature_major_f32_to_test_vindex, make_model_gate_vindex,
        make_starcoder2_test_weights, make_test_vindex, make_test_weights,
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

    /// Full forward through the ladder lands on `full_mmap` (step 7 —
    /// `up_features.bin` + `down_features.bin`, no interleaved) and
    /// matches the dense `WeightFfn` — output AND activation.
    #[test]
    fn full_mmap_matches_dense_weight_ffn() {
        let weights = make_test_weights();
        let mut index = make_model_gate_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        assert!(index.has_full_mmap_ffn());
        let walk = WalkFfn::new_unlimited(&weights, &index).with_dispatch_trace();
        let dense = WeightFfn { weights: &weights };
        let input = x(2, weights.hidden_size);

        for layer in 0..weights.num_layers {
            let (out_w, obs_w) = walk.forward_observed(layer, &input);
            let act_w = obs_w
                .into_dense()
                .expect("dense walk path observes densely");
            let (out_d, obs_d) = dense.forward_observed(layer, &input);
            let act_d = obs_d.into_dense().expect("WeightFfn observes densely");
            for (w, d) in act_w.iter().zip(act_d.iter()) {
                assert!(
                    (w - d).abs() <= 1e-5 * d.abs().max(1.0),
                    "layer {layer}: activation {w} vs dense {d}"
                );
            }
            for (w, d) in out_w.iter().zip(out_d.iter()) {
                assert!(
                    (w - d).abs() <= 1e-5 * d.abs().max(1.0),
                    "layer {layer}: output {w} vs dense {d}"
                );
            }
            assert!(out_d.iter().any(|v| v.abs() > 0.0), "non-trivial reference");
        }
        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), weights.num_layers);
        assert!(
            trace.iter().all(|e| e.path == "full_mmap"),
            "up+down feature-major storage must route via step 7, got {trace:?}"
        );
    }

    /// Without the up/down feature-major mmaps the path declines so the
    /// ladder can fall through.
    #[test]
    fn full_mmap_declines_without_native_up_down() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let walk = WalkFfn::new_unlimited(&weights, &index);
        assert!(walk
            .walk_ffn_full_mmap(0, &x(1, weights.hidden_size))
            .is_none());
    }

    /// The down-bias arm: StarCoder2 carries `ffn_down_bias`, so the
    /// output must equal `activation · downᵀ + bias` — reconstructed
    /// here from the returned activation and the source tensors.
    #[test]
    fn full_mmap_applies_down_bias_from_arch() {
        let weights = make_starcoder2_test_weights();
        let mut index = make_model_gate_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let walk = WalkFfn::new_unlimited(&weights, &index);
        let input = x(1, weights.hidden_size);
        let (out, activation) = walk
            .walk_ffn_full_mmap(0, &input)
            .expect("full-mmap walk runs on the starcoder2 fixture");

        let down = &weights.tensors[&weights.arch.ffn_down_key(0)]; // [hidden, inter]
        let bias = &weights.vectors[&weights
            .arch
            .ffn_down_bias_key(0)
            .expect("starcoder2 has a down bias")];
        assert!(
            bias.iter().any(|b| *b != 0.0),
            "fixture bias must be non-zero"
        );
        let expected = activation.dot(&down.t());
        for h in 0..weights.hidden_size {
            let e = expected[[0, h]] + bias[h];
            assert!(
                (out[[0, h]] - e).abs() <= 1e-5 * e.abs().max(1.0),
                "hidden {h}: {} vs activation·downᵀ+bias {e}",
                out[[0, h]]
            );
        }
    }
}
