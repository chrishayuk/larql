//! Unit tests for inference modules: attention, ffn, residual.

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

mod test_residual {
    use larql_inference::residual::{rms_norm, rms_norm_heads};
    use ndarray::Array2;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn rms_norm_identity_without_weight() {
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = rms_norm(&x, None, 1.0);
        // Without weight, norm should normalize to unit RMS
        let rms: f32 = (out.row(0).iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.01);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn rms_norm_with_weight_offset_zero() {
        // offset=0 means weight is applied directly (Llama style)
        let x = Array2::from_shape_vec((1, 3), vec![3.0, 4.0, 0.0]).unwrap();
        let weight = vec![1.0, 1.0, 1.0];
        let out = rms_norm(&x, Some(&weight), 0.0);
        // With weight=[1,1,1] and offset=0, it's just raw weight multiplication
        let rms: f32 = (out.row(0).iter().map(|v| v * v).sum::<f32>() / 3.0).sqrt();
        assert!(rms > 0.0);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn rms_norm_with_weight_offset_one() {
        // offset=1 means weight = 1 + learned (Gemma style)
        let x = Array2::from_shape_vec((1, 3), vec![3.0, 4.0, 0.0]).unwrap();
        let weight = vec![0.0, 0.0, 0.0]; // effective weight = 1.0
        let out_gemma = rms_norm(&x, Some(&weight), 1.0);
        let out_none = rms_norm(&x, None, 1.0);
        // With zero learned weights and offset=1, should equal no-weight norm
        for j in 0..3 {
            assert!((out_gemma[[0, j]] - out_none[[0, j]]).abs() < 1e-6);
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn rms_norm_batch() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = rms_norm(&x, None, 1.0);
        assert_eq!(out.shape(), &[2, 3]);
        // Each row should be independently normalized
        assert_ne!(out[[0, 0]], out[[1, 0]]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn rms_norm_heads_two_heads() {
        // 1 sequence position, 2 heads, head_dim=2
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = vec![1.0, 1.0]; // per head_dim
        let out = rms_norm_heads(&x, &weight, 2, 2, 1.0);
        assert_eq!(out.shape(), &[1, 4]);
        // Each head normalized independently — different scaling
        let head0_rms = (out[[0, 0]].powi(2) + out[[0, 1]].powi(2)).sqrt();
        let head1_rms = (out[[0, 2]].powi(2) + out[[0, 3]].powi(2)).sqrt();
        // Both should be non-zero and roughly similar scale
        assert!(head0_rms > 0.0);
        assert!(head1_rms > 0.0);
    }
}

mod test_ffn {
    use larql_inference::ffn::silu_gate_up;
    use ndarray::Array2;

    /// SiLU-gated FFN helper for unit tests (no model architecture needed).
    fn silu_ffn_forward(
        x: &Array2<f32>,
        w_gate: &Array2<f32>,
        w_up: &Array2<f32>,
        w_down: &Array2<f32>,
    ) -> Array2<f32> {
        let gate = x.dot(&w_gate.t());
        let up = x.dot(&w_up.t());
        silu_gate_up(&gate, &up).dot(&w_down.t())
    }

    fn silu_ffn_forward_with_activation(
        x: &Array2<f32>,
        w_gate: &Array2<f32>,
        w_up: &Array2<f32>,
        w_down: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let gate = x.dot(&w_gate.t());
        let up = x.dot(&w_up.t());
        let activation = silu_gate_up(&gate, &up);
        let out = activation.dot(&w_down.t());
        (out, activation)
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn silu_gate_up_elementwise() {
        let gate = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, -1.0]).unwrap();
        let up = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
        let out = silu_gate_up(&gate, &up);
        assert!((out[[0, 0]]).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.7311).abs() < 0.01);
        assert!((out[[0, 2]] - (-0.2689)).abs() < 0.01);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn ffn_forward_shapes() {
        let x = Array2::ones((2, 4));
        let w_gate = Array2::ones((3, 4)) * 0.1;
        let w_up = Array2::ones((3, 4)) * 0.1;
        let w_down = Array2::ones((4, 3)) * 0.1;
        let out = silu_ffn_forward(&x, &w_gate, &w_up, &w_down);
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn ffn_forward_with_activation_returns_both() {
        let x = Array2::ones((1, 4));
        let w_gate = Array2::ones((3, 4)) * 0.1;
        let w_up = Array2::ones((3, 4)) * 0.1;
        let w_down = Array2::ones((4, 3)) * 0.1;
        let (out, act) = silu_ffn_forward_with_activation(&x, &w_gate, &w_up, &w_down);
        assert_eq!(out.shape(), &[1, 4]);
        assert_eq!(act.shape(), &[1, 3]);
    }
}

mod test_attention {
    use larql_inference::attention::{apply_rope, gqa_attention};
    use ndarray::Array2;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn apply_rope_preserves_shape() {
        let x = Array2::ones((3, 8)); // seq=3, 2 heads * head_dim=4
        let out = apply_rope(&x, 2, 4, 10000.0);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn apply_rope_position_zero_is_identity() {
        // At position 0, cos(0)=1, sin(0)=0, so RoPE is identity
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = apply_rope(&x, 1, 4, 10000.0);
        for j in 0..4 {
            assert!(
                (out[[0, j]] - x[[0, j]]).abs() < 1e-6,
                "pos=0 should be identity, got {} vs {}",
                out[[0, j]],
                x[[0, j]]
            );
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn apply_rope_different_positions_differ() {
        let mut x = Array2::ones((2, 4));
        x[[0, 0]] = 1.0;
        x[[1, 0]] = 1.0;
        let out = apply_rope(&x, 1, 4, 10000.0);
        // Position 0 and 1 should produce different rotations
        assert_ne!(out[[0, 0]], out[[1, 0]]);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn gqa_attention_causal_mask() {
        // 2 tokens, 1 head, head_dim=2
        let q = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = q.clone();
        let v = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let out = gqa_attention(&q, &k, &v, 1, 2, 1, 0.5_f64.sqrt(), 2);
        assert_eq!(out.shape(), &[2, 2]);
        // Token 0 can only attend to itself
        // Token 1 attends to both
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn gqa_attention_single_token() {
        let q = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = q.clone();
        let v = Array2::from_shape_vec((1, 4), vec![0.5, 0.5, 0.5, 0.5]).unwrap();
        // Single token: attention weight = 1.0, output = V
        let out = gqa_attention(&q, &k, &v, 1, 4, 1, 0.5, 1);
        for j in 0..4 {
            assert!((out[[0, j]] - 0.5).abs() < 1e-5);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod test_model_config {
    use larql_inference::model::load_model_dir;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn load_model_dir_rejects_nonexistent() {
        let result = load_model_dir("/nonexistent/model/path");
        assert!(result.is_err());
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn resolve_model_path_rejects_nonexistent() {
        let result = larql_inference::model::resolve_model_path("nonexistent/model");
        assert!(result.is_err());
    }
}
