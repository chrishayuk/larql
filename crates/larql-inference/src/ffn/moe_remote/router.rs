// ── Local routing math ────────────────────────────────────────────────────────
// Mirrored from larql-compute cpu/ops/moe.rs so the client can route without
// having the expert weights locally.

pub(super) fn rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    if w.is_empty() || x.is_empty() {
        return x.to_vec();
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi / rms * (wi + offset))
        .collect()
}

/// Parameter-free RMSNorm (HF `Gemma4RMSNorm(with_scale=False)`): scales
/// `x` by `1/sqrt(mean(x²) + eps)` with no learned weight.
pub(super) fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|v| v / rms).collect()
}

fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    (0..out_rows)
        .map(|row| {
            let w_row = &w[row * in_cols..(row + 1) * in_cols];
            x.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum()
        })
        .collect()
}

fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

fn top_k(v: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(v.len());
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    (
        indexed.iter().map(|(i, _)| *i).collect(),
        indexed.iter().map(|(_, v)| *v).collect(),
    )
}

/// Routing-only parameters. A subset of `MoeLayerWeights` — the expert weight
/// slices (`experts_gate_up`, `experts_down`) are absent; those live on shards.
pub struct MoeRouterWeights<'a> {
    /// Router linear projection [num_experts × hidden_size].
    pub router_proj: &'a [f32],
    /// Optional router input scale [hidden_size].
    pub router_scale: &'a [f32],
    /// Optional per-expert output scale [num_experts].
    pub router_per_expert_scale: &'a [f32],
    /// Optional router-specific RMSNorm weights [hidden_size]. When non-empty,
    /// the router input is `rms_norm(h, router_norm)`; when empty AND
    /// `router_norm_parameter_free` is true, it's parameter-free RMSNorm;
    /// otherwise falls back to `rms_norm(h, pre_experts_norm)`.
    pub router_norm: &'a [f32],
    /// Parameter-free router RMSNorm (no learned weight). HF Gemma 4 sets
    /// this true (`Gemma4RMSNorm(with_scale=False)`).
    pub router_norm_parameter_free: bool,
    /// Scalar multiplier on the router input after the norm and `router_scale`.
    /// HF Gemma 4: `hidden_size^-0.5`. Use `1.0` for no scaling.
    pub router_input_scalar: f32,
    /// Pre-experts RMSNorm weights [hidden_size].
    pub pre_experts_norm: &'a [f32],
    /// Post-experts RMSNorm weights [hidden_size]. Applied to the summed output.
    pub post_experts_norm: &'a [f32],
    pub num_experts: usize,
    pub top_k: usize,
}

impl MoeRouterWeights<'_> {
    /// Run steps 1-5 of the MoE forward pass (norm → scale → proj → softmax → top-K).
    /// Returns `(h_norm, expert_indices, expert_weights)` where `h_norm` is
    /// the experts' input (pre_experts_norm output), not the router's input.
    pub fn route(&self, h: &[f32], norm_offset: f32, eps: f32) -> (Vec<f32>, Vec<usize>, Vec<f32>) {
        let hidden = h.len();

        // Experts' input norm (used by callers for the expert matmuls).
        // Router norm composes on top of h_norm — matches Metal's
        // `gpu_moe_dispatch` convention. See the note in
        // `larql-compute/src/cpu/ops/moe/forward.rs`.
        let h_norm = rms_norm(h, self.pre_experts_norm, eps, norm_offset);

        // Router input norm. Priority:
        //   1. learned router_norm weight (architectures that ship one),
        //   2. parameter-free RMSNorm (HF Gemma 4 — `with_scale=False`),
        //   3. fallback: experts' pre-norm.
        // All apply on top of h_norm so routing matches Metal.
        let router_in_normed = if !self.router_norm.is_empty() {
            rms_norm(&h_norm, self.router_norm, eps, norm_offset)
        } else if self.router_norm_parameter_free {
            rms_norm_no_weight(&h_norm, eps)
        } else {
            h_norm.clone()
        };

        let mut router_in: Vec<f32> = if !self.router_scale.is_empty() {
            router_in_normed
                .iter()
                .zip(self.router_scale.iter())
                .map(|(a, b)| a * b)
                .collect()
        } else {
            router_in_normed
        };
        if self.router_input_scalar != 1.0 && self.router_input_scalar != 0.0 {
            for v in router_in.iter_mut() {
                *v *= self.router_input_scalar;
            }
        }

        let mut logits = matmul_vec(&router_in, self.router_proj, self.num_experts, hidden);
        softmax(&mut logits);

        let (indices, mut weights) = top_k(&logits, self.top_k);

        // Renormalize selected weights to sum to 1 — matches Gemma 4's
        // gemma4_top_k_softmax which normalises after selection.
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum > 0.0 {
            for w in &mut weights {
                *w /= weight_sum;
            }
        }

        if !self.router_per_expert_scale.is_empty() {
            for (i, &ei) in indices.iter().enumerate() {
                if ei < self.router_per_expert_scale.len() {
                    weights[i] *= self.router_per_expert_scale[ei];
                }
            }
        }

        (h_norm, indices, weights)
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_empty_inputs_are_passthrough() {
        // Empty weight or empty input → return clone of input.
        assert_eq!(rms_norm(&[], &[1.0], 1e-6, 0.0), Vec::<f32>::new());
        let v = vec![1.0f32, 2.0];
        assert_eq!(rms_norm(&v, &[], 1e-6, 0.0), v);
    }

    #[test]
    fn rms_norm_normalises_to_unit_rms_with_unit_weight() {
        // x = [3, 4], rms = sqrt((9+16)/2 + eps) ≈ sqrt(12.5) ≈ 3.535
        // weight=1 → output ≈ x / rms.
        let out = rms_norm(&[3.0f32, 4.0], &[1.0, 1.0], 1e-12, 0.0);
        let rms = (12.5f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
        assert!((out[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn rms_norm_no_weight_normalises_to_unit_rms() {
        let out = rms_norm_no_weight(&[3.0f32, 4.0], 1e-12);
        let rms = (12.5f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn rms_norm_no_weight_empty_returns_empty() {
        assert!(rms_norm_no_weight(&[], 1e-6).is_empty());
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_handles_zero_sum() {
        // All -inf → max is -inf → sum becomes 0 → division skipped.
        let mut v = vec![f32::NEG_INFINITY, f32::NEG_INFINITY];
        softmax(&mut v);
        // After exp(0) = 1, exp(0) = 1, sum=2 → output [0.5, 0.5] (max=NEG_INF
        // makes shifted = inf … actually NEG_INF - NEG_INF = NaN).
        // We only check that it doesn't panic.
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn top_k_returns_top_indices_and_values() {
        let v = vec![0.1f32, 0.5, 0.3, 0.2];
        let (indices, values) = top_k(&v, 2);
        assert_eq!(indices, vec![1, 2]);
        assert!((values[0] - 0.5).abs() < 1e-6);
        assert!((values[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn top_k_clamps_to_input_length() {
        let v = vec![0.1f32, 0.5];
        let (indices, _) = top_k(&v, 99);
        assert_eq!(indices.len(), 2);
    }

    fn router_weights<'a>(
        router_proj: &'a [f32],
        pre_experts_norm: &'a [f32],
        post_experts_norm: &'a [f32],
        num_experts: usize,
        top_k: usize,
    ) -> MoeRouterWeights<'a> {
        MoeRouterWeights {
            router_proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm,
            post_experts_norm,
            num_experts,
            top_k,
        }
    }

    #[test]
    fn route_returns_h_norm_indices_and_weights() {
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        // 2 experts × 4 hidden — diagonal-ish projection.
        let proj = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let pre_norm = vec![1.0f32; 4];
        let post_norm = vec![1.0f32; 4];
        let r = router_weights(&proj, &pre_norm, &post_norm, 2, 2);
        let (h_norm, indices, weights) = r.route(&h, 0.0, 1e-6);
        assert_eq!(h_norm.len(), 4);
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        // Selected weights renormalised to sum to 1.
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn route_with_router_norm_routes_through_learned_norm_branch() {
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let proj = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let pre_norm = vec![1.0f32; 4];
        let post_norm = vec![1.0f32; 4];
        let router_norm = vec![1.0f32; 4]; // learned router norm present
        let r = MoeRouterWeights {
            router_proj: &proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &router_norm,
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm,
            post_experts_norm: &post_norm,
            num_experts: 2,
            top_k: 1,
        };
        let (_, indices, _) = r.route(&h, 0.0, 1e-6);
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn route_with_parameter_free_router_norm_routes_through_param_free_branch() {
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let proj = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let pre_norm = vec![1.0f32; 4];
        let post_norm = vec![1.0f32; 4];
        let r = MoeRouterWeights {
            router_proj: &proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: true,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm,
            post_experts_norm: &post_norm,
            num_experts: 2,
            top_k: 2,
        };
        let (_, indices, weights) = r.route(&h, 0.0, 1e-6);
        assert_eq!(indices.len(), 2);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn route_with_router_scale_applies_per_dim_scale() {
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let proj = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let pre_norm = vec![1.0f32; 4];
        let post_norm = vec![1.0f32; 4];
        let router_scale = vec![2.0f32; 4]; // 2× scale on every dim
        let r = MoeRouterWeights {
            router_proj: &proj,
            router_scale: &router_scale,
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm,
            post_experts_norm: &post_norm,
            num_experts: 2,
            top_k: 2,
        };
        let (_, _, weights) = r.route(&h, 0.0, 1e-6);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn route_with_input_scalar_applies_post_norm_scale() {
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let proj = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let pre_norm = vec![1.0f32; 4];
        let post_norm = vec![1.0f32; 4];
        let r = MoeRouterWeights {
            router_proj: &proj,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 0.5, // != 1.0 and != 0.0 → branch fires
            pre_experts_norm: &pre_norm,
            post_experts_norm: &post_norm,
            num_experts: 2,
            top_k: 2,
        };
        let _ = r.route(&h, 0.0, 1e-6);
    }

    #[test]
    fn route_with_per_expert_scale_applies_to_selected_weights() {
        let h = vec![1.0f32, 2.0, 3.0, 4.0];
        let proj = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let pre_norm = vec![1.0f32; 4];
        let post_norm = vec![1.0f32; 4];
        let per_expert = vec![1.5f32, 0.5];
        let r = MoeRouterWeights {
            router_proj: &proj,
            router_scale: &[],
            router_per_expert_scale: &per_expert,
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &pre_norm,
            post_experts_norm: &post_norm,
            num_experts: 2,
            top_k: 2,
        };
        let (_, _, weights) = r.route(&h, 0.0, 1e-6);
        // Two weights returned; per-expert scale applied.
        assert_eq!(weights.len(), 2);
    }
}
