//! Full MoE block forward pass: router → top-k → weighted sum of expert outputs.
//!
//! Flow is controlled by `MoeLayerWeights::routing_policy` and
//! `weight_layout`, so Gemma-style hybrid MoE choices are metadata rather
//! than hidden branches in the hot path.

use crate::MoeLayerWeights;

use super::cache::try_cached_dequant;
use super::expert::{run_single_expert_q4k_q8k_into, ExpertScratch};
use super::math::{gelu_tanh, matmul_vec, silu, softmax};
use super::{
    moe_expert_input, moe_post_expert_output, moe_route_from_router_input, moe_router_input,
};
use crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use crate::options;

/// Run the MoE expert block for one token.
///
/// `h` — residual stream at this layer (hidden_size f32 values).
/// Returns the expert block contribution to add to the dense FFN output.
/// If `moe` is missing required fields, returns a zero vector of hidden_size.
pub fn cpu_moe_forward(
    h: &[f32],
    moe: &MoeLayerWeights<'_>,
    norm_offset: f32,
    eps: f32,
) -> Vec<f32> {
    // Per-stage timing for bottleneck diagnosis.  Enable with
    // `LARQL_MOE_FWD_TIMING=1`.  Cached in TLS to avoid syscalls
    // per call on the hot path.
    thread_local! {
        static FWD_TIMING: bool = options::env_flag(options::ENV_MOE_FWD_TIMING);
    }
    let timing = FWD_TIMING.with(|t| *t);
    let t_start = std::time::Instant::now();

    let hidden = h.len();
    let num_experts = moe.num_experts;
    let top_k_val = moe.top_k;
    let inter = moe.intermediate_size;

    if num_experts == 0 || top_k_val == 0 || inter == 0 {
        return vec![0.0f32; hidden];
    }
    if moe.router_proj.is_empty() || moe.experts_gate_up.is_empty() || moe.experts_down.is_empty() {
        return vec![0.0f32; hidden];
    }
    // Diagnostic: bypass the expert block entirely. Dense FFN alone flows
    // through the normal path; if this produces legible output, the MoE
    // block is the broken piece. If still garbage, look upstream.
    if options::skip_moe_enabled() {
        return vec![0.0f32; hidden];
    }

    let expert_input = moe_expert_input(h, moe, norm_offset, eps);
    let router_in = moe_router_input(h, &expert_input, moe, norm_offset, eps);
    let (expert_indices, expert_weights) = moe_route_from_router_input(&router_in, moe);
    let debug_logits = if options::moe_debug_enabled() {
        let mut logits = matmul_vec(&router_in, moe.router_proj, num_experts, hidden);
        softmax(&mut logits);
        Some(logits)
    } else {
        None
    };

    // Debug: print routing per layer if MOE_DEBUG=1
    static DEBUG_LAYER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if let Some(logits) = debug_logits.as_ref() {
        let layer_n = DEBUG_LAYER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 30;
        let h_rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32).sqrt();
        let hn_rms =
            (expert_input.iter().map(|v| v * v).sum::<f32>() / expert_input.len() as f32).sqrt();
        let ri_rms =
            (router_in.iter().map(|v| v * v).sum::<f32>() / router_in.len().max(1) as f32).sqrt();
        let logit_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logit_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let pnorm_rms = (moe.pre_experts_norm.iter().map(|v| v * v).sum::<f32>()
            / moe.pre_experts_norm.len().max(1) as f32)
            .sqrt();
        let rnorm_rms = (moe.router_norm.iter().map(|v| v * v).sum::<f32>()
            / moe.router_norm.len().max(1) as f32)
            .sqrt();
        let rscale_rms = (moe.router_scale.iter().map(|v| v * v).sum::<f32>()
            / moe.router_scale.len().max(1) as f32)
            .sqrt();
        eprintln!("[L{layer_n:02}] h_rms={h_rms:.2} hn_rms={hn_rms:.2} router_in_rms={ri_rms:.2} | pnorm_rms={pnorm_rms:.2} rnorm_rms={rnorm_rms:.2} rscale_rms={rscale_rms:.2} scalar={:.4} | logits [{logit_min:.3}..{logit_max:.3}] | experts:{expert_indices:?}", moe.router_input_scalar);
    }

    // Run each selected expert's gated FFN (BF16 dequant on demand).
    //    Experts are independent — their only shared input is `expert_input` and
    //    their outputs are summed. Parallelise across the top-K experts with
    //    rayon so BLAS-accelerated gemv on each core overlaps. `moe.activation`
    //    is a plain enum (Copy), and `cached_dequant` hands out shared
    //    Arc<Vec<f32>> values that are Sync, so the closure is Send+Sync.
    //
    //    gate_up layout: [num_experts, 2*inter, hidden]  (gate rows first, then up rows)
    //    down layout:    [num_experts, hidden, inter]
    let activation = moe.activation;
    let format = moe.expert_data_format;
    // Storage layout per Gemma 4 26B-A4B (and the per-layer Q4_K writer):
    //   gate_up: [2*inter, hidden]              — never padded; quantises
    //                                             cleanly because hidden is
    //                                             already a 256-multiple.
    //   down:    [hidden, inter_padded]         — Q4_K pads `inter` up to
    //                                             the next 256 super-block
    //                                             (704 → 768). BF16 stores
    //                                             un-padded.
    // Mirror Metal's `inter_padded` handling (`metal/moe_dispatch.rs`):
    // dequant down at the padded width, zero-pad the hidden_state so
    // the matmul reads `inter_padded` columns with the padding
    // contributing zero.
    let inter_padded = moe.inter_padded();

    let t_pre_par = t_start.elapsed();

    // Q4_K direct-from-mmap path: quantise expert_input to Q8_K once per layer
    // (shared across all K active experts) and use the SDOT-based integer
    // matvec.  Bypasses the f32 dequant cache entirely — at Gemma 4 26B-A4B
    // sizes the f32 cache is 5.7 GB walked per token and DRAM-bandwidth
    // bound; direct-Q4K is ~1.4 GB.  Set `LARQL_DISABLE_Q4K_DIRECT=1` to
    // fall back to the BLAS-on-cached-f32 path for kernel-debug A/B runs.
    let q4k_direct = matches!(format, crate::QuantFormat::Q4_K)
        && hidden.is_multiple_of(256)
        && !super::q4k_direct_disabled();
    let t_q8k_quant_start = std::time::Instant::now();
    let expert_input_q8k = q4k_direct.then(|| quantize_x_to_q8k(&expert_input));
    let t_q8k_quant = t_q8k_quant_start.elapsed();
    let t_par_start = std::time::Instant::now();

    // Per-rayon-thread scratch buffers (gate_out / up_out / act / act_q8k /
    // out).  Allocated lazily on first hit, reused across all subsequent
    // expert calls on the same worker.  Replaces the prior pattern of
    // `vec![0; ...]` allocs per expert call (5 distinct heap allocs per
    // call × K=8 × 30 layers = 1200 allocs/token, with occasional 150 µs
    // spikes from the allocator's slow path that drag par_iter wall up).
    thread_local! {
        static SCRATCH: std::cell::RefCell<Option<ExpertScratch>> =
            const { std::cell::RefCell::new(None) };
    }

    use rayon::prelude::*;
    let expert_out = expert_indices
        .par_iter()
        .zip(expert_weights.par_iter())
        .filter(|(_, &w)| w != 0.0)
        .fold(
            || vec![0.0f32; hidden],
            |mut acc, (&ei, &w)| {
                let Some(&gate_up_bytes) = moe.experts_gate_up.get(ei) else {
                    return acc;
                };
                let Some(&down_bytes) = moe.experts_down.get(ei) else {
                    return acc;
                };

                SCRATCH.with(|cell| {
                    let mut borrow = cell.borrow_mut();
                    let scratch = borrow
                        .get_or_insert_with(|| ExpertScratch::new(hidden, inter, inter_padded));
                    if scratch.gate_out.len() != inter
                        || scratch.act.len() != inter_padded
                        || scratch.out.len() != hidden
                    {
                        *scratch = ExpertScratch::new(hidden, inter, inter_padded);
                    }

                    if let Some(q8k) = expert_input_q8k.as_ref() {
                        // Q4_K direct path — single source of truth in
                        // `expert::run_single_expert_q4k_q8k_into`.  Reuses
                        // the scratch's act_q8k buffer too.
                        let h2 = run_single_expert_q4k_q8k_into(
                            scratch,
                            q8k,
                            gate_up_bytes,
                            down_bytes,
                            inter,
                            activation,
                        );
                        for (a, &v) in acc.iter_mut().zip(h2.iter()) {
                            *a += w * v;
                        }
                        return;
                    }

                    // Fallback: BF16 / F32 / Q4_K-with-disable — original
                    // f32 cache path.  Inlined here to avoid pulling the
                    // per-call rms_norm / format dispatch from the legacy
                    // `run_single_expert_into` that doesn't share scratch.
                    let gate_up_w = try_cached_dequant(gate_up_bytes, format, 2 * inter * hidden)
                        .unwrap_or_else(|err| panic!("{err}"));
                    if gate_up_w.is_empty() {
                        return;
                    }
                    let gate_w = &gate_up_w[..inter * hidden];
                    let up_w = &gate_up_w[inter * hidden..2 * inter * hidden];

                    let gate_out = matmul_vec(&expert_input, gate_w, inter, hidden);
                    let up_out = matmul_vec(&expert_input, up_w, inter, hidden);

                    for j in 0..inter {
                        let g = gate_out[j];
                        let u = up_out[j];
                        scratch.act[j] = match activation {
                            crate::Activation::GeluTanh => gelu_tanh(g) * u,
                            _ => silu(g) * u,
                        };
                    }

                    let down_w = try_cached_dequant(down_bytes, format, hidden * inter_padded)
                        .unwrap_or_else(|err| panic!("{err}"));
                    if down_w.is_empty() {
                        return;
                    }
                    let expert_contribution =
                        matmul_vec(&scratch.act, &down_w, hidden, inter_padded);
                    for (a, &v) in acc.iter_mut().zip(expert_contribution.iter()) {
                        *a += w * v;
                    }
                });
                acc
            },
        )
        .reduce(
            || vec![0.0f32; hidden],
            |mut a, b| {
                for (x, &y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        );

    let t_par = t_par_start.elapsed();
    let t_sum = std::time::Duration::ZERO;

    // Post-experts output policy (Gemma 4: `post_feedforward_layernorm_2`)
    let t_post_start = std::time::Instant::now();
    let result = moe_post_expert_output(&expert_out, moe, norm_offset, eps);
    let t_post = t_post_start.elapsed();

    if timing {
        eprintln!(
            "[cpu_moe_forward] K={} pre_par={:.0}us q8k_quant={:.0}us \
             par_iter={:.0}us sum={:.0}us post_norm={:.0}us total={:.0}us",
            expert_indices.len(),
            t_pre_par.as_secs_f64() * 1e6,
            t_q8k_quant.as_secs_f64() * 1e6,
            t_par.as_secs_f64() * 1e6,
            t_sum.as_secs_f64() * 1e6,
            t_post.as_secs_f64() * 1e6,
            t_start.elapsed().as_secs_f64() * 1e6,
        );
    }

    if options::moe_debug_enabled() {
        let pre_rms =
            (expert_out.iter().map(|v| v * v).sum::<f32>() / expert_out.len() as f32).sqrt();
        let post_rms = (result.iter().map(|v| v * v).sum::<f32>() / result.len() as f32).sqrt();
        let pnorm2_rms = (moe.post_experts_norm.iter().map(|v| v * v).sum::<f32>()
            / moe.post_experts_norm.len().max(1) as f32)
            .sqrt();
        eprintln!(
            "  pre_norm_rms={pre_rms:.3} post_norm2_rms={pnorm2_rms:.3} moe_out_rms={post_rms:.3}"
        );
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::quantize_q4_k;
    use crate::{Activation, QuantFormat};

    fn bf16_fill(len: usize, val: f32) -> Vec<u8> {
        let b = ((val.to_bits() >> 16) as u16).to_le_bytes();
        let mut bytes = vec![0u8; len * 2];
        for i in 0..len {
            bytes[i * 2] = b[0];
            bytes[i * 2 + 1] = b[1];
        }
        bytes
    }

    fn one_expert_moe<'a>(
        _hidden: usize,
        inter: usize,
        experts_gate_up: Vec<&'a [u8]>,
        experts_down: Vec<&'a [u8]>,
        router: &'a [f32],
        format: QuantFormat,
    ) -> MoeLayerWeights<'a> {
        MoeLayerWeights {
            experts_gate_up,
            experts_down,
            routing_policy: crate::MoeRoutingPolicy::default(),
            weight_layout: crate::MoeWeightLayout::default(),
            router_proj: router,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: 1,
            top_k: 1,
            intermediate_size: inter,
            activation: Activation::Silu,
            expert_data_format: format,
        }
    }

    #[test]
    fn empty_selected_expert_weight_slices_are_skipped() {
        let hidden = 8;
        let inter = 2;
        let router = vec![1.0f32; hidden];
        let h = vec![1.0f32; hidden];
        let gate_up = bf16_fill(2 * inter * hidden, 1.0);
        let down = bf16_fill(hidden * inter, 1.0);

        let missing_gate_up = one_expert_moe(
            hidden,
            inter,
            vec![&[]],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        );
        assert_eq!(
            cpu_moe_forward(&h, &missing_gate_up, 0.0, 1e-6),
            vec![0.0; hidden]
        );

        let missing_down = one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![&[]],
            &router,
            QuantFormat::BF16,
        );
        assert_eq!(
            cpu_moe_forward(&h, &missing_down, 0.0, 1e-6),
            vec![0.0; hidden]
        );
    }

    #[test]
    fn selected_expert_with_missing_down_table_is_skipped() {
        let hidden = 8;
        let inter = 2;
        let num_experts = 4;
        let gate_up = bf16_fill(2 * inter * hidden, 1.0);
        let down = bf16_fill(hidden * inter, 1.0);
        let experts_gate_up = vec![
            gate_up.as_slice(),
            gate_up.as_slice(),
            gate_up.as_slice(),
            gate_up.as_slice(),
        ];
        let experts_down = vec![down.as_slice()];
        let mut router = vec![0.0f32; num_experts * hidden];
        router[3 * hidden..4 * hidden].fill(10.0);
        let moe = MoeLayerWeights {
            experts_gate_up,
            experts_down,
            routing_policy: crate::MoeRoutingPolicy::default(),
            weight_layout: crate::MoeWeightLayout::default(),
            router_proj: &router,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k: 1,
            intermediate_size: inter,
            activation: Activation::Silu,
            expert_data_format: QuantFormat::BF16,
        };

        assert_eq!(
            cpu_moe_forward(&vec![1.0; hidden], &moe, 0.0, 1e-6),
            vec![0.0; hidden]
        );
    }

    #[test]
    fn q4k_cached_dequant_fallback_runs_for_non_256_hidden() {
        let hidden = 128;
        let inter = 1;
        let inter_padded = 256;
        let gate_up_f32: Vec<f32> = (0..2 * inter * hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let down_f32: Vec<f32> = (0..hidden * inter_padded)
            .map(|i| {
                if i % inter_padded == 0 {
                    ((i / inter_padded) as f32 % 13.0 - 6.0) * 0.01
                } else {
                    0.0
                }
            })
            .collect();
        let gate_up = quantize_q4_k(&gate_up_f32);
        let down = quantize_q4_k(&down_f32);
        let router = vec![1.0f32; hidden];
        let h: Vec<f32> = (0..hidden).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let moe = one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::Q4_K,
        );

        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);

        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_per_expert_scale_filters_selected_expert() {
        let hidden = 8;
        let inter = 2;
        let gate_up = bf16_fill(2 * inter * hidden, 1.0);
        let down = bf16_fill(hidden * inter, 1.0);
        let router = vec![1.0f32; hidden];
        let zero_scale = [0.0f32];
        let moe = MoeLayerWeights {
            router_per_expert_scale: &zero_scale,
            ..one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &router,
                QuantFormat::BF16,
            )
        };

        assert_eq!(
            cpu_moe_forward(&vec![1.0; hidden], &moe, 0.0, 1e-6),
            vec![0.0; hidden]
        );
    }

    // ENV_LOCK protects tests that set process-wide env vars. The cpu_moe_forward
    // hot path reads `LARQL_SKIP_MOE`, `LARQL_MOE_DEBUG`, and (once per thread,
    // through a TLS cache) `LARQL_MOE_FWD_TIMING`.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_env<T>(vars: &[(&'static str, Option<&'static str>)], f: impl FnOnce() -> T) -> T {
        // Recover from prior-test panics so an unrelated failure in one
        // env-sensitive test doesn't cascade-fail the others.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous: Vec<_> = vars
            .iter()
            .map(|(name, _)| (*name, std::env::var_os(name)))
            .collect();
        for (name, value) in vars {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
        let result = f();
        for (name, value) in previous {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
        result
    }

    fn trivial_moe_inputs() -> (usize, usize, Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
        let hidden = 8;
        let inter = 2;
        let gate_up = bf16_fill(2 * inter * hidden, 1.0);
        let down = bf16_fill(hidden * inter, 1.0);
        let router = vec![1.0f32; hidden];
        let h = vec![1.0f32; hidden];
        (hidden, inter, gate_up, down, router, h)
    }

    #[test]
    fn returns_zero_vec_when_num_experts_zero() {
        let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
        let moe = MoeLayerWeights {
            num_experts: 0,
            ..one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &router,
                QuantFormat::BF16,
            )
        };
        assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
    }

    #[test]
    fn returns_zero_vec_when_top_k_zero() {
        let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
        let moe = MoeLayerWeights {
            top_k: 0,
            ..one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &router,
                QuantFormat::BF16,
            )
        };
        assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
    }

    #[test]
    fn returns_zero_vec_when_intermediate_size_zero() {
        let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
        let moe = MoeLayerWeights {
            intermediate_size: 0,
            ..one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &router,
                QuantFormat::BF16,
            )
        };
        assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
    }

    #[test]
    fn returns_zero_vec_when_router_proj_empty() {
        let (hidden, inter, gate_up, down, _, h) = trivial_moe_inputs();
        let empty: [f32; 0] = [];
        let moe = MoeLayerWeights {
            router_proj: &empty,
            ..one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &empty,
                QuantFormat::BF16,
            )
        };
        assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
    }

    #[test]
    fn returns_zero_vec_when_experts_gate_up_table_empty() {
        let (hidden, inter, _, down, router, h) = trivial_moe_inputs();
        let moe = one_expert_moe(
            hidden,
            inter,
            vec![],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        );
        assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
    }

    #[test]
    fn returns_zero_vec_when_experts_down_table_empty() {
        let (hidden, inter, gate_up, _, router, h) = trivial_moe_inputs();
        let moe = one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![],
            &router,
            QuantFormat::BF16,
        );
        assert_eq!(cpu_moe_forward(&h, &moe, 0.0, 1e-6), vec![0.0; hidden]);
    }

    #[test]
    fn skip_moe_env_returns_zero_vec_before_running_experts() {
        let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();
        let moe = one_expert_moe(
            hidden,
            inter,
            vec![gate_up.as_slice()],
            vec![down.as_slice()],
            &router,
            QuantFormat::BF16,
        );
        let out = with_env(&[(crate::options::ENV_SKIP_MOE, Some("1"))], || {
            cpu_moe_forward(&h, &moe, 0.0, 1e-6)
        });
        assert_eq!(out, vec![0.0; hidden]);
    }

    #[test]
    fn moe_debug_env_exercises_diagnostic_branches() {
        // Pre-experts + router norm + scale arrays populated so the debug
        // branch reads non-empty buffers (rms calculations).
        let hidden = 8;
        let inter = 2;
        let gate_up = bf16_fill(2 * inter * hidden, 1.0);
        let down = bf16_fill(hidden * inter, 1.0);
        let router = vec![0.5f32; hidden];
        let pre = vec![1.0f32; hidden];
        let rnorm = vec![1.0f32; hidden];
        // `router_scale` is element-wise multiplied with `router_in` (zip), so
        // it must match `hidden` in length to avoid silently truncating the
        // router input vector.
        let rscale = vec![1.0f32; hidden];
        let post = vec![1.0f32; hidden];
        let moe = MoeLayerWeights {
            pre_experts_norm: &pre,
            router_norm: &rnorm,
            router_scale: &rscale,
            post_experts_norm: &post,
            ..one_expert_moe(
                hidden,
                inter,
                vec![gate_up.as_slice()],
                vec![down.as_slice()],
                &router,
                QuantFormat::BF16,
            )
        };
        let h = vec![0.25f32; hidden];

        let out = with_env(&[(crate::options::ENV_MOE_DEBUG, Some("1"))], || {
            cpu_moe_forward(&h, &moe, 0.0, 1e-6)
        });
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn moe_fwd_timing_env_prints_timing_block() {
        // FWD_TIMING is a TLS-cached read of the env var, evaluated lazily on
        // first hit per thread.  Run the entire call on a fresh
        // `std::thread::spawn` so the TLS init sees the env var set to "1"
        // instead of inheriting an earlier `false` from this test process's
        // main thread.
        let (hidden, inter, gate_up, down, router, h) = trivial_moe_inputs();

        let result = with_env(&[(crate::options::ENV_MOE_FWD_TIMING, Some("1"))], || {
            std::thread::spawn(move || {
                let moe = one_expert_moe(
                    hidden,
                    inter,
                    vec![gate_up.as_slice()],
                    vec![down.as_slice()],
                    &router,
                    QuantFormat::BF16,
                );
                cpu_moe_forward(&h, &moe, 0.0, 1e-6)
            })
            .join()
            .expect("timing thread did not panic")
        });
        assert_eq!(result.len(), hidden);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn scratch_reallocates_when_dimensions_change_between_calls() {
        // Force rayon's worker TLS to hold a scratch for one shape, then call
        // again with different (hidden, inter) so the size-mismatch branch
        // (`*scratch = ExpertScratch::new(...)`) is taken.  We don't assert
        // the path runs on every worker — rayon may schedule onto a
        // freshly-warmed thread — but the par_iter is non-empty on either
        // shape, so at least one worker hits the branch.
        let first_h = 8usize;
        let first_inter = 2usize;
        let first_gate_up = bf16_fill(2 * first_inter * first_h, 1.0);
        let first_down = bf16_fill(first_h * first_inter, 1.0);
        let first_router = vec![1.0f32; first_h];
        let first_moe = one_expert_moe(
            first_h,
            first_inter,
            vec![first_gate_up.as_slice()],
            vec![first_down.as_slice()],
            &first_router,
            QuantFormat::BF16,
        );
        let out_a = cpu_moe_forward(&vec![1.0; first_h], &first_moe, 0.0, 1e-6);
        assert_eq!(out_a.len(), first_h);

        let second_h = 16usize;
        let second_inter = 4usize;
        let second_gate_up = bf16_fill(2 * second_inter * second_h, 1.0);
        let second_down = bf16_fill(second_h * second_inter, 1.0);
        let second_router = vec![1.0f32; second_h];
        let second_moe = one_expert_moe(
            second_h,
            second_inter,
            vec![second_gate_up.as_slice()],
            vec![second_down.as_slice()],
            &second_router,
            QuantFormat::BF16,
        );
        let out_b = cpu_moe_forward(&vec![1.0; second_h], &second_moe, 0.0, 1e-6);
        assert_eq!(out_b.len(), second_h);
        assert!(out_b.iter().all(|v| v.is_finite()));
    }
}
