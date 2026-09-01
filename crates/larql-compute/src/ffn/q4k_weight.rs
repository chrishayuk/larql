//! Q4_K dense FFN backend — the quantised sibling of [`super::weight::WeightFfn`].
//!
//! Weights are quantised once at construction (row-major Q4_K
//! super-blocks). Execution is **regime-split**, because the TTS
//! funnel's benchmarks (`docs/tts-funnel.md`) proved the two shapes
//! want opposite strategies on this hardware:
//!
//! * **decode (one row):** the packed representation feeds the
//!   Q4K·Q8K integer-dot kernels directly — dequantise-then-BLAS is
//!   dramatically slower when the dequant cannot amortise;
//! * **prefill (many rows):** dequantise each projection once and run
//!   fp32 BLAS GEMM — the integer kernels are issue-bound at
//!   ~650-700 GFLOPS in every loop structure, while AMX GEMM runs the
//!   same work at ~2.2 TFLOPS with the dequant amortised across rows.
//!
//! Scope: gated-SiLU FFNs without biases (the Qwen3 family, which covers
//! both MOSS transformers). Construction refuses anything else rather
//! than silently computing a different function.

use ndarray::Array2;

use larql_models::{Activation, FfnType, ModelWeights};

use crate::cpu::ops::q4_common::{dequantize_q4_k, quantize_q4_k};
use crate::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_parallel;
use crate::quantize_x_to_q8k;

use super::silu_gate_up;
use super::FfnBackend;

/// Q4_K super-blocks span this many weights; every matrix dimension fed
/// to [`quantize_q4_k`] must divide into them row by row.
const Q4K_BLOCK_ELEMS: usize = 256;
/// Registry tag for the packed format, as the kernel dispatch expects.
const Q4K_FORMAT_TAG: &str = "Q4_K";

struct Q4kLayer {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    hidden: usize,
    intermediate: usize,
}

/// Dense gated-SiLU FFN over Q4_K-packed weights.
pub struct Q4kFfn {
    layers: Vec<Q4kLayer>,
}

impl Q4kFfn {
    /// Quantise `weights`' FFN projections. Errors (by message) on
    /// non-gated / non-SiLU architectures, missing tensors, biased FFNs,
    /// or dimensions that don't divide into Q4_K blocks — every case
    /// where "quantised" would quietly mean "different math".
    pub fn quantize_from(weights: &ModelWeights) -> Result<Self, String> {
        let arch = &*weights.arch;
        if arch.ffn_type() != FfnType::Gated || arch.activation() != Activation::Silu {
            return Err(format!(
                "Q4kFfn supports gated-SiLU FFNs; {} is {:?}/{:?}",
                arch.family(),
                arch.ffn_type(),
                arch.activation()
            ));
        }
        let mut layers = Vec::with_capacity(weights.num_layers);
        for layer in 0..weights.num_layers {
            if arch.ffn_up_bias_key(layer).is_some() || arch.ffn_down_bias_key(layer).is_some() {
                return Err("Q4kFfn does not support FFN biases".to_string());
            }
            let pack = |key: String| -> Result<(Vec<u8>, usize, usize), String> {
                let tensor = weights
                    .tensors
                    .get(&key)
                    .ok_or_else(|| format!("missing FFN tensor {key}"))?;
                let (rows, cols) = tensor.dim();
                if cols % Q4K_BLOCK_ELEMS != 0 {
                    return Err(format!(
                        "{key}: {cols} columns not a multiple of {Q4K_BLOCK_ELEMS}"
                    ));
                }
                let data = tensor
                    .as_standard_layout()
                    .as_slice()
                    .expect("standard layout")
                    .to_vec();
                Ok((quantize_q4_k(&data), rows, cols))
            };
            let (gate, intermediate, hidden) = pack(arch.ffn_gate_key(layer))?;
            let (up, up_rows, up_cols) = pack(arch.ffn_up_key(layer))?;
            let (down, down_rows, down_cols) = pack(arch.ffn_down_key(layer))?;
            if (up_rows, up_cols) != (intermediate, hidden)
                || (down_rows, down_cols) != (hidden, intermediate)
            {
                return Err(format!(
                    "layer {layer}: inconsistent FFN shapes \
                     gate [{intermediate},{hidden}] up [{up_rows},{up_cols}] \
                     down [{down_rows},{down_cols}]"
                ));
            }
            layers.push(Q4kLayer {
                gate,
                up,
                down,
                hidden,
                intermediate,
            });
        }
        Ok(Self { layers })
    }
}

impl FfnBackend for Q4kFfn {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let l = &self.layers[layer];
        let rows = x.nrows();
        if rows > 1 {
            return self.forward_block(l, x);
        }
        let mut gate = Array2::<f32>::zeros((rows, l.intermediate));
        let mut up = Array2::<f32>::zeros((rows, l.intermediate));
        for (row, x_row) in x.rows().into_iter().enumerate() {
            let x_vec: Vec<f32> = x_row.to_vec();
            let q8 = quantize_x_to_q8k(&x_vec);
            q4k_q8k_matvec_parallel(
                gate.row_mut(row).as_slice_mut().expect("contiguous"),
                &q8,
                &l.gate,
                l.intermediate,
                l.hidden,
                Q4K_FORMAT_TAG,
            )
            .unwrap_or_else(|e| panic!("Q4kFfn::forward layer {layer}: {e}"));
            q4k_q8k_matvec_parallel(
                up.row_mut(row).as_slice_mut().expect("contiguous"),
                &q8,
                &l.up,
                l.intermediate,
                l.hidden,
                Q4K_FORMAT_TAG,
            )
            .unwrap_or_else(|e| panic!("Q4kFfn::forward layer {layer}: {e}"));
        }
        let activation = silu_gate_up(&gate, &up);
        let mut out = Array2::<f32>::zeros((rows, l.hidden));
        for (row, act_row) in activation.rows().into_iter().enumerate() {
            let act_vec: Vec<f32> = act_row.to_vec();
            let q8 = quantize_x_to_q8k(&act_vec);
            q4k_q8k_matvec_parallel(
                out.row_mut(row).as_slice_mut().expect("contiguous"),
                &q8,
                &l.down,
                l.hidden,
                l.intermediate,
                Q4K_FORMAT_TAG,
            )
            .unwrap_or_else(|e| panic!("Q4kFfn::forward layer {layer}: {e}"));
        }
        out
    }

    fn name(&self) -> &str {
        "q4k-weights"
    }
}

impl Q4kFfn {
    /// Multi-row (prefill-shaped) forward: dequantise each projection
    /// once and run fp32 BLAS GEMM. Prefill is not "decode over a
    /// batch of rows" — and, measured at the MOSS 343-row shapes
    /// (`moss_prefill_bench`), it is not an integer-GEMM problem on
    /// this hardware either: the Q4K·Q8K kernel is issue-bound at
    /// ~650-700 GFLOPS in every loop structure tried (row-at-a-time,
    /// weight-chunk blocked, activation-parallel blocked — see
    /// `q4k_q8k_dot::gemm`), while dequant-whole + AMX BLAS runs the
    /// same work 1.6x faster (~2.2 TFLOPS on the GEMM, dequant
    /// amortised across the rows). The decode doctrine
    /// ("dequant-then-BLAS is dramatically slower") is a one-row
    /// truth; it inverts at prefill shape.
    ///
    /// Numerics: slightly *closer* to the dense reference than the
    /// decode path (no Q8K activation quantisation). Batch and
    /// incremental sessions share this code, so the step-5
    /// batch ≡ incremental gate is unaffected.
    fn forward_block(&self, l: &Q4kLayer, x: &Array2<f32>) -> Array2<f32> {
        let dense = |packed: &[u8], rows: usize, cols: usize| -> Array2<f32> {
            let values = dequantize_q4_k(packed, rows * cols);
            Array2::from_shape_vec((rows, cols), values).expect("dequant shape")
        };
        let gate_w = dense(&l.gate, l.intermediate, l.hidden);
        let up_w = dense(&l.up, l.intermediate, l.hidden);
        let gate = x.dot(&gate_w.t());
        let up = x.dot(&up_w.t());
        let activation = silu_gate_up(&gate, &up);
        let down_w = dense(&l.down, l.hidden, l.intermediate);
        activation.dot(&down_w.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::WeightFfn;
    use larql_models::test_fixtures::make_test_weights;
    use larql_models::WeightArray;
    use std::collections::HashMap;

    /// A 1-layer gated-SiLU model with Q4K-divisible dims, values small
    /// enough that Q4_K error stays well-behaved.
    fn q4k_divisible_weights() -> ModelWeights {
        const HIDDEN: usize = 256;
        const INTER: usize = 512;
        let arch_json = serde_json::json!({
            "model_type": "tinymodel",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 1,
            "intermediate_size": INTER,
            "head_dim": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
        });
        let arch = larql_models::detect_from_json(&arch_json);
        let mut state = 0x1234_5678_u64;
        let mut mat = |rows: usize, cols: usize| -> WeightArray {
            let data: Vec<f32> = (0..rows * cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / u32::MAX as f32 - 0.25) * 0.2
                })
                .collect();
            Array2::from_shape_vec((rows, cols), data)
                .unwrap()
                .into_shared()
        };
        let mut tensors: HashMap<String, WeightArray> = HashMap::new();
        tensors.insert(arch.ffn_gate_key(0), mat(INTER, HIDDEN));
        tensors.insert(arch.ffn_up_key(0), mat(INTER, HIDDEN));
        tensors.insert(arch.ffn_down_key(0), mat(HIDDEN, INTER));
        let embed = mat(32, HIDDEN);
        ModelWeights {
            tensors,
            vectors: HashMap::new(),
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            per_layer_ffn_format: HashMap::new(),
            per_layer_ffn_arrangement: HashMap::new(),
            embed: embed.clone(),
            lm_head: embed,
            position_embed: None,
            num_layers: 1,
            hidden_size: HIDDEN,
            intermediate_size: INTER,
            vocab_size: 32,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 2,
            rope_base: 10000.0,
            arch,
        }
    }

    #[test]
    fn q4k_ffn_close_to_dense() {
        let weights = q4k_divisible_weights();
        let dense = WeightFfn { weights: &weights };
        let q4k = Q4kFfn::quantize_from(&weights).unwrap();

        let x = Array2::from_shape_fn((3, weights.hidden_size), |(r, c)| {
            ((r * 31 + c) % 17) as f32 * 0.02 - 0.15
        });
        let reference = dense.forward(0, &x);
        let quantised = q4k.forward(0, &x);

        assert_eq!(reference.dim(), quantised.dim());
        // Quantisation noise on a gated FFN with near-cancelling sums is
        // genuinely nonzero; what distinguishes "kernel wrong" from
        // "4-bit weights" is the overall direction and energy of the
        // output, not per-element deltas.
        let (mut dot, mut ref_sq, mut q_sq, mut diff_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in reference.iter().zip(quantised.iter()) {
            dot += a as f64 * b as f64;
            ref_sq += (a as f64).powi(2);
            q_sq += (b as f64).powi(2);
            diff_sq += (a as f64 - b as f64).powi(2);
        }
        let cosine = dot / (ref_sq.sqrt() * q_sq.sqrt());
        let relative = (diff_sq / ref_sq).sqrt();
        assert!(
            cosine > 0.98 && relative < 0.2,
            "Q4K FFN drifted: cosine {cosine:.4}, relative error {relative:.4}"
        );
    }

    #[test]
    fn refuses_non_divisible_dims() {
        let weights = make_test_weights();
        assert!(Q4kFfn::quantize_from(&weights).is_err());
    }

    #[test]
    fn multi_row_forward_tracks_dense_at_least_as_closely_as_decode_path() {
        // The prefill path (dequant + BLAS) uses different arithmetic
        // than the decode path (Q8K integer dot) — no bit contract
        // between them. The contract is against the dense reference:
        // the multi-row path must be at least as close to fp32 truth as
        // the decode path is, and deterministic.
        let weights = q4k_divisible_weights();
        let dense = WeightFfn { weights: &weights };
        let q4k = Q4kFfn::quantize_from(&weights).unwrap();
        let x = Array2::from_shape_fn((5, weights.hidden_size), |(r, c)| {
            ((r * 31 + c) % 17) as f32 * 0.02 - 0.15
        });

        let reference = dense.forward(0, &x);
        let multi = q4k.forward(0, &x);
        assert_eq!(
            multi,
            q4k.forward(0, &x),
            "prefill path must be deterministic"
        );

        let relative_error = |got: &Array2<f32>| -> f64 {
            let (mut diff_sq, mut ref_sq) = (0.0f64, 0.0f64);
            for (&a, &b) in reference.iter().zip(got.iter()) {
                diff_sq += (a as f64 - b as f64).powi(2);
                ref_sq += (a as f64).powi(2);
            }
            (diff_sq / ref_sq).sqrt()
        };
        let mut decode_rows = Array2::<f32>::zeros(reference.dim());
        for row in 0..x.nrows() {
            let single = q4k.forward(0, &x.slice(ndarray::s![row..=row, ..]).to_owned());
            decode_rows.row_mut(row).assign(&single.row(0));
        }
        let multi_error = relative_error(&multi);
        let decode_error = relative_error(&decode_rows);
        assert!(
            multi_error <= decode_error * 1.05,
            "prefill path drifted past the decode path: {multi_error:.5} vs {decode_error:.5}"
        );
    }
}
