//! Q4_0 interleaved walk. C kernel with `vdotq_s32` for gate/up, scalar
//! kernel for down. Reads ~44 MB per layer (vs 315 MB for f32
//! interleaved) — 7× less data to page in, same BLAS speed warm.
//!
//! Metal Q4 path (when `self.backend.supports_quant(::larql_compute::QuantFormat::Q4_K)`): one GPU submission
//! for gate+up across all seq positions, followed by one vecmat per
//! position for down. C kernel path is the CPU fallback.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_q4_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        use larql_compute::cpu::ops::{q4_matvec, q4_vecmat};

        let q4_mmap = self.index.interleaved_q4_mmap_ref()?;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];

        let q4_bytes_per_matrix = larql_compute::QuantFormat::Q4_0
            .packed_matrix_bytes(intermediate, hidden)
            .expect("Q4_0 interleaved FFN format must have packed geometry");
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
        let layer_start = layer * q4_bytes_per_layer;

        let gate_q4 = &q4_mmap[layer_start..layer_start + q4_bytes_per_matrix];
        let up_q4 =
            &q4_mmap[layer_start + q4_bytes_per_matrix..layer_start + 2 * q4_bytes_per_matrix];
        let down_q4 =
            &q4_mmap[layer_start + 2 * q4_bytes_per_matrix..layer_start + 3 * q4_bytes_per_matrix];

        self.index.prefetch_interleaved_q4_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        let metal_q4 = self.backend.and_then(|be| {
            if be.supports_quant(::larql_compute::QuantFormat::Q4_K) {
                Some(be)
            } else {
                None
            }
        });

        if let Some(be) = metal_q4 {
            // Metal: ONE GPU submission for all gate+up across ALL seq positions
            let x_flat = x.as_slice().unwrap();
            let (all_gate, all_up) = be
                .q4_matvec_pair_batch(gate_q4, up_q4, x_flat, seq_len, intermediate, hidden)
                .unwrap();

            let mut all_activation: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            for s in 0..seq_len {
                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = all_gate[s][i];
                    let u = all_up[s][i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }
                all_activation.push(activation);
            }

            for (s, activation_row) in all_activation.iter().enumerate().take(seq_len) {
                let down_result = be
                    .q4_vecmat(activation_row, down_q4, intermediate, hidden)
                    .unwrap();
                let mut out_row = out.row_mut(s);
                for j in 0..hidden {
                    out_row[j] = down_result[j];
                }
            }
            self.trace_path(layer, "interleaved_q4:metal");
        } else {
            for s in 0..seq_len {
                let x_row = x.row(s);
                let x_slice = x_row.as_slice().unwrap();

                let gate_scores = q4_matvec::dispatch(gate_q4, x_slice, intermediate, hidden);
                let up_scores = q4_matvec::dispatch(up_q4, x_slice, intermediate, hidden);

                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = gate_scores[i];
                    let u = up_scores[i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }

                let down_result = q4_vecmat::dispatch(&activation, down_q4, intermediate, hidden);
                let mut out_row = out.row_mut(s);
                for j in 0..hidden {
                    out_row[j] = down_result[j];
                }
            }
            self.trace_path(layer, "interleaved_q4:cpu");
        }

        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }
}
