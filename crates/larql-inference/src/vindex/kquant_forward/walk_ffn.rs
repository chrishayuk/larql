use larql_compute::cpu::ops::q4k_q8k_dot::{
    q4k_q8k_gate_up_into, q4k_q8k_matvec_into, quantize_x_to_q8k, Q8KActivation,
};
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::dequant::dequantize_matrix;

/// Run one layer's FFN forward on a Q4_K vindex, dequantising gate/up/down
/// for just this layer and applying the architecture's activation gate.
pub fn kquant_ffn_forward_layer(
    arch: &dyn larql_models::ModelArchitecture,
    index: &VectorIndex,
    layer: usize,
    x: &Array2<f32>,
) -> Array2<f32> {
    use crate::ffn::{gelu_tanh_gate_up, silu_gate_up};
    use crate::forward::dot_proj;

    let hidden = x.shape()[1];
    let intermediate = index.num_features(layer);

    let ffn = index
        .interleaved_kquant_layer_data(layer)
        .unwrap_or_else(|| {
            panic!(
                "interleaved_kquant layer data missing for layer {layer} - \
             server must call `load_interleaved_kquant` before serving walk-ffn"
            )
        });

    let gate = if let Some(arc) = index.kquant_ffn_layer_once(layer, 0) {
        let w_gate =
            ndarray::ArrayView2::from_shape((intermediate, hidden), &arc[..intermediate * hidden])
                .expect("gate cache shape");
        x.dot(&w_gate.t())
    } else {
        let w_gate = dequantize_matrix(ffn[0].0, ffn[0].1, intermediate, hidden);
        dot_proj(x, &w_gate)
    };
    let up = if let Some(arc) = index.kquant_ffn_layer_once(layer, 1) {
        let w_up =
            ndarray::ArrayView2::from_shape((intermediate, hidden), &arc[..intermediate * hidden])
                .expect("up cache shape");
        x.dot(&w_up.t())
    } else {
        let w_up = dequantize_matrix(ffn[1].0, ffn[1].1, intermediate, hidden);
        dot_proj(x, &w_up)
    };
    let activation = match arch.activation() {
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => {
            gelu_tanh_gate_up(&gate, &up)
        }
        _ => silu_gate_up(&gate, &up),
    };
    // Down projection: use LRU dequant cache (component=2 stores feature-major = w_down^T).
    let n = intermediate * hidden;
    if let Some(arc) = index.kquant_ffn_layer_once(layer, 2) {
        let w_down_t = ndarray::ArrayView2::from_shape((intermediate, hidden), &arc[..n])
            .expect("down cache shape");
        activation.dot(&w_down_t)
    } else {
        let inter_padded = intermediate.div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
            * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        let w_down = if inter_padded != intermediate {
            let w = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, inter_padded);
            w.slice(ndarray::s![.., ..intermediate]).to_owned()
        } else {
            dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate)
        };
        dot_proj(&activation, &w_down)
    }
}

/// Q4_K × Q8_K variant: accepts a pre-quantised Q8_K activation vector
/// (already RMS-normed by the client) and skips the dequant of gate/up by
/// using the NEON/AVX2 `q4k_q8k_gate_up_into` kernel.  Down projection
/// still goes through the f32 dequant path (no Q6K×Q8K kernel yet).
///
/// `h_q8k.qs.len()` must equal `hidden` (= `x.ncols()`), which is a
/// multiple of 256 (Q8_K block size).
///
/// Returns the FFN delta only — same semantics as `kquant_ffn_forward_layer`.
pub fn kquant_ffn_forward_layer_q8k(
    arch: &dyn larql_models::ModelArchitecture,
    index: &VectorIndex,
    layer: usize,
    h_q8k: &Q8KActivation,
) -> Array2<f32> {
    use crate::ffn::{gelu_tanh_gate_up, silu_gate_up};
    use crate::forward::dot_proj;

    let hidden = h_q8k.qs.len(); // = n_blocks * 256
    let intermediate = index.num_features(layer);

    let ffn = index
        .interleaved_kquant_layer_data(layer)
        .unwrap_or_else(|| {
            panic!(
                "interleaved_kquant layer data missing for layer {layer} - \
             server must call `load_interleaved_kquant` before serving walk-ffn-q8k"
            )
        });

    // gate + up via the fused Q4K×Q8K kernel (shared activation load).
    let mut gate_flat = vec![0.0f32; intermediate];
    let mut up_flat = vec![0.0f32; intermediate];
    q4k_q8k_gate_up_into(
        &mut gate_flat,
        &mut up_flat,
        h_q8k,
        ffn[0].0, // gate Q4K bytes
        ffn[1].0, // up Q4K bytes
        intermediate,
        hidden,
    );

    // Wrap into Array2 for the shared activation + down path.
    let gate = Array2::from_shape_vec((1, intermediate), gate_flat).expect("gate shape");
    let up = Array2::from_shape_vec((1, intermediate), up_flat).expect("up shape");

    let activation = match arch.activation() {
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu => {
            gelu_tanh_gate_up(&gate, &up)
        }
        _ => silu_gate_up(&gate, &up),
    };

    // Down projection: Q4K×Q8K NEON — quantise the f32 activation once,
    // then call the NEON matvec directly on the mmap Q4K bytes.
    // No dequant, no large f32 allocation, no BLAS thread-pool collision.
    // Guard: intermediate must be Q8K-block-aligned (multiple of the
    // Q4_K/Q8_K super-block size).
    // For non-aligned sizes (rare, non-production) fall back to OnceLock cache.
    if intermediate.is_multiple_of(crate::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS) {
        let activation_flat = activation.as_slice().expect("activation contiguous");
        let act_q8k = quantize_x_to_q8k(activation_flat);
        let mut out = vec![0.0f32; hidden];
        q4k_q8k_matvec_into(&mut out, &act_q8k, ffn[2].0, hidden, intermediate);
        Array2::from_shape_vec((1, hidden), out).expect("down output shape")
    } else {
        // Fallback: OnceLock cache + ndarray dot for non-256-aligned intermediate.
        let n = intermediate * hidden;
        if let Some(arc) = index.kquant_ffn_layer_once(layer, 2) {
            let w_down_t = ndarray::ArrayView2::from_shape((intermediate, hidden), &arc[..n])
                .expect("down cache shape");
            activation.dot(&w_down_t)
        } else {
            let inter_padded = intermediate
                .div_ceil(larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS)
                * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
            let w_down = if inter_padded != intermediate {
                let w = dequantize_matrix(ffn[2].0, ffn[2].1, hidden, inter_padded);
                w.slice(ndarray::s![.., ..intermediate]).to_owned()
            } else {
                dequantize_matrix(ffn[2].0, ffn[2].1, hidden, intermediate)
            };
            dot_proj(&activation, &w_down)
        }
    }
}
