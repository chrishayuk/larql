//! The FFN down projection's operand.
//!
//! `down` multiplies the activation, which is `inter` wide. The Q4_0 / Q8_0
//! kernels take that operand as Q8, so it has to be quantised from the
//! activation into staging of its own. The FFN input's Q8 buffers hold the
//! normed hidden state, `hidden` wide, and are the wrong operand. Binding
//! them made the full pipeline compute `down(q8(norm(h)))` and read
//! `inter - hidden` bytes past their end, so its output depended on whatever
//! sat next to them in GPU memory.

use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState};

use super::quant_matvec;

/// Q8 staging for a Q4_0 / Q8_0 down projection.
///
/// Those kernels take a Q8 operand, and the down projection's operand is
/// the activation, `inter` wide. It cannot share the FFN input's Q8 buffers
/// (`ffn_q8_in`), which hold the normed hidden state, `hidden` wide: a down
/// matvec bound to them multiplies the wrong vector and reads past its end.
/// `q8` holds `seq_len * inter` bytes and `q8s` `seq_len * inter / 32` f32
/// scales. Unused for formats that take f32 input.
#[derive(Clone, Copy)]
pub struct DownQ8Stage<'a> {
    pub q8_quant_pipeline: &'a ComputePipelineState,
    pub q8: &'a Buffer,
    pub q8s: &'a Buffer,
}

/// Whether `format`'s matvec consumes a Q8 operand rather than f32.
pub fn takes_q8_operand(format: larql_compute::QuantFormat) -> bool {
    matches!(
        format,
        larql_compute::QuantFormat::Q4_0 | larql_compute::QuantFormat::Q8_0
    )
}

/// `down_out[pos] = W_down · act[pos]` for every position, quantising the
/// activation to Q8 first when the down format needs it.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_down(
    enc: &ComputeCommandEncoderRef,
    pipes: &quant_matvec::Pipelines<'_>,
    down_q8: DownQ8Stage<'_>,
    down_format: larql_compute::QuantFormat,
    down_buf: &Buffer,
    act_scratch: &Buffer,
    down_out: &Buffer,
    seq_len: usize,
    inter: usize,
    hidden: usize,
    h_stride_bytes: u64,
    inter_stride_bytes: u64,
) {
    let q8_stride = inter as u64;
    let q8s_stride = (inter.div_ceil(LEGACY_BLOCK_ELEMS) * 4) as u64;
    for pos in 0..seq_len {
        let h_off = pos as u64 * h_stride_bytes;
        let inter_off = pos as u64 * inter_stride_bytes;
        let q8_off = pos as u64 * q8_stride;
        let q8s_off = pos as u64 * q8s_stride;
        if takes_q8_operand(down_format) {
            quant_matvec::encode_q8_quant(
                enc,
                down_q8.q8_quant_pipeline,
                act_scratch,
                inter_off,
                down_q8.q8,
                q8_off,
                down_q8.q8s,
                q8s_off,
                inter,
            );
        }
        quant_matvec::encode(
            enc,
            down_format,
            down_buf,
            act_scratch,
            inter_off,
            down_q8.q8,
            q8_off,
            down_q8.q8s,
            q8s_off,
            down_out,
            h_off,
            pipes,
            hidden,
            inter,
        );
    }
}
