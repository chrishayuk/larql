mod device_tables;
mod kernels;

use std::sync::Arc;

pub use cuda_core::CudaContext;
use cuda_core::{DeviceBuffer, LaunchConfig};
use cuda_host::{cuda_launch, load_kernel_module};

use crate::{KvFormat, QuantizedKv, RotorQuantError};
use kernels::{
    __iso3_dequantize_block_CudaKernel, __iso4_dequantize_block_CudaKernel,
    __iso4_quantize_rows_CudaKernel, __planar3_dequantize_block_CudaKernel,
    __planar3_quantize_rows_CudaKernel, __planar4_dequantize_block_CudaKernel,
    __planar4_quantize_rows_CudaKernel,
};

pub fn quantize(
    ctx: &Arc<CudaContext>,
    format: KvFormat,
    data: &[f32],
    n_rows: usize,
    head_dim: usize,
) -> Result<QuantizedKv, RotorQuantError> {
    validate_quantize_input(format, data, n_rows, head_dim)?;

    let stream = ctx.default_stream();
    let input_dev = DeviceBuffer::from_host(&stream, data).map_err(cuda_err)?;
    let mut codes_dev = DeviceBuffer::<u8>::zeroed(&stream, code_len(format, n_rows, head_dim)?)
        .map_err(cuda_err)?;
    let mut norms_dev = DeviceBuffer::<f32>::zeroed(&stream, n_rows).map_err(cuda_err)?;
    let mut rotations_dev =
        DeviceBuffer::<u16>::zeroed(&stream, n_rows * (head_dim / format.block_size()))
            .map_err(cuda_err)?;
    let module = load_kernel_module(ctx, "larql_rotorquant").map_err(cuda_err)?;
    let config = LaunchConfig::for_num_elems(n_rows as u32);

    match format {
        KvFormat::Planar3 => {
            cuda_launch! {
                kernel: planar3_quantize_rows,
                stream: stream,
                module: module,
                config: config,
                args: [
                    slice(input_dev),
                    slice_mut(codes_dev),
                    slice_mut(norms_dev),
                    slice_mut(rotations_dev),
                    head_dim
                ]
            }
        }
        KvFormat::Planar4 => {
            cuda_launch! {
                kernel: planar4_quantize_rows,
                stream: stream,
                module: module,
                config: config,
                args: [
                    slice(input_dev),
                    slice_mut(codes_dev),
                    slice_mut(norms_dev),
                    slice_mut(rotations_dev),
                    head_dim
                ]
            }
        }
        KvFormat::Iso4 => {
            cuda_launch! {
                kernel: iso4_quantize_rows,
                stream: stream,
                module: module,
                config: config,
                args: [
                    slice(input_dev),
                    slice_mut(codes_dev),
                    slice_mut(norms_dev),
                    slice_mut(rotations_dev),
                    head_dim
                ]
            }
        }
        KvFormat::Iso3 => {
            return Err(RotorQuantError::InvalidBuffer(
                "cuda-oxide quantize currently supports Planar3, Planar4, and Iso4".into(),
            ));
        }
    }
    .map_err(cuda_err)?;

    Ok(QuantizedKv {
        format,
        n_rows,
        head_dim,
        codes: codes_dev.to_host_vec(&stream).map_err(cuda_err)?,
        norms: norms_dev.to_host_vec(&stream).map_err(cuda_err)?,
        rotation_indices: rotations_dev.to_host_vec(&stream).map_err(cuda_err)?,
    })
}

pub fn dequantize(ctx: &Arc<CudaContext>, qkv: &QuantizedKv) -> Result<Vec<f32>, RotorQuantError> {
    validate(qkv)?;

    let stream = ctx.default_stream();
    let codes_dev = DeviceBuffer::from_host(&stream, &qkv.codes).map_err(cuda_err)?;
    let norms_dev = DeviceBuffer::from_host(&stream, &qkv.norms).map_err(cuda_err)?;
    let rotations_dev =
        DeviceBuffer::from_host(&stream, &qkv.rotation_indices).map_err(cuda_err)?;
    let mut out_dev =
        DeviceBuffer::<f32>::zeroed(&stream, qkv.n_rows * qkv.head_dim).map_err(cuda_err)?;
    let module = load_kernel_module(ctx, "larql_rotorquant").map_err(cuda_err)?;
    let config = LaunchConfig::for_num_elems((qkv.n_rows * qkv.head_dim) as u32);

    match qkv.format {
        KvFormat::Planar3 => {
            cuda_launch! {
                kernel: planar3_dequantize_block,
                stream: stream,
                module: module,
                config: config,
                args: [
                    slice(codes_dev),
                    slice(norms_dev),
                    slice(rotations_dev),
                    slice_mut(out_dev),
                    qkv.head_dim
                ]
            }
        }
        KvFormat::Planar4 => {
            cuda_launch! {
                kernel: planar4_dequantize_block,
                stream: stream,
                module: module,
                config: config,
                args: [
                    slice(codes_dev),
                    slice(norms_dev),
                    slice(rotations_dev),
                    slice_mut(out_dev),
                    qkv.head_dim
                ]
            }
        }
        KvFormat::Iso3 => {
            cuda_launch! {
                kernel: iso3_dequantize_block,
                stream: stream,
                module: module,
                config: config,
                args: [
                    slice(codes_dev),
                    slice(norms_dev),
                    slice(rotations_dev),
                    slice_mut(out_dev),
                    qkv.head_dim
                ]
            }
        }
        KvFormat::Iso4 => {
            cuda_launch! {
                kernel: iso4_dequantize_block,
                stream: stream,
                module: module,
                config: config,
                args: [
                    slice(codes_dev),
                    slice(norms_dev),
                    slice(rotations_dev),
                    slice_mut(out_dev),
                    qkv.head_dim
                ]
            }
        }
    }
    .map_err(cuda_err)?;

    out_dev.to_host_vec(&stream).map_err(cuda_err)
}

pub fn dequantize_iso3(
    ctx: &Arc<CudaContext>,
    qkv: &QuantizedKv,
) -> Result<Vec<f32>, RotorQuantError> {
    validate_iso3(qkv)?;

    let stream = ctx.default_stream();
    let codes_dev = DeviceBuffer::from_host(&stream, &qkv.codes).map_err(cuda_err)?;
    let norms_dev = DeviceBuffer::from_host(&stream, &qkv.norms).map_err(cuda_err)?;
    let rotations_dev =
        DeviceBuffer::from_host(&stream, &qkv.rotation_indices).map_err(cuda_err)?;
    let mut out_dev =
        DeviceBuffer::<f32>::zeroed(&stream, qkv.n_rows * qkv.head_dim).map_err(cuda_err)?;
    let module = load_kernel_module(ctx, "larql_rotorquant").map_err(cuda_err)?;

    cuda_launch! {
        kernel: iso3_dequantize_block,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems((qkv.n_rows * qkv.head_dim) as u32),
        args: [
            slice(codes_dev),
            slice(norms_dev),
            slice(rotations_dev),
            slice_mut(out_dev),
            qkv.head_dim
        ]
    }
    .map_err(cuda_err)?;

    out_dev.to_host_vec(&stream).map_err(cuda_err)
}

fn validate_iso3(qkv: &QuantizedKv) -> Result<(), RotorQuantError> {
    if qkv.format != KvFormat::Iso3 {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "cuda-oxide Iso3 dequantize requires Iso3, got {:?}",
            qkv.format
        )));
    }
    validate(qkv)
}

fn validate(qkv: &QuantizedKv) -> Result<(), RotorQuantError> {
    let block_size = qkv.format.block_size();
    if !qkv.head_dim.is_multiple_of(block_size) {
        return Err(RotorQuantError::HeadDimNotDivisible {
            format: qkv.format,
            head_dim: qkv.head_dim,
            block_size,
        });
    }
    if !qkv.head_dim.is_multiple_of(4) {
        if matches!(qkv.format, KvFormat::Iso3 | KvFormat::Iso4) {
            return Err(RotorQuantError::HeadDimNotDivisible {
                format: qkv.format,
                head_dim: qkv.head_dim,
                block_size: 4,
            });
        }
    }
    let n_codes = qkv.n_rows * qkv.head_dim;
    let expected_code_bytes = (n_codes * usize::from(qkv.format.bits())).div_ceil(8);
    if qkv.codes.len() != expected_code_bytes {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "codes.len {} != expected {:?} bytes {}",
            qkv.codes.len(),
            qkv.format,
            expected_code_bytes
        )));
    }
    if qkv.norms.len() != qkv.n_rows {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "norms.len {} != n_rows {}",
            qkv.norms.len(),
            qkv.n_rows
        )));
    }
    let expected_rotations = qkv.n_rows * (qkv.head_dim / block_size);
    if qkv.rotation_indices.len() != expected_rotations {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "rotation_indices.len {} != expected {}",
            qkv.rotation_indices.len(),
            expected_rotations
        )));
    }
    Ok(())
}

fn validate_quantize_input(
    format: KvFormat,
    data: &[f32],
    n_rows: usize,
    head_dim: usize,
) -> Result<(), RotorQuantError> {
    let block_size = format.block_size();
    if !head_dim.is_multiple_of(block_size) {
        return Err(RotorQuantError::HeadDimNotDivisible {
            format,
            head_dim,
            block_size,
        });
    }
    let expected = n_rows * head_dim;
    if data.len() != expected {
        return Err(RotorQuantError::InputLengthMismatch {
            got: data.len(),
            n_rows,
            head_dim,
            expected,
        });
    }
    if format.bits() == 3 && !(head_dim * usize::from(format.bits())).is_multiple_of(8) {
        return Err(RotorQuantError::InvalidBuffer(format!(
            "cuda-oxide {:?} quantize requires row code bits to be byte-aligned, got head_dim {}",
            format, head_dim
        )));
    }
    Ok(())
}

fn code_len(format: KvFormat, n_rows: usize, head_dim: usize) -> Result<usize, RotorQuantError> {
    let n_codes = n_rows * head_dim;
    Ok((n_codes * usize::from(format.bits())).div_ceil(8))
}

fn cuda_err<E: std::fmt::Display>(err: E) -> RotorQuantError {
    RotorQuantError::CudaOxide(err.to_string())
}
