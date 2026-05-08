mod device_tables;
mod kernels;

use std::sync::Arc;

pub use cuda_core::CudaContext;
use cuda_core::{DeviceBuffer, LaunchConfig};
use cuda_host::{cuda_launch, load_kernel_module};

use crate::{KvFormat, QuantizedKv, RotorQuantError};
use kernels::{
    __iso3_dequantize_block_CudaKernel, __iso4_dequantize_block_CudaKernel,
    __planar3_dequantize_block_CudaKernel, __planar4_dequantize_block_CudaKernel,
};

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

fn cuda_err<E: std::fmt::Display>(err: E) -> RotorQuantError {
    RotorQuantError::CudaOxide(err.to_string())
}
