//! CUDA backend init errors.
//!
//! The full surface lands with `cuda-f32-baseline`; this enum exists so
//! callers can distinguish "no driver" / "no devices" / "toolkit
//! mismatch" / "not implemented yet" without changing types later.

use std::fmt;

#[derive(Debug)]
pub enum CudaInitError {
    /// `cudarc` reported no CUDA driver. Usually means the host has no
    /// NVIDIA GPU or the driver kmod isn't loaded.
    DriverMissing(String),
    /// The driver is present but reports zero CUDA-capable devices.
    NoDevices,
    /// Driver version older than the build's minimum.
    ToolkitMismatch { found: String, need: &'static str },
    /// The requested CUDA kernel surface is not wired up yet.
    NotImplemented(&'static str),
}

impl fmt::Display for CudaInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaInitError::DriverMissing(s) => {
                write!(f, "CUDA driver not reachable: {s}")
            }
            CudaInitError::NoDevices => {
                write!(f, "CUDA driver loaded but reports no devices")
            }
            CudaInitError::ToolkitMismatch { found, need } => {
                write!(f, "CUDA toolkit mismatch: found {found}, need {need}+")
            }
            CudaInitError::NotImplemented(what) => {
                write!(
                    f,
                    "CUDA backend stub: {what} is not yet implemented \
                     (see openspec/changes/cuda-and-rotorquant-kv/)"
                )
            }
        }
    }
}

impl std::error::Error for CudaInitError {}
