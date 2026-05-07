//! CUDA implementation hook for RotorQuant. Phase-1 placeholder —
//! every entrypoint defers to the CPU reference. A future change
//! lands real PTX kernels for the rotation + quantise loops; the
//! public API surface (`quantize_k`, `dequantize_v_with_inverse_rotation`,
//! ...) stays identical so callers don't change.

#![cfg(feature = "cuda")]

// Reserved for the on-device kernels and module loaders. Currently
// empty — the public API in lib.rs always routes through cpu_ref.
