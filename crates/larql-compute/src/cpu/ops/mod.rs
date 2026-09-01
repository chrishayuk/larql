//! CPU operation dispatch — one file per operation type.
//!
//! Mirrors the Metal ops/ structure for consistent API across backends.
//! Each module handles dispatch for one category of compute operation.

pub mod attention;
pub mod f32_matmul;
pub mod geglu;
pub mod kernel_shape;
pub mod linalg;
pub mod moe;
pub mod outer_combine;
pub mod q4_common;
pub mod q4_matvec;
pub mod q4_vecmat;
pub mod q4k_matvec;
pub mod q4k_q8k_dot;
pub mod q6k_matvec;
pub mod q8_matvec;
pub mod ternary_matvec;
pub mod vector;

pub use kernel_shape::KernelShapeError;
