//! Metal GPU operation dispatch — one file per operation type.
//!
//! Each module handles dispatch for one category of compute operation:
//! - `q4_matvec`: Q4×Q8 matrix-vector (gate scoring, up projection)
//! - `q4_vecmat`: Q4 vector-matrix (scatter-accumulate down projection)
//! - `q4_f32_matvec`: Q4×f32 matrix-vector (transposed down)
//! - `q4_batched`: Batched operations (pair_batch, multi-layer pipeline)
//! - `q4_quantize`: Helper to quantize f32→Q8 on GPU
//!
//! All operations use the shared `BufferCache` for weight caching
//! and `ComputePipelineState` from shader compilation.

pub mod full_layer;
pub mod full_pipeline;
pub mod kv_cache;
pub mod q4_batched;
pub mod q4_common;
pub mod q4_f32_matvec;
pub mod q4_matvec;
pub mod q4_vecmat;
