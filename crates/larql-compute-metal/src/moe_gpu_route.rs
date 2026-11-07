//! Descriptor-driven MoE layer encode — rung E of the GPU-dataflow
//! routing ladder.
//!
//! The GPU twin of `moe_zero_copy::encode_experts_and_combine_zero_copy`:
//! same kernels, same dispatch geometry, same slot-aligned scratch — but
//! every ROUTE-DEPENDENT input is a GPU-resident buffer:
//!
//! - expert offsets: `moe_descriptor_gather` (was: CPU resolve + `set_bytes`)
//! - gate/up biases: `moe_bias_stage` (was: CPU memcpy loop)
//! - down biases:    `moe_down_bias_stage` (was: CPU memcpy loop)
//! - routing weights: rung B's `selected_weights` buffer bound with
//!   `set_buffer` (was: `set_bytes` of CPU-computed weights)
//! - bias presence: a LAYER fact read from the descriptor table's bank
//!   presence (was: `expert_mlp(selected_id)` — a fact accessed through
//!   an expert is not thereby an expert fact)
//!
//! E's contract: after the residual enters the GPU router, no CPU
//! operation may inspect, transform, stage, resolve, or combine anything
//! whose value depends on which experts won. Route-INDEPENDENT host work
//! (x staging, dims, layer uniforms) is permitted — removing it is F's
//! subject (scheduling), not E's (semantics). The `route_witness`
//! counters hold this path to that contract: it must not move them.
//!
//! Q6_K + ContiguousHalves only for now — rung G extends the descriptor
//! bindings to MXFP4; other formats keep the legacy path by explicit
//! caller choice (complete or refuse, never a silent partial arm).

mod encode;
mod forward;
mod options;
mod rows;
mod transform;

// The forward wrappers and `MoeTokenScheduleStats` surface as inherent
// items on `MetalBackend` (tests receive the struct by return value);
// only the route gate + transform resolver are consumed by path.
pub use rows::MoeBlockScratch;
pub(crate) use transform::gpu_route_enabled;
