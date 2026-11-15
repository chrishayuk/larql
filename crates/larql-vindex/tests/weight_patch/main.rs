//! Weight patch: edit stored weights of a VINDEX3 container into a new one.
//!
//! The fixture is a small BF16 container with a Q8_0 pack compiled by the
//! production compiler, so every test exercises the real derive path.

mod apply;
mod errors;
mod fixture;
mod golden;
mod hf_cache_layout;
mod inspection;
mod lifecycle;
mod live_exec;
mod live_fused;
mod live_read_path;
mod live_refusals;
mod live_support;
mod live_view;
mod low_rank_op;
mod multi;
mod op;
mod op_target;
mod overlay;
mod qualified_names;
mod real_scale;
mod refusals;
mod rounding;
mod sources;
mod structured_op;
mod structured_overlay;
mod structured_role;
mod vlp_numerics;
mod vlp_numerics_v2;
