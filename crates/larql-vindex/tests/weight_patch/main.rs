//! Weight patch: edit stored weights of a VINDEX3 container into a new one.
//!
//! The fixture is a small BF16 container with a Q8_0 pack compiled by the
//! production compiler, so every test exercises the real derive path.

mod apply;
mod errors;
mod fixture;
mod golden;
mod inspection;
mod low_rank_op;
mod multi;
mod op;
mod overlay;
mod refusals;
mod rounding;
mod sources;
mod structured_op;
mod structured_overlay;
mod structured_role;
mod vlp_numerics;
mod vlp_numerics_v2;
