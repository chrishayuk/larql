//! Tests for the CPU executor: the kernels, the plan that pairs a format
//! with one, and the ledger that records what ran.

mod arithmetic;
mod arm_selection;
mod cost;
mod executor;
mod fp8_slab;
mod integer;
mod kernels;
mod kquant_plan;
mod ledger;
mod nvfp4_slab;
mod physical;
mod projection_cost;
mod q8;
mod regime_stability;
mod sdot;
mod stationary;
