//! The CPU executor: LARQL owns the threads, kernels own the arithmetic.
//!
//! CPU-1B measured the hazard this module exists to remove. Accelerate's
//! `sgemv` arrives **already internally threaded** — partitioning rows on
//! top of it wins 1.14x and sometimes loses — while the fused BF16 kernel
//! starts serial and scales 3.56x to twelve workers. Two paths with
//! opposite threading needs, and a decode that mixes them.
//!
//! If each kernel grew its own Rayon parallelism, a stack running both
//! would oversubscribe: nested pools, 12 workers each spawning 12. So the
//! rule here is one sentence:
//!
//! > **At most one layer of parallelism owns the machine for a
//! > primitive.**
//!
//! A kernel therefore never calls `par_iter`. It exposes a ROW RANGE
//! primitive — "compute these output rows from this slab of weight" — and
//! the executor decides how many workers to use and how to cut the rows.
//! A kernel that wants no external threading says so
//! ([`CpuParallelism::LibraryOwned`]) and is called once.

pub mod arithmetic;
pub mod cost;
pub mod environment;
pub mod executor;
pub mod integer;
pub mod kernels;
pub mod ledger;
pub mod physical;
pub mod projector;
pub mod replay;
pub mod stationary;

pub use arithmetic::{AccumulatorRep, ActivationRep, Arithmetic, ScaleSpan, WeightRep};
pub use cost::{budget, predicted_tokens_per_second, ProjectionBudget, SYNTHETIC_TO_REAL};
pub use environment::Environment;
pub use executor::{shared, CpuExecutor};
pub use integer::{quantise_activation, Bf16xQ8, Q4xQ8, Q8xQ8, QuantisedActivation};
pub use kernels::{BlasF32, FusedBf16, FusedQ4, FusedQ8, ScalarF32};
pub use ledger::{ledger, thread_projection_calls, PlanTally, ProjectionLedger};
pub use physical::PhysicalProjectionPlan;
pub use projector::{CpuParallelism, DenseProjector, WeightRows};
pub use replay::{replay, start_capture, take_capture, Captured, ReplayOrder};

#[cfg(test)]
mod tests;
