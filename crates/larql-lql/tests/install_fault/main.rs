//! Forcing a COMPOSE install to fail at chosen points, and what the session looks like afterwards.
//!
//! The first group of tests is the instrumentation's own contract: it is inert unless armed and a
//! fault fires once. The second group is the requirement it measures, Gate B's B5: a failed or
//! refused INSERT leaves the session exactly as it found it, on both backends. See `atomicity`.

mod atomicity;
mod contract;
mod support;
