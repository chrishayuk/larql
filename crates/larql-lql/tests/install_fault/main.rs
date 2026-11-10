//! Forcing a COMPOSE install to fail at chosen points, and what the session looks like afterwards.
//!
//! The first group of tests is the instrumentation's own contract: it is inert unless armed and a
//! fault fires once. The second group records what a failed install leaves behind **today**. That
//! is a known defect, not a desired behaviour: see `atomicity`.

mod atomicity;
mod contract;
mod support;
