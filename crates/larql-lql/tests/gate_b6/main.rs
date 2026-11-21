//! Gate B, B6 (`docs/gate-b-b6.md` r2): COMPILE and COMPACT INTO VINDEX
//! write deterministic outputs and publish them atomically and durably.

mod determinism;
mod hard_links;
mod into_model;
mod publish_faults;
mod replay;
mod support;
