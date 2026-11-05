//! `UPDATE WEIGHTS`: editing stored weights through LQL, directly or as part
//! of a `.vlp` patch, and compiling the result into a new container.
//!
//! The fixture is a small BF16 container with a Q8_0 pack compiled by the
//! production compiler, so every test goes through the real derive path.

mod direct;
mod golden;
mod parsing;
mod refusals;
mod support;
mod vlp;
