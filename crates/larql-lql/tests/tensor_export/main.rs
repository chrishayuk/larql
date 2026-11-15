//! `SHOW TENSORS` and `EXPORT WEIGHT`: every tensor of a VINDEX3 container named so a caller can read it, and its
//! stored values written to a NumPy file with a sidecar that says where they came from.
//!
//! The fixtures are the small BF16 container with a Q8_0 pack compiled by the production compiler (its embedding,
//! final norm and output head are all named `weight`, as in a real checkpoint) and the F32 miniature container.
//! Every exported value is checked against the segment's bytes, read here directly.

mod export_weight;
mod npy;
mod parsing;
mod partial_output;
mod refusals;
mod show_tensors;
mod support;
