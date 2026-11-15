//! Stored tensors: every tensor listed by its qualified name, and any plain float tensor read exactly as stored,
//! whole or by rows and columns.
//!
//! The fixtures are the small BF16 container with a Q8_0 pack compiled by the production compiler (its embedding,
//! final norm and output head are all named `weight`, as in a real checkpoint) and the F32 miniature container.

mod listing;
mod locate;
mod read;
mod support;
