//! Permanent security regressions for `ApolloStore::load`, against filesystem
//! semantics as well as bytes: sparse files, devices, FIFOs and aliases.
//!
//! Each load runs on its own watched thread, so a hang fails its test instead of
//! hanging the suite, and a thread-local allocator measures it and can refuse it.
//! An abort still kills this process, which is also a failure.
//!
//! Issue #679 and the review on PR #684 are the source of these cases.

mod allocation;
mod filesystem;
mod fixtures;
mod fuzz;
mod instrument;
mod malformed;

#[global_allocator]
static ALLOCATOR: instrument::ThreadCountingAllocator = instrument::ThreadCountingAllocator;
