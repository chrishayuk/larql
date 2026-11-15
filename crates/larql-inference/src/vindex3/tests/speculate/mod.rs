//! Greedy speculative generation over a block target.
//!
//! Two layers, as elsewhere here: the loop's semantics (ledger, stops,
//! limits, proposals accepted always, sometimes or never, corrupted
//! proposals) against a scripted target whose greedy model is a known
//! function; then exact ids and frame accounting against the real routed
//! fixture with in-process workers.

mod fixture;
mod semantics;
