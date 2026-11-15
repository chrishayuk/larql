//! Greedy speculation policy, shared by every speculative decode loop.
//!
//! The policy is what to propose, how much of a proposal the target
//! agreed with, and where generation ends. It owns no execution and no
//! continuation state: each loop runs its own session (the local Metal
//! session, or the remote routed block session), keeps its own K/V, and
//! calls these functions to decide what to verify and what to keep.

pub mod greedy;
pub mod prompt_lookup;
pub mod stop;

pub use greedy::greedy_id;
pub use prompt_lookup::{accepted_prefix, propose, MAX_NGRAM, MIN_NGRAM};
pub use stop::truncate_at_stop;

#[cfg(test)]
mod tests;
