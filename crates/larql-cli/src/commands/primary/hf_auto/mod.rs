//! `hf://` for a raw Hugging Face checkpoint: encode to VINDEX3 on first
//! use, cache it, run it.
//!
//! `larql run hf://Qwen/Qwen3-0.6B "…"` used to work only when the repo
//! was a published vindex. A checkpoint (`config.json` + safetensors, no
//! `index.json`) is now encoded once with the `larql vindex3 encode`
//! pipeline — headers staged, plan gated on text generation, only the
//! tensors the plan needs fetched by range — into the encoded cache
//! (`~/.cache/larql/encoded/`, see [`store`]), keyed by repo and commit.
//! Every later run is answered from that cache with no network, and the
//! entry lists, removes and runs by shorthand like any other cache entry.
//!
//! Repos that ARE vindexes, local paths and existing shorthands take
//! exactly the paths they took before; see [`resolve`] for the order.
//!
//! An encode is never refreshed behind the user's back: `run` stays
//! offline-first. `larql pull` is the explicit refresh ([`refresh`]).

pub mod classify;
pub mod encode;
mod live_runtime;
pub mod refresh;
pub mod refusal;
pub mod resolve;
pub mod store;

#[cfg(test)]
mod tests;

pub use live_runtime::{pull_hf_model, resolve_hf_model};
pub use resolve::Resolution;
pub use store::EncodedStore;

/// URI scheme naming a Hugging Face repo.
pub const HF_SCHEME: &str = "hf://";
