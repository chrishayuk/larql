//! KV-cache engine implementations.
//!
//! Each engine implements [`crate::KvEngine`] (which lives in
//! `larql-inference::kv_engine` and is re-exported here) — a common
//! interface for prefill + autoregressive decode that manages inference
//! state differently:
//!
//! ## Engine ladder (Gemma 3 4B @ 370K tokens)
//!
//! | Engine | Mechanism | Memory | Accuracy |
//! |---|---|---|---|
//! | [`standard`] | Production K/V tensor cache (default) | O(seq) f32 K/V | exact — the reference |
//! | [`no_cache`] | Full re-forward per step | O(seq) token IDs | exact — correctness fallback |
//! | [`markov_residual`] | Residual-stream replacement | ~171 MB | exact (KL=0.0) under contract |
//! | [`unlimited_context`] | Per-window K/V checkpoints | ~193 MB | exact within window |
//! | [`turbo_quant`] | WHT + Lloyd-Max 3/4-bit codec | ~12.7 GB | cos≈0.991 |
//! | [`apollo`] | Boundary store + residual injection | ~11 MB | task accuracy |
//!
//! ## Selecting an engine
//!
//! ```text
//! larql bench gemma3-4b-q4k --engine standard
//! larql bench gemma3-4b-q4k --engine standard:window=1024
//! larql bench gemma3-4b-q4k --engine no-cache
//! larql bench gemma3-4b-q4k --engine markov-rs:window=512
//! larql bench gemma3-4b-q4k --engine unlimited-context:window=256
//! larql bench gemma3-4b-q4k --engine turbo-quant:bits=3
//! larql bench gemma3-4b-q4k --engine apollo:layer=25,coef=8.0
//! ```
//!
//! See [`crate::EngineKind::from_name`] for the full parameter syntax.
//!
//! ## Architecture notes
//!
//! - **Metal Q4K path** (`prefill_quant` / `decode_step_quant`): all four engines
//!   use the Metal `decode_token` full pipeline when a Q4K VectorIndex and a
//!   Metal backend are available. This gives 93-95 tok/s — matching or exceeding
//!   the standard larql-metal path (76 tok/s) because the engine bench uses
//!   faster Metal lm_head KNN rather than a full vocab matmul.
//!
//! - **CPU fallback**: when Metal is unavailable, engines fall back to a CPU
//!   path using dequantised attention tensors (lazily inserted into
//!   `weights.tensors`) and `WalkFfn` for Q4K FFN.
//!
//! - **Apollo compressed path**: when the store has boundary residuals captured
//!   at `crystal_layer` (default 30), `forward_from_layer` runs only
//!   `crystal_layer..num_layers` layers (~4 instead of 34), ~8.5× faster per step.

pub mod apollo;
pub mod boundary_kv;
pub mod boundary_per_layer;
pub mod markov_residual;
pub mod markov_residual_codec;
pub mod no_cache;
pub mod standard;
pub mod turbo_quant;
pub mod unlimited_context;
