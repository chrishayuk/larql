//! Accuracy test suite for KV-cache engines.
//!
//! The suite splits results along three axes so engine trade-offs
//! surface explicitly instead of being averaged into one number:
//!
//! | Axis | Where the answer lives | Module |
//! |---|---|---|
//! | **Parametric** | In the model's weights — short cue recovers it. Survives any K/V strategy because the weights are unchanged. | [`prompts`] |
//! | **In-context** | Planted in the prompt; the K/V state has to preserve it. At risk under sliding windows / residual-stream replacement / quantised K/V / boundary checkpoints. | [`needle`] |
//! | **Conflict** | Prompt's in-context claim contradicts pretraining. Score: did the engine follow the prompt (`followed_context`) or fall back to weights (`parametric_fallback`)? The most engine-discriminating axis. | [`conflict`] |
//!
//! [`runner`] holds the result types, drivers
//! ([`runner::evaluate_parametric`], [`runner::evaluate_in_context`],
//! [`runner::evaluate_conflict`]), and the [`runner::format_strategy_split`]
//! "video frame" table:
//!
//! ```text
//!                     Param %   Param bits   InCtx %   InCtx bits   Follow %   Fallback %
//! Standard KV         100.0%         0.42     100.0%        1.10        ...        ...
//! Markov RS           100.0%         0.43      88.6%        3.71        ...        ...
//! TurboQuant 4-bit     99.0%         0.55     100.0%        1.45        ...        ...
//! ```
//!
//! Each row mixes two complementary scorers:
//! - **Top-1 match** (`Param %`, `InCtx %`) — binary verdict from argmax.
//! - **Shannon bits** (`Param bits`, `InCtx bits`) —
//!   `-log2(P(expected | prompt))`. Continuous; separates "barely
//!   confident in Paris" from "highly confident in Paris" on the same
//!   `match=true`. Lower is better; ≤2 bits = strong recall.
//!
//! Inline unit tests cover the result types, formatters, and Shannon
//! math against synthetic logits. End-to-end runs against real models
//! live in `tests/accuracy_suite_real_model.rs` (gated on
//! `LARQL_MODEL`).
//!
//! Origin: lifted from the retired `kv-cache-benchmark` crate
//! (2026-05-16). The original `KvStrategy`-based runner was replaced
//! with a `KvEngine`-trait driver and the parametric/in-context/conflict
//! taxonomy was added.

pub mod conflict;
pub mod measurement;
pub mod needle;
pub mod prompts;
pub mod runner;
