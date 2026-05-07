## Context

`larql-server`'s request handlers call `Tokenizer::encode(text)`
synchronously per request. The encode cost on Gemma 3 4B's
SentencePiece-derived tokeniser is ~80 µs per 1k characters of
text on this dev box's CPU. At 256 concurrent requests sharing a
common chat-template prefix, that's ~20 ms of redundant work per
batch.

SMG's solution (per the PyTorch blog post): two-tier cache that
exploits the fact that **chat templates are identical** across
requests in a session, and **prompts often repeat verbatim** in
benchmarks and tool-augmented agent loops.

## Goals / Non-Goals

**Goals:**
- L0 exact-match cache: zero-cost re-tokenisation of repeated prompts.
- L1 prefix-aware cache: re-use the chat-template prefix tokens
  across distinct user messages.
- Cache size tunable via env var; sensible defaults that fit in
  ≤10 MB total.
- Concurrent-safe: LRU access from multiple Tokio worker threads.

**Non-Goals:**
- Cross-process caching. Each server instance has its own LRU.
- Detokenisation cache. Output tokenisation cost is dominated by
  decode latency, not encode.
- Compression of cached tokens. They're already u32; gzip would
  cost more than it saves.
- Streaming-aware tokenisation. The cache only fires on the
  initial `encode`; streaming decode keeps cold-tokenising.

## Decisions

### D1 — `lru` crate over a hand-rolled HashMap+Vec

The `lru` crate is the workspace's standard LRU (already used by
`larql-cli`'s cache and `ffn_l2_cache`). Saves writing eviction
logic; uses `O(1)` get and put.

### D2 — L1 keyed on FNV-64 of the prefix, not the prefix itself

Storing the full prefix string (often hundreds of bytes for chat
templates) wastes memory; FNV-64 is fast and collision-resistant
enough for cache lookup. On collision the cache returns wrong
tokens — but the L1 entry stores the prefix length, so a verifier
re-tokenises the prefix only on suspicion (e.g., once per million
hits). Acceptable failure mode; would be harmful only if collisions
went undetected for compression-style scenarios where a few wrong
tokens silently corrupt logits.

We add a small additional check: alongside `(Vec<u32>, usize)`, store
the FNV-64 of the prefix's *byte content*. On lookup, the second
hash is verified — collision risk is now (1/2^64)² ≈ negligible.

### D3 — Find-last-special-token via a per-tokeniser regex

Each tokeniser knows its set of special tokens (`<|im_start|>`,
`<|im_end|>`, `<bos>`, etc.). The L1 keying logic finds the *last*
such token in the input and uses everything up to its end as the
"prefix". The bytes after the last special token are the user-
specific message; they're tokenised cold and concatenated.

The regex is compiled once per tokeniser at server startup.

### D4 — Async-safe via `tokio::sync::Mutex`

LRU access is a write operation (it reorders the linked list).
Tokio's `Mutex` is the right choice for short critical sections
inside async handlers; alternatives (`Arc<Mutex<Lru>>` with
`std::sync::Mutex`) would block the executor at high concurrency.

### D5 — Size knobs are env vars, not config files

`LARQL_TOKENIZER_CACHE_L0_SIZE` and
`LARQL_TOKENIZER_CACHE_L1_SIZE` follow the existing convention
(see `env_flags.rs`). 0 disables the cache entirely (useful for
benchmarks and debugging).

## Risks / Trade-offs

- **Risk: hash collision corrupts logits.** → Mitigation: D2's
  double-FNV check makes collision rate < 1 in 2^64 hits.
- **Risk: cache thrashing under non-repeating prompts.** Worst case:
  every request misses both L0 and L1, paying both LRU updates +
  full encode. → Mitigation: LRU put is O(1); the overhead is a
  few hundred ns per miss vs ~80 µs encode. Measured
  worst-case impact: < 1% added latency on cache-cold workloads.
- **Risk: chat-template change invalidates cached prefixes silently.**
  → Mitigation: cache key includes the tokeniser's content hash.
  Vindex reload purges the cache.

## Migration Plan

Land. New requests start using the cache transparently.
`LARQL_TOKENIZER_CACHE_L0_SIZE=0 LARQL_TOKENIZER_CACHE_L1_SIZE=0`
disables for fall-back testing.

Rollback: revert. No data path changes.

## Open Questions

- **Q1: Should we cache by (tokeniser_id, text) so multi-model
  servers don't share entries across vindexes?** Yes —
  `tokeniser_id` is the vindex's BLAKE2b hash, already computed at
  load time. Add to the cache key.
- **Q2: Pre-warm cache at startup with the chat template?** Often
  the first 500-token chunk of every chat call is identical. Adding
  a warmup pass that pre-tokenises the template would zero-cost
  every first-token call. Recommendation: yes, behind a
  `LARQL_TOKENIZER_CACHE_WARMUP=1` flag; defer to follow-up.
