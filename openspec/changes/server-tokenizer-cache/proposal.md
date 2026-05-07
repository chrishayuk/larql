## Why

Each `larql-server` request currently hits `Tokenizer::encode` cold.
At high concurrency that becomes a measurable host-side cost — every
request pays full BPE tokenisation, even when the chat-template
prefix (`<|im_start|>system\n…<|im_end|>\n<|im_start|>user\n`) is
identical across thousands of requests.

The SMG project (LightSeek) reports **216k tokenisations/s** with
**99% memory reduction** (180 KB → 1.4 KB per node) by adding a
two-tier cache in front of their tokenizer. Their numbers translate
to a measured **23% TTFT reduction** at 256 concurrency on
Llama-3.3-70B. This is a cheap win that doesn't touch any GPU code.

This is **not** on the critical path for the CUDA + RotorQuant
workstream — it's a server-side optimisation we revisit once the
tensor-math layer stops being the bottleneck.

## What Changes

- ADD `crates/larql-server/src/tokenizer_cache.rs` with two LRUs:
  - **L0 exact-match**: `LruCache<String, Vec<u32>>` keyed on the
    full input string. Hits return tokens with zero BPE work.
  - **L1 prefix-trie**: `LruCache<u64, (Vec<u32>, usize)>` keyed on
    the FNV-64 hash of the *prefix up to the last special-token
    boundary*. Hits return the shared prefix tokens; the suffix
    after the last `<|im_start|>`-style marker is tokenised cold
    and concatenated.
- MODIFY route handlers (`/v1/infer`, `/v1/select`, `/v1/embed`,
  OpenAI `/v1/chat/completions`) to consult the cache before
  falling back to `Tokenizer::encode`.
- ADD env-var tuning: `LARQL_TOKENIZER_CACHE_L0_SIZE` (default
  4096), `LARQL_TOKENIZER_CACHE_L1_SIZE` (default 16384).
- ADD a small set of inline tests covering: L0 hit, L1 prefix hit,
  cache miss falls back to `Tokenizer::encode`, eviction order is
  LRU.
- MODIFY `server-vindex-loading` capability to declare the cache
  contract in spec form.

This is non-breaking. Cache miss path is identical to today's
behaviour.

## Capabilities

### New Capabilities

(none — implements scenarios on `server-vindex-loading`.)

### Modified Capabilities

- `server-vindex-loading`: adds a tokenizer-cache requirement with
  scenarios for L0/L1 hit semantics and LRU eviction.

## Impact

- **Affected files**: `crates/larql-server/src/tokenizer_cache.rs`
  (new); `routes/{infer,embed,openai/...}` (call-site updates);
  `Cargo.toml` adds `lru` dep (already in workspace via
  `larql-cli` cache); env-flag listed in `env_flags.rs`.
- **Affected systems**: server-only. Inference, attention, KV cache
  untouched.
- **Memory budget**: at default sizes (4k L0 entries × ~100 bytes,
  16k L1 entries × ~200 bytes) the cache is ~5 MB per server
  process — negligible vs the multi-GB vindex.
- **Provenance**: derived from the SMG technique. We don't vendor
  any of their code; the algorithm is straightforward.
- **Out of scope**: caching tokeniser output across server
  instances (Redis/memcached); caching detokenisation; chat-template
  awareness beyond special-token boundaries.
