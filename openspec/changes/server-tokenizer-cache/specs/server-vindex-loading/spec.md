## ADDED Requirements

### Requirement: server SHALL maintain a two-tier tokenizer cache

`larql-server` SHALL expose a tokenizer cache with two tiers:

- **L0 exact-match**: keyed on the full input string. A hit returns
  the cached `Vec<u32>` with zero BPE work.
- **L1 prefix-aware**: keyed on the FNV-64 hash of the prefix
  ending at the last recognised special token (`<|im_start|>`,
  `<|im_end|>`, `<bos>`, `<eos>`, etc.). A hit returns the shared
  prefix tokens; only the suffix after the last special token is
  tokenised cold.

Sizes SHALL be tunable via `LARQL_TOKENIZER_CACHE_L0_SIZE` (default
4096) and `LARQL_TOKENIZER_CACHE_L1_SIZE` (default 16384). Setting
either to 0 SHALL disable that tier.

#### Scenario: L0 hit returns cached tokens
- **WHEN** the same input string is encoded twice
- **THEN** the second call SHALL return the cached `Vec<u32>` without invoking `Tokenizer::encode`
<!-- test: unbacked -->

#### Scenario: L1 hit shares the chat-template prefix
- **WHEN** two requests differ only in their user-message suffix and the chat-template prefix is identical up to the last special token
- **THEN** the L1 cache SHALL serve the prefix tokens and only the suffix is tokenised cold
<!-- test: unbacked -->

#### Scenario: cache miss falls back to Tokenizer::encode
- **WHEN** an input has neither an L0 nor an L1 hit
- **THEN** the handler SHALL call `Tokenizer::encode` and the result SHALL be stored in both tiers
<!-- test: unbacked -->

### Requirement: cache MUST be safe at high concurrency

The cache SHALL use `tokio::sync::Mutex` to guard LRU updates so
concurrent request handlers don't corrupt the linked list. Hit
latency on a warm cache SHALL be ≤ 5 µs even at 256 concurrent
requests.

#### Scenario: parallel hits don't deadlock or corrupt
- **WHEN** 256 concurrent requests hit the same L0 entry simultaneously
- **THEN** every request SHALL receive the correct tokens and no panic / deadlock SHALL occur
<!-- test: unbacked -->

### Requirement: hash collision MUST NOT corrupt tokens

The L1 cache SHALL guard against FNV-64 collisions by storing a
second 64-bit hash of the prefix's byte content alongside the
tokens. On lookup the second hash is verified before returning a
hit. Collision rate < 1 in 2^64 effective.

#### Scenario: synthesised collision is detected
- **WHEN** a test injects two distinct prefixes with the same primary FNV-64 hash and looks up the second after the first
- **THEN** the cache SHALL return a miss (because the secondary hash differs) rather than the wrong tokens
<!-- test: unbacked -->
