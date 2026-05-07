## 1. New module

- [ ] 1.1 `crates/larql-server/src/tokenizer_cache.rs` with two LRUs
      and a `compile_special_token_regex` helper.
- [ ] 1.2 `pub struct TokenizerCache { l0, l1 }` with
      `get(text) -> Option<Vec<u32>>` and `insert(text, tokens)`.

## 2. Wire-up

- [ ] 2.1 Add `lru` to `larql-server/Cargo.toml` (workspace dep).
- [ ] 2.2 Construct `TokenizerCache` once per `LoadedModel` and
      stash on the `state::AppState`.
- [ ] 2.3 Wrap the existing `Tokenizer::encode` call in routes
      (`/v1/infer`, `/v1/select`, OpenAI `/v1/chat/completions`,
      `/v1/embed`).
- [ ] 2.4 Document env vars in `crates/larql-server/src/env_flags.rs`.

## 3. Tests

- [ ] 3.1 `l0_hit_returns_cached_tokens`.
- [ ] 3.2 `l1_hit_shares_chat_template_prefix`.
- [ ] 3.3 `cache_miss_falls_back_to_encode`.
- [ ] 3.4 `parallel_hits_no_corruption`.
- [ ] 3.5 `hash_collision_synthesised_returns_miss`.

## 4. Validation

- [ ] 4.1 `openspec validate server-tokenizer-cache --strict` passes.
- [ ] 4.2 `cargo test -p larql-server --lib tokenizer_cache` passes (5 tests).
- [ ] 4.3 `make traceability-check` and `make openspec-validate` pass.
- [ ] 4.4 Optional: micro-benchmark via Criterion confirming ≥10× speedup on the L0-hit path.
