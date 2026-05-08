# `attention-service-routes` A.2 — handoff

> _Last updated: this commit. Use this doc to pick up A.2 without
> re-reading the whole conversation history._

## What's already on `main` (do not redo)

- **C** — loud `503 quantized_model_unsupported` guard in
  `routes/attention.rs::handle_prefill` and `handle_decode` plus the
  gRPC siblings. Verified against real `chrishayuk/gemma-3-4b-it-vindex`.
- **A.1** — prefill response contract pivoted to:
  - default → `{"final_hidden": [seq][hidden], …}`
  - `?capture=layer-residuals` → also adds
    `{"post_attention_residuals": [layers][seq][hidden]}` (FP32-only).
- **A.2 helper landed (not yet wired)** — new public function
  `larql_inference::vindex::prefill_q4k_from_embeddings(weights, h0,
  index, moe_remote) -> (Array2<f32>, Vec<Option<SharedKV>>)`.

## What's left for A.2

Wire the helper into the two route handlers (HTTP + gRPC) so Q4K
models stop returning 503 on the default contract.

### Concrete changes needed

1. **`crates/larql-server/src/routes/attention.rs::handle_prefill`** —
   right now the FP32 guard returns 503 when
   `check_fp32_attention_loaded(&weights)` errors. Replace that
   branch with:

   ```rust
   if check_fp32_attention_loaded(&weights).is_ok() {
       // FP32 path — current code, unchanged. Honours
       // ?capture=layer-residuals.
   } else {
       // Q4K path: fall through to a new helper that calls
       // larql_inference::vindex::prefill_q4k_from_embeddings.
       // ?capture=layer-residuals on Q4K must 501 with
       // "layer_residuals_unsupported_on_quantized" — the dequant
       // loop doesn't expose intermediate residuals without a
       // hook variant.
   }
   ```

   The Q4K branch needs:
   - A **write** lock on weights (`model.lock_weights_for_gen()`
     instead of `get_or_load_weights()`) — `prefill_q4k_from_embeddings`
     transiently inserts/removes tensors per layer.
   - To run inside `tokio::task::spawn_blocking` (it's CPU-bound).
   - To populate `session.cache.set_layer(layer, kv)` from the
     `kvs` Vec the helper returns.
   - To return `final_hidden` from the helper's `Array2<f32>` output.
   - To 501 if `weights.arch.has_per_layer_embeddings()` (Gemma 4 E2B).
     The helper skips PLE; an embedding-input variant that respects
     PLE would need token_ids alongside the embeddings.

2. **`crates/larql-server/src/grpc_attention.rs::prefill`** — same
   refactor, mirroring the HTTP path. The streaming PrefillEvent
   shape doesn't fit "single final_hidden" cleanly; for Q4K, emit
   one terminal event with `done=true` and the full final_hidden
   in `post_attention.rows[seq][hidden]` (one event total).

3. **Decode handler (HTTP + gRPC)** — Q4K decode is harder because
   `prefill_q4k_from_embeddings` does the full forward from
   embeddings, but decode advances by ONE position. The cleanest
   path for Q4K decode is to call `prefill_q4k_from_embeddings`
   with the new query embedding stacked onto the old K/V cache
   somehow — but that's not what the helper does. **Recommended:**
   keep decode at 503 quantized_model_unsupported for now and ship
   a separate `decode_q4k_one_step_from_embedding` helper later.
   That keeps A.2 scope to prefill-only, which is the path
   `make attention-smoke` exercises.

4. **Tests:**
   - Replace the `prefill_returns_503_quantized_model_unsupported`
     test with one that asserts a 200 response on the q4k-like
     model (attn_q tensors stripped) — confirming the new dispatch
     path fires. The existing synthetic-Q4K test fixture (strip
     `attn_q_key` from `weights.tensors`) won't load Q4K data
     from a real vindex, so this end-to-end test is best run
     against `output/gemma-3-4b-it-vindex` as an `#[ignore]`d
     real-model integration test (analogue to
     `prefill_gemma_shaped_residuals_match_local_reference`).
   - Add a smoke-test scenario that calls
     `make attention-smoke` against the live Gemma 3 4B server
     and asserts `final_hidden` is non-zero.

5. **Spec wiring** — flip the corresponding scenario in
   `openspec/changes/cuda-and-rotorquant-kv/specs/server-attention-service/spec.md`
   from `<!-- test: unbacked -->` to the new test.

## Reference implementation

The helper's body is in
`crates/larql-inference/src/vindex/q4k_forward/hidden.rs`
(`pub fn prefill_q4k_from_embeddings(...)`). It mirrors
`predict_q4k_hidden` exactly except:

- Takes pre-embedded `h0: Array2<f32>` instead of `token_ids`.
- Skips `precompute_per_layer_inputs` (PLE).
- Returns the K/V cache vec alongside the final hidden state.

That helper is **already public and exported** as
`larql_inference::vindex::prefill_q4k_from_embeddings`. It compiles.
It hasn't been validated against a real Gemma 3 4B run end-to-end —
that validation is the first step of A.2 wiring.

## Verification command (after wiring)

```bash
# Boot the server with real Gemma 3 4B
cargo run --release -p larql-server -- \
    --role attention --port 8081 \
    output/gemma-3-4b-it-vindex

# In another terminal:
LARQL_ATTN_URL=http://localhost:8081 \
LARQL_MODEL_ID=gemma-3-4b-it \
  python3 scripts/attention-service-smoke.py \
    --hidden-dim 2560 --seq-len 8 --kv-format fp32

# Expected: every endpoint round-trips, final_hidden is non-zero,
# snapshot bytes_len reflects 8 prefill positions × 34 layers ×
# (kv_dim × 4 bytes × 2) = ~5 MB (not 338).
```

## OpenSpec hooks

- `openspec/changes/cuda-and-rotorquant-kv/specs/server-attention-service/spec.md`
  has the "Prefill of 1024 tokens populates the KV cache" scenario.
- `openspec/changes/attention-service-routes/specs/server-attention-service/spec.md`
  has "decode residual matches local reference" — partially backed
  by `test_attention_validation`; flip the second `unbacked` once a
  real-model decode validation lands.
