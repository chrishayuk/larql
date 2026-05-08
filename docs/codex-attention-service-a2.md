# Codex prompt — finish `attention-service-routes` A.2

> _Point codex at this file. It's self-contained — you don't need
> to read prior chat history. Repo is clean, `main` is up to date,
> `make ci` is green._

## What you're doing

Wire the Q4_K-quantised model path through the
`/v1/attention/prefill` handler so it stops returning
`503 quantized_model_unsupported` and starts returning real
`final_hidden` output. The helper that does the per-layer Q4_K
dequant is already public on `main`; you only need to call it
from the route.

End-state success: `make attention-smoke` against a running
`larql-server` loaded with `output/gemma-3-4b-it-vindex` runs
**every endpoint** to 200 with non-zero `final_hidden` (currently
prefill 503s).

## Background you need

- `larql-server`'s attention service was built FP32-only. The
  prefill runner (`run_layer_with_ffn_capturing_h_post_attn`)
  reads `weights.tensors.get(attn_q_key)?` — that map is **empty**
  for Q4_K-quantised models because attention weights live in
  `index.attn_q4k_layer_data(layer)` and `weights.packed_mmaps`.
- A previous session (a) added a loud 503 guard
  (`check_fp32_attention_loaded`) so the silent failure stops, and
  (b) pivoted the response contract so `final_hidden` is the
  default field, with `?capture=layer-residuals` as the opt-in
  for per-layer captures (FP32-only).
- The Q4_K-aware helper landed at
  `larql_inference::vindex::prefill_q4k_from_embeddings`. It's
  public, exported, and compiles. It hasn't been called from
  anywhere yet.
- `/v1/completions` already works on this same Q4_K vindex via
  `larql_inference::vindex::predict_q4k_hidden` (the helper above
  is the embedding-input variant of that path).

The full handoff narrative is in
[`docs/attention-service-a2-handoff.md`](attention-service-a2-handoff.md).

## The work

### 1. Wire HTTP prefill — `crates/larql-server/src/routes/attention.rs`

Find this block in `handle_prefill` (search for `err_quantized_unsupported`):

```rust
if let Err(missing_layer) = check_fp32_attention_loaded(&weights_guard) {
    return err_quantized_unsupported(missing_layer).into_response();
}
```

Replace with a dispatch:

- **FP32 path** (current code): when
  `check_fp32_attention_loaded(&weights_guard).is_ok()`, run the
  existing `run_layer_with_ffn_capturing_h_post_attn` loop. Honour
  `?capture=layer-residuals`. No change to the FP32 numerics.

- **Q4_K path** (new): when the check fails AND
  `weights.arch.has_per_layer_embeddings()` is `false`, dispatch
  to `larql_inference::vindex::prefill_q4k_from_embeddings`. Key
  differences from the FP32 dispatch:
  - **Use `model.lock_weights_for_gen()`** (write guard), not
    `get_or_load_weights()` (read guard). The helper transiently
    inserts and removes dequantised tensors in
    `weights.tensors` per layer.
  - Pass the input `Array2<f32>` (the reshape of `req.token_embeddings`
    that the existing handler already does), the `&model.patched.read().await.base()`
    (or just `model.patched.blocking_read().base()` inside the
    spawn_blocking — the block_on RwLock idiom in
    `routes/openai/completions.rs` is the template), and
    `model.moe_remote.as_deref()` for the fourth arg.
  - The helper returns `(Array2<f32>, Vec<Option<SharedKV>>)`.
    Stash the K/V into the session cache same as the FP32 path,
    set `final_h` to the helper's first return value, and skip
    the per-layer residual collection entirely.
  - When `?capture=layer-residuals` is set on a Q4_K model, return
    `501` with `error: "layer_residuals_unsupported_on_quantized"`,
    detail explaining that per-layer captures need an FP32 path.

- **PLE-aware path** (Gemma 4 E2B): when the model declares
  `weights.arch.has_per_layer_embeddings()`, return `501` with
  `error: "per_layer_embeddings_unsupported"`, detail noting that
  the helper currently skips PLE and a token_ids-aware variant
  needs to land first.

The blocking task should still happen via `tokio::task::spawn_blocking`.
The reshape of `req.token_embeddings` into `Array2` happens before
spawn_blocking same as today. Inside the closure, you'll need to
hold the write guard and the patched-read guard simultaneously —
use the `routes/openai/completions.rs::stream_completion_blocking`
function as the structural template.

### 2. Wire gRPC prefill — `crates/larql-server/src/grpc_attention.rs`

Same dispatch logic, but:
- The streaming response contract emits one `PrefillEvent` per
  layer. For the Q4_K path (no per-layer capture), emit a single
  terminal event with `done = true`, `layer = num_layers - 1`,
  `tokens_processed`, `latency_ms`, and the full final_hidden in
  `post_attention.rows[seq][hidden]`. The semantics are "the only
  layer event is the final one"; clients can detect Q4_K by seeing
  exactly one event whose `layer == num_layers - 1`. Document this
  in the proto comment for `PrefillEvent`.

### 3. Decode handler stays at 503 (for now)

Decode is harder — `prefill_q4k_from_embeddings` does the full
forward from embeddings, but decode advances by ONE position
against an existing K/V cache. A clean Q4_K decode-step helper
needs a different shape. **Do not attempt decode in this PR.**
Leave `handle_decode` returning 503 quantized_model_unsupported
for Q4_K and add a TODO comment pointing at a future
`decode_q4k_one_step_from_embedding` helper.

### 4. Tests

Add three tests to `crates/larql-server/tests/test_attention_validation.rs`:

- `prefill_q4k_unsupported_layer_residuals_capture_returns_501`:
  on a synthetic Q4_K-like LoadedModel (the existing
  `build_state_with_q4k_like_weights` helper builds one), call
  prefill with `?capture=layer-residuals` and assert HTTP 501
  with `error == "layer_residuals_unsupported_on_quantized"`.
- `prefill_q4k_default_returns_200_against_real_vindex`:
  marked `#[ignore]`, expects `output/gemma-3-4b-it-vindex/` on
  disk. Boots an in-process router with that vindex, runs prefill
  at `seq_len = 4`, asserts HTTP 200 and that
  `final_hidden[0][0..4]` are non-zero floats. This is the
  end-to-end validation that closes the loop on the original bug.
- _Optional_:
  `prefill_q4k_default_against_real_vindex_matches_completions`,
  also `#[ignore]`. Runs prefill, then decodes one token via
  `/v1/completions` against the same prompt, and asserts
  consistency. This is the "numerical correctness" validation —
  it's harder because completions does sampling on top, so you'd
  need to drop temperature to zero and compare token strings, not
  residuals.

### 5. Spec scenarios

Flip the following from `<!-- test: unbacked -->` to point at
the new tests:
- `openspec/changes/cuda-and-rotorquant-kv/specs/server-attention-service/spec.md` —
  "Prefill of 1024 tokens populates the KV cache" → wire to
  `larql_server::test_attention_validation::prefill_q4k_default_returns_200_against_real_vindex`.

### 6. Update the smoke script

`scripts/attention-service-smoke.py` currently uses
`?capture=layer-residuals`. After A.2 lands, the default (no
query param) should also work against Q4_K models. Add a `--capture`
flag (default off) and only append the query string when set.
Drop the assertion that `post_attention_residuals` is populated
when the flag isn't set.

## Constraints

- **Do NOT skip hooks** (`--no-verify`, `--no-gpg-sign`). Keep
  every commit signed and let the precommit hook run.
- **Do NOT push to main directly.** Create a branch
  `feat/attention-service-a2`, push, open a PR. (`gh` may not be
  authed in your env; if so, just push the branch and tell the
  user the PR URL to click.)
- **Do NOT modify the FP32 path** other than adding the dispatch
  fork. The existing `test_attention_validation` tests must keep
  passing.
- **Do NOT widen the API contract.** Q4_K returns
  `final_hidden` only — it does NOT populate
  `post_attention_residuals` even when `?capture=layer-residuals`
  is set; that combination is a 501.
- **Do NOT remove the `err_quantized_unsupported` helper.** The
  PLE-501 path still uses an analogous typed-error pattern; reuse
  the helper structure. It's also still the right answer for
  models that hit a "neither FP32 nor a recognised Q4_K layout"
  edge case.

## Verification

```bash
# After your changes, on the dev box:
cargo build --release -p larql-server
RUST_LOG=info ./target/release/larql-server \
    --role attention --port 8081 \
    output/gemma-3-4b-it-vindex \
    > /tmp/larql-server.log 2>&1 &

sleep 6
curl -fsS http://localhost:8081/v1/health   # expect status:ok

LARQL_ATTN_URL=http://localhost:8081 \
LARQL_MODEL_ID=gemma-3-4b-it \
  python3 scripts/attention-service-smoke.py --hidden-dim 2560 --seq-len 8

# Expected output: every endpoint round-trips, prefill returns
# non-zero final_hidden, snapshot_bytes >> 338. Currently it
# 503s on prefill.

pkill -9 larql-server
```

`make ci` must remain green. `make attention-validate` (FP32
synthetic) and `make attention-validate-gemma` (FP32 Gemma-shaped)
must continue to pass without modification.

## Reference files (read these first)

- `docs/attention-service-a2-handoff.md` — concrete pointers
- `crates/larql-inference/src/vindex/q4k_forward/hidden.rs` —
  the helper you're calling (and `predict_q4k_hidden` for the
  pattern it mirrors)
- `crates/larql-server/src/routes/openai/completions.rs` —
  template for the `lock_weights_for_gen` + `patched.blocking_read`
  + `spawn_blocking` idiom
- `crates/larql-server/src/routes/attention.rs` — where you're
  editing; search for `A.2 follow-up seam:` for the marked
  insertion point
- `crates/larql-server/tests/test_attention_validation.rs` —
  test patterns

## Pre-existing bugs you'll see (out of scope)

- `/v1/infer` returns garbage on Q4_K models (predicts " is"
  with prob 1.0 for "The capital of France is"). Same root
  cause — uses FP32-only `predict()`. File a separate task.
- `cargo bench -p larql-server --bench attention_service` works
  on synthetic models but not against real Gemma — no benchmark
  variant for real-vindex prefill yet. Worth a follow-up.

Good luck.
