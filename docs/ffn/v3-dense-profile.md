# Exact dense V3 provider timing

`larql run --v3-profile PATH` writes a new JSONL file for a CPU continuation
run, either fully local or using `--v3-ffn-shards`. It does not change the FFN
carrier, routing, binding checks, or numerical provider. Binary f32 is the
default remote wire; `--v3-ffn-wire json` selects the historical JSON control
([wire contract](v3-ffn-wire.md)). It refuses an existing
output file, chat mode, Metal, and full-prefix layer-worker replay.

For a two-layer dense artifact, start the existing workers:

```bash
larql-server model.vindex3 --ffn-only --layers 0-0 --port 9181
larql-server model.vindex3 --ffn-only --layers 1-1 --port 9182
```

Run the same input through each placement:

```bash
larql run model.vindex3 "Hello" --max-tokens 32 --emit-ids --v3-profile local.jsonl
larql run model.vindex3 "Hello" --max-tokens 32 --emit-ids --v3-profile loopback.jsonl \
  --v3-ffn-shards http://localhost:9181,http://localhost:9182
python3 scripts/v3_ffn_profile.py loopback.jsonl
```

Choose worker ranges covering **every layer** for a larger artifact. For LAN,
change only the worker URLs and output filename. Use the same artifact and
build on all machines. These commands illustrate collection, not a completed
performance comparison.

## What is measured

The first JSONL line identifies the schema, artifact path, placement, and whether
the run completed. Following lines identify each executed input position and
token ID (`null` for an embedding input). Prompt positions are included. The
last sampled output token is not executed when generation stops, so the number
of records is not the number of emitted tokens. Timing excludes loading,
tokenization, sampling, detokenization, and output I/O. Records are buffered on
the calling thread and written after execution, including partial failed runs.

Each position partitions elapsed wall time into four disjoint intervals:

| Field | Boundary |
|---|---|
| `attention_ns` | Layer entry, including pre-attention normalization, through attention and KV append |
| `ffn_ns` | FFN application; local computation or the full remote provider call including validation |
| `reentry_ns` | Post-attention update, FFN normalization, and post-FFN normalization/residual update |
| `other_ns` | Embedding, final norm/head, and remaining traversal overhead |
| `total_ns` | Sum of the four intervals; one canonical decode traversal |

The `provider_calls` array contains ordered per-layer HTTP diagnostics. These
are **nested inside `ffn_ns`**, not additional top-level time:

| Field | Meaning |
|---|---|
| `request_bytes`, `response_bytes` | Actual body lengths: binary header/carrier or JSON binding/carrier; excludes HTTP headers, TLS and TCP overhead |
| `encode_ns` | Client request construction and wire encoding |
| `roundtrip_ns` | Client HTTP send through complete response body receipt |
| `decode_ns` | Client response wire decoding |
| `worker.decode_ns` | Worker request body receipt and wire decoding/admission |
| `worker.queue_ns` | Wait for the blocking worker task |
| `worker.execute_ns` | JSON: authority validation plus FFN/response construction; binary: admitted transform |
| `worker.ffn_ns` | Prepared worker transform, including its input/output and provider checks; nested within `execute_ns` |
| `worker.encode_ns` | Worker response body encoding |
| `worker.handler_ns` | Request extraction through response body encoding, including task wakeup/validation overhead |
| `transport_remainder_ns` | Round trip minus worker handler time, when nonnegative |

The transport remainder includes client/server HTTP overhead, task scheduling,
and network transfer. It is **not a pure RTT or one-way latency measurement**.
No synchronized clocks are assumed. Missing worker diagnostics are unknown,
not zero; a negative subtraction is recorded as `null`. Timings travel only
when requested via `x-larql-ffn-profile: 1`, in a response header of the same
name. Profiling leaves numerical response bodies unchanged. Worker diagnostics are not
execution authority or an attestation of remote performance.

`complete: false` marks a failed traversal or transport call. The numerical
binding/shape/finiteness checks still run, and a remote failure still invalidates
the continuation. A successfully decoded RPC can therefore appear within a
failed token record if the coordinator subsequently rejects its binding.

## Measurement discipline

Profiling adds clocks and diagnostic serialization; its overhead is included.
Do not promote these runs as uninstrumented throughput. Keep the exact local
and exact remote arms on the same input IDs and build. Record the artifact
identity, machine details, worker placement, thread settings, and build revision
with the results. Compare prompt and continuation positions separately.

For a performance claim, establish exclusivity with every peer session on every
machine, warm to plateau, then use local / remote / local brackets. Discard a
block if its local controls disagree by more than about 1%. Skipping some trace
rows with `--skip` alone does not establish warmup or exclusivity. Preserve the
raw nanoseconds; do not evaluate a percentage gate using rounded display data.

The HTTP integration gate can save a diagnostic fixture trace:

```bash
LARQL_V3_FFN_SMOKE_PROFILE=/tmp/v3-ffn-smoke.jsonl \
  cargo test -p larql-server --no-default-features --test test_vindex3_serve \
  v3_dense_ffn_workers_over_http_preserve_local_continuation -- --exact
python3 scripts/v3_ffn_profile.py /tmp/v3-ffn-smoke.jsonl
```

This is a tiny two-layer debug fixture with bitwise local/remote parity and
timing-accounting checks. It measures actual loopback HTTP but is **not** model
performance evidence. The output file must not already exist.
