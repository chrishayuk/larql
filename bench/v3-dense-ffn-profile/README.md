# Dense V3 FFN: real-model loopback diagnostics

Collected September 25, 2026 (London), using binary revision `8ce05698` on an
Apple M3 Max / Mac15,8 with 128 GiB RAM. Optimized CPU-only build, Rust 1.98.0.
Two persistent HTTP workers split each model's layers evenly; the CLI retains
attention, row KV, norms, residuals and head. Carrier: full f32 JSON.

**Provisional diagnostics, not a promoted performance result.** Other coding
sessions were visible and peer exclusivity was not confirmed. The first bracket
for each model passed the numerical control-drift check; both second brackets
failed it and are excluded from comparisons. Raw data for every run is retained.

## Observations

First measured local / remote / local brackets; local is the midpoint of its
two controls. Time is the mean over 39 matched continuation positions, excluding
the prompt and first eight continuation steps. Throughput is reciprocal mean
profiled traversal time, excluding model load, tokenization, sampling and I/O.

| Model | Layers / hidden | Local ms/token | Remote ms/token | Local tok/s | Remote tok/s | Added ms/token | Control drift |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B | 28 / 1024 | 22.856 | 39.976 | 43.75 | 25.02 | 17.120 | 0.378% |
| Gemma 3 4B | 34 / 2560 | 61.479 | 93.629 | 16.27 | 10.68 | 32.150 | 0.438% |

All 20 CLI runs (including warmups) produced the same prompt IDs, generated IDs
and executed input IDs within each model. This is real-model **token agreement**;
full-logit bitwise parity remains the separate synthetic integration gate.

The binding records show BF16 source representations for both models. Qwen's
FFN lowering is `Decode(BlasF32)`; Gemma's is `Requantise(FusedQ8)`, both through
`cpu-production/v1`. The Gemma container and result directory contain `q4k` in
their names because that container also stores Q4_K. **This run selected BF16
source weights, not its stored Q4_K representation.** The existing weight
lowering is shared by the local and remote arms; no carrier quantization was
introduced.

## Where the time goes

Mean per token in the first measured remote arm. The first five rows form a
disjoint partition; the RPC details below them are nested inside the FFN call.

| Interval | Qwen ms | Gemma ms |
|---|---:|---:|
| Local attention and entry | 10.028 | 20.102 |
| FFN provider call, inclusive | 28.356 | 66.663 |
| Local residual / re-entry | 0.082 | 0.578 |
| Embedding/head/other traversal | 1.510 | 6.286 |
| **Total** | **39.976** | **93.629** |
| Client request encoding | 1.987 | 4.411 |
| HTTP round trip through response body receipt | 23.957 | 57.307 |
| Worker FFN transform, inside that round trip | 11.085 | 34.756 |
| Worker response encoding | 1.466 | 3.706 |
| Client response decoding | 1.995 | 4.187 |
| Transport/HTTP remainder after worker handler | 4.845 | 8.004 |

Local FFN computation averages 11.116 ms and 34.546 ms respectively. The worker
transform remains close to that cost; relocating it adds protocol work around
the same computation. Worker validation and response construction additionally
take about 2.495 ms/token for Qwen and 3.928 ms/token for Gemma, measured as
`worker.execute_ns - worker.ffn_ns`. The transport remainder includes HTTP and
scheduling overhead; it is not a pure network RTT measurement.

| HTTP body traffic per token, decimal MB | Qwen | Gemma |
|---|---:|---:|
| Sent | 1.317 | 2.475 |
| Received | 1.322 | 2.537 |
| Binding JSON alone, repeated per direction | 1.000 | 1.484 |
| Binding share of request body | 76.0% | 60.0% |

Actual transmitted body lengths come from the HTTP client. Binding-only sizes
were calculated from compact serialization of the saved binding descriptors,
weighted by each worker's owned layer count. HTTP/TLS/TCP headers are excluded.
The repeated descriptors are a substantial wire cost before any carrier
compression is considered. These measurements do not estimate the gain from
changing that protocol.

## Run discipline and receipts

Both arms use `LARQL_CPU_WORKERS=8`, `RAYON_NUM_THREADS=8`,
`VECLIB_MAXIMUM_THREADS=1`, `TOKIO_WORKER_THREADS=2`. Four warmup runs precede
two local / remote / local brackets. Each CLI invocation starts fresh local KV;
workers remain resident. Warmup is recorded, but a short run and unconfirmed
peer exclusivity do not establish sustained plateau or machine exclusivity.
Profiling overhead is included in the observations.

Second-bracket control drift was **2.562% for Qwen** and **1.128% for Gemma**.
Their comparison ratios are null in `brackets.json`; they were not averaged
into the table. No LAN, Metal coordinator, routed experts, carrier compression,
long-context scaling, or uninstrumented throughput was measured here.

- [Qwen receipts](results/qwen3-0.6b-20260925/manifest.json),
  [brackets](results/qwen3-0.6b-20260925/brackets.json),
  [raw traces](results/qwen3-0.6b-20260925/raw-traces.tar.gz).
- [Gemma receipts](results/gemma3-4b-q4k-20260925/manifest.json),
  [brackets](results/gemma3-4b-q4k-20260925/brackets.json),
  [raw traces](results/gemma3-4b-q4k-20260925/raw-traces.tar.gz).

Each result directory also contains worker bindings, per-run commands and
summaries, stdout/stderr, server logs, binary/metadata hashes and
`trace-archive.json` with hashes of every raw JSONL file. Archive contents were
verified against the originals. Workers were stopped after each model's run.

Reproduce collection with [run_loopback.py](run_loopback.py), using a new output
directory. This driver deliberately labels its output diagnostic; a confirmed
peer handshake and a fuller plateau/replication protocol are needed for a
performance claim. See [field definitions](../../docs/ffn/v3-dense-profile.md).
