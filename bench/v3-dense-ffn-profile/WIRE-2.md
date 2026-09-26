# V3-FFN-WIRE-2: exact persistent stream versus binary HTTP

Implementation **`dfba977f`**, September 25, 2026. The experimental
`--v3-ffn-wire stream` uses one WebSocket per worker on the existing authenticated
HTTP listener. It carries the same VFF1/VFR1 f32 payloads and uses the same bound
CPU transform and per-operation blocking dispatch as binary HTTP. **Binary HTTP
remains the default.** See the [protocol and lifetime contract](../../docs/ffn/v3-ffn-wire.md).

The diagnostic supports lower transport overhead, but **does not establish a
consistent Gemma end-to-end speedup**. Qwen improved in both measured brackets.
Gemma's first candidate was slower overall despite lower transport overhead;
the two subsequent faster candidates failed the HTTP-control drift check.
No rejected bracket is pooled or promoted.

## Method and limits

Same M3 Max / 128 GiB, model containers, prompt and thread settings as
[WIRE-1](WIRE-1.md). All arms use the same optimized build
(`cargo build --release -p larql-cli -p larql-server --no-default-features`).
Qwen's CPU FFN is BF16-source `Decode(BlasF32)`; Gemma's is BF16-source
`Requantise(FusedQ8)`, despite the container's `q4k` filename. Carriers are exact
f32 throughout. Attention and row-KV stay on the CPU coordinator for this test.

Each primary collection has six warmup runs followed by two brackets:
local → binary HTTP → stream → binary HTTP → local. Each CLI generates 48 tokens;
19 prompt positions and the first eight continuation steps are excluded from
summaries, leaving 39 matched positions per run. The Gemma reversal prompted
one predefined replicate: six warmups and one bracket, with all earlier results
retained. There was no repeat-until-pass procedure or relaxed threshold.

**Peer exclusivity was not confirmed.** Other agent sessions were present;
process inspection and an empty child-agent list do not establish their
exclusivity. Warmups do not establish a sustained plateau under these conditions.
All timings remain provisional diagnostics, including brackets whose numerical
drift checks pass. These are profiled CPU loopback runs, not LAN, Metal or
uninstrumented production measurements.

## Every measured bracket

Times are mean ms per executed continuation position. HTTP drift compares the
two inner controls; local drift compares the outer pair. A comparison exceeding
1% is rejected, including the replicate's narrowly failing 1.019%.

| Model / bracket | HTTP before | Stream | HTTP after | HTTP drift | Local drift | HTTP/stream comparison |
|---|---:|---:|---:|---:|---:|---|
| Qwen / 0 | 27.859 | 26.272 | 28.119 | 0.926% | 0.763% | Numerical checks pass; provisional |
| Qwen / 1 | 27.848 | 25.919 | 27.813 | 0.124% | 3.858% | Inner check passes; local comparison rejected |
| Gemma / 0 | 68.954 | 70.256 | 68.792 | 0.236% | 2.746% | Inner check passes; stream slower; local comparison rejected |
| Gemma / 1 | 68.584 | 68.200 | 69.453 | 1.259% | 0.164% | Rejected |
| Gemma replicate / 0 | 69.729 | 67.724 | 70.444 | 1.019% | 1.805% | Rejected |

Qwen stream observations are **38.06 and 38.58 tok/s**, versus HTTP controls
35.56–35.95 tok/s. Relative to each bracket's HTTP midpoint, stream saves
**1.717 and 1.912 ms/token**. These are individual bracket observations, not a
pooled or promoted performance estimate. The three measured Gemma stream arms
range from 14.23 to 14.77 tok/s and cannot support a consistent improvement claim.

## What moved inside the provider

Mean per-token transport remainder, including scheduling outside worker handler:

| Model / bracket | HTTP midpoint, ms | Stream, ms |
|---|---:|---:|
| Qwen / 0 | 3.886 | 2.373 |
| Qwen / 1 | 3.894 | 2.408 |
| Gemma / 0 | 6.025 | 3.664 |
| Gemma / 1 (drift rejected) | 6.042 | 3.714 |
| Gemma replicate (drift rejected) | 6.146 | 3.746 |

This remainder is **not pure RTT**. HTTP worker decode includes receiving the
body; stream worker decode begins after a complete message arrives. Stream
message receipt therefore lies in its remainder. Sending timing text and the
response also lies outside the stream's handler clock. The profile definitions
are preserved in the protocol document; subtracting these remainders does not
isolate one kernel, scheduler or network component.

Gemma bracket 0 illustrates why lower transport overhead did not ensure a faster
token. Stream worker FFN took **36.274 ms**, versus **34.485 ms** at the HTTP
midpoint. Local attention took **21.497 ms**, versus **20.035 ms**. Other local
work also rose. This is a measured slowdown in useful work across both processes;
its cause was not established. It must not be erased by substituting the faster
warmups or by presenting the rejected replicate as a clean performance result.

Qwen bracket 0 worker FFN was 11.192 ms on stream versus 11.261 ms at the HTTP
midpoint. Client encode/decode totaled about 0.039 ms/token. The observations are
consistent with a modest transport improvement while leaving substantial
per-layer synchronization and dispatch overhead. No new numerical approximation
or compute algorithm is responsible for the change.

## Exact bytes and parity

All **43 CLI runs** matched prompt IDs, generated IDs and executed-position IDs
within each model. Separate fixture tests check bitwise logits through a
sliding-window boundary across local, JSON, binary HTTP and stream providers.
Every remote position in the archived traces was checked for complete calls,
exact numerical payload counts and exact top-level phase partitioning.

| Model | Numerical request bytes/token | Numerical response bytes/token | WS framing bytes/token with profiling | Timing text payload bytes/token, measured arms |
|---|---:|---:|---:|---:|
| Qwen3 0.6B | 115,696 | 115,696 | 448 | 3,705–3,707 mean |
| Gemma 3 4B | 349,384 | 349,384 | 544 | 4,577–4,596 mean |

Numerical counts include every 36-byte operation header. WS framing includes
client masking, numerical message headers and timing-message headers. The first
options message is separately counted as `stream_setup_bytes`; cold binding and
HTTP upgrade, TCP and TLS overhead are excluded. Timing text is absent when
profiling is disabled. These are body/framing counts, not packet-capture totals.

## Validation and reproduction

Passed: router fault tests (corrupt/miscorrelated/oversized replies and peer close
permanently invalidate the stream), all 22 V3 server integration tests (including
stale handles, repeated sequences, invalid layers/carriers and continuation
parity), 20 V3 CLI tests, affected-crate clippy with warnings denied, formatting
and documentation links. Loopback WS was exercised; WSS is implemented using
native TLS but was not deployment-tested. All benchmark worker processes were
stopped by the driver after collection.

```bash
python3 bench/v3-dense-ffn-profile/run_loopback.py \
  /Users/christopherhay/chris-models/qwen3-0.6b.vindex3 /tmp/wire2-qwen-new \
  --stream-comparison
python3 bench/v3-dense-ffn-profile/run_loopback.py \
  /Users/christopherhay/chris-models/gemma3-4b-it.q4k.vindex3 /tmp/wire2-gemma-new \
  --stream-comparison --port 19183
# The predefined replicate additionally used --blocks 1.
```

Evidence: [Qwen](results/wire-2-qwen3-0.6b-20260925/summary.tsv),
[Gemma](results/wire-2-gemma3-4b-20260925/summary.tsv),
[Gemma replicate](results/wire-2-gemma3-4b-replicate-20260925/summary.tsv).
Each directory includes the source revision, binary/artifact hashes, commands,
settings, bindings, stdout/stderr, worker logs, trial and bracket JSON, plus
`profiles.tar.gz` containing every raw JSONL trace. `trace-archive.json` records
archive and individual trace SHA-256 hashes, verified after writing;
`verification.json` records the parity and byte-accounting checks. Plain JSONL
copies remain local and ignored. No earlier dataset was overwritten.
