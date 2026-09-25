# GPT-OSS exact routed profile: provisional loopback result

In the bracket-passing blocks, one CPU worker adds **7.939 ms per decode
position** and two workers splitting experts within every layer add **1.223 ms**.
The corresponding FFN-boundary deltas are **7.903 ms** and **2.354 ms**.
These are workload-specific diagnostics, not promoted performance claims:
there is no all-peer exclusivity handshake, each layout has only one passing
block, and the frozen continuation is highly repetitive.

All **41 profiled CLI trials** matched the saved uninstrumented control's
151 prompt IDs and 64 generated IDs. This establishes token parity for this
workload, not bitwise equality of all real-model intermediates or output quality.
The separate fixture gates cover bitwise arithmetic and frame parity.

## Controls and limits

Measured September 25, 2026, on an Apple M3 Max, 16 logical CPUs, 128 GiB memory,
macOS 15.7.4. Both workers run on the same host over loopback; this is not a LAN
scaling experiment. Each process uses eight CPU workers/eight Rayon threads,
one Accelerate thread and two Tokio workers. The split changes available
concurrency and contention as well as placement.

Runtime source and CPU release binaries: `8e0c2bd8`. The uninstrumented control
is the saved `85e85100` binary, checked against its previous receipt before the
new build. Binary and artifact metadata hashes are in the manifests. GPT-OSS
20B supplies packed MXFP4 source banks; this CPU provider widens owned matrix
rows to F32 for execution. This does not measure a native packed-MXFP4 kernel.

The [frozen protocol](PROFILE-1.md) uses local / remote / local, warming each
arm with up to four trials and requiring successive means within 1%. Each
measured trial discards the first eight decode positions, leaving 55 positions
(159–213, zero-based), all beyond the 128-token sliding window. Loading,
tokenization, prefill and output are outside recorded decode wall time. Every
CLI trial is a fresh process; workers persist across their candidate arm and
stop before the next local arm. Full local and worker banks are never resident
simultaneously. Timings include optional instrumentation.

The 64-token continuation has only **six distinct IDs**, including long repeated
runs. Prompt and continuation were frozen before profiling; no replacement
was selected after observing timings. Repeated trials are not independent
language tasks and cannot establish general-chat throughput or model quality.
No hardware DRAM counters or physical network counters were collected.

## Brackets

All times are mean milliseconds per measured decode position. Control drift
uses full-precision nanoseconds and the symmetric relative difference. The
local reference for deltas is the mean of that block's two controls.

| Layout / block | Local before | Remote | Local after | Control drift | Timing gate |
|---|---:|---:|---:|---:|---|
| One worker / 0 | 115.974 | 124.251 | 116.651 | 0.582% | Pass |
| One worker / 1 | 114.169 | 119.801 | 116.431 | 1.962% | Void: controls |
| Expert split / 0 | 115.853 | 117.380 | 116.462 | 0.525% | Pass |
| Expert split / 1 | 116.372 | 117.453 | 113.185 | 2.777% | Void: warm-up and controls |

Void blocks are retained individually and never pooled. The last split control
failed to reach a warm-up plateau; its final measurement does not rescue that
gate. Full warm histories are in the audit receipt. Passing this table's gate
does not establish exclusivity. Descriptively, the passing remote samples are
8.05 and 8.52 decode positions/s, against their local references of about 8.60
and 8.61; these reciprocals exclude startup and prefill.

## Compute and synchronization

These are the two passing block-0 samples. Measurements are nested; summed
worker resource time must not be added to coordinator wall time.

| Metric, ms/position | One worker | Expert split |
|---|---:|---:|
| Local reference: expert transforms | 88.039 | 87.738 |
| Remote: all worker transforms summed | 86.391 | 131.159 |
| Remote: compute on last-finishing shard, summed across layers | 86.391 | 80.966 |
| Remote: maximum shard compute, summed across layers | 86.391 | 81.000 |
| Fan-out wall time | 95.946 | 90.104 |
| Last-shard completion offsets, summed across layers | 94.928 | 89.193 |
| Collection after final shard completion | 1.018 | 0.911 |
| Critical-shard HTTP round trip minus worker handler | 5.562 | 5.320 |
| Client encode + decode, all requests summed | 0.333 | 0.407 |
| Worker decode + encode, all requests summed | 0.526 | 0.784 |
| Worker blocking-pool queue, all requests summed | 0.249 | 0.550 |
| Local router | 0.196 | 0.198 |
| Local ordered reduction, including validation | 0.155 | 0.154 |
| Net FFN-boundary delta | +7.903 | +2.354 |
| Net full-position delta | +7.939 | +1.223 |

One-worker expert arithmetic takes roughly the same time as local arithmetic.
The split overlaps transforms and reduces critical-path compute, while total
worker resource time rises. Sharing the same CPU and memory system is a
plausible contributor; these timings do not isolate its cause. The split's
attention measured 21.106 ms versus 22.280 ms in its local reference, so about
1.174 ms of its lower total delta comes from attention timing, not remote FFN.

The transport remainder includes HTTP/runtime scheduling and timing-header
handling, not just network transit. Fan-out includes grouping, thread launch,
validation and collection. The latest completion offset, rather than join
order or largest compute duration, identifies the critical shard. Expert
transforms dominate these samples; this does not prove how many bytes came
from DRAM or predict performance on independent LAN hosts.

## Exact carrier bytes

Counts are HTTP body bytes only, excluding OPEN/binding, HTTP headers, TCP
overhead and telemetry headers. No quantization or prediction was used.

| Per measured position | One worker | Expert split |
|---|---:|---:|
| Requests | 24 | 46.109 average |
| Request body bytes | 277,824 | 533,405.091 average |
| Response body bytes | 1,107,264 | 1,108,148.364 average |
| Total body bytes | 1,385,088 | 1,641,553.455 average |

Hidden width is 2,880; four experts are selected at each of 24 layers. A shard
request is `40 + 4 × selected_count + 4 × hidden` bytes. A response is
`40 + selected_count × (4 + 4 × hidden)` bytes. Responses return every
unweighted expert output for deterministic local accumulation. Splitting adds
input-carrier copies for additional contacted shards, while expert-output
bytes stay fixed apart from extra headers. The analyzer checks these formulas
for every measured request.

## Receipts and validation

- [Single-worker receipt](profile-results/one-20260925/verification.json):
  19 trials; [blocks](profile-results/one-20260925/blocks.json).
- [Expert-split receipt](profile-results/experts-20260925/verification.json):
  22 trials; [blocks](profile-results/experts-20260925/blocks.json).
- [Uninstrumented IDs](profile-results/control/ids.json) and
  [combined audit](profile-results/audit.json).

The audit verifies every original receipt hash, replays the analyzer over all
41 raw profiles, checks IDs and position counts, and independently reconstructs
warm-up and bracket verdicts. Original receipts remain unchanged. Raw JSONL
files are preserved in `one-20260925-profiles.tar.gz` and
`experts-20260925-profiles.tar.gz` under `profile-results/`; each archive member
was checked against its original SHA-256. Extract either archive from that
directory to restore the corresponding receipt-relative paths.

The harness originally saved the final measured trial before attaching its
warm-up summary. Block verdicts and raw trials were correct; the separate audit
reconstructs every arm's summary. The harness now persists it immediately for
future runs. No runtime code changed after measurements.

Pre-run validation passed: 3 analyzer tests, 2 profile-capture tests, 30
production-expert tests, 1 HTTP client telemetry test, 23 V3 HTTP integration
tests, 20 V3 CLI tests, and clippy with warnings denied for the six changed
Rust crates. HTTP coverage includes profiled/unprofiled body equality and
remote/local bitwise fixture logits. CPU release build, formatting and
documentation link checks passed.

This closes the first exact routed profiling rung. Metal coordination, LAN
hosts, KDA/MLA and K3 latent operators remain separate work. These results give
them a measured CPU control without changing the exact provider contract.
