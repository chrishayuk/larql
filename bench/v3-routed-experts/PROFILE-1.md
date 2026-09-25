# Exact routed profiling

Execution was authorized after the draft hold. Validation and real-model
measurements follow the protocol below. The earlier correctness receipts in
this directory remain separate from the profiling results.

Completed measurements: [GPT-OSS loopback results](PROFILE-1-results.md).
External-worker setup and the next measurement: [physical LAN protocol](LAN-1.md).

## Measurement contract

Keep binary HTTP, bind-once authority, exact F32 frames and expert arithmetic.
`--v3-profile` gains routed-operation records and explicitly labels remote
routed placement. No new representation, stream, batching or scheduling policy
is selected by the profiler. Optional worker timings travel in
`x-larql-expert-profile`; input and output frame layouts remain unchanged.

Each measured position records attention, FFN, residual re-entry, other and
total time through the existing decoder profile. Each routed layer additionally
records local routing, selected-transform dispatch, local ordered reduction
(including validation), and local transform compute for the local control.

Each selected shard records input/output body bytes, client encoding/decoding,
HTTP round trip, worker decode, blocking-pool queue, transform compute, worker
execution including validation, encoding and handler time. Missing worker
timings invalidate analysis; they never become zero compute. Diagnostics are
not execution authority.

Dispatch start and finish offsets use the coordinator's monotonic clock,
relative to entry into that layer's fan-out. The latest finish identifies the
critical shard, independently of which worker does the most compute. Parent
collection order cannot select the critical shard. Worker durations use that
worker's clock; no clock synchronization between hosts is assumed.

| Metric | Interpretation |
|---|---|
| `worker_experts_sum_ns` | Sum of all shard transform durations: resource time, not token wall time |
| `worker_experts_max_ns` | Sum across layers of the largest shard compute duration per layer |
| `critical_worker_experts_ns` | Compute on the shard that finishes last in each layer |
| `slowest_shard_wait_ns` | Sum across sequential layers of the latest shard completion offset |
| `fanout_wall_ns` | Grouping, thread launch, request completion, validation and collection |
| `fanout_post_completion_ns` | Parent collection and instrumentation after the final dispatch returns |
| `critical_transport_remainder_ns` | Critical-shard round trip minus its handler duration |
| `net_total_delta_ns` | Candidate position time minus the mean of agreeing local controls |
| `net_ffn_delta_ns` | Same difference at the FFN boundary |

These are nested, often overlapping measurements. Do **not** add aggregate
worker time to token wall time. The transport remainder includes HTTP/runtime
scheduling and timing-header handling; it is not a direct measurement of
physical network latency. The net distribution delta also includes changes in
parallel compute and contention. No hardware DRAM counters are collected, so
transform time alone cannot establish the volume of DRAM traffic.

## Run design

First format/build and run the prepared tests. Then build CPU release binaries.
Use the 151-token prompt and 64 generated tokens, discarding the first eight
decode positions. Every measured position is beyond the 128-token sliding
window. Every trial must retain identical prompt and generated IDs.

The harness runs warmed **local / candidate / local** blocks. Each arm gets
at least two identical warm trials, up to four until consecutive per-position
means agree within 1%. It then runs a measured trial. A block is void if any arm
fails to reach that plateau or the two measured local controls differ by more
than 1%. The gate uses unrounded nanoseconds. Void blocks remain individual
diagnostics and are never pooled into a promoted result.

Local controls and worker banks are never resident simultaneously. Worker
startup/shutdown and CLI weight preparation are outside per-position timings.
The changing process residency is part of this protocol and must be disclosed.
Optional saved peer acknowledgements are retained in the manifest; supplying
a file does not by itself prove exclusivity. Review the acknowledgements for
every peer session and the entire measurement window before making a claim.

Command for an authorized measurement window:

```bash
python3 bench/v3-routed-experts/profile_exact.py --execute \
  --model /path/to/gpt-oss-20b.vindex3 --out /tmp/gpt-oss-routed-profile \
  --topology experts --blocks 2 --reference /path/to/uninstrumented-ids.json \
  --peer-handshake /path/to/peer-acknowledgements.txt
```

`one` and `mixed` are separate candidate layouts, each requiring its own
brackets. This first profile should compare one worker with the two-way expert
split before considering wire changes. No automatic build or model execution
occurs on module import or without `--execute`.

## Validation scope

- Format and clippy for the changed Rust crates.
- Profile capture thread-isolation tests.
- Production routed numerical and placement tests.
- HTTP integration: profiled remote logits versus unprofiled local logits,
  exact frame byte counts, nested timing bounds and prior failure semantics.
- HTTP client: missing optional telemetry preserves output and remains missing.
- Analysis tests: distinguish aggregate compute, maximal compute and the last
  completing shard; reject missing timing and use unrounded bracket gates.
- CLI profile/parity regression tests and a real-model instrumentation control.

Validation outcomes and timing results are recorded separately with their
source and binary identities. A protocol description alone is not a test result.

The pre-measurement checks passed: three analysis tests, two capture tests,
30 production-expert tests, the HTTP client corruption test, 23 V3 HTTP tests,
20 V3 CLI tests, and clippy with warnings denied across six changed crates.
The HTTP test also compares profiled and unprofiled response bodies byte for
byte. Real-model instrumentation parity uses `--reference` from a saved
uninstrumented binary, independently of the profiled local timing controls.
