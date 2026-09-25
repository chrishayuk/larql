# Exact routed experts on physical LAN workers

Status: harness prepared; physical hosts have not been supplied and no LAN
model runs have been performed. The coordinator remains CPU-only for this
experiment. Metal coordination is a separate implementation and control.

The question is whether the worker critical path decreases on independent
hosts, and how much real LAN transport adds back. Compare each LAN sample with
its own warmed local / candidate / local bracket. The previous loopback result
is a historical diagnostic, not a matched control for a new machine or day.
Direct LAN-versus-loopback attribution would require contemporaneous controls.

## Worker setup

Start LAN-1 on a fresh branch from the merged distributed-FFN baseline on
`main`, and use the same recorded source revision on coordinator and workers.
`8e0c2bd8` identifies the historical loopback measurement runtime; its receipts
remain unchanged. Build for each host's target; different platforms have different
binary hashes. Preserve source revision, binary hash, CPU, memory, OS, thread
settings, model metadata hashes and the NIC/link configuration for each host.
Binding equality does not establish physical host identity, CPU ISA parity,
hardware exclusivity or binary provenance.

Prepare the same GPT-OSS 20B VINDEX3 artifact on both hosts. These CPU workers
widen owned MXFP4 matrix rows to F32, so size RAM for prepared weights, not just
the packed source files. Start the workers on their LAN interfaces with the
same process settings as the loopback control:

```bash
# Worker A: replace LAN_ADDRESS_A and /path/to/model.vindex3.
LARQL_CPU_WORKERS=8 RAYON_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=1 \
TOKIO_WORKER_THREADS=2 LARQL_KV_ENGINE= \
target/release/larql-server /path/to/model.vindex3 \
  --host LAN_ADDRESS_A --port 9181 --ffn-only --layers 0-23 --experts 0-15

# Worker B: replace LAN_ADDRESS_B and /path/to/model.vindex3.
LARQL_CPU_WORKERS=8 RAYON_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=1 \
TOKIO_WORKER_THREADS=2 LARQL_KV_ENGINE= \
target/release/larql-server /path/to/model.vindex3 \
  --host LAN_ADDRESS_B --port 9181 --ffn-only --layers 0-23 --experts 16-31
```

Record peer-session exclusivity acknowledgements for the coordinator and both
worker hosts over the measurement window. Each worker has its own eight-thread
pool; the harness sets coordinator environment variables only. External worker
processes stay resident throughout the brackets and their lifecycle belongs to
the operator. This differs from loopback, where the harness stops workers
before loading the full local control to limit shared-host memory residency.

## Coordinator run

Use the harness and runtime from that recorded merged baseline. This preserves
exact binary HTTP and F32 carriers, local routing, and production-order
reduction. Token parity against the saved control remains a required gate.

```bash
python3 bench/v3-routed-experts/profile_exact.py --execute \
  --model /path/to/model.vindex3 --out /path/to/new-lan-results \
  --topology experts --blocks 2 \
  --worker-url http://LAN_ADDRESS_A:9181 \
  --worker-url http://LAN_ADDRESS_B:9181 \
  --reference bench/v3-routed-experts/profile-results/control/ids.json \
  --peer-handshake /path/to/peer-acknowledgements.txt
```

URLs follow topology order: experts 0–15 first, then 16–31. Before starting any
local model control, the harness fetches and saves the advertised bindings and
checks their shape, ownership ranges and mutual program identity. It checks
the complete advertisements again before each candidate arm. The Rust client
still validates the worker against the local artifact and numerical provider
when opening execution. The harness never substitutes its checks for that
authority, never starts/stops external workers, and never silently falls back
to loopback. URL hostnames alone are not proof of distinct physical hosts.

The manifest records external endpoints and only the local CLI binary hash;
remote binary/environment provenance must be recorded separately. A saved
handshake file requires manual review and cannot auto-promote a timing claim.
An unreachable/mismatched worker, absent timing telemetry or differing token
IDs stops the run and retains diagnostics. Cross-architecture numerical
differences must be investigated rather than relaxing parity after the fact.

## Interpretation

Retain the [existing profiling protocol](PROFILE-1.md): frozen 151-token prompt,
64 generated IDs, 55 measured decode positions, exact body-byte checks,
warm-up and full-precision 1% bracket gates. The continuation is repetitive;
results apply to this workload. Void blocks are never pooled.

Report total and FFN-boundary deltas separately. Include last-shard completion,
critical-shard expert compute, aggregate worker compute, local reduction,
request/response bytes and critical HTTP round trip minus handler time. The
latter includes networking and runtime scheduling, not pure NIC latency.
Compute overlap is measurable; DRAM read volume needs hardware counters.
The same CPU settings across different CPU types do not isolate memory
bandwidth as the cause of a worker speed change.

No Q8, prediction, changed expert arithmetic or Metal coordinator belongs in
this first comparison. Those require separately identified arms and controls.
