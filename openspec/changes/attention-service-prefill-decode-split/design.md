## Context

The parent change designs `attention-service-routes` with three
session-state endpoints (create / prefill / decode) plus
KV-cache snapshot / restore. The session is server-side; prefill
and decode both happen against the same in-VRAM cache.

SMG observed (and the academic literature confirms — Sarathi-Serve
2024, DistServe 2024) that prefill and decode have orthogonal
resource profiles:

|  | prefill | decode |
|---|---|---|
| Compute | high (O(n²) GEMM) | low (O(n) gemv) |
| Memory bandwidth | low | high (one KV read per token) |
| Latency target | TTFT (one-shot) | tok/s (steady-state) |
| Batching | within request | across requests |

Co-locating the two on a single GPU forces a compromise. SMG
reports 20–30% TTFT win when the prefill side is a separate pool.

The architectural change to support both modes:
- Prefill becomes **stateless** — it accepts tokens, returns
  residuals + a snapshot blob. No session needed.
- Decode becomes **session-bound** — it accepts a snapshot at
  session create time, holds it in VRAM, and serves decode RPCs
  against it.

The snapshot blob is the natural handoff. The parent change
already specs `/v1/kv-cache/snapshot` and `/v1/kv-cache/restore`;
this change just makes them the wire format between PD pools.

## Goals / Non-Goals

**Goals:**
- A single binary that can run as `--mode prefill`,
  `--mode decode`, or `--mode both` (default).
- Stateless prefill: accept tokens, return KV-snapshot blob.
- Session create accepts an optional `restore_from_snapshot` to
  initialise from prefill output.
- Router can route prefill RPCs to one pool and decode to another
  via a sub-capability tag.

**Non-Goals:**
- Microbatching prefill across requests (own change).
- Predictive co-location heuristics (queue-length-based migration
  between modes — out of scope; for a future change).
- Cross-region snapshot transport (snapshot blobs are large and
  best kept within a single rack).

## Decisions

### D1 — Snapshot blob format reuses the parent change's

`/v1/kv-cache/snapshot` already defines a wire format. The
PD-split case just runs that endpoint immediately after prefill
returns. We don't introduce a parallel format.

### D2 — `--mode prefill` rejects decode with HTTP 503

Returning 503 with a body containing
`{ role: "prefill", missing: "decode" }` lets the router
rebalance gracefully. 405 Method Not Allowed would be technically
correct but less helpful — the route exists, it's just not served
by this instance.

### D3 — Session create accepts the snapshot inline

Body of `/v1/attention/session`:

```json
{
  "model": "gemma-3-4b",
  "kv_format": "iso3",
  "max_seq_len": 8192,
  "restore_from_snapshot": "<base64 snapshot blob>"   // optional
}
```

The blob can be passed as base64 (smaller payloads) or via a
binary subroute (`/v1/attention/session/{id}/restore`) for blobs
> 10 MB. Current default: base64 inline; binary path lands when
the first user hits the size cap.

### D4 — Router gets two sub-capability tags

`"attention-prefill"` and `"attention-decode"` in addition to the
existing `"attention"`. A `--mode both` shard advertises all three;
a `--mode prefill` shard advertises only `"attention" + "attention-prefill"`.
Clients call `route_for_capability(_, _, "attention-prefill")` to
get a prefill-capable shard.

### D5 — Snapshot transport over loopback first, then network

Initial deployment: prefill and decode pools on the same host (two
processes, two GPUs). Snapshot transport stays in-process via
shared memory (a future fast-path optimisation). Network transport
becomes relevant when the pools are on different boxes, at which
point we revisit serialisation.

## Risks / Trade-offs

- **Risk: snapshot serialisation overhead eats the prefill win.**
  Snapshot is ~50 MB for a 4B model with 1k context at iso3
  compression. At 1 GB/s network it's 50 ms — significant fraction
  of a 200 ms prefill. → Mitigation: shared-memory fast-path on the
  same host; reserve network transport for cross-box deployments
  where the latency budget is bigger anyway.
- **Risk: imbalanced PD pools waste hardware.** A workload skew
  (long contexts → prefill bound; many users → decode bound) can
  leave one pool idle. → Mitigation: defer to autoscaling;
  document in `deploy/docker/README.md` that operators tune pool
  ratios per workload.
- **Risk: client-side code knows about PD split.** → Mitigation:
  the router hides the split. Clients hit `route_for_capability(_,
  _, "attention")`; the router synthesises the prefill+decode
  sequence under the hood. Clients see one logical session.

## Migration Plan

Land **after** `attention-service-routes` ships. Default
`--mode both` collapses to the baseline behaviour; nothing
breaks. Operators opt into the split by running two pools and
configuring the router with both sub-capability tags.

Rollback: revert. The baseline single-pool design keeps working.

## Open Questions

- **Q1: Should the snapshot blob include the chat template's
  cached prefix tokens?** Tokenizer cache + prefix-aware routing
  + PD-split together suggest a `BootstrapBundle` that ships
  cached tokens + KV snapshot. → Defer to a future
  `attention-bootstrap-bundle` change once we have at least two
  of the three components live.
- **Q2: How long is a snapshot valid?** When the underlying
  vindex changes (new patches, model swap), every snapshot
  becomes invalid. → Snapshot includes a `vindex_version` field
  in its header; restore rejects a snapshot whose version
  doesn't match the loaded vindex.
