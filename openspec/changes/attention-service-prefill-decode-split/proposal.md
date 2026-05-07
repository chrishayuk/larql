## Why

The parent change `cuda-and-rotorquant-kv` plans an
`attention-service-routes` capability with `/v1/attention/session`,
`/v1/attention/prefill`, and `/v1/attention/decode` endpoints. As
designed today these all live in a single GPU shard role.

SMG (LightSeek) reports **20–30% TTFT improvement** by separating
prefill and decode onto **different worker pools**. Prefill is
compute-bound (large GEMMs across long sequences); decode is
memory-bound (single-token attention with large KV cache).
Co-locating them on the same GPU contends for compute units and
memory bandwidth in opposite ways.

This proposal extends the planned `attention-service-routes`
design to support optional PD-disaggregation: prefill is **stateless**
(returns a KV-snapshot blob); decode is **session-bound** (accepts a
snapshot to restore from). One container can run both modes; a
production deployment can run them on separate GPUs.

This is **not** on the critical path for the CUDA + RotorQuant
workstream — it's a deployment-topology refinement we revisit
once `attention-service-routes` ships its baseline (single-mode)
design and we have real session data flowing.

## What Changes

- MODIFY `server-attention-service` capability:
  - **`POST /v1/attention/prefill`** — body: token embeddings.
    Response: KV-snapshot blob + per-layer post-attention
    residuals. Stateless: no session id required, no in-VRAM
    state retained after the call.
  - **`POST /v1/attention/session`** with optional
    `restore_from_snapshot: bytes` — rehydrate the KV cache from
    the snapshot before decode begins.
  - **`POST /v1/attention/decode`** — unchanged from the baseline
    design (session-bound, returns one residual).
- ADD a `--mode` server flag with values `prefill | decode | both`
  (default `both` for backwards compat). `prefill` mode rejects
  decode RPCs; `decode` mode rejects prefill RPCs (returns 503 with
  a useful body).
- ADD a router-side hint: when
  `route_for_capability(_, _, "attention-prefill")` is requested,
  pick a shard whose role includes prefill; symmetric for
  `"attention-decode"`.
- DOCUMENT the deployment topology in
  `deploy/docker/README.md`: a high-throughput stack runs prefill
  workers on one GPU pool and decode workers on another; the
  router glues them together via the snapshot blob.

This is non-breaking against the planned
`attention-service-routes` design — it just adds knobs. Default
`both` mode collapses to the baseline behaviour.

## Capabilities

### New Capabilities

(none — extends the planned `server-attention-service` capability.)

### Modified Capabilities

- `server-attention-service`: adds requirements for the
  `--mode` flag, the stateless `/v1/attention/prefill`
  contract, and the `restore_from_snapshot` parameter on
  session create.
- `router-grid`: adds optional `"attention-prefill"` and
  `"attention-decode"` capability sub-tags so the router can
  honour the split.

## Impact

- **Affected files**: requires `attention-service-routes` to land
  first; this change layers on top of those route definitions.
- **Affected systems**: server, router. No GPU code changes.
- **Provenance**: derived from SMG's PD-disaggregation pattern.
  We don't vendor their code; the architectural pattern is
  well-documented in the literature (Sarathi-Serve, DistServe).
- **Out of scope**: actual SLA-driven autoscaling of the
  prefill / decode pools (that's a Kubernetes / Fly autoscaler
  concern); micro-batching across requests within a prefill worker
  (in scope for a future `attention-prefill-microbatching` change).
