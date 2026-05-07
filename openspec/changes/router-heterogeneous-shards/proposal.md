## Why

Sixth sub-change of [`cuda-and-rotorquant-kv`](../cuda-and-rotorquant-kv/proposal.md).
The two-container topology (CPU FFN + GPU attention) only works if
the router can dispatch attention RPCs to GPU shards and FFN RPCs to
CPU shards — without a way to filter by what each shard can serve
the router has to assume every shard handles every RPC.

This change lands the **router-side** half of the heterogeneous-shards
contract: `ServerEntry` gains a `capabilities` field, and a new
`route_for_capability` method returns a least-loaded shard that
advertises the requested capability on the requested layer. The
**announce-side** half (extending `AnnounceMsg` so real shards can
declare their capability set) lands with the
`attention-service-routes` change where the GPU shard actually
needs to declare itself as `["attention"]` only.

Backwards-compat is preserved: shards that announce without an
explicit capability set get the default `["attention", "expert"]`,
which means they continue to receive every RPC they did before.

## What Changes

- ADD `ServerEntry::capabilities: Vec<String>` field.
- ADD `ServerEntry::default_capabilities()` returning `["attention",
  "expert"]` and `ServerEntry::supports(cap)` for case-insensitive
  membership check.
- ADD `GridState::route_for_capability(model_id, layer, capability)`
  — a capability-filtered variant of `route()`.
- MODIFY both existing `ServerEntry { ... }` construction sites
  (announce handler in `grid.rs`, test helper) to default the new
  field to `default_capabilities()`.
- ADD four inline tests covering:
  - default capability set advertises both,
  - capability filter routes correctly between an FFN-only and an
    attention-only shard,
  - missing capability returns `None`,
  - legacy default-caps shard matches both filters (back-compat).

This is non-breaking. The `route()` method is unchanged; the new
`route_for_capability` is purely additive. The proto definitions
are not touched (that's the `attention-service-routes` change's
remit).

## Capabilities

### New Capabilities

(none — implements scenarios already on the parent change's
`router-grid` capability.)

### Modified Capabilities

- `router-grid`: scenarios for capability-tagged shards / capability
  filtering / backwards-compat default get real test annotations on
  `larql_router::grid::tests::*`.

## Impact

- **Affected files**: `crates/larql-router/src/grid.rs` (~70 added
  lines incl. tests).
- **Affected systems**: router-only. CPU/server/inference paths
  untouched.
- **Out of scope**: extending `AnnounceMsg` to carry capabilities
  on the wire (with `attention-service-routes`); per-hop deadline
  enforcement (that's an additive change once we have real
  attention RPCs flowing).
