## Context

`larql-router` registers shards via a gRPC AnnounceMsg, builds a
`(model_id, layer) → [server_id]` route table, and dispatches via
`GridState::route()` (least-loaded among replicas).

The parent change's design says shards declare capabilities and the
router routes by capability. Today every shard implicitly handles
both attention and expert RPCs. Splitting requires a place to
filter. This change adds that place.

## Goals / Non-Goals

**Goals:**
- A `capabilities: Vec<String>` slot on `ServerEntry`.
- A capability-filtered route fn that picks the least-loaded shard
  whose capability set covers the requested op.
- Backwards-compat: a shard that registered without capabilities
  defaults to "everything" and continues to work.

**Non-Goals:**
- Touching the AnnounceMsg proto. Real shards still register without
  capabilities until `attention-service-routes` lands.
- Per-hop deadline. The 5 s timeout the parent design talks about
  belongs in the dispatch path, not the registration path.
- HTTP-side filtering of incoming requests. The router currently
  routes by layer only; extending the request handlers to pick a
  capability based on the URL (`/v1/attention/*` → "attention") is
  a separate change.

## Decisions

### D1 — `Vec<String>` over an enum

Three options:

1. **Enum**: `pub enum Capability { Attention, Expert }`.
   Type-safe, but every new capability requires touching this
   crate.
2. **Bitflags**: tight memory, but unfriendly for log lines and
   the proto extension we'll do next.
3. **`Vec<String>`**: lossy but flexible. New capabilities (e.g.
   `"rotorquant-iso3"`) can be declared without code changes.

Chose `Vec<String>`. The router doesn't enforce known names; it
just filters. Type-safety on the consumer side is the announce-msg
extension's concern.

### D2 — Default to "everything" for back-compat

Shards announced before this change have no capability info on the
wire. The grid keeps them working by defaulting their entry to
`{"attention", "expert"}`. A future change that wants "deny by
default" semantics can switch the default — but explicitly, with
its own design rationale.

### D3 — `route_for_capability` is additive, not replacing

`route()` stays as-is. The new method is the right entry point for
capability-aware dispatch. Existing call sites can migrate
incrementally.

## Risks / Trade-offs

- **Risk: drift between `route` and `route_for_capability`**. Two
  near-identical methods can drift. → Mitigation: kept the second
  method tiny by reusing the same lookup table; the only delta is
  the `.filter()` on capability.
- **Risk: case-sensitivity confusion.** `supports("attention")`
  vs `supports("ATTENTION")` should match. → Mitigation: the impl
  uses `eq_ignore_ascii_case`.
- **Risk: empty capabilities list silently drops the shard from
  every filter.** → Mitigation: defaulting to "everything" on
  registration prevents this; explicit empty lists are a future
  attention-service-routes concern (where they MAY be intentional).

## Migration Plan

Land. Existing call sites use `route` and continue to work. The
attention-service-routes change adds the proto field, populates
real capability sets at announce time, and migrates the HTTP
handlers to call `route_for_capability` based on the request URL.

Rollback: revert. `route` and the test suite keep passing.

## Open Questions

- **Q1: Should default capabilities be configurable?** A future
  cluster operator might want `route("attention")` to fail closed.
  Recommendation: env var override on `larql-router` startup —
  follow-up if/when needed.
