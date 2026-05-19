# ADR-0018 — MoE Expert-Level Routing

**Status:** Proposed — implementation pending.
**Depends on:** ADR-0003 (router base), ADR-0004 (grid),
ADR-0011 (replication), ADR-0013 (routing comparator).
**Affects:** `larql-router-protocol`, `larql-router/src/grid/*`,
`larql-router/src/http.rs`, `larql-router/src/dispatch.rs`.

---

## Context

The grid today routes at **layer granularity**: a server owns a
contiguous range `[layer_start, layer_end]` of a model. For dense
models (Gemma 3 4B, Llama 3) every layer is monolithic, so a server
owning layers 0-14 handles 100% of those layers' FFN compute.

MoE models break that assumption. A single layer in a Mixture-of-
Experts model contains N experts (e.g. 8 in Mixtral, 64 in DeepSeek-V2).
At inference time a gate scorer picks a sparse subset (typically
top-2 to top-8) of experts per token; only the picked experts run.

Two consequences for the router:

1. **An MoE shard can be smaller than "one layer."** A server might
   own experts 0-7 of layers 0-14 (a quarter of the FFN compute for
   that layer range). Layer ownership is no longer a complete shard
   description.
2. **The hot path becomes per-token fan-out.** For each token the
   router needs to dispatch N independent sub-requests (one per
   picked expert) and merge the results. This is where ADR-0010
   (QUIC) HTTP/3 stream independence finally matters — HoL blocking
   on a single multiplexed stream serialises the fan-out.

Status quo: routing has no concept of experts. An MoE model would
have to be served as a single shard per layer range with all experts
co-located on the same host. That defeats the whole point of MoE
sharding (load balancing distinct experts across distinct hardware).

---

## Decision

### Proto extension: contiguous expert range alongside layer range

`AnnounceMsg`, `ReadyMsg`, and `AssignMsg` grow two optional fields:

```proto
message AnnounceMsg {
  string model_id    = 1;
  uint32 layer_start = 2;
  uint32 layer_end   = 3;
  uint64 ram_bytes   = 4;
  string listen_url  = 5;
  string vindex_hash = 6;

  // ADR-0018 — MoE expert ownership. Both default to 0 when the
  // server is dense (every layer is monolithic). For MoE shards
  // these advertise the *contiguous expert ID range* this server
  // owns *across all layers it covers*. expert_count==0 means
  // "dense", expert_count>0 means "MoE shard."
  uint32 expert_start = 7;  // inclusive (only meaningful when expert_count>0)
  uint32 expert_end   = 8;  // inclusive (only meaningful when expert_count>0)
}
```

Same two fields on `ReadyMsg` and `AssignMsg`. `ServerEntry` mirrors
them as `expert_start: u32, expert_end: u32` with `expert_count() ->
u32 = if expert_end >= expert_start && (expert_start != 0 ||
expert_end != 0) { expert_end - expert_start + 1 } else { 0 }`.

The **contiguous range** choice (vs `repeated uint32 expert_ids`) is
the same call we made for layers — uniform sharding is the
overwhelmingly common case, and arbitrary-set ownership can be
expressed by registering multiple ranges from the same server (one
announce per non-contiguous sub-range).

### Single dimension of ownership-per-server, not a matrix

A server owns one contiguous expert range across one contiguous layer
range. Two patterns this covers naturally:

**Pattern A — small/mid MoE, shard by expert across full layer range:**

```
server-a: layers 0-14, experts 0-3   (a "quarter slice" of layers 0-14)
server-b: layers 0-14, experts 4-7   (the other quarter slice)
server-c: layers 15-29, experts 0-7  (the second half, all experts)
```

**Pattern B — large MoE, shard by (single layer, expert subset):**

For large MoE models (DeepSeek-V2 236B with 60 layers × 160 experts,
Mixtral 8x22B with 32 layers × 8 experts, etc.) a single host typically
has RAM for only **one layer's worth** of expert weights. The
deployment pattern is `layer_start == layer_end` per server, with
different hosts owning different expert subsets of the same layer:

```
server-a:  layer_start=12, layer_end=12, expert_start=0,   expert_end=39
server-b:  layer_start=12, layer_end=12, expert_start=40,  expert_end=79
server-c:  layer_start=12, layer_end=12, expert_start=80,  expert_end=119
server-d:  layer_start=12, layer_end=12, expert_start=120, expert_end=159
server-e:  layer_start=13, layer_end=13, expert_start=0,   expert_end=39
...
```

This is the **canonical large-model pattern**, not an edge case.
Operators announce one shard per (layer, expert-range) tuple. A 60-
layer × 4-expert-shards-per-layer DeepSeek-V2 deployment is 240
distinct `ServerEntry` records — possibly fewer physical hosts if
each host has enough RAM for multiple shards, in which case the host
opens multiple `Join` streams (one per shard).

What's **rejected**: a layer-by-layer expert map from a single server
(server owns experts {1,3,5} on layer 0, {2,4,6} on layer 1, …).
That flexibility isn't needed because operators always split big
models into many small shards anyway — each shard fits the simple
`(layer_start..=layer_end, expert_start..=expert_end)` shape.
Allowing the matrix form would complicate the route table by an
order of magnitude (`(model_id, layer, expert_id) → server_ids` vs
`(model_id, layer) → server_ids` post-filtered by
`expert_start..=expert_end`) without earning anything operationally.

### Routing: two new accessors, no comparator change

```rust
impl GridState {
    /// Existing API, unchanged. For MoE models this returns the
    /// owner of layer L *ignoring expert ownership*. Useful for
    /// "any host that has this layer's weights" queries (rare).
    pub fn route(&self, model_id: Option<&str>, layer: u32) -> Option<String>;

    /// ADR-0018 — pick the best replica that owns `(layer, expert_id)`.
    /// For dense models (expert_count == 0 on every ServerEntry for the
    /// layer) returns the same answer as `route()`. For MoE models
    /// filters the candidate set to servers where
    /// `expert_start <= expert_id <= expert_end`.
    pub fn route_expert(
        &self,
        model_id: Option<&str>,
        layer: u32,
        expert_id: u32,
    ) -> Option<String>;

    /// Batched form. Returns Ok(layer_expert -> url) or Err((layer, expert))
    /// for the first uncovered pair.
    pub fn route_all_experts(
        &self,
        model_id: Option<&str>,
        layer_experts: &[(usize, u32)],
    ) -> Result<HashMap<(usize, u32), String>, (usize, u32)>;
}
```

The three-tier comparator from ADR-0013 (GT3 → RTT → in-flight)
applies unchanged. The only difference is which replicas are in the
candidate set — `route()` includes every replica covering `layer`,
`route_expert()` filters to replicas where `expert_start..=expert_end`
contains `expert_id`.

### Route table layout: layered, not expert-keyed

The internal `route_table: HashMap<(model_id, layer), Vec<server_id>>`
stays keyed on `(model_id, layer)`. Expert filtering happens
**inside** `route_expert` by iterating the layer's `Vec<server_id>`
and rejecting entries whose expert range doesn't cover the requested
expert. Reasons:

- A model with 8 experts × 30 layers × 1 model_id × 3 replicas would
  generate 8 × 30 × 3 = 720 keys at `(model, layer, expert)`
  granularity vs the current 30 × 3 = 90 keys at `(model, layer)`.
  Memory matters less than the rebuild_route_table cost, which is
  already O(N × L) per join — multiplying by `expert_count` would
  push N=100 servers × L=62 layers × E=8 experts to 49,600 inserts
  per join.
- Most candidate sets are small (target_replicas × experts_per_shard).
  Linear filter through `Vec<server_id>` of size ~3-10 is faster than
  hash lookup over expert ID, especially with the routing comparator's
  per-candidate work.
- The comparator already iterates the candidate set, so the filter
  costs effectively nothing in the hot path.

### HTTP shape: extend `/v1/walk-ffn`, no new endpoint

Existing JSON request body (today):

```json
{ "layer": 5 }                      // single layer
{ "layers": [5, 6, 7] }             // multi-layer fan-out
{ "layer": 5, "model_id": "..." }   // optional model selector
```

New optional MoE shape (additive):

```json
{ "layer": 5, "experts": [0, 3, 7] }              // single layer, picked experts
{ "layer_experts": [{"layer": 5, "experts": [0, 3]}, {"layer": 6, "experts": [1, 5]}] }
```

When the body has `experts` or `layer_experts`:
- Router resolves each `(layer, expert)` pair via `route_expert`.
- Groups by destination URL (same as today).
- Builds the sub-request body with `{layer: L, experts: [E, ...]}`
  for that shard's pairs.
- Merges responses identically.

When the body has no `experts` field, the existing dense path runs
unchanged — the router never invokes `route_expert`.

### Binary protocol: stays single-dimension for now

The current binary peek decodes `[layer]` or `[BATCH_MARKER, count,
layer_0, …, layer_{count-1}]`. Adding expert IDs would mean a v2 wire
format. Rather than break compatibility today:

- Binary requests stay **dense-only**. An MoE request must use JSON.
- ADR-0009 (wire-format evolution) covers the future v2 binary format
  with expert IDs as a separate spec.

This is the documented gap, not a bug — most clients are JSON anyway,
and the binary protocol is a perf optimisation for FFN-heavy dense
walks where the wire-byte savings dominate.

### Replication: per-(layer-range, expert-range)

`under_replicated_ranges()` and `over_replicated_ranges()` already
key on `(model_id, layer_start, layer_end)`. Extend the key to
`(model_id, layer_start, layer_end, expert_start, expert_end)`. Two
servers that own different expert ranges of the *same* layer range
are **distinct shards** for replication purposes — pulling a spare
to back up server-a (experts 0-3) doesn't fill the gap if server-b
(experts 4-7) drops.

`find_origin_for(model_id, layer_start, layer_end)` extends to
`find_origin_for(model_id, layer_start, layer_end, expert_start,
expert_end)` so Mode B assigns the right slice.

### Hot-shard elevation: per-expert-range too

`hot_layer_ranges` already returns `Vec<(model_id, u32, u32)>`. The
new form returns `Vec<(model_id, u32, u32, u32, u32)>` with the
expert range appended. `elevated_ranges: HashSet<(String, u32, u32,
u32, u32)>`. Elevation can fire on the (layers 0-14, experts 0-3)
shard alone without affecting the (layers 0-14, experts 4-7) shard.

### Expert affinity: deferred

A natural extension is "route same `expert_id` to the same host
repeatedly so the host's expert MLP cache stays warm." This is a
real optimisation but adds state (per-`expert_id` last-routed-host
map) and a fourth tier in the comparator.

**Not in this ADR.** Ship expert routing with the existing 3-tier
comparator; affinity is a future ADR if real workload data shows
the cache-warmth signal is meaningful.

---

## Target deployments

The design pressure for ADR-0018 is **trillion-parameter MoE
models** that no single host can hold: Kimi K2 / K2.6 (~1T-class,
~128 experts), DeepSeek-V3 (671B total / 37B active, 256 experts per
layer × 61 layers), DeepSeek-V4 (post-V3, ≥1T expected). These have
two structural traits that constrain the router:

1. **Per-layer expert count is large** — 128 to 256+. Sharding by
   expert range is the only way to fit a layer.
2. **Number of layers × experts is in the thousands** — a V3-style
   deployment with 4-expert-shards-per-layer × 61 layers = 244
   distinct expert shards. K2.6 with 8-expert-shards × ~80 layers =
   ~640 shards. The route table must stay tractable.

### V3-class deployment sketch

For DeepSeek-V3 with 256 experts per MoE layer and a 4-way expert
split per layer:

```
host-001:  layer_start=0,  layer_end=0,  expert_start=0,   expert_end=63
host-002:  layer_start=0,  layer_end=0,  expert_start=64,  expert_end=127
host-003:  layer_start=0,  layer_end=0,  expert_start=128, expert_end=191
host-004:  layer_start=0,  layer_end=0,  expert_start=192, expert_end=255
host-005:  layer_start=1,  layer_end=1,  expert_start=0,   expert_end=63
...                          (61 layers × 4 hosts/layer = 244 shards)
```

With `target_replicas=2` for survivability: 488 hosts total. The
route table holds `(model_id, layer) → Vec<server_id>` with 61
entries × ~8 server_ids per entry (4 shards × 2 replicas) — small
HashMaps, fits cleanly.

### Attention is bundled with experts, not its own shard kind

Real MoE models have **attention + FFN** per transformer block. The
attention sub-layer is dense (every token, same weights) while the
FFN sub-layer is routed (top-K experts per token).

The router models this by treating attention weights as **co-resident
with whichever expert shards live on the host**. Every server that
owns a piece of layer N's MoE FFN also has layer N's full attention
weights loaded — attention is comparatively cheap (a single
`hidden_size × hidden_size`-ish matrix per layer vs N expert MLPs)
so replicating attention across all of a layer's expert-shards is
the standard trick (DeepSpeed-MII, vLLM, etc.).

The proto and routing don't need a separate "attention shard" concept.
A `walk-ffn` call for `(layer N, experts {E1, E2})` dispatches to the
expert-shard owners; each owner runs its expert's MLP path. Attention
runs in the upstream / downstream of the FFN call, on whichever host
holds the residual stream at that moment (typically the host serving
the first MoE layer of the block).

### Per-token fan-out cost at K2.6 scale

A K2.6 forward pass might pick top-8 experts per MoE layer across 80
layers. Naive routing would be **80 × 8 = 640 sub-requests per
token**. Two mitigations baked into this design:

1. **Layer batching.** The `layer_experts` JSON shape lets one
   `/v1/walk-ffn` call carry multiple (layer, expert-list) tuples
   to the same shard host. If `host-001` owns `(layer=0, experts
   0-63)` AND `(layer=20, experts 0-63)`, a single call delivers
   both layers' work for whichever experts the gate picked.
2. **Expert affinity (future, ADR TBD).** Sticky routing reuses the
   same host for the same expert_id across consecutive tokens.
   Lowers cache-miss cost; doesn't reduce raw call count.

For K2.6 specifically, the dominant cost is **wire round-trip across
80 layers**. ADR-0010 (QUIC) HTTP/3 multi-stream independence is the
unblock — the 8 expert sub-requests per layer can issue as 8 parallel
streams without HoL stalls. This is the **real** reason HTTP/3 sits
on the P3 list; the dense path tolerates HTTP/2-over-QUIC fine, but
the MoE fan-out doesn't.

## Backward compatibility

### Dense routing is unchanged

A server that announces with `expert_start = 0, expert_end = 0`
(the proto3 default) is **dense**. Every helper path special-cases
this:

- `route_expert(model, layer, expert)` falls through to `route` if
  every candidate `ServerEntry` for that layer has `expert_count() == 0`.
- The new HTTP fields (`experts`, `layer_experts`) are optional;
  bodies without them go straight to the existing dense dispatch.
- `under_replicated_ranges` and friends with `(0, 0)` expert ranges
  are equivalent to the old `(model_id, layer_start, layer_end)`
  keys.

### Dense-only regression test surface

The implementation includes:

1. A regression test that runs the existing `routing.rs` bench
   suite (production-shape `route()`, `route_all`, `register`,
   etc.) and confirms numbers are within ±10% of the pre-ADR
   baseline. Filter through `route_expert` adds a constant factor
   even on dense paths — must stay negligible.
2. A regression test that exercises every existing integration
   path (`tests/test_grid_service.rs`, `test_admin_rpcs.rs`,
   `test_http_handlers.rs`) with dense-only servers, asserting
   that no proto fields, no routing semantics, no HTTP behaviour
   changed.
3. The existing 184 tests must remain green without modification.

### gRPC wire compatibility

Adding fields to proto3 messages is **backward-compatible**: an old
client that doesn't know about `expert_start/end` will send 0/0 (the
proto3 default for missing fields) and the router will treat it as
dense. An old router receiving an MoE announce from a new server
will see the new fields as zero in its parsing (proto3 unknown
fields are preserved on the wire but ignored by the typed view) —
the router routes the server as dense, which is the safe fallback.

---

## Alternatives Considered

### Per-layer expert map (`map<uint32, ExpertList>`)

A server announces a full `(layer → expert_ids)` mapping. Maximum
flexibility — useful if a deployment loads experts non-uniformly
across layers. Rejected because:

- No real deployment we've seen requires this.
- The route table inflates by `experts_per_layer` × `layers`.
- Operators can still express non-uniform layouts by registering
  multiple ranges from one server.

### New `expert_id` dimension in the route table key

`HashMap<(model_id, layer, expert_id), Vec<server_id>>`. Constant-
time expert lookup. Rejected because:

- Memory blowup (see "Route table layout" above).
- `rebuild_route_table` cost multiplies by `experts_per_layer` per
  join — already O(N × L), would become O(N × L × E).
- Linear filter through small candidate sets is cheap.

### Separate `MoeService` proto

A new gRPC service for MoE routing instead of extending the existing
one. Rejected because:

- Operators don't want two services to monitor.
- The replication / Mode B / drain machinery is identical for MoE
  shards; reusing it means less code.
- Additive proto changes preserve existing clients.

### Make `/v1/walk-moe-experts` a separate endpoint

Symmetric with the proto choice. Rejected — same reasoning. One
endpoint, optional fields, dispatcher branches on shape.

### Expert affinity in this ADR

See "Expert affinity: deferred" above. Not rejected, just out of
scope for this round.

---

## Consequences

### Positive

- MoE models become first-class. Operators can shard expert ranges
  across hosts the same way they shard layer ranges today.
- Per-token expert fan-out (the canonical MoE inference shape) maps
  directly to the router's `route_expert` + dispatch surface.
- Replication, hot-shard, drain-then-reassign all extend naturally
  to per-expert shards.
- Dense models keep working with no client-side changes.

### Negative

- More proto fields. `AnnounceMsg`/`ReadyMsg`/`AssignMsg` each grow
  two `uint32`s. Backwards-compatible but bigger.
- The replication tick's key tuple becomes 5 fields wide. Test
  helpers + ad-hoc inspection code get marginally more verbose.
- One more thing to remember: when adding a server, decide whether
  it's a dense or MoE shard. Docs need to spell this out.
- Binary wire stays dense-only. Documented gap, not blocking.

### Neutral

- Routing comparator unchanged.
- Bench shape unchanged for dense; new MoE bench scenarios to add.
- Hot-shard elevation set widens from 3-tuple to 5-tuple. Internal.

---

## Implementation plan (sketch)

Eight commits, all on top of ADR-0017's `/metrics` work:

1. **Proto extension.** Add `expert_start`/`expert_end` to
   `AnnounceMsg`/`ReadyMsg`/`AssignMsg`. Regenerate tonic stubs.
2. **`ServerEntry` + `GridState` mutations.** Carry expert range,
   default 0/0 for dense.
3. **`route_expert` + `route_all_experts`.** Linear filter inside
   the existing `route_table` candidate sets.
4. **HTTP path.** Add `experts` / `layer_experts` parsing in
   `extract_layers_and_model_id` (rename to
   `extract_layer_experts_and_model_id`). Dispatch branches on
   shape.
5. **Replication keys widen** to include expert range. Update
   `under/over_replicated_ranges`, `find_origin_for`,
   `try_assign_gap`.
6. **Hot-shard elevation set widens**. `hot_layer_ranges` →
   `hot_shard_ranges`, returning 5-tuples.
7. **Metrics**. New `expert_count` label cardinality consideration
   (bounded: small enum like `dense`, `1-8`, `9-32`, `33+`).
8. **Dense regression suite**. Bench rerun + existing-test pass
   without changes.

Each commit must keep the build green + tests green. The dense
regression suite (step 8) is the gate before merging.

---

## Open questions

1. **Should `route_expert` accept `model_id = None` for the
   single-model case?** Same logic as `route()` — yes, fall through
   to `any_model_table`.
2. **Should the HTTP shape allow `{layer: 5}` for an MoE model
   (no `experts`)?** Two options: (a) implicit "all experts on the
   owning shards" — the router would have to fan out to every
   `(layer, expert)` permutation, which loses the point of MoE
   sparsity. (b) Reject with a 400. **Choose (b)**: MoE requests
   must specify experts; dense requests must not.
3. **What about expert IDs that don't exist for a model?**
   `route_expert` returns `None` for any expert that no server
   owns; the dispatcher 503s the request. Same shape as a layer
   gap today.
