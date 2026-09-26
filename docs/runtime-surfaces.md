# Runtime and interface boundaries

**Class: CURRENT.** A crate dependency does not imply that every capability of
that dependency is exposed by its consumer. This map describes the public
surfaces and the checks that decide what a particular artifact can do.

| Surface | Implemented role | Boundary |
|---|---|---|
| `vindex` | Plan/encode/inspect/verify, representation compilation and supported export | No inference session or observation runner |
| `larql vindex3` | Container operations, execution, representation instruments and observation/intervention | Backend, operator and representation support are explicit |
| LQL | Parsed queries, lifecycle, mutation, tracing and generation-specific dispatch | Capability profile and artifact capabilities apply after parsing |
| Python bindings | Native graph, direct index, LQL-session, walk and trace wrappers | LQL sessions bind V2/V3; direct NumPy arrays require V2; no general V3 runtime wrapper |
| HTTP server | Bind V2/V3 artifacts, generation, query/profile-specific services and distributed execution | Mounted routes and loaded capabilities determine availability |
| Router | Shard fan-out, grid membership and capability-based API proxying | A partial shard is not a whole-model generation backend |
| Factory | Recipe validation, identity, estimation and staged build/publish execution | Build preflight refuses unmet numeric/Hub verification requirements; MIRROR/REGISTER are external |
| Observatory | Import, validate and replay recorded evidence | Import does not execute a model or independently prove parity |

## State and generation

`larql-inference` defines the session/generation seam. `larql-kv` contains
model-weight `KvEngine` implementations and a distinct `CanonicalKvState`
provider for the V3 interpreter. Storage reductions, derivative-state recovery
and semantic continuation equivalence are separate claims. The
[KV state policy](../crates/larql-kv/docs/state-policy.md) and
[V3 runtime](vindex3-runtime.md) explain the relevant contracts.

Observation is a subscription to canonical execution. The CLI recorder binds
provenance, scope and receipts; `--heads` and `--intervene` have narrower
backend/operator support than ordinary generation. A server request or Python
array access does not implicitly produce the same evidence record. Use
[observation and intervention](vindex3/observation-and-intervention.md).

## Serving and distributed execution

Query `/v1/capabilities` to discover the mounted server profile. The route
assembly and capability code are authoritative; a reserved path or protocol
message is not proof of an exposed operation. V3 defaults to production CPU execution. The macOS `vindex3-metal` feature adds
explicit Metal selection at startup (`--v3-backend`) and dynamic load (`backend`).
The selected backend is reported in `/v1/runtime`. `metal-experts` alone still
serves the separate V2 expert path. CPU V3 `--layers` prepares a stateless
layer-prefix worker; the `larql run --v3-shards` coordinator validates complete
coverage and loads only the stack endpoints. Workers refuse whole-model
completion requests. See [inputs, state and workers](vindex3/runtime-followups.md)
for the supported scopes and remaining distributed work.

The router's whole-model API proxy uses capable grid registrations, while
FFN/expert fan-out distributes partial work. Static layer maps do not supply
whole-model capability announcements. Continuation affinity, patch state and
local sessions remain distinct from stateless request routing.

The [server README](../crates/larql-server/README.md),
[router README](../crates/larql-router/README.md) and
[protocol README](../crates/larql-router-protocol/README.md) link their source
contracts. Transport options change delivery, not the artifact's semantics.

## Python and tooling

Python sessions bind V2/V3 through LQL; `session.vindex` lazily opens a V2-only
array view and explicitly refuses on V3. Rebinding with `USE` invalidates that
view. See the [source build instructions](../crates/larql-python/README.md).

Factory build preflight refuses required reconstruction, logit-match and Hub
verification that the driver cannot execute. A checksum pass cannot authorize
publication under those requirements. The [Factory README](../crates/larql-factory/README.md)
describes the operational consequence and the remaining verifier work.

The nested WASM experts and `model-compute` solver library have different ABIs
and host implementations. Both execute explicit structured requests; a direct
solver success says nothing about a model's ability to select the right call.
