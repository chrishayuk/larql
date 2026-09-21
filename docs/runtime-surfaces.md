# Runtime and interface boundaries

**Class: CURRENT.** A crate dependency does not imply that every capability of
that dependency is exposed by its consumer. This map describes the public
surfaces and the checks that decide what a particular artifact can do.

| Surface | Implemented role | Boundary |
|---|---|---|
| `vindex` | Plan/encode/inspect/verify, representation compilation and supported export | No inference session or observation runner |
| `larql vindex3` | Container operations, execution, representation instruments and observation/intervention | Backend, operator and representation support are explicit |
| LQL | Parsed queries, lifecycle, mutation, tracing and generation-specific dispatch | Capability profile and artifact capabilities apply after parsing |
| Python bindings | Native graph, direct index, LQL-session, walk and trace wrappers | Direct index/session wrappers still open `VectorIndex`; no general V3 runtime wrapper |
| HTTP server | Bind V2/V3 artifacts, generation, query/profile-specific services and distributed execution | Mounted routes and loaded capabilities determine availability |
| Router | Shard fan-out, grid membership and capability-based API proxying | A partial shard is not a whole-model generation backend |
| Factory | Recipe validation, identity, estimation and staged build/publish execution | Current VERIFY is checksum integrity; MIRROR/REGISTER are external |
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
message is not proof of an exposed operation. The current V3 server binding
uses production CPU execution. Its `metal-experts` feature serves the separate
V2 expert path and must not be advertised as V3 Metal serving.

The router's whole-model API proxy uses capable grid registrations, while
FFN/expert fan-out distributes partial work. Static layer maps do not supply
whole-model capability announcements. Continuation affinity, patch state and
local sessions remain distinct from stateless request routing.

The [server README](../crates/larql-server/README.md),
[router README](../crates/larql-router/README.md) and
[protocol README](../crates/larql-router-protocol/README.md) link their source
contracts. Transport options change delivery, not the artifact's semantics.

## Python and tooling

The Python session constructor first executes LQL `USE` and then opens a
`PyVindex` for direct arrays. That second step constrains its current artifact
support even when Rust LQL can open V3. Use the
[source build instructions](../crates/larql-python/README.md) and tests rather
than interpreting the old draft API proposal as a shipped interface.

The nested WASM experts and `model-compute` solver library have different ABIs
and host implementations. Both execute explicit structured requests; a direct
solver success says nothing about a model's ability to select the right call.
