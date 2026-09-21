# LARQL stack architecture

**Class: CURRENT.** [Manifest-derived crate facts](generated/workspace-facts.md)
are the dependency/feature inventory. [VINDEX3 facts](generated/current-facts.md)
cover schemas and CLI commands. Neither is a release-availability claim.

VINDEX3 is implemented primarily inside `larql-vindex`; there is no standalone
`vindex3` workspace crate. The surrounding crates separate source description,
numerical execution, artifact semantics, runtime state and user-facing adapters.
The dependency graph is not a single linear chain.

## Responsibility map

| Layer | Crates and boundaries |
|---|---|
| Contract and utility leaves | [vindex-spec](../crates/larql-vindex-spec/README.md) owns the manifest contract; [execution](../crates/larql-execution/README.md) owns refusal classification; [core](../crates/larql-core/README.md) owns generic knowledge graphs; [boundary](../crates/larql-boundary/README.md) owns residual codecs; [router-protocol](../crates/larql-router-protocol/README.md) owns transport contracts |
| Source description | [models](../crates/larql-models/README.md) owns architecture/config, inventory, weight formats and multimodal descriptions |
| Numerical substrate | [compute](../crates/larql-compute/README.md) owns shared traits and CPU math; [compute-metal](../crates/larql-compute-metal/README.md) owns the Metal implementation and shaders |
| Artifact semantics | [vindex](../crates/larql-vindex/README.md) owns lifecycle, V3 graph/plans/representations and the canonical interpreter, alongside V2 index APIs |
| Runtime and state | [inference](../crates/larql-inference/README.md) composes sessions/generation/records; [kv](../crates/larql-kv/README.md) provides engine/state implementations |
| Language and tools | [lql](../crates/larql-lql/README.md), [larql CLI](../crates/larql-cli/README.md), [vindex CLI](../crates/vindex-cli/README.md) and [Python](../crates/larql-python/README.md) expose different subsets of the system |
| Services | [server](../crates/larql-server/README.md) binds artifacts and serves requests; [router](../crates/larql-router/README.md) coordinates distributed nodes and capability-based proxying |
| Build and examples | [Factory](../crates/larql-factory/README.md) orchestrates recipe builds; [demos](../crates/larql-demos/README.md) declares runnable examples |
| Explicit compute tools | [model-compute](../crates/model-compute/README.md) provides native/solver calls; the separate [experts workspace](../crates/larql-experts/README.md) builds JSON-ABI WASM guests |

## Boundaries that matter

`larql-core` is a dependency of `larql-vindex`, with no normal `larql-*`
dependencies of its own. Its entity/edge graph is not the VINDEX3 system graph.
`larql-vindex-spec` owns a shared manifest contract; it does not define all V3
semantics merely because its name includes “spec.”

CPU and Metal are peers as numerical implementations. In Cargo, Metal depends
on `larql-compute` for the shared trait surface and CPU fallback. “Peer” does
not mean the two crates have no dependency edge. The substrate reaches index
storage through `KvIndex`; it must not import the concrete `VectorIndex`.

The V3 `PlanBackend` seam is separate from the compute traits. The interpreter
owns program meaning, a prepared image binds operands/realizations, and runtime
sessions own continuation. A source architecture adapter is not consulted to
reinterpret an already encoded V3 program.

Normal dependencies, dev dependencies and optional backends answer different
questions. A test-only edge from vindex into inference is not production
ownership. The generated JSON records those edge kinds separately; Cargo's
resolved graph for a chosen target/features remains the build authority.

## Interfaces are not interchangeable

The [runtime surface map](runtime-surfaces.md) explains which APIs expose V3,
V2, observation, state and distributed execution. In particular, Python direct
vindex access is not the V3 runtime API, and the KV engine selector is not an
automatic selector for V3 continuation providers.

The nested experts workspace has its own manifests and checks. Its
`larql_call` ABI is hosted by `larql-inference::experts`; `model-compute`'s
`solve` ABI is a different host/guest contract. Root workspace checks do not
cover the nested workspace.

See [compute/source integration](compute-substrate.md),
[VINDEX3 architecture](vindex3/architecture.md), [status](vindex3/status.md)
and the [documentation policy](documentation-policy.md).
