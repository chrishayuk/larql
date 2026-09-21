# larql-server

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

HTTP, streaming and gRPC services for loaded model artifacts, plus distributed
FFN/expert and shard services. Bootstrap distinguishes V2 indexes from VINDEX3
containers. V3 requests execute the artifact's declared program through the
VINDEX3 runtime rather than rebuilding V2 weights.

## Start and inspect capabilities

```bash
cargo build --release -p larql-server
larql-server model.vindex3 --port 8080
curl http://localhost:8080/v1/capabilities
curl http://localhost:8080/v1/models
```

Add Cargo's release directory to PATH or invoke the binary there. The model
path is a placeholder. `larql serve` delegates to this binary. Use
`larql-server --help` for authentication, bind address, TLS, CORS, concurrency,
artifact-loading and shard options. The crate has no default GPU feature;
`metal-experts` is an explicit macOS option for the supported V2 expert path.
The current V3 server binding uses production CPU execution.

## Interface boundaries

| Surface | Scope |
|---|---|
| `/v1/capabilities` | Advertised routes/capabilities for the actual server profile |
| `/v1/components`, `/v1/representations`, `/v1/provenance`, `/v1/authority` | Capability-gated container facts |
| `/v1/plan` | Source planning; a plan is not an encoded artifact |
| Completion/chat/Responses APIs | Generation and streaming under the loaded model's runtime support |
| Query, session and patch routes | Generation- and profile-dependent LQL/graph operations |
| FFN, expert and shard services | Partial/distributed execution and artifact handoff, not a universal whole-model endpoint |

The [route assembly](src/routes/mod.rs), [path constants](src/routes/paths.rs)
and [capability implementation](src/capabilities.rs) define the server surface.
A reserved path constant is not proof that a route is mounted: encoding remains
a separate CLI operation. API compatibility does not imply every V2 option is
implemented by the V3 branch, or that server generation exports observation
records. Use the [runtime interface guide](../../docs/runtime-surfaces.md).

## Ownership

[bootstrap](src/bootstrap/) opens artifacts; [state](src/state/) owns model
bindings and lifecycle; [routes](src/routes/) adapts HTTP requests;
[session](src/session/) and response-state modules manage continuation;
[grpc](src/grpc.rs) and expert modules adapt distributed calls.
[larql-router](../larql-router/README.md) owns shard selection and grid control.

```bash
cargo test -p larql-server
```

For detailed generation/state behavior see [VINDEX3 runtime](../../docs/vindex3-runtime.md).
The [original server draft](docs/server-spec.md) and
[router design](docs/router-spec.md) retain their versioned scopes.
Prior deployment recipes, endpoint examples and measurements remain accessible
through [the archived README](../../docs/archive/README.md); use current command
help and route capabilities before reproducing them.
