# larql-server

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

HTTP, streaming and gRPC services for loaded model artifacts, plus distributed
FFN/expert and shard services. Bootstrap distinguishes V2 indexes from VINDEX3
containers. V3 requests execute the artifact's declared program through the
VINDEX3 runtime rather than rebuilding V2 weights.

For V3 CPU softmax stacks, `--layers START-END` prepares only that inclusive
layer range. `GET/POST /v1/vindex3/layers` exposes a stateless prefix worker in
single-model mode, and `/v1/runtime` reports `layer_shard`. Whole-model
generation on a worker refuses. See [worker commands and scope](../../docs/vindex3/runtime-followups.md).

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
V3 defaults to production CPU execution. On macOS, build with
`--features vindex3-metal` and select `--v3-backend metal` to use the interpreter's
Metal F16 projection provider. The interpreter still performs its CPU-side
operations; this is separate from the CLI's `metal-lowered` executor.
No device or unsupported operator means refusal, never a silent CPU fallback.
The server's explicit backend composition in [vindex3.rs](src/vindex3.rs)
registers its configured device provider, then opens the runtime by that
provider's identity. This is an additional permitted construction site in the
[lowering closure check](../larql-vindex/tests/lowering_closure.rs); the original
LOWERING-PLUGIN-1 experimental construction inventory remains frozen.

Dynamic loading accepts `POST /v1/runtime/model` with
`{"path":"model.vindex3","backend":"metal"}` (backend defaults to `cpu`).
`/v1/runtime` reports `backend.selected`; changing an existing binding's backend
requires unloading it first. `/v1/capabilities` advertises the compiled V3 choices;
device availability is checked at load time.

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
