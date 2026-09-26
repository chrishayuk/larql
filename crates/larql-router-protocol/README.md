# larql-router-protocol

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Shared wire contracts and transport wrappers for the router/server grid.
This crate has no normal `larql-*` dependencies. It owns generated tonic/prost
types and optional transport adapters; model execution and placement policy
remain with consumers.

| Contract | Responsibility |
|---|---|
| [vindex3](src/vindex3.rs) | Versioned JSON binding and rows for stateless CPU layer-prefix execution |
| [grid.proto](proto/grid.proto) | Registration, heartbeats, assignment, status and drain control |
| [expert.proto](proto/expert.proto) | Remote expert dispatch |
| [shard.proto](proto/shard.proto) | Sharded index-query service |
| [transport](src/transport/) | Optional QUIC and HTTP/3 adapters |

[build.rs](build.rs) generates the Rust types. Linux/macOS use the vendored
protobuf build dependency; Windows requires `protoc` on PATH. Consumers should
use the generated types rather than independently copying message layouts.

The `quic` and `http3` features are explicit opt-ins. Identity, TLS pinning and
message framing belong to these transports; enabling them does not enable an
unsupported model, representation or router operation. Registration and
capability announcements are interpreted by [larql-router](../larql-router/README.md)
and [larql-server](../larql-server/README.md).

```bash
cargo test -p larql-router-protocol
```

The [distributed interface guide](../../docs/runtime-surfaces.md) explains
service boundaries. Proto lint and consumer integration tests complement
[V3 worker integration](../../docs/vindex3/runtime-followups.md),
crate tests; wire-format changes must update both sides deliberately.
