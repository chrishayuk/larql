# larql-router

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Layer/expert routing and grid coordination for distributed `larql-server`
deployments. The router selects serving nodes, tracks coverage and proxies
supported requests. It does not own VINDEX3 model semantics or replace the
canonical interpreter.

## Modes

```bash
cargo build --release -p larql-router
larql-router --shards 0-14=http://shard-a:9181,15-29=http://shard-b:9182 --port 9090
```

The static map example assumes those shards are already serving compatible
artifacts. Grid mode instead uses `--grid-port` and server registration via
`--join`; consult each binary's help and the [protocol crate](../larql-router-protocol/README.md).

[dispatch.rs](src/dispatch.rs) and [shards.rs](src/shards.rs) implement fan-out;
[grid](src/grid/) owns live membership/routing; [tasks](src/tasks/) owns RTT
probing and rebalancing. Administrative operations and bounded-cardinality
metrics have their own modules.

The [OpenAI-compatible proxy](src/openai/) uses grid-announced capabilities.
A static shard map alone supplies no `serves_openai` signal. Partial layer or
expert service must not be advertised as whole-model generation. Responses
routing retains backend affinity for continuation; other per-server sessions,
patches and operational endpoints are not transparently federated by the proxy.

The optional `quic` and `http3` features expose transport variants, not new
model-execution capabilities. Transport schemas and implementation live in
`larql-router-protocol`; generation availability still depends on the selected
server's artifact and runtime.

```bash
cargo test -p larql-router
```

See [runtime interfaces](../../docs/runtime-surfaces.md),
[multi-host demo](docs/multi-host-demo.md), [hot-shard demo](docs/hot-shard-demo.md)
and the [versioned router design](../larql-server/docs/router-spec.md).
Historical throughput numbers are topology-specific records.
