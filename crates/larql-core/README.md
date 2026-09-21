# larql-core

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Typed knowledge graphs, graph algorithms and serialization. This is a
workspace dependency leaf: it has no normal `larql-*` dependencies.
`larql-vindex` consumes it. The `Graph` of subject/relation/object edges is
separate from VINDEX3's `SystemGraph` and `ComponentOpPlan`.

## API

```rust
use larql_core::{Edge, Graph, shortest_path};

let mut graph = Graph::new();
graph.add_edge(Edge::new("France", "capital", "Paris"));
graph.add_edge(Edge::new("Paris", "river", "Seine"));
let path = shortest_path(&graph, "France", "Seine");
```

The [core](src/core/) module owns indexed edges, schema and metadata.
[algo](src/algo/) provides traversal, shortest path, PageRank, components,
merge, diff, filtering and walks. [io](src/io/) owns serialization;
[engine](src/engine/) owns provider-driven graph extraction helpers.

`add_edge` retains legacy duplicate-skipping behavior; `try_add_edge` reports
insertion versus duplicate, and `insert_edge` can replace an exact triple.
Metadata and edge provenance describe the graph's evidence. A stored edge or
a graph walk does not by itself establish causal influence in model execution.

The default `http` and `msgpack` features enable their optional dependencies;
consumers such as vindex can disable default features. These are graph-library
features, not VINDEX3 representation codecs.

```bash
cargo test -p larql-core
```

Runnable examples live in [the core demo catalog](../larql-demos/examples/core/README.md).
For model structure and execution, use [the VINDEX3 architecture](../../docs/vindex3/architecture.md).
