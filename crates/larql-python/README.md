# larql-python

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

PyO3 bindings packaged as `larql._native`, with Python convenience modules
for graph operations, direct vindex arrays, LQL sessions, walks and traces.
This is a native extension, not a subprocess wrapper around the CLI.

## Build from this checkout

```bash
cd crates/larql-python
uv sync --no-install-project --group dev
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
```

The repository toolchain applies to the Rust extension. Use this source-build
workflow when checking APIs against the current checkout; package metadata does
not prove a matching wheel has been published.

## What is exposed

[python/larql/__init__.py](python/larql/__init__.py) exports the supported Python
surface; [src/lib.rs](src/lib.rs) registers native types/functions. Direct
`Vindex` access is implemented in [src/vindex.rs](src/vindex.rs), LQL sessions
in [src/session.rs](src/session.rs), walks and traces in their respective modules.

```python
import larql

graph = larql.Graph()
graph.add_edge(larql.Edge("France", "capital", "Paris"))
```

Direct `larql.load(...)` uses the `VectorIndex` path. The Python session wrapper
also opens a direct `PyVindex` after its LQL `USE`, so it must not be advertised
as a generic VINDEX3 session merely because the underlying Rust LQL layer can
open V3. No Python wrapper here exposes the complete `Vindex3Runtime`,
`observe --heads` or intervention-record API. Use the supported Rust/CLI
interfaces for those workflows.

See the [Python interface guide](../../docs/larql-python.md),
[tests](tests/) and [runtime surface map](../../docs/runtime-surfaces.md).
Optional MLX/streaming helpers retain their own model and dependency requirements.
