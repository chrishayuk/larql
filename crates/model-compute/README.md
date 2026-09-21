# model-compute

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Portable bounded compute primitives with no `larql-*` dependencies. Native
arithmetic/datetime kernels and an optional Wasmtime solver host are explicit
library calls; this crate does not load model weights or choose inference routes.

| Feature | API | Scope |
|---|---|---|
| `native` (default) | [KernelRegistry](src/native/registry.rs) | Built-in arithmetic and datetime kernels |
| `wasm` | [SolverRuntime](src/wasm/runtime.rs) | Compile a solver module and invoke it in a bounded session |

```rust
use model_compute::native::KernelRegistry;
let registry = KernelRegistry::with_defaults();
let answer = registry.invoke("arithmetic", "sum(1..101)");
```

Native bounds and accepted syntax live in [arithmetic](src/native/arithmetic.rs)
and [datetime](src/native/datetime.rs). The WASM host uses fuel/memory limits
and a fresh store per session. Its solver ABI is `alloc`, `solve`,
`solution_ptr` and `solution_len`, as implemented in [session.rs](src/wasm/session.rs).

This ABI is distinct from the nested [larql-experts](../larql-experts/README.md)
workspace's `larql_call`/`larql_metadata` ABI. That registry's host currently
lives in `larql-inference::experts`; this crate is not its automatic adapter.
A caller may consume a computed answer when compiling an edge, but weight
mutation and model semantics remain the caller's responsibility.

```bash
cargo test -p model-compute
cargo test -p model-compute --features wasm
```

Use the repository path dependency when developing here. The package version
in the generated inventory does not establish availability on a package registry.
