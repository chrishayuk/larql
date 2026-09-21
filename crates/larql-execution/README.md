# larql-execution

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Shared execution-refusal vocabulary. This small dependency leaf lets compute,
container and engine layers classify a refusal without introducing a cycle.
It has no `larql-*` dependencies and owns no scheduler, transport or model state.

| `RefusalKind` | Meaning | Appropriate response |
|---|---|---|
| `Residency` | A valid operand is unavailable here | Fetch, load or reroute |
| `Unsupported` | This route cannot execute the valid operation | Select a capable executor/representation |
| `BindingDefect` | The artifact or bound plan violates its contract | Reject and repair the binding/artifact |

The [Rust contract](src/lib.rs) defines the enum, stable names and classification
helpers. Concrete errors retain the cause and evidence. A network timeout is
an execution-attempt failure belonging to the transport; it is not automatically
a new semantic refusal kind. `BindingDefect` must never become silent fallback.

```bash
cargo test -p larql-execution
```
