# larql-lql

**Class: CURRENT.** [Stack architecture](../../docs/architecture-stack.md) ·
[manifest-derived dependencies and features](../../docs/generated/workspace-facts.md).

Lazarus Query Language: AST, parser, capability-aware executor, local REPL and
remote client. LQL exposes container and graph operations through a SQL-like
surface; the Rust CLI and Python session are callers of the same language layer.

```bash
larql repl
larql lql 'USE "model.vindex"; DESCRIBE "France";'
larql lql 'HELP;'                      # every statement's syntax
larql lql 'HELP REMOVE DIRECTION;'     # one statement, a group (HELP WEIGHTS;) or the process model (HELP BATCH;)
```

`larql lql` runs its `;`-separated statements in one process and one session;
nothing carries to the next invocation except files (`.vlp` patches, compiled
containers). It exits 0 when every statement succeeded, 1 when any failed, and
2 when a verification contract was violated and nothing failed. `larql repl`
always exits 0.

The path is a placeholder. Generation, extraction tier, representation and
runtime capabilities determine which statements are admissible. A parsed
statement is not evidence that its requested operation is available on the
loaded artifact. V2 patch operations preserve immutable base files.

## Source map

| Source | Responsibility |
|---|---|
| [ast.rs](src/ast.rs), [parser](src/parser/) | Statement definitions and parsing |
| [capability.rs](src/capability.rs) | Capability checks for the selected session/artifact |
| [help](src/help/) | `HELP`: the catalogue (one page per statement, checked against the parser), lookup and rendering; also the source of the `USE` banner and of parse-error hints |
| [executor](src/executor/) | Lifecycle, query, mutation, introspection, tracing and VINDEX3 dispatch |
| [repl](src/repl.rs) | Interactive, batch and one-shot entry points |
| [executor/remote](src/executor/remote/) | `USE REMOTE` client operations |

The [language specification](docs/spec.md) is versioned; the
[VINDEX3 generation policy](../../docs/vindex-generation-policy.md) describes
coexistence. Use [the LQL guide](../../docs/lql-guide.md) for examples and
[the interface map](../../docs/runtime-surfaces.md) for boundaries between LQL,
format CLI, executor recording and HTTP. `observe --heads` and `--intervene`
are CLI interfaces; do not invent LQL statements from the conceptual verbs.

```bash
cargo test -p larql-lql
```

Parser and executor coverage must both change when adding a statement, and
the statement needs a help page: `Statement::help_topic` and the catalogue are
exhaustive matches, so the crate does not compile without one. Runtime
sessions and backend arithmetic remain in their owning crates.
