# Contributing to LARQL

Thanks for helping. This page gets you from a fresh clone to a pull request.
[AGENTS.md](AGENTS.md) is the detailed reference for workspace layout and
invariants; you don't need to read it all before a first change.

## Set up

```bash
git clone https://github.com/chrishayuk/larql && cd larql
cargo build -p larql-cli            # Linux/Windows: add --no-default-features
```

The repository pins its Rust toolchain (`rust-toolchain.toml`), so `cargo`
installs and uses the right version automatically. On macOS the default
features build the Metal GPU backend; elsewhere use `--no-default-features`
for the portable CPU build.

To try your build on a real model, follow
[Getting started](docs/getting-started.md) using `./target/debug/larql` or
`./target/release/larql`. Debug builds are much slower at inference; use
`--release` for anything model-backed.

## Find your way around

| If you're changing… | Look in |
|---|---|
| The `larql` command line | `crates/larql-cli/src/commands/` (one module per verb) and `crates/larql-cli/src/main.rs` |
| LQL, the query language | `crates/larql-lql/` (lexer, parser, executor) |
| The HTTP server | `crates/larql-server/` |
| The VINDEX3 format, encoding, planning | `crates/larql-vindex/src/format/vindex3/` |
| Running models (sessions, engines, chat templates) | `crates/larql-inference/` |
| CPU / Metal kernels | `crates/larql-compute/`, `crates/larql-compute-metal/` |
| Documentation | `docs/`, plus `docs/` folders inside crates |

[Stack architecture](docs/architecture-stack.md) explains how the crates fit
together and which way dependencies may point.

## Your first change: a walkthrough

Here is how an LQL statement such as `STATS;` flows through the code. Adding a
new statement means touching the same five places:

1. **Keyword**: `crates/larql-lql/src/lexer.rs`. Add it to `enum Keyword` and
   to the string-to-keyword match (`"STATS" => Some(Self::Stats)`).
2. **Syntax tree**: `crates/larql-lql/src/ast.rs`. Add a `Statement` variant
   and its name in the statement-name match.
3. **Parser**: `crates/larql-lql/src/parser/mod.rs` dispatches on the
   keyword; the parse function lives in the topical file (`STATS` is in
   `parser/introspection.rs`).
4. **Executor**: `crates/larql-lql/src/executor/mod.rs` dispatches
   `Statement::Stats` to `exec_stats` in `executor/lifecycle/stats.rs`. If
   the statement should also work over `USE REMOTE`, add the remote arm
   further down the same file.
5. **Capabilities**: `crates/larql-lql/src/capability.rs` lists what each
   profile allows. Add the statement where it belongs.

Then add tests, in test files rather than inline: parser tests in
`crates/larql-lql/src/parser/tests/`, executor and integration tests in
`crates/larql-lql/tests/`. Run:

```bash
cargo test -p larql-lql
make larql-lql-ci          # fmt, clippy, tests, examples, bench smoke
```

## House rules

These are enforced in review and, mostly, by CI:

- **Tests live in test files** (`tests/` folders or `*_tests.rs` modules),
  not large inline `mod tests` blocks.
- **Coverage is gated per file** at 90% for the crates with a coverage
  policy. New public API needs tests in its *own* crate; tests in a
  downstream crate don't count towards it.
- **No magic values.** Name constants; keep modules small and focused.
- **No hard-coded model families.** Behaviour comes from the model's own
  declarations (config, container graph), not `if family == "llama"`.
- **Refuse, don't guess.** When LARQL can't do something correctly it says so
  by name rather than falling back silently. Keep that property in new code.
- **Docs move with code.** CI checks that every relative link and inline
  code reference (paths, `file.rs:LINE`, `make` targets, env vars) resolves.

## Before you open a pull request

```bash
cargo fmt --all -- --check
cargo clippy -p <crate> --all-targets -- -D warnings
cargo test -p <crate>
python3 scripts/check_doc_links.py
python3 scripts/check_doc_references.py --strict
python3 scripts/current_facts.py --check     # if you changed CLI commands or format constants
```

`make ci` runs the full workspace format, lint and test suite. It's slow;
the per-crate `make <crate>-ci` targets are usually enough locally.

Open the pull request against `main`. Commit messages and PR descriptions
must not contain AI-tool attribution lines (`Co-Authored-By: Claude`,
"Generated with Claude Code" and similar); the `commit-messages` check
rejects them.

## Reporting bugs

Open an issue with the command you ran, the full output, your OS and how you
installed LARQL (release binary or source build, and the commit). For model
problems, include the Hugging Face model ID and the output of
`larql vindex3 plan hf://<model>`.
