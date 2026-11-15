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

`larql.session(path)` binds V2 or VINDEX3 through LQL. For example:

```python
session = larql.session("model.vindex3")
print(session.query_text("STATS"))
print(session.query_text('INFER "Hello" GENERATE 16'))
```

Direct `larql.load(...)` and `session.vindex` expose V2 `VectorIndex` arrays.
The session loads that view lazily; accessing it on V3 raises
`NotImplementedError` with guidance to use `query()` or the typed operations
below. A successful `USE` changes which artifact the session refers to and
invalidates its cached array view.

### VINDEX3 operations

On a VINDEX3 binding the session also has typed methods. Each calls the same
Rust operation as the LQL statement in the second column (see
`larql_lql::operations`), so its result equals what the statement prints or
writes: Python builds no LQL text and reads no console output.

| Method | Statement | Returns |
|---|---|---|
| `s.tensors(filter=None)` | `SHOW TENSORS` | list of dicts: `name` (qualified), `representation`, `object`, `tensor`, `canonical`, `dtype`, `shape`, `bare` |
| `s.tensor(name, rows=None, cols=None, bits=False)` | `EXPORT WEIGHT` | `Readout`: `.array` (float32, or uint16/uint32 stored bits) and `.provenance` |
| `s.tokenize(text, raw=False)`, `s.decode(ids)` | the encoders `INFER` and `CAPTURE` use | ids, text |
| `s.capture(site, prompt, layer=, layers=, continue_text=, continue_tokens=, generate=, positions="LAST")` | `CAPTURE` | `Readout` |
| `s.infer(prompt, top=5)`, `s.generate(prompt, n)` | `INFER … TOP`, `INFER … GENERATE` | dicts of ids, tokens and probabilities, or generated ids and text |
| `s.begin_patch(path)`, `s.save_patch()` | `BEGIN PATCH`, `SAVE PATCH` | `None`, a dict of counts |
| `s.add_low_rank(tensor, a, b, columns, numerics)` | `ADD LOW RANK` | a dict: `digest`, `changed_elements`, `sham`, `pending_structured_ops`, `origin` |
| `s.update_weights(tensor, row, cols, delta)` | `UPDATE WEIGHTS` | a dict: `digest`, `elements` (BF16 bits before and after), `pending_elements` |
| `s.apply_patch(path)`, `s.remove_patch(path)`, `s.patches()` | `APPLY PATCH`, `REMOVE PATCH`, `SHOW PATCHES` | dicts |
| `s.discard_weight_edits()` | `DISCARD WEIGHT EDITS` | a dict: `discarded`, `unrecorded`, `from_patches` |

```python
import numpy as np
import larql

s = larql.session("model.vindex3")
logits = s.capture("logits", "The capital of France is").array   # [1, vocab]
row = s.tensor("target.embedding@BF16::weight", rows=[9079])
row.provenance["index_sha256"]

a = np.full((64, 2), 0.125)          # rows x rank
b = np.ones((2, 3))                  # rank x len(columns)
with larql.recording(s, "edit.vlp"):
    s.add_low_rank("1.self_attn.o_proj.weight", a, b, [8, 9, 10], numerics="f64-seq-rne1/v1")
```

- **Provenance** is the dict the statement's sidecar holds, with `statement`
  `None` and `npy_sha256` the digest of the `.npy` file the statement would
  write for the same values.
- **Tokens.** `tokenize(text)` is INFER's raw-prompt encoder: the tokenizer's
  own special tokens, plus the container's declared BOS where the tokenizer did
  not add it; no chat template. `tokenize(text, raw=True)` is the text's tokens
  alone, how `CAPTURE … CONTINUE "<text>"` encodes its continuation. `decode`
  keeps special tokens, as a capture's per-position `token` does.
- **Arrays out** are owned copies. `tensor` reads the container's stored bytes;
  edits staged or applied in the session are not included. `capture`,
  `infer` and `generate` execute the session's program, edits included.
- **Arrays in.** `add_low_rank` takes `a` (`rows × rank`) and `b`
  (`rank × len(columns)`) as float64 or float32 arrays and converts them, in
  Rust, to the operand bits of `numerics`: an f64-working profile takes
  float64 (float32 is widened exactly), the f32 profile takes float32 (float64
  is refused). The op's origin records `python arrays` with each array's shape
  and the SHA-256 of its bits, so the op is byte-identical to the operand-file
  form of the same bits and origin.
- `larql.recording(s, path)` is a context manager: `begin_patch` on entry,
  `save_patch` on a normal exit (its result is the yielded dict's `saved`).
- These methods are not judged by an LQL capability profile, which applies to
  statements; a Python session runs under the full profile.
- `larql._native._write_test_container(dir)` builds the tiny synthetic
  container the tests run on.

`s.discard_weight_edits()` (`DISCARD WEIGHT EDITS`) drops the weight ops staged
directly in the session and their copies in an open recording; ops an applied
patch carries stay with it.

### Out of process: `larql.connect`

`larql.connect(socket_path)` talks to a `larql serve-session` process and
returns an object with the same method names, arguments and return types as
the session methods above (arrays come back as `Readout`s with `.array` and
`.provenance`), plus `budget()`, the edited-model executions left. It is pure
Python (standard library and NumPy) and does not import the native extension,
so it works where `larql._native` is not installed; `import larql` then still
succeeds and only the native names raise `ImportError`. Paths name files in the
service's workdir. The service's refusals raise `larql.ServiceError` with its
`kind` and `reason`. See [the session service](../../docs/lql-session-service.md),
including what it does not enforce: it is not an OS sandbox.

```python
s = larql.connect("/path/to/larql.sock")
s.capture("logits", "The capital of France is").array
s.apply_patch("edit.vlp")
s.budget()
```

`tests/test_session_service.py` runs the `larql` binary (`$LARQL_BIN`, or
`target/{release,debug}/larql`; build it with `cargo build -p larql-cli`).

The full `Vindex3Runtime`, observation and intervention-record APIs remain
available through Rust/CLI rather than separate Python wrappers.

## Threads

Loads, forward passes, generation, trace capture/projection, KNN, WALK,
DESCRIBE, INSERT, LQL statements and the VINDEX3 operations release the GIL
while they run, so
other Python threads keep going. `Vindex`, `WalkModel`, `ResidualTrace` and
`Session` can be shared between threads: a `Vindex`'s mutations (`insert`,
`delete`, `set_*`) take a write lock and are atomic with respect to
concurrent reads, concurrent `infer()` calls on one `Vindex` run one at a
time, and one `Session` executes one statement at a time. `Graph` and its
algorithms still hold the GIL. [tests/test_threading.py](tests/test_threading.py)
measures the release.

See the [Python interface guide](../../docs/larql-python.md),
[tests](tests/) and [runtime surface map](../../docs/runtime-surfaces.md).
Optional MLX/streaming helpers retain their own model and dependency requirements.
