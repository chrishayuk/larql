# LQL model introspection: LARQL owns the model, Python helps the agent think (r4, approved direction 2026-11-15)

**r2 supersedes r1.** The owner's decision on 2026-11-15: Python is allowed for the maths. What is not acceptable is
an agent building its own inference engine, as in WSP-002-E0, where the agent re-implemented Gemma's forward pass
in torch and read tensor bytes directly. r1's array language (`LET`, expressions, solve, ridge, `FORWARD FROM`) is
dropped as a prerequisite.

## The contract

> The agent may write any analysis code, including Python, NumPy and SciPy. But every model weight, activation,
> logit and inference result it uses must come from LARQL, and every change to the model's weights is made by
> LARQL.

| Capability | LARQL | The agent's Python |
|---|---|---|
| Read weights | **only LARQL** | never reads VINDEX3, HF or GGUF bytes |
| Tokenise, chat template, decode | **LARQL** | uses the ids LARQL returns |
| Run the model | **only LARQL** | no forward pass of its own, no model runtime |
| Activations, residuals, logits | **only LARQL** | analyses what LARQL exported |
| Arithmetic, regression, selection, building operands | — | **yes** |
| Apply/remove edits, save `.vlp`, run the edited model | **only LARQL** | scores LARQL's outputs |

Python may compute a counterfactual (for example, the logits from an exported activation through an exported
head) but never a full forward pass. The protocol meters counterfactuals: see "E1 enforcement".

## What exists (origin/main, read 2026-11-15)

- **VINDEX3 execution already exposes the internals, through callbacks:**
  - `PlaneEvent::Layer { post_attention, ffn_input, post_layer }` (`opplan/exec/trace.rs:110-125, 270`);
  - `StepObserver::ffn_down_input`, the real down_proj input (`observe.rs:330`);
  - `DownInputCapture` (`down_input_capture.rs:23`);
  - logits as the step's return value.
- **LQL uses none of it for readout.** VINDEX3 `TRACE` prints event names only and refuses all clauses
  (`executor/trace.rs:30-39`).
- **The tied embedding/head can't be named.** `"weight"` is ambiguous between `target.embedding@BF16` and
  `target.final_norm@BF16`, and `target.embedding::weight` is refused (E0 tooling gap 3).
- **`ADD LOW RANK` already reads a Python-friendly operand file:** JSON with hex bits (`low_rank.rs:14-56`). It
  stays as it is.

## What to build

### 1. Tensor reads
```
EXPORT WEIGHT "<tensor>" [OF "<representation>"] [ROWS (…)] [COLUMNS (…)] TO "<file>.npy";
```
- **Unambiguous naming of every tensor:** `<representation>::<tensor>`, with `SHOW TENSORS [OF …]` listing them,
  the tied embedding/head and the norm gains included. The same naming should be accepted by weight edits, so the
  head becomes editable (gap 3).
- **Exact values:** BF16 stored values are widened exactly to f32. `AS BITS` writes the raw u16.

### 2. Tokens
- `TOKENS "<text>" [RAW] TO "<file>"` gives the ids, with the container's chat template unless RAW.
- `DECODE (…)` turns ids back into text.
- `EXPORT VOCAB TO "<file>"` writes every id with its decoded text, so Python can search it.

### 3. Capture
```
CAPTURE <site> [AT LAYER <n> | AT LAYERS (…)] FOR "<prompt>" [CONTINUE "<text>" | CONTINUE TOKENS (…) | GENERATE <n>]
        [POSITIONS PROMPT_END | REPLY | ALL | (…)] TO "<file>.npy";
```
- **Sites:**
  - `residual` (after a layer);
  - `post_attention`;
  - `ffn.input`;
  - `ffn.down_input`;
  - `ffn.down_output`;
  - `final`;
  - `logits` (full vocabulary, f32).
- **Teacher forcing.** `CONTINUE` feeds a given reply, so an agent can capture along a baseline reply. `GENERATE`
  captures along the model's own greedy path and records the tokens.
- **One execution path.** It runs on the session's live state, so it captures under an applied patch, or the base
  when none is applied. It uses the **same** executor as INFER (`step_observed` and `PlaneEvent`), never a second
  forward implementation. Captured logits at a position must be **bit-identical** to INFER's there.

### 4. The export format
Each export is a standard `.npy` file (f32, or u16 bits), plus a sidecar `<file>.json`:
- `site`, `layer`, `positions` (and the token id at each), `shape`, `dtype`;
- the model: container `index_sha256` and `model`;
- the **patches in effect**, by `.vlp` sha256 and op digests;
- the prompt's token ids, the statement text, and the LARQL version.

So every number the agent analyses can be traced to a model state.

### 5. BF16 preview
```
PREVIEW LOW RANK "<operand.json>" TO "<tensor>" COLUMNS (…) [NUMERICS …];
```
- It changes nothing.
- It reports the elements changed after BF16 rounding, the ratio of effective to declared change per column, and
  whether the op would be refused.
- It can export the effective delta. It uses the existing low-rank resolution code (the "N element(s) would
  change" path), not new arithmetic.

### Not built
- `LET`, array expressions, solve, ridge, `FORWARD FROM`, gradients.
- These may come later on their own merits, but they are not prerequisites for E1.

## Testing
- **Capture against INFER.** Logits bit-identical to INFER at matching positions; `ffn.down_input` bit-identical
  to `DownInputCapture`; captures under an applied patch equal captures from the compiled container.
- **Capture against an oracle.** On real Gemma 3 4B with fresh, neutral prompts (not E0's), captures agree within
  calibrated tolerances with Hugging Face transformers fp32 on the same checkpoint. The oracle is test code, never
  the agent's tooling.
- **Exports.** Weights equal the stored bits exactly. The `.npy` files load in NumPy with the right shape and dtype.
  The sidecar's digests match.
- **Preview.** Its counts equal what `ADD LOW RANK` then stages and what COMPILE writes.
- **End to end, neutral.** A scripted session on a neutral task (not code fences or JSON):
  - LARQL exports weights and captures;
  - a short Python script computes an operand from those exports alone;
  - LARQL previews, applies live, infers, saves the `.vlp`, removes it, and compiles.
  - A file-access check proves the script read nothing but the exports.

## HELP
- Each statement gets a page with neutral examples: a capital city, a toy tensor.
- No page mentions code fences, JSON, layer 33, gates at the first reply position, or ridge-fitting a switch.
- The pages are reviewed for recipe leakage before E1.

## E1 enforcement (for the protocol)
- **The agent can't read the model directly.** Its Python runs without access to the container, the HF cache or
  any GGUF: they are kept outside its readable paths, or it runs as another user. A wrapper is the only route to
  larql.
- **No model runtime.** No torch, transformers or llama.cpp bindings in the agent's environment. Its Python imports
  are logged.
- **Metering.**
  - Full evaluations of the edited model are counted, as in E0.
  - Captures under an applied patch count as evaluations of the edited model: they run it.
  - Python counterfactuals can't be metered by the harness. Either the protocol caps them by rule and the agent
    logs each one, or it says plainly that they are unmetered. To be decided in the E1 protocol.

## The Python bridge (r3 addition, 2026-11-15)

The owner chose Option A: LARQL used from Python, with LARQL as the model runtime and Python as the workbench. Python
inside LQL is not built.

**It already exists in part.** `crates/larql-python` (PyO3 0.29 plus `numpy`) exposes `larql.session(path)`, which
binds VINDEX3 and runs LQL through `query` and `query_text`. It has no capture, tensor or patch calls, and
`session.vindex` refuses on V3.

**The rule (r4): one typed Rust operation, several interfaces.**
- Each capability is a typed Rust function: `read_tensor`, `tokenize`, `capture`, `infer`, `stage_low_rank`,
  `preview_low_rank`, `apply_patch`, `remove_patch`.
- The LQL executor and the PyO3 methods both call it. Python never builds an LQL string or parses console output.
- Every capability is in both interfaces, so none is Python-only. Tests hold the two interfaces to equal results.

| Method | LQL it wraps | Returns |
|---|---|---|
| `larql.open(path)` | `USE` | a session |
| `s.tensor(name, rows=, cols=)` | `EXPORT WEIGHT` | owned `ndarray`, plus `.provenance` |
| `s.tokenize(text, raw=False)`, `s.decode(ids)`, `s.vocab()` | `TOKENS`, `DECODE`, `EXPORT VOCAB` | ids or text |
| `s.capture(site, layer=, prompt=, continue_=, generate=, positions=)` | `CAPTURE` | an owned array with positions, ids and provenance |
| `s.infer(prompt, top=)`, `s.generate(prompt, n)` | `INFER` | structured results |
| `s.patch(path)`, as a context manager | `BEGIN PATCH` … `SAVE PATCH` | a patch handle |
| `p.add_low_rank(tensor, a, b, columns, numerics=)` | `ADD LOW RANK` | stage report |
| `p.preview_low_rank(…)` | `PREVIEW LOW RANK` | counts and the effective delta |
| `s.apply(path)`, `s.remove(path)`, `s.patches()` | `APPLY`/`REMOVE PATCH`, `SHOW PATCHES` | — |

- **Arrays passed in.** NumPy arrays given to `add_low_rank` are converted to the operand bits in Rust, by the same
  parser and type the file form uses. The op origin records "from python arrays", with their digests.
- **Arrays passed out.** Owned copies first. Read-only zero-copy views come later, once lifetimes are proven: a view
  pins its buffer, and a later INFER or patch change can't touch it.

**Transport, and why it matters for E1.** An in-process PyO3 module needs read access to the container, so the
agent's own Python process could also read the model files. In-process can't enforce E1's "no direct model
access" rule. So the bridge gets two transports behind the same API:
- **in-process** (`larql.open`), for users and tests;
- **out-of-process** (`larql.connect(socket)`), talking to a **restricted model-access service**. This is a
  dedicated LARQL service on the same typed operations; it is not LQL syntax and not an arbitrary LQL executor.
  - **What it enforces itself:** the one permitted model, the permitted patch paths, the permitted op kinds, and the
    experiment's evaluation budget, with every captured or run state of an edited model counted.
  - **How it talks:** JSON-lines requests; arrays travel as `.npy` bytes with a provenance header.

E1 uses only the service. The socket alone is not isolation. The harness also enforces, at the operating-system
level, that the agent's process can't read the model directory or any copy of the checkpoint, and has no other
model runtime. A Python library is not a security boundary.

## PRs (each with its own gate)
| PR | Content | Gate |
|---|---|---|
| 1 | Live VLP execution (the existing spike) | real-model parity, normal CI |
| 2 | Tensor naming and `EXPORT WEIGHT`, `SHOW TENSORS`, `TOKENS`/`DECODE`/`EXPORT VOCAB` | exact bits, CI |
| 3 | `CAPTURE` at every site, the `.npy` and sidecar format | bit-identity to INFER, the HF oracle on real 4B |
| 4 | `PREVIEW LOW RANK`; head editable through the new names | preview counts equal staged/compiled |
| 5a | Python API: extend the existing PyO3 session over the shared typed operations, with NumPy in and out | Python results equal LQL results, operation for operation |
| 5b | Restricted session service and `larql.connect`; server-side enforcement of model, paths, ops and budget | its own security review; an isolation test that the agent's process cannot read model files |
| 6 | HELP pages and Python docstrings, the recipe-leak review, the neutral end-to-end through the bridge | E1-ready |

GGUF export (Gemma 3 4B, text-only, BF16) is a separate, parallel workstream with its own two gates: export
fidelity against `convert_hf_to_gguf.py`, and the edited model's behaviour in llama.cpp.

## Roadmap (not v0.3.0, not E1)
The long-term vision is a complete weight-space programming language in LARQL:
**observe → analyse → compute → edit → verify → export**. It needs `LET` and typed session arrays, vector and
matrix expressions, solvers with a numerical contract (singular, ill-conditioned and non-positive-definite inputs
specified, never left to a library), `FORWARD FROM` (counterfactual, kept apart from full evaluation and metered),
and possibly gradients. r1 of this document sketched it. It deserves its own design and review, and may be a
future video ("can the agent drop Python entirely?").
