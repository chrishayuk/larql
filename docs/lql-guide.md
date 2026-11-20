# LQL Quick Start Guide

**LQL** (Lazarus Query Language) is the query language for neural network weights treated as a graph database. One binary, no Python, no GPU.

## Launch the REPL

```bash
cargo run -p larql-cli -- repl
```

Features: arrow keys, command history (`~/.larql_history`), Ctrl-R search, Ctrl-C cancel, Ctrl-D exit.

One or more statements, without the REPL:

```bash
cargo run -p larql-cli -- lql 'SHOW MODELS;'
cargo run -p larql-cli -- lql 'USE "model.vindex3"; STATS; SHOW LAYERS;'
```

The language documents itself. No model needs to be bound:

```bash
larql lql 'HELP;'                    # every statement's syntax, one line each
larql lql 'HELP REMOVE DIRECTION;'   # one statement: syntax, examples, requirements, refusals
larql lql 'HELP WEIGHTS;'            # one group of statements
larql lql 'HELP BATCH;'              # how statements share a session; HELP EXIT CODES; for the status
larql lql --help                     # the same process model, from the command line
```

An unknown topic is an error that names the nearest ones, and a mistyped statement says what was
expected: `REMOVE DIRECTON …` answers `after REMOVE expected DIRECTION or PATCH, got DIRECTON. Did you
mean REMOVE DIRECTION? Run HELP REMOVE DIRECTION; for the syntax.`

## Getting Started

### 1. Extract a model

```sql
-- Default: a VINDEX3 container, executable and queryable, session left bound
EXTRACT MODEL "Qwen/Qwen3-0.6B" INTO "qwen3-0.6b.vindex3";

-- VINDEX2 (the compatibility generation) must be asked for by name.
-- Browse-only (~3 GB f16 / ~6 GB f32, fast queries, no inference)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" FORMAT VINDEX2;

-- VINDEX2 with inference support (~6 GB f16 / ~12 GB f32, enables INFER)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH INFERENCE FORMAT VINDEX2;

-- VINDEX2 full (~10 GB f16 / ~18 GB f32, enables COMPILE for recompilation)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL FORMAT VINDEX2;
```

With no `FORMAT`, the [extraction-generation policy](vindex-generation-policy.md)
decides, and since 2026-11-06 it decides VINDEX3. A VINDEX3 extraction
encodes the checkpoint's own bytes into a container (HF checkpoint artifacts
only — GGUF sources refuse by name rather than transcoding on the way in),
carries the tokenizer and HF metadata in beside it, and leaves the session
bound: no `USE` needed. `WITH BROWSE | INFERENCE | ALL` shape a VINDEX2
extraction only. Not every architecture is admitted yet — check with
`larql vindex3 plan <model>`; Gemma 3, for example, still needs `FORMAT VINDEX2`.

CLI equivalent: `larql extract Qwen/Qwen3-0.6B -o qwen3-0.6b.vindex3`, or
`larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --generation v2 --level inference`
for VINDEX2.

### 2. Connect

```sql
-- Use a pre-extracted vindex (fast, all operations)
USE "gemma3-4b.vindex";
STATS;

-- Or, from `larql lql` / `larql repl`, a model already in the cache by name
-- (the name `larql run` takes; never downloads)
USE "Qwen3-0.6B";

-- Or point directly at model weights (no extraction needed)
USE MODEL "google/gemma-3-4b-it";
STATS;
-- Supports: INFER, EXPLAIN INFER, STATS
-- For WALK/DESCRIBE/SELECT/INSERT: extract into a vindex first
```

### 3. Browse knowledge

```sql
-- What does the model know about France?
-- Verbose by default: relation labels, also-tokens, layer ranges
DESCRIBE "France";

-- Compact view: top edges, primary layer only
DESCRIBE "France" BRIEF;

-- No labels — pure model signal
DESCRIBE "France" RAW;

-- Show all layer bands (syntax + knowledge + output)
DESCRIBE "France" ALL LAYERS;

-- Single layer
DESCRIBE "Mozart" AT LAYER 26;

-- Feature scan: which features fire for a prompt?
WALK "The capital of France is" TOP 10;

-- Per-layer trace
EXPLAIN WALK "The capital of France is" LAYERS 24-33;

-- SQL-style edge queries
SELECT entity, target FROM EDGES WHERE relation = "capital" LIMIT 10;

-- Synonym-robust relation filter (FR3): if "seat" matches no stored relation
-- exactly, it is resolved to a known relation by meaning ("seat" → capital,
-- "money" → currency) via a trained residual probe, then re-run against the
-- canonical relation. Needs a vindex with model weights + relation labels.
SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;
```

### 4. Run inference

Requires model weights: either a vindex built with `WITH INFERENCE` / `WITH ALL`,
or a `USE MODEL` session (direct weight access).

```sql
-- Next-token prediction with attention
INFER "The capital of France is" TOP 5;

-- Compare walk (no attention) vs infer (with attention)
INFER "The capital of France is" TOP 5 COMPARE;

-- Full inference trace
EXPLAIN INFER "The capital of France is" TOP 5;
```

For facts installed with `INSERT … MODE KNN`, the `ROUTE` clause selects how the
KNN side-channel decides whether to override the model's answer:

```sql
-- FR1 verified router: among the top-k activation candidates, override only
-- with a fact whose entity the prompt actually names — otherwise abstain and
-- let the model answer. Fixes the legacy "confident-wrong" injection. The safe
-- default for open queries. TOPK sets the candidate pool (default 5).
INFER "The capital of Atlantis is" ROUTE VERIFY TOP 3;

-- FR2 two-tier router: VERIFY first, then an activation-alias fallback for when
-- the prompt names an alias rather than the stored entity (e.g. "Persia" → Iran).
-- WARNING: the fallback has no entity-name guard, so on a query about a
-- non-stored entity it can confident-wrong like the legacy gate — use FALLBACK
-- only for queries known to be aliases of stored entities.
INFER "The capital of Persia is" ROUTE VERIFY FALLBACK TOPK 8 TOP 3;

-- EXIT: retrieval-augmented early exit. When the verified hit fires, the forward
-- short-circuits at the resolved layer — the stored target is emitted and the
-- remaining layers + lm_head are skipped (parity-exact, ~1.4× faster on
-- fact-lookup answer tokens). Verified-only; ignored with FALLBACK.
INFER "The capital of Atlantis is" ROUTE VERIFY EXIT TOP 3;
```

With no `ROUTE` clause, INFER inherits the global default from the `LARQL_KNN_*`
env vars (`LARQL_KNN_VERIFY` → verified, `+ LARQL_KNN_FALLBACK` → two-tier;
`LARQL_KNN_TOPK`, `LARQL_KNN_MIN_COS` tune the knobs). Unset → legacy top-1 gate.

### 5. Edit knowledge

```sql
-- Insert a fact (default KNN retrieval override). A KNN target must be
-- one token: the entry recalls one token id.
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "London");

-- Insert with all COMPOSE knobs: choose the layer, set confidence,
-- and dial the down-vector override strength
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.95
    ALPHA 0.30
    MODE COMPOSE;

-- Verify
DESCRIBE "John Coyle";

-- An inserted fact has one identity: entity + relation address it.
-- SELECT lists it above the discovered rows; UPDATE and DELETE reach it,
-- and never a discovered row.
SELECT * FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";
UPDATE EDGES SET target = "Paris"
    WHERE entity = "John Coyle" AND relation = "lives-in";
DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";

-- Discovered rows (features the model already has) are edited by slot.
DELETE FROM EDGES WHERE layer = 26 AND feature = 8821;

-- Update metadata by entity (top-token match over discovered rows)
UPDATE EDGES SET target = "London" WHERE entity = "John Coyle";

-- Update by (layer, feature) — fast-path, bypasses the entity scan
UPDATE EDGES SET target = "London", confidence = 0.95
    WHERE layer = 26 AND feature = 8821;
```

INSERT defaults to `MODE KNN`, which records a retrieval override and ignores
`ALPHA`. Use `MODE COMPOSE` when you want an FFN overlay that participates in
inference and can be compiled into vindex/model bytes; its default `ALPHA` is
0.10, with the validated range around 0.05-0.30. Relation predicates on
DELETE/UPDATE require relation labels in the active vindex; otherwise target
by `(layer, feature)` or omit `relation`.

**Asserted facts** (FACT-ID-1, `docs/fact-id-1.md`). `WHERE entity = … AND
relation = …`, with nothing else, addresses a fact LQL inserted, by its key
(case, `-`/`_` and spacing do not matter). Each fact has a stable id
(`f_…`), a version, a realization (`knn` or `compose`) and a status;
`Session::asserted_facts()` returns them as structured values.

- Inserting a key that already exists is refused; change it with UPDATE.
- UPDATE of a KNN fact keeps its retrieval key and replaces its target id
  and text, so INFER answers the new target.
- Compose facts are listed but read-only by identity. REMOVE PATCH of the
  patch that installed one is its exact rollback.
- APPLY PATCH refuses a patch that asserts a live fact, or that changes one
  the session does not hold.
- After COMPILE, KNN facts keep their identity (in `knn_store.bin`).
  Compose facts read back as discovered rows.

### 6. Patches

Patches are lightweight knowledge diffs — portable JSON files that modify a vindex without touching the base files.

```sql
-- Start recording a patch
BEGIN PATCH "medical-knowledge.vlp";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");

-- Save (base vindex NOT modified)
SAVE PATCH;

-- Apply a patch
APPLY PATCH "medical-knowledge.vlp";

-- Stack multiple patches
APPLY PATCH "fix-hallucinations.vlp";

-- See active patches
SHOW PATCHES;

-- Remove a patch (instantly reverts)
REMOVE PATCH "fix-hallucinations.vlp";

-- Extract diff as a patch
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

### 7. Recompile

```sql
-- See what changed
DIFF "gemma3-4b.vindex" CURRENT;

-- Bake the patches into a fresh standalone vindex (fast: unchanged
-- weight files are hard-linked from the source; a file that gets
-- override columns is copied first and the copy is patched, so the
-- source is never written).
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";

-- Use the compiled vindex like any other — INFER produces the new
-- facts with no patch overlay loaded.
USE "gemma3-4b-medical.vindex";
INFER "The capital of Atlantis is" TOP 5;

```

`COMPILE INTO VINDEX` writes every edit or refuses by name. A deleted slot
is written with a zero gate row and down vector, so it contributes nothing,
as in the session. `down_meta.bin` stores one token id per record and no
text: a multi-token target reloads as its first token, and the compile
lists each such record. An `UPDATE` that changes only a slot's text, a
quantised or per-layer source, a file the manifest names and the source
lacks, and a `FORMAT` clause are refused.

`COMPILE CURRENT INTO MODEL` is refused while the session holds edits the
export would not contain: any `DELETE`, `UPDATE` or KNN-mode `INSERT`, and
any compose-mode `INSERT` unless `LARQL_MEMIT_ENABLE=1`. `FORMAT GGUF` is
refused. The export itself does not currently complete for any source (it
fails while writing the model); that is tracked separately.

**How an output is published.** `COMPILE … INTO VINDEX`, `COMPILE … INTO
MODEL` and `COMPACT INTO VINDEX` write their output into a hidden staging
directory beside the destination, named `.<name>.tmp.<pid>.<n>`. Then:

1. every staged file, and on Unix every staged directory, is synced to disk;
2. the staged output is reopened and checked: a VINDEX2 vindex's checksums
   must match its bytes, a VINDEX3 container must open with every segment
   hashing to its directory entry, and a compacted container must mean
   exactly what its source means (an empty `DIFF`);
3. the staging directory is renamed to the destination in one step;
4. on Unix, the destination's parent directory is synced, so the rename
   survives a crash.

What this guarantees, and what it does not:

- **Identical inputs give identical bytes.** Two sessions in the same state
  write byte-identical outputs, every file included.
- **A failure before the rename changes nothing.** The destination stays as
  it was, the staging directory is removed, and the source and the session
  are untouched. If the staging directory cannot be removed either, the
  error names it so you can delete it.
- **A failure of the final sync is reported, not undone.** The output is in
  place and complete, but its durability is not confirmed. The error says
  "published … but its durability is not confirmed"; nothing is deleted.
- **The destination must be new.** An existing non-empty directory, a
  symlink, the source itself, or a path nested with the source is refused
  before anything is written.
- **Unchanged files are hard-links.** The output shares those files with the
  source. No LQL statement writes a shared file in place, so editing or
  compiling the output later never changes the source.
- **Platforms.** Durability is complete on Unix local filesystems. On other
  platforms files are synced, but directories cannot be, so the rename itself
  is not made durable.
- **Not a defence against concurrent writers.** The destination is checked,
  then renamed into. Another process that swaps a parent directory for a
  symlink in between can redirect the rename. The one race that is closed is
  a destination that gains contents: the rename then fails rather than
  deleting them.
- **An interrupted run** (a crash or a kill) can leave a staging directory
  behind. It is never published and nothing deletes it automatically; remove
  it by hand.

## Residual Stream Trace

Trace decomposes a forward pass into attention and FFN contributions at every layer.

```sql
-- What does the model predict at each layer?
TRACE "The capital of France is";

-- Track a specific answer through all layers
TRACE "The capital of France is" FOR "Paris";
-- Shows rank, probability, attn/FFN logit contribution, who pushes the answer

-- Attention vs FFN decomposition at the phase transition
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Save the trace to an mmap'd file
TRACE "The capital of France is" SAVE "france.trace";

-- Trace all token positions (not just last)
TRACE "The capital of France is" POSITIONS ALL SAVE "france_all.trace";
```

TRACE requires model weights (`WITH ALL` or `WITH INFERENCE` during EXTRACT). It uses the same WalkFfn as INFER — INSERT/DELETE mutations are reflected.

## Introspection

```sql
-- Discovered relation types
SHOW RELATIONS WITH EXAMPLES;

-- Layer summary
SHOW LAYERS;
SHOW LAYERS RANGE 14-27;

-- Feature details
SHOW FEATURES 26 LIMIT 20;

-- Available vindexes in current directory
SHOW MODELS;

-- Active patches
SHOW PATCHES;

-- Knowledge graph coverage
STATS;
```

## Layer Bands

DESCRIBE groups features into three bands based on the model's layer structure:

| Band | Gemma 3 4B | Llama 3 8B | What it contains |
|------|-----------|-----------|-----------------|
| Syntax | L0-13 | L0-12 | Morphological, syntactic, code |
| Knowledge | L14-27 | L13-25 | Factual relations (default view) |
| Output | L28-33 | L26-31 | Formatting, token selection |

```sql
DESCRIBE "France";              -- Verbose: relation labels, also-tokens, layer ranges (default)
DESCRIBE "France" BRIEF;        -- Compact: top edges, primary layer only
DESCRIBE "France" RAW;          -- No labels, pure model signal
DESCRIBE "France" SYNTAX;       -- Syntax band only
DESCRIBE "France" OUTPUT;       -- Output band only
DESCRIBE "France" ALL LAYERS;   -- All three bands
```

Bands are model-specific — computed automatically during EXTRACT from known architecture boundaries.

## VINDEX3 Containers

`USE` binds a VINDEX3 container directly — the container's own
generation marker decides, and the binding holds the executable
program, never a reconstructed model:

```
larql> USE "gpt-oss-20b.v3";
Using: gpt-oss-20b.v3 (VINDEX3, model: gpt-oss-20b, component target, 24 layers, execution closed)
Supported: USE, EXTRACT, STATS, SHOW COMPONENTS, …, INFER, WALK, EXPLAIN, TRACE, UPDATE WEIGHTS, REMOVE DIRECTION, ADD LOW RANK, …, HELP, PIPE. Tokenizer: present.
Syntax: HELP; lists every statement. HELP <statement>; prints one page.

larql> STATS;                      -- the container's own authority
larql> SELECT * FROM FEATURES WHERE layer = 0 LIMIT 8;
larql> WALK "France" TOP 5;        -- role feature_gate scan
larql> DESCRIBE "France";
larql> SHOW LAYERS;                -- per-layer facts off the plan
larql> INFER "The capital of France is" TOP 5;
larql> INFER "The capital of France is" GENERATE 16;   -- greedy continuation
larql> EXPLAIN INFER "x";          -- the executable plan, statically
larql> TRACE "x";                  -- observe the executor while it runs

larql> INSERT INTO EDGES (entity, relation, target)
   ..>     VALUES ("France", "capital", "Paris");     -- default MODE KNN
larql> DESCRIBE "France";          -- the edit is immediately visible
larql> INFER "The capital of France is" TOP 3;        -- …and overrides top-1
larql> UPDATE EDGES SET target = "Lyon" WHERE layer = 20 AND feature = 105;
larql> DELETE FROM EDGES WHERE layer = 20 AND feature = 106;
larql> MERGE "other-knowledge.vindex";                -- V2 source, V3 target
larql> BEGIN PATCH "france.vlp";   -- names the patch; the edits above are kept
larql> SAVE PATCH;                 -- persists; the container is untouched
larql> COMPILE CURRENT INTO VINDEX "gpt-oss-france.v3";  -- bake: clean container
larql> DIFF "gpt-oss-20b.v3" CURRENT;   -- logical: knowledge edges + feature slots
larql> DIFF "gpt-oss-20b.v3" "gpt-oss-france.v3" PHYSICAL;  -- segment-level
larql> COMPACT INTO VINDEX "gpt-oss-min.v3";  -- same meaning, cleaner storage
```

The container stays immutable on disk: edits live in a knowledge
overlay addressed by semantic identity, the KNN key is captured from
the plan's own execution taps, and the `.vlp` patch is the same
logical format V2 writes — one patch applies to either backend.
Compose-mode INSERT executes too: the FFN slot lands in the overlay
and execution observes it through the operand-source seam, running
V2's full install pipeline (capture, decoy-suppressed refine, balance,
cross-fact regression) with cross-format parity gated. `COMPILE … INTO MODEL`,
`DIFF … INTO PATCH`, `COMPACT MINOR` and `COMPACT MAJOR` refuse on a VINDEX3
container with a message naming the supported statements — a capability
boundary, not a missing feature apology. See the spec's §4.2/§4.4 for the
full contract.

### Editing stored weights

On a VINDEX3 container three statements edit the stored weights themselves, not the knowledge overlay:
`UPDATE WEIGHTS`, `REMOVE DIRECTION` and `ADD LOW RANK`. This is the whole loop:

```
USE  →  stage one or more ops  →  [BEGIN PATCH … SAVE PATCH]  →  COMPILE  →  VERIFY
```

Five facts shape everything below.

- **Staging writes nothing, and the session runs the edit at once.** A staged op is checked and remembered;
  no file is written and the bound container is never modified. From the next statement, `INFER` and `TRACE`
  in that session compute with the edited tensor: its bytes are produced by the path `COMPILE` writes them
  with and held in memory, and the reply lists each such tensor with the SHA-256 of its bytes. `DESCRIBE`,
  `WALK`, `SELECT` and `DIFF … CURRENT` still read the stored weights. `COMPILE CURRENT INTO VINDEX` writes a
  container that stores the edit; `DISCARD WEIGHT EDITS` and `REMOVE PATCH` withdraw it, and the stored
  weights run again.
- **An edit the session cannot run exactly is refused, not approximated.** While such an op is pending,
  `INFER` and `TRACE` are refused with the reason and do not fall back to the stored weights. That happens
  when the session's representation source is not `stored`, when a knowledge edit is pending as well, when
  the tensor is one execution reads past the edited bytes (an expert bank, a routed layer, a mixer other
  than softmax attention), or when `COMPILE` would refuse the op (step 7 shows one).
- **State lives in the process.** `larql lql 'A; B; C;'` runs its statements in one process and one session,
  so `USE`, the staged ops and applied patches carry from `A` to `C`. A second `larql lql` starts unbound
  with nothing staged. Only files carry over: a `.vlp` patch and a compiled container.
- **A compiled edit cannot be edited again.** A container written by a stored-weight `COMPILE` carries a
  ledger, `weight_patches.json`. Stage every op on the unedited base and compile once.
- **`COMPILE` writes to a path that does not exist yet, or refuses.** Nothing is ever replaced, and a
  stored-weight compile refuses an existing directory even when it is empty.

Every example below was run with `larql lql` on a two-layer toy container with no tokenizer (the conformance
fixture `verifier/conformance/structured_patch/base`, copied to `model.vindex3`, with the operand files
`u.json` and `a_b.json` of `verifier/conformance/weight_patches_v4/mixed/operands/`), and the replies are the
ones it printed. The `USE` banner is left out after the first example.

The toy container has no tokenizer, so `INFER` cannot run on it (step 9). What the replies below say about
`INFER` and `TRACE` is therefore not shown here by running them. It rests on the tests that do run them, on
a two-layer container of the same shape with a tokenizer added:
`crates/larql-lql/tests/weight_patch_lql/live.rs` (what `INFER`, `INFER … GENERATE` and `TRACE` print with an
op pending, after it is withdrawn, and against a fresh binding of the compiled container) and
`crates/larql-vindex/tests/weight_patch/live_exec.rs` (the same comparisons by token id, on every logit, bit
for bit).

#### 1. Bind, and find the tensor names

```
$ larql lql 'USE "model.vindex3"; STATS;'
Using: model.vindex3 (VINDEX3, model: wsp, component target, 2 layers, execution closed)
Supported: USE, EXTRACT, STATS, SHOW COMPONENTS, …, UPDATE WEIGHTS, REMOVE DIRECTION, ADD LOW RANK, …, HELP, PIPE. Tokenizer: absent (token-id capability only).
Refused without a tokenizer: SHOW RELATIONS, SHOW FEATURES, SHOW ENTITIES, DESCRIBE, SELECT, INFER, WALK, TRACE, INSERT, DELETE, UPDATE.
Syntax: HELP; lists every statement. HELP <statement>; prints one page.
Execution: representation source stored (the stored weights execute; nothing is manufactured at load).
Model:           wsp (VINDEX3)
…
```

A weight edit names a tensor as the container stores it. `EXPLAIN INFER "x";` prints the plan without
running a token and needs no tokenizer; the name is the part after `::`:

```
$ larql lql 'USE "model.vindex3"; EXPLAIN INFER "x";'
  layer 0
    …
    ffn        kind dense
      gate         target.decoder_stack::0.mlp.gate_proj.weight @BF16
      up           target.decoder_stack::0.mlp.up_proj.weight @BF16
      down         target.decoder_stack::0.mlp.down_proj.weight @BF16
```

#### 2. `UPDATE WEIGHTS`: a delta on named elements

```
$ larql lql 'USE "model.vindex3";
UPDATE WEIGHTS SET DELTA = 0.0625
  WHERE TENSOR = "0.mlp.down_proj.weight" AND ROW = 5 AND COL IN (12, 13);
SHOW PATCHES;
COMPILE CURRENT INTO VINDEX "delta.vindex3";'
Staged: 2 element(s) of 0.mlp.down_proj.weight, row 5, delta 0.0625 (BF16, nearest-even of the exact sum)
  [5, 12]: -0.0277100 -> 0.0346680
  [5, 13]: -0.0476074 -> 0.0148926
  op digest 45d6eeb173923883
  2 weight element(s) pending in this session
  In effect for INFER, TRACE and CAPTURE in this session: 1 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    0.mlp.down_proj.weight [64, 256]  sha256 74c3f8e2d76b0d5b7bf5d6fb008a63b1fc9526b27a8c09cfe8914354b92d12d0
  Nothing on disk has changed: COMPILE CURRENT INTO VINDEX writes a container that stores exactly these bytes. DESCRIBE, WALK, SELECT and DIFF read the stored weights.
  (no patches applied)
  Stored-weight edits: 2 element(s) in 1 op(s), pending COMPILE
  In effect for INFER, TRACE and CAPTURE in this session: 1 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    0.mlp.down_proj.weight [64, 256]  sha256 74c3f8e2d76b0d5b7bf5d6fb008a63b1fc9526b27a8c09cfe8914354b92d12d0
  Nothing on disk has changed: COMPILE CURRENT INTO VINDEX writes a container that stores exactly these bytes. DESCRIBE, WALK, SELECT and DIFF read the stored weights.
Compiled: delta.vindex3 (VINDEX3, clean container; weight patch of 2 element(s) of 0.mlp.down_proj.weight)
  re-derived: target.decoder_stack@Q8_0
  edited source (canonical) payload 22c8dfe3b3429bcc
  ledger weight_patches.json weight-patches/v2 (1 patch op(s) as origin)
```

Each element becomes the bfloat16 nearest to stored value + delta (ties to even), rounded once. `COL = <n>`
names one column and `COL IN (…)` several. The three indented lines after the staging reply are the session's
live view: the tensor `INFER` and `TRACE` now compute with, and the SHA-256 of its whole patched bytes, which
is the digest that tensor has in `delta.vindex3`. `SHOW PATCHES` shows what a `COMPILE` would write and prints
the live view again. The quantised representation (`@Q8_0`) is re-derived from the edited weights. The session
executes the canonical weights, not that pack. The later examples leave the three live lines out as `…`.

#### 3. `REMOVE DIRECTION`: project a direction out of chosen columns

The direction is an operand file, not a number in the statement: JSON
`{"bits": ["<16 lowercase hex digits>", … one per row of the tensor], "origin": "<optional note>"}`. Each
string is the bit pattern of an f64 (`"3ff0000000000000"` is 1.0), never a decimal.

```
$ larql lql 'USE "model.vindex3";
BEGIN PATCH "removal.vlp";
REMOVE DIRECTION "u.json" FROM "1.mlp.down_proj.weight" COLUMNS (33, 34)
  SCALE 0.5 NUMERICS "f64-seq-rne1/v1";
SAVE PATCH;
COMPILE CURRENT INTO VINDEX "removed.vindex3";'
Patch session started: removal.vlp
Staged: remove direction (64 component(s)) from 1.mlp.down_proj.weight, column(s) [33, 34], scale 0.5 under f64-seq-rne1/v1; 2 element(s) would change
  op digest ada59dc5924ba662
  results are BF16 as the profile defines them; no f32 delta is involved
  1 structured op(s) pending COMPILE in this session
  In effect for INFER, TRACE and CAPTURE in this session: 1 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    1.mlp.down_proj.weight [64, 256]  sha256 f6ffb0885d65d312a7ecfc3e27b5b3be5879912e1f2cb022c693784ca2135408
  …
Saved: removal.vlp (0 inserts, 1 updates, 0 deletes, 1 weight edit(s) bound to the source container)
Compiled: removed.vindex3 (VINDEX3, clean container; weight patch of 2 element(s) of 1.mlp.down_proj.weight)
  re-derived: target.decoder_stack@Q8_0
  edited source (canonical) payload 47ae4d9cea976afb
  ledger weight_patches.json weight-patches/v3 (1 patch op(s) as origin)
```

`COLUMNS` must be strictly increasing. `NUMERICS` is `f64-seq-rne1/v1` or `f64-seq-rne2/v1`. The patch
is recorded here because `VERIFY PATCH` (step 8) judges this kind of compile from the saved `.vlp`. The
ops stay staged after `SAVE PATCH`, so the `COMPILE` in the same batch still writes them.

#### 4. `ADD LOW RANK`: add `a · b` to chosen columns

The operand file is JSON `{"rank": K, "a": […], "b": […], "origin": "<optional note>"}`. `a` is rows × K
and `b` is K × the number of `COLUMNS`, both row-major, as bit patterns in the profile's working width:
16 hex digits for the f64 profiles, 8 for `f32-seq-rne1/v1`. There is no `SCALE`: fold it into `a` or `b`.

```
$ larql lql 'USE "model.vindex3";
ADD LOW RANK "a_b.json" TO "1.mlp.down_proj.weight" COLUMNS (8, 9)
  NUMERICS "f64-seq-rne1/v1";
COMPILE CURRENT INTO VINDEX "lowrank.vindex3";'
Staged: add rank-1 update to 1.mlp.down_proj.weight, 2 column(s) 8..=9 under f64-seq-rne1/v1; 128 element(s) would change
  op digest 3ae85f53b8765b65
  results are BF16 as the profile defines them; an element whose terms all have a zero factor keeps its bits
  1 structured op(s) pending COMPILE in this session
  In effect for INFER, TRACE and CAPTURE in this session: 1 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    1.mlp.down_proj.weight [64, 256]  sha256 5fbf52eff4d7cc89ad8f47887c25d6e9c6285b4cc58b806d86605dc20270fb54
  …
Compiled: lowrank.vindex3 (VINDEX3, clean container; weight patch of 128 element(s) of 1.mlp.down_proj.weight)
  re-derived: target.decoder_stack@Q8_0
  edited source (canonical) payload 2b1fa2caf8ef0f34
  ledger weight_patches.json weight-patches/v4 (1 patch op(s) as origin)
```

The three compiles above wrote three ledger schemas. The schema follows the ops:

| Ops in the compile | Ledger |
|---|---|
| `UPDATE WEIGHTS` only | `weight-patches/v2` |
| a `REMOVE DIRECTION`, no `ADD LOW RANK` | `weight-patches/v3` |
| any `ADD LOW RANK` | `weight-patches/v4` |

#### 5. Several ops, recorded as a patch

One compile takes several ops on several tensors. Two structured ops may share a tensor on disjoint
columns; a tensor takes `UPDATE WEIGHTS` or structured ops, not both. `BEGIN PATCH` records what is staged
after it, and `SAVE PATCH` writes the file:

```
$ larql lql 'USE "model.vindex3";
BEGIN PATCH "edit.vlp";
ADD LOW RANK "a_b.json" TO "1.mlp.down_proj.weight" COLUMNS (8, 9) NUMERICS "f64-seq-rne1/v1";
REMOVE DIRECTION "u.json" FROM "1.mlp.down_proj.weight" COLUMNS (33, 34) SCALE 0.5 NUMERICS "f64-seq-rne1/v1";
UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "0.mlp.down_proj.weight" AND ROW = 5 AND COL IN (12, 13);
SAVE PATCH;'
Patch session started: edit.vlp
Staged: add rank-1 update to 1.mlp.down_proj.weight, …
Staged: remove direction (64 component(s)) from 1.mlp.down_proj.weight, …
Staged: 2 element(s) of 0.mlp.down_proj.weight, row 5, delta 0.0625 (BF16, nearest-even of the exact sum)
  …
  recorded into patch "edit.vlp" (3 ops pending)
  In effect for INFER, TRACE and CAPTURE in this session: 2 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    1.mlp.down_proj.weight [64, 256]  sha256 0e8aa74721d59ff096a196f37bd8bd058f4b8401258f0d66446ff12487b8627b
    0.mlp.down_proj.weight [64, 256]  sha256 74c3f8e2d76b0d5b7bf5d6fb008a63b1fc9526b27a8c09cfe8914354b92d12d0
  …
Saved: edit.vlp (0 inserts, 3 updates, 0 deletes, 4 weight edit(s) bound to the source container)
```

The live view lists a tensor once, with every op on it applied: `1.mlp.down_proj.weight` carries the low-rank
update and the removal, so its digest is neither the one step 3 printed nor the one step 4 did.

That process has ended, and nothing was compiled. A new process applies the patch, which puts the same two
tensors in effect, and compiles it:

```
$ larql lql 'USE "model.vindex3";
APPLY PATCH "edit.vlp";
SHOW PATCHES;
COMPILE CURRENT INTO VINDEX "edited.vindex3";'
Applied: edit.vlp (3 operations: 0 inserts, 3 updates, 0 deletes; 2 stored-weight edit(s), bound to this container and pending COMPILE; 2 structured weight op(s), bound to this container and pending COMPILE)
  In effect for INFER, TRACE and CAPTURE in this session: 2 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    1.mlp.down_proj.weight [64, 256]  sha256 0e8aa74721d59ff096a196f37bd8bd058f4b8401258f0d66446ff12487b8627b
    0.mlp.down_proj.weight [64, 256]  sha256 74c3f8e2d76b0d5b7bf5d6fb008a63b1fc9526b27a8c09cfe8914354b92d12d0
  …
  1. edit.vlp                                 3 ops (0 ins, 3 upd, 0 del)
  Total: 3 from files, 0 in session
  Stored-weight edits: 2 element(s) in 3 op(s), pending COMPILE
  …
Compiled: edited.vindex3 (VINDEX3, clean container; weight patch of 132 element(s) of 1.mlp.down_proj.weight, 0.mlp.down_proj.weight)
  re-derived: target.decoder_stack@Q8_0
  edited source (canonical) payload bd899be7f3522094
  ledger weight_patches.json weight-patches/v4 (3 patch op(s) as origin)
```

A stored-weight op in a patch is bound to the exact container it was written against. Applied to any
other container, the compiled one included, it is refused:

```
$ larql lql 'USE "edited.vindex3"; APPLY PATCH "edit.vlp";'
Error: Execution error: failed to apply patch: this edit was written against another artifact: expected index/weights 1f68b2cf5f93 / fc7c2807decd, this container is ee0b1086a3ce / bd899be7f352
```

`DIFF` compares the base with the result. It needs nothing bound:

```
$ larql lql 'DIFF "model.vindex3" "edited.vindex3";
DIFF "model.vindex3" "edited.vindex3" PHYSICAL;'
Diff (logical): model.vindex3 vs edited.vindex3
FEATURES
  L0 F12   down_col changed
  L0 F13   down_col changed
  L1 F8   down_col changed
  L1 F9   down_col changed
  L1 F33   down_col changed
  L1 F34   down_col changed
SUMMARY
  feature slots changed: 6, knowledge +0/−0, other tensors: 0
Diff (physical): model.vindex3 vs edited.vindex3
  segment target.decoder_stack@BF16: 1ae6b2638229 → df00da3c14f8
  segment target.decoder_stack@Q8_0: ce9d8624980e → 5fdfe7049a7b
```

`DIFF` reads stored weights. With the patch applied and not compiled, `CURRENT` is still the base, and the
reply says so:

```
$ larql lql 'USE "model.vindex3";
APPLY PATCH "edit.vlp";
DIFF CURRENT "model.vindex3";'
Applied: edit.vlp (3 operations: …)
  …
Diff (logical): CURRENT vs model.vindex3
  no semantic differences — the models are equivalent
  note: 3 stored-weight op(s) are pending COMPILE and are not part of CURRENT here (INFER in this session executes them; DIFF does not see them); compile them and DIFF the container to compare the edited weights
```

#### 6. Changing your mind

`DISCARD WEIGHT EDITS` drops the ops staged directly in the session. `REMOVE PATCH` drops an applied patch
and its ops. Each takes effect at once: the reply says what `INFER` and `TRACE` compute with afterwards. A
second `USE` drops everything, without a warning.

```
$ larql lql 'USE "model.vindex3";
UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "0.mlp.down_proj.weight" AND ROW = 5 AND COL = 12;
DISCARD WEIGHT EDITS;
APPLY PATCH "edit.vlp";
REMOVE PATCH "edit.vlp";
SHOW PATCHES;'
Staged: 1 element(s) of 0.mlp.down_proj.weight, row 5, delta 0.0625 (BF16, nearest-even of the exact sum)
  …
  In effect for INFER, TRACE and CAPTURE in this session: 1 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    0.mlp.down_proj.weight [64, 256]  sha256 d96553fd4081236158e02b744cfbe485545fb0c1ace7655b0ffd6ac798dc2177
  …
Discarded: 1 stored-weight op(s) staged in this session
  No stored-weight op is in effect: INFER, TRACE and CAPTURE execute the container's stored weights.
Applied: edit.vlp (3 operations: …)
  In effect for INFER, TRACE and CAPTURE in this session: 2 tensor(s) execute from patched bytes held in memory, not from the container's stored bytes.
    1.mlp.down_proj.weight [64, 256]  sha256 0e8aa74721d59ff096a196f37bd8bd058f4b8401258f0d66446ff12487b8627b
    0.mlp.down_proj.weight [64, 256]  sha256 74c3f8e2d76b0d5b7bf5d6fb008a63b1fc9526b27a8c09cfe8914354b92d12d0
  …
Removed patch #1
  No stored-weight op is in effect: INFER, TRACE and CAPTURE execute the container's stored weights.
  (no patches applied)
```

After the discard nothing of the one-element edit is left: the patch's `0.mlp.down_proj.weight` has the
digest step 5 printed, not one that also carries the discarded op.

Without the `DISCARD` the `APPLY PATCH` is refused, because the patch edits element (5, 12) too:

```
$ larql lql 'USE "model.vindex3";
UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "0.mlp.down_proj.weight" AND ROW = 5 AND COL = 12;
APPLY PATCH "edit.vlp";'
Staged: 1 element(s) of 0.mlp.down_proj.weight, row 5, delta 0.0625 (BF16, nearest-even of the exact sum)
  …
Error: Execution error: failed to apply patch: parse error: weight patch refused: element (5, 12) of `0.mlp.down_proj.weight` is already edited by this session; edits to one element are not composed
```

#### 7. A compiled container cannot be edited again

`UPDATE WEIGHTS` is refused when it is staged. `REMOVE DIRECTION` and `ADD LOW RANK` stage, and both
execution and the `COMPILE` are refused:

```
$ larql lql 'USE "edited.vindex3";
UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "0.mlp.down_proj.weight" AND ROW = 6 AND COL = 12;'
Error: Execution error: UPDATE WEIGHTS refused: the source already carries a weight-patch ledger; patching a patched container is not supported

$ larql lql 'USE "edited.vindex3";
ADD LOW RANK "a_b.json" TO "1.mlp.down_proj.weight" COLUMNS (20, 21) NUMERICS "f64-seq-rne1/v1";
COMPILE CURRENT INTO VINDEX "twice.vindex3";'
Staged: add rank-1 update to 1.mlp.down_proj.weight, 2 column(s) 20..=21 under f64-seq-rne1/v1; 128 element(s) would change
  …
  NOT executable in this session, so INFER, TRACE and CAPTURE are refused while it is pending: COMPILE would refuse these ops: the source already carries a weight-patch ledger; patching a patched container is not supported
  Nothing on disk has changed.
Error: Execution error: COMPILE refused: the source already carries a weight-patch ledger; patching a patched container is not supported
```

The staging reply already says the op cannot be executed, and why: what `COMPILE` would refuse the session
does not run, and it does not run the stored weights in its place either.

To change an edit, go back to the base: apply the saved patch or re-stage the ops there, add the new op,
and compile to a new path.

#### 8. Verifying

LQL never judges an edit itself. A `VERIFY` statement runs an external verifier and reports its answer. It
takes paths, so it needs nothing bound. Which statement fits depends on the ledger:

| Ledger | Statement |
|---|---|
| `weight-patches/v4` (any `ADD LOW RANK`) | `VERIFY LEDGER`: the verifier recomputes every op the ledger records |
| `weight-patches/v3` (a `REMOVE DIRECTION`, no `ADD LOW RANK`) | `VERIFY PATCH`, from the saved `.vlp`, when it holds only `REMOVE DIRECTION` ops |
| `weight-patches/v2` (`UPDATE WEIGHTS` only) | Neither statement judges it |

`VERIFY LEDGER` passes the verifier the contract, the base and the container, and no patch file.

With no verifier configured a `VERIFY` statement is refused; it never passes by default. No verifier was
configured for this guide, so this is what the two statements printed (and both runs exited 1):

```
$ larql lql 'VERIFY LEDGER "edited.vindex3" USING CONTRACT "WEIGHT_PATCHES_V4" FROM "model.vindex3";'
Error: Execution error: VERIFY LEDGER needs an independent verifier and found none: install verifier/ from this repository, or set LARQL_VERIFIER. LARQL does not evaluate contracts itself.

$ larql lql 'VERIFY PATCH "removal.vlp" USING CONTRACT "STRUCTURED_REMOVE_DIRECTION" FROM "model.vindex3" AGAINST "removed.vindex3";'
Error: Execution error: VERIFY PATCH needs an independent verifier and found none: install verifier/ from this repository, or set LARQL_VERIFIER. LARQL does not evaluate contracts itself.
```

With a verifier, the statement prints its report. `satisfied` and `violated` are both answers: a violation
does not fail the statement, and `larql lql` exits 2 when nothing else failed. `could_not_evaluate` is an
error. How a verifier is found is in [the contract protocol](verification/contract-protocol.md) and in
`HELP VERIFIER;`.

#### 9. The process model

- **A failed statement does not stop a batch.** Later statements still run, and each failure is one
  `Error: …` line on standard output.
- **After a failure, nothing is written.** Every later statement that would write an artifact (`COMPILE`,
  `SAVE PATCH`, `EXTRACT`, `COMPACT INTO VINDEX`, `COMPACT MAJOR`, `DIFF … INTO PATCH`, `TRACE … SAVE`) is
  refused:

  ```
  $ larql lql 'USE "model.vindex3";
  UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "no.such.tensor" AND ROW = 5 AND COL = 12;
  COMPILE CURRENT INTO VINDEX "never.vindex3";'
  Error: Execution error: UPDATE WEIGHTS refused: tensor `no.such.tensor` is not held by any canonical representation of the source
  Error: refused `COMPILE CURRENT INTO VINDEX "never.vindex3"`: 1 earlier statement(s) in this batch failed, and a batch writes nothing after a failure
  ```

- **The exit status of `larql lql`** is 0 when every statement succeeded; 1 when any failed (a parse error,
  a refusal, an execution error, a contract that could not be evaluated, a missing verifier); 2 when a
  `VERIFY` contract was violated and nothing else failed. Every run in this section that printed `Error:`
  exited 1.
- **`larql repl` exits 0** whatever happened, and does not refuse writes after a failure. Use `larql lql`
  when a script must act on the result.
- **`COMPILE` refuses an existing output**:

  ```
  $ larql lql 'USE "model.vindex3"; COMPILE CURRENT INTO VINDEX "edited.vindex3";'
  Error: Execution error: compile: bake failed: refusing to write output `edited.vindex3`: the directory already exists and is not empty; existing contents are never replaced — choose a new path, or remove the directory yourself first
  ```

- **A container without `tokenizer.json`** binds and can be edited, compiled and inspected, and refuses
  the statements that need text. On the toy container:

  ```
  $ larql lql 'USE "model.vindex3"; INFER "The capital of France is" TOP 3;'
  Error: Execution error: INFER needs a tokenizer and this container carries no tokenizer.json — token-id capability only
  ```

  The refusal is the same with an op pending: the tokenizer is asked for first.

  ```
  $ larql lql 'USE "model.vindex3";
  UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = "0.mlp.down_proj.weight" AND ROW = 5 AND COL = 12;
  INFER "The capital of France is" TOP 3;'
  Staged: 1 element(s) of 0.mlp.down_proj.weight, row 5, delta 0.0625 (BF16, nearest-even of the exact sum)
    …
  Error: Execution error: INFER needs a tokenizer and this container carries no tokenizer.json — token-id capability only
  ```

  So nothing in this guide runs `INFER` on an edited session. That a staged op is executed is shown here only
  by the engine's reply (`In effect for INFER, TRACE and CAPTURE in this session`), and is established by the tests
  named at the top of this section.

`HELP <statement>;` gives each statement's refusals in full, and the spec (§3.5.1 to §3.5.4, §3.9) the
limits and what is and is not verified.

## Statement Reference

| Category | Statements |
|----------|-----------|
| Lifecycle | EXTRACT, COMPILE, DIFF, USE |
| Browse | WALK, DESCRIBE, SELECT, EXPLAIN WALK |
| Inference | INFER (with TOP, GENERATE, ROUTE, COMPARE), EXPLAIN INFER |
| Trace | TRACE (with FOR, DECOMPOSE, LAYERS, POSITIONS, SAVE; plain form on VINDEX3) |
| Mutation | INSERT, DELETE, UPDATE, MERGE, REBALANCE |
| Stored weights (VINDEX3) | UPDATE WEIGHTS, REMOVE DIRECTION, ADD LOW RANK, DISCARD WEIGHT EDITS |
| Patches | BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH |
| Verification | VERIFY REPRESENTATION, VERIFY OPERATOR, VERIFY PATCH, VERIFY LEDGER (external verifier) |
| Introspection | SHOW RELATIONS/LAYERS/FEATURES/ENTITIES/MODELS/PATCHES, SHOW COMPONENTS/REPRESENTATIONS/TENSORS/PROVENANCE/AUTHORITY (VINDEX3), STATS |
| Weight reads (VINDEX3) | EXPORT WEIGHT (one stored tensor, or rows and columns of it, to `.npy` with a `.npy.json` sidecar) |
| Activation reads (VINDEX3) | CAPTURE (what the session's program computed at a site, layer and positions, to `.npy` with a `.npy.json` sidecar) |
| Help | HELP, HELP \<topic\> |
| Pipe | `\|>` chains two statements |

`HELP;` prints the syntax of every statement above from the running binary, and is the list to trust
when this table and the binary disagree. See [the LQL specification](../crates/larql-lql/docs/spec.md)
for the full language specification.
