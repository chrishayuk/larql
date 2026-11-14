# LQL Quick Start Guide

**LQL** (Lazarus Query Language) is the query language for neural network weights treated as a graph database. One binary, no Python, no GPU.

## Launch the REPL

```bash
cargo run -p larql-cli -- repl
```

Features: arrow keys, command history (`~/.larql_history`), Ctrl-R search, Ctrl-C cancel, Ctrl-D exit.

Single statement:

```bash
cargo run -p larql-cli -- lql 'SHOW MODELS;'
```

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
-- Insert a fact (default KNN retrieval override)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

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

-- List the edges, then delete the slot the listing shows.
-- (A mutation does not combine entity and relation; SELECT is its preview.)
SELECT * FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";
DELETE FROM EDGES WHERE layer = 26 AND feature = 8821;

-- Update by entity (works on both heap and mmap-loaded vindexes)
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

-- Bake the patches into a fresh standalone vindex (instant on APFS:
-- weight files hardlinked from source, only down_weights.bin gets the
-- override columns rewritten in place).
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
Supported: SELECT, DESCRIBE, WALK, INFER [TOP n] [GENERATE n], INSERT [MODE KNN], patches, … Tokenizer: present.

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
cross-fact regression) with cross-format parity gated. COMPILE/DIFF/COMPACT still refuse with a message naming the
supported statements — a capability boundary, not a missing feature
apology. See the spec's §4.2/§4.4 for the full contract.

### Editing stored weights

On a VINDEX3 container, `UPDATE WEIGHTS` edits real elements of one stored tensor. The container is never
modified; `COMPILE` writes a new one.

```sql
USE "qwen3-0.6b-q8_0.vindex3";

-- Optional: record the edit so it can be saved, shared and reapplied.
BEGIN PATCH "wsp001.vlp";

UPDATE WEIGHTS SET DELTA = 0.0078125
  WHERE TENSOR = "26.mlp.down_proj.weight" AND ROW = 100
    AND COL IN (200, 201, 202, 203, 204, 206, 207);

SAVE PATCH;                                   -- writes wsp001.vlp (version 2)
COMPILE CURRENT INTO VINDEX "qwen3-edited.vindex3";
```

In another session, the same edit applies to the same container and produces the same bytes:

```sql
USE "qwen3-0.6b-q8_0.vindex3";
APPLY PATCH "wsp001.vlp";       -- refused for any other container, or if the file was altered
COMPILE CURRENT INTO VINDEX "qwen3-edited.vindex3";
REMOVE PATCH "wsp001.vlp";      -- before COMPILE: nothing remains staged
```

The delta is added to the stored value and rounded once to bfloat16 (ties to even). Applying a patch twice, or
two patches that touch one element, is refused. Quantised representations are re-derived from the edited
weights. See the spec (§3.5.1) for the limits and for what is and is not verified.

## Statement Reference

| Category | Statements |
|----------|-----------|
| Lifecycle | EXTRACT, COMPILE, DIFF, USE |
| Browse | WALK, DESCRIBE, SELECT, EXPLAIN WALK |
| Inference | INFER (with TOP, GENERATE, ROUTE, COMPARE), EXPLAIN INFER |
| Trace | TRACE (with FOR, DECOMPOSE, LAYERS, POSITIONS, SAVE; plain form on VINDEX3) |
| Mutation | INSERT, DELETE, UPDATE, UPDATE WEIGHTS (VINDEX3), MERGE |
| Patches | BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH |
| Introspection | SHOW RELATIONS/LAYERS/FEATURES/MODELS/PATCHES, STATS |
| Pipe | `\|>` chains two statements |

See [the LQL specification](../crates/larql-lql/docs/spec.md) for the full language specification.
