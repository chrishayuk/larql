# LQL — Lazarus Query Language Specification

**Version:** 0.4  
**Date:** 2026-04-11  
**Status:** Draft  
**Implementation target:** `larql-lql` crate (Rust)  
**Companion:** `larql-knowledge` spec (data pipeline)

---

## 1. Design Principles

LQL is a query language for neural network weights treated as a graph database. It is not SQL. It is not SPARQL. It borrows from both but serves a different purpose: decompiling, inspecting, editing, and recompiling neural networks

**Principles:**

1. **Weights are rows.** Every W_gate row is a record. Every W_down column is a record. Every embedding vector is a record. The model IS the database.
2. **Two backends, one language.** LQL operates on either a `.vindex` (pre-extracted, fast) or directly on model weights via safetensors (live, no extraction needed). The vindex is preferred for production — sub-millisecond lookups from a pre-built index. Direct weight access is for exploration — point at any model, start querying immediately. Same statements, same results, different performance.
3. **Statements, not scripts.** Each LQL statement is self-contained. No variables that persist across statements (except `USE` context). Pipe results with `|>`.
4. **Three verbs for the demo.** The video needs exactly: `EXTRACT`, `DESCRIBE`, `INSERT`, `COMPILE`. Everything else is power-user.
5. **Rust-native.** The parser lives in `larql-lql`. No Python dependency. No runtime. One binary.
6. **Labels are external.** Relation labels come from the `larql-knowledge` project (probes, Wikidata triples, WordNet, AST pairs). The engine reads label files. It does not contain ingestion or probing code.

---

## 2. Statement Categories

### 2.1 Model Lifecycle

| Statement | Purpose |
|---|---|
| `EXTRACT` | Decompile model weights → vindex |
| `COMPILE` | Recompile vindex → model weights |
| `DIFF` | Compare two vindexes |
| `USE` | Set active vindex / model context |

### 2.2 Knowledge Browser (pure vindex, no model needed)

| Statement | Purpose |
|---|---|
| `WALK` | Feature scan — what gate features fire for a token's embedding |
| `SELECT` | Query edges by entity, relation, layer |
| `DESCRIBE` | Show all knowledge for an entity, grouped by layer band |
| `EXPLAIN WALK` | Feature trace without attention (pure vindex) |

### 2.3 Inference (requires model weights in vindex)

| Statement | Purpose |
|---|---|
| `INFER` | Full forward pass with attention — actual next-token prediction |
| `EXPLAIN INFER` | Full inference with per-layer feature trace |

### 2.4 Knowledge Mutation

| Statement | Purpose |
|---|---|
| `INSERT` | Add an edge (mode `KNN` default, `COMPOSE` FFN-overlay) |
| `DELETE` | Remove edge(s) from the vindex |
| `UPDATE` | Modify existing edge(s) |
| `MERGE` | Merge edges from another vindex |
| `REBALANCE` | Global fixed-point rebalance over compose installs |
| `COMPACT MINOR` | Promote L0 KNN entries to L1 compose edges |
| `COMPACT MAJOR` | Promote L1 compose edges to L2 MEMIT (optional `FULL`, `WITH LAMBDA`) |

### 2.5 Patches

| Statement | Purpose |
|---|---|
| `BEGIN PATCH` | Start recording edits into a patch file |
| `SAVE PATCH` | Save and close the current patch |
| `APPLY PATCH` | Apply a patch to the current vindex (stacks) |
| `REMOVE PATCH` | Remove a patch from the stack |
| `SHOW PATCHES` | List active patches |
| `DIFF ... INTO PATCH` | Extract diff between two vindexes as a patch |
| `COMPILE ... INTO VINDEX` | Bake patches into a new clean vindex |

### 2.6 Schema Introspection

| Statement | Purpose |
|---|---|
| `SHOW RELATIONS` | List discovered relation types |
| `SHOW LAYERS` | Layer-by-layer summary |
| `SHOW FEATURES` | Feature details at a layer |
| `SHOW ENTITIES` | Distinct named entities across the loaded layers |
| `SHOW MODELS` | List available vindexes |
| `SHOW PATCHES` | List active patches (see §2.5) |
| `SHOW COMPACT STATUS` | Storage-engine status (L0/L1/L2 tiers, epoch) |
| `STATS` | Counts, coverage, size |

### 2.7 Residual Stream Trace (requires model weights in vindex)

| Statement | Purpose |
|---|---|
| `TRACE` | Capture residual stream decomposition for a prompt |
| `TRACE ... FOR <token>` | Track a specific target token's rank/contribution per layer |
| `TRACE ... DECOMPOSE` | Show attention vs FFN delta per layer |
| `TRACE ... SAVE <path>` | Write trace to a file |

> Planned (not yet implemented): `TRACE ... DIFF`, boundary stores, tiered
> context stores. See §11.

---

## 3. Grammar

### 3.1 Notation

```
UPPERCASE  = keyword (case-insensitive in parser)
<name>     = required parameter
[name]     = optional parameter
{a | b}    = choice
...        = repeatable
```

### 3.2 Model Lifecycle Statements

```
EXTRACT MODEL <model_id> INTO <vindex_path>
    [COMPONENTS <component_list>]
    [LAYERS <range>]
    [WITH {INFERENCE | ALL}]

-- Decompile a HuggingFace model (or local path) into a vindex.
-- 
-- Default (no WITH clause): builds the browse-only vindex.
--   Extracts: gate_vectors, embeddings, down_meta, labels
--   Enables: WALK, DESCRIBE, SELECT, EXPLAIN WALK
--   Size: ~3 GB (f16)
--
-- WITH INFERENCE: adds attention weights for INFER.
--   Adds: attn_weights (Q, K, V, O per layer)
--   Enables: + INFER, EXPLAIN INFER
--   Size: ~6 GB (f16)
--
-- WITH ALL: adds all weights for COMPILE.
--   Adds: + up_weights, norms, lm_head
--   Enables: + COMPILE
--   Size: ~10 GB (f16)
--
-- Layers: 0-33 (default: all)
-- No data is duplicated — gate_vectors IS W_gate, embeddings IS W_embed.

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex";
-- Browse-only: ~3 GB

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex"
    WITH INFERENCE;
-- Browse + inference: ~6 GB

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex"
    WITH ALL;
-- Full: ~10 GB, supports COMPILE
```

```
COMPILE {<vindex_path> | CURRENT} INTO MODEL <output_path>
    [FORMAT {safetensors | gguf}]

-- Recompile a vindex back to model weights.
-- Round-trip: EXTRACT then COMPILE should produce identical weights.
--
-- INSERT/DELETE/UPDATE mutations applied to the vindex are baked into
-- the canonical down_weights.bin (column-rewrite at the inserted slots),
-- so the exported safetensors / gguf file already contains the
-- constellation in the standard down_proj tensors. No special loader
-- support is needed — load the output in HuggingFace Transformers, GGUF
-- runtimes, etc. and the inserted facts fire through the standard FFN.

COMPILE CURRENT
    INTO MODEL "gemma3-4b-edited/"
    FORMAT safetensors;
```

```
DIFF <vindex_a> {<vindex_b> | CURRENT}
    [LAYER <n>]
    [RELATION <type>]
    [LIMIT <n>]

-- Show edges that differ between two vindexes.

DIFF "gemma3-4b.vindex" CURRENT;

DIFF "gemma3-4b.vindex" "gemma3-4b-edited.vindex"
    RELATION "capital"
    LIMIT 20;
```

```
USE <vindex_path>;
USE MODEL <model_id> [AUTO_EXTRACT];
USE REMOTE <url>;

-- Set the active backend for subsequent statements.
-- USE with a .vindex path: fast, pre-extracted, all statements available.
-- USE MODEL: live weight access, reads safetensors directly. Slower but
--   zero setup — point at any model, start querying immediately.
-- USE REMOTE: HTTP client to a `larql serve` instance. All read queries
--   are forwarded over the wire. Mutations create a local patch overlay
--   (the remote vindex is not modified).
-- AUTO_EXTRACT: create vindex automatically on first mutation.
--
-- Mutation (INSERT/DELETE/UPDATE) and COMPILE require a vindex.
-- INFER requires model weights (either via --include-weights or USE MODEL).

USE "gemma3-4b.vindex";

USE MODEL "google/gemma-3-4b-it";

USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;

USE REMOTE "https://models.example.com/larql";
```

### 3.3 Knowledge Browser Statements

```
WALK <prompt>
    [TOP <n>]
    [LAYERS {<range> | ALL}]
    [MODE {hybrid | pure | dense}]
    [COMPARE]

-- Feature scan: what gate features fire for the last token's embedding.
-- This is a knowledge browser operation, NOT inference.
-- No attention is used. The query is the raw token embedding.
-- Returns per-layer top features with relation labels and gate scores.

WALK "France" TOP 10;

WALK "The capital of France is"
    TOP 10
    LAYERS 24-33;
```

```
DESCRIBE <entity>
    [{ALL LAYERS | SYNTAX | KNOWLEDGE | OUTPUT}]
    [AT LAYER <n>]
    [RELATIONS ONLY]
    [{VERBOSE | BRIEF | RAW}]

-- Show all knowledge for an entity. Groups by layer band:
--   SYNTAX:     morphological, syntactic, code structure features
--   KNOWLEDGE:  semantic/factual features (default view)
--   OUTPUT:     formatting/output features
--   ALL LAYERS: show all three bands
--
-- Display modes:
--   BRIEF (default): compact view — top edges only, primary layer, no also-tokens.
--   VERBOSE: relation labels in [brackets], also-tokens,
--            layer ranges, multi-layer hit counts.
--            Labelled features show [relation], unlabelled show [—].
--   RAW:     no probe/cluster labels — pure model signal with also-tokens.
--
-- Layer band boundaries are model-specific, stored in index.json:
--   Gemma 3 4B:  syntax=0-13, knowledge=14-27, output=28-33
--   Llama 3 8B:  syntax=0-15, knowledge=16-28, output=29-31
--   (boundaries discoverable via SHOW LAYERS)
--
-- Labels come from multiple sources (highest priority first):
--   1. Probe-confirmed (model inference verified this feature)
--   2. Wikidata output matching (L14-27 only)
--   3. WordNet output matching (L0-13 only)
--   4. AST output matching (L0-13 only)
--   5. Entity pattern detection (country, language, month, number)
--   6. Morphological detection (short suffixes/prefixes)
--   7. TF-IDF top tokens (fallback)
--
-- Each edge shows: relation label, target token, gate score,
-- layer range, occurrence count, and "also:" variant tokens.

DESCRIBE "France";
-- Brief by default: compact, top edges, primary layer only

DESCRIBE "France" VERBOSE;
-- Full detail: relation labels, also-tokens, layer ranges

DESCRIBE "France" RAW;
-- No labels — pure model signal

DESCRIBE "France" ALL LAYERS;
-- Shows syntax (L0-13) + knowledge (L14-27) + output (L28-33)

DESCRIBE "def" SYNTAX;
-- Shows L0-13 features only — code/linguistic structure

DESCRIBE "France" KNOWLEDGE;
-- Explicit: same as default

DESCRIBE "France" OUTPUT;
-- Shows L28-33 only — output formatting features

DESCRIBE "Mozart" AT LAYER 26;
-- Single layer

DESCRIBE "France" RELATIONS ONLY;
-- Only show edges with a confirmed relation label (probe or matched)
```

```
SELECT [<fields>]
    FROM EDGES
    [NEAREST TO <entity> AT LAYER <n>]
    [WHERE <conditions>]
    [ORDER BY <field> {ASC | DESC}]
    [LIMIT <n>]

-- Query edges in the vindex.
-- Fields: entity, relation, target, layer, feature, confidence, gate
--
-- WHERE relation = "<word>" matches the stored relation label exactly. If an
-- exact match finds nothing, FR3 synonym resolution kicks in: the relation word
-- is resolved to a known relation BY MEANING (a trained residual probe — "seat"
-- → capital, "money" → currency) and the query re-runs against the canonical
-- relation, printing a note. Needs a vindex with model weights + relation
-- labels; otherwise exact-only (no change). One-time probe build is cached.

SELECT entity, relation, target, confidence
    FROM EDGES
    WHERE entity = "France"
    ORDER BY confidence DESC
    LIMIT 10;

-- Synonym-robust relation filter (FR3): "seat" resolves to "capital".
SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;

SELECT entity, target
    FROM EDGES
    WHERE relation = "capital"
    AND confidence > 0.5;

SELECT *
    FROM EDGES
    WHERE layer = 27
    AND feature = 9515;

SELECT entity, target, distance
    FROM EDGES
    NEAREST TO "Mozart"
    AT LAYER 26
    LIMIT 20;
```

```
EXPLAIN WALK <prompt>
    [LAYERS <range>]
    [VERBOSE]
    [TOP <n>]

-- Show the per-layer feature trace from a pure vindex walk.
-- No attention. Each layer shows top-K features that fire,
-- with relation labels, gate scores, and output tokens.

EXPLAIN WALK "The capital of France is";

EXPLAIN WALK "The capital of France is"
    LAYERS 24-33 TOP 3 VERBOSE;
```

### 3.4 Inference Statements

```
INFER <prompt>
    [ROUTE VERIFY [FALLBACK] [TOPK <n>] [EXIT]]
    [TOP <n>]
    [GENERATE <n>]
    [COMPARE]

-- Full forward pass with attention. Requires attention weights.
-- Vindex must be built with WITH INFERENCE or WITH ALL.
-- Or use USE MODEL for live weight access.
-- Uses walk FFN (gate KNN from vindex) as the FFN backend,
-- but runs real attention for token routing.
-- COMPARE: also run dense inference and show both.
--
-- ROUTE selects the KnnStore router (FR1/FR2) for INSERT … MODE KNN facts.
-- Default (no clause): legacy top-1 + fixed 0.75 cosine gate — fires on
--   near-rank-1 collisions and can inject a confident-wrong fact.
-- ROUTE VERIFY (FR1): top-k candidates + a verifier that only overrides with a
--   fact whose entity the PROMPT names, else abstains. Fixes confident-wrong.
--   TOPK <n> sets the candidate count (default 5).
-- ROUTE VERIFY FALLBACK (FR2): adds an activation alias fallback — if no
--   candidate's entity is named (alias/paraphrase, e.g. "Persia"→Iran), route
--   by the top-1 activation key. Recovers aliases exact-string can't, at a
--   fuzzy ~0.7-0.9 rate. WARNING: the fallback has no entity-name guard, so on
--   an open query about a NON-stored entity it confident-wrongs like the legacy
--   gate (measured 0/20 distractor-safe vs VERIFY's 20/20). Use FALLBACK only
--   for queries known to be aliases of stored entities; VERIFY is the safe
--   open default.
-- ROUTE VERIFY EXIT: retrieval-augmented EARLY EXIT. When the verified router
--   fires, the forward short-circuits at the resolved layer (~0.7 depth) — the
--   stored target is emitted and the remaining layers + lm_head are SKIPPED.
--   Parity-exact (the emitted token is byte-identical to the full forward) and
--   ~1.4× faster on fact-lookup answer tokens. Verified-only: EXIT is ignored
--   when FALLBACK is set (the FR2 fallback needs the full forward to stay
--   parity-identical). On a verify-miss the forward completes normally.
-- (The LARQL_KNN_VERIFY / LARQL_KNN_FALLBACK / LARQL_KNN_TOPK / LARQL_KNN_MIN_COS
--  env vars set the default when no ROUTE clause is present;
--  LARQL_KNN_EARLY_EXIT turns on EXIT for the env-default verified path.)

INFER "The capital of France is" TOP 5;

INFER "The capital of France is" TOP 5 COMPARE;

INFER "The capital of Atlantis is" ROUTE VERIFY TOP 3;

INFER "The capital of Persia is" ROUTE VERIFY FALLBACK TOPK 8 TOP 3;

-- Early-exit: emit the stored target the moment the verified hit fires.
INFER "The capital of Atlantis is" ROUTE VERIFY EXIT TOP 3;

-- GENERATE <n>: greedy autoregressive continuation of <n> tokens
--   through the bound runtime. Currently VINDEX3 bindings only (the
--   proven seam: batch prefill -> caller-owned KV state -> resumed
--   decode); other backends refuse with a clear message. Output is
--   the generated ids, the detokenised text, and timings.
INFER "The capital of France is" GENERATE 16;
```

```
EXPLAIN INFER <prompt>
    [LAYERS <range>]
    [VERBOSE]
    [TOP <n>]
    [WITH ATTENTION]

-- Full inference with per-layer feature trace.
-- Shows which features fire WITH attention context.
-- Requires model weights.
--
-- WITH ATTENTION: also print per-layer attention head attributions
--                 alongside the feature trace.

EXPLAIN INFER "The capital of France is" TOP 5;

EXPLAIN INFER "The capital of France is" TOP 5 WITH ATTENTION;
```

### 3.5 Knowledge Mutation Statements

```
INSERT INTO EDGES
    (entity, relation, target)
    VALUES (<entity>, <relation>, <target>)
    [AT LAYER <n>]
    [CONFIDENCE <float>]
    [ALPHA <float>]
    [MODE {KNN | COMPOSE}]

-- Add an edge to the vindex.
-- The relation should be one of the known types (from probe/cluster labels).
--
-- INSERT has two install modes. Default: KNN.
--
-- MODE KNN (default) — Architecture B retrieval override:
--   Captures the model's residual at the install layer (or the entity
--   embedding if no weights are available) and stores it as a key in
--   the per-layer KnnStore alongside the target token. INFER overrides
--   the model's top-1 when a stored key matches the inference residual
--   at cos > 0.75. Scales freely (validated at 25K edges, 87 edges/s,
--   100% same-prompt retrieval) — independent entries, no cross-fact
--   interference. Doesn't participate in the forward pass.
--
-- MODE COMPOSE — FFN-overlay install:
--   Writes a single-layer slot via the install_compiled_slot pipeline
--   (gate_scale=30, up = gate_dir × u_ref, down = target_embed_unit ×
--   d_ref × alpha). Features participate in the forward pass and can
--   chain for multi-hop. Has a Hopfield-style cap at ~5–10 facts per
--   layer under template-shared prompts — run REBALANCE after a batch
--   install to fixed-point the down magnitudes. Validated 10/10
--   retrieval, 0/4 regression bleed on Gemma 3 4B (exp 14).
--
-- AT LAYER N: pins the install to a single layer. Default:
--   `knowledge.hi − 1` (L26 on Gemma 4B). Single-layer only —
--   earlier drafts used an 8-layer span but the install_compiled_slot
--   (×30 gate) math makes multi-layer spans hijack unrelated prompts.
-- CONFIDENCE: stored on the inserted features (default: 0.9 for
--   COMPOSE, 1.0 for KNN).
-- ALPHA:      COMPOSE only — per-layer down-vector scale (default:
--   0.10). Validated range ~0.05–0.30 in the Python reference.
--   Larger values push the new fact harder but dilute neighbouring
--   facts; smaller values reduce neighbour degradation.
-- MODE:       KNN (default) or COMPOSE.

INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "occupation", "engineer")
    AT LAYER 26 CONFIDENCE 0.8 ALPHA 0.3 MODE COMPOSE;
```

```
REBALANCE
    [UNTIL CONVERGED]
    [MAX <n>]
    [FLOOR <float>]
    [CEILING <float>]

-- Global fixed-point rebalance over compose-mode installs. Iterates
-- every `installed_edges` entry, INFERs its canonical prompt, and
-- scales its `down_col` until its target probability lands in the
-- [FLOOR, CEILING] band. Per-INSERT local balance is greedy and
-- breaks past N ≈ 5 on template-shared prompts; this pass converges
-- the full batch jointly.
--
-- Defaults: MAX 16, FLOOR 0.30, CEILING 0.90.
-- UNTIL CONVERGED is accepted but redundant (the loop always runs
-- until convergence or MAX iters).
-- KNN installs don't need rebalancing.

REBALANCE;
REBALANCE UNTIL CONVERGED MAX 16 FLOOR 0.30 CEILING 0.90;
```

```
DELETE FROM EDGES
    WHERE <conditions>

DELETE FROM EDGES
    WHERE entity = "John Coyle"
    AND relation = "lives-in";
```

`relation` predicates require relation labels in the active vindex
(`relation_clusters.json` / probe labels). On unlabeled vindexes, target by
`layer` + `feature` or omit the relation predicate.

```
UPDATE EDGES
    SET <field> = <value> [, <field> = <value>]...
    WHERE <conditions>

-- Update by entity (matches the feature whose top_token contains the entity).
-- Works on both heap-resident and mmap-loaded vindexes.
UPDATE EDGES
    SET target = "London"
    WHERE entity = "John Coyle"
    AND relation = "lives-in";

-- Fast path: explicit (layer, feature) bypasses the entity scan and
-- targets the slot directly. Use this when you already know which
-- feature slot you want to overwrite (e.g. from `SHOW FEATURES` output).
UPDATE EDGES
    SET target = "London", confidence = 0.95
    WHERE layer = 26 AND feature = 8821;
```

```
MERGE <source_vindex>
    [INTO <target_vindex>]
    [ON CONFLICT {KEEP_SOURCE | KEEP_TARGET | HIGHEST_CONFIDENCE}]

MERGE "medical-knowledge.vindex"
    INTO "gemma3-4b.vindex"
    ON CONFLICT HIGHEST_CONFIDENCE;
```

```
BEGIN PATCH <patch_path>

-- Start a patch session. All subsequent INSERT/DELETE/UPDATE operations
-- are captured into the patch file. The base vindex is NOT modified.
-- End with SAVE PATCH.

BEGIN PATCH "medical-knowledge.vlp";
```

```
SAVE PATCH

-- Save the current patch session to disk and end patch mode.
-- The base vindex remains unchanged.

SAVE PATCH;
-- Saved: medical-knowledge.vlp (3 operations, 30 KB)
```

```
APPLY PATCH <patch_path>

-- Apply a patch to the current vindex. Patches stack in order.
-- The base vindex files are not modified — patches are an overlay.

APPLY PATCH "medical-knowledge.vlp";
APPLY PATCH "fix-hallucinations.vlp";
```

```
REMOVE PATCH <patch_path>

-- Remove a previously applied patch from the stack.

REMOVE PATCH "fix-hallucinations.vlp";
```

```
SHOW PATCHES

-- List all currently applied patches.

SHOW PATCHES;
-- 1. medical-knowledge.vlp     (3 inserts, 30 KB)
-- 2. company-facts.vlp         (200 inserts, 2 MB)
```

```
DIFF <vindex_a> <vindex_b>
    INTO PATCH <patch_path>

-- Extract the difference between two vindexes as a portable patch file.

DIFF "gemma3-4b.vindex" "gemma3-4b-medical.vindex"
    INTO PATCH "medical-changes.vlp";
```

```
COMPILE {CURRENT | <vindex_path>} INTO VINDEX <output_path>
    [ON CONFLICT {LAST_WINS | HIGHEST_CONFIDENCE | FAIL}]

-- Flatten the current session's applied patches into a new clean vindex.
-- Path form loads that source vindex from disk as-is; use CURRENT when
-- you need to include unsaved or applied session overlays.
-- The result is a fully self-contained vindex with no overlay or sidecar:
-- the inserted features' down/gate/up vectors are written into the
-- canonical weight files (column-rewrite at the inserted slots), and
-- every other weight file is hard-linked from the source (instant, free
-- on APFS — same inode, same bytes).
--
-- INSERT handles bleed defense at install time (batch refine against
-- cached decoy residuals), so no compile-time refine step is needed.
--
-- ON CONFLICT controls how to resolve (layer, feature) slots that are
-- written by more than one applied patch:
--   LAST_WINS (default):  the last applied patch wins.
--   HIGHEST_CONFIDENCE:   accepted for forward compatibility but currently
--                         resolves like LAST_WINS for down vectors —
--                         see implementation note below.
--   FAIL:                 abort if any slot has a conflicting write.
-- ON CONFLICT is only valid for COMPILE INTO VINDEX, not COMPILE INTO MODEL.
--
-- A subsequent USE on the compiled vindex needs no special loader code:
-- it loads like any other vindex and INFER produces the inserted facts
-- through the standard dense FFN path. From this point you can also run
-- COMPILE INTO MODEL to export to safetensors / gguf — the constellation
-- is already in the bytes that get exported.

COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";

COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex"
    ON CONFLICT FAIL;
```

```
COMPILE {CURRENT | <vindex_path>} INTO MODEL <output_path> [FORMAT safetensors|gguf]

-- Compile the current vindex (with patches) or a path-loaded vindex into
-- plain model weights.
-- If the patch overlay contains INSERT operations, MEMIT closed-form
-- weight editing is used to bake the inserted facts into W_down at the
-- install layer(s). The output is a standard safetensors / gguf file
-- with no vindex dependency at inference time.
--
-- Requires model weights in the vindex (EXTRACT ... WITH ALL).

COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

> **Implementation note (HIGHEST_CONFIDENCE).** Down vectors are
> synthesised at INSERT time and stored on the base index in
> last-wins-collapsed form, so re-resolving them from raw patches at
> compile time would require regenerating the synthesised vectors.
> The compile path currently keeps last-wins semantics for down vectors
> regardless of the strategy chosen, but FAIL fully detects collisions
> and HIGHEST_CONFIDENCE is accepted for forward compatibility.

### 3.6 Schema Introspection Statements

```
SHOW RELATIONS
    [AT LAYER <n>]
    [WITH EXAMPLES]
    [{VERBOSE | BRIEF | RAW}]

-- Display modes:
--   BRIEF (default): probe-confirmed relations only.
--   VERBOSE: probe-confirmed relations + raw output tokens with scores/layers.
--   RAW:     raw output tokens only, no probe labels.

SHOW LAYERS
    [RANGE <start>-<end>]
    [{<start>-<end>}]

SHOW FEATURES <layer>
    [WHERE <conditions>]
    [LIMIT <n>]

SHOW ENTITIES
    [AT LAYER <n> | <n>]
    [LIMIT <n>]

-- Scan the loaded layers for distinct named-entity-shaped tokens
-- (uppercase start, alphabetic, length ≥ 3, not a stop word) and
-- report them sorted by feature count. Works on browse-only vindexes
-- — no model weights needed.

SHOW MODELS;

SHOW COMPONENTS;

-- VINDEX3: the system graph's components — id, role, layer count,
-- hidden size, and the per-operator attention census — reconstructed
-- purely from the container. Nothing is inferred from tensor names.

SHOW REPRESENTATIONS ["<object>"];

-- VINDEX3: the physically present representation directory, optionally
-- filtered to entries whose logical object (or entry id) contains the
-- given substring. Presence is physical: a variant listed here exists
-- as bytes; selecting one not listed fails closed (§9.1 of the ABI).

SHOW PROVENANCE ["<object>"];

-- VINDEX3: hashes and lineage per directory entry — payload and
-- segment SHA-256 (printed whole), what an entry was compiled from,
-- and what model the container derives from.

SHOW AUTHORITY;

-- VINDEX3: the container's own authority declaration — canonical
-- (source bytes present; derived representations recompilable) or
-- derived (executable; not re-compilable) — and its declared profiles.

SHOW PATCHES;

SHOW COMPACT STATUS;

-- Report the storage engine's current tier occupancy (L0 WAL/KNN,
-- L1 arch-A compose, L2 MEMIT) and the session epoch. L2 is only
-- reported when hidden_dim ≥ 1024 (MEMIT requirement).

STATS [<vindex_path>];
```

`COMPACT MINOR` and `COMPACT MAJOR` are the two tier-promotion verbs —
they live alongside `REBALANCE` in the mutation family (§3.5) but
read like introspection from the user's perspective:

```
COMPACT MINOR;
-- Promote every L0 (KNN) entry into L1 (arch-A compose edges).
-- Uses exec_insert internally with MODE COMPOSE; requires model
-- weights. A no-op when L0 is empty.

COMPACT MAJOR [FULL] [WITH LAMBDA = <float>];
-- Promote L1 compose edges to L2 MEMIT-decomposed (key, down) pairs.
-- FULL re-decomposes every L1 edge; default does only the ones the
-- current compose overlay owns. LAMBDA is the MEMIT ridge (default
-- 1e-3).
```

### 3.7 Residual Stream Trace Statements

```
TRACE <prompt>
    [FOR <token>]
    [DECOMPOSE]
    [LAYERS <start>-<end>]
    [POSITIONS {LAST | ALL}]
    [SAVE <path>]

-- Capture a residual stream trace through the forward pass.
-- Decomposes each layer into attn_delta + ffn_delta against the residual.
-- Requires model weights in the vindex (WITH INFERENCE or WITH ALL).
--
-- FOR <token>:    track this target token's rank, probability, and the
--                 attn/FFN logit contributions per layer.
-- DECOMPOSE:      print the per-layer attn vs FFN attribution table.
-- LAYERS s-e:     restrict the trace to the inclusive layer range.
-- POSITIONS LAST: trace only the last token (default).
-- POSITIONS ALL:  trace every token position in the prompt.
-- SAVE <path>:    write the captured trace to a file.

TRACE "The capital of France is";
-- Default trace, last token only

TRACE "The capital of France is" FOR "Paris";
-- Layer  Rank   Prob    Attn      FFN
-- L22     50   0.002   +22.2    +34.4   BOTH ↑
-- L23     10   0.024   -16.9    +55.9   FFN ↑
-- L24      1   0.714  +105.7    +24.4   BOTH ↑   ← phase transition
-- L25      1   0.997    +4.3    +94.4   FFN ↑

TRACE "The capital of France is"
    DECOMPOSE
    LAYERS 22-27;
-- Per-layer attn vs FFN delta table

TRACE "The capital of France is"
    POSITIONS ALL
    SAVE "france.trace";
-- All token positions, all layers, written to file
```

**VINDEX3 bindings**: plain `TRACE <prompt>` runs the *observational*
trace — subscribers on the canonical executor's operation boundaries
(embed, per-layer attention/FFN completion, output head), followed by
the greedy next token. Observation is subscription, never a second
executor: the observed and unobserved paths are gated bit-identical.
The richer clauses above (`FOR`, `DECOMPOSE`, `LAYERS`, `POSITIONS`,
`SAVE`) are V2 residual-trace features and refuse on a V3 binding
rather than silently no-op; they arrive later as further detail
levels on the same observation seam.

> Planned: `TRACE ... DIFF <prompt_b>` (cross-prompt comparison),
> tiered SAVE formats (`FORMAT boundary | context`, `TIER`, `WINDOW`),
> and `BOUNDARY OPEN` / `BOUNDARY <path> AT <n>` boundary stores.
> See §11. The wire format and capture machinery already exist in
> `larql-inference`; the LQL surface for them has not landed.

### 3.8 Comparison Operators

```
=    -- equal
!=   -- not equal
>    -- greater than
<    -- less than
>=   -- greater than or equal
<=   -- less than or equal
LIKE -- pattern match (% wildcard)
IN   -- set membership

WHERE entity = "France"
WHERE confidence > 0.5
WHERE entity LIKE "Fran%"
WHERE entity IN ("France", "Germany")
WHERE layer >= 20 AND layer <= 30
```

---

## 4. Backend Architecture

LQL abstracts over three local backends (plus the remote HTTP
forwarder). The backend is decided ONCE, at `USE` time, from the
artifact itself; no statement executor re-detects the format
afterwards.

### 4.1 The Backends

```
┌──────────────────────────────────────────────────────────────┐
│                        LQL Parser                            │
│         (same AST, same statements, same output)             │
└───────────────────────┬──────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────┐
│  VindexBackend │ │  WeightBackend │ │  Vindex3Backend    │
│                │ │                │ │                    │
│  Pre-extracted │ │ Live           │ │ Executable program │
│  KNN index     │ │ safetensors    │ │ (ComponentOpPlan)  │
│  0.98ms/layer  │ │ Dense matmul   │ │ + operand store    │
│  Read + write  │ │ Read only      │ │ Read + execute     │
│  No model      │ │ Model in       │ │ No ModelWeights,   │
│  needed        │ │ memory         │ │ no VectorIndex, no │
│                │ │                │ │ ModelArchitecture  │
└────────────────┘ └────────────────┘ └────────────────────┘
```

`USE "path"` binds a VINDEX3 container when the container's own
generation marker (`index.json` schema 3) says so — never by falling
back from a failed V2 load. The bound runtime exposes declared facts
(the executable plan) and capabilities; every V3 statement below
consumes those, and nothing reconstructs architecture from weights or
family metadata.

### 4.2 Backend Capabilities

| Statement | Vindex | Direct Weights | VINDEX3 |
|---|---|---|---|
| WALK (feature scan) | ✅ KNN (0.98ms/layer) | ✅ Dense matmul (~6ms/layer) | ✅ role `feature_gate` scan |
| DESCRIBE | ✅ Pre-computed edges + labels | ✅ On-the-fly per entity | ✅ via semantic roles |
| SELECT | ✅ Index lookup | ✅ Live gate×embedding scan | ✅ via semantic roles |
| EXPLAIN WALK | ✅ Walk trace from index | ✅ Walk trace from matmul | ✅ via semantic roles |
| INFER | ✅ With `--include-weights` | ✅ Full forward pass | ✅ Batch-prefill logits, top-k |
| INFER … GENERATE n | ❌ | ❌ | ✅ Greedy continuation (runtime seam) |
| EXPLAIN INFER | ✅ With `--include-weights` | ✅ Full forward pass + trace | ✅ Static plan explanation (§4.4) |
| TRACE | ✅ Residual decomposition | ✅ | ✅ Observational (plain form only) |
| SHOW RELATIONS | ✅ From label cache | ✅ Cluster on-the-fly (slow) | ✅ raw tokens (no labels yet) |
| SHOW LAYERS | ✅ From metadata | ✅ Computed from weights | ✅ Per-layer plan facts |
| SHOW FEATURES | ✅ Index lookup | ✅ Dense scan per layer | ✅ via semantic roles |
| STATS | ✅ Instant | ✅ Computed | ✅ Container's own authority |
| SHOW COMPONENTS | ❌ V3 concept | ❌ V3 concept | ✅ from the system graph |
| SHOW REPRESENTATIONS | ❌ V3 concept | ❌ V3 concept | ✅ the physical directory |
| SHOW PROVENANCE | ❌ V3 concept | ❌ V3 concept | ✅ digests + lineage |
| SHOW AUTHORITY | ❌ V3 concept | ❌ V3 concept | ✅ the index's declaration |
| INSERT (MODE KNN, default) | ✅ | ❌ Error: "requires vindex" | ✅ key from plan taps (§4.4) |
| INSERT MODE COMPOSE | ✅ | ❌ Error: "requires vindex" | ✅ via the operand-source seam¹ |
| DELETE | ✅ | ❌ Error: "requires vindex" | ✅ overlay tombstones |
| UPDATE | ✅ | ❌ Error: "requires vindex" | ✅ overlay meta overrides |
| BEGIN/SAVE/APPLY/REMOVE PATCH | ✅ | ❌ Error: "requires vindex" | ✅ vector-free patches (all-or-nothing) |
| SHOW PATCHES | ✅ | ❌ | ✅ overlay's applied list |
| COMPILE INTO VINDEX | ✅ | ❌ Error: "requires vindex" | ✅ bake via the operand-source seam² |
| COMPILE INTO MODEL | ✅ | ❌ Error: "requires vindex" | ❌ container→checkpoint export, later |
| DIFF | ✅ | ⚠️ One side can be weights | ✅ logical-first³ (+ PHYSICAL mode) |
| MERGE | ✅ | ❌ Error: "requires vindex" | ✅ V2 source into the overlay |
| COMPACT MINOR/MAJOR | ✅ LSM tier promotion | ❌ | ❌ tier model is V2's; later |
| COMPACT INTO VINDEX | — (V3 statement) | ❌ | ✅ physical reorganisation⁴ |

A VINDEX3 refusal is a *capability* statement, not a format apology:
the error names what the binding supports. Since V3-LQL-3A the browse
surface (SELECT / DESCRIBE / WALK / EXPLAIN WALK / SHOW …) executes on
V3 through the container's own semantic roles. Since V3-LQL-3B the
default (KNN) INSERT, DELETE, UPDATE, MERGE, and the patch lifecycle
execute too: the container stays immutable on disk and edits live in a
**knowledge overlay** addressed by semantic identity — entity-keyed
retrieval entries plus feature-slot meta overrides and tombstones
carrying V2's tombstone/resurrection contract — with the KNN key
captured from the plan's own execution taps and the same `.vlp` patch
language V2 speaks: one patch file applies to either format
— vector-bearing compose edits included since the compose rung: the
overlay's gate/up rows and down columns reach execution through the
**operand-source seam** (base representation + overlay override →
effective operand), so a compose install alters what the program
computes, WALK surfaces the slot, TRACE observes the same effective
program, and REMOVE PATCH returns execution bit-for-bit to baseline.
¹ The V3 compose install runs V2's full pipeline — clean-base capture
(normed FFN-input tap), decoy suppression, batch refine from raw
captures, probability-band balance, and the cross-fact regression
check — with V2 as the staged parity oracle (identical scripts emit
equivalent patches; directions within 1e-5, identical balance
decisions). Remaining refusals: COMPILE/DIFF/COMPACT (lifecycle). The
² `COMPILE CURRENT INTO VINDEX` on V3 materialises the overlay's
effective operands into rewritten segments (untouched segments
hard-linked), carries L0 knowledge as `knn_store.bin`, and the result
binds with a zero-override overlay — behaviour comes from the stored
bytes (equivalence-gated: INFER/GENERATE/TRACE/WALK exact). Refusals
follow the derived-annotation authority: tombstones and meta-only
relabels have no clean-container form and refuse rather than drop.
³ V3 `DIFF` reports **model facts** — knowledge edges, feature-slot
value changes (gate row / up row / down column), representation-level
tensor changes, metadata — comparing EFFECTIVE states (`CURRENT` = the
bound container plus its overlay). Gated as the COMPILE oracle in
reverse: an overlay and its bake diff identically against any third
side and diff each other as equivalent, while `DIFF … PHYSICAL` (the
subordinate segment-hash report) shows the rewrite. Mixed-generation
diffs and `INTO PATCH` are later rungs.
⁴ `COMPACT INTO VINDEX` preserves meaning exactly while reorganising
storage (today: garbage collection — unreferenced files dropped and
named, referenced segments carried byte-identically). The logical
DIFF is its proof instrument (`SemanticDiff(input, output) = ∅`), and
it refuses while the session holds overlay state: COMPILE materialises
meaning, COMPACT reorganises it — never a second compiler.
The whole-language sweep test
(`tests/vindex3_lql.rs::every_statement_is_sensible_on_a_v3_binding`)
pins that every statement either executes or refuses this way — never
a panic, never a misleading "no backend loaded".

### 4.3 Promotion: Weights → Vindex

Direct weight access is the on-ramp. When someone hits a mutation, LQL nudges:

```
larql> USE MODEL "google/gemma-3-4b-it";
Using model: google/gemma-3-4b-it (8.1 GB, live weights)

larql> INSERT INTO EDGES (entity, relation, target)
   ...   VALUES ("John Coyle", "lives-in", "Colchester");
Error: INSERT requires a vindex. Run:
  EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";
  USE "gemma3-4b.vindex";
```

### 4.4 The VINDEX3 Binding

A VINDEX3 container is an **executable model program**: a closed,
operand-verified `ComponentOpPlan` plus the operand bytes it names.
The binding holds the opened runtime and (when the container carries
`tokenizer.json`) the text capability — structurally no
`ModelWeights`, `VectorIndex`, or `ModelArchitecture`, so the V2
paths are unreachable from it, not merely avoided.

Statement semantics on a V3 binding:

- `USE "container"` — binds on the container's generation marker and
  reports the container's self-declared name (`index.model`).
- `STATS` — the program's authority: generation, component, execution
  closure, layer count, sliding/full attention split, plan-derived KV
  geometry, output head, tokenizer capability, supported statements.
- `SHOW LAYERS` — per-layer attention facts off the plan (mode,
  window, Q/KV heads, head dim).
- **Browse (V3-LQL-3A)** — `SELECT` / `DESCRIBE` / `WALK` /
  `EXPLAIN WALK` / `SHOW RELATIONS/FEATURES/ENTITIES` run against the
  container's **query surface**
  (`format::vindex3::knowledge::KnowledgeView`): semantic roles
  (`feature_gate`, `feature_down`, `embedding`) bound to the
  executable plan's operands and loaded through the same operand
  store execution uses. No `VectorIndex` is manufactured and no V2
  loader runs. Annotation semantics match the V2 extractor's
  user-visible contract (`embed · feature_down`, top logit as
  `c_score`), pinned by the **V2→V3 LQL compatibility gate**
  (`tests/vindex3_v2_parity.rs`): one checkpoint realised as both
  formats must report the same logical results for the same script —
  reads (feature space, walks, DESCRIBE), mutations (identical
  INSERT/DELETE/UPDATE/MERGE scripts leave identical logical state),
  and patch stacking (replay order determines visible state). The gate
  is the release criterion: VINDEX3 does not become the default
  binding until every statement family V2 promises is green in it.
- `INFER "…" [TOP n]` — single-step top-k next-token prediction from
  batch-prefill logits. `INFER "…" GENERATE n` — greedy continuation
  through the proven runtime seam (batch prefill into caller-owned
  continuation state, resumed decode). Requires the tokenizer
  capability; a tokenizer-less container binds and answers
  `STATS`/`EXPLAIN` but refuses text inference.
- `EXPLAIN INFER "…"` — static, deterministic rendering of the
  structured `ExplainPlan` (larql-inference): execution-ordered ops
  per layer, attention geometry, FFN kind, continuation geometry, and
  operand provenance (`object → tensor → dtype/shape` — the exact
  coordinates the operand store resolves at execution time). No
  tokens run.
- `TRACE "…"` — the observational trace: subscribers on the canonical
  executor's step boundaries, then the greedy next token. Gated
  bit-identical to untraced execution.

Everything else refuses with the capability list. The refusals are
the seed of a future `ModelCapabilities` surface, deliberately left
as plain declared facts until `WALK`/`PATCH`-class pressure shapes
the type.

---

## 5. Label Architecture

Feature labels come from the `larql-knowledge` project and are stored in the vindex as JSON files. The engine reads them; it does not produce them.

### 5.1 Label Sources (Priority Order)

| Priority | Source | Confidence | Layer Band | Description |
|----------|--------|------------|------------|-------------|
| 1 | Probe-confirmed | Highest | Knowledge | Model inference confirmed this feature encodes this relation |
| 2 | Wikidata output matching | High | Knowledge | Cluster outputs match Wikidata objects |
| 3 | WordNet output matching | High | Syntax | Cluster outputs match WordNet pairs |
| 4 | AST output matching | High | Syntax | Cluster outputs match code AST pairs |
| 5 | Entity pattern detection | Medium | Any | Cluster members match known lists (country, language, month, number) |
| 6 | Morphological detection | Medium | Syntax | Cluster members are short suffixes/prefixes |
| 7 | TF-IDF top tokens | Low | Any | Fallback: most distinctive tokens in the cluster |

### 5.2 Vindex File Layout

The vindex IS the model, reorganised for queryability. No data is duplicated — each weight matrix is stored once in its optimal format.

```
gemma3-4b.vindex/

  # ═══ Query Index (the vindex core) ═══
  # These are the model's FFN + embedding weights in a queryable format.
  # WALK, DESCRIBE, SELECT, EXPLAIN WALK use only these files.
  
  gate_vectors.bin            # W_gate rows per layer (KNN index)
                              # ~3.3 GB (f32) or ~1.7 GB (f16)
  
  embeddings.bin              # W_embed matrix (token lookup)
                              # ~2.5 GB (f32) or ~1.3 GB (f16)
  
  down_meta.bin               # W_down decoded: per-feature top token IDs + scores
                              # ~2 MB (binary) — replaces 160 MB JSONL
  
  # ═══ Inference Weights (for INFER, not duplicated) ═══
  # Only the weights NOT already in the query index.
  # INFER loads these in addition to the query index.
  
  attn_weights.bin            # Q, K, V, O attention matrices per layer
                              # ~3 GB (f16)
  
  # ═══ Compile Weights (for COMPILE, not duplicated) ═══
  # Additional weights needed to reconstruct full safetensors.
  
  up_weights.bin              # W_up per layer
                              # ~3.3 GB (f16)
  
  norms.bin                   # LayerNorm parameters per layer
                              # ~1 MB
  
  lm_head.bin                 # Output projection (unembed)
                              # ~1.3 GB (f16) — or shared with embeddings if tied
  
  # ═══ Metadata & Labels ═══
  
  index.json                  # Config: layers, hidden_size, vocab_size,
                              # component manifest, build info
  tokenizer.json              # Tokenizer (~32 MB)
  relation_clusters.json      # Cluster centres, labels, counts
  feature_clusters.jsonl      # Per-feature cluster assignments
  feature_labels.json         # Probe-confirmed labels (from larql-knowledge)
```

**Size by use case:**

| Use Case | Files Loaded | Size (f16) | Size (f32 current) |
|----------|-------------|------------|-------------------|
| Browse only (WALK, DESCRIBE, SELECT) | gate + embed + down_meta + labels | ~3 GB | ~6 GB |
| Browse + Inference (+ INFER) | Above + attn_weights | ~6 GB | ~9 GB |
| Full (+ COMPILE) | All files | ~10 GB | ~16 GB |

**Compared to original model:**

```
Original HuggingFace model (f16):    ~8 GB
Vindex browse-only (f16):            ~3 GB  (62% smaller)
Vindex browse + infer (f16):         ~6 GB  (25% smaller)
Vindex full (f16):                   ~10 GB (25% larger, but queryable + compilable)
```

Each component loads lazily — DESCRIBE never touches attention weights, INFER never touches up_weights. The vindex grows from 3GB to 10GB as you need more capabilities, but the browse-only core is always small.

**Deduplication principle:** gate_vectors.bin IS W_gate. embeddings.bin IS W_embed. They are not copies — they are the canonical storage. COMPILE reads gate_vectors.bin to reconstruct W_gate in safetensors format. No data is stored twice.

### 5.3 Layer-Aware Matching

| Layer Band | Name | Reference Databases | Typical Relations |
|------------|------|--------------------|--------------------|
| Syntax (early layers) | Morphological + Syntactic | Morphological lexicon, WordNet, English grammar, AST pairs | plural, gerund, synonym, determiner→noun, py:function_def |
| Knowledge (middle layers) | Factual + Relational | Wikidata triples, probe labels | capital, language, continent, occupation, genre |
| Output (late layers) | Formatting | None (formatting) | TF-IDF fallback only |

Layer band boundaries are model-specific and stored in `index.json`:

```json
{
  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  }
}
```

Default boundaries are computed during EXTRACT by analysing feature distributions. Models with more layers have wider bands. The boundaries can be overridden manually.

**Example boundaries by model:**

| Model | Layers | Syntax | Knowledge | Output |
|-------|--------|--------|-----------|--------|
| Gemma 3 4B | 34 | 0-13 | 14-27 | 28-33 |
| Llama 3 8B | 32 | 0-12 | 13-25 | 26-31 |
| Llama 3 70B | 80 | 0-30 | 31-65 | 66-79 |
| Mistral 7B | 32 | 0-12 | 13-25 | 26-31 |
| GPT-2 | 12 | 0-4 | 5-9 | 10-11 |

These are estimates. The actual boundaries are discoverable via `SHOW LAYERS` — the layer where factual features start appearing marks the syntax/knowledge boundary.

### 5.4 DESCRIBE Output Format

**Brief (default):** compact, top edges only.

```
larql> DESCRIBE "France";
France
  Edges (L14-27):
                 → Paris                    9.2  L27
                 → français                14.7  L23
                 → Europe                  14.4  L25
```

**Verbose:** relation labels in brackets, also-tokens, layer ranges.

```
larql> DESCRIBE "France" VERBOSE;
France
  Edges (L14-27):
    [capital]      → Paris                    9.2  L27      1x  also: Francia, París
    [language]     → français                14.7  L23      1x  also: French, oiseaux
    [continent]    → Europe                  14.4  L10-25   4x  also: Belgique, Lesotho
    [—]            → Spain                   13.3  L18      1x  also: España, Germany
    [—]            → Senegal                  9.4  L16      1x  also: Algeria, Moroccan
    [—]            → Channel                  7.2  L26      1x  also: channels, Channels
  Output (L28-33):
    [—]            → French                  35.2  L15-32   8x  also: Conseil
    [—]            → European                11.9  L11-33   7x  also: EUROPE, Euro, German
```

Labelled features show `[relation]` (from probe or cluster). Unlabelled features show `[—]` — model-discovered associations the probes didn't cover. The `also:` column shows what cluster each feature belongs to.

**Raw:** no labels, pure model signal.

```
larql> DESCRIBE "France" RAW;
France
  Edges (L14-27):
                 → Paris                    9.2  L27      1x  also: Francia, París
                 → français                14.7  L23      1x  also: French, oiseaux
                 → Europe                  14.4  L10-25   4x  also: Belgique, Lesotho
```

---

## 6. Pipe Operator

LQL supports `|>` to chain statements. The output of the left statement becomes context for the right.

```sql
WALK "The capital of France is" TOP 5
    |> EXPLAIN WALK "The capital of France is";

DESCRIBE "France"
    |> DIFF WITH "llama3-8b.vindex";
```

---

## 7. The Demo Script

One terminal. One language. The full loop.

```sql
-- ═══════════════════════════════════════════════════════
-- ACT 1: DECOMPILE
-- ═══════════════════════════════════════════════════════

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex"
    WITH ALL;
-- Extracts browse index + inference weights + compile weights (~10 GB)

USE "gemma3-4b.vindex";
STATS;

-- ═══════════════════════════════════════════════════════
-- ACT 2: INSPECT
-- ═══════════════════════════════════════════════════════

SHOW RELATIONS WITH EXAMPLES;

DESCRIBE "France" VERBOSE;
-- France
--   Edges (L14-27):
--     [capital]      → Paris                    9.2  L27      1x  also: Francia, París
--     [language]     → français                14.7  L23      1x  also: French, oiseaux
--     [continent]    → Europe                  14.4  L10-25   4x  also: Belgique, Lesotho
--     [—]            → Spain                   13.3  L18      1x  also: España, Germany

DESCRIBE "Einstein" VERBOSE;
-- Einstein
--   Edges (L14-27):
--     [—]            → phys                     7.1  L27      1x  also: physics, quantum
--     [—]            → astronomy                6.1  L25      1x  also: planets, science

DESCRIBE "def" SYNTAX;
-- def
--   py:function_def → init, forward, main  (ast)

-- ═══════════════════════════════════════════════════════
-- ACT 3: WALK + INFER
-- ═══════════════════════════════════════════════════════

WALK "France" TOP 10;

EXPLAIN WALK "The capital of France is";

INFER "The capital of France is" TOP 5 COMPARE;
-- Walk prediction:  Is (99.88%)  — no attention, wrong
-- Infer prediction: Paris (97.91%) — with attention, correct

-- ═══════════════════════════════════════════════════════
-- ACT 4: EDIT
-- ═══════════════════════════════════════════════════════

DESCRIBE "John Coyle";
-- John Coyle
--   (no edges found)

INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

DESCRIBE "John Coyle";
-- John Coyle
--   lives-in → Colchester  (inserted)

-- ═══════════════════════════════════════════════════════
-- ACT 5: RECOMPILE
-- ═══════════════════════════════════════════════════════

DIFF "gemma3-4b.vindex" CURRENT;
-- 1 edge added: John Coyle → lives-in → Colchester

COMPILE CURRENT
    INTO MODEL "gemma3-4b-edited/"
    FORMAT safetensors;
-- Compiling 348,161 features across 34 layers...
-- Written: gemma3-4b-edited/model.safetensors
```

---

## 8. Implementation Notes

### 8.1 Parser Architecture

```
Input string
    → Lexer (tokenise keywords, strings, numbers, operators)
    → Parser (recursive descent, one statement at a time)
    → AST (Statement enum with variants per statement type)
    → Executor (dispatches to larql-core / larql-inference / larql-models)
```

The parser and executor live in `larql-lql`, organized as modular
subfiles. Each LQL verb gets its own file; orchestrator modules are
kept tight so the top-down flow reads clearly.

```
src/
  lib.rs                    Module exports
  ast.rs                    AST definitions (Statement enum + support types)
  error.rs                  Error types
  lexer.rs                  Tokenizer (90+ keywords)
  relations.rs              Relation classifier + label loader
  repl.rs                   REPL & batch runner
  parser/
    mod.rs                  Parser struct, dispatch, parse()
    lifecycle.rs            EXTRACT, COMPILE, DIFF, USE, COMPACT
    query.rs                WALK, INFER, SELECT, DESCRIBE, EXPLAIN
    mutation.rs             INSERT (ALPHA, MODE), DELETE, UPDATE, MERGE, REBALANCE
    patch.rs                BEGIN/SAVE/APPLY/REMOVE PATCH + DIFF INTO PATCH
    trace.rs                TRACE
    introspection.rs        SHOW + STATS (includes COMPACT STATUS, ENTITIES)
    helpers.rs              Token utilities, value/field/condition parsers
    tests.rs                Parser tests (146)
  executor/
    mod.rs                  Session + execute() dispatch + patch session helpers
    backend.rs              Backend enum (Vindex/Weight/Remote) + require_* accessors
    helpers.rs              format_number/format_bytes/dir_size, content-token filter
    compact.rs              COMPACT MINOR / MAJOR (tier promotion)
    remote.rs               HTTP forwarding for the Remote backend
    trace.rs                TRACE executor
    introspection.rs        SHOW + STATS + SHOW COMPACT STATUS + SHOW ENTITIES
    lifecycle/
      mod.rs                submodule declarations
      use_cmd.rs            USE
      extract.rs            EXTRACT
      stats.rs              STATS
      diff.rs               DIFF [INTO PATCH]
      compile/
        mod.rs              exec_compile dispatch + MEMIT fact collector
        into_model.rs       COMPILE INTO MODEL (MEMIT-gated via LARQL_MEMIT_ENABLE)
        into_vindex.rs      COMPILE INTO VINDEX + collision detection
        bake.rs             patch_{down,gate,up}_weights + apply_memit_deltas + tests
    query/
      mod.rs                shared resolve_bands helper
      walk.rs               WALK
      infer.rs              INFER
      describe.rs           DESCRIBE (MoE path + describe_* helpers)
      select.rs             SELECT {EDGES, FEATURES, ENTITIES} + NEAREST TO
      explain.rs            EXPLAIN WALK
      infer_trace.rs        EXPLAIN INFER (attention + logit lens)
    mutation/
      mod.rs                submodule declarations
      delete.rs             DELETE
      update.rs             UPDATE
      merge.rs              MERGE
      rebalance.rs          REBALANCE
      insert/
        mod.rs              exec_insert orchestrator
        knn.rs              MODE KNN (KnnStore retrieval override)
        plan.rs             Phase 1 — target embed + layer selection
        capture.rs          Phase 1b — canonical + decoy residual capture
        compose.rs          Phase 2 — install_slots + cliff-breaker refine + tests
        balance.rs          Phase 3 — balance + cross-fact regression check
    tests.rs                Executor tests (93 integration + 17 in-module unit)
```

### 8.2 AST

Canonical definition lives in `crates/larql-lql/src/ast.rs`. The shape
below is kept in sync with the real enum — if they diverge, the source
file wins.

```rust
pub enum Statement {
    // ── Lifecycle ──
    Extract { model: String, output: String,
              components: Option<Vec<Component>>, layers: Option<Range>,
              extract_level: ExtractLevel },
    Compile { vindex: VindexRef, output: String,
              format: Option<OutputFormat>, target: CompileTarget,
              on_conflict: Option<CompileConflict> },
    Diff { a: VindexRef, b: VindexRef, layer: Option<u32>,
           relation: Option<String>, limit: Option<u32>,
           into_patch: Option<String> },
    Use { target: UseTarget },

    // ── Query ──
    Walk { prompt: String, top: Option<u32>, layers: Option<Range>,
           mode: Option<WalkMode>, compare: bool },
    Infer { prompt: String, top: Option<u32>, compare: bool },
    Select { source: SelectSource, fields: Vec<Field>,
             conditions: Vec<Condition>, nearest: Option<NearestClause>,
             order: Option<OrderBy>, limit: Option<u32> },
    Describe { entity: String, band: Option<LayerBand>,
               layer: Option<u32>, relations_only: bool,
               mode: DescribeMode },
    Explain { prompt: String, mode: ExplainMode, layers: Option<Range>,
              band: Option<LayerBand>, verbose: bool, top: Option<u32>,
              relations_only: bool, with_attention: bool },

    // ── Mutation ──
    Insert { entity: String, relation: String, target: String,
             layer: Option<u32>, confidence: Option<f32>,
             alpha: Option<f32>, mode: InsertMode },
    Delete { conditions: Vec<Condition> },
    Update { set: Vec<Assignment>, conditions: Vec<Condition> },
    Merge { source: String, target: Option<String>,
            conflict: Option<ConflictStrategy> },
    Rebalance { max_iters: Option<u32>, floor: Option<f32>,
                ceiling: Option<f32> },

    // ── Introspection ──
    ShowRelations { layer: Option<u32>, with_examples: bool,
                    mode: DescribeMode },
    ShowLayers { range: Option<Range> },
    ShowFeatures { layer: u32, conditions: Vec<Condition>,
                   limit: Option<u32> },
    ShowEntities { layer: Option<u32>, limit: Option<u32> },
    ShowModels,
    Stats { vindex: Option<String> },
    ShowCompactStatus,
    CompactMinor,
    CompactMajor { full: bool, lambda: Option<f32> },

    // ── Patch ──
    BeginPatch { path: String },
    SavePatch,
    ApplyPatch { path: String },
    ShowPatches,
    RemovePatch { path: String },

    // ── Trace ──
    Trace { prompt: String, answer: Option<String>, decompose: bool,
            layers: Option<Range>, positions: Option<TracePositionMode>,
            save: Option<String> },

    // ── Pipe ──
    Pipe { left: Box<Statement>, right: Box<Statement> },
}

pub enum VindexRef { Path(String), Current }

pub enum UseTarget {
    Vindex(String),
    Model { id: String, auto_extract: bool },
    Remote(String),
}

pub enum LayerBand { Syntax, Knowledge, Output, All }

// DESCRIBE's output default is Brief (compact). Verbose / Raw are
// opt-in via the VERBOSE / RAW keyword on the DESCRIBE statement.
pub enum DescribeMode { Verbose, Brief, Raw }

pub enum ExplainMode { Walk, Infer }
pub enum WalkMode { Hybrid, Pure, Dense }
pub enum OutputFormat { Safetensors, Gguf }
pub enum CompileTarget { Model, Vindex }
pub enum CompileConflict { LastWins, HighestConfidence, Fail }
pub enum ConflictStrategy { KeepSource, KeepTarget, HighestConfidence }
pub enum Component { FfnGate, FfnDown, FfnUp, Embeddings, AttnOv, AttnQk }
pub enum SelectSource { Edges, Features, Entities }
pub enum TracePositionMode { Last, All }

// INSERT install mode. Default is Knn (Architecture B retrieval
// override). Compose is the FFN-overlay install validated in
// ~/chris-source/chris-experiments/compilation/14_vindex_compilation
pub enum InsertMode { Knn, Compose }

pub enum ExtractLevel {
    Browse,     // Default: gate + embed + down_meta (~3 GB)
    Inference,  // + attention weights (~6 GB)
    All,        // + up, norms, lm_head (~10 GB, enables COMPILE)
}
```

> **Note.** `Rebalance`, `ShowEntities`, `ShowCompactStatus`,
> `CompactMinor`, `CompactMajor`, the full patch family, `Trace`, and
> `UseTarget::Remote` were added after the first-cut AST; `InsertMode`,
> `alpha` on `Insert`, `on_conflict` on `Compile`, `into_patch` on
> `Diff`, and `SelectSource` on `Select` are likewise additions. Any
> earlier external doc that lists a smaller enum is stale.

### 8.3 Crate Mapping

| Statement | Crate | Function |
|---|---|---|
| EXTRACT | `larql-models` | Read safetensors → write vindex |
| COMPILE | `larql-models` | Read vindex → write safetensors |
| WALK | `larql-inference` | Gate KNN on VectorIndex |
| INFER | `larql-inference` | predict_with_ffn (attention + walk FFN) |
| SELECT | `larql-core` | Edge query on graph |
| INSERT/DELETE/UPDATE | `larql-core` | Graph mutation |
| DESCRIBE | `larql-inference` | Multi-layer gate KNN + label lookup |
| EXPLAIN | `larql-inference` | Walk/infer with trace capture |
| MERGE | `larql-core` | Graph union |
| DIFF | `larql-core` | Graph comparison |
| SHOW/STATS | `larql-core` + `larql-models` | Metadata queries |
| USE | `larql-lql` | Session state |

### 8.4 Implementation Status

| Component | Status |
|---|---|
| LQL Parser | ✅ Done — recursive descent, 90+ keywords, modular subfiles |
| REPL | ✅ Done — rustyline, history, multi-line, help |
| USE / STATS | ✅ Done — vindex loading, stats display |
| SHOW (RELATIONS, LAYERS, FEATURES, MODELS) | ✅ Done |
| SELECT / DESCRIBE | ✅ Done — vindex edge query, layer band grouping |
| DESCRIBE layer bands (SYNTAX/KNOWLEDGE/OUTPUT/ALL) | ✅ Done — all bands, per-family boundaries from config |
| WALK / EXPLAIN WALK | ✅ Done — gate KNN, per-layer feature trace |
| INFER | ✅ Done — full forward pass with walk FFN (requires `--include-weights`) |
| EXPLAIN INFER | ✅ Done — inference trace with relation labels |
| Label loading (feature_labels.json) | ✅ Done — probe-confirmed labels override cluster labels |
| Cluster-based labels (relation_clusters.json) | ✅ Done — k=512, offset clustering, Wikidata + WordNet + pattern matching |
| EXTRACT | ✅ Done — full pipeline: gate, embed, down_meta, clustering, split weights |
| INSERT | ✅ Done — cluster-centre gate synthesis, auto-layer, patch overlay (base readonly) |
| DELETE | ✅ Done — by layer+feature or entity match, patch overlay |
| UPDATE | ✅ Done — target/confidence update, patch overlay |
| COMPILE INTO VINDEX | ✅ Done — bake_down patches into clean vindex |
| COMPILE INTO MODEL | ✅ Done — reconstructs safetensors from split weight files |
| DIFF | ✅ Done — feature-level comparison, INTO PATCH export |
| MERGE | ✅ Done — graph union with KeepSource/KeepTarget/HighestConfidence strategies |
| BEGIN/SAVE/APPLY/SHOW/REMOVE PATCH | ✅ Done — full patch lifecycle |
| Auto-patch on mutation | ✅ Done — INSERT/DELETE/UPDATE auto-start anonymous patch session |
| INSERT MODE {KNN, COMPOSE} | ✅ Done — KNN default (Architecture B), COMPOSE FFN-overlay validated exp 14 |
| REBALANCE | ✅ Done — global fixed-point rebalance over compose installs |
| COMPACT MINOR / MAJOR | ✅ Done — L0 → L1 → L2 tier promotion, `SHOW COMPACT STATUS` |
| SHOW ENTITIES | ✅ Done — named-entity scan across loaded layers |
| TRACE | ✅ Done — residual decomposition with FOR/DECOMPOSE/POSITIONS/SAVE |
| Readonly base | ✅ Done — base vindex files never modified, all edits via PatchedVindex overlay |
| Split weight files | ✅ Done — attn, up, down, norms, lm_head (no gate duplication) |
| f16 storage | ✅ Done — `--f16` flag, halves file sizes |
| Vindexfile | ✅ Done — declarative builds (FROM + PATCH + INSERT), `larql build` CLI |
| USE REMOTE | ✅ Done — HTTP client to larql-server, all queries forwarded, local patch overlay |
| `larql serve` | ✅ Done — HTTP/gRPC server, all endpoints, multi-model, per-session patches |
| WeightBackend (USE MODEL) | ✅ Done — direct safetensors, INFER/EXPLAIN INFER/STATS; browse ops guide to EXTRACT |
| GGUF output format | 🔴 Planned — COMPILE INTO MODEL FORMAT gguf |
| MXFP4 browse quality | 🟡 Known limitation — gate KNN noisy for 4-bit quantized MoE; INFER works correctly |
| Gated KNN for MoE | 🔴 Planned — use SiLU(gate)×up instead of raw gate dot product for MXFP4 models |
| Residual-based DESCRIBE | 🔴 Planned — capture actual residuals for accurate MoE knowledge browse |

### 8.5 INSERT Semantics — How Edge Becomes Vector

When you `INSERT ("John Coyle", "lives-in", "Colchester")`:

1. **Find the relation direction.** Look up the "lives-in" cluster centre from schema discovery. The probe labels provide the geometric direction for known relations.
2. **Find the entity embedding.** Look up "John" and "Coyle" token embeddings. Combine to get the entity vector.
3. **Find the target embedding.** Look up "Colchester" token embedding.
4. **Synthesise the gate vector.** Gate direction ≈ entity embedding, scaled to match existing gate magnitudes at the target layer.
5. **Synthesise the down vector.** Down direction ≈ target embedding, scaled to match existing down magnitudes.
6. **Synthesise the up vector.** Up direction ≈ gate direction (for simple facts).
7. **Find a free feature slot.** Use an unused feature (low activation across all entities).
8. **Write the vectors.** Update gate_vectors.bin, down metadata, and up vectors in the vindex.

The relation type determines which layer to write to. Probe data shows which layers each relation type occupies (e.g., capital features cluster at L24-27, language features at L22-32).

### 8.6 Priority Order for Implementation

1. ~~LQL Parser + REPL~~ — **done**
2. ~~USE / STATS / SHOW~~ — **done**
3. ~~SELECT / DESCRIBE~~ — **done**
4. ~~WALK / EXPLAIN WALK~~ — **done**
5. ~~INFER / EXPLAIN INFER~~ — **done**
6. ~~Label loading (probe + cluster)~~ — **done**
7. ~~DESCRIBE SYNTAX / ALL LAYERS~~ — **done**
8. ~~EXTRACT~~ — **done**
9. ~~W_up extraction~~ — **done**
10. ~~INSERT~~ — **done**
11. ~~COMPILE~~ — **done**
12. ~~DIFF / DELETE / UPDATE / MERGE~~ — **done**
13. ~~WeightBackend (USE MODEL)~~ — **done**

---

## 9. The REPL

```
$ larql repl

   ╦   ╔═╗ ╦═╗ ╔═╗ ╦
   ║   ╠═╣ ╠╦╝ ║═╬╗║
   ╩═╝ ╩ ╩ ╩╚═ ╚═╝╚╩═╝
   Lazarus Query Language v0.1

larql> USE "output/gemma3-4b-full.vindex";
Using: output/gemma3-4b-full.vindex (34 layers, 348.2K features,
  model: google/gemma-3-4b-it, relations: 512 types, 143 probe-confirmed)

larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital        → Paris           gate=1436.9  L27-27  1x  (probe)
    language       → French          gate=35.2    L24-32  4x  (probe)
    continent      → Europe          gate=14.4    L25-25  1x  (probe)
    borders        → Spain           gate=13.3    L18-18  1x  (probe)
    country        → Australia       gate=25.1    L26-26  1x  also: Italy, Germany, Spain
  Output (L28-33):
                   → German          gate=15.0    L30-30  1x  also: Dutch, Italian
                   → European        gate=11.9    L33-33  1x  also: Europe, Europeans

larql> INFER "The capital of France is" TOP 3;
  1. Paris                (97.91%)
  2. the                  (0.42%)
  3. a                    (0.31%)

larql> INSERT INTO EDGES (entity, relation, target)
   ...   VALUES ("John Coyle", "lives-in", "Colchester");
Inserted 1 edge. Feature F8821@L26 allocated.

larql> COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;
Compiling 348,161 features across 34 layers...
Written: edited/model.safetensors (8.1 GB)
```

---

## 10. Integration with larql-knowledge

The `larql-knowledge` project produces all label artifacts. The engine consumes them.

### 10.1 Label Merge Command

```bash
# After running probes in larql-knowledge:
larql label <vindex_path> \
  --probes <path_to_feature_labels.json> \
  --triples <path_to_wikidata_triples.json> \
  --wordnet <path_to_wordnet_relations.json> \
  --ast <path_to_ast_dir>
```

### 10.2 What the Engine Reads

| File | Produced by | Used for |
|------|-------------|----------|
| `feature_labels.json` | `larql-knowledge` probe pipeline | Probe-confirmed per-feature labels |
| `relation_clusters.json` | `larql` vindex build + `larql-knowledge` triples/WordNet | Cluster-based labels |
| `feature_clusters.jsonl` | `larql` vindex build | Per-feature cluster assignments |

### 10.3 Additive Updates

Labels are additive. New probes add new feature labels without removing existing ones. New triples improve cluster labels for unprobed features. The `larql label` command merges incrementally:

```bash
# Add new probe results — keeps existing, adds new
larql label gemma3-4b.vindex \
  --probes probes/batch2/feature_labels.json

# Re-cluster with new triples — improves cluster labels only
larql label gemma3-4b.vindex \
  --triples data/wikidata_triples_v2.json
```

---

## 11. Future Extensions (Not for V1)

### 11.1 Cross-Model DIFF

Compare what one model knows versus another about the same entity or relation. The Procrustes alignment work (0.946 cosine across models) means knowledge coordinates map between architectures.

```sql
-- What does Llama know about France that Gemma doesn't?
DIFF "gemma3-4b.vindex" "llama3-8b.vindex"
    ENTITY "France";
-- Gemma has:  capital→Paris, language→French, continent→Europe
-- Llama adds: population→67M, anthem→La Marseillaise, motto→Liberté

-- Which model has more knowledge about music?
DIFF "gemma3-4b.vindex" "llama3-8b.vindex"
    RELATION "instrument";
-- Gemma: 4 features (guitar, piano, saxophone, drums)
-- Llama: 11 features (+ violin, cello, flute, trumpet, bass, harmonica, banjo)

-- Full knowledge comparison report
DIFF "gemma3-4b.vindex" "llama3-70b.vindex"
    INTO REPORT "model-comparison.md";
```

This answers a question nobody can answer today: "what does this model know that that model doesn't?" The vindex makes it a database query instead of a research project.

### 11.2 Search Across Entities

Reverse DESCRIBE — instead of "what does the model know about France?" ask "which entities have this relation?"

```sql
-- Which entities have a capital?
SELECT entity, target FROM EDGES
    WHERE relation = "capital"
    ORDER BY confidence DESC;
-- France→Paris, Germany→Berlin, Japan→Tokyo, UK→London, ...

-- What does the model think is in Europe?
SELECT entity FROM EDGES
    WHERE relation = "continent"
    AND target = "Europe";
-- France, Germany, Spain, Italy, Poland, ...

-- Find all composers the model knows
SELECT entity, target FROM EDGES
    WHERE relation = "occupation"
    AND target LIKE "%composer%";
-- Mozart→composer, Beethoven→composer, Bach→composer, ...
```

This is full-text search across the model's entire knowledge graph. No inference, no prompting — just index lookups against the gate vectors and labels.

### 11.3 Export to Standard Knowledge Graph Formats

Dump the model's knowledge as standard graph formats, making it queryable by every graph database, SPARQL endpoint, and visualisation tool in the world.

```sql
EXPORT CURRENT
    INTO "gemma3-4b-knowledge.ttl"
    FORMAT turtle;
-- RDF triples: <France> <capital> <Paris> .
--              <France> <language> <French> .

EXPORT CURRENT
    INTO "gemma3-4b-knowledge.csv"
    FORMAT neo4j;
-- CSV import files for Neo4j: nodes.csv, relationships.csv

EXPORT CURRENT
    INTO "gemma3-4b-knowledge.jsonld"
    FORMAT json-ld;
-- JSON-LD linked data format

EXPORT CURRENT
    INTO "gemma3-4b-knowledge.graphml"
    FORMAT graphml;
-- GraphML for visualisation in Gephi, yEd, Cytoscape
```

The export includes only probe-confirmed and high-confidence cluster labels — not raw TF-IDF fallbacks. The resulting knowledge graph is a curated extraction of what the model actually knows, in formats the entire graph database ecosystem can consume.

### 11.4 Temporal Snapshots

Compare vindexes built from different training checkpoints to see what the model learned between runs.

```sql
-- Build vindexes from two checkpoints of the same model
-- (or two different versions: Llama 2 vs Llama 3)

DIFF "llama2-7b.vindex" "llama3-8b.vindex"
    INTO REPORT "llama-evolution.md";
-- New relations in Llama 3: 47 types not in Llama 2
-- Removed relations: 3 types (deprecated knowledge)
-- Strengthened: capital edges +40% confidence
-- New entities: 12,000 entities Llama 3 knows that Llama 2 doesn't

DIFF "gemma2-2b.vindex" "gemma3-4b.vindex"
    RELATION "capital";
-- Gemma 2: 80 countries with capital edges
-- Gemma 3: 180 countries with capital edges (100 new)
```

Model archaeology. Track how knowledge accumulates across training runs, model families, and parameter scales. See which facts are stable across versions and which are volatile.

### 11.5 Streaming Queries

For large models with hundreds of edges per entity, return results progressively:

```sql
-- Stream edges as they're found, layer by layer
DESCRIBE "France" STREAM;
-- L14: (nothing yet)
-- L15: language → French (probe)
-- L18: borders → Spain (probe)
-- ...results appear as each layer is scanned

-- Useful for remote vindexes — start seeing results before the full scan completes
USE REMOTE "https://models.example.com/llama-70b.vindex";
DESCRIBE "Einstein" STREAM LIMIT 20;
```

The layer-level byte offsets in gate_vectors.bin enable this — each layer can be fetched and scanned independently. For remote vindexes, the client sees results from L14 while L15-27 are still downloading.

### 11.6 Planned LQL Surfaces (machinery exists, language doesn't)

These are not aspirational research — the underlying capabilities live in
`larql-inference` and `larql-vindex` today. They are listed here because
the LQL surface for them has not yet landed, and the spec previously
described grammars that did not match the parser.

- **`TRACE ... DIFF <prompt_b> [AT LAYER <n>]`** — cross-prompt comparison
  of two captured traces (cosine, delta_norm, side-by-side top-1).
- **Tiered SAVE formats on TRACE** — `FORMAT {trace | boundary | context}`,
  `TIER {1..4}`, `WINDOW <n>` clauses to drive the existing trace stores
  (see §11.8). The on-disk formats already exist; only the LQL surface
  is missing.
- **`BOUNDARY OPEN <path>` / `BOUNDARY <path> AT <n>`** — open a boundary
  store for querying and read a specific boundary residual.
- **DESCRIBE STREAM** — progressive layer-by-layer DESCRIBE, particularly
  useful with `USE REMOTE`.
- **End-to-end validation of the Rust refine + decoy pipeline on a real
  model.** `COMPILE INTO VINDEX` bakes gate/up/down overlays into a
  standalone vindex (validated 10/10 retrieval, 0/4 bleed on Gemma 3 4B).
  INSERT handles bleed defense at install time via batch refine against
  cached decoy residuals, so no compile-time refine step is needed.
  `COMPILE INTO MODEL` uses MEMIT closed-form weight editing to write
  inserted facts into W_down at the install layer(s), producing plain
  safetensors with no vindex dependency.

### 11.7 Additional Future Work

- **STEER** — relation steering via ±Δ vectors at a layer. Proven (8/8) but complex to expose cleanly.
- **CHAIN** — multi-hop via L24 residual injection. Proven (KL<0.06) but requires model loaded.
- **LIFT** — analogy operation (same edge, different node). Proven but needs model.
- **GRAPH ALGORITHMS** — PageRank, community detection, shortest path across the knowledge graph.
- **INGEST** — document → context graph extraction (Mode 5, the Apollo demo).
- **Training from graph** — compile an edited knowledge graph back to weights, bypassing gradient descent entirely.

### 11.8 Attention Template Cache (Partially Proven)

99% of attention heads produce fixed patterns across entity substitutions for the same query template (confirmed). Within a known entity set, 11D projection replaces attention perfectly (15/15, K=11). However, generalization to unseen entities fails (0/21) — the entity signal is a nonlinear function of the residual stream, not the raw embedding. Full attention replacement remains open research.

**Status:** Template fixedness confirmed. Within-set replacement proven. Generalization blocked by the nonlinear embedding→residual mapping. The residual stream trace (Section 3.7) provides the ground-truth decomposition for continued research.

### 11.9 Tiered Context Store (Proven)

The residual stream trace enables infinite context without KV cache. Boundary residuals (10 KB per 200-token window) carry the complete Markov state. Tiered storage provides accuracy/storage tradeoffs:

- **Tier 1:** 10 KB/window, 3100× compression vs KV cache (boundary residual only)
- **Tier 4 int8:** 58 KB/window, 511× compression (bit-perfect reconstruction, no replay)

370K tokens (Apollo 11 transcript): 55-110 MB vs 56 GB KV cache.

**Status:** Implemented in `trace/` module. File formats: `.bin` (full chains), `.bndx` (boundaries), `.ctxt` (tiered context). Mmap'd, append-only, zero-copy. See `docs/residual-trace.md` and `docs/residual-trace.md`.
