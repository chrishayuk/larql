# LARQL

The model IS the database. Query neural network weights like a graph database. No GPU required.

LARQL decompiles transformer models into a queryable format called a **vindex** (vector index), then provides **LQL** (Lazarus Query Language) to browse, edit, and recompile the model's knowledge.

**VINDEX3 — the successor container — is specified and explained at [vindex3.org](https://vindex3.org).** The site is where the format is taught, explored and held to its evidence (the Record); this repository is its reference implementation. Install the format-native reader from [vindex3.org/get-started](https://vindex3.org/get-started), and cite the specification via [vindex3.org/cite](https://vindex3.org/cite).

```sql
larql> USE "gemma3-4b.vindex";
Using: gemma3-4b.vindex (34 layers, 348.2K features, relations: 512 types)

larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27  (probe)
    language    → French               35.2  L24  (probe)
    continent   → Europe               14.4  L25  (probe)
    borders     → Spain                13.3  L18  (probe)

larql> INSERT INTO EDGES (entity, relation, target)
   ...   VALUES ("John Coyle", "lives-in", "Colchester");
Inserted 1 edge. Feature F8821@L26 allocated.

larql> INFER "The capital of France is" TOP 3;
  1. Paris                (97.91%)
  2. the                  (0.42%)
  3. a                    (0.31%)
```

## Table of Contents

- [Quick Start](#quick-start)
    - [Serve it over HTTP + gRPC](#serve-it-over-http--grpc)
    - [Run attention locally, FFN on another machine](#run-attention-locally-ffn-on-another-machine)
    - [MoE expert sharding — experts on CPU-only remote machines](#moe-expert-sharding--experts-on-cpu-only-remote-machines)
    - [Publish to HuggingFace — full + slices + collections](#publish-to-huggingface--full--slices--collections)
    - [Vindex Factory — recipe-driven builds](#vindex-factory--recipe-driven-builds)
    - [Pull with slice awareness](#pull-with-slice-awareness)
    - [Query via LQL](#query-via-lql)
    - [Research / interpretability tools](#research--interpretability-tools)
- [What is a Vindex?](#what-is-a-vindex)
- [Architecture](#architecture)
    - [larql-vindex](#larql-vindex)
    - [larql-kv](#larql-kv)
    - [larql-lql](#larql-lql)
- [LQL Reference](#lql-reference)
    - [Key Statements](#key-statements)
- [Patches](#patches)
- [Vindexfile](#vindexfile)
- [Model Support](#model-support)
- [Realtime speech (MOSS-TTS-Realtime)](#realtime-speech-moss-tts-realtime)
- [Benchmarks](#benchmarks)
    - [Vindex Operations](#vindex-operations)
    - [Inference Engine (Gemma 3 4B, Apple Silicon M3 Max)](#inference-engine-gemma-3-4b-apple-silicon-m3-max)
    - [MoE / grid (Gemma 4 26B A4B, M3 Max)](#moe--grid-gemma-4-26b-a4b-m3-max)
    - [Pure MoE (GPT-OSS-20B Q6_K, M3 Max)](#pure-moe-gpt-oss-20b-q6_k-m3-max)
    - [Load-bearing environment flags (serving & measurement)](#load-bearing-environment-flags-serving--measurement)
    - [Dense remote-FFN (Gemma 4 31B Q4K, M3 Max, localhost)](#dense-remote-ffn-gemma-4-31b-q4k-m3-max-localhost)
- [Residual Stream Trace](#residual-stream-trace)
    - [Tiered Context (infinite context without KV cache)](#tiered-context-infinite-context-without-kv-cache)
- [Mechanistic interpretability surface](#mechanistic-interpretability-surface)
- [Documentation](#documentation)
- [Platform Support](#platform-support)
- [Install](#install)
- [Building & Testing](#building--testing)
    - [Cross-engine correctness check](#cross-engine-correctness-check)
- [License](#license)

---

## Quick Start

```bash
# Get the binaries — prebuilt release (see "Install" below for other platforms)
curl -fsSL https://github.com/chrishayuk/larql/releases/latest/download/larql-x86_64-unknown-linux-gnu.tar.gz | tar xz
# …or build from source
cargo build --release

# Pull a pre-built vindex from HuggingFace
larql pull hf://chrishayuk/gemma-3-4b-it-vindex

# List what's cached
larql list

# Run it — one-shot or chat
larql run gemma-3-4b-it-vindex "The capital of France is"
larql run gemma-3-4b-it-vindex          # drops into chat mode

# Multi-modal — describe an image (Gemma 3 + SigLIP, prefix-only)
larql run gemma3-4b-v2 --image photo.jpg \
    --mm-weights ~/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/<hash> \
    "Describe this image in one sentence."

# Or extract locally — inference-ready at f16 by default
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex
larql run gemma3-4b.vindex "Einstein is known for"
```

`larql extract` defaults to `--level inference` (full local forward
pass) stored at f16. No flags needed for the common case.

<details>
<summary>Extract tiers and options</summary>

```bash
# Browse-only — gate KNN + embeddings, no forward pass (~3 GB for 4B)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level browse

# Attention-only — client-side slice for `run --ffn URL` (Act 2 demo)
larql extract google/gemma-3-4b-it -o gemma3-4b.attn.vindex --level attention

# Inference (default) — full local forward pass
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level inference

# All — +lm_head +COMPILE extras (largest)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level all

# Q4_K/Q6_K inline (Ollama-compatible, smallest disk footprint)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --quant q4k

# Maximum size reduction on Q4K — drop gate_vectors.bin, rebuild from
# interleaved_kquant.bin at load (~1.6 s cost on 4B, ~12 s on 31B)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex \
  --quant q4k --drop-gate-vectors

# Uniform Q4_K on FFN — gate + up + down all Q4_K (default stores
# down as Q6_K). ~30 MB/layer smaller, ~1.5–1.7× faster decode down
# matmul. Adds ~1.5 % softmax drift; top-1 / top-5 preserved.
larql extract google/gemma-4-31b-it -o gemma4-31b.vindex \
  --quant q4k --down-q4k

# Opt out of f16 (rarely wanted — doubles file sizes)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --f32

# Convert from GGUF instead of extracting from safetensors
larql convert gguf-to-vindex model.gguf -o model.vindex
```

`extract-index` is kept as a backwards-compatible alias of `extract`.

</details>

### Serve it over HTTP + gRPC

```bash
larql serve gemma3-4b.vindex --port 8080
```

Grid traffic uses **f16 wire format** by default (50% bandwidth vs f32). Opt out with `LARQL_F16_WIRE_DISABLE=1`. Enable i8 symmetric quantised residuals (75% bandwidth, opt-in) with `LARQL_I8_WIRE=1`. Wire format is negotiated per-request via `Accept`/`Content-Type` headers — and the two directions are independent as of 2026-07-24 (ADR-0025): `Content-Type` declares the inbound residual format (f32/f16/i8 request encodings), `Accept` the return format, so asymmetric pairs like f16-in/i8-return are first-class. Responses can carry an opt-in server-latency trailer (`x-larql-timing: 1`), and `larql dec-bench drift` scores any wire config's bits/char fidelity against an in-run f32 baseline (0.5% gate). Non-grid clients receive f32 unchanged.

**WebSocket streaming** on `WS /v1/stream`:
```json
// Token-by-token generation — send:
{"type": "generate", "prompt": "The capital of France is", "max_tokens": 50}
// Receive one frame per token:
{"type": "token", "text": " Paris", "index": 0}
// Final frame:
{"type": "done", "tokens": 1, "latency_ms": 48.2}
// Abort mid-generation:
{"type": "cancel"}
```
SSE token streaming is also available on `POST /v1/chat/completions` with `"stream": true`.

### Run attention locally, FFN on another machine

```bash
# Extract once, then carve deployment slices with `larql slice`.
# Either --preset or --parts a,b,c works; `--dry-run` previews.
larql extract google/gemma-4-31b-it -o gemma4-31b.vindex --quant q4k

# Client slice (7.4 GB for 31B Q4_K — attn + embed + norms + tokenizer)
larql slice gemma4-31b.vindex --preset client -o gemma4-31b.client.vindex

# Server slice (27 GB — gate + interleaved FFN + down_meta, no attention)
larql slice gemma4-31b.vindex --preset server -o gemma4-31b.server.vindex

# Server (holds the FFN half):
larql serve gemma4-31b.server.vindex --port 8080 --ffn-only

# Client (laptop — runs attention locally, FFN over HTTP):
larql run gemma4-31b.client.vindex --ffn http://server.local:8080 \
  "The capital of France is"
```

Other presets: `browse` (DESCRIBE/WALK only, no forward pass), `router`
(MoE router weights only), `expert-server` (MoE expert weights for remote
CPU serving — see below), `all` (full clone). See `larql slice --help`
for the explicit part list.

### MoE expert sharding — experts on CPU-only remote machines

For Mixture-of-Experts models (Gemma 4 26B A4B, Mixtral, etc.), the expert
bank can be served from **CPU-only machines with no GPU and no VRAM**. The
laptop runs attention and the router (hot path); the expert servers hold the
dormant majority as memory-mapped data.

```bash
# Carve the client slice (attn + embed + router — 2.1 GB for 26B A4B Q4_K)
larql slice gemma4-26b-a4b.vindex --preset expert-server \
  -o gemma4-26b-a4b.expert-server.vindex

# Two expert servers — experts 0-63 on one machine, 64-127 on another
larql serve gemma4-26b-a4b.vindex --port 8081 --experts 0-63
larql serve gemma4-26b-a4b.vindex --port 8082 --experts 64-127

# Client dispatches expert calls directly
larql run gemma4-26b-a4b.vindex \
  --moe-shards "0-63=http://expert-a:8081,64-127=http://expert-b:8082" \
  "The capital of France is"
```

The `expert-server` preset includes everything the server needs to boot and
serve `POST /v1/expert/batch` calls: embeddings, norms, the interleaved Q4K
dense FFN, the per-layer expert weights (`layers/`), tokenizer, and manifest.

**Single server** (simplest — one machine holds all experts):

```bash
larql serve gemma4-26b-a4b.vindex --port 8080
larql run  gemma4-26b-a4b.vindex --moe-shards "0-127=http://server:8080" "..."
```

**2D layer × expert grid.** Layer shards can themselves fan out to expert
servers, so both axes scale independently:

```bash
# Layer shard — runs attention for layers 0-14, delegates experts to CPU tier
larql serve gemma4-26b-a4b.vindex --port 8091 --layers 0-14 \
  --moe-shards "0-63=http://expert-a:8081,64-127=http://expert-b:8082"

# larql-router routes by layer range; client just sends --ffn to the router
larql-router --port 9090 \
  --shards "0-14=http://layer-a:8091,15-29=http://layer-b:8092"

larql run gemma4-26b-a4b.vindex --ffn http://router:9090 "..."
```

**Deploy expert servers to fly.io** (CPU-only, no GPU, tested):

```bash
# Publish the expert-server slice to HuggingFace first
larql publish gemma4-26b-a4b.expert-server.vindex \
  --repo myorg/gemma-4-26b-a4b-vindex-expert-server --slices none

# Then deploy — start.sh auto-downloads the vindex on first boot
fly deploy --app larql-expert-server --config deploy/fly/fly.toml --remote-only
```

See [`deploy/fly/`](deploy/fly/) for the Dockerfile, `fly.toml`, and startup
script. First boot downloads the vindex from HuggingFace to the persistent
volume (~2 min on fly's network); subsequent restarts are instant.

Live demo: `https://larql-expert-server.fly.dev` serves
`hf://chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server` — a real CPU-only
expert server on fly.io that you can point `--moe-shards` at.

**3-tier topology (ADR-0008).** When laptop RAM matters, split the
embedding table out to its own server:

```bash
# Attention-only client (no embed, no FFN — ~310 MB on 4B, 10× smaller than `client`)
larql slice gemma3-4b.vindex --preset attn -o gemma3-4b.attn.vindex

# Embed server slice (embed + tokenizer; paired with ADR-0008 embed-server)
larql slice gemma3-4b.vindex --preset embed -o gemma3-4b.embed.vindex
```

The 3-tier client + embed server + FFN server split unlocks the
"laptop in ~1 GB" version of the dense-remote topology for small
models. Full rationale in
[`docs/adr/0007-vindex-distribution.md`](docs/adr/0007-vindex-distribution.md)
and [`docs/adr/0008-embed-server.md`](docs/adr/0008-embed-server.md).

### Publish to HuggingFace — full + slices + collections

Every published vindex carries a versioned on-disk contract. The
`crates/larql-vindex-spec` crate defines the v1 manifest schema —
hardened provenance (pinned upstream commit + per-shard safetensors
digests), closed enums for `extract_level` / `dtype` / `quant`, and a
20 GiB shard cap. Repos stamp `library_name: larql` in their model
card so the Hub filters them at
[`huggingface.co/models?library=larql`](https://huggingface.co/models?library=larql).
The contract lives in [`crates/larql-vindex-spec/SPEC.md`](crates/larql-vindex-spec/SPEC.md).

`larql publish` combines `slice` + `hf publish` and adds HuggingFace
**collections**: one run uploads six sibling repos and files them into
three nested collections (model / family / library) for discovery.

```bash
# One command. Six repos (full + client + attn + embed + server + browse).
# Three collections (model / family / library).
larql publish gemma4-31b.vindex --repo chrishayuk/gemma-4-31b-it-vindex

# Preview without touching HF
larql publish gemma4-31b.vindex --repo chrishayuk/gemma-4-31b-it-vindex --dry-run
```

**Skip-if-unchanged.** Each upload compares the local SHA256 against the
remote `lfs.oid`. Files that already match skip the transfer. Re-publishing
a ~27 GB server slice where nothing changed re-uploads only the manifest —
not 27 GB of weights. Override with `--force-upload`.

**Streaming + progress.** Uploads stream the file (no 27 GB-into-RAM pre-read)
and report live progress via a per-file bar. An interrupted run picks up
on the next invocation: completed files skip via SHA, the interrupted
file re-uploads.

Flags: `--no-full`, `--slices client,server`, `--collections model,family`,
`--model-title`, `--family`, `--library-title`, `--slice-repo-template`,
`--force-upload`, `--dry-run`. Requires `HF_TOKEN` or
`~/.huggingface/token`.

### Vindex Factory — recipe-driven builds

[docs/vindex-factory.md](docs/vindex-factory.md) turns the above into a
reviewable pipeline: a recipe file merged to `main` is a promise that a
specific vindex exists on the Hub, built from a pinned upstream by a
pinned `larql`, verified before it went public. Recipe schema,
`build_id`, structural validation, capability manifest, card generation,
the size/cost estimate, and the PREFLIGHT→RELEASE build driver are all
built (`crates/larql-factory`); MIRROR and REGISTER stay external
(§7 — a rig worker's job, not this binary's).

```bash
# Structural validation — schema shape, thresholds, no network I/O
larql recipe validate my-recipe.yaml

# The content hash that decides no-op / verify-only / rebuild
larql recipe build-id my-recipe.yaml

# Upstream size, per-output size, executor recommendation, cost band
larql recipe estimate my-recipe.yaml

# Fetch, extract, slice, verify checksums, publish private, release public
larql recipe build my-recipe.yaml

# Which architectures this larql release recognises, and what each supports
larql capabilities

# A Hub model card from a recipe + manifest + verification report
larql card render --recipe my-recipe.yaml --manifest index.json --verification verification.json
```

### Pull with slice awareness

`larql pull` mirrors `publish` on the download side: pick a specific
sibling, pull them all, or pull a whole collection. Each file gets an
indicatif progress bar; hf-hub resumes interrupted downloads from the
`.incomplete` partial on the next run.

```bash
# Plain pull — the full vindex. Shows a hint at the end listing
# any `-client` / `-attn` / `-embed` / `-server` / `-browse` siblings
# that exist on HF.
larql pull chrishayuk/gemma-4-31b-it-vindex

# Pull just the client slice (laptop side of `run --ffn URL`)
larql pull chrishayuk/gemma-4-31b-it-vindex --preset client

# Pull full + every default sibling in one command
larql pull chrishayuk/gemma-4-31b-it-vindex --all-slices

# Pull every dataset in an HF collection — works on the collection URL
# from larql publish or the slug alone.
larql pull --collection chrishayuk/gemma-4-31b-it-larql-vindex-abc123
```

**Bounding server RSS.** `--ffn-only` skips the eager gate warmup at
startup (55 GB → 5.6 GB on 31B Q4_K). For steady-state bounds, layer
each of these on as needed:

```bash
larql serve gemma4-31b.vindex --port 8080 --ffn-only \
  --layers 0-19                    \  # hard bound: this shard serves only layers 0-19
  --max-gate-cache-layers 4        \  # LRU cap on decoded f16 gate heap
  --release-mmap-after-request        # madvise(DONTNEED) post-request (Linux strict)
```

`--layers` is the reliable hard bound on both Linux and macOS.
`--release-mmap-after-request` is strict on Linux, advisory on Darwin.
See `docs/adr/0005-ffn-service-memory-bounds.md` for the measured
ceilings under each combination.

### Query via LQL

```bash
larql repl
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
larql lql 'USE "hf://chrishayuk/gemma-3-4b-it-vindex"; DESCRIBE "France";'
```

### Research / interpretability tools

All under `larql dev <subcmd>` (weight extraction, QK rank analysis,
OV→gate projection, circuit discovery, trajectory tracing, 20+ others):

```bash
larql dev --help
larql dev walk --prompt "The capital of France is" --index gemma3-4b.vindex --predict
```

Legacy invocation `larql walk …` still works and transparently trampolines
to `larql dev walk …`.

## What is a Vindex?

A vindex is a directory containing a model's weights reorganised for queryability. Gate vectors become a KNN index. Embeddings become token lookups. Down projections become edge labels. The model IS the database.

```
gemma3-4b.vindex/
  gate_vectors.bin         # W_gate rows (KNN index, 3.3 GB)
  embeddings.bin           # W_embed matrix (token lookup, 2.5 GB)
  down_meta.bin            # Per-feature output metadata (binary)
  index.json               # Config, layer bands, provenance
  tokenizer.json           # Tokenizer
  relation_clusters.json   # Discovered relation types
  feature_labels.json      # Probe-confirmed labels
```

**Container generations.** `index.json`'s `version` is the sole discriminator —
schemas 1–2 are **VINDEX2** (what `extract` writes by default, and what every
published vindex is today), schemas 3–4 are **VINDEX3**, the 3.0 Candidate
container for model systems. One binary reads both; `larql show` and
`larql verify` dispatch on the version and describe each generation in its own
terms rather than flattening one into the other.

**VINDEX3 is a 3.0 Candidate Specification — executable, servable, and
reachable from every extraction surface; the *default* extraction is still
VINDEX2.** The `larql vindex3` command family (`plan`, `encode`, `inspect`,
`verify`, `ops`, `exec` — `larql-cli` `commands/primary/vindex3_cmd/`) plans
a model system from HF checkpoints, encodes it into a self-contained
container, proves source ≡ encoded, and executes the container's own
declared operator program with no architecture registry. Since graph
schema 6, **surfaces follow the program**: a component's attention and FFN
surface groups exist iff its declared per-layer operators run those
operations — proven live by a pure-SSM witness (mamba2-780m: 48 declared
Mamba2 operators, zero attention surfaces, generating through ordinary LQL
with the source checkpoint deleted). Whole production models — gpt-oss-20b,
Gemma 4 26B-A4B, Granite 4.1 3B/8B/30B — encode and execute
byte-identically to their HF sources, and `larql serve` serves a V3
container over `/v1/completions` via the V3 runtime (see
[`docs/vindex3-runtime.md`](docs/vindex3-runtime.md)), while
`larql run <container> [prompt]` executes one from the command line, text
in and text out, with the tokenizer the container carries. VINDEX3 extraction
is available on request (`larql extract --generation v3`, LQL
`EXTRACT ... FORMAT VINDEX3`); the *default* remains VINDEX2 until the M4
flip — a named decision, made in one place, not yet made (see
[`docs/vindex-generation-policy.md`](docs/vindex-generation-policy.md)).
See
[`crates/larql-vindex/docs/vindex3-format-spec.md`](crates/larql-vindex/docs/vindex3-format-spec.md)
(the 3.0 Candidate ABI) and [`docs/vindex3-format.md`](docs/vindex3-format.md)
(the living spec) — both explained chapter by chapter at
[vindex3.org](https://vindex3.org), with the version and its six claims at
[vindex3.org/3.0](https://vindex3.org/3.0) and the status of every claim at
[vindex3.org/ladder](https://vindex3.org/ladder).

Three extraction levels:

| Level | CLI Flag | LQL Syntax | Size (f16) | Enables |
|-------|----------|-----------|-----------|---------|
| Browse | `--level browse` | `EXTRACT MODEL ... INTO ...` (LQL default) | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | `--level inference` (CLI default) | `... WITH INFERENCE` | ~6 GB | + INFER |
| All | `--level all` | `... WITH ALL` | ~10 GB | + COMPILE |

Defaults differ by surface: `larql extract` defaults to `--level
inference` (`larql-cli` `extract_index_cmd.rs`), matching the quick
start above; a bare LQL `EXTRACT MODEL` statement defaults to browse
(`larql-lql` `parser/lifecycle.rs`). Storage is f16 by default — pass
`--f32` to opt out (doubles file sizes).

## Architecture

Two crate families. LARQL-specific crates own the vindex + LQL + server stack;
portable `model-*` crates carry primitives that any neural-model compiler
(LARQL, TinyModel, others) can consume.

```
# LARQL-specific
larql-models          Model config, architecture traits, weight loading, quant/dequant
    ↓
larql-vindex          Vindex lifecycle: extract, load, query, mutate, patch, save
    ↓
larql-core            Graph algorithms, merge, diff
larql-compute         Compute backend trait + CPU kernels; GPU backends are siblings
larql-compute-metal   Metal GPU backend (Apple Silicon, metal-rs + MSL shaders);
                      non-Mac hosts skip the build entirely
larql-inference       Forward pass, BLAS-fused attention, WalkFfn, V3 runtime
    ↓
larql-kv              Pluggable KV-cache engines — 10 implementations, state-policy
                      classified (canonical vs derivative), W10 mask cascade
    ↓
larql-lql             LQL parser, executor, REPL, USE REMOTE client
    ↓
larql-server          HTTP/gRPC server: serve vindexes over the network
larql-router          Layer-sharding router for distributed larql-server deployments
larql-router-protocol gRPC protocol types for the self-assembling grid
larql-cli             CLI commands (extract-index, build, serve, repl, convert, hf,
                      verify, vindex3)
larql-factory         Vindex Factory driver: recipe schema, build_id, capabilities,
                      card generator, build-stage driver (PREFLIGHT→RELEASE)
larql-boundary        Confidence-gated BOUNDARY ref codec (final-layer residuals →
                      contract-bearing protocol objects)
larql-python          Python bindings (PyO3) for the graph engine and vindex
larql-demos           Runnable demos of shipped capabilities (all `--example`
                      demos live here; per-crate benches stay in their crates)
larql-experts         Nested workspace of WASM virtual experts (wasm32-wasip1
                      cdylibs, JSON ABI) the engine can dispatch to

# Portable (no larql-* deps; extract to sibling repo later)
model-compute         bounded compute: native kernels (default) + wasmtime (opt-in)
larql-vindex-spec     public vindex on-disk contract: Rust types, JSON Schema,
                      validation thresholds
larql-execution       execution-refusal semantics (RefusalKind) shared across the
                      runtime crates
```

The portable crate never imports `larql-*`. Flow is one-way: LARQL consumes
it (e.g. compile-time resolution of `sum(1..100)` via `model_compute::native`).
See [crates/model-compute/README.md](crates/model-compute/README.md).

### larql-vindex

Owns the vindex lifecycle. Streaming extraction (mmap, no full model load), KNN via BLAS matmul,
zero-copy mmap loading, split weight files, readonly base with patch overlay, clustering, f16 storage.

```rust
// Load (readonly base)
let index = VectorIndex::load_vindex(&path, &mut cb)?;
let patched = PatchedVindex::new(index);

// Query
let hits = patched.gate_knn(layer, &query, 10);  // exact BLAS gemv — ~2.6ms/layer at Gemma 3 4B shape (10240×2560)
let trace = patched.walk(&query, &layers, 10);    // multi-layer scan

// Mutate (patch overlay — base files never modified)
patched.insert_feature(layer, feature, gate_vec, meta);
patched.apply_patch(VindexPatch::load("edits.vlp")?);
```

### larql-kv

**LARQL KV engines separate model continuation state from execution
cache.** Standard engines store K/V as state. Residual-state engines
store the residual stream and derive K/V only when execution needs
it. The choice changes how the engine composes with the dispatch
hot path — and as of 2026-05-21, the three derivative-K/V engines
match `standard`'s fused-kernel speed because they can elide the
GPU→CPU state bridge entirely (W10 mask cascade, default-on).

State Policy classifies every engine as a triple `(canonical_state,
derivative_state, correctness_contract)` — the same compression
ratio with K/V slotted canonical vs derivative gives a 13% tok/s
delta on Metal, which the per-engine bench numbers confirm.

| Engine | Canonical state | K/V role | Contract | Bench (tok/s) |
|---|---|---|---|---:|
| `standard` | K/V tensors | canonical | exact logits | 97.6 |
| `no-cache` | tokens | recomputed | exact logits | (debug) |
| `markov-rs` | residual stream | derivative | exact logits under arch contract | **98.0** |
| `markov-rs-codec` | compressed residuals | derivative | bounded KL | **98.1** |
| `boundary-per-layer` | per-layer codec residuals | derivative | bounded KL per-layer | **98.7** |
| `windowed-checkpoint` | KV (within window) + checkpoints | derivative | exact within window | 94.2 |
| `turbo-quant` | quantised K/V | canonical (destructive) | bounded KL | 85.0 |
| `boundary-kv` | K/V + boundary frames | canonical | exact logits | composes `standard` |
| `apollo` | boundary retrieval store | n/a (retrieval) | task-level | orthogonal |
| `semantic-promotion` | semantic authorities over a base exact engine | policy wrapper | exact decode via base engine | composes `standard` |

Gemma 3 4B Q4K, Metal, M3 Max, 50 decode tokens, W10 default-on
(2026-05-21).

> *KV cache is an implementation detail. Continuation state is the
> real abstraction.*

```bash
# Pick an engine — same trait, different state policy
larql run gemma3-4b "The capital of France is" --engine markov-rs
larql run gemma3-4b "The capital of France is" --engine markov-rs-codec
larql run gemma3-4b "The capital of France is" --engine boundary-per-layer:window=512

# Bench the ladder
larql bench gemma3-4b-q4k-v2 --engine "standard;markov-rs;boundary-per-layer:layers=34"
```

See [crates/larql-kv/README.md](crates/larql-kv/README.md) for the
full engine catalog, [crates/larql-kv/docs/state-policy.md](crates/larql-kv/docs/state-policy.md)
for the `(canonical, derivative, contract)` framing, and
[crates/larql-kv/PERFORMANCE.md](crates/larql-kv/PERFORMANCE.md)
for the bench protocol + W10 mask cascade detail.

### larql-lql

LQL parser and executor. 20+ statement types across 5 categories:

- **Lifecycle**: EXTRACT, COMPILE, DIFF, USE
- **Browse**: WALK, DESCRIBE, SELECT, EXPLAIN WALK
- **Inference**: INFER, EXPLAIN INFER
- **Mutation**: INSERT, DELETE, UPDATE, MERGE
- **Patches**: BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH
- **Introspection**: SHOW RELATIONS/LAYERS/FEATURES/MODELS/PATCHES, STATS

## LQL Reference

See [crates/larql-lql/docs/spec.md](crates/larql-lql/docs/spec.md) for the full language specification and [docs/lql-guide.md](docs/lql-guide.md) for a quick start guide.

### Key Statements

```sql
-- Decompile a model
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;

-- Browse knowledge (no GPU needed)
USE "gemma3-4b.vindex";
DESCRIBE "France";                      -- verbose by default: [relation] labels, also-tokens
DESCRIBE "Einstein" ALL LAYERS;
DESCRIBE "France" BRIEF;                -- compact view
WALK "The capital of France is" TOP 10;

-- Run inference (needs model weights in vindex)
INFER "The capital of France is" TOP 5 COMPARE;

-- Trace the residual stream (decomposed forward pass)
TRACE "The capital of France is" FOR "Paris";
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;
TRACE "The capital of France is" SAVE "france.trace";

-- Edit knowledge (auto-patch: base files never modified)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");
-- "Auto-patch started (use SAVE PATCH to persist)"

-- Insert with all knobs (multi-layer constellation, validated regime)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.95
    ALPHA 0.30;

-- Patches (lightweight, shareable knowledge diffs)
BEGIN PATCH "medical.vlp";
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");
SAVE PATCH;
APPLY PATCH "medical.vlp";

-- Bake the patches into a fresh standalone vindex (instant on APFS:
-- weight files are hardlinked from source, only down_weights.bin gets
-- the override columns rewritten in place).
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";

-- Or recompile back to standard HuggingFace / GGUF format. The
-- constellation is in the standard down_proj tensors, so loading in
-- Transformers or GGUF runtimes Just Works — no special loader code.
COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;
```

## Patches

Patches are lightweight JSON files (.vlp) that capture INSERT/DELETE/UPDATE operations. They overlay an immutable base vindex without modifying it.

```sql
-- Create a patch
BEGIN PATCH "medical-knowledge.vlp";
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
SAVE PATCH;

-- Apply patches (stackable, reversible)
APPLY PATCH "medical-knowledge.vlp";
APPLY PATCH "fix-hallucinations.vlp";
SHOW PATCHES;
REMOVE PATCH "fix-hallucinations.vlp";

-- Extract diff between two vindexes as a patch
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

A single fact is ~10 KB. A 1,000-fact domain patch is ~10 MB. Compared to the full model at 8 GB, that's 1/800th the size. No fine-tuning, no GPU, no retraining.

The base vindex is always readonly. INSERT/DELETE/UPDATE automatically create a patch overlay. Edits are never written to base files.

## Vindexfile

Declarative model builds. Like a Dockerfile for model knowledge.

```dockerfile
# Vindexfile
FROM hf://chrishayuk/gemma-3-4b-it-vindex
PATCH hf://medical-ai/drug-interactions@2.1.0
PATCH ./patches/company-facts.vlp
INSERT ("Acme Corp", "headquarters", "London")
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest
EXPOSE browse inference
```

```bash
larql build .                          # build from Vindexfile
larql build . --stage prod             # named stage
larql build . --output custom.vindex   # custom output path
```

## Model Support

Input formats: **safetensors** (HuggingFace), **GGUF** (llama.cpp, dequantized to f32), **MLX** (Apple, same safetensors layout).

| Family | Models | FFN Type |
|--------|--------|----------|
| Gemma | Gemma 2/3/4 (2B-31B) | Gated (GeGLU); Gemma 4 26B-A4B is MoE |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) |
| Mistral | Mistral 7B | Gated (SiLU) |
| Mixtral | Mixtral 8x7B, 8x22B | MoE (8 experts) |
| Qwen | Qwen 2/2.5 (0.5B-72B) | Gated (SiLU); dense and MoE variants |
| Granite | Granite 3B/8B/30B (dense + MoE) | Gated (SiLU); MoE variants routed |
| OLMoE | OLMoE | MoE (Qwen3-MoE layout, experts sized from `intermediate_size`) |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) |
| DeepSeek-V4 | DeepSeek-V4 | MoE + MLA + MXFP4, HCA attention |
| GPT-OSS | GPT-OSS-20B/120B | MoE (32/128 experts, MXFP4 → Q6_K lossless) |
| GPT-2 | GPT-2 (117M-1.5B) | Standard (GELU-tanh, vindex extraction only) |
| StarCoder2 | StarCoder2 | Standard |
| Muse-Glimmer | Muse-Glimmer text | Gated; judged gate/QK-norm semantics |
| BitNet | BitNet b1.58 | Gated (native-ternary inference via the larql-inference ternary path) |
| TinyModel | TinyModel (research-scale decoder) | Gated |
| MOSS-TTS-Realtime | 2.3B speech (RVQ audio tokens out) | Gated (SiLU) ×2 + depth transformer — see [Realtime speech](#realtime-speech-moss-tts-realtime) |

Detection is `model_type`-driven (`crates/larql-models/src/detect/mod.rs`);
anything unrecognised — Phi included — falls to `GenericArch`, which works
for standard Llama-shaped dense models but carries no family-specific
semantics.

Dense and full-precision MoE models support all operations (DESCRIBE, WALK, INFER). MXFP4-quantized MoE models (GPT-OSS) can be extracted and served but DESCRIBE/WALK produce noisy results due to 4-bit weight precision — use INFER for accurate knowledge queries. See [operations spec](crates/larql-vindex/docs/operations-spec.md) for details.

**GPT-OSS-20B is served end to end (2026-08-10).** `larql run
gpt-oss-20b-q4k.vindex` generates coherent harmony-format output on CPU
(~60 ms/token) and on Metal with the identical greedy trajectory — see
the [Pure MoE benchmark](#pure-moe-gpt-oss-20b-q6_k-m3-max) and
[`docs/k3-funnel.md`](docs/k3-funnel.md) §4.11. The experts serve from a
lossless MXFP4→Q6_K transcode; sliding-window + YaRN attention, per-layer
sinks, and all four projection biases are applied on both backends.

**Native MXFP4 experts are served end to end (2026-08-14).** `larql run
<vindex> --routed-from <container> --metal` (with `LARQL_GPU_ROUTE=1`)
composes the Q6_K spine with a VINDEX3 container's native MXFP4 expert
banks — payloads, paired e8m0 scale streams, layout and format all owned
by the container, admission complete-or-refuse with tamper-tested
negative controls. Same machine, same session, paired: Q6_K **68.3
tok/s**, native MXFP4 **77.2 tok/s** (12.95 ms/token) through the
GPU-resident route (one command buffer per token, zero route-dependent
host work).

**GPT-OSS attention sinks (2026-07-29).** GPT-OSS attention uses a learned per-head *sink* logit that competes in the softmax and is then discarded, so attention weights over real positions deliberately sum to less than one. Until 2026-07-29 larql neither extracted nor applied it — along with all four projection biases — so 5 of 11 attention tensors per layer were silently dropped and the forward pass was systematically wrong. Both are now extracted and applied on the CPU and Metal paths, with numerical parity tests against the reference implementation. Details and the measured sink magnitudes are in [`docs/k3-funnel.md`](docs/k3-funnel.md) §4.6.

**Known gap:** `larql shannon score` (and therefore `shannon verify`) uses a dense-only FFN backend, so it cannot score **any** MoE model, GPT-OSS included. Cross-engine bits/char verification is unavailable for MoE architectures until the scorer routes its FFN by architecture — see [`docs/k3-funnel.md`](docs/k3-funnel.md) §4.6.8.

GPT-2 status: GGUF conversion (`larql convert gguf-to-vindex`) lands canonical
weights — the loader transparently re-orients non-standard FFN layouts, splits
the fused `attn_qkv` projection into per-head q/k/v, and surfaces learned
`wpe` positional embeddings on `ModelWeights::position_embed`. Forward-pass
inference still requires wiring `position_embed` into the residual init and
the LayerNorm-with-bias / FFN-with-bias paths through the run-time stack;
extraction-only flows (DESCRIBE, KNN, vindex publish) work today.

## Realtime speech (MOSS-TTS-Realtime)

LARQL executes the generative half of
[MOSS-TTS-Realtime](https://github.com/OpenMOSS/MOSS-TTS) — text tokens
in, 16-codebook RVQ audio tokens out — as an ordinary model in the same
engine, **with per-token parity against the reference implementation**
(the full greedy 138-frame fixture replays token-exactly; the sampler
replicates the reference's non-standard top-p literally). The codec
stays external: LARQL emits audio tokens, `--codec-cmd` turns them into
a WAV.

```bash
# Speak, cloning the voice in aru-12.tokens (codec-encoded reference audio)
larql run <moss-tts-realtime-dir> --speak "Good evening. All systems are normal." \
  --voice aru-12.tokens --q4 \
  --codec-cmd 'python moss_codec_cli.py {tokens} {wav}' --play
```

Measured on M3 Max CPU (Q4_K weights, quiet machine, interleaved runs —
see `docs/tts-funnel.md` for the full gate log and methodology):

| Metric | Value |
|---|---|
| Steady-state generation | ~31 ms/frame p50 vs the 80 ms frame budget (**~1.9× realtime**) |
| First-turn TTFA | ~1.25 s (prefill-dominated; `<500 ms` gate in progress) |
| Streaming envelope | 0 ms minimum pre-buffer, zero cold-start underruns |
| Incremental ≡ batch | token-identical at every text chunking (gated in CI + on-checkpoint) |

The incremental session (`MossSession`) implements the reference's
push-text protocol — text ids in as they become available, frames out as
they are earned — so an LLM's token stream and a typed prompt are the
same input. Honest scope note: these are speech-**token** generation
numbers; end-to-end audio (incremental codec + PCM ring + device
output) is tracked separately on the roadmap.

Speech also drove a general engine rule now used across LARQL
(`ROADMAP.md`, "physical planning"): the same logical operator gets
different physical plans per execution phase — packed Q4K·Q8K integer
matvec for single-row decode, dequant-once + BLAS GEMM and batched
attention for multi-row prefill — selected by measured shape, not
tensor format. All of it behind the same token-exact parity oracle.

## Benchmarks

### Vindex Operations

Criterion (`cargo bench -p larql-vindex --bench vindex_ops`), M3 Max,
synthetic data. All KNN/walk rows are the **exact brute-force BLAS
gemv** — the walk hot path never consults HNSW (`enable_hnsw` only
affects gate-KNN serving consumers such as the browse `walk()` and
server KNN endpoints; pinned by `gate_walk_ignores_hnsw_toggle`).
An earlier revision of this table quoted `0.008 ms`/layer and a
`0.3 ms` 34-layer walk — those came from the reduced 1024×256
synthetic shape in the pre-2026-04-05 `vindex_bench` example
("reduced from 10240/2560/34 for bench speed"), not the production
shape below.

| Operation | Latency |
|---|---|
| Gate KNN, per layer (1024f × 256h synthetic) | 22.7 µs |
| Gate KNN, per layer (10240f × 2560h — Gemma 3 4B shape) | 2.64 ms |
| Walk (8L × 1024f × 256h synthetic) | 216 µs |
| Walk (8L × 10240f × 2560h — Gemma band) | 21.2 ms |
| Feature meta lookup | ~245 ns |
| Save gates (8 MB) | 2.0 ms |
| Load vindex (mmap) | 261 µs |
| Mutate (meta + gate) | 301 ns |

Full tables (per-shape KNN, Q4/Metal, HNSW-vs-brute, W2 feature-major
down): [crates/larql-vindex/README.md](crates/larql-vindex/README.md)
and [crates/larql-vindex/PERFORMANCE.md](crates/larql-vindex/PERFORMANCE.md).

### Inference Engine (Gemma 3 4B, Apple Silicon M3 Max)

| Operation | Latency | tok/s |
|---|---|---|
| **GPU Q4K decode (Metal, 34L, KV cache)** | **11.4ms** | **88.1** |
| **CPU Q4K decode (StandardEngine, KV cache, 8 threads)** | **~38ms** | **~26.4** |
| Walk prediction (CPU, no attention) | 33ms | 30 |
| INFER walk (CPU, with attention, mmap FFN) | 517ms | 1.9 |
| INFER dense (CPU, all matmul) | 535ms | 1.9 |
| DESCRIBE (knowledge browse) | 33ms | — |

GPU decode per-stage breakdown (post 2026-05-09 QKV defuse, ADR-016):

| Component | Time | % of total |
|---|---|---|
| GPU forward (34 layers, Q4K/Q6K, defused norm+QKV) | 11.40 ms | 86% |
| LM head (Q4_K production path) | 1.85 ms | 14% |
| Embed + norm + detokenize | <0.1ms | <1% |

vs ollama gemma3:4b on the same machine: ~103 tok/s steady → **gap 1.17×**, was 1.18× pre QKV defuse, 1.30× pre 2026-05-02 dispatch fix. Acceptance criterion (~85 tok/s, ~1.16×) effectively met.

**CPU vs llama.cpp** (reconciled 2026-06-02, M3 Max, 8 threads, warm): larql **26.4** (StandardEngine) / 23.5 (legacy `bench --cpu`) vs **llama.cpp `-ngl 0` 43.0** tok/s → **gap ~1.6–1.8×**. The gap is per-core kernel quality — both attention and FFN already run the int8 Q8_K SDOT kernel; closing it is C12 (hand-asm; an opt-in `LARQL_Q4K_ASM=1` v1 lands +~4% isolated). `larql bench --cpu` now reports both the legacy and production-StandardEngine rows; `--ollama-cpu` forces a true CPU ollama baseline (default `--ollama` runs on Metal GPU). The earlier 1.5×/1.9× spread was two measurement confounds (path mismatch + an unwarmed-ollama artifact), not a regression — see `bench/baselines/c10_gemma3-4b_cpu_reconciled.json`.

**Memory-bandwidth roofline** (measured 2026-07-29, M3 Max, AC, quiet — [`docs/diagnoses/memory-bandwidth-roofline.md`](docs/diagnoses/memory-bandwidth-roofline.md)): attainable **read** bandwidth is **127 GB/s on the CPU cluster** (saturating at *two* threads) vs **367 GB/s on the GPU**, against a 400 GB/s SoC spec — so the CPU reaches 31% of the chip's bandwidth and the GPU 92%, a **2.9× structural advantage** that no CPU kernel can close. Against that denominator the production CPU matvec runs at **50–91% of attainable** (the long-quoted "~47 GB/s effective" came from the bench's rayon arm, not the shipping path — the same shape is **98.7 GB/s** through `q4k_q8k_matvec_parallel`), and the Metal kernels at 74–85%. Practical reading: **CPU bandwidth work is essentially finished**; the bandwidth-bound half belongs on the GPU. Probe: `cargo run --release -p larql-compute --example membw_probe` (read its **spread** column before its GB/s column — a contended machine reports nonsense). KV-cache traffic was priced on the same ledger and came out small: the `sliding_window` caps 25 of 30 layers on the 26B, so f16 KV would buy only 3–8% of a token at chat-typical context and cannot explain the long-context decay — it is a client-density lever, not a speed one.

**CPU prefill** (2026-06-22): the per-layer f32 dequant — long the dominant prefill cost (~2.7 s / ~2 tok/s on the 5-token prompt) — is gone. Q/K/V/O **and** gate/up/down now project straight from the Q4_K/Q6_K vindex bytes via amortised `q4k_matmul` / `q6k_matmul` (the Q6_K twin handles the default Q6_K `v_proj` / `down_proj`) with a hand-written aarch64 NEON inner dot. Gemma 3 4B Q4_K CPU prefill: **2746 ms → 233 ms (11.8×)**, closing the gap to llama.cpp `pp5` from ~55× to **~3×**; the NEON `q4k_matmul` at seq=5 beats f32 AMX sgemm while still skipping the dequant. See `bench/baselines/cpu/COMPARISON.md`.

**Cross-arch coverage (2026-05-09)**: Gemma 3, Gemma 4 31B dense, Llama 2 7B, Mistral 7B all dispatch correctly through Metal. Gemma 4 E2B currently falls back to CPU (Per-Layer Embeddings not yet in Metal — ROADMAP D-METAL-PLE). See [crates/larql-compute/docs/architecture-shader-map.md](crates/larql-compute/docs/architecture-shader-map.md) for the per-architecture shader dispatch table.

CPU walk breakdown:

| Component | Time | % of total |
|---|---|---|
| Logits (262K vocab gemv) | 221ms | 41% |
| FFN × 34 layers (walk) | 194ms | 36% |
| Attention × 34 layers | 84ms | 16% |

Walk is **faster than dense** (517ms vs 535ms). GPU Q4K decode is **23× faster** than CPU walk. FFN down projection in walk reads from mmap'd vindex (zero-copy BLAS). Walk only needs ~3.5GB of model weights (attention + embeddings), not 16.6GB. No quantization. See [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md) for architecture and [docs/inference-engine.md](docs/inference-engine.md) for engine details.

### MoE / grid (Gemma 4 26B A4B, M3 Max)

| Topology | tok/s | Notes |
|---|---|---|
| **Local Metal MoE** | **18.9** → **23.6** | 2026-05-04 → re-baselined 2026-07-23 (KV append-in-place + spin-pool landings) |
| Remote-FFN loopback (streaming) | 27.8–28.6 | Re-baselined 2026-07-23 (DEC-0 arm M); f32/f16/i8 wire, ~34% of step in FFN round-trips |
| 1-shard CPU/grid (loopback) | 18.3 | 2026-05-04; NEON Q4_K matvec on shard server, gRPC fan-in |
| 2-shard CPU/grid (loopback) | 17.3 | Parallel collect + parallel fire (`std::thread::scope` + `rayon::par_iter`) |
| `LARQL_SKIP_MOE=1` ceiling | 56.8 | Attention + dense FFN only; theoretical max |

**DEC-0 arm M (2026-07-23, `docs/dec-funnel.md` §3)**: the loopback batch curve is measured — expert-tier step time is sub-linear through batch 32 (×1.6–1.8 on batch dispatch), aggregate tier throughput ~1,050 tok/s at B64 (~25× single-stream), movement ratio 1.2–1.9 × 10⁻³. Run records under `bench/dec0/`; system of record is the `dec0-loopback-mac` experiment in the registry.

**Wire format (2026-05-07)**: grid traffic uses f16 by default (50% bandwidth). Set `LARQL_I8_WIRE=1` for i8 symmetric quantisation (75% bandwidth, opt-in). Both are architecture-agnostic — `hidden_size` is read from vindex config at runtime. Per-layer latency is tracked via `HeartbeatMsg.layer_stats` (EMA + p99); the router uses it to route replicated layers to the lowest-latency server. Use `make bench-wire` to measure codec throughput and `make bench-routing` for routing hot-path.

### Pure MoE (GPT-OSS-20B Q6_K, M3 Max)

`larql run gpt-oss-20b-q4k.vindex --metal` — single stream, greedy, short
context; experts are the lossless MXFP4→Q6_K transcode (6.56 bpw). Metal
produces the **identical greedy trajectory to CPU** — parity re-verified at
every rung of this ladder (2026-08-10, `docs/k3-funnel.md` §4.11):

| ms/token | tok/s | Rung |
|---:|---:|---|
| 97.8 | 10.2 | staged expert copies — top-4 × ~22 MB × 24 layers ≈ 2.1 GB/token of CPU memcpy |
| 25.9 | 38.7 | **zero-copy expert regions** — each layer mmap bound once (`newBufferWithBytesNoCopy`), experts addressed as byte offsets |
| 22.7 | 44.0 | **grouped expert kernels** — all selected experts in one 2-D dispatch (η 0.64→0.90) |
| 22.6 | — | **fused attention** (no-QK-norm + QKV biases in `attn_fused`): attention GPU 8→3.3 ms, wall unmoved — the sync structure was the term |
| 16.7 | **59.8** | **merged command buffers** — GPU MoE combine; a layer's experts + the next layer's attention share one CB (one wait/layer) |
| 14.6 | 68.3 | **GPU-resident routing** (`LARQL_GPU_ROUTE=1`) — router + top-k + expert dispatch as GPU dataflow; **one command buffer per token**, witness counters prove zero route-dependent host work |
| 13.4 | 74.6 | **native MXFP4 expert banks** (`--routed-from` a VINDEX3 container) — 4.25 bpw experts replace the 6.56 bpw Q6_K transcode; expert reads drop 1 959 → 1 269 MB/token |
| 12.95 | **77.2** | **vectorised MXFP4 kernel** (`uint4` group loads; arm `a2`, the default) — the three rows above are one paired same-session A/B/A ladder (2026-08-14) |

CPU decode on the same vindex: ~60 ms/token (15-17 tok/s). Comparison
framing: native-MXFP4 engines (oMLX, ~4.25 bpw experts) report ~83-90 tok/s
on this class of machine at 1k-4k context. At the time of this ladder
(2026-08-14) the comparison was **77.2 vs ~83** — a ~0.9 ms/token gap,
fully attributed (~1.09 ms of lm_head Q4_K matvec at the bandwidth
ceiling, ~0.5 ms of command-buffer boundary). The VINDEX3 decode path has
since closed and passed it — see the table below. Kernel-level
attribution for the MXFP4 rungs: measured grouped-kernel bandwidth
predicts the expert-read delta within 0.1 ms of the e2e number, so
nothing is hiding in integration.

### VINDEX3 decode (stage-profiled, M3 Max, 2026-08-20)

`larql vindex3 exec --generate N` through the Metal lowering, with the
stage-level GPU ledger (`larql-compute-metal` `lowering/profile.rs`,
commit `d053ac52`). Same greedy ids on every arm; levers priced per stage
against the byte floor (bytes / 367 GB/s):

| Model | Before (tok/s) | After (tok/s) | Levers |
|---|---:|---:|---|
| gpt-oss-20b | 91 | **106** | encode-ahead + NVFP4 QKV/gate-up fusion |
| Granite 4.1 3B | 97 | **114** | same |
| Gemma 4 26B-A4B | 62 | **74** | same + `rms_norm_multi3` |
| Glimmer 30B | 17.8 | **20.1** | same |

V3 meets or beats V2 on every model where both exist (gpt-oss 91-106 vs
83-93; Gemma 4 26B-A4B 62-74 vs ~19-23.6 on the older MoE path above).
The residual over the byte floor is roughly constant across
representations — per-kernel fixed cost, not bytes — which is what the
per-stage ledger exists to attribute.

### Load-bearing environment flags (serving & measurement)

Every env var is `LARQL_`-prefixed (canonical since 2026-07-22 — the historical
unprefixed `SKIP_MOE` / `SKIP_OUTER_NORM` / `DECODE_DEBUG` still work as loud
deprecated aliases). `larql-server` logs the kernel class **and** the decode
flag state at startup, so no benchmark number is recorded against unlogged
flag state. The ones that change what a measured number means:

| Flag | Default | Effect |
|---|---|---|
| `LARQL_SKIP_MOE=1` | off | Bypass MoE entirely (the ceiling-arm flag above) — local AND grid paths |
| `LARQL_GPU_ROUTE=1` | off | GPU-resident MoE routing (one CB/token). Off, decode falls back to the legacy CPU-routed path — a healthy-looking run at ~2/3 the speed, so any recorded MoE number must state this flag |
| `LARQL_MXFP4_ARM=<a\|a2\|b\|c\|d>` | `a2` | MXFP4 grouped-expert kernel arm. `a2` (vectorised split, exact) is the default and demotes itself to `a` per layer bank when payload offsets are not 16-byte aligned — loudly, never silently |
| `LARQL_Q4K_DIRECT_ATTN` / `_ATTN_INT8` / `_LM_HEAD` / `_DIRECT_FFN` / `_Q4K_ASM` / `_SPIN_POOL` | **on** (`=0` opts out) | CPU decode fast-path stages; A/B against the f32 path |
| `LARQL_COMPUTE_CONCURRENCY=N` | auto (physical cores) | Expert-server batch compute parallelism (`layer_batch`) — load-bearing for tier throughput |
| `LARQL_DISABLE_Q8K_WIRE=1` | off | Fall back from the Q8K predispatch wire to f32/f16 |
| `LARQL_I8_WIRE=1` | off | i8 wire arm (also required server-side for dec-bench's i8 sweep arm) |
| `LARQL_MOE_TOP_K=N` | arch value | Override MoE top-k on the grid — changes routing, not just speed |
| `LARQL_SKIP_OUTER_NORM=1` | off | Debug-only outer-norm bypass — corrupts output; never in a recorded run |

Diagnostics (`LARQL_*_TIMING`, `LARQL_MOE_DEBUG`, `LARQL_DECODE_DEBUG`, …) print
to stderr and do not change results. The full inventory lives in
`docs/audits/dec-readiness-review-2026-07-22.md` §5.

### Dense remote-FFN (Gemma 4 31B Q4K, M3 Max, localhost)

| Topology | tok/s | Notes |
|---|---|---|
| **Remote-FFN batch, Metal GPU server** | **6.5** | `larql bench --ffn URL --ffn-dispatch batch`; `--features metal-experts` on server. 153ms/tok: 92ms attn local + 60ms FFN remote. |
| Remote-FFN batch, CPU server | 1.6 | Same path, server uses CPU NEON instead of Metal. |
| Remote-FFN streaming (60 sequential HTTP) | 0.6 | Q8K wire format via `/v1/walk-ffn-q8k`, NEON down projection. |
| Local Metal | blocked | Heterogeneous attention (L5/L11/…/L59 head_dim=512 vs sliding head_dim=256) — A1-A3 roadmap. Est. ~12-15 tok/s after fix. |

**Metal GPU FFN server** (`larql serve --ffn-only --features metal-experts`): pre-loads Q4K weight bytes into Metal buffers at startup via zero-copy mmap; dispatches `q4k_ffn_gate_up_8sg` + `geglu_gelu_tanh` + `q4k_matvec` per Q8K batch request — same shaders as local decode. **Build separation required**: `larql-cli` must be built WITHOUT `--features metal-experts` (adding it causes a 10.7 vs 18.9 tok/s regression on Gemma 4 26B-A4B due to Metal pipeline init overhead in the standard decode path). Only the server binary uses that flag.

The grid path is the load-bearing primitive for the **"split large models in grids"** axis — Kimi K2.6 / DeepSeek V4-class models (1T params, ~600 GB Q4_K) only fit on a multi-shard deployment. See [`crates/larql-server/ROADMAP.md` §G-SCALE](crates/larql-server/ROADMAP.md) for the path forward.

## Residual Stream Trace

Capture the complete record of inference — every layer, every contribution, queryable.

```sql
-- LQL: answer trajectory through all layers
larql> TRACE "The capital of France is" FOR "Paris";
  Layer   Rank     Prob      Attn       FFN      Who
    L22     50    0.002     +22.2     +34.4   BOTH ↑
    L23     10    0.024     -16.9     +55.9    FFN ↑
    L24      1    0.714    +105.7     +24.4   BOTH ↑  ← phase transition
    L25      1    0.997      +4.3     +94.4    FFN ↑
    L26      1    0.999     +83.1     +18.7   BOTH ↑

-- Attn vs FFN decomposition at the phase transition
larql> TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Persist for later analysis
larql> TRACE "The capital of France is" SAVE "france.trace";
```

```python
# Python: same trace, programmatic access
import larql

wm = larql.WalkModel("gemma3-4b.vindex")
t = wm.trace("The capital of France is")
t.answer_trajectory("Paris")   # rank, prob, attn/ffn logits per layer
t.top_k(24)                    # [('Paris', 0.714), ...]
t.save("trace.bin")            # mmap'd store
```

### Tiered Context (infinite context without KV cache)

| Storage | Per window | 370K tokens | vs KV cache |
|---|---|---|---|
| Boundary residual | 10 KB | 18.9 MB | 3,100x |
| Tier 4 int8 (bit-perfect) | 58 KB | 110 MB | 511x |
| KV cache | ~30 MB | 56,000 MB | 1x |

```python
from larql._native import BoundaryWriter, BoundaryStore

# Write boundary residuals — one per 200-token window
writer = BoundaryWriter("context.bndx", hidden_size=2560, window_size=200)
writer.append(token_offset=0, window_tokens=200, residual=boundary_vec)
writer.finish()

# Mmap'd read — OS pages on demand, RSS ≈ one boundary
store = BoundaryStore("context.bndx")
store.residual(42)  # zero-copy from mmap
```

See [docs/residual-trace.md](docs/residual-trace.md) for the full writeup.

## Mechanistic interpretability surface

LARQL exposes a programmatic forward-hook system for capture, ablation,
steering, activation patching, logit lens, and KV-cache surgery — the
primitives lazarus-style MCP servers (e.g. `chuk-mcp-lazarus`) build on
top of. All of it works on real models and on synthetic weights, with
zero overhead when no hook is registered.

```rust
use larql_inference::forward::{
    RecordHook, SteerHook, ZeroAblateHook, trace_forward_full_hooked,
    capture_donor_state, patch_and_trace, logit_lens_topk, embedding_neighbors,
};

// 1. Capture residuals at chosen layers (read-only).
let mut record = RecordHook::for_layers([12, 18, 24]);
trace_forward_full_hooked(&weights, &tokens, &[12, 18, 24],
    /*activations=*/ false, 0, /*attention=*/ false, &ffn, &mut record);
let residual_at_18 = record.post_layer.get(&18).unwrap();

// 2. Logit lens at any layer — top-k, single-token tracking, full race.
let top_k     = logit_lens_topk(&weights, residual_at_18.row(0).as_slice().unwrap(), 5);
let neighbors = embedding_neighbors(&weights, &query_vec, 10);

// 3. Ablate or steer mid-forward.
let mut ablate = ZeroAblateHook::for_layers([14usize]);
let mut steer  = SteerHook::new().add(20, steer_vec, 0.5);

// 4. Activation patching — donor → recipient at chosen (layer, position) coords.
let donor   = capture_donor_state(&weights, &donor_tokens, &[(10, 4)]);
let patched = patch_and_trace(&weights, &recipient_tokens, &donor, &[28]);
```

From Python via `larql._native.WalkModel`:
`capture_residuals`, `forward_with_capture`, `forward_ablate`,
`forward_steer`, `patch_activations`, `logit_lens`, `track_token_at`,
`track_race`, `embedding_neighbors`, `project_through_unembed`,
`embedding_for`, `unembedding_for`, `generate_with_hooks`. Returned
tensors are numpy arrays.

**Backend split.** Hooks during single-forward (`trace_forward_full_hooked`,
all the capture/ablate/steer/patch primitives above) are zero-cost when
no hook is registered and run on the existing CPU forward path. Hooks
during **multi-token generation** (`generate_cached_hooked` /
`WalkModel.generate_with_hooks`) also use the CPU KV-cache path — the
Metal-fast `predict` is hook-free by design (kernels are fused; threading
hooks through would split the fast path even when unused). Mech-interp
tools want correctness over throughput, so the CPU-when-hooks-active
trade is the right one.

End-to-end walkthrough on synthetic weights (no vindex required):

```bash
cargo run --release -p larql-demos --example mech_interp_demo
```

The full surface is documented in [crates/larql-inference/ROADMAP.md](crates/larql-inference/ROADMAP.md) §
"P0: Mechanistic hooks (lazarus parity)".

## Documentation

| Doc | Description |
|---|---|
| [crates/larql-lql/docs/spec.md](crates/larql-lql/docs/spec.md) | LQL language specification (v0.4) |
| [crates/larql-vindex/docs/format-spec.md](crates/larql-vindex/docs/format-spec.md) | Vindex file format specification (v0.4, ~98% implemented) |
| [vindex3.org](https://vindex3.org) | VINDEX3 explained — the specification as chapters, an explorer, and the Record of what is demonstrated; [vindex3.org/3.0](https://vindex3.org/3.0) is the citable version page |
| [crates/larql-vindex/docs/vindex3-format-spec.md](crates/larql-vindex/docs/vindex3-format-spec.md) | Vindex3 container ABI (`larql-vindex` side — bytes, sections, admission) |
| [docs/vindex3-format.md](docs/vindex3-format.md) | Vindex3 model-system container spec — the actively updated spec (plan/encode/verify semantics); the ABI doc above governs the on-disk bytes |
| [docs/vindex3-runtime.md](docs/vindex3-runtime.md) | Vindex3 runtime stack — `Vindex3Runtime`, `LogitsSession`, the KV seam, and V3 serving over `/v1/completions` |
| [crates/larql-vindex/docs/operations-spec.md](crates/larql-vindex/docs/operations-spec.md) | Vindex operations, API, patches (~98% implemented) |
| [crates/larql-vindex/docs/ecosystem-spec.md](crates/larql-vindex/docs/ecosystem-spec.md) | Distributed hosting, HuggingFace, Vindexfile (~85% implemented) |
| [crates/larql-vindex-spec/SPEC.md](crates/larql-vindex-spec/SPEC.md) | Vindex v1 public contract — manifest schema, sharding rule, validation thresholds, model card tags |
| [crates/larql-vindex-spec/schema/vindex-v1.schema.json](crates/larql-vindex-spec/schema/vindex-v1.schema.json) | JSON Schema 2020-12 mirror of the v1 manifest |
| [docs/vindex-factory.md](docs/vindex-factory.md) | Vindex Factory — recipe-driven, verified, remote-executed vindex builds; `larql-factory` crate (recipe schema, `build_id`, validator, capability manifest, card generator, PREFLIGHT→RELEASE build driver) |
| [docs/model-publishing.md](docs/model-publishing.md) | Republishing models — why the 2026-08 recovery could not use the Vindex Factory, what that cost, the affected-vindex audit rule, and the recipes that need to exist so the next one is `larql recipe build` |
| [crates/larql-factory/README.md](crates/larql-factory/README.md) | `larql-factory` crate reference — module layout, CLI usage, what's not built yet |
| [docs/lql-guide.md](docs/lql-guide.md) | LQL quick start guide |
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/inference-engine.md](docs/inference-engine.md) | Inference engine — BLAS-fused attention, Metal GPU, auto-calibration |
| [crates/larql-kv/README.md](crates/larql-kv/README.md) | **KV engines** — 10 pluggable implementations, state-policy classified, W10 mask cascade |
| [crates/larql-kv/docs/state-policy.md](crates/larql-kv/docs/state-policy.md) | **State Policy** — `(canonical_state, derivative_state, correctness_contract)` framing; why the K/V slot choice predicts perf |
| [crates/larql-kv/PERFORMANCE.md](crates/larql-kv/PERFORMANCE.md) | KV engine bench protocol, W10 default-on result (2026-05-21), per-engine perf decomposition |
| [crates/larql-inference/docs/specs/kv-engine-unification.md](crates/larql-inference/docs/specs/kv-engine-unification.md) | KV engine unification — single `KvEngine` trait dispatch through `larql run` / `walk` / `bench` |
| [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md) | FFN graph layer — mmap walk faster than dense (517ms vs 535ms), all 34 layers |
| [docs/walk-boundary-sweep.md](docs/walk-boundary-sweep.md) | Walk boundary sweep — correctness proof across all layer boundaries |
| [docs/residual-trace.md](docs/residual-trace.md) | Residual stream trace — decomposition, storage, tiered context |
| [docs/mech-interp.md](docs/mech-interp.md) | Mechanistic interp surface — hooks, lens, vocab proj, patching, KV surgery (Rust + Python) |
| [crates/larql-inference/docs/trace-format.md](crates/larql-inference/docs/trace-format.md) | Trace file format specification (.bin, .bndx, .ctxt) |
| [docs/adr/0009-wire-format-evolution.md](docs/adr/0009-wire-format-evolution.md) | Wire format: f16 default, i8 opt-in, Accept/Content-Type negotiation |
| [docs/adr/0010-quic-grid-transport.md](docs/adr/0010-quic-grid-transport.md) | QUIC transport for grid (planned) |
| [docs/adr/0011-grid-self-balancing.md](docs/adr/0011-grid-self-balancing.md) | Grid Mode B + dynamic rebalancing (planned) |
| [docs/adr/0012-grid-benchmarking.md](docs/adr/0012-grid-benchmarking.md) | Grid benchmarking infrastructure — criterion + CLI + CI gate |
| [docs/diagnoses/shannon-cross-engine-divergence.md](docs/diagnoses/shannon-cross-engine-divergence.md) | Forward-pass correctness diagnostic via `larql shannon verify` — three-engine bits/char comparison against HF/PyTorch and MLX, plus the three bugs it surfaced |
| [scripts/README_shannon_score.md](scripts/README_shannon_score.md) | Cross-engine Shannon scorers — `larql shannon verify` + standalone scripts for MLX and HF |
| [docs/audits/dec-readiness-review-2026-07-22.md](docs/audits/dec-readiness-review-2026-07-22.md) | DEC-readiness review of the server/router/remote-FFN data plane ahead of the DEC funnel programme — silent-corruption + security findings, remediation tracked in ROADMAP |
| [docs/dec-funnel.md](docs/dec-funnel.md) | **DEC funnel** — decoupled attention/weights serving at batch and frontier scale; the experiment ladder, the claims table, and standing rules R0–R10 (numbers that don't travel between conventions) |
| [docs/k3-funnel.md](docs/k3-funnel.md) | **K3 adapter ladder** — GPT-OSS-20B → Kimi Linear → K3, the three rungs that get a 2.8T hybrid-linear MoE onto the engine |
| [docs/quant-obs.md](docs/quant-obs.md) | **Quant-Obs** — observer-metric ladder for quantisation sensitivity |
| [docs/vindex3-experiments.md](docs/vindex3-experiments.md) | Pre-registered VINDEX3 experimental programme (the V2-0..V2-4 gates) |
| [docs/lyrw-v2.md](docs/lyrw-v2.md) | LYRW v2 — the K3 routed-layer physical-layout gate |
| [docs/kv-residency-contract.md](docs/kv-residency-contract.md) | The KV residency contract — window vs storage vs residency, disentangled |
| [docs/kv-attention-scaling.md](docs/kv-attention-scaling.md) | KV attention scaling — measurement schema + run-hygiene rules |
| [docs/metal-kernel-capabilities.md](docs/metal-kernel-capabilities.md) | Metal kernel capability table (Phase B ground-truth audit) |
| [docs/ffn/README.md](docs/ffn/README.md) | FFN backend family — WeightFfn, SparseFfn, WalkFfn, distributed sharding |
| [docs/ffn-cache.md](docs/ffn-cache.md) | FFN activation cache — skip recomputation of repeated feature sets |
| [docs/multi-modal.md](docs/multi-modal.md) | Multi-modal support — Phase 0–2 shipped, phases 3–6 design-only |
| [docs/virtual-experts-dispatch.md](docs/virtual-experts-dispatch.md) | Virtual experts — bounded routing into typed, sandboxed WASM compute units |
| [docs/fleet-routing-extensions.md](docs/fleet-routing-extensions.md) | Fleet routing extensions FR1–FR4 — spec + frozen pre-registrations |
| [docs/authority-control-plane.md](docs/authority-control-plane.md) | Authority control plane (EXP-26..38) — layer-mechanism branch closed |
| [docs/dec-funnel-v0.4.md](docs/dec-funnel-v0.4.md) | DEC funnel v0.4.1 — superseded by docs/dec-funnel.md |
| [docs/dec-funnel-v0.2.md](docs/dec-funnel-v0.2.md) | DEC funnel v0.2 — archived; control plane and gates inherited by reference |
| [docs/training-free-insert.md](docs/training-free-insert.md) | Training-free knowledge insertion — residual capture + feature writes |
| [docs/weight-extraction.md](docs/weight-extraction.md) | Weight extraction pipeline — model weights → vindex, no bulk forward passes |
| [docs/circuit-types.md](docs/circuit-types.md) | Circuit type analysis — gate/down cosine classifies feature roles |
| [docs/confidence.md](docs/confidence.md) | Confidence scoring for query results |
| [docs/fhg.md](docs/fhg.md) | FHG — Fourier heuristic graph programme (behavioural, model-agnostic) |
| [docs/knowledge-pipeline.md](docs/knowledge-pipeline.md) | Stub — placeholder for the knowledge pipeline spec |

The full index, including docs not tabled here, is
[docs/README.md](docs/README.md).

## Platform Support

| Platform | Compiles | GPU | BLAS |
|----------|----------|-----|------|
| macOS arm64 (M-series) | ✓ | Metal (`--features gpu`) | Accelerate |
| Linux arm64 / x86_64 | ✓ | — (CPU fallback) | OpenBLAS |
| Windows arm64 / x86_64 | ✓ | — (CPU fallback) | OpenBLAS |

macOS gets Metal GPU acceleration. Linux and Windows run the same CPU path (BLAS-fused attention + mmap walk FFN). All platforms require OpenBLAS on Linux/Windows — install via your system package manager (`apt install libopenblas-dev`, `vcpkg install openblas`).

## Install

Tagged releases carry prebuilt `larql` + `larql-server` binaries for the three
platforms CI validates ([ADR-0026](docs/adr/0026-tagged-release-binaries.md)).
Each archive holds both binaries under a triple-named directory.

| Platform | Asset |
|---|---|
| macOS arm64 | `larql-aarch64-apple-darwin.tar.gz` |
| Linux x86_64 | `larql-x86_64-unknown-linux-gnu.tar.gz` |
| Windows x86_64 | `larql-x86_64-pc-windows-msvc.zip` |

```bash
V=v0.2.0   # or use /releases/latest/download/… to track the newest
curl -fsSL "https://github.com/chrishayuk/larql/releases/download/$V/larql-aarch64-apple-darwin.tar.gz" | tar xz
./larql-aarch64-apple-darwin/larql --version
```

These are build artifacts versioned by tag — **not** a crates.io release and no
API-stability commitment. The crates.io entries under the `larql*` names are
`0.0.0` placeholders held to prevent squatting; `cargo install larql` will not
give you a working binary.

Prefer these over a source build on any rented or ephemeral host: a cold
`cargo build --release` costs 20–40 minutes, and on a GPU-provisioned box that
is pure CPU work with the GPU idle. The DEC stage drivers enforce this via
`scripts/lib/larql-binaries.sh`.

## Building & Testing

```bash
cargo build --release                    # optimised build
cargo build --release --features gpu     # with GPU backend (Metal on macOS today; Vulkan/CUDA later)
cargo test                               # all tests across all crates
.venv/bin/python scripts/diagnose_models.py    # cross-engine correctness sweep — see below
cargo test -p larql-inference            # inference engine tests (109 tests)
cargo test -p larql-inference --features gpu    # + GPU tests (115 tests)
cargo test -p larql-lql                  # LQL parser + executor tests (272 tests)
cargo test -p larql-vindex               # vindex storage + patch tests (525 tests as of 2026-05-08)

# Crate-local CI shortcuts
make larql-vindex-ci                     # fmt, clippy, tests, examples, benches, coverage policy
make larql-vindex-test                   # cargo test -p larql-vindex
make larql-vindex-fmt-check              # cargo fmt -p larql-vindex -- --check
make larql-vindex-lint                   # cargo clippy -p larql-vindex --all-targets -- -D warnings
make larql-vindex-examples               # cargo check -p larql-vindex --examples
make larql-vindex-bench-test             # cargo test -p larql-vindex --benches
make larql-vindex-coverage-summary       # aggregate + per-file coverage ratchet
make larql-vindex-coverage-html          # HTML report plus the same policy gate

# Inference engine examples (demos live in `larql-demos`, benches in `larql-inference`)
cargo run --release -p larql-demos --example attention_demo        # fused attention demo
cargo run --release -p larql-demos --example mech_interp_demo      # capture / lens / ablate / steer / patch (synthetic — no vindex)
cargo run --release -p larql-inference --example bench_attention   # attention benchmarks
cargo run --release -p larql-demos --example backend_demo --features gpu       # backend demo
cargo run --release -p larql-inference --example bench_backend --features gpu  # backend benchmarks
cargo run --release -p larql-inference --example bench_inference   # full inference benchmarks

# Vindex tools (build once, enables mmap walk)
cargo run --release -p larql-vindex --example build_convert_gates_f32 -- path/to/vindex   # f16→f32 gate vectors
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex  # feature-major down vectors
cargo run --release -p larql-vindex --example build_up_features -- path/to/vindex    # feature-major up vectors

# Server (walk inference over HTTP)
cargo run --release -p larql-server -- path/to/vindex --port 8080
cargo run -p larql-demos --example server_demo              # synthetic HTTP surface demo
cargo run -p larql-demos --example embed_demo               # synthetic embed/logits/token demo
cargo run --release -p larql-server --example server_bench  # synthetic server operation benchmark
cargo run --release -p larql-server --example bench_embed_server -- path/to/vindex
cargo test -p larql-router                                  # static router + grid route-table checks

# Vindex and LQL demos (synthetic — run in CI; all demos live in `larql-demos`)
cargo run -p larql-demos --example demo_features                     # vindex feature showcase
cargo run --release -p larql-demos --example mmap_demo               # mmap RAM behaviour + scaling table
cargo run --release -p larql-demos --example q4k_demo                # streaming Q4_K: size ratio, manifests, dequant round-trip
cargo run --release -p larql-demos --example demo_memit_solve        # MEMIT decomposition + MemitStore round-trip
cargo run -p larql-demos --example parser_demo                       # parser demo (24/24 statements)
cargo run -p larql-demos --example lql_demo                          # LQL spec compliance (61/61)
cargo run --release -p larql-demos --example compact_demo            # LSM storage tier walkthrough

# Model-dependent demos (require real vindex, skip gracefully otherwise)
cargo run --release -p larql-demos --example compile_demo            # end-to-end COMPILE INTO VINDEX on real Gemma 4B
cargo run --release -p larql-demos --example refine_demo             # 10-fact INSERT + COMPILE (exp 14 reproduction, 10/10 retrieval)
cargo run --release -p larql-demos --example trace_demo              # TRACE residual decomposition on real Gemma 4B

# Criterion benches (use --quick for a fast sweep, omit for full sample sizes)
cargo bench -p larql-lql    --bench parser               # parse_single × 18 + parse_batch
cargo bench -p larql-lql    --bench executor             # SELECT, SHOW, DELETE, UPDATE, patch lifecycle
cargo bench -p larql-lql    --bench compile              # COMPILE INTO VINDEX bake cost
cargo bench -p larql-vindex --bench vindex_ops           # KNN, walk, save/load, mutate, MoE
make larql-vindex-bench                                  # shortcut for vindex_ops
cargo bench -p larql-vindex --bench vindex_scaling       # production-dim KNN (Gemma/Llama/Mixtral)
cargo bench -p larql-vindex --bench memit_solve          # ridge decomposition throughput
cargo bench -p larql-vindex --bench extract_throughput   # streaming extract: f32 vs Q4K write-path
cargo bench -p larql-vindex --bench q4k_vs_f32           # per-layer attn retrieval: f32 memcpy vs Q4K dequant
cargo bench -p larql-compute --bench linalg              # CPU/Metal linalg (matmul) backends
cargo bench -p larql-inference --bench wire_codec        # f32/f16/i8 encode+decode throughput (MB/s)
cargo bench -p larql-router --bench routing              # route/heartbeat/rebuild hot-path (ns/op)
make bench-all                                           # all of the above in one shot
```

The `compile_demo` example proves the full flow on a real Gemma 4B
vindex: `INSERT Atlantis → Poseidon`, `COMPILE CURRENT INTO VINDEX`,
then `USE` the compiled vindex in a fresh session and verify
`INFER "The capital of Atlantis is" → Pose 56.91%` and
`INFER "The capital of France is" → Paris 67.34%` (neighbour
preserved). The constellation is baked into `down_weights.bin`
column-wise — no overlay or sidecar needed at load time.

Bench HTML reports go to `target/criterion/`. The `parser` bench
parses 100 mixed statements in ~78 µs (1.28 M stmts/s); `vindex_ops`
runs production-sized Gemma 4B gate KNN in ~2.78 ms/layer; `compile`
runs `COMPILE INTO VINDEX` in ~1.84 ms (no patches) to 2.41 ms (with
`down_weights.bin`).

### Cross-engine correctness check

`larql shannon verify` runs the LARQL Rust forward pass alongside HF/PyTorch
and MLX reference scorers on the same corpus and prints a bits/char delta
table — the strongest unit-of-observable check that LARQL's forward path
matches the canonical references end-to-end.

```bash
# Single model.
larql shannon verify google/gemma-3-4b-it \
    --corpus data/gutenberg/frankenstein.txt \
    --bytes 1024 \
    --threshold 0.5

# All supported architectures (SmolLM2, Llama 3.2, Mistral 7B, Gemma 3 4B)
# + the Q4K Metal vindex path for models with a local vindex.
.venv/bin/python scripts/diagnose_models.py
```

PyTorch and `mlx_lm` are required in `.venv` for the reference scorers
(see [`scripts/README_shannon_score.md`](scripts/README_shannon_score.md)).

When the verifier reports a real divergence, the bisection methodology
and the env-var diagnostic instruments are documented in
[`docs/diagnoses/shannon-cross-engine-divergence.md`](docs/diagnoses/shannon-cross-engine-divergence.md).
The 2026-05-15 sweep identified — and the loader fix in `larql-models`
landed by 2026-05-16 closed — three config-loading bugs (unparsed
`rms_norm_eps`, missing per-layer-type `rope_scaling` for Gemma 3,
missing `llama3` rope_scaling for Llama 3.x). Post-fix, all four
reference architectures match HF F32 to <0.01% bits/char with no env
vars set.

The CI workflow at
[`.github/workflows/shannon-verify.yml`](.github/workflows/shannon-verify.yml)
runs `larql shannon verify` against HF/PyTorch on SmolLM2-135M for every
PR + push to main. Any future regression in the Rust forward path that
drifts past 0.5% bits/char trips the gate before merge.

## License

Apache-2.0
