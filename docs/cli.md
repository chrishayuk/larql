# LARQL CLI Reference

```
larql <COMMAND> [OPTIONS]
```

## Primary commands

Ollama-style day-to-day verbs. Models can be referenced by cache
shorthand (`gemma-3-4b-it-vindex`), `owner/name`, `hf://owner/name`, or
a local directory path — see [Model resolution](#model-resolution) below.

| Command | Description |
|---|---|
| `run <model> [prompt]` | Run inference. One-shot if prompt given; chat loop if not. |
| `chat <model>` | Alias for `run <model>` with no prompt. |
| `pull <model>` | Download a vindex from HuggingFace and cache locally. |
| `model <subcmd>` | Manage HuggingFace **model** repos (companion to `pull`, which is vindex-only). |
| `link <path>` | Register a local vindex directory with the cache so `run` / `list` / `show` find it by shorthand. |
| `list` | Show cached vindexes (model, size, layers, hidden). |
| `show <model>` | Vindex metadata and file inventory. |
| `slice <source>` | Carve a subset of a vindex (`client` / `attn` / `embed` / `server` / `browse` / `router` / `expert-server`). |
| `publish <source>` | Publish a vindex to HuggingFace — full + slice siblings + collections. |
| `rm <model>` | Evict a cached vindex. |
| `bench <model>` | Benchmark decode throughput on a real vindex (Metal / CPU / Ollama). |
| `shannon <subcmd>` | Next-token bit scoring, slot probes, repetition probes, layer lens, demo arithmetic coding. |
| `serve <model>` | Serve a vindex over HTTP + gRPC. |

## Build / extract

| Command | Description |
|---|---|
| `extract <model-id>` | Build a .vindex from a HuggingFace model (safetensors/GGUF/MLX → queryable). |
| `extract-index` | Backwards-compat alias of `extract`. |
| `build` | Build a custom vindex from a Vindexfile (FROM + PATCH + INSERT). |
| `compile` | Compile vindex patches into model weights (AOT). |
| `convert` | Convert between formats (GGUF ↔ vindex, safetensors → vindex; `quantize` for FP4/Q4_K_M). |
| `hf` | HuggingFace Hub: download / publish a vindex. |
| `verify` | Verify vindex file integrity (SHA256 checksums). |
| `diag` | Engine diagnostic — print which kernel paths fire for a vindex, validate Q4_K/Q6_K strides, optional `--probe` runs a real forward pass. |
| `parity` | Cross-backend numerical diff (`reference` / `cpu` / `metal`) at well-known checkpoints. |

## LQL

| Command | Description |
|---|---|
| `repl` | Launch the LQL interactive REPL. |
| `lql '<stmt>'` | Execute a one-shot LQL statement. |

## Research / interpretability tools — `larql dev <subcmd>`

All extraction / probing / benchmark tooling lives under `larql dev`.
The pre-redesign top-level invocations (`larql walk …`,
`larql weight-extract …`, etc.) are rewritten to `larql dev <name>`
transparently by an argv trampoline, so existing scripts continue to
work.

```
larql dev --help
larql dev walk --index X.vindex --prompt "..." --predict
```

See [Research commands (dev)](#research-commands-dev) below for the full
list.

## Model resolution

The `<model>` argument on `run`, `chat`, `show`, `rm`, and `pull`
resolves in this order:

1. **`hf://owner/name[@rev]`** — download (if not cached) via HF hub API,
   return the cache path.
2. **Existing local directory** — use as-is.
3. **`owner/name`** — cache lookup first; fall back to HF download.
4. **Plain name** — search the cache for a unique
   `datasets--<*>--<name>` entry. Ambiguous shorthands error out and
   list candidates.

`rm` never downloads — it only resolves against the cache.

### `larql run`

One-shot inference or interactive chat.

```
larql run <MODEL> [PROMPT] [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Vindex dir, `hf://owner/name`, `owner/name`, or cache shorthand | — |
| `[PROMPT]` | Prompt text; omit to enter chat mode | — |
| `-n, --max-tokens <N>` | Max tokens to generate autoregressively | 64 |
| `--top <N>` | Show the top-K prediction table per step instead of just the argmax (implied by `--verbose`) | 1 |
| `--kv-cache <KIND>` | KV cache strategy: `standard` (full FP32, unbounded), `markov-bounded` (sliding window), `none` (no cache, O(N²)) | standard |
| `--context-window <N>` | Sliding-window size when `--kv-cache markov-bounded`; `0` = unbounded | 0 |
| `--metal` | Force the Metal GPU path (macOS only). Default is CPU on non-macOS, Metal on macOS when available | auto |
| `--ffn <URL>` | Route dense FFN to a remote larql-server. Attention runs locally; each layer's FFN is a round trip | — |
| `--ffn-timeout-secs <N>` | HTTP timeout for `--ffn` | 60 |
| `--ffn-dispatch <streaming\|batch>` | Dense FFN dispatch when `--ffn` is set. `streaming` runs N sequential round-trips per token; `batch` predispatches all layers in parallel and refines in a second Metal pass | streaming |
| `--ffn-predispatch-iters <N>` | Refinement iterations for `--ffn-dispatch batch`. Higher = more accurate, slower | 1 |
| `--moe-shards <SPEC>` | MoE expert dispatch: `"0-63=URL1,64-127=URL2"`. Client runs the router locally; expert calls fan out | — |
| `--moe-units-manifest <PATH>` | Fine-grained per-(layer,expert) shard map from JSON. Mutually exclusive with `--moe-shards` | — |
| `--moe-dispatch <streaming\|batch>` | Same shape as `--ffn-dispatch`, applied to MoE expert calls | streaming |
| `--moe-predispatch-iters <N>` | Refinement iterations for `--moe-dispatch batch` | 1 |
| `--experts` | Enable WASM-expert dispatch (gcd, base64, …) at FFN slots | false |
| `--experts-dir <PATH>` | Directory of `.wasm` experts (overrides default lookup) | — |
| `--ops <NAME[,…]>` | Restrict `--experts` to a comma-separated subset of op names | all |
| `--constrained` | Run constrained-decoding mode (require all generated tokens to satisfy declared ops) | false |
| `-v, --verbose` | Verbose load / timing output | false |

Examples:

```bash
larql run gemma-3-4b-it-vindex "The capital of France is"
larql run chrishayuk/gemma-3-4b-it-vindex           # chat mode
larql run hf://chrishayuk/gemma-3-4b-it-vindex      # explicit HF
larql run gemma4-31b.vindex --ffn http://server:8080 "…"
```

### `larql chat`

Interactive chat. Alias for `run <model>` with no prompt.

```
larql chat <MODEL> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Vindex dir, `hf://owner/name`, or cache shorthand | — |
| `-n, --max-tokens <N>` | Max tokens per chat response | 64 |
| `--ffn <URL>` | Route FFN to a remote larql-server | — |
| `--ffn-timeout-secs <N>` | HTTP timeout for `--ffn` | 60 |
| `-v, --verbose` | Verbose load / timing output | false |

`larql chat` is a thin shim — under the hood it constructs the same `RunArgs`
as `larql run` (with `prompt: None`) and the chat-loop UX is implemented in
`run_cmd::run`. The richer flag surface above (sampling, KV cache, MoE) is
available via `larql run <model>` with no prompt.

### `larql bench`

Benchmark decode throughput on a real vindex. Reports tok/s for each backend
(default `metal`) and optionally a side-by-side against a local Ollama server
or a remote larql-server FFN grid.

```
larql bench <MODEL> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Vindex dir, `hf://owner/name`, or cache shorthand | — |
| `--prompt <TEXT>` | Prompt to time (kept short to keep prefill consistent across runs) | "The capital of France is" |
| `-n, --tokens <N>` | Number of decode steps to measure | 50 |
| `--warmup <N>` | Discarded warmup steps before measurement (smooths first-call allocation effects) | 3 |
| `--backends <LIST>` | Comma-separated backend list. Supported: `metal`, `cpu` | metal |
| `--ollama <MODEL>` | Also query a local Ollama server with this model name (e.g. `gemma3:4b`). Requires `ollama serve` running | — |
| `--engine <LIST>` | Comma-separated KV engines to bench alongside the GPU path. Supported: `markov-rs`, `unlimited-context` | — |
| `--ffn <URL>` | Bench the grid path: route FFN to a remote larql-server | — |
| `--ffn-dispatch <streaming\|batch>` | Same shape as `larql run --ffn-dispatch` | streaming |
| `--moe-shards <SPEC>` | Bench the remote MoE expert path | — |
| `--moe-dispatch <streaming\|batch>` | Same shape as `larql run --moe-dispatch` | streaming |
| `--moe-predispatch-iters <N>` | Refinement iterations for batch dispatch | 2 |
| `--profile` | Print per-stage timing breakdown for each engine (markov-rs only for now) | false |
| `--wire <f32,f16,i8>` | Comma-separated wire formats to compare end-to-end. Requires `--ffn` | — |
| `--bench-grid` | Shard-count scaling sweep: with `--moe-shards`, reruns with 1..N shards from the provided map | false |
| `--concurrent <N>` | Simulate N concurrent clients; reports aggregate tok/s and per-client p99 | 1 |
| `--output <json>` | Emit machine-readable JSON alongside the table output | — |
| `--output-file <PATH>` | Write JSON output to this file instead of stdout | — |

Examples:

```bash
larql bench gemma3-4b-it-vindex --backends metal,cpu
larql bench gemma3-4b-it-vindex --ollama gemma3:4b
larql bench gemma4-26b-a4b.vindex --moe-shards "0-63=http://a:8081,64-127=http://b:8082"
```

### `larql model`

Manage HuggingFace **model** repos (raw safetensors + tokenizer + config).
Companion to `pull`, which is vindex-only.

```
larql model pull <model-id> [OPTIONS]
```

| Subcommand | Description |
|---|---|
| `pull <model-id>` | Download a HuggingFace model repo. Stages a raw HF model on disk for `convert safetensors-to-vindex` (or for any non-LARQL tool that wants the original weights). |

Use `larql pull` for vindex artifacts; use `larql model pull` to get the
upstream HF safetensors that an extraction would consume.

### `larql link`

Register a local vindex directory with the cache so `run` / `list` / `show`
can find it by shorthand.

```
larql link <PATH> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<PATH>` | Path to a vindex directory (must contain `index.json`) | — |
| `--as <NAME>` | Override the registered name (defaults to the directory basename with any `.vindex` suffix stripped) | basename |
| `-f, --force` | Replace an existing link of the same name | false |

### `larql shannon`

Shannon-style measurement tools for scripted demos. These use the dense
transformer forward pass to score the actual next token as
`-log2 p(token | context)`. They are measurement tools, not production
compressors.

```bash
larql shannon score google/gemma-3-4b-it --corpus frankenstein.txt --bytes 50000
larql shannon slot google/gemma-3-4b-it --prefix "The capital of France is " --answer Paris
larql shannon repeat google/gemma-3-4b-it --text frankenstein.txt --needle "created"
larql shannon encode google/gemma-3-4b-it --in frankenstein_4kb.txt --out compressed.lsc
larql shannon decode google/gemma-3-4b-it --in compressed.lsc --out recovered.txt
larql shannon encode google/gemma-3-4b-it --vindex ./gemma-q4k.vindex --metal --in frankenstein_4kb.txt --out compressed.lsc
larql shannon decode google/gemma-3-4b-it --vindex ./gemma-q4k.vindex --metal --in compressed.lsc --out recovered.txt
```

| Subcommand | Description |
|---|---|
| `score` | Score a corpus and print bits/token, bits/char, bits/byte, and total bits. |
| `slot` | Score an answer span after a prefix and show top predictions before the slot. |
| `repeat` | Score each occurrence of a string in its real preceding context. |
| `layers` | Per-layer Shannon bits via the final-norm logit lens. At every layer L (embed plus each post-block residual), project through `final_norm + lm_head` and report bits/token, KL-to-final, and adjacent `bits_saved[L]` deltas. |
| `encode` | Write a real arithmetic-coded bitstream driven by model probabilities. Intended for short excerpts. |
| `decode` | Reconstruct text from `encode` output using the same model. |
| `verify` | Three-engine cross-engine correctness check (LARQL Rust ↔ HF/PyTorch ↔ MLX) on the same corpus. Pretty delta table, exits non-zero if any pair-wise delta exceeds `--threshold`. See [Cross-engine verify](#cross-engine-verify) below. |

Without `--vindex`, `encode` / `decode` rerun the dense model for each
recovered token and are intended only for short excerpts. With `--vindex
--metal`, Q4K vindexes use the Metal KV-cache path and a full-vocabulary
LM-head query for each forced token. The vindex codec is segmented into
512-token arithmetic blocks so encode/decode stay byte-exact despite tiny GPU
float drift. The payload is real entropy-coded data; the file also includes a
small header with the first token, token count, original byte count, context
size, and payload length.

#### Cross-engine verify

`larql shannon verify` orchestrates the three independent bits/char scorers
(LARQL Rust in-process, plus MLX and HF/PyTorch as Python subprocesses) on
the same corpus and prints a delta table. Exits non-zero if any pair-wise
delta exceeds `--threshold`.

```bash
larql shannon verify google/gemma-3-4b-it \
    --corpus data/gutenberg/frankenstein.txt \
    --bytes 1024 \
    --context 512 --stride 256 \
    --threshold 0.5
```

Output:

```
engine     tokens   bits/token    bits/char     total bits   Δ vs ref    elapsed
--------------------------------------------------------------------------------
rust          247       4.3239       1.0712       1068.014    +0.000%       6.6s
mlx           247       4.3264       1.0718       1068.611    +0.056%       5.2s
hf            247       4.3239       1.0712       1068.011          —       6.8s

max pair-wise delta: 0.056% (mlx vs hf)
threshold:           0.500%
PASS
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--engines mlx,hf` | both | Comma list. `rust` always runs. |
| `--threshold 0.5` | 0.5% | PASS/FAIL boundary on max pair-wise delta. |
| `--python` | `.venv/bin/python` | Interpreter for the MLX and HF scripts. |
| `--mlx-script` | `scripts/shannon_score_mlx.py` | Path to the MLX scorer. |
| `--hf-script` | `scripts/shannon_score_hf.py` | Path to the HF scorer. |
| `--hf-device` | `cpu` | `cpu` is deterministic; `mps` is faster on Apple Silicon. |

Apparent quirks worth knowing about (also in
[`scripts/README_shannon_score.md`](../scripts/README_shannon_score.md)):

- The corpus is CRLF-normalized internally (Python `read_text` strips `\r`
  silently; LARQL Rust doesn't, and the difference would split tokenization).
- BOS handling matches LARQL's `encode_prompt(..., add_special_tokens=true)`
  semantics; don't strip BOS from the Python scripts manually.
- F32 throughout — the Python references cast bf16/fp16 weights to F32 on
  load. Mixed-precision would introduce its own noise.

When verify reports a real divergence, see
[`docs/diagnoses/shannon-cross-engine-divergence.md`](diagnoses/shannon-cross-engine-divergence.md)
for the bisection methodology and the three diagnostic env vars
(`LARQL_FORCE_GLOBAL_LAYERS`, `LARQL_ROPE_POS_DIVISOR_GLOBAL`,
`LARQL_LLAMA3_ROPE_SCALING`, `LARQL_NORM_EPS_OVERRIDE`) used to localise
the bugs in LARQL's forward path.

For a multi-architecture sweep covering F32 reference and the Q4K Metal
path in one shot:

```bash
.venv/bin/python scripts/diagnose_models.py
```

That runs `shannon verify` against every architecture LARQL supports
(SmolLM2, Llama, Mistral, Gemma 3) and additionally runs `shannon encode
--vindex --metal` for models with a local Q4K vindex to verify the GPU
path stays within typical Q4K_M quantization gap (~30–40%).

### `larql pull`

Download a vindex from HuggingFace into the HF hub cache
(`~/.cache/huggingface/hub/`).

```
larql pull <MODEL>
```

Accepts `hf://owner/name`, `owner/name`, or a local path (no-op). Prints
the resolved cache directory and basic metadata.

### `larql list`

Show every cached vindex, one row per entry.

```
larql list
```

Columns: `MODEL`, `SIZE (MB)`, `LAYERS`, `HIDDEN`. Scans the HF hub
cache for `datasets--<owner>--<name>/snapshots/<sha>/index.json`.

### `larql show`

Vindex metadata plus file inventory.

```
larql show <MODEL>
```

Prints layer count, hidden size, dtype, quant format, and each file in
the vindex with size. Resolves the same way as `run`.

### `larql rm`

Remove a cached vindex. Cache-only — never downloads.

```
larql rm <MODEL> [-y]
```

Accepts `owner/name` or cache shorthand. Prompts for confirmation unless
`-y` is passed.

### `larql serve`

Serve a vindex over HTTP. Loads a vindex into memory and exposes REST endpoints for knowledge queries, feature walks, edge selection, and patch management.

```
larql serve <VINDEX_PATH> [OPTIONS]
larql serve --dir <DIR> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<VINDEX_PATH>` | Path to .vindex directory or `hf://` URL | — |
| `--dir <DIR>` | Serve all .vindex directories in folder | — |
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--no-infer` | Disable inference endpoint (browse-only, saves memory) | false |
| `--ffn-only` | Run as an FFN-service endpoint for `larql run --ffn URL` clients. Implies `--no-infer`; advertises `mode: ffn-service` in `/v1/stats`. | false |
| `--cors` | Enable CORS headers for browser access | false |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--rate-limit <SPEC>` | Per-IP rate limit (e.g., "100/min", "10/sec") | — |
| `--trust-forwarded-for` | Use the first `X-Forwarded-For` IP for rate limiting. Enable only behind a trusted proxy. | false |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--cache-ttl <SECS>` | Cache TTL for DESCRIBE results (0 = disabled) | 0 |
| `--layers <START-END>` | Only load and serve layers in this range (e.g. `0-14`). Pages outside the range are never faulted in; RSS scales with shard size. | — |
| `--experts <START-END>` | Only serve expert IDs in this range (e.g. `0-63`). MoE shard filter. Mutually exclusive with `--units`. | — |
| `--units <PATH>` | Fine-grained per-(layer,expert) ownership manifest (JSON). Mutually exclusive with `--experts`. | — |
| `--moe-shards <SPEC>` | Server-side MoE expert dispatch: `"0-63=URL1,64-127=URL2"`. When set, the `walk-ffn` handler fans out MoE expert calls to remote shard servers. Combine with `--layers` for 2D layer × expert sharding. | — |
| `--moe-units-manifest <PATH>` | Fine-grained per-(layer,expert) server-side shard map. Mutually exclusive with `--moe-shards`. | — |
| `--join <ADDRS>` | Join one or more router grids (comma-separated gRPC addresses, e.g. `grpc://router:50052`). Self-assembling grid. Requires `--public-url`. | — |
| `--public-url <URL>` | Public HTTP URL for this server (used with `--join`). | — |
| `--grid-key <SECRET>` | Shared secret for grid auth (also `LARQL_GRID_KEY` env var). | — |
| `--max-gate-cache-layers <N>` | LRU cap on decoded f16 gate layers (0 = unlimited). | 0 |
| `--release-mmap-after-request` | `madvise(DONTNEED)` on all mmaps post-request. Linux: strict. Darwin: advisory. | false |
| `--embed-only` | Load only embeddings + lm_head (embed-server mode, ADR-0008). | false |
| `--grpc-port <PORT>` | Enable gRPC server on this port | — |
| `--uds-path <PATH>` | Bind a Unix domain socket alongside TCP for same-host MoE shard clients. | — |
| `--warmup-hnsw` | Eager-build HNSW index for every owned layer at startup. Requires `--hnsw`. | false |
| `--warmup-walk-ffn` | Pre-load inference weights and prefetch all owned layer mmap pages at boot. | false |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key for HTTPS | — |
| `--log-level <LEVEL>` | Logging level | info |

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/v1/describe?entity=France` | Knowledge edges (with probe relation labels) |
| GET | `/v1/walk?prompt=...&top=5` | Feature scan for a prompt |
| POST | `/v1/select` | SQL-style edge query |
| POST | `/v1/infer` | Full forward pass (walk/dense/compare) |
| GET | `/v1/relations` | List known relation types |
| GET | `/v1/stats` | Model and index statistics |
| POST | `/v1/patches/apply` | Apply a patch in-memory |
| GET | `/v1/patches` | List active patches |
| DELETE | `/v1/patches/{name}` | Remove a patch |
| POST | `/v1/walk-ffn` | Decoupled inference. Two modes — see below. |
| WS | `/v1/stream` | WebSocket streaming (layer-by-layer DESCRIBE) |
| GET | `/v1/health` | Health check (auth exempt) |
| GET | `/v1/models` | List loaded models |

**`POST /v1/walk-ffn`** has two modes:

- **Features-only (default).** Client POSTs a `[hidden]` residual; server
  runs gate KNN only and returns feature indices + scores. Client still
  needs `up_features.bin` + `down_features.bin` locally to compute the
  FFN output.
- **Full-output (`"full_output": true`).** Client POSTs a
  `[seq_len × hidden]` row-major residual plus `"seq_len": N`; server
  runs the architecture-correct `WalkFfn` path (gate KNN → activation →
  up gather → down projection) and returns the hidden-size FFN output
  for each requested layer. This is what the `larql run --ffn URL`
  client uses — the server holds all FFN weights, the client holds only
  attention.

Example (full-output):
```bash
curl -X POST http://server:8080/v1/walk-ffn \
  -H 'Content-Type: application/json' \
  -d '{
        "layers": [0, 1, 2],
        "residual": [/* seq_len * hidden floats */],
        "seq_len": 4,
        "full_output": true
      }'
```

Response shape: `{ "results": [{ "layer": N, "output": [...], "seq_len": 4 }, ...] }`.

The gRPC `WalkFfn` RPC mirrors the HTTP endpoint — see `vindex.proto`.

**Multi-model:** When using `--dir`, each model gets its own namespace: `/v1/{model_id}/describe`, etc.

**Examples:**

```bash
# Development — single model
larql serve output/gemma3-4b-v2.vindex --port 8080

# Multi-model server
larql serve --dir ./vindexes/ --port 8080

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080

# Browse-only with CORS for web clients
larql serve output/gemma3-4b-v2.vindex --no-infer --cors --port 8080

# With auth + HTTPS
larql serve output/gemma3-4b.vindex --api-key "sk-abc123" --tls-cert cert.pem --tls-key key.pem

# With rate limiting + DESCRIBE cache
larql serve output/gemma3-4b.vindex --rate-limit "100/min" --cache-ttl 300

# With rate limiting behind a trusted reverse proxy
larql serve output/gemma3-4b.vindex --rate-limit "100/min" --trust-forwarded-for

# Query from the REPL
larql repl
> USE REMOTE "http://localhost:8080";
> DESCRIBE "France";
> INFER "The capital of France is" TOP 5;
> APPLY PATCH "local-facts.vlp";    -- stays client-side
```

**Sessions:** Clients can send `X-Session-Id` header to isolate patches per session. Without the header, patches apply globally.

See `crates/larql-server/README.md` for full API documentation.

### `larql repl`

Launch the LQL interactive REPL. Tab completion, history, multi-line input.

```bash
larql repl
```

```sql
larql> USE "gemma3-4b.vindex";
larql> DESCRIBE "France";
larql> WALK "The capital of France is" TOP 5;
larql> INSERT INTO EDGES (entity, relation, target) VALUES ("John", "lives-in", "London");
larql> SAVE PATCH "edits.vlp";
```

Direct model access (no extraction needed):
```sql
larql> USE MODEL "google/gemma-3-4b-it";
larql> INFER "The capital of France is" TOP 5;
-- WALK/DESCRIBE/SELECT require EXTRACT into a vindex first
```

Remote mode:
```sql
larql> USE REMOTE "http://localhost:8080";
larql> DESCRIBE "France";
```

### `larql lql`

Execute a single LQL statement from the command line.

```bash
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
larql lql 'USE "gemma3-4b.vindex"; WALK "Einstein" TOP 10;'
```

## Research commands (dev)

These commands live under `larql dev <subcmd>`. They predate the REPL
and vindex format and remain available for low-level extraction,
debugging, and interpretability research.

The pre-redesign top-level invocations — `larql walk …`,
`larql weight-extract …`, `larql qk-templates …`, etc. — are still
accepted and rewritten to `larql dev <name>` transparently so existing
scripts keep working. Running any of them with `--help` prints the new
`Usage: larql dev <subcmd> …` form to signal the canonical home.

### `larql dev weight-extract`

Extract edges from FFN weight matrices. Zero forward passes. Pure matrix multiplication.

```
larql dev weight-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID (e.g. `google/gemma-3-4b-it`) |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per feature [default: 5] |
| `--min-score <MIN_SCORE>` | Minimum raw activation score for top-k selection [default: 0.02] |
| `--min-confidence <MIN_CONFIDENCE>` | Minimum normalized confidence [0-1] to keep an edge [default: 0.0] |
| `--stats <STATS>` | Write layer statistics to a separate JSON file |

**Model resolution:** Accepts a local directory path or a HuggingFace model ID. Model IDs are resolved from the HuggingFace cache at `~/.cache/huggingface/hub/` (or `$HF_HOME/hub/`).

**Resume:** If the output file already exists, completed layers are detected from edge metadata and skipped. Saves after each layer. Safe to interrupt and re-run.

**Confidence scoring:** Each edge gets a normalized confidence `c` in [0, 1], computed as `(c_in × c_out) / max(c_in × c_out)` per layer. Raw scores `c_in` (input selectivity) and `c_out` (output strength) are stored in metadata.

**Examples:**

```bash
# Full extraction
larql dev weight-extract google/gemma-3-4b-it -o knowledge.larql.json

# Single layer test
larql dev weight-extract google/gemma-3-4b-it --layer 26 -o L26.larql.json

# Filtered extraction with stats
larql dev weight-extract google/gemma-3-4b-it \
    -o knowledge.larql.json \
    --min-confidence 0.1 \
    --stats stats.json

# MessagePack output (smaller, faster)
larql dev weight-extract google/gemma-3-4b-it -o knowledge.larql.bin
```

### `larql dev attention-extract`

Extract routing edges from attention OV circuits. Zero forward passes.

```
larql dev attention-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per head [default: 3] |
| `--min-score <MIN_SCORE>` | Minimum score [default: 0.0] |

**How it works:** For each attention head, computes the OV circuit (`O_h @ V_h`), projects all vocab tokens through it, finds the most amplified inputs, and decodes what output tokens each produces.

**Resume:** Same as `weight-extract` — detects completed layers and skips them.

**Examples:**

```bash
larql dev attention-extract google/gemma-3-4b-it -o attention.larql.json
larql dev attention-extract google/gemma-3-4b-it --layer 12 -o attention-L12.larql.json
```

### `larql dev predict`

Run a full transformer forward pass from extracted safetensors weights and return top-k next-token predictions. Pure Rust inference — no MLX, no PyTorch.

```
larql dev predict <MODEL> --prompt <PROMPT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-p, --prompt <PROMPT>` | Prompt text to predict the next token for |
| `-k, --top-k <TOP_K>` | Number of top predictions to show [default: 10] |

**How it works:** Loads all safetensors weights, tokenizes the prompt (with BOS token), runs the full forward pass through all layers (embedding, attention with RoPE, GQA, QK norm, FFN with SiLU gating, all layer norms), projects the final residual against the embedding matrix, and returns softmax probabilities.

**Architecture awareness:** Uses the `ModelArchitecture` trait to handle model-specific behavior. Gemma 3 gets +1 norm offset, sqrt(hidden) embedding scale, QK normalization, and 4 norms per layer. Llama/generic gets standard behavior.

**Performance:** ~800ms per query for Gemma 3 4B on Apple Silicon (BLAS-accelerated, 34 layers, 6 tokens).

**Examples:**

```bash
# Basic prediction
larql dev predict google/gemma-3-4b-it --prompt "The capital of France is" -k 5
# 1. Paris (99.67%)

# Factual queries
larql dev predict google/gemma-3-4b-it --prompt "The largest planet is" -k 3
# 1. Jupiter (99.86%)

# Works with any HuggingFace model in cache
larql dev predict google/gemma-3-4b-it -p "Water freezes at" -k 10
```

For day-to-day inference against a `.vindex`, use
[`larql run`](#larql-run) — same forward pass, slimmer flag surface,
ollama-style ergonomics.

### `larql dev index-gates`

Build a precomputed gate index for graph-based FFN. Offline step — run once per model. Eliminates the gate matmul at inference time.

```
larql dev index-gates <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output index file (`.gate-index.jsonl`) |
| `--features-per-token <N>` | Features to index per token per layer [default: 100] |
| `--top-tokens <N>` | Top tokens to match at runtime [default: 10] |
| `--layers <LAYERS>` | Layers to index (e.g. `0-33` or `26,27,28`). Default: all |

**Examples:**

```bash
larql dev index-gates google/gemma-3-4b-it -o gates.gate-index.jsonl
larql dev index-gates google/gemma-3-4b-it -o gates.gate-index.jsonl --layers 24-33
```

### `larql dev walk`

Walk the model as a local vector index — gate KNN followed by down token lookup. No forward pass needed when using a `.vindex`. This is the research-grade inference path with the full flag surface; for everyday use prefer [`larql run`](#larql-run).

```
larql dev walk --prompt <PROMPT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-p, --prompt <PROMPT>` | Prompt text to walk through the model |
| `--index <PATH>` | Path to a `.vindex` directory (self-contained, no model needed) |
| `-m, --model <MODEL>` | Model path or HuggingFace model ID |
| `--gate-vectors <PATH>` | Path to extracted ffn_gate vectors (alternative to `--index`) |
| `--down-vectors <PATH>` | Path to extracted ffn_down vectors (alternative to `--index`) |
| `-k, --top-k <N>` | Top-K features per layer for gate KNN [default: 10] |
| `-l, --layers <LAYERS>` | Layers to walk. Comma-separated or range. Default: all |
| `--predict-top-k <N>` | Number of top predictions to show [default: 10] |
| `--predict` | Run full forward pass with walk FFN and show predictions (requires `--model`) |
| `--compare` | Compare walk FFN predictions against dense ground truth (requires `--model`). Incompatible with `--ffn-remote`. |
| `--down-top-k <N>` | Number of down tokens to show per feature [default: 5] |
| `-v, --verbose` | Show verbose loading and timing info |
| `--ffn-remote <URL>` | Route FFN to a remote `larql-server` via `POST /v1/walk-ffn` (`full_output: true`). Attention still runs locally; all layers are sent in a single binary batch round trip (`application/x-larql-ffn`, little-endian f32). Falls back to JSON if the server does not support binary. Same wire protocol that [`larql run --ffn`](#larql-run) uses. |
| `--ffn-remote-timeout-secs <N>` | Per-request HTTP timeout for `--ffn-remote` [default: 60] |

**Examples:**

```bash
# Walk with a pre-built .vindex
larql dev walk --prompt "The capital of France is" --index model.vindex

# Walk with loose vector files
larql dev walk --prompt "The capital of France is" \
    --gate-vectors vectors/ffn_gate.vectors.jsonl \
    --down-vectors vectors/ffn_down.vectors.jsonl

# Walk + compare against ground truth
larql dev walk --prompt "The capital of France is" --index model.vindex --model google/gemma-3-4b-it --compare

# Walk + FFN on a remote server (Act 2 of the demo)
larql dev walk --prompt "The capital of France is" --index client.vindex \
    --model google/gemma-3-4b-it --predict \
    --ffn-remote http://server:8080
```

### `larql dev attention-capture`

Capture and compare attention patterns across multiple prompts. Shows which heads attend similarly or differently.

```
larql dev attention-capture <MODEL> --prompts <PROMPTS> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-p, --prompts <PROMPTS>` | Prompts to compare (comma-separated) |
| `-l, --layers <LAYERS>` | Layers to capture. Comma-separated or range. Default: all |
| `--threshold <F>` | Attention threshold — only show heads with max attention above this [default: 0.1] |
| `-v, --verbose` | Show verbose per-head details |

**Examples:**

```bash
larql dev attention-capture google/gemma-3-4b-it \
    --prompts "The capital of France is,The capital of Germany is,The capital of Japan is"

larql dev attention-capture google/gemma-3-4b-it \
    --prompts "The capital of France is,The language of France is" \
    --layers 20-33 --threshold 0.2
```

### `larql dev qk-templates`

Extract attention template circuits from QK weight decomposition. Identifies which heads are "fixed" (same pattern regardless of entity) vs "variable".

```
larql dev qk-templates <MODEL> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-t, --templates <TEMPLATES>` | Template prompts (format: `relation:prompt`, comma-separated). Uses built-in templates if omitted |
| `-l, --layers <LAYERS>` | Layers to analyze. Default: all |
| `--threshold <F>` | Correlation threshold below which a head is "variable" [default: 0.95] |
| `--top-components <N>` | Number of top SVD components to show per head [default: 5] |

**Examples:**

```bash
larql dev qk-templates google/gemma-3-4b-it
larql dev qk-templates google/gemma-3-4b-it --layers 20-33 --threshold 0.90
```

### `larql dev ov-gate`

Map attention OV circuits to FFN gate features. Shows what each attention head activates in the next layer's FFN.

```
larql dev ov-gate <MODEL> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-l, --layers <LAYERS>` | Layers to analyze. Default: all |
| `-k, --top-k <N>` | Top-K gate features to show per head [default: 10] |
| `--heads <HEADS>` | Only show specific heads (for focused analysis) |
| `-v, --verbose` | Show verbose per-feature details |

**Examples:**

```bash
larql dev ov-gate google/gemma-3-4b-it --layers 25,26,27
larql dev ov-gate google/gemma-3-4b-it --layers 26 --heads 0,1,2 -k 20 -v
```

### `larql dev vector-extract`

Extract full weight vectors to intermediate NDJSON files.

```
larql dev vector-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output directory for `.vectors.jsonl` files |
| `--components <COMPONENTS>` | Components to extract (comma-separated): `ffn_down`, `ffn_gate`, `ffn_up`, `attn_ov`, `attn_qk`, `embeddings` |
| `--layers <LAYERS>` | Layers to extract (comma-separated). Default: all |
| `--top-k <TOP_K>` | Top-k tokens for metadata per vector [default: 10] |
| `--resume` | Resume from existing output files |

**Examples:**

```bash
# Extract all components
larql dev vector-extract google/gemma-3-4b-it -o vectors/

# Extract only FFN down projections from layers 25-33
larql dev vector-extract google/gemma-3-4b-it -o vectors/ \
    --components ffn_down --layers 25,26,27,28,29,30,31,32,33
```

### `larql dev residuals capture`

Capture residual stream vectors for entities via forward passes. The residuals are the hidden state at a specific layer — the signal that the next layer's features actually see during inference.

```
larql dev residuals capture <MODEL> --entities <ENTITIES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-e, --entities <ENTITIES>` | Comma-separated entities, or path to a text file (one per line) |
| `-l, --layer <LAYER>` | Layer(s) to capture at. Can specify multiple times. [default: 25] |
| `--all-layers` | Capture at every layer |
| `-o, --output <OUTPUT>` | Output directory for NDJSON files |
| `--template <TEMPLATE>` | Prompt template. `{entity}` is replaced. Default: bare entity name |
| `--activations` | Also capture sparse FFN activations (top-K features per layer) |
| `--activation-top-k <N>` | Number of top features to record per layer [default: 50] |

**How it works:** Tokenizes each entity, runs a full forward pass through the transformer up to the target layer(s), and saves the last-token hidden state as a vector in NDJSON format.

**Use case:** Capture real residual vectors for a small set of entities (50–100). These can be used as query vectors for vindex gate KNN lookups to discover factual edges without additional forward passes.

**Examples:**

```bash
# L25 residuals for seed entities
larql dev residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o residuals-L25.vectors.ndjson

# Multiple layers
larql dev residuals capture google/gemma-3-4b-it \
    --entities entities.txt \
    --layer 25 --layer 26 --layer 29 \
    -o residuals.vectors.ndjson

# Full trajectory (all layers)
larql dev residuals capture google/gemma-3-4b-it \
    --entities "France" --all-layers \
    -o residuals-full.vectors.ndjson

# With prompt template
larql dev residuals capture google/gemma-3-4b-it \
    --entities "France,Germany" \
    --layer 25 \
    --template "The capital of {entity} is" \
    -o residuals-capital.vectors.ndjson
```

**Output format:** Same NDJSON as `vector-extract`:

```json
{"_header": true, "component": "residuals", "model": "google/gemma-3-4b-it", "dimension": 2560}
{"id": "France_L25", "layer": 25, "feature": 0, "vector": [...], "top_token": "Paris", "c_score": 12.4, ...}
```

### `larql dev bfs`

BFS extraction from a running model endpoint.

```
larql dev bfs --seeds <SEEDS> --templates <TEMPLATES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-s, --seeds <SEEDS>` | Comma-separated seed entities |
| `-t, --templates <TEMPLATES>` | Path to templates JSON file |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-e, --endpoint <ENDPOINT>` | Model endpoint URL [default: `http://localhost:11434/v1`] |
| `-m, --model <MODEL>` | Model name for the endpoint |
| `--mock` | Use mock provider instead of HTTP |
| `--mock-knowledge <PATH>` | Path to mock knowledge JSON (with `--mock`) |
| `--max-depth <N>` | Maximum BFS depth [default: 3] |
| `--max-entities <N>` | Maximum entities to probe [default: 1000] |
| `--min-confidence <F>` | Minimum edge confidence [default: 0.3] |
| `--resume <PATH>` | Resume from a checkpoint file |

**Requires:** Templates JSON file defining prompt templates for each relation. See [format.md](format.md) for template format.

**Examples:**

```bash
# Against Ollama
larql dev bfs \
    --seeds "France,Germany,Japan" \
    --templates templates.json \
    --endpoint http://localhost:11434/v1 \
    --model gemma3:4b-it \
    -o knowledge.larql.json

# With mock provider
larql dev bfs \
    --seeds "France,Germany" \
    --templates templates.json \
    --mock --mock-knowledge mock.json \
    -o knowledge.larql.json
```

### Other `larql dev` subcommands

The following research subcommands exist and respond to `--help` but are
not documented in detail above. They are stable enough to use but are
mostly driven by the comments in their args structs and the experiment
write-ups in `experiments/`.

| Subcommand | One-line summary |
|---|---|
| `qk-rank` | SVD rank analysis of attention QK products. |
| `qk-modes` | Extract interpretable modes from low-rank QK heads via SVD → gate projection. |
| `ov-rd` | OV rate-distortion + residual-table attention compilation experiments. |
| `circuit-discover` | Discover attention → FFN circuits from weight decomposition. |
| `attn-bottleneck` | Bottleneck analysis of attention components. |
| `ffn-bottleneck` | Bottleneck analysis of FFN components. |
| `ffn-overlap` | Measure overlap between entity-routed and ground-truth gate features. |
| `kg-bench` | Knowledge graph retrieval benchmark. |
| `trajectory-trace` | Trace residual stream trajectories on the sphere across layers. |
| `projection-test` | Test rank-k projection through the residual stream. |
| `fingerprint-extract` | Extract OV fingerprint basis from attention weights. |
| `bottleneck-test` | Test rule-based bottleneck — if-else rules replace early layers. |
| `embedding-jump` | Embedding jump — raw token embeddings → projected L13 → decoder. |
| `ffn-latency` | Measure round-trip latency breakdown against a remote FFN server. |

Run `larql dev <subcmd> --help` for the full flag surface of any of
these.

## Build commands

Top-level verbs for producing, converting, and publishing vindexes.

### `larql extract-index`

Build a `.vindex` — the model decompiled to a standalone vector index.
Can be queried with [`larql run`](#larql-run) or
[`larql dev walk`](#larql-dev-walk) without the original model.

`larql extract` is the canonical form; `larql extract-index` is kept as
a backwards-compatible alias.

```
larql extract-index [MODEL] --output <OUTPUT> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Model path or HuggingFace model ID (not needed with `--from-vectors`) | — |
| `-o, --output <OUTPUT>` | Output path for the `.vindex` directory | — |
| `--level <LEVEL>` | `browse` / `attention` / `inference` / `all` — strict increasing tiers. See below. | `inference` |
| `--f32` | Opt out of f16 on side-channel tensors. Rarely wanted — doubles file sizes. | off (f16) |
| `--quant <FORMAT>` | Inline-quantise forward-pass weights: `none`, `q4k`, or `kquant` (alias). The k-quant family emits Q4_K/Q6_K Ollama-compatible blocks; implies `--level all` + f16 side-channels. | `none` |
| `--compact` | Skip `up_weights.bin` + `down_weights.bin`; FFN weights live only in feature-major files. `WalkFfn`-only. | off |
| `--drop-gate-vectors` | Skip `gate_vectors.bin` entirely; loader rebuilds gate from `interleaved_kquant.bin` (or legacy `interleaved_q4k.bin`) at load. Only with `--quant q4k` / `kquant`. | off |
| `--down-q4k` | Quantise FFN down-proj as Q4_K instead of Q6_K. Saves ~1.8 GB on 31B, cuts down-matmul cost ~1.5–1.7× at decode. Introduces ~2.5× more probability-redistribution noise (top-1 + top-5 preserved). Validated by `walk_correctness`, which auto-relaxes its prob-delta gate from 0.02 to 0.035 when it detects Q4_K down. Only with `--quant q4k` / `kquant`. | off |
| `--from-vectors <PATH>` | Build from already-extracted NDJSON vector files instead of model weights | — |
| `--down-top-k <N>` | Top-K tokens per feature in down metadata | 10 |
| `--include-weights` | Alias for `--level all` (deprecated — use `--level` directly) | — |
| `--resume` | Skip stages that already have output files | off |

**Extract tiers (`--level`).** Each tier is a strict superset of the
previous:

| Tier | Adds | Enables |
|---|---|---|
| `browse` | gate + embed + down_meta + tokenizer | WALK / DESCRIBE / SELECT (no forward pass) |
| `attention` | + attention + norms | client-side of `run --ffn URL` (Act 2 demo) |
| **`inference` (default)** | + FFN up/down | full local forward pass (INFER) |
| `all` | + lm_head + COMPILE extras | COMPILE |

**Examples:**

```bash
# Default — inference-ready, f16 (~6 GB for 4B)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex

# Browse-only (no forward pass, ~3 GB)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level browse

# Attention-only slice for Act 2 of the demo — carve out the
# client-side half of an FFN-over-HTTP pair
larql extract google/gemma-4-31b-it -o gemma4-31b.client.vindex --level attention

# All (+ lm_head for COMPILE)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level all

# Q4_K/Q6_K quantised inline — smallest disk footprint, Ollama-compat
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --quant q4k

# Maximum size reduction on Q4K — drop the redundant f16 gate, rebuild
# from the Q4K at load time
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex \
  --quant q4k --drop-gate-vectors

# All-Q4K FFN — gate + up + down all Q4_K, faster down projection at
# decode (~1.5–1.7× on CPU), ~30 MB/layer smaller. Trades ~2.5× more
# softmax-probability drift; validate with `walk_correctness`.
larql extract google/gemma-4-31b-it -o gemma4-31b.vindex \
  --quant q4k --down-q4k

# Legacy name still works
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex

# Build from pre-extracted vectors
larql extract -o gemma3-4b.vindex --from-vectors vectors/

# Resume an interrupted build
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --resume
```

**`--quant q4k` (alias `--quant kquant`) details:**

- Q/K/O + gate/up: Q4_K (148 bytes per 256 values)
- V + down: Q6_K (210 bytes per 256 values), or Q4_K with `--down-q4k`
- Writers emit `attn_weights_kquant.bin`, `interleaved_kquant.bin`,
  `lm_head_kquant.bin` (plus matching `*_manifest.json` sidecars).
  Readers also accept the legacy `*_q4k.bin` / `lm_head_q4.bin`
  filenames so pre-rename vindexes keep loading. `index.json` records
  `"quant": "q4k"` for v1 vindexes; readers accept `"kquant"` too.
- `--level browse` is implicitly promoted to `--level all` — the
  k-quant writer materialises all of attention, FFN, norms, and
  `lm_head` in one pass, so a browse-only k-quant vindex would be
  incoherent.
- Side-channel tensors that k-quant doesn't quantise — `gate_vectors.bin`
  and `embeddings.bin` — are stored at f16 by default under `--quant q4k`.
  Leaving them at f32 pairs 4-bit main weights with 32-bit lookup
  tables, roughly doubling the vindex footprint for no accuracy gain.
- V-shares-K fallback: on Gemma 4 31B global layers where `v_proj` is
  absent, K's bytes are stored in V's slot (still tagged `Q6_K`, still
  keyed by the V tensor name) so downstream 4-per-layer indexing
  stays valid.
- `VindexConfig.quant = Q4k` is written to `index.json`; loaders
  should dispatch on this field rather than sniffing filenames.

### `larql build`

Build a custom model from a Vindexfile (declarative: FROM + PATCH + INSERT).

```
larql build [DIR] [OPTIONS]
```

| Flag | Description |
|---|---|
| `[DIR]` | Directory containing the Vindexfile [default: `.`] |
| `--stage <NAME>` | Build stage (e.g. `dev`, `prod`, `edge`) |
| `-o, --output <PATH>` | Output directory for the built vindex [default: `./build/vindex/`] |
| `--compile <FORMAT>` | Compile output to a model format after building |

**Examples:**

```bash
# Build from Vindexfile in current directory
larql build .

# Build a specific stage
larql build . --stage prod

# Build to a custom output path
larql build . --output custom.vindex
```

### `larql convert`

Convert between model formats.

```
larql convert <SUBCOMMAND>
```

| Subcommand | Description |
|---|---|
| `gguf-to-vindex` | Convert a GGUF model to a vindex (dequantized to f32) |
| `safetensors-to-vindex` | Convert safetensors model to a vindex |
| `gguf-info` | Show GGUF file metadata and detected architecture |
| `quantize fp4` | Quantise an existing f32/f16 vindex to the LARQL FP4/FP8 format |
| `quantize q4k` | Quantise an existing f32/f16 vindex to GGML Q4_K_M (Ollama-compatible) |

**Examples:**

```bash
# Convert GGUF to vindex
larql convert gguf-to-vindex model-Q4_K_M.gguf -o model.vindex --f16

# Show GGUF metadata
larql convert gguf-info model-Q4_K_M.gguf

# Convert safetensors to vindex
larql convert safetensors-to-vindex ./model/ -o model.vindex --level inference --f16

# Quantise an existing f16 vindex to FP4 (Option B: source-dtype gate + FP4 up + FP8 down)
larql convert quantize fp4 \
    --input  output/gemma3-4b-f16.vindex \
    --output output/gemma3-4b-fp4.vindex

# Quantise an existing f16 vindex to Q4_K_M (attn Q/K/O + FFN gate/up at Q4_K, V + FFN down at Q6_K)
larql convert quantize q4k \
    --input  output/gemma3-4b-f16.vindex \
    --output output/gemma3-4b-q4k.vindex

# Q4_K_M with FFN down also at Q4_K (saves ~30 MB/layer on 31B at modest precision cost)
larql convert quantize q4k \
    --input  output/gemma4-31b-f16.vindex \
    --output output/gemma4-31b-q4k.vindex \
    --down-q4k
```

Supported GGUF quantization types for reading: F32, F16, BF16, Q4_0, Q4_1, Q8_0. All tensors are dequantized to f32 during conversion.

**`quantize` family** — see [`docs/specs/quantize-cli-spec.md`](specs/quantize-cli-spec.md) for the full surface (flags, exit codes, output layout, atomic-rename semantics). Both subcommands require the source vindex to carry full model weights (`--level inference` or `--level all`); browse-only sources are rejected with a clear error.

### `larql hf`

HuggingFace Hub: download or publish vindexes.

```
larql hf <SUBCOMMAND>
```

| Subcommand | Description |
|---|---|
| `download` | Download a vindex from HuggingFace to local cache |
| `publish` | Upload a local vindex to HuggingFace |

**Examples:**

```bash
# Download a vindex
larql hf download chrishayuk/gemma-3-4b-it-vindex

# Download to a specific directory
larql hf download chrishayuk/gemma-3-4b-it-vindex -o ./gemma3-4b.vindex

# Download a specific version
larql hf download chrishayuk/gemma-3-4b-it-vindex --revision v2.0

# Publish a vindex
larql hf publish ./gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
```

HuggingFace vindexes can also be used directly from the REPL:

```sql
larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
larql> DESCRIBE "France";
```

Publishing requires `HF_TOKEN` environment variable or `huggingface-cli login`.

### `larql verify`

Verify vindex file integrity against stored SHA256 checksums.

```
larql verify <VINDEX>
```

**Example:**

```bash
larql verify gemma3-4b.vindex
# gate_vectors.bin ... OK (1.66 GB)
# embeddings.bin ... OK (1.25 GB)
# down_meta.bin ... OK (2.0 MB)
# All 3 files verified.
```

### `larql compile`

Compile vindex patches into model weights (AOT — bake patches at install
time so live inference doesn't need to consult the patches). Used to
materialise compiled fact / passage edges into a deployable model.

```
larql compile --base <BASE> --output <OUTPUT> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `--base <PATH>` | Path to the base model (directory with safetensors, or HF model ID) | — |
| `--vindex <PATH>` | Path to the vindex with patches to compile (not needed for fact mode) | — |
| `-o, --output <PATH>` | Output directory for the compiled model safetensors | — |
| `--gate-scale <F>` | Gate scale for compiled edges (1.0 keeps natural usage clean on Gemma 3 4B; previous default 30.0 saturated silu and leaked the edge into unrelated queries) | 1.0 |
| `--alpha <F>` | Initial write-magnitude multiplier; the balancer refines this after install | 0.3 |
| `--floor <F>` | Minimum target-token probability before the balancer stops scaling up | 0.40 |
| `--ceiling <F>` | Maximum target-token probability before the balancer scales down | 0.85 |
| `--max-iters <N>` | Balancer iterations (`0` = opt-out, install at `--alpha` / `--gate-scale` and trust the caller's defaults) | 0 |
| `--no-chat-template` | Skip applying the base model's chat template before tokenising. Default behaviour wraps the prompt so the captured trigger residual matches a chat-wrapped deployment | off |
| `--prompt <TEXT>` | Prompt whose residual becomes the trigger direction | — |
| `--answer <TEXT>` | Correct answer token to compile into the weights | — |
| `--layer <N>` | Layer to install the compiled edge at | 30 |
| `--slot <N>` | FFN slot to install the compiled edge at | 9000 |

The balancer is opt-in (`--max-iters 0` by default) because
`larql_inference::forward::predict` is systematically peakier than HF's
`forward` on the same weights — leaving the balancer off installs at the
hand-tuned `g=1.0, α=0.3` sweet spot from the paraphrase sweep.

### `larql diag`

Engine diagnostic. Loads a vindex through the production path and prints
which kernel paths the loader picks (lm_head fast/slow, attn fused/per-proj),
validates Q4_K/Q6_K manifest strides against the canonical 144-byte GGUF
layout, and surfaces silent-slowdown classes (stale 148-byte stride,
`vocab_size=0`) at a glance.

```
larql diag <MODEL> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Vindex dir, `hf://owner/name`, `owner/name`, or cache shorthand | — |
| `--probe` | Run a real forward pass and print per-stage timings | off |
| `--probe-tokens <N>` | Token count for `--probe` (caps at 100 to keep the diagnostic snappy) | 5 |

**Examples:**

```bash
# Static check only — no forward pass
larql diag gemma3-4b-q4k-v2.vindex

# Static check + 50-token probe with per-stage timing breakdown
larql diag gemma3-4b-q4k-v2.vindex --probe --probe-tokens 50
```

Two-pass output:
1. **Static** — config (`hidden_size`, `vocab_size`, dtype), file inventory,
   stride validation. Doesn't load the vindex; safe for huge models.
2. **Loaded** — opens via `open_inference_vindex`, reports which kernels
   would actually fire (lm_head fast path, attention fused/per-proj, etc.).

### `larql parity`

Cross-backend numerical parity diff. Runs the same input through multiple
backends (`reference`, `cpu`, `metal`) and reports the first checkpoint where
they diverge beyond `--tolerance`. Catches "I refactored quantization /
activation / norm and silently broke something" regressions that latency
benches and synthetic-weight unit tests miss.

```
larql parity <MODEL> --component <C> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Vindex dir, `hf://` URL, or cache shorthand | — |
| `--component <C>` | Inference checkpoint to diff: `moe-expert`, `moe-block`, `lm-head`, `layer` | moe-block |
| `--layer <N>` | Layer index | 0 |
| `--expert <N>` | Expert index (used when `--component moe-expert`) | 0 |
| `--backends <LIST>` | Comma-separated backends. First is the reference; others are diffed against it | reference,cpu |
| `--prompt <TEXT>` | Prompt for `--component layer` (drives the actual forward pass). For `moe-*`, seeds a synthetic residual; otherwise a deterministic sin pattern is used | — |
| `--seed <N>` | Random-ish seed for the synthetic residual. Ignored when `--prompt` is set | 0 |
| `--tolerance <F>` | Max element-wise abs diff before declaring divergence (per-expert ≈ 1e-3; full forward needs more headroom for accumulated f32 noise) | 1e-3 |
| `-v, --verbose` | Print intermediate values at each checkpoint, not just diffs | off |

**Components:**

| Component | What it diffs |
|---|---|
| `moe-expert` | Single expert forward (gate matmul, up matmul, gelu_tanh, down matmul) |
| `moe-block` | Full MoE block, one layer (router → top-K → K experts → weighted sum → post-norm) |
| `lm-head` | Final projection parity (Q4_K vs f32 reference). Backend-agnostic; works on any vindex with an lm_head |
| `layer` | Full transformer layer end-to-end. Reads per-layer `metal_layer_NN_h_out.f32` / `metal_layer_NN_h_post_attn.f32` dumps; works on dense models too |

Requires the `metal` feature on macOS — Metal is the reference backend the
CPU output is compared against.

```bash
# Smoke-test MoE block on layer 0
larql parity gemma4-26b-a4b.vindex --component moe-block --layer 0

# Diff lm_head Q4_K against f32 reference on a dense vindex
larql parity gemma3-4b-q4k.vindex --component lm-head

# Per-layer residual diff between CPU and Metal across a real prompt
larql parity gemma4-31b.vindex --component layer --prompt "The capital of France is"
```

## Graph-file commands

These operate on the NDJSON/MessagePack knowledge-graph files produced
by `larql dev weight-extract` / `larql dev bfs` — the pre-vindex output
format. For a vindex, use LQL (`DESCRIBE` / `SELECT` / `STATS` in the
REPL) instead.

### `larql query`

Select edges from a subject, optionally filtered by relation.

```
larql query --graph <GRAPH> <SUBJECT> [RELATION]
```

```bash
larql query --graph knowledge.larql.json France
larql query --graph knowledge.larql.json France capital-of
```

### `larql describe`

Show all outgoing and incoming edges for an entity.

```
larql describe --graph <GRAPH> <ENTITY>
```

```bash
larql describe --graph knowledge.larql.json France
```

### `larql stats`

Show graph statistics: entity count, edge count, relation count, connected components, average degree, average confidence, source distribution.

```
larql stats <GRAPH>
```

```bash
larql stats knowledge.larql.json
```

### `larql validate`

Check a graph file for issues: zero-confidence edges, self-loops, empty subjects/objects.

```
larql validate <GRAPH>
```

```bash
larql validate knowledge.larql.json
```

### `larql merge`

Merge multiple graph files into one.

```
larql merge <INPUT>... --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<INPUT>...` | Input graph files to merge (at least 2) |
| `-o, --output <OUTPUT>` | Output merged graph file |
| `--strategy <STRATEGY>` | Merge strategy: `union`, `max_confidence`, `source_priority` [default: `union`] |

**Examples:**

```bash
larql merge weights.larql.json attention.larql.json -o combined.larql.json
larql merge weights.larql.json bfs.larql.json -o combined.larql.json --strategy max_confidence
```

## Templates format

Used by `larql bfs`. A JSON array of prompt templates:

```json
[
  {
    "relation": "capital-of",
    "template": "The capital of {subject} is",
    "multi_token": true,
    "stop_tokens": [".", "\n", ",", ";"]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `relation` | string | Relation name for edges produced by this template |
| `template` | string | Prompt text. `{subject}` is replaced with the entity name |
| `multi_token` | bool | Chain multiple forward passes for multi-token answers |
| `reverse_template` | string? | Optional reverse probe (`{object}` placeholder) |
| `stop_tokens` | char[] | Characters that terminate multi-token chaining |

## Mock knowledge format

Used by `larql bfs --mock`. A JSON array:

```json
[
  {"prompt": "The capital of France is", "answer": "Paris", "probability": 0.89}
]
```

## Distribution: `slice`, `publish`, `pull`

The three commands form the distribution pipeline: extract once, carve
the deployment variants you need, push them to HuggingFace as sibling
repos under a shared collection, pull them on the consumer side with
progress + resume.

Architectural rationale (why each slice exists, how collections get
composed, SHA256 skip semantics, empty-gate loader for attention-only
clients) is in
[`docs/adr/0007-vindex-distribution.md`](adr/0007-vindex-distribution.md).
Command reference below.

### `slice`

Carve a built vindex into deployment variants. Pure file I/O + `index.json`
rewrite — no re-extract.

| Flag | Description | Default |
|---|---|---|
| `<SRC>` | Source vindex: directory, `hf://owner/name`, cache shorthand | — |
| `-o, --output <DST>` | Destination directory. Must not exist unless `--force`. | — |
| `--preset <NAME>` | `client`, `attn`, `embed`, `server`, `browse`, `router`, `expert-server`, `all` | — |
| `--parts <list>` | Explicit parts (embed, norms, attn, gate, down_meta, ffn, expert_layers, lm_head, router, tokenizer, manifest, labels, readme). `index.json` is always copied. | — |
| `--force` | Overwrite `<DST>` if it exists | false |
| `--dry-run` | Preview what would be copied | false |

**Preset sizes (Gemma 3 4B Q4_K measured; 31B figures scaled):**

| Preset | Topology | 4B | 31B Q4K | 26B MoE | Pairs with |
|---|---|---|---|---|---|
| `client` | 2-tier | 3.0 GB | 7.4 GB | 2.1 GB | `larql run --ffn URL` |
| `attn` | 3-tier | 310 MB | 4.8 GB | — | `larql run --embed URL --ffn URL` (ADR-0008) |
| `embed` | 3-tier | 1.28 GB | 2.6 GB | — | `larql serve --embed-only` (ADR-0008) |
| `server` | either | 1.8 GB | 27 GB | — | `larql serve --ffn-only` |
| `browse` | — | 1.3 GB | 16 GB | — | DESCRIBE/WALK only |
| `expert-server` | MoE | — | — | 14.1 GB | `larql serve --experts START-END` |
| `full` | — | 1.3 GB | 32 GB | 16 GB | everything |

`expert-server` includes embed, norms, dense FFN (`interleaved_q4k.bin`),
and the per-layer expert weights (`layers/`). Everything `larql serve` needs
to boot and serve `POST /v1/expert/batch` calls on a CPU-only machine.

Use `attn` + `embed` when laptop RAM matters and you can run an embed
server alongside the FFN server. `attn` alone is 10× smaller than
`client` on 4B because the embedding table (2.7 GB) is the biggest
piece of a client vindex.

### `publish`

Upload the full vindex plus sibling slices to HuggingFace and file them
into three nested collections.

| Flag | Description | Default |
|---|---|---|
| `<SRC>` | Source vindex | — |
| `--repo <OWNER/NAME>` | HF repo ID for the full vindex. Siblings derive from `--slice-repo-template`. | required |
| `--full` / `--no-full` | Upload the full vindex | `--full` |
| `--slices <list>` | Presets to upload alongside the full vindex. `none` to skip. Covers both 2-tier (`client`) and 3-tier (`attn` + `embed`) topologies by default. | `client,attn,embed,server,browse` |
| `--slice-repo-template <T>` | `{repo}` → `--repo`, `{preset}` → preset. | `{repo}-{preset}` |
| `--collections <list>` | `model`, `family`, `library`. `none` to skip. | `model,family,library` |
| `--model-title <T>` | Override per-model collection title | derived |
| `--family <NAME>` | Override family collection group | derived |
| `--library-title <T>` | Override library collection title | `LARQL Vindex Library` |
| `--force-upload` | Re-upload every file; ignore SHA256 skip | false |
| `--tmp-dir <DIR>` | Staging directory for slice carving | system temp |
| `--dry-run` | Preview, no HF writes | false |

**Skip-if-unchanged** (default on): before each upload the client fetches
the repo's LFS file index and compares `lfs.oid` against the local
SHA256. Matches skip. Small json files always re-upload. Re-publishing
a 27 GB server slice where nothing changed transfers only a few KB.

**Streaming + progress**: uploads stream the file via a counting
`Read` adapter that ticks a per-file indicatif bar as bytes flow out.
No whole-file-into-RAM pre-read. An interrupted `publish` restarts via
the SHA256 skip on completed files; the interrupted file re-uploads.

### `pull`

Download a vindex (or a slice, or a whole collection) with per-file
progress bars. hf-hub handles `.incomplete` partial-file resume
internally.

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | `hf://owner/name[@rev]`, `owner/name`, or local path. Omit with `--collection`. | — |
| `--preset <NAME>` | Pull `{repo}-{preset}` instead of the named repo. | — |
| `--all-slices` | Full + every default sibling (`-client`, `-attn`, `-embed`, `-server`, `-browse`). Missing siblings warn, don't fail. | false |
| `--collection <SLUG\|URL>` | Pull every dataset in an HF collection. | — |
| `--sibling-template <T>` | Must match `publish --slice-repo-template`. | `{repo}-{preset}` |
| `--output <PATH>` | Download to this path instead of the default local cache. Idempotent: skips if `index.json` already present. Use in container startup scripts. | cache |

After a plain `pull <repo>`, `larql` HEAD-probes for standard siblings
and prints an "also available" hint if any exist — so the sliced layout
is self-announcing from a single repo URL.

```bash
larql pull chrishayuk/gemma-4-31b-it-vindex
# → progress bars per file, then:
#   Also available on HuggingFace:
#     --preset client   → hf://chrishayuk/gemma-4-31b-it-vindex-client
#     --preset attn     → hf://chrishayuk/gemma-4-31b-it-vindex-attn
#     --preset embed    → hf://chrishayuk/gemma-4-31b-it-vindex-embed
#     --preset server   → hf://chrishayuk/gemma-4-31b-it-vindex-server
#     --preset browse   → hf://chrishayuk/gemma-4-31b-it-vindex-browse
#   Use `larql pull <repo> --all-slices` to grab them all.
```

Requires `HF_TOKEN` or `~/.huggingface/token` only for private repos
and collections; public pulls work unauthenticated.
