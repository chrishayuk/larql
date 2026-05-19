# Roadmap — larql-cli

For shipped work, see [CHANGELOG.md](CHANGELOG.md).

## Current state

Primary verbs: `run`, `chat`, `pull`, `model`, `link`, `list`, `show`, `slice`,
`publish`, `rm`, `bench`, `shannon`, `serve`. Build verbs: `extract`, `build`,
`compile`, `convert`, `verify`, `hf`, plus diagnostic verbs `diag` and `parity`.
Legacy research commands gated under `larql dev <subcmd>` for backwards-compat.
Dual cache (HuggingFace hub + `~/.cache/larql/local/`) with shorthand resolution
(`larql run gemma3-4b-it-vindex`).

The `shannon` family was extended in 2026-05-16 with **`larql shannon
verify`** — a cross-engine bits/char correctness check that orchestrates
the LARQL Rust forward (in-process) against HF/PyTorch and MLX reference
scorers (subprocesses driving
`scripts/shannon_score_{hf,mlx}.py`). Prints a delta table, exits
non-zero if any pair-wise delta exceeds `--threshold` (default 0.5%).
First serious application surfaced four config-loading bugs in
`larql-models` (rms_norm_eps not parsed; Gemma 3 per-layer-type
rope_scaling missing; llama3 rope_scaling missing; StarCoder2
norm_epsilon alias). The CI gate at
[`.github/workflows/shannon-verify.yml`](../../.github/workflows/shannon-verify.yml)
runs this on every PR. Per-arch sweep:
[`scripts/diagnose_models.py`](../../scripts/diagnose_models.py). See
[`docs/cli.md#cross-engine-verify`](../../docs/cli.md#cross-engine-verify)
and
[`docs/diagnoses/shannon-cross-engine-divergence.md`](../../docs/diagnoses/shannon-cross-engine-divergence.md).

---

## P1: Generation UX

### Sampling flags
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`
Add `--temperature F`, `--top-p F`, `--top-k N`, `--repetition-penalty F` to
the `run` / `chat` subcommands. Values are threaded through to `generate.rs`
logit post-processing (tracked in larql-inference P0).

### `--max-context N`
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`
Expose `--max-context N` (default 8192). Thread through to `KVCache::new_per_layer`
in `generate.rs`. `larql chat` should also respect this for multi-turn state.

### Auto-extract on `larql run hf://`
**Status**: Not started
**Files**: `src/commands/primary/cache.rs` (resolver)
If the shorthand looks like `hf://owner/name` and no cached vindex is found, offer
to run `larql extract` inline (confirm prompt or `--yes`). Collapses the three-step
`extract → link → run` flow to one command. Today only **vindex** `hf://` paths
resolve via the cache; raw HF model paths still need an explicit `extract`.

### OpenAI-compatible surface — CLI side
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`
After the server-side `/v1/chat/completions` endpoint lands (larql-server P0),
expose `larql run --openai-url URL` to send prompts to any OpenAI-compatible
endpoint (including the local `larql serve` instance). Useful for round-trip
testing without a client library.

---

## P2: parity polish

`larql parity` is wired and shipping (see CHANGELOG 2026-05-10). Remaining
open scoping work from the original 2026-04-27 design:

### `--json` output
**Files**: `src/commands/diagnostics/parity.rs`
Human-readable table by default; `--json` emits machine-parseable diff records
for CI consumption (`max_diff`, `index_of_first_divergence`, `checkpoint_name`).

### `--from-recording <path>` replay
**Files**: `src/commands/diagnostics/parity.rs`
Replay a previously captured trace without reloading the model. Useful for
repeated diffs against the same recorded reference run; pairs naturally with
HF sidecar captures once those exist.

### Per-component tolerance defaults
**Files**: `src/commands/diagnostics/parity.rs`
`forward` after 30 layers will accumulate to ~1e-2 even for "correct"
backends; `--tolerance` should default per-component instead of a single
`1e-3`.

### Trace-point infrastructure (larql-inference side)
**Files**: `larql-inference/src/diagnostics/` (new module)
Today `parity` runs each backend end-to-end and compares outputs. The
designed-but-unbuilt extension is named trace points (`post_pre_norm`,
`post_router_softmax`, `post_gate_matmul`, `post_activation`,
`post_down_matmul`, `post_combine`, `post_post_norm`) emitted to a
registered `TraceSink`. Walking the merged traces would let the diagnostic
print the **first divergence** with full surrounding context. Gated on a
`diagnostics` cargo feature in `larql-inference` so release builds pay zero
overhead. Scoped here because the CLI is the primary consumer; the
underlying work belongs to larql-inference.

### `hf` backend for parity
**Files**: `tools/hf_capture.py` + `src/commands/diagnostics/parity.rs`
A Python sidecar that runs `model.forward` with intermediate captures and
writes `.safetensors`; Rust harness loads and compares. The third backend
column (after `reference` and `cpu`/`metal`).

---

## P2: MoE / expert routing

### `--experts` flag (sampling, not WASM)
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`, the `serve` glue
`larql run --experts '0-31=http://host1,32-63=http://host2'` — MoE counterpart
to `--ffn URL`. Maps expert ID ranges to remote URLs; passed through to
`RemoteExpertBackend` in larql-inference. Distinct from the existing
`--experts` flag in `run_cmd.rs` which gates WASM-op dispatch (gcd, base64,
…). Naming overlap to be resolved when this lands. See also
`larql-lql/ROADMAP.md` Phase 3 for the LQL grammar surface.
