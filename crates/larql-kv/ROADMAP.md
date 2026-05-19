# Roadmap — larql-kv

## Current state (as of 2026-05-17)

- Crate extracted from `larql-inference::engines` on 2026-05-09 — see
  [`CHANGELOG.md`](CHANGELOG.md).
- **Seven engines shipped** as of 2026-05-17:
  - Original four: `standard`, `no_cache`, `markov_residual`,
    `unlimited_context`, `turbo_quant`, `apollo`.
  - Three new: `boundary_kv`, `markov_residual_codec`, `boundary_per_layer`.
    Specs in `crates/larql-inference/docs/specs/`:
    [boundary-kv-engine.md](../larql-inference/docs/specs/boundary-kv-engine.md),
    [markov-residual-codec-engine.md](../larql-inference/docs/specs/markov-residual-codec-engine.md),
    [boundary-per-layer-engine.md](../larql-inference/docs/specs/boundary-per-layer-engine.md).
- Consumers wired:
  - `larql-cli bench --engine <spec>` (selector dispatch)
  - `larql-cli bench --via-executor` opts into the new `LayerExecutor`
    surface; falls through to legacy path for unmigrated engines.
  - in-crate `benches/engine_decode.rs` (criterion: dispatch helpers + Standard parity)
- Coverage policy: 90 % line coverage per source file (see
  `coverage-policy.json`); CI gate at `make larql-kv-coverage-policy`.
  Workspace `larql-kv` lib total: **95.55% lines, 95.40% regions, 94.49%
  functions** (2026-05-17 evening, up from 92.12% earlier the same day).
  **All 43 files now ≥90% lines; debt baselines cleared from policy
  file.** The post-Phase-2 push lifted `accuracy_suite/measurement.rs`
  (77→99%), `accuracy_suite/runner.rs` (75→94%), `vindex_compare.rs`
  (65→97%), and `engines/apollo/store.rs` (89→92%) by adding
  formatter + driver tests against the synthetic fixtures; the
  Phase-2 engines (`unlimited_context/engine.rs`, `turbo_quant/engine.rs`,
  `apollo/engine.rs`) all land ≥92% on the new `*_via_executor` methods.

## Architectural cuts (2026-05-17)

Substantive refactors landed; specs reflect the new boundaries.

### Naming hygiene — renamed for honesty

- **`metal_fused_prefill` / `metal_fused_decode_step`** → `fused_prefill`
  / `fused_decode_step`. The "metal" was a lie — `CpuBackend` implements
  `prefill_q4` and `decode_token` via its C Q4 kernel and also takes the
  fused path on `--cpu`. The aliases in `unlimited_context::engine`
  (`quant_prefill_metal`, `quant_decode_token`) follow.
- **`KvEngine::prefill_q4k` / `decode_step_q4k`** → `prefill_quant` /
  `decode_step_quant`. The `_q4k` suffix baked one format into the trait
  surface; the trait is quant-agnostic (dispatches on `index`'s format).
  Internals that are genuinely Q4K-specific (`prefill_q4k_moe`,
  `cpu_q4k_cache_*`, `run_ffn_decode_step_q4k_direct`) keep their names.
- **`ComputeBackend::has_q4()` → `supports_quant(format: QuantFormat)`.**
  Per-format predicate; `CpuBackend` reports support for `Q4_0`, `Q4_K`,
  `Q4_KF`, `Q6_K`; `MetalBackend` adds `Q8_0`. Backends can advertise new
  format support without trait extension.
- **Storage slots `q4k` → `kquant` for K-family fields.** `attn_q4k`,
  `interleaved_q4k`, `set_attn_q4k`, `load_attn_q4k`, etc. — these hold
  K-family quant bytes (Q4_K, Q4_KF, Q6_K — manifest tag picks). Q4_0
  (`attn_q4`) and Q8 (`attn_q8`) slots stay — genuinely format-specific.

### Engine state vs execution — new abstraction

Spec: [engine-state-vs-execution.md](../larql-inference/docs/specs/engine-state-vs-execution.md).

The engines were re-coupling backend / FFN / format decisions into their
state-management code. The new shape:

- **`LayerExecutor` trait** (in `larql-inference::layer_executor`) —
  per-layer execution surface with `run_prefill_layer` /
  `run_decode_layer` returning `(hidden, SharedKV)`. Dispatch kind
  (`Fused` / `PerLayer`) is explicit.
- **`LocalWalkExecutor`** — wraps `run_attention_with_kv_backend` +
  the caller's `&dyn FfnBackend`. The critical decoupling: the executor
  does **not** construct its own `WalkFfn` — it uses whatever the engine
  was handed.
- **Engine trait extension:** `KvEngine::prefill_via_executor`,
  `decode_step_via_executor`, `prefill_quant_via_executor`,
  `decode_step_quant_via_executor`. Default impls fall through to the
  legacy methods so unmigrated engines work unchanged.

### Engines on the new surface

Every engine now runs its own state-policy code; there is no hidden
fall-through to the backend's fused kernel from per-layer engines.
`standard` (and by delegation `boundary_kv`) is the **only** engine
that exercises the fused fast path — via
`ComputeBackend::coarse_prefill` / `coarse_decode_step`, which on
Metal calls `larql_inference::vindex::fused_prefill`.

| Engine | Default dispatch | `*_via_executor` override | Honors FFN backend | Tok/s (Gemma 3 4B Q4K, Metal) | Hot state |
|---|---|---|---|---:|---:|
| `standard` | `ComputeBackend::coarse_prefill` (fused fast path) | n/a (no per-layer code to migrate) | n/a | 104 | 0 MB (backend owns K/V) |
| `boundary_kv` | Delegates to `standard` + emits boundary frames | n/a | n/a | ≈104 | 0 MB |
| `markov_residual` | Per-layer walk via `rs_prefill_walk` | ✅ | ✅ counter test | 3.6 | 6.0 MB |
| `markov_residual_codec` | Per-layer walk via `rs_prefill_codec_walk` (bf16 cold) | ✅ | ✅ counter test | 4.3 | 6.0 MB |
| `unlimited_context` | Windowed checkpoint extension via `process_q4k` | ✅ | ✅ counter test | 25.6 | 4.8 MB |
| `turbo_quant` | Per-layer WHT + Lloyd-Max compression cycle | ✅ | ✅ counter test | 3.9 | 0.6 MB |
| `boundary_per_layer` | Per-layer walk with per-layer codec policy | ✅ (dense) | ✅ counter test | — | matches markov_residual_codec |
| `apollo` | Whole-forward through `forward_layer_range` (boundary prefix + perturb) | ✅ | ✅ counter test | requires store | scales with store |
| `no_cache` | Full re-forward per step (O(N²) wall-time) | ✅ | ✅ already did on legacy `prefill` | — | token list only |

## Coverage debt

None as of 2026-05-17 evening — every file in `crates/larql-kv/src/`
passes the 90 % per-file floor; `coverage-policy.json` ships with an
empty `per_file_line_min_percent` map. Workspace total 95.55 % lines.
Any future regression below 90 % must be fixed with tests, not by
re-introducing a debt baseline.

## Open work

### P0 — correctness / performance

- **Close the 25-30× standard-vs-per-layer gap.** After the
  2026-05-17 fused-bypass strip, the honest numbers are: standard 104
  tok/s, markov-rs 3.6, codec 4.3, unlimited-context 25.6, turbo-quant
  3.9 (Gemma 3 4B Q4K, Metal). Previously every per-layer engine was
  silently running standard's kernel and posting ~103 tok/s, so this
  gap was invisible. Each per-layer engine's per-step cost should be
  decomposed (`larql bench --profile` already does this for
  markov-rs; wire it for the others — see "Engine-level profiler
  coverage" below). Likely culprits: per-layer Metal command-buffer
  overhead (each layer = one submit), per-layer dequant on the
  attn tensors, codec encode/decode work in the inner loop. None of
  these were targetable while the bypass hid them.
- **`LocalFusedExecutor`.** Phase 2 of the
  [engine-state-vs-execution spec](../larql-inference/docs/specs/engine-state-vs-execution.md)
  needs a fused executor for `standard` + `boundary_kv` to migrate
  without losing Metal fast path performance. Open design question
  (spec §9): `KvHandle` opaque cache vs `SharedKV` tuple for fused
  executor's return shape. Probably needs sibling methods on the
  `LayerExecutor` trait (`run_prefill_fused` / `run_decode_step_fused`)
  with default-None for per-layer executors.
- **`BoundaryKvEngine::resume`.** Spec §6.3 describes restoring from a
  frame chain via `MarkovResidualEngine::recompute_kv`. The frame
  emission half is shipped; resume is not. Restore-parity test fixture
  needed (capture frame, verify first-5-tokens agreement under
  `D-@high`).
- **D-METAL-PLE** *(carries from larql-compute roadmap)*: Per-Layer
  Embeddings not implemented in Metal. Engines on Gemma 4 E2B fall through
  the deliberate CPU fallback in `gpu.rs:372-374`, costing ~30× decode.
  Fix is a 1-2 day Metal port of `forward/ple.rs`. Engines themselves are
  PLE-agnostic; the gain accrues through the shared `decode_token` Metal
  path.
- **Engine-level profiler coverage.** `markov_residual` records
  per-stage `embed/recompute_cold/recompute_hot/attention/ffn/total`. The
  other engines do not yet populate `EngineProfiler`; they return `None`
  from `stage_summary()`. Wire them so `larql bench --profile` produces
  comparable breakdowns.

### P1 — capability extensions

- **Wire `--ffn http://...` through the executor surface.** The
  existing `--ffn` flag uses `run_concurrent_ffn` (separate path that
  routes through the `larql-metal` reference, not the engines). Once
  the four remaining engines (P0) are on `*_via_executor`, the bench
  should be able to compose `--engine markov-rs-codec:window=512
  --ffn http://shard:8080` and have the codec engine drive remote FFN
  with bounded local memory. Spec §7 calls this out as a primary use
  case.
- **Auto-rewind variant of `boundary_kv`.** Discussed mid-session as the
  only way to combine Metal's fast-path tok/s with bounded memory: emit
  boundary frame every N chunks, reset Metal's K/V cache, re-prefill
  from the last frame. Bounded memory at ~99% of fast-path tok/s with
  periodic re-prefill spikes. Would need an `evict_after_chunks` config
  on `BoundaryKvEngineConfig` plus a `backend.reset_kv_cache()` call
  after the capture.
- **Per-layer codec calibration sweep harness.** `BoundaryPerLayerEngine`
  ships with `BoundaryCalibrationStore` trait + `InMemoryCalibrationStore`,
  but the actual sweep tool that populates records (per-layer fragility
  measurement → policy generation → end-to-end KL validation) is not in
  tree. Per spec Phase 1 of
  [boundary-per-layer-engine.md](../larql-inference/docs/specs/boundary-per-layer-engine.md).
- **Page-aligned KV slabs for `unlimited_context`.** The current
  `CheckpointStore` uses owned `Vec<f32>` per layer per checkpoint; a
  hugepage-backed slab would cut allocation churn and improve thermal
  steadiness during 370K-token replays.
- **Apollo store on disk.** `apollo` currently expects an in-memory
  `ApolloStore`. Add an mmap loader that reads the constellation map +
  boundary residuals from the same vindex-style on-disk layout as
  `down_meta.bin`, so apollo can serve ~10⁵-entry stores without RAM cost.
- **TurboQuant SIMD packing.** The Lloyd-Max codec works at scalar f32
  today; the rotation step is amenable to NEON / AVX2 vectorisation. The
  encoder is bandwidth-bound at ~95 tok/s decode but the corpus-prep step
  pays N×L×kv_dim of full-precision encode; that's the right place to
  spend the SIMD budget if/when the corpus grows.

### P2 — research / sequencing

- **Non-`Bf16` codecs in `markov_residual_codec`.** v0.1 ships `Bf16`
  only as the safely-defaultable cold codec. `Int8Clip3Sigma`,
  `AdaptiveBlockG32`, `PerGroupInt4G128` are present in `larql-boundary`
  but Exp 46 showed mid-layer failure for `Int8Clip3Sigma`. The
  per-architecture calibration sweep (P1) gates their promotion to
  defaults. Until then, `BoundaryPerLayerEngine` with a custom policy
  is the way to use them.
- **`MarkovResidualCodecEngine` cold tier on actual Q4K deployments.**
  Bench results confirm 50% cold tier saving on dense models and on
  Q4K Gemma with `--via-executor`. Production deployment scenario:
  long-context decode (10k+ tokens) on a 64 GB consumer Mac with a
  large model — the codec's bf16 cold tier is the difference between
  fits-in-RAM and OOM. No technical work blocking this; needs a
  recipe / docs.
- **Cross-engine comparator.** Today `larql bench --engine <spec>` runs one
  engine at a time and `benches/engine_decode.rs` exercises Standard vs the
  parity oracle. The synthesis question is: which engine wins for which
  prompt regime (long-context QA vs short-prompt multi-turn vs streaming
  generation)? A criterion harness sweeping prompt length × decode length ×
  batch size against the production `KvEngine` impls would surface this —
  the retired `kv-cache-benchmark::kv_strategies` synthetic comparator
  measured the wrong thing (encode/decode of random vectors, not real
  decode steady-state).
- **Compositional engines.** `apollo + turbo_quant` would put quantised
  K/V inside the boundary windows; `markov_residual + apollo` would let
  the residual recompute path read pre-projected boundary residuals.
  `markov_residual_codec + boundary_kv` would give bounded cold +
  cross-session resume. Neither is wired today; the trait already
  supports composition because engines hold the persistent state, not
  the dispatch — but the executor + state-policy separation (Phase 2
  spec) makes composition cleaner.

## Closed (recent)

- **2026-05-17 night — Fused-bypass strip: engines are now engines.**
  Every per-layer engine (`markov_residual`, `markov_residual_codec`,
  `unlimited_context`, `turbo_quant`) had a hidden
  `if let Some(h) = fused_prefill(...) { return Some(h); }` short-
  circuit at the top of `prefill_quant` / `decode_step_quant`. The
  short-circuit meant `--engine markov-rs` on Metal silently ran
  `StandardEngine`'s fused kernel instead — five engines tied at
  ~103 tok/s with `hot=0.0MB`, masking every state-policy difference
  and making per-layer optimization invisible. Cut: removed every
  short-circuit; deleted dead `metal_prefill_done` + `force_walk`
  fields and `with_force_walk` builders; dropped the pub(crate)
  `fused_prefill`/`fused_decode_step` re-exports from
  `unlimited_context::engine` (only `StandardEngine::coarse_prefill`
  uses the underlying `larql_inference::vindex::fused_prefill` now,
  via `ComputeBackend::coarse_prefill`). `StandardEngine` remains the
  default engine and the only home of the fused fast path. Bench now
  reports honest numbers: standard 104 tok/s, markov-rs 3.6, codec
  4.3, unlimited-context 25.6, turbo-quant 3.9 — every per-layer
  engine reports non-zero `hot=` memory because their state
  structures actually materialise. The 25-30× standard-vs-per-layer
  gap is the new optimization frontier; previously it was invisible
  because every engine was running the same kernel under different
  labels.
- **2026-05-17 evening — Phase-2 migration completed for the remaining
  three engines.** `unlimited_context`, `turbo_quant`, and `apollo` all
  override `*_via_executor` methods and honor the caller-supplied
  `FfnBackend`. `CountingFfn` stub tests prove per-(token, layer)
  dispatch through the caller's backend. Same push cleared every
  `coverage-policy.json` debt baseline: all 43 files in src/ at ≥90%
  lines, workspace total 95.55%. `larql bench --ffn http://shard:8080`
  now routes through the remote shard for every per-layer engine
  instead of silently constructing a local `WalkFfn`.
- **2026-05-17 — Phase 2 engine migration to `LayerExecutor`.** Four
  engines (`markov_residual`, `markov_residual_codec`,
  `boundary_per_layer`, `no_cache`) override `*_via_executor` methods.
  They drive per-layer dispatch through `executor.run_*_layer` and
  honor the caller's `FfnBackend`. `CountingFfn` stub tests prove the
  FFN parameter is no longer silently ignored. Bench has
  `--via-executor` flag; demoed on Gemma 3 4B Q4K showing the codec
  engine's 50% cold tier saving (22.9 MB → 11.5 MB).
- **2026-05-17 — `LayerExecutor` trait + `LocalWalkExecutor`.** New
  abstraction in `larql-inference::layer_executor` separating state
  policy (engine concern) from execution strategy (executor concern).
  Spec at
  [engine-state-vs-execution.md](../larql-inference/docs/specs/engine-state-vs-execution.md).
- **2026-05-17 — `q4k` → `kquant` storage rename.** K-family storage
  slots (`attn_q4k`, `interleaved_q4k`, manifests, setters, loaders)
  renamed for consistency with accessor naming (`attn_kquant_layer_data`).
  Q4_0 and Q8 slots unchanged. ~60 sites touched.
- **2026-05-17 — `has_q4()` → `supports_quant(format)`.** Per-format
  predicate on `ComputeBackend`. 79 call sites migrated to
  `supports_quant(QuantFormat::Q4_K)`. Enables future Q6_K / FP4
  fused-pipeline backends without trait extension.
- **2026-05-17 — `KvEngine::prefill_q4k` / `decode_step_q4k` →
  `prefill_quant` / `decode_step_quant`.** Trait surface naming made
  quant-agnostic. 112 sites updated. Internals that are genuinely
  Q4K-specific kept their names.
- **2026-05-17 — `metal_fused_*` → `fused_*` rename.** The "metal"
  prefix was a lie: `CpuBackend` implements `prefill_q4` and
  `decode_token` via its C Q4 kernel. Aliases in
  `unlimited_context::engine` follow.
- **2026-05-17 — `BoundaryKvEngine`, `MarkovResidualCodecEngine`,
  `BoundaryPerLayerEngine` shipped.** All three new engines have
  contracts in `crates/larql-inference/docs/specs/`. Per-file coverage
  ≥94 % lines on every new file. Bench demoed end-to-end on Gemma 3 4B,
  Gemma 4 E2B, 26B-A4B, 31B, Qwen3 0.6B (dense + Q4K).
- **2026-05-09 — Initial extraction.** `engines/` carved out of
  `larql-inference` into the new `larql-kv` crate. ~5,540 LOC moved with
  no semantic changes. All four engines + `KvEngine` + accuracy /
  profiler helpers now ship from this crate.

## Non-goals

- **Sampling.** Engines return hidden states; sampling lives in
  `larql_inference::layer_graph::generate::Sampler`. Don't add sampling
  helpers here.
- **Tokenisation / chat templates.** Out of scope; the engines operate on
  `&[u32]` token IDs already produced by `larql_inference::tokenizer` /
  `chat`.
- **Generic K/V backends for non-transformer architectures.** The
  `KvEngine` trait references `ModelWeights` directly. Generalising to
  state-space models or RNNs is not on this roadmap; rebuilds are cheap
  and that effort would belong in larql-inference's layer-graph surface.
